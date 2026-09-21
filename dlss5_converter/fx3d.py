"""Volumetric FX and particles drawn over the splat scene, all on the GPU.

Adapted from Depth Animator's renderer3d (VOLUME_SHADER / PARTICLE_SHADER). The
difference is where scene depth comes from: there it was a hardware depth
buffer under a mesh; here the splat pass writes coverage-weighted view depth
into a second colour target, and both passes read that.

Volumes are raymarched inside an oriented box: 40 steps of fbm noise shaped per
kind (fog, smoke, fire, cloud, god rays), stopping at the first splat surface
and thinning as they approach it so fog touches a wall instead of being cut by
it. Particles are camera-facing sprites animated entirely in the vertex shader
(no CPU state per particle), hidden where a surface is nearer.

This runs through wgpu (D3D12/Vulkan), which on an NVIDIA card is the same GPU
CUDA would use. CUDA would add a large dependency and a copy between the two
APIs every frame, for no speed gain on this workload.
"""

from __future__ import annotations

import math

import numpy as np

VOLUME_KINDS = {"fog": 0.0, "smoke": 1.0, "fire": 2.0, "cloud": 3.0, "godrays": 4.0}
PARTICLE_KINDS = {"smoke": 0.0, "fire": 1.0, "embers": 2.0, "dust": 3.0, "snow": 4.0, "clouds": 5.0}
MAX_PARTICLES = 20000

VOLUME_SHADER = """
struct VU {
    vp: mat4x4<f32>,
    camera: vec4<f32>,
    centre: vec4<f32>,
    half_size: vec4<f32>,
    shape: vec4<f32>,        // x density, y noise scale, z detail, w speed
    colour: vec4<f32>,       // rgb colour, w emission
    light: vec4<f32>,        // xyz key direction, w light response
    ambient_kind: vec4<f32>, // rgb ambient, w kind
    timing: vec4<f32>,       // x time, y seed
    rot0: vec4<f32>,         // rows of the local->world rotation
    rot1: vec4<f32>,
    rot2: vec4<f32>,
    forward: vec4<f32>,      // xyz camera forward, w soft-fade distance
};
@group(0) @binding(0) var<uniform> u: VU;
@group(0) @binding(1) var scene_depth: texture_2d<f32>;

fn to_world(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(u.rot0.xyz, v), dot(u.rot1.xyz, v), dot(u.rot2.xyz, v));
}
fn to_local(v: vec3<f32>) -> vec3<f32> {
    return u.rot0.xyz * v.x + u.rot1.xyz * v.y + u.rot2.xyz * v.z;
}

// Distance along this pixel's ray to the splat surface. The depth target holds
// (view depth * coverage, coverage); a pixel with almost no coverage is open
// sky as far as the fog is concerned.
fn scene_distance(frag: vec2<f32>, dir: vec3<f32>) -> f32 {
    let d = textureLoad(scene_depth, vec2<i32>(frag), 0);
    if (d.a < 0.05) { return 1.0e9; }
    let planar = d.r / d.a;
    return planar / max(dot(dir, u.forward.xyz), 1e-3);
}

const CUBE = array<vec3<f32>, 36>(
 vec3(-1.,-1., 1.),vec3( 1.,-1., 1.),vec3( 1., 1., 1.), vec3(-1.,-1., 1.),vec3( 1., 1., 1.),vec3(-1., 1., 1.),
 vec3( 1.,-1.,-1.),vec3(-1.,-1.,-1.),vec3(-1., 1.,-1.), vec3( 1.,-1.,-1.),vec3(-1., 1.,-1.),vec3( 1., 1.,-1.),
 vec3(-1.,-1.,-1.),vec3(-1.,-1., 1.),vec3(-1., 1., 1.), vec3(-1.,-1.,-1.),vec3(-1., 1., 1.),vec3(-1., 1.,-1.),
 vec3( 1.,-1., 1.),vec3( 1.,-1.,-1.),vec3( 1., 1.,-1.), vec3( 1.,-1., 1.),vec3( 1., 1.,-1.),vec3( 1., 1., 1.),
 vec3(-1., 1., 1.),vec3( 1., 1., 1.),vec3( 1., 1.,-1.), vec3(-1., 1., 1.),vec3( 1., 1.,-1.),vec3(-1., 1.,-1.),
 vec3(-1.,-1.,-1.),vec3( 1.,-1.,-1.),vec3( 1.,-1., 1.), vec3(-1.,-1.,-1.),vec3( 1.,-1., 1.),vec3(-1.,-1., 1.)
);
struct Out { @builtin(position) position: vec4<f32>, @location(0) world: vec3<f32> };
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> Out {
    var out: Out;
    out.world = u.centre.xyz + to_world(CUBE[i] * u.half_size.xyz);
    out.position = u.vp * vec4(out.world, 1.0);
    return out;
}
fn hash31(p: vec3<f32>) -> f32 { return fract(sin(dot(p, vec3(127.1,311.7,74.7))) * 43758.5453); }
fn noise(p: vec3<f32>) -> f32 {
    let i=floor(p); let f=fract(p); let q=f*f*(3.0-2.0*f);
    return mix(mix(mix(hash31(i),hash31(i+vec3(1,0,0)),q.x),mix(hash31(i+vec3(0,1,0)),hash31(i+vec3(1,1,0)),q.x),q.y),
               mix(mix(hash31(i+vec3(0,0,1)),hash31(i+vec3(1,0,1)),q.x),mix(hash31(i+vec3(0,1,1)),hash31(i+vec3(1,1,1)),q.x),q.y),q.z);
}
fn fbm(p0: vec3<f32>) -> f32 {
    var p=p0; var value=0.0; var amp=0.55;
    for(var i=0;i<4;i=i+1){ value += noise(p)*amp; p=p*2.03+vec3(1.7,3.1,2.4); amp*=0.5; }
    return value;
}
// Drawn with FRONT faces culled: the back faces of the box are always on
// screen, including when the camera is inside the fog, which is exactly when
// front-face drawing would make the volume vanish.
@fragment fn fs_main(in: Out) -> @location(0) vec4<f32> {
    let world_dir = normalize(in.world - u.camera.xyz);
    let ro = to_local(u.camera.xyz - u.centre.xyz) / u.half_size.xyz;
    let rd = to_local(world_dir) / u.half_size.xyz;
    let inv=1.0/rd;
    let lo=min((-vec3(1.0)-ro)*inv,(vec3(1.0)-ro)*inv);
    let hi=max((-vec3(1.0)-ro)*inv,(vec3(1.0)-ro)*inv);
    var t=max(max(lo.x,lo.y),max(lo.z,0.0));
    var end=min(hi.x,min(hi.y,hi.z));
    let wall = scene_distance(in.position.xy, world_dir);
    end = min(end, wall);
    if(end<=t){discard;}
    let dt=(end-t)/40.0;
    let fade = max(u.forward.w, 1e-4);
    var trans=1.0; var accum=vec3(0.0);
    for(var i=0;i<40;i=i+1){
        let travelled = t + dt * (f32(i) + 0.5);
        let world=u.camera.xyz+world_dir*travelled;
        let p=to_local(world-u.centre.xyz)/u.half_size.xyz;
        let radial=max(0.0,1.0-dot(p,p)*0.34);
        let drift=vec3(0.13,-u.shape.w,0.09)*u.timing.x;
        let n=fbm(p*u.shape.y+drift+u.timing.y);
        let kind=u.ambient_kind.w;
        var d=u.shape.x*radial*smoothstep(0.22,0.82,n);
        var c=u.colour.rgb;
        if(kind>1.5 && kind<2.5){
            let flame=max(0.0,1.0-(p.y+1.0)*0.42);
            d*=flame; c=mix(vec3(1.0,0.04,0.005),vec3(1.0,0.75,0.08),clamp(n+flame*0.5,0.0,1.0));
        } else if(kind>2.5 && kind<3.5){ d*=smoothstep(0.08,0.46,n); }
        else if(kind>3.5){
            let view_to_light=max(dot(normalize(u.light.xyz),normalize(u.camera.xyz-world)),0.0);
            let shafts=pow(view_to_light,5.0)*(0.3+0.7*smoothstep(0.35,0.8,n));
            d*=shafts*2.2;
        }
        d *= clamp((wall - travelled) / fade, 0.0, 1.0);
        let step_alpha=1.0-exp(-d*dt*2.4);
        let normal=normalize(vec3(noise(p*7.0+vec3(.05,0,0))-noise(p*7.0-vec3(.05,0,0)), .35, noise(p*7.0+vec3(0,0,.05))-noise(p*7.0-vec3(0,0,.05))));
        let direct=max(dot(normal,normalize(u.light.xyz)),0.0)*u.light.w;
        let lit=u.ambient_kind.rgb+vec3(direct);
        let sample_colour=c*lit+c*u.colour.w;
        accum+=trans*sample_colour*step_alpha; trans*=1.0-step_alpha;
        if(trans<0.015){break;}
    }
    return vec4(accum,1.0-trans);   // premultiplied
}
"""

PARTICLE_SHADER = """
struct PU {
    vp: mat4x4<f32>,
    right: vec4<f32>, up: vec4<f32>, position: vec4<f32>, extent: vec4<f32>,
    motion: vec4<f32>,       // x time, y lifetime, z particle size, w speed
    physics: vec4<f32>,      // x spread, y gravity (along dir), z turbulence, w seed
    colour: vec4<f32>,       // rgb, w emission
    ambient_kind: vec4<f32>, // rgb ambient, w kind
    view_z: vec4<f32>,       // row 2 of the view matrix
    style: vec4<f32>,        // x opacity, y plane count, z bounce, w growth
    dir: vec4<f32>,          // xyz travel direction, w drag
    planes: array<vec4<f32>, 4>,   // collision planes n.p + c >= 0 is free space
};
@group(0) @binding(0) var<uniform> u: PU;
@group(0) @binding(1) var scene_depth: texture_2d<f32>;
fn hash(n:f32)->f32{return fract(sin(n)*43758.5453);}
struct Out{@builtin(position) position:vec4<f32>,@location(0) uv:vec2<f32>,@location(1) fade:f32,
           @location(2) depth:f32};

// Stateless physics: every particle's position is a closed-form function of
// its age, so nothing is simulated per frame on the CPU and a frame at any
// time (scrubbing, export) is exact.
@vertex fn vs_main(@builtin(vertex_index) vi:u32)->Out{
    let id=f32(vi/6u); let corner=vi%6u;
    var quad=array<vec2<f32>,6>(vec2(-1,-1),vec2(1,-1),vec2(1,1),vec2(-1,-1),vec2(1,1),vec2(-1,1));
    let q=quad[corner]; let phase=hash(id*19.19+u.physics.w);
    let age=fract(u.motion.x/max(u.motion.y,0.01)+phase); let s=age*u.motion.y;

    // Travel axis chosen by the user, and two axes across it for the cone.
    let dir=normalize(u.dir.xyz + vec3(1e-6, 0.0, 0.0));
    var helper=vec3(1.0,0.0,0.0);
    if (abs(dir.x) > 0.9) { helper=vec3(0.0,0.0,1.0); }
    let side=normalize(cross(dir,helper));
    let side2=cross(dir,side);

    let random=vec3(hash(id*7.1)-.5,hash(id*13.7)-.5,hash(id*31.3)-.5);
    // Spawn anywhere in the emitter's box, exactly the box drawn in the
    // editor; spread only fans particles out as they travel.
    var p=u.position.xyz+random*u.extent.xyz;

    // Launch speed dies away under drag: distance = v (1 - e^-kt) / k. Then
    // a constant push along the axis (buoyancy, or gravity when negative).
    let k=max(u.dir.w,1e-3);
    let launch=u.motion.w*(1.0-exp(-k*s))/k;
    p+=dir*(launch+0.5*u.physics.y*s*s);
    // Each particle leaves at its own angle; the plume widens with age,
    // faster for kinds that grow (smoke spreads as it rises).
    let cone=(hash(id*3.3)-.5)*side+(hash(id*5.9)-.5)*side2;
    p+=cone*(abs(launch)*0.6+0.05*s)*u.physics.x*(1.0+u.style.w*age);

    // Turbulence: a swirling field of layered sines through space and time,
    // stronger as a particle ages, so smoke curls and breaks up instead of
    // rising in straight lines.
    let tq=u.physics.z;
    let f=p*1.7+dir*(s*0.6);
    let curl=vec3(sin(f.y*2.1+s*1.3+id)+0.5*sin(f.z*3.7+s*0.7),
                  sin(f.z*1.9+s*1.1)+0.5*sin(f.x*2.9+id*0.3),
                  sin(f.x*2.3+s*0.9)+0.5*sin(f.y*3.1+id));
    p+=curl*tq*0.09*(0.25+age);

    // Collisions with the scene's floor, walls and ceiling. A particle that
    // ends up on the wrong side is put back on the surface (slides along it)
    // or mirrored off it (bounces), by the bounce amount.
    let n_planes=i32(u.style.y);
    for (var i=0; i<4; i=i+1) {
        if (i >= n_planes) { break; }
        let pl=u.planes[i];
        let d=dot(pl.xyz,p)+pl.w;
        if (d<0.0) { p-=pl.xyz*d*(1.0+u.style.z); }
    }

    let size=u.motion.z*(0.35+sin(age*3.14159)*0.9)*(1.0+u.style.w*age);
    let centre_depth = -(dot(u.view_z.xyz, p) + u.view_z.w);
    p+=u.right.xyz*q.x*size+u.up.xyz*q.y*size;
    var out:Out; out.position=u.vp*vec4(p,1); out.uv=q; out.fade=sin(age*3.14159);
    out.depth=centre_depth; return out;
}
@fragment fn fs_main(in:Out)->@location(0) vec4<f32>{
    let r=length(in.uv); if(r>1.0){discard;}
    // Hidden behind nearer splats, with a soft edge so a particle drifting
    // into a wall fades rather than being sliced.
    let d = textureLoad(scene_depth, vec2<i32>(in.position.xy), 0);
    var occl = 1.0;
    if (d.a > 0.05) {
        let surface = d.r / d.a;
        occl = clamp((surface - in.depth) / 0.08, 0.0, 1.0);
    }
    let soft=pow(1.0-smoothstep(.15,1.0,r),1.4)*in.fade*occl;
    let lit=u.ambient_kind.rgb+vec3(0.25);
    let rgb=u.colour.rgb*(lit+vec3(u.colour.w));
    return vec4(rgb,soft*0.72*u.style.x);
}
"""


def euler_matrix(rotation) -> np.ndarray:
    """Local->world rotation for Euler XYZ radians, applied X then Y then Z."""
    rx, ry, rz = (float(v) for v in rotation)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float32)
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float32)
    return (mz @ my @ mx).astype(np.float32)


def lighting_values(effects):
    """(ambient, key colour, key direction) from the effects' lighting."""
    if effects is None or not effects.lighting.enabled:
        return (np.array((0.55, 0.60, 0.72), np.float32),
                np.array((1.0, 0.94, 0.84), np.float32),
                np.array((-0.4, 0.6, 0.7), np.float32))
    lighting = effects.lighting
    gain = max(0.0, lighting.strength) * (2.0 ** lighting.exposure)
    direction = np.asarray(lighting.key_direction, np.float32)
    angle = math.radians(lighting.rotation)
    c, s = math.cos(angle), math.sin(angle)
    direction = np.array((c * direction[0] - s * direction[2], direction[1],
                          s * direction[0] + c * direction[2]), np.float32)
    return (np.asarray(lighting.ambient_colour, np.float32) * gain,
            np.asarray(lighting.key_colour, np.float32) * gain, direction)


class FxPass:
    """Pipelines and per-frame encoding for volumes and particles."""

    def __init__(self, device, colour_format: str = "rgba16float") -> None:
        import wgpu
        self._wgpu = wgpu
        self.device = device
        layout = device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": wgpu.TextureSampleType.unfilterable_float}}])
        self._layout = layout
        pl = device.create_pipeline_layout(bind_group_layouts=[layout])

        def pipe(code: str, cull, colour_src: str):
            mod = device.create_shader_module(code=code)
            return device.create_render_pipeline(
                layout=pl,
                vertex={"module": mod, "entry_point": "vs_main", "buffers": []},
                primitive={"topology": wgpu.PrimitiveTopology.triangle_list, "cull_mode": cull},
                fragment={"module": mod, "entry_point": "fs_main", "targets": [{
                    "format": colour_format,
                    "blend": {"color": {"src_factor": colour_src,
                                        "dst_factor": "one-minus-src-alpha", "operation": "add"},
                              "alpha": {"src_factor": "one",
                                        "dst_factor": "one-minus-src-alpha", "operation": "add"}}}]})

        # The raymarch returns premultiplied colour; particles return straight
        # alpha, so their colour is scaled by alpha in the blend instead.
        self._volume = pipe(VOLUME_SHADER, wgpu.CullMode.front, "one")
        self._particle = pipe(PARTICLE_SHADER, wgpu.CullMode.none, "src-alpha")
        self._pool: dict[str, list] = {"v": [], "p": []}

    def _group(self, kind: str, index: int, values: np.ndarray, depth_view):
        wgpu = self._wgpu
        pool = self._pool[kind]
        while len(pool) <= index:
            pool.append(self.device.create_buffer(
                size=512, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST))
        buf = pool[index]
        self.device.queue.write_buffer(buf, 0, values.astype(np.float32).tobytes())
        return self.device.create_bind_group(layout=self._layout, entries=[
            {"binding": 0, "resource": {"buffer": buf}},
            {"binding": 1, "resource": depth_view}])

    def encode(self, enc, colour_view, depth_view, view: np.ndarray, proj: np.ndarray,
               camera_pos, effects, time_seconds: float, planes=None) -> None:
        if effects is None or (not effects.volumes and not effects.emitters):
            return
        wgpu = self._wgpu
        vp = proj @ view
        ambient, _key, key_dir = lighting_values(effects)
        ambient = np.clip(ambient, 0.0, 8.0)
        forward = -view[2, :3]
        rp = enc.begin_render_pass(color_attachments=[{
            "view": colour_view, "load_op": wgpu.LoadOp.load, "store_op": wgpu.StoreOp.store}])
        rp.set_pipeline(self._volume)
        slot = 0
        for vol in effects.volumes:
            if not vol.enabled or vol.density <= 0.0:
                continue
            v = np.zeros(64, np.float32)
            v[:16] = np.ascontiguousarray(vp.T).ravel()
            v[16:19] = camera_pos
            v[20:23] = vol.position
            v[24:27] = np.maximum(np.asarray(vol.size, np.float32) * 0.5, 0.001)
            v[28:32] = (vol.density, vol.noise_scale, vol.detail, vol.speed)
            v[32:35] = vol.colour
            v[35] = vol.emission
            v[36:39] = key_dir
            v[39] = vol.light_response
            v[40:43] = ambient
            v[43] = VOLUME_KINDS.get(vol.kind, 0.0)
            v[44:46] = (time_seconds, vol.seed)
            rot = euler_matrix(getattr(vol, "rotation", (0.0, 0.0, 0.0)))
            v[48:51], v[52:55], v[56:59] = rot[0], rot[1], rot[2]
            v[60:63] = forward
            v[63] = max(float(np.min(vol.size)) * 0.25, 0.05)
            rp.set_bind_group(0, self._group("v", slot, v, depth_view))
            slot += 1
            rp.draw(36, 1, 0, 0)
        rp.set_pipeline(self._particle)
        slot = 0
        # User planes replace the detected ones: their normal already points
        # into free space. The first floor also defines "up" for directions.
        user_planes, floor_rot = [], None
        for pl in getattr(effects, "planes", []):
            if not pl.enabled:
                continue
            n = np.asarray(pl.normal(), np.float32)
            user_planes.append((n, -float(n @ np.asarray(pl.position, np.float32))))
            if floor_rot is None and pl.kind == "floor":
                floor_rot = euler_matrix(pl.rotation)
        for em in effects.emitters:
            if not em.enabled or em.count <= 0:
                continue
            v = np.zeros(80, np.float32)
            v[:16] = np.ascontiguousarray(vp.T).ravel()
            v[16:19] = view[0, :3]
            v[20:23] = view[1, :3]
            v[24:27] = em.position
            v[28:31] = em.size
            v[32:36] = (time_seconds * em.rate, em.lifetime, em.particle_size, em.speed)
            v[36:40] = (em.spread, em.gravity, em.turbulence, em.seed)
            v[40:43] = em.colour
            v[43] = em.emission
            v[44:47] = ambient * em.light_response
            v[47] = PARTICLE_KINDS.get(em.kind, 0.0)
            v[48:52] = view[2]
            v[52] = float(np.clip(getattr(em, "opacity", 1.0), 0.0, 1.0))
            v[54] = float(np.clip(getattr(em, "bounce", 0.0), 0.0, 1.0))
            v[55] = max(0.0, float(getattr(em, "growth", 0.0)))
            direction = np.asarray(getattr(em, "direction", (0.0, 1.0, 0.0)), np.float32)
            if floor_rot is not None:
                direction = floor_rot @ direction     # "up" is the user's floor
            v[56:59] = direction
            v[59] = max(0.0, float(getattr(em, "drag", 0.6)))
            if getattr(em, "collide", True) and user_planes:
                count = 0
                for n, c in user_planes[:4]:
                    v[60 + 4 * count: 63 + 4 * count] = n
                    v[63 + 4 * count] = c
                    count += 1
                v[53] = count
            elif getattr(em, "collide", True) and planes:
                # Which side of each plane is free space. A horizontal plane
                # below the photo's camera is a floor (keep particles above
                # it) and one above it a ceiling (keep them below), whatever
                # side the emitter was dropped on; judging floors by the
                # emitter trapped particles under the ground when a new
                # emitter spawned slightly low. Walls keep the emitter's side.
                origin = np.asarray(em.position, np.float32)
                count = 0
                for n, c in planes[:4]:
                    n = np.asarray(n, np.float32)
                    c = float(c)
                    if abs(float(n[1])) > 0.85:
                        if n[1] < 0.0:
                            n, c = -n, -c                  # normal up
                        floor = (-c / float(n[1])) < 0.0   # plane height below camera
                        if not floor:
                            n, c = -n, -c                  # ceiling: free side down
                    elif float(n @ origin + c) < 0.0:
                        n, c = -n, -c
                    v[60 + 4 * count: 63 + 4 * count] = n
                    v[63 + 4 * count] = c
                    count += 1
                v[53] = count
            rp.set_bind_group(0, self._group("p", slot, v, depth_view))
            slot += 1
            rp.draw(min(int(em.count), MAX_PARTICLES) * 6, 1, 0, 0)
        rp.end()
