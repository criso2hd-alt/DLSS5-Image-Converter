"""Volumetric FX and particles drawn over the splat scene, all on the GPU.

Adapted from Depth Animator's renderer3d (VOLUME_SHADER / PARTICLE_SHADER). The
difference is where scene depth comes from: there it was a hardware depth
buffer under a mesh; here the splat pass writes coverage-weighted view depth
into a second colour target, and both passes read that.

Volumes are raymarched inside an oriented box: 40 steps of fbm noise shaped per
kind (fog, smoke, fire, cloud, god rays), stopping at the first splat surface
and thinning as they approach it so fog touches a wall instead of being cut by
it. Particles are camera-facing sprites animated entirely in the vertex shader
(no CPU state per particle), hidden where a surface is nearer. Each kind has
its own sprite shape and colour over life (embers cool from white hot to ash),
and glowing ones blend additively.

This runs through wgpu (D3D12/Vulkan), which on an NVIDIA card is the same GPU
CUDA would use. CUDA would add a large dependency and a copy between the two
APIs every frame, for no speed gain on this workload.
"""

from __future__ import annotations

import math

import numpy as np

VOLUME_KINDS = {"fog": 0.0, "smoke": 1.0, "fire": 2.0, "cloud": 3.0, "godrays": 4.0}
PARTICLE_KINDS = {"smoke": 0.0, "fire": 1.0, "embers": 2.0, "dust": 3.0, "snow": 4.0, "clouds": 5.0,
                  "rain": 6.0}
MAX_PARTICLES = 20000
#: One strike lasts this long, flicker included.
STRIKE_SECONDS = 0.45

BOLT_SHADER = """
struct BU { vp: mat4x4<f32>, colour: vec4<f32>, view_z: vec4<f32> };
@group(0) @binding(0) var<uniform> u: BU;
@group(0) @binding(1) var scene_depth: texture_2d<f32>;
struct VIn{@location(0) pos:vec3<f32>, @location(1) across:f32, @location(2) strength:f32};
struct Out{@builtin(position) position:vec4<f32>, @location(0) across:f32,
           @location(1) strength:f32, @location(2) depth:f32};
@vertex fn vs_main(v:VIn)->Out{
    var o:Out;
    o.position=u.vp*vec4(v.pos,1.0);
    o.across=v.across; o.strength=v.strength;
    o.depth=-(dot(u.view_z.xyz,v.pos)+u.view_z.w);
    return o;
}
// A white-hot core inside a wide coloured glow, added to the frame. Hidden
// behind nearer surfaces like particles are, so a strike behind a building
// lights the sky around it instead of drawing over it.
@fragment fn fs_main(i:Out)->@location(0) vec4<f32>{
    let d=textureLoad(scene_depth, vec2<i32>(i.position.xy), 0);
    var occl=1.0;
    if (d.a>0.05) { occl=clamp((d.r/d.a-i.depth)/0.3,0.0,1.0); }
    let a=i.across;
    // The ribbon is mostly glow: the core is a thin line down its middle.
    let core=exp(-a*a*120.0);
    let glow=exp(-a*a*4.0);
    let rgb=(vec3(1.0)*core*0.45+u.colour.rgb*glow*0.05)*u.colour.w*i.strength*occl;
    return vec4(rgb,0.0);
}
"""


def _rng(*keys):
    return np.random.default_rng([int(abs(k) * 1000) & 0xFFFFFFFF for k in keys])


#: Strike indices at or above this are the user's placed strikes, so their
#: bolt shapes never repeat a random strike's.
MANUAL_INDEX = 1_000_000


def _flicker(rng, dt: float) -> float:
    """Brightness `dt` seconds into a strike: two or three quick flashes."""
    pulses = [0.0, 0.06 + 0.05 * rng.random(), 0.18 + 0.1 * rng.random()]
    pulses = pulses[:1 + int(rng.integers(1, 3))]
    return max((math.exp(-(dt - p) / 0.05) for p in pulses if dt >= p), default=0.0)


def strike_at(item, t: float) -> tuple[float, int]:
    """(brightness 0..1, strike index) of a lightning item at time t.

    Random strikes: time is cut into slots of 60/rate seconds and each slot
    may hold one strike at a random moment inside it. Placed strikes happen
    exactly at the user's times. Everything comes from the seed and the slot
    or time, so the storm is the same on every playback."""
    best = (0.0, -1)
    rate = float(item.rate)
    if rate > 0.0:
        slot = 60.0 / rate
        k0 = int(math.floor(t / slot))
        for k in (k0 - 1, k0):
            if k < 0:
                continue
            rng = _rng(item.seed, k + 1)
            if rng.random() > 0.8:
                continue                   # the odd gap keeps it irregular
            start = k * slot + rng.random() * max(slot - STRIKE_SECONDS, 0.0)
            dt = t - start
            if 0.0 <= dt < STRIKE_SECONDS:
                env = _flicker(rng, dt)
                if env > best[0]:
                    best = (env, k)
    for i, start in enumerate(sorted(getattr(item, "strike_times", []) or [])):
        dt = t - float(start)
        if 0.0 <= dt < STRIKE_SECONDS:
            rng = _rng(item.seed, start * 1000.0 + 1.0, 3)
            rng.random()                   # keep the same draw order as random strikes
            env = _flicker(rng, dt)
            if env > best[0]:
                best = (env, MANUAL_INDEX + i)
    return best


def _jagged(a: np.ndarray, b: np.ndarray, rng, levels: int, rough: float) -> np.ndarray:
    """Midpoint displacement: split every segment and push the midpoint
    sideways by a share of its length, level after level."""
    pts = [a, b]
    for _ in range(levels):
        out = [pts[0]]
        for p, q in zip(pts, pts[1:]):
            seg = q - p
            length = float(np.linalg.norm(seg))
            off = rng.normal(size=3)
            off -= seg * float(off @ seg) / (length * length + 1e-9)
            out += [(p + q) * 0.5 + off * length * rough, q]
        pts = out
    return np.asarray(pts, np.float32)


def bolt_paths(item, index: int) -> list[tuple[np.ndarray, float, float]]:
    """The bolt of strike `index`: (points, width, strength) per branch.
    The main channel runs from the sky down to the target; a few thinner
    forks leave it partway and die out."""
    rng = _rng(item.seed, index + 1, 7)
    target = np.asarray(item.position, np.float32)
    h = max(float(item.height), 0.1)
    top = target + np.array([rng.normal() * 0.25 * h, h, rng.normal() * 0.25 * h], np.float32)
    main = _jagged(top, target, rng, 7, 0.2)
    paths = [(main, 1.0, 1.0)]
    for _ in range(int(rng.integers(2, 5))):
        i = int(rng.integers(len(main) // 6, len(main) * 2 // 3))
        start = main[i]
        rest = target - start
        side = rng.normal(size=3).astype(np.float32)
        side -= rest * float(side @ rest) / (float(rest @ rest) + 1e-9)
        side /= float(np.linalg.norm(side)) + 1e-9
        end = start + rest * rng.uniform(0.25, 0.5) + side * float(np.linalg.norm(rest)) * 0.35
        paths.append((_jagged(start, end, rng, 5, 0.25), 0.45, 0.55))
    return paths


def bolt_vertices(item, index: int, camera_pos, brightness: float) -> np.ndarray:
    """Camera-facing ribbons along every branch: (x, y, z, across, strength)."""
    cam = np.asarray(camera_pos, np.float32)
    half = 0.06 + 0.01 * max(float(item.height), 0.1)
    rows = []
    for pts, width, strength in bolt_paths(item, index):
        n = len(pts) - 1
        for j in range(n):
            a, b = pts[j], pts[j + 1]
            side = np.cross(b - a, cam - a)
            side = side / (float(np.linalg.norm(side)) + 1e-9) * half * width
            # Branches fade towards their tips.
            s = brightness * strength * (1.0 - 0.7 * j / max(n, 1) if width < 1.0 else 1.0)
            for p, x in ((a - side, -1.0), (a + side, 1.0), (b + side, 1.0),
                         (a - side, -1.0), (b + side, 1.0), (b - side, -1.0)):
                rows.append((p[0], p[1], p[2], x, s))
    return np.asarray(rows, np.float32)


def scene_flash(effects, t: float) -> float:
    """How much the whole frame lights up at time t, from every strike."""
    total = 0.0
    for item in getattr(effects, "strikes", []) if effects is not None else []:
        if item.enabled and item.flash > 0.0:
            total += strike_at(item, t)[0] * float(item.flash)
    return min(total, 3.0)

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
    light: vec4<f32>,        // xyz key light direction (towards the light)
};
@group(0) @binding(0) var<uniform> u: PU;
@group(0) @binding(1) var scene_depth: texture_2d<f32>;

const SMOKE: i32 = 0;
const FIRE: i32 = 1;
const EMBERS: i32 = 2;
const DUST: i32 = 3;
const SNOW: i32 = 4;
const CLOUDS: i32 = 5;
const RAIN: i32 = 6;
// Embers glow until COOL_START of their life and are fully ash by ASH_AT.
const COOL_START: f32 = 0.18;
const ASH_AT: f32 = 0.62;

fn hash(n:f32)->f32{return fract(sin(n)*43758.5453);}
fn h2(p:vec2<f32>)->f32{return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
fn vnoise(p:vec2<f32>)->f32{
    let i=floor(p); let f=fract(p); let w=f*f*(3.0-2.0*f);
    return mix(mix(h2(i),h2(i+vec2(1.0,0.0)),w.x),
               mix(h2(i+vec2(0.0,1.0)),h2(i+vec2(1.0,1.0)),w.x),w.y);
}
fn fbm(p:vec2<f32>)->f32{
    var a=0.5; var s=0.0; var q=p;
    for (var i=0; i<4; i=i+1) { s+=a*vnoise(q); q=q*2.03+vec2(1.7,9.2); a*=0.5; }
    return s;
}
fn kind()->i32{return i32(u.ambient_kind.w+0.5);}

struct Axes{dir:vec3<f32>, side:vec3<f32>, side2:vec3<f32>};
fn axes()->Axes{
    let dir=normalize(u.dir.xyz + vec3(1e-6, 0.0, 0.0));
    var helper=vec3(1.0,0.0,0.0);
    if (abs(dir.x) > 0.9) { helper=vec3(0.0,0.0,1.0); }
    let side=normalize(cross(dir,helper));
    return Axes(dir, side, cross(dir,side));
}

// Stateless physics: every particle's position is a closed-form function of
// its age, so nothing is simulated per frame on the CPU and a frame at any
// time (scrubbing, export) is exact. A function rather than inline code so
// the vertex stage can evaluate it twice and get a velocity for streaks.
// w is 1 when a collision plane stopped the particle (rain splashes).
fn position_at(id:f32, s:f32)->vec4<f32>{
    let ax=axes();
    let age=clamp(s/max(u.motion.y,0.01),0.0,1.0);
    let random=vec3(hash(id*7.1)-.5,hash(id*13.7)-.5,hash(id*31.3)-.5);
    // Spawn anywhere in the emitter's box, exactly the box drawn in the
    // editor; spread only fans particles out as they travel.
    var p=u.position.xyz+random*u.extent.xyz;

    // Launch speed dies away under drag: distance = v (1 - e^-kt) / k. Then
    // a constant push along the axis (buoyancy, or gravity when negative).
    let k=max(u.dir.w,1e-3);
    let launch=u.motion.w*(1.0-exp(-k*s))/k;
    p+=ax.dir*(launch+0.5*u.physics.y*s*s);
    // Each particle leaves at its own angle; the plume widens with age,
    // faster for kinds that grow (smoke spreads as it rises).
    let cone=(hash(id*3.3)-.5)*ax.side+(hash(id*5.9)-.5)*ax.side2;
    p+=cone*(abs(launch)*0.6+0.05*s)*u.physics.x*(1.0+u.style.w*age);

    // Turbulence: a swirling field of layered sines through space and time,
    // stronger as a particle ages, so smoke curls and breaks up instead of
    // rising in straight lines.
    let tq=u.physics.z;
    let f=p*1.7+ax.dir*(s*0.6);
    let curl=vec3(sin(f.y*2.1+s*1.3+id)+0.5*sin(f.z*3.7+s*0.7),
                  sin(f.z*1.9+s*1.1)+0.5*sin(f.x*2.9+id*0.3),
                  sin(f.x*2.3+s*0.9)+0.5*sin(f.y*3.1+id));
    p+=curl*tq*0.09*(0.25+age);

    // Ash: a cooled ember loses the heat that carried it up. From the moment
    // it turns to ash it is pulled back down (against the travel direction,
    // which for embers is up) and flutters side to side like a flake, so it
    // settles on the floor instead of rising forever.
    if (kind()==EMBERS) {
        let t=max(s-ASH_AT*u.motion.y,0.0);
        p-=ax.dir*(0.5*(max(u.physics.y,0.0)+0.35)*t*t + u.motion.w*0.25*t);
        p+=(ax.side*sin(t*4.1+id)+ax.side2*cos(t*3.3+id*1.7))*0.05*t;
    }

    // Collisions with the scene's floor, walls and ceiling. A particle that
    // ends up on the wrong side is put back on the surface (slides along it)
    // or mirrored off it (bounces), by the bounce amount.
    let n_planes=i32(u.style.y);
    var hit=0.0;
    for (var i=0; i<4; i=i+1) {
        if (i >= n_planes) { break; }
        let pl=u.planes[i];
        let d=dot(pl.xyz,p)+pl.w;
        if (d<0.0) { p-=pl.xyz*d*(1.0+u.style.z); hit=1.0; }
    }
    return vec4(p,hit);
}

struct Out{@builtin(position) position:vec4<f32>,@location(0) uv:vec2<f32>,
           @location(1) age:f32, @location(2) depth:f32, @location(3) rnd:f32,
           @location(4) hit:f32};

@vertex fn vs_main(@builtin(vertex_index) vi:u32)->Out{
    let id=f32(vi/6u); let corner=vi%6u;
    var quad=array<vec2<f32>,6>(vec2(-1,-1),vec2(1,-1),vec2(1,1),vec2(-1,-1),vec2(1,1),vec2(-1,1));
    let q=quad[corner]; let phase=hash(id*19.19+u.physics.w);
    let age=fract(u.motion.x/max(u.motion.y,0.01)+phase); let s=age*u.motion.y;
    let at=position_at(id,s);
    let p=at.xyz;
    let kd=kind();
    // A raindrop that has reached the floor becomes a splash lying on it.
    let splash=kd==RAIN && at.w>0.5;

    var size=u.motion.z*(0.35+sin(age*3.14159)*0.9)*(1.0+u.style.w*age);
    if (kd==EMBERS) {
        // Sparks stay small and shrink as they cool; the ash flake left
        // behind is a little bigger than the spark was.
        let ash=smoothstep(COOL_START,ASH_AT,age);
        size=u.motion.z*mix(1.0-0.35*age,1.35,ash);
    } else if (kd==SNOW || kd==DUST) {
        size=u.motion.z*(0.8+0.4*hash(id*2.7));
    } else if (kd==RAIN) {
        size=u.motion.z*(0.8+0.4*hash(id*2.7));
        if (splash) { size=u.motion.z*14.0; }
    }

    // Motion streak: stretch the sprite along its on-screen velocity, the
    // way a camera shutter smears anything fast and bright. Only glowing
    // embers and fire; ash and smoke are too slow to smear.
    var along=vec2(1.0,0.0); var stretch=1.0;
    if ((kd==EMBERS || kd==FIRE || kd==RAIN) && !splash) {
        let dt=1.0/30.0;
        let v=(p-position_at(id,max(s-dt,0.0)).xyz)/dt;
        let sv=vec2(dot(v,u.right.xyz),dot(v,u.up.xyz));
        let speed=length(sv);
        let hot=1.0-smoothstep(COOL_START,ASH_AT*0.8,age);
        if (speed>1e-4) { along=sv/speed; }
        if (kd==FIRE) {
            // Flames are tall whatever their speed: a tongue licking along
            // the direction the fire rises. q.x runs along it, tip at +x.
            stretch=1.9;
        } else if (kd==RAIN) {
            // Rain is only ever seen as streaks: a drop falls several of its
            // own lengths during one exposure.
            stretch=1.0+min(speed*(1.0/40.0)/max(size,1e-4),40.0);
        } else {
            stretch=1.0+min(speed*(1.0/48.0)/max(size,1e-4),6.0)*hot;
        }
    }
    let across=vec2(-along.y,along.x);
    var o=along*q.x*size*stretch+across*q.y*size;
    // A splash lies on the floor, so seen from a normal eye height its
    // ring is a flat ellipse rather than a circle facing the camera.
    if (splash) { o.y*=0.3; }

    let centre_depth = -(dot(u.view_z.xyz, p) + u.view_z.w);
    var out:Out;
    out.position=u.vp*vec4(p+u.right.xyz*o.x+u.up.xyz*o.y,1.0);
    out.uv=q; out.age=age; out.depth=centre_depth; out.rnd=hash(id*1.37+u.physics.w);
    out.hit=select(0.0,1.0,splash);
    // A splash sits exactly on the surface, so half its ring would fail the
    // depth test against that same surface. Judge it as slightly nearer.
    if (splash) { out.depth=centre_depth-0.12; }
    return out;
}

// Returns premultiplied colour. The alpha written is coverage times
// (1 - glow): a glowing particle adds light like fire does and a cool one
// covers like smoke, and one ember can pass from the first to the second.
fn shade(rgb:vec3<f32>, cover:f32, glow:f32)->vec4<f32>{
    let a=clamp(cover,0.0,1.0)*u.style.x;
    return vec4(rgb*a, a*(1.0-glow));
}

// Soft lighting for a round puff: treat the sprite as a sphere facing the
// camera and wrap the key light around it, so smoke has a lit side.
fn puff_light(uv:vec2<f32>, r:f32)->f32{
    let n=vec3(uv.x,uv.y,sqrt(max(1.0-r*r,0.0)));
    let l=normalize(vec3(dot(u.right.xyz,u.light.xyz),dot(u.up.xyz,u.light.xyz),
                         -dot(u.view_z.xyz,u.light.xyz))+vec3(0.0,1e-5,0.0));
    return 0.55+0.45*dot(n,l);
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
    let age=in.age; let t=u.motion.x; let kd=kind();
    let lit=u.ambient_kind.rgb+vec3(0.25);
    let life=smoothstep(0.0,0.04,age)*(1.0-smoothstep(0.85,1.0,age));

    if (kd==EMBERS) {
        // Temperature 1 = white hot, 0 = cold. It falls quickly at first,
        // the way a spark flashes and then dulls.
        let temp=1.0-smoothstep(0.0,ASH_AT,age);
        let hotc=mix(u.colour.rgb,vec3(1.0,0.86,0.6),smoothstep(0.85,1.0,temp)*0.8);
        let dull=u.colour.rgb*vec3(0.55,0.16,0.07);
        let ash_c=vec3(0.16,0.15,0.14)*lit*1.4;
        var c=mix(dull,hotc,smoothstep(0.35,0.8,temp));
        let ash=smoothstep(COOL_START,ASH_AT,age);
        c=mix(c*(1.0+u.colour.w*0.6*temp*temp*temp),ash_c,ash);
        // A spark is a soft glowing core; ash is a jagged flake with a hard
        // edge, spinning. The shape changes with the colour.
        let core=exp(-r*r*5.0);
        let ang=in.rnd*6.2832+t*(1.5+in.rnd*2.0);
        let rot=mat2x2(cos(ang),-sin(ang),sin(ang),cos(ang))*in.uv;
        let edge=0.55+0.35*vnoise(vec2(atan2(rot.y,rot.x)*1.6+in.rnd*40.0,in.rnd*9.0));
        let flake=1.0-smoothstep(edge-0.08,edge,length(rot*vec2(1.0,1.8)));
        let cover=mix(core*1.2,flake*0.9,ash)*life*occl;
        return shade(c,cover,1.0-ash);
    }
    if (kd==FIRE) {
        // A flame tongue along uv.x (the rise direction, tip at +x): wide
        // at the root, narrowing to a point, its edge eaten by noise that
        // scrolls towards the tip. Hottest low in the core. Late in life it
        // burns out into a wisp of dark smoke that covers rather than glows.
        let along01=in.uv.x*0.5+0.5;
        let width=mix(0.95,0.18,along01*along01);
        let n=fbm(vec2(in.uv.x*1.6-t*3.2,in.uv.y*2.4)+vec2(in.rnd*17.0,in.rnd*5.0));
        let shape=length(vec2(in.uv.x*0.9,in.uv.y/width));
        let flame=1.0-smoothstep(0.35,1.0,shape+(n-0.5)*0.55);
        let heat=flame*(1.0-along01*0.6)*(1.0-smoothstep(0.1,0.7,age));
        var c=mix(u.colour.rgb*vec3(0.75,0.3,0.12),u.colour.rgb,smoothstep(0.1,0.45,heat));
        c=mix(c,vec3(1.0,0.82,0.5),smoothstep(0.62,0.95,heat)*0.85);
        c=c*(1.0+u.colour.w*0.35*heat*heat);
        let smoke=smoothstep(0.45,0.85,age);
        let smoke_c=vec3(0.12,0.11,0.1)*lit;
        let cover=mix(flame*0.75,flame*0.3,smoke)*life*occl;
        return shade(mix(c,smoke_c,smoke),cover,1.0-smoke);
    }
    if (kd==RAIN) {
        if (in.hit>0.5) {
            // Splash: a thin ring spreading out and fading, restarting now
            // and then so a wet floor keeps rippling while rain lands on it.
            // Only some landed drops ripple at a time, or the floor turns
            // into a pattern of rings.
            if (in.rnd>0.45) { discard; }
            let ring_t=fract(t*1.3+in.rnd*7.0);
            let ring=1.0-smoothstep(0.0,0.1,abs(r-ring_t));
            let flat_ring=ring*(1.0-ring_t)*(1.0-ring_t)*0.8;
            return shade(u.colour.rgb*(lit+vec3(0.3)),flat_ring*occl,0.2);
        }
        // A streak: thin across, fading at both ends, with a faint sheen
        // where it catches light.
        let across=1.0-smoothstep(0.15,1.0,abs(in.uv.y));
        let ends=1.0-smoothstep(0.4,1.0,abs(in.uv.x));
        let c=u.colour.rgb*(lit+vec3(0.35+u.colour.w));
        return shade(c,across*ends*0.45*life*occl,0.25);
    }
    if (kd==SNOW) {
        let flake=1.0-smoothstep(0.45,0.7,r);
        let glint=0.85+0.15*sin(t*6.0+in.rnd*40.0);
        return shade(u.colour.rgb*lit*glint*(1.0+u.colour.w),flake*life*occl,0.0);
    }
    if (kd==DUST) {
        // Motes that catch the light now and then.
        let mote=exp(-r*r*6.0);
        let glint=0.55+0.45*pow(0.5+0.5*sin(t*2.3+in.rnd*60.0),6.0);
        return shade(u.colour.rgb*(lit+vec3(0.35))*(glint+0.35+u.colour.w),mote*life*occl,glint*0.3);
    }
    // Smoke and cloud puffs: billowing noise inside a soft round edge,
    // turning slowly, lit from the key light's side.
    let ang=in.rnd*6.2832+age*(in.rnd-0.5)*2.0;
    let rot=mat2x2(cos(ang),-sin(ang),sin(ang),cos(ang))*in.uv;
    let n=fbm(rot*1.8+vec2(in.rnd*23.0,age*1.5));
    let body=(1.0-smoothstep(0.25,1.0,r+(0.5-n)*0.45));
    var dens=0.72;
    if (kd==CLOUDS) { dens=0.6; }
    let c=u.colour.rgb*(lit*puff_light(in.uv,r)+vec3(u.colour.w));
    return shade(c,body*dens*sin(age*3.14159)*occl,0.0);
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

        def pipe(code: str, cull, colour_src: str, buffers=()):
            mod = device.create_shader_module(code=code)
            return device.create_render_pipeline(
                layout=pl,
                vertex={"module": mod, "entry_point": "vs_main", "buffers": list(buffers)},
                primitive={"topology": wgpu.PrimitiveTopology.triangle_list, "cull_mode": cull},
                fragment={"module": mod, "entry_point": "fs_main", "targets": [{
                    "format": colour_format,
                    "blend": {"color": {"src_factor": colour_src,
                                        "dst_factor": "one-minus-src-alpha", "operation": "add"},
                              "alpha": {"src_factor": "one",
                                        "dst_factor": "one-minus-src-alpha", "operation": "add"}}}]})

        # Both return premultiplied colour. Particles also lower their alpha
        # as they glow, which turns the same blend additive for fire and hot
        # embers while smoke and ash still cover what is behind them.
        self._volume = pipe(VOLUME_SHADER, wgpu.CullMode.front, "one")
        self._particle = pipe(PARTICLE_SHADER, wgpu.CullMode.none, "one")
        self._bolt = pipe(BOLT_SHADER, wgpu.CullMode.none, "one", [{
            "array_stride": 20, "step_mode": wgpu.VertexStepMode.vertex,
            "attributes": [
                {"format": wgpu.VertexFormat.float32x3, "offset": 0, "shader_location": 0},
                {"format": wgpu.VertexFormat.float32, "offset": 12, "shader_location": 1},
                {"format": wgpu.VertexFormat.float32, "offset": 16, "shader_location": 2}]}])
        self._pool: dict[str, list] = {"v": [], "p": [], "b": []}

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
        if effects is None or not (effects.volumes or effects.emitters
                                   or getattr(effects, "strikes", [])):
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
            v = np.zeros(84, np.float32)
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
            v[80:83] = key_dir
            rp.set_bind_group(0, self._group("p", slot, v, depth_view))
            slot += 1
            rp.draw(min(int(em.count), MAX_PARTICLES) * 6, 1, 0, 0)
        rp.set_pipeline(self._bolt)
        slot = 0
        for item in getattr(effects, "strikes", []):
            if not item.enabled or not item.show_bolt:
                continue
            brightness, index = strike_at(item, time_seconds)
            if brightness < 0.02:
                continue
            verts = bolt_vertices(item, index, camera_pos, brightness)
            if not len(verts):
                continue
            v = np.zeros(24, np.float32)
            v[:16] = np.ascontiguousarray(vp.T).ravel()
            v[16:19] = item.colour
            v[19] = item.emission
            v[20:24] = view[2]
            buf = self.device.create_buffer_with_data(
                data=verts.tobytes(), usage=wgpu.BufferUsage.VERTEX)
            rp.set_bind_group(0, self._group("b", slot, v, depth_view))
            rp.set_vertex_buffer(0, buf)
            slot += 1
            rp.draw(len(verts), 1, 0, 0)
        rp.end()
