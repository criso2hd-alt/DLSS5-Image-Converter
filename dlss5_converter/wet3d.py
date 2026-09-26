"""Wet surfaces and puddles for the splat scene, as one full-screen pass.

It runs after the splats and before particles, over the colour and depth the
splat pass wrote, so rain and splashes are drawn on top of the wet ground.

From the depth it rebuilds each pixel's position and surface normal, which
says how much the surface faces up (ground) versus sideways (walls). Then:

- Wet: surfaces darken and their colour deepens, strongest on the ground.
- Shine: ground reflects the scene through screen-space reflections, with a
  Fresnel falloff (strongest at grazing angles, as real wet asphalt is).
- Puddles: noise patches on the ground become near-mirrors, darker still.
- Ripples: rain rings disturb the normal inside puddles and on wet ground.

Screen-space reflections can only show what is already in frame; a ray
that leaves the frame or finds nothing simply gives no reflection, fading
out near the edges rather than cutting off. This is the same limit games
accept for the same effect.
"""

from __future__ import annotations

import numpy as np

WET_SHADER = """
struct WU {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    size: vec4<f32>,     // x width, y height, z time
    amount: vec4<f32>,   // x wetness, y puddles, z ripples, w puddle size
    up: vec4<f32>,       // xyz world up, w 1 = mirror fallback in puddles
    snow: vec4<f32>,     // x snow cover 0..1 at this moment
};
@group(0) @binding(0) var<uniform> u: WU;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dep: texture_2d<f32>;

@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    return vec4(p[i], 0.0, 1.0);
}

// Integer hash (PCG): exact on every GPU. The old fract(sin(x) * 43758.5)
// runs out of float precision for larger x, which made neighbouring cells and
// particle ids get near-identical "random" values: square blotches in snow
// and puddles, particles marching in lines.
fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
fn u01(v: u32) -> f32 { return f32(v >> 8u) * (1.0 / 16777216.0); }
fn h2(p:vec2<f32>)->f32{
    let i = vec2<i32>(floor(p));
    return u01(pcg(bitcast<u32>(i.x) ^ pcg(bitcast<u32>(i.y) + 0x9E3779B9u)));
}
// Gradient (Perlin-style) noise, 0..1. Value noise on a square grid shows the
// grid wherever it is thresholded into shapes; gradient noise has no visible
// cells.
fn grad2(i:vec2<f32>, f:vec2<f32>)->f32{
    let a = h2(i) * 6.2831853;
    return dot(vec2(cos(a), sin(a)), f);
}
fn vnoise(p:vec2<f32>)->f32{
    let i=floor(p); let f=fract(p);
    let w=f*f*f*(f*(f*6.0-15.0)+10.0);
    let n=mix(mix(grad2(i,f),grad2(i+vec2(1.0,0.0),f-vec2(1.0,0.0)),w.x),
              mix(grad2(i+vec2(0.0,1.0),f-vec2(0.0,1.0)),grad2(i+vec2(1.0,1.0),f-vec2(1.0,1.0)),w.x),w.y);
    return clamp(0.5+n*0.95,0.0,1.0);
}
// Each octave turned against the last, so no layer lines up with the floor.
fn fbm(p:vec2<f32>)->f32{
    var a=0.5; var s=0.0; var q=p;
    let r=mat2x2<f32>(0.8,0.6,-0.6,0.8);
    for (var i=0; i<4; i=i+1) { s+=a*vnoise(q); q=r*q*2.03+vec2(1.7,9.2); a*=0.5; }
    return s;
}

fn dims()->vec2<i32>{ return vec2<i32>(i32(u.size.x), i32(u.size.y)); }

// View distance at a pixel, or -1 where nothing was drawn (open sky).
fn dist_at(c:vec2<i32>)->f32{
    let d=textureLoad(dep, clamp(c, vec2<i32>(0), dims()-1), 0);
    if (d.a<0.05) { return -1.0; }
    return d.r/d.a;
}

fn view_pos(c:vec2<i32>, z:f32)->vec3<f32>{
    let ndc=vec2((f32(c.x)+0.5)/u.size.x*2.0-1.0, 1.0-(f32(c.y)+0.5)/u.size.y*2.0);
    return vec3(ndc.x*z/u.proj[0][0], ndc.y*z/u.proj[1][1], -z);
}

// Neighbour for the normal: the nearer-depth side, so a silhouette edge does
// not blend foreground and background into a nonsense slope.
fn side(c:vec2<i32>, step:vec2<i32>, z:f32, pc:vec3<f32>)->vec3<f32>{
    let za=dist_at(c+step); let zb=dist_at(c-step);
    let da=select(1e9, abs(za-z), za>0.0); let db=select(1e9, abs(zb-z), zb>0.0);
    if (da<=db) { return view_pos(c+step, za)-pc; }
    return pc-view_pos(c-step, zb);
}

@fragment fn fs_main(@builtin(position) fp: vec4<f32>) -> @location(0) vec4<f32> {
    let c=vec2<i32>(fp.xy);
    let here=textureLoad(src, c, 0);
    let z=dist_at(c);
    if (z<0.0 || here.a<0.02) { return here; }
    var rgb=here.rgb/max(here.a,1e-4);
    let pc=view_pos(c,z);
    // Normal over a 3-pixel baseline: one splat per pixel makes adjacent
    // depths noisy enough to flip a 1-pixel normal (speckle in the shine).
    var n=normalize(cross(side(c,vec2(3,0),z,pc), side(c,vec2(0,3),z,pc)));
    if (dot(n,-pc)<0.0) { n=-n; }
    // A wider-baseline normal too: splat depth jitters pixel to pixel, and a
    // 3-pixel normal alone makes reflections sparkle as the camera moves.
    var n8=normalize(cross(side(c,vec2(8,0),z,pc), side(c,vec2(0,8),z,pc)));
    if (dot(n8,-pc)<0.0) { n8=-n8; }
    var n16=normalize(cross(side(c,vec2(16,0),z,pc), side(c,vec2(0,16),z,pc)));
    if (dot(n16,-pc)<0.0) { n16=-n16; }
    let n_steady=normalize(n+n8*2.0+n16*2.0);
    let up_v=normalize((u.view*vec4(u.up.xyz,0.0)).xyz);
    let facing=dot(n,up_v);
    // Ground is judged on the steady normal: with the 3-pixel one, pixels
    // flip between wet-dark and dry as the camera moves (shimmer).
    let ground=smoothstep(0.5,0.85,dot(n_steady,up_v));

    // Coordinates on the ground plane, for puddle shapes and ripple cells.
    let world=(u.inv_view*vec4(pc,1.0)).xyz;
    let upw=normalize(u.up.xyz);
    var t1=normalize(cross(upw, vec3(0.0,0.0,1.0)));
    if (abs(upw.z)>0.9) { t1=normalize(cross(upw, vec3(1.0,0.0,0.0))); }
    let t2=cross(upw,t1);
    let gp=vec2(dot(world,t1), dot(world,t2));

    let wet=u.amount.x;
    let puddle_n=fbm(gp*(1.6/max(u.amount.w,0.05))+vec2(3.1,7.7));
    let puddle=ground*smoothstep(1.0-u.amount.y*0.75-0.1, 1.0-u.amount.y*0.75+0.1, puddle_n)
               *step(0.001,u.amount.y);
    let dampness=max(wet*(0.25+0.75*ground), puddle);

    // Wet colour: darker and more saturated, as water fills the surface.
    let lum=dot(rgb,vec3(0.299,0.587,0.114));
    rgb=mix(vec3(lum),rgb,1.0+0.35*dampness);
    rgb=rgb*(1.0-0.38*dampness)*(1.0-0.3*puddle);

    // Rain ripples: expanding rings, one per cell, perturbing the normal.
    let cell=gp/0.22;
    let ci=floor(cell); let cf=fract(cell)-0.5;
    let ph=h2(ci);
    let ring_t=fract(u.size.z*1.3+ph);
    // Kept inside its cell (offset + radius < 0.5), or the ring is cut off
    // at the cell edge and the grid shows as straight lines.
    let off=vec2(h2(ci+17.0)-0.5, h2(ci+31.0)-0.5)*0.3;
    let rv=cf-off;
    let rd=length(rv);
    let ring=exp(-pow((rd-ring_t*0.3)/0.03,2.0))*(1.0-ring_t)*u.amount.z;
    let rdir=rv/max(rd,1e-4);
    // A little micro-roughness away from puddles, so wet ground's reflection
    // is soft and a puddle's is sharp.
    // Faded out with distance: past a few metres the grain is finer than a
    // pixel and would only shimmer.
    let rough=(1.0-puddle)*0.035*clamp(1.0-(z-2.0)/6.0,0.0,1.0);
    let jitter=vec2(vnoise(gp*14.0)-0.5, vnoise(gp*14.0+9.0)-0.5)*rough;
    let bend=(rdir*ring*0.5+jitter);
    let tv1=normalize((u.view*vec4(t1,0.0)).xyz); let tv2=normalize((u.view*vec4(t2,0.0)).xyz);
    let n2=normalize(n_steady+(tv1*bend.x+tv2*bend.y)*ground);

    let v=normalize(pc);
    let r=reflect(v,n2);
    let cosv=clamp(dot(-v,n2),0.0,1.0);
    let fresnel=0.04+0.96*pow(1.0-cosv,5.0);
    let strength=clamp(fresnel*3.0,0.12,1.0)*(ground*wet*0.5+puddle*0.95);

    var refl=vec3(0.0); var got=0.0;
    // The last on-screen point the ray passed over. A ray that finds no
    // exact hit still takes that colour, weaker, so neighbouring pixels do
    // not flip between reflection and none (which reads as speckle).
    var last=vec3(0.0); var last_w=0.0;
    if (strength>0.01) {
        var t=0.04;
        var prev_t=0.0;
        for (var i=0; i<48; i=i+1) {
            let p=pc+r*t;
            if (-p.z<0.05) { break; }
            let clip=u.proj*vec4(p,1.0);
            if (clip.w<=0.0) { break; }
            let ndc=clip.xy/clip.w;
            let px=vec2((ndc.x*0.5+0.5)*u.size.x, (0.5-ndc.y*0.5)*u.size.y);
            if (px.x<0.0 || px.y<0.0 || px.x>=u.size.x || px.y>=u.size.y) { break; }
            let sd=dist_at(vec2<i32>(px));
            let pz=-p.z;
            if (sd>0.0) {
                let ls=textureLoad(src, vec2<i32>(px), 0);
                last=ls.rgb/max(ls.a,1e-4);
                let edge=min(min(px.x,u.size.x-px.x), min(px.y,u.size.y-px.y));
                last_w=clamp(edge/(0.06*u.size.y),0.0,1.0)*ls.a*0.25;
            }
            if (sd>0.0 && pz>sd+0.005 && pz-sd<max(0.12,t*0.25)) {
                // Refine between the previous step and this one, so the hit
                // lands where the ray actually crosses the surface; without
                // it the step size shows as banding across the floor.
                var lo=prev_t; var hi=t; var hp=px;
                for (var k=0; k<5; k=k+1) {
                    let mid=(lo+hi)*0.5;
                    let q=pc+r*mid;
                    let qc=u.proj*vec4(q,1.0);
                    let qn=qc.xy/qc.w;
                    let qp=vec2((qn.x*0.5+0.5)*u.size.x, (0.5-qn.y*0.5)*u.size.y);
                    let qd=dist_at(vec2<i32>(qp));
                    if (qd>0.0 && -q.z>qd) { hi=mid; hp=qp; } else { lo=mid; }
                }
                // A small average around the hit, not one pixel: a single
                // pixel flips colour with every tiny camera move (shimmer).
                // Wider on damp ground, tight in puddles so they stay sharp.
                let blur=mix(9.0,1.0,puddle);
                var acc=vec3(0.0); var wsum=0.0;
                for (var k=0; k<9; k=k+1) {
                    let o=vec2(f32(k%3)-1.0, f32(k/3)-1.0)*blur;
                    let tap=clamp(hp+o, vec2(0.0), u.size.xy-1.0);
                    let ts=textureLoad(src, vec2<i32>(tap), 0);
                    acc+=ts.rgb/max(ts.a,1e-4)*ts.a; wsum+=ts.a;
                }
                let s=textureLoad(src, vec2<i32>(hp), 0);
                refl=acc/max(wsum,1e-4);
                // Fade near the frame edges, where the reflection would
                // otherwise stop at a hard line.
                let edge=min(min(px.x,u.size.x-px.x), min(px.y,u.size.y-px.y));
                got=clamp(edge/(0.06*u.size.y),0.0,1.0)*s.a;
                break;
            }
            prev_t=t;
            t=t*1.1+0.012;
        }
    }
    if (got<=0.0 && puddle>0.01 && u.up.w>0.5) {
        // Mirror fallback: flip the image across the horizon, the classic fake
        // reflection. It shows what is above the puddle even when the true ray
        // leaves the frame (sky, rooftops), which real photos do constantly.
        let fwd=vec3(0.0,0.0,-1.0);
        let along=normalize(fwd-up_v*dot(fwd,up_v)+vec3(0.0,1e-5,0.0));
        let hc=u.proj*vec4(along*1000.0,1.0);
        let horizon=(0.5-(hc.y/hc.w)*0.5)*u.size.y;
        let mx=clamp(fp.x+bend.x*40.0, 0.0, u.size.x-1.0);
        let my=2.0*horizon-fp.y;
        // Mirrored past the top of the frame: keep using the top rows (the
        // classic trick) rather than fading, or low puddles stay empty.
        let top_fade=mix(0.7,1.0,clamp((my+0.1*u.size.y)/(0.1*u.size.y),0.0,1.0));
        let mc=vec2<i32>(i32(mx), i32(clamp(my,2.0,u.size.y-1.0)));
        let ms=textureLoad(src, mc, 0);
        if (ms.a>0.05 && my<fp.y) {
            refl=ms.rgb/max(ms.a,1e-4);
            got=puddle*top_fade*ms.a;
        }
    }
    if (got<=0.0) { refl=last; got=last_w; }
    rgb=mix(rgb, refl, strength*got);

    // Snow: settles on whatever faces up, flattest first, broken up by noise
    // so it gathers in patches before it closes into a blanket. It keeps the
    // scene's own light and shadow (a snowy floor in shade is grey, not glowing)
    // and hides the wet shine underneath.
    let snow=u.snow.x;
    if (snow>0.001) {
        // Faces up at two scales, or it is a noisy normal on a wall: a
        // single 3-pixel normal flickers upward often enough to speckle walls.
        let facing_snow=min(facing, dot(n8,up_v));
        let flat_up=smoothstep(0.3,0.85,facing_snow);
        let breakup=fbm(gp*2.3+vec2(11.0,5.0))-0.5+(vnoise(gp*11.0)-0.5)*0.35;
        let cover=smoothstep(0.0,1.0,clamp((flat_up*1.25-(1.0-snow)*1.15+breakup*0.7)*2.2,0.0,1.0))*smoothstep(0.15,0.3,facing_snow);
        let lum_here=dot(here.rgb/max(here.a,1e-4),vec3(0.299,0.587,0.114));
        let shade=clamp(0.3+lum_here*1.7,0.25,1.0);
        let glint=step(0.985,h2(floor(gp*90.0)))*0.25*shade;
        let snow_rgb=vec3(0.90,0.93,0.98)*shade+vec3(glint);
        rgb=mix(rgb,snow_rgb,cover);
    }
    return vec4(rgb*here.a, here.a);
}
"""


def active(effects) -> bool:
    lighting = getattr(effects, "lighting", None) if effects is not None else None
    return lighting is not None and (getattr(lighting, "wetness", 0.0) > 0.0
                                     or getattr(lighting, "puddles", 0.0) > 0.0
                                     or getattr(lighting, "snow_cover", 0.0) > 0.0)


class WetPass:
    """Owns the pipeline and a copy of the colour target to read from."""

    def __init__(self, device) -> None:
        import wgpu
        self._wgpu = wgpu
        self.device = device
        mod = device.create_shader_module(code=WET_SHADER)
        tex = {"sample_type": wgpu.TextureSampleType.unfilterable_float}
        self._layout = device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": tex},
            {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": tex}])
        self._pipe = device.create_render_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._layout]),
            vertex={"module": mod, "entry_point": "vs_main", "buffers": []},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            fragment={"module": mod, "entry_point": "fs_main",
                      "targets": [{"format": "rgba16float"}]})
        self._uniform = device.create_buffer(
            size=512, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._copy = None
        self._size = None

    def encode(self, enc, colour_tex, depth_view, view: np.ndarray, proj: np.ndarray,
               size, effects, time_seconds: float, up=(0.0, 1.0, 0.0)) -> None:
        wgpu = self._wgpu
        w, h = int(size[0]), int(size[1])
        if self._size != (w, h):
            self._copy = self.device.create_texture(
                size=(w, h, 1), format="rgba16float",
                usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING)
            self._size = (w, h)
        # The pass reads the splats and writes the wet result over them, and a
        # texture cannot be both, so it reads from a copy.
        enc.copy_texture_to_texture({"texture": colour_tex}, {"texture": self._copy}, (w, h, 1))
        lighting = effects.lighting
        v = np.zeros(68, np.float32)
        v[0:16] = np.ascontiguousarray(view.T, np.float32).ravel()
        v[16:32] = np.ascontiguousarray(proj.T, np.float32).ravel()
        v[32:48] = np.ascontiguousarray(np.linalg.inv(view).T, np.float32).ravel()
        v[48:51] = (w, h, time_seconds)
        v[52:56] = (float(lighting.wetness), float(lighting.puddles),
                    float(lighting.ripples), float(lighting.puddle_size))
        v[56:59] = up
        v[59] = 1.0 if getattr(lighting, "puddle_mirror", True) else 0.0
        from .effects3d import snow_amount
        v[60] = snow_amount(lighting, time_seconds)
        self.device.queue.write_buffer(self._uniform, 0, v.tobytes())
        bind = self.device.create_bind_group(layout=self._layout, entries=[
            {"binding": 0, "resource": {"buffer": self._uniform}},
            {"binding": 1, "resource": self._copy.create_view()},
            {"binding": 2, "resource": depth_view}])
        rp = enc.begin_render_pass(color_attachments=[{
            "view": colour_tex.create_view(), "load_op": wgpu.LoadOp.load,
            "store_op": wgpu.StoreOp.store}])
        rp.set_pipeline(self._pipe)
        rp.set_bind_group(0, bind)
        rp.draw(3, 1, 0, 0)
        rp.end()
