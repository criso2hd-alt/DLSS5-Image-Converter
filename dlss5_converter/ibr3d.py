"""Renders a ShotScene: the best few screenshots, blended per pixel, on the GPU.

Per frame:
1. The PER_FRAME shots nearest the view are drawn, each as a textured depth
   mesh (its own pixels at the game's depth), into its own colour + distance
   target.
2. One full-screen pass blends them per pixel: only layers that see the
   nearest surface there take part (a depth test across layers), weighted by
   how close each shot's viewing direction is to the camera's, by trust, and
   by a soft fade near each shot's frame edge. Things that moved between shots
   come from the nearest shot only.
3. The blend writes the same colour and depth targets the splat renderer does,
   so wet ground, particles, lightning and the finish pass work unchanged.

Same interface as `splat3d.SplatRenderer.render_u8`.
"""

from __future__ import annotations

import numpy as np

from .splat3d import FINISH_SHADER

PER_FRAME = 4
RESIDENT = 8           # shots kept on the GPU; the rest are rebuilt on demand

MESH_SHADER = """
struct U { view: mat4x4<f32>, proj: mat4x4<f32>, flags: vec4<f32> };   // flags.x: 1 = owner of moved things
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32>,
              @location(1) feather: f32, @location(2) moved: f32, @location(3) z: f32 };
@vertex fn vs_main(@location(0) pos: vec3<f32>, @location(1) uv: vec2<f32>,
                   @location(2) extra: vec2<f32>) -> VOut {
    var o: VOut;
    let v = u.view * vec4<f32>(pos, 1.0);
    o.clip = u.proj * v;
    o.uv = uv; o.feather = extra.x; o.moved = extra.y; o.z = -v.z;
    return o;
}
struct FOut { @location(0) colour: vec4<f32>, @location(1) dist: vec4<f32> };
@fragment fn fs_main(i: VOut) -> FOut {
    // Something that moved between shots is taken from one shot only; in the
    // others it is a hole the owner fills.
    if (i.moved > 0.5 && u.flags.x < 0.5) { discard; }
    var o: FOut;
    o.colour = vec4<f32>(textureSampleLevel(tex, samp, i.uv, 0.0).rgb, i.feather);
    o.dist = vec4<f32>(i.z, 0.0, 0.0, 1.0);
    return o;
}
"""

BLEND_SHADER = """
struct B { inv_view: mat4x4<f32>, proj: vec4<f32>,      // proj: x p00, y p11, z w, w h
           eye: vec4<f32>, shots: array<vec4<f32>, 4> }; // shots[k]: eye.xyz, trust (0 = unused slot)
@group(0) @binding(0) var<uniform> b: B;
@group(0) @binding(1) var c0: texture_2d<f32>;
@group(0) @binding(2) var c1: texture_2d<f32>;
@group(0) @binding(3) var c2: texture_2d<f32>;
@group(0) @binding(4) var c3: texture_2d<f32>;
@group(0) @binding(5) var d0: texture_2d<f32>;
@group(0) @binding(6) var d1: texture_2d<f32>;
@group(0) @binding(7) var d2: texture_2d<f32>;
@group(0) @binding(8) var d3: texture_2d<f32>;
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    return vec4(p[i], 0.0, 1.0);
}
struct FOut { @location(0) colour: vec4<f32>, @location(1) depth: vec4<f32> };
fn sample_c(k: i32, c: vec2<i32>) -> vec4<f32> {
    if (k == 0) { return textureLoad(c0, c, 0); }
    if (k == 1) { return textureLoad(c1, c, 0); }
    if (k == 2) { return textureLoad(c2, c, 0); }
    return textureLoad(c3, c, 0);
}
fn sample_d(k: i32, c: vec2<i32>) -> vec4<f32> {
    if (k == 0) { return textureLoad(d0, c, 0); }
    if (k == 1) { return textureLoad(d1, c, 0); }
    if (k == 2) { return textureLoad(d2, c, 0); }
    return textureLoad(d3, c, 0);
}
@fragment fn fs_main(@builtin(position) pos: vec4<f32>) -> FOut {
    let c = vec2<i32>(pos.xy);
    var nearest = 1e30;
    for (var k = 0; k < 4; k++) {
        let d = sample_d(k, c);
        if (b.shots[k].w > 0.0 && d.a > 0.5) { nearest = min(nearest, d.r); }
    }
    var o: FOut;
    if (nearest > 1e29) {
        o.colour = vec4<f32>(0.0); o.depth = vec4<f32>(0.0);
        return o;
    }
    // This pixel's surface point in the world.
    let ndc = vec2<f32>((pos.x / b.proj.z) * 2.0 - 1.0, 1.0 - (pos.y / b.proj.w) * 2.0);
    let vp = vec3<f32>(ndc.x * nearest / b.proj.x, ndc.y * nearest / b.proj.y, -nearest);
    let world = (b.inv_view * vec4<f32>(vp, 1.0)).xyz;
    let to_cam = normalize(b.eye.xyz - world);
    var rgb = vec3<f32>(0.0);
    var total = 0.0;
    for (var k = 0; k < 4; k++) {
        let d = sample_d(k, c);
        if (b.shots[k].w <= 0.0 || d.a < 0.5 || d.r > nearest * 1.015 + 0.005) { continue; }
        let s = sample_c(k, c);
        let to_shot = normalize(b.shots[k].xyz - world);
        let angle = acos(clamp(dot(to_shot, to_cam), -1.0, 1.0));
        // Closest viewing direction wins, smoothly, so sources hand over
        // gradually as the camera moves. Feather^6: a shot's frame edge
        // barely counts while another shot sees the same spot properly.
        let w = (exp(-pow(angle / 0.1047, 2.0)) + 1e-4) * b.shots[k].w * pow(max(s.a, 0.02), 6.0);
        rgb += s.rgb * w;
        total += w;
    }
    o.colour = vec4<f32>(rgb / max(total, 1e-9), 1.0);
    o.depth = vec4<f32>(nearest, 0.0, 0.0, 1.0);
    return o;
}
"""


class _Layer:
    __slots__ = ("vbuf", "ibuf", "count", "texture", "bind", "uniform", "eye", "trust")


class IBRRenderer:
    """Holds a wgpu device and renders a ShotScene."""

    def __init__(self, device=None) -> None:
        import wgpu
        self._wgpu = wgpu
        if device is None:
            from . import gpus
            adapter = gpus.wgpu_adapter()
            limits = adapter.limits
            device = adapter.request_device_sync(required_limits={
                "max-storage-buffer-binding-size": limits["max-storage-buffer-binding-size"],
                "max-buffer-size": limits["max-buffer-size"]})
            self.adapter = adapter
        self.device = device
        mesh = device.create_shader_module(code=MESH_SHADER)
        self._mesh_layout = device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {"sample_type": "float"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {"type": "filtering"}}])
        self._mesh_pipe = device.create_render_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._mesh_layout]),
            vertex={"module": mesh, "entry_point": "vs_main", "buffers": [{
                "array_stride": 7 * 4, "step_mode": "vertex", "attributes": [
                    {"format": "float32x3", "offset": 0, "shader_location": 0},
                    {"format": "float32x2", "offset": 12, "shader_location": 1},
                    {"format": "float32x2", "offset": 20, "shader_location": 2}]}]},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list, "cull_mode": wgpu.CullMode.none},
            depth_stencil={"format": "depth32float", "depth_write_enabled": True, "depth_compare": "less"},
            fragment={"module": mesh, "entry_point": "fs_main", "targets": [
                {"format": "rgba16float"}, {"format": "rgba16float"}]})
        self._sampler = device.create_sampler(mag_filter="linear", min_filter="linear")
        blend = device.create_shader_module(code=BLEND_SHADER)
        tex = {"sample_type": wgpu.TextureSampleType.unfilterable_float}
        self._blend_layout = device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": wgpu.BufferBindingType.uniform}}] + [
            {"binding": i, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": tex} for i in range(1, 9)])
        self._blend_pipe = device.create_render_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._blend_layout]),
            vertex={"module": blend, "entry_point": "vs_main", "buffers": []},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            fragment={"module": blend, "entry_point": "fs_main", "targets": [
                {"format": "rgba16float"}, {"format": "rgba16float"}]})
        self._blend_uniform = device.create_buffer(
            size=(16 + 4 + 4 + 16) * 4, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        from .fx3d import FxPass
        self._fx = FxPass(device)
        fin = device.create_shader_module(code=FINISH_SHADER)
        self._fin_layout = device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": tex},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": wgpu.BufferBindingType.uniform}}])
        self._fin_pipe = device.create_render_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._fin_layout]),
            vertex={"module": fin, "entry_point": "vs_main", "buffers": []},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            fragment={"module": fin, "entry_point": "fs_main", "targets": [{"format": "rgba8unorm"}]})
        self._fin_uniform = device.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._targets = None
        self._scene = None
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(max_workers=2)
        self._pending: dict = {}           # shot -> future of its CPU-side layer data
        self._layers: dict[int, _Layer] = {}
        self._lru: list[int] = []
        self._planes: list = []

    # --- scene ------------------------------------------------------------------
    def set_scene(self, scene) -> None:
        self._scene = scene
        self._layers.clear()
        self._lru.clear()
        self._pending.clear()
        self._planes = list(scene.planes or [])

    def _layer(self, k: int) -> _Layer:
        wgpu = self._wgpu
        if k in self._layers:
            self._lru.remove(k)
            self._lru.append(k)
            return self._layers[k]
        if len(self._lru) >= RESIDENT:
            self._layers.pop(self._lru.pop(0))
        future = self._pending.pop(k, None)
        data = future.result() if future is not None else self._scene.layer(k)
        w, h = data["size"]
        verts = np.concatenate([data["positions"], data["uv"], data["feather"][:, None],
                                data["moved"][:, None]], 1).astype(np.float32)
        L = _Layer()
        L.vbuf = self.device.create_buffer_with_data(data=verts.tobytes(), usage=wgpu.BufferUsage.VERTEX)
        L.ibuf = self.device.create_buffer_with_data(data=data["indices"].ravel().tobytes(),
                                                     usage=wgpu.BufferUsage.INDEX)
        L.count = int(data["indices"].size)
        L.texture = self.device.create_texture(
            size=(w, h, 1), format="rgba8unorm",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self.device.queue.write_texture({"texture": L.texture}, np.ascontiguousarray(data["colour"]).tobytes(),
                                        {"bytes_per_row": w * 4, "rows_per_image": h}, (w, h, 1))
        L.uniform = self.device.create_buffer(size=36 * 4, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        L.bind = self.device.create_bind_group(layout=self._mesh_layout, entries=[
            {"binding": 0, "resource": {"buffer": L.uniform}},
            {"binding": 1, "resource": L.texture.create_view()},
            {"binding": 2, "resource": self._sampler}])
        L.eye = self._scene.eye(k)
        L.trust = float(self._scene.trust[k])
        self._layers[k] = L
        self._lru.append(k)
        return L

    def _prefetch(self, eye, fwd) -> None:
        """Prepare the shots the camera is heading toward on a worker thread,
        so they are ready before they are needed (reading and meshing a shot
        takes a moment; uploading it is quick)."""
        if self._scene is None:
            return
        for k in self._scene.best_shots(eye, fwd, RESIDENT):
            if k not in self._layers and k not in self._pending and len(self._pending) < 4:
                self._pending[k] = self._pool.submit(self._scene.layer, k)

    # --- targets ----------------------------------------------------------------
    def _ensure_targets(self, w: int, h: int) -> None:
        if self._targets == (w, h):
            return
        wgpu = self._wgpu
        usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_SRC
        mk = lambda fmt: self.device.create_texture(size=(w, h, 1), format=fmt, usage=usage)  # noqa: E731
        self._lc = [mk("rgba16float") for _ in range(PER_FRAME)]
        self._ld = [mk("rgba16float") for _ in range(PER_FRAME)]
        self._zbuf = self.device.create_texture(size=(w, h, 1), format="depth32float",
                                                usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
        self._color = mk("rgba16float")
        self._depth = mk("rgba16float")
        self._final = self.device.create_texture(size=(w, h, 1), format="rgba8unorm",
                                                 usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
        self._blend_bind = self.device.create_bind_group(layout=self._blend_layout, entries=[
            {"binding": 0, "resource": {"buffer": self._blend_uniform}}] + [
            {"binding": 1 + i, "resource": t.create_view()} for i, t in enumerate(self._lc + self._ld)])
        self._fin_bind = self.device.create_bind_group(layout=self._fin_layout, entries=[
            {"binding": 0, "resource": self._color.create_view()},
            {"binding": 1, "resource": {"buffer": self._fin_uniform}}])
        self._targets = (w, h)

    # --- drawing ----------------------------------------------------------------
    def _draw(self, view, proj, size, effects=None, time_seconds: float = 0.0) -> None:
        wgpu = self._wgpu
        w, h = int(size[0]), int(size[1])
        self._ensure_targets(w, h)
        inv_view = np.linalg.inv(view)
        eye = inv_view[:3, 3]
        fwd = -inv_view[:3, 2]
        shots = self._scene.best_shots(eye, fwd, PER_FRAME) if self._scene is not None else []
        owner = shots[0] if shots else -1
        self._prefetch(eye, fwd)
        enc = self.device.create_command_encoder()
        slots = np.zeros((4, 4), np.float32)
        for i in range(PER_FRAME):
            clear = [{"view": self._lc[i].create_view(), "clear_value": (0, 0, 0, 0),
                      "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store},
                     {"view": self._ld[i].create_view(), "clear_value": (0, 0, 0, 0),
                      "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store}]
            rp = enc.begin_render_pass(color_attachments=clear, depth_stencil_attachment={
                "view": self._zbuf.create_view(), "depth_clear_value": 1.0,
                "depth_load_op": wgpu.LoadOp.clear, "depth_store_op": wgpu.StoreOp.store})
            if i < len(shots):
                L = self._layer(shots[i])
                u = np.zeros(36, np.float32)
                u[0:16] = np.ascontiguousarray(view.T, np.float32).ravel()
                u[16:32] = np.ascontiguousarray(proj.T, np.float32).ravel()
                u[32] = 1.0 if shots[i] == owner else 0.0
                self.device.queue.write_buffer(L.uniform, 0, u.tobytes())
                rp.set_pipeline(self._mesh_pipe)
                rp.set_bind_group(0, L.bind)
                rp.set_vertex_buffer(0, L.vbuf)
                rp.set_index_buffer(L.ibuf, wgpu.IndexFormat.uint32)
                rp.draw_indexed(L.count, 1, 0, 0, 0)
                slots[i, :3] = L.eye
                slots[i, 3] = max(L.trust, 1e-3)
            rp.end()
        bu = np.zeros(40, np.float32)
        bu[0:16] = np.ascontiguousarray(inv_view.T, np.float32).ravel()
        bu[16:20] = (proj[0, 0], proj[1, 1], w, h)
        bu[20:23] = eye
        bu[24:40] = slots.ravel()
        self.device.queue.write_buffer(self._blend_uniform, 0, bu.tobytes())
        colour_view = self._color.create_view()
        rp = enc.begin_render_pass(color_attachments=[
            {"view": colour_view, "clear_value": (0, 0, 0, 0), "load_op": wgpu.LoadOp.clear,
             "store_op": wgpu.StoreOp.store},
            {"view": self._depth.create_view(), "clear_value": (0, 0, 0, 0), "load_op": wgpu.LoadOp.clear,
             "store_op": wgpu.StoreOp.store}])
        rp.set_pipeline(self._blend_pipe)
        rp.set_bind_group(0, self._blend_bind)
        rp.draw(3, 1, 0, 0)
        rp.end()
        from . import wet3d
        if wet3d.active(effects):
            if getattr(self, "_wet", None) is None:
                self._wet = wet3d.WetPass(self.device)
            self._wet.encode(enc, self._color, self._depth.create_view(), view, proj, (w, h), effects,
                             time_seconds, self._ground_up(effects))
        if effects is not None:
            self._fx.encode(enc, colour_view, self._depth.create_view(), view, proj, eye, effects,
                            time_seconds, self._planes)
        self.device.queue.submit([enc.finish()])

    def _ground_up(self, effects) -> tuple[float, float, float]:
        for pl in getattr(effects, "planes", []) or []:
            if pl.enabled and pl.kind == "floor":
                return tuple(float(v) for v in pl.normal())
        for n, _c in self._planes:
            n = np.asarray(n, np.float32)
            if abs(float(n[1])) > 0.85:
                n = n if n[1] > 0 else -n
                return tuple(float(v) for v in n / (np.linalg.norm(n) + 1e-9))
        return (0.0, 1.0, 0.0)

    def render_u8(self, view: np.ndarray, proj: np.ndarray, size,
                  background=(0.02, 0.021, 0.026), effects=None, time_seconds: float = 0.0,
                  mode: int = 0) -> np.ndarray:
        wgpu = self._wgpu
        w, h = int(size[0]), int(size[1])
        self._draw(view, proj, (w, h), effects if mode == 0 else None, time_seconds)
        flash = 0.0
        if effects is not None and mode == 0:
            from .fx3d import scene_flash
            flash = scene_flash(effects, time_seconds)
        self.device.queue.write_buffer(self._fin_uniform, 0, np.array([*background, flash], np.float32).tobytes())
        enc = self.device.create_command_encoder()
        rp = enc.begin_render_pass(color_attachments=[{
            "view": self._final.create_view(), "clear_value": (0, 0, 0, 1),
            "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store}])
        rp.set_pipeline(self._fin_pipe)
        rp.set_bind_group(0, self._fin_bind)
        rp.draw(3, 1, 0, 0)
        rp.end()
        self.device.queue.submit([enc.finish()])
        raw = self.device.queue.read_texture({"texture": self._final},
                                             {"bytes_per_row": w * 4, "rows_per_image": h}, (w, h, 1))
        return np.ascontiguousarray(np.frombuffer(raw, np.uint8).reshape(h, w, 4)[:, :, :3])

    def render(self, view, proj, size, background=(0.02, 0.021, 0.026), effects=None,
               time_seconds: float = 0.0, mode: int = 0) -> np.ndarray:
        return self.render_u8(view, proj, size, background, effects, time_seconds, mode).astype(np.float32) / 255.0
