"""GPU renderer for the 3D tab: a connected, textured depth mesh via wgpu.

Adapted from the Depth Animator viewer. The depth map is unprojected into a
real 3-D point surface, triangulated into a *connected* mesh, and rasterised on
the GPU with a z-buffer and 4x MSAA. That is what fixes the loose-splat gaps and
the low resolution of the earlier CPU approach: the texture interpolates across
triangles, so hard edges stretch a continuous surface instead of tearing into
holes, and it renders crisply at full size.

Three primitive modes share one vertex buffer:
  - solid: textured triangles,
  - wireframe: deduplicated triangle edges as lines,
  - point cloud: a strided subset of vertices as 1-px points (sparse, like the
    loading reveal), not every vertex.

wgpu targets Vulkan/D3D12/Metal, so this is GPU-accelerated everywhere without
an OpenGL context. Depth is left to the shader's fog; particles/flare are
composited on top by the caller.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

SAMPLE_COUNT = 4

SHADER = """
struct U {
    mvp: mat4x4<f32>,
    fog: vec4<f32>,      // rgb colour, w = density
    params: vec4<f32>,   // x fog_start, y unused, z unused, w unused
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) dist: f32,
};

@vertex
fn vs_main(@location(0) pos: vec3<f32>, @location(1) uv: vec2<f32>) -> VOut {
    var o: VOut;
    o.clip = u.mvp * vec4<f32>(pos, 1.0);
    o.uv = uv;
    o.dist = o.clip.w;                 // view-space distance from camera
    return o;
}

fn fogged(rgb: vec3<f32>, dist: f32) -> vec3<f32> {
    let f = clamp((dist - u.params.x) * u.fog.w, 0.0, 1.0);
    return mix(rgb, u.fog.xyz, f);
}

@fragment
fn fs_main(i: VOut) -> @location(0) vec4<f32> {
    let c = textureSample(tex, samp, i.uv).rgb;
    return vec4<f32>(fogged(c, i.dist), 1.0);
}

@fragment
fn fs_point(i: VOut) -> @location(0) vec4<f32> {
    let c = textureSample(tex, samp, i.uv).rgb;
    return vec4<f32>(fogged(c, i.dist), 1.0);
}

@fragment
fn fs_wire(i: VOut) -> @location(0) vec4<f32> {
    return vec4<f32>(0.35, 0.82, 1.0, 1.0);
}
"""


def disparity_to_depth(disp: np.ndarray, near: float, far: float) -> np.ndarray:
    """Normalised inverse depth [0,1] -> view-space Z, interpolated in disparity
    so parallax stays physically even (screen motion ~ 1/Z)."""
    disp = np.clip(disp.astype(np.float32), 0.0, 1.0)
    inv = (1.0 / far) + ((1.0 / near) - (1.0 / far)) * disp
    return (1.0 / np.maximum(inv, 1e-6)).astype(np.float32)


def build_grid_mesh(depth: np.ndarray, stride: int, fov_deg: float,
                    near: float, far: float):
    """Connected displaced grid mesh. Returns (positions Nx3, uvs Nx2, tris Mx3)."""
    d = depth[::stride, ::stride]
    h, w = d.shape
    z = disparity_to_depth(d, near, far)
    focal = h / (2.0 * math.tan(math.radians(fov_deg) * 0.5))
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (xs - cx) * z / focal
    y = -(ys - cy) * z / focal
    pos = np.stack([x, y, -z], -1).reshape(-1, 3).astype(np.float32)
    uv = np.stack([xs / max(w - 1, 1), ys / max(h - 1, 1)], -1).reshape(-1, 2).astype(np.float32)
    idx = np.arange(h * w, dtype=np.uint32).reshape(h, w)
    tl, tr = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    bl, br = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    tris = np.concatenate([
        np.stack([tl, bl, tr], -1),
        np.stack([tr, bl, br], -1),
    ]).astype(np.uint32)
    return pos, uv, tris, (h, w)


def _grid_edges(gh: int, gw: int, step: int) -> np.ndarray:
    """Line-list edges of a coarse sub-grid: every `step`-th row and column,
    connecting neighbours along that row/column. Reads as a clean wireframe."""
    idx = np.arange(gh * gw, dtype=np.uint32).reshape(gh, gw)
    segs = []
    rows = idx[::step, :]
    segs.append(np.stack([rows[:, :-1], rows[:, 1:]], -1).reshape(-1, 2))
    cols = idx[:, ::step]
    segs.append(np.stack([cols[:-1, :], cols[1:, :]], -1).reshape(-1, 2))
    return np.concatenate(segs).astype(np.uint32).ravel()


def _unique_edges(tris: np.ndarray) -> np.ndarray:
    t = tris.reshape(-1, 3).astype(np.uint32)
    pairs = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    lo = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.uint64)
    hi = np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.uint64)
    keys = np.unique((lo << np.uint64(32)) | hi)
    out = np.empty((keys.size, 2), np.uint32)
    out[:, 0] = (keys >> np.uint64(32)).astype(np.uint32)
    out[:, 1] = (keys & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    return out.ravel()


class MeshRenderer:
    """Holds a wgpu device and renders the current mesh to an RGB numpy array."""

    def __init__(self) -> None:
        import wgpu
        self._wgpu = wgpu
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = self.adapter.request_device_sync()
        self._shader = self.device.create_shader_module(code=SHADER)
        self._sampler = self.device.create_sampler(
            mag_filter="linear", min_filter="linear", mipmap_filter="linear",
            address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge")
        self._uniform = self.device.create_buffer(
            size=96, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._pipes: dict[str, object] = {}
        self._targets = None
        self._tex_view = None
        self._bind = None
        self._grid_hw = (0, 0)

    @property
    def adapter_name(self) -> str:
        return str(self.adapter.info.get("description", "GPU"))

    # -- resources -----------------------------------------------------------

    def set_texture(self, image_rgb: np.ndarray) -> None:
        wgpu = self._wgpu
        img = np.ascontiguousarray(image_rgb, np.uint8)
        h, w = img.shape[:2]
        rgba = np.dstack([img, np.full((h, w, 1), 255, np.uint8)])
        levels = [np.ascontiguousarray(rgba)]
        lw, lh = w, h
        while lw > 1 or lh > 1:
            lw, lh = max(1, lw // 2), max(1, lh // 2)
            levels.append(np.ascontiguousarray(
                cv2.resize(levels[-1], (lw, lh), interpolation=cv2.INTER_AREA)))
        tex = self.device.create_texture(
            size=(w, h, 1), format="rgba8unorm", mip_level_count=len(levels),
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        for lvl, data in enumerate(levels):
            dh, dw = data.shape[:2]
            self.device.queue.write_texture(
                {"texture": tex, "mip_level": lvl}, data,
                {"bytes_per_row": dw * 4, "rows_per_image": dh}, (dw, dh, 1))
        self._tex_view = tex.create_view()
        self._bind = None

    def set_mesh(self, pos: np.ndarray, uv: np.ndarray, tris: np.ndarray,
                 grid_hw, point_stride: int = 3) -> None:
        wgpu = self._wgpu
        verts = np.ascontiguousarray(np.hstack([pos, uv]), np.float32)
        self._vbuf = self.device.create_buffer_with_data(
            data=verts, usage=wgpu.BufferUsage.VERTEX)
        tri_idx = np.ascontiguousarray(tris.ravel(), np.uint32)
        self._ibuf = self.device.create_buffer_with_data(
            data=tri_idx, usage=wgpu.BufferUsage.INDEX)
        self._icount = int(tri_idx.size)
        # Wireframe from a COARSE grid subset, not every triangle edge, so it
        # reads as a wire lattice over the form instead of a solid fill.
        gh, gw = grid_hw
        ws = max(1, min(gh, gw) // 60)
        edges = _grid_edges(gh, gw, ws)
        self._ebuf = self.device.create_buffer_with_data(
            data=edges, usage=wgpu.BufferUsage.INDEX)
        self._ecount = int(edges.size)
        # Sparse points: a strided subset of the grid, so the cloud reads as
        # dots (like the loading reveal) rather than a solid wall of pixels.
        gh, gw = grid_hw
        pidx = np.arange(gh * gw, dtype=np.uint32).reshape(gh, gw)
        pidx = pidx[::point_stride, ::point_stride].ravel()
        self._pbuf = self.device.create_buffer_with_data(
            data=np.ascontiguousarray(pidx), usage=wgpu.BufferUsage.INDEX)
        self._pcount = int(pidx.size)

    # -- pipelines / targets -------------------------------------------------

    def _layout(self):
        wgpu = self._wgpu
        if not hasattr(self, "_bind_layout"):
            self._bind_layout = self.device.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                 "buffer": {"type": wgpu.BufferBindingType.uniform}},
                {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {}},
                {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {}},
            ])
        return self._bind_layout

    def _pipe(self, topology: str, entry: str):
        key = topology + entry
        if key in self._pipes:
            return self._pipes[key]
        wgpu = self._wgpu
        layout = self.device.create_pipeline_layout(bind_group_layouts=[self._layout()])
        pipe = self.device.create_render_pipeline(
            layout=layout,
            vertex={"module": self._shader, "entry_point": "vs_main", "buffers": [{
                "array_stride": 5 * 4, "step_mode": wgpu.VertexStepMode.vertex,
                "attributes": [
                    {"format": "float32x3", "offset": 0, "shader_location": 0},
                    {"format": "float32x2", "offset": 12, "shader_location": 1}]}]},
            primitive={"topology": topology, "cull_mode": wgpu.CullMode.none},
            depth_stencil={"format": "depth32float", "depth_write_enabled": True,
                           "depth_compare": wgpu.CompareFunction.less},
            multisample={"count": SAMPLE_COUNT},
            fragment={"module": self._shader, "entry_point": entry,
                      "targets": [{"format": "rgba8unorm"}]})
        self._pipes[key] = pipe
        return pipe

    def _ensure_targets(self, w: int, h: int) -> None:
        wgpu = self._wgpu
        if self._targets == (w, h):
            return
        self._msaa = self.device.create_texture(
            size=(w, h, 1), format="rgba8unorm", sample_count=SAMPLE_COUNT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
        self._resolve = self.device.create_texture(
            size=(w, h, 1), format="rgba8unorm",
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
        self._depth = self.device.create_texture(
            size=(w, h, 1), format="depth32float", sample_count=SAMPLE_COUNT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
        self._targets = (w, h)

    # -- render --------------------------------------------------------------

    def render(self, mvp: np.ndarray, size, mode: str,
               fog_color, fog_density: float, fog_start: float,
               background=(0.04, 0.055, 0.085)) -> np.ndarray:
        wgpu = self._wgpu
        w, h = int(size[0]), int(size[1])
        self._ensure_targets(w, h)
        u = np.zeros(24, np.float32)
        u[:16] = np.ascontiguousarray(mvp.T, np.float32).ravel()  # WGSL column-major
        u[16:19] = fog_color
        u[19] = max(0.0, fog_density)
        u[20] = fog_start
        self.device.queue.write_buffer(self._uniform, 0, u.tobytes())
        if self._bind is None:
            self._bind = self.device.create_bind_group(layout=self._layout(), entries=[
                {"binding": 0, "resource": {"buffer": self._uniform, "offset": 0, "size": 96}},
                {"binding": 1, "resource": self._tex_view},
                {"binding": 2, "resource": self._sampler}])
        enc = self.device.create_command_encoder()
        rp = enc.begin_render_pass(
            color_attachments=[{
                "view": self._msaa.create_view(),
                "resolve_target": self._resolve.create_view(),
                "clear_value": (*background, 1.0),
                "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store}],
            depth_stencil_attachment={
                "view": self._depth.create_view(), "depth_clear_value": 1.0,
                "depth_load_op": wgpu.LoadOp.clear, "depth_store_op": wgpu.StoreOp.store})
        rp.set_bind_group(0, self._bind)
        rp.set_vertex_buffer(0, self._vbuf)
        if mode == "Wireframe":
            rp.set_pipeline(self._pipe(wgpu.PrimitiveTopology.line_list, "fs_wire"))
            rp.set_index_buffer(self._ebuf, wgpu.IndexFormat.uint32)
            rp.draw_indexed(self._ecount, 1, 0, 0, 0)
        elif mode == "Point cloud":
            rp.set_pipeline(self._pipe(wgpu.PrimitiveTopology.point_list, "fs_point"))
            rp.set_index_buffer(self._pbuf, wgpu.IndexFormat.uint32)
            rp.draw_indexed(self._pcount, 1, 0, 0, 0)
        else:
            rp.set_pipeline(self._pipe(wgpu.PrimitiveTopology.triangle_list, "fs_main"))
            rp.set_index_buffer(self._ibuf, wgpu.IndexFormat.uint32)
            rp.draw_indexed(self._icount, 1, 0, 0, 0)
        rp.end()
        self.device.queue.submit([enc.finish()])
        raw = self.device.queue.read_texture(
            {"texture": self._resolve}, {"bytes_per_row": w * 4, "rows_per_image": h},
            (w, h, 1))
        frame = np.frombuffer(raw, np.uint8).reshape(h, w, 4)
        return np.ascontiguousarray(frame[:, :, :3]).astype(np.float32) / 255.0


def is_available() -> bool:
    try:
        import wgpu
        return wgpu.gpu.request_adapter_sync() is not None
    except Exception:  # noqa: BLE001
        return False
