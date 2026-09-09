"""The 3D tab's scene: a GPU-rendered depth mesh plus atmosphere.

The geometry and camera work is done by `mesh3d` (a connected, textured depth
mesh rasterised on the GPU via wgpu) and `camera3d` (a real perspective camera).
This module builds the mesh from an image + its depth, drives the camera from a
preset over the loop, and composites the lighter effects (fog is in the mesh
shader; flare and drifting particles are added here) on top of the rendered RGB.

Particles are projected through the same camera matrix so they move with the
scene. They are a stop-gap 2-D-over-3-D composite; the plan is to make fog and
particles true 3-D volumes with in-viewport placement, like Depth Animator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from . import camera3d, layers, mesh3d

ASPECTS: dict[str, float] = {
    "Original": 0.0,                      # 0 = keep the source's own aspect
    "Portrait (4:5)": 4 / 5,
    "Portrait (9:16)": 9 / 16,
    "Square (1:1)": 1.0,
    "Landscape (16:9)": 16 / 9,
}
VIEW_MODES = ("Solid", "Wireframe", "Point cloud")
PRESETS = camera3d.PRESET_NAMES

FOV = 55.0
NEAR, FAR = 1.0, 6.0        # scene depth range in view metres
_RENDERER: mesh3d.MeshRenderer | None = None


def get_renderer() -> mesh3d.MeshRenderer:
    """Lazily create the shared GPU renderer (on the calling thread)."""
    global _RENDERER
    if _RENDERER is None:
        _RENDERER = mesh3d.MeshRenderer()
    return _RENDERER


@dataclass
class CreativeSettings:
    aspect: str = "Original"
    preset: str = "Orbit"
    view: str = "Solid"
    depth_intensity: float = 1.0          # camera-move strength

    fog: float = 0.25
    fog_plane: float = 0.5
    fog_color: tuple[int, int, int] = (150, 156, 168)

    flare: float = 0.2

    embers: float = 0.0
    ember_plane: float = 0.5
    ember_dir: float = 0.0
    ember_speed: float = 0.4
    ember_size: float = 0.5

    dust: float = 0.25
    dust_plane: float = 0.5
    dust_dir: float = 90.0
    dust_speed: float = 0.25
    dust_size: float = 0.4

    frames: int = 48
    fps: int = 24


def reframe(image_rgb: np.ndarray, inv_depth: np.ndarray, aspect: float):
    if aspect <= 0:                       # "Original": keep the source framing
        return np.ascontiguousarray(image_rgb), np.ascontiguousarray(inv_depth)
    h, w = image_rgb.shape[:2]
    if w / h > aspect:
        nw = int(round(h * aspect)); x0 = (w - nw) // 2
        sl = (slice(None), slice(x0, x0 + nw))
    else:
        nh = int(round(w / aspect)); y0 = (h - nh) // 2
        sl = (slice(y0, y0 + nh), slice(None))
    return np.ascontiguousarray(image_rgb[sl]), np.ascontiguousarray(inv_depth[sl])


class Renderer:
    """One image's 3-D scene: builds the mesh once, renders any loop frame."""

    def __init__(self, image_rgb: np.ndarray, inv_depth: np.ndarray,
                 long_side: int = 900, inpainter=None):
        img = image_rgb.astype(np.float32)
        if img.max() > 1.5:
            img /= 255.0
        # Work at a capped resolution; the GPU render is the output size.
        h, w = img.shape[:2]
        scale = long_side / max(h, w)
        if scale < 1.0:
            w2, h2 = int(w * scale), int(h * scale)
            w2 -= w2 % 2; h2 -= h2 % 2
            img = cv2.resize(img, (w2, h2), interpolation=cv2.INTER_AREA)
        self.h, self.w = img.shape[:2]
        self.rgb8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        d = inv_depth.astype(np.float32)
        d = (d - d.min()) / (np.ptp(d) + 1e-6)
        if d.shape != (self.h, self.w):
            d = cv2.resize(d, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        # Relief: sharpen local depth so surfaces round out instead of reading
        # as flat cards. A mild unsharp on the depth adds within-object
        # curvature without inventing structure that is not there.
        blur = cv2.GaussianBlur(d, (0, 0), max(1.0, self.w / 90.0))
        d = np.clip(d + 0.6 * (d - blur), 0.0, 1.0)
        self.depth = d

        # Layered slices: separate the image into depth-ordered layers, each its
        # own isolated mesh at its depth (with rolled edges), and fill the hidden
        # backplate. A near layer then parallaxes over the FILLED background
        # instead of tearing a hole. layer_count is an upper bound: the builder
        # collapses to as many layers as the depth actually supports (auto).
        stride = max(1, int(round(math.sqrt(self.w * self.h / 200_000))))
        scene = layers.build_layered_scene(
            self.rgb8, d, layer_count=6, fov_degrees=FOV, near=NEAR, far=FAR,
            stride=stride, detail=1.0, colour_inpainter=inpainter)
        self._scene_layers = []
        allpos = []
        for layer, mesh in zip(scene.layers, scene.meshes):
            if mesh.triangle_count == 0:
                continue
            self._scene_layers.append((mesh.positions, mesh.uvs, mesh.indices,
                                       (mesh.height, mesh.width), layer.colour))
            allpos.append(mesh.positions)
        pos = np.concatenate(allpos) if allpos else np.zeros((1, 3), np.float32)
        self.n_layers = len(self._scene_layers)
        self.pivot_z = -float((NEAR + FAR) * 0.5)
        # Scene bounds at the pivot plane, for placing particles in the volume.
        self._xmax = float(np.abs(pos[:, 0]).max()) or 1.0
        self._ymax = float(np.abs(pos[:, 1]).max()) or 1.0

        self._upload()
        self._rng = np.random.default_rng(7)
        self._parts: dict[str, np.ndarray] = {}

    # -- particles -----------------------------------------------------------

    def _ensure_parts(self, kind: str, n: int) -> np.ndarray:
        key = f"{kind}:{n}"
        if key not in self._parts:
            rng = self._rng
            p = np.zeros((n, 8), np.float32)
            p[:, 0] = rng.uniform(-self._xmax, self._xmax, n)
            p[:, 1] = rng.uniform(-self._ymax, self._ymax, n)
            p[:, 2] = rng.uniform(-0.35, 0.35, n)          # z jitter around plane
            p[:, 3:7] = rng.uniform(0, 1, (n, 4))
            p[:, 7] = rng.uniform(0.6, 1.4, n)
            self._parts[key] = p
        return self._parts[key]

    def _draw_particles(self, canvas, mvp, fpx, kind, amount, plane, dir_deg,
                        speed, size, phase):
        if amount <= 0:
            return
        n = int(amount * (260 if kind == "dust" else 190))
        if n <= 0:
            return
        p = self._ensure_parts(kind, n)
        t = phase; tau = 2 * np.pi
        x0, y0, zoff = p[:, 0], p[:, 1], p[:, 2]
        ph1, ph2, ph3, ph4, sj = p[:, 3], p[:, 4], p[:, 5], p[:, 6], p[:, 7]
        th = np.deg2rad(dir_deg)
        trav = (0.9 if kind == "embers" else 0.6) * speed
        vx, vy = np.sin(th) * trav, np.cos(th) * trav
        swirl = (0.10 if kind == "embers" else 0.03) * (0.5 + sj)
        ox = swirl * np.sin((t + ph1) * tau) + 0.5 * swirl * np.sin((2 * t + ph2) * tau)
        oy = swirl * np.cos((t + ph1) * tau) + 0.5 * swirl * np.cos((2 * t + ph3) * tau)
        X = ((x0 + vx * t + ox + self._xmax) % (2 * self._xmax)) - self._xmax
        Y = ((y0 + vy * t + oy + self._ymax) % (2 * self._ymax)) - self._ymax
        Z = self.pivot_z + (plane - 0.5) * (FAR - NEAR) + zoff
        pts = np.stack([X, Y, np.full_like(X, Z)], 1)
        ph = np.concatenate([pts, np.ones((len(pts), 1), np.float32)], 1)
        clip = ph @ mvp.T
        wv = clip[:, 3]
        ok = wv > 1e-3
        ndc = clip[:, :3] / np.where(ok, wv, 1.0)[:, None]
        sx = (ndc[:, 0] * 0.5 + 0.5) * self.w
        sy = (1 - (ndc[:, 1] * 0.5 + 0.5)) * self.h
        world = (0.004 + size * 0.03) * sj
        with np.errstate(divide="ignore", invalid="ignore"):
            rad = np.clip(fpx * world / np.maximum(wv, 1e-3), 0.0, 40.0)
        twin = 0.55 + 0.45 * np.sin((2 * t + ph4) * tau)
        col = (np.array([1.0, 0.55, 0.18], np.float32) if kind == "embers"
               else np.array([0.85, 0.9, 1.0], np.float32))
        glow = kind == "embers"
        # Vectorised, at half resolution: bucket particles by screen radius into
        # a few size tiers and scatter+blur each tier once, instead of a Python
        # loop of circle draws. Keeps 3-D size variation, an order faster.
        h2, w2 = self.h // 2, self.w // 2
        xi = (sx * 0.5).astype(int); yi = (sy * 0.5).astype(int)
        good = ok & (xi >= 0) & (xi < w2) & (yi >= 0) & (yi < h2) & (ndc[:, 2] < 1.0)
        rad_h = rad * 0.5
        tier_sigma = (1.0, 3.4)          # small vs large motes; two blurs/kind
        edges = np.array([2.2])
        tiers = np.digitize(rad_h, edges)
        out = np.zeros((h2, w2), np.float32)
        for ti, sig in enumerate(tier_sigma):
            sel = good & (tiers == ti)
            if not sel.any():
                continue
            lay = np.zeros((h2, w2), np.float32)
            np.add.at(lay, (yi[sel], xi[sel]), twin[sel])
            out += cv2.GaussianBlur(lay, (0, 0), sig) * (2.0 * np.pi * sig * sig)
        out = cv2.resize(out, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        canvas += out[..., None] * col * (0.22 if glow else 0.10)

    def _apply_flare(self, canvas, amount):
        h2, w2 = self.h // 2, self.w // 2
        small = cv2.resize(canvas, (w2, h2), interpolation=cv2.INTER_AREA)
        lum = small.mean(2)
        mask = np.clip((lum - 0.82) / 0.18, 0.0, 1.0)[..., None] * small
        bloom = cv2.GaussianBlur(mask, (0, 0), max(2.0, w2 / 55.0))
        canvas += cv2.resize(bloom, (self.w, self.h), interpolation=cv2.INTER_LINEAR) * (amount * 0.9)

    # -- one frame -----------------------------------------------------------

    def frame(self, s: CreativeSettings, index: int) -> np.ndarray:
        phase = index / s.frames
        rig = camera3d.preset_rig(s.preset, phase, self.pivot_z,
                                  strength=s.depth_intensity, base_fov=FOV)
        cam = rig.to_camera()
        aspect = self.w / self.h
        mvp = cam.view_projection(aspect)

        fog_start = NEAR + float(s.fog_plane) * (FAR - NEAR)
        density = s.fog * 2.0 / (FAR - NEAR)
        fc = np.array(s.fog_color, np.float32) / 255.0
        rgb = get_renderer().render(mvp, (self.w, self.h), s.view,
                                    tuple(fc), density, fog_start)
        canvas = rgb.copy()

        fpx = 0.5 * self.h / math.tan(math.radians(cam.fov_degrees) * 0.5)
        self._draw_particles(canvas, mvp, fpx, "dust", s.dust, s.dust_plane,
                             s.dust_dir, s.dust_speed, s.dust_size, phase)
        self._draw_particles(canvas, mvp, fpx, "embers", s.embers, s.ember_plane,
                             s.ember_dir, s.ember_speed, s.ember_size, phase)
        if s.flare > 0:
            self._apply_flare(canvas, s.flare)
        return np.clip(canvas, 0.0, 1.0)

    def _upload(self) -> None:
        get_renderer().set_scene(self._scene_layers, point_stride=1)

    def render_loop(self, s: CreativeSettings) -> list[np.ndarray]:
        return [self.frame(s, i) for i in range(s.frames)]
