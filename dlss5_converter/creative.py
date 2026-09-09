"""The 3D tab's renderer: a 2.5-D "living photo" built from the depth we
already compute for every image.

This is deliberately *not* a 3D engine. It takes one image plus its inverse
depth and produces a short looping clip: a small, amplitude-capped camera move
through the depth (parallax), optional depth-aware fog, and drifting particles
(embers / dust) that sit in the scene and are occluded by the subject. No new
models, no downloads. Everything is numpy + cv2 so it runs anywhere the app
already runs.

Design notes that matter:
- Motion is capped hard. Big parallax tears the single image open (there is no
  data behind the foreground), so every preset stays within a few percent of the
  frame and we over-scan the base so the moving frame never reveals an edge.
- The loop is rendered once to a frame cache and played back by the UI, so
  playback is always smooth without a GPU path. Re-render happens only when a
  control changes.
- Depth intensity is one knob that scales both the parallax separation and how
  far into the scene particles sit, so fog and embers always share the photo's
  space.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# Aspect ratios offered in the UI, as width / height.
ASPECTS: dict[str, float] = {
    "Square (1:1)": 1.0,
    "Portrait (4:5)": 4 / 5,
    "Portrait (9:16)": 9 / 16,
    "Landscape (16:9)": 16 / 9,
}

# Motion presets: each maps a loop phase p in [0, 1) to a normalised camera
# offset (tx, ty, zoom_extra). Amplitudes here are tiny on purpose; the actual
# pixel amount is these times AMP_PX / the frame, so nothing ever tears.
AMP_PX = 26.0            # peak parallax travel in pixels at depth-intensity 1.0
OVERSCAN = 1.10          # base zoom so the moving frame never shows a border


def _loop(p: float) -> float:
    """A smooth 0 -> 1 -> 0 envelope over one loop (ease in/out, seamless)."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * p)


# Each preset maps loop phase p -> (tx, ty, zoom, dolly). `dolly` is the
# Vertigo/dolly-zoom term: it zooms the far planes more than the near, so the
# background diverges from a held foreground.
def _orbit(p: float):
    a = 2.0 * np.pi * p
    return np.cos(a), np.sin(a), 0.0, 0.0       # gentle circular sway


def _push(p: float):
    e = _loop(p)
    return 0.0, -0.15 * e, 0.9 * e, 0.0         # dolly in, slight rise


def _float(p: float):
    return 0.35 * np.sin(2 * np.pi * p), 0.5 * _loop(p) - 0.25, 0.2 * _loop(p), 0.0


def _sway(p: float):
    return np.sin(2 * np.pi * p), 0.12 * np.cos(2 * np.pi * p), 0.0, 0.0


def _dolly(p: float):
    # Vertigo: hold the framing, diverge the depth planes. Pure Z, no pan, so
    # the background rushes relative to the foreground. The user's favourite.
    e = _loop(p)
    return 0.0, 0.0, 0.5 * e, 1.0 * e


PRESETS: dict[str, callable] = {
    "Float": _float,
    "Push in": _push,
    "Dolly zoom": _dolly,
    "Orbit": _orbit,
    "Sway": _sway,
}


@dataclass
class CreativeSettings:
    aspect: str = "Portrait (4:5)"
    preset: str = "Float"
    depth_intensity: float = 1.0          # 0..2, scales parallax separation

    # Fog: a depth plane it begins at, so it can sit behind the subject or fill
    # the whole scene. plane 0 = fog everywhere, 1 = only the deepest distance.
    fog: float = 0.35                     # 0..1 density
    fog_plane: float = 0.5                # 0 near .. 1 far, where fog starts
    fog_color: tuple[int, int, int] = (150, 156, 168)  # RGB, cool haze

    # Embers: warm motes. dir is degrees (0 up, 90 right, 180 down, 270 left).
    embers: float = 0.0                   # 0..1 amount
    ember_plane: float = 0.5              # depth the embers sit at (0 near..1 far)
    ember_dir: float = 0.0                # drift direction, degrees
    ember_speed: float = 0.4              # 0..1
    ember_size: float = 0.5               # 0..1 base particle size

    # Dust: cooler, slower motes.
    dust: float = 0.35
    dust_plane: float = 0.5
    dust_dir: float = 90.0
    dust_speed: float = 0.25
    dust_size: float = 0.4

    frames: int = 48                      # loop length
    fps: int = 24


def reframe(image_rgb: np.ndarray, inv_depth: np.ndarray, aspect: float
            ) -> tuple[np.ndarray, np.ndarray]:
    """Centre-crop the image and its depth to the target aspect ratio."""
    h, w = image_rgb.shape[:2]
    cur = w / h
    if cur > aspect:                      # too wide: trim sides
        nw = int(round(h * aspect)); x0 = (w - nw) // 2
        sl = (slice(None), slice(x0, x0 + nw))
    else:                                 # too tall: trim top/bottom
        nh = int(round(w / aspect)); y0 = (h - nh) // 2
        sl = (slice(y0, y0 + nh), slice(None))
    return np.ascontiguousarray(image_rgb[sl]), np.ascontiguousarray(inv_depth[sl])


class Renderer:
    """Holds the per-image maps and renders any loop frame cheaply.

    Building the coordinate grids and the depth-derived fields once, then
    reusing them for every frame, is what keeps a whole loop render fast enough
    to feel instant on a control change.
    """

    def __init__(self, image_rgb: np.ndarray, inv_depth: np.ndarray):
        self.img = image_rgb.astype(np.float32)
        if self.img.max() > 1.5:
            self.img /= 255.0
        h, w = self.img.shape[:2]
        self.h, self.w = h, w
        d = inv_depth.astype(np.float32)
        d = (d - d.min()) / (np.ptp(d) + 1e-6)      # 0 far .. 1 near
        # A touch of blur so parallax edges do not shimmer on hard depth steps.
        self.near = cv2.GaussianBlur(d, (0, 0), max(1.0, w / 400.0))
        self.far = 1.0 - self.near
        self.X, self.Y = np.meshgrid(np.arange(w, dtype=np.float32),
                                     np.arange(h, dtype=np.float32))
        self.cx, self.cy = w / 2.0, h / 2.0
        self._rng = np.random.default_rng(7)
        self._particles: dict[str, np.ndarray] = {}

    # -- camera / parallax ---------------------------------------------------

    def _warp(self, tx: float, ty: float, zoom: float, amp: float,
              dolly: float = 0.0) -> np.ndarray:
        """The image seen from a small camera offset, near moving more than far."""
        # Backward map: for output p, sample base at the de-zoomed, de-parallaxed
        # position. Parallax scales with `near` so foreground shifts most.
        map_x, map_y = self._maps(tx, ty, zoom, amp, dolly)
        return cv2.remap(self.img, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)

    def _maps(self, tx: float, ty: float, zoom: float, amp: float,
              dolly: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        z = OVERSCAN + zoom * 0.10
        # Dolly zoom: far pixels get an extra per-pixel zoom, so the background
        # scales while the near subject holds. dolly 0 collapses to a plain zoom.
        zz = z * (1.0 + dolly * self.far)
        shift = amp * self.near
        map_x = (self.cx + (self.X - self.cx) / zz - tx * shift).astype(np.float32)
        map_y = (self.cy + (self.Y - self.cy) / zz - ty * shift).astype(np.float32)
        return np.ascontiguousarray(map_x), np.ascontiguousarray(map_y)

    def _warp_scalar(self, field: np.ndarray, tx, ty, zoom, amp, dolly=0.0) -> np.ndarray:
        map_x, map_y = self._maps(tx, ty, zoom, amp, dolly)
        return cv2.remap(field, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)

    # -- particles -----------------------------------------------------------

    def _ensure_particles(self, kind: str, n: int) -> np.ndarray:
        """Lazily create a particle set. Columns:
        0 x0, 1 y0 (start position), 2 depth offset around the chosen plane,
        3..6 turbulence phases, 7 per-particle size jitter.
        Plane / direction / speed / size are applied at draw time, so those
        controls are live without regenerating the set.
        """
        key = f"{kind}:{n}"
        if key not in self._particles:
            r = self._rng
            p = np.zeros((n, 8), np.float32)
            p[:, 0] = r.uniform(0, self.w, n)
            p[:, 1] = r.uniform(0, self.h, n)
            p[:, 2] = r.uniform(-0.22, 0.22, n)       # spread around the plane
            p[:, 3:7] = r.uniform(0.0, 1.0, (n, 4))   # turbulence phases
            p[:, 7] = r.uniform(0.6, 1.4, n)          # size jitter
            self._particles[key] = p
        return self._particles[key]

    def _draw_particles(self, canvas: np.ndarray, kind: str, amount: float,
                        plane: float, dir_deg: float, speed: float, size: float,
                        phase: float, tx: float, ty: float, amp: float) -> None:
        if amount <= 0:
            return
        n = int(amount * (240 if kind == "dust" else 180))
        if n <= 0:
            return
        p = self._ensure_particles(kind, n)
        t = phase
        L = max(self.h, self.w)
        tau = 2.0 * np.pi
        x0, y0, zoff = p[:, 0], p[:, 1], p[:, 2]
        ph1, ph2, ph3, ph4, sj = p[:, 3], p[:, 4], p[:, 5], p[:, 6], p[:, 7]

        # Directional drift (0 deg = up), toroidal so the density loops.
        theta = np.deg2rad(dir_deg)
        travel = 0.85 * L * speed
        vx, vy = np.sin(theta) * travel, -np.cos(theta) * travel
        # Vortex: a per-particle circular orbit plus a faster wobble, both
        # periodic in the loop so it stays seamless. Embers swirl hard, dust
        # drifts. This is what makes embers read as real, not on rails.
        swirl = L * (0.06 if kind == "embers" else 0.018) * (0.5 + sj)
        ox = swirl * np.sin((t + ph1) * tau) + 0.5 * swirl * np.sin((2 * t + ph2) * tau)
        oy = swirl * np.cos((t + ph1) * tau) + 0.5 * swirl * np.cos((2 * t + ph3) * tau)
        x = (x0 + vx * t + ox) % self.w
        y = (y0 + vy * t + oy) % self.h
        z = np.clip(plane + zoff, 0.0, 1.0)

        # Parallax: particles shift with the camera scaled by their own depth.
        sx = x + tx * amp * z
        sy = y + ty * amp * z
        xi = np.clip(sx.astype(int), 0, self.w - 1)
        yi = np.clip(sy.astype(int), 0, self.h - 1)
        visible = self.near[yi, xi] <= z + 0.05      # hidden behind nearer scene
        twinkle = 0.55 + 0.45 * np.sin((2 * t + ph4) * tau)

        # True 3-D size: near particles bigger, far ones down to a single pixel.
        maxr = L / 70.0
        rad = (0.5 + size * maxr) * (0.25 + 1.15 * z) * sj

        col = (np.array([1.0, 0.55, 0.18], np.float32) if kind == "embers"
               else np.array([0.85, 0.9, 1.0], np.float32))
        glow = kind == "embers"
        layer = np.zeros((self.h, self.w), np.float32)
        for k in np.nonzero(visible)[0]:
            xk, yk = int(sx[k]), int(sy[k])
            if not (0 <= xk < self.w and 0 <= yk < self.h):
                continue
            b = float(twinkle[k]) * (0.6 * float(z[k]) + 0.4)
            r = int(round(float(rad[k])))
            if r < 1:                                  # sub-pixel motes: 1 px
                layer[yk, xk] += b
            else:
                cv2.circle(layer, (xk, yk), r, float(b), -1, cv2.LINE_AA)
        sigma = max(0.6, L / 800.0) * (1.3 if glow else 1.0)
        layer = cv2.GaussianBlur(layer, (0, 0), sigma)
        canvas += layer[..., None] * col * (1.5 if glow else 0.7)

    # -- one frame -----------------------------------------------------------

    def frame(self, s: CreativeSettings, index: int) -> np.ndarray:
        phase = index / s.frames
        tx, ty, zoom, dolly = PRESETS[s.preset](phase)
        amp = AMP_PX * s.depth_intensity
        out = self._warp(tx, ty, zoom, amp, dolly)
        if s.fog > 0:
            # `far` is 0 at the nearest pixel, 1 at the deepest. Fog begins at
            # the chosen plane and deepens with distance beyond it.
            far = self._warp_scalar(self.far, tx, ty, zoom, amp, dolly)
            p0 = float(s.fog_plane)
            band = np.clip((far - p0) / (1.0 - p0 + 1e-3), 0.0, 1.0) ** 1.3
            fog = band[..., None] * (s.fog * 0.9)
            fc = np.array(s.fog_color, np.float32) / 255.0
            out = out * (1.0 - fog) + fc * fog
        self._draw_particles(out, "dust", s.dust, s.dust_plane, s.dust_dir,
                             s.dust_speed, s.dust_size, phase, tx, ty, amp)
        self._draw_particles(out, "embers", s.embers, s.ember_plane, s.ember_dir,
                             s.ember_speed, s.ember_size, phase, tx, ty, amp)
        return np.clip(out, 0.0, 1.0)

    def render_loop(self, s: CreativeSettings) -> list[np.ndarray]:
        return [self.frame(s, i) for i in range(s.frames)]
