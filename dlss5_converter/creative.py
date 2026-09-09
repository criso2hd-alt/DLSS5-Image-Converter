"""The 3D tab's renderer: a real depth-reconstructed 3-D scene, not a 2-D warp.

The earlier version slid pixels around in 2-D and tore at the slightest camera
move. This projects the depth map as an actual 3-D surface through a perspective
camera, the way the 3D-from-Image app's viewer does, except the geometry comes
from our own depth (a displaced point-surface) instead of an ML model, so there
are no new downloads.

Because it is a genuine camera in 3-D space:
- parallax, dolly and a true dolly-zoom (pull back while narrowing FOV) are all
  geometrically correct and hold up under motion;
- particles live in the same 3-D volume, orbit with the scene, size by real
  distance, and occlude against the surface's own depth buffer;
- three view modes fall out naturally: solid (splatted surface), wireframe
  (depth-grid edges) and point cloud.

Honest limit: one depth map is a 2.5-D sheet. Hard silhouettes stretch a little
under motion (there is no real backside to a single photo). Surface connectivity
makes that a soft stretch rather than a hole, and small moves hide it. Filling
the true backside would need a model, which we deliberately do not add.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ASPECTS: dict[str, float] = {
    "Square (1:1)": 1.0,
    "Portrait (4:5)": 4 / 5,
    "Portrait (9:16)": 9 / 16,
    "Landscape (16:9)": 16 / 9,
}

VIEW_MODES = ("Solid", "Wireframe", "Point cloud")

CAM_DIST = 3.2          # base camera distance, scene half-height ~1
FOV_DEG = 34.0          # base vertical field of view
RELIEF = 0.85           # how far depth pushes points in Z at depth_intensity 1


# Camera presets: loop phase p -> (yaw, pitch, dolly, focal_mul). Angles in
# radians, kept small so the 2.5-D sheet does not stretch. dolly adds to the
# camera distance; focal_mul zooms the lens (used for the real dolly-zoom).
def _loop(p: float) -> float:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * p)


def _float(p):
    a = 2 * np.pi * p
    return 0.10 * np.sin(a), 0.06 * np.sin(a + 1.3) + 0.02, 0.0, 1.0


def _orbit(p):
    a = 2 * np.pi * p
    return 0.16 * np.sin(a), 0.10 * np.cos(a), 0.0, 1.0


def _sway(p):
    return 0.18 * np.sin(2 * np.pi * p), 0.0, 0.0, 1.0


def _push(p):
    return 0.0, 0.0, -0.7 * _loop(p), 1.0            # move the camera in


def _dolly(p):
    # True Vertigo: pull the camera back while zooming the lens in, so the
    # subject holds size and the background compresses. This is the real thing,
    # correct now that there is an actual camera.
    e = _loop(p)
    return 0.0, 0.0, 1.4 * e, 1.0 + 1.3 * e


PRESETS: dict[str, callable] = {
    "Float": _float,
    "Orbit": _orbit,
    "Sway": _sway,
    "Push in": _push,
    "Dolly zoom": _dolly,
}


@dataclass
class CreativeSettings:
    aspect: str = "Portrait (4:5)"
    preset: str = "Float"
    view: str = "Solid"
    depth_intensity: float = 1.0          # 0..2, 3-D relief + camera swing

    # Fog (depth-based haze). plane 0 = everywhere, 1 = only the deepest.
    fog: float = 0.30
    fog_plane: float = 0.5
    fog_color: tuple[int, int, int] = (150, 156, 168)

    # Flare: bloom on highlights (0..1).
    flare: float = 0.25

    # Embers: warm motes. dir degrees (0 up, 90 right, 180 down, 270 left).
    embers: float = 0.0
    ember_plane: float = 0.5
    ember_dir: float = 0.0
    ember_speed: float = 0.4
    ember_size: float = 0.5

    dust: float = 0.30
    dust_plane: float = 0.5
    dust_dir: float = 90.0
    dust_speed: float = 0.25
    dust_size: float = 0.4

    frames: int = 48
    fps: int = 24


def reframe(image_rgb: np.ndarray, inv_depth: np.ndarray, aspect: float
            ) -> tuple[np.ndarray, np.ndarray]:
    """Centre-crop image + depth to the target aspect ratio."""
    h, w = image_rgb.shape[:2]
    if w / h > aspect:
        nw = int(round(h * aspect)); x0 = (w - nw) // 2
        sl = (slice(None), slice(x0, x0 + nw))
    else:
        nh = int(round(w / aspect)); y0 = (h - nh) // 2
        sl = (slice(y0, y0 + nh), slice(None))
    return np.ascontiguousarray(image_rgb[sl]), np.ascontiguousarray(inv_depth[sl])


def _look_at(cam, target, up=(0.0, 1.0, 0.0)):
    """Return camera basis (right, up, forward). forward points at the target."""
    f = target - cam
    f = f / (np.linalg.norm(f) + 1e-9)
    up = np.asarray(up, np.float32)
    r = np.cross(f, up); r = r / (np.linalg.norm(r) + 1e-9)
    u = np.cross(r, f)
    return r, u, f


class Renderer:
    """Depth -> 3-D point-surface, rendered through a perspective camera."""

    def __init__(self, image_rgb: np.ndarray, inv_depth: np.ndarray):
        img = image_rgb.astype(np.float32)
        if img.max() > 1.5:
            img /= 255.0
        h, w = img.shape[:2]
        self.h, self.w = h, w

        d = inv_depth.astype(np.float32)
        d = (d - d.min()) / (np.ptp(d) + 1e-6)          # 0 far .. 1 near
        self.near_map = cv2.GaussianBlur(d, (0, 0), max(1.0, w / 400.0))

        # Build the surface point grid. Subsample to a budget so projection and
        # splatting stay interactive; the render buffer is still full size.
        budget = 150_000
        step = max(1, int(round(np.sqrt(w * h / budget))))
        self.step = step
        ys = np.arange(0, h, step)
        xs = np.arange(0, w, step)
        gx, gy = np.meshgrid(xs, ys)
        self.gh, self.gw = gx.shape
        near = self.near_map[gy, gx]
        # Plane coords: x in [-aspect, aspect], y in [-1, 1], z from depth.
        ax = w / h
        px = (gx / w - 0.5) * 2.0 * ax
        py = -(gy / h - 0.5) * 2.0
        self._px, self._py = px.astype(np.float32), py.astype(np.float32)
        self._near = near.astype(np.float32)
        self.colors = img[gy, gx].reshape(-1, 3).astype(np.float32)
        self.ax = ax
        self._rng = np.random.default_rng(7)
        self._parts: dict[str, np.ndarray] = {}

    # -- camera --------------------------------------------------------------

    def _camera(self, yaw, pitch, dolly, focal_mul):
        dist = CAM_DIST + dolly
        cam = np.array([
            dist * np.sin(yaw) * np.cos(pitch),
            dist * np.sin(pitch),
            dist * np.cos(yaw) * np.cos(pitch),
        ], np.float32)
        r, u, f = _look_at(cam, np.zeros(3, np.float32))
        fpx = 0.5 * self.h / np.tan(np.deg2rad(FOV_DEG) / 2.0) * focal_mul
        return cam, r, u, f, fpx

    def _project(self, X, Y, Z, cam):
        """World points -> (sx, sy, zc). zc is distance in front of the camera."""
        campos, r, u, f, fpx = cam
        rx = X - campos[0]; ry = Y - campos[1]; rz = Z - campos[2]
        xc = rx * r[0] + ry * r[1] + rz * r[2]
        yc = rx * u[0] + ry * u[1] + rz * u[2]
        zc = rx * f[0] + ry * f[1] + rz * f[2]
        zc = np.where(np.abs(zc) < 1e-3, 1e-3, zc)
        sx = self.w * 0.5 + fpx * xc / zc
        sy = self.h * 0.5 - fpx * yc / zc
        return sx, sy, zc

    def _surface_z(self, di):
        return (self._near - 0.5) * RELIEF * di

    # -- particles (3-D volume) ---------------------------------------------

    def _ensure_parts(self, kind, n):
        key = f"{kind}:{n}"
        if key not in self._parts:
            r = self._rng
            p = np.zeros((n, 8), np.float32)
            p[:, 0] = r.uniform(-self.ax, self.ax, n)   # x
            p[:, 1] = r.uniform(-1.0, 1.0, n)           # y
            p[:, 2] = r.uniform(-0.3, 0.3, n)           # z offset around plane
            p[:, 3:7] = r.uniform(0, 1, (n, 4))         # turbulence phases
            p[:, 7] = r.uniform(0.6, 1.4, n)            # size jitter
            self._parts[key] = p
        return self._parts[key]

    def _draw_particles(self, canvas, zbuf, kind, amount, plane, dir_deg,
                        speed, size, phase, cam, di):
        if amount <= 0:
            return
        n = int(amount * (260 if kind == "dust" else 190))
        if n <= 0:
            return
        p = self._ensure_parts(kind, n)
        t = phase; tau = 2 * np.pi
        x0, y0, zoff = p[:, 0], p[:, 1], p[:, 2]
        ph1, ph2, ph3, ph4, sj = p[:, 3], p[:, 4], p[:, 5], p[:, 6], p[:, 7]
        # Drift in the image plane, direction in degrees (0 up).
        th = np.deg2rad(dir_deg)
        trav = 1.6 * speed
        vx, vy = np.sin(th) * trav, np.cos(th) * trav
        swirl = (0.14 if kind == "embers" else 0.05) * (0.5 + sj)
        ox = swirl * np.sin((t + ph1) * tau) + 0.5 * swirl * np.sin((2 * t + ph2) * tau)
        oy = swirl * np.cos((t + ph1) * tau) + 0.5 * swirl * np.cos((2 * t + ph3) * tau)
        X = ((x0 + vx * t + ox + self.ax) % (2 * self.ax)) - self.ax
        Y = ((y0 + vy * t + oy + 1.0) % 2.0) - 1.0
        # Z: sit around the chosen plane in the same range the surface spans.
        Zc = ((plane - 0.5) * RELIEF * di) + zoff * RELIEF
        sx, sy, zc = self._project(X, Y, Zc, cam)
        fpx = cam[4]
        # Screen radius of a small world-space mote at distance zc: fpx*size/zc.
        # Near motes are naturally bigger, far ones sub-pixel. Clamped so a mote
        # that drifts close to the camera cannot balloon over the frame.
        world = (0.003 + size * 0.03) * sj
        with np.errstate(divide="ignore", invalid="ignore"):
            rad = np.clip(fpx * world / np.maximum(zc, 1e-3), 0.0, 48.0)
        twin = 0.55 + 0.45 * np.sin((2 * t + ph4) * tau)
        col = (np.array([1.0, 0.55, 0.18], np.float32) if kind == "embers"
               else np.array([0.85, 0.9, 1.0], np.float32))
        glow = kind == "embers"
        layer = np.zeros((self.h, self.w), np.float32)
        xi = sx.astype(int); yi = sy.astype(int)
        ok = (xi >= 0) & (xi < self.w) & (yi >= 0) & (yi < self.h) & (zc > 0)
        # Occlude against the surface depth buffer (with a little tolerance).
        ok &= zc <= zbuf[np.clip(yi, 0, self.h - 1), np.clip(xi, 0, self.w - 1)] + 0.04
        for k in np.nonzero(ok)[0]:
            r = int(round(float(rad[k])))
            b = float(twin[k])
            xk, yk = int(sx[k]), int(sy[k])
            if r < 1:
                layer[yk, xk] += b
            else:
                cv2.circle(layer, (xk, yk), r, b, -1, cv2.LINE_AA)
        sigma = max(0.6, self.w / 800.0) * (1.3 if glow else 1.0)
        layer = cv2.GaussianBlur(layer, (0, 0), sigma)
        canvas += layer[..., None] * col * (1.5 if glow else 0.7)

    # -- one frame -----------------------------------------------------------

    def frame(self, s: CreativeSettings, index: int) -> np.ndarray:
        phase = index / s.frames
        yaw, pitch, dolly, fmul = PRESETS[s.preset](phase)
        di = s.depth_intensity
        cam = self._camera(yaw * di, pitch * di, dolly, fmul)

        Z = self._surface_z(di)
        sx, sy, zc = self._project(self._px, self._py, Z, cam)

        bg = np.array([0.04, 0.055, 0.085], np.float32)
        canvas = np.tile(bg, (self.h, self.w, 1))
        zbuf = np.full((self.h, self.w), np.inf, np.float32)

        if s.view == "Wireframe":
            self._render_wire(canvas, sx, sy, zc, zbuf)
        elif s.view == "Point cloud":
            self._render_points(canvas, sx, sy, zc, zbuf, radius=0)
        else:
            self._render_points(canvas, sx, sy, zc, zbuf, radius=1)

        if s.fog > 0:
            self._apply_fog(canvas, zbuf, s)
        self._draw_particles(canvas, zbuf, "dust", s.dust, s.dust_plane,
                             s.dust_dir, s.dust_speed, s.dust_size, phase, cam, di)
        self._draw_particles(canvas, zbuf, "embers", s.embers, s.ember_plane,
                             s.ember_dir, s.ember_speed, s.ember_size, phase, cam, di)
        if s.flare > 0:
            self._apply_flare(canvas, s.flare)
        return np.clip(canvas, 0.0, 1.0)

    # -- render modes --------------------------------------------------------

    def _render_points(self, canvas, sx, sy, zc, zbuf, radius):
        sx = sx.ravel(); sy = sy.ravel(); zc = zc.ravel()
        xi = sx.astype(np.int32); yi = sy.astype(np.int32)
        ok = (xi >= 0) & (xi < self.w) & (yi >= 0) & (yi < self.h) & (zc > 0)
        xi, yi, z = xi[ok], yi[ok], zc[ok]
        cols = self.colors[ok]
        order = np.argsort(-z)                          # far first, near overwrites
        xi, yi, z, cols = xi[order], yi[order], z[order], cols[order]
        flat = canvas.reshape(-1, 3)
        zf = zbuf.reshape(-1)
        offs = ([(0, 0)] if radius == 0 else
                [(dx, dy) for dy in range(-radius, radius + 1)
                 for dx in range(-radius, radius + 1)])
        for dx, dy in offs:
            xx = np.clip(xi + dx, 0, self.w - 1)
            yy = np.clip(yi + dy, 0, self.h - 1)
            idx = yy * self.w + xx
            flat[idx] = cols
            zf[idx] = z

    def _render_wire(self, canvas, sx, sy, zc, zbuf):
        # Fill a faint solid first so the wireframe reads against the form, then
        # stroke a coarse subset of the depth grid's edges.
        self._render_points(canvas, sx, sy, zc, zbuf, radius=1)
        canvas *= 0.35
        gw, gh = self.gw, self.gh
        SX = sx.reshape(gh, gw); SY = sy.reshape(gh, gw); ZC = zc.reshape(gh, gw)
        stepx = max(1, gw // 48); stepy = max(1, gh // 48)
        cwire = (0.35, 0.82, 1.0)
        for j in range(0, gh, stepy):
            pts = np.stack([SX[j, ::stepx], SY[j, ::stepx]], 1)
            vis = ZC[j, ::stepx] > 0
            self._polyline(canvas, pts, vis, cwire)
        for i in range(0, gw, stepx):
            pts = np.stack([SX[::stepy, i], SY[::stepy, i]], 1)
            vis = ZC[::stepy, i] > 0
            self._polyline(canvas, pts, vis, cwire)

    def _polyline(self, canvas, pts, vis, color):
        for k in range(len(pts) - 1):
            if not (vis[k] and vis[k + 1]):
                continue
            a = (int(pts[k, 0]), int(pts[k, 1]))
            b = (int(pts[k + 1, 0]), int(pts[k + 1, 1]))
            cv2.line(canvas, a, b, color, 1, cv2.LINE_AA)

    # -- atmosphere ----------------------------------------------------------

    def _apply_fog(self, canvas, zbuf, s):
        z = zbuf.copy()
        finite = np.isfinite(z)
        if not finite.any():
            return
        lo, hi = np.percentile(z[finite], [2, 98])
        far = np.clip((z - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        far[~finite] = 1.0                              # empty space reads as far
        p0 = float(s.fog_plane)
        band = np.clip((far - p0) / (1.0 - p0 + 1e-3), 0.0, 1.0) ** 1.3
        fog = band[..., None] * (s.fog * 0.9)
        fc = np.array(s.fog_color, np.float32) / 255.0
        canvas *= (1.0 - fog)
        canvas += fc * fog

    def _apply_flare(self, canvas, amount):
        # Bloom: isolate the brightest highlights only, blur wide, add a little
        # back. Kept gentle: a glow around bright spots, never a wash.
        lum = canvas.mean(2)
        mask = np.clip((lum - 0.82) / 0.18, 0.0, 1.0)[..., None] * canvas
        sigma = max(2.0, self.w / 110.0)
        bloom = cv2.GaussianBlur(mask, (0, 0), sigma)
        canvas += bloom * (amount * 0.9)

    def render_loop(self, s: CreativeSettings) -> list[np.ndarray]:
        return [self.frame(s, i) for i in range(s.frames)]
