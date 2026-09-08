"""Depth-driven point-cloud reveal.

The neural pass gives no intermediate image, so "working…" needs its own visual.
Instead of the old grey→colour sweep, this reconstructs the photo as a real 3D
point cloud — each pixel placed at its depth in Z — and slowly orbits the camera
so the foreground genuinely swings past the background (true perspective
parallax, not a 2D fake). A scan pass rides top→bottom on a loop, lifting the
band it crosses toward the camera. The result reads as a sci-fi depth
reconstruction rather than a flat image shattered into tiles.

Kept as pure numpy (no Qt) so it renders offline for tuning and unit tests; the
widget converts the returned RGB buffer to a QImage and blits it in one call,
which is the only way this stays cheap at animation rate.
"""

from __future__ import annotations

import numpy as np

#: Points across the widest edge. Kept sparse on purpose: a dense wall reads as a
#: broken image, a sparser cloud reads as 3D structure you can see through.
_POINTS_ACROSS = 150

#: Reveal accent (cyan), matching the app's signal colour, ridden by the scan.
_ACCENT = np.asarray([90, 220, 255], np.float32)

#: Near-black ground the cloud sits on.
_BG = np.asarray([4, 6, 10], np.float32)

#: Half-width of the scan pass as a fraction of image height.
_SCAN_WIDTH = 0.14

#: Scene geometry, in arbitrary world units. EXTRUDE is how far apart near and
#: far pixels sit in Z — the whole point of the effect, so it is large; CAM_DIST
#: keeps the cloud comfortably in front of the camera; YAW_AMP is how far the
#: orbit swings (radians ≈ 23°), which is what makes the parallax obvious.
_EXTRUDE = 1.7
_CAM_DIST = 1.8
_FOCAL = 1.3
#: Held orbit: rotate right by this much (a little less than before), and tilt
#: the view slightly *up* (negative pitch) — the downward tilt read unpleasant.
_YAW_AMP = 0.30
_PITCH = -0.14


def render_point_cloud_frame(
    colour_rgb: np.ndarray,
    depth: np.ndarray,
    out_w: int,
    out_h: int,
    scan: float,
    phase: float = 0.0,
    flatten: float = 0.0,
    ripple: float = 0.0,
) -> np.ndarray:
    """One frame of the reveal, as an (out_h, out_w, 3) uint8 RGB buffer.

    ``colour_rgb`` is the source photo (H×W×3, uint8) and ``depth`` its inverse
    depth (H×W, 0..1, near = 1). ``scan`` in [0, 1] is the vertical centre of the
    top→bottom scan pass; ``phase`` (radians) drives the camera orbit. ``flatten``
    in [0, 1] eases the whole cloud back to a head-on flat grid (orbit, pitch and
    depth all collapse to zero) — the landing that resolves into the final image.
    ``ripple`` in [0, 1] adds a horizontal wave that shifts the points side to
    side inside the scan band, so the pulse displaces pixels rather than only
    lighting them — kept off (0) during a landing so it stays pixel-aligned.
    """
    out_w = max(1, int(out_w))
    out_h = max(1, int(out_h))
    frame = np.empty((out_h, out_w, 3), np.float32)
    frame[:] = _BG

    if colour_rgb.size == 0 or depth.size == 0:
        return frame.astype(np.uint8)

    src_w = depth.shape[1]
    step = max(1, src_w // _POINTS_ACROSS)
    col = colour_rgb[::step, ::step, :3].astype(np.float32)
    dz = np.clip(depth[::step, ::step].astype(np.float32), 0.0, 1.0)
    gh, gw = dz.shape

    gx = np.broadcast_to(np.linspace(0.0, 1.0, gw), (gh, gw))
    gy = np.broadcast_to(np.linspace(0.0, 1.0, gh)[:, None], (gh, gw))
    aspect = out_w / out_h

    # World coordinates: the image plane centred on the origin, pushed back into
    # Z by (1 - depth) so near pixels (dz→1) sit close to the camera and far ones
    # recede. This Z spread is what a 2D grid never had.
    wx = (gx - 0.5) * aspect
    wy = (0.5 - gy)
    wz = (1.0 - dz) * _EXTRUDE

    # The scan pass lifts its band toward the camera (reduces Z), and brightens
    # it — a travelling ridge running down the cloud.
    push = np.exp(-(((gy - float(scan)) / _SCAN_WIDTH) ** 2))
    wz = wz - push * (_EXTRUDE * 0.22)

    # Landing: ease the depth extrusion (and, below, the orbit) to zero so the
    # cloud collapses head-on into a flat grid that matches the final image.
    flat = float(np.clip(flatten, 0.0, 1.0))
    ease = flat * flat * (3.0 - 2.0 * flat)  # smoothstep, so it settles softly
    wz = wz * (1.0 - ease)

    # Camera move: turn once to a held angle, then back — no oscillation. The
    # angle is tied to (1 - ease): zero when flat, full through the orbit hold,
    # eased back to zero as it lands. A fixed downward pitch rakes the plane away.
    cz = _EXTRUDE * 0.5
    turn = 1.0 - ease
    yaw = _YAW_AMP * turn
    cy_, sy_ = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(_PITCH * turn), np.sin(_PITCH * turn)
    zc = wz - cz
    xr = wx * cy_ + zc * sy_
    zr = -wx * sy_ + zc * cy_
    yr = wy * cp - zr * sp
    zr = wy * sp + zr * cp
    z_cam = zr + cz + _CAM_DIST

    valid = z_cam > 0.05
    inv_z = np.where(valid, 1.0 / np.maximum(z_cam, 0.05), 0.0)

    # Perspective projection, with the scale derived so the near plane fills the
    # frame edge-to-edge.
    scale = out_h * ((cz + _CAM_DIST) / _FOCAL)
    cx = (out_w - 1) * 0.5
    cyc = (out_h - 1) * 0.5
    px = cx + _FOCAL * xr * inv_z * scale
    py = cyc - _FOCAL * yr * inv_z * scale

    # Land exactly onto the final image: as it flattens, blend the perspective
    # projection into a plain full-frame grid, so the dissipating cloud is
    # pixel-aligned with the result it fades into — no zoom pop at the seam.
    px_flat = gx * (out_w - 1)
    py_flat = gy * (out_h - 1)
    px_final = px * (1.0 - ease) + px_flat * ease
    py_final = py * (1.0 - ease) + py_flat * ease

    # Ripple: inside the scan band the points wave — mostly side to side, with a
    # smaller vertical wobble — travelling down with the band, so the pulse
    # displaces pixels rather than only lighting them. Tied to `push` so it is
    # zero away from the band.
    rip = float(np.clip(ripple, 0.0, 1.0))
    if rip > 0.0:
        s2 = float(scan) * (2.0 * np.pi * 2.0)
        px_final = px_final + (rip * out_w * 0.02) * push * np.sin(
            gy * (2.0 * np.pi * 3.0) - s2
        )
        py_final = py_final + (rip * out_h * 0.012) * push * np.cos(
            gx * (2.0 * np.pi * 2.5) - s2
        )

    ox = np.rint(px_final).astype(np.int32)
    oy = np.rint(py_final).astype(np.int32)

    # Near points brighter/warmer, far dimmer/cooled; the scan brightens its band.
    # Lifted overall so single-pixel dots stay visible even on a dark night frame.
    bright = ((0.55 + 0.85 * dz) * (1.0 + 1.1 * push))[..., None]
    cool = np.stack([np.ones_like(dz), 0.95 + 0.05 * dz, 0.85 + 0.15 * dz], axis=-1)
    rgb = col * bright * cool
    # The scan crest reads as a travelling ridge of light, and a faint cyan wash
    # rides the whole band so the pulse is clear even across dark regions.
    glow = (push * push)[..., None]
    rgb = rgb * (1.0 - 0.5 * glow) + _ACCENT * 0.8 * glow
    rgb = rgb + _ACCENT * (0.12 * push)[..., None]

    flat_valid = valid.ravel()
    ox_f = np.clip(ox.ravel()[flat_valid], 0, out_w - 1)
    oy_f = np.clip(oy.ravel()[flat_valid], 0, out_h - 1)
    rgb_f = rgb.reshape(-1, 3)[flat_valid]
    z_f = z_cam.ravel()[flat_valid]
    if ox_f.size == 0:
        return frame.astype(np.uint8)
    # Every point is a single pixel, near or far — a true point cloud, not splats.
    # Drawn far→near (numpy keeps the last write at a duplicated index) so nearer
    # points win the pixel where two land on the same spot.
    order = np.argsort(-z_f)
    frame[oy_f[order], ox_f[order]] = rgb_f[order]

    return np.clip(frame, 0.0, 255.0).astype(np.uint8)
