"""Scene from shots, main-shot style: one hero shot, its neighbours as surroundings.

The user picks the main shot. It is reconstructed with SHARP exactly as the
single-image 3D tab does, so near its camera the scene looks like that
photo. The shots taken around it are reconstructed too, but only to fill in
what the main shot cannot see: past its frame edges and behind the things it
looks past. Their splats never cover the main shot's view.

Everything is merged once into one static splat scene. Nothing switches while
the camera moves (switching shots per frame is what made the per-pixel
approach pop), and it renders with the ordinary splat renderer, effects and
all.

Qt-free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import measured, multishot, shotsolve

#: Neighbours on each side of the main shot, in capture order, that are
#: solved and used for the surroundings.
WINDOW = 8

#: A neighbour splat within this fraction of the main shot's surface (or in
#: front of it) is something the main shot already shows: dropped.
HERO_MARGIN = 0.03

#: Neighbour splats snap to the game's depth; ones SHARP placed more than
#: this far from it are hidden guesses that would float in the open.
SNAP = 0.15

#: One neighbour per cell of this size (renderer units): the nearest to the
#: main shot keeps it, so neighbours never stack copies of the same surface.
CELL = 0.03


class Cancelled(Exception):
    pass


def _window(paths: list[Path], main: Path) -> list[Path]:
    i = [str(p) for p in paths].index(str(main))
    return paths[max(0, i - WINDOW):i + WINDOW + 1]


def _sharp_splats(scene, k: int, hero: bool):
    """SHARP for one shot, placed by the game's depth, in the capture world.
    Returns (positions, colours 0..1, opacity, covariances 3x3)."""
    from . import sharp3d

    name = scene.names[k]
    full = cv2.cvtColor(cv2.imdecode(np.fromfile(name, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    H, W = full.shape[:2]
    K = scene.cameras[k]
    means, sizes, quats, linear, alpha = sharp3d.predict(full)
    alpha = alpha.ravel()
    zmap = scene.depth(k, (W, H))
    nx, ny = means[:, 0] / means[:, 2], means[:, 1] / means[:, 2]
    u = ((nx + 1) * 0.5 * W).astype(int)
    v = ((ny + 1) * 0.5 * H).astype(int)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    zg = np.full(len(means), np.nan)
    zg[inside] = zmap[v[inside], u[inside]]
    zg[zg >= measured.SKY / 2] = np.nan
    fit = inside & np.isfinite(zg) & (alpha > 0.3)
    if fit.sum() < 100:
        return None
    k_scale = float(np.median(zg[fit] / means[fit, 2]))
    stretch = np.array([(W / 2) / K[0, 0] * k_scale, (H / 2) / K[1, 1] * k_scale, k_scale])
    cam = means * stretch
    # Snap each splat along its own ray onto the game's surface. The main
    # shot keeps its hidden layers (SHARP's guess of what is behind things)
    # at SHARP's own depth; for neighbours those would float in the open.
    ratio = zg / np.maximum(cam[:, 2], 1e-6)
    on = np.isfinite(ratio) & (np.abs(ratio - 1) < SNAP)
    ratio = np.where(on, ratio, 1.0)
    cam = cam * ratio[:, None]
    keep = (alpha > 0.05) & (on | hero)
    qw, qx, qy, qz = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    rot = np.stack([
        np.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)], -1),
        np.stack([2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)], -1),
        np.stack([2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)], -1)], 1)
    cov = rot @ (sizes[:, :, None] ** 2 * np.transpose(rot, (0, 2, 1)))
    cov = cov * (stretch[None, :, None] * stretch[None, None, :]) * (ratio ** 2)[:, None, None]
    R, C = scene.rotations[k], scene.centres[k]
    world = C + cam[keep] @ R
    cov_w = R.T[None] @ cov[keep] @ R[None]
    # Colour in linear light, with this shot's auto-exposure undone.
    lin = np.clip(linear[keep], 0, None) * scene.gain[k][None, :]
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return world, np.clip(srgb, 0, 1), alpha[keep], cov_w


def build(paths, main, excluded=None, progress: Callable[[str], None] = lambda _m: None,
          cancelled: Callable[[], bool] = lambda: False):
    """Returns (SplatScene | None, Report)."""
    from . import sharp3d, splat3d

    excluded = excluded or set()
    paths = [Path(p) for p in paths if str(p) not in excluded]
    main = Path(main) if main else paths[len(paths) // 2]
    if str(main) not in [str(p) for p in paths]:
        main = paths[len(paths) // 2]
    if not sharp3d.is_downloaded():
        report = multishot.Report()
        report.advice = "Download SHARP (High quality) first: each shot is rebuilt with it."
        return None, report
    window = _window(paths, main)
    try:
        shots, report, _moved = shotsolve.build(window, set(), progress, cancelled)
    except shotsolve.Cancelled:
        raise Cancelled from None
    if shots is None:
        return None, report
    names = [str(n) for n in shots.names]
    if str(main) not in names or not shots.used[names.index(str(main))]:
        report.advice = ("The main shot could not be lined up with its neighbours. Pick another main "
                         "shot, or take the shots around it in smaller steps.")
        return None, report
    h = names.index(str(main))

    # The renderer world is the main shot's camera: it sits at the origin
    # looking down -z, and its typical depth is TARGET_DEPTH, as a single-image
    # scene. The 3D tab's default view is then exactly the main shot.
    R_h, C_h = shots.rotations[h], shots.centres[h]
    fw, fh = shots.size
    z_h = shots.depth(h, (fw, fh))
    typical = float(np.median(z_h[z_h < measured.SKY / 2]))
    scale = multishot.TARGET_DEPTH / max(typical, 1e-6)
    axes = np.stack([R_h[0], -R_h[1], -R_h[2]])

    def to_r(p):
        return (p - C_h) @ axes.T * scale

    K_h = shots.cameras[h]
    order = [h] + sorted((k for k in np.flatnonzero(shots.used) if k != h),
                         key=lambda k: np.linalg.norm(shots.centres[k] - C_h)
                         + np.arccos(np.clip(shots.rotations[k][2] @ R_h[2], -1, 1)))
    parts, occupied = [], set()
    for n, k in enumerate(order, 1):
        if cancelled():
            raise Cancelled
        progress(f"Rebuilding shots with SHARP… {n}/{len(order)}")
        got = _sharp_splats(shots, int(k), hero=(k == h))
        if got is None:
            continue
        world, colour, alpha, cov = got
        if k != h:
            # Only what the main shot cannot see: outside its frame, or behind
            # the surface it shows.
            c = (world - C_h) @ R_h.T
            zc = np.where(c[:, 2] > 1e-6, c[:, 2], 1e-6)
            u = K_h[0, 0] * c[:, 0] / zc + K_h[0, 2]
            v = K_h[1, 1] * c[:, 1] / zc + K_h[1, 2]
            seen = (c[:, 2] > 0) & (u >= 0) & (u < fw) & (v >= 0) & (v < fh)
            zs = np.full(len(world), np.inf)
            zs[seen] = z_h[v[seen].astype(int), u[seen].astype(int)]
            covered = seen & (c[:, 2] < zs * (1 + HERO_MARGIN))
            keep = ~covered
            # And not what a nearer neighbour already filled.
            pr = to_r(world)
            cells = np.floor(pr / CELL).astype(np.int64)
            key = (cells[:, 0] * 73856093) ^ (cells[:, 1] * 19349663) ^ (cells[:, 2] * 83492791)
            taken = np.fromiter((x in occupied for x in key.tolist()), bool, len(key))
            keep &= ~taken
            occupied.update(key[keep & (alpha > 0.5)].tolist())
            world, colour, alpha, cov = world[keep], colour[keep], alpha[keep], cov[keep]
        pr = to_r(world)
        cr = (axes[None] @ cov @ axes.T[None]) * scale * scale
        parts.append((pr.astype(np.float32), colour.astype(np.float32), alpha.astype(np.float32),
                      np.stack([cr[:, 0, 0], cr[:, 0, 1], cr[:, 0, 2], cr[:, 1, 1], cr[:, 1, 2], cr[:, 2, 2]],
                               -1).astype(np.float32)))
        if k == h:
            n_front = len(pr)
    positions = np.concatenate([p[0] for p in parts])
    colours = np.concatenate([p[1] for p in parts])
    opacity = np.concatenate([p[2] for p in parts])
    covs = np.concatenate([p[3] for p in parts])
    far = np.linalg.norm(positions, axis=1) < 55.0
    planes = []
    for nrm, c in shots.planes:
        # Planes came in the solver's own world; carry them into this one.
        p0 = -c * nrm
        q_world = (p0 - shots.offset) / shots.scale @ shots.axes + shots.target
        n_world = shots.axes.T @ nrm
        n_r = axes @ n_world
        planes.append((n_r, float(-n_r @ to_r(q_world[None])[0])))
    scene = splat3d.SplatScene(positions[far], colours[far], opacity[far], covs[far],
                               int(min(n_front, far.sum())), photo_z=None,
                               focal=float(K_h[0, 0]) * multishot.WORK_WIDTH / fw, planes=planes)
    report.sharp = True
    report.key_shots = len(parts)
    img = cv2.cvtColor(cv2.imdecode(np.fromfile(str(main), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    report.root_image = cv2.resize(img, (multishot.WORK_WIDTH, round(fh * multishot.WORK_WIDTH / fw)),
                                   interpolation=cv2.INTER_AREA)
    report.advice = (report.advice + " " if report.advice else "") + (
        f"Main shot {Path(main).stem.split('_')[-1]}; {len(parts) - 1} shots around it fill the surroundings.")
    return scene, report
