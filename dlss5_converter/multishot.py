"""Scene from shots: turn a folder of game captures into one 3D splat scene.

This is the glue between the capture side (the DLSS5 Scene Capture ReShade
add-on, or plain screenshots) and the 3D tab's renderer. It finds the shooting
sessions in a folder, places every shot (``measured`` when the shots carry the
game's real depth, ``multiview``/``atoms`` with estimated depth otherwise),
fuses them, and hands back a ``splat3d.SplatScene`` the 3D tab draws like any
other, plus a per-shot report the UI shows as badges.

No Qt here, so it can be tested and run headless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Callable

import cv2
import numpy as np

from . import game_depth, measured
from . import multiview as mv

SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

#: Shots further apart in time than this belong to different sessions. A
#: capture pass has gaps of seconds to a minute between shots (the Stray room
#: had 20 s gaps); two different shoots in the same folder are minutes apart.
SESSION_GAP_SECONDS = 300.0

#: Working width. Depth, cameras and fusion run at this size; atoms are found
#: at full resolution (see ``measured.describe``).
WORK_WIDTH = 1600

#: Upper bound on splats handed to the renderer. A single-image scene is about
#: this size, so a multi-shot scene draws at the same speed.
MAX_SPLATS = 3_000_000

#: Where the root shot's typical depth lands in the renderer's space. The 3D
#: tab's camera and pivot are laid out for scenes about this far away.
TARGET_DEPTH = 3.5

#: One shot in this many gets SHARP. A third was the first test (11 of 33 on
#: the Stray room): the room closed up from every side and SHARP added about a
#: minute. Every shot still helps place the cameras.
KEY_EVERY = 3

#: SHARP splats kept per key shot must sit within this fraction of the game's
#: measured depth. SHARP fills what a photo cannot see with long streaks that
#: look right from that photo's camera and smear from anywhere else; with
#: eleven shots every view was covered in other shots' streaks. Keeping only
#: surface splats lets the other shots cover those areas with real views.
SURFACE_TOLERANCE = 0.06

#: Largest splat kept, in pixels at its own distance, for the same reason.
MAX_SPLAT_PIXELS = 6.0

#: Ownership cell size in the renderer's units: where several key shots cover
#: the same cell, the closest shot's splats win and the others are dropped.
OWNER_CELL = 0.02


# --- sessions ------------------------------------------------------------------

@dataclass
class Session:
    """One shooting pass: shots taken close together in time."""

    paths: list[Path]
    started: float                   # epoch seconds of the first shot
    with_depth: int                  # shots that carry the game's depth

    @property
    def game(self) -> str:
        # The add-on names files "<game exe> <date> <time>_<n>.png".
        stem = self.paths[0].stem
        return stem.split(" 20")[0] if " 20" in stem else stem

    @property
    def label(self) -> str:
        when = time.strftime("%d %b %H:%M", time.localtime(self.started))
        depth = (f"{self.with_depth} with game depth" if self.with_depth
                 else "no game depth")
        return f"{self.game}, {when}: {len(self.paths)} shots, {depth}"


def find_sessions(folder: str | Path) -> list[Session]:
    """Group a folder's images into sessions, newest first."""
    folder = Path(folder)
    images = [p for p in folder.iterdir() if p.suffix.lower() in SUFFIXES and p.is_file()]
    if not images:
        return []
    images.sort(key=lambda p: (p.stat().st_mtime, p.name))
    sessions: list[list[Path]] = [[images[0]]]
    for previous, path in zip(images, images[1:]):
        if path.stat().st_mtime - previous.stat().st_mtime > SESSION_GAP_SECONDS:
            sessions.append([])
        sessions[-1].append(path)
    out = [Session(group, group[0].stat().st_mtime,
                   sum(1 for p in group if game_depth.sidecars(p)[0].is_file()))
           for group in sessions]
    out.sort(key=lambda s: s.started, reverse=True)
    return out


# --- per-shot report ---------------------------------------------------------------

PLACED = "placed"
NO_DEPTH = "no_depth"
NO_OVERLAP = "no_overlap"
EXCLUDED = "excluded"
UNREADABLE = "unreadable"

#: What the UI shows under each thumbnail, and in its tooltip.
STATUS_TEXT = {
    PLACED: ("Placed", "This shot is part of the scene."),
    NO_DEPTH: ("No depth", "This shot has no game depth (or it was empty). Capture it "
               "again with the DLSS5 Scene Capture add-on."),
    NO_OVERLAP: ("No overlap", "Not enough of this shot matches the others to place it. "
                 "Take shots in smaller steps so each shares most of its view with a "
                 "neighbour."),
    EXCLUDED: ("Excluded", "You left this shot out. Click it to include it again."),
    UNREADABLE: ("Unreadable", "This image could not be opened."),
}


@dataclass
class Report:
    statuses: dict[str, str] = field(default_factory=dict)   # path -> status
    fov_degrees: float = 0.0
    agreement: float = 0.0          # mean dense surface agreement of the links used
    estimated: bool = False         # built from estimated depth (no game depth)
    advice: str = ""
    root_image: object = None       # the shot whose camera the scene is seen from
    sharp: bool = False             # built from SHARP key shots
    key_shots: int = 0
    saved: str = ""                 # folder the scene was saved to

    @property
    def placed(self) -> int:
        return sum(1 for s in self.statuses.values() if s == PLACED)

    @property
    def considered(self) -> int:
        return sum(1 for s in self.statuses.values() if s != EXCLUDED)


class Cancelled(Exception):
    pass


# --- building ---------------------------------------------------------------------

def _load(paths, report: Report, progress, cancelled):
    shots, sources = [], []
    for n, path in enumerate(paths, 1):
        if cancelled():
            raise Cancelled
        progress(f"Reading shots… {n}/{len(paths)}")
        raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if raw is None:
            report.statuses[str(path)] = UNREADABLE
            continue
        full = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        height = round(full.shape[0] * WORK_WIDTH / full.shape[1])
        image = cv2.resize(full, (WORK_WIDTH, height), interpolation=cv2.INTER_AREA)
        shot = mv.Shot(str(path), image)
        shot.disparity = game_depth.load(path, (WORK_WIDTH, height))
        # An all-zero depth (a capture taken while the depth buffer was cleared)
        # is no depth at all.
        if shot.disparity is not None and float(shot.disparity.max()) <= 0.0:
            shot.disparity = None
        shots.append(shot)
        sources.append(full)
    return shots, sources


def _focal(shots, pair_matches, progress, cancelled) -> float:
    """Focal length from the shots themselves: the one most atoms agree under.

    Scored on a spread-out subset with the cheap rigid check only (the dense
    refinement is not needed to tell a right focal length from a wrong one),
    coarse first, then finer around the best. Matches come from the shared
    per-pair cache, since they do not depend on the focal length.
    """
    pick = np.linspace(0, len(shots) - 1, min(len(shots), 8)).round().astype(int)
    pick = sorted(set(pick.tolist()))
    width, height = shots[0].size
    z_maps = {i: measured.distances(shots[i]) for i in pick}

    def score(focal: float) -> int:
        camera = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
        rng = np.random.default_rng(0)
        total = 0
        for a_index, a in enumerate(pick):
            for b in pick[a_index + 1:]:
                found = measured.link(a, b, shots, None, camera,
                                      {a: z_maps[a], b: z_maps[b]}, rng,
                                      refine_pose=False, matches=pair_matches(a, b))
                total += found.agree if found else 0
        return total

    coarse = np.linspace(0.25, 1.2, 11) * width
    fine_step = (coarse[1] - coarse[0]) / 4
    steps = len(coarse) + 6
    scores = {}
    for n, focal in enumerate(coarse, 1):
        if cancelled():
            raise Cancelled
        progress(f"Finding the field of view… {n}/{steps}")
        scores[float(focal)] = score(float(focal))
    best = max(scores, key=scores.get)
    for n, offset in enumerate((-3, -2, -1, 1, 2, 3), len(coarse) + 1):
        if cancelled():
            raise Cancelled
        progress(f"Finding the field of view… {n}/{steps}")
        focal = best + offset * fine_step
        if focal > 0:
            scores[float(focal)] = score(float(focal))
    return max(scores, key=scores.get)


def _level(placed) -> tuple[np.ndarray, np.ndarray, float]:
    """Gravity-aligned axes, the point of interest, and its distance.

    Game cameras almost never roll, so every shot's "right" direction is
    horizontal and true up is the one direction perpendicular to all of them.
    Without this the world is whatever the first shot's camera happened to be,
    and in Stray that camera looked steeply down at the cat, so the whole
    scene, and every orbit around it, came out tilted.

    The point of interest is where the cameras' view lines pass closest to one
    another: the subject of an orbit. The 3D tab's orbit preset circles its
    pivot, so putting this point there makes the default move circle what the
    player circled.
    """
    rights = np.array([s.rotation.T @ np.array([1.0, 0.0, 0.0]) for s in placed])
    ups = np.array([s.rotation.T @ np.array([0.0, -1.0, 0.0]) for s in placed])
    if len(placed) >= 3:
        _, _, vt = np.linalg.svd(rights)
        up = vt[-1]
    else:
        up = ups.mean(axis=0)
    up /= max(np.linalg.norm(up), 1e-9)
    if up @ ups.mean(axis=0) < 0:
        up = -up

    root = _root(placed)
    forwards = [s.rotation.T @ np.array([0.0, 0.0, 1.0]) for s in placed]
    a = sum(np.eye(3) - np.outer(f, f) for f in forwards)
    b = sum((np.eye(3) - np.outer(f, f)) @ s.centre for s, f in zip(placed, forwards))
    target = np.linalg.lstsq(a, b, rcond=None)[0]
    root_forward = root.rotation.T @ np.array([0.0, 0.0, 1.0])
    ahead = float((target - root.centre) @ root_forward)
    valid = root.depth[root.depth < measured.SKY / 2]
    typical = float(np.median(valid)) if valid.size else 1.0
    # A walk-through has no common subject: the view lines barely meet, and the
    # "target" lands behind the cameras or absurdly far away. Look ahead of the
    # first shot instead.
    if not (0.2 * typical < ahead < 5.0 * typical):
        target = root.centre + root_forward * typical

    back = root.centre - target
    back -= up * (back @ up)
    if np.linalg.norm(back) < 1e-6:
        back = -root_forward - up * (-root_forward @ up)
    z_axis = back / max(np.linalg.norm(back), 1e-9)
    x_axis = np.cross(up, z_axis)
    axes = np.stack([x_axis, up, z_axis])     # rows: world (capture) -> renderer
    return axes, target, float(np.linalg.norm(root.centre - target))


def _root(placed):
    """The shot whose camera defines the world: identity pose."""
    for shot in placed:
        if np.allclose(shot.rotation, np.eye(3)) and np.allclose(shot.centre, 0.0):
            return shot
    return placed[0] if placed else None


def _to_splats(shots, camera) -> "splat3d.SplatScene":
    """Fused points as surface-aligned discs in the renderer's space.

    The capture world is OpenCV-style (x right, y down, z forward). The
    renderer's world is levelled (y up), with the point of interest at the 3D
    tab's pivot, the first shot's side facing the default camera, and scaled
    so the capture orbit has the same radius as the tab's orbit preset. Each
    shot's discs are built in its own camera, where surface normals come
    straight from its depth map, then carried into that world.
    """
    from . import splat3d

    placed = [s for s in shots if s.depth is not None]
    axes, target, distance = _level(placed)
    scale = TARGET_DEPTH / max(distance, 1e-6)
    offset = np.array([0.0, 0.0, -TARGET_DEPTH])

    def to_renderer(points_world):
        return (points_world - target) @ axes.T * scale + offset

    total = sum(int((s.depth < measured.SKY / 2).sum()) for s in placed)
    stride = max(1, int(np.ceil(np.sqrt(total / MAX_SPLATS))))
    focal = float(camera[0, 0])
    positions, colors, covs = [], [], []
    for shot in placed:
        z = shot.depth[::stride, ::stride]
        rgb = shot.image[::stride, ::stride].astype(np.float32) / 255.0
        h, w = z.shape
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        xs, ys = xs * stride, ys * stride
        valid = z < measured.SKY / 2
        zz = np.where(valid, z, 1.0)
        cam = np.dstack([(xs - camera[0, 2]) / focal * zz, (ys - camera[1, 2]) / focal * zz, zz])
        # Disc size: one splat covers `stride` pixels at its distance.
        foot = (zz / focal) * 0.62 * stride
        cov_cam = splat3d._surface_cov((cam * scale).astype(np.float32),
                                       (foot * scale).astype(np.float32),
                                       np.zeros(z.shape, bool))
        keep = valid.reshape(-1)
        world = shot.centre + cam.reshape(-1, 3)[keep] @ shot.rotation
        c6 = cov_cam[keep]
        full = np.zeros((len(c6), 3, 3), np.float32)
        full[:, 0, 0], full[:, 0, 1], full[:, 0, 2] = c6[:, 0], c6[:, 1], c6[:, 2]
        full[:, 1, 0], full[:, 1, 1], full[:, 1, 2] = c6[:, 1], c6[:, 3], c6[:, 4]
        full[:, 2, 0], full[:, 2, 1], full[:, 2, 2] = c6[:, 2], c6[:, 4], c6[:, 5]
        turn = (axes @ shot.rotation.T).astype(np.float32)   # shot camera -> renderer
        rotated = turn @ full @ turn.T
        covs.append(np.stack([rotated[:, 0, 0], rotated[:, 0, 1], rotated[:, 0, 2],
                              rotated[:, 1, 1], rotated[:, 1, 2], rotated[:, 2, 2]], -1))
        positions.append(to_renderer(world))
        colors.append(rgb.reshape(-1, 3)[keep])

    positions = np.concatenate(positions).astype(np.float32)
    colors = np.concatenate(colors).astype(np.float32)
    covs = np.concatenate(covs).astype(np.float32)
    # The 3D tab's camera draws out to 60 units; beyond that is distant city
    # that only costs sort time.
    near_enough = np.linalg.norm(positions - offset, axis=1) < 55.0
    positions, colors, covs = positions[near_enough], colors[near_enough], covs[near_enough]

    root = _root(placed)
    z = root.depth
    valid = z < measured.SKY / 2
    h, w = z.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    zz = np.where(valid, z, 1.0)
    cam = np.dstack([(xs - camera[0, 2]) / focal * zz, (ys - camera[1, 2]) / focal * zz, zz])
    grid = to_renderer(root.centre + cam.reshape(-1, 3) @ root.rotation).reshape(h, w, 3)
    planes = splat3d.fit_planes(grid.astype(np.float32), valid)
    return splat3d.SplatScene(positions, colors, np.ones(len(positions), np.float32), covs,
                              len(positions), photo_z=None, focal=focal, planes=planes)


def _planes(root, camera, to_renderer):
    """Scene planes (floor, walls) from the root shot's measured depth, for
    the effects that need a ground: wet surfaces, rain splashes."""
    from . import splat3d

    focal = float(camera[0, 0])
    z = root.depth
    valid = z < measured.SKY / 2
    h, w = z.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    zz = np.where(valid, z, 1.0)
    cam = np.dstack([(xs - camera[0, 2]) / focal * zz, (ys - camera[1, 2]) / focal * zz, zz])
    grid = to_renderer(root.centre + cam.reshape(-1, 3) @ root.rotation).reshape(h, w, 3)
    return splat3d.fit_planes(grid.astype(np.float32), valid)


def _sharp_splats(shots, sources, camera, report, progress, cancelled):
    """Apple's SHARP on key shots, aligned with the game's depth, placed by the
    solved cameras. The multi-view solve decides where each piece sits; SHARP
    provides clean, full-detail surfaces for it.

    SHARP's own depth matched the game's up to a single scale (1.5% median
    error on a Stray shot), so one robust number per shot aligns it and its
    shapes stay untouched: no warping, which is what made fused depth smeary.
    """
    from . import sharp3d, splat3d

    placed = [(s, full) for s, full in zip(shots, sources) if s.depth is not None]
    axes, target, distance = _level([s for s, _ in placed])
    scale = TARGET_DEPTH / max(distance, 1e-6)
    offset = np.array([0.0, 0.0, -TARGET_DEPTH])
    width, height = placed[0][0].size
    focal = float(camera[0, 0])
    keys = placed[::KEY_EVERY]
    report.key_shots = len(keys)

    positions, colours, opacity, covs, scores, owners = [], [], [], [], [], []
    for owner, (shot, full) in enumerate(keys):
        if cancelled():
            raise Cancelled
        progress(f"Running SHARP on key shots… {owner + 1}/{len(keys)}")
        means, sizes, quats, linear, alpha = sharp3d.predict(full)
        alpha = alpha.ravel()
        nx, ny = means[:, 0] / means[:, 2], means[:, 1] / means[:, 2]
        u = ((nx + 1) * 0.5 * width).astype(int)
        v = ((ny + 1) * 0.5 * height).astype(int)
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        z_game = np.full(len(means), np.nan)
        z_game[inside] = shot.depth[v[inside], u[inside]]
        z_game[z_game >= measured.SKY / 2] = np.nan
        fit = inside & np.isfinite(z_game) & (alpha > 0.3)
        if fit.sum() < 100:
            continue
        k = float(np.median(z_game[fit] / means[fit, 2]))

        # X = kx * (mx / mz) * Z with Z = k * mz is a per-axis scale.
        stretch = np.array([(width / 2) / focal * k, (height / 2) / focal * k, k], np.float32)
        cam = means * stretch
        qw, qx, qy, qz = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        rot = np.stack([
            np.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)], -1),
            np.stack([2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)], -1),
            np.stack([2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)], -1)], 1)
        cov = rot @ (sizes[:, :, None] ** 2 * np.transpose(rot, (0, 2, 1)))
        cov = cov * (stretch[None, :, None] * stretch[None, None, :])

        with np.errstate(invalid="ignore"):
            surface = np.abs(k * means[:, 2] - z_game) / z_game < SURFACE_TOLERANCE
        largest = np.sqrt(np.max(np.linalg.eigvalsh(cov), axis=1))
        small = largest < MAX_SPLAT_PIXELS * cam[:, 2] / focal
        keep = (alpha > 0.05) & np.nan_to_num(surface, nan=False) & small

        turn = (axes @ shot.rotation.T).astype(np.float32)
        world = (shot.centre + cam[keep] @ shot.rotation - target) @ axes.T * scale + offset
        rotated = (turn @ cov[keep] @ turn.T) * scale * scale
        srgb = np.where(linear <= 0.0031308, linear * 12.92,
                        1.055 * np.power(np.maximum(linear, 0.0), 1 / 2.4) - 0.055)
        positions.append(world.astype(np.float32))
        colours.append(np.clip(srgb[keep], 0, 1).astype(np.float32))
        opacity.append(alpha[keep].astype(np.float32))
        covs.append(np.stack([rotated[:, 0, 0], rotated[:, 0, 1], rotated[:, 0, 2],
                              rotated[:, 1, 1], rotated[:, 1, 2], rotated[:, 2, 2]], -1)
                    .astype(np.float32))
        scores.append((cam[keep, 2] * scale).astype(np.float32))
        owners.append(np.full(int(keep.sum()), owner, np.int32))

    progress("Fusing the scene…")
    positions = np.concatenate(positions)
    colours = np.concatenate(colours)
    opacity = np.concatenate(opacity)
    covs = np.concatenate(covs)
    scores = np.concatenate(scores)
    owners = np.concatenate(owners)

    # Closest key shot wins each cell; the others' splats there are dropped,
    # so near-identical copies never stack.
    cells = np.floor(positions / OWNER_CELL).astype(np.int64)
    cell_key = (cells[:, 0] * 73856093) ^ (cells[:, 1] * 19349663) ^ (cells[:, 2] * 83492791)
    order = np.lexsort((scores, cell_key))
    sorted_keys = cell_key[order]
    first = np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]
    winner = owners[order][first][np.cumsum(first) - 1]
    keep = np.zeros(len(positions), bool)
    keep[order[owners[order] == winner]] = True
    keep &= np.linalg.norm(positions - offset, axis=1) < 55.0

    def to_renderer(points_world):
        return (points_world - target) @ axes.T * scale + offset

    planes = _planes(_root([s for s, _ in placed]), camera, to_renderer)
    return splat3d.SplatScene(positions[keep], colours[keep], opacity[keep], covs[keep],
                              int(keep.sum()), photo_z=None, focal=focal, planes=planes)


def read_gaussians_ply(path: str | Path, focal: float = 0.0):
    """A standard 3D Gaussian Splatting .ply (as Brush writes it) as a SplatScene.

    The format stores colour as spherical-harmonic coefficients, opacity as a
    logit, size as log-scales and orientation as a quaternion. Only the base
    colour term is used: the renderer draws view-independent colour.
    """
    from . import splat3d

    with open(path, "rb") as fh:
        header, count, fields = [], 0, []
        while True:
            line = fh.readline().decode("ascii", "replace").strip()
            header.append(line)
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property"):
                kind, name = line.split()[1], line.split()[-1]
                if kind not in ("float", "float32"):
                    raise ValueError(f"unsupported property type {kind} for {name}")
                fields.append(name)
            elif line == "end_header":
                break
        if "binary_little_endian" not in " ".join(header):
            raise ValueError("only binary little-endian .ply files are supported")
        table = np.frombuffer(fh.read(count * len(fields) * 4), "<f4").reshape(count, len(fields))
    col = {name: i for i, name in enumerate(fields)}
    positions = table[:, [col["x"], col["y"], col["z"]]].astype(np.float32)
    c0 = 0.28209479177387814
    colours = np.clip(0.5 + c0 * table[:, [col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]]], 0, 1)
    opacity = 1.0 / (1.0 + np.exp(-table[:, col["opacity"]]))
    scales = np.exp(table[:, [col["scale_0"], col["scale_1"], col["scale_2"]]])
    q = table[:, [col["rot_0"], col["rot_1"], col["rot_2"], col["rot_3"]]]
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rot = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)
    cov = rot @ (scales[:, :, None] ** 2 * np.transpose(rot, (0, 2, 1)))
    cov6 = np.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2],
                     cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], -1).astype(np.float32)
    return splat3d.SplatScene(positions, colours.astype(np.float32), opacity.astype(np.float32),
                              cov6, count, photo_z=None, focal=focal)


# --- saved scenes ---------------------------------------------------------------------

@dataclass
class SavedScene:
    folder: Path
    label: str
    created: float
    splats: int
    placed: int
    shots: int
    sharp: bool

    @property
    def thumbnail(self) -> Path:
        return self.folder / "thumb.jpg"


def save(scene, report: Report, label: str, root: Path | None = None, progress=None) -> Path:
    """Write a built scene to disk: splats, a thumbnail, and what it came from."""
    import json
    from . import paths

    root = Path(root) if root is not None else paths.scenes_dir()
    stamp = time.strftime("%Y-%m-%d %H-%M-%S")
    # Session labels read "Stray, 23 Sep 15:32: 33 shots, ..."; the game is
    # the part before the first comma, and the time goes in the stamp.
    game = label.split(",")[0]
    safe = "".join(c if c.isalnum() or c in " -_." else "_" for c in game).strip()
    folder = root / f"{safe or 'Scene'} {stamp}"
    folder.mkdir(parents=True, exist_ok=True)
    from . import shotscene
    if isinstance(scene, shotscene.ShotScene):
        scene.save(folder, getattr(scene, "pending_moved", None), progress)
        _save_thumb(folder, report)
        meta = {"label": label, "created": time.time(), "kind": "shots", "splats": 0,
                "placed": int(scene.used.sum()), "shots": report.considered, "sharp": False,
                "fov_degrees": report.fov_degrees, "agreement": report.agreement}
        (folder / "scene.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return folder
    planes = np.array([list(n) + [c] for n, c in (scene.planes or [])], np.float32).reshape(-1, 4)
    np.savez_compressed(
        folder / "scene.npz",
        positions=scene.positions.astype(np.float32),
        colors=np.round(np.clip(scene.colors, 0, 1) * 255).astype(np.uint8),
        opacity=np.round(np.clip(scene.opacity, 0, 1) * 255).astype(np.uint8),
        cov=scene.cov.astype(np.float32),
        planes=planes,
        focal=np.float32(scene.focal))
    _save_thumb(folder, report)
    meta = {"label": label, "created": time.time(), "splats": len(scene),
            "placed": report.placed, "shots": report.considered, "sharp": report.sharp,
            "key_shots": report.key_shots, "fov_degrees": report.fov_degrees,
            "agreement": report.agreement}
    (folder / "scene.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return folder


def _save_thumb(folder: Path, report: Report) -> None:
    if report.root_image is not None:
        image = np.asarray(report.root_image)
        h, w = image.shape[:2]
        thumb = cv2.resize(image, (320, max(1, round(h * 320 / w))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
        if ok:
            buf.tofile(str(folder / "thumb.jpg"))


def list_saved(root: Path | None = None) -> list[SavedScene]:
    """Saved scenes, newest first. A damaged entry is skipped, never fatal."""
    import json
    from . import paths

    root = Path(root) if root is not None else paths.scenes_dir()
    out = []
    for folder in root.iterdir() if root.is_dir() else []:
        try:
            meta = json.loads((folder / "scene.json").read_text(encoding="utf-8"))
            if not ((folder / "scene.npz").is_file() or (folder / "shots.json").is_file()):
                continue
            out.append(SavedScene(folder, meta.get("label", folder.name), float(meta.get("created", 0)),
                                  int(meta.get("splats", 0)), int(meta.get("placed", 0)),
                                  int(meta.get("shots", 0)), bool(meta.get("sharp", False))))
        except Exception:  # noqa: BLE001 - one bad folder must not hide the rest
            continue
    out.sort(key=lambda s: s.created, reverse=True)
    return out


def load(folder: str | Path):
    """A saved scene back, ready for the 3D tab: a ShotScene, or a SplatScene
    for scenes built without game depth."""
    from . import shotscene, splat3d

    if shotscene.is_shot_scene(folder):
        return shotscene.load(folder)

    data = np.load(Path(folder) / "scene.npz")
    planes = [(row[:3], float(row[3])) for row in data["planes"]] if "planes" in data else None
    return splat3d.SplatScene(
        data["positions"], data["colors"].astype(np.float32) / 255.0,
        data["opacity"].astype(np.float32) / 255.0, data["cov"], len(data["positions"]),
        photo_z=None, focal=float(data["focal"]), planes=planes or None)


def build(paths: list[str | Path], excluded: set[str] | None = None,
          progress: Callable[[str], None] = lambda _m: None,
          cancelled: Callable[[], bool] = lambda: False, main: str | Path | None = None):
    """Place and fuse a session's shots. Returns (SplatScene | None, Report)."""
    excluded = excluded or set()
    report = Report()
    wanted = []
    for path in paths:
        key = str(path)
        if key in excluded:
            report.statuses[key] = EXCLUDED
        else:
            wanted.append(Path(path))

    # Shots with game depth become a shot scene: every screenshot kept as
    # itself and blended per pixel (see shotscene). Only captures without
    # depth still go through the estimated-depth splat path below.
    if main is not None and any(game_depth.sidecars(p)[0].is_file() for p in wanted):
        # Main-shot scene: the chosen shot rebuilt with SHARP, its
        # neighbours only filling what it cannot see (see herosolve).
        from . import herosolve
        try:
            scene, hero_report = herosolve.build(wanted, main, set(), progress, cancelled)
        except herosolve.Cancelled:
            raise Cancelled from None
        hero_report.statuses.update({k: v for k, v in report.statuses.items() if v == EXCLUDED})
        return scene, hero_report
    if any(game_depth.sidecars(p)[0].is_file() for p in wanted):
        from . import shotsolve
        try:
            scene, shot_report, moved = shotsolve.build(wanted, set(), progress, cancelled)
        except shotsolve.Cancelled:
            raise Cancelled from None
        shot_report.statuses.update({k: v for k, v in report.statuses.items() if v == EXCLUDED})
        if scene is not None:
            scene.pending_moved = moved
        return scene, shot_report

    shots, sources = _load(wanted, report, progress, cancelled)
    measured_shots = [s for s in shots if s.disparity is not None]

    if len(measured_shots) >= 2:
        for shot in shots:
            if shot.disparity is None:
                report.statuses[shot.name] = NO_DEPTH
        keep = [i for i, s in enumerate(shots) if s.disparity is not None]
        shots = [shots[i] for i in keep]
        sources = [sources[i] for i in keep]
        described = []
        for n, full in enumerate(sources, 1):
            if cancelled():
                raise Cancelled
            progress(f"Finding atoms… {n}/{len(sources)}")
            described.append(measured.describe(full, WORK_WIDTH))
        cache: dict[tuple[int, int], tuple] = {}

        def pair_matches(i: int, j: int):
            if (i, j) not in cache:
                cache[(i, j)] = measured.match_points(described[i], described[j])
            return cache[(i, j)]

        focal = _focal(shots, pair_matches, progress, cancelled)
        width, height = shots[0].size
        camera = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
        candidates = measured.candidate_pairs(shots)
        pairs = len(candidates)
        counter = {"n": 0}

        def on_pair():
            counter["n"] += 1
            if cancelled():
                raise Cancelled
            if counter["n"] % 4 == 0 or counter["n"] == pairs:
                progress(f"Linking shots… {counter['n']}/{pairs}")

        used = measured.solve(shots, camera, described=described, on_pair=on_pair,
                              pair_matches=pair_matches, pairs=candidates)
        report.agreement = float(np.mean([l.surface for l in used])) if used else 0.0
    else:
        # No game depth: fall back to estimated depth and the atom solver.
        # Works, but each photo's depth is a guess, so the scene is softer.
        report.estimated = True
        from . import atoms
        from .onnx_depth import OnnxDepthEngine
        from .settings import AppSettings
        engine = OnnxDepthEngine()
        engine.load(AppSettings().depth.model_id)
        for n, shot in enumerate(shots, 1):
            if cancelled():
                raise Cancelled
            progress(f"Estimating depth… {n}/{len(shots)}")
            shot.disparity = engine.infer(shot.image)
        width, height = shots[0].size if shots else (WORK_WIDTH, 1)
        camera = mv.intrinsics(width, height, 80.0)
        progress("Placing shots from estimated depth…")
        mv.solve_graph(shots, camera)
        _, _, _, camera = atoms.solve(shots, camera)
        focal = float(camera[0, 0])

    for shot in shots:
        report.statuses[shot.name] = PLACED if shot.depth is not None else NO_OVERLAP
    width = shots[0].size[0] if shots else WORK_WIDTH
    report.fov_degrees = float(np.degrees(2 * np.arctan(width / 2 / max(focal, 1e-6))))

    placed = [s for s in shots if s.depth is not None]
    if len(placed) < 2:
        report.advice = ("Not enough shots could be placed. Take more shots, each moving a "
                         "little from the last, so neighbours share most of their view.")
        return None, report
    no_depth = [Path(p).stem.split("_")[-1] for p, s in report.statuses.items() if s == NO_DEPTH]
    lost = [Path(p).stem.split("_")[-1] for p, s in report.statuses.items() if s == NO_OVERLAP]
    if report.estimated:
        report.advice = ("These shots have no game depth, so depth was estimated and the "
                         "scene is softer. Capture with the DLSS5 Scene Capture add-on for "
                         "real depth.")
    elif no_depth:
        report.advice = f"Shots {', '.join(no_depth)} have no depth. Capture them again."
    elif lost:
        report.advice = (f"Shots {', '.join(lost)} did not overlap enough with the others. "
                         "Smaller steps between shots help.")

    report.root_image = _root(placed).image
    from . import sharp3d
    if not report.estimated and sharp3d.is_downloaded():
        report.sharp = True
        scene = _sharp_splats(shots, sources, camera, report, progress, cancelled)
    else:
        progress("Fusing the scene…")
        scene = _to_splats(shots, camera)
        if not report.estimated and not report.advice:
            report.advice = ("Download SHARP (High quality) for far cleaner scenes: it rebuilds "
                             "each key shot in full detail.")
    return scene, report
