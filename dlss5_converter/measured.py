"""Place shots that carry the game's real depth: rigid 3D alignment.

``multiview`` and ``atoms`` were built for guessed depth. They recover camera
motion from 2D matches alone (the essential matrix), which needs parallax,
fails on a pure rotation, and knows distance only up to scale.

With the depth the DLSS5 Scene Capture add-on saves, none of that applies.
Every matched atom already has a 3D position in each shot, so placing one shot
relative to another is aligning two point sets: a rotation and a translation,
solved in closed form (Kabsch) inside RANSAC. A camera that only turned is no
longer a problem, and scale needs no guessing, because the game's depth is
``near / distance`` with the same near plane in every shot.

The one unknown left is the focal length: photo modes have their own FOV
slider and nothing records it. A wrong focal length bends every unprojected
point, so shots stop agreeing. ``estimate_focal`` searches for the value under
which the most atoms agree across shots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import multiview as mv

#: RANSAC tolerance, as a fraction of the point's distance. Depth buffers are
#: precise, but a feature's pixel position is not, and that error grows with
#: distance, so a fixed tolerance would reject every far match.
TOLERANCE = 0.03

#: Matched atoms that must agree before a placement is believed.
MIN_AGREE = 20

#: A rigid fit must agree with an independent 2D estimate of the same rotation
#: (essential matrix, or a homography when the camera only turned) to within
#: this many degrees. On the first Cyberpunk orbit several fits passed on atom
#: count alone while being tens of degrees off, and one bad link in the chain
#: wrecked every shot placed after it.
MAX_ROTATION_DISAGREEMENT = 8.0

#: The agreeing atoms must span real depth (farthest / nearest). A fit whose
#: inliers all sit on one wall at one distance scores well and pins the
#: rotation down badly.
MIN_DEPTH_SPREAD = 1.3

#: After dense refinement, the share of overlapping depth pixels that must sit
#: flush with the other shot's surface. Below this the two shots do not
#: describe the same geometry, whatever the atoms said.
MIN_SURFACE_AGREEMENT = 0.5


@dataclass
class Link:
    first: int
    second: int
    agree: int
    rotation: np.ndarray       # second-camera frame -> first-camera frame
    translation: np.ndarray
    surface: float = 0.0       # share of overlapping depth pixels that agree


#: Shared unit for measured depth: a hundredth of the game's near-plane
#: distance per unit. Arbitrary, but the same in every shot, and it keeps a
#: street scene (hundreds of near-planes deep) inside ``multiview.unproject``'s
#: far cut instead of discarding it as sky.
UNIT = 0.01

#: Stand-in distance for sky. Far beyond any far cut, so it is dropped, rather
#: than 0, which would put a point at the camera itself.
SKY = 1e9


def distances(shot: mv.Shot) -> np.ndarray:
    """Camera-space depth (z) in UNIT near-planes; SKY where nothing was drawn."""
    d = shot.disparity
    with np.errstate(divide="ignore"):
        return np.where(d > 1e-6, UNIT / np.maximum(d, 1e-6), SKY)


def lift(points: np.ndarray, z_map: np.ndarray, camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixels to camera-space 3D points using measured depth. Also a validity mask."""
    z = mv.sample(z_map, points)
    ok = z < SKY / 2
    x = (points[:, 0] - camera[0, 2]) / camera[0, 0] * z
    y = (points[:, 1] - camera[1, 2]) / camera[1, 1] * z
    return np.column_stack([x, y, z]), ok


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R and translation t minimising |R @ source + t - target|."""
    cs, ct = source.mean(axis=0), target.mean(axis=0)
    h = (source - cs).T @ (target - ct)
    u, _, vt = np.linalg.svd(h)
    fix = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    rotation = vt.T @ fix @ u.T
    return rotation, ct - rotation @ cs


def _kabsch_batch(source: np.ndarray, target: np.ndarray):
    """Kabsch for many point triples at once: (n,k,3) x2 -> R (n,3,3), t (n,3)."""
    cs, ct = source.mean(axis=1), target.mean(axis=1)
    h = np.einsum("nki,nkj->nij", source - cs[:, None], target - ct[:, None])
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(np.transpose(vt, (0, 2, 1)) @ np.transpose(u, (0, 2, 1))))
    fix = np.zeros((len(source), 3, 3))
    fix[:, 0, 0] = fix[:, 1, 1] = 1.0
    fix[:, 2, 2] = d
    rotation = np.transpose(vt, (0, 2, 1)) @ fix @ np.transpose(u, (0, 2, 1))
    return rotation, ct - np.einsum("nij,nj->ni", rotation, cs)


def align(source: np.ndarray, target: np.ndarray, rng: np.random.Generator,
          iterations: int = 400):
    """RANSAC rigid fit; returns (R, t, inlier mask) or None.

    All hypotheses are solved and scored in one batch: done one at a time in
    Python, the 400 small SVDs per pair were most of a thirty-shot build.
    """
    if len(source) < 3:
        return None
    tolerance = TOLERANCE * np.linalg.norm(target, axis=1)
    picks = np.stack([rng.choice(len(source), 3, replace=False) for _ in range(iterations)])
    rotations, translations = _kabsch_batch(source[picks], target[picks])
    best, best_count = None, -1
    # Score in slices so a big match set does not build a huge array at once.
    for start in range(0, iterations, 100):
        r, t = rotations[start:start + 100], translations[start:start + 100]
        moved = np.einsum("hij,nj->hni", r, source) + t[:, None]
        counts_mask = np.linalg.norm(moved - target[None], axis=2) < tolerance[None]
        counts = counts_mask.sum(axis=1)
        top = int(np.argmax(counts))
        if counts[top] > best_count:
            best_count, best = int(counts[top]), counts_mask[top]
    if best is None or best_count < 3:
        return None
    rotation, translation = kabsch(source[best], target[best])
    error = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    inliers = error < tolerance
    return rotation, translation, inliers


def candidate_pairs(shots, window: int = 2, similar: int = 4) -> list[tuple[int, int]]:
    """The pairs worth linking: neighbours in capture order, and look-alikes.

    Checking every pair grows with the square of the shot count (528 pairs for
    33 shots) while each shot only really overlaps a handful of others. Shots
    taken one after another almost always overlap; the look-alikes (compared as
    tiny grey thumbnails) catch the loop back to the start and any revisit.
    """
    import cv2

    count = len(shots)
    pairs = {(i, j) for i in range(count) for j in range(i + 1, min(count, i + window + 1))}
    if count > 2 and similar > 0:
        thumbs = []
        for shot in shots:
            grey = cv2.cvtColor(shot.image, cv2.COLOR_RGB2GRAY)
            tiny = cv2.resize(grey, (32, 16), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
            tiny -= tiny.mean()
            thumbs.append(tiny / max(float(np.linalg.norm(tiny)), 1e-6))
        likeness = np.array(thumbs) @ np.array(thumbs).T
        np.fill_diagonal(likeness, -np.inf)
        for i in range(count):
            for j in np.argsort(likeness[i])[::-1][:similar]:
                pairs.add((min(i, int(j)), max(i, int(j))))
    return sorted(pairs)


class _Point:
    """A keypoint reduced to what matching needs: its (scaled) position."""

    __slots__ = ("pt",)

    def __init__(self, x: float, y: float):
        self.pt = (x, y)


def describe(full_rgb: np.ndarray, working_width: int, cap: int = 20000):
    """Atoms found at full resolution, positioned in working-resolution pixels.

    Two things multiplied the matches 5-9x on a night-time Cyberpunk orbit:
    looking at the full 3840 px instead of the 1600 px working copy, and a
    local contrast boost (CLAHE) first. Photo-mode shots at night are mostly
    near-black, where SIFT finds almost nothing without it. Depth, cameras and
    fusion stay at the working size; only detection needs the detail.
    """
    import cv2

    grey = cv2.cvtColor(full_rgb, cv2.COLOR_RGB2GRAY)
    grey = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(grey)
    keypoints, descriptors = cv2.SIFT_create(nfeatures=cap).detectAndCompute(grey, None)
    scale = working_width / full_rgb.shape[1]
    return [_Point(k.pt[0] * scale, k.pt[1] * scale) for k in keypoints], descriptors


def image_rotation(points_i: np.ndarray, points_j: np.ndarray, camera: np.ndarray):
    """Rotation from shot i's camera to shot j's, from pixels alone.

    The independent witness for a rigid fit. When one homography explains the
    matches the camera only turned, and the rotation is K^-1 H K; otherwise the
    essential matrix gives it. Returns None when neither can be estimated.
    """
    import cv2

    if len(points_i) < 8:
        return None
    homography, h_mask = cv2.findHomography(points_i, points_j, cv2.USAC_MAGSAC, 2.0)
    essential, e_mask = cv2.findEssentialMat(points_i, points_j, camera,
                                             method=cv2.USAC_MAGSAC, prob=0.9999, threshold=1.0)
    h_count = int(h_mask.sum()) if h_mask is not None else 0
    e_count = int(e_mask.sum()) if e_mask is not None else 0
    if homography is not None and h_count >= 0.8 * max(e_count, 1):
        rotation = np.linalg.inv(camera) @ homography @ camera
        u, _, vt = np.linalg.svd(rotation)
        rotation = u @ vt
        return rotation if np.linalg.det(rotation) > 0 else None
    if essential is None or essential.shape != (3, 3):
        return None
    keep = e_mask.ravel() > 0
    _, rotation, _, _ = cv2.recoverPose(essential, points_i[keep], points_j[keep], camera)
    return rotation


def _angle(rotation: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0))))


def normals(points: np.ndarray) -> np.ndarray:
    """Per-pixel surface normals of an HxWx3 camera-space point map."""
    dx = np.zeros_like(points)
    dy = np.zeros_like(points)
    dx[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dy[1:-1] = points[2:] - points[:-2]
    n = np.cross(dx, dy)
    length = np.linalg.norm(n, axis=2, keepdims=True)
    return np.where(length > 1e-12, n / np.maximum(length, 1e-12), 0.0)


def point_map(z_map: np.ndarray, camera: np.ndarray) -> np.ndarray:
    """Every pixel lifted to camera space: HxWx3, NaN where there is sky."""
    height, width = z_map.shape
    ys, xs = np.mgrid[0:height, 0:width]
    z = np.where(z_map < SKY / 2, z_map, np.nan)
    return np.dstack([(xs - camera[0, 2]) / camera[0, 0] * z,
                      (ys - camera[1, 2]) / camera[1, 1] * z, z])


def refine(rotation: np.ndarray, translation: np.ndarray, source_z: np.ndarray,
           target_z: np.ndarray, camera: np.ndarray, iterations: int = 25,
           stride: int = 4, far: float = 60.0):
    """Tighten a source -> target rigid pose with every depth pixel.

    Point-to-plane ICP with projective association (the KinectFusion trick):
    each source pixel is moved by the current pose and projected into the
    target camera, and the target's own pixel there is its partner. No nearest-
    neighbour search, so no SciPy, and it is fast. A handful of atoms fixes the
    heading well but leaves the tilt loose; the ground alone is hundreds of
    thousands of pixels constraining exactly that.

    Returns the refined (rotation, translation) and the share of overlapping
    pixels that ended up agreeing, which is the link's confidence.
    """
    source = point_map(source_z, camera)[::stride, ::stride].reshape(-1, 3)
    source = source[np.isfinite(source).all(axis=1) & (source[:, 2] < far)]
    target_points = point_map(target_z, camera)
    target_normals = normals(target_points)
    height, width = target_z.shape
    agree = 0.0
    for step in range(iterations):
        moved = source @ rotation.T + translation
        front = moved[:, 2] > 1e-3
        u = np.round(camera[0, 0] * moved[:, 0] / np.where(front, moved[:, 2], 1) + camera[0, 2]).astype(int)
        v = np.round(camera[1, 1] * moved[:, 1] / np.where(front, moved[:, 2], 1) + camera[1, 2]).astype(int)
        inside = front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if inside.sum() < 100:
            return rotation, translation, 0.0
        partner = target_points[v[inside], u[inside]]
        normal = target_normals[v[inside], u[inside]]
        mine = moved[inside]
        usable = np.isfinite(partner).all(axis=1) & (np.abs(normal).sum(axis=1) > 0)
        # Tolerance shrinks as the fit settles: generous first, so a tilted
        # start can find its partners, strict at the end, so nothing across a
        # depth edge is paired.
        tolerance = (0.10 if step < iterations // 2 else 0.03) * mine[:, 2]
        gap = np.einsum("ij,ij->i", mine - np.nan_to_num(partner), normal)
        close = usable & (np.abs(gap) < tolerance)
        agree = float(close.sum()) / float(max(usable.sum(), 1))
        if close.sum() < 100:
            return rotation, translation, 0.0
        p, n, g = mine[close], normal[close], gap[close]
        # Linearised point-to-plane step: small rotation w and shift d with
        # (p + w x p + d - q) . n = 0, i.e. [p x n, n] . [w, d] = -gap.
        # Each row is divided by its distance. Unweighted, the far background
        # (most of the pixels, and the largest absolute gaps) decides alone,
        # and far geometry barely feels a camera move: on the first Cyberpunk
        # orbit that collapsed a 1-unit orbit into a camera spinning on the
        # spot. Relative error puts near and far on an equal footing.
        weight = 1.0 / np.maximum(p[:, 2], 1e-6)
        a = np.hstack([np.cross(p, n), n]) * weight[:, None]
        w_d = np.linalg.lstsq(a, -g * weight, rcond=None)[0]
        w, d = w_d[:3], w_d[3:]
        angle = np.linalg.norm(w)
        if angle > 1e-12:
            k = w / angle
            kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            delta = np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx
        else:
            delta = np.eye(3)
        rotation, translation = delta @ rotation, delta @ translation + d
        if angle < 1e-6 and np.linalg.norm(d) < 1e-6:
            break
    return rotation, translation, agree


def match_points(described_i, described_j):
    """Ratio-tested matches between two shots as two Nx2 pixel arrays.

    A FLANN tree rather than brute force: with twenty thousand atoms per shot,
    brute-force matching was most of a thirty-shot build. Matching does not
    depend on the camera, so callers cache the result per pair.
    """
    import cv2

    if described_i[1] is None or described_j[1] is None or len(described_i[1]) < 2 \
            or len(described_j[1]) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2))
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
    pairs = matcher.knnMatch(described_i[1], described_j[1], k=2)
    found = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < mv.RATIO * n.distance]
    return mv.matched_points(described_i[0], described_j[0], found)


def link(i: int, j: int, shots, described, camera, z_maps, rng,
         refine_pose: bool = True, matches=None) -> Link | None:
    """How shot j sits relative to shot i, from atoms both see.

    ``refine_pose=False`` skips the dense refinement: enough to score a focal
    length, and far cheaper when that is run for every candidate.
    """
    if matches is None:
        matches = match_points(described[i], described[j])
    points_i, points_j = matches
    if len(points_i) < MIN_AGREE:
        return None
    in_i, ok_i = lift(points_i, z_maps[i], camera)
    in_j, ok_j = lift(points_j, z_maps[j], camera)
    ok = ok_i & ok_j
    if ok.sum() < MIN_AGREE:
        return None
    fit = align(in_j[ok], in_i[ok], rng)
    if fit is None:
        return None
    rotation, translation, inliers = fit
    if inliers.sum() < MIN_AGREE:
        return None
    depth = in_i[ok][inliers][:, 2]
    if depth.max() / max(depth.min(), 1e-9) < MIN_DEPTH_SPREAD:
        return None
    witness = image_rotation(points_i, points_j, camera)
    # The rigid fit maps j -> i and the witness i -> j, so agreeing rotations
    # compose to (nearly) the identity.
    if witness is None or _angle(witness @ rotation) > MAX_ROTATION_DISAGREEMENT:
        return None
    if not refine_pose:
        return Link(i, j, int(inliers.sum()), rotation, translation)
    rotation, translation, agreement = refine(rotation, translation, z_maps[j], z_maps[i], camera)
    if agreement < MIN_SURFACE_AGREEMENT:
        return None
    found = Link(i, j, int(inliers.sum()), rotation, translation)
    found.surface = agreement
    return found


def solve(shots: list[mv.Shot], camera: np.ndarray, seed: int = 0, described=None,
          on_pair=None, pair_matches=None, pairs=None):
    """Place every shot that measured depth can reach. Returns the links used.

    Every pair is scored by agreeing atoms, then a greedy maximum spanning tree
    from the best-connected shot places the rest. Each shot's metric depth is
    written to ``shot.depth`` so ``multiview.fuse`` can build the cloud.
    """
    rng = np.random.default_rng(seed)
    described = described or [mv.features(shot.image) for shot in shots]
    z_maps = [distances(shot) for shot in shots]
    links: dict[tuple[int, int], Link] = {}
    todo = pairs if pairs is not None else [(i, j) for i in range(len(shots))
                                            for j in range(i + 1, len(shots))]
    for i, j in todo:
        cached = pair_matches(i, j) if pair_matches is not None else None
        found = link(i, j, shots, described, camera, z_maps, rng, matches=cached)
        if found is not None:
            links[(i, j)] = found
        if on_pair is not None:
            on_pair()

    for shot in shots:
        shot.depth = None
    if not links:
        return []
    score = np.zeros(len(shots))
    for (i, j), found in links.items():
        score[i] += found.agree
        score[j] += found.agree
    root = int(np.argmax(score))
    shots[root].rotation, shots[root].centre = np.eye(3), np.zeros(3)
    shots[root].depth = z_maps[root]
    placed, used = {root}, []

    while True:
        best = None
        for (i, j), found in links.items():
            if (i in placed) != (j in placed) and (best is None or found.agree > best.agree):
                best = found
        if best is None:
            break
        if best.first in placed:
            parent, child = best.first, best.second
            rotation, translation = best.rotation, best.translation       # child -> parent
        else:
            parent, child = best.second, best.first
            rotation = best.rotation.T                                    # invert
            translation = -best.rotation.T @ best.translation
        # Parent camera frame -> world is (R_p^T, C_p). Child frame -> parent
        # frame is (rotation, translation). Compose, then store world -> child.
        to_world = shots[parent].rotation.T @ rotation
        shots[child].rotation = to_world.T
        shots[child].centre = shots[parent].rotation.T @ translation + shots[parent].centre
        shots[child].depth = z_maps[child]
        placed.add(child)
        used.append(best)
    return used


def estimate_focal(shots: list[mv.Shot], width: int, height: int,
                   fractions=np.linspace(0.2, 1.2, 21), described=None) -> tuple[float, dict]:
    """Focal length (pixels) under which the most atoms agree across shots.

    Photo modes change FOV freely and nothing records it. Unprojecting with the
    wrong focal length bends every shot differently, so fewer matched atoms can
    be made to line up; the right one maximises agreement.
    """
    described = described or [mv.features(shot.image) for shot in shots]
    z_maps = [distances(shot) for shot in shots]
    scores = {}
    for fraction in fractions:
        focal = fraction * width
        camera = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
        rng = np.random.default_rng(0)
        total = 0
        for i in range(len(shots)):
            for j in range(i + 1, len(shots)):
                found = link(i, j, shots, described, camera, z_maps, rng)
                total += found.agree if found else 0
        scores[float(focal)] = total
    best = max(scores, key=scores.get)
    return best, scores
