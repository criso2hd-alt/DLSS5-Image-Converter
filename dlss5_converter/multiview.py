"""Turn several photo-mode shots of one place into a single 3D scene.

A one-photo scene is only honest from the camera that took it: everything the
lens could not see is invented, so orbiting far enough always reaches made-up
geometry. Two shots of the same wall from different positions remove the guess
for every surface both of them saw.

The renderer in ``splat3d`` already draws whatever splats it is handed, so the
work here is entirely in the *builder*: recover where each shot was taken from,
put every shot's depth into one shared world, and fuse.

Why this is possible without a photogrammetry stack
---------------------------------------------------
Game screenshots are better input than real photographs. There is no lens
distortion, no rolling shutter, no sensor noise, and with depth of field off,
everything is in focus. A plain pinhole model is not an approximation here, it
is exactly right. So OpenCV's essential-matrix pose recovery, which struggles on
phone snapshots, has an easy time.

The load-bearing problem is scale. Essential-matrix recovery gives a translation
direction but not a distance, and Depth Anything gives *relative* inverse depth
whose scale differs from shot to shot. Neither is usable alone. Together they
pin each other down: triangulated feature points supply true relative geometry
at a fixed baseline, the depth map is fitted onto those points, and the fitted
depth then measures the *next* baseline. Scale walks along the sequence from one
arbitrary unit fixed at the first pair.

This is a builder, not a trainer. It produces a fused point cloud, not an
optimised Gaussian scene, so it will not match a real 3DGS pipeline. It will
however be geometrically honest everywhere two shots overlap, which is the part
a single image cannot do at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import cv2
import numpy as np


#: Ratio-test threshold for SIFT matches. Lowe's 0.8 lets too much through on
#: the repeated textures games are full of (brick, foliage, tiling floors).
RATIO = 0.75

#: Below this many surviving inliers a pair is not trustworthy enough to chain
#: a pose through, however good the matches looked.
MIN_INLIERS = 40

#: Median triangulation angle, in degrees, under which a pair carries almost no
#: depth information. This is the "they only rotated the camera" detector: pure
#: rotation gives perfect matches and zero parallax, and without this check the
#: pipeline happily returns a confident, meaningless scene.
MIN_PARALLAX = 1.5

#: A pair whose matches a single homography explains this well is a rotation
#: or a flat surface, whatever its measured parallax says. Parallax alone was
#: not enough on real captures: a spin in place in Cyberpunk's photo mode
#: produced a handful of pairs whose noise pushed them over MIN_PARALLAX, and
#: each one "placed" a shot thousands of units away. The homography test is
#: the one real SfM systems use for exactly this.
MAX_HOMOGRAPHY_SHARE = 0.8

#: How far the fitted depth may disagree with a pair's own triangulated points
#: (median absolute log ratio; 0.35 is about +/-40%) before the pair's scale is
#: not trusted. A wildly inconsistent ratio means a bad pose, and carrying it
#: forward multiplies the error into every shot placed after it.
MAX_SCALE_SPREAD = 0.35

#: The first pair fixes the world unit, so it needs a real baseline. COLMAP
#: insists on far more for its initial pair; this is a floor, not a target.
SEED_PARALLAX = 2.0


@dataclass
class Shot:
    """One photo-mode capture and everything we work out about it."""

    name: str
    image: np.ndarray                      # RGB uint8, HxWx3
    disparity: np.ndarray | None = None    # Depth Anything output, 1.0 = near
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    centre: np.ndarray = field(default_factory=lambda: np.zeros(3))
    depth: np.ndarray | None = None        # metric-ish depth in world units
    #: Per-pixel confidence a fused point needs to survive, from the trust
    #: tiers; NaN where the pixel is beyond the hard cut and always dropped.
    required: np.ndarray | None = None

    @property
    def size(self) -> tuple[int, int]:
        height, width = self.image.shape[:2]
        return width, height

    def translation(self) -> np.ndarray:
        """World-to-camera translation, the form OpenCV wants."""
        return -self.rotation @ self.centre


@dataclass
class PairReport:
    """What a single image pair contributed, for the diagnosis panel."""

    first: str
    second: str
    matches: int
    inliers: int
    parallax: float
    scale: float = 1.0
    ok: bool = True
    note: str = ""


def intrinsics(width: int, height: int, fov_degrees: float = 70.0,
               horizontal: bool = True) -> np.ndarray:
    """Pinhole camera matrix for a game screenshot.

    Games report field of view inconsistently (some horizontal, some vertical,
    some vertical-at-16:9 then Hor+ scaled), so this is a starting guess the
    caller can override. Being 5 degrees out warps the scene slightly but does
    not break pose recovery, because the same wrong focal length is used for
    every shot and the error is largely absorbed into scale.
    """
    span = width if horizontal else height
    focal = (span / 2.0) / math.tan(math.radians(fov_degrees) / 2.0)
    return np.array([[focal, 0.0, width / 2.0],
                     [0.0, focal, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _grey(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def features(image: np.ndarray, cap: int = 8000):
    """SIFT keypoints and descriptors.

    SIFT rather than ORB: game scenes repeat textures aggressively and ORB's
    binary descriptors produce far more plausible-looking wrong matches on
    them. Speed is not the constraint when the whole job runs once.
    """
    sift = cv2.SIFT_create(nfeatures=cap)
    return sift.detectAndCompute(_grey(image), None)


def match(descriptors_a, descriptors_b) -> list:
    """Ratio-tested mutual matches between two descriptor sets."""
    if descriptors_a is None or descriptors_b is None:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    return [m for m, n in (p for p in pairs if len(p) == 2)
            if m.distance < RATIO * n.distance]


def matched_points(keypoints_a, keypoints_b, matches) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([keypoints_a[m.queryIdx].pt for m in matches], np.float64)
    b = np.array([keypoints_b[m.trainIdx].pt for m in matches], np.float64)
    return a.reshape(-1, 2), b.reshape(-1, 2)


def relative_pose(points_a: np.ndarray, points_b: np.ndarray, camera: np.ndarray):
    """Rotation and *unit* translation taking camera A's frame to camera B's.

    MAGSAC over the plain RANSAC default: it is markedly better at holding on
    when a chunk of the frame is a moving element (water, an NPC, blowing
    foliage) that matches well and disagrees with the true camera motion.
    """
    if len(points_a) < 8:
        return None, None, np.zeros(len(points_a), bool)
    essential, mask = cv2.findEssentialMat(
        points_a, points_b, camera, method=cv2.USAC_MAGSAC,
        prob=0.9999, threshold=1.0)
    if essential is None or essential.shape != (3, 3):
        return None, None, np.zeros(len(points_a), bool)
    inliers = mask.ravel().astype(bool) if mask is not None else np.ones(len(points_a), bool)
    _, rotation, translation, pose_mask = cv2.recoverPose(
        essential, points_a[inliers], points_b[inliers], camera)
    kept = np.zeros(len(points_a), bool)
    kept[np.flatnonzero(inliers)[pose_mask.ravel() > 0]] = True
    return rotation, translation.reshape(3), kept


def homography_share(points_a: np.ndarray, points_b: np.ndarray,
                     essential_inliers: int, threshold: float = 1.5) -> float:
    """Inliers one homography explains, relative to the essential matrix's.

    Near 1.0 means the two views are related by a pure rotation or the scene is
    one flat surface. Both are degenerate for essential-matrix recovery: it
    still returns an answer, and the answer is noise. A user who panned their
    camera instead of moving it needs to be told that, not handed a scene.
    """
    if len(points_a) < 8 or essential_inliers <= 0:
        return 0.0
    _, mask = cv2.findHomography(points_a, points_b, cv2.USAC_MAGSAC, threshold)
    if mask is None:
        return 0.0
    return float(mask.ravel().astype(bool).sum()) / float(essential_inliers)


def triangulate(camera: np.ndarray,
                rotation_a: np.ndarray, translation_a: np.ndarray,
                rotation_b: np.ndarray, translation_b: np.ndarray,
                points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """World-space points from one correspondence set, as an Nx3 array."""
    projection_a = camera @ np.hstack([rotation_a, translation_a.reshape(3, 1)])
    projection_b = camera @ np.hstack([rotation_b, translation_b.reshape(3, 1)])
    homogeneous = cv2.triangulatePoints(projection_a, projection_b,
                                        points_a.T, points_b.T)
    w = homogeneous[3]
    w[np.abs(w) < 1e-12] = 1e-12
    return (homogeneous[:3] / w).T


def parallax_degrees(points: np.ndarray, centre_a: np.ndarray,
                     centre_b: np.ndarray) -> float:
    """Median angle each point subtends between the two camera centres.

    This is the honest measure of how much a pair actually tells us. A shot
    taken from the same spot with the camera merely turned scores ~0 no matter
    how many features match.
    """
    if len(points) == 0:
        return 0.0
    to_a = points - centre_a
    to_b = points - centre_b
    to_a /= np.maximum(np.linalg.norm(to_a, axis=1, keepdims=True), 1e-9)
    to_b /= np.maximum(np.linalg.norm(to_b, axis=1, keepdims=True), 1e-9)
    cosine = np.clip(np.sum(to_a * to_b, axis=1), -1.0, 1.0)
    return float(np.degrees(np.median(np.arccos(cosine))))


def sample(field_map: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Nearest-pixel lookup of a per-pixel map at sub-pixel feature positions."""
    height, width = field_map.shape[:2]
    x = np.clip(np.round(points[:, 0]).astype(int), 0, width - 1)
    y = np.clip(np.round(points[:, 1]).astype(int), 0, height - 1)
    return field_map[y, x]


def fit_disparity(disparity: np.ndarray, distance: np.ndarray):
    """Least-squares fit of ``disparity ~= a / distance + b``.

    Depth Anything's output is inverse depth put through an unknown affine
    transform, which is exactly this two-parameter family. Fitting against
    triangulated points is what converts a relative map into one that agrees
    with the other shots, and it is linear in 1/distance, so there is nothing
    iterative to go wrong.
    """
    usable = (distance > 1e-6) & np.isfinite(distance) & np.isfinite(disparity)
    if usable.sum() < 8:
        return None
    inverse = 1.0 / distance[usable]
    design = np.column_stack([inverse, np.ones(usable.sum())])
    solution, *_ = np.linalg.lstsq(design, disparity[usable], rcond=None)
    a, b = float(solution[0]), float(solution[1])
    if a <= 1e-9:                       # inverted fit: the map disagrees entirely
        return None
    return a, b


def apply_fit(disparity: np.ndarray, fit: tuple[float, float],
              far: float = 1e4) -> np.ndarray:
    """Turn a relative disparity map into distances using a fitted a, b."""
    a, b = fit
    denominator = disparity - b
    # Sky and anything at or beyond the fitted horizon goes to `far` rather than
    # negative or enormous distances, which would otherwise spray points behind
    # the camera and wreck the fused cloud.
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = np.where(denominator > 1e-6, a / denominator, far)
    return np.clip(np.nan_to_num(distance, nan=far, posinf=far), 1e-3, far)


def place(first: Shot, second: Shot, described_a, described_b,
          camera: np.ndarray) -> PairReport:
    """Put `second` into the world using `first`, which must already be placed.

    If `first` has no fitted depth yet it is the root of the whole scene: its
    depth map is fitted to this pair's unit-baseline geometry, and that choice
    fixes the world unit for everything placed after it. Otherwise its existing
    depth measures this pair's baseline, which is how scale propagates rather
    than being re-guessed at every step.
    """
    keypoints_a, descriptors_a = described_a
    keypoints_b, descriptors_b = described_b
    found = match(descriptors_a, descriptors_b)
    report = PairReport(first.name, second.name, len(found), 0, 0.0)

    if len(found) < MIN_INLIERS:
        report.ok, report.note = False, "too few matches; shots may not overlap"
        return report

    points_a, points_b = matched_points(keypoints_a, keypoints_b, found)
    rotation, direction, inliers = relative_pose(points_a, points_b, camera)
    report.inliers = int(inliers.sum())
    share = homography_share(points_a, points_b, max(report.inliers, 1))
    if share > MAX_HOMOGRAPHY_SHARE or rotation is None and share > 0.5:
        report.ok = False
        report.note = ("the camera rotated but barely moved, so there is no "
                       "parallax to work from (or the whole view is one flat surface)")
        return report
    if rotation is None or report.inliers < MIN_INLIERS:
        report.ok, report.note = False, "pose recovery failed"
        return report

    points_a, points_b = points_a[inliers], points_b[inliers]

    # Solve the pair on its own, with a unit baseline, in shot A's frame.
    local = triangulate(camera, np.eye(3), np.zeros(3), rotation, direction,
                        points_a, points_b)
    in_front = local[:, 2] > 0
    local, points_a, points_b = local[in_front], points_a[in_front], points_b[in_front]
    report.parallax = parallax_degrees(local, np.zeros(3), -rotation.T @ direction)
    if report.parallax < MIN_PARALLAX:
        report.ok = False
        report.note = (f"only {report.parallax:.2f} deg of parallax; the camera "
                       "rotated but barely moved")
        return report

    if first.depth is None:
        if first.disparity is None:
            report.ok, report.note = False, "no depth map for the first shot"
            return report
        fit = fit_disparity(sample(first.disparity, points_a), local[:, 2])
        if fit is None:
            report.ok, report.note = False, "depth map does not fit the geometry"
            return report
        first.depth = apply_fit(first.disparity, fit)

    known = sample(first.depth, points_a)
    usable = (local[:, 2] > 1e-6) & np.isfinite(known)
    if usable.sum() < 8:
        report.ok, report.note = False, "not enough overlap to carry the scale"
        return report
    ratios = np.log(known[usable] / local[usable, 2])
    centre_ratio = float(np.median(ratios))
    spread = float(np.median(np.abs(ratios - centre_ratio)))
    if spread > MAX_SCALE_SPREAD:
        report.ok = False
        report.note = (f"the depth map and the geometry disagree (spread {spread:.2f}); "
                       "likely reflections or moving people")
        return report
    scale = float(np.exp(centre_ratio))
    report.scale = scale

    second.rotation = rotation @ first.rotation
    second.centre = first.centre + first.rotation.T @ (-rotation.T @ direction * scale)

    # Fit shot B's own depth map against the same points, now in B's frame.
    if second.disparity is not None:
        world = first.centre + (first.rotation.T @ (local * scale).T).T
        in_b = (second.rotation @ (world - second.centre).T).T
        fit = fit_disparity(sample(second.disparity, points_b), in_b[:, 2])
        if fit is not None:
            second.depth = apply_fit(second.disparity, fit)

    return report


def solve(shots: list[Shot], camera: np.ndarray) -> list[PairReport]:
    """Place shots strictly in capture order, stopping at the first break.

    The simple case, kept because it is what a careful orbit produces and its
    failures are easy to read. Real captures rarely behave like this; see
    ``solve_graph`` for the version that tolerates jumps.
    """
    reports: list[PairReport] = []
    if len(shots) < 2:
        return reports
    described = [features(shot.image) for shot in shots]
    shots[0].rotation = np.eye(3)
    shots[0].centre = np.zeros(3)
    for index in range(len(shots) - 1):
        report = place(shots[index], shots[index + 1],
                       described[index], described[index + 1], camera)
        reports.append(report)
        if not report.ok:
            break
    return reports


def overlap(shots: list[Shot], camera: np.ndarray, described=None) -> np.ndarray:
    """Pose-consistent match count for every pair of shots.

    Inliers after essential-matrix recovery, not raw matches: a raw count is
    easily inflated by wet ground and neon, whose reflections match well from
    anywhere while agreeing with no single camera motion.
    """
    described = described or [features(shot.image) for shot in shots]
    count = len(shots)
    table = np.zeros((count, count), int)
    for i in range(count):
        for j in range(i + 1, count):
            found = match(described[i][1], described[j][1])
            if len(found) < MIN_INLIERS:
                continue
            points_a, points_b = matched_points(described[i][0], described[j][0], found)
            rotation, _, inliers = relative_pose(points_a, points_b, camera)
            if rotation is not None:
                table[i, j] = table[j, i] = int(inliers.sum())
    return table


def solve_graph(shots: list[Shot], camera: np.ndarray):
    """Place as many shots as the overlap allows, in whatever order works.

    A real capture does not arrive as a tidy orbit: people jump across the
    scene, climb stairs, look back the way they came. Chaining in capture order
    dies at the first jump and orphans everything after it. Here every pair is
    scored, the strongest pair seeds the scene, and each step adds the unplaced
    shot with the strongest link to any shot already placed (a greedy maximum
    spanning tree). Shots nothing overlaps are left out and named.

    Returns the placement reports and the overlap table, so the caller can show
    the user which shots were orphaned and why.
    """
    reports: list[PairReport] = []
    if len(shots) < 2:
        return reports, np.zeros((len(shots), len(shots)), int)

    described = [features(shot.image) for shot in shots]
    table = overlap(shots, camera, described)
    if table.max() < MIN_INLIERS:
        return reports, table

    placed: set[int] = set()
    rejected: set[tuple[int, int]] = set()
    # Seed from the strongest pair that is also well conditioned. The pair with
    # the most matches is usually two near-identical shots: plenty of inliers,
    # almost no baseline, and a world unit so small every later scale is noise.
    candidates = sorted(
        ((int(table[i, j]), i, j) for i in range(len(shots))
         for j in range(len(shots)) if i != j and table[i, j] >= MIN_INLIERS),
        reverse=True)
    for _, seed_a, seed_b in candidates:
        root, child = shots[seed_a], shots[seed_b]
        root.rotation, root.centre, root.depth = np.eye(3), np.zeros(3), None
        report = place(root, child, described[seed_a], described[seed_b], camera)
        if report.ok and report.parallax >= SEED_PARALLAX and child.depth is not None:
            reports.append(report)
            placed = {seed_a, seed_b}
            break
        root.depth = child.depth = None
    if not placed:
        return reports, table

    while True:
        best, link = 0, None
        for parent in placed:
            for child in range(len(shots)):
                if child in placed or (parent, child) in rejected:
                    continue
                if table[parent, child] > best:
                    best, link = table[parent, child], (parent, child)
        if link is None or best < MIN_INLIERS:
            break
        parent, child = link
        report = place(shots[parent], shots[child],
                       described[parent], described[child], camera)
        reports.append(report)
        if report.ok and shots[child].depth is not None:
            placed.add(child)
        else:
            # This edge failed (too little parallax, bad fit); another placed
            # shot may still reach the child, so only the edge is ruled out.
            rejected.add(link)

    return reports, table


def unproject(shot: Shot, camera: np.ndarray, stride: int = 2,
              far_cut: float = 200.0, with_pixels: bool = False):
    """Every stride'th pixel of one shot as world points plus colours.

    With ``with_pixels`` the source (y, x) of each point comes back too, so a
    per-pixel map such as the trust tiers can be applied to the result.
    """
    empty = (np.zeros((0, 3)), np.zeros((0, 3), np.uint8))
    if shot.depth is None:
        return empty + ((np.zeros(0, int), np.zeros(0, int)),) if with_pixels else empty
    depth = shot.depth[::stride, ::stride]
    colour = shot.image[::stride, ::stride]
    height, width = depth.shape
    ys, xs = np.mgrid[0:height, 0:width]
    xs = xs * stride
    ys = ys * stride

    # Sky and the fitted horizon sit at `far`; carrying them into the cloud
    # would dwarf the scene and make every viewport frame useless.
    keep = depth < far_cut
    if not keep.any():
        return empty + ((np.zeros(0, int), np.zeros(0, int)),) if with_pixels else empty

    z = depth[keep]
    x = (xs[keep] - camera[0, 2]) / camera[0, 0] * z
    y = (ys[keep] - camera[1, 2]) / camera[1, 1] * z
    in_camera = np.column_stack([x, y, z])
    world = shot.centre + (shot.rotation.T @ in_camera.T).T
    if with_pixels:
        return world, colour[keep], (ys[keep], xs[keep])
    return world, colour[keep]


def agreement(points: np.ndarray, source: Shot, others: list[Shot],
              camera: np.ndarray, tolerance: float = 0.08):
    """How many other shots confirm each point, and how many could have.

    A point is *confirmed* by a shot when it projects inside that shot's frame
    and that shot's own depth at that pixel is within `tolerance` (relative) of
    the point's distance. It is *contradicted* when it projects inside the frame
    and the depth there is clearly behind it: the other shot looked straight
    through where this point claims a surface is. That is what a ghost is.

    Single-image depth is only affine-corrected here, so shots never agree
    exactly; what survives voting is the geometry that persists across views,
    which is the part worth trusting.
    """
    confirmed = np.zeros(len(points), np.int32)
    contradicted = np.zeros(len(points), np.int32)
    for other in others:
        if other is source or other.depth is None:
            continue
        cam = (other.rotation @ (points - other.centre).T).T
        z = cam[:, 2]
        front = z > 1e-3
        u = np.full(len(points), -1)
        v = np.full(len(points), -1)
        u[front] = np.round(camera[0, 0] * cam[front, 0] / z[front] + camera[0, 2]).astype(int)
        v[front] = np.round(camera[1, 1] * cam[front, 1] / z[front] + camera[1, 2]).astype(int)
        height, width = other.depth.shape
        inside = front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        seen = other.depth[v[inside], u[inside]]
        relative = (seen - z[inside]) / np.maximum(z[inside], 1e-6)
        index = np.flatnonzero(inside)
        confirmed[index[np.abs(relative) < tolerance]] += 1
        # The other shot saw clearly *past* this point: it cannot be a surface.
        contradicted[index[relative > 3 * tolerance]] += 1
    return confirmed, contradicted


def fuse(shots: list[Shot], camera: np.ndarray, stride: int = 2,
         voxel: float = 0.0, consistent: bool = False,
         tolerance: float = 0.08, min_confidence: float = 0.9):
    """Every solved shot's points merged into one cloud.

    ``voxel`` above zero collapses the cloud onto a grid, which both shrinks it
    and removes the duplicate surfaces that overlapping shots inevitably
    produce: the same wall seen three times becomes one wall.
    """
    clouds, colours = [], []
    placed = [shot for shot in shots if shot.depth is not None]
    for shot in placed:
        points, colour, (ys, xs) = unproject(shot, camera, stride, with_pixels=True)
        if len(points) and consistent and len(placed) > 1:
            # Keep what another shot confirms, drop what another shot saw
            # through. Points no other shot could see are kept: they are the
            # unique coverage multi-view exists to add, not ghosts.
            confirmed, contradicted = agreement(points, shot, placed, camera, tolerance)
            votes = confirmed + contradicted
            with np.errstate(divide="ignore", invalid="ignore"):
                confidence = np.where(votes > 0, confirmed / np.maximum(votes, 1), 1.0)
            if shot.required is not None:
                # Trust tiers: the bar depends on how far this pixel is from
                # trusted atoms, and NaN (beyond the hard cut) always drops it.
                bar = shot.required[ys, xs]
                keep = np.isfinite(bar) & (confidence >= np.nan_to_num(bar, nan=2.0))
                # A point no other shot could see has no evidence against it,
                # and it is exactly the coverage multi-view exists to add, so the
                # distance cut does not apply to it (the user's decision).
                keep |= votes == 0
            else:
                keep = confidence >= min_confidence
            points, colour = points[keep], colour[keep]
        if len(points):
            clouds.append(points)
            colours.append(colour)
    if not clouds:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    points = np.vstack(clouds)
    colour = np.vstack(colours)
    if voxel > 0:
        keys = np.round(points / voxel).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        points, colour = points[first], colour[first]
    return points, colour


def diagnose(reports: list[PairReport], shots: list[Shot]) -> list[str]:
    """Plain-language problems with a capture, for the user rather than the log.

    Most failures here are capture mistakes, not software faults, and the only
    useful thing to do about them is say which shot went wrong and why.
    """
    notes: list[str] = []
    if len(shots) < 2:
        notes.append("Add more shots: one image cannot show the parts it hides.")
        return notes
    solved = sum(1 for shot in shots if shot.depth is not None)
    if solved < len(shots):
        notes.append(f"Only {solved} of {len(shots)} shots could be placed.")
    for report in reports:
        if not report.ok:
            notes.append(f"{report.first} to {report.second}: {report.note}")
    weak = [r for r in reports if r.ok and r.parallax < 3.0]
    if weak:
        notes.append("Some shots are very close together. Move further between "
                     "captures for more of the scene to be recovered.")
    return notes


def write_ply(path, points: np.ndarray, colours: np.ndarray) -> None:
    """A coloured point cloud any 3D tool will open, for eyeballing a result."""
    header = (f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    rows = np.column_stack([points, colours.astype(np.int32)])
    with open(path, "w", encoding="ascii") as handle:
        handle.write(header)
        for row in rows:
            handle.write(f"{row[0]:.5f} {row[1]:.5f} {row[2]:.5f} "
                         f"{int(row[3])} {int(row[4])} {int(row[5])}\n")
