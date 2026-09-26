"""Atomise the shots: solve for the *items*, and let the cameras follow.

The first multi-view attempt trusted each shot's depth map and only asked the
other shots to vote afterwards. Poses came out close, but every single-image
depth map is warped in its own way, so the same monorail landed at a slightly
different depth in each shot and the fused scene ghosted.

This turns it around. Every shot is broken into atoms (SIFT features: a corner,
a letter on a sign, the foot of a lamp post), the same atom is followed across
every shot that sees it, and its 3D position is solved from *all* of those
sightings at once. An atom is trusted only when at least 90% of the shots that
see it agree on where it is, and it must be seen by at least three, since two
sightings always agree with each other and prove nothing.

Those trusted atoms then do three jobs:

  1. Place shots that no single pair could: a shot looking down at a flat floor
     fails the two-view test, but if it sees forty atoms that are already
     positioned, PnP places it directly.
  2. Refine every camera against the shared atoms, and re-solve the atoms from
     the refined cameras, until the two stop moving. A bundle adjustment done by
     alternation, without pulling in SciPy for a joint solver.
  3. Bend each shot's depth map onto the atoms it sees. Not one affine for the
     whole map, which is what could not fix the warping, but a smooth field
     that varies across the image.

The dense points built from the bent depth are then held to the same 90% rule
by the other shots.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import multiview as mv


#: The user's rule: keep what at least 90% of the evidence agrees on.
MIN_CONFIDENCE = 0.9

#: Two sightings always agree; confidence only means something from three.
MIN_VIEWS = 3

#: Reprojection error, in pixels, under which a sighting agrees with an atom.
AGREE_PX = 2.0

#: An atom whose sightings all come from nearly the same direction has an
#: undetermined depth, however well they agree.
MIN_ATOM_ANGLE = 1.5

#: Correspondences needed before PnP is allowed to place a shot.
MIN_PNP = 40

#: Trust tiers (the user's design). Atoms that pass the full rule are tier 0.
#: An atom next to a tier-0 atom only needs 80% agreement, one next to that
#: 70%, and so on, until the requirement would fall below HARD_CUT, where
#: nothing is accepted however well connected. Trust spreads outward from the
#: solid core and weakens with every step away from it.
TIER_STEP = 0.1
HARD_CUT = 0.5

#: Two atoms are neighbours when some shot sees them this close together in
#: the image (at the working resolution) ...
NEIGHBOUR_PX = 60.0

#: ... and at a similar depth there. Without this, a person standing in front
#: of a far wall would lend their trust to the wall, or the wall to them.
NEIGHBOUR_DEPTH = 0.25


def tier_threshold(tier: int) -> float:
    """Agreement an atom of this tier needs; below HARD_CUT means rejected."""
    return MIN_CONFIDENCE - TIER_STEP * tier


@dataclass
class Atom:
    """One item followed across shots: a list of (shot, keypoint) sightings."""

    sightings: list[tuple[int, int]]
    position: np.ndarray | None = None
    confidence: float = 0.0
    views: int = 0            # sightings in shots that are currently placed
    tier: int = -1            # -1: not accepted; 0: trusted core; n: n steps out

    @property
    def trusted(self) -> bool:
        return (self.position is not None and self.views >= MIN_VIEWS
                and self.confidence >= MIN_CONFIDENCE)

    @property
    def usable(self) -> bool:
        """Good enough to help *place* a shot, not yet to be kept.

        Two agreeing sightings with a real angle between them. Without this
        the scene cannot grow past its first pair: no atom can be seen by three
        cameras while only two are placed, so nothing is ever trusted.
        """
        return (self.position is not None and self.views >= 2
                and self.confidence >= MIN_CONFIDENCE)


@dataclass
class AtomReport:
    matches: int = 0
    atoms: int = 0
    trusted: int = 0
    placed_by_pairs: int = 0
    placed_by_atoms: int = 0
    reprojection: float = 0.0
    bent: int = 0
    focal: float = 0.0
    tiers: dict | None = None


# --- building atoms ---------------------------------------------------------

def verified_matches(described: list, camera: np.ndarray) -> dict:
    """Epipolar-consistent keypoint matches for every pair of shots.

    Verified with a fundamental matrix rather than a pose, on purpose: a pair
    that only rotated is useless for *placing* a camera, but its matches are
    still perfectly good evidence that two pixels are the same item. Those are
    exactly the links that tie a floor shot into the rest of the scene.
    """
    pairs: dict[tuple[int, int], np.ndarray] = {}
    count = len(described)
    for i in range(count):
        for j in range(i + 1, count):
            found = mv.match(described[i][1], described[j][1])
            if len(found) < mv.MIN_INLIERS // 2:
                continue
            points_a, points_b = mv.matched_points(described[i][0], described[j][0], found)
            _, mask = cv2.findFundamentalMat(points_a, points_b, cv2.USAC_MAGSAC, 1.5, 0.9999)
            if mask is None:
                continue
            keep = mask.ravel().astype(bool)
            if keep.sum() < mv.MIN_INLIERS // 2:
                continue
            index = np.array([(m.queryIdx, m.trainIdx) for m in found], np.int64)
            pairs[(i, j)] = index[keep]
    return pairs


def build_atoms(described: list, pairs: dict) -> list[Atom]:
    """Chain pairwise matches into atoms seen by any number of shots.

    Union-find over (shot, keypoint) nodes. An atom that ends up with two
    different keypoints in the same shot is a chain of wrong matches gluing two
    items together, and is dropped whole rather than guessed at.
    """
    offsets = np.cumsum([0] + [len(d[0]) for d in described])
    parent = np.arange(offsets[-1])

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for (i, j), index in pairs.items():
        for a, b in index:
            ra, rb = root(offsets[i] + a), root(offsets[j] + b)
            if ra != rb:
                parent[rb] = ra

    groups: dict[int, list[tuple[int, int]]] = {}
    for shot in range(len(described)):
        for keypoint in range(len(described[shot][0])):
            groups.setdefault(root(offsets[shot] + keypoint), []).append((shot, keypoint))

    atoms = []
    for sightings in groups.values():
        if len(sightings) < 2:
            continue
        shots_seen = [s for s, _ in sightings]
        if len(set(shots_seen)) != len(shots_seen):
            continue
        atoms.append(Atom(sightings))
    return atoms


# --- positioning atoms ------------------------------------------------------

def _projection(shot: mv.Shot, camera: np.ndarray) -> np.ndarray:
    return camera @ np.hstack([shot.rotation, shot.translation().reshape(3, 1)])


def locate(atom: Atom, shots: list[mv.Shot], described: list,
           camera: np.ndarray, placed: set[int]) -> None:
    """Solve an atom's position from every placed shot that sees it.

    Linear least squares over all sightings at once (multi-view DLT), then the
    confidence: the share of those sightings that agree with the answer.
    """
    rows, views = [], []
    for shot_index, keypoint in atom.sightings:
        if shot_index not in placed:
            continue
        x, y = described[shot_index][0][keypoint].pt
        projection = _projection(shots[shot_index], camera)
        rows.append(x * projection[2] - projection[0])
        rows.append(y * projection[2] - projection[1])
        views.append((shot_index, x, y))
    atom.views = len(views)
    if len(views) < 2:
        atom.position, atom.confidence = None, 0.0
        return

    _, _, vt = np.linalg.svd(np.array(rows))
    homogeneous = vt[-1]
    if abs(homogeneous[3]) < 1e-12:
        atom.position, atom.confidence = None, 0.0
        return
    point = homogeneous[:3] / homogeneous[3]

    agree, directions = 0, []
    for shot_index, x, y in views:
        shot = shots[shot_index]
        in_camera = shot.rotation @ (point - shot.centre)
        if in_camera[2] <= 1e-6:
            continue
        u = camera[0, 0] * in_camera[0] / in_camera[2] + camera[0, 2]
        v = camera[1, 1] * in_camera[1] / in_camera[2] + camera[1, 2]
        if (u - x) ** 2 + (v - y) ** 2 < AGREE_PX ** 2:
            agree += 1
            ray = point - shot.centre
            directions.append(ray / max(np.linalg.norm(ray), 1e-9))

    atom.confidence = agree / len(views)
    if len(directions) >= 2:
        stack = np.array(directions)
        widest = np.degrees(np.arccos(np.clip(stack @ stack.T, -1.0, 1.0)).max())
        if widest < MIN_ATOM_ANGLE:
            atom.confidence = 0.0      # agrees, but the depth is undetermined
    atom.position = point


def _pnp(atoms: list[Atom], shot_index: int, described: list,
         camera: np.ndarray, guess: mv.Shot | None = None):
    """Place one shot from the trusted atoms it can see."""
    objects, images = [], []
    for atom in atoms:
        if not atom.usable:
            continue
        for index, keypoint in atom.sightings:
            if index == shot_index:
                objects.append(atom.position)
                images.append(described[shot_index][0][keypoint].pt)
                break
    if len(objects) < MIN_PNP:
        return None
    objects = np.array(objects, np.float64)
    images = np.array(images, np.float64)

    if guess is None:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objects, images, camera, None, reprojectionError=3.0,
            confidence=0.999, iterationsCount=3000)
        if not ok or inliers is None or len(inliers) < 0.9 * MIN_PNP:
            return None
        keep = inliers.ravel()
    else:
        rvec, _ = cv2.Rodrigues(guess.rotation)
        tvec = guess.translation().reshape(3, 1)
        keep = np.arange(len(objects))
    ok, rvec, tvec = cv2.solvePnP(objects[keep], images[keep], camera, None,
                                  rvec, tvec, useExtrinsicGuess=True,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    return rotation, (-rotation.T @ tvec).ravel()


def _adjust(shots, atoms, described, camera, placed, root, rejected):
    """One bundle adjustment over the placed shots and their usable atoms.

    Writes the refined cameras, atom positions and focal length back, and
    returns the new camera matrix and the median reprojection error in pixels.
    """
    from . import bundle

    order = sorted(placed)
    slot = {index: position for position, index in enumerate(order)}
    used, camera_index, point_index, observed = [], [], [], []
    for atom_index, atom in enumerate(atoms):
        if atom_index in rejected or atom.position is None or atom.views < 2:
            continue
        sightings = [(i, k) for i, k in atom.sightings if i in placed]
        if any((shots[i].rotation @ (atom.position - shots[i].centre))[2] <= 1e-6
               for i, _ in sightings):
            continue
        point_slot = len(used)
        used.append(atom_index)
        for shot_index, keypoint in sightings:
            camera_index.append(slot[shot_index])
            point_index.append(point_slot)
            observed.append(described[shot_index][0][keypoint].pt)
    if len(used) < 10:
        return camera, float("nan")

    rotations = np.array([cv2.Rodrigues(shots[i].rotation)[0].ravel() for i in order])
    translations = np.array([shots[i].translation() for i in order])
    problem = bundle.Problem(
        rotations, translations, np.array([atoms[i].position for i in used]),
        float(camera[0, 0]), (float(camera[0, 2]), float(camera[1, 2])),
        np.array(camera_index), np.array(point_index), np.array(observed, np.float64),
        fixed=slot[root])
    result = bundle.adjust(problem)

    for position, index in enumerate(order):
        rotation = cv2.Rodrigues(problem.rotations[position])[0]
        shots[index].rotation = rotation
        shots[index].centre = -rotation.T @ problem.translations[position]
    for position, atom_index in enumerate(used):
        atoms[atom_index].position = problem.points[position]
    refined = camera.copy()
    refined[0, 0] = refined[1, 1] = problem.focal
    return refined, result["median_px"]


def _sightings_in(shot_index: int, atoms: list[Atom], shots, described, members):
    """Pixel position and depth, in one shot, of the given atoms it sees."""
    shot = shots[shot_index]
    rows = []
    for atom_index in members:
        atom = atoms[atom_index]
        for index, keypoint in atom.sightings:
            if index == shot_index:
                depth = (shot.rotation @ (atom.position - shot.centre))[2]
                if depth > 1e-6:
                    x, y = described[index][0][keypoint].pt
                    rows.append((atom_index, x, y, depth))
                break
    return rows


def assign_tiers(atoms: list[Atom], shots, described, placed: set[int],
                 hard_cut: float = HARD_CUT) -> dict[int, int]:
    """Grow trust outward from the core, one tier per step, to the hard cut.

    Breadth first: tier n is every not-yet-accepted atom that sits next to a
    tier n-1 atom in some shot and meets tier n's lower bar. An atom's tier is
    therefore its distance from the core, and a neighbour of a better atom is
    never penalised for also touching a worse one. Returns a tier histogram.
    """
    for atom in atoms:
        atom.tier = 0 if atom.trusted else -1
    frontier = {i for i, atom in enumerate(atoms) if atom.tier == 0}
    counts = {0: len(frontier)}
    tier = 1
    while frontier and tier_threshold(tier) >= hard_cut - 1e-9:
        bar = tier_threshold(tier)
        candidates = {i for i, atom in enumerate(atoms)
                      if atom.tier < 0 and atom.position is not None
                      and atom.views >= 2 and atom.confidence >= bar}
        if not candidates:
            break
        reached: set[int] = set()
        for shot_index in placed:
            near = _sightings_in(shot_index, atoms, shots, described, frontier)
            far = _sightings_in(shot_index, atoms, shots, described, candidates - reached)
            if not near or not far:
                continue
            near_xy = np.array([(x, y) for _, x, y, _ in near])
            near_depth = np.array([d for *_, d in near])
            for atom_index, x, y, depth in far:
                gap = np.hypot(near_xy[:, 0] - x, near_xy[:, 1] - y)
                close = gap < NEIGHBOUR_PX
                if close.any() and (np.abs(near_depth[close] - depth)
                                    / np.maximum(near_depth[close], 1e-6)
                                    < NEIGHBOUR_DEPTH).any():
                    reached.add(atom_index)
        for atom_index in reached:
            atoms[atom_index].tier = tier
        if reached:
            counts[tier] = len(reached)
        frontier = reached
        tier += 1
    return counts


def requirement_map(shot_index: int, atoms: list[Atom], shots, described,
                    hard_cut: float = HARD_CUT) -> np.ndarray | None:
    """Per-pixel confidence a fused point from this shot must reach.

    The tiers carried into the dense points: a pixel takes the tier of its
    nearest accepted atom in this shot, plus one more tier for every
    NEIGHBOUR_PX further away. Past the hard cut the pixel is NaN and every
    point from it is dropped, so geometry no chain of evidence reaches cannot
    survive however unopposed it is.
    """
    shot = shots[shot_index]
    accepted = [i for i, atom in enumerate(atoms) if atom.tier >= 0]
    rows = _sightings_in(shot_index, atoms, shots, described, accepted)
    if not rows:
        return None
    height, width = shot.image.shape[:2]
    # Nearest accepted atom per pixel, via a distance transform over a map
    # holding each atom's tier (lower tiers win where two share a pixel).
    tiers = np.full((height, width), 255, np.uint8)
    for atom_index, x, y, _ in sorted(rows, key=lambda r: -atoms[r[0]].tier):
        tiers[min(int(y), height - 1), min(int(x), width - 1)] = atoms[atom_index].tier
    seeds = (tiers != 255).astype(np.uint8)
    distance, labels = cv2.distanceTransformWithLabels(
        1 - seeds, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    seed_y, seed_x = np.nonzero(seeds)
    order = np.zeros(labels.max() + 1, np.int32)
    order[labels[seed_y, seed_x]] = tiers[seed_y, seed_x]
    tier = order[labels] + np.floor(distance / NEIGHBOUR_PX).astype(np.int32)
    bar = MIN_CONFIDENCE - TIER_STEP * tier
    return np.where(bar >= hard_cut - 1e-9, bar, np.nan).astype(np.float32)


# --- bending depth onto the atoms -------------------------------------------

def bend_field(shape: tuple[int, int], pixels: np.ndarray, log_ratio: np.ndarray,
               grid: tuple[int, int] = (16, 8)) -> np.ndarray:
    """A smooth per-pixel log-scale correction through sparse samples.

    Each grid cell takes the median of the samples inside it; empty cells are
    filled from their neighbours; the result is smoothed and scaled up to the
    image. Medians, not means: a few atoms on a reflection must not drag a
    whole region of wall with them.
    """
    height, width = shape
    columns, rows = grid
    total = np.zeros((rows, columns), np.float32)
    known = np.zeros((rows, columns), np.float32)
    cell_x = np.clip((pixels[:, 0] / width * columns).astype(int), 0, columns - 1)
    cell_y = np.clip((pixels[:, 1] / height * rows).astype(int), 0, rows - 1)
    for y in range(rows):
        for x in range(columns):
            inside = (cell_x == x) & (cell_y == y)
            if inside.sum() >= 3:
                total[y, x] = np.median(log_ratio[inside])
                known[y, x] = 1.0
    if known.sum() == 0:
        return np.zeros(shape, np.float32)

    field, weight = total * known, known.copy()
    for _ in range(max(rows, columns)):
        if (weight > 0).all():
            break
        spread_value = cv2.blur(field, (3, 3), borderType=cv2.BORDER_REPLICATE)
        spread_weight = cv2.blur(weight, (3, 3), borderType=cv2.BORDER_REPLICATE)
        empty = weight == 0
        filled = spread_weight > 0
        field[empty & filled] = spread_value[empty & filled] / spread_weight[empty & filled]
        weight[empty & filled] = 1.0
    field = cv2.GaussianBlur(field, (3, 3), 0)
    return cv2.resize(field, (width, height), interpolation=cv2.INTER_CUBIC)


def bend(shot: mv.Shot, index: int, atoms: list[Atom], described: list,
         camera: np.ndarray) -> bool:
    """Refit a shot's depth to the trusted atoms it sees. False if too few."""
    if shot.disparity is None:
        return False
    pixels, distances = [], []
    for atom in atoms:
        if atom.tier < 0:
            continue
        for shot_index, keypoint in atom.sightings:
            if shot_index == index:
                in_camera = shot.rotation @ (atom.position - shot.centre)
                if in_camera[2] > 1e-6:
                    pixels.append(described[index][0][keypoint].pt)
                    distances.append(in_camera[2])
                break
    if len(pixels) < MIN_PNP:
        return False
    pixels = np.array(pixels)
    distances = np.array(distances)
    fit = mv.fit_disparity(mv.sample(shot.disparity, pixels), distances)
    if fit is None:
        return False
    base = mv.apply_fit(shot.disparity, fit)
    ratio = np.log(distances / mv.sample(base, pixels))
    field = bend_field(base.shape, pixels, np.clip(ratio, -1.5, 1.5))
    shot.depth = base * np.exp(field)
    return True


# --- the whole thing --------------------------------------------------------

def solve(shots: list[mv.Shot], camera: np.ndarray, rounds: int = 4):
    """Place, refine, and bend. Expects ``mv.solve_graph`` to have run first.

    The pair solver provides the starting cluster and the world scale; from
    there on everything is driven by the atoms. Returns the report, the atoms
    and the per-shot features, so a viewer can draw the atoms themselves.
    """
    report = AtomReport()
    described = [mv.features(shot.image) for shot in shots]
    pairs = verified_matches(described, camera)
    report.matches = sum(len(v) for v in pairs.values())
    atoms = build_atoms(described, pairs)
    report.atoms = len(atoms)

    placed = {i for i, shot in enumerate(shots) if shot.depth is not None}
    report.placed_by_pairs = len(placed)
    if len(placed) < 2:
        return report, atoms, described, camera
    root = min(placed)
    rejected: set[int] = set()

    for round_number in range(rounds):
        for atom in atoms:
            locate(atom, shots, described, camera, placed)

        # Solve every placed camera, every usable atom and the focal length at
        # once. This is what makes the cameras agree with each other rather
        # than only with whichever shot they were chained from.
        camera, adjusted = _adjust(shots, atoms, described, camera, placed, root, rejected)
        report.reprojection = adjusted

        for index, atom in enumerate(atoms):
            locate(atom, shots, described, camera, placed)
            # The 90% rule: an atom three or more shots see, which they do not
            # agree on, is thrown out of every later solve.
            if atom.views >= MIN_VIEWS and atom.confidence < MIN_CONFIDENCE:
                rejected.add(index)

        grew = False
        for index in range(len(shots)):
            if index in placed:
                continue
            pose = _pnp(atoms, index, described, camera)
            if pose is not None:
                shots[index].rotation, shots[index].centre = pose
                placed.add(index)
                grew = True
                report.placed_by_atoms += 1
        if not grew and round_number > 0:
            break

    camera, report.reprojection = _adjust(shots, atoms, described, camera, placed,
                                          root, rejected)
    report.focal = float(camera[0, 0])
    for atom in atoms:
        locate(atom, shots, described, camera, placed)
    trusted = [atom for atom in atoms if atom.trusted]
    report.trusted = len(trusted)

    for atom in atoms:
        locate(atom, shots, described, camera, placed)
    report.tiers = assign_tiers(atoms, shots, described, placed)

    for index in placed:
        shots[index].required = requirement_map(index, atoms, shots, described)
        if bend(shots[index], index, atoms, described, camera):
            report.bent += 1
        else:
            shots[index].depth = None     # an unbent map would reintroduce ghosts
    return report, atoms, described, camera


def atom_cloud(atoms: list[Atom], shots: list[mv.Shot], described: list):
    """Trusted atom positions coloured from one of their sightings."""
    points, colours = [], []
    for atom in atoms:
        if not atom.trusted:
            continue
        index, keypoint = atom.sightings[0]
        x, y = described[index][0][keypoint].pt
        points.append(atom.position)
        colours.append(shots[index].image[int(y), int(x)])
    return np.array(points).reshape(-1, 3), np.array(colours, np.uint8).reshape(-1, 3)
