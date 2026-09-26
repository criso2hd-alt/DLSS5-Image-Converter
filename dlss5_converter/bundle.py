"""Bundle adjustment: every camera and every atom, solved together.

Placing cameras one pair at a time gives poses that agree two at a time and
disagree as soon as a third shot weighs in: on the Cyberpunk capture, atoms
seen by three chained cameras reprojected up to 40 px apart. The fix is to stop
solving cameras individually and minimise the reprojection error of all
sightings of all atoms at once, cameras and atoms and the lens together.

Written on numpy because the app does not ship SciPy and this is the only thing
that would want it. It is a textbook Levenberg-Marquardt with the Schur
complement: the point blocks are 3x3 and invert independently, which reduces
the system to one dense camera-sized solve. With at most a few dozen shots that
system is a couple of hundred unknowns, so numeric Jacobians are fast enough and
far less error-prone than hand-derived ones.

The lens is shared by every shot. By default only its focal length is solved:
games rarely state their field of view reliably, and a wrong focal length is
otherwise indistinguishable from wrong geometry. ``lens=True`` also solves
separate horizontal and vertical focal lengths, the optical centre and two
radial distortion terms. On the Stray capture the error grew from 1.7 px at
the image centre to 6.1 px in the corners under the one-focal pinhole: the
signature of a lens model that does not match how the game projects at 100
degrees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The full lens, in this order: fx, fy, cx, cy, k1, k2.
LENS_SIZE = 6


@dataclass
class Problem:
    """Everything the solver moves, plus the sightings it must explain."""

    rotations: np.ndarray       # (cameras, 3) Rodrigues vectors, world -> camera
    translations: np.ndarray    # (cameras, 3)
    points: np.ndarray          # (atoms, 3) world positions
    focal: float
    centre: tuple[float, float]
    camera_index: np.ndarray    # (sightings,) which camera saw it
    point_index: np.ndarray     # (sightings,) which atom it is
    observed: np.ndarray        # (sightings, 2) pixel position
    fixed: int = 0              # this camera stays put: it anchors the world
    distortion: tuple[float, float] = (0.0, 0.0)   # k1, k2
    focal_y: float | None = None                   # None: same as focal

    def lens(self) -> np.ndarray:
        fy = self.focal if self.focal_y is None else self.focal_y
        return np.array([self.focal, fy, self.centre[0], self.centre[1],
                         self.distortion[0], self.distortion[1]], float)

    def set_lens(self, lens: np.ndarray) -> None:
        self.focal, self.focal_y = float(lens[0]), float(lens[1])
        self.centre = (float(lens[2]), float(lens[3]))
        self.distortion = (float(lens[4]), float(lens[5]))


def rodrigues(vectors: np.ndarray) -> np.ndarray:
    """Batched axis-angle to rotation matrix, (n, 3) -> (n, 3, 3)."""
    angle = np.linalg.norm(vectors, axis=1)
    small = angle < 1e-12
    axis = vectors / np.where(small, 1.0, angle)[:, None]
    axis[small] = (1.0, 0.0, 0.0)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zero = np.zeros_like(x)
    cross = np.stack([np.stack([zero, -z, y], -1),
                      np.stack([z, zero, -x], -1),
                      np.stack([-y, x, zero], -1)], 1)
    cos, sin = np.cos(angle)[:, None, None], np.sin(angle)[:, None, None]
    outer = axis[:, :, None] * axis[:, None, :]
    return cos * np.eye(3) + (1.0 - cos) * outer + sin * cross


def project(rotations, translations, points, focal, centre, distortion=(0.0, 0.0),
            focal_y=None):
    """Pixel positions for per-sighting camera and point parameters.

    Radial distortion is the usual polynomial on the normalised radius:
    ``x_d = x (1 + k1 r^2 + k2 r^4)``. With both terms zero this is the plain
    pinhole projection.
    """
    in_camera = np.einsum("nij,nj->ni", rodrigues(rotations), points) + translations
    depth = np.where(np.abs(in_camera[:, 2]) < 1e-9, 1e-9, in_camera[:, 2])
    x, y = in_camera[:, 0] / depth, in_camera[:, 1] / depth
    k1, k2 = distortion
    if k1 or k2:
        r2 = x * x + y * y
        scale = 1.0 + k1 * r2 + k2 * r2 * r2
        x, y = x * scale, y * scale
    fy = focal if focal_y is None else focal_y
    return np.column_stack([focal * x + centre[0], fy * y + centre[1]])


def _project_lens(rotations, translations, points, lens):
    return project(rotations, translations, points, lens[0], (lens[2], lens[3]),
                   (lens[4], lens[5]), lens[1])


def residuals(problem: Problem) -> np.ndarray:
    c, p = problem.camera_index, problem.point_index
    predicted = _project_lens(problem.rotations[c], problem.translations[c],
                              problem.points[p], problem.lens())
    return predicted - problem.observed


def _huber_weights(errors: np.ndarray, delta: float) -> np.ndarray:
    """Down-weight big residuals so a few wrong matches cannot bend the scene."""
    norm = np.linalg.norm(errors, axis=1)
    return np.where(norm <= delta, 1.0, delta / np.maximum(norm, 1e-12))


def _cost(errors: np.ndarray, delta: float) -> float:
    norm = np.linalg.norm(errors, axis=1)
    return float(np.sum(np.where(norm <= delta, 0.5 * norm ** 2,
                                 delta * (norm - 0.5 * delta))))


def _lens_steps(lens: np.ndarray, eps: float) -> np.ndarray:
    """Finite-difference step per lens parameter, sized to its magnitude."""
    return np.array([eps * max(lens[0], 1.0), eps * max(lens[1], 1.0),
                     eps * max(lens[2], 1.0), eps * max(lens[3], 1.0), 1e-6, 1e-6])


def _jacobians(problem: Problem, lens_params: list[int], eps: float = 1e-6):
    """Per-sighting derivatives: camera (n,2,6), point (n,2,3), lens (n,2,m).

    ``lens_params`` lists which lens entries are solved. In focal-only mode
    that is one shared focal length that moves fx and fy together.
    """
    c, p = problem.camera_index, problem.point_index
    rotations, translations = problem.rotations[c], problem.translations[c]
    points = problem.points[p]
    lens = problem.lens()
    base = _project_lens(rotations, translations, points, lens)

    camera_jacobian = np.empty((len(c), 2, 6))
    for k in range(3):
        moved = rotations.copy()
        moved[:, k] += eps
        camera_jacobian[:, :, k] = (_project_lens(moved, translations, points, lens) - base) / eps
        moved = translations.copy()
        moved[:, k] += eps
        camera_jacobian[:, :, 3 + k] = (_project_lens(rotations, moved, points, lens) - base) / eps
    point_jacobian = np.empty((len(c), 2, 3))
    for k in range(3):
        moved = points.copy()
        moved[:, k] += eps
        point_jacobian[:, :, k] = (_project_lens(rotations, translations, moved, lens) - base) / eps

    steps = _lens_steps(lens, eps)
    lens_jacobian = np.empty((len(c), 2, max(len(lens_params), 1)))
    for column, index in enumerate(lens_params):
        moved = lens.copy()
        if index == -1:                       # shared focal: fx and fy together
            moved[0] += steps[0]
            moved[1] += steps[0]
            step = steps[0]
        else:
            moved[index] += steps[index]
            step = steps[index]
        lens_jacobian[:, :, column] = (_project_lens(rotations, translations, points, moved) - base) / step
    return camera_jacobian, point_jacobian, lens_jacobian


def adjust(problem: Problem, iterations: int = 40, delta: float = 4.0,
           solve_focal: bool = True, lens: bool = False) -> dict:
    """Levenberg-Marquardt over all cameras, atoms and the lens.

    ``lens=False`` (default) solves one shared focal length, as before;
    ``lens=True`` solves fx, fy, cx, cy, k1 and k2. ``solve_focal=False``
    freezes the lens entirely.
    """
    lens_params = list(range(LENS_SIZE)) if lens else [-1]
    if not solve_focal:
        lens_params = []
    m = len(lens_params)

    cameras = len(problem.rotations)
    atoms = len(problem.points)
    free = [i for i in range(cameras) if i != problem.fixed]
    offset = np.full(cameras, -1)
    offset[free] = np.arange(len(free)) * 6
    lens_start = 6 * len(free)
    size = lens_start + max(m, 1)
    lens_slots = lens_start + np.arange(max(m, 1))

    camera_slots = np.where(offset[problem.camera_index, None] >= 0,
                            offset[problem.camera_index, None] + np.arange(6),
                            lens_slots[0])
    slots = np.concatenate([camera_slots,
                            np.broadcast_to(lens_slots, (len(problem.camera_index), max(m, 1)))], 1)
    is_fixed = offset[problem.camera_index] < 0

    damping = 1e-3
    errors = residuals(problem)
    cost = _cost(errors, delta)
    start = cost
    improvement = 1.0

    for _ in range(iterations):
        weights = _huber_weights(errors, delta)
        camera_jacobian, point_jacobian, lens_jacobian = _jacobians(problem, lens_params)
        camera_jacobian[is_fixed] = 0.0
        if m == 0:
            lens_jacobian = np.zeros_like(lens_jacobian)
        block = np.concatenate([camera_jacobian, lens_jacobian], 2)     # (n, 2, 6+m)

        w = weights[:, None, None]
        u_blocks = np.einsum("nki,nkj->nij", block * w, block)
        v_blocks = np.einsum("nki,nkj->nij", point_jacobian * w, point_jacobian)
        w_blocks = np.einsum("nki,nkj->nij", block * w, point_jacobian)
        gradient_c = -np.einsum("nki,nk->ni", block * w, errors)
        gradient_p = -np.einsum("nki,nk->ni", point_jacobian * w, errors)

        u = np.zeros((size, size))
        np.add.at(u, (slots[:, :, None], slots[:, None, :]), u_blocks)
        g_c = np.zeros(size)
        np.add.at(g_c, slots, gradient_c)
        v = np.zeros((atoms, 3, 3))
        np.add.at(v, problem.point_index, v_blocks)
        g_p = np.zeros((atoms, 3))
        np.add.at(g_p, problem.point_index, gradient_p)
        coupling = np.zeros((atoms, size, 3))
        np.add.at(coupling, (problem.point_index[:, None], slots), w_blocks)

        while True:
            u_damped = u + damping * np.diag(np.diag(u) + 1e-9)
            v_damped = v + damping * (v * np.eye(3)) + 1e-9 * np.eye(3)
            v_inverse = np.linalg.inv(v_damped)
            y = np.einsum("pij,pjk->pik", coupling, v_inverse)
            schur = u_damped - np.einsum("pij,pkj->ik", y, coupling)
            rhs = g_c - np.einsum("pij,pj->i", y, g_p)
            try:
                step_c = np.linalg.solve(schur + 1e-12 * np.eye(size), rhs)
            except np.linalg.LinAlgError:
                damping *= 10
                continue
            step_p = np.einsum("pij,pj->pi", v_inverse,
                               g_p - np.einsum("pji,j->pi", coupling, step_c))

            trial = Problem(problem.rotations.copy(), problem.translations.copy(),
                            problem.points + step_p, problem.focal, problem.centre,
                            problem.camera_index, problem.point_index,
                            problem.observed, problem.fixed, problem.distortion,
                            problem.focal_y)
            for camera in free:
                o = offset[camera]
                trial.rotations[camera] += step_c[o:o + 3]
                trial.translations[camera] += step_c[o + 3:o + 6]
            if m:
                new_lens = problem.lens()
                for column, index in enumerate(lens_params):
                    delta_value = step_c[lens_start + column]
                    if index == -1:
                        new_lens[0] += delta_value
                        new_lens[1] += delta_value
                    else:
                        new_lens[index] += delta_value
                new_lens[0] = max(new_lens[0], 1.0)
                new_lens[1] = max(new_lens[1], 1.0)
                trial.set_lens(new_lens)
                if not lens:
                    trial.focal_y = problem.focal_y if problem.focal_y is None else trial.focal_y
            trial_errors = residuals(trial)
            trial_cost = _cost(trial_errors, delta)
            if np.isfinite(trial_cost) and trial_cost < cost:
                problem.rotations, problem.translations = trial.rotations, trial.translations
                problem.points = trial.points
                problem.focal, problem.focal_y = trial.focal, trial.focal_y
                problem.centre, problem.distortion = trial.centre, trial.distortion
                improvement = (cost - trial_cost) / max(cost, 1e-12)
                errors, cost = trial_errors, trial_cost
                damping = max(damping / 3, 1e-9)
                break
            damping *= 4
            if damping > 1e8:
                improvement = 0.0
                break
        if damping > 1e8 or improvement < 1e-7:
            break

    norm = np.linalg.norm(errors, axis=1)
    return {"start_cost": start, "cost": cost,
            "median_px": float(np.median(norm)) if len(norm) else 0.0,
            "focal": problem.focal, "lens": problem.lens()}
