"""Build a ShotScene from game screenshots with depth.

1. Read every shot once: features for matching, and half-resolution grey and
   depth for the dense stage. The 4K image is not kept.
2. Rough cameras: focal length and poses from feature matches lifted by the
   game depth (`measured`).
3. Dense refinement at half resolution: each camera re-solved against its
   neighbours from per-pixel matches (neighbour warped through its depth,
   leftover misalignment measured by optical flow), a few sweeps. Along the
   way each shot's own depth conversion 1/z = a*d + b is fitted (games change
   their near plane between screenshots), and a shot that still disagrees
   tries its own focal length (photo modes zoom).
4. Grading: a shot whose leftover misalignment stays large is not used.
5. Colour gains (auto-exposure), moved-object masks, planes for effects, and
   the levelled renderer world.

Qt-free; progress(str) and cancelled() hooks as in `multishot.build`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import game_depth, measured, multishot
from . import multiview as mv
from .shotscene import ShotScene

SUB = 2                 # dense work at half the screenshot resolution
NEIGHBOURS = 4
SWEEPS = 3
GOOD_PX = 1.2           # fully trusted at or below this (full-resolution pixels)
REJECT_PX = 6.0         # never used above this
MOVED_MARGIN = 1.08     # a neighbour seeing 8% further along a pixel's line: it moved


class Cancelled(Exception):
    pass


class _Shots:
    """Per-shot arrays for the dense stage."""

    def __init__(self):
        self.names: list[str] = []
        self.grey: list[np.ndarray] = []
        self.disp: list[np.ndarray] = []       # raw buffer value, 0 where empty
        self.described: list = []
        self.work: list[mv.Shot] = []


def _read(paths, report, progress, cancelled) -> _Shots:
    s = _Shots()
    for n, path in enumerate(paths, 1):
        if cancelled():
            raise Cancelled
        progress(f"Reading shots… {n}/{len(paths)}")
        raw = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        if raw is None:
            report.statuses[str(path)] = multishot.UNREADABLE
            continue
        full = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        fh, fw = full.shape[:2]
        wh = round(fh * multishot.WORK_WIDTH / fw)
        disp_work = game_depth.load(path, (multishot.WORK_WIDTH, wh))
        if disp_work is None or float(disp_work.max()) <= 0.0:
            report.statuses[str(path)] = multishot.NO_DEPTH
            continue
        shot = mv.Shot(str(path), cv2.resize(full, (multishot.WORK_WIDTH, wh), interpolation=cv2.INTER_AREA))
        shot.disparity = disp_work
        s.work.append(shot)
        s.names.append(str(path))
        s.described.append(measured.describe(full, multishot.WORK_WIDTH))
        hw, hh = fw // SUB, fh // SUB
        s.grey.append(cv2.resize(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), (hw, hh), interpolation=cv2.INTER_AREA))
        d = game_depth.load(path, (hw, hh))
        s.disp.append(np.where(d > 1e-6, d, 0.0).astype(np.float32))
        s.full_size = (fw, fh)
    return s


class _Dense:
    """Cameras, depth conversions and dense matching at half resolution."""

    def __init__(self, shots: _Shots, Rs, Cs, K_half):
        self.s = shots
        self.n = len(shots.names)
        self.Rs = [np.asarray(r, np.float64).copy() for r in Rs]
        self.Cs = [np.asarray(c, np.float64).copy() for c in Cs]
        self.Ks = [K_half.copy() for _ in range(self.n)]
        self.a = np.full(self.n, 1.0 / measured.UNIT)
        self.b = np.zeros(self.n)
        self.H, self.W = shots.grey[0].shape
        self.yy, self.xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        self.textured = [np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)) > 16 for g in shots.grey]

    def depth(self, i):
        d = self.s.disp[i]
        return np.where(d > 0, 1.0 / np.maximum(self.a[i] * d + self.b[i], 1e-9), measured.SKY)

    def lift(self, i):
        z = self.depth(i)
        K = self.Ks[i]
        cam = np.stack([(self.xx - K[0, 2]) / K[0, 0] * z, (self.yy - K[1, 2]) / K[1, 1] * z, z], -1)
        return self.Cs[i] + cam.reshape(-1, 3) @ self.Rs[i], (z < measured.SKY / 2).ravel()

    def project(self, j, world):
        K = self.Ks[j]
        c = (world - self.Cs[j]) @ self.Rs[j].T
        z = np.where(c[:, 2] > 1e-6, c[:, 2], 1e-6)
        shape = (self.H, self.W)
        return ((K[0, 0] * c[:, 0] / z + K[0, 2]).reshape(shape).astype(np.float32),
                (K[1, 1] * c[:, 1] / z + K[1, 2]).reshape(shape).astype(np.float32), c[:, 2].reshape(shape))

    def bilinear(self, img, x, y):
        x = np.clip(x, 0, self.W - 1.001)
        y = np.clip(y, 0, self.H - 1.001)
        x0, y0 = x.astype(int), y.astype(int)
        ax, ay = x - x0, y - y0
        return (img[y0, x0] * (1 - ax) * (1 - ay) + img[y0, x0 + 1] * ax * (1 - ay)
                + img[y0 + 1, x0] * (1 - ax) * ay + img[y0 + 1, x0 + 1] * ax * ay)

    def pair(self, i, j):
        world, ok = self.lift(i)
        u, v, zj = self.project(j, world)
        dj = cv2.remap(self.depth(j).astype(np.float32), u, v, cv2.INTER_NEAREST)
        vis = ok.reshape(self.H, self.W) & (u >= 0) & (u < self.W - 1) & (v >= 0) & (v < self.H - 1) & \
            (np.abs(dj - zj) < 0.03 * zj)
        warped = cv2.remap(self.s.grey[j], u, v, cv2.INTER_LINEAR)
        flow = cv2.calcOpticalFlowFarneback(self.s.grey[i], warped, None, 0.5, 4, 21, 5, 7, 1.5, 0)
        return world, u, v, vis, flow

    def error(self, i, j) -> float:
        _, _, _, vis, flow = self.pair(i, j)
        m = vis & self.textured[i]
        if m.sum() < 2000:
            return np.inf
        return float(np.median(np.hypot(flow[..., 0], flow[..., 1])[m])) * SUB

    def neighbours(self, i, count=NEIGHBOURS):
        fwd = np.array([R[2] for R in self.Rs])
        cen = np.array(self.Cs)
        spread = np.median(np.linalg.norm(cen - cen.mean(0), axis=1)) + 1e-9
        score = np.linalg.norm(cen - cen[i], axis=1) / spread + np.arccos(np.clip(fwd @ fwd[i], -1, 1))
        score[i] = np.inf
        return [int(j) for j in np.argsort(score)[:count]]

    def shot_error(self, i) -> float:
        return float(np.median([self.error(i, j) for j in self.neighbours(i, 2)]))

    def matches(self, j):
        """3D points from j's neighbours matched to their true pixels in j."""
        obj, pts = [], []
        for i in self.neighbours(j):
            world, u, v, vis, flow = self.pair(i, j)
            m = vis & self.textured[i] & (np.hypot(flow[..., 0], flow[..., 1]) < 10)
            sel = np.zeros_like(m)
            sel[::3, ::3] = True
            ys, xs = np.nonzero(m & sel)
            fx, fy = xs + flow[ys, xs, 0], ys + flow[ys, xs, 1]
            obj.append(world.reshape(self.H, self.W, 3)[ys, xs])
            pts.append(np.stack([self.bilinear(u, fx, fy), self.bilinear(v, fx, fy)], -1))
        if not obj:
            return np.zeros((0, 3)), np.zeros((0, 2))
        obj, pts = np.concatenate(obj), np.concatenate(pts)
        if len(obj) > 50000:
            pick = np.random.default_rng(j).choice(len(obj), 50000, replace=False)
            obj, pts = obj[pick], pts[pick]
        return obj.astype(np.float64), pts.astype(np.float64)

    def solve_pose(self, j, obj, pts) -> bool:
        if len(obj) < 500:
            return False
        rvec = cv2.Rodrigues(self.Rs[j])[0]
        tvec = (-self.Rs[j] @ self.Cs[j]).reshape(3, 1)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, pts, self.Ks[j], None, rvec, tvec, True, iterationsCount=150,
                                                 reprojectionError=0.75, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inl is None or len(inl) < 400:
            return False
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], pts[inl], self.Ks[j], None, rvec, tvec)
        self.Rs[j] = cv2.Rodrigues(rvec)[0]
        self.Cs[j] = (-self.Rs[j].T @ tvec).ravel()
        self._fit_depth(j, obj[inl], pts[inl])
        return True

    def _fit_depth(self, j, obj, pts) -> None:
        """This shot's own depth conversion, from the depth its (independently
        solved) camera predicts at the matched pixels."""
        zpred = ((obj - self.Cs[j]) @ self.Rs[j].T)[:, 2]
        d = self.bilinear(self.s.disp[j], pts[:, 0], pts[:, 1])
        ok = (d > 1e-6) & (zpred > 0)
        if ok.sum() < 200:
            return
        d, inv = d[ok], 1.0 / zpred[ok]
        rng = np.random.default_rng(j)
        best, best_n = (self.a[j], self.b[j]), -1
        for _ in range(200):
            p, q = rng.choice(len(d), 2, replace=False)
            if abs(d[p] - d[q]) < 1e-9:
                continue
            a_ = (inv[p] - inv[q]) / (d[p] - d[q])
            b_ = inv[p] - a_ * d[p]
            cnt = int((np.abs(a_ * d + b_ - inv) < 0.02 * inv).sum())
            if cnt > best_n:
                best, best_n = (a_, b_), cnt
        a_, b_ = best
        good = np.abs(a_ * d + b_ - inv) < 0.02 * inv
        if good.sum() >= 100:
            a_, b_ = np.polyfit(d[good], inv[good], 1)
        if a_ > 0:
            self.a[j], self.b[j] = a_, b_

    def try_own_focal(self, j, err) -> float:
        obj, pts = self.matches(j)
        if len(obj) < 500:
            return err
        rvec = cv2.Rodrigues(self.Rs[j])[0]
        tvec = (-self.Rs[j] @ self.Cs[j]).reshape(3, 1)
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
                 | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
        try:
            _, Kj, _, rv, tv = cv2.calibrateCamera([obj.astype(np.float32)], [pts.astype(np.float32)],
                                                   (self.W, self.H), self.Ks[j].copy(), np.zeros(5), [rvec], [tvec],
                                                   flags=flags)
        except cv2.error:
            return err
        old = (self.Ks[j].copy(), self.Rs[j].copy(), self.Cs[j].copy())
        self.Ks[j], self.Rs[j] = Kj, cv2.Rodrigues(rv[0])[0]
        self.Cs[j] = (-self.Rs[j].T @ tv[0]).ravel()
        new = self.shot_error(j)
        if new < err * 0.8:
            return new
        self.Ks[j], self.Rs[j], self.Cs[j] = old
        return err


def _gains(dense: _Dense, use: np.ndarray) -> np.ndarray:
    """Per-shot colour gain from overlapping pixels, solved for all shots at
    once and robust to lights and mismatches (median of per-pixel ratios,
    Huber re-weighting)."""
    n = dense.n
    idx = {k: t for t, k in enumerate(use)}
    cache: dict[int, np.ndarray] = {}

    def lin(k):
        # Only the shots in use right now: all of them at once is gigabytes.
        if k not in cache:
            if len(cache) >= 8:
                cache.pop(next(iter(cache)))
            img = cv2.imdecode(np.fromfile(dense.s.names[k], np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(cv2.resize(img, (dense.W, dense.H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            c = img.astype(np.float32) / 255.0
            cache[k] = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return cache[k]

    rows, rhs, wts = [], [], []
    for i in use:
        world, ok = dense.lift(i)
        for j in dense.neighbours(i):
            if j not in idx:
                continue
            u, v, zj = dense.project(j, world)
            dj = cv2.remap(dense.depth(j).astype(np.float32), u, v, cv2.INTER_NEAREST)
            vis = ok.reshape(dense.H, dense.W) & (u >= 1) & (u < dense.W - 2) & (v >= 1) & (v < dense.H - 2) & \
                (np.abs(dj - zj) < 0.02 * zj)
            vis[::2] = False                     # half the rows is plenty
            if vis.sum() < 3000:
                continue
            a = lin(i)[vis]
            b = cv2.remap(lin(j), u, v, cv2.INTER_LINEAR)[vis]
            mid = (a.mean(1) > 0.01) & (a.mean(1) < 0.8) & (b.mean(1) > 0.01) & (b.mean(1) < 0.8)
            if mid.sum() < 2000:
                continue
            lr = np.log(np.maximum(a[mid], 1e-4)) - np.log(np.maximum(b[mid], 1e-4))
            med = np.median(lr, axis=0)
            mad = np.median(np.abs(lr - med), axis=0)
            if mad.mean() > 0.35:
                continue
            for ch in range(3):
                r = np.zeros(3 * len(use))
                r[3 * idx[j] + ch], r[3 * idx[i] + ch] = 1.0, -1.0
                rows.append(r); rhs.append(med[ch]); wts.append(np.sqrt(mid.sum()) / (mad[ch] + 0.05))
    gain = np.ones((n, 3))
    if not rows:
        return gain
    M, y, w0 = np.array(rows), np.array(rhs), np.array(wts)
    pin = np.zeros((3, 3 * len(use)))
    for ch in range(3):
        pin[ch, ch::3] = 1e3
    sol = np.linalg.lstsq(np.vstack([M * w0[:, None], pin]), np.r_[y * w0, np.zeros(3)], rcond=None)[0]
    for _ in range(8):
        w = w0 * np.minimum(1.0, 0.15 / np.maximum(np.abs(M @ sol - y), 1e-9))
        sol = np.linalg.lstsq(np.vstack([M * w[:, None], pin]), np.r_[y * w, np.zeros(3)], rcond=None)[0]
    gain[use] = np.exp(sol).reshape(len(use), 3)
    return gain


def _moved(dense: _Dense, i: int, use) -> np.ndarray:
    world, ok = dense.lift(i)
    votes = np.zeros(dense.H * dense.W, np.int32)
    for j in dense.neighbours(i):
        if j not in use:
            continue
        u, v, zj = dense.project(j, world)
        inside = ok & (zj.ravel() > 0) & (u.ravel() >= 0) & (u.ravel() < dense.W - 1) & \
            (v.ravel() >= 0) & (v.ravel() < dense.H - 1)
        dj = cv2.remap(dense.depth(j).astype(np.float32), u, v, cv2.INTER_NEAREST).ravel()
        votes += (inside & (dj > zj.ravel() * MOVED_MARGIN)).astype(np.int32)
    m = (votes >= 1).reshape(dense.H, dense.W).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.dilate(m, np.ones((9, 9), np.uint8)) > 0


def _planes(dense: _Dense, use, to_renderer, count=4) -> list:
    """Large flat surfaces (floor, walls, ceiling) from all shots' depth, for
    effects to collide with. Plain RANSAC with a least-squares refit."""
    pts = []
    for k in use:
        # A sparse grid is all a plane fit needs; lifting every pixel of
        # every shot made this step take many minutes on a large capture.
        z = dense.depth(k)[8::16, 8::16]
        K = dense.Ks[k]
        ys, xs = np.mgrid[8:dense.H:16, 8:dense.W:16].astype(np.float64)
        ok = z < measured.SKY / 2
        cam = np.stack([(xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z], -1)[ok]
        pts.append(dense.Cs[k] + cam @ dense.Rs[k])
    if not pts:
        return []
    left = to_renderer(np.concatenate(pts))
    total = len(left)
    rng = np.random.default_rng(0)
    planes = []
    for _ in range(count):
        if len(left) < 300:
            break
        best, best_n = None, 0
        for _ in range(800):
            a, b, c = left[rng.choice(len(left), 3, replace=False)]
            nrm = np.cross(b - a, c - a)
            if np.linalg.norm(nrm) < 1e-9:
                continue
            nrm /= np.linalg.norm(nrm)
            cnt = int((np.abs((left - a) @ nrm) < 0.025).sum())
            if cnt > best_n:
                best, best_n = (nrm, a), cnt
        if best is None or best_n < 0.03 * total:
            break
        nrm, a = best
        inl = np.abs((left - a) @ nrm) < 0.025
        cen = left[inl].mean(0)
        nrm = np.linalg.svd(left[inl] - cen)[2][2]
        planes.append((nrm, float(-nrm @ cen)))      # n.p + c = 0
        left = left[~inl]
    return planes


def build(paths, excluded=None, progress: Callable[[str], None] = lambda _m: None,
          cancelled: Callable[[], bool] = lambda: False):
    """Returns (ShotScene | None, Report, moved masks {shot: bool array})."""
    excluded = excluded or set()
    report = multishot.Report()
    wanted = []
    for p in paths:
        if str(p) in excluded:
            report.statuses[str(p)] = multishot.EXCLUDED
        else:
            wanted.append(Path(p))
    shots = _read(wanted, report, progress, cancelled)
    if len(shots.names) < 2:
        report.advice = ("These shots have no game depth. Capture them with the DLSS5 Scene Capture add-on "
                         "(F10) so each screenshot has its depth beside it.")
        return None, report, {}

    # Rough cameras from features + game depth.
    cache = {}

    def pair_matches(i, j):
        if (i, j) not in cache:
            cache[(i, j)] = measured.match_points(shots.described[i], shots.described[j])
        return cache[(i, j)]

    work = shots.work
    focal = multishot._focal(work, pair_matches, progress, cancelled)
    width, height = work[0].size
    camera = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
    candidates = measured.candidate_pairs(work)
    counter = {"n": 0}

    def on_pair():
        counter["n"] += 1
        if cancelled():
            raise Cancelled
        if counter["n"] % 4 == 0:
            progress(f"Placing shots… {counter['n']}/{len(candidates)}")

    measured.solve(work, camera, described=shots.described, on_pair=on_pair, pair_matches=pair_matches,
                   pairs=candidates)
    placed = [k for k, s in enumerate(work) if s.depth is not None]
    for k, s in enumerate(work):
        report.statuses[s.name] = multishot.PLACED if k in placed else multishot.NO_OVERLAP
    if len(placed) < 2:
        report.advice = ("Not enough shots could be placed. Take more shots, each moving a little from the "
                         "last, so neighbours share most of their view.")
        return None, report, {}
    # Unplaced shots are dropped from here on.
    keep = placed
    shots.names = [shots.names[k] for k in keep]
    shots.grey = [shots.grey[k] for k in keep]
    shots.disp = [shots.disp[k] for k in keep]
    shots.described = [shots.described[k] for k in keep]
    work = [work[k] for k in keep]
    fw, fh = shots.full_size
    K_half = camera.copy()
    K_half[:2] *= (fw / SUB) / width
    dense = _Dense(shots, [s.rotation for s in work], [s.centre for s in work], K_half)

    # Dense refinement, outward from the best-agreeing shot.
    errs = []
    for k in range(dense.n):
        if cancelled():
            raise Cancelled
        progress(f"Checking alignment… {k + 1}/{dense.n}")
        errs.append(dense.shot_error(k))
    anchor = int(np.argmin(errs))
    order = sorted(range(dense.n), key=lambda k: np.linalg.norm(dense.Cs[k] - dense.Cs[anchor]))
    for sweep in range(SWEEPS):
        for step, j in enumerate(order, 1):
            if cancelled():
                raise Cancelled
            progress(f"Aligning cameras, pass {sweep + 1} of {SWEEPS}… {step}/{dense.n}")
            if j == anchor:
                continue
            obj, pts = dense.matches(j)
            dense.solve_pose(j, obj, pts)
    errs = []
    for k in range(dense.n):
        if cancelled():
            raise Cancelled
        progress(f"Grading shots… {k + 1}/{dense.n}")
        e = dense.shot_error(k)
        if e > GOOD_PX and k != anchor:
            e = dense.try_own_focal(k, e)
        errs.append(e)
    errs = np.array(errs)
    used = errs <= REJECT_PX
    trust = np.clip((REJECT_PX - errs) / (REJECT_PX - GOOD_PX), 0.15, 1.0)
    rejected = [Path(shots.names[k]).stem.split("_")[-1] for k in np.flatnonzero(~used)]
    for k in np.flatnonzero(~used):
        report.statuses[shots.names[k]] = multishot.NO_OVERLAP
    use = np.flatnonzero(used)
    if len(use) < 2:
        report.advice = "The shots could not be lined up. Take smaller steps between shots."
        return None, report, {}

    progress("Matching brightness between shots…")
    gain = _gains(dense, use)
    moved = {}
    for n_, k in enumerate(use, 1):
        if cancelled():
            raise Cancelled
        progress(f"Finding things that moved… {n_}/{len(use)}")
        moved[int(k)] = _moved(dense, int(k), set(use.tolist()))

    # The levelled renderer world, as the 3D tab expects.
    level_shots = []
    for k in use:
        s = work[k]
        s.rotation, s.centre = dense.Rs[k], dense.Cs[k]
        level_shots.append(s)
    axes, target, distance = multishot._level(level_shots)
    scale = multishot.TARGET_DEPTH / max(distance, 1e-6)
    offset = np.array([0.0, 0.0, -multishot.TARGET_DEPTH])
    to_r = lambda p: (p - target) @ axes.T * scale + offset  # noqa: E731
    progress("Finding floors and walls…")
    planes = _planes(dense, use, to_r)
    # Effects treat n.p + c >= 0 as free space: point each normal toward the
    # cameras, which are always on the open side.
    eyes = to_r(np.array([dense.Cs[k] for k in use])).mean(0)
    planes = [(n, c) if n @ eyes + c >= 0 else (-n, -c) for n, c in planes]

    K_full = np.array([K * np.array([[SUB], [SUB], [1.0]]) for K in dense.Ks])
    scene = ShotScene(names=shots.names, rotations=np.array(dense.Rs), centres=np.array(dense.Cs),
                      cameras=K_full, size=(fw, fh), depth_a=dense.a, depth_b=dense.b, gain=gain, trust=trust,
                      used=used, axes=axes, target=target, scale=scale, offset=offset, planes=planes)
    report.fov_degrees = float(np.degrees(2 * np.arctan(fw / 2 / K_full[anchor][0, 0])))
    report.agreement = float(np.median(errs[used]))
    report.root_image = work[anchor].image
    if rejected:
        report.advice = (f"Shots {', '.join(rejected)} could not be lined up and are not used. Smaller steps "
                         "between shots help, and keep moving things out of the middle.")
    return scene, report, moved
