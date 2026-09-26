"""Simulated particles: the physics mode of a particle emitter.

The default particles are closed-form (a formula of age), which is cheap and
exact at any frame but has no memory: smoke cannot pool under a ceiling and
snow cannot pile up. With physics on, an emitter is simulated step by step
instead: buoyancy or gravity, air drag toward the wind, turbulence,
collisions, spreading along surfaces, snow that sticks where it lands.

Scrubbing stays fast through a cache of checkpoints: the full state every
CHECKPOINT steps, so any frame is at most a few steps from a stored one.
Export reads the same simulation, so the video matches the preview.

Keyframed values on a physics emitter are applied as they are at the frame
being shown (the simulation assumes them constant over its span).

Qt-free, numpy only.
"""

from __future__ import annotations

import math

import numpy as np

STEP = 1.0 / 30.0          # simulation step, scene seconds
CHECKPOINT = 10            # full state stored every this many steps
MAX_CHECKPOINTS = 3000     # ~16 minutes of scene time at 30 steps a second

#: Which kinds stick to what they land on (snow settles; the rest slide).
STICKY = {"snow"}


def _hash(x: np.ndarray) -> np.ndarray:
    return np.modf(np.sin(x) * 43758.5453)[0] % 1.0


class _State:
    __slots__ = ("p", "v", "born", "stuck", "stuck_at", "hit", "prev", "settled", "settled_at")

    def copy(self) -> "_State":
        s = _State()
        for name in self.__slots__:
            setattr(s, name, getattr(self, name).copy())
        return s


class ParticleSim:
    """One emitter's simulation, from `-preroll` seconds onward."""

    def __init__(self, emitter, lighting, planes, direction) -> None:
        self.em = emitter
        self.planes = [(np.asarray(n, np.float64), float(c)) for n, c in planes]
        self.dir = np.asarray(direction, np.float64)
        self.dir /= max(np.linalg.norm(self.dir), 1e-9)
        helper = np.array([0.0, 0.0, 1.0]) if abs(self.dir[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
        self.side = np.cross(self.dir, helper)
        self.side /= np.linalg.norm(self.side)
        self.side2 = np.cross(self.dir, self.side)
        self.lighting = lighting
        self.n = int(min(max(emitter.count, 1), 20000))
        ids = np.arange(self.n, dtype=np.float64)
        self.phase = _hash(ids * 19.19 + float(emitter.seed))
        self.rand = np.stack([_hash(ids * 7.1), _hash(ids * 13.7), _hash(ids * 31.3)], 1) - 0.5
        self.cone = (_hash(ids * 3.3) - 0.5)[:, None] * self.side + (_hash(ids * 5.9) - 0.5)[:, None] * self.side2
        self.ids = ids
        self.life = max(float(emitter.lifetime), 0.05)
        self.rate = max(float(emitter.rate), 1e-3)
        # One lifetime early, so the emitter is already flowing at frame 0
        # (as the formula particles are); pre-roll goes further back, for
        # snow already settled or a room already filled with smoke.
        self.t0 = -(max(0.0, float(getattr(emitter, "preroll", 0.0))) + self.life / self.rate)
        self.sticky = emitter.kind in STICKY or float(getattr(emitter, "settle", 0.0)) > 0.0
        self.settle = float(getattr(emitter, "settle", 0.0))
        # A pile of up to this many landed flakes (oldest melt first).
        self.max_settled = int(min(self.n * 6, 60000))
        self.checkpoints: dict[int, _State] = {}
        start = self._fresh()
        self.checkpoints[0] = start

    # --- birth ------------------------------------------------------------
    def _spawn(self, idx: np.ndarray, s: _State, t: float) -> None:
        em = self.em
        s.p[idx] = np.asarray(em.position, np.float64) + self.rand[idx] * np.asarray(em.size, np.float64)
        launch = self.dir * float(em.speed)
        s.v[idx] = launch + self.cone[idx] * float(em.spread) * (abs(float(em.speed)) * 0.6 + 0.05)
        s.born[idx] = t
        s.stuck[idx] = False
        s.hit[idx] = False
        s.prev[idx] = s.p[idx]

    def _fresh(self) -> _State:
        s = _State()
        s.p = np.zeros((self.n, 3))
        s.v = np.zeros((self.n, 3))
        s.born = np.zeros(self.n)
        s.stuck = np.zeros(self.n, bool)
        s.stuck_at = np.zeros(self.n)
        s.hit = np.zeros(self.n, bool)
        s.prev = np.zeros((self.n, 3))
        # Landed flakes live here, not in the falling pool, so a pile can grow
        # while snow keeps falling.
        s.settled = np.zeros((0, 3))
        s.settled_at = np.zeros(0)
        # Staggered births so the emitter starts mid-flow, like the formula
        # particles do: particle i was born phase_i of a lifetime ago.
        self._spawn(np.arange(self.n), s, self.t0)
        s.born = self.t0 - self.phase * self.life / self.rate
        return s

    # --- one step -------------------------------------------------------------
    def _wind(self, t: float) -> np.ndarray:
        from .effects3d import wind_vector
        now, _avg, _turb = wind_vector(self.lighting, t)
        return now * float(getattr(self.em, "wind_response", 1.0))

    def _step(self, s: _State, t: float) -> None:
        em = self.em
        dt = STEP * self.rate                      # the emitter's own clock
        s.prev = s.p.copy()
        free = ~s.stuck
        # Respawn particles whose life is over (stuck ones stay for `settle`).
        age = (t - s.born) * self.rate / self.life
        expired = (age >= 1.0) & free
        if len(s.settled_at) and self.settle > 0:
            keep = (t - s.settled_at) < self.settle
            s.settled, s.settled_at = s.settled[keep], s.settled_at[keep]
        idx = np.flatnonzero(expired)
        if len(idx):
            self._spawn(idx, s, t)
            free = ~s.stuck
        f = np.flatnonzero(free)
        if not len(f):
            return
        p, v = s.p[f], s.v[f]
        # Buoyancy (lift > 0 along the direction) or gravity (< 0).
        v = v + self.dir * float(em.gravity) * dt
        # Air drag relaxes the velocity toward the wind's plus the particle's
        # own steady drift along its direction (a flake's falling speed, a
        # plume's rise): terminal velocity. Still air leaves that drift,
        # moving air carries it along.
        k = max(float(em.drag), 0.05)
        wind = self._wind(t)
        target = wind + self.dir * float(em.speed)
        v = target + (v - target) * math.exp(-k * dt)
        # Turbulence: a swirling field through space and time.
        tq = float(em.turbulence) + float(getattr(self.lighting, "wind_turbulence", 0.0)) * float(np.linalg.norm(wind)) * 0.5
        if tq > 0:
            q = p * 1.7
            ph = self.ids[f]
            curl = np.stack([np.sin(q[:, 1] * 2.1 + t * 1.3 + ph) + 0.5 * np.sin(q[:, 2] * 3.7 + t * 0.7),
                             np.sin(q[:, 2] * 1.9 + t * 1.1) + 0.5 * np.sin(q[:, 0] * 2.9 + ph * 0.3),
                             np.sin(q[:, 0] * 2.3 + t * 0.9) + 0.5 * np.sin(q[:, 1] * 3.1 + ph)], 1)
            v = v + curl * tq * 0.35 * dt
        p = p + v * dt
        # Collisions: put back on the surface; bounce the normal part, keep
        # the rest (slides along), and note who is touching a surface.
        hit = np.zeros(len(f), bool)
        bounce = float(np.clip(getattr(em, "bounce", 0.0), 0.0, 1.0))
        stick = np.zeros(len(f), bool)
        for n, c in self.planes if getattr(em, "collide", True) else []:
            d = p @ n + c
            inside = d < 0
            if not inside.any():
                continue
            p[inside] -= np.outer(d[inside], n)
            vn = v[inside] @ n
            into = vn < 0
            vi = v[inside]
            vi[into] -= np.outer(vn[into] * (1.0 + bounce), n)
            v[inside] = vi
            hit |= inside
            if self.sticky and n[1] > 0.5:          # lands on something facing up
                stick |= inside
        # Spreading along surfaces: particles pressed against a floor or
        # ceiling push away from crowded spots, so smoke fans out and pools
        # under a ceiling instead of piling into one point.
        touching = hit & ~stick
        if touching.sum() > 8:
            cell = max(float(em.particle_size) * 3.0, 0.08)
            key = np.floor(p[touching][:, [0, 2]] / cell).astype(np.int64)
            lo = key.min(0)
            grid_ix = key - lo
            shape = grid_ix.max(0) + 3
            if shape[0] * shape[1] < 4_000_000:
                counts = np.zeros(shape)
                np.add.at(counts, (grid_ix[:, 0] + 1, grid_ix[:, 1] + 1), 1.0)
                gx = (counts[2:, 1:-1] - counts[:-2, 1:-1]) * 0.5
                gz = (counts[1:-1, 2:] - counts[1:-1, :-2]) * 0.5
                push = np.stack([-gx[grid_ix[:, 0], grid_ix[:, 1]], np.zeros(len(key)),
                                 -gz[grid_ix[:, 0], grid_ix[:, 1]]], 1)
                v[touching] = v[touching] + push * 1.5 * dt
        s.p[f], s.v[f], s.hit[f] = p, v, hit
        if stick.any():
            # Rest a flake on top of the surface, not in it (the depth test
            # would hide it inside the floor), move it to the settled store and
            # send a new one falling in its place.
            st = f[stick]
            lift = float(em.particle_size) * 0.6
            s.settled = np.concatenate([s.settled, s.p[st] + np.array([0.0, lift, 0.0])])[-self.max_settled:]
            s.settled_at = np.concatenate([s.settled_at, np.full(len(st), t)])[-self.max_settled:]
            self._spawn(st, s, t)

    # --- frames ---------------------------------------------------------------
    def state_at(self, t: float) -> _State:
        steps = max(0, int(round((t - self.t0) / STEP)))
        base = min(steps // CHECKPOINT, MAX_CHECKPOINTS)
        while base not in self.checkpoints:
            base -= 1
        s = self.checkpoints[base].copy()
        k = base * CHECKPOINT
        while k < steps:
            self._step(s, self.t0 + (k + 1) * STEP)
            k += 1
            if k % CHECKPOINT == 0 and k // CHECKPOINT not in self.checkpoints \
                    and k // CHECKPOINT <= MAX_CHECKPOINTS:
                self.checkpoints[k // CHECKPOINT] = s.copy()
        return s

    def buffer_at(self, t: float) -> np.ndarray:
        """Per particle two vec4s for the GPU: (position, age 0..1) and
        (previous position, 1 if on a surface)."""
        s = self.state_at(t)
        age = np.clip((t - s.born) * self.rate / self.life, 0.0, 0.999)
        out = np.zeros((self.n + len(s.settled), 8), np.float32)
        out[:self.n, 0:3] = s.p
        out[:self.n, 3] = age
        out[:self.n, 4:7] = s.prev
        out[:self.n, 7] = s.hit.astype(np.float32)
        # Settled flakes: full size (mid-life), not moving, on a surface.
        out[self.n:, 0:3] = s.settled
        out[self.n:, 3] = 0.5
        out[self.n:, 4:7] = s.settled
        out[self.n:, 7] = 1.0
        return out


_SIMS: dict[str, tuple[tuple, ParticleSim]] = {}


def _signature(em, lighting, planes, direction) -> tuple:
    fields = ("kind", "count", "rate", "lifetime", "particle_size", "speed", "spread", "gravity",
              "turbulence", "drag", "bounce", "collide", "seed", "preroll", "settle", "wind_response")
    vals = tuple(round(float(getattr(em, f)), 5) if not isinstance(getattr(em, f), str) else getattr(em, f)
                 for f in fields if hasattr(em, f))
    geo = tuple(np.round(np.concatenate([np.asarray(em.position, float), np.asarray(em.size, float),
                                         np.asarray(direction, float)]), 5))
    wind = tuple(round(float(getattr(lighting, f, 0.0)), 5)
                 for f in ("wind_direction", "wind_strength", "wind_gusts", "wind_turbulence"))
    pl = tuple(np.round(np.concatenate([np.r_[n, c] for n, c in planes]), 5)) if planes else ()
    return vals + geo + wind + pl


def simulation(em, lighting, planes, direction) -> ParticleSim:
    """The cached simulation for this emitter, rebuilt when anything that
    shapes it changes."""
    sig = _signature(em, lighting, planes, direction)
    held = _SIMS.get(em.id)
    if held is None or held[0] != sig:
        held = (sig, ParticleSim(em, lighting, planes, direction))
        _SIMS[em.id] = held
    return held[1]


def clear() -> None:
    _SIMS.clear()
