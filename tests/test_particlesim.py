"""Physics-mode particles: simulated, cached, deterministic."""

from __future__ import annotations

import numpy as np

from dlss5_converter import effects3d as fx
from dlss5_converter import particlesim

FLOOR = (np.array([0.0, 1.0, 0.0]), 0.8)          # y = -0.8, free side up
CEILING = (np.array([0.0, -1.0, 0.0]), 1.0)       # y = 1.0, free side down


def _sim(em, planes, direction=(0.0, 1.0, 0.0), lighting=None):
    particlesim.clear()
    return particlesim.simulation(em, lighting or fx.LightingSettings(), planes, direction)


def test_scrubbing_back_gives_the_same_frame():
    em = fx.emitter_preset("smoke")
    em.physics = True
    sim = _sim(em, [FLOOR, CEILING])
    later = sim.buffer_at(6.0)
    sim.buffer_at(2.0)                             # scrub back, then forward again
    assert np.allclose(sim.buffer_at(6.0), later)
    fresh = _sim(em, [FLOOR, CEILING]).buffer_at(6.0)
    assert np.allclose(fresh, later)


def test_smoke_pools_and_spreads_under_a_ceiling():
    em = fx.emitter_preset("smoke")
    em.physics, em.gravity, em.lifetime = True, 0.6, 12.0
    open_air = _sim(em, [FLOOR]).buffer_at(10.0)
    roofed = _sim(em, [FLOOR, CEILING]).buffer_at(10.0)
    assert roofed[:, 1].max() <= 1.0 + 1e-6              # nothing passes the ceiling
    at_top = roofed[roofed[:, 1] > 0.9]
    spread = np.linalg.norm(at_top[:, [0, 2]] - np.array([0.0, -4.0]), axis=1)
    assert len(at_top) > 50 and np.median(spread) > 0.6
    assert open_air[:, 1].max() > 1.5                     # without it, it keeps rising


def test_snow_piles_up_while_it_keeps_falling():
    em = fx.emitter_preset("snow")
    em.physics, em.settle = True, 60.0
    sim = _sim(em, [FLOOR], direction=(0.0, -1.0, 0.0))
    early, late = sim.buffer_at(3.0), sim.buffer_at(20.0)
    assert len(late) > len(early) > em.count             # landed flakes are extra
    settled = late[em.count:]
    assert np.all(settled[:, 1] > -0.8) and np.all(settled[:, 1] < -0.7)
    falling = late[:em.count]
    assert (falling[:, 1] > -0.7).mean() > 0.5            # the fall goes on


def test_preroll_starts_with_snow_already_down():
    em = fx.emitter_preset("snow")
    em.physics, em.settle = True, 60.0
    plain = len(_sim(em, [FLOOR], (0.0, -1.0, 0.0)).buffer_at(0.0))
    em.preroll = 30.0
    rolled = len(_sim(em, [FLOOR], (0.0, -1.0, 0.0)).buffer_at(0.0))
    assert rolled > plain


def test_wind_carries_simulated_particles():
    em = fx.emitter_preset("smoke")
    em.physics = True
    calm = _sim(em, [FLOOR]).buffer_at(5.0)
    windy = _sim(em, [FLOOR], lighting=fx.LightingSettings(wind_strength=2.0, wind_direction=90.0,
                                                            wind_gusts=0.0)).buffer_at(5.0)
    assert np.median(windy[:, 0]) > np.median(calm[:, 0]) + 0.5
