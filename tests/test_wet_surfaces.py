"""Wet surfaces: only the ground-facing part of the scene changes."""
from __future__ import annotations

import numpy as np
import pytest

from dlss5_converter import splat3d

pytestmark = pytest.mark.skipif(not splat3d.is_available(), reason="no GPU adapter")


def _render(wetness: float, puddles: float) -> np.ndarray:
    from dlss5_converter.camera3d import Camera
    from dlss5_converter.creative_page import camera_matrices
    from dlss5_converter.effects3d import EffectsState
    h, w = 120, 200
    yy = np.linspace(0, 1, h)[:, None].repeat(w, 1)
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[yy < 0.6] = (230, 140, 50)                     # a bright wall
    rgb[yy >= 0.6] = (90, 85, 80)                      # the ground
    disp = np.where(yy < 0.6, 0.3, 0.3 + 0.6 * (yy - 0.6) / 0.4).astype(np.float32)
    r = splat3d.SplatRenderer()
    r.set_scene(splat3d.build_splats(rgb, disp, backfill=False))
    view, proj = camera_matrices(Camera(target=(0, -0.25, -4.0)), (w, h))
    fx = EffectsState()
    fx.lighting.wetness, fx.lighting.puddles = wetness, puddles
    return r.render_u8(view, proj, (w, h), (0.0, 0.0, 0.0), fx, 1.0).astype(np.float32)


def test_wet_ground_changes_far_more_than_walls():
    dry, wet = _render(0.0, 0.0), _render(0.9, 0.6)
    wall = (slice(10, 50), slice(40, 160))
    ground = (slice(95, 115), slice(40, 160))
    # Walls only darken a little (soaked, but nothing to reflect on them).
    wall_change = np.abs(wet[wall] - dry[wall]).mean() / dry[wall].mean()
    ground_change = np.abs(wet[ground] - dry[ground]).mean() / dry[ground].mean()
    assert wall_change < 0.18
    assert ground_change > 1.5 * wall_change
    # The wall is orange, so its reflection makes wet ground warmer.
    assert (wet[ground][..., 0] - wet[ground][..., 2]).mean() > \
           (dry[ground][..., 0] - dry[ground][..., 2]).mean()
