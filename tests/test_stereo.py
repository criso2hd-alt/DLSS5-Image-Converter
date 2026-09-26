"""3D video: depth smoothing, eye views, packing formats."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dlss5_converter import stereo
from dlss5_converter.settings import AppSettings, StereoSettings


def _frame(h=60, w=120):
    rgb = np.zeros((h, w, 3), np.float32)
    rgb[:, :, 0] = np.linspace(0, 1, w)[None, :]          # a horizontal ramp
    depth = np.zeros((h, w), np.float32)
    depth[20:40, 50:70] = 1.0                              # a near box on a far wall
    return rgb, depth


@pytest.mark.parametrize("fmt", list(stereo.FORMATS))
def test_every_format_packs_to_its_declared_size(fmt):
    rgb, depth = _frame()
    out = stereo.frame(rgb, depth, StereoSettings(enabled=True, format=fmt), stereo.DepthSmoother(0.0))
    w, h = stereo.output_size(fmt, (120, 60))
    assert out.shape == (h, w, 3)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_near_things_shift_between_the_eyes_and_far_ones_less():
    rgb, depth = _frame()
    rgb[20:40, 50:70] = (0.0, 1.0, 0.0)                     # green box, near
    # Pop-out 1: the far wall sits at the screen and the near box comes out
    # in front of it, so the box shifts between the eyes and the wall does not.
    left, right = stereo.views(rgb, depth, strength=1.0, pop_out=1.0)
    box_l = np.flatnonzero(left[30, :, 1] > 0.5).mean()
    box_r = np.flatnonzero(right[30, :, 1] > 0.5).mean()
    assert box_l > box_r + 1.0                              # left eye sees it further right
    assert np.allclose(left[5, 5:30, 0], rgb[5, 5:30, 0], atol=0.02)
    # Pop-out 0: the nearest thing sits at the screen, everything else behind.
    left, right = stereo.views(rgb, depth, strength=1.0, pop_out=0.0)
    assert np.flatnonzero(left[30, :, 1] > 0.5).mean() == pytest.approx(
        np.flatnonzero(right[30, :, 1] > 0.5).mean(), abs=0.5)


def test_smoothing_steadies_depth_but_resets_at_a_cut():
    rgb, depth = _frame()
    smooth = stereo.DepthSmoother(0.8)
    smooth(depth, rgb)
    jitter = depth.copy()
    jitter[20:40, 50:70] = 0.5
    steadied = smooth(jitter, rgb)
    assert steadied[30, 60] > 0.8                           # mostly the previous frame
    cut = np.ones_like(rgb)                                 # a completely different shot
    assert smooth(jitter, cut)[30, 60] == pytest.approx(stereo.DepthSmoother(0.0)(jitter, cut)[30, 60])


def test_stereo_settings_survive_a_restart(tmp_path):
    s = AppSettings()
    s.stereo = StereoSettings(enabled=True, format="tb_half", strength=0.8, pop_out=0.1, run_dlss=False)
    path = tmp_path / "settings.json"
    s.save(path)
    back = AppSettings.load(path)
    assert back.stereo == s.stereo
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["stereo"]["from_the_future"] = 1                   # unknown keys are ignored
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert AppSettings.load(path).stereo.format == "tb_half"
