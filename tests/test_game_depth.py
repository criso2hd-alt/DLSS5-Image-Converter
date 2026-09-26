"""Reading the ReShade add-on's depth sidecars."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dlss5_converter import game_depth
from dlss5_converter import multiview as mv


def _write(tmp_path, raw, **extra):
    """Sidecars exactly as the add-on writes them."""
    image = tmp_path / "shot.png"
    image.write_bytes(b"")
    height, width = raw.shape
    raw.astype("<f4").tofile(tmp_path / "shot.depth.f32")
    info = {"width": width, "height": height,
            "fraction_at_zero": float(np.mean(raw <= 1e-7)),
            "fraction_at_one": float(np.mean(raw >= 1 - 1e-7))}
    info.update(extra)
    (tmp_path / "shot.depth.json").write_text(json.dumps(info), encoding="utf-8")
    return image


def _reversed(distance, near=0.1):
    """Reversed Z, infinite far plane: what modern games store."""
    return np.where(np.isfinite(distance), near / distance, 0.0).astype(np.float32)


def test_sidecar_names_sit_beside_the_screenshot(tmp_path):
    depth, info = game_depth.sidecars(tmp_path / "Cyberpunk 2077 shot.png")
    assert depth.name == "Cyberpunk 2077 shot.depth.f32"
    assert info.name == "Cyberpunk 2077 shot.depth.json"


def test_missing_sidecar_falls_back(tmp_path):
    image = tmp_path / "plain.png"
    image.write_bytes(b"")
    assert game_depth.load(image) is None


def test_damaged_sidecar_falls_back(tmp_path):
    image = _write(tmp_path, np.zeros((4, 6), np.float32))
    (tmp_path / "shot.depth.f32").write_bytes(b"\x00" * 10)      # truncated
    assert game_depth.load(image) is None


def test_reversed_z_comes_back_unchanged(tmp_path):
    distance = np.full((20, 30), np.inf)
    distance[5:15, 5:25] = 4.0
    raw = _reversed(distance)
    out = game_depth.load(_write(tmp_path, raw))
    assert out == pytest.approx(raw)


def test_standard_z_is_flipped_into_the_same_shape(tmp_path):
    distance = np.full((20, 30), np.inf)
    distance[5:15, 5:25] = 4.0
    reversed_ = _reversed(distance)
    out = game_depth.load(_write(tmp_path, 1.0 - reversed_))
    assert out == pytest.approx(reversed_, abs=1e-6)


def test_indoor_shot_without_sky_is_still_classified():
    """No pixel at either end: the median decides."""
    reversed_ = np.full((10, 10), 0.02, np.float32)
    info = {"fraction_at_zero": 0.0, "fraction_at_one": 0.0}
    assert game_depth.is_reversed(info, reversed_)
    assert not game_depth.is_reversed(info, 1.0 - reversed_)


def test_resize_uses_nearest_so_edges_stay_sharp(tmp_path):
    raw = np.zeros((4, 4), np.float32)
    raw[:, 2:] = 0.5
    out = game_depth.load(_write(tmp_path, raw), size=(8, 4))
    assert set(np.unique(out)) <= {0.0, 0.5}


def test_game_depth_fits_the_multiview_model_exactly():
    """near / distance is a / distance + b with a = near and b = 0."""
    distance = np.linspace(1.0, 50.0, 500)
    fit = mv.fit_disparity(_reversed(distance, near=0.1).astype(np.float64), distance)
    assert fit[0] == pytest.approx(0.1, rel=1e-5)
    assert fit[1] == pytest.approx(0.0, abs=1e-6)
