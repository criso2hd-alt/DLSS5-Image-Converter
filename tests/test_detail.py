"""Detail recovery: the Preserve blend and the unsharp fallback."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from dlss5_converter import detail
from dlss5_converter.settings import AppSettings, DetailSettings


def _sharp_lines(size=128):
    yy, xx = np.mgrid[0:size, 0:size]
    pat = ((xx // 3) % 2).astype(np.float32)  # crisp 3px vertical bars
    return np.repeat(pat[:, :, None], 3, axis=2)


def _sharpness(img):
    g = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(g.astype(np.float32), cv2.CV_32F).var()


def test_preserve_amount_zero_is_a_passthrough():
    result = _sharp_lines()
    assert detail.preserve_detail(result, result, amount=0.0) is result


def test_preserve_restores_softened_detail():
    source = _sharp_lines()
    softened = cv2.GaussianBlur(source, (0, 0), 1.5)  # stand in for DLAA softening
    restored = detail.preserve_detail(softened, source, amount=1.0)
    # The restored image is much sharper than the softened one, and close to the
    # source it lifted the detail from.
    assert _sharpness(restored) > _sharpness(softened) * 1.5
    assert _sharpness(restored) == pytest.approx(_sharpness(source), rel=0.15)


def test_preserve_stays_in_range_and_never_overshoots_beyond_source():
    source = _sharp_lines()
    softened = cv2.GaussianBlur(source, (0, 0), 1.5)
    restored = detail.preserve_detail(softened, source, amount=1.0)
    assert restored.min() >= 0.0 and restored.max() <= 1.0


def test_preserve_requires_matching_sizes():
    a = _sharp_lines(128)
    b = _sharp_lines(64)
    with pytest.raises(ValueError, match="matching sizes"):
        detail.preserve_detail(a, b, amount=1.0)


def test_preserve_range_keeps_hdr_highlights():
    source = _sharp_lines() * 3.0  # scene-referred, above 1.0
    softened = cv2.GaussianBlur(source, (0, 0), 1.5)
    restored = detail.preserve_detail(softened, source, amount=1.0, preserve_range=True)
    assert restored.max() > 1.5  # not crushed to white


def test_sharpen_amount_zero_is_a_passthrough():
    img = _sharp_lines()
    assert detail.sharpen(img, amount=0.0) is img


def test_sharpen_increases_acutance():
    img = cv2.GaussianBlur(_sharp_lines(), (0, 0), 1.2)
    assert _sharpness(detail.sharpen(img, amount=1.5)) > _sharpness(img)


def test_boost_keeps_the_requested_factor_above_the_old_8k_cap():
    from dlss5_converter.pipeline import _boost_target
    assert _boost_target(4, 3840, 2160) == (15360, 8640)


def test_boost_refuses_only_the_d3d12_texture_limit():
    from dlss5_converter.pipeline import _boost_target
    with pytest.raises(RuntimeError, match="D3D12 textures stop at 16384"):
        _boost_target(8, 3840, 2160)


def test_boost_vram_preflight_uses_current_free_memory(monkeypatch):
    from dlss5_converter import hardware, pipeline
    monkeypatch.setattr(
        hardware,
        "query_nvidia_vram",
        lambda: hardware.VramInfo("RTX test", 16 * 1024**3, 15 * 1024**3, 1024**3),
    )
    with pytest.raises(RuntimeError, match="RTX test has 1.0 GB free"):
        pipeline._preflight_boost_vram(8000, 8000)


def test_boost_vram_preflight_allows_an_unknown_query(monkeypatch):
    from dlss5_converter import hardware, pipeline
    messages = []
    monkeypatch.setattr(hardware, "query_nvidia_vram", lambda: None)
    pipeline._preflight_boost_vram(8000, 8000, messages.append)
    assert "letting D3D12 decide" in messages[0]


def test_detail_settings_round_trip(tmp_path):
    path = tmp_path / "settings.json"
    a = AppSettings()
    a.detail.mode = "boost"
    a.detail.amount = 0.9
    a.detail.supersample = 8
    a.save(path)
    b = AppSettings.load(path)
    assert b.detail.mode == "boost"
    assert b.detail.amount == pytest.approx(0.9)
    assert b.detail.supersample == 8
    assert AppSettings().detail.is_neutral  # default is off
