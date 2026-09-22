"""The Ultra streaming core: tiling → merge → BigTIFF on disk, no GPU.

`_run_ultra_stream` takes the DLSS evaluation as an injected callback, so the
whole SR/tile/merge/stream pipeline can be exercised with an identity 'eval' and
no harness. The GPU path is covered by the user's own run; this pins the plumbing.
"""

from __future__ import annotations

import numpy as np
import pytest

from dlss5_converter import pipeline
from dlss5_converter.settings import AppSettings

tifffile = pytest.importorskip("tifffile")


def _identity_eval(linear, depth, tile):
    # Stand in for the DLSS pass: return the (already linear) input untouched.
    return linear


def test_ultra_stream_writes_bigtiff_and_returns_native(tmp_path):
    settings = AppSettings()
    settings.detail.mode = "ultra"
    settings.detail.sr_enabled = False   # engine=None below → Lanczos path
    settings.detail.ultra_factor = 2.0

    source = np.full((200, 300, 3), 0.4, np.float32)   # constant → survives resample
    depth = np.zeros((200, 300), np.float32)
    dest = tmp_path / "full.tiff"

    (big_w, big_h), native = pipeline._run_ultra_stream(
        source, depth, settings=settings, scratch=tmp_path,
        dest_tiff=dest, evaluate=_identity_eval, engine=None,
    )

    assert (big_w, big_h) == (600, 400)
    assert dest.exists()
    back = tifffile.imread(str(dest))
    assert back.shape == (400, 600, 3)
    assert back.dtype == np.uint16
    # A constant 0.4 sRGB image round-trips through linear→srgb to ~0.4.
    assert abs(int(back[200, 300, 0]) - round(0.4 * 65535)) <= 300

    assert native.shape == (200, 300, 3)
    assert np.allclose(native, 0.4, atol=2e-3)


def test_ultra_stream_cleans_scratch_and_tiles_when_large(tmp_path, monkeypatch):
    # Pin free memory: the planner clamps the factor to what RAM allows, so on
    # a machine that happens to be busy this asserted a size it never asked for.
    from dlss5_converter import hardware
    monkeypatch.setattr(hardware, "query_system_ram", lambda: 64 * 1024**3)
    monkeypatch.setattr(hardware, "query_nvidia_vram", lambda: None)
    settings = AppSettings()
    settings.detail.mode = "ultra"
    settings.detail.sr_enabled = False
    settings.detail.ultra_factor = 4.0

    # 3000×2000 at ×4 = 12000×8000 → multiple tiles (>7680 ceiling), exercising
    # the multi-tile merge and the memmap streaming.
    source = np.full((2000, 3000, 3), 0.6, np.float32)
    depth = np.zeros((2000, 3000), np.float32)
    dest = tmp_path / "big.tiff"

    (big_w, big_h), native = pipeline._run_ultra_stream(
        source, depth, settings=settings, scratch=tmp_path,
        dest_tiff=dest, evaluate=_identity_eval, engine=None,
    )
    assert (big_w, big_h) == (12000, 8000)
    assert dest.exists()
    assert native.shape == (2000, 3000, 3)
    assert np.allclose(native, 0.6, atol=3e-3)
    # The memmap accumulator scratch is cleaned up.
    assert not list(tmp_path.glob("*.dat"))
