"""Exporting at a size other than the one DLSS ran at."""

from __future__ import annotations

import numpy as np
import pytest

from dlss5_converter import resample
from dlss5_converter.contract import srgb_to_linear


def test_fit_keeps_the_aspect_ratio():
    assert resample.fit((3840, 2160), 1920) == (1920, 1080)
    assert resample.fit((2160, 3840), 1920) == (1080, 1920)


def test_fit_never_rounds_a_side_to_zero():
    """A panorama shrunk hard still has to be at least one pixel tall."""
    assert resample.fit((10000, 3), 100) == (100, 1)


def test_match_completes_the_other_side():
    assert resample.match((1600, 900), 800, None) == (800, 450)
    assert resample.match((1600, 900), None, 450) == (800, 450)
    assert resample.match((1600, 900), 800, 800) == (800, 800), "both given wins"
    assert resample.match((1600, 900), None, None) == (1600, 900)


def test_native_size_is_returned_untouched():
    """Saving at native size must not quietly resample."""
    image = np.random.default_rng(0).random((32, 48, 3)).astype(np.float32)
    assert resample.resize(image, 48, 32) is image


def test_resize_produces_the_exact_size_asked_for():
    image = np.zeros((100, 200, 3), np.float32)
    assert resample.resize(image, 400, 200).shape == (200, 400, 3)
    assert resample.resize(image, 50, 25).shape == (25, 50, 3)


def test_a_flat_colour_survives_a_round_trip():
    """The transfer curve has to be applied and undone, not just applied."""
    image = np.full((16, 16, 3), 0.5, np.float32)
    scaled = resample.resize(image, 64, 64)
    assert np.allclose(scaled, 0.5, atol=1e-3)


def test_downscaling_averages_in_linear_light():
    """The whole reason this module exists rather than one call to cv2.resize.

    A checkerboard of black and white is half the light. Averaged in sRGB it
    comes out at 0.5 - which is only 21% of the light - and the image darkens.
    Averaged correctly it lands near 0.735, the encoding of linear 0.5.
    """
    board = np.indices((64, 64)).sum(axis=0) % 2
    image = np.repeat(board[:, :, None].astype(np.float32), 3, axis=2)

    scaled = resample.resize(image, 1, 1)
    linear = float(srgb_to_linear(scaled)[0, 0, 0])
    assert linear == pytest.approx(0.5, abs=0.02), "should preserve total light"
    assert float(scaled[0, 0, 0]) > 0.7, "the naive sRGB average would be 0.5"


def test_upscaling_stays_in_range():
    """Lanczos overshoots at edges; nothing may leave 0..1 or go NaN."""
    image = np.zeros((32, 32, 3), np.float32)
    image[:, 16:] = 1.0
    scaled = resample.resize(image, 128, 128)
    assert np.isfinite(scaled).all()
    assert scaled.min() >= 0.0 and scaled.max() <= 1.0


def test_an_absurd_size_is_refused_rather_than_attempted():
    image = np.zeros((8, 8, 3), np.float32)
    with pytest.raises(ValueError, match="capped"):
        resample.resize(image, 40000, 40000)
    with pytest.raises(ValueError):
        resample.resize(image, 0, 100)


def test_describe_names_the_filter():
    assert "no resampling" in resample.describe((100, 100), (100, 100))
    assert "Lanczos" in resample.describe((100, 100), (200, 200))
    assert "area" in resample.describe((100, 100), (50, 50))
    assert "2.00x" in resample.describe((100, 100), (200, 200))
