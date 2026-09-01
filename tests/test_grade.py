"""The post-pass colour grade.

Applied to the finished image, in linear light. The tests that matter are the
boring ones: neutral must be a true no-op, and the maths must not produce NaN,
because a grade that quietly corrupts an 8K render is worse than no grade.
"""

from __future__ import annotations

import numpy as np
import pytest

from dlss5_converter.grade import GradeSettings, apply


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    return rng.random((32, 32, 3)).astype(np.float32)


def test_neutral_is_a_true_no_op(image):
    """Not "close enough" - the same object, with no transfer round trip."""
    assert apply(image, GradeSettings()) is image


def test_exposure_is_a_stop(image):
    """+1 stop is a doubling in linear light, not of the sRGB value."""
    from dlss5_converter.contract import linear_to_srgb, srgb_to_linear

    graded = apply(image, GradeSettings(exposure=1.0))
    expected = np.clip(linear_to_srgb(srgb_to_linear(image) * 2.0), 0.0, 1.0)
    assert np.allclose(graded, expected, atol=1e-5)


def test_exposure_brightens_and_negative_darkens(image):
    assert apply(image, GradeSettings(exposure=0.5)).mean() > image.mean()
    assert apply(image, GradeSettings(exposure=-0.5)).mean() < image.mean()


def test_contrast_pivots_around_middle_grey():
    """Middle grey must stay put; that is what makes it contrast."""
    from dlss5_converter.contract import linear_to_srgb

    grey = np.full((4, 4, 3), linear_to_srgb(np.float32(0.18)), dtype=np.float32)
    for amount in (-0.5, 0.5, 1.0):
        graded = apply(grey, GradeSettings(contrast=amount))
        assert np.allclose(graded, grey, atol=1e-4)


def test_contrast_spreads_the_histogram(image):
    assert apply(image, GradeSettings(contrast=0.5)).std() > image.std()
    assert apply(image, GradeSettings(contrast=-0.5)).std() < image.std()


def test_full_desaturation_is_grey(image):
    graded = apply(image, GradeSettings(saturation=-1.0))
    spread = graded.max(axis=-1) - graded.min(axis=-1)
    assert spread.max() < 1e-3


def test_vibrance_protects_what_is_already_saturated():
    """The whole point of vibrance over saturation: skin moves less than sky."""
    muted = np.full((4, 4, 3), 0.5, dtype=np.float32)
    muted[..., 0] = 0.55  # barely coloured
    vivid = np.zeros((4, 4, 3), dtype=np.float32)
    vivid[..., 0] = 1.0  # fully saturated red

    grade = GradeSettings(vibrance=1.0)
    muted_shift = np.abs(apply(muted, grade) - muted).mean()
    vivid_shift = np.abs(apply(vivid, grade) - vivid).mean()
    assert muted_shift > vivid_shift


def test_black_does_not_become_nan():
    """A power curve about a pivot hits log(0) on pure black if unguarded."""
    black = np.zeros((4, 4, 3), dtype=np.float32)
    graded = apply(black, GradeSettings(contrast=1.0, exposure=-1.0, vibrance=1.0))
    assert np.isfinite(graded).all()


def test_output_stays_in_range(image):
    extreme = GradeSettings(exposure=2.0, contrast=1.0, saturation=1.0, vibrance=1.0)
    graded = apply(image, extreme)
    assert graded.min() >= 0.0
    assert graded.max() <= 1.0
    assert np.isfinite(graded).all()
