"""A small colour grade applied to the finished image.

Deliberately *after* the neural pass, not before. Grading the input would change
what the model sees — the pass reasons about light transport, so lifting
exposure first changes the result rather than just the look — and every change
would cost a fresh harness launch. Grading the output is free, instant, and
undoable, which is what "this came out a bit washed out" actually needs.

Everything happens in **linear** light. Exposure and contrast applied straight to
sRGB values are the classic way to get muddy midtones and hue shifts in
saturated colour, because a stop of light is a multiply in linear and something
else entirely once the transfer curve is in the way.

All four controls are neutral at 0.0, and a fully neutral grade returns the
input untouched rather than round-tripping it through the transfer curve for
nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contract import linear_to_srgb, srgb_to_linear

#: Middle grey in linear light. Contrast pivots here so the image gets more
#: contrast rather than simply brighter or darker.
_PIVOT = 0.18

#: Rec.709 luma weights, which is the primary set the rest of the pipeline
#: assumes.
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


#: Encode table for the interactive path, indexed by ``sqrt(linear)`` rather
#: than by linear directly. A linear index wastes almost all of its resolution
#: on the highlights and bands visibly in the shadows — the square root is close
#: enough to perceptual that 4096 entries are indistinguishable from the real
#: curve, while costing one np.sqrt instead of a full np.power over the image.
_LUT_SIZE = 4096
_ENCODE_LUT = np.round(
    np.clip(
        linear_to_srgb(np.linspace(0.0, 1.0, _LUT_SIZE, dtype=np.float32) ** 2), 0.0, 1.0
    )
    * 255.0
).astype(np.uint8)


def _encode_fast(linear: np.ndarray) -> np.ndarray:
    """Linear float to 8-bit sRGB, for previews. ~10x quicker than the curve."""
    index = np.sqrt(np.clip(linear, 0.0, 1.0)) * (_LUT_SIZE - 1)
    return _ENCODE_LUT[index.astype(np.int32)]


@dataclass
class GradeSettings:
    """Neutral at all zeros."""

    #: Stops of exposure. A multiply in linear light.
    exposure: float = 0.0
    #: S-curve strength around middle grey. Negative flattens.
    contrast: float = 0.0
    #: Uniform saturation. -1 is greyscale.
    saturation: float = 0.0
    #: Saturation that backs off where colour is already strong, so skies and
    #: foliage lift without taking skin with them. The one to reach for first
    #: on a portrait.
    vibrance: float = 0.0

    @property
    def is_neutral(self) -> bool:
        return not any((self.exposure, self.contrast, self.saturation, self.vibrance))


def apply(image_rgb: np.ndarray, grade: GradeSettings) -> np.ndarray:
    """Grade 0..1 sRGB float RGB, returning 0..1 sRGB float RGB."""
    if grade.is_neutral:
        return image_rgb
    return apply_to_linear(srgb_to_linear(np.clip(image_rgb, 0.0, 1.0).astype(np.float32)), grade)


def apply_preview(linear: np.ndarray, grade: GradeSettings) -> np.ndarray:
    """Grade linear RGB straight to 8-bit sRGB, for the interactive preview.

    Same maths as apply_to_linear, but encodes through the lookup table and
    returns the uint8 the display wants anyway, skipping a float round trip.
    """
    return _encode_fast(_grade_linear(linear, grade))


def _grade_linear(linear: np.ndarray, grade: GradeSettings) -> np.ndarray:
    """The grade itself, linear in and linear out."""
    if grade.is_neutral:
        return linear

    if grade.exposure:
        linear = linear * np.float32(2.0**grade.exposure)

    if grade.contrast:
        # A power curve about middle grey. Guarding the zero keeps the log/pow
        # away from -inf on pure black, which otherwise comes back as NaN and
        # shows up as scattered black pixels in the shadows.
        gamma = np.float32(1.0 + grade.contrast)
        safe = np.maximum(linear, 1e-6)
        linear = np.float32(_PIVOT) * np.power(safe / np.float32(_PIVOT), gamma)

    if grade.saturation or grade.vibrance:
        luma = (linear * _LUMA).sum(axis=-1, keepdims=True)
        if grade.saturation:
            linear = luma + (linear - luma) * np.float32(1.0 + grade.saturation)
        if grade.vibrance:
            # How colourful this pixel already is, 0..1. Scaling the boost by
            # its inverse is what makes vibrance protect skin: faces sit at a
            # moderate saturation and get much less of the lift than a sky does.
            peak = linear.max(axis=-1, keepdims=True)
            floor = linear.min(axis=-1, keepdims=True)
            current = np.clip(peak - floor, 0.0, 1.0)
            amount = np.float32(grade.vibrance) * (1.0 - current)
            linear = luma + (linear - luma) * (1.0 + amount)

    return np.maximum(linear, 0.0)


def apply_to_linear(linear: np.ndarray, grade: GradeSettings) -> np.ndarray:
    """Grade already-linear RGB, returning 0..1 sRGB float."""
    return np.clip(linear_to_srgb(_grade_linear(linear, grade)), 0.0, 1.0)
