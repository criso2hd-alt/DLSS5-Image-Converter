"""Resize the finished image on the way out.

People kept reading **Max size** as "the resolution I get", and it is not — it
is the resolution DLSS *runs at*, chosen for VRAM and time. What they actually
wanted was to export bigger than that, which is a separate question and belongs
at save time.

Deliberately not an AI upscaler. The neural pass has already run; a second model
on top of it would fight the first and produce the waxy compounding that already
catches people out on a second pass. This is a plain, correct resampler.

Two things make it better than the obvious one-liner:

**It resamples in linear light.** Averaging sRGB-encoded values averages the
wrong quantity - the encoding is a power curve, so the mean of two encoded
samples is not the encoding of their mean. The visible result is edges that
darken as they soften, worst on exactly the high-contrast detail this tool
sharpens. Decoding first costs one pass over the image and removes the error.

**It picks the filter from the direction.** Lanczos reconstructs detail when
enlarging and is the right choice there. Shrinking with it undersamples - it
reads a handful of source pixels per output pixel and simply misses the rest,
which aliases. Area averaging looks at every source pixel that lands in the
output pixel, which is what downscaling means.
"""

from __future__ import annotations

import cv2
import numpy as np

from .contract import linear_to_srgb, srgb_to_linear

#: Refuse sizes that would allocate absurd amounts. 16K on the long edge is
#: already 1.5 GB of float32 mid-resize; past that a typo costs a swap storm.
MAX_EDGE = 16384


def fit(size: tuple[int, int], long_edge: int) -> tuple[int, int]:
    """Scale `size` so its longest side is `long_edge`, keeping the aspect."""
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("Size must be positive.")
    scale = long_edge / max(width, height)
    return proportional(size, scale)


def proportional(size: tuple[int, int], scale: float) -> tuple[int, int]:
    """Scale both sides by `scale`, never rounding a side away to nothing."""
    width, height = size
    return (max(1, round(width * scale)), max(1, round(height * scale)))


def match(size: tuple[int, int], width: int | None, height: int | None) -> tuple[int, int]:
    """Complete a partly-specified size, keeping the original aspect ratio.

    Given one side, the other follows. This is what the locked aspect ratio in
    the export dialog is doing on every keystroke.
    """
    original_width, original_height = size
    if width and height:
        return (max(1, width), max(1, height))
    if width:
        return (max(1, width), max(1, round(width * original_height / original_width)))
    if height:
        return (max(1, round(height * original_width / original_height)), max(1, height))
    return size


def resize(image_rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resample 0..1 sRGB float RGB to exactly `width` x `height`.

    Returns the input untouched when the size already matches, so saving at
    native size costs nothing and cannot introduce a resample of its own.
    """
    source_height, source_width = image_rgb.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("Export size must be positive.")
    if width > MAX_EDGE or height > MAX_EDGE:
        raise ValueError(f"Export size is capped at {MAX_EDGE} px on a side.")
    if (width, height) == (source_width, source_height):
        return image_rgb

    enlarging = width * height > source_width * source_height
    # INTER_AREA has no meaning when enlarging - OpenCV falls back to bilinear -
    # so the choice really is per direction rather than a preference.
    interpolation = cv2.INTER_LANCZOS4 if enlarging else cv2.INTER_AREA

    linear = srgb_to_linear(np.clip(image_rgb, 0.0, 1.0).astype(np.float32))
    scaled = cv2.resize(linear, (width, height), interpolation=interpolation)
    # Lanczos rings: it overshoots on both sides of a hard edge, and negative
    # linear light is not a colour. Clamping here rather than after encoding
    # keeps the overshoot from turning into NaN in the transfer curve.
    scaled = np.clip(scaled, 0.0, None)
    return np.clip(linear_to_srgb(scaled), 0.0, 1.0)


def describe(size: tuple[int, int], target: tuple[int, int]) -> str:
    """One line for the dialog, naming the operation rather than the numbers."""
    width, height = size
    new_width, new_height = target
    if (width, height) == (new_width, new_height):
        return "Native size - no resampling."
    scale = max(new_width / width, new_height / height)
    how = "Lanczos, in linear light" if scale > 1 else "area averaged, in linear light"
    return f"{scale:.2f}x - {how}."
