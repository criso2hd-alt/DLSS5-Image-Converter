"""Detail recovery for the DLSS result.

DLAA is an anti-aliaser: on a real photo it treats genuine fine texture — brick
courses, perforations, railings — as aliasing to smooth, so the result comes
back softer than the source (measured ~9% on a test render, worse with jitter
and multiple passes). This module gives that detail back, three ways, from
faithful to punchy:

- **Preserve** re-injects the *source's own* high-frequency band onto the DLSS
  result. It is the ground truth — the real detail, not a guess — so nothing
  beats it for fidelity, and it is free (a couple of blurs). It needs the source
  and the result at the same size, which is the native/DLAA case.

- **Sharpen** is a plain unsharp mask, for when there is no clean source to lift
  detail from — after a supersample upscale, the boost path's crispening step.

(An off-the-shelf AI sharpener was trialled and dropped: bake-offs showed the
learned restorers, trained to reverse bicubic downscaling, do not recognise
DLAA's softening and add nothing over the free blend — see the project notes.)

Everything here operates on display-referred sRGB float in ``[0, 1]`` — the same
space the grade and effects use — and, like them, a neutral setting returns the
input untouched so it costs nothing until asked for.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Default frequency split. A ~2px Gaussian separates the fine texture (brick
#: mortar, perforation edges) from the tone/relight the neural pass legitimately
#: changed, so Preserve restores the former without undoing the latter.
DEFAULT_RADIUS = 2.0


def preserve_detail(
    result: np.ndarray,
    source: np.ndarray,
    amount: float = 1.0,
    radius: float = DEFAULT_RADIUS,
    preserve_range: bool = False,
) -> np.ndarray:
    """Re-inject the source's high-frequency detail onto the DLSS result.

    Keeps the result's low/mid frequencies — the colour, tone and relighting the
    neural pass produced — and swaps its softened high-frequency band for the
    source's crisp one. ``amount`` blends: 0 is the untouched result, 1 is full
    detail restored. ``result`` and ``source`` must be the same size, both sRGB
    float in ``[0, 1]`` (or scene-referred with ``preserve_range``).

    This is a frequency-domain graft, not a sharpen: it adds no acutance of its
    own and cannot overshoot, because the high band it lays down is one that
    genuinely existed in the photograph.
    """
    if amount <= 0.0:
        return result
    if source.shape != result.shape:
        raise ValueError(
            f"preserve_detail needs matching sizes; got result {result.shape} "
            f"and source {source.shape}."
        )
    result = np.asarray(result, dtype=np.float32)
    source = np.asarray(source, dtype=np.float32)
    radius = max(0.4, float(radius))

    source_hf = source - cv2.GaussianBlur(source, (0, 0), radius)
    result_hf = result - cv2.GaussianBlur(result, (0, 0), radius)
    # Replace the result's high band with the source's, scaled by amount.
    out = result + np.float32(amount) * (source_hf - result_hf)
    return np.maximum(out, 0.0) if preserve_range else np.clip(out, 0.0, 1.0)


def sharpen(
    image: np.ndarray,
    amount: float = 0.0,
    radius: float = DEFAULT_RADIUS,
    preserve_range: bool = False,
) -> np.ndarray:
    """Unsharp mask. For the boost path, where there is no source to preserve.

    ``amount`` is the weight of the high-pass (0 leaves the image untouched, ~1
    is a strong crispen). Unlike :func:`preserve_detail` this *can* overshoot
    into halos on high-contrast edges — that is the tradeoff for working with no
    reference — which is why the AI sharpener is the better crispening step where
    it is available.
    """
    if amount <= 0.0:
        return image
    image = np.asarray(image, dtype=np.float32)
    radius = max(0.4, float(radius))
    high = image - cv2.GaussianBlur(image, (0, 0), radius)
    out = image + np.float32(amount) * high
    return np.maximum(out, 0.0) if preserve_range else np.clip(out, 0.0, 1.0)
