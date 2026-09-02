"""Carrying an HDR image through a pipeline that has to show it on a monitor.

Three separate jobs, and conflating them is how HDR support goes wrong:

**Which files are HDR.** `.jxr` (and its older `.wdp`/`.hdp` names) is what an
HDR screenshot is on Windows. `.exr` and `.hdr` are linear by convention and
always have been - they were readable here before, but were being treated as
though they carried an sRGB curve, which quietly crushed them.

**What the numbers mean.** Everything here is scRGB: linear light, sRGB
primaries, **1.0 at diffuse white**, and larger values for anything brighter
than a sheet of white paper in the scene. A 4.0 is not "clipped white", it is a
highlight four times brighter than white, and the whole reason to keep an HDR
image in float rather than clamping it on the way in.

That convention is also what DLSS 5 expects. The add-on's paper white, HDR
transfer and colour sliders exist because the neural pass is built to work in
this space - so feeding it a real HDR image is not an accommodation, it is the
input the model was designed for.

**How to show it.** A monitor gets 0..1. Tone mapping is unavoidable for the
preview and for any SDR export, and it must never touch the data on its way to
DLSS - only the copy being displayed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .contract import linear_to_srgb
from .wic import SUFFIXES as _WIC_SUFFIXES

#: scRGB's anchor. 1.0 in these images is 80 nits, the sRGB reference white.
#: Only used to explain the numbers - nothing here converts to absolute nits,
#: because the add-on's paper-white slider is what decides that downstream.
PAPER_WHITE_NITS = 80.0

#: Everything that arrives already linear and may exceed 1.0.
SUFFIXES = frozenset(_WIC_SUFFIXES | {".exr", ".hdr"})

#: Rec.709 luminance, matching the weights the grade already uses.
_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def is_hdr_source(path: str | Path) -> bool:
    """Whether this file should be read as linear light rather than sRGB."""
    return Path(path).suffix.lower() in SUFFIXES


def white_point(linear_rgb: np.ndarray, percentile: float = 99.9) -> float:
    """The luminance that tone mapping should bring down to 1.0.

    A percentile rather than the maximum: a single specular pixel at 300x
    diffuse white is common in a game capture, and letting it set the white
    point drags the entire image into the floor to make room for one dot.

    Never below 1.0, so an image that is HDR only in name is left alone rather
    than being stretched to fill a range it does not use.
    """
    if linear_rgb.size == 0:
        return 1.0
    luma = (linear_rgb[..., :3] * _LUMA).sum(axis=-1)
    return float(max(1.0, np.percentile(luma, percentile)))


def tonemap(linear_rgb: np.ndarray, white: float | None = None) -> np.ndarray:
    """Linear HDR RGB to 0..1 sRGB float, for display or an SDR export.

    Extended Reinhard on luminance, with the colour carried through by ratio.
    Two properties earn it its place over the alternatives:

    - it is the identity below the point where it has to start compressing, so
      an image whose highlights barely exceed 1.0 comes out looking like the
      SDR version rather than washed out;
    - it maps `white` exactly to 1.0 and nothing above it, so no highlight is
      ever clipped flat, which is the failure people actually notice.

    Operating on luminance rather than per channel is what stops bright
    saturated colours drifting towards white as they are compressed.
    """
    linear = np.maximum(linear_rgb[..., :3].astype(np.float32), 0.0)
    if white is None:
        white = white_point(linear)
    white = max(float(white), 1e-3)

    luma = (linear * _LUMA).sum(axis=-1, keepdims=True)
    # Extended Reinhard: L * (1 + L/W^2) / (1 + L). At L = W this is exactly 1.
    mapped = luma * (1.0 + luma / np.float32(white * white)) / (1.0 + luma)
    # Where the image is black there is no ratio to preserve; 1.0 leaves it black.
    scale = np.divide(mapped, luma, out=np.ones_like(luma), where=luma > 1e-6)

    return np.clip(linear_to_srgb(linear * scale), 0.0, 1.0)


def to_linear_preview(linear_rgb: np.ndarray, white: float | None = None) -> np.ndarray:
    """Tone map for the on-screen preview. Same thing, named for its caller."""
    return tonemap(linear_rgb, white)


def describe(linear_rgb: np.ndarray) -> str:
    """One line for the status bar, in the terms the image is actually in."""
    peak = float(np.max(linear_rgb[..., :3])) if linear_rgb.size else 0.0
    if peak <= 1.0:
        return "HDR container, but nothing above diffuse white"
    return f"HDR, peak {peak:.1f}x diffuse white"
