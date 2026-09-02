"""Image sequences: finding them, and reading depth that came from a renderer.

Two jobs, both small, both easy to get subtly wrong.

**Finding the sequence.** Given one frame, work out which files belong with it.
Renderers name frames with a zero-padded counter — ``shot_0001.png`` — so the
rule is: split the trailing digits off the stem, and take every sibling that
matches the same prefix, the same digit width and the same extension. Digit
width matters: without it ``shot_1.png`` and ``shot_0001.png`` end up in the same
sequence, and a folder holding two renders at different padding silently
interleaves them.

**Reading supplied depth.** This is the whole reason sequence mode can be
temporally stable. Depth Anything is estimated per frame and wobbles slightly
between them; a depth pass out of Blender or Maya is geometrically exact and
rock steady, and feeding that instead removes the main source of flicker.

The catch is that nobody agrees which way up a depth map goes. This module
normalises to the one convention the rest of the pipeline uses — **near = 1.0,
far = 0.0**, the same as Depth Anything and the same as reversed-Z — and leaves
the choice of whether to flip to the caller, because it cannot be inferred
reliably and the app shows the result so a human can just look.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

#: What we will treat as a frame. EXR is included because renderers emit depth
#: that way and it is the only common format that survives real Z values. JPEG
#: XR because that is what an HDR screenshot is on Windows.
SEQUENCE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".hdr", ".webp", ".bmp",
    ".jxr", ".wdp", ".hdp",
}

_TRAILING_DIGITS = re.compile(r"^(?P<prefix>.*?)(?P<digits>\d+)$")


def split_frame_number(path: Path) -> tuple[str, int, int] | None:
    """`shot_0042.png` -> ("shot_", 42, 4). None if the stem has no counter."""
    match = _TRAILING_DIGITS.match(path.stem)
    if match is None:
        return None
    digits = match.group("digits")
    return match.group("prefix"), int(digits), len(digits)


def find_sequence(path: Path) -> list[Path]:
    """Every frame belonging with `path`, in frame order.

    Returns just `[path]` when the name carries no counter, so a single still
    dropped into sequence mode behaves like a one-frame sequence rather than an
    error.
    """
    path = Path(path)
    parts = split_frame_number(path)
    if parts is None:
        return [path]
    prefix, _, width = parts

    frames: list[tuple[int, Path]] = []
    for candidate in path.parent.iterdir():
        if candidate.suffix.lower() != path.suffix.lower():
            continue
        found = split_frame_number(candidate)
        if found is None:
            continue
        other_prefix, number, other_width = found
        # Same prefix and the same padding: two renders in one folder at
        # different widths are two sequences, not one.
        if other_prefix == prefix and other_width == width:
            frames.append((number, candidate))

    frames.sort()
    return [frame for _, frame in frames]


def load_depth_map(path: Path, invert: bool = False) -> np.ndarray:
    """A renderer's depth pass as 0..1 float32, near = 1.0.

    Normalised per frame against its own range. That is a deliberate trade: it
    handles metric EXR Z, 16-bit PNG and 8-bit alike without asking anyone for a
    near/far plane, at the cost of the mapping shifting if the depth range of the
    shot changes a lot between frames. For the fixed-camera work this is mostly
    used for, the range is stable and this is invisible.
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if image is None:
        raise OSError(f"Could not read the depth map {path}")

    if image.ndim == 3:
        # A depth pass written as RGB carries the same value in each channel;
        # averaging is harmless and survives a stray alpha.
        image = image[..., :3].mean(axis=2)
    depth = image.astype(np.float32)

    # Renderers write "no geometry here" as a huge value or as infinity, and one
    # such pixel would flatten the entire useful range into nothing.
    finite = np.isfinite(depth)
    if not finite.all():
        depth = np.where(finite, depth, np.nan)
        far = np.nanmax(depth)
        depth = np.nan_to_num(depth, nan=far if np.isfinite(far) else 0.0)

    low = float(depth.min())
    high = float(depth.max())
    if high - low < 1e-9:
        return np.zeros_like(depth, dtype=np.float32)
    depth = (depth - low) / (high - low)

    # After normalising, 1.0 is whatever was largest. For a mist or Z pass that
    # is the *far* plane, so it has to be flipped to match near = 1.0.
    return (1.0 - depth if invert else depth).astype(np.float32)


def describe(frames: list[Path]) -> str:
    """One line for the UI: how many frames, and which."""
    if not frames:
        return "no frames"
    if len(frames) == 1:
        return f"1 frame — {frames[0].name}"
    return f"{len(frames)} frames — {frames[0].name} … {frames[-1].name}"
