"""Read the real depth the DLSS5 Scene Capture add-on saves beside screenshots.

The ReShade add-on (native/scene_capture) writes, beside ``shot.png``:

    shot.depth.f32    raw depth buffer values, float32, top row first
    shot.depth.json   width, height, and a few statistics about the values

Why this matters: Depth Anything guesses depth from each photo on its own, and
each guess is warped differently, which is what made multi-shot scenes ghost.
The game's depth buffer is a measurement, consistent from shot to shot.

It also slots straight into the existing pipeline. Modern games store
reversed Z with an infinite far plane, ``near / distance``, which is exactly
the inverse-depth shape ``multiview`` already expects from Depth Anything
(``a / distance + b``), with ``a`` the near plane and ``b`` zero. So this module
returns a "disparity" map, and everything downstream fits it as before, only
now the fit is exact and there is nothing left for the depth bending to fix.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def sidecars(image_path: str | Path) -> tuple[Path, Path]:
    """Where the add-on puts the depth and its description for a screenshot."""
    path = Path(image_path)
    return path.with_suffix(".depth.f32"), path.with_suffix(".depth.json")


def is_reversed(info: dict, raw: np.ndarray) -> bool:
    """True when the buffer is reversed Z (near = 1, sky = 0).

    Sky and far geometry pile up at one end of the range: at 0 in reversed Z,
    at 1 in standard Z. The add-on records both fractions; if neither end is
    populated (an indoor shot with no sky), the median decides, since reversed Z
    crowds almost every surface towards 0 and standard Z towards 1.
    """
    at_zero = float(info.get("fraction_at_zero", np.mean(raw <= 1e-7)))
    at_one = float(info.get("fraction_at_one", np.mean(raw >= 1.0 - 1e-7)))
    if abs(at_zero - at_one) > 0.001:
        return at_zero > at_one
    return float(np.median(raw)) < 0.5


def load(image_path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray | None:
    """The game's depth for a screenshot as inverse depth (1 = near), or None.

    ``size`` is the (width, height) the caller is working at; the map is
    resized to it with nearest-neighbour sampling, because blending across a
    depth edge would invent surfaces halfway between a face and a far wall.
    Returns None when the screenshot has no sidecar, or the sidecar is damaged,
    so callers fall back to Depth Anything instead of failing.
    """
    depth_path, info_path = sidecars(image_path)
    if not depth_path.is_file() or not info_path.is_file():
        return None
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        width, height = int(info["width"]), int(info["height"])
        raw = np.fromfile(depth_path, dtype="<f4")
        if raw.size != width * height:
            return None
        raw = raw.reshape(height, width)
    except Exception:  # noqa: BLE001 - a bad sidecar must degrade, never block
        return None
    if not np.isfinite(raw).all():
        raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0)

    # Standard Z with an infinite far plane stores 1 - near / distance, so one
    # subtraction puts it in the same shape as reversed Z.
    disparity = raw if is_reversed(info, raw) else 1.0 - raw
    disparity = np.clip(disparity, 0.0, 1.0).astype(np.float32)

    if size is not None and (width, height) != tuple(size):
        disparity = cv2.resize(disparity, tuple(size), interpolation=cv2.INTER_NEAREST)
    return disparity
