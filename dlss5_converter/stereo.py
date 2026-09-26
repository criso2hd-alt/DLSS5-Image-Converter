"""3D video from ordinary frames: depth per frame, then a view for each eye.

Each finished frame gets a depth estimate (Depth Anything), smoothed over
time, and every pixel is shifted left or right by how near it is: near things
move more between the two eyes than far ones, which is what reads as depth.
The two views are then packed into a 3D format a TV, headset or player takes.

Two things make or break converted 3D:
- Flicker. A per-frame depth model jitters, and in 3D that is surfaces
  wobbling toward and away from you. Depth is normalised per frame and blended
  with the previous frames; a scene cut resets it, so a cut does not smear one
  shot's depth into the next.
- Gaps. Shifting pixels uncovers thin strips behind edges that the camera
  never saw. Sampling backward from each eye's view stretches the background
  side into them, which is stable from frame to frame (an inpainting model
  would invent something different every frame and shimmer).

Qt-free.
"""

from __future__ import annotations

import cv2
import numpy as np

from .settings import StereoSettings

#: Packing formats: key -> (label, what plays it).
FORMATS: dict[str, tuple[str, str]] = {
    "sbs_half": ("Side by side (half width)",
                 "3D TVs and projectors, most VR video players. Same frame size as the source."),
    "sbs_full": ("Side by side (full width)",
                 "VR headsets and PC 3D players. Double width, the sharpest."),
    "tb_half": ("Top and bottom (half height)",
                "3D TVs and players that prefer over-under. Same frame size as the source."),
    "tb_full": ("Top and bottom (full height)",
                "VR players that prefer over-under. Double height."),
    "anaglyph": ("Anaglyph (red / cyan)",
                 "Any screen, with red/cyan glasses."),
    "depth": ("Depth video",
              "A greyscale depth pass (near is white) for editors and 3D tools."),
    "rgbd": ("Colour + depth side by side",
             "Looking Glass displays and players that build the 3D themselves."),
}

#: Largest shift between the two eyes at full strength, as a fraction of the
#: frame width. 3% is on the strong side of comfortable on a TV.
MAX_DISPARITY = 0.03


def output_size(fmt: str, size: tuple[int, int]) -> tuple[int, int]:
    w, h = size
    if fmt in ("sbs_full", "rgbd"):
        return w * 2, h
    if fmt == "tb_full":
        return w, h * 2
    return w, h


class DepthSmoother:
    """Steadies per-frame depth: normalised, then blended with the previous
    frames, reset at a scene cut."""

    def __init__(self, smoothing: float) -> None:
        self.alpha = float(np.clip(smoothing, 0.0, 0.95))
        self._depth = None
        self._thumb = None

    def __call__(self, disparity: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        d = disparity.astype(np.float32)
        # Depth models give relative depth whose range drifts frame to frame;
        # pinning the 2nd..98th percentiles to 0..1 removes that drift.
        lo, hi = np.percentile(d, (2, 98))
        d = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        thumb = cv2.resize(rgb.astype(np.float32), (32, 18), interpolation=cv2.INTER_AREA)
        cut = self._thumb is None or float(np.abs(thumb - self._thumb).mean()) > 0.12
        self._thumb = thumb
        if cut or self._depth is None or self._depth.shape != d.shape:
            self._depth = d
        else:
            self._depth = self.alpha * self._depth + (1.0 - self.alpha) * d
        return self._depth.astype(np.float32)


def views(rgb: np.ndarray, depth: np.ndarray, strength: float, pop_out: float):
    """Left and right eye views. `rgb` float 0..1 HxWx3, `depth` 0..1 (near=1)."""
    depth = depth.astype(np.float32)          # remap wants 32-bit maps
    h, w = depth.shape
    # Grow the near side of every edge by a couple of pixels, so the
    # foreground keeps its silhouette and the stretch happens on the
    # background, where it is least noticed.
    near = cv2.dilate(depth, np.ones((5, 5), np.uint8))
    # Positive shift = toward the viewer; the screen plane is at depth pop_out.
    shift = ((near - float(np.clip(1.0 - pop_out, 0.0, 1.0))) * float(strength) * MAX_DISPARITY * w * 0.5
             ).astype(np.float32)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    out = []
    for sign in (1.0, -1.0):                 # left eye sees near things shifted right
        # Backward warp, refined once: look up where each output pixel came
        # from, using the shift found at that source position.
        src = xs - sign * shift
        s2 = cv2.remap(shift, src, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        src = xs - sign * np.maximum(shift, s2)
        out.append(cv2.remap(rgb.astype(np.float32), src.astype(np.float32), ys, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE))
    return out[0], out[1]


def pack(fmt: str, rgb: np.ndarray, depth: np.ndarray, left: np.ndarray | None = None,
         right: np.ndarray | None = None) -> np.ndarray:
    h, w = rgb.shape[:2]
    if fmt == "depth":
        return np.repeat(depth[..., None], 3, axis=2)
    if fmt == "rgbd":
        return np.concatenate([rgb, np.repeat(depth[..., None], 3, axis=2)], axis=1)
    if fmt == "sbs_full":
        return np.concatenate([left, right], axis=1)
    if fmt == "tb_full":
        return np.concatenate([left, right], axis=0)
    if fmt == "sbs_half":
        half = (w // 2, h)
        return np.concatenate([cv2.resize(left, half, interpolation=cv2.INTER_AREA),
                               cv2.resize(right, (w - w // 2, h), interpolation=cv2.INTER_AREA)], axis=1)
    if fmt == "tb_half":
        return np.concatenate([cv2.resize(left, (w, h // 2), interpolation=cv2.INTER_AREA),
                               cv2.resize(right, (w, h - h // 2), interpolation=cv2.INTER_AREA)], axis=0)
    if fmt == "anaglyph":
        # Half-colour anaglyph: the left eye's red from its luminance keeps
        # the retinal rivalry of pure red/cyan down while leaving most colour.
        lum = left @ np.array([0.299, 0.587, 0.114], np.float32)
        return np.stack([lum, right[..., 1], right[..., 2]], axis=-1)
    raise ValueError(f"Unknown 3D format {fmt!r}")


def needs_views(fmt: str) -> bool:
    return fmt not in ("depth", "rgbd")


def frame(rgb: np.ndarray, disparity: np.ndarray, settings: StereoSettings, smoother: DepthSmoother) -> np.ndarray:
    """One finished frame as a packed 3D frame."""
    depth = smoother(disparity, rgb)
    if needs_views(settings.format):
        left, right = views(rgb, depth, settings.strength, settings.pop_out)
        return np.clip(pack(settings.format, rgb, depth, left, right), 0.0, 1.0)
    return np.clip(pack(settings.format, rgb, depth), 0.0, 1.0)
