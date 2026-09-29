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

Frames may be 0..1 float, 8-bit, or 16-bit (HDR, PQ-encoded). Video
conversion passes integer frames straight through: warping and packing them
is several times faster than floats at 4K, and nothing is lost, as the
encoder takes 8 or 10 bits anyway.

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
        # pinning the 2nd..98th percentiles to 0..1 removes that drift. Every
        # 4th pixel each way is plenty to find them (a full 4K sort cost more
        # than the rest of this class put together).
        lo, hi = np.percentile(d[::4, ::4], (2, 98))
        d = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        thumb = cv2.resize(rgb, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32)
        if rgb.dtype.kind == "u":
            thumb /= float(np.iinfo(rgb.dtype).max)
        cut = self._thumb is None or float(np.abs(thumb - self._thumb).mean()) > 0.12
        self._thumb = thumb
        if cut or self._depth is None or self._depth.shape != d.shape:
            self._depth = d
        else:
            self._depth = self.alpha * self._depth + (1.0 - self.alpha) * d
        return self._depth.astype(np.float32)


_GRIDS: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinate maps, made once per frame size: two 4K float maps
    per frame are a measurable share of the warp."""
    grid = _GRIDS.get((h, w))
    if grid is None:
        _GRIDS.clear()
        grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        _GRIDS[(h, w)] = grid
    return grid


def shift_map(depth: np.ndarray, size: tuple[int, int], strength: float,
              pop_out: float) -> np.ndarray:
    """Per-pixel eye shift in pixels at `size` (w, h), from 0..1 depth of any
    size. Separate from the warp so video can make it on another thread."""
    w, h = size
    depth = depth.astype(np.float32)          # remap wants 32-bit maps
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    # Grow the near side of every edge by a couple of pixels, so the
    # foreground keeps its silhouette and the stretch happens on the
    # background, where it is least noticed.
    near = cv2.dilate(depth, np.ones((5, 5), np.uint8))
    # Positive shift = toward the viewer; the screen plane is at depth pop_out.
    # OpenCV's element-wise calls rather than numpy expressions here: they are
    # multi-threaded, and at 4K this function runs once per video frame.
    scale = float(strength) * MAX_DISPARITY * w * 0.5
    return cv2.addWeighted(near, scale, near, 0.0, -float(np.clip(1.0 - pop_out, 0.0, 1.0)) * scale)


def warp(rgb: np.ndarray, shift: np.ndarray):
    """Left and right eye views from a frame and its shift_map."""
    h, w = shift.shape
    xs, ys = _grid(h, w)
    image = rgb if rgb.dtype.kind == "u" else rgb.astype(np.float32)
    out = []
    for sign in (1.0, -1.0):                 # left eye sees near things shifted right
        # Backward warp, refined once: look up where each output pixel came
        # from, using the shift found at that source position.
        src = cv2.scaleAdd(shift, -sign, xs)
        s2 = cv2.remap(shift, src, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        src = cv2.scaleAdd(cv2.max(shift, s2), -sign, xs)
        out.append(cv2.remap(image, src, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    return out[0], out[1]


def views(rgb: np.ndarray, depth: np.ndarray, strength: float, pop_out: float):
    """Left and right eye views. `rgb` float 0..1 or integer HxWx3 (the views
    come back the same type), `depth` 0..1 (near=1)."""
    h, w = rgb.shape[:2]
    return warp(rgb, shift_map(depth, (w, h), strength, pop_out))


def pack(fmt: str, rgb: np.ndarray, depth: np.ndarray, left: np.ndarray | None = None,
         right: np.ndarray | None = None) -> np.ndarray:
    h, w = rgb.shape[:2]
    if rgb.dtype.kind == "u":
        top = float(np.iinfo(rgb.dtype).max)
        depth = (depth * top + 0.5).astype(rgb.dtype)
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
        if left.dtype.kind == "u":
            lum = np.clip(lum + 0.5, 0, np.iinfo(left.dtype).max).astype(left.dtype)
        return np.stack([lum, right[..., 1], right[..., 2]], axis=-1)
    raise ValueError(f"Unknown 3D format {fmt!r}")


def needs_views(fmt: str) -> bool:
    return fmt not in ("depth", "rgbd")


#: Longest edge the depth model is fed at. It works at ~518 px internally,
#: so a 4K input only costs a bigger resize for the same depth; smoothing is
#: done at this size too and the result is scaled up once, in compose.
DEPTH_EDGE = 1024


def depth_input(rgb8: np.ndarray) -> np.ndarray:
    """The frame at the size the depth model is given."""
    h, w = rgb8.shape[:2]
    scale = min(1.0, DEPTH_EDGE / float(max(h, w)))
    if scale >= 1.0:
        return rgb8
    return cv2.resize(rgb8, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def compose(rgb: np.ndarray, depth: np.ndarray, settings: StereoSettings,
            shift: np.ndarray | None = None) -> np.ndarray:
    """A frame and its steadied depth (any size) as a packed 3D frame.
    `shift` is the frame's shift_map when it was made ahead of time."""
    h, w = rgb.shape[:2]
    # Full-size depth is only drawn by the depth and colour+depth formats
    # (and read by shift_map below when no shift was made ahead).
    if depth.shape != (h, w) and not (shift is not None and settings.format in ("sbs_half", "sbs_full",
                                                                               "tb_half", "tb_full")):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    if needs_views(settings.format):
        if shift is None:
            shift = shift_map(depth, (w, h), settings.strength, settings.pop_out)
        left, right = warp(rgb, shift)
        packed = pack(settings.format, rgb, depth, left, right)
    else:
        packed = pack(settings.format, rgb, depth)
    # Integers cannot leave their range; only a float frame needs the clamp.
    return packed if packed.dtype.kind == "u" else np.clip(packed, 0.0, 1.0)


def frame(rgb: np.ndarray, disparity: np.ndarray, settings: StereoSettings, smoother: DepthSmoother) -> np.ndarray:
    """One finished frame as a packed 3D frame."""
    return compose(rgb, smoother(disparity, rgb), settings)
