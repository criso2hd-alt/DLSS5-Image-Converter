"""Turn one photo into the DLAA frame contract DLSS expects.

This module is the whole trick, so it is worth stating plainly what it is doing.

DLSS 5's neural renderer is not an image model with an image input. It is an NGX
snippet that rides along on a DLSS Super Resolution evaluation — the RenoDX
add-on hooks ``NGX_D3D12_EVALUATE_DLSS`` and injects the neural pass there. To
run it on a photo we therefore do not "call DLSS 5"; we hand DLSS a frame that
looks like it came out of a game, in DLAA mode (render size == output size), and
let the add-on take it from there.

A DLAA evaluation wants colour, depth, motion vectors, and a jitter offset:

* **Colour** — the photo, sRGB-decoded to linear and stored as RGBA16F. Linear
  matters: the neural pass reasons about light transport, and feeding it gamma
  values makes it read shadows as mid-grey and over-lift them.
* **Depth** — see ``to_hardware_depth`` below.
* **Motion vectors** — zeros. Nothing moved between our identical frames, and
  claiming otherwise makes DLSS reject its own history and smear.
* **Jitter** — optional, and the one place where a still image can be made to
  behave like a real render. See ``halton``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Contract:
    """The three planes plus the metadata the harness needs to interpret them."""

    colour: np.ndarray  # (h, w, 4) float16, linear, alpha = 1
    depth: np.ndarray  # (h, w)    float32, reversed-Z hardware depth
    motion: np.ndarray  # (h, w, 2) float16, pixels, all zero for a still
    jitter: list[tuple[float, float]]  # per-frame sub-pixel offsets

    @property
    def size(self) -> tuple[int, int]:
        height, width = self.depth.shape[:2]
        return width, height


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """Decode 0..1 sRGB to linear with the real piecewise curve.

    The 2.2-power shortcut is wrong by up to 4% in the deep shadows, which is
    precisely the range the neural relighting pass amplifies. Cheap to do
    properly; visibly better in dark portraits.
    """
    low = image / 12.92
    high = np.power((image + 0.055) / 1.055, 2.4)
    return np.where(image <= 0.04045, low, high).astype(np.float32)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    low = image * 12.92
    high = 1.055 * np.power(np.maximum(image, 0.0), 1.0 / 2.4) - 0.055
    return np.where(image <= 0.0031308, low, high).astype(np.float32)


def halton(index: int, base: int) -> float:
    """One term of the Halton low-discrepancy sequence, in [0, 1)."""
    result, fraction = 0.0, 1.0
    while index > 0:
        fraction /= base
        result += fraction * (index % base)
        index //= base
    return result


def jitter_sequence(frames: int) -> list[tuple[float, float]]:
    """The (2, 3) Halton pattern every DLSS integration guide recommends.

    Offsets are in pixels, centred on zero. The first frame is deliberately
    *not* forced to (0, 0): starting at the sequence origin and only then
    spreading out biases the accumulated result towards the unjittered sample.
    """
    return [
        (halton(i + 1, 2) - 0.5, halton(i + 1, 3) - 0.5) for i in range(max(1, frames))
    ]


def to_hardware_depth(inverse_depth: np.ndarray, contrast: float = 1.0) -> np.ndarray:
    """Depth Anything's normalised inverse depth as a reversed-Z depth buffer.

    Games overwhelmingly render with reversed-Z and an infinite far plane, where
    the value written to the depth buffer is ``near / z_view``: 1.0 at the near
    plane, decaying towards 0.0 at infinity. Depth Anything V2 emits normalised
    *inverse* relative depth — 1.0 nearest, 0.0 furthest. Those are the same
    curve, which is why this function is a remap and not a reprojection: there
    is no metric depth to recover and no camera to reconstruct.

    The harness passes ``--reversed-depth`` so NGX interprets the range the same
    way ReShade's ``RESHADE_DEPTH_INPUT_IS_REVERSED`` does in a real game.

    ``contrast`` reshapes the near-far distribution without flipping it. Values
    above 1.0 pull the scene towards the near plane so more of the frame reads
    as foreground; the exponent form keeps the endpoints pinned at 0 and 1, so
    the reversed-Z contract still holds whatever the user does with the slider.
    """
    depth = np.clip(inverse_depth.astype(np.float32), 0.0, 1.0)
    contrast = float(np.clip(contrast, 0.2, 5.0))
    if abs(contrast - 1.0) > 1e-3:
        depth = np.power(depth, 1.0 / contrast)
    # Never hand DLSS an exact 0.0 or 1.0 plane. Some depth paths treat the far
    # value as "nothing was rendered here" and skip the pixel entirely, which
    # shows up as untouched rectangles in flat sky.
    return np.clip(depth, 1.0 / 65504.0, 1.0 - 1e-6).astype(np.float32)


def fit_to_budget(image: np.ndarray, max_edge: int) -> np.ndarray:
    """Downscale so the longest edge fits, and make both edges even.

    DLSS allocates internal buffers from the dimensions it is given and odd
    sizes fail on some builds during the internal half-resolution passes. INTER_AREA
    for the downscale because it is the only filter here that does not
    pre-sharpen the input and hand the neural pass a false edge to chase.
    """
    height, width = image.shape[:2]
    scale = min(1.0, float(max_edge) / float(max(height, width)))
    target_w = max(64, int(round(width * scale)) & ~1)
    target_h = max(64, int(round(height * scale)) & ~1)
    if (target_w, target_h) == (width, height):
        return image
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (target_w, target_h), interpolation=interpolation)


def load_image(path: str | Path) -> np.ndarray:
    """Read any supported image as 0..1 float32 RGB.

    ``IMREAD_UNCHANGED`` so 16-bit PNG/TIFF and HDR sources keep their range
    instead of being silently crushed to 8 bits on the way in.
    """
    source = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if source is None:
        raise ValueError(f"Could not read {Path(path).name} — unsupported or corrupt.")
    if source.ndim == 2:
        source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    source = source[:, :, :3]
    if np.issubdtype(source.dtype, np.integer):
        source = source.astype(np.float32) / float(np.iinfo(source.dtype).max)
    else:
        source = np.nan_to_num(source.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    return np.ascontiguousarray(source[:, :, ::-1])  # BGR -> RGB


def build(
    image_rgb: np.ndarray,
    inverse_depth: np.ndarray,
    *,
    depth_contrast: float = 1.0,
    frames: int = 8,
    jitter: bool = True,
    already_linear: bool = False,
) -> Contract:
    """Assemble the planes. `image_rgb` is 0..1 float32 RGB at final size."""
    height, width = image_rgb.shape[:2]
    if inverse_depth.shape[:2] != (height, width):
        inverse_depth = cv2.resize(
            inverse_depth, (width, height), interpolation=cv2.INTER_LINEAR
        )

    linear = image_rgb if already_linear else srgb_to_linear(np.clip(image_rgb, 0.0, 1.0))
    colour = np.empty((height, width, 4), np.float16)
    colour[..., :3] = linear.astype(np.float16)
    colour[..., 3] = np.float16(1.0)

    offsets = jitter_sequence(frames) if jitter else [(0.0, 0.0)] * max(1, frames)

    return Contract(
        colour=np.ascontiguousarray(colour),
        depth=np.ascontiguousarray(to_hardware_depth(inverse_depth, depth_contrast)),
        # Zero, not merely small. A non-zero motion vector on a static frame
        # tells DLSS its history is stale and it discards the accumulation we
        # are running multiple frames specifically to build.
        motion=np.zeros((height, width, 2), np.float16),
        jitter=offsets,
    )


def shift_subpixel(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Resample by a fractional pixel offset, matching a jittered projection.

    A real renderer gets new samples by nudging the projection matrix. We can
    only resample what the photo already has, so this adds no information — but
    it does give the temporal accumulator the phase diversity it expects, which
    is what stops multi-frame evaluation from just re-deriving frame one.

    Lanczos rather than bicubic: over eight accumulated frames bicubic's
    overshoot compounds into visible ringing on high-contrast edges.
    """
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return image
    matrix = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )


def write_planes(contract: Contract, folder: Path) -> dict[str, Path]:
    """Dump the planes as raw binary for the native harness.

    Raw rather than a container format on purpose: the harness is C++ with no
    dependencies, and every byte of parsing there is a byte that can go wrong at
    a layer we cannot debug from Python. Dimensions travel on the command line.
    """
    folder.mkdir(parents=True, exist_ok=True)
    paths = {
        "colour": folder / "colour.bin",
        "depth": folder / "depth.bin",
        "motion": folder / "motion.bin",
    }
    contract.colour.tofile(paths["colour"])
    contract.depth.tofile(paths["depth"])
    contract.motion.tofile(paths["motion"])
    return paths


def read_output(path: Path, width: int, height: int) -> np.ndarray:
    """Read the harness's RGBA16F result back as linear float32 RGB."""
    expected = width * height * 4
    data = np.fromfile(path, dtype=np.float16)
    if data.size != expected:
        raise ValueError(
            f"DLSS returned {data.size} values, expected {expected}. "
            "The harness and the contract disagree about the frame size."
        )
    frame = data.reshape(height, width, 4).astype(np.float32)
    return np.nan_to_num(frame[..., :3], nan=0.0, posinf=1.0, neginf=0.0)
