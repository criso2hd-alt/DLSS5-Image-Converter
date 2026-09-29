"""HDR video in and out: PQ (HDR10) and HLG sources, HDR10 output.

DLSS takes linear light, and in a game HDR frame the highlights simply run
above 1.0. So an HDR video is brought to exactly that: decoded at 16 bits,
its transfer curve undone, BT.2020 turned into BT.709 primaries, and scaled
so SDR reference white (203 nits, ITU-R BT.2408) sits at 1.0. Everything
brighter stays above 1.0 in the half-float plane, as a game would send it.

Out, the same steps run backwards into 16-bit PQ BT.2020, which the writer
tags and encodes as 10-bit HDR10. A codec that cannot carry HDR (H.264, VP9)
gets a tone-mapped SDR frame instead, with the highlights rolled off rather
than clipped.

Every curve here is a table: input codes are 16-bit and DLSS returns half
floats, so each has at most 65536 values, and a table lookup is a small
fraction of the cost of evaluating PQ's powers over 25 million samples.

Qt-free.
"""

from __future__ import annotations

import threading
from functools import lru_cache

import cv2
import numpy as np

from .contract import table_lookup

#: Linear 1.0 in the DLSS plane, in nits: BT.2408's HDR reference white,
#: where SDR white is placed in an HDR programme.
SDR_WHITE_NITS = 203.0
#: Nominal peak of an HLG display; the HLG curve is relative to it.
HLG_PEAK_NITS = 1000.0

# SMPTE ST 2084 (PQ) constants.
_M1 = 2610.0 / 16384.0
_M2 = 2523.0 / 4096.0 * 128.0
_C1 = 3424.0 / 4096.0
_C2 = 2413.0 / 4096.0 * 32.0
_C3 = 2392.0 / 4096.0 * 32.0

# ARIB STD-B67 (HLG) constants.
_HLG_A = 0.17883277
_HLG_B = 1.0 - 4.0 * _HLG_A
_HLG_C = 0.5 - _HLG_A * np.log(4.0 * _HLG_A)

#: Linear-light primaries conversion (rows produce R, G, B).
BT2020_TO_BT709 = np.array([[1.6605, -0.5876, -0.0728],
                            [-0.1246, 1.1329, -0.0083],
                            [-0.0182, -0.1006, 1.1187]], np.float32)
BT709_TO_BT2020 = np.array([[0.6274, 0.3293, 0.0433],
                            [0.0691, 0.9195, 0.0114],
                            [0.0164, 0.0880, 0.8956]], np.float32)

#: FFmpeg's AVColorTransferCharacteristic values for the two HDR curves.
TRC_PQ = 16
TRC_HLG = 18


def kind_of(color_trc) -> str:
    """'pq', 'hlg' or '' (SDR) from a stream or frame's transfer tag."""
    try:
        value = int(color_trc)
    except (TypeError, ValueError):
        return ""
    return {TRC_PQ: "pq", TRC_HLG: "hlg"}.get(value, "")


def pq_to_nits(signal: np.ndarray) -> np.ndarray:
    e = np.power(np.clip(signal, 0.0, 1.0), 1.0 / _M2)
    return 10000.0 * np.power(np.maximum(e - _C1, 0.0) / (_C2 - _C3 * e), 1.0 / _M1)


def nits_to_pq(nits: np.ndarray) -> np.ndarray:
    y = np.power(np.clip(nits / 10000.0, 0.0, 1.0), _M1)
    return np.power((_C1 + _C2 * y) / (1.0 + _C3 * y), _M2)


def hlg_to_nits(signal: np.ndarray) -> np.ndarray:
    """HLG signal to display light. The OOTF is applied per channel (gamma
    1.2 at a 1000 nit display), close to the luminance-based original and
    what makes this a table rather than a per-pixel calculation."""
    e = np.clip(signal, 0.0, 1.0)
    with np.errstate(over="ignore"):
        scene = np.where(e <= 0.5, e * e / 3.0, (np.exp((e - _HLG_C) / _HLG_A) + _HLG_B) / 12.0)
    return HLG_PEAK_NITS * np.power(np.clip(scene, 0.0, 1.0), 1.2)


@lru_cache(maxsize=2)
def decode_lut(kind: str) -> np.ndarray:
    """16-bit code -> linear light with SDR white at 1.0 (float32, 65536)."""
    signal = np.arange(65536, dtype=np.float64) / 65535.0
    nits = hlg_to_nits(signal) if kind == "hlg" else pq_to_nits(signal)
    return (nits / SDR_WHITE_NITS).astype(np.float32)


def _half_values() -> np.ndarray:
    values = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(np.float16).astype(np.float64)
    return np.nan_to_num(values, nan=0.0, posinf=65504.0, neginf=0.0)


@lru_cache(maxsize=1)
def half_to_pq_lut() -> np.ndarray:
    """Half-float bits (linear, 1.0 = SDR white) -> 16-bit PQ code."""
    nits = np.maximum(_half_values(), 0.0) * SDR_WHITE_NITS
    return (nits_to_pq(nits) * 65535.0 + 0.5).astype(np.uint16)


def tone_map(linear: np.ndarray) -> np.ndarray:
    """HDR linear (1.0 = SDR white) to 0..1 SDR linear. Untouched up to 0.8,
    then a smooth shoulder that reaches white instead of clipping at it, so
    skies and lamps keep their shape in an SDR file."""
    knee = 0.8
    x = np.maximum(linear, 0.0)
    over = x > knee
    y = x.copy()
    y[over] = knee + (1.0 - knee) * np.tanh((x[over] - knee) / (1.0 - knee))
    return y


@lru_cache(maxsize=1)
def half_to_sdr8_lut() -> np.ndarray:
    """Half-float bits (HDR linear) -> tone-mapped 8-bit sRGB."""
    from .contract import linear_to_srgb

    srgb = np.clip(linear_to_srgb(tone_map(_half_values()).astype(np.float32)), 0.0, 1.0)
    return (srgb * 255.0 + 0.5).astype(np.uint8)


_REFORMATTERS = threading.local()


def _reformatter():
    """One swscale converter per thread, kept for the whole clip; a fresh one
    per frame sets up the 4K scaler again every time."""
    reformatter = getattr(_REFORMATTERS, "value", None)
    if reformatter is None:
        from av.video.reformatter import VideoReformatter

        reformatter = _REFORMATTERS.value = VideoReformatter()
    return reformatter


def decode_frame(frame, kind: str) -> np.ndarray:
    """An HDR PyAV frame -> linear BT.709 float32 RGB, SDR white at 1.0.

    Colours outside BT.709 come out slightly negative and are clamped: DLSS
    expects non-negative light, and the loss is limited to the most saturated
    wide-gamut colours.
    """
    range_ = "JPEG" if int(getattr(frame, "color_range", 0) or 0) == 2 else "MPEG"
    # Output tags are left alone on purpose: requesting different ones makes
    # swscale convert transfer and primaries, which is the job done below.
    rgb48 = _reformatter().reformat(frame, format="rgb48le", src_colorspace="bt2020",
                                    dst_colorspace="bt2020", src_color_range=range_,
                                    dst_color_range="JPEG").to_ndarray()
    linear2020 = table_lookup(decode_lut(kind), rgb48)
    linear709 = cv2.transform(linear2020, BT2020_TO_BT709)
    return np.maximum(linear709, 0.0, out=linear709)


def to_pq16(linear709: np.ndarray) -> np.ndarray:
    """Linear BT.709 float (SDR white 1.0) -> 16-bit PQ BT.2020 RGB.

    A float RGBA frame is taken too (alpha ignored). The matrix, then half
    floats (OpenCV's conversion: bit-identical to numpy's, ~6x faster), then
    the PQ table on those bits.
    """
    image = np.ascontiguousarray(linear709, np.float32)
    matrix = BT709_TO_BT2020
    if image.shape[-1] == 4:
        matrix = np.hstack([matrix, np.zeros((3, 1), np.float32)])
    linear2020 = cv2.transform(image, matrix)
    return table_lookup(half_to_pq_lut(), cv2.convertFp16(linear2020).view(np.uint16))


def half_bits_to_pq16(bits: np.ndarray) -> np.ndarray:
    """DLSS's RGBA half-float output bits (linear BT.709) -> 16-bit PQ BT.2020 RGB."""
    return to_pq16(cv2.convertFp16(np.ascontiguousarray(bits).view(np.int16)))


@lru_cache(maxsize=1)
def pq16_to_sdr8_lut() -> np.ndarray:
    """16-bit PQ code -> tone-mapped 8-bit sRGB (primaries left as they are:
    this is for depth estimation and previews, where a small hue shift in
    the most saturated colours does not matter)."""
    from .contract import linear_to_srgb

    linear = tone_map(decode_lut("pq").astype(np.float64)).astype(np.float32)
    return (np.clip(linear_to_srgb(linear), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def pq16_to_sdr8(pq16: np.ndarray) -> np.ndarray:
    """An 8-bit SDR look at a PQ frame, for the depth model and previews."""
    return table_lookup(pq16_to_sdr8_lut(), pq16)
