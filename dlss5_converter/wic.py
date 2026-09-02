"""Read and write JPEG XR through Windows' own imaging codecs.

`.jxr` is what an HDR screenshot is on Windows — Xbox Game Bar and NVIDIA's
capture both write it — and nothing in the Python stack opens one. OpenCV
returns None, Pillow does not recognise the format, and the packages that do
handle it would add tens of megabytes to a release whose whole shape is "ship
nothing you can avoid shipping".

Windows already has the codec. `WindowsCodecs.dll` has decoded and encoded
JPEG XR since Vista, it is present on every machine this app runs on, and WIC
is a COM API, which ctypes can call. So this module is a few hundred lines of
vtable arithmetic instead of a dependency.

**Everything here is linear.** WIC defines `128bppRGBAFloat` as scRGB: linear
light, sRGB primaries, 1.0 at diffuse white and values above it for anything
brighter. Converting *into* that format applies whatever transfer curve the
source used, so an ordinary 8-bit JXR arrives linearised and an HDR one arrives
with its highlights intact. That is exactly the space the rest of the pipeline
wants, so no transfer curve is applied on either side of this module.

The vtable indices below are counted from the interface declarations in
`wincodec.h`, IUnknown's three methods first. They are the one thing here that
cannot be checked at runtime - a wrong index calls the wrong function with the
wrong arguments - so they are named rather than written inline.
"""

from __future__ import annotations

import ctypes
from ctypes import POINTER, byref, c_double, c_int, c_uint32, c_void_p, c_wchar_p
from pathlib import Path

import numpy as np

#: Containers the JPEG XR codec answers to. `.wdp` and `.hdp` are the older
#: Windows Media Photo / HD Photo extensions for the same bitstream.
SUFFIXES = frozenset({".jxr", ".wdp", ".hdp"})

_S_OK = 0
_GENERIC_WRITE = 0x40000000
_GENERIC_READ = 0x80000000
_CLSCTX_INPROC_SERVER = 1
_COINIT_MULTITHREADED = 0
_RPC_E_CHANGED_MODE = 0x80010106

# -- vtable indices ----------------------------------------------------------
# IUnknown occupies 0..2 in every one of these.

_FACTORY_CREATE_DECODER_FROM_FILENAME = 3
_FACTORY_CREATE_ENCODER = 8
_FACTORY_CREATE_FORMAT_CONVERTER = 10
_FACTORY_CREATE_STREAM = 14

_DECODER_GET_FRAME = 13

_SOURCE_GET_SIZE = 3
_SOURCE_COPY_PIXELS = 7

_CONVERTER_INITIALIZE = 8

# IWICStream extends IStream, whose own methods run to 13.
_STREAM_INITIALIZE_FROM_FILENAME = 15

_ENCODER_INITIALIZE = 3
_ENCODER_CREATE_NEW_FRAME = 10
_ENCODER_COMMIT = 11

_FRAME_INITIALIZE = 3
_FRAME_SET_SIZE = 4
_FRAME_SET_PIXEL_FORMAT = 6
_FRAME_WRITE_PIXELS = 10
_FRAME_COMMIT = 12

_RELEASE = 2


class WicError(RuntimeError):
    """A WIC call failed. Carries the HRESULT, which is the whole diagnosis."""

    def __init__(self, what: str, hresult: int) -> None:
        self.hresult = hresult & 0xFFFFFFFF
        super().__init__(f"{what} failed (0x{self.hresult:08X})")


class _GUID(ctypes.Structure):
    _fields_ = [
        ("data1", ctypes.c_uint32),
        ("data2", ctypes.c_uint16),
        ("data3", ctypes.c_uint16),
        ("data4", ctypes.c_ubyte * 8),
    ]


def _guid(text: str) -> _GUID:
    value = _GUID()
    hresult = ctypes.windll.ole32.CLSIDFromString(c_wchar_p("{" + text + "}"), byref(value))
    if hresult != _S_OK:
        raise WicError(f"parsing GUID {text}", hresult)
    return value


_CLSID_FACTORY = _guid("cacaf262-9370-4615-a13b-9f5539da4c0a")
_IID_FACTORY = _guid("ec5ec8a9-c395-4314-9c77-54d7a935ff70")
#: scRGB: linear light, sRGB primaries, unbounded above 1.0.
_FMT_RGBA_FLOAT = _guid("6fddc324-4e03-4bfe-b185-3d77768dc919")
_CONTAINER_WMP = _guid("57a37caa-367a-4540-916b-f183c5093a4b")


def _invoke(obj: c_void_p, index: int, *signature):
    """Bind vtable slot `index` on `obj`. `signature` is the argument types.

    The return type is a plain int rather than `ctypes.HRESULT`. HRESULT makes
    ctypes raise an OSError of its own the moment a call fails, which sounds
    helpful and is not: it happens before `_check` ever sees the code, so the
    message loses which operation failed and callers have two exception types
    to catch instead of one.
    """
    vtable = ctypes.cast(obj, POINTER(POINTER(c_void_p))).contents
    prototype = ctypes.WINFUNCTYPE(ctypes.c_int32, c_void_p, *signature)
    return prototype(vtable[index])


def _check(hresult: int, what: str) -> None:
    # Negative is failure; S_FALSE (1) is a success nothing here treats
    # specially.
    if hresult < 0:
        raise WicError(what, hresult)


def _release(obj: c_void_p | None) -> None:
    if obj:
        try:
            _invoke(obj, _RELEASE)(obj)
        except Exception:  # noqa: BLE001 - releasing must never mask a real error
            pass


def _initialise_com() -> None:
    """Join the COM apartment, tolerating a thread that is already in one.

    The GUI calls this from a worker thread, and Qt may have put that thread in
    a single-threaded apartment already. RPC_E_CHANGED_MODE means exactly that
    and is not a failure - WIC works either way.
    """
    hresult = ctypes.windll.ole32.CoInitializeEx(None, _COINIT_MULTITHREADED)
    if (hresult & 0xFFFFFFFF) not in (_S_OK, 1, _RPC_E_CHANGED_MODE):
        raise WicError("CoInitializeEx", hresult)


def _factory() -> c_void_p:
    _initialise_com()
    instance = c_void_p()
    hresult = ctypes.windll.ole32.CoCreateInstance(
        byref(_CLSID_FACTORY), None, _CLSCTX_INPROC_SERVER, byref(_IID_FACTORY), byref(instance)
    )
    _check(hresult, "creating the WIC imaging factory")
    return instance


def available() -> bool:
    """Whether Windows' imaging codecs can be reached at all."""
    try:
        factory = _factory()
    except Exception:  # noqa: BLE001 - a missing codec must degrade, not raise
        return False
    _release(factory)
    return True


def handles(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUFFIXES


def read(path: str | Path) -> np.ndarray:
    """Decode an image to linear scRGB float32 RGB.

    Values above 1.0 are real and are preserved; that is the entire point.
    """
    filename = str(Path(path))
    factory = decoder = frame = converter = None
    try:
        factory = _factory()

        decoder = c_void_p()
        _check(
            _invoke(
                factory, _FACTORY_CREATE_DECODER_FROM_FILENAME,
                c_wchar_p, c_void_p, ctypes.c_uint32, c_int, POINTER(c_void_p),
            )(factory, filename, None, _GENERIC_READ, 0, byref(decoder)),
            f"opening {Path(path).name}",
        )

        frame = c_void_p()
        _check(
            _invoke(decoder, _DECODER_GET_FRAME, c_uint32, POINTER(c_void_p))(
                decoder, 0, byref(frame)
            ),
            "reading the first frame",
        )

        width, height = c_uint32(), c_uint32()
        _check(
            _invoke(frame, _SOURCE_GET_SIZE, POINTER(c_uint32), POINTER(c_uint32))(
                frame, byref(width), byref(height)
            ),
            "reading the image size",
        )
        if not width.value or not height.value:
            raise WicError("the image has no pixels", _S_OK)

        # Convert rather than demand: the source may be 8-bit BGR, 16-bit, or
        # already half-float, and the converter applies the right transfer
        # curve for each on its way to linear.
        converter = c_void_p()
        _check(
            _invoke(factory, _FACTORY_CREATE_FORMAT_CONVERTER, POINTER(c_void_p))(
                factory, byref(converter)
            ),
            "creating a format converter",
        )
        _check(
            _invoke(
                converter, _CONVERTER_INITIALIZE,
                c_void_p, POINTER(_GUID), c_int, c_void_p, c_double, c_int,
            )(converter, frame, byref(_FMT_RGBA_FLOAT), 0, None, 0.0, 0),
            "converting to linear float",
        )

        stride = width.value * 16  # four float32 channels
        size = stride * height.value
        buffer = (ctypes.c_ubyte * size)()
        _check(
            _invoke(
                converter, _SOURCE_COPY_PIXELS,
                c_void_p, c_uint32, c_uint32, c_void_p,
            )(converter, None, stride, size, buffer),
            "copying pixels",
        )

        pixels = np.frombuffer(bytes(buffer), np.float32).reshape(
            height.value, width.value, 4
        )
        # Drop alpha and copy: the ctypes buffer dies with this function.
        rgb = np.ascontiguousarray(pixels[:, :, :3])
        # A codec may hand back negatives for out-of-gamut colours. scRGB
        # allows them; nothing downstream does.
        return np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0).clip(0.0, None)
    finally:
        _release(converter)
        _release(frame)
        _release(decoder)
        _release(factory)


def write(path: str | Path, linear_rgb: np.ndarray) -> None:
    """Encode linear scRGB float RGB as JPEG XR.

    Stored as half-float, which is what the format carries and is plenty: it
    holds the full HDR range and the precision loss is far below the noise the
    neural pass leaves behind.
    """
    if linear_rgb.ndim != 3 or linear_rgb.shape[2] < 3:
        raise ValueError("Expected an RGB image.")
    height, width = linear_rgb.shape[:2]

    rgba = np.ones((height, width, 4), np.float32)
    rgba[:, :, :3] = np.nan_to_num(
        linear_rgb[:, :, :3].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    ).clip(0.0, None)
    payload = np.ascontiguousarray(rgba).tobytes()

    filename = str(Path(path))
    factory = stream = encoder = frame = None
    properties = c_void_p()
    try:
        factory = _factory()

        stream = c_void_p()
        _check(
            _invoke(factory, _FACTORY_CREATE_STREAM, POINTER(c_void_p))(factory, byref(stream)),
            "creating a stream",
        )
        _check(
            _invoke(stream, _STREAM_INITIALIZE_FROM_FILENAME, c_wchar_p, ctypes.c_uint32)(
                stream, filename, _GENERIC_WRITE
            ),
            f"creating {Path(path).name}",
        )

        encoder = c_void_p()
        _check(
            _invoke(factory, _FACTORY_CREATE_ENCODER, POINTER(_GUID), c_void_p, POINTER(c_void_p))(
                factory, byref(_CONTAINER_WMP), None, byref(encoder)
            ),
            "creating the JPEG XR encoder",
        )
        _check(
            _invoke(encoder, _ENCODER_INITIALIZE, c_void_p, c_int)(encoder, stream, 2),
            "initialising the encoder",
        )

        frame = c_void_p()
        _check(
            _invoke(encoder, _ENCODER_CREATE_NEW_FRAME, POINTER(c_void_p), POINTER(c_void_p))(
                encoder, byref(frame), byref(properties)
            ),
            "creating a frame",
        )
        _check(
            _invoke(frame, _FRAME_INITIALIZE, c_void_p)(frame, properties),
            "initialising the frame",
        )
        _check(
            _invoke(frame, _FRAME_SET_SIZE, c_uint32, c_uint32)(frame, width, height),
            "setting the size",
        )

        # SetPixelFormat is in/out: the encoder writes back what it will
        # actually store. Asking for float and getting something else would
        # silently throw the HDR range away, so it is checked rather than
        # assumed.
        wanted = _GUID.from_buffer_copy(_FMT_RGBA_FLOAT)
        _check(
            _invoke(frame, _FRAME_SET_PIXEL_FORMAT, POINTER(_GUID))(frame, byref(wanted)),
            "setting the pixel format",
        )
        if bytes(wanted) != bytes(_FMT_RGBA_FLOAT):
            raise WicError("the encoder refused a floating-point pixel format", _S_OK)

        stride = width * 16
        _check(
            _invoke(frame, _FRAME_WRITE_PIXELS, c_uint32, c_uint32, c_uint32, c_void_p)(
                frame, height, stride, len(payload), payload
            ),
            "writing pixels",
        )
        _check(_invoke(frame, _FRAME_COMMIT)(frame), "committing the frame")
        _check(_invoke(encoder, _ENCODER_COMMIT)(encoder), "committing the file")
    finally:
        _release(properties)
        _release(frame)
        _release(encoder)
        _release(stream)
        _release(factory)
