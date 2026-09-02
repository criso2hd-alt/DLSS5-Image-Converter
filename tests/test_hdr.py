"""HDR sources: JPEG XR in, linear through the pipeline, HDR back out.

The thing worth guarding here is not the file format - it is that the range
survives. Every stage between the decoder and the encoder has a clip in it for
good reasons, and any one of them silently reintroduced turns an HDR conversion
back into an SDR one that still says HDR on the tin.
"""

from __future__ import annotations

import numpy as np
import pytest

from dlss5_converter import contract, grade, hdr, pipeline, resample, wic
from dlss5_converter.settings import GradeSettings

wic_available = pytest.mark.skipif(not wic.available(), reason="Windows imaging codecs")


def hdr_image(height: int = 8, width: int = 8) -> np.ndarray:
    """Linear RGB with a highlight well above diffuse white."""
    image = np.full((height, width, 3), 0.25, np.float32)
    image[:, width // 2 :, 0] = 8.0  # a specular highlight, 8x white
    return image


# -- the decoder -------------------------------------------------------------


@wic_available
def test_jpeg_xr_survives_a_round_trip(tmp_path):
    path = tmp_path / "shot.jxr"
    original = hdr_image()
    wic.write(path, original)
    decoded = wic.read(path)

    assert decoded.shape == original.shape
    assert decoded.dtype == np.float32
    # Half-float storage, so exactness is not on offer - the range is.
    assert decoded.max() == pytest.approx(8.0, rel=1e-3)
    assert np.allclose(decoded, original, rtol=1e-2, atol=1e-3)


@wic_available
def test_a_missing_file_fails_with_the_hresult(tmp_path):
    with pytest.raises(wic.WicError) as caught:
        wic.read(tmp_path / "nope.jxr")
    assert "0x" in str(caught.value), "the HRESULT is the whole diagnosis"


def test_only_the_hdr_extensions_are_treated_as_linear():
    assert hdr.is_hdr_source("a.jxr") and hdr.is_hdr_source("a.EXR")
    assert hdr.is_hdr_source("a.hdr") and hdr.is_hdr_source("a.wdp")
    assert not hdr.is_hdr_source("a.png")
    assert not hdr.is_hdr_source("a.jpg")


# -- loading -----------------------------------------------------------------


@wic_available
def test_loading_a_jxr_keeps_the_highlights(tmp_path):
    path = tmp_path / "shot.jxr"
    wic.write(path, hdr_image())

    loaded = contract.load_source(path)
    assert loaded.hdr
    assert loaded.linear.max() == pytest.approx(8.0, rel=1e-3)
    # The display copy is bounded, and is not merely the linear one clipped.
    assert loaded.rgb.max() <= 1.0
    assert loaded.white > 1.0


def test_loading_an_ordinary_png_is_unchanged(tmp_path):
    import cv2

    path = tmp_path / "plain.png"
    cv2.imwrite(str(path), np.full((8, 8, 3), 128, np.uint8))

    loaded = contract.load_source(path)
    assert not loaded.hdr
    assert loaded.rgb.max() <= 1.0
    assert loaded.white == 1.0
    # linear is the sRGB decode of rgb, which for mid grey is about 0.216.
    assert float(loaded.linear.mean()) == pytest.approx(0.2158, abs=0.002)


def test_an_hdr_container_holding_nothing_bright_is_not_hdr(tmp_path):
    """A .jxr of an ordinary photo should not pay for a tone map."""
    path = tmp_path / "sdr.exr"
    import cv2

    cv2.imwrite(str(path), np.full((8, 8, 3), 0.5, np.float32))
    loaded = contract.load_source(path)
    assert not loaded.hdr


def test_exr_is_read_as_linear_not_as_srgb(tmp_path):
    """EXR is linear by convention; treating it as encoded crushed it."""
    import cv2

    path = tmp_path / "render.exr"
    cv2.imwrite(str(path), np.full((8, 8, 3), 4.0, np.float32))

    loaded = contract.load_source(path)
    assert loaded.hdr
    assert float(loaded.linear.max()) == pytest.approx(4.0, rel=1e-3)


# -- tone mapping ------------------------------------------------------------


def test_the_white_point_maps_exactly_to_one():
    linear = np.full((4, 4, 3), 6.0, np.float32)
    mapped = hdr.tonemap(linear, white=6.0)
    assert float(mapped.max()) == pytest.approx(1.0, abs=1e-3)


def test_nothing_is_clipped_flat():
    """The failure people notice is a sky that goes uniformly white."""
    linear = np.zeros((1, 4, 3), np.float32)
    linear[0, :, :] = np.array([[1.0], [2.0], [4.0], [8.0]], np.float32)
    mapped = hdr.tonemap(linear, white=8.0)
    values = mapped[0, :, 0]
    assert np.all(np.diff(values) > 0), "brighter input must stay brighter output"


def test_black_stays_black_and_nothing_goes_nan():
    linear = np.zeros((4, 4, 3), np.float32)
    mapped = hdr.tonemap(linear, white=4.0)
    assert np.isfinite(mapped).all()
    assert float(mapped.max()) == 0.0


def test_one_hot_pixel_does_not_set_the_white_point():
    """A single specular dot at 300x would drag the whole image into the floor."""
    linear = np.full((64, 64, 3), 1.0, np.float32)
    linear[0, 0] = 300.0
    assert hdr.white_point(linear) < 10.0


def test_the_white_point_never_drops_below_one():
    assert hdr.white_point(np.full((8, 8, 3), 0.1, np.float32)) == 1.0


# -- grading and resampling stay unbounded -----------------------------------


def test_grading_linear_keeps_values_above_one():
    linear = hdr_image()
    graded = grade.apply_linear(linear, GradeSettings(exposure=1.0))
    assert float(graded.max()) == pytest.approx(16.0, rel=1e-3)


def test_resizing_linear_keeps_values_above_one():
    scaled = resample.resize_linear(hdr_image(64, 64), 128, 128)
    assert scaled.shape == (128, 128, 3)
    assert float(scaled.max()) > 7.0


def test_resizing_linear_at_native_size_is_a_no_op():
    image = hdr_image()
    assert resample.resize_linear(image, 8, 8) is image


# -- saving ------------------------------------------------------------------


@wic_available
def test_saving_hdr_to_jxr_keeps_the_range(tmp_path):
    path = tmp_path / "out.jxr"
    pipeline.save_image(hdr_image(), path, linear=True)
    assert wic.read(path).max() == pytest.approx(8.0, rel=1e-3)


def test_saving_hdr_to_exr_keeps_the_range(tmp_path):
    path = tmp_path / "out.exr"
    pipeline.save_image(hdr_image(), path, linear=True)

    import cv2

    back = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert back is not None
    assert float(back.max()) == pytest.approx(8.0, rel=1e-2)


def test_saving_hdr_to_png_tone_maps_rather_than_clips(tmp_path):
    """Clipping is what turns a bright sky into a white shape."""
    import cv2

    path = tmp_path / "out.png"
    linear = np.zeros((1, 4, 3), np.float32)
    linear[0, :, :] = np.array([[1.0], [2.0], [4.0], [8.0]], np.float32)
    pipeline.save_image(linear, path, linear=True)

    back = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    reds = back[0, :, 2].astype(int)  # BGR on the way back
    assert len(set(reds)) == 4, "every input level must remain distinguishable"


@wic_available
def test_saving_an_sdr_image_as_jxr_encodes_it_to_linear(tmp_path):
    """JPEG XR stores scRGB, so sRGB values cannot be written straight in."""
    path = tmp_path / "out.jxr"
    pipeline.save_image(np.full((8, 8, 3), 0.5, np.float32), path)
    # sRGB 0.5 is linear 0.214.
    assert float(wic.read(path).mean()) == pytest.approx(0.214, abs=0.005)


def test_output_names_follow_the_source_format(tmp_path):
    assert pipeline.hdr_output_path(tmp_path, "shot", tmp_path / "shot.jxr").suffix == ".jxr"
    assert pipeline.hdr_output_path(tmp_path, "shot", tmp_path / "shot.png").suffix == ".png"
