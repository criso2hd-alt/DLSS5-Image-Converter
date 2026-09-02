"""Sequence discovery and renderer depth maps."""

from __future__ import annotations

# Imported first: dlss5_converter/__init__ enables OpenCV's OpenEXR codec, and
# it only takes effect if it runs before cv2 is imported anywhere.
import dlss5_converter  # noqa: F401
import cv2
import numpy as np
import pytest

from dlss5_converter.sequence import (
    describe,
    find_sequence,
    load_depth_map,
    split_frame_number,
)


def touch(folder, name):
    path = folder / name
    path.write_bytes(b"")
    return path


def test_frame_numbers_are_split_off_the_stem(tmp_path):
    assert split_frame_number(tmp_path / "shot_0042.png") == ("shot_", 42, 4)
    assert split_frame_number(tmp_path / "render1.png") == ("render", 1, 1)
    assert split_frame_number(tmp_path / "beauty.png") is None


def test_a_sequence_is_found_and_ordered_numerically(tmp_path):
    """Not alphabetically: frame 10 must not sort before frame 9."""
    for n in (1, 2, 9, 10, 11):
        touch(tmp_path, f"shot_{n:04d}.png")
    frames = find_sequence(tmp_path / "shot_0002.png")
    assert [f.name for f in frames] == [
        "shot_0001.png", "shot_0002.png", "shot_0009.png",
        "shot_0010.png", "shot_0011.png",
    ]


def test_padding_width_separates_two_renders_in_one_folder(tmp_path):
    touch(tmp_path, "shot_0001.png")
    touch(tmp_path, "shot_0002.png")
    touch(tmp_path, "shot_1.png")
    assert len(find_sequence(tmp_path / "shot_0001.png")) == 2
    assert len(find_sequence(tmp_path / "shot_1.png")) == 1


def test_other_names_and_extensions_are_not_swept_in(tmp_path):
    touch(tmp_path, "shot_0001.png")
    touch(tmp_path, "shot_0002.png")
    touch(tmp_path, "depth_0001.png")   # different prefix
    touch(tmp_path, "shot_0003.jpg")    # different extension
    assert len(find_sequence(tmp_path / "shot_0001.png")) == 2


def test_a_still_without_a_counter_is_a_one_frame_sequence(tmp_path):
    single = touch(tmp_path, "beauty.png")
    assert find_sequence(single) == [single]


def write_depth(path, values):
    cv2.imwrite(str(path), values)
    return path


def test_depth_is_normalised_with_near_at_one(tmp_path):
    """The convention the whole pipeline uses, and reversed-Z's."""
    ramp = np.linspace(0, 255, 64, dtype=np.uint8)[None, :].repeat(16, 0)
    path = write_depth(tmp_path / "d.png", ramp)
    depth = load_depth_map(path)
    assert depth.min() == pytest.approx(0.0)
    assert depth.max() == pytest.approx(1.0)
    # Brightest input stays nearest without inversion.
    assert depth[0, -1] > depth[0, 0]


def test_invert_flips_a_mist_pass(tmp_path):
    ramp = np.linspace(0, 255, 64, dtype=np.uint8)[None, :].repeat(16, 0)
    path = write_depth(tmp_path / "d.png", ramp)
    assert load_depth_map(path, invert=True)[0, 0] > load_depth_map(path, invert=True)[0, -1]


def test_sixteen_bit_depth_keeps_its_precision(tmp_path):
    ramp = np.linspace(0, 65535, 256, dtype=np.uint16)[None, :].repeat(4, 0)
    depth = load_depth_map(write_depth(tmp_path / "d.png", ramp))
    assert len(np.unique(depth)) > 200


def test_a_flat_depth_map_does_not_divide_by_zero(tmp_path):
    flat = np.full((8, 8), 128, np.uint8)
    depth = load_depth_map(write_depth(tmp_path / "d.png", flat))
    assert np.isfinite(depth).all()
    assert depth.shape == (8, 8)


def test_infinities_do_not_flatten_the_range(tmp_path):
    """Renderers write "no geometry" as inf; one pixel must not eat the range."""
    values = np.linspace(1.0, 10.0, 64, dtype=np.float32)[None, :].repeat(8, 0).copy()
    values[0, 0] = np.inf
    path = tmp_path / "d.exr"
    if not cv2.imwrite(str(path), values):
        pytest.skip("this OpenCV build cannot write EXR")
    depth = load_depth_map(path)
    assert np.isfinite(depth).all()
    assert depth.max() == pytest.approx(1.0)
    assert len(np.unique(depth)) > 10


def test_describe_reads_sensibly(tmp_path):
    assert describe([]) == "no frames"
    one = [tmp_path / "a_0001.png"]
    assert "1 frame" in describe(one)
    many = [tmp_path / "a_0001.png", tmp_path / "a_0009.png"]
    assert "2 frames" in describe(many)


def test_mismatched_depth_count_is_refused_before_any_gpu_work(tmp_path):
    """Caught up front, not 200 frames into a render.

    The check deliberately sits before runtime detection, so it raises on a
    machine with no DLSS runtime at all - which is also what makes it testable.
    """
    from dlss5_converter import pipeline
    from dlss5_converter.settings import AppSettings

    frames = [tmp_path / f"a_{i:04d}.png" for i in range(1, 5)]
    depth = [tmp_path / f"d_{i:04d}.png" for i in range(1, 3)]

    with pytest.raises(ValueError, match="one to one"):
        # A generator: nothing runs until it is iterated.
        list(
            pipeline.convert_sequence(
                frames, AppSettings(), None, tmp_path / "out", depth_frames=depth
            )
        )


def test_an_empty_sequence_is_a_no_op(tmp_path):
    from dlss5_converter import pipeline
    from dlss5_converter.settings import AppSettings

    assert list(pipeline.convert_sequence([], AppSettings(), None, tmp_path)) == []
