from __future__ import annotations

import numpy as np
import pytest

from dlss5_converter import contract


def test_hardware_depth_keeps_reversed_z_orientation():
    """Near must stay near. Flipping this silently inverts the whole effect."""
    inverse = np.array([[0.0, 0.5, 1.0]], np.float32)
    depth = contract.to_hardware_depth(inverse)
    assert depth[0, 0] < depth[0, 1] < depth[0, 2]


def test_hardware_depth_never_emits_the_exact_far_plane():
    """A literal 0.0 reads as 'nothing rendered here' on some depth paths."""
    depth = contract.to_hardware_depth(np.zeros((4, 4), np.float32))
    assert np.all(depth > 0.0)
    depth = contract.to_hardware_depth(np.ones((4, 4), np.float32))
    assert np.all(depth < 1.0)


@pytest.mark.parametrize("contrast", [0.5, 1.0, 2.0, 3.0])
def test_depth_contrast_pins_the_endpoints(contrast):
    """The reversed-Z contract must hold wherever the user drags the slider."""
    inverse = np.linspace(0.0, 1.0, 32, dtype=np.float32).reshape(4, 8)
    depth = contract.to_hardware_depth(inverse, contrast)
    assert depth.min() == pytest.approx(0.0, abs=1e-3)
    assert depth.max() == pytest.approx(1.0, abs=1e-3)
    # Monotonic: contrast reshapes the distribution, it never reorders it.
    flat = depth.reshape(-1)
    assert np.all(np.diff(flat) >= -1e-6)


def test_srgb_round_trip():
    values = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    assert np.allclose(contract.linear_to_srgb(contract.srgb_to_linear(values)), values, atol=1e-5)


def test_jitter_sequence_is_centred_and_bounded():
    offsets = contract.jitter_sequence(16)
    assert len(offsets) == 16
    assert all(-0.5 <= x < 0.5 and -0.5 <= y < 0.5 for x, y in offsets)
    # Low-discrepancy, so the mean should sit near zero rather than drifting.
    mean_x = sum(x for x, _ in offsets) / len(offsets)
    assert abs(mean_x) < 0.1


def test_jitter_sequence_does_not_start_at_the_origin():
    """Starting at (0, 0) biases the accumulation towards the unjittered sample."""
    first = contract.jitter_sequence(8)[0]
    assert first != (0.0, 0.0)


def test_fit_to_budget_produces_even_dimensions():
    image = np.zeros((1001, 4097, 3), np.float32)
    fitted = contract.fit_to_budget(image, 3840)
    height, width = fitted.shape[:2]
    assert width <= 3840 and height <= 3840
    assert width % 2 == 0 and height % 2 == 0


def test_fit_to_budget_is_a_no_op_when_already_small():
    image = np.zeros((480, 640, 3), np.float32)
    assert contract.fit_to_budget(image, 3840) is image


def test_build_produces_the_formats_the_harness_expects():
    source = np.random.default_rng(0).random((32, 48, 3)).astype(np.float32)
    depth = np.random.default_rng(1).random((32, 48)).astype(np.float32)
    plan = contract.build(source, depth, frames=4)

    assert plan.colour.dtype == np.float16 and plan.colour.shape == (32, 48, 4)
    assert plan.depth.dtype == np.float32 and plan.depth.shape == (32, 48)
    assert plan.motion.dtype == np.float16 and plan.motion.shape == (32, 48, 2)
    assert plan.size == (48, 32)
    assert len(plan.jitter) == 4


def test_motion_vectors_are_exactly_zero():
    """Non-zero motion on a static frame makes DLSS discard its history."""
    source = np.zeros((8, 8, 3), np.float32)
    plan = contract.build(source, np.zeros((8, 8), np.float32), frames=2)
    assert not plan.motion.any()


def test_build_resizes_a_mismatched_depth_map():
    source = np.zeros((32, 32, 3), np.float32)
    plan = contract.build(source, np.zeros((16, 16), np.float32), frames=1)
    assert plan.depth.shape == (32, 32)


def test_planes_round_trip_through_disk(tmp_path):
    source = np.random.default_rng(2).random((16, 24, 3)).astype(np.float32)
    plan = contract.build(source, np.zeros((16, 24), np.float32), frames=1)
    written = contract.write_planes(plan, tmp_path)
    assert np.array_equal(
        np.fromfile(written["colour"], np.float16).reshape(16, 24, 4), plan.colour
    )
    assert np.array_equal(np.fromfile(written["depth"], np.float32).reshape(16, 24), plan.depth)


def test_read_output_rejects_a_size_mismatch(tmp_path):
    path = tmp_path / "out.bin"
    np.zeros((4, 4, 4), np.float16).tofile(path)
    with pytest.raises(ValueError, match="expected"):
        contract.read_output(path, 8, 8)


def test_shift_subpixel_is_identity_at_zero():
    image = np.random.default_rng(3).random((16, 16, 3)).astype(np.float32)
    assert contract.shift_subpixel(image, 0.0, 0.0) is image


def test_shift_subpixel_actually_moves_the_image():
    image = np.zeros((16, 16, 3), np.float32)
    image[8, 8] = 1.0
    shifted = contract.shift_subpixel(image, 0.5, 0.0)
    assert not np.allclose(shifted, image)
