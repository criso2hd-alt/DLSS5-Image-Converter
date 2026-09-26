"""Placing shots that carry the game's real depth."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from dlss5_converter import measured

W, H = 320, 180
CAMERA = np.array([[160.0, 0, W / 2], [0, 160.0, H / 2], [0, 0, 1.0]])


def _rotation(yaw_degrees, pitch_degrees=0.0):
    return cv2.Rodrigues(np.radians([pitch_degrees, yaw_degrees, 0.0]))[0]


def test_kabsch_recovers_an_exact_motion():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(50, 3))
    truth = _rotation(30, 10)
    shift = np.array([0.4, -0.2, 1.0])
    rotation, translation = measured.kabsch(source, source @ truth.T + shift)
    assert np.allclose(rotation, truth, atol=1e-9)
    assert np.allclose(translation, shift, atol=1e-9)


def test_kabsch_never_returns_a_mirror():
    rng = np.random.default_rng(1)
    source = rng.normal(size=(30, 3))
    mirrored = source * np.array([-1.0, 1.0, 1.0])
    rotation, _ = measured.kabsch(source, mirrored)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_align_ignores_wrong_matches():
    rng = np.random.default_rng(2)
    source = rng.uniform([-3, -1, 4], [3, 1, 20], (200, 3))
    truth, shift = _rotation(-45), np.array([0.9, 0.0, 0.4])
    target = source @ truth.T + shift
    bad = rng.choice(200, 60, replace=False)
    target[bad] += rng.normal(0, 3.0, (60, 3))
    rotation, translation, inliers = measured.align(source, target, rng)
    assert np.allclose(rotation, truth, atol=1e-6)
    assert inliers.sum() >= 135


def _depth_of_box(rotation, centre):
    """Distance map of the inside of a room, seen from a camera pose.

    Floor, back wall and a side wall at different depths, so every degree of
    freedom is constrained, as in a real street scene.
    """
    ys, xs = np.mgrid[0:H, 0:W]
    rays = np.stack([(xs - W / 2) / 160.0, (ys - H / 2) / 160.0, np.ones_like(xs, float)], -1)
    world_rays = rays @ rotation          # camera -> world: R^T applied to row vectors
    planes = [(np.array([0, 1.0, 0]), 1.5),   # floor at y = 1.5 (image y points down)
              (np.array([0, 0, 1.0]), 12.0),  # back wall
              (np.array([1.0, 0, 0]), 4.0)]   # side wall
    best = np.full((H, W), np.inf)
    for normal, offset in planes:
        denom = world_rays @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (offset - centre @ normal) / denom
        best = np.where((t > 0) & (t < best), t, best)
    return np.where(np.isfinite(best), best, measured.SKY)   # ray scale is z, so t is z


def test_dense_refinement_tightens_a_rough_pose():
    """Atoms leave a few degrees and a few centimetres of error; pixels remove it."""
    first = _depth_of_box(np.eye(3), np.zeros(3))
    true_rotation = _rotation(8)
    true_centre = np.array([0.5, 0.0, 0.2])
    second = _depth_of_box(true_rotation, true_centre)
    # second-camera frame -> first-camera frame
    rotation, translation = true_rotation.T, true_centre
    rough_rotation = _rotation(2.5, 1.5) @ rotation
    rough_translation = translation + np.array([0.08, -0.05, 0.1])

    refined_r, refined_t, agree = measured.refine(rough_rotation, rough_translation,
                                                  second, first, CAMERA, stride=2)

    error = np.degrees(np.arccos(np.clip((np.trace(refined_r @ rotation.T) - 1) / 2, -1, 1)))
    assert error < 0.3
    assert np.linalg.norm(refined_t - translation) < 0.03
    assert agree > 0.9


def test_sky_is_never_lifted_to_a_point():
    z = np.full((H, W), measured.SKY)
    z[50, 60] = 5.0
    points, ok = measured.lift(np.array([[60.0, 50.0], [10.0, 10.0]]), z, CAMERA)
    assert ok.tolist() == [True, False]
    assert points[0, 2] == pytest.approx(5.0)


def test_distances_follow_reversed_z():
    shot = type("S", (), {})()
    shot.disparity = np.array([[0.5, 0.0]], np.float32)
    z = measured.distances(shot)
    assert z[0, 0] == pytest.approx(measured.UNIT / 0.5)
    assert z[0, 1] == measured.SKY
