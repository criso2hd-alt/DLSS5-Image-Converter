"""The numpy bundle adjuster, checked where the right answer is known."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from dlss5_converter import bundle


def _scene(seed=0, cameras=5, atoms=300, focal=900.0):
    rng = np.random.default_rng(seed)
    points = rng.uniform([-4, -2, 6], [4, 2, 14], (atoms, 3))
    rotations, translations = [], []
    for i in range(cameras):
        centre = np.array([i * 0.6 - 1.2, rng.normal(0, 0.05), rng.normal(0, 0.1)])
        rotation = cv2.Rodrigues(np.array([0.0, (i - 2) * 0.05, 0.0]))[0]
        rotations.append(cv2.Rodrigues(rotation)[0].ravel())
        translations.append(-rotation @ centre)
    rotations, translations = np.array(rotations), np.array(translations)
    cam_index = np.repeat(np.arange(cameras), atoms)
    pt_index = np.tile(np.arange(atoms), cameras)
    observed = bundle.project(rotations[cam_index], translations[cam_index],
                              points[pt_index], focal, (800.0, 333.0))
    return rotations, translations, points, cam_index, pt_index, observed


def test_rodrigues_matches_opencv():
    vectors = np.random.default_rng(1).normal(size=(20, 3))
    ours = bundle.rodrigues(vectors)
    for vector, matrix in zip(vectors, ours):
        assert np.allclose(matrix, cv2.Rodrigues(vector)[0], atol=1e-10)


def test_rodrigues_handles_zero_rotation():
    assert np.allclose(bundle.rodrigues(np.zeros((1, 3)))[0], np.eye(3))


def test_perfect_scene_has_zero_error():
    rotations, translations, points, c, p, observed = _scene()
    problem = bundle.Problem(rotations, translations, points, 900.0, (800.0, 333.0),
                             c, p, observed)
    assert np.abs(bundle.residuals(problem)).max() < 1e-9


def test_recovers_from_scrambled_cameras_points_and_focal():
    """Chained pair poses come in this wrong; the adjuster must pull them in."""
    rotations, translations, points, c, p, observed = _scene()
    rng = np.random.default_rng(5)
    bad_rotations = rotations + rng.normal(0, 0.01, rotations.shape)
    bad_translations = translations + rng.normal(0, 0.05, translations.shape)
    bad_rotations[0], bad_translations[0] = rotations[0], translations[0]   # anchor
    problem = bundle.Problem(bad_rotations, bad_translations,
                             points + rng.normal(0, 0.15, points.shape),
                             820.0, (800.0, 333.0), c, p, observed)
    before = np.median(np.linalg.norm(bundle.residuals(problem), axis=1))

    result = bundle.adjust(problem, iterations=60)

    assert before > 5.0
    assert result["median_px"] < 0.5
    assert result["focal"] == pytest.approx(900.0, rel=0.03)


def test_wrong_matches_do_not_bend_the_scene():
    """Huber weighting: 10% garbage sightings must not drag the answer."""
    rotations, translations, points, c, p, observed = _scene(seed=2)
    rng = np.random.default_rng(3)
    corrupted = observed.copy()
    bad = rng.choice(len(observed), len(observed) // 10, replace=False)
    corrupted[bad] += rng.uniform(-80, 80, (len(bad), 2))
    problem = bundle.Problem(rotations + rng.normal(0, 0.005, rotations.shape),
                             translations.copy(), points.copy(), 900.0, (800.0, 333.0),
                             c, p, corrupted)
    problem.rotations[0] = rotations[0]
    bundle.adjust(problem, iterations=60)

    clean = np.setdiff1d(np.arange(len(observed)), bad)
    error = np.linalg.norm(bundle.residuals(problem)[clean], axis=1)
    assert np.median(error) < 1.0


def test_anchor_camera_does_not_move():
    rotations, translations, points, c, p, observed = _scene(seed=4)
    problem = bundle.Problem(rotations + 0.01, translations + 0.05, points.copy(), 900.0,
                             (800.0, 333.0), c, p, observed)
    anchor_r, anchor_t = problem.rotations[0].copy(), problem.translations[0].copy()
    bundle.adjust(problem, iterations=10)
    assert np.allclose(problem.rotations[0], anchor_r)
    assert np.allclose(problem.translations[0], anchor_t)


def test_full_lens_is_recovered_from_a_pinhole_guess():
    """Game lenses at wide angles are not a one-focal pinhole: the corners give
    it away. Starting from a plain pinhole, the lens model must converge."""
    rotations, translations, points, c, p, _ = _scene(seed=6, cameras=6, atoms=400)
    true_lens = dict(focal=900.0, focal_y=880.0, centre=(812.0, 326.0), distortion=(-0.08, 0.01))
    observed = bundle.project(rotations[c], translations[c], points[p], true_lens["focal"],
                              true_lens["centre"], true_lens["distortion"], true_lens["focal_y"])
    problem = bundle.Problem(rotations.copy(), translations.copy(), points.copy(), 890.0,
                             (800.0, 333.0), c, p, observed)
    focal_only = bundle.Problem(rotations.copy(), translations.copy(), points.copy(), 890.0,
                                (800.0, 333.0), c, p, observed)
    bundle.adjust(focal_only, iterations=60)
    result = bundle.adjust(problem, iterations=80, lens=True)

    assert result["median_px"] < 0.1
    assert np.median(np.linalg.norm(bundle.residuals(focal_only), axis=1)) > 5 * result["median_px"]
    assert problem.focal == pytest.approx(900.0, rel=0.02)
    assert problem.focal_y == pytest.approx(880.0, rel=0.02)
    assert problem.distortion[0] == pytest.approx(-0.08, abs=0.02)


def test_default_mode_still_keeps_one_focal():
    rotations, translations, points, c, p, observed = _scene(seed=7)
    problem = bundle.Problem(rotations.copy(), translations.copy(), points.copy(), 850.0,
                             (800.0, 333.0), c, p, observed)
    bundle.adjust(problem, iterations=40)
    assert problem.focal_y is None and problem.distortion == (0.0, 0.0)
