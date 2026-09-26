"""Multi-shot reconstruction, checked against a scene whose answer we know.

Real photo-mode captures cannot be committed to the repo and cannot be checked
objectively anyway. So the scene here is synthetic: two textured planes at known
depths, photographed by cameras at known positions. Every number the pipeline
recovers has a right answer to compare against, including the one that worries
me most, the scale carried from one pair to the next.
"""

from __future__ import annotations

import numpy as np
import pytest

from dlss5_converter import multiview as mv


WIDTH, HEIGHT = 960, 540
NEAR_Z, FAR_Z = 5.0, 12.0
CAMERA = mv.intrinsics(WIDTH, HEIGHT, 70.0)


def _texture(seed: int, size: int = 1024) -> np.ndarray:
    """Dense non-repeating detail, so SIFT has honest work to do."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 255, (size // 8, size // 8, 3), dtype=np.uint8)
    import cv2

    return cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC)


def _plane_homography(centre: np.ndarray, depth: float, half: float) -> np.ndarray:
    """Image-from-texture mapping for a fronto-parallel plane at `depth`.

    The plane spans world X and Y in [-half, half]; texture pixel coordinates
    are mapped onto it linearly, so a plain homography places it in any camera.
    """
    scale = 2.0 * half
    # Texture (u, v) in [0, 1] -> world (X, Y, depth).
    basis = np.array([[scale, 0.0, -half],
                      [0.0, scale, -half],
                      [0.0, 0.0, depth]])
    shift = np.array([[1.0, 0.0, -centre[0]],
                      [0.0, 1.0, -centre[1]],
                      [0.0, 0.0, 1.0]])
    translated = basis.copy()
    translated[0, 2] -= centre[0]
    translated[1, 2] -= centre[1]
    translated[2, 2] -= centre[2]
    del shift
    return CAMERA @ translated


def _render(centre: np.ndarray):
    """One synthetic capture plus the depth map a perfect estimator would give."""
    import cv2

    far = cv2.warpPerspective(
        _texture(1), _plane_homography(centre, FAR_Z, 14.0) @ _unit(1024),
        (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    near_texture = _texture(2, 512)
    homography = _plane_homography(centre, NEAR_Z, 1.6) @ _unit(512)
    near = cv2.warpPerspective(near_texture, homography, (WIDTH, HEIGHT),
                               flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((512, 512), 255, np.uint8), homography,
                               (WIDTH, HEIGHT), flags=cv2.INTER_NEAREST) > 0

    image = far.copy()
    image[mask] = near[mask]
    depth = np.full((HEIGHT, WIDTH), FAR_Z, np.float32)
    depth[mask] = NEAR_Z
    return image, depth


def _unit(size: int) -> np.ndarray:
    """Pixels to the unit square the plane mapping expects."""
    return np.array([[1.0 / size, 0.0, 0.0],
                     [0.0, 1.0 / size, 0.0],
                     [0.0, 0.0, 1.0]])


def _disparity(depth: np.ndarray, seed: int) -> np.ndarray:
    """Depth Anything's output shape: inverse depth through an unknown affine.

    Each shot gets a *different* affine, which is the whole difficulty. If the
    pipeline only worked when every shot shared one scale, it would not work on
    anything real.
    """
    rng = np.random.default_rng(seed)
    gain, bias = rng.uniform(2.0, 6.0), rng.uniform(-0.2, 0.2)
    return (gain / depth + bias).astype(np.float32)


def _shots(centres, with_depth=True):
    made = []
    for index, centre in enumerate(centres):
        image, depth = _render(np.asarray(centre, float))
        disparity = _disparity(depth, index) if with_depth else None
        made.append(mv.Shot(f"shot{index}", image, disparity))
    return made


# --- the pieces, checked on their own -------------------------------------

def test_intrinsics_centre_and_focal():
    camera = mv.intrinsics(1920, 1080, 90.0)
    assert camera[0, 2] == 960 and camera[1, 2] == 540
    assert camera[0, 0] == pytest.approx(960.0, rel=1e-6)   # tan(45) = 1


def test_disparity_fit_recovers_the_affine():
    distance = np.linspace(2.0, 30.0, 200)
    fit = mv.fit_disparity(3.7 / distance - 0.11, distance)
    assert fit is not None
    assert fit[0] == pytest.approx(3.7, rel=1e-6)
    assert fit[1] == pytest.approx(-0.11, abs=1e-6)


def test_apply_fit_inverts_it():
    distance = np.array([2.0, 5.0, 20.0])
    disparity = 4.0 / distance + 0.05
    assert mv.apply_fit(disparity, (4.0, 0.05)) == pytest.approx(distance, rel=1e-5)


def test_apply_fit_sends_sky_far_away():
    """Anything at or past the fitted horizon must not land behind the camera."""
    out = mv.apply_fit(np.array([0.05, 0.0, -0.3], np.float32), (4.0, 0.2), far=1e4)
    assert np.all(out > 0)
    assert np.all(out[1:] >= 1e4 - 1)


def test_parallax_is_zero_without_movement():
    points = np.random.default_rng(0).normal(size=(50, 3)) + [0, 0, 10]
    assert mv.parallax_degrees(points, np.zeros(3), np.zeros(3)) == pytest.approx(0.0)


def test_parallax_grows_with_baseline():
    points = np.random.default_rng(0).normal(size=(50, 3)) + [0, 0, 10]
    near = mv.parallax_degrees(points, np.zeros(3), np.array([0.2, 0, 0]))
    far = mv.parallax_degrees(points, np.zeros(3), np.array([2.0, 0, 0]))
    assert 0 < near < far


# --- the whole chain ------------------------------------------------------

def test_two_shots_recover_the_baseline_direction():
    shots = _shots([(0, 0, 0), (0.5, 0, 0)])
    reports = mv.solve(shots, CAMERA)

    assert len(reports) == 1 and reports[0].ok, reports[0].note
    assert reports[0].inliers >= mv.MIN_INLIERS
    assert reports[0].parallax > mv.MIN_PARALLAX
    # The camera stepped along +X and did not turn.
    direction = shots[1].centre / np.linalg.norm(shots[1].centre)
    assert direction[0] == pytest.approx(1.0, abs=0.05)
    assert np.allclose(shots[1].rotation, np.eye(3), atol=0.02)


def test_scale_carries_across_three_shots():
    """The point of the whole design: pair two is measured, not re-guessed.

    Equal real steps must come back as equal recovered steps, whatever unit the
    first pair happened to fix.
    """
    shots = _shots([(0, 0, 0), (0.5, 0, 0), (1.0, 0, 0)])
    reports = mv.solve(shots, CAMERA)

    assert all(report.ok for report in reports), [r.note for r in reports]
    first_step = np.linalg.norm(shots[1].centre - shots[0].centre)
    second_step = np.linalg.norm(shots[2].centre - shots[1].centre)
    assert second_step == pytest.approx(first_step, rel=0.2)


def test_fitted_depth_matches_the_real_geometry():
    """Depth is only useful fused if every shot ends up in the same units."""
    shots = _shots([(0, 0, 0), (0.5, 0, 0)])
    mv.solve(shots, CAMERA)

    # World unit = the first baseline (0.5 real), so the near plane at 5.0
    # should come back at about 10.
    depth = shots[0].depth
    assert depth is not None
    near = np.median(depth[depth < np.median(depth)])
    assert near == pytest.approx(NEAR_Z / 0.5, rel=0.15)


def test_a_camera_that_only_turns_is_rejected():
    """The failure mode that would otherwise produce a confident wrong scene."""
    shots = _shots([(0, 0, 0), (0.0005, 0, 0)])
    reports = mv.solve(shots, CAMERA)

    assert not reports[0].ok
    assert "barely moved" in reports[0].note
    assert any("barely moved" in note or "placed" in note
               for note in mv.diagnose(reports, shots))


def test_unrelated_shots_are_rejected():
    shots = _shots([(0, 0, 0)])
    other, depth = _render(np.array([0.0, 0.0, 0.0]))
    rng = np.random.default_rng(9)
    shots.append(mv.Shot("elsewhere", rng.integers(0, 255, other.shape, dtype=np.uint8),
                         _disparity(depth, 7)))
    reports = mv.solve(shots, CAMERA)
    assert not reports[0].ok


def test_fuse_merges_both_shots_into_one_cloud():
    shots = _shots([(0, 0, 0), (0.5, 0, 0)])
    mv.solve(shots, CAMERA)
    points, colours = mv.fuse(shots, CAMERA, stride=8)

    assert len(points) == len(colours) > 1000
    assert np.isfinite(points).all()
    # Both cameras look down +Z, so nothing should end up behind them.
    assert (points[:, 2] > 0).mean() > 0.99


def test_voxel_merge_collapses_the_duplicate_surfaces():
    shots = _shots([(0, 0, 0), (0.5, 0, 0)])
    mv.solve(shots, CAMERA)
    loose, _ = mv.fuse(shots, CAMERA, stride=8)
    merged, _ = mv.fuse(shots, CAMERA, stride=8, voxel=0.25)
    assert 0 < len(merged) < len(loose)


def test_diagnose_speaks_to_the_user_not_the_log():
    shots = _shots([(0, 0, 0), (0.0005, 0, 0)])
    notes = mv.diagnose(mv.solve(shots, CAMERA), shots)
    assert notes and all(note[0].isupper() or ":" in note for note in notes)


def test_single_shot_is_not_a_scene():
    assert mv.diagnose([], _shots([(0, 0, 0)])) == [
        "Add more shots: one image cannot show the parts it hides."]


def test_write_ply_round_trips(tmp_path):
    path = tmp_path / "cloud.ply"
    points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    mv.write_ply(path, points, np.array([[1, 2, 3], [4, 5, 6]], np.uint8))
    text = path.read_text(encoding="ascii")
    assert "element vertex 2" in text
    assert text.strip().endswith("3.00000 4.00000 5.00000 4 5 6")


def test_points_no_other_shot_sees_are_exempt_from_the_tier_cut():
    """Beyond the hard cut everywhere, only unopposed coverage may survive."""
    shots = _shots([(0, 0, 0), (0.5, 0, 0)])
    mv.solve(shots, CAMERA)
    loose, _ = mv.fuse(shots, CAMERA, stride=8, consistent=True)
    for shot in shots:
        shot.required = np.full(shot.image.shape[:2], np.nan, np.float32)
    cut, _ = mv.fuse(shots, CAMERA, stride=8, consistent=True)
    assert 0 < len(cut) < len(loose)
