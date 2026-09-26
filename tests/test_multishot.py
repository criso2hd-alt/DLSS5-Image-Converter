"""Scene from shots: sessions, levelling, and the 3D tab's shots mode."""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pytest

from dlss5_converter import multishot
from dlss5_converter import multiview as mv


def _capture(folder, name, when, depth=True):
    """A capture as the add-on writes it, with a chosen file time."""
    image = folder / f"{name}.png"
    cv2.imwrite(str(image), np.full((20, 40, 3), 90, np.uint8))
    paths = [image]
    if depth:
        raw = np.full((20, 40), 0.01, np.float32)
        raw.astype("<f4").tofile(folder / f"{name}.depth.f32")
        (folder / f"{name}.depth.json").write_text(json.dumps(
            {"width": 40, "height": 20, "fraction_at_zero": 0.0, "fraction_at_one": 0.0}))
        paths += [folder / f"{name}.depth.f32", folder / f"{name}.depth.json"]
    for path in paths:
        os.utime(path, (when, when))
    return image


# --- sessions ---------------------------------------------------------------------

def test_shots_minutes_apart_are_separate_sessions(tmp_path):
    for n in range(3):
        _capture(tmp_path, f"Stray 2026-09-23 15-30-0{n}_{n}", 1_000_000 + n * 20)
    for n in range(2):
        _capture(tmp_path, f"Cyberpunk2077 2026-09-23 10-00-0{n}_{n}", 1_000_000 - 3600 + n * 20,
                 depth=False)
    sessions = multishot.find_sessions(tmp_path)
    assert [len(s.paths) for s in sessions] == [3, 2]            # newest first
    assert sessions[0].game == "Stray"
    assert sessions[0].with_depth == 3 and sessions[1].with_depth == 0
    assert "3 shots, 3 with game depth" in sessions[0].label
    assert "no game depth" in sessions[1].label


def test_sidecars_are_not_mistaken_for_shots(tmp_path):
    _capture(tmp_path, "Stray 2026-09-23 15-30-00_1", 1_000_000)
    sessions = multishot.find_sessions(tmp_path)
    assert len(sessions) == 1 and len(sessions[0].paths) == 1


def test_empty_folder_has_no_sessions(tmp_path):
    assert multishot.find_sessions(tmp_path) == []


# --- levelling ----------------------------------------------------------------------

def _ring(count=8, radius=2.0, pitch_degrees=35.0, tilt=None):
    """Cameras orbiting a subject at the origin, looking down at it.

    `tilt` rotates the whole capture, as the first shot's arbitrary frame does.
    """
    shots = []
    tilt = np.eye(3) if tilt is None else tilt
    for k in range(count):
        a = 2 * np.pi * k / count
        centre = np.array([radius * np.sin(a), -radius * np.tan(np.radians(pitch_degrees)) * 0.5,
                           -radius * np.cos(a)])   # y down: cameras sit above the subject
        forward = -centre / np.linalg.norm(centre)
        right = np.cross(np.array([0.0, 1.0, 0.0]), forward)   # horizontal: no roll
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack([right, down, forward]) @ tilt.T
        shot = mv.Shot(f"s{k}", np.zeros((10, 10, 3), np.uint8))
        shot.rotation = rotation
        shot.centre = tilt @ centre
        shot.depth = np.full((10, 10), radius, np.float32)
        shots.append(shot)
    # The world is the first shot's camera, as the solvers produce it.
    root_r, root_c = shots[0].rotation.copy(), shots[0].centre.copy()
    for shot in shots:
        shot.centre = root_r @ (shot.centre - root_c)
        shot.rotation = shot.rotation @ root_r.T
    return shots


def test_level_finds_true_up_under_a_tilted_first_shot():
    shots = _ring()
    axes, _, _ = multishot._level(shots)
    ups = [s.rotation.T @ np.array([0.0, -1.0, 0.0]) for s in shots]
    rights = [s.rotation.T @ np.array([1.0, 0.0, 0.0]) for s in shots]
    # Every camera's right vector is horizontal in the levelled frame...
    for right in rights:
        assert abs(axes[1] @ right) < 1e-6
    # ...and up points the way the cameras' tops do on average.
    assert axes[1] @ np.mean(ups, axis=0) > 0


def test_level_puts_the_subject_at_the_target():
    shots = _ring()
    _, target, distance = multishot._level(shots)
    # The subject sits where all view lines meet; in the root frame that is
    # straight ahead of the root camera at the orbit radius.
    root_forward = shots[0].rotation.T @ np.array([0.0, 0.0, 1.0])
    expected = shots[0].centre + root_forward * np.linalg.norm(
        np.array([2.0, 2.0 * np.tan(np.radians(35.0)) * 0.5, 0.0]))
    assert target == pytest.approx(expected, abs=1e-6)
    assert distance == pytest.approx(np.linalg.norm(expected - shots[0].centre), rel=1e-6)


def test_level_axes_are_a_rotation():
    axes, _, _ = multishot._level(_ring())
    assert axes @ axes.T == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(axes) == pytest.approx(1.0)


# --- the 3D tab in shots mode -----------------------------------------------------------

@pytest.fixture(scope="module")
def qt_app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def page(qt_app):
    from dlss5_converter.creative_page import CreativePage

    p = CreativePage()
    yield p
    # Stop the page's worker thread: a page dropped with it running crashes
    # Qt later in the session.
    p.shutdown()
    p.deleteLater()
    qt_app.processEvents()


def test_switching_source_swaps_the_rail_and_strip(page):
    page.show()
    assert page.scene_card.isVisible() and not page.shots_card.isVisible()
    page.mode_switch.set_index(1)
    page._mode_changed(1)
    assert page.shots_card.isVisible() and not page.scene_card.isVisible()
    assert page.shot_strip.isVisible()
    assert "folder of captures" in page.preview.text()
    page._mode_changed(0)
    assert page.scene_card.isVisible() and not page.shot_strip.isVisible()


def test_a_single_image_arriving_does_not_disturb_shots_mode(page):
    page._mode_changed(1)
    page.set_source(np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8), np.float32))
    assert page._image_rgb8 is not None          # remembered for One image
    assert page._rgb8 is None                    # but shots mode is untouched
    assert "folder of captures" in page.preview.text()


def test_loading_a_folder_fills_the_strip_and_left_out_shots_count(page, tmp_path):
    for n in range(4):
        _capture(tmp_path, f"Stray 2026-09-23 15-30-0{n}_{n + 1}", 1_000_000 + n * 20)
    page._mode_changed(1)
    page.load_shots_folder(str(tmp_path))
    assert page.shot_strip.count() == 4
    assert "4 shots, 4 with game depth" in page.shots_info.text()
    assert page.build_shots_button.isEnabled()

    page._toggle_shot(page.shot_strip.item(0))
    assert "3 shots (1 left out)" in page.shots_info.text()
    assert "Excluded" in page.shot_strip.item(0).text()
    page._toggle_shot(page.shot_strip.item(0))
    assert "Excluded" not in page.shot_strip.item(0).text()


def test_one_shot_left_is_not_enough_to_build(page, tmp_path):
    _capture(tmp_path, "Stray 2026-09-23 15-30-00_1", 1_000_000)
    page._mode_changed(1)
    page.load_shots_folder(str(tmp_path))
    assert not page.build_shots_button.isEnabled()


def test_report_badges_say_why(page, tmp_path):
    paths = [_capture(tmp_path, f"Stray 2026-09-23 15-30-0{n}_{n + 1}", 1_000_000 + n * 20)
             for n in range(3)]
    page._mode_changed(1)
    page.load_shots_folder(str(tmp_path))
    # A queued job for this session, as Build scene makes one (without
    # starting a real build on the worker).
    page._jobs.append({"id": 7, "label": "test", "status": "Queued",
                       "paths": {str(p) for p in paths}})
    page._on_shots_started(7)
    report = multishot.Report(statuses={str(paths[0]): multishot.PLACED,
                                        str(paths[1]): multishot.PLACED,
                                        str(paths[2]): multishot.NO_OVERLAP},
                              fov_degrees=101.0, agreement=0.91)
    page._on_shots_done(None, report, 7)
    assert "Placed" in page.shot_strip.item(0).text()
    assert "No overlap" in page.shot_strip.item(2).text()
    assert "Placed 2 of 3 shots, 91% agreement" in page.shots_result.text()
    assert "101°" in page.shots_result.text()
    assert page.build_shots_button.text() == "Build scene"
    assert not page._jobs and not page.cancel_shots_button.isVisibleTo(page)


# --- saved scenes ---------------------------------------------------------------------------

def _tiny_scene():
    from dlss5_converter import splat3d

    n = 50
    rng = np.random.default_rng(0)
    cov = np.tile(np.array([1e-4, 0, 0, 1e-4, 0, 1e-4], np.float32), (n, 1))
    return splat3d.SplatScene(rng.normal(size=(n, 3)).astype(np.float32),
                              rng.uniform(size=(n, 3)).astype(np.float32),
                              np.full(n, 0.8, np.float32), cov, n, focal=640.0,
                              planes=[(np.array([0.0, 1.0, 0.0]), -1.5)])


def test_saved_scene_round_trips(tmp_path):
    scene = _tiny_scene()
    report = multishot.Report(statuses={"a": multishot.PLACED, "b": multishot.PLACED},
                              sharp=True, key_shots=1,
                              root_image=np.zeros((20, 40, 3), np.uint8))
    folder = multishot.save(scene, report, "Stray, 23 Sep 15:32: 2 shots", root=tmp_path)
    assert (folder / "thumb.jpg").is_file()

    saved = multishot.list_saved(tmp_path)
    assert len(saved) == 1 and saved[0].sharp and saved[0].placed == 2

    back = multishot.load(folder)
    assert back.positions == pytest.approx(scene.positions)
    assert back.colors == pytest.approx(scene.colors, abs=1 / 255)
    assert back.cov == pytest.approx(scene.cov)
    assert back.planes[0][1] == pytest.approx(-1.5)


def test_a_damaged_saved_scene_is_skipped(tmp_path):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "scene.json").write_text("{not json")
    assert multishot.list_saved(tmp_path) == []
