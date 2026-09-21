"""3D tab export must not send a new user to the Video tab for PyAV."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


def test_export_installs_video_support_when_missing(monkeypatch):
    from dlss5_converter import bootstrap, creative_page, video

    state = {"installed": False, "calls": 0}
    monkeypatch.setattr(video, "is_available", lambda: state["installed"])

    def fake_install(on_bytes=None, on_text=None):
        state["calls"] += 1
        on_bytes and on_bytes(50, 100)
        state["installed"] = True

    monkeypatch.setattr(bootstrap, "install_av", fake_install)
    monkeypatch.setattr(bootstrap, "activate_av", lambda: None)
    worker = creative_page._Worker()
    messages = []
    worker.progress.connect(messages.append)
    worker._ensure_video_support()
    assert state["calls"] == 1
    assert any("50%" in m for m in messages)
    worker._ensure_video_support()          # already there: no second download
    assert state["calls"] == 1


def test_export_reports_when_install_does_not_load(monkeypatch):
    from dlss5_converter import bootstrap, creative_page, video

    monkeypatch.setattr(video, "is_available", lambda: False)
    monkeypatch.setattr(bootstrap, "install_av", lambda on_bytes=None, on_text=None: None)
    monkeypatch.setattr(bootstrap, "activate_av", lambda: None)
    with pytest.raises(RuntimeError, match="could not be loaded"):
        creative_page._Worker()._ensure_video_support()


def test_particle_directions_round_trip():
    from dlss5_converter.creative_page import (
        DIRECTIONS, angles_to_direction, direction_name, direction_to_angles)

    assert direction_name((0.0, 1.0, 0.0)) == "Up"
    assert direction_name((0.0, -1.0, 0.0)) == "Down"
    assert direction_name((1.0, 0.0, 0.0)) == "Right"
    assert direction_name((0.0, 0.0, 1.0)) == "Toward camera"
    for name, (tilt, heading) in DIRECTIONS.items():
        d = angles_to_direction(tilt, heading)
        assert direction_name(d) == name
        t2, h2 = direction_to_angles(d)
        assert abs(t2 - tilt) < 1e-6
    assert direction_name(angles_to_direction(30.0, 45.0)) == "Custom"


def test_output_sizes_follow_video_standards():
    from dlss5_converter.creative_page import ASPECTS, RESOLUTIONS, output_size

    hd = RESOLUTIONS["1080p / Full HD"]
    expect = {
        "16:9 HD": (1920, 1080),
        "9:16 Vertical": (1080, 1920),
        "4:3": (1440, 1080),
        "1:1 Square": (1080, 1080),
        "4:5 Vertical (social)": (1080, 1350),
        "1.85:1 Flat (cinema)": (1920, 1038),
        "2.39:1 CinemaScope": (1920, 804),
    }
    for name, size in expect.items():
        assert output_size(ASPECTS[name], hd) == size, name
    assert output_size(16 / 9, RESOLUTIONS["4K / UHD"]) == (3840, 2160)
    for name, ratio in ASPECTS.items():
        if ratio:
            w, h = output_size(ratio, RESOLUTIONS["4K / UHD"])
            assert w % 2 == 0 and h % 2 == 0


def test_lens_round_trip_and_aspect_crops_not_widens():
    import math
    from dlss5_converter.animation3d import CameraKey
    from dlss5_converter.creative_page import lens_fov, lens_mm, shot_matrices

    assert abs(lens_fov(lens_mm(40.0)) - 40.0) < 1e-6
    assert 22.0 < lens_mm(55.0) < 24.0
    key = CameraKey(time=0.0, fov_degrees=55.0)
    _v, p_src = shot_matrices(key, (1600, 900), 16 / 9)
    _v, p_scope = shot_matrices(key, (1920, 804), 16 / 9)
    # Same horizontal coverage (the photo's width); scope only trims height.
    half_w_src = 1.0 / p_src[0, 0]
    half_w_scope = 1.0 / p_scope[0, 0]
    assert abs(half_w_src - half_w_scope) < 1e-4
    _v, p_vert = shot_matrices(key, (1080, 1920), 16 / 9)
    assert abs(1.0 / p_vert[1, 1] - math.tan(math.radians(27.5))) < 1e-4   # height kept
