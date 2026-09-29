"""The Video tab's own DLSS settings, independent of the Single image sidebar."""

from __future__ import annotations

import json
import os

import pytest

from dlss5_converter.settings import VIDEO_MAX_EDGE, VIDEO_PASSES, AppSettings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_existing_users_start_with_their_photo_look(tmp_path):
    """No video settings saved yet: they begin as a copy of the photo ones."""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"neural": {"style": 1, "skin": 0.7}}), encoding="utf-8")
    loaded = AppSettings.load(path)
    assert loaded.video.neural.style == 1 and loaded.video.neural.skin == 0.7
    loaded.video.neural.skin = 1.5                       # a copy, not the same object
    assert loaded.neural.skin == 0.7


def test_video_settings_round_trip_independently(tmp_path):
    path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.neural.style = 0
    settings.video.neural.style = 2
    settings.video.neural.paper_white = 16.0
    settings.save(path)
    back = AppSettings.load(path)
    assert back.neural.style == 0
    assert back.video.neural.style == 2 and back.video.neural.paper_white == 16.0


@pytest.fixture
def window(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QApplication

    from dlss5_converter import app as gui

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(QThread, "start", lambda self, *a, **k: None)
    # Never read or write the real settings file from a test.
    monkeypatch.setattr(gui.paths, "settings_path", lambda: tmp_path / "settings.json")
    win = gui.MainWindow(first_run_setup=False)
    yield win
    win.deleteLater()


def test_video_conversion_uses_only_the_video_tab(window):
    s = window.settings
    s.neural.style, s.neural.intensity = 0, 0.3            # the photo sidebar
    s.evaluation.frames, s.evaluation.max_edge = 8, 7680
    s.depth.model_id = "depth-anything/Depth-Anything-V2-Large-hf"
    s.grade.exposure = 1.0 if hasattr(s.grade, "exposure") else 0.0
    window.video_page.style_box.changed.emit(2)            # the Video tab's style
    run = window.video_run_settings()
    assert run.neural.style == 2 and run.neural.intensity == s.video.neural.intensity
    assert run.evaluation.frames == VIDEO_PASSES and run.evaluation.max_edge == VIDEO_MAX_EDGE
    assert run.depth.model_id.endswith("Small-hf") and run.detail.mode == "off"
    assert run.grade.is_neutral
    # The photo side is untouched.
    assert s.neural.style == 0 and s.evaluation.frames == 8


def test_video_tab_shows_its_own_cards_in_order(window):
    page = window.video_page
    order = [page._cards_col.itemAt(i).widget() for i in range(page._cards_col.count())]
    cards = [page.source_card, page.export_card, page.neural_card, page.hdr_card, page.stereo_card]
    assert [order.index(c) for c in cards] == sorted(order.index(c) for c in cards)
    assert not hasattr(page, "mode_box") and not hasattr(page, "estimate_depth")
    assert page.neural_params in window._chip_groups


def _sdr_clip(path, frames=12, size=(160, 96)):
    av = pytest.importorskip("av")
    import numpy as np

    w, h = size
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=12)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        for i in range(frames):
            img = np.full((h, w, 3), i * 20, np.uint8)       # brightness marks the frame
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_frame_at_lands_on_the_requested_frame(tmp_path):
    from dlss5_converter import video

    clip = tmp_path / "clip.mp4"
    _sdr_clip(clip)
    first = video.frame_at(clip, 0.0)
    later = video.frame_at(clip, 0.5)                       # 12 fps: frame 6
    assert first.dtype.name == "uint8" and first.shape == (96, 160, 3)
    assert abs(float(later.mean()) - 120) < 12 and float(first.mean()) < 12
    small = video.frame_at(clip, 0.0, max_edge=80)
    assert small.shape[1] == 80 and small.shape[0] % 2 == 0


def test_preview_lands_in_the_wipe_unless_the_video_is_playing(window):
    import numpy as np

    page = window.video_page
    page.source = gui_path = __import__("pathlib").Path("clip.mp4")
    assert gui_path
    before = np.zeros((20, 30, 3), np.float32)
    after = np.ones((20, 30, 3), np.float32)
    window._vpreview_show = True
    window._video_preview_done((before, after, 1.5))
    assert page.stack.currentWidget() is page.dlss_view
    assert page.view_switch.isEnabled() and "1.5 s" in page.preview_status.text()
    page.show_video()
    assert page.stack.currentWidget() is page.video_widget


def test_preview_without_dlss_files_says_so(window, monkeypatch):
    from dlss5_converter import app as gui

    class _NotReady:
        ready = False

    monkeypatch.setattr(gui.runtime, "detect", lambda *_a, **_k: _NotReady())
    page = window.video_page
    page.source = __import__("pathlib").Path("clip.mp4")
    window._video_preview(seconds=0.0)
    assert window._vpreview_thread is None
    assert "DLSS files" in page.preview_status.text()
