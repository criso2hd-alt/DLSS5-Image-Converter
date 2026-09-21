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
