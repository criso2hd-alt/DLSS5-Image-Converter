"""The freeze detector writes where the UI thread was stuck, and only then."""

from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from dlss5_converter.freeze_watch import FreezeWatch  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


def _pump(app, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.02)


def a_stuck_ui_thread(seconds):
    time.sleep(seconds)          # stands in for a blocking call on the UI thread


def test_a_stall_is_logged_with_the_stuck_function(qt_app, tmp_path):
    watch = FreezeWatch(tmp_path, freeze_seconds=1.5, version="test")
    watch.start()
    try:
        _pump(qt_app, 0.8)
        a_stuck_ui_thread(3.0)
        _pump(qt_app, 1.5)       # recover, so the watcher notes it
    finally:
        watch.stop()
    logs = list(tmp_path.glob("freeze_*.txt"))
    assert len(logs) == 1
    text = logs[0].read_text(encoding="utf-8")
    assert "a_stuck_ui_thread" in text and "recovered" in text


def test_a_responsive_ui_writes_nothing(qt_app, tmp_path):
    watch = FreezeWatch(tmp_path, freeze_seconds=1.0)
    watch.start()
    try:
        _pump(qt_app, 2.5)
    finally:
        watch.stop()
    assert not list(tmp_path.glob("freeze_*.txt"))
