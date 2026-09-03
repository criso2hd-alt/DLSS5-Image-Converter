"""The video timeline: frame/pixel mapping, zoom, and In/Out brackets.

The interactive parts (dragging, painting) are verified by eye, but the mapping
and clamping are pure arithmetic and are exactly where an off-by-one puts an In
point on the wrong frame or a zoom that walks off the end of the clip.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from dlss5_converter.widgets import TimelineWidget  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def timeline(qt_app):
    w = TimelineWidget()
    w.resize(1016, 56)  # 1000 px track after the 8 px handles each side
    w.set_duration(5000, 30.0)
    yield w
    w.deleteLater()


def test_duration_sets_a_full_range(timeline):
    assert timeline.in_out() == (0, 4999)
    assert timeline._view_lo == 0 and timeline._view_hi == 5000


def test_frame_and_pixel_round_trip(timeline):
    for frame in (0, 1234, 4999):
        x = timeline._frame_to_x(frame)
        assert abs(timeline._x_to_frame(x) - frame) <= 1


def test_in_out_clamps_and_orders(timeline):
    timeline.set_in_out(4000, 1000)  # out below in
    a, b = timeline.in_out()
    assert a <= b, "out is never left below in"
    timeline.set_in_out(-50, 99999)
    assert timeline.in_out() == (0, 4999), "clamped to the clip"


def test_playhead_clamps(timeline):
    timeline.set_playhead(999999)
    assert timeline._playhead == 4999
    timeline.set_playhead(-10)
    assert timeline._playhead == 0


def test_range_length_in_seconds(timeline):
    timeline.set_in_out(300, 899)  # 600 frames at 30 fps = 20 s
    assert timeline.range_seconds() == pytest.approx(20.0, abs=0.05)


def test_zoom_narrows_the_window_and_stays_in_bounds(timeline):
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtCore import Qt

    def scroll(x, steps):
        at = QPointF(x, 20)
        timeline.wheelEvent(QWheelEvent(
            at, at, QPoint(0, 0), QPoint(0, 120 * steps),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase, False,
        ))

    before = timeline._view_hi - timeline._view_lo
    scroll(500, 5)  # zoom in near the middle
    after = timeline._view_hi - timeline._view_lo
    assert after < before, "scrolling up zooms in"
    assert timeline._view_lo >= 0 and timeline._view_hi <= timeline._total

    for _ in range(50):
        scroll(500, -1)  # zoom all the way back out
    assert timeline._view_lo == 0 and timeline._view_hi == timeline._total


def test_zoom_keeps_the_frame_under_the_cursor(timeline):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    x = 300.0
    pivot = timeline._x_to_frame(x)
    at = QPointF(x, 20)
    timeline.wheelEvent(QWheelEvent(
        at, at, QPoint(0, 0), QPoint(0, 360),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    ))
    assert abs(timeline._x_to_frame(x) - pivot) <= 3, "the cursor stays over its frame"


def test_a_zero_length_clip_does_not_blow_up(qt_app):
    w = TimelineWidget()
    w.resize(400, 56)
    w.set_duration(0, 24.0)
    assert w.in_out() == (0, 0)
    assert w.range_seconds() == 0.0
    w.deleteLater()
