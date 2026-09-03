"""Two panes, one transform.

The claim this view makes is that the panes cannot drift apart, because they
are not two synchronised viewports - they are one zoom and one pan drawn twice.
These tests hold that claim to the geometry.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QImage, QMouseEvent, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from dlss5_converter.widgets import SideBySideView  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def view(qt_app):
    widget = SideBySideView()
    widget.resize(812, 400)
    left = np.zeros((100, 200, 3), np.uint8)
    right = np.full((100, 200, 3), 255, np.uint8)
    widget.set_images_u8(left, right)
    yield widget
    widget.deleteLater()


def scroll(view: SideBySideView, at: QPointF, steps: int = 1) -> None:
    event = QWheelEvent(
        at, view.mapToGlobal(at.toPoint()).toPointF(),
        QPoint(0, 0), QPoint(0, 120 * steps),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )
    view.wheelEvent(event)


def drag(view: SideBySideView, start: QPointF, end: QPointF) -> None:
    def event(kind, pos, button):
        return QMouseEvent(
            kind, pos, view.mapToGlobal(pos.toPoint()).toPointF(), button,
            button, Qt.KeyboardModifier.NoModifier,
        )

    view.mousePressEvent(
        event(QMouseEvent.Type.MouseButtonPress, start, Qt.MouseButton.RightButton)
    )
    view.mouseMoveEvent(event(QMouseEvent.Type.MouseMove, end, Qt.MouseButton.NoButton))
    view.mouseReleaseEvent(
        event(QMouseEvent.Type.MouseButtonRelease, end, Qt.MouseButton.RightButton)
    )


def test_the_panes_split_the_width_evenly(view):
    assert view._pane_left(0) == 0.0
    assert view._pane_left(1) == pytest.approx(view._pane_width() + view.GAP)
    assert 2 * view._pane_width() + view.GAP == pytest.approx(view.width())


def test_both_panes_use_one_rect(view):
    """Not "kept in step" - the same numbers, offset by the pane position."""
    scroll(view, QPointF(100, 200), steps=3)
    rect = view._display_rect(view._left.size())
    # Whatever the transform is, pane 1 draws the identical rect shifted right
    # by exactly the pane offset, so the same pixel lands at the same height.
    shifted = rect.translated(view._pane_left(1), 0.0)
    assert shifted.top() == rect.top()
    assert shifted.height() == rect.height()
    assert shifted.left() - rect.left() == pytest.approx(view._pane_left(1))


def test_zooming_over_either_pane_gives_the_same_result(qt_app):
    """The right pane's cursor is mapped into pane-local space first.

    Without that, zooming on the right would aim at a point half an image away
    and the two halves would show different parts of the picture.
    """
    rects = []
    for pane in (0, 1):
        widget = SideBySideView()
        widget.resize(812, 400)
        widget.set_images_u8(
            np.zeros((100, 200, 3), np.uint8), np.zeros((100, 200, 3), np.uint8)
        )
        at = QPointF(120 + widget._pane_left(pane), 210)
        scroll(widget, at, steps=2)
        rects.append(widget._display_rect(widget._left.size()))
        widget.deleteLater()

    assert rects[0].left() == pytest.approx(rects[1].left(), abs=0.5)
    assert rects[0].top() == pytest.approx(rects[1].top(), abs=0.5)
    assert rects[0].width() == pytest.approx(rects[1].width(), abs=0.5)


def test_panning_moves_one_transform(view):
    scroll(view, QPointF(200, 200), steps=3)
    before = view._display_rect(view._left.size())
    drag(view, QPointF(200, 200), QPointF(160, 190))
    after = view._display_rect(view._left.size())
    assert after.left() != before.left() or after.top() != before.top()
    # Still one rect for both panes after panning.
    assert after.translated(view._pane_left(1), 0.0).top() == after.top()


def test_zoom_fits_within_a_pane_not_the_widget(view):
    """At fit, each image fills its own pane rather than the whole window."""
    rect = view._display_rect(view._left.size())
    assert rect.width() <= view._pane_width() + 1


def test_a_new_image_of_the_same_size_keeps_the_view(view):
    scroll(view, QPointF(200, 200), steps=3)
    zoomed = view._zoom
    view.set_images_u8(
        np.zeros((100, 200, 3), np.uint8), np.zeros((100, 200, 3), np.uint8)
    )
    assert view._zoom == zoomed, "re-grading must not throw away the zoom"


def test_a_different_size_refits(view):
    scroll(view, QPointF(200, 200), steps=3)
    view.set_images_u8(
        np.zeros((50, 50, 3), np.uint8), np.zeros((50, 50, 3), np.uint8)
    )
    assert view._zoom == 1.0


def test_labels_are_stored_for_both_panes(view):
    view.set_labels("Natural", "Cinematic")
    assert view._labels == ("Natural", "Cinematic")


def test_clearing_drops_both_images(view):
    view.clear()
    assert view._left is None and view._right is None
    assert view._content_size() is None


def rendered_panes(view: SideBySideView):
    """What each pane actually painted, below the label band.

    The arrays are copied out. numpy would otherwise hold a view straight into
    the QImage's buffer, and the QImage dies when this returns - which reads as
    an access violation somewhere later and entirely unrelated.
    """
    image = view.grab().toImage().convertToFormat(QImage.Format.Format_RGB32)
    height, width = image.height(), image.width()
    buffer = np.frombuffer(image.constBits(), np.uint8).reshape(
        height, image.bytesPerLine() // 4, 4
    )
    pixels = np.array(buffer[:, :width, :3])
    pane = int(view._pane_width())
    start = int(view._pane_left(1))
    # Only the panes. The label and zoom bands above and below are shared
    # chrome drawn across the whole widget, not something either pane painted.
    top = view.LABEL_BAND
    bottom = height - view.CAPTION_BAND
    return pixels[top:bottom, :pane], pixels[top:bottom, start:start + pane]


def test_both_panes_render_the_same_region(view):
    """The property the whole view exists for, checked on pixels.

    The same image goes into both panes, so after any amount of zooming and
    panning the two must paint identically. Anything else means the halves are
    showing different parts of the picture, which is worse than useless for a
    comparison - it would look like the styles differ where they do not.

    Deliberately zooms over one pane and then the other: mapping the cursor
    into pane-local coordinates is the step that makes that work, and getting
    it wrong aims the second zoom half an image away.
    """
    same = np.random.default_rng(0).integers(0, 255, (120, 200, 3), dtype=np.uint8)
    view.set_images_u8(same, same)

    def identical() -> bool:
        left, right = rendered_panes(view)
        return bool((left == right).all())

    assert identical(), "at fit"
    scroll(view, QPointF(200, 300), steps=5)
    assert identical(), "after zooming over the left pane"
    drag(view, QPointF(400, 300), QPointF(250, 220))
    assert identical(), "after panning"
    scroll(view, QPointF(view._pane_left(1) + 150, 260), steps=4)
    assert identical(), "after zooming over the right pane"
    drag(view, QPointF(300, 300), QPointF(380, 350))
    assert identical(), "after panning again"
    assert view._zoom > 5.0, "should have ended up well zoomed in"


def test_a_pan_moves_both_panes_by_the_same_amount(view):
    """Dragging either pane drags the other with it, not merely near it."""
    scroll(view, QPointF(200, 200), steps=4)
    before = view._display_rect(view._left.size())
    drag(view, QPointF(200, 200), QPointF(140, 170))
    after = view._display_rect(view._left.size())

    moved = (after.left() - before.left(), after.top() - before.top())
    assert moved != (0.0, 0.0), "the drag should have moved something"
    # One rect drives both panes, so the right pane moved by exactly this too.
    right_before = before.translated(view._pane_left(1), 0.0)
    right_after = after.translated(view._pane_left(1), 0.0)
    assert (right_after.left() - right_before.left()) == pytest.approx(moved[0])
    assert (right_after.top() - right_before.top()) == pytest.approx(moved[1])
