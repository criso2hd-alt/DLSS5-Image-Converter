"""The mouse wheel must scroll side panels, never change a control (v0.4.2)."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from dlss5_converter.widgets import WheelGuard
    guard = WheelGuard(app)
    app.installEventFilter(guard)
    yield app
    app.removeEventFilter(guard)


def _wheel(widget, steps=-3):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication
    pos = QPointF(widget.width() / 2, widget.height() / 2)
    ev = QWheelEvent(pos, widget.mapToGlobal(pos), QPoint(0, 0), QPoint(0, 120 * steps),
                     Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                     Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(widget, ev)


def test_wheel_scrolls_the_panel_not_the_controls(qt_app):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QScrollArea, QSlider,
                                   QVBoxLayout, QWidget)

    inner = QWidget()
    col = QVBoxLayout(inner)
    slider = QSlider(Qt.Orientation.Horizontal); slider.setRange(0, 100); slider.setValue(50)
    spin = QDoubleSpinBox(); spin.setRange(0, 10); spin.setValue(5)
    combo = QComboBox(); combo.addItems(["a", "b", "c"]); combo.setCurrentIndex(1)
    for w in (slider, spin, combo):
        col.addWidget(w)
    col.addSpacing(3000)                      # tall enough to scroll
    area = QScrollArea(); area.setWidget(inner); area.setWidgetResizable(True)
    area.resize(300, 200); area.show(); qt_app.processEvents()

    before = area.verticalScrollBar().value()
    for w in (slider, spin, combo):
        w.setFocus()                          # even when focused
        _wheel(w)
    qt_app.processEvents()
    assert slider.value() == 50
    assert spin.value() == 5
    assert combo.currentIndex() == 1
    assert area.verticalScrollBar().value() > before     # the panel scrolled instead
    area.close()


def test_plain_widgets_keep_their_own_wheel(qt_app):
    """Image zoom, the timelines and the 3D viewport are plain QWidgets."""
    from PySide6.QtWidgets import QWidget

    got = []

    class Zoomer(QWidget):
        def wheelEvent(self, event):  # noqa: N802
            got.append(event.angleDelta().y())

    z = Zoomer(); z.resize(100, 100); z.show(); qt_app.processEvents()
    _wheel(z, 1)
    assert got == [120]
    z.close()


def test_depth_strength_applies_on_release_only(qt_app):
    """Dragging Depth strength must not rebuild the scene until the button is let go."""
    from dlss5_converter.creative_page import CreativePage

    page = CreativePage()
    changes = []
    page.contrast.valueChanged.connect(changes.append)
    page.contrast.setSliderDown(True)       # user is dragging
    page.contrast.setSliderPosition(150)
    page.contrast.setSliderPosition(180)
    assert changes == []                     # nothing while dragging
    page.contrast.setSliderDown(False)      # released
    assert changes == [180]
    page.shutdown()


def test_download_buttons_hidden_from_the_start_when_installed(qt_app, monkeypatch):
    """A fresh start must not offer to download SHARP/LaMa that are installed."""
    from dlss5_converter import creative_page, inpaint, sharp3d

    monkeypatch.setattr(sharp3d, "is_downloaded", lambda: True)
    monkeypatch.setattr(inpaint, "is_downloaded", lambda: True)
    page = creative_page.CreativePage()          # no scene built yet
    assert page.sharp_button.isHidden()
    assert page.lama_button.isHidden()
    page.shutdown()

    monkeypatch.setattr(sharp3d, "is_downloaded", lambda: False)
    monkeypatch.setattr(inpaint, "is_downloaded", lambda: False)
    page = creative_page.CreativePage()
    assert not page.sharp_button.isHidden()
    assert not page.lama_button.isHidden()
    page.shutdown()
