"""The export size chooser.

Small dialog, but the aspect lock is two spin boxes writing to each other, and
that shape reliably grows a feedback loop or an off-by-one. Worth pinning.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from dlss5_converter.app import ExportDialog  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def parent(qt_app):
    widget = QWidget()
    yield widget
    widget.deleteLater()


def dialog(parent, size=(3840, 2160)) -> ExportDialog:
    return ExportDialog(parent, size)


def choose(box: ExportDialog, label: str) -> None:
    index = box.preset.findText(label)
    assert index >= 0, f"no preset called {label}"
    box.preset.setCurrentIndex(index)


def test_it_opens_at_native_size(parent):
    """The default must cost nothing, or every save grows a decision."""
    box = dialog(parent)
    assert box.chosen() == (3840, 2160)
    assert "no resampling" in box.hint.text()


def test_a_multiplier_scales_both_sides(parent):
    box = dialog(parent)
    choose(box, "2x")
    assert box.chosen() == (7680, 4320)


def test_a_long_edge_preset_respects_orientation(parent):
    """A portrait image's long edge is its height."""
    box = dialog(parent, (900, 1600))
    choose(box, "3840 px long edge")
    assert box.chosen() == (2160, 3840)


def test_typing_a_width_follows_the_height(parent):
    box = dialog(parent)
    box.width_box.setValue(1000)
    assert box.chosen() == (1000, 562)


def test_typing_a_height_follows_the_width(parent):
    box = dialog(parent)
    box.height_box.setValue(1080)
    assert box.chosen() == (1920, 1080)


def test_typing_a_size_clears_the_preset(parent):
    """Leaving "2x" selected next to a hand-typed size would be a lie."""
    box = dialog(parent)
    choose(box, "2x")
    box.width_box.setValue(1000)
    assert box.preset.currentIndex() == -1
    assert box.chosen() == (1000, 562), "clearing the preset must not resize"


def test_unlocking_lets_the_sides_move_independently(parent):
    box = dialog(parent)
    box.lock.setChecked(False)
    box.width_box.setValue(500)
    box.height_box.setValue(500)
    assert box.chosen() == (500, 500)


def test_going_back_to_native_returns_the_original(parent):
    box = dialog(parent)
    choose(box, "4x")
    choose(box, "Native")
    assert box.chosen() == (3840, 2160)
