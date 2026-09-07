from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from dlss5_converter import onboarding, paths  # noqa: E402
from dlss5_converter.settings import (  # noqa: E402
    ONBOARDING_VERSION,
    AppSettings,
)


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


def test_probe_requires_the_whole_neural_path():
    good = "\n".join((
        "dlss_available: 1",
        "neural_addon_loaded: 1",
        "reshade_proxy_loaded: 1",
        "dlssnr_module_loaded: 1",
        "test_evaluation: ok",
    ))
    assert onboarding.probe_succeeded(good)
    assert not onboarding.probe_succeeded(good.replace("neural_addon_loaded: 1", "neural_addon_loaded: 0"))


def test_new_settings_need_onboarding_but_old_files_are_migrated(tmp_path):
    assert AppSettings().onboarding_version == 0
    old = tmp_path / "settings.json"
    old.write_text(json.dumps({"theme": "Neural Cyan"}), encoding="utf-8")
    assert AppSettings.load(old).onboarding_version == ONBOARDING_VERSION


def test_onboarding_version_round_trips(tmp_path):
    settings = AppSettings(onboarding_version=ONBOARDING_VERSION)
    saved = tmp_path / "settings.json"
    settings.save(saved)
    assert AppSettings.load(saved).onboarding_version == ONBOARDING_VERSION


def test_bad_onboarding_version_does_not_break_settings(tmp_path):
    saved = tmp_path / "settings.json"
    saved.write_text(json.dumps({"onboarding_version": "not-a-number"}), encoding="utf-8")
    assert AppSettings.load(saved).onboarding_version == ONBOARDING_VERSION


def test_tutorial_asset_is_bundled():
    assert paths.onboarding_image().is_file()


def test_spotlight_walks_each_real_target_and_finishes(qt_app):
    parent = QWidget()
    parent.resize(900, 600)
    first = QWidget(parent)
    first.setGeometry(20, 80, 400, 300)
    second = QWidget(parent)
    second.setGeometry(650, 100, 200, 220)
    overlay = onboarding.SpotlightOverlay(parent, [
        onboarding.TourStep("Image", "Open one.", first),
        onboarding.TourStep("Controls", "Tune them.", second),
    ])
    completed = []
    overlay.finished.connect(lambda: completed.append(True))
    overlay.start()

    assert overlay.step_index == 0
    assert overlay._target.intersects(first.geometry())
    old_target = overlay._target
    overlay._next()
    overlay._spot_animation.setCurrentTime(overlay._spot_animation.duration() // 2)
    halfway = overlay._target
    # The ring is one continuously reshaping rectangle: halfway through it is
    # neither the old target nor the new one, and both its position and size
    # are in flight. This guards against regressing to a jump/cross-fade.
    assert halfway != old_target
    assert halfway != second.geometry().adjusted(-7, -7, 7, 7)
    assert halfway.width() < old_target.width()
    assert halfway.left() > old_target.left()
    overlay._spot_animation.setCurrentTime(overlay._spot_animation.duration())
    assert overlay.step_index == 1
    assert overlay._target.intersects(second.geometry())
    assert overlay.next_button.text() == "Done"
    overlay._next()
    assert completed == [True]
