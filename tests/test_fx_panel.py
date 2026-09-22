"""The 3D tab's effects panel: every control of every effect type works."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def page(qt_app):
    from dlss5_converter.creative_page import CreativePage
    p = CreativePage()
    yield p
    # Stop the page's worker thread: a page dropped with it running crashes
    # Qt later in the session, in whichever test happens to be next.
    p.shutdown()
    p.deleteLater()
    qt_app.processEvents()


ADDERS = [("volume", "_add_volume", "Fog"), ("particles", "_add_emitter", "Rain"),
          ("particles", "_add_emitter", "Embers"), ("lightning", "_add_lightning", None),
          ("plane", "_add_plane", "Floor")]


@pytest.mark.parametrize("kind, adder, arg", ADDERS)
def test_every_slider_edits_and_keys(page, kind, adder, arg):
    panel = page.fx_panel
    getattr(panel, adder)(*([arg] if arg else []))
    item = panel.base_item()
    assert item is not None and panel.props_card.isVisibleTo(page)
    for field, slider in list(panel._sliders.items()):
        if field in ("tilt", "heading"):
            continue
        p = slider.prop
        slider.slider.setValue(slider.STEPS // 2)        # a user drag
        if field != "size":
            assert getattr(item, field) == pytest.approx(slider.value.value(), abs=10 ** -p.decimals)
        slider.diamond.click()                           # key it
        assert panel._key_state((field,)) == "keyed"
        slider.diamond.click()                           # and unkey it
        assert panel._key_state((field,)) == "none"


def test_animated_property_auto_keys(page):
    panel = page.fx_panel
    panel._add_emitter("Smoke")
    item = panel.base_item()
    page.time = 0.0
    panel._toggle_key(("opacity",))
    page.time = 2.0
    panel.set_value("opacity", 0.1)          # animated, so this adds a key
    assert page.effects_track.is_keyed(2.0, f"{item.id}.opacity")
    assert page.effects_at(0.0).item(item.id).opacity == pytest.approx(item.opacity)
    assert page.effects_at(2.0).item(item.id).opacity == pytest.approx(0.1)
    # The base value, used where nothing is keyed, is untouched.
    assert item.opacity != pytest.approx(0.1)


def test_colour_and_placement_key_every_component(page):
    panel = page.fx_panel
    panel._add_lightning()
    item = panel.base_item()
    panel._toggle_key(("colour",))
    panel._toggle_key(("position", "rotation"))
    track = page.effects_track
    for path in [f"{item.id}.colour.{i}" for i in range(3)] + \
                [f"{item.id}.position.{i}" for i in range(3)]:
        assert track.is_keyed(page.time, path)


def test_duplicate_and_remove(page):
    panel = page.fx_panel
    panel._add_emitter("Snow")
    panel._duplicate()
    assert len(page.effects.emitters) == 2
    assert page.effects.emitters[0].id != page.effects.emitters[1].id
    panel._remove()
    panel._remove()
    assert page.effects.emitters == [] and not panel.props_card.isVisibleTo(page)


def test_unticking_hides_without_removing(page):
    from PySide6.QtCore import Qt
    panel = page.fx_panel
    panel._add_volume("Smoke")
    panel.list.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert page.effects.volumes and not page.effects.volumes[0].enabled
