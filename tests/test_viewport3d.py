"""The 3D tab's effect gizmo: handles follow the arrow on screen, and scale."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _viewport_with_volume(qt_app, yaw: float):
    from dlss5_converter.effects3d import EffectsState, volume_preset
    from dlss5_converter.viewport3d import EditorViewport

    vp = EditorViewport()
    vp.resize(800, 600)
    vp.yaw = yaw
    fx = EffectsState()
    item = volume_preset("fog", -3.5)
    fx.volumes.append(item)
    vp.effects = fx
    vp.show_effect_widgets = True
    vp.selected_effect_id = item.id
    return vp, item


def _drag_toward_tip(vp, item, axis: str, pixels: int = 20):
    """Drag along the axis handle as it is drawn on screen."""
    cam = vp._editor_camera()
    m = cam.view_projection(vp.width() / vp.height())
    handles = vp._gizmo_handles(m, vp.size(), np.asarray(item.position, np.float32))
    c, t = handles["view"], handles[axis]
    d = np.array([t.x() - c.x(), t.y() - c.y()], float)
    d = d / np.linalg.norm(d) * pixels
    vp._fx_handle = axis
    vp._drag_effect_handle(int(round(d[0])), int(round(d[1])))


@pytest.mark.parametrize("yaw", [0.55, -0.55, 2.4])
def test_every_axis_handle_moves_toward_its_arrow(qt_app, yaw):
    # The blue (Z) handle used to follow the horizontal drag whatever way its
    # arrow pointed, so from most angles it moved the effect the wrong way.
    vp, item = _viewport_with_volume(qt_app, yaw)
    for i, axis in enumerate("xyz"):
        before = item.position[i]
        _drag_toward_tip(vp, item, axis)
        assert item.position[i] > before, axis


def test_space_cycles_move_rotate_scale(qt_app):
    vp, _item = _viewport_with_volume(qt_app, 0.55)
    seen = []
    for _ in range(3):
        seen.append(vp.gizmo_mode)
        vp.cycle_gizmo_mode()
    assert seen == ["move", "rotate", "scale"]
    assert vp.gizmo_mode == "move"


def test_scale_axis_and_uniform(qt_app):
    vp, item = _viewport_with_volume(qt_app, 0.55)
    vp.set_gizmo_mode("scale")
    sx, sy, sz = item.size
    _drag_toward_tip(vp, item, "z")
    assert item.size[2] > sz and item.size[0] == sx and item.size[1] == sy
    before = item.size
    vp._fx_handle = "view"
    vp._drag_effect_handle(15, -15)          # right and up grows uniformly
    assert all(a > b for a, b in zip(item.size, before))
    ratio_before = before[0] / before[1]
    assert abs(item.size[0] / item.size[1] - ratio_before) < 1e-6


def test_floor_plane_redefines_up_and_collides(qt_app):
    """A user floor tilts particle 'up' with it; the gizmo can move it."""
    import math
    from dlss5_converter.effects3d import plane_preset
    from dlss5_converter.fx3d import euler_matrix

    floor = plane_preset("floor", -3.5)
    floor.rotation = (0.0, 0.0, math.radians(20))
    up = euler_matrix(floor.rotation) @ np.array([0.0, 1.0, 0.0], np.float32)
    assert np.allclose(up, floor.normal(), atol=1e-5)
    assert up[0] < -0.3            # tilted: "up" leans with the floor

    vp, _item = _viewport_with_volume(qt_app, 0.55)
    vp.effects.planes.append(floor)
    vp.selected_effect_id = floor.id
    before = floor.position[1]
    _drag_toward_tip(vp, floor, "y")
    assert floor.position[1] > before


def test_key_easing_is_per_key_like_after_effects():
    from dlss5_converter.animation3d import (
        CameraKey, CameraTrack, Easing, key_ease_label, set_key_ease)
    from dlss5_converter.timeline3d import key_sides

    t = CameraTrack()
    keys = [t.add(CameraKey(time=float(i), easing=Easing.LINEAR), tolerance=0.0) for i in range(3)]
    mid = keys[1]
    set_key_ease(t, mid, "Ease In")          # slow arrival into the middle key only
    assert key_ease_label(t, mid) == "Ease In"
    assert key_sides(keys[0], mid) == ("ease", "linear")
    assert key_ease_label(t, keys[0]) == "Linear"
    set_key_ease(t, mid, "Easy Ease")
    assert key_sides(keys[0], mid) == ("ease", "ease")
    set_key_ease(t, mid, "Ease Out")
    assert key_sides(keys[0], mid) == ("linear", "ease")
    set_key_ease(t, mid, "Hold")
    assert key_ease_label(t, mid) == "Hold"
    assert key_sides(mid, keys[2])[0] == "hold"


@pytest.mark.parametrize("yaw", [0.0, 1.2, 2.6])
def test_middle_drag_pans_along_the_screen(qt_app, yaw):
    """Dragging right moves the view's pivot left on screen, from any angle."""
    from PySide6.QtCore import QPoint, Qt

    vp, _item = _viewport_with_volume(qt_app, yaw)
    right = vp._editor_camera().view_matrix()[0, :3].copy()
    vp._button = Qt.MouseButton.MiddleButton
    vp._last = QPoint(100, 100)

    class Ev:
        def position(self):
            from PySide6.QtCore import QPointF
            return QPointF(140, 100)

    vp.mouseMoveEvent(Ev())
    moved = vp.pan
    assert float(moved @ right) < 0          # along the camera's own right axis
    assert abs(float(np.linalg.norm(moved) - abs(moved @ right))) < 1e-5
