"""Interactive 3D viewport for inspecting the scene and the camera path.

Ported from Depth Animator (viewport.py); draws through splat3d.

The render view shows what the video will be. This shows the *scene*: the mesh
from any angle, where the animated camera sits, and the path it travels. Without
it the camera move can only be judged from its own output, which is why the old
single-preview layout felt like editing blind.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from .animation3d import CameraTrack, key_to_camera
from .camera3d import Camera

#: Names map to `Renderer3D.render(mode=...)` lowercased. "Point cloud" needs
#: its space stripped, so `EditorViewport.render_mode` does that rather than
#: leaving the renderer to guess.
VIEW_MODES = ["Textured", "Depth", "Point cloud"]

#: Axis colours, shared by the viewport gizmo and the sidebar transform fields
#: so the two can never drift apart. X red, Y green, Z blue, as in every 3D app.
AXIS_COLOURS = {"x": "#ff5f6b", "y": "#8bd450", "z": "#4aa8ff"}


def _project(point: np.ndarray, view_projection: np.ndarray, size) -> QPoint | None:
    """World point to widget pixel, or None when behind the camera."""
    clip = view_projection @ np.array([point[0], point[1], point[2], 1.0], np.float32)
    if clip[3] <= 1e-5:
        return None
    ndc = clip[:3] / clip[3]
    x = (ndc[0] * 0.5 + 0.5) * size.width()
    y = (0.5 - ndc[1] * 0.5) * size.height()
    return QPoint(int(round(x)), int(round(y)))


class EditorViewport(QWidget):
    """Orbitable scene view with a camera gizmo drawn over it.

    Right mouse orbits, middle mouse pans, and the wheel zooms. These editor
    controls never modify the animated camera.
    """

    camera_moved = Signal()
    keyframe_added = Signal(float)
    effect_moved = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.renderer = None
        self.pivot_z = -4.0
        self.track: CameraTrack | None = None
        self.current_time = 0.0
        self.duration = 4.0
        self.mode = "Textured"
        self.show_gizmo = True
        self.scan_phase = 0.0
        self.effects = None
        self.show_effect_widgets = False
        self.show_reconstruction = False
        self.scene_model = None
        self.selected_effect_id: str | None = None
        self._active_effect = None
        self._fx_handle: str | None = None

        self.yaw = 0.55
        self.pitch = 0.30
        self.distance_scale = 1.9
        self.pan = np.zeros(2, dtype=np.float32)

        self._last: QPoint | None = None
        self._button = None
        self.gizmo_mode = "move"
        self.active_handle: str | None = None
        self.hovered_handle: str | None = None
        self._frame: np.ndarray | None = None
        self.setMinimumSize(360, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # -- state ---------------------------------------------------------------

    def set_renderer(self, renderer, pivot_z: float) -> None:
        self.renderer = renderer
        self.pivot_z = pivot_z
        self.invalidate()

    def set_track(self, track: CameraTrack | None, duration: float) -> None:
        self.track = track
        self.duration = max(0.1, duration)
        self.update()

    def set_effects(self, effects) -> None:
        self.effects = effects
        self.invalidate()

    def set_effect_selection(self, item_id: str | None) -> None:
        self.selected_effect_id = item_id
        self.update()

    def set_time(self, seconds: float) -> None:
        self.current_time = seconds
        self.update()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.invalidate()

    def set_scene_model(self, model) -> None:
        """Orientation applied to the scene, shared with the camera preview."""
        self.scene_model = model
        self.invalidate()

    def set_show_reconstruction(self, enabled: bool) -> None:
        """Tint invented texture red here too, not only in the camera view."""
        self.show_reconstruction = bool(enabled)
        self.invalidate()

    @property
    def render_mode(self) -> str:
        """The dropdown label as the renderer names it."""
        return self.mode.lower().replace(" ", "")

    def advance_scan(self, amount: float = 0.008) -> None:
        """Sweep the point-cloud scan plane from the camera to the backdrop."""
        self.scan_phase = (self.scan_phase + amount) % 1.0
        if self.render_mode == "pointcloud":
            self.invalidate()

    def invalidate(self) -> None:
        self._frame = None
        self.update()

    def reset_view(self) -> None:
        self.yaw, self.pitch, self.distance_scale = 0.55, 0.30, 1.9
        self.pan[:] = 0.0
        self.invalidate()

    # -- camera --------------------------------------------------------------

    def _editor_camera(self) -> Camera:
        distance = abs(self.pivot_z) * self.distance_scale
        pivot = np.array([self.pan[0], self.pan[1], self.pivot_z], np.float32)
        eye = pivot + np.array(
            [
                distance * math.sin(self.yaw) * math.cos(self.pitch),
                distance * math.sin(self.pitch),
                distance * math.cos(self.yaw) * math.cos(self.pitch),
            ],
            np.float32,
        )
        return Camera(
            position=tuple(float(v) for v in eye),
            target=tuple(float(v) for v in pivot),
            fov_degrees=50.0,
            far=400.0,
        )

    # -- painting ------------------------------------------------------------

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#080b11"))

        if self.renderer is None:
            painter.setPen(QColor("#657087"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Convert an image on the Single image tab first",
            )
            return

        # Render at device pixels, not logical ones: on a scaled display the
        # logical-size frame was drawn into a larger rect and resampled, which
        # cost the viewport its sharpness for no saving.
        ratio = max(1.0, float(self.devicePixelRatioF()))
        size = (
            max(2, min(3840, round(self.width() * ratio))),
            max(2, min(2160, round(self.height() * ratio))),
        )
        camera = self._editor_camera()
        if self._frame is None or self._frame.shape[:2] != (size[1], size[0]):
            self._frame = self.renderer.render_view(
                camera,
                size,
                self.render_mode,
                hologram_phase=self.scan_phase,
                effects=self.effects,
                time_seconds=self.current_time,
                show_reconstruction=self.show_reconstruction,
                model=self.scene_model,
            )
        frame = np.ascontiguousarray(self._frame)
        image = QImage(
            frame.data, frame.shape[1], frame.shape[0], frame.strides[0],
            QImage.Format.Format_RGB888,
        )
        painter.drawPixmap(self.rect(), QPixmap.fromImage(image.copy()))

        if self.show_gizmo:
            self._paint_gizmo(painter, camera)
        if self.show_effect_widgets:
            self._paint_effect_widgets(painter, camera)
        self._paint_hud(painter)

    def _effect_items(self):
        if self.effects is None:
            return []
        return [*self.effects.volumes, *self.effects.emitters]

    def _paint_effect_widgets(self, painter: QPainter, camera: Camera) -> None:
        vp = camera.view_projection(self.width() / max(self.height(), 1))
        for item in self._effect_items():
            if not item.enabled:
                continue
            centre = _project(np.asarray(item.position, np.float32), vp, self.size())
            if centre is None:
                continue
            selected = item.id == self.selected_effect_id
            colour = QColor("#ff7a42" if item.kind in {"fire", "embers"} else "#75dbff")
            colour.setAlpha(245 if selected else 150)
            painter.setPen(QPen(colour, 3 if selected else 1))
            painter.setBrush(QColor(colour.red(), colour.green(), colour.blue(), 34))
            radius = 13 if selected else 9
            self._paint_bounds(painter, vp, item, colour, selected)
            if selected:
                # Transform handles on the selected effect, so it can be moved
                # and rotated in place rather than only slid across the view.
                previous = self.active_handle
                self.active_handle = self._fx_handle
                self._painting_effect = True
                self._paint_transform_gizmo(
                    painter, vp, self.size(), np.asarray(item.position, np.float32)
                )
                self._painting_effect = False
                self.active_handle = previous
            painter.drawEllipse(centre, radius, radius)
            painter.drawLine(centre-QPoint(radius+5, 0), centre+QPoint(radius+5, 0))
            painter.drawLine(centre-QPoint(0, radius+5), centre+QPoint(0, radius+5))

    #: The 12 edges of a cube, as index pairs into the corner ordering below.
    _BOX_EDGES = (
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )

    def _box_corners(self, item) -> list[np.ndarray]:
        """World-space corners, oriented by the effect's own rotation."""
        from .fx3d import euler_matrix

        half = np.asarray(item.size, np.float32) * 0.5
        origin = np.asarray(item.position, np.float32)
        rotation = euler_matrix(getattr(item, "rotation", (0.0, 0.0, 0.0)))
        corners = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    local = half * np.array((sx, sy, sz), np.float32)
                    corners.append(origin + rotation @ local)
        return corners

    def _paint_bounds(self, painter: QPainter, vp, item, colour, selected: bool) -> None:
        """Draw the volume bounds as a real wireframe box.

        This used to take the min/max of the projected corners and draw their
        screen-space bounding rectangle, which discarded all the 3D: the box
        resized as the camera orbited but never tilted, so it read as a flat
        overlay rather than a volume sitting in the scene.
        """
        projected = [_project(c, vp, self.size()) for c in self._box_corners(item)]
        if any(p is None for p in projected):
            return
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(colour, 2 if selected else 1))
        for a, b in self._BOX_EDGES:
            painter.drawLine(projected[a], projected[b])

    def selected_effect(self):
        if not self.selected_effect_id:
            return None
        return next(
            (i for i in self._effect_items() if i.id == self.selected_effect_id), None
        )

    def _hit_effect_handle(self, point: QPoint) -> str | None:
        """Pick a transform handle belonging to the selected effect."""
        item = self.selected_effect()
        if item is None or not self.show_effect_widgets:
            return None
        origin = np.asarray(item.position, np.float32)
        if self.gizmo_mode == "rotate":
            return self._hit_rotate_ring(point, origin)
        camera = self._editor_camera()
        vp = camera.view_projection(self.width() / max(self.height(), 1))
        handles = self._gizmo_handles(vp, self.size(), origin)
        best, best_distance = None, 1e9
        # Whole shaft, not just the tip, as for the camera gizmo.
        centre = handles.get("view")
        if centre is not None and float((centre - point).manhattanLength()) < 14:
            return "view"
        px, py = float(point.x()), float(point.y())
        for name in ("x", "y", "z"):
            tip = handles.get(name)
            if centre is None or tip is None:
                continue
            vx, vy = tip.x() - centre.x(), tip.y() - centre.y()
            length2 = vx * vx + vy * vy
            if length2 <= 1e-6:
                continue
            along = max(0.0, min(1.0, ((px - centre.x()) * vx + (py - centre.y()) * vy) / length2))
            distance = math.hypot(px - (centre.x() + along * vx), py - (centre.y() + along * vy))
            if distance < 12 and distance < best_distance:
                best, best_distance = name, distance
        if best is not None:
            return best
        for name, position in handles.items():
            distance = float((position - point).manhattanLength())
            limit = 16 if name != "view" else 14
            if distance < limit and distance < best_distance:
                best, best_distance = name, distance
        return best

    def _along_handle(self, dx: int, dy: int, origin, axis: str) -> float:
        """Mouse movement projected onto an axis handle's on-screen direction,
        in pixels: positive when dragging toward the arrow tip."""
        camera = self._editor_camera()
        vp = camera.view_projection(self.width() / max(self.height(), 1))
        handles = self._gizmo_handles(vp, self.size(), np.asarray(origin, np.float32))
        centre, tip = handles.get("view"), handles.get(axis)
        if centre is None or tip is None:
            return 0.0
        vx, vy = tip.x() - centre.x(), tip.y() - centre.y()
        length = math.hypot(vx, vy)
        if length < 1e-6:
            return 0.0
        return (dx * vx + dy * vy) / length

    def _drag_effect_handle(self, dx: int, dy: int) -> None:
        """Apply a constrained drag to the selected effect."""
        item = self.selected_effect()
        if item is None:
            return
        if self.gizmo_mode == "rotate":
            rx, ry, rz = getattr(item, "rotation", (0.0, 0.0, 0.0))
            step = 0.006
            if self._fx_handle == "x":
                rx -= dy * step
            elif self._fx_handle == "y":
                ry += dx * step
            elif self._fx_handle == "z":
                rz += dx * step
            else:
                ry += dx * step
                rx -= dy * step
            item.rotation = (rx, ry, rz)
        elif self.gizmo_mode == "scale":
            # Axis handle stretches that axis only; the centre scales all three.
            # Dragging along the handle's on-screen direction grows it, against
            # it shrinks, exponentially so it feels the same at any size.
            size = list(item.size)
            if self._fx_handle in ("x", "y", "z"):
                i = "xyz".index(self._fx_handle)
                amount = self._along_handle(dx, dy, item.position, self._fx_handle)
                size[i] = max(0.02, size[i] * math.exp(amount * 0.01))
            else:
                amount = (dx - dy) * 0.01
                size = [max(0.02, s * math.exp(amount)) for s in size]
            item.size = tuple(size)
        else:
            speed = abs(self.pivot_z) * 0.0025 * self.distance_scale
            x, y, z = item.position
            if self._fx_handle in ("x", "y", "z"):
                # Follow the arrow as drawn on screen. The old code moved Z by
                # the horizontal drag whatever the arrow's direction, so the
                # blue handle went the wrong way from most angles.
                amount = self._along_handle(dx, dy, item.position, self._fx_handle) * speed
                pos = [x, y, z]
                pos["xyz".index(self._fx_handle)] += amount
                x, y, z = pos
            else:
                # Centre handle: move in the plane facing the editor camera.
                cam = self._editor_camera()
                view = cam.view_matrix()
                right, up = view[0, :3], view[1, :3]
                delta = right * dx * speed - up * dy * speed
                x, y, z = x + float(delta[0]), y + float(delta[1]), z + float(delta[2])
            item.position = (x, y, z)
        self.effect_moved.emit(item.id)
        self.invalidate()

    def _hit_effect(self, point: QPoint):
        if not self.show_effect_widgets:
            return None
        camera = self._editor_camera()
        vp = camera.view_projection(self.width() / max(self.height(), 1))
        best, distance = None, 18.0
        for item in self._effect_items():
            centre = _project(np.asarray(item.position, np.float32), vp, self.size())
            if centre is None:
                continue
            d = math.hypot(centre.x()-point.x(), centre.y()-point.y())
            if d < distance:
                best, distance = item, d
        return best

    def _paint_gizmo(self, painter: QPainter, camera: Camera) -> None:
        if self.track is None or not self.track.keys:
            return
        vp = camera.view_projection(self.width() / max(self.height(), 1))
        size = self.size()

        # The path the camera travels, sampled finely enough to read as a curve.
        points: list[QPoint] = []
        for i in range(81):
            t = self.duration * i / 80.0
            key = self.track.evaluate(t)
            if key is None:
                continue
            projected = _project(np.array(key.position, np.float32), vp, size)
            if projected is not None:
                points.append(projected)
        if len(points) > 1:
            painter.setPen(QPen(QColor(255, 180, 90, 110), 4))
            painter.drawPolyline(points)
            painter.setPen(QPen(QColor(255, 200, 120, 235), 2))
            painter.drawPolyline(points)

        # Keys along the path.
        painter.setBrush(QColor("#ffb45a"))
        painter.setPen(QPen(QColor("#2a1d08"), 1))
        for key in self.track.sorted_keys():
            projected = _project(np.array(key.position, np.float32), vp, size)
            if projected is not None:
                painter.drawEllipse(projected, 4, 4)

        current = self.track.evaluate(self.current_time)
        if current is None:
            return
        cam = key_to_camera(current, self.pivot_z)
        self._paint_frustum(painter, vp, size, cam)
        self._paint_transform_gizmo(painter, vp, size, np.array(current.position, np.float32))

    # -- transform gizmo -----------------------------------------------------

    def _axis_scale(self) -> float:
        return max(abs(self.pivot_z) * 0.16 * self.distance_scale, 0.1)

    def _gizmo_handles(self, vp, size, origin: np.ndarray) -> dict[str, QPoint]:
        """Screen positions of each axis tip, for both drawing and hit-testing."""
        length = self._axis_scale()
        handles: dict[str, QPoint] = {}
        centre = _project(origin, vp, size)
        if centre is not None:
            handles["view"] = centre
        for name, direction in (
            ("x", (1.0, 0.0, 0.0)),
            ("y", (0.0, 1.0, 0.0)),
            ("z", (0.0, 0.0, 1.0)),
        ):
            tip = origin + np.array(direction, np.float32) * length
            projected = _project(tip, vp, size)
            if projected is not None:
                handles[name] = projected
        return handles

    def _axis_ring(
        self, origin: np.ndarray, axis: str, segments: int = 72
    ) -> list[np.ndarray]:
        """Points of the rotation circle lying in the plane normal to `axis`."""
        radius = self._axis_scale()
        basis = {
            "x": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            "y": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
            "z": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        }[axis]
        u = np.array(basis[0], np.float32)
        v = np.array(basis[1], np.float32)
        points = []
        for i in range(segments + 1):
            angle = math.tau * i / segments
            points.append(origin + (u * math.cos(angle) + v * math.sin(angle)) * radius)
        return points

    def _ring_screen_radius(self, vp, size, origin: np.ndarray) -> int:
        """Approximate on-screen radius of the rings, for the trackball circle."""
        centre = _project(origin, vp, size)
        if centre is None:
            return 40
        best = 0
        for axis in ("x", "y", "z"):
            for point in self._axis_ring(origin, axis, segments=8):
                projected = _project(point, vp, size)
                if projected is not None:
                    best = max(best, int((projected - centre).manhattanLength() * 0.75))
        return max(28, min(best, max(size.width(), size.height())))

    def _hit_rotate_ring(self, point: QPoint, origin: np.ndarray) -> str | None:
        """Pick the ring whose projected curve passes nearest the cursor."""
        camera = self._editor_camera()
        vp = camera.view_projection(self.width() / max(self.height(), 1))
        size = self.size()
        best, best_distance = None, 14.0
        for axis in ("x", "y", "z"):
            for world in self._axis_ring(origin, axis, segments=48):
                projected = _project(world, vp, size)
                if projected is None:
                    continue
                delta = projected - point
                distance = math.hypot(delta.x(), delta.y())
                if distance < best_distance:
                    best, best_distance = axis, distance
        return best

    def _paint_transform_gizmo(self, painter: QPainter, vp, size, origin) -> None:
        handles = self._gizmo_handles(vp, size, origin)
        centre = handles.get("view")
        if centre is None:
            return
        colours = {axis: QColor(value) for axis, value in AXIS_COLOURS.items()}

        if self.gizmo_mode == "rotate":
            # Real 3D rings, projected — so each one tilts with the view the way
            # Blender's do. Flat screen-space circles gave no sense of which
            # plane an axis actually rotates in.
            eye = np.array(self._editor_camera().position, np.float32)
            for axis in ("x", "y", "z"):
                ring = self._axis_ring(origin, axis)
                front, back = [], []
                origin_distance = float(np.linalg.norm(origin - eye))
                for point in ring:
                    projected = _project(point, vp, size)
                    if projected is None:
                        continue
                    nearer = float(np.linalg.norm(point - eye)) <= origin_distance
                    (front if nearer else back).append(projected)
                colour = colours[axis]
                active = self.active_handle == axis or self.hovered_handle == axis
                # The far half is drawn faint, which is what makes the ring read
                # as a circle in space rather than an outline on the screen.
                faded = QColor(colour)
                faded.setAlpha(70)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                if len(back) > 1:
                    painter.setPen(QPen(faded, 2))
                    painter.drawPolyline(back)
                if len(front) > 1:
                    painter.setPen(QPen(colour, 4 if active else 2))
                    painter.drawPolyline(front)
            # Trackball ring for free rotation, facing the viewer.
            radius = self._ring_screen_radius(vp, size, origin)
            painter.setPen(QPen(QColor(220, 226, 240, 150), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(centre, radius + 12, radius + 12)
            return

        # Move gizmo: an arrow per axis, plus a view-plane handle at the centre.
        for axis in ("x", "y", "z"):
            tip = handles.get(axis)
            if tip is None:
                continue
            active = self.active_handle == axis or self.hovered_handle == axis
            painter.setPen(QPen(colours[axis], 4 if active else 3))
            painter.drawLine(centre, tip)
            painter.setBrush(colours[axis])
            painter.setPen(QPen(colours[axis], 1))
            r = 6 if active else 5
            if self.gizmo_mode == "scale" and getattr(self, "_painting_effect", False):
                painter.drawRect(tip.x() - r, tip.y() - r, 2 * r, 2 * r)   # scale: cubes
            else:
                painter.drawEllipse(tip, r, r)
        centre_active = self.active_handle == "view" or self.hovered_handle == "view"
        painter.setBrush(QColor(235, 240, 250, 90 if centre_active else 40))
        painter.setPen(QPen(QColor(220, 226, 240, 200), 2))
        painter.drawEllipse(centre, 11, 11)

    def _paint_frustum(self, painter: QPainter, vp, size, cam: Camera) -> None:
        eye = np.array(cam.position, np.float32)
        target = np.array(cam.target, np.float32)
        forward = target - eye
        length = float(np.linalg.norm(forward))
        if length < 1e-6:
            return
        forward /= length
        right = np.cross(forward, np.array([0.0, 1.0, 0.0], np.float32))
        if float(np.linalg.norm(right)) < 1e-6:
            right = np.array([1.0, 0.0, 0.0], np.float32)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        reach = max(length * 0.35, 0.3)
        half = math.tan(math.radians(cam.fov_degrees) * 0.5) * reach
        centre = eye + forward * reach
        corners = [
            centre + right * half * 1.4 + up * half,
            centre - right * half * 1.4 + up * half,
            centre - right * half * 1.4 - up * half,
            centre + right * half * 1.4 - up * half,
        ]
        apex = _project(eye, vp, size)
        projected = [_project(c, vp, size) for c in corners]
        if apex is None or any(p is None for p in projected):
            return

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(120, 230, 255, 230), 2))
        painter.drawPolygon(projected)
        painter.setPen(QPen(QColor(120, 230, 255, 150), 1))
        for corner in projected:
            painter.drawLine(apex, corner)
        painter.setBrush(QColor("#78e6ff"))
        painter.setPen(QPen(QColor("#06222b"), 1))
        painter.drawEllipse(apex, 4, 4)

    def _paint_hud(self, painter: QPainter) -> None:
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(8, 11, 17, 190))
        box = QRect(8, self.height() - 26, 430, 18)
        painter.drawRoundedRect(box, 5, 5)
        painter.setPen(QColor("#8e99ad"))
        painter.drawText(
            box.adjusted(6, 0, 0, 0),
            Qt.AlignmentFlag.AlignVCenter,
            f"{self.gizmo_mode.title()} (Space / G R S) · Right drag orbit · "
            "Middle drag pan · Wheel zoom",
        )

    # -- interaction ---------------------------------------------------------

    def _hit_handle(self, point: QPoint) -> str | None:
        if self.track is None or not self.track.keys or not self.show_gizmo:
            return None
        key = self.track.evaluate(self.current_time)
        if key is None:
            return None
        origin = np.array(key.position, np.float32)
        if self.gizmo_mode == "rotate":
            # Rotate picks the ring under the cursor, not an axis tip — the
            # tips are meaningless once the handles are circles.
            return self._hit_rotate_ring(point, origin)
        camera = self._editor_camera()
        vp = camera.view_projection(self.width() / max(self.height(), 1))
        handles = self._gizmo_handles(vp, self.size(), origin)
        centre = handles.get("view")
        if centre is None:
            return None
        if float((centre - point).manhattanLength()) < 14:
            return "view"
        best, best_distance = None, 12.0
        # Pick the complete shaft, not only its endpoint. A generous invisible
        # hit region keeps thin projected axes usable on high-DPI displays.
        px, py = float(point.x()), float(point.y())
        for name in ("x", "y", "z"):
            tip = handles.get(name)
            if tip is None:
                continue
            ax, ay = float(centre.x()), float(centre.y())
            bx, by = float(tip.x()), float(tip.y())
            vx, vy = bx - ax, by - ay
            length2 = vx * vx + vy * vy
            if length2 <= 1e-6:
                continue
            along = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / length2))
            distance = math.hypot(px - (ax + along * vx), py - (ay + along * vy))
            if distance < best_distance:
                best, best_distance = name, distance
        return best

    def set_gizmo_mode(self, mode: str) -> None:
        self.gizmo_mode = mode if mode in ("move", "rotate", "scale") else "move"
        self.update()

    def cycle_gizmo_mode(self) -> None:
        order = ("move", "rotate", "scale")
        self.set_gizmo_mode(order[(order.index(self.gizmo_mode) + 1) % len(order)])

    def keyPressEvent(self, event) -> None:
        # Space cycles move/rotate/scale as in Unreal; G/R/S jump straight there
        # as in Blender, so either muscle memory works. The camera has no size,
        # so for it Scale behaves as Move; only effects get scale handles.
        if event.key() == Qt.Key.Key_Space:
            self.cycle_gizmo_mode()
        elif event.key() == Qt.Key.Key_G:
            self.set_gizmo_mode("move")
        elif event.key() == Qt.Key.Key_R:
            self.set_gizmo_mode("rotate")
        elif event.key() == Qt.Key.Key_S:
            self.set_gizmo_mode("scale")
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:
        self._last = event.position().toPoint()
        self._button = event.button()
        self.setFocus()
        if event.button() == Qt.MouseButton.LeftButton:
            # Handles of the selected effect win over picking a different one,
            # otherwise grabbing an axis that overlaps another widget would
            # silently change the selection instead of transforming.
            self._fx_handle = self._hit_effect_handle(self._last)
            if self._fx_handle is None:
                self._active_effect = self._hit_effect(self._last)
                if self._active_effect is not None:
                    self.selected_effect_id = self._active_effect.id
                    self.effect_moved.emit(self._active_effect.id)
                else:
                    self.active_handle = self._hit_handle(self._last)
            else:
                self.update()
            if self.active_handle:
                self.update()

    def mouseReleaseEvent(self, _event) -> None:
        self._last = None
        self._button = None
        self._active_effect = None
        if self._fx_handle:
            self._fx_handle = None
            self.update()
        if self.active_handle:
            self.active_handle = None
            self.update()

    def _drag_gizmo(self, dx: int, dy: int) -> None:
        """Apply a drag to the camera key at the playhead."""
        if self.track is None:
            return
        key = self.track.nearest(self.current_time, tolerance=0.08)
        created = False
        if key is None:
            evaluated = self.track.evaluate(self.current_time)
            if evaluated is None:
                return
            evaluated.time = self.current_time
            key = self.track.add(evaluated)
            created = True

        if self.gizmo_mode == "rotate":
            # Each ring drives the rotation about its own axis: Y yaws, X
            # pitches, Z rolls — matching the ring the user grabbed.
            if self.active_handle == "y":
                key.yaw += dx * 0.006
            elif self.active_handle == "x":
                key.pitch = float(np.clip(key.pitch - dy * 0.006, -1.5, 1.5))
            elif self.active_handle == "z":
                key.roll += dx * 0.006
            else:
                key.yaw += dx * 0.006
                key.pitch = float(np.clip(key.pitch - dy * 0.006, -1.5, 1.5))
        else:
            # Screen-space drag scaled by viewing distance, so the handle keeps
            # up with the cursor at any zoom level.
            speed = abs(self.pivot_z) * 0.0025 * self.distance_scale
            if self.active_handle in {"x", "y", "z"}:
                camera = self._editor_camera()
                vp = camera.view_projection(self.width() / max(self.height(), 1))
                handles = self._gizmo_handles(
                    vp, self.size(), np.array(key.position, np.float32)
                )
                centre, tip = handles.get("view"), handles.get(self.active_handle)
                if centre is not None and tip is not None:
                    vx, vy = tip.x() - centre.x(), tip.y() - centre.y()
                    length = max(math.hypot(vx, vy), 1e-6)
                    amount = (dx * vx + dy * vy) / length * speed
                    setattr(key, self.active_handle, getattr(key, self.active_handle) + amount)
            else:  # view plane
                key.x += dx * speed
                key.y -= dy * speed
        self.camera_moved.emit()
        if created:
            self.keyframe_added.emit(key.time)
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._last is None:
            hovered = self._hit_handle(event.position().toPoint())
            if hovered != self.hovered_handle:
                self.hovered_handle = hovered
                self.update()
            return
        pos = event.position().toPoint()
        dx = pos.x() - self._last.x()
        dy = pos.y() - self._last.y()
        self._last = pos
        if self.active_handle:
            self._drag_gizmo(dx, dy)
            return
        if self._fx_handle is not None:
            self._drag_effect_handle(dx, dy)
            return
        if self._active_effect is not None:
            speed = abs(self.pivot_z) * 0.0025 * self.distance_scale
            x, y, z = self._active_effect.position
            self._active_effect.position = (x + dx * speed, y - dy * speed, z)
            self.effect_moved.emit(self._active_effect.id)
            self.invalidate()
            return
        if self._button == Qt.MouseButton.MiddleButton:
            scale = abs(self.pivot_z) * 0.002
            self.pan[0] -= dx * scale
            self.pan[1] += dy * scale
        elif self._button == Qt.MouseButton.RightButton:
            self.yaw -= dx * 0.008
            self.pitch = float(np.clip(self.pitch + dy * 0.008, -1.35, 1.35))
        else:
            return
        self.invalidate()

    def wheelEvent(self, event) -> None:
        steps = event.angleDelta().y() / 120.0
        self.distance_scale = float(
            np.clip(self.distance_scale * (0.9**steps), 0.25, 6.0)
        )
        self.invalidate()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._frame = None
