"""Expandable camera keyframe timeline.

Ported from Depth Animator (timeline.py).
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from .animation3d import CameraTrack, Easing
from .effects3d import EffectsTrack

RULER_HEIGHT = 22
ROW_HEIGHT = 25
KEY_RADIUS = 5
MARGIN = 10
LABEL_WIDTH = 92
PROPERTY_ROWS = (
    ("Orbit", "yaw"),
    ("Height", "pitch"),
    ("Dolly", "z"),
    ("Lens", "fov_degrees"),
)


def key_sides(prev, key) -> tuple[str, str]:
    """How the move arrives at and leaves a key: "linear", "ease" or "hold".

    A key's easing shapes the segment AFTER it. Ease In starts that segment
    slowly (so the key's right side is eased); Ease Out ends it slowly (so the
    NEXT key's left side is eased); Hold freezes it."""
    right = "linear"
    if key.easing is Easing.STEP:
        right = "hold"
    elif key.easing in (Easing.EASE_IN, Easing.EASE_IN_OUT):
        right = "ease"
    left = "linear"
    if prev is not None:
        if prev.easing is Easing.STEP:
            left = "hold"
        elif prev.easing in (Easing.EASE_OUT, Easing.EASE_IN_OUT):
            left = "ease"
    return left, right


def key_shape(x: int, y: int, r: int, left: str, right: str) -> list[QPoint]:
    """Keyframe glyph, After Effects style, built from two halves:
    linear = the diamond's point, ease = the hourglass half (flat outside,
    pinched at the centre), hold = a square half."""
    def half(sign: int, kind: str) -> list[QPoint]:
        # Points from the top centre, round the outside, to the bottom centre.
        if kind == "ease":
            return [QPoint(x + sign * r, y - r), QPoint(x, y), QPoint(x + sign * r, y + r)]
        if kind == "hold":
            return [QPoint(x, y - r), QPoint(x + sign * r, y - r),
                    QPoint(x + sign * r, y + r), QPoint(x, y + r)]
        return [QPoint(x, y - r), QPoint(x + sign * r, y), QPoint(x, y + r)]

    right_pts = half(1, right)
    left_pts = list(reversed(half(-1, left)))
    return right_pts + left_pts


class TimelineWidget(QWidget):
    """Scrubbable ruler with a collapsible Camera property group.

    Camera keys currently store a complete pose, so a key appears in every
    property row. The hierarchy nevertheless makes the authored values visible
    and gives the data model room for sparse per-property curves later.
    """

    time_changed = Signal(float)
    key_selected = Signal(float)
    property_selected = Signal(str)
    keys_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.track = CameraTrack()
        self.effects_track = EffectsTrack()
        #: Event times drawn as ticks on the FX row (placed lightning strikes).
        self.markers: list[float] = []
        self.duration = 4.0
        self.fps = 30
        self.time = 0.0
        self.selected: float | None = None
        self.selection: set[float] = set()
        self.selected_property = "camera"
        self.expanded = False
        self._dragging_key: float | None = None
        self._scrubbing = False
        self._marquee_origin: QPoint | None = None
        self._marquee_rect: QRect | None = None
        self._marquee_base_selection: set[float] = set()
        self._update_height()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _rows(self) -> list[tuple[str, str]]:
        rows = [("Camera", "camera")]
        if self.expanded:
            rows.extend(PROPERTY_ROWS)
        rows.append(("FX & Lighting", "effects"))
        return rows

    def _update_height(self) -> None:
        row_count = 2 + (len(PROPERTY_ROWS) if self.expanded else 0)
        height = RULER_HEIGHT + ROW_HEIGHT * row_count + 14
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)
        self.updateGeometry()

    def set_expanded(self, expanded: bool) -> None:
        self.expanded = bool(expanded)
        self._update_height()
        self.update()

    def _usable(self) -> QRect:
        return self.rect().adjusted(LABEL_WIDTH, 0, -MARGIN, 0)

    def _x_for(self, seconds: float) -> int:
        area = self._usable()
        fraction = 0.0 if self.duration <= 0 else seconds / self.duration
        return area.left() + int(round(fraction * max(area.width() - 1, 1)))

    def _time_for(self, x: float) -> float:
        area = self._usable()
        fraction = (x - area.left()) / max(area.width() - 1, 1)
        return max(0.0, min(1.0, fraction)) * self.duration

    def _track_y(self, row: int = 0) -> int:
        return RULER_HEIGHT + row * ROW_HEIGHT + ROW_HEIGHT // 2

    def _row_at(self, y: float) -> int | None:
        row = int((y - RULER_HEIGHT) // ROW_HEIGHT)
        return row if 0 <= row < len(self._rows()) else None

    def set_duration(self, duration: float, fps: int) -> None:
        self.duration = max(0.1, float(duration))
        self.fps = max(1, int(fps))
        self.time = min(self.time, self.duration)
        self.update()

    def set_time(self, seconds: float) -> None:
        seconds = max(0.0, min(self.duration, float(seconds)))
        if abs(seconds - self.time) > 1e-6:
            self.time = seconds
            self.update()
            self.time_changed.emit(self.time)

    def set_track(self, track: CameraTrack) -> None:
        self.track = track
        if self.selected is not None and track.nearest(self.selected, 1e-3) is None:
            self.selected = None
        self.selection = {
            time for time in self.selection if track.nearest(time, 1e-3) is not None
        }
        self.update()

    def set_effects_track(self, track: EffectsTrack) -> None:
        self.effects_track = track
        self.update()

    def set_markers(self, times) -> None:
        self.markers = sorted(float(t) for t in times)
        self.update()

    def select_property(self, name: str) -> None:
        if name not in {row[1] for row in self._rows()}:
            return
        self.selected_property = name
        if name != "camera" and not self.expanded:
            self.set_expanded(True)
        self.update()

    def snap(self, seconds: float) -> float:
        frame = round(seconds * self.fps)
        return max(0.0, min(self.duration, frame / self.fps))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0f141d"))
        area = self._usable()
        self._paint_ruler(painter, area)
        self._paint_rows(painter, area)
        self._paint_playhead(painter, area)
        if self._marquee_rect is not None:
            painter.setPen(QPen(QColor("#9b8cff"), 1))
            painter.setBrush(QColor(124, 105, 255, 45))
            painter.drawRect(self._marquee_rect)

    def _paint_ruler(self, painter: QPainter, area: QRect) -> None:
        painter.setFont(QFont("Segoe UI", 7))
        for step in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0):
            if self.duration / step <= 12:
                break
        seconds = 0.0
        bottom = RULER_HEIGHT + ROW_HEIGHT * len(self._rows())
        while seconds <= self.duration + 1e-6:
            x = self._x_for(seconds)
            painter.setPen(QPen(QColor("#2c3547"), 1))
            painter.drawLine(x, RULER_HEIGHT - 6, x, bottom)
            painter.setPen(QColor("#7f8aa3"))
            painter.drawText(
                QRect(x - 26, 2, 52, RULER_HEIGHT - 8),
                Qt.AlignmentFlag.AlignCenter,
                f"{seconds:g}s",
            )
            seconds += step

    def _paint_rows(self, painter: QPainter, area: QRect) -> None:
        for row_index, (label, property_name) in enumerate(self._rows()):
            keys = (
                sorted(self.effects_track.keys, key=lambda key: key.time)
                if property_name == "effects" else self.track.sorted_keys()
            )
            top = RULER_HEIGHT + row_index * ROW_HEIGHT
            rect = QRect(area.left(), top, area.width(), ROW_HEIGHT - 1)
            selected_row = self.selected_property == property_name
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#1a2030") if selected_row else QColor("#151b26"))
            painter.drawRoundedRect(rect, 4, 4)
            painter.setPen(QColor("#d5d9e4") if selected_row else QColor("#8792a8"))
            indent = 18 if row_index else 0
            prefix = ("▾ " if self.expanded else "▸ ") if row_index == 0 else ""
            painter.drawText(
                QRect(8 + indent, top, LABEL_WIDTH - 12 - indent, ROW_HEIGHT),
                Qt.AlignmentFlag.AlignVCenter,
                prefix + label,
            )
            y = self._track_y(row_index)
            if property_name == "effects" and self.markers:
                # Lightning strikes: a yellow bolt tick at each placed moment.
                painter.setPen(QPen(QColor("#ffd24a"), 2))
                for t in self.markers:
                    x = self._x_for(t)
                    painter.drawLine(x, top + 3, x - 3, y)
                    painter.drawLine(x - 3, y, x + 2, y)
                    painter.drawLine(x + 2, y, x - 1, top + ROW_HEIGHT - 4)
            for index, key in enumerate(keys):
                x = self._x_for(key.time)
                if index + 1 < len(keys):
                    next_x = self._x_for(keys[index + 1].time)
                    colour = QColor("#41b9c6") if property_name == "effects" else (
                        QColor("#4a5570") if key.easing is Easing.STEP else QColor("#7464e8")
                    )
                    painter.setPen(QPen(colour, 2 if row_index else 3))
                    painter.drawLine(x, y, next_x, y)
                chosen = property_name != "effects" and any(abs(key.time - time) < 1e-6 for time in self.selection)
                painter.setPen(QPen(QColor("#ffffff") if chosen else QColor("#b7adff"), 1))
                painter.setBrush(QColor("#8b79ff") if chosen else QColor("#2a2450"))
                radius = KEY_RADIUS if row_index == 0 else KEY_RADIUS - 1
                if property_name == "effects":
                    left = right = "linear"
                else:
                    prev = keys[index - 1] if index > 0 else None
                    left, right = key_sides(prev, key)
                painter.drawPolygon(key_shape(x, y, radius + 1, left, right))

    def _paint_playhead(self, painter: QPainter, area: QRect) -> None:
        x = self._x_for(self.time)
        bottom = RULER_HEIGHT + ROW_HEIGHT * len(self._rows())
        painter.setPen(QPen(QColor("#ff7a59"), 2))
        painter.drawLine(x, 2, x, bottom)
        painter.setBrush(QColor("#ff7a59"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon([QPoint(x - 5, 2), QPoint(x + 5, 2), QPoint(x, 10)])
        painter.setPen(QColor("#e6e9f0"))
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.DemiBold))
        frame = int(round(self.time * self.fps))
        painter.drawText(
            QRect(area.left(), bottom, area.width(), 14),
            Qt.AlignmentFlag.AlignRight,
            f"{self.time:.2f}s · frame {frame}",
        )

    def _key_at(self, pos) -> tuple[float, int] | None:
        row = self._row_at(pos.y())
        if row is None or abs(pos.y() - self._track_y(row)) > KEY_RADIUS * 2.2:
            return None
        if self._rows()[row][1] == "effects":
            return None
        for key in self.track.sorted_keys():
            if abs(self._x_for(key.time) - pos.x()) <= KEY_RADIUS * 1.8:
                return key.time, row
        return None

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        self.setFocus()
        row = self._row_at(pos.y())
        if pos.x() < LABEL_WIDTH and row is not None:
            if row == 0:
                self.set_expanded(not self.expanded)
            else:
                self.selected_property = self._rows()[row][1]
                self.property_selected.emit(self.selected_property)
                self.update()
            return
        hit = self._key_at(pos)
        if hit is not None:
            time, row = hit
            additive = bool(
                event.modifiers()
                & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier)
            )
            if additive:
                if time in self.selection:
                    self.selection.remove(time)
                else:
                    self.selection.add(time)
            else:
                self.selection = {time}
            self.selected = time
            self.selected_property = self._rows()[row][1]
            self._dragging_key = time
            self.set_time(time)
            self.key_selected.emit(time)
            self.property_selected.emit(self.selected_property)
            self.update()
            return
        if pos.y() < RULER_HEIGHT:
            self._scrubbing = True
            self.set_time(self._time_for(pos.x()))
        else:
            self._marquee_origin = pos.toPoint()
            self._marquee_rect = QRect(self._marquee_origin, self._marquee_origin)
            additive = bool(
                event.modifiers()
                & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier)
            )
            self._marquee_base_selection = set(self.selection) if additive else set()
            if not additive:
                self.selection.clear()
                self.selected = None
            self.update()

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        if self._dragging_key is not None:
            key = self.track.nearest(self._dragging_key, tolerance=1e-3)
            if key is not None:
                previous_time = key.time
                key.time = self.snap(self._time_for(pos.x()))
                self._dragging_key = key.time
                self.selected = key.time
                self.selection.discard(previous_time)
                self.selection.add(key.time)
                self.track.keys.sort(key=lambda item: item.time)
                self.set_time(key.time)
                self.keys_changed.emit()
                self.update()
            return
        if self._scrubbing:
            self.set_time(self._time_for(pos.x()))
            return
        if self._marquee_origin is not None:
            self._marquee_rect = QRect(
                self._marquee_origin, pos.toPoint()
            ).normalized()
            rows = self._rows()
            # Recompute from the selection that existed at mouse-down. Using
            # the previous move's result made a shrinking marquee unable to
            # deselect keys it no longer enclosed.
            selected = set(self._marquee_base_selection)
            for row_index in range(len(rows)):
                y = self._track_y(row_index)
                for key in self.track.sorted_keys():
                    if self._marquee_rect.contains(QPoint(self._x_for(key.time), y)):
                        selected.add(key.time)
            self.selection = selected
            self.selected = min(selected) if selected else None
            self.update()
            return
        self.setCursor(
            Qt.CursorShape.SizeHorCursor if self._key_at(pos) else Qt.CursorShape.PointingHandCursor
        )

    def mouseReleaseEvent(self, _event) -> None:
        if self._marquee_origin is not None and self._marquee_rect is not None:
            if self._marquee_rect.width() < 4 and self._marquee_rect.height() < 4:
                self.set_time(self._time_for(self._marquee_origin.x()))
            elif self.selected is not None:
                self.key_selected.emit(self.selected)
        self._dragging_key = None
        self._scrubbing = False
        self._marquee_origin = None
        self._marquee_rect = None
        self._marquee_base_selection.clear()
        self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            removed = False
            for time in list(self.selection):
                removed = self.track.remove_at(time, tolerance=1e-3) or removed
            if removed:
                self.selection.clear()
                self.selected = None
                self.keys_changed.emit()
                self.update()
            return
        super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        hit = self._key_at(event.position())
        if hit is not None and self.track.remove_at(hit[0], tolerance=1e-3):
            self.selected = None
            self.selection.discard(hit[0])
            self.keys_changed.emit()
            self.update()
