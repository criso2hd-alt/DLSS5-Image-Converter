"""First-run introduction and the in-product spotlight tour.

The real setup work lives in :mod:`bootstrap`, :mod:`discovery`, and
:mod:`runtime`.  This module is deliberately only presentation: it explains
those steps, then points at the real widgets the person will use.  Keeping the
overlay here stops an already-large main window from accumulating custom paint
and placement code.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QPoint,
    QPointF,
    QPropertyAnimation,
    QRect,
    Qt,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .widgets import FONT_DISPLAY, FONT_MONO, apply_font


def _css_cubic_bezier(x1: float, y1: float, x2: float, y2: float) -> QEasingCurve:
    """Build the same timing curves used by the browser prototype's CSS."""
    curve = QEasingCurve(QEasingCurve.Type.BezierSpline)
    curve.addCubicBezierSegment(QPointF(x1, y1), QPointF(x2, y2), QPointF(1.0, 1.0))
    return curve


def probe_succeeded(report: str) -> bool:
    """Whether the native probe proved the whole neural path is live."""
    required = (
        "dlss_available: 1",
        "neural_addon_loaded: 1",
        "reshade_proxy_loaded: 1",
        "dlssnr_module_loaded: 1",
        "test_evaluation: ok",
    )
    return all(line in report for line in required)


@dataclass(frozen=True)
class TourStep:
    """One piece of the real window to explain."""

    title: str
    body: str
    target: QWidget


class ComparisonPreview(QWidget):
    """A small, honest visual explanation of Preserve rather than a fake result.

    The right half is the bundled source at full detail; the left is the same
    image downsampled and restored, standing in for the texture DLAA commonly
    softens.  It is an illustration in the introduction, never presented as an
    actual conversion or used by the pipeline.
    """

    def __init__(
        self, image_path: Path, accent: str = "#42dcf5",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._sharp = QPixmap(str(image_path))
        self._accent = QColor(accent)
        self._split = 0.18
        self.setMinimumHeight(180)
        self.setMaximumHeight(230)
        # Same slow back-and-forth reveal as the mockup's introduction: the
        # cyan divider travels from 18% to 88% and back over 4.6 seconds.
        self._wipe = QVariantAnimation(self)
        self._wipe.setStartValue(0.18)
        self._wipe.setKeyValueAt(0.5, 0.88)
        self._wipe.setEndValue(0.18)
        self._wipe.setDuration(4600)
        self._wipe.setEasingCurve(_css_cubic_bezier(0.42, 0.0, 0.58, 1.0))
        self._wipe.setLoopCount(-1)
        self._wipe.valueChanged.connect(self._move_split)
        self._wipe.start()

    def _move_split(self, value: object) -> None:
        self._split = float(value)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        area = self.rect().adjusted(1, 1, -1, -1)
        if self._sharp.isNull():
            painter.fillRect(area, QColor("#0a111c"))
            painter.setPen(QColor("#8ea1ba"))
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, "Fine-detail preview")
            return

        sharp = self._sharp.scaled(
            area.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        # A small round trip removes high frequencies without inventing a
        # second image.  The comparison therefore teaches the right concept
        # while remaining clearly illustrative.
        tiny = sharp.scaled(
            max(1, sharp.width() // 5), max(1, sharp.height() // 5),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        soft = tiny.scaled(
            sharp.size(), Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        x = area.x() + (area.width() - sharp.width()) // 2
        y = area.y() + (area.height() - sharp.height()) // 2
        split = area.left() + int(area.width() * self._split)
        painter.save()
        painter.setClipRect(area)
        painter.drawPixmap(x, y, soft)
        painter.setClipRect(QRect(split, area.y(), area.right() - split + 1, area.height()))
        painter.drawPixmap(x, y, sharp)
        painter.restore()

        painter.setPen(QPen(self._accent, 2))
        painter.drawLine(split, area.top(), split, area.bottom())
        painter.setPen(QPen(QColor("#29425d"), 1))
        painter.drawRoundedRect(area, 9, 9)

        painter.setPen(QColor("#d8e5f4"))
        painter.drawText(area.adjusted(12, 0, 0, -10),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
                         "SOURCE · SOFTENED")
        painter.drawText(area.adjusted(0, 0, -12, -10),
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
                         "DETAIL PRESERVED")


class FirstConversionDialog(QDialog):
    """Bridge from successful setup into the hands-on tour."""

    def __init__(
        self, image_path: Path, parent: QWidget | None = None,
        palette: dict[str, str] | None = None,
    ) -> None:
        super().__init__(parent)
        colours = palette or {
            "panel_hi": "#101b2c", "base": "#0d1625", "line": "#29405d",
            "ink": "#f5f9ff", "ink_dim": "#9bacc2", "signal": "#42dcf5",
            "signal_light": "#69eaff", "signal_deep": "#159bc3",
            "on_accent": "#041017",
        }
        self.setWindowTitle("First-run setup")
        self.setModal(True)
        self.setMinimumWidth(590)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 24, 26, 24)
        layout.setSpacing(13)

        eyebrow = QLabel("READY  ·  STEP 3 OF 3")
        eyebrow.setObjectName("onboardingEyebrow")
        apply_font(eyebrow, family=FONT_MONO, size=8.5, spacing=2.0, caps=True)
        layout.addWidget(eyebrow)

        title = QLabel("Let's do your first conversion.")
        title.setObjectName("onboardingTitle")
        apply_font(title, family=FONT_DISPLAY, size=19)
        layout.addWidget(title)

        body = QLabel(
            "Choose an image and the short tour will show you where to tune the "
            "neural look, preserve detail, compare the result, and convert."
        )
        body.setObjectName("onboardingBody")
        body.setWordWrap(True)
        layout.addWidget(body)
        layout.addWidget(ComparisonPreview(image_path, colours["signal"]))

        buttons = QHBoxLayout()
        self.start_button = QPushButton("Choose an image && start")
        self.start_button.setObjectName("onboardingPrimary")
        self.skip_button = QPushButton("Skip tutorial")
        self.skip_button.setObjectName("secondary")
        self.start_button.clicked.connect(self.accept)
        self.skip_button.clicked.connect(self.reject)
        buttons.addWidget(self.start_button, 1)
        buttons.addWidget(self.skip_button)
        layout.addLayout(buttons)

        self.setStyleSheet(f"""
            QDialog {{ background: {colours['panel_hi']}; color: {colours['ink']}; }}
            QLabel#onboardingEyebrow {{ color: {colours['signal']}; }}
            QLabel#onboardingTitle {{ color: {colours['ink']}; font-weight: 700; }}
            QLabel#onboardingBody {{ color: {colours['ink_dim']}; font-size: 12px; }}
            QPushButton {{ min-height: 36px; border-radius: 8px; padding: 0 16px; }}
            QPushButton#onboardingPrimary {{
                color: {colours['on_accent']}; font-weight: 700;
                border: 1px solid {colours['signal_light']};
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {colours['signal_light']}, stop:1 {colours['signal_deep']});
            }}
            QPushButton#secondary {{
                color: {colours['ink_dim']}; border: 1px solid {colours['line']};
                background: {colours['base']};
            }}
        """)


class SpotlightOverlay(QWidget):
    """Dim the window and teach against one real control at a time."""

    finished = Signal()
    skipped = Signal()
    step_changed = Signal(int)

    def __init__(
        self, parent: QWidget, steps: list[TourStep],
        palette: dict[str, str] | None = None,
    ) -> None:
        super().__init__(parent)
        colours = palette or {
            "panel_hi": "#101b2c", "base": "#0d1625", "line": "#29405d",
            "ink": "#f3f8ff", "ink_dim": "#a8b7c9", "ink_faint": "#61748d",
            "signal": "#42dcf5", "signal_light": "#69eaff",
            "signal_deep": "#159bc3", "on_accent": "#041017",
        }
        self._accent = QColor(colours["signal"])
        self._steps = steps
        self._index = 0
        self._target = QRect()
        self._veil_opacity = 0.0
        self._pulse_phase = 0.0
        self.setObjectName("spotlightOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self.card = QFrame(self)
        self.card.setObjectName("tourCard")
        self.card.setFixedWidth(310)
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(18, 16, 18, 15)
        card_layout.setSpacing(9)

        self.eyebrow = QLabel()
        self.eyebrow.setObjectName("tourEyebrow")
        apply_font(self.eyebrow, family=FONT_MONO, size=8, spacing=1.8, caps=True)
        self.title = QLabel()
        self.title.setObjectName("tourTitle")
        apply_font(self.title, family=FONT_DISPLAY, size=13)
        self.body = QLabel()
        self.body.setObjectName("tourBody")
        self.body.setWordWrap(True)
        card_layout.addWidget(self.eyebrow)
        card_layout.addWidget(self.title)
        card_layout.addWidget(self.body)

        footer = QHBoxLayout()
        self.counter = QLabel()
        self.counter.setObjectName("tourCounter")
        self.skip_button = QPushButton("Skip tour")
        self.skip_button.setObjectName("tourSecondary")
        self.next_button = QPushButton("Next")
        self.next_button.setObjectName("tourPrimary")
        self.skip_button.clicked.connect(self._skip)
        self.next_button.clicked.connect(self._next)
        footer.addWidget(self.counter)
        footer.addStretch(1)
        footer.addWidget(self.skip_button)
        footer.addWidget(self.next_button)
        card_layout.addLayout(footer)

        self._card_opacity = QGraphicsOpacityEffect(self.card)
        self.card.setGraphicsEffect(self._card_opacity)
        self._card_opacity.setOpacity(0.0)

        # The prototype uses 320 ms for the spotlight and 300 ms for the card.
        # Both keep their identity while travelling to the next control; a cut
        # or a fade-out/fade-in loses the delightful "guided camera" feeling.
        self._spot_animation = QPropertyAnimation(self, b"spotlightRect", self)
        self._spot_animation.setDuration(320)
        self._spot_animation.setEasingCurve(_css_cubic_bezier(0.25, 0.1, 0.25, 1.0))
        self._card_animation = QPropertyAnimation(self.card, b"geometry", self)
        self._card_animation.setDuration(300)
        self._card_animation.setEasingCurve(_css_cubic_bezier(0.25, 0.1, 0.25, 1.0))
        self._fade_animation = QPropertyAnimation(self, b"veilOpacity", self)
        self._fade_animation.setDuration(350)
        self._fade_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._card_fade = QPropertyAnimation(self._card_opacity, b"opacity", self)
        self._card_fade.setDuration(300)
        self._card_fade.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._pulse = QVariantAnimation(self)
        self._pulse.setStartValue(0.0)
        self._pulse.setEndValue(math.tau)
        self._pulse.setDuration(1800)
        self._pulse.setLoopCount(-1)
        self._pulse.valueChanged.connect(self._set_pulse)

        self.setStyleSheet(f"""
            QFrame#tourCard {{
                background: {colours['panel_hi']}; border: 1px solid {colours['line']};
                border-radius: 12px;
            }}
            QLabel#tourEyebrow {{ color: {colours['signal']}; }}
            QLabel#tourTitle {{ color: {colours['ink']}; font-weight: 700; }}
            QLabel#tourBody {{ color: {colours['ink_dim']}; font-size: 11px; }}
            QLabel#tourCounter {{
                color: {colours['ink_faint']}; font-family: "IBM Plex Mono";
            }}
            QPushButton {{ min-height: 30px; border-radius: 7px; padding: 0 11px; }}
            QPushButton#tourSecondary {{
                color: {colours['ink_dim']}; background: {colours['base']};
                border: 1px solid {colours['line']};
            }}
            QPushButton#tourPrimary {{
                color: {colours['on_accent']}; font-weight: 700;
                border: 1px solid {colours['signal_light']};
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {colours['signal_light']}, stop:1 {colours['signal_deep']});
            }}
        """)
        self._show_step()

    @property
    def step_index(self) -> int:
        return self._index

    def start(self) -> None:
        self.setGeometry(self.parentWidget().rect())
        self.raise_()
        self.show()
        self._show_step(animate=False)
        self._fade_animation.setStartValue(0.0)
        self._fade_animation.setEndValue(1.0)
        self._fade_animation.start()
        self._card_fade.setStartValue(0.0)
        self._card_fade.setEndValue(1.0)
        self._card_fade.start()
        self._pulse.start()

    def _show_step(self, *, animate: bool = True) -> None:
        if not self._steps:
            self._finish()
            return
        step = self._steps[self._index]
        self.eyebrow.setText(f"TUTORIAL  ·  {self._index + 1} / {len(self._steps)}")
        self.title.setText(f"{self._index + 1} · {step.title}")
        self.body.setText(step.body)
        self.counter.setText(f"Step {self._index + 1} of {len(self._steps)}")
        self.next_button.setText("Done" if self._index == len(self._steps) - 1 else "Next")
        self.card.adjustSize()
        self.step_changed.emit(self._index)
        self._sync_geometry(animate=animate)
        self.update()

    def refresh_target(self) -> None:
        """Recalculate after a scroll area reveals the next highlighted card."""
        self._sync_geometry(animate=True)
        self.update()

    def _sync_geometry(self, *, animate: bool = False) -> None:
        if not self._steps:
            return
        target = self._steps[self._index].target
        top_left = self.mapFromGlobal(target.mapToGlobal(QPoint(0, 0)))
        raw = QRect(top_left, target.size()).adjusted(-7, -7, 7, 7)
        target_rect = raw.intersected(self.rect().adjusted(5, 5, -5, -5))

        margin = 16
        width = self.card.width()
        height = self.card.sizeHint().height()
        right_x = target_rect.right() + margin
        left_x = target_rect.left() - width - margin
        if right_x + width <= self.width() - margin:
            x = right_x
        elif left_x >= margin:
            x = left_x
        else:
            x = max(margin, min(self.width() - width - margin,
                                target_rect.center().x() - width // 2))

        below_y = target_rect.bottom() + margin
        above_y = target_rect.top() - height - margin
        if below_y + height <= self.height() - margin:
            y = below_y
        elif above_y >= margin:
            y = above_y
        else:
            y = max(margin, min(self.height() - height - margin,
                                target_rect.center().y() - height // 2))
        card_rect = QRect(x, y, width, height)
        if animate and not self._target.isEmpty():
            self._spot_animation.stop()
            self._spot_animation.setStartValue(self._target)
            self._spot_animation.setEndValue(target_rect)
            self._spot_animation.start()
            self._card_animation.stop()
            self._card_animation.setStartValue(self.card.geometry())
            self._card_animation.setEndValue(card_rect)
            self._card_animation.start()
        else:
            self._target = target_rect
            self.card.setGeometry(card_rect)

    def _get_spotlight_rect(self) -> QRect:
        return self._target

    def _set_spotlight_rect(self, rect: QRect) -> None:
        self._target = rect
        self.update()

    spotlightRect = Property(QRect, _get_spotlight_rect, _set_spotlight_rect)

    def _get_veil_opacity(self) -> float:
        return self._veil_opacity

    def _set_veil_opacity(self, value: float) -> None:
        self._veil_opacity = float(value)
        self.update()

    veilOpacity = Property(float, _get_veil_opacity, _set_veil_opacity)

    def _set_pulse(self, value: object) -> None:
        self._pulse_phase = float(value)
        self.update()

    def _next(self) -> None:
        if self._index >= len(self._steps) - 1:
            self._finish()
            return
        self._index += 1
        self._show_step()

    def _skip(self) -> None:
        self._stop_animations()
        self.hide()
        self.skipped.emit()
        self.deleteLater()

    def _finish(self) -> None:
        self._stop_animations()
        self.hide()
        self.finished.emit()
        self.deleteLater()

    def _stop_animations(self) -> None:
        """Never leave a native animation callback queued after the overlay dies."""
        for animation in (
            self._spot_animation,
            self._card_animation,
            self._fade_animation,
            self._card_fade,
            self._pulse,
        ):
            animation.stop()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt name
        self._sync_geometry()
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        shade = QPainterPath()
        shade.setFillRule(Qt.FillRule.OddEvenFill)
        shade.addRect(self.rect())
        if not self._target.isEmpty():
            shade.addRoundedRect(self._target, 10, 10)
        painter.fillPath(shade, QColor(2, 7, 14, int(205 * self._veil_opacity)))
        if not self._target.isEmpty():
            # The mockup's 1.8 s breathing glow makes the highlighted control
            # feel live even while the user is reading. Multiple translucent
            # strokes approximate its CSS box-shadow without a bitmap effect.
            pulse = (math.sin(self._pulse_phase) + 1.0) * 0.5
            glow = QColor(self._accent)
            glow.setAlpha(int((26 + 30 * pulse) * self._veil_opacity))
            painter.setPen(QPen(glow, 9 + 7 * pulse))
            painter.drawRoundedRect(self._target.adjusted(-3, -3, 3, 3), 13, 13)
            painter.setPen(QPen(self._accent, 2))
            painter.drawRoundedRect(self._target, 10, 10)
