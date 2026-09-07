"""Reusable pieces of the interface."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QPointF,
    QProcess,
    QRect,
    QRectF,
    QSize,
    Qt,
    QTimer,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QImage,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QProgressBar,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".exr",
             ".hdr", ".jxr", ".wdp", ".hdp"}

#: The bundled type families (registered in app._load_bundled_fonts). Named here
#: so the whole UI reaches for the same three faces the mockup uses.
FONT_DISPLAY = "Archivo SemiBold"          # wordmark, card titles, Convert
FONT_SANS = "IBM Plex Sans"                # body: labels, chips, buttons
FONT_MONO = "IBM Plex Mono"                # small-caps tags, values, readouts


def apply_font(
    widget,
    *,
    family: str | None = None,
    size: float | None = None,
    weight=None,
    spacing: float | None = None,
    caps: bool = False,
) -> None:
    """Set face, size, weight, letter-spacing and caps on a widget in one call.

    Letter-spacing is the reason this exists: Qt Style Sheets silently ignore the
    ``letter-spacing`` property, so the mockup's spaced-out caps — the wordmark,
    the card titles, the RUNTIME pill — can only be had by setting it on the
    QFont here, per widget.
    """
    from PySide6.QtGui import QFont

    font = widget.font()
    if family:
        font.setFamily(family)
    if size is not None:
        font.setPointSizeF(size)
    if weight is not None:
        font.setWeight(weight)
    if spacing is not None:
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    if caps:
        font.setCapitalization(QFont.Capitalization.AllUppercase)
    widget.setFont(font)


def stage_placeholder(icon: str, title: str, subtitle: str = "") -> QWidget:
    """A centred icon + title + hint, for the empty state of a preview stage.

    So the Video and Sequence tabs read as "your clip / frames appear here"
    rather than a black void — the same welcoming empty state the single-image
    drop zone gives, in the same visual language.
    """
    holder = QWidget()
    box = QVBoxLayout(holder)
    box.setAlignment(Qt.AlignmentFlag.AlignCenter)
    box.setSpacing(8)
    glyph = QLabel(icon)
    glyph.setAlignment(Qt.AlignmentFlag.AlignCenter)
    apply_font(glyph, size=34)
    head = QLabel(title)
    head.setObjectName("phTitle")
    head.setAlignment(Qt.AlignmentFlag.AlignCenter)
    box.addWidget(glyph)
    box.addWidget(head)
    if subtitle:
        sub = QLabel(subtitle)
        sub.setObjectName("hint")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box.addWidget(sub)
    return holder


class ModuleCard(QFrame):
    """A titled card: the title sits *inside* a header strip with a divider under
    it, then a padded body — the mockup's ``.mod`` block, not a QGroupBox.

    QGroupBox hangs its title on the border and cannot draw the divider or carry
    a right-aligned tag, which is most of why the sidebar read as a plain form
    rather than the mockup's instrument panel. The optional ``tag`` is the little
    mono caption on the right of the header (RENODX · DLAA, NEW).
    """

    #: Emitted when a checkable card's header toggle changes.
    toggled = Signal(bool)

    def __init__(
        self, title: str, tag: str = "", checkable: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("modCard")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        head = QFrame()
        head.setObjectName("modHead")
        head_row = QHBoxLayout(head)
        head_row.setContentsMargins(16, 12, 16, 11)
        head_row.setSpacing(9)

        self._check: "QCheckBox | None" = None
        if checkable:
            # The title itself is the on/off control, like the old checkable
            # group box — a checkbox whose label is the card title.
            from PySide6.QtWidgets import QCheckBox

            self._check = QCheckBox(title)
            self._check.setObjectName("modTitle")
            apply_font(self._check, family=FONT_DISPLAY, size=9.5, spacing=2.4, caps=True)
            self._check.toggled.connect(self._on_toggle)
            head_row.addWidget(self._check)
            self.title_label = self._check
        else:
            self.title_label = QLabel(title)
            self.title_label.setObjectName("modTitle")
            apply_font(self.title_label, family=FONT_DISPLAY, size=9.5, spacing=2.4, caps=True)
            head_row.addWidget(self.title_label)
        head_row.addStretch(1)
        if tag:
            tag_label = QLabel(tag)
            tag_label.setObjectName("modTag")
            apply_font(tag_label, family=FONT_MONO, size=7.5, spacing=1.4, caps=True)
            head_row.addWidget(tag_label)
        outer.addWidget(head)

        self._body_widget = QWidget()
        self.body = QVBoxLayout(self._body_widget)
        self.body.setContentsMargins(16, 14, 16, 15)
        self.body.setSpacing(13)
        outer.addWidget(self._body_widget)

    def add(self, widget: QWidget) -> QWidget:
        self.body.addWidget(widget)
        return widget

    def add_layout(self, layout) -> "object":
        self.body.addLayout(layout)
        return layout

    # -- checkable proxy (so a card can stand in for a checkable QGroupBox) --

    def _on_toggle(self, on: bool) -> None:
        # Grey the body when off, exactly like a checkable group box.
        self._body_widget.setEnabled(on)
        self.toggled.emit(on)

    def setChecked(self, on: bool) -> None:  # noqa: N802 - Qt-style name
        if self._check is not None:
            self._check.setChecked(on)
            # Sync the body even when the state did not change (so an initial
            # unchecked card greys its body without relying on a toggle signal).
            self._body_widget.setEnabled(on)

    def isChecked(self) -> bool:  # noqa: N802 - Qt-style name
        return self._check.isChecked() if self._check is not None else True


def to_qimage_u8(image_rgb: np.ndarray) -> QImage:
    """8-bit RGB to a QImage that owns its buffer.

    The copy at the end is not redundant: QImage wraps the numpy buffer without
    taking a reference, so returning an uncopied view hands Qt a pointer that is
    freed as soon as the temporary array goes out of scope.
    """
    data = np.ascontiguousarray(image_rgb, dtype=np.uint8)
    height, width = data.shape[:2]
    return QImage(data.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()


def to_qimage(image_rgb: np.ndarray) -> QImage:
    """0..1 float RGB to an 8-bit QImage that owns its buffer."""
    return to_qimage_u8(np.clip(image_rgb, 0.0, 1.0) * 255.0)


def first_supported(paths: Iterable[str]) -> Path | None:
    for candidate in paths:
        path = Path(candidate)
        if path.suffix.lower() in SUPPORTED and path.is_file():
            return path
    return None


class DropZone(QFrame):
    """Drag-and-drop target that doubles as the empty state."""

    opened = Signal(Path)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("dropZone")
        self.setMinimumHeight(240)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label = QLabel("Drop a photo here, or paste one")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint = QLabel("PNG · JPEG · TIFF · EXR — Ctrl+V, or click to browse")
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setObjectName("hint")
        layout.addWidget(self._label)
        layout.addWidget(self._hint)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt name
        urls = event.mimeData().urls()
        if first_supported(url.toLocalFile() for url in urls):
            event.acceptProposedAction()
            self.setProperty("hovering", True)
            self._restyle()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802 - Qt name
        self.setProperty("hovering", False)
        self._restyle()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt name
        self.setProperty("hovering", False)
        self._restyle()
        path = first_supported(url.toLocalFile() for url in event.mimeData().urls())
        if path is not None:
            self.opened.emit(path)
            event.acceptProposedAction()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt name
        from PySide6.QtWidgets import QFileDialog

        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Choose an image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.exr *.hdr *.jxr *.wdp *.hdp)",
        )
        if chosen:
            self.opened.emit(Path(chosen))

    def _restyle(self) -> None:
        # Qt does not re-evaluate property selectors on its own.
        self.style().unpolish(self)
        self.style().polish(self)


def desaturated(pixmap: QPixmap) -> QPixmap:
    """A greyscale copy, built once and reused while a sweep runs."""
    return QPixmap.fromImage(
        pixmap.toImage().convertToFormat(QImage.Format.Format_Grayscale8)
    )


def paint_sweep(
    painter: QPainter,
    rect: QRectF,
    colour: QPixmap,
    grey: QPixmap,
    progress: float,
    highlight: QColor,
) -> None:
    """Grey above the line, full colour below it, sweeping upward.

    Shared by every view that can be recomputed - the single image, the
    before/after wipe, and each pane of the style comparison - so "this is
    being worked on" looks the same wherever it appears. A different treatment
    per view would read as different states rather than the same one.
    """
    painter.drawPixmap(rect, grey, QRectF(grey.rect()))
    # Knock the grey back as well as desaturating it. Grey alone at full
    # brightness reads as a deliberate black-and-white treatment rather than
    # as something unfinished.
    painter.fillRect(rect, QColor(6, 8, 12, 90))

    line = rect.bottom() - rect.height() * progress
    if progress > 0.0:
        painter.save()
        painter.setClipRect(QRectF(rect.left(), line, rect.width(), rect.bottom() - line))
        painter.drawPixmap(rect, colour, QRectF(colour.rect()))
        painter.restore()

    # A soft edge on the boundary, so the sweep reads as a moving front rather
    # than as an image cut in half.
    if 0.0 < progress < 1.0:
        glow = QLinearGradient(0.0, line - 26.0, 0.0, line + 2.0)
        faded = QColor(highlight)
        faded.setAlpha(0)
        edge = QColor(highlight)
        edge.setAlpha(120)
        glow.setColorAt(0.0, faded)
        glow.setColorAt(1.0, edge)
        painter.fillRect(QRectF(rect.left(), line - 26.0, rect.width(), 28.0), QBrush(glow))
        painter.setPen(QPen(highlight, 1))
        painter.drawLine(QPointF(rect.left(), line), QPointF(rect.right(), line))


# -- on-image overlays --------------------------------------------------------
#
# The prototype floats a thin instrument layer over the picture: SOURCE / DLSS
# pills in the top corners, a glowing divider with a round grip, and a
# resolution + zoom readout bottom-right. None of it is reachable from QSS - the
# canvas views draw their own pixels - so it is reproduced here in the painter,
# matching the mockup's mono caps, translucent plates and cyan accent.

#: Overlay plate colours, straight from the mockup (--tag background, --line).
_OVERLAY_PLATE = QColor(6, 10, 18, 190)
_OVERLAY_LINE = QColor(35, 48, 74)
_OVERLAY_INK = QColor(201, 213, 230)      # --ink-dim, the non-accent pill text
_OVERLAY_READ = QColor(147, 162, 188)     # --ink-faint-ish, the readout text


def _overlay_font(painter: QPainter, size: float, *, caps: bool = True) -> None:
    """Set the painter to the mockup's overlay face: mono, spaced, small caps.

    Explicitly mono rather than the widget's inherited face because these chips
    are readouts - a resolution, a state - and the prototype sets them in IBM
    Plex Mono so the digits line up and the caps read as instrument labels.
    """
    font = QFont(FONT_MONO)
    font.setPointSizeF(size)
    font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 118)
    font.setCapitalization(
        QFont.Capitalization.AllUppercase if caps else QFont.Capitalization.MixedCase
    )
    painter.setFont(font)


def _draw_chip(
    painter: QPainter, rect: QRectF, text: str, *, accent: bool, signal: QColor,
    ink: QColor = _OVERLAY_INK, radius: float = 7.0,
) -> None:
    """Fill one overlay chip: translucent plate, hairline border, centred text.

    The accent variant is the DLSS side - cyan text over a cyan-tinted border -
    so the eye reads "this half is the neural result" without a legend.
    """
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(_OVERLAY_PLATE)
    painter.drawRoundedRect(rect, radius, radius)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    if accent:
        border = QColor(signal)
        border.setAlpha(160)
        painter.setPen(QPen(border, 1))
    else:
        painter.setPen(QPen(_OVERLAY_LINE, 1))
    painter.drawRoundedRect(rect, radius, radius)
    painter.setPen(signal if accent else ink)
    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


def _corner_label(
    painter: QPainter, text: str, image: QRectF, widget: QWidget, align_left: bool,
    *, accent: bool = False, signal: QColor | None = None,
) -> None:
    """Draw a pill at a top corner of `image`, clamped inside `widget`.

    The anchor is the image's own corner, so at fit the pills sit on the picture
    - top-left and top-right - wherever it is letterboxed. Clamping to the widget
    is what makes them sticky: zoom in and the image corners leave the screen,
    but the pill holds at the visible edge rather than scrolling away with the
    pixels it is describing.

    The translucent plate keeps mono caps legible over a blown-out sky or a
    white wall, which a plain light label would disappear into.
    """
    if not text:
        return
    if signal is None:
        signal = widget.palette().highlight().color()
    _overlay_font(painter, 8.5)
    metrics = painter.fontMetrics()
    pad_x, pad_y = 10.0, 5.0
    chip_w = metrics.horizontalAdvance(text) + pad_x * 2
    chip_h = metrics.height() + pad_y * 2
    margin = 12.0

    top = min(max(image.top() + margin, margin), widget.height() - chip_h - margin)
    if align_left:
        x = min(max(image.left() + margin, margin), widget.width() - chip_w - margin)
    else:
        x = max(min(image.right() - margin - chip_w, widget.width() - chip_w - margin),
                margin)
    _draw_chip(painter, QRectF(x, top, chip_w, chip_h), text, accent=accent, signal=signal)


def _paint_divider(
    painter: QPainter, x: float, top: float, bottom: float, signal: QColor,
) -> None:
    """The prototype's glowing seam: a vertical gradient bar with a round grip.

    A flat line reads as a crop mark. The gradient fading at the ends, the halo
    and the grabbable disc read instead as a control - which is what it is, the
    thing the user drags to compare - so it invites the drag rather than just
    marking where the two halves meet.
    """
    grad = QLinearGradient(x, top, x, bottom)
    clear = QColor(signal)
    clear.setAlpha(0)
    grad.setColorAt(0.0, clear)
    grad.setColorAt(0.5, signal)
    grad.setColorAt(1.0, clear)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(grad))
    painter.drawRect(QRectF(x - 1.0, top, 2.0, bottom - top))

    centre = QPointF(x, (top + bottom) / 2.0)
    # Concentric translucent rings stand in for the mockup's CSS box-shadow glow,
    # the same trick the tutorial spotlight uses - the painter has no box-shadow.
    for radius, alpha in ((23.0, 45), (18.0, 80)):
        halo = QColor(signal)
        halo.setAlpha(alpha)
        painter.setPen(QPen(halo, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(centre, radius, radius)
    painter.setBrush(QColor(8, 17, 28))
    painter.setPen(QPen(signal, 1.5))
    painter.drawEllipse(centre, 16.0, 16.0)

    # Two chevrons for the grip, drawn rather than set as a glyph so the arrow
    # never depends on a font that may not carry ⟺.
    grip = QPen(signal, 1.6)
    grip.setCapStyle(Qt.PenCapStyle.RoundCap)
    grip.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(grip)
    cy = centre.y()
    left = QPolygonF([QPointF(x - 3, cy - 4), QPointF(x - 7, cy), QPointF(x - 3, cy + 4)])
    right = QPolygonF([QPointF(x + 3, cy - 4), QPointF(x + 7, cy), QPointF(x + 3, cy + 4)])
    painter.drawPolyline(left)
    painter.drawPolyline(right)


def _paint_readout(
    painter: QPainter, widget: QWidget, text: str, *, band: float | None = None,
) -> None:
    """A resolution + zoom chip pinned bottom-right, like the mockup's readout.

    Bottom-right rather than centred: it is a passive instrument reading, so it
    belongs out of the way at the corner, not floating over the middle of the
    picture the user is judging.

    ``band`` confines the chip to a reserved strip of that height at the very
    bottom, for the side-by-side view: there the panes must stay pixel-identical
    for comparison, so the chip has to sit clear of the image rather than float
    over it as it does on the single and wipe views.
    """
    if not text:
        return
    _overlay_font(painter, 8.5, caps=False)
    metrics = painter.fontMetrics()
    chip_w = metrics.horizontalAdvance(text) + 22.0
    margin = 12.0
    if band is not None:
        chip_h = min(metrics.height() + 8.0, band - 2.0)
        top = widget.height() - band + (band - chip_h) / 2.0
    else:
        chip_h = metrics.height() + 12.0
        top = widget.height() - chip_h - margin
    rect = QRectF(widget.width() - chip_w - margin, top, chip_w, chip_h)
    _draw_chip(
        painter, rect, text, accent=False, signal=widget.palette().highlight().color(),
        ink=_OVERLAY_READ, radius=9.0,
    )


class CanvasView(QWidget):
    """Shared zoom and pan for the image views.

    Wheel zooms about the cursor, right-drag pans, double-click fits again.
    Right rather than left because the comparison view already uses left-drag
    for its divider, and losing that to panning would be a bad trade.

    Zooming matters more here than in most viewers: people are running 6K and 8K
    renders through this and judging changes — pore detail, fabric weave, hair
    silhouettes — that simply are not visible in a fit-to-window view.
    """

    MIN_ZOOM = 1.0
    MAX_ZOOM = 32.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._panning = False
        self._pan_from = QPointF(0.0, 0.0)
        # The overlay layer - pills and the resolution readout - is hidden until
        # the pointer is over the picture, then faded in by the StageHost. It
        # starts invisible so a converted image is unobstructed the instant it
        # lands, and the instruments arrive only when the person reaches for them.
        self._chrome_opacity = 0.0
        self._hover_listener = None
        self.setMinimumSize(480, 320)

    # -- hover-revealed overlay chrome ---------------------------------------

    def set_chrome_opacity(self, value: float) -> None:
        """Fade the pills/readout in or out, driven by the StageHost's hover."""
        value = float(value)
        if value != self._chrome_opacity:
            self._chrome_opacity = value
            self.update()

    def set_hover_listener(self, callback) -> None:
        """Let the StageHost hear when the pointer enters or leaves the picture."""
        self._hover_listener = callback

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt name
        if self._hover_listener is not None:
            self._hover_listener(True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt name
        if self._hover_listener is not None:
            self._hover_listener(False)
        super().leaveEvent(event)

    # -- geometry ------------------------------------------------------------

    def _viewport(self) -> QRectF:
        """The area one image is laid out in.

        The whole widget, unless a subclass splits it into panes. Everything
        below works in this space rather than in widget coordinates, which is
        what lets a two-pane view drive both panes from a single zoom and pan
        - they are not synchronised, they are literally the same numbers.
        """
        return QRectF(0.0, 0.0, float(self.width()), float(self.height()))

    def _to_viewport(self, point: QPointF) -> QPointF:
        """Widget coordinates into the space `_viewport` describes."""
        return point

    def _fit_rect(self, size) -> QRectF:
        """The image at 100% fit, centred, ignoring zoom and pan."""
        view = self._viewport()
        if size.width() <= 0 or size.height() <= 0 or view.isEmpty():
            return QRectF()
        scale = min(view.width() / size.width(), view.height() / size.height())
        width, height = size.width() * scale, size.height() * scale
        return QRectF(
            view.left() + (view.width() - width) / 2,
            view.top() + (view.height() - height) / 2,
            width,
            height,
        )

    def _display_rect(self, size) -> QRectF:
        base = self._fit_rect(size)
        if base.isEmpty():
            return base
        width, height = base.width() * self._zoom, base.height() * self._zoom
        left = base.center().x() - width / 2 + self._pan.x()
        top = base.center().y() - height / 2 + self._pan.y()
        return QRectF(left, top, width, height)

    def _content_size(self):
        """Subclasses return the pixmap size they are drawing, or None."""
        return None

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()

    def _clamp_pan(self) -> None:
        """Keep some of the image on screen at all times."""
        size = self._content_size()
        if size is None:
            return
        rect = self._display_rect(size)
        view = self._viewport()
        margin_x = max(0.0, (rect.width() - view.width()) / 2)
        margin_y = max(0.0, (rect.height() - view.height()) / 2)
        self._pan.setX(float(np.clip(self._pan.x(), -margin_x, margin_x)))
        self._pan.setY(float(np.clip(self._pan.y(), -margin_y, margin_y)))

    # -- interaction ---------------------------------------------------------

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt name
        size = self._content_size()
        if size is None:
            return
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        previous = self._zoom
        self._zoom = float(np.clip(previous * (1.25**steps), self.MIN_ZOOM, self.MAX_ZOOM))
        if self._zoom == previous:
            return

        # Keep whatever is under the cursor under the cursor. Without this,
        # zooming always creeps towards the centre and you lose the detail you
        # were aiming at.
        cursor = self._to_viewport(event.position())
        before = self._display_rect(size)
        if before.width() > 0 and before.height() > 0:
            u = (cursor.x() - before.left()) / before.width()
            v = (cursor.y() - before.top()) / before.height()
            self._pan = QPointF(0.0, 0.0) + self._pan  # copy
            after = self._display_rect(size)
            self._pan.setX(self._pan.x() + cursor.x() - (after.left() + u * after.width()))
            self._pan.setY(self._pan.y() + cursor.y() - (after.top() + v * after.height()))
        if self._zoom <= self.MIN_ZOOM:
            self._pan = QPointF(0.0, 0.0)
        self._clamp_pan()
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt name
        if event.button() == Qt.MouseButton.RightButton:
            self._panning = True
            self._pan_from = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt name
        if self._panning:
            delta = event.position() - self._pan_from
            self._pan_from = event.position()
            self._pan += delta
            self._clamp_pan()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt name
        if event.button() == Qt.MouseButton.RightButton:
            self._panning = False
            self.unsetCursor()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt name
        self.reset_view()

    def _zoom_caption(self) -> str:
        return "" if self._zoom <= 1.001 else f"{self._zoom:.1f}x  ·  right-drag to pan"

    def _readout_text(self, size) -> str:
        """The bottom-right instrument reading: native resolution, then zoom.

        The resolution is the picture's real pixel size, not the on-screen rect,
        because that is what the person cares about when judging an 8K render;
        the zoom is appended only when it is doing something.
        """
        if size is None or size.width() <= 0 or size.height() <= 0:
            return ""
        dims = f"{size.width()} × {size.height()}"
        return dims if self._zoom <= 1.001 else f"{dims}   ·   {self._zoom:.1f}×"


class ImageView(CanvasView):
    """One image, scaled to fit and centred.

    Used for the source photo and the depth mask. Deliberately not a QLabel with
    a scaled pixmap: rescaling on every repaint keeps the depth preview sharp
    when the window is resized, and the depth map is the thing the user is
    squinting at to judge silhouettes.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._caption = ""
        self._progress: float | None = None
        self._grey: QPixmap | None = None

    def _content_size(self):
        return self._pixmap.size() if self._pixmap is not None else None

    # -- progress ------------------------------------------------------------

    def set_progress(self, fraction: float | None) -> None:
        """Show the image draining of colour, refilling as work completes.

        `fraction` is how much is done, 0..1, or None to stop. The height of
        the colour is the progress - it is a progress bar that happens to be
        the picture, not an animation playing next to one. That distinction is
        the whole reason to do it this way: an idle spinner tells you the app
        has not frozen, this tells you how much longer.

        Nothing here claims the revealed part is finished output. DLSS gives us
        no intermediate image, so the colour returning is the source image
        coming back, and the caption underneath says what is actually running.
        """
        if fraction is None:
            self._progress = None
            self._grey = None
            self.update()
            return
        if self._grey is None and self._pixmap is not None:
            # Built once per run rather than per repaint: at 8K this is a
            # 33-megapixel conversion and the bar moves eight times.
            self._grey = desaturated(self._pixmap)
        self._progress = float(np.clip(fraction, 0.0, 1.0))
        self.update()

    def set_pixmap(self, pixmap: QPixmap | None, caption: str = "") -> None:
        # A new image at the old zoom would land the viewport somewhere
        # arbitrary, so start fitted. Re-rendering the *same* image at a new
        # depth contrast goes through here too, but that only changes colour.
        changed = self._pixmap is None or pixmap is None or (
            self._pixmap.size() != pixmap.size()
        )
        self._pixmap = pixmap
        self._caption = caption
        # A new picture invalidates the desaturated copy the progress sweep
        # draws; rebuilt on the next set_progress rather than eagerly, since
        # most images never see one.
        self._grey = None
        if changed:
            self.reset_view()
        self.update()

    def set_image_u8(self, image_rgb: np.ndarray, caption: str = "") -> None:
        self.set_pixmap(QPixmap.fromImage(to_qimage_u8(image_rgb)), caption)

    def set_image(self, image_rgb: np.ndarray, caption: str = "") -> None:
        self.set_pixmap(QPixmap.fromImage(to_qimage(image_rgb)), caption)

    def clear(self) -> None:
        self.set_pixmap(None)

    def _paint_progress(self, painter: QPainter, rect: QRectF) -> None:
        assert self._pixmap is not None and self._grey is not None
        paint_sweep(
            painter, rect, self._pixmap, self._grey,
            self._progress or 0.0, self.palette().highlight().color(),
        )

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt name
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        if self._pixmap is None:
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = self._display_rect(self._pixmap.size())
        if self._progress is not None and self._grey is not None:
            self._paint_progress(painter, rect)
        else:
            painter.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))
        if self._caption:
            painter.setPen(self.palette().text().color())
            painter.drawText(
                QRectF(0, self.height() - 26, self.width(), 22),
                Qt.AlignmentFlag.AlignCenter,
                self._caption,
            )
        if self._chrome_opacity > 0.01:
            painter.save()
            painter.setOpacity(self._chrome_opacity)
            _paint_readout(painter, self, self._readout_text(self._pixmap.size()))
            painter.restore()


class WipeView(CanvasView):
    """Before/after comparison with a divider the user drags.

    A wipe rather than side-by-side panes because the differences DLSS 5 makes —
    skin subsurface, hair lighting, fabric sheen — are local and low-amplitude.
    Two half-size images side by side hide exactly that kind of change; sliding
    one image over another in place makes it obvious.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._before: QPixmap | None = None
        self._after: QPixmap | None = None
        self._split = 0.5
        self._dragging = False
        self._labels: tuple[str, str] | None = None
        self._accent_right = True
        self._progress: float | None = None
        self._grey: QPixmap | None = None
        self.setMouseTracking(True)

    def set_progress(self, fraction: float | None) -> None:
        """Sweep the *after* half while a new result is being computed.

        Only the after half, deliberately. The before image is not being
        recomputed, and greying it out would suggest otherwise; leaving it
        alone also means the comparison stays readable throughout - you keep
        the previous result to look at rather than an empty view.
        """
        if fraction is None:
            self._progress = None
            self._grey = None
            self.update()
            return
        if self._grey is None and self._after is not None:
            self._grey = desaturated(self._after)
        self._progress = float(np.clip(fraction, 0.0, 1.0))
        self.update()

    def set_labels(self, left: str = "", right: str = "", *, accent_right: bool = True) -> None:
        """Name the two halves and say whether the right one is the result.

        ``accent_right`` tints the right pill cyan, marking it as the neural
        result - correct for source-versus-DLSS, but wrong for a style-versus-
        style compare where neither half is "the output", so that caller turns
        it off and both pills read as neutral names.
        """
        self._labels = (left, right) if (left or right) else None
        self._accent_right = accent_right
        self.update()

    def _content_size(self):
        return self._before.size() if self._before is not None else None

    def set_images(self, before: np.ndarray, after: np.ndarray) -> None:
        # Only refit when the image itself changes shape. Re-grading redraws
        # this constantly, and snapping back to fit on every slider tick would
        # make the grade impossible to judge while zoomed in.
        changed = self._before is None or self._before.size() != QPixmap.fromImage(
            to_qimage(before)
        ).size()
        self._before = QPixmap.fromImage(to_qimage(before))
        self._after = QPixmap.fromImage(to_qimage(after))
        self._grey = None
        if changed:
            self.reset_view()
        self.update()

    def set_images_u8(
        self, before: np.ndarray, after: np.ndarray, keep_view: bool = False
    ) -> None:
        """Same as set_images but for 8-bit arrays, which the grade produces.

        ``keep_view`` holds the current zoom and pan even though the pixmap size
        changed. That is what lets the same result be swapped between a fast
        low-res preview (while a colour slider is moving) and the full-resolution
        image (once it settles) without the view snapping back to fit - the two
        fit to the same rect, so the zoom multiplier still frames the same place,
        just sharper.
        """
        pixmap = QPixmap.fromImage(to_qimage_u8(before))
        changed = self._before is None or self._before.size() != pixmap.size()
        self._before = pixmap
        self._after = QPixmap.fromImage(to_qimage_u8(after))
        self._grey = None
        if changed and not keep_view:
            self.reset_view()
        self.update()

    def clear(self) -> None:
        self._before = self._after = None
        self.update()

    def _target_rect(self) -> QRectF:
        if self._before is None:
            return QRectF()
        return self._display_rect(self._before.size())

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt name
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        if self._before is None or self._after is None:
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = self._target_rect()
        if self._progress is not None and self._grey is not None:
            paint_sweep(
                painter, rect, self._after, self._grey,
                self._progress, self.palette().highlight().color(),
            )
        else:
            painter.drawPixmap(rect, self._after, QRectF(self._after.rect()))

        # The "before" half is clipped rather than drawn scaled-and-cropped, so
        # both sides stay pixel-aligned and the seam does not shimmer as it moves.
        split_x = rect.left() + rect.width() * self._split
        painter.save()
        painter.setClipRect(QRectF(rect.left(), rect.top(), split_x - rect.left(), rect.height()))
        painter.drawPixmap(rect, self._before, QRectF(self._before.rect()))
        painter.restore()

        signal = self.palette().highlight().color()
        # The divider stays visible whether or not the pointer is over the
        # picture: it is the comparison control, and hiding it would leave no
        # sign that the two halves can be wiped between.
        _paint_divider(painter, split_x, rect.top(), rect.bottom(), signal)

        # The pills and readout are hover chrome, faded in and out by the
        # StageHost so the picture is unobstructed while it is being studied.
        if self._chrome_opacity <= 0.01:
            return
        painter.save()
        painter.setOpacity(self._chrome_opacity)
        # Name the two halves. Explicit labels win (a style-versus-style compare
        # is not obvious); otherwise the default source/result pair, with the
        # right half accented so the eye reads it as the neural result. Each is
        # pinned to the image's own top corner and clamped to the widget - so
        # zoomed in, when the corner has slid off screen, the pill holds at the
        # visible edge instead of vanishing - then clipped to its own half.
        if self._labels is not None:
            left, right = self._labels
            accent_right = self._accent_right
        else:
            left, right, accent_right = "SOURCE", "DLSS 5", True
        painter.save()
        painter.setClipRect(QRectF(0, 0, split_x, self.height()))
        _corner_label(painter, left, rect, self, align_left=True, signal=signal)
        painter.restore()
        painter.save()
        painter.setClipRect(QRectF(split_x, 0, self.width() - split_x, self.height()))
        _corner_label(
            painter, right, rect, self, align_left=False, accent=accent_right, signal=signal,
        )
        painter.restore()
        _paint_readout(painter, self, self._readout_text(self._before.size()))
        painter.restore()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt name
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._move_split(event.position().x())
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt name
        if self._dragging:
            self._move_split(event.position().x())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt name
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
        else:
            super().mouseReleaseEvent(event)

    def _move_split(self, x: float) -> None:
        rect = self._target_rect()
        if rect.width() <= 0:
            return
        self._split = float(np.clip((x - rect.left()) / rect.width(), 0.0, 1.0))
        self.update()


class SideBySideView(CanvasView):
    """Two or three full images in a row, locked to one zoom and one pan.

    The wipe is better at spotting a change; this is better at judging one.
    Sliding a divider back and forth answers "did that move?", but choosing
    between neural styles is a question about the whole frame at once - which
    of these pictures do I want - and that needs all of them on screen.

    The panes do not synchronise with each other. They share a single zoom and
    pan, applied within each pane, so they cannot drift apart: there is nothing
    to keep in step. Zooming about the cursor works from whichever pane the
    cursor is in, and the others land on the same part of the image.
    """

    #: Space between the panes, in pixels. Wide enough to read as separate
    #: images rather than one wide one.
    GAP = 12
    #: Strips kept clear above and below the panes, for the labels and the zoom
    #: readout. Reserved rather than drawn over the images: a caption centred on
    #: the widget lands half on one picture and half on the next, which looks
    #: broken, and a label sitting on the image obscures what is being judged.
    LABEL_BAND = 26
    CAPTION_BAND = 26

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._panes: list[QPixmap] = []
        self._labels: list[str] = []
        self._progress: list[float | None] = []
        self._grey: list[QPixmap | None] = []

    # -- progress ------------------------------------------------------------

    def set_pane_progress(self, index: int, fraction: float | None) -> None:
        """Sweep one pane while its version is being computed.

        Per pane rather than per view, because the styles are converted one
        after another - the add-on reads its configuration once at startup, so
        each style is its own harness. Showing them all sweeping together would
        claim work is happening that has not started.
        """
        while len(self._progress) < len(self._panes):
            self._progress.append(None)
            self._grey.append(None)
        if not 0 <= index < len(self._progress):
            return
        if fraction is None:
            self._progress[index] = None
            self._grey[index] = None
        else:
            if self._grey[index] is None and index < len(self._panes):
                self._grey[index] = desaturated(self._panes[index])
            self._progress[index] = float(np.clip(fraction, 0.0, 1.0))
        self.update()

    def clear_progress(self) -> None:
        self._progress = [None] * len(self._panes)
        self._grey = [None] * len(self._panes)
        self.update()

    # -- geometry ------------------------------------------------------------

    def count(self) -> int:
        return max(1, len(self._panes))

    def _pane_width(self) -> float:
        """Whole pixels, deliberately.

        A fractional pane width puts every pane at a fractional offset, and the
        same image then rasterises slightly differently in each one - sub-pixel,
        invisible, and enough to mean the panes are not actually showing you
        identical geometry. Rounding down and letting a pixel or two go spare on
        the right costs nothing and makes them genuinely identical.
        """
        panes = self.count()
        return max(1.0, float((self.width() - self.GAP * (panes - 1)) // panes))

    def _pane_left(self, index: int) -> float:
        return float(index) * (self._pane_width() + self.GAP)

    def _viewport(self) -> QRectF:
        # Pane-local: every pane is the same size, so one rect describes them
        # all, and drawing that one rect at each pane offset is what makes the
        # zoom and pan shared rather than merely synchronised.
        height = max(1.0, self.height() - self.LABEL_BAND - self.CAPTION_BAND)
        return QRectF(0.0, float(self.LABEL_BAND), self._pane_width(), height)

    def _to_viewport(self, point: QPointF) -> QPointF:
        # Whichever pane the cursor is over, in that pane's own coordinates -
        # so zoom-about-cursor aims at the same pixel of the image in all of
        # them, rather than at a point one or two panes away.
        stride = self._pane_width() + self.GAP
        index = int(np.clip(point.x() // stride, 0, self.count() - 1))
        return QPointF(point.x() - self._pane_left(index), point.y())

    def _content_size(self):
        return self._panes[0].size() if self._panes else None

    # -- content -------------------------------------------------------------

    def set_panes_u8(self, images: list[np.ndarray]) -> None:
        """Replace the row. Any number of panes; two or three in practice."""
        pixmaps = [QPixmap.fromImage(to_qimage_u8(image)) for image in images]
        # Refit when the layout changes, not on every re-grade: snapping back
        # to fit each time a colour slider moved would make the grade
        # impossible to judge while zoomed in.
        changed = (
            len(pixmaps) != len(self._panes)
            or not self._panes
            or pixmaps[0].size() != self._panes[0].size()
        )
        # New pictures, so any sweep state belongs to images that are gone. The
        # caller re-arms whatever is still outstanding; keeping it here would
        # leave a finished pane greyed because it used to be working.
        self._panes = pixmaps
        self._progress = [None] * len(pixmaps)
        self._grey = [None] * len(pixmaps)
        if changed:
            self.reset_view()
        self.update()

    def set_images_u8(self, left: np.ndarray, right: np.ndarray) -> None:
        """Two-pane convenience, kept because most callers only ever want two."""
        self.set_panes_u8([left, right])

    def set_labels(self, *labels: str) -> None:
        self._labels = [text for text in labels]
        self.update()

    def clear(self) -> None:
        self._panes = []
        self._progress = []
        self._grey = []
        self.update()

    # -- painting ------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt name
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        if not self._panes:
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        rect = self._display_rect(self._panes[0].size())
        view = self._viewport()
        for index, pixmap in enumerate(self._panes):
            offset = self._pane_left(index)
            pane = QRectF(offset, view.top(), self._pane_width(), view.height())
            painter.save()
            painter.setClipRect(pane)
            here = rect.translated(offset, 0.0)
            progress = self._progress[index] if index < len(self._progress) else None
            grey = self._grey[index] if index < len(self._grey) else None
            if progress is not None and grey is not None:
                paint_sweep(
                    painter, here, pixmap, grey,
                    progress, self.palette().highlight().color(),
                )
            else:
                painter.drawPixmap(here, pixmap, QRectF(pixmap.rect()))
            painter.restore()

            label = self._labels[index] if index < len(self._labels) else ""
            if label and self._chrome_opacity > 0.01:
                # Centred in the reserved band above the pane, as a pill so it
                # reads in the same instrument language as the wipe's corner tags.
                # Hover chrome, so faded with the readout by the StageHost.
                painter.save()
                painter.setOpacity(self._chrome_opacity)
                _overlay_font(painter, 8.5)
                metrics = painter.fontMetrics()
                chip_w = metrics.horizontalAdvance(label) + 20.0
                chip_h = metrics.height() + 8.0
                chip_x = offset + (self._pane_width() - chip_w) / 2.0
                chip_y = (self.LABEL_BAND - chip_h) / 2.0
                _draw_chip(
                    painter, QRectF(chip_x, chip_y, chip_w, chip_h), label,
                    accent=False, signal=self.palette().highlight().color(),
                )
                painter.restore()

        # Seams, so similar images do not read as one wide one.
        painter.setPen(QPen(self.palette().highlight().color(), 1))
        for index in range(1, len(self._panes)):
            middle = self._pane_left(index) - self.GAP / 2
            painter.drawLine(QPointF(middle, view.top()), QPointF(middle, view.bottom()))

        if self._chrome_opacity > 0.01:
            painter.save()
            painter.setOpacity(self._chrome_opacity)
            _paint_readout(
                painter, self, self._readout_text(self._panes[0].size()),
                band=float(self.CAPTION_BAND),
            )
            painter.restore()


class HoverBar(QFrame):
    """A translucent control bar that reports pointer enter/leave to its host.

    It floats over the picture and carries the view controls. Because it tells
    the StageHost when the pointer is on it, moving from the image onto the bar
    keeps the whole overlay revealed rather than flickering it away.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("viewBar")
        self._hover_listener = None

    def set_hover_listener(self, callback) -> None:
        self._hover_listener = callback

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt name
        if self._hover_listener is not None:
            self._hover_listener(True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt name
        if self._hover_listener is not None:
            self._hover_listener(False)
        super().leaveEvent(event)


class StageHost(QWidget):
    """Holds the image stack and floats a hover-revealed control bar over it.

    The whole overlay layer - the SOURCE/DLSS pills, the resolution readout and
    the view bar - fades in when the pointer is over the picture and fades out
    when it leaves, so the image is unobstructed while it is being studied and
    the instruments are there the instant they are reached for. The fade is one
    animation driving both the painted chrome (on the views) and the bar, so the
    two can never disagree about whether they are shown.
    """

    #: How long the overlay takes to fade in or out. Short enough to feel
    #: immediate, long enough not to read as a hard cut.
    FADE_MS = 160

    def __init__(
        self, stack: QStackedWidget, bar: HoverBar, parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._stack = stack
        self._bar = bar
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(stack)

        bar.setParent(self)
        self._effect = QGraphicsOpacityEffect(bar)
        bar.setGraphicsEffect(self._effect)
        self._effect.setOpacity(0.0)
        bar.setVisible(False)

        self._opacity = 0.0
        self._target = 0.0
        self._pending_hide = False
        self._fade = QVariantAnimation(self)
        self._fade.setDuration(self.FADE_MS)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade.valueChanged.connect(self._apply)

        # The views exist before the host is built, so bind their hover once.
        self._views = stack.findChildren(CanvasView)
        for view in self._views:
            view.set_hover_listener(self._hover)
        bar.set_hover_listener(self._hover)

    def _position_bar(self) -> None:
        """Bottom-left, matching the mockup's stagefoot; the readout is painted
        bottom-right by the view, so the two share the strip without colliding."""
        margin = 14
        self._bar.adjustSize()
        self._bar.move(margin, max(margin, self.height() - self._bar.height() - margin))
        self._bar.raise_()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt name
        self._position_bar()
        super().resizeEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt name
        self._position_bar()
        super().showEvent(event)

    def _hover(self, entered: bool) -> None:
        if entered:
            self._pending_hide = False
            self._reveal(True)
        else:
            # Moving from a view onto the bar (or a bar button) fires a leave
            # before the next enter, so never hide on the spot: wait, then hide
            # only if the pointer really is off the whole stage by then.
            self._pending_hide = True
            QTimer.singleShot(60, self._maybe_hide)

    def _maybe_hide(self) -> None:
        if not self._pending_hide:
            return
        inside = self.rect().contains(self.mapFromGlobal(QCursor.pos()))
        if not inside:
            self._reveal(False)

    def _reveal(self, shown: bool) -> None:
        self._target = 1.0 if shown else 0.0
        if self._target == self._opacity:
            return
        if shown:
            self._bar.setVisible(True)
            self._position_bar()
        self._fade.stop()
        self._fade.setStartValue(self._opacity)
        self._fade.setEndValue(self._target)
        self._fade.start()

    def _apply(self, value: object) -> None:
        self._opacity = float(value)
        self._effect.setOpacity(self._opacity)
        if self._opacity <= 0.001 and self._target == 0.0:
            self._bar.setVisible(False)
        for view in self._views:
            view.set_chrome_opacity(self._opacity)


def format_bytes(count: float) -> str:
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return "less than a second"
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f} min {seconds % 60:.0f} s"
    return f"{minutes / 60:.1f} hours"


class DownloadDialog(QDialog):
    """First-run download, with the two numbers people actually want.

    Rate is averaged over a trailing window rather than since-the-start: these
    downloads resume, and a run that picks up at 80% would otherwise show a
    wildly optimistic rate for its whole life. The window also keeps the
    estimate from lurching every time a file finishes.
    """

    #: Seconds of history used for the rate estimate.
    WINDOW = 8.0

    #: Emitted when the user asks to abandon the step (only if enable_cancel was
    #: called). The owner is responsible for stopping the work and closing this.
    cancelled = Signal()

    def __init__(self, title: str, subtitle: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(460)
        # No close button: the work continues regardless of the dialog, and a
        # titlebar X that silently does nothing is worse than not having one.
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)

        self._heading = QLabel(title)
        layout.addWidget(self._heading)
        if subtitle:
            note = QLabel(subtitle)
            note.setObjectName("hint")
            note.setWordWrap(True)
            layout.addWidget(note)

        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setTextVisible(False)
        layout.addWidget(self._bar)

        self._detail = QLabel("Starting…")
        self._detail.setObjectName("hint")
        layout.addWidget(self._detail)

        self._history: list[tuple[float, int]] = []

    def enable_cancel(self, text: str = "Skip") -> None:
        """Add a button that lets the user abandon this step.

        Off by default: the first-run downloads carry on whether the dialog is
        up or not, so a Cancel there would be a lie. The runtime check is the
        opposite - it can wedge the GPU, and the user needs a way out that is not
        force-quitting the whole app. Clicking it only emits `cancelled`; the
        owner stops the work and closes the dialog, so the button can never look
        like it did something while the process is actually still running.
        """
        row = QHBoxLayout()
        row.addStretch(1)
        button = QPushButton(text)
        button.setObjectName("secondary")
        button.clicked.connect(self.cancelled.emit)
        row.addWidget(button)
        self.layout().addLayout(row)

    def set_heading(self, text: str) -> None:
        self._heading.setText(text)

    def set_status(self, text: str) -> None:
        self._detail.setText(text)

    def set_busy(self) -> None:
        """Show indeterminate progress for work with no meaningful percentage."""
        self._bar.setRange(0, 0)

    def mark_complete(self) -> None:
        """Fill the bar on success.

        The folder-watching reporter can deliver a final sample slightly under
        the total — files are renamed out of ``.incomplete`` as they land, so
        the measured size dips at the very end — and a bar that stops at 95%
        looks like a download that gave up.
        """
        self._bar.setRange(0, 1000)
        self._bar.setValue(1000)

    def update_bytes(self, done: int, total: int) -> None:
        now = time.monotonic()
        self._history.append((now, done))
        while len(self._history) > 2 and now - self._history[0][0] > self.WINDOW:
            self._history.pop(0)

        if total > 0:
            self._bar.setRange(0, 1000)
            self._bar.setValue(int(min(1000, done / total * 1000)))
        else:
            # Unknown total: a busy indicator beats a bar stuck at zero.
            self._bar.setRange(0, 0)

        parts = [f"{format_bytes(done)} of {format_bytes(total)}" if total else format_bytes(done)]
        span = now - self._history[0][0]
        moved = done - self._history[0][1]
        if span >= 1.0 and moved > 0:
            rate = moved / span
            parts.append(f"{format_bytes(rate)}/s")
            if total > done:
                parts.append(f"about {format_duration((total - done) / rate)} left")
        self._detail.setText("  —  ".join(parts))


class SliderRow(QWidget):
    """A slider over 0..`maximum` with a live numeric readout."""

    def __init__(
        self,
        label: str,
        value: float,
        on_change: Callable[[float], None],
        tooltip: str = "",
        parent: QWidget | None = None,
        maximum: float = 1.0,
        minimum: float = 0.0,
    ) -> None:
        super().__init__(parent)
        self._on_change = on_change
        self._maximum = float(maximum)
        self._minimum = float(minimum)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(2)

        header = QHBoxLayout()
        self._name = QLabel(label)
        self._value = QLabel(f"{value:+.2f}" if minimum < 0 else f"{value:.2f}")
        self._value.setObjectName("hint")
        header.addWidget(self._name)
        header.addStretch(1)
        header.addWidget(self._value)
        layout.addLayout(header)

        # Sliders are integers. The step is fixed at 0.01 of a unit rather than
        # a fraction of the range, so a 0..2 slider gets 200 positions and the
        # readout stays aimable at the same precision as a 0..1 one.
        self._steps = max(1, int(round((self._maximum - self._minimum) * 100)))
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, self._steps)
        self._slider.setValue(self._to_raw(value))
        self._slider.valueChanged.connect(self._changed)
        layout.addWidget(self._slider)

        if tooltip:
            self.setToolTip(tooltip)
            self._name.setToolTip(tooltip)

    def _to_raw(self, value: float) -> int:
        span = self._maximum - self._minimum
        scaled = (float(value) - self._minimum) / span * self._steps
        return int(round(min(self._steps, max(0, scaled))))

    def _from_raw(self, raw: int) -> float:
        return self._minimum + raw / self._steps * (self._maximum - self._minimum)

    def _changed(self, raw: int) -> None:
        value = self._from_raw(raw)
        # A bipolar control reads much better with the sign shown: "+0.35" and
        # "-0.35" are obviously opposites in a way "0.35" is not.
        self._value.setText(f"{value:+.2f}" if self._minimum < 0 else f"{value:.2f}")
        self._on_change(value)

    def set_value(self, value: float) -> None:
        self._slider.setValue(self._to_raw(value))

    def compact(self) -> "SliderRow":
        """Drop the name from the header, keeping just the live value readout.

        Used inside a ChipSliderGroup, where the selected chip already carries
        the name — repeating it under the chip is noise. Returns self so it can
        be chained at construction.
        """
        self._name.hide()
        return self

    def bare(self) -> "SliderRow":
        """Hide the whole header, leaving just the slider.

        For a ChipSliderGroup that shows each value on its chip: the name and the
        value both live on the chip above, so the row under it is only the track.
        """
        return self.set_header(False, False)

    def set_header(self, name: bool, value: bool) -> "SliderRow":
        """Show or hide the name and value parts of the header independently.

        Lets one SliderRow serve both a ChipSliderGroup's compact mode (header
        off, the chip carries name and value) and its full mode (header on, the
        row stands alone). Returns self so it can be chained at construction.
        """
        self._name.setVisible(name)
        self._value.setVisible(value)
        return self

    def formatted(self, value: float) -> str:
        return f"{value:+.2f}" if self._minimum < 0 else f"{value:.2f}"


class FlowLayout(QLayout):
    """A layout that wraps its children onto new rows when they run out of width.

    Qt ships no wrapping layout, and a row of parameter chips must not overflow a
    fixed-width sidebar — four chips fit on one line on a wide window and fold to
    two lines on a narrow one, rather than being clipped. This is the canonical
    Qt FlowLayout, trimmed to what the chip rows need.
    """

    def __init__(self, parent: QWidget | None = None, spacing: int = 6) -> None:
        super().__init__(parent)
        self._items: list = []
        self._spacing = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:  # noqa: N802 - Qt name
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802 - Qt name
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):  # noqa: N802 - Qt name
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802 - Qt name
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt name
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt name
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 - Qt name
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt name
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt name
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        x, y = rect.x(), rect.y()
        line_height = 0
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + self._spacing
            if next_x - self._spacing > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + self._spacing
                next_x = x + hint.width() + self._spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y()


class ChipSliderGroup(QWidget):
    """A row of parameter chips over one shared slider.

    Rather than stack four sliders — the "wall of sliders" that made the sidebar
    read as cluttered — the parameters become a segmented row of chips, exactly
    like the Photo/Result/Depth switch under the image, and only the selected
    chip's slider is shown below it. Same controls, a fraction of the height, and
    one pattern reused everywhere a group holds several sliders.

    Each ``param`` is ``(label, value, on_change, tooltip, minimum, maximum)``.
    ``rows`` exposes the SliderRow per label so callers can still push a value in
    (a reset, a preset change) through ``set_value``.
    """

    def __init__(
        self, params, mode: str = "compact", columns: int = 2,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 2, 0, 2)
        outer.setSpacing(9)

        # A two-column grid rather than a wrapping flow: equal-width chips that
        # pack 2×2, so four value-bearing chips take two tidy rows instead of
        # four ragged ones. Columns stretch so the pair fills the sidebar width.
        self._chip_bar = QWidget()
        grid = QGridLayout(self._chip_bar)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        columns = max(1, columns)
        for c in range(columns):
            grid.setColumnStretch(c, 1)
        outer.addWidget(self._chip_bar)

        # Rows live in a plain box, not a stack, so Full mode can show them all at
        # once and Compact can hide all but the selected one — one set of widgets,
        # two layouts, switched live.
        rows_holder = QWidget()
        self._rows_box = QVBoxLayout(rows_holder)
        self._rows_box.setContentsMargins(0, 0, 0, 0)
        self._rows_box.setSpacing(2)
        outer.addWidget(rows_holder)

        self._buttons = QButtonGroup(self)
        self._buttons.setExclusive(True)
        self.rows: dict[str, SliderRow] = {}
        self._chips: dict[str, QPushButton] = {}
        self._order: list[str] = []
        self._mode = mode

        for index, (label, value, on_change, tooltip, minimum, maximum) in enumerate(params):
            chip = QPushButton()
            chip.setObjectName("chip")
            chip.setCheckable(True)
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            if tooltip:
                chip.setToolTip(tooltip)
            grid.addWidget(chip, index // columns, index % columns)
            self._buttons.addButton(chip, index)

            # The chip carries the value as well as the name, so all read at a
            # glance even with one slider showing. The setter is wrapped to keep
            # the chip's number in step with its slider as it drags.
            row = SliderRow(
                label, value, self._chip_setter(label, on_change), tooltip,
                maximum=maximum, minimum=minimum,
            )
            chip.setText(f"{label}  {row.formatted(value)}")
            self._rows_box.addWidget(row)
            self.rows[label] = row
            self._chips[label] = chip
            self._order.append(label)

        self._buttons.idClicked.connect(self._select)
        if params:
            self._buttons.button(0).setChecked(True)
        self.set_mode(mode)

    def _chip_setter(self, label: str, on_change):
        def wrapped(value: float) -> None:
            chip = self._chips.get(label)
            row = self.rows.get(label)
            if chip is not None and row is not None:
                chip.setText(f"{label}  {row.formatted(value)}")
            on_change(value)
        return wrapped

    def _select(self, index: int) -> None:
        """Compact only: reveal the chosen row, hide the rest."""
        if self._mode != "compact":
            return
        for i, label in enumerate(self._order):
            self.rows[label].setVisible(i == index)

    def set_mode(self, mode: str) -> None:
        """Switch between 'compact' (chips + one slider) and 'full' (all sliders).

        Full drops the chip bar and shows every row with its own name and value;
        Compact brings the chip bar back and leaves only the selected row's bare
        track under it. The chips stay wired in both, so a value pushed in from
        outside lands the same way.
        """
        self._mode = "full" if mode == "full" else "compact"
        full = self._mode == "full"
        self._chip_bar.setVisible(not full)
        selected = max(0, self._buttons.checkedId())
        for i, label in enumerate(self._order):
            row = self.rows[label]
            row.set_header(full, full)
            row.setVisible(full or i == selected)

    def set_value(self, label: str, value: float) -> None:
        row = self.rows.get(label)
        if row is not None:
            row.set_value(value)  # fires the wrapped setter, so the chip updates too


class SegmentedControl(QWidget):
    """A row of equal-width, mutually exclusive buttons — a styled radio group.

    The mockup replaces the Style and Detail-mode dropdowns with these: the two
    or three choices are all visible and one tap wide, instead of hidden behind a
    combo. ``options`` are ``str`` or ``(label, data)``; ``changed`` carries the
    selected index and ``current_data`` returns the chosen payload.
    """

    changed = Signal(int)

    def __init__(self, options, current: int = 0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._data: list = []
        for index, option in enumerate(options):
            label, data = option if isinstance(option, tuple) else (option, option)
            button = QPushButton(label)
            button.setObjectName("chip")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            row.addWidget(button, 1)
            self._group.addButton(button, index)
            self._data.append(data)
        self._group.idClicked.connect(self.changed)
        if options:
            self._group.button(min(current, len(options) - 1)).setChecked(True)

    def current_index(self) -> int:
        return self._group.checkedId()

    def current_data(self):
        index = self._group.checkedId()
        return self._data[index] if 0 <= index < len(self._data) else None

    def set_index(self, index: int) -> None:
        button = self._group.button(index)
        if button is not None:
            button.setChecked(True)


class GpuGauge(QFrame):
    """The command bar's GPU / VRAM readout — name, a usage bar, and the numbers.

    Read from ``nvidia-smi`` rather than by importing torch, deliberately: torch
    is a multi-second, gigabyte import this app keeps lazy (see CLAUDE.md), and
    the gauge must be up the instant the window paints. nvidia-smi ships with the
    driver, answers in milliseconds, and is run through a QProcess so it never
    blocks the GUI thread. No GPU, no smi — the gauge just reads a dash.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("gauge")
        row = QHBoxLayout(self)
        row.setContentsMargins(13, 6, 13, 6)
        row.setSpacing(11)

        name_col = QVBoxLayout()
        name_col.setSpacing(1)
        lab = QLabel("GPU")
        lab.setObjectName("gaugeLab")
        apply_font(lab, family=FONT_MONO, size=7.5, spacing=1.6, caps=True)
        self._name = QLabel("detecting…")
        self._name.setObjectName("gpuName")
        apply_font(self._name, family=FONT_DISPLAY, size=10.5)
        name_col.addWidget(lab)
        name_col.addWidget(self._name)
        row.addLayout(name_col)

        vram_col = QVBoxLayout()
        vram_col.setSpacing(3)
        self._bar = QProgressBar()
        self._bar.setObjectName("vramBar")
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(5)
        self._bar.setFixedWidth(118)
        self._read = QLabel("VRAM  —")
        self._read.setObjectName("vramRead")
        apply_font(self._read, family=FONT_MONO, size=8.5)
        vram_col.addWidget(self._bar)
        vram_col.addWidget(self._read)
        row.addLayout(vram_col)

        self._proc = QProcess(self)
        self._proc.finished.connect(self._parsed)
        self._proc.errorOccurred.connect(lambda _err: self._offline())
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(3000)
        QTimer.singleShot(0, self._poll)

    def _poll(self) -> None:
        if self._proc.state() == QProcess.ProcessState.NotRunning:
            self._proc.start("nvidia-smi", [
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ])

    def _offline(self) -> None:
        # errorOccurred can fire during window teardown, after the C++ labels are
        # gone; touching them then raises. A dead gauge has nothing to update.
        try:
            self._name.setText("—")
            self._read.setText("VRAM  —")
            self._timer.stop()  # smi is missing; polling it again will not help
        except RuntimeError:
            pass

    def _parsed(self) -> None:
        try:
            self._parse()
        except RuntimeError:
            pass  # widget torn down between the process finishing and this slot

    def _parse(self) -> None:
        raw = bytes(self._proc.readAllStandardOutput()).decode("utf-8", "ignore").strip()
        line = raw.splitlines()[0] if raw else ""
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            return
        name = parts[0].replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
        try:
            total = float(parts[1]) / 1024.0
            used = float(parts[2]) / 1024.0
        except ValueError:
            return
        self._name.setText(name)
        self._read.setText(f"VRAM  {used:.1f} / {total:.1f} GB")
        if total > 0:
            self._bar.setValue(int(min(1000, used / total * 1000)))


class TimelineWidget(QWidget):
    """A scrub bar with Premiere-style In/Out brackets and scroll-to-zoom.

    Frames are the unit throughout - the playhead, the In and Out points, and
    the visible window are all frame indices - because the conversion works in
    frames and a range that does not land on exact frames would convert a
    slightly different span than the one shown. Time is only ever a label.

    Zoom is a visible window ``[_view_lo, _view_hi)`` of the whole ``0..total``
    range; the wheel narrows or widens it about the cursor, so you can place an
    In point on frame 4137 of a five-thousand-frame clip without fighting a bar
    where every frame is a third of a pixel.
    """

    seeked = Signal(int)       # playhead moved by the user
    in_changed = Signal(int)
    out_changed = Signal(int)

    _HANDLE = 8.0              # half-width of a draggable bracket, in pixels
    _MIN_SPAN = 2             # never zoom tighter than this many frames

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(56)
        self.setMouseTracking(True)
        self._total = 0
        self._fps = 24.0
        self._playhead = 0
        self._in = 0
        self._out = 0
        self._range_mode = False
        self._view_lo = 0
        self._view_hi = 0          # exclusive; == total when fully zoomed out
        self._drag: str | None = None   # "playhead" | "in" | "out" | "pan"
        self._pan_anchor = 0.0

    # -- state ---------------------------------------------------------------

    def set_duration(self, total_frames: int, fps: float) -> None:
        self._total = max(0, int(total_frames))
        self._fps = fps or 24.0
        self._playhead = 0
        self._in = 0
        self._out = max(0, self._total - 1)
        self._view_lo = 0
        self._view_hi = self._total
        self.update()

    def set_playhead(self, frame: int) -> None:
        """Called from the player as it advances; does not emit seeked."""
        self._playhead = int(np.clip(frame, 0, max(0, self._total - 1)))
        self.update()

    def set_range_mode(self, on: bool) -> None:
        self._range_mode = bool(on)
        self.update()

    def set_in_out(self, in_frame: int, out_frame: int) -> None:
        last = max(0, self._total - 1)
        self._in = int(np.clip(in_frame, 0, last))
        self._out = int(np.clip(out_frame, self._in, last))
        self.update()

    def in_out(self) -> tuple[int, int]:
        return self._in, self._out

    def range_seconds(self) -> float:
        return (self._out - self._in + 1) / self._fps if self._total else 0.0

    # -- geometry ------------------------------------------------------------

    def _track(self) -> QRectF:
        return QRectF(self._HANDLE, 6.0, max(1.0, self.width() - 2 * self._HANDLE), 26.0)

    def _span(self) -> int:
        return max(1, self._view_hi - self._view_lo)

    def _frame_to_x(self, frame: float) -> float:
        track = self._track()
        return track.left() + (frame - self._view_lo) / self._span() * track.width()

    def _x_to_frame(self, x: float) -> int:
        track = self._track()
        if track.width() <= 0:
            return self._view_lo
        frac = (x - track.left()) / track.width()
        return int(round(self._view_lo + frac * self._span()))

    # -- interaction ---------------------------------------------------------

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt name
        if self._total <= 0:
            return
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        pivot = self._x_to_frame(event.position().x())
        factor = 0.8 ** steps   # scroll up zooms in
        new_span = int(np.clip(round(self._span() * factor), self._MIN_SPAN, self._total))
        # Keep the frame under the cursor under the cursor.
        frac = (pivot - self._view_lo) / self._span()
        lo = int(round(pivot - frac * new_span))
        lo = int(np.clip(lo, 0, max(0, self._total - new_span)))
        self._view_lo = lo
        self._view_hi = min(self._total, lo + new_span)
        self.update()

    def _nearest_handle(self, x: float) -> str:
        candidates = [("playhead", self._playhead)]
        if self._range_mode:
            candidates += [("in", self._in), ("out", self._out)]
        best, best_dist = "playhead", 1e9
        for name, frame in candidates:
            dist = abs(self._frame_to_x(frame) - x)
            if dist < best_dist:
                best, best_dist = name, dist
        # Grab a bracket only when genuinely near it; otherwise treat a click as
        # a seek, which is what a click on empty track should do.
        return best if best_dist <= self._HANDLE * 2 else "seek"

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt name
        if self._total <= 0:
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._drag = "pan"
            self._pan_anchor = event.position().x()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        target = self._nearest_handle(event.position().x())
        if target == "seek":
            self._drag = "playhead"
            self._apply_drag(event.position().x())
        else:
            self._drag = target
            self._apply_drag(event.position().x())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt name
        if self._drag == "pan":
            track = self._track()
            dx = event.position().x() - self._pan_anchor
            self._pan_anchor = event.position().x()
            shift = int(round(-dx / track.width() * self._span()))
            lo = int(np.clip(self._view_lo + shift, 0, max(0, self._total - self._span())))
            self._view_lo, self._view_hi = lo, lo + self._span()
            self.update()
        elif self._drag:
            self._apply_drag(event.position().x())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt name
        self._drag = None

    def _apply_drag(self, x: float) -> None:
        frame = int(np.clip(self._x_to_frame(x), 0, max(0, self._total - 1)))
        if self._drag == "playhead":
            self._playhead = frame
            self.seeked.emit(frame)
        elif self._drag == "in":
            self._in = min(frame, self._out)
            self.in_changed.emit(self._in)
        elif self._drag == "out":
            self._out = max(frame, self._in)
            self.out_changed.emit(self._out)
        self.update()

    # -- painting ------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.palette().window())
        if self._total <= 0:
            return
        track = self._track()
        painter.fillRect(track, QColor(38, 42, 51))

        highlight = self.palette().highlight().color()
        if self._range_mode:
            x_in = self._frame_to_x(self._in)
            x_out = self._frame_to_x(self._out)
            sel = QRectF(x_in, track.top(), max(1.0, x_out - x_in), track.height())
            fill = QColor(highlight)
            fill.setAlpha(70)
            painter.fillRect(sel, fill)
            painter.setPen(QPen(highlight, 2))
            for x in (x_in, x_out):
                painter.drawLine(QPointF(x, track.top() - 3), QPointF(x, track.bottom() + 3))
                # A little bracket foot so it reads as a handle.
                foot = 6.0 if x == x_in else -6.0
                painter.drawLine(QPointF(x, track.top() - 3), QPointF(x + foot, track.top() - 3))
                painter.drawLine(QPointF(x, track.bottom() + 3), QPointF(x + foot, track.bottom() + 3))

        # Playhead.
        px = self._frame_to_x(self._playhead)
        painter.setPen(QPen(QColor(240, 240, 240), 1))
        painter.drawLine(QPointF(px, track.top() - 4), QPointF(px, track.bottom() + 4))
        painter.setBrush(QColor(240, 240, 240))
        painter.drawPolygon(
            QPolygonF([
                QPointF(px - 5, track.top() - 4),
                QPointF(px + 5, track.top() - 4),
                QPointF(px, track.top() + 2),
            ])
        )

        # Labels: current time, and the selection length when ranging.
        painter.setPen(self.palette().text().color())
        painter.drawText(
            QRectF(0, track.bottom() + 6, self.width(), 16),
            Qt.AlignmentFlag.AlignLeft,
            f"  {_fmt_tc(self._playhead, self._fps)}",
        )
        right = (
            f"In {_fmt_tc(self._in, self._fps)}  Out {_fmt_tc(self._out, self._fps)}  "
            f"({self.range_seconds():.1f}s)  "
            if self._range_mode
            else f"{_fmt_tc(max(0, self._total - 1), self._fps)}  "
        )
        painter.drawText(
            QRectF(0, track.bottom() + 6, self.width(), 16),
            Qt.AlignmentFlag.AlignRight,
            right,
        )


def _fmt_tc(frame: int, fps: float) -> str:
    total_seconds = frame / (fps or 24.0)
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    frames = int(round(frame % (fps or 24.0)))
    return f"{minutes:02d}:{seconds:02d}.{frames:02d}"
