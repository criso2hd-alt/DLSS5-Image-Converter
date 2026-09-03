"""Reusable pieces of the interface."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSlider,
    QVBoxLayout,
    QWidget,
)

SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".exr",
             ".hdr", ".jxr", ".wdp", ".hdp"}


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
        self.setMinimumSize(480, 320)

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

    def _content_size(self):
        return self._pixmap.size() if self._pixmap is not None else None

    def set_pixmap(self, pixmap: QPixmap | None, caption: str = "") -> None:
        # A new image at the old zoom would land the viewport somewhere
        # arbitrary, so start fitted. Re-rendering the *same* image at a new
        # depth contrast goes through here too, but that only changes colour.
        changed = self._pixmap is None or pixmap is None or (
            self._pixmap.size() != pixmap.size()
        )
        self._pixmap = pixmap
        self._caption = caption
        if changed:
            self.reset_view()
        self.update()

    def set_image_u8(self, image_rgb: np.ndarray, caption: str = "") -> None:
        self.set_pixmap(QPixmap.fromImage(to_qimage_u8(image_rgb)), caption)

    def set_image(self, image_rgb: np.ndarray, caption: str = "") -> None:
        self.set_pixmap(QPixmap.fromImage(to_qimage(image_rgb)), caption)

    def clear(self) -> None:
        self.set_pixmap(None)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt name
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        if self._pixmap is None:
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = self._display_rect(self._pixmap.size())
        painter.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))
        zoom = self._zoom_caption()
        if zoom:
            painter.setPen(self.palette().text().color())
            painter.drawText(
                QRectF(0, 6, self.width(), 20), Qt.AlignmentFlag.AlignCenter, zoom
            )
        if self._caption:
            painter.setPen(self.palette().text().color())
            painter.drawText(
                QRectF(0, self.height() - 26, self.width(), 22),
                Qt.AlignmentFlag.AlignCenter,
                self._caption,
            )


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
        self.setMouseTracking(True)

    def set_labels(self, left: str = "", right: str = "") -> None:
        """Name the two halves, for a comparison that is not before/after.

        Source-versus-result needs no labels: which side is which is obvious
        from the wipe itself. Two neural styles are not obvious at all - the
        whole difficulty is that they look similar - so an unlabelled wipe
        would be a puzzle rather than a comparison.
        """
        self._labels = (left, right) if (left or right) else None
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
        if changed:
            self.reset_view()
        self.update()

    def set_images_u8(self, before: np.ndarray, after: np.ndarray) -> None:
        """Same as set_images but for 8-bit arrays, which the grade produces."""
        pixmap = QPixmap.fromImage(to_qimage_u8(before))
        changed = self._before is None or self._before.size() != pixmap.size()
        self._before = pixmap
        self._after = QPixmap.fromImage(to_qimage_u8(after))
        if changed:
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
        painter.drawPixmap(rect, self._after, QRectF(self._after.rect()))

        # The "before" half is clipped rather than drawn scaled-and-cropped, so
        # both sides stay pixel-aligned and the seam does not shimmer as it moves.
        split_x = rect.left() + rect.width() * self._split
        painter.save()
        painter.setClipRect(QRectF(rect.left(), rect.top(), split_x - rect.left(), rect.height()))
        painter.drawPixmap(rect, self._before, QRectF(self._before.rect()))
        painter.restore()

        painter.setPen(QPen(self.palette().highlight().color(), 2))
        painter.drawLine(QPointF(split_x, rect.top()), QPointF(split_x, rect.bottom()))

        if self._labels is not None:
            left, right = self._labels
            painter.setPen(self.palette().text().color())
            # Positioned against the widget, not the image. Zoomed in, the
            # image rect runs off every edge and image-relative labels go with
            # it - which is precisely when someone is comparing detail and most
            # needs to know which half they are looking at.
            band = QRectF(10, 8, self.width() - 20, 20)
            # Each label is clipped to its own half, so it stays correct
            # wherever the divider has been dragged to.
            painter.setClipRect(QRectF(0, 0, split_x, self.height()))
            painter.drawText(band, Qt.AlignmentFlag.AlignLeft, left)
            painter.setClipRect(QRectF(split_x, 0, self.width() - split_x, self.height()))
            painter.drawText(band, Qt.AlignmentFlag.AlignRight, right)
            painter.setClipping(False)

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
    """Two full images in two panes, locked to one zoom and one pan.

    The wipe is better at spotting a change; this is better at judging one.
    Sliding a divider back and forth answers "did that move?", but comparing
    two neural styles is a question about the whole frame at once - which of
    these two pictures do I want - and that needs both of them on screen.

    The panes do not synchronise with each other. They share a single zoom and
    pan, applied within each pane, so they cannot drift apart: there is nothing
    to keep in step. Zooming about the cursor works from whichever pane the
    cursor is in, and the other pane lands on the same part of the image.
    """

    #: Space between the panes, in pixels. Wide enough to read as two images.
    GAP = 12

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._left: QPixmap | None = None
        self._right: QPixmap | None = None
        self._labels = ("", "")

    # -- geometry ------------------------------------------------------------

    def _pane_width(self) -> float:
        return max(1.0, (self.width() - self.GAP) / 2.0)

    def _pane_left(self, index: int) -> float:
        return 0.0 if index == 0 else self._pane_width() + self.GAP

    def _viewport(self) -> QRectF:
        # Pane-local: both panes are the same size, so one rect describes both.
        return QRectF(0.0, 0.0, self._pane_width(), float(self.height()))

    def _to_viewport(self, point: QPointF) -> QPointF:
        # Whichever pane the cursor is over, expressed in that pane's own
        # coordinates - so zoom-about-cursor aims at the same pixel of the
        # image in both.
        index = 0 if point.x() < self._pane_left(1) else 1
        return QPointF(point.x() - self._pane_left(index), point.y())

    def _content_size(self):
        return self._left.size() if self._left is not None else None

    # -- content -------------------------------------------------------------

    def set_images_u8(self, left: np.ndarray, right: np.ndarray) -> None:
        pixmap = QPixmap.fromImage(to_qimage_u8(left))
        changed = self._left is None or self._left.size() != pixmap.size()
        self._left = pixmap
        self._right = QPixmap.fromImage(to_qimage_u8(right))
        if changed:
            self.reset_view()
        self.update()

    def set_labels(self, left: str = "", right: str = "") -> None:
        self._labels = (left, right)
        self.update()

    def clear(self) -> None:
        self._left = self._right = None
        self.update()

    # -- painting ------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt name
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        if self._left is None or self._right is None:
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        rect = self._display_rect(self._left.size())
        for index, pixmap in enumerate((self._left, self._right)):
            offset = self._pane_left(index)
            pane = QRectF(offset, 0.0, self._pane_width(), float(self.height()))
            painter.save()
            painter.setClipRect(pane)
            painter.drawPixmap(rect.translated(offset, 0.0), pixmap, QRectF(pixmap.rect()))
            painter.restore()

            label = self._labels[index]
            if label:
                painter.setPen(self.palette().text().color())
                painter.drawText(
                    QRectF(offset, 8.0, self._pane_width(), 20.0),
                    Qt.AlignmentFlag.AlignCenter,
                    label,
                )

        # A seam, so two similar images do not read as one wide one.
        painter.setPen(QPen(self.palette().highlight().color(), 1))
        middle = self._pane_width() + self.GAP / 2
        painter.drawLine(QPointF(middle, 0.0), QPointF(middle, float(self.height())))

        zoom = self._zoom_caption()
        if zoom:
            painter.setPen(self.palette().text().color())
            painter.drawText(
                QRectF(0, self.height() - 26, self.width(), 22),
                Qt.AlignmentFlag.AlignCenter,
                zoom,
            )


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

    def set_heading(self, text: str) -> None:
        self._heading.setText(text)

    def set_status(self, text: str) -> None:
        self._detail.setText(text)

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
        name = QLabel(label)
        self._value = QLabel(f"{value:+.2f}" if minimum < 0 else f"{value:.2f}")
        self._value.setObjectName("hint")
        header.addWidget(name)
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
            name.setToolTip(tooltip)

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
