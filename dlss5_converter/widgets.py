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

SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".exr", ".hdr"}


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
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.exr *.hdr)",
        )
        if chosen:
            self.opened.emit(Path(chosen))

    def _restyle(self) -> None:
        # Qt does not re-evaluate property selectors on its own.
        self.style().unpolish(self)
        self.style().polish(self)


class ImageView(QWidget):
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
        self.setMinimumSize(480, 320)

    def set_pixmap(self, pixmap: QPixmap | None, caption: str = "") -> None:
        self._pixmap = pixmap
        self._caption = caption
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
        size = self._pixmap.size()
        scale = min(self.width() / size.width(), self.height() / size.height())
        width, height = size.width() * scale, size.height() * scale
        rect = QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)
        painter.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))
        if self._caption:
            painter.setPen(self.palette().text().color())
            painter.drawText(
                QRectF(0, self.height() - 26, self.width(), 22),
                Qt.AlignmentFlag.AlignCenter,
                self._caption,
            )


class WipeView(QWidget):
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
        self.setMinimumSize(480, 320)
        self.setMouseTracking(True)

    def set_images(self, before: np.ndarray, after: np.ndarray) -> None:
        self._before = QPixmap.fromImage(to_qimage(before))
        self._after = QPixmap.fromImage(to_qimage(after))
        self.update()

    def clear(self) -> None:
        self._before = self._after = None
        self.update()

    def _target_rect(self) -> QRectF:
        if self._before is None:
            return QRectF()
        size = self._before.size()
        scale = min(self.width() / size.width(), self.height() / size.height())
        width, height = size.width() * scale, size.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

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

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt name
        self._dragging = True
        self._move_split(event.position().x())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt name
        if self._dragging:
            self._move_split(event.position().x())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt name
        self._dragging = False

    def _move_split(self, x: float) -> None:
        rect = self._target_rect()
        if rect.width() <= 0:
            return
        self._split = float(np.clip((x - rect.left()) / rect.width(), 0.0, 1.0))
        self.update()


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
    ) -> None:
        super().__init__(parent)
        self._on_change = on_change
        self._maximum = float(maximum)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(2)

        header = QHBoxLayout()
        name = QLabel(label)
        self._value = QLabel(f"{value:.2f}")
        self._value.setObjectName("hint")
        header.addWidget(name)
        header.addStretch(1)
        header.addWidget(self._value)
        layout.addLayout(header)

        # Sliders are integers. The step is fixed at 0.01 of a unit rather than
        # a fraction of the range, so a 0..2 slider gets 200 positions and the
        # readout stays aimable at the same precision as a 0..1 one.
        self._steps = max(1, int(round(self._maximum * 100)))
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, self._steps)
        self._slider.setValue(self._to_raw(value))
        self._slider.valueChanged.connect(self._changed)
        layout.addWidget(self._slider)

        if tooltip:
            self.setToolTip(tooltip)
            name.setToolTip(tooltip)

    def _to_raw(self, value: float) -> int:
        scaled = float(value) / self._maximum * self._steps
        return int(round(min(self._steps, max(0, scaled))))

    def _changed(self, raw: int) -> None:
        value = raw / self._steps * self._maximum
        self._value.setText(f"{value:.2f}")
        self._on_change(value)

    def set_value(self, value: float) -> None:
        self._slider.setValue(self._to_raw(value))
