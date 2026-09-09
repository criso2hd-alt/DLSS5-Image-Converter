"""The "3D" tab: turn the already-processed still into a looping 2.5-D clip.

Architecture note (important): this tab does NOT create or process an image. The
Single-image tab owns creation: it runs DLSS and computes depth. This tab is a
*sub-process* of that finished image. It receives the enhanced image and its
depth from the main window and only ever renders parallax + effects over them,
so every control change is instant, never a re-processing step.

Depth is never recomputed here. The loop re-renders on a worker thread at a
reduced preview size (so changes feel immediate) and plays back from a frame
cache; export re-renders at full quality and writes a looping MP4.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget,
)

from . import creative, video

PREVIEW_LONG = 600      # long side used for the live preview render
EXPORT_LONG = 1440      # long side used for the exported MP4


def _fit(image: np.ndarray, depth: np.ndarray, long_side: int
         ) -> tuple[np.ndarray, np.ndarray]:
    """Scale image+depth so the long side is `long_side` (depth matched to it)."""
    h, w = image.shape[:2]
    scale = long_side / max(h, w)
    if scale < 1.0:
        nw, nh = int(round(w * scale)), int(round(h * scale))
        nw -= nw % 2; nh -= nh % 2
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    if depth.shape[:2] != image.shape[:2]:
        depth = cv2.resize(depth, (image.shape[1], image.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
    return image, depth


class _Worker(QObject):
    """Lives on a background thread. Renders loops from a held source image."""

    frames_ready = Signal(object)          # list[np.ndarray]
    export_done = Signal(str)
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._img: np.ndarray | None = None
        self._dep: np.ndarray | None = None

    def set_source(self, image: object, depth: object) -> None:
        # Stored full-resolution; preview and export scale from these. Copies,
        # because the arrays belong to the main window and it may reuse them.
        self._img = None if image is None else np.ascontiguousarray(image)
        self._dep = None if depth is None else np.ascontiguousarray(depth)

    def render(self, s: creative.CreativeSettings) -> None:
        try:
            if self._img is None:
                return
            img, dep = _fit(self._img, self._dep, PREVIEW_LONG)
            img, dep = creative.reframe(img, dep, creative.ASPECTS[s.aspect])
            self.frames_ready.emit(creative.Renderer(img, dep).render_loop(s))
        except Exception as error:  # noqa: BLE001 - surfaced in the UI
            self.failed.emit(str(error))

    def export(self, s: creative.CreativeSettings, path: str) -> None:
        try:
            if self._img is None:
                raise RuntimeError("No image to export.")
            img, dep = _fit(self._img, self._dep, EXPORT_LONG)
            img, dep = creative.reframe(img, dep, creative.ASPECTS[s.aspect])
            frames = creative.Renderer(img, dep).render_loop(s)
            fh, fw = frames[0].shape[:2]
            writer = video.VideoWriter(Path(path), video.CODECS_BY_KEY["h264"],
                                       float(s.fps), (fw, fh))
            for _ in range(2):             # twice, so a short clip reads as a loop
                for f in frames:
                    writer.write(f)
            writer.close()
            self.export_done.emit(path)
        except Exception as error:  # noqa: BLE001
            self.failed.emit(str(error))


class CreativePage(QWidget):
    """3D tab widget. Inherits the processed image from the main window."""

    _request_set_source = Signal(object, object)
    _request_render = Signal(object)
    _request_export = Signal(object, str)

    def __init__(self) -> None:
        super().__init__()
        self.settings = creative.CreativeSettings()
        self._frames: list[np.ndarray] = []
        self._play_index = 0
        self._has_source = False

        self._thread = QThread(self)
        self._worker = _Worker()
        self._worker.moveToThread(self._thread)
        self._request_set_source.connect(self._worker.set_source)
        self._request_render.connect(self._worker.render)
        self._request_export.connect(self._worker.export)
        self._worker.frames_ready.connect(self._on_frames)
        self._worker.export_done.connect(self._on_export_done)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(90)
        self._debounce.timeout.connect(self._render)

        self._playback = QTimer(self)
        self._playback.timeout.connect(self._advance)

        self._build()

    # -- UI ------------------------------------------------------------------

    def _build(self) -> None:
        row = QHBoxLayout(self)
        row.setContentsMargins(16, 16, 16, 16)
        row.setSpacing(16)

        self.view = QLabel(self._empty_text())
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setWordWrap(True)
        self.view.setMinimumSize(480, 480)
        self.view.setStyleSheet(
            "QLabel { background: #0a0f18; border-radius: 8px; color: #6b7688; padding: 24px; }")
        row.addWidget(self.view, 1)

        # Controls live in a scroll area: there are enough of them now that a
        # fixed rail would clip on a short window.
        inner = QWidget()
        col = QVBoxLayout(inner)
        col.setContentsMargins(0, 0, 8, 0)
        col.setSpacing(10)
        self._controls: list[QWidget] = []

        blurb = QLabel("Bring the converted image to life: a gentle 2.5-D move "
                       "through its depth, with atmosphere. Loops seamlessly, "
                       "ready to post.")
        blurb.setWordWrap(True)
        blurb.setStyleSheet("color: #8a93a6;")
        col.addWidget(blurb)

        disclaimer = QLabel(
            "Heads up: this is a 2.5-D parallax effect, not true 3-D. It reshapes "
            "the one photo along its estimated depth, so it is for experimenting "
            "with depth and motion, and not every image will hold up. Keep the "
            "moves gentle.")
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet(
            "color: #8590a3; font-size: 11px; font-style: italic; "
            "background: #10151f; border: 1px solid #1c2534; border-radius: 6px; padding: 8px;")
        col.addWidget(disclaimer)

        self.aspect_box = self._combo(list(creative.ASPECTS), self.settings.aspect,
                                      self._aspect_changed)
        self.preset_box = self._combo(list(creative.PRESETS), self.settings.preset,
                                      self._preset_changed)
        col.addLayout(self._labeled("Frame", self.aspect_box))
        col.addLayout(self._labeled("Motion", self.preset_box))
        col.addLayout(self._labeled("Depth strength",
                                    self._fslider("depth_intensity", 2.0)))

        col.addWidget(self._section("Fog"))
        col.addLayout(self._labeled("Amount", self._fslider("fog", 1.0)))
        col.addLayout(self._labeled("Plane (near → far)",
                                    self._fslider("fog_plane", 1.0)))

        col.addWidget(self._section("Embers"))
        col.addLayout(self._labeled("Amount", self._fslider("embers", 1.0)))
        col.addLayout(self._labeled("Plane (near → far)",
                                    self._fslider("ember_plane", 1.0)))
        col.addLayout(self._labeled("Direction", self._dirslider("ember_dir")))
        col.addLayout(self._labeled("Speed", self._fslider("ember_speed", 1.0)))
        col.addLayout(self._labeled("Size", self._fslider("ember_size", 1.0)))

        col.addWidget(self._section("Dust"))
        col.addLayout(self._labeled("Amount", self._fslider("dust", 1.0)))
        col.addLayout(self._labeled("Plane (near → far)",
                                    self._fslider("dust_plane", 1.0)))
        col.addLayout(self._labeled("Direction", self._dirslider("dust_dir")))
        col.addLayout(self._labeled("Speed", self._fslider("dust_speed", 1.0)))
        col.addLayout(self._labeled("Size", self._fslider("dust_size", 1.0)))

        col.addStretch(1)

        rail = QScrollArea()
        rail.setWidget(inner)
        rail.setWidgetResizable(True)
        rail.setFrameShape(QFrame.Shape.NoFrame)
        rail.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        rail.setFixedWidth(320)

        railcol = QWidget()
        rc = QVBoxLayout(railcol)
        rc.setContentsMargins(0, 0, 0, 0)
        rc.setSpacing(10)
        rc.addWidget(rail, 1)
        self.status = QLabel("")
        self.status.setStyleSheet("color: #8a93a6;")
        self.status.setWordWrap(True)
        rc.addWidget(self.status)
        self.export_button = QPushButton("Export loop…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export)
        rc.addWidget(self.export_button)
        railcol.setFixedWidth(320)

        row.addWidget(railcol)
        self._set_controls_enabled(False)

    def _section(self, title: str) -> QLabel:
        lab = QLabel(title.upper())
        lab.setStyleSheet("color: #6f7a8e; font-size: 11px; font-weight: 600; "
                          "letter-spacing: 1px; margin-top: 6px;")
        return lab

    def _fslider(self, field: str, hi: float) -> QSlider:
        """A slider over a float field 0..hi, mapped through 0..100 ticks."""
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(0, 100)
        s.setValue(int(round(getattr(self.settings, field) / hi * 100)))
        s.valueChanged.connect(lambda v: self._set(field, v / 100 * hi))
        self._controls.append(s)
        return s

    def _dirslider(self, field: str) -> QSlider:
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(0, 360)
        s.setValue(int(getattr(self.settings, field)))
        s.valueChanged.connect(lambda v: self._set(field, float(v)))
        self._controls.append(s)
        return s

    def _empty_text(self) -> str:
        return ("Convert or open an image in the Single image tab.\n\n"
                "It carries straight into here, no reprocessing.")

    def _combo(self, items, current, on_change) -> QComboBox:
        box = QComboBox(); box.addItems(items); box.setCurrentText(current)
        box.currentTextChanged.connect(on_change)
        return box

    def _labeled(self, text, widget) -> QVBoxLayout:
        box = QVBoxLayout(); box.setSpacing(4)
        lab = QLabel(text); lab.setStyleSheet("color: #b9c1d1; font-size: 12px;")
        box.addWidget(lab); box.addWidget(widget)
        return box

    def _set_controls_enabled(self, on: bool) -> None:
        for w in [self.aspect_box, self.preset_box, *self._controls]:
            w.setEnabled(on)

    # -- source intake -------------------------------------------------------

    def set_source(self, image, depth) -> None:
        """Receive the finished image + its depth from the main window.

        `image` is 0..1 float RGB (the enhanced result, or the source if not yet
        converted); `depth` is the inverse depth already computed for it. No
        processing happens here.
        """
        if image is None or depth is None:
            self._has_source = False
            self._set_controls_enabled(False)
            self.export_button.setEnabled(False)
            self._playback.stop()
            self.view.setText(self._empty_text())
            self._request_set_source.emit(None, None)
            return
        self._has_source = True
        self._set_controls_enabled(True)
        self._request_set_source.emit(np.asarray(image), np.asarray(depth))
        self.status.setText("Rendering…")
        self._request_render.emit(self._snapshot())

    # -- events --------------------------------------------------------------

    def _snapshot(self) -> creative.CreativeSettings:
        return creative.CreativeSettings(**vars(self.settings))

    def _set(self, field, value) -> None:
        setattr(self.settings, field, value)
        if self._has_source:
            self._debounce.start()

    def _aspect_changed(self, text) -> None:
        self.settings.aspect = text
        if self._has_source:
            self._debounce.start()

    def _preset_changed(self, text) -> None:
        self.settings.preset = text
        if self._has_source:
            self._debounce.start()

    def _render(self) -> None:
        if self._has_source:
            self.status.setText("Rendering…")
            self._request_render.emit(self._snapshot())

    def _on_frames(self, frames) -> None:
        self._frames = list(frames)
        self._play_index = 0
        self.export_button.setEnabled(True)
        self.status.setText("")
        self._playback.start(int(1000 / max(1, self.settings.fps)))

    def _advance(self) -> None:
        if not self._frames:
            return
        frame = self._frames[self._play_index % len(self._frames)]
        self._play_index += 1
        self.view.setPixmap(self._to_pixmap(frame))

    def _to_pixmap(self, frame: np.ndarray) -> QPixmap:
        data = np.ascontiguousarray((np.clip(frame, 0, 1) * 255).astype(np.uint8))
        h, w = data.shape[:2]
        img = QImage(data.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(img.copy()).scaled(
            self.view.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)

    def _export(self) -> None:
        if not video.is_available():
            QMessageBox.information(
                self, "Video engine needed",
                "The video engine isn't installed yet. Open the Video tab once "
                "to set it up, then export here.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export loop", "living_photo.mp4",
                                              "MP4 video (*.mp4)")
        if not path:
            return
        self.status.setText("Exporting…")
        self.export_button.setEnabled(False)
        self._request_export.emit(self._snapshot(), path)

    def _on_export_done(self, path: str) -> None:
        self.export_button.setEnabled(True)
        self.status.setText(f"Saved {Path(path).name}")

    def _on_failed(self, message: str) -> None:
        self.status.setText(message)
        self.export_button.setEnabled(self._has_source)

    def shutdown(self) -> None:
        self._playback.stop()
        self._thread.quit()
        self._thread.wait(2000)
