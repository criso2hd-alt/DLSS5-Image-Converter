"""The "3D" tab: turn a still into a short looping 2.5-D clip.

A thin UI over `creative.Renderer`. Depth is computed once per image on a worker
thread; the loop re-renders (also on the worker) whenever a control changes, and
the finished frames are cached and played back by a QTimer so playback is always
smooth. Export re-renders at the chosen output size and writes an MP4.

Kept deliberately small: presets, a few sliders, no free camera, no new models.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QSlider, QVBoxLayout, QWidget,
)

from . import creative, video
from .imaging import imread
from .onnx_depth import SMALL, OnnxDepthEngine


class _Worker(QObject):
    """Lives on a background thread. Computes depth once, renders on request."""

    depth_progress = Signal(str)
    frames_ready = Signal(object)          # list[np.ndarray]
    export_done = Signal(str)
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._engine: OnnxDepthEngine | None = None
        self._renderer: creative.Renderer | None = None

    def load(self, path: str) -> None:
        try:
            self.depth_progress.emit("Reading image…")
            bgr = imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("Could not read that image.")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if self._engine is None:
                self._engine = OnnxDepthEngine()
                self._engine.load(SMALL)
            self.depth_progress.emit("Estimating depth…")
            inv = self._engine.infer(rgb)
            self._renderer = None
            self._rgb = rgb.astype(np.float32) / 255.0
            self._inv = inv
            self.depth_progress.emit("")
        except Exception as error:  # noqa: BLE001 - surfaced in the UI
            self.failed.emit(str(error))

    def render(self, s: creative.CreativeSettings) -> None:
        try:
            if not hasattr(self, "_rgb"):
                return
            img, dep = creative.reframe(self._rgb, self._inv, creative.ASPECTS[s.aspect])
            self._renderer = creative.Renderer(img, dep)
            self.frames_ready.emit(self._renderer.render_loop(s))
        except Exception as error:  # noqa: BLE001
            self.failed.emit(str(error))

    def export(self, s: creative.CreativeSettings, path: str, long_side: int) -> None:
        try:
            if not hasattr(self, "_rgb"):
                raise RuntimeError("Open an image first.")
            # Reframe, then scale so the long side hits the requested size.
            img, dep = creative.reframe(self._rgb, self._inv, creative.ASPECTS[s.aspect])
            h, w = img.shape[:2]
            scale = long_side / max(h, w)
            if scale != 1.0:
                nw, nh = int(round(w * scale)), int(round(h * scale))
                nw -= nw % 2; nh -= nh % 2
                img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
                dep = cv2.resize(dep, (nw, nh), interpolation=cv2.INTER_AREA)
            renderer = creative.Renderer(img, dep)
            frames = renderer.render_loop(s)
            fh, fw = frames[0].shape[:2]
            writer = video.VideoWriter(Path(path), video.CODECS_BY_KEY["h264"],
                                       float(s.fps), (fw, fh))
            # Play the loop twice so a short clip reads as a loop on autoplay.
            for _ in range(2):
                for f in frames:
                    writer.write(f)
            writer.close()
            self.export_done.emit(path)
        except Exception as error:  # noqa: BLE001
            self.failed.emit(str(error))


class CreativePage(QWidget):
    """3D tab widget."""

    _request_load = Signal(str)
    _request_render = Signal(object)
    _request_export = Signal(object, str, int)

    def __init__(self) -> None:
        super().__init__()
        self.settings = creative.CreativeSettings()
        self._frames: list[np.ndarray] = []
        self._play_index = 0
        self._loaded = False

        self._thread = QThread(self)
        self._worker = _Worker()
        self._worker.moveToThread(self._thread)
        self._request_load.connect(self._worker.load)
        self._request_render.connect(self._worker.render)
        self._request_export.connect(self._worker.export)
        self._worker.depth_progress.connect(self._on_progress)
        self._worker.frames_ready.connect(self._on_frames)
        self._worker.export_done.connect(self._on_export_done)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

        # Debounce control changes into one render.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(140)
        self._debounce.timeout.connect(self._render)

        self._playback = QTimer(self)
        self._playback.timeout.connect(self._advance)

        self._build()

    # -- UI ------------------------------------------------------------------

    def _build(self) -> None:
        row = QHBoxLayout(self)
        row.setContentsMargins(16, 16, 16, 16)
        row.setSpacing(16)

        # Preview.
        self.view = QLabel("Open an image to begin.")
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumSize(480, 480)
        self.view.setStyleSheet("QLabel { background: #0a0f18; border-radius: 8px; color: #6b7688; }")
        row.addWidget(self.view, 1)

        # Controls rail.
        rail = QWidget()
        rail.setFixedWidth(300)
        col = QVBoxLayout(rail)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(12)

        blurb = QLabel("Bring a still to life: a gentle 2.5-D move through its "
                       "depth, with atmosphere. Loops seamlessly, ready to post.")
        blurb.setWordWrap(True)
        blurb.setStyleSheet("color: #8a93a6;")
        col.addWidget(blurb)

        self.open_button = QPushButton("Open image…")
        self.open_button.clicked.connect(self._open)
        col.addWidget(self.open_button)

        self.aspect_box = self._combo(list(creative.ASPECTS), self.settings.aspect,
                                      self._aspect_changed)
        self.preset_box = self._combo(list(creative.PRESETS), self.settings.preset,
                                      self._preset_changed)
        col.addLayout(self._labeled("Frame", self.aspect_box))
        col.addLayout(self._labeled("Motion", self.preset_box))

        self.depth_slider = self._slider(0, 200, int(self.settings.depth_intensity * 100),
                                         lambda v: self._set("depth_intensity", v / 100))
        self.fog_slider = self._slider(0, 100, int(self.settings.fog * 100),
                                       lambda v: self._set("fog", v / 100))
        self.embers_slider = self._slider(0, 100, int(self.settings.embers * 100),
                                          lambda v: self._set("embers", v / 100))
        self.dust_slider = self._slider(0, 100, int(self.settings.dust * 100),
                                        lambda v: self._set("dust", v / 100))
        col.addLayout(self._labeled("Depth", self.depth_slider))
        col.addLayout(self._labeled("Fog", self.fog_slider))
        col.addLayout(self._labeled("Embers", self.embers_slider))
        col.addLayout(self._labeled("Dust", self.dust_slider))

        col.addStretch(1)

        self.status = QLabel("")
        self.status.setStyleSheet("color: #8a93a6;")
        self.status.setWordWrap(True)
        col.addWidget(self.status)

        self.export_button = QPushButton("Export loop…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export)
        col.addWidget(self.export_button)

        row.addWidget(rail)

    def _combo(self, items: list[str], current: str, on_change) -> QComboBox:
        box = QComboBox()
        box.addItems(items)
        box.setCurrentText(current)
        box.currentTextChanged.connect(on_change)
        return box

    def _slider(self, lo: int, hi: int, value: int, on_change) -> QSlider:
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(value)
        s.valueChanged.connect(on_change)
        return s

    def _labeled(self, text: str, widget: QWidget) -> QVBoxLayout:
        box = QVBoxLayout()
        box.setSpacing(4)
        lab = QLabel(text)
        lab.setStyleSheet("color: #b9c1d1; font-size: 12px;")
        box.addWidget(lab)
        box.addWidget(widget)
        return box

    # -- events --------------------------------------------------------------

    def load_image(self, path: str | Path) -> None:
        """Public: load an image (also callable by the main window)."""
        self.view.setText("Estimating depth…")
        self._loaded = False
        self.export_button.setEnabled(False)
        self._playback.stop()
        self._request_load.emit(str(path))
        # Kick a render right after load finishes: the worker processes load then
        # render in order on its own thread, so queue the render now.
        self._request_render.emit(self._snapshot())

    def _open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)")
        if path:
            self.load_image(path)

    def _snapshot(self) -> creative.CreativeSettings:
        return creative.CreativeSettings(**vars(self.settings))

    def _set(self, field: str, value) -> None:
        setattr(self.settings, field, value)
        self._debounce.start()

    def _aspect_changed(self, text: str) -> None:
        self.settings.aspect = text
        self._debounce.start()

    def _preset_changed(self, text: str) -> None:
        self.settings.preset = text
        self._debounce.start()

    def _render(self) -> None:
        self.status.setText("Rendering…")
        self._request_render.emit(self._snapshot())

    def _on_progress(self, text: str) -> None:
        if text:
            self.view.setText(text)
        self.status.setText(text)

    def _on_frames(self, frames: object) -> None:
        self._frames = list(frames)
        self._loaded = True
        self._play_index = 0
        self.export_button.setEnabled(True)
        self.status.setText("")
        interval = int(1000 / max(1, self.settings.fps))
        self._playback.start(interval)

    def _advance(self) -> None:
        if not self._frames:
            return
        frame = self._frames[self._play_index % len(self._frames)]
        self._play_index += 1
        self.view.setPixmap(self._to_pixmap(frame))

    def _to_pixmap(self, frame: np.ndarray) -> QPixmap:
        data = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        data = np.ascontiguousarray(data)
        h, w = data.shape[:2]
        img = QImage(data.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img.copy())
        return pix.scaled(self.view.size(), Qt.AspectRatioMode.KeepAspectRatio,
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
        # Export at a tasteful size: long side 1440, good for social without
        # being enormous.
        self._request_export.emit(self._snapshot(), path, 1440)

    def _on_export_done(self, path: str) -> None:
        self.export_button.setEnabled(True)
        self.status.setText(f"Saved {Path(path).name}")

    def _on_failed(self, message: str) -> None:
        self._playback.stop()
        self.view.setText("Something went wrong.")
        self.status.setText(message)
        self.export_button.setEnabled(self._loaded)

    def shutdown(self) -> None:
        self._playback.stop()
        self._thread.quit()
        self._thread.wait(2000)
