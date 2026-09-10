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
    QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget,
)

from . import creative, video

PREVIEW_LONG = 720      # long side used for the live preview render
EXPORT_LONG = 1440      # long side used for the exported MP4


class _Worker(QObject):
    """Lives on a background thread. Renders single frames on demand.

    Rendering one frame per playback tick (rather than precomputing a whole
    loop) is what keeps the preview responsive: a control change is reflected on
    the very next frame, and the GPU readback cost is paid one frame at a time.
    """

    frame_ready = Signal(int, object)      # index, np.ndarray
    export_done = Signal(str)
    lama_ready = Signal(bool, str)         # ok, message
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._img: np.ndarray | None = None
        self._dep: np.ndarray | None = None
        self._renderer = None
        self._rkey = None
        self._settings = creative.CreativeSettings()

    def set_source(self, image: object, depth: object) -> None:
        self._img = None if image is None else np.ascontiguousarray(image)
        self._dep = None if depth is None else np.ascontiguousarray(depth)
        self._renderer = None

    def set_settings(self, s: creative.CreativeSettings) -> None:
        self._settings = s

    def _inpainter(self, s):
        from . import inpaint
        return inpaint.build_inpainter(s.use_lama)

    def download_lama(self) -> None:
        try:
            from . import inpaint
            inpaint.download(progress=lambda m: self.lama_ready.emit(True, m))
            self._renderer = None                      # rebuild with LaMa next
            self.lama_ready.emit(True, "")
        except Exception as error:  # noqa: BLE001
            self.lama_ready.emit(False, str(error))

    def render_one(self, index: int) -> None:
        try:
            if self._img is None:
                return
            s = self._settings
            key = (s.aspect, s.use_lama, round(s.depth_contrast, 3))
            if self._renderer is None or self._rkey != key:
                img, dep = creative.reframe(self._img, self._dep,
                                            creative.ASPECTS[s.aspect])
                self._renderer = creative.Renderer(
                    img, dep, long_side=PREVIEW_LONG, inpainter=self._inpainter(s),
                    depth_contrast=s.depth_contrast)
                self._rkey = key
            self.frame_ready.emit(index, self._renderer.frame(s, index % s.frames))
        except Exception as error:  # noqa: BLE001 - surfaced in the UI
            self.failed.emit(str(error))

    def export(self, s: creative.CreativeSettings, path: str) -> None:
        try:
            if self._img is None:
                raise RuntimeError("No image to export.")
            img, dep = creative.reframe(self._img, self._dep,
                                        creative.ASPECTS[s.aspect])
            frames = creative.Renderer(
                img, dep, long_side=EXPORT_LONG, inpainter=self._inpainter(s),
                depth_contrast=s.depth_contrast).render_loop(s)
            fh, fw = frames[0].shape[:2]
            # The cached preview mesh is now overwritten on the shared GPU
            # renderer; force a rebuild on the next preview.
            self._renderer = None
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
    _request_settings = Signal(object)
    _request_frame = Signal(int)
    _request_export = Signal(object, str)
    _request_download_lama = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.settings = creative.CreativeSettings()
        self._play_index = 0
        self._has_source = False
        self._in_flight = False

        self._thread = QThread(self)
        self._worker = _Worker()
        self._worker.moveToThread(self._thread)
        self._request_set_source.connect(self._worker.set_source)
        self._request_settings.connect(self._worker.set_settings)
        self._request_frame.connect(self._worker.render_one)
        self._request_export.connect(self._worker.export)
        self._request_download_lama.connect(self._worker.download_lama)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.export_done.connect(self._on_export_done)
        self._worker.lama_ready.connect(self._on_lama)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

        # Pull-based playback: request the next frame only after the last one
        # arrives, paced to the target fps. Playback runs at whatever rate the
        # GPU sustains, and any control change is picked up on the next frame.
        self._pacer = QTimer(self)
        self._pacer.setSingleShot(True)
        self._pacer.timeout.connect(self._next_frame)

        # Structural changes (depth contrast, aspect) rebuild the mesh, so debounce
        # them: only apply after the user pauses, not on every slider tick.
        self._struct_debounce = QTimer(self)
        self._struct_debounce.setSingleShot(True)
        self._struct_debounce.setInterval(350)
        self._struct_debounce.timeout.connect(self._push_settings)

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
            "Heads up: this reconstructs 3-D from a single depth map, so it is a "
            "real camera move but over one surface, not a full model. Hard edges "
            "can stretch under big moves, and not every image holds up. Keep the "
            "moves gentle and experiment.")
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet(
            "color: #8590a3; font-size: 11px; font-style: italic; "
            "background: #10151f; border: 1px solid #1c2534; border-radius: 6px; padding: 8px;")
        col.addWidget(disclaimer)

        self.aspect_box = self._combo(list(creative.ASPECTS), self.settings.aspect,
                                      self._aspect_changed)
        self.preset_box = self._combo(list(creative.PRESETS), self.settings.preset,
                                      self._preset_changed)
        self.view_box = self._combo(list(creative.VIEW_MODES), self.settings.view,
                                    self._view_changed)
        self.lama_check = QCheckBox("Rebuild background (LaMa · ~207 MB)")
        self.lama_check.setStyleSheet("color: #b9c1d1;")
        self.lama_check.setToolTip(
            "Reconstruct what's hidden behind foreground objects with a learned "
            "model, instead of the built-in blur fill. Downloads once.")
        self.lama_check.toggled.connect(self._lama_toggled)
        lama_row = QVBoxLayout(); lama_row.addWidget(self.lama_check)
        col.addWidget(self._card("Scene", [
            self._labeled("Frame", self.aspect_box),
            self._labeled("Camera move", self.preset_box),
            self._labeled("View", self.view_box),
            self._labeled("Depth strength", self._fslider("depth_intensity", 2.0)),
            self._labeled("Depth contrast", self._struct_slider("depth_contrast", 2.0)),
            lama_row,
        ]))
        col.addWidget(self._card("Fog", [
            self._labeled("Amount", self._fslider("fog", 1.0)),
            self._labeled("Plane (near → far)", self._fslider("fog_plane", 1.0)),
        ]))
        col.addWidget(self._card("Flare", [
            self._labeled("Bloom", self._fslider("flare", 1.0)),
        ]))
        col.addWidget(self._card("Embers", [
            self._labeled("Amount", self._fslider("embers", 1.0)),
            self._labeled("Plane (near → far)", self._fslider("ember_plane", 1.0)),
            self._labeled("Direction", self._dirslider("ember_dir")),
            self._labeled("Speed", self._fslider("ember_speed", 1.0)),
            self._labeled("Size", self._fslider("ember_size", 1.0)),
        ]))
        col.addWidget(self._card("Dust", [
            self._labeled("Amount", self._fslider("dust", 1.0)),
            self._labeled("Plane (near → far)", self._fslider("dust_plane", 1.0)),
            self._labeled("Direction", self._dirslider("dust_dir")),
            self._labeled("Speed", self._fslider("dust_speed", 1.0)),
            self._labeled("Size", self._fslider("dust_size", 1.0)),
        ]))
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

    def _card(self, title: str, rows: list) -> QFrame:
        """A titled card grouping one effect's controls, so the rail reads as a
        few tidy sections instead of a wall of sliders."""
        card = QFrame()
        card.setObjectName("creativeCard")
        card.setStyleSheet(
            "QFrame#creativeCard { background: #10151f; border: 1px solid #1c2534; "
            "border-radius: 8px; }")
        box = QVBoxLayout(card)
        box.setContentsMargins(12, 10, 12, 12)
        box.setSpacing(8)
        head = QLabel(title.upper())
        head.setStyleSheet("color: #6f7a8e; font-size: 11px; font-weight: 600; "
                           "letter-spacing: 1px; border: none;")
        box.addWidget(head)
        for r in rows:
            box.addLayout(r)
        return card

    def _fslider(self, field: str, hi: float) -> QSlider:
        """A slider over a float field 0..hi, mapped through 0..100 ticks."""
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(0, 100)
        s.setValue(int(round(getattr(self.settings, field) / hi * 100)))
        s.valueChanged.connect(lambda v: self._set(field, v / 100 * hi))
        self._controls.append(s)
        return s

    def _struct_slider(self, field: str, hi: float) -> QSlider:
        """Like _fslider, but for a field that rebuilds the mesh: debounced."""
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(0, 100)
        s.setValue(int(round(getattr(self.settings, field) / hi * 100)))
        s.valueChanged.connect(lambda v: self._set_struct(field, v / 100 * hi))
        self._controls.append(s)
        return s

    def _set_struct(self, field, value) -> None:
        setattr(self.settings, field, value)
        if self._has_source:
            self._struct_debounce.start()

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
        for w in [self.aspect_box, self.preset_box, self.view_box,
                  self.lama_check, *self._controls]:
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
            self._pacer.stop()
            self.view.setText(self._empty_text())
            self._request_set_source.emit(None, None)
            return
        self._has_source = True
        self._set_controls_enabled(True)
        self.export_button.setEnabled(True)
        self._request_set_source.emit(np.asarray(image), np.asarray(depth))
        self._push_settings()
        self.status.setText("Rendering…")
        self._play_index = 0
        if not self._in_flight:                     # kick the playback loop
            self._next_frame()

    # -- events --------------------------------------------------------------

    def _push_settings(self) -> None:
        self._request_settings.emit(creative.CreativeSettings(**vars(self.settings)))

    def _set(self, field, value) -> None:
        setattr(self.settings, field, value)
        if self._has_source:
            self._push_settings()

    def _aspect_changed(self, text) -> None:
        self.settings.aspect = text
        if self._has_source:
            self._push_settings()

    def _preset_changed(self, text) -> None:
        self.settings.preset = text
        if self._has_source:
            self._push_settings()

    def _view_changed(self, text) -> None:
        self.settings.view = text
        if self._has_source:
            self._push_settings()

    def _lama_toggled(self, checked: bool) -> None:
        self.settings.use_lama = checked
        if checked:
            from . import inpaint
            if not inpaint.is_downloaded():
                self.status.setText("Downloading LaMa background model "
                                    "(~207 MB, one time)…")
                self.lama_check.setEnabled(False)
                self._request_download_lama.emit()
        if self._has_source:
            self._push_settings()               # cache key changes -> rebuild

    def _on_lama(self, ok: bool, message: str) -> None:
        if not ok:
            self.lama_check.setEnabled(True)
            self.lama_check.setChecked(False)
            self.status.setText(f"LaMa unavailable: {message}")
            return
        if message:                             # progress text
            self.status.setText(message)
            return
        self.lama_check.setEnabled(True)         # download finished
        self.status.setText("Background model ready.")
        if self._has_source:
            self._push_settings()

    def _next_frame(self) -> None:
        if not self._has_source:
            self._in_flight = False
            return
        self._in_flight = True
        self._request_frame.emit(self._play_index)

    def _on_frame(self, index: int, frame) -> None:
        self.status.setText("")
        self.view.setPixmap(self._to_pixmap(frame))
        self._play_index = (index + 1) % max(1, self.settings.frames)
        # Pace to the target fps; if a render already took longer, go again now.
        self._pacer.start(int(1000 / max(1, self.settings.fps)))

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
        self._request_export.emit(creative.CreativeSettings(**vars(self.settings)), path)

    def _on_export_done(self, path: str) -> None:
        self.export_button.setEnabled(True)
        self.status.setText(f"Saved {Path(path).name}")

    def _on_failed(self, message: str) -> None:
        self._in_flight = False
        self.status.setText(message)
        self.export_button.setEnabled(self._has_source)

    def shutdown(self) -> None:
        self._pacer.stop()
        self._thread.quit()
        self._thread.wait(2000)
