"""The "3D" tab: a Gaussian-splat scene editor built from the converted image.

Architecture note (important): this tab does NOT create or process an image. The
Single-image tab owns creation: it runs DLSS and computes depth. This tab
receives the finished image and its depth from the main window and builds a
splat scene from them (splat3d), then lets the user direct a camera through it
with keyframes, place volumetric fog and particles, and export a video.

Layout follows Depth Animator: the camera's view and a free 3-D scene view side
by side (the scene view draws the camera's frustum and path, and its gizmo moves
the camera or the selected effect), a keyframe timeline spanning the bottom,
and a control rail on the right.

Threads: wgpu objects belong to the thread that made them. The GUI thread owns
the renderer that draws both viewports. The worker thread builds scenes (pure
numpy), bakes the background fill and exports, each with its own renderer, so a
long job never freezes the editor.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QScrollArea, QSlider, QSplitter, QVBoxLayout, QWidget,
)

from . import splat3d, video
from .animation3d import CameraKey, CameraTrack, Easing, key_to_camera, track_from_preset
from .effects3d import EffectsState, EffectsTrack, emitter_preset, volume_preset
from .timeline3d import TimelineWidget
from .viewport3d import VIEW_MODES, EditorViewport
from .widgets import Spinner

#: Long side the splat scene is built at. Every source pixel becomes a splat,
#: so this sets both detail and GPU memory: 2560 is ~3.7M splats (~240 MB),
#: sharp at 1440p and still clean at 4K.
SCENE_LONG = 2560
#: Long side of the renders used to find and fill holes along the camera move.
BAKE_LONG = 1280
#: Long side SHARP scenes are built for. SHARP always sees a 1536 square; this
#: only sets the photo-camera depth map the fill works against.
SHARP_LONG = 1600
SOURCES = {"Standard": "standard", "High quality (SHARP)": "sharp"}
PIVOT_Z = -(splat3d.NEAR + splat3d.FAR) * 0.5
PRESETS = ["Orbit", "Drift", "Push in", "Pull out", "Vertigo", "Static"]
RESOLUTIONS = {
    "4K  3840 x 2160": (3840, 2160),
    "1440p  2560 x 1440": (2560, 1440),
    "1080p  1920 x 1080": (1920, 1080),
    "720p  1280 x 720": (1280, 720),
}
VOLUME_TYPES = ["Fog", "Smoke", "Fire", "Cloud", "Godrays"]
PARTICLE_TYPES = ["Embers", "Dust", "Snow", "Smoke", "Fire", "Clouds"]
BACKGROUND = (0.02, 0.021, 0.026)


def _to_rgb8(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        if img.max() <= 1.5:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.dstack([img] * 3)
    return np.ascontiguousarray(img[:, :, :3])


def fit_size(aspect: float, box: tuple[int, int]) -> tuple[int, int]:
    """Largest even size of `aspect` inside `box`, with the box's long side
    matched to the image's long side (a portrait image exports portrait)."""
    bw, bh = box
    if aspect < 1.0:
        bw, bh = bh, bw
    w = bw
    h = int(round(w / aspect))
    if h > bh:
        h = bh
        w = int(round(h * aspect))
    return max(2, w - w % 2), max(2, h - h % 2)


def camera_matrices(camera, size) -> tuple[np.ndarray, np.ndarray]:
    camera.near, camera.far = 0.05, 60.0
    return camera.view_matrix(), camera.projection_matrix(size[0] / max(size[1], 1))


class _SceneView:
    """What EditorViewport expects of a renderer: render_view(camera, size...)."""

    def __init__(self, renderer: splat3d.SplatRenderer) -> None:
        self.renderer = renderer

    def render_view(self, camera, size, mode="textured", effects=None, time_seconds=0.0,
                    **_ignored) -> np.ndarray:
        view, proj = camera_matrices(camera, size)
        code = {"depth": 1, "pointcloud": 2}.get(mode, 0)
        return self.renderer.render_u8(view, proj, size, BACKGROUND,
                                       effects if code == 0 else None, time_seconds, code)


class _Worker(QObject):
    """Background jobs. Owns its own renderer, created on first use."""

    scene_ready = Signal(object, int)       # SplatScene, generation
    progress = Signal(str)
    bake_done = Signal(object, int)         # SplatScene, generation
    export_done = Signal(str)
    lama_ready = Signal(bool, str)
    sharp_ready = Signal(bool, str)
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._renderer: splat3d.SplatRenderer | None = None
        self.cancel = False

    def _get_renderer(self) -> splat3d.SplatRenderer:
        if self._renderer is None:
            self._renderer = splat3d.SplatRenderer()
        return self._renderer

    def build(self, rgb8: np.ndarray, depth: np.ndarray, contrast: float, gen: int,
              mode: str = "standard") -> None:
        try:
            if mode == "sharp":
                from . import sharp3d
                self.progress.emit("Building the scene with SHARP… (about 20 s)")
                h, w = rgb8.shape[:2]
                s = min(1.0, SHARP_LONG / max(h, w))
                if s < 1.0:
                    rgb8 = cv2.resize(rgb8, (int(w * s) // 2 * 2, int(h * s) // 2 * 2),
                                      interpolation=cv2.INTER_AREA)
                scene = sharp3d.build_scene(rgb8, self._get_renderer(), contrast)
                self.scene_ready.emit(scene, gen)
                return
            self.progress.emit("Building the splat scene…")
            h, w = rgb8.shape[:2]
            s = min(1.0, SCENE_LONG / max(h, w))
            if s < 1.0:
                rgb8 = cv2.resize(rgb8, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            d = cv2.resize(depth.astype(np.float32), (rgb8.shape[1], rgb8.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
            scene = splat3d.build_splats(rgb8, d, None, depth_contrast=contrast, backfill=False)
            self.scene_ready.emit(scene, gen)
        except Exception as error:  # noqa: BLE001 - surfaced in the UI
            self.failed.emit(f"Could not build the scene: {error}")

    def bake(self, scene: splat3d.SplatScene, poses_fn, size, gen: int) -> None:
        try:
            from . import inpaint
            inp = inpaint.build_inpainter(True)
            r = self._get_renderer()
            r.set_scene(scene)
            poses = poses_fn(size)
            label = "LaMa" if inp is not None else "quick fill"
            splat3d.fill_along_path(
                scene, r, poses, size, inp,
                progress=lambda i, n: self.progress.emit(
                    f"Filling what the move reveals ({label})… {i}/{n}"))
            self.bake_done.emit(scene, gen)
        except Exception as error:  # noqa: BLE001
            self.failed.emit(f"Background fill failed: {error}")

    def download_sharp(self) -> None:
        try:
            from . import sharp3d
            sharp3d.download(progress=lambda m: self.progress.emit(m))
            self.sharp_ready.emit(True, "")
        except Exception as error:  # noqa: BLE001
            self.sharp_ready.emit(False, str(error))

    def download_lama(self) -> None:
        try:
            from . import inpaint
            inpaint.download(progress=lambda m: self.progress.emit(m))
            self.progress.emit("Preparing LaMa for the GPU…")
            inpaint.LamaInpainter()._ensure_session()
            self.lama_ready.emit(True, "")
        except Exception as error:  # noqa: BLE001
            self.lama_ready.emit(False, str(error))

    def export(self, scene, job: dict) -> None:
        try:
            self.cancel = False
            r = self._get_renderer()
            r.set_scene(scene)
            w, h = job["size"]
            fps = job["fps"]
            total = max(1, int(round(job["duration"] * fps)))
            codec = video.CODECS_BY_KEY[job["codec"]]
            writer = video.VideoWriter(Path(job["path"]), codec, float(fps), (w, h))
            try:
                for i in range(total):
                    if self.cancel:
                        raise RuntimeError("Export cancelled.")
                    t = i / fps
                    key = job["track"].evaluate(t)
                    cam = key_to_camera(key)
                    view, proj = camera_matrices(cam, (w, h))
                    fx = job["effects_track"].evaluate(t, job["effects"])
                    writer.write(r.render_u8(view, proj, (w, h), BACKGROUND, fx, t))
                    if i % 4 == 0:
                        self.progress.emit(f"Exporting… frame {i + 1}/{total}")
            finally:
                writer.close()
            self.export_done.emit(job["path"])
        except Exception as error:  # noqa: BLE001
            self.failed.emit(str(error))


class CreativePage(QWidget):
    """3D tab widget. Inherits the processed image from the main window."""

    _request_build = Signal(object, object, float, int, str)
    _request_sharp = Signal()
    _request_bake = Signal(object, object, object, int)
    _request_export = Signal(object, object)
    _request_lama = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._rgb8: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._scene: splat3d.SplatScene | None = None
        # The scene as built from the photo, before any fill. Every bake starts
        # from this, so re-baking after a camera change replaces the old fill
        # instead of stacking a second one on top of it.
        self._base_scene: splat3d.SplatScene | None = None
        self._gen = 0
        self._renderer: splat3d.SplatRenderer | None = None
        self._renderer_error = ""
        self.track = CameraTrack()
        self.effects = EffectsState()
        self.effects_track = EffectsTrack()
        self.duration = 6.0
        self.fps = 30
        self.time = 0.0
        self._playing = False
        self._baked = False

        self._thread = QThread(self)
        self._worker = _Worker()
        self._worker.moveToThread(self._thread)
        self._request_build.connect(self._worker.build)
        self._request_bake.connect(self._worker.bake)
        self._request_export.connect(self._worker.export)
        self._request_lama.connect(self._worker.download_lama)
        self._request_sharp.connect(self._worker.download_sharp)
        self._worker.sharp_ready.connect(self._on_sharp)
        self._worker.scene_ready.connect(self._on_scene)
        self._worker.bake_done.connect(self._on_baked)
        self._worker.export_done.connect(self._on_exported)
        self._worker.lama_ready.connect(self._on_lama)
        self._worker.progress.connect(self._set_status)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

        self._play_timer = QTimer(self)
        self._play_timer.setInterval(16)
        self._play_timer.timeout.connect(self._advance)
        self._redraw = QTimer(self)
        self._redraw.setSingleShot(True)
        self._redraw.timeout.connect(self._render_preview)
        self._rebuild = QTimer(self)
        self._rebuild.setSingleShot(True)
        self._rebuild.setInterval(400)
        self._rebuild.timeout.connect(self._start_build)
        self._build_ui()
        self._reset_track()

    # -- UI ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)
        body = QHBoxLayout()
        body.setSpacing(10)
        outer.addLayout(body, 1)

        stage = QVBoxLayout()
        views = QSplitter(Qt.Orientation.Horizontal)
        views.setChildrenCollapsible(False)
        views.setHandleWidth(8)

        cam_box = QWidget()
        cam_col = QVBoxLayout(cam_box)
        cam_col.setContentsMargins(0, 0, 0, 0)
        cam_col.addWidget(self._heading("Camera"))
        self.preview = QLabel(self._empty_text())
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setWordWrap(True)
        self.preview.setMinimumSize(360, 260)
        self.preview.setStyleSheet(
            "QLabel { background: #090c12; border-radius: 8px; color: #6b7688; padding: 16px; }")
        cam_col.addWidget(self.preview, 1)
        views.addWidget(cam_box)

        scene_box = QWidget()
        scene_col = QVBoxLayout(scene_box)
        scene_col.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        head.addWidget(self._heading("3D scene"))
        head.addStretch()
        self.view_mode = QComboBox()
        self.view_mode.addItems(VIEW_MODES)
        self.view_mode.currentTextChanged.connect(lambda m: self.viewport.set_mode(m))
        head.addWidget(self.view_mode)
        reset = QPushButton("Reset view")
        reset.clicked.connect(lambda: self.viewport.reset_view())
        head.addWidget(reset)
        scene_col.addLayout(head)
        self.viewport = EditorViewport()
        self.viewport.pivot_z = PIVOT_Z
        self.viewport.show_effect_widgets = True
        self.viewport.camera_moved.connect(self._camera_edited)
        self.viewport.keyframe_added.connect(lambda _t: self._keys_changed())
        self.viewport.effect_moved.connect(self._effect_moved)
        scene_col.addWidget(self.viewport, 1)
        views.addWidget(scene_box)
        views.setSizes([600, 600])
        stage.addWidget(views, 1)

        transport = QHBoxLayout()
        self.play_button = QPushButton("▶  Play")
        self.play_button.clicked.connect(self._toggle_play)
        add_key = QPushButton("◆ Add key")
        add_key.setToolTip("Keyframe the camera at the playhead. Then move it with the "
                           "gizmo in the 3D view (G move, R rotate).")
        add_key.clicked.connect(self._add_key)
        del_key = QPushButton("Delete key")
        del_key.clicked.connect(self._delete_key)
        self.easing = QComboBox()
        self.easing.addItems([e.label for e in Easing])
        self.easing.setToolTip("Easing out of the selected keyframe")
        self.easing.currentTextChanged.connect(self._easing_changed)
        self.preset = QComboBox()
        self.preset.addItems(PRESETS)
        apply_preset = QPushButton("Use preset")
        apply_preset.setToolTip("Replace the camera keys with this preset, as ordinary "
                                "editable keys.")
        apply_preset.clicked.connect(self._apply_preset)
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1.0, 60.0)
        self.duration_spin.setValue(self.duration)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.valueChanged.connect(self._duration_changed)
        self.fps_box = QComboBox()
        self.fps_box.addItems(["24", "30", "60"])
        self.fps_box.setCurrentText(str(self.fps))
        self.fps_box.currentTextChanged.connect(self._fps_changed)
        self.time_label = QLabel()
        for wdg in (self.play_button, add_key, del_key, QLabel("Easing"), self.easing,
                    self.preset, apply_preset, QLabel("Length"), self.duration_spin,
                    QLabel("fps"), self.fps_box):
            transport.addWidget(wdg)
        transport.addStretch()
        transport.addWidget(self.time_label)
        stage.addLayout(transport)
        body.addLayout(stage, 1)

        body.addWidget(self._build_rail())

        self.timeline = TimelineWidget()
        self.timeline.set_track(self.track)
        self.timeline.set_effects_track(self.effects_track)
        self.timeline.time_changed.connect(self._seek)
        self.timeline.key_selected.connect(self._key_selected)
        self.timeline.keys_changed.connect(self._keys_changed)
        outer.addWidget(self.timeline)

        status_row = QHBoxLayout()
        self.spinner = Spinner(14, self)
        self.spinner.hide()
        self.status = QLabel("")
        self.status.setStyleSheet("color: #8a93a6;")
        status_row.addWidget(self.spinner)
        status_row.addWidget(self.status, 1)
        outer.addLayout(status_row)
        self._set_enabled(False)

    def _heading(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setStyleSheet("color: #c9d1e0; font-weight: 600;")
        return lab

    def _card(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setStyleSheet("QFrame#card { background: #121824; border-radius: 8px; }")
        frame.setObjectName("card")
        col = QVBoxLayout(frame)
        col.setContentsMargins(12, 10, 12, 12)
        col.setSpacing(6)
        col.addWidget(self._heading(title))
        return frame, col

    def _build_rail(self) -> QWidget:
        inner = QWidget()
        col = QVBoxLayout(inner)
        # Right margin wider than the app's scrollbar: it overlays the content,
        # and at 6 px it sat on the combo arrows and spin buttons.
        col.setContentsMargins(0, 0, 16, 0)
        col.setSpacing(10)
        self._controls: list[QWidget] = []

        card, c = self._card("Scene")
        c.addWidget(QLabel("Scene quality"))
        self.source_box = QComboBox()
        self.source_box.addItems(list(SOURCES))
        self.source_box.setToolTip(
            "Standard: built from this app's depth map, no download.\n"
            "High quality: Apple's SHARP model predicts the 3D scene itself. Much "
            "cleaner around people and objects and more solid from other angles; a "
            "little softer in fine texture. One-time download, about 20 s per image.")
        self.source_box.currentTextChanged.connect(self._source_changed)
        c.addWidget(self.source_box)
        from . import sharp3d
        self.sharp_button = QPushButton(f"Download SHARP for High quality ({sharp3d.SIZE_LABEL})")
        self.sharp_button.setToolTip("Apple's SHARP model (research licence). Downloaded once.")
        self.sharp_button.clicked.connect(self._download_sharp)
        c.addWidget(self.sharp_button)
        c.addWidget(QLabel("Depth strength"))
        self.contrast = QSlider(Qt.Orientation.Horizontal)
        self.contrast.setRange(40, 250)
        self.contrast.setValue(100)
        self.contrast.setToolTip("How far apart near and far things sit. Raise it for "
                                 "flat-looking images, lower it if edges tear.")
        self.contrast.valueChanged.connect(lambda _v: self._rebuild.start())
        c.addWidget(self.contrast)
        self.bake_button = QPushButton("Fill background for this move")
        self.bake_button.setToolTip(
            "Walks the camera along its path, finds everything the move reveals behind "
            "objects and past the frame edges, and paints it in as part of the scene. "
            "Run it again after changing the camera move.")
        self.bake_button.clicked.connect(self._bake)
        c.addWidget(self.bake_button)
        self.lama_button = QPushButton("Download LaMa for better fills (207 MB)")
        self.lama_button.clicked.connect(self._download_lama)
        c.addWidget(self.lama_button)
        self.bake_note = QLabel("")
        self.bake_note.setWordWrap(True)
        self.bake_note.setStyleSheet("color: #8a93a6;")
        c.addWidget(self.bake_note)
        self._controls += [self.contrast, self.bake_button, self.source_box]
        col.addWidget(card)

        card, c = self._card("Atmosphere")
        row = QHBoxLayout()
        self.volume_type = QComboBox()
        self.volume_type.addItems(VOLUME_TYPES)
        add_vol = QPushButton("Add volume")
        add_vol.clicked.connect(self._add_volume)
        row.addWidget(self.volume_type, 1)
        row.addWidget(add_vol)
        c.addLayout(row)
        row = QHBoxLayout()
        self.particle_type = QComboBox()
        self.particle_type.addItems(PARTICLE_TYPES)
        add_em = QPushButton("Add particles")
        add_em.clicked.connect(self._add_emitter)
        row.addWidget(self.particle_type, 1)
        row.addWidget(add_em)
        c.addLayout(row)
        self.fx_list = QListWidget()
        self.fx_list.setMaximumHeight(110)
        self.fx_list.currentItemChanged.connect(self._fx_selected)
        c.addWidget(self.fx_list)
        self.fx_panel = QWidget()
        fp = QVBoxLayout(self.fx_panel)
        fp.setContentsMargins(0, 0, 0, 0)
        self.fx_enabled = QCheckBox("Enabled")
        self.fx_enabled.toggled.connect(lambda v: self._fx_set("enabled", v))
        fp.addWidget(self.fx_enabled)
        self.fx_amount = self._spin(fp, "Amount", 0.0, 5000.0, 0.05, "amount")
        self.fx_speed = self._spin(fp, "Speed", -3.0, 3.0, 0.02, "speed")
        self.fx_scale = self._spin(fp, "Size", 0.05, 20.0, 0.05, "scale")
        self.fx_glow = self._spin(fp, "Glow", 0.0, 10.0, 0.1, "emission")
        row = QHBoxLayout()
        colour = QPushButton("Colour…")
        colour.clicked.connect(self._fx_colour)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._fx_remove)
        key_fx = QPushButton("◆ Key FX")
        key_fx.setToolTip("Keyframe every effect's current settings at the playhead.")
        key_fx.clicked.connect(self._key_fx)
        row.addWidget(colour)
        row.addWidget(key_fx)
        row.addWidget(remove)
        fp.addLayout(row)
        hint = QLabel("Drag an effect's box in the 3D view to place it.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8a93a6;")
        fp.addWidget(hint)
        self.fx_panel.setEnabled(False)
        c.addWidget(self.fx_panel)
        self._controls += [add_vol, add_em]
        col.addWidget(card)

        card, c = self._card("Export")
        self.res_box = QComboBox()
        self.res_box.addItems(list(RESOLUTIONS))
        self.res_box.setCurrentIndex(2)
        self.codec_box = QComboBox()
        for cd in video.CODECS:
            self.codec_box.addItem(cd.label, cd.key)
            self.codec_box.setItemData(self.codec_box.count() - 1, cd.note,
                                       Qt.ItemDataRole.ToolTipRole)
        c.addWidget(QLabel("Resolution"))
        c.addWidget(self.res_box)
        c.addWidget(QLabel("Format"))
        c.addWidget(self.codec_box)
        self.export_button = QPushButton("Export video…")
        self.export_button.clicked.connect(self._export)
        c.addWidget(self.export_button)
        self._controls += [self.export_button]
        col.addWidget(card)
        col.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(340)
        return scroll

    def _spin(self, layout, label: str, lo: float, hi: float, step: float, field: str):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        sp.setDecimals(2)
        sp.valueChanged.connect(lambda v, f=field: self._fx_set(f, v))
        row.addStretch()
        row.addWidget(sp)
        layout.addLayout(row)
        return sp

    def _empty_text(self) -> str:
        return ("Convert an image on the Single image tab, then come back here to "
                "direct a camera through it in 3D.")

    def _set_enabled(self, on: bool) -> None:
        for wdg in getattr(self, "_controls", []):
            wdg.setEnabled(on)
        self.play_button.setEnabled(on)

    def _set_status(self, text: str) -> None:
        self.status.setText(text)
        self.spinner.setVisible("…" in text)

    # -- source & scene --------------------------------------------------------

    def set_source(self, image: object, depth: object) -> None:
        self._stop()
        if image is None or depth is None:
            self._rgb8 = self._depth = self._scene = self._base_scene = None
            self.preview.setPixmap(QPixmap())
            self.preview.setText(self._empty_text())
            self.viewport.set_renderer(None, PIVOT_Z)
            self._set_enabled(False)
            return
        self._rgb8 = _to_rgb8(image)
        d = np.asarray(depth, np.float32)
        self._depth = (d - d.min()) / (np.ptp(d) + 1e-6)
        self._baked = False
        self._start_build()

    def _start_build(self) -> None:
        if self._rgb8 is None:
            return
        self._gen += 1
        self._set_enabled(False)
        mode = SOURCES.get(self.source_box.currentText(), "standard")
        from . import sharp3d
        if mode == "sharp" and not sharp3d.is_downloaded():
            mode = "standard"
            self.source_box.blockSignals(True)
            self.source_box.setCurrentIndex(0)
            self.source_box.blockSignals(False)
        self._request_build.emit(self._rgb8, self._depth, self.contrast.value() / 100.0,
                                 self._gen, mode)

    def _ensure_renderer(self) -> bool:
        if self._renderer is None and not self._renderer_error:
            try:
                self._renderer = splat3d.SplatRenderer()
            except Exception as error:  # noqa: BLE001 - no usable GPU adapter
                self._renderer_error = str(error)
        if self._renderer is None:
            self.preview.setText("The 3D view needs a GPU with Vulkan or DirectX 12.\n\n"
                                 + self._renderer_error)
            return False
        return True

    def _on_scene(self, scene, gen: int) -> None:
        if gen != self._gen or not self._ensure_renderer():
            return
        self._scene = scene
        self._base_scene = scene
        self._baked = False
        self._renderer.set_scene(scene)
        self.viewport.set_renderer(_SceneView(self._renderer), PIVOT_Z)
        self.viewport.set_effects(self.effects)
        self._set_enabled(True)
        self._set_status("")
        self._update_bake_note()
        self._request_redraw()

    def _update_bake_note(self) -> None:
        from . import inpaint
        from . import sharp3d
        has_lama = inpaint.is_downloaded()
        self.lama_button.setVisible(not has_lama)
        self.sharp_button.setVisible(not sharp3d.is_downloaded())
        if self._baked:
            self.bake_note.setText("Background filled for the current move.")
        else:
            self.bake_note.setText("Black gaps appear where the camera sees past objects. "
                                   "Fill them once you are happy with the move.")

    def _bake(self) -> None:
        if self._scene is None:
            return
        self._stop()
        aspect = self._rgb8.shape[1] / self._rgb8.shape[0]
        size = fit_size(aspect, (BAKE_LONG, BAKE_LONG))
        track = CameraTrack(keys=[CameraKey(**{f: getattr(k, f) for f in (
            "time", "x", "y", "z", "yaw", "pitch", "roll", "fov_degrees", "easing")})
            for k in self.track.keys], smooth=self.track.smooth)
        duration = self.duration

        def poses(sz):
            # Every key plus evenly spaced samples: keys are where a move
            # usually peaks, samples catch what a curve sweeps past between.
            times = sorted({*(k.time for k in track.keys),
                            *(duration * i / 11 for i in range(12))})
            out = []
            for t in times:
                key = track.evaluate(t)
                if key is not None:
                    out.append(camera_matrices(key_to_camera(key), sz))
            return out

        self._set_enabled(False)
        self._request_bake.emit(self._copy_scene(self._base_scene), poses, size, self._gen)

    @staticmethod
    def _copy_scene(s: splat3d.SplatScene) -> splat3d.SplatScene:
        return splat3d.SplatScene(s.positions.copy(), s.colors.copy(), s.opacity.copy(),
                                  s.cov.copy(), s.n_front, s.photo_z, s.focal, s.planes)

    def _on_baked(self, scene, gen: int) -> None:
        if gen != self._gen:
            return
        self._scene = scene
        self._renderer.set_scene(scene)
        self._baked = True
        self._set_enabled(True)
        self._set_status("")
        self._update_bake_note()
        self.viewport.invalidate()
        self._request_redraw()

    def _source_changed(self, text: str) -> None:
        from . import sharp3d
        if SOURCES.get(text) == "sharp" and not sharp3d.is_downloaded():
            answer = QMessageBox.question(
                self, "Download SHARP",
                f"High quality uses Apple's SHARP model, a one-time {sharp3d.SIZE_LABEL} "
                "download (research licence). Download it now?")
            if answer == QMessageBox.StandardButton.Yes:
                self._download_sharp()
            else:
                self.source_box.blockSignals(True)
                self.source_box.setCurrentIndex(0)
                self.source_box.blockSignals(False)
            return
        self._baked = False
        self._start_build()

    def _download_sharp(self) -> None:
        self.sharp_button.setEnabled(False)
        self._request_sharp.emit()

    def _on_sharp(self, ok: bool, message: str) -> None:
        self.sharp_button.setEnabled(True)
        self._set_status("" if ok else f"SHARP download failed: {message}")
        self._update_bake_note()
        if ok:
            self.source_box.blockSignals(True)
            self.source_box.setCurrentText("High quality (SHARP)")
            self.source_box.blockSignals(False)
            self._start_build()

    def _download_lama(self) -> None:
        self.lama_button.setEnabled(False)
        self._request_lama.emit()

    def _on_lama(self, ok: bool, message: str) -> None:
        self.lama_button.setEnabled(True)
        self._set_status("" if ok else f"LaMa download failed: {message}")
        self._update_bake_note()

    def _on_failed(self, message: str) -> None:
        self._set_enabled(self._scene is not None)
        self._set_status(message)

    # -- camera & timeline ---------------------------------------------------

    def _reset_track(self) -> None:
        self.track = track_from_preset("Orbit", self.duration, PIVOT_Z, strength=0.6)
        self._track_changed()

    def _track_changed(self) -> None:
        self.timeline.set_track(self.track)
        self.timeline.set_duration(self.duration, self.fps)
        self.viewport.set_track(self.track, self.duration)
        self._request_redraw()

    def _apply_preset(self) -> None:
        self.track = track_from_preset(self.preset.currentText(), self.duration, PIVOT_Z,
                                       strength=0.6)
        self._track_changed()
        self._camera_move_changed()

    def _add_key(self) -> None:
        key = self.track.evaluate(self.time) or CameraKey(time=self.time)
        key.time = self.time
        self.track.add(key)
        self._keys_changed()

    def _delete_key(self) -> None:
        times = list(self.timeline.selection) or [self.time]
        removed = any([self.track.remove_at(t, tolerance=0.05) for t in times])
        if removed:
            self.timeline.selection.clear()
            self._keys_changed()

    def _key_selected(self, t: float) -> None:
        key = self.track.nearest(t, 1e-3)
        if key is not None:
            self.easing.blockSignals(True)
            self.easing.setCurrentText(key.easing.label)
            self.easing.blockSignals(False)

    def _easing_changed(self, label: str) -> None:
        for t in self.timeline.selection:
            key = self.track.nearest(t, 1e-3)
            if key is not None:
                key.easing = Easing(label)
        self._keys_changed()

    def _keys_changed(self) -> None:
        self.timeline.set_track(self.track)
        self.viewport.set_track(self.track, self.duration)
        self._camera_move_changed()
        self._request_redraw()

    def _camera_edited(self) -> None:
        self.timeline.update()
        self._camera_move_changed()
        self._request_redraw()

    def _camera_move_changed(self) -> None:
        if self._baked:
            self.bake_note.setText("The camera move changed. Fill the background again to "
                                   "cover anything new it reveals.")

    def _duration_changed(self, v: float) -> None:
        old = self.duration
        self.duration = float(v)
        # Stretch the keys with the clip, so a longer clip is the same move slower.
        for k in self.track.keys:
            k.time *= self.duration / old
        self.time = min(self.time, self.duration)
        self._track_changed()

    def _fps_changed(self, text: str) -> None:
        self.fps = int(text)
        self.timeline.set_duration(self.duration, self.fps)

    def _seek(self, t: float) -> None:
        self.time = float(t)
        self.viewport.set_time(self.time)
        self._request_redraw()

    def _toggle_play(self) -> None:
        if self._playing:
            self._stop()
        else:
            self._playing = True
            self.play_button.setText("❚❚  Pause")
            self._clock = QtClock()
            self._start_time = self.time
            self._play_timer.start()

    def _stop(self) -> None:
        self._playing = False
        self._play_timer.stop()
        self.play_button.setText("▶  Play")

    def _advance(self) -> None:
        t = (self._start_time + self._clock.elapsed()) % max(self.duration, 1e-3)
        self.timeline.set_time(t)      # emits time_changed -> _seek

    # -- effects -------------------------------------------------------------

    def _add_volume(self) -> None:
        item = volume_preset(self.volume_type.currentText(), PIVOT_Z)
        self.effects.volumes.append(item)
        self._fx_added(item)

    def _add_emitter(self) -> None:
        item = emitter_preset(self.particle_type.currentText(), PIVOT_Z)
        self.effects.emitters.append(item)
        self._fx_added(item)

    def _fx_added(self, item) -> None:
        li = QListWidgetItem(item.name)
        li.setData(Qt.ItemDataRole.UserRole, item.id)
        self.fx_list.addItem(li)
        self.fx_list.setCurrentItem(li)
        self._fx_changed()

    def _selected_fx(self):
        li = self.fx_list.currentItem()
        return None if li is None else self.effects.item(li.data(Qt.ItemDataRole.UserRole))

    def _fx_selected(self, *_args) -> None:
        item = self._selected_fx()
        self.fx_panel.setEnabled(item is not None)
        self.viewport.set_effect_selection(item.id if item is not None else None)
        if item is None:
            return
        is_volume = hasattr(item, "density")
        widgets = (self.fx_enabled, self.fx_amount, self.fx_speed, self.fx_scale, self.fx_glow)
        for wdg in widgets:
            wdg.blockSignals(True)
        self.fx_enabled.setChecked(item.enabled)
        self.fx_amount.setValue(item.density if is_volume else float(item.count))
        self.fx_amount.setSingleStep(0.05 if is_volume else 25)
        self.fx_speed.setValue(item.speed)
        self.fx_scale.setValue(float(np.mean(item.size)))
        self.fx_glow.setValue(item.emission)
        for wdg in widgets:
            wdg.blockSignals(False)

    def _fx_set(self, field: str, value) -> None:
        item = self._selected_fx()
        if item is None:
            return
        if field == "amount":
            if hasattr(item, "density"):
                item.density = float(value)
            else:
                item.count = int(value)
        elif field == "scale":
            cur = float(np.mean(item.size)) or 1.0
            item.size = tuple(float(s) * float(value) / cur for s in item.size)
        else:
            setattr(item, field, value)
        self._fx_changed()

    def _fx_colour(self) -> None:
        item = self._selected_fx()
        if item is None:
            return
        r, g, b = (int(c * 255) for c in item.colour)
        chosen = QColorDialog.getColor(QColor(r, g, b), self, "Effect colour")
        if chosen.isValid():
            item.colour = (chosen.redF(), chosen.greenF(), chosen.blueF())
            self._fx_changed()

    def _fx_remove(self) -> None:
        item = self._selected_fx()
        if item is None:
            return
        self.effects.volumes = [v for v in self.effects.volumes if v.id != item.id]
        self.effects.emitters = [e for e in self.effects.emitters if e.id != item.id]
        self.fx_list.takeItem(self.fx_list.currentRow())
        self._fx_changed()

    def _key_fx(self) -> None:
        self.effects_track.add(self.time, self.effects.clone())
        self.timeline.set_effects_track(self.effects_track)

    def _effect_moved(self, item_id: str) -> None:
        for i in range(self.fx_list.count()):
            if self.fx_list.item(i).data(Qt.ItemDataRole.UserRole) == item_id:
                self.fx_list.setCurrentRow(i)
        self._request_redraw()

    def _fx_changed(self) -> None:
        self.viewport.set_effects(self.effects)
        self._request_redraw()

    def _effects_at(self, t: float) -> EffectsState:
        if not self.effects_track.keys:
            return self.effects
        return self.effects_track.evaluate(t, self.effects)

    # -- drawing ---------------------------------------------------------------

    def _request_redraw(self) -> None:
        if not self._redraw.isActive():
            self._redraw.start(0)

    def _render_preview(self) -> None:
        self.time_label.setText(f"{self.time:.2f}s / {self.duration:.2f}s")
        if self._scene is None or self._renderer is None:
            return
        key = self.track.evaluate(self.time)
        if key is None:
            return
        ratio = max(1.0, float(self.preview.devicePixelRatioF()))
        box = (max(64, int(self.preview.width() * ratio)), max(64, int(self.preview.height() * ratio)))
        aspect = self._rgb8.shape[1] / self._rgb8.shape[0]
        # Fit the image aspect inside the label; the render is the display size.
        w = box[0]
        h = int(w / aspect)
        if h > box[1]:
            h = box[1]
            w = int(h * aspect)
        w, h = max(2, w - w % 2), max(2, h - h % 2)
        view, proj = camera_matrices(key_to_camera(key), (w, h))
        frame = self._renderer.render_u8(view, proj, (w, h), BACKGROUND,
                                         self._effects_at(self.time), self.time)
        img = QImage(frame.data, w, h, frame.strides[0], QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(img)
        pix.setDevicePixelRatio(ratio)
        self.preview.setPixmap(pix)
        self.viewport.invalidate()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._request_redraw()

    # -- export ----------------------------------------------------------------

    def _export(self) -> None:
        if self._scene is None:
            return
        codec = video.CODECS_BY_KEY[self.codec_box.currentData()]
        if not video.is_available():
            QMessageBox.information(
                self, "Video support needed",
                "Exporting needs the video add-on. Open the Video tab once to set it up, "
                "then export here.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export video", f"scene{codec.suffix}", f"{codec.label} (*{codec.suffix})")
        if not path:
            return
        aspect = self._rgb8.shape[1] / self._rgb8.shape[0]
        job = {
            "path": path, "codec": codec.key, "fps": self.fps, "duration": self.duration,
            "size": fit_size(aspect, RESOLUTIONS[self.res_box.currentText()]),
            "track": CameraTrack(keys=[CameraKey(**{f: getattr(k, f) for f in (
                "time", "x", "y", "z", "yaw", "pitch", "roll", "fov_degrees", "easing")})
                for k in self.track.keys], smooth=self.track.smooth),
            "effects": self.effects.clone(),
            "effects_track": EffectsTrack(keys=list(self.effects_track.keys)),
        }
        self._stop()
        self._set_enabled(False)
        self._request_export.emit(self._copy_scene(self._scene), job)

    def _on_exported(self, path: str) -> None:
        self._set_enabled(True)
        self._set_status(f"Saved {Path(path).name}")

    def shutdown(self) -> None:
        self._stop()
        self._worker.cancel = True
        self._thread.quit()
        self._thread.wait(3000)


class QtClock:
    """Seconds since construction, for wall-clock playback."""

    def __init__(self) -> None:
        from PySide6.QtCore import QElapsedTimer
        self._t = QElapsedTimer()
        self._t.start()

    def elapsed(self) -> float:
        return self._t.elapsed() / 1000.0
