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
from .animation3d import (KEY_EASES, CameraKey, CameraTrack, Easing, key_ease_label,
                          key_to_camera, set_key_ease, track_from_preset)
from .effects3d import (EffectsState, EffectsTrack, emitter_preset, plane_preset,
                        volume_preset)
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
#: Resolution classes: (width, height) of the 16:9 frame in each class. The
#: aspect ratio then sets the shape within the class (see output_size).
RESOLUTIONS = {
    "4K / UHD": (3840, 2160),
    "1440p / QHD": (2560, 1440),
    "1080p / Full HD": (1920, 1080),
    "720p / HD": (1280, 720),
}
#: Output aspect ratios, width / height. None follows the source image.
ASPECTS = {
    "Source image": None,
    "16:9 HD": 16 / 9,
    "9:16 Vertical": 9 / 16,
    "4:3": 4 / 3,
    "3:4 Vertical": 3 / 4,
    "1:1 Square": 1.0,
    "4:5 Vertical (social)": 4 / 5,
    "1.85:1 Flat (cinema)": 1.85,
    "2.39:1 CinemaScope": 2.39,
    "2.76:1 Ultra Panavision": 2.76,
}


def output_size(aspect: float, klass: tuple[int, int]) -> tuple[int, int]:
    """Standard video frame for an aspect within a resolution class.

    Mastering convention: ratios wider than 16:9 keep the class width and
    letterbox the height (1080p scope is 1920x804, flat 1920x1038); narrower
    landscape keeps the class height (4:3 is 1440x1080); vertical and square
    frames use the class height as their width (9:16 is 1080x1920, 4:5 is
    1080x1350). Even numbers throughout, as encoders require."""
    cw, ch = klass
    if aspect >= cw / ch:
        w, h = cw, cw / aspect
    elif aspect >= 1.0:
        w, h = ch * aspect, ch
    else:
        w, h = ch, ch / aspect
    # Nearest EVEN size (1920 / 2.39 = 803.3 -> 804, the standard scope frame).
    return max(2, 2 * int(round(w / 2))), max(2, 2 * int(round(h / 2)))


def lens_mm(vfov_degrees: float) -> float:
    """Full-frame-equivalent focal length for a vertical field of view."""
    return 12.0 / math.tan(math.radians(vfov_degrees) * 0.5)


def lens_fov(mm: float) -> float:
    return math.degrees(2.0 * math.atan(12.0 / max(mm, 1e-3)))


def shot_matrices(key, size, source_aspect: float) -> tuple[np.ndarray, np.ndarray]:
    """View and projection for the camera at `key`, framed to `size`.

    The aspect ratio crops the camera's frame, it never changes the camera:
    the key's field of view describes the frame at the SOURCE image's shape,
    and a wider output trims top and bottom (a taller one, the sides) instead
    of reaching past the photo's edges."""
    cam = key_to_camera(key)
    out_aspect = size[0] / max(size[1], 1)
    if out_aspect > source_aspect:
        half = math.tan(math.radians(cam.fov_degrees) * 0.5) * source_aspect / out_aspect
        cam.fov_degrees = math.degrees(2.0 * math.atan(half))
    return camera_matrices(cam, size)
VOLUME_TYPES = ["Fog", "Smoke", "Fire", "Cloud", "Godrays"]
PARTICLE_TYPES = ["Embers", "Dust", "Snow", "Rain", "Smoke", "Fire", "Clouds"]
BACKGROUND = (0.02, 0.021, 0.026)


#: Named particle directions as (tilt, heading) in degrees. Tilt 0 is scene up;
#: heading 0 is away from the camera, 90 to its right.
DIRECTIONS = {
    "Up": (0.0, 0.0),
    "Down": (180.0, 0.0),
    "Left": (90.0, 270.0),
    "Right": (90.0, 90.0),
    "Toward camera": (90.0, 180.0),
    "Away from camera": (90.0, 0.0),
}


def angles_to_direction(tilt: float, heading: float) -> tuple[float, float, float]:
    t, hd = math.radians(tilt), math.radians(heading)
    return (math.sin(t) * math.sin(hd), math.cos(t), -math.sin(t) * math.cos(hd))


def direction_to_angles(direction) -> tuple[float, float]:
    x, y, z = (float(v) for v in direction)
    length = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / length, y / length, z / length
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, y))))
    heading = math.degrees(math.atan2(x, -z)) % 360.0 if abs(y) < 0.9999 else 0.0
    return tilt, heading


def direction_name(direction) -> str:
    tilt, heading = direction_to_angles(direction)
    for name, (t, hd) in DIRECTIONS.items():
        dh = abs((heading - hd + 180.0) % 360.0 - 180.0)
        if abs(tilt - t) < 0.5 and (dh < 0.5 or t in (0.0, 180.0)):
            return name
    return "Custom"


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

    def _ensure_video_support(self) -> None:
        """Fetch the video add-on (PyAV) here if it is missing.

        It is normally installed the first time the Video tab is used, but a
        new user can go straight from converting an image to the 3D tab; asking
        them to visit another tab first, just to download a component, was a
        dead end. Same installer the Video tab uses."""
        if video.is_available():
            return
        from . import bootstrap

        def on_bytes(done, total):
            if total:
                self.progress.emit(f"Downloading video support… {100 * done // total}%")

        bootstrap.install_av(on_bytes=on_bytes, on_text=self.progress.emit)
        bootstrap.activate_av()
        if not video.is_available():
            raise RuntimeError("Video support was downloaded but could not be loaded. "
                               "Restart the app and try the export again.")

    def export(self, scene, job: dict) -> None:
        try:
            self.cancel = False
            self._ensure_video_support()
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
                    view, proj = shot_matrices(key, (w, h), job["source_aspect"])
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
        cam_head = QHBoxLayout()
        cam_head.addWidget(self._heading("Camera"))
        cam_head.addStretch()
        cam_head.addWidget(QLabel("Aspect"))
        self.aspect_box = QComboBox()
        self.aspect_box.addItems(list(ASPECTS))
        self.aspect_box.setToolTip(
            "Shape of the video. The camera view shows exactly this frame; a wider "
            "ratio trims top and bottom, a taller one the sides. Export sizes follow "
            "video standards for the chosen resolution.")
        self.aspect_box.currentTextChanged.connect(lambda _t: self._request_redraw())
        cam_head.addWidget(self.aspect_box)
        cam_head.addWidget(QLabel("Lens"))
        self.lens_spin = QDoubleSpinBox()
        self.lens_spin.setRange(8.0, 300.0)
        self.lens_spin.setDecimals(1)
        self.lens_spin.setSingleStep(1.0)
        self.lens_spin.setSuffix(" mm")
        self.lens_spin.setToolTip(
            "Focal length (full-frame equivalent). Changing it keys the lens at the "
            "playhead, so two keys with different lenses make an animated zoom.")
        self.lens_spin.valueChanged.connect(self._lens_changed)
        cam_head.addWidget(self.lens_spin)
        cam_col.addLayout(cam_head)
        self.frame_label = QLabel("")
        self.frame_label.setStyleSheet("color: #8a93a6;")
        self.preview = QLabel(self._empty_text())
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setWordWrap(True)
        self.preview.setMinimumSize(360, 260)
        self.preview.setStyleSheet(
            "QLabel { background: #090c12; border-radius: 8px; color: #6b7688; padding: 16px; }")
        cam_col.addWidget(self.preview, 1)
        cam_col.addWidget(self.frame_label)
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
        self.easing.addItems(list(KEY_EASES))
        self.easing.setToolTip(
            "Easing of the selected keyframe, as in After Effects: Ease In slows the "
            "arrival into it, Ease Out the departure, Easy Ease both, Hold freezes "
            "until the next key. The key's shape on the timeline shows it.")
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
        # Decided now, not only once a scene has been built: otherwise a fresh
        # start showed "Download SHARP" (and LaMa below) while the first scene
        # was still building, even with the model already installed.
        self.sharp_button.setVisible(not sharp3d.is_downloaded())
        c.addWidget(self.sharp_button)
        c.addWidget(QLabel("Depth strength"))
        self.contrast = QSlider(Qt.Orientation.Horizontal)
        self.contrast.setRange(40, 250)
        self.contrast.setValue(100)
        self.contrast.setToolTip("How far apart near and far things sit. Raise it for "
                                 "flat-looking images, lower it if edges tear.")
        # Rebuilding the scene takes a moment, so it waits until the slider is
        # let go: with tracking off, valueChanged fires on release (and on a
        # click or key press), not continuously while dragging.
        self.contrast.setTracking(False)
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
        from . import inpaint
        self.lama_button.setVisible(not inpaint.is_downloaded())
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
        row = QHBoxLayout()
        self.plane_type = QComboBox()
        self.plane_type.addItems(["Floor", "Ceiling", "Wall"])
        add_plane = QPushButton("Add plane")
        add_plane.setToolTip(
            "A surface particles collide with, placed with the gizmo. A floor starts "
            "on the detected ground and shows which way gravity pulls; tilt it to "
            "tilt 'up' for every particle effect. Adding any plane replaces the "
            "automatically detected ones.")
        add_plane.clicked.connect(self._add_plane)
        row.addWidget(self.plane_type, 1)
        row.addWidget(add_plane)
        c.addLayout(row)
        self._controls.append(add_plane)
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
        # Particles only: the size of each individual dot (Size above is the
        # emitter's box) and how solid they are.
        # Particles only, in one box that is hidden for volumes.
        self.particle_box = QWidget()
        pb = QVBoxLayout(self.particle_box)
        pb.setContentsMargins(0, 0, 0, 0)
        self.fx_psize = self._spin(pb, "Particle size", 0.002, 1.0, 0.005, "particle_size")
        self.fx_psize.setDecimals(3)
        self.fx_opacity = self._spin(pb, "Opacity", 0.0, 1.0, 0.05, "opacity")
        self.fx_turb = self._spin(pb, "Turbulence", 0.0, 4.0, 0.05, "turbulence")
        row = QHBoxLayout()
        row.addWidget(QLabel("Direction"))
        self.fx_dir = QComboBox()
        self.fx_dir.addItems(list(DIRECTIONS) + ["Custom"])
        self.fx_dir.setToolTip("Which way the particles travel. The photo decides what "
                               "'up' is, so pick it here, or use Custom with Tilt/Heading.")
        self.fx_dir.currentTextChanged.connect(self._fx_dir_preset)
        row.addStretch()
        row.addWidget(self.fx_dir)
        pb.addLayout(row)
        self.fx_tilt = self._spin(pb, "Tilt", 0.0, 180.0, 5.0, "tilt")
        self.fx_tilt.setSuffix("°")
        self.fx_tilt.setToolTip("0° travels up, 90° sideways, 180° down.")
        self.fx_heading = self._spin(pb, "Heading", 0.0, 360.0, 5.0, "heading")
        self.fx_heading.setSuffix("°")
        self.fx_heading.setToolTip("Sideways direction: 0° away from the camera, 90° right, "
                                   "180° toward the camera, 270° left.")
        self.fx_collide = QCheckBox("Collide with floor, walls and ceiling")
        self.fx_collide.setToolTip("Stop at the surfaces the scene analysis found.")
        self.fx_collide.toggled.connect(lambda v: self._fx_set("collide", v))
        pb.addWidget(self.fx_collide)
        self.fx_bounce = self._spin(pb, "Bounce", 0.0, 1.0, 0.05, "bounce")
        self.fx_bounce.setToolTip("0 slides along a surface (smoke pooling on the floor), "
                                  "1 bounces straight back.")
        fp.addWidget(self.particle_box)
        self.fx_glow = self._spin(fp, "Glow", 0.0, 10.0, 0.1, "emission")
        row = QHBoxLayout()
        colour = QPushButton("Colour…")
        colour.clicked.connect(self._fx_colour)
        self.fx_colour_button = colour
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
        self.res_box.currentTextChanged.connect(lambda _t: self._request_redraw())
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
        text = QLabel(label)
        row.addWidget(text)
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        sp.setDecimals(2)
        sp._label = text
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
        self._sync_lens()
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
            self.easing.setCurrentText(key_ease_label(self.track, key))
            self.easing.blockSignals(False)

    def _easing_changed(self, label: str) -> None:
        for t in self.timeline.selection:
            key = self.track.nearest(t, 1e-3)
            if key is not None:
                set_key_ease(self.track, key, label)
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
        self._sync_lens()
        self._request_redraw()

    def _sync_lens(self) -> None:
        key = self.track.evaluate(self.time)
        if key is None:
            return
        self.lens_spin.blockSignals(True)
        self.lens_spin.setValue(lens_mm(key.fov_degrees))
        self.lens_spin.blockSignals(False)

    def _lens_changed(self, mm: float) -> None:
        # Auto-key: write the lens into the key at the playhead, making one
        # from the current pose if there is none, as editing any keyed value
        # does in After Effects.
        key = self.track.nearest(self.time, tolerance=0.5 / max(self.fps, 1))
        if key is None:
            key = self.track.evaluate(self.time) or CameraKey(time=self.time)
            key.time = self.time
            key = self.track.add(key)
        key.fov_degrees = lens_fov(mm)
        self._keys_changed()

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

    def _add_plane(self) -> None:
        kind = self.plane_type.currentText()
        ground = None
        for n, c in (self._scene.planes if self._scene is not None and self._scene.planes else []):
            if abs(float(n[1])) > 0.85:
                ground = (n, c)
                break
        item = plane_preset(kind, PIVOT_Z, ground)
        self.effects.planes.append(item)
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
        is_plane = getattr(item, "kind", "") in ("floor", "ceiling", "wall")
        # A plane has only a place, a tilt and an on/off switch.
        for sp in (self.fx_amount, self.fx_speed, self.fx_glow):
            sp.setVisible(not is_plane)
            sp._label.setVisible(not is_plane)
        self.fx_colour_button.setVisible(not is_plane)
        if is_plane:
            self.particle_box.setVisible(False)
            self.fx_enabled.blockSignals(True)
            self.fx_enabled.setChecked(item.enabled)
            self.fx_enabled.blockSignals(False)
            self.fx_scale.blockSignals(True)
            self.fx_scale.setValue(float(np.mean([item.size[0], item.size[2]])))
            self.fx_scale.blockSignals(False)
            return
        is_volume = hasattr(item, "density")
        widgets = (self.fx_enabled, self.fx_amount, self.fx_speed, self.fx_scale, self.fx_glow,
                   self.fx_psize, self.fx_opacity, self.fx_turb, self.fx_dir, self.fx_tilt,
                   self.fx_heading, self.fx_collide, self.fx_bounce)
        for wdg in widgets:
            wdg.blockSignals(True)
        self.fx_enabled.setChecked(item.enabled)
        self.fx_amount.setValue(item.density if is_volume else float(item.count))
        self.fx_amount.setSingleStep(0.05 if is_volume else 25)
        self.fx_speed.setValue(item.speed)
        self.fx_scale.setValue(float(np.mean(item.size)))
        self.fx_glow.setValue(item.emission)
        if not is_volume:
            self.fx_psize.setValue(item.particle_size)
            self.fx_opacity.setValue(getattr(item, "opacity", 1.0))
            self.fx_turb.setValue(item.turbulence)
            tilt, heading = direction_to_angles(item.direction)
            self.fx_tilt.setValue(tilt)
            self.fx_heading.setValue(heading)
            self.fx_dir.setCurrentText(direction_name(item.direction))
            self.fx_collide.setChecked(bool(getattr(item, "collide", True)))
            self.fx_bounce.setValue(float(getattr(item, "bounce", 0.0)))
        self.particle_box.setVisible(not is_volume)
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
        elif field in ("tilt", "heading"):
            item.direction = angles_to_direction(self.fx_tilt.value(), self.fx_heading.value())
            self.fx_dir.blockSignals(True)
            self.fx_dir.setCurrentText(direction_name(item.direction))
            self.fx_dir.blockSignals(False)
        elif field == "scale" and getattr(item, "kind", "") in ("floor", "ceiling", "wall"):
            cur = float(np.mean([item.size[0], item.size[2]])) or 1.0
            k = float(value) / cur
            item.size = (item.size[0] * k, item.size[1], item.size[2] * k)
        elif field == "scale":
            cur = float(np.mean(item.size)) or 1.0
            item.size = tuple(float(s) * float(value) / cur for s in item.size)
        else:
            setattr(item, field, value)
        self._fx_changed()

    def _fx_dir_preset(self, name: str) -> None:
        if name not in DIRECTIONS:
            return
        tilt, heading = DIRECTIONS[name]
        for sp, v in ((self.fx_tilt, tilt), (self.fx_heading, heading)):
            sp.blockSignals(True)
            sp.setValue(v)
            sp.blockSignals(False)
        self._fx_set("tilt", tilt)

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
        self.effects.planes = [p for p in self.effects.planes if p.id != item.id]
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
        src_aspect = self._rgb8.shape[1] / self._rgb8.shape[0]
        aspect = self._output_aspect()
        out_w, out_h = output_size(aspect, RESOLUTIONS[self.res_box.currentText()])
        self.frame_label.setText(f"{self.aspect_box.currentText()}  ·  export {out_w} x {out_h}")
        # Fit the output frame inside the label; the render is the display size.
        w = box[0]
        h = int(w / aspect)
        if h > box[1]:
            h = box[1]
            w = int(h * aspect)
        w, h = max(2, w - w % 2), max(2, h - h % 2)
        view, proj = shot_matrices(key, (w, h), src_aspect)
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
        path, _ = QFileDialog.getSaveFileName(
            self, "Export video", f"scene{codec.suffix}", f"{codec.label} (*{codec.suffix})")
        if not path:
            return
        job = {
            "path": path, "codec": codec.key, "fps": self.fps, "duration": self.duration,
            "size": output_size(self._output_aspect(), RESOLUTIONS[self.res_box.currentText()]),
            "source_aspect": self._rgb8.shape[1] / self._rgb8.shape[0],
            "track": CameraTrack(keys=[CameraKey(**{f: getattr(k, f) for f in (
                "time", "x", "y", "z", "yaw", "pitch", "roll", "fov_degrees", "easing")})
                for k in self.track.keys], smooth=self.track.smooth),
            "effects": self.effects.clone(),
            "effects_track": EffectsTrack(keys=list(self.effects_track.keys)),
        }
        self._stop()
        self._set_enabled(False)
        self._request_export.emit(self._copy_scene(self._scene), job)

    def _output_aspect(self) -> float:
        chosen = ASPECTS.get(self.aspect_box.currentText())
        if chosen is None and self._rgb8 is not None:
            return self._rgb8.shape[1] / self._rgb8.shape[0]
        return chosen or 16 / 9

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
