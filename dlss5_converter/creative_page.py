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
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QSize, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QScrollArea, QSlider, QSplitter, QVBoxLayout, QWidget,
)

from . import splat3d, video
from .animation3d import (KEY_EASES, CameraKey, CameraTrack, Easing, key_ease_label,
                          key_to_camera, set_key_ease, track_from_preset)
from .effects3d import EffectsState, EffectsTrack
from .timeline3d import TimelineWidget
from .viewport3d import VIEW_MODES, EditorViewport
from .widgets import ModuleCard, SegmentedControl, Spinner
from .fx_panel import (DIRECTIONS, PARTICLE_TYPES, VOLUME_TYPES, FxPanel,  # noqa: F401
                       angles_to_direction, direction_name, direction_to_angles)

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
    shots_done = Signal(object, object, int)    # SplatScene | None, Report | None, job id
    shots_started = Signal(int)                 # job id
    saved_loaded = Signal(object, object, int)  # SplatScene, thumbnail RGB | None, token
    thumb_ready = Signal(str, object, int)      # path, small RGB image, token

    def __init__(self) -> None:
        super().__init__()
        self._renderer: splat3d.SplatRenderer | None = None
        self.cancel = False

    def _get_renderer(self) -> splat3d.SplatRenderer:
        if self._renderer is None:
            self._renderer = splat3d.SplatRenderer()
        return self._renderer

    def _renderer_for(self, scene):
        """Shot scenes (screenshots blended per pixel) have their own renderer."""
        from . import shotscene
        if isinstance(scene, shotscene.ShotScene):
            if getattr(self, "_ibr", None) is None:
                from . import ibr3d
                self._ibr = ibr3d.IBRRenderer()
            return self._ibr
        return self._get_renderer()

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

    def build_shots(self, paths, excluded, main, label: str, job: int) -> None:
        """Place, fuse and save a session of captures (Scene from shots).

        Jobs queue up on this thread, so several sessions can be sent at once;
        each is written to disk when done, so a finished scene is never lost.
        """
        from . import multishot
        self.shots_started.emit(job)
        try:
            self.cancel = False
            scene, report = multishot.build(paths, excluded, progress=self.progress.emit,
                                            cancelled=lambda: self.cancel, main=main)
            if scene is not None:
                self.progress.emit("Saving the scene…")
                report.saved = str(multishot.save(scene, report, label, progress=self.progress.emit))
            self.shots_done.emit(scene, report, job)
        except multishot.Cancelled:
            self.shots_done.emit(None, None, job)
        except Exception as error:  # noqa: BLE001 - surfaced in the UI
            self.failed.emit(f"Could not build the scene: {error}")
            self.shots_done.emit(None, None, job)

    def load_saved(self, folder: str, token: int) -> None:
        from . import multishot
        try:
            scene = multishot.load(folder)
            thumb = None
            raw = cv2.imread(str(Path(folder) / "thumb.jpg"), cv2.IMREAD_COLOR)
            if raw is not None:
                thumb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            self.saved_loaded.emit(scene, thumb, token)
        except Exception as error:  # noqa: BLE001
            self.failed.emit(f"Could not open that scene: {error}")

    def thumbnails(self, paths, token: int) -> None:
        """Small previews for the shot strip, decoded off the UI thread: a 4K
        PNG takes long enough that thirty of them froze the tab."""
        for path in paths:
            raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if raw is None:
                continue
            h, w = raw.shape[:2]
            small = cv2.resize(raw, (THUMB_W, max(1, round(h * THUMB_W / w))),
                               interpolation=cv2.INTER_AREA)
            self.thumb_ready.emit(str(path), cv2.cvtColor(small, cv2.COLOR_BGR2RGB), token)

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
            r = self._renderer_for(scene)
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


#: Shot strip thumbnail width, in pixels.
THUMB_W = 120

#: Badge colours for the shot strip, by status. Semantic, so fixed rather than
#: themed: green placed, amber fixable, red failed, grey left out.
STATUS_COLOURS = {
    "placed": "#5DCAA5", "no_depth": "#EF9F27", "no_overlap": "#F09595",
    "unreadable": "#F09595", "excluded": "#6b7688",
}


class CreativePage(QWidget):
    """3D tab widget. Inherits the processed image from the main window."""

    _request_build = Signal(object, object, float, int, str)
    _request_sharp = Signal()
    _request_bake = Signal(object, object, object, int)
    _request_export = Signal(object, object)
    _request_lama = Signal()
    _request_shots = Signal(object, object, object, str, int)
    _request_load = Signal(str, int)
    _request_thumbs = Signal(object, int)

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
        self._renderers: dict = {}
        self._renderer_error = ""
        self.track = CameraTrack()
        self.effects = EffectsState()
        self.effects_track = EffectsTrack()
        self.duration = 6.0
        self.fps = 30
        self.time = 0.0
        self._playing = False
        self._baked = False
        # Two sources, each with its own scene, so switching between them is
        # instant and a long multi-shot build is never thrown away by a look at
        # the single image.
        self._mode = "image"
        self._image_rgb8: np.ndarray | None = None
        self._image_depth: np.ndarray | None = None
        self._image_scene: splat3d.SplatScene | None = None
        self._image_base: splat3d.SplatScene | None = None
        self._shots_scene: splat3d.SplatScene | None = None
        self._shots_rgb8: np.ndarray | None = None
        self._shots_gen = 0
        self._shots_building = False
        self._shots_folder = str(Path.home())
        self._sessions: list = []
        self._excluded: set[str] = set()
        self._main_shot: str | None = None
        self._statuses: dict[str, str] = {}
        self._strip_items: dict[str, QListWidgetItem] = {}
        self._thumb_token = 0
        # Build queue: jobs in order, each {"id", "label", "status"}.
        self._jobs: list[dict] = []
        self._job_counter = 0
        self._running_job: int | None = None

        self._thread = QThread(self)
        self._worker = _Worker()
        self._worker.moveToThread(self._thread)
        self._request_build.connect(self._worker.build)
        self._request_bake.connect(self._worker.bake)
        self._request_export.connect(self._worker.export)
        self._request_lama.connect(self._worker.download_lama)
        self._request_sharp.connect(self._worker.download_sharp)
        self._request_shots.connect(self._worker.build_shots)
        self._request_thumbs.connect(self._worker.thumbnails)
        self._worker.shots_done.connect(self._on_shots_done)
        self._worker.shots_started.connect(self._on_shots_started)
        self._worker.saved_loaded.connect(self._on_saved_loaded)
        self._request_load.connect(self._worker.load_saved)
        self._worker.progress.connect(self._job_progress)
        self._worker.thumb_ready.connect(self._on_thumb)
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

        # Scene from shots: one thumbnail per capture, its badge saying whether
        # it made it into the scene and, if not, why. Click to leave one out.
        self.shot_strip = QListWidget()
        self.shot_strip.setViewMode(QListWidget.ViewMode.IconMode)
        self.shot_strip.setFlow(QListWidget.Flow.LeftToRight)
        self.shot_strip.setWrapping(False)
        self.shot_strip.setMovement(QListWidget.Movement.Static)
        self.shot_strip.setIconSize(QSize(THUMB_W, THUMB_W * 5 // 12))
        self.shot_strip.setSpacing(4)
        self.shot_strip.setFixedHeight(112)
        self.shot_strip.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.shot_strip.setToolTip("Click a shot to leave it out of the scene, or include it again. "
                                   "Right-click to make it the main shot.")
        self.shot_strip.itemClicked.connect(self._toggle_shot)
        self.shot_strip.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.shot_strip.customContextMenuRequested.connect(self._shot_menu)
        self.shot_strip.hide()
        stage.addWidget(self.shot_strip)

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
        # The app's titled card, so this rail reads like the rest of the app.
        # Tighter side padding than the main sidebar's: this rail is narrower,
        # and the card's own 16 px each side pushed slider values and their
        # keyframe diamonds past the right edge.
        card = ModuleCard(title)
        card.body.setContentsMargins(12, 12, 12, 13)
        card.body.setSpacing(8)
        return card, card.body

    def _build_rail(self) -> QWidget:
        inner = QWidget()
        col = QVBoxLayout(inner)
        # Right margin wider than the app's scrollbar: it overlays the content,
        # and at 6 px it sat on the combo arrows and spin buttons.
        col.setContentsMargins(0, 0, 16, 0)
        col.setSpacing(10)
        self._controls: list[QWidget] = []

        # Where the scene comes from. One image is the original 3D tab; Scene
        # from shots fuses a folder of game captures into one scene.
        self.mode_switch = SegmentedControl(["One image", "Scene from shots"])
        self.mode_switch.setToolTip(
            "One image: the picture from the Single image tab, in 3D.\n"
            "Scene from shots: several captures of one place, fused into a scene you "
            "can see from any side. Best with the DLSS5 Scene Capture add-on, which "
            "saves the game's real depth with every shot.")
        self.mode_switch.changed.connect(self._mode_changed)
        col.addWidget(self.mode_switch)

        card, c = self._card("Scene")
        self.scene_card = card
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

        card, c = self._card("Shots")
        self.shots_card = card
        self.folder_button = QPushButton("Choose folder…")
        # Secondary: Build scene is the one action this card leads to.
        self.folder_button.setObjectName("secondary")
        self.folder_button.setToolTip("A folder of captures. Shots taken in one sitting are "
                                      "grouped into a session; pick one below.")
        self.folder_button.clicked.connect(self._choose_folder)
        c.addWidget(self.folder_button)
        self.session_box = QComboBox()
        self.session_box.currentIndexChanged.connect(self._session_changed)
        self.session_box.hide()
        c.addWidget(self.session_box)
        self.shots_info = QLabel("Choose a folder of captures to begin.")
        self.shots_info.setWordWrap(True)
        self.shots_info.setStyleSheet("color: #8a93a6;")
        c.addWidget(self.shots_info)
        self.build_shots_button = QPushButton("Build scene")
        self.build_shots_button.setToolTip(
            "Finds where every shot was taken, lines them up, and fuses them into one "
            "scene. A few minutes for thirty shots; the rest of the app stays usable.")
        self.build_shots_button.clicked.connect(self._build_shots)
        self.build_shots_button.setEnabled(False)
        self.cancel_shots_button = QPushButton("Cancel")
        self.cancel_shots_button.setObjectName("secondary")
        self.cancel_shots_button.setToolTip("Stop the scene that is building now. Queued ones "
                                            "still run.")
        self.cancel_shots_button.clicked.connect(self._cancel_shots)
        self.cancel_shots_button.hide()
        build_row = QHBoxLayout()
        build_row.addWidget(self.build_shots_button, 1)
        build_row.addWidget(self.cancel_shots_button)
        c.addLayout(build_row)
        from . import sharp3d as _sharp
        self.shots_sharp_button = QPushButton(
            f"Download SHARP for best quality ({_sharp.SIZE_LABEL})")
        self.shots_sharp_button.setObjectName("secondary")
        self.shots_sharp_button.setToolTip(
            "Apple's SHARP rebuilds each key shot in full detail and fills what it hides. "
            "Without it, scenes are built from the game's depth alone and look much rougher.")
        self.shots_sharp_button.clicked.connect(self._download_sharp)
        self.shots_sharp_button.setVisible(not _sharp.is_downloaded())
        c.addWidget(self.shots_sharp_button)
        self.shots_result = QLabel("")
        self.shots_result.setWordWrap(True)
        c.addWidget(self.shots_result)
        self.shots_advice = QLabel("")
        self.shots_advice.setWordWrap(True)
        self.shots_advice.setStyleSheet("color: #8a93a6;")
        c.addWidget(self.shots_advice)
        # The capture add-on: without it there is no game depth, so no scene.
        addon_row = QHBoxLayout()
        self.install_addon_button = QPushButton("Install capture add-on in a game…")
        self.install_addon_button.setObjectName("secondary")
        self.install_addon_button.setToolTip(
            "Adds the DLSS5 Scene Capture add-on to a game that has ReShade, so F10 saves "
            "each screenshot with the game's depth. Pick the game's .exe.")
        self.install_addon_button.clicked.connect(self._install_addon)
        self.remove_addon_button = QPushButton("Remove…")
        self.remove_addon_button.setObjectName("secondary")
        self.remove_addon_button.setToolTip("Take the capture add-on out of a game again.")
        self.remove_addon_button.clicked.connect(self._remove_addon)
        addon_row.addWidget(self.install_addon_button, 1)
        addon_row.addWidget(self.remove_addon_button)
        c.addLayout(addon_row)
        c.addWidget(QLabel("Scenes"))
        self.scene_list = QListWidget()
        self.scene_list.setIconSize(QSize(80, 34))
        self.scene_list.setMinimumHeight(150)
        self.scene_list.setToolTip("Built scenes are saved automatically. Click one to open it.")
        self.scene_list.itemClicked.connect(self._scene_clicked)
        c.addWidget(self.scene_list)
        card.hide()
        col.addWidget(card)

        self.fx_panel = FxPanel(self)
        col.addWidget(self.fx_panel)

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
        # Wide enough for a slider row's label, value and keyframe diamond
        # without clipping; the effects panel needs more room than the old
        # plain-form rail did.
        scroll.setFixedWidth(392)
        return scroll

    def _empty_text(self) -> str:
        if self._mode == "shots":
            return ("Choose a folder of captures on the right, then Build scene to see "
                    "them fused into one place you can move through.")
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
        if image is None or depth is None:
            self._image_rgb8 = self._image_depth = None
            self._image_scene = self._image_base = None
            if self._mode == "image":
                self._stop()
                self._rgb8 = self._depth = None
                self._show_scene(None)
            return
        rgb8 = _to_rgb8(image)
        d = np.asarray(depth, np.float32)
        self._image_rgb8 = rgb8
        self._image_depth = (d - d.min()) / (np.ptp(d) + 1e-6)
        self._image_scene = self._image_base = None
        if self._mode != "image":
            return          # built when the user switches back to One image
        self._stop()
        self._rgb8, self._depth = self._image_rgb8, self._image_depth
        self._baked = False
        self._start_build()

    def _show_scene(self, scene, base=None) -> None:
        """Put a scene in the views, or the empty state when there is none."""
        self.viewport.empty_text = ("Build a scene from your shots first" if self._mode == "shots"
                                    else "Convert an image on the Single image tab first")
        if scene is None:
            self._scene = self._base_scene = None
            self.preview.setPixmap(QPixmap())
            self.preview.setText(self._empty_text())
            self.viewport.set_renderer(None, PIVOT_Z)
            self._set_enabled(False)
            return
        from . import shotscene
        if not self._ensure_renderer("ibr" if isinstance(scene, shotscene.ShotScene) else "splat"):
            return
        self._scene = scene
        self._base_scene = base if base is not None else scene
        self._renderer.set_scene(scene)
        self.viewport.main_zone = self._main_zone(scene)
        self.viewport.set_renderer(_SceneView(self._renderer), PIVOT_Z)
        self.viewport.set_effects(self.effects)
        self._set_enabled(True)
        self._request_redraw()

    def _main_zone(self, scene):
        """A scene from shots built around a main shot sits in that shot's
        camera (origin, looking down -z); show where it looks its best."""
        from . import multishot
        if self._mode != "shots" or not isinstance(scene, splat3d.SplatScene) or not scene.focal:
            return None
        rgb = self._shots_rgb8
        aspect = rgb.shape[1] / rgb.shape[0] if rgb is not None else 2.4
        height = multishot.WORK_WIDTH / aspect
        return {"vfov": math.degrees(2 * math.atan(height / 2 / scene.focal)), "aspect": aspect}

    def _mode_changed(self, index: int) -> None:
        self._stop()
        self._mode = "shots" if index == 1 else "image"
        shots = self._mode == "shots"
        self.scene_card.setVisible(not shots)
        self.shots_card.setVisible(shots)
        self.shot_strip.setVisible(shots)
        if shots:
            self._refresh_scenes()
            self._rgb8, self._depth = self._shots_rgb8, None
            self._show_scene(self._shots_scene)
            return
        self._rgb8, self._depth = self._image_rgb8, self._image_depth
        if self._image_scene is not None:
            self._baked = self._image_scene is not self._image_base
            self._show_scene(self._image_scene, self._image_base)
            self._update_bake_note()
        elif self._image_rgb8 is not None:
            self._show_scene(None)
            self._start_build()
        else:
            self._show_scene(None)

    # -- scene from shots --------------------------------------------------------

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Folder with your captures",
                                                  self._shots_folder)
        if folder:
            self.load_shots_folder(folder)

    def load_shots_folder(self, folder: str) -> None:
        from . import multishot
        self._shots_folder = folder
        try:
            self._sessions = multishot.find_sessions(folder)
        except OSError as error:
            self._sessions = []
            self.shots_info.setText(f"That folder could not be read: {error}")
        self.session_box.blockSignals(True)
        self.session_box.clear()
        for session in self._sessions:
            self.session_box.addItem(session.label)
        self.session_box.blockSignals(False)
        self.session_box.setVisible(len(self._sessions) > 1)
        if not self._sessions:
            self.shot_strip.clear()
            self._strip_items = {}
            self.build_shots_button.setEnabled(False)
            self.shots_info.setText("No images in that folder. Choose the folder your "
                                    "captures are saved to.")
            return
        self.session_box.setCurrentIndex(0)
        self._session_changed(0)

    def _current_session(self):
        index = self.session_box.currentIndex()
        return self._sessions[index] if 0 <= index < len(self._sessions) else None

    def _session_changed(self, _index: int) -> None:
        session = self._current_session()
        if session is None:
            return
        self._excluded = set()
        self._statuses = {}
        # The main shot: rebuilt in full, the others only fill around it.
        # The middle of the session until the user picks one.
        self._main_shot = str(session.paths[len(session.paths) // 2]) if session.paths else None
        self.shots_result.setText("")
        self.shots_advice.setText("")
        self.shot_strip.clear()
        self._strip_items = {}
        placeholder = QPixmap(THUMB_W, THUMB_W * 5 // 12)
        placeholder.fill(QColor("#101825"))
        for path in session.paths:
            item = QListWidgetItem(QIcon(placeholder), "")
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.shot_strip.addItem(item)
            self._strip_items[str(path)] = item
        self._thumb_token += 1
        self._request_thumbs.emit([str(p) for p in session.paths], self._thumb_token)
        self._refresh_strip()
        self._update_shots_info()

    def _update_shots_info(self) -> None:
        session = self._current_session()
        if session is None:
            return
        chosen = len(session.paths) - len(self._excluded)
        text = f"{session.game}: {chosen} shots"
        if self._excluded:
            text += f" ({len(self._excluded)} left out)"
        if session.with_depth:
            text += f", {session.with_depth} with game depth."
        else:
            text += (". No game depth, so it will be estimated and the scene will be "
                     "softer. The DLSS5 Scene Capture add-on saves real depth.")
        self.shots_info.setText(text)
        self.build_shots_button.setEnabled(chosen >= 2)

    def _on_thumb(self, path: str, rgb, token: int) -> None:
        item = self._strip_items.get(path)
        if token != self._thumb_token or item is None:
            return
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        item.setIcon(QIcon(QPixmap.fromImage(image)))

    def _refresh_strip(self) -> None:
        from . import multishot
        for path, item in self._strip_items.items():
            number = Path(path).stem.rsplit("_", 1)[-1]
            status = multishot.EXCLUDED if path in self._excluded else self._statuses.get(path)
            if status is None:
                main = path == self._main_shot
                item.setText(f"{number}  MAIN" if main else number)
                item.setToolTip(Path(path).name)
                item.setForeground(QColor("#EF9F27" if main else "#8a93a6"))
                continue
            label, why = multishot.STATUS_TEXT.get(status, (status, ""))
            if path == self._main_shot:
                label = f"MAIN · {label}"
            item.setText(f"{number}  {label}")
            item.setToolTip(f"{Path(path).name}\n{why}")
            item.setForeground(QColor(STATUS_COLOURS.get(status, "#8a93a6")))

    def _pick_game(self, title: str) -> str | None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, title, "", "Game (*.exe)")
        return path or None

    def _install_addon(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        from . import capture_install
        game = self._pick_game("Pick the game's .exe (the one ReShade is installed next to)")
        if game is None:
            return
        result = capture_install.install(game)
        text = result.message
        if result.installed:
            text += "\n\nFiles placed:\n" + "\n".join(str(p) for p in result.installed)
        (QMessageBox.information if result.ok else QMessageBox.warning)(
            self, "Capture add-on", text)

    def _remove_addon(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        from . import capture_install
        game = self._pick_game("Pick the game's .exe to remove the add-on from")
        if game is None:
            return
        result = capture_install.uninstall(game)
        (QMessageBox.information if result.ok else QMessageBox.warning)(
            self, "Capture add-on", result.message)

    def _shot_menu(self, pos) -> None:
        item = self.shot_strip.itemAt(pos)
        if item is None:
            return
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self.shot_strip)
        action = menu.addAction("Use as main shot")
        if menu.exec(self.shot_strip.mapToGlobal(pos)) is action:
            path = item.data(Qt.ItemDataRole.UserRole)
            self._main_shot = path
            self._excluded.discard(path)
            self._refresh_strip()
            self._update_shots_info()

    def _toggle_shot(self, item: QListWidgetItem) -> None:
        # Fine while a build runs: each queued job took its own copy of the
        # selection when it was queued.
        path = item.data(Qt.ItemDataRole.UserRole)
        if path in self._excluded:
            self._excluded.discard(path)
        elif path != self._main_shot:           # the main shot cannot be left out
            self._excluded.add(path)
        self._statuses.pop(path, None)
        self._refresh_strip()
        self._update_shots_info()

    def _build_shots(self) -> None:
        """Queue the current session. Builds run one at a time in the
        background, so the rest of the app, and more queuing, stay free."""
        session = self._current_session()
        if session is None:
            return
        self._job_counter += 1
        job = {"id": self._job_counter, "label": session.label, "status": "Queued",
               "paths": {str(p) for p in session.paths}}
        self._jobs.append(job)
        self.shots_result.setText("")
        self.shots_advice.setText("")
        self._request_shots.emit([str(p) for p in session.paths], set(self._excluded),
                                 self._main_shot, session.label, job["id"])
        self._refresh_scenes()

    def _cancel_shots(self) -> None:
        if self._running_job is not None:
            self._worker.cancel = True
            self.cancel_shots_button.setText("Cancelling…")

    def _job(self, job_id: int):
        return next((j for j in self._jobs if j["id"] == job_id), None)

    def _on_shots_started(self, job_id: int) -> None:
        self._running_job = job_id
        self._shots_building = True
        job = self._job(job_id)
        if job is not None:
            job["status"] = "Starting…"
        self.cancel_shots_button.setText("Cancel")
        self.cancel_shots_button.show()
        self._refresh_scenes()

    def _job_progress(self, text: str) -> None:
        job = self._job(self._running_job) if self._running_job is not None else None
        if job is not None and job["status"] != text:
            job["status"] = text
            row = job.get("row")
            if row is not None:
                row.setText(f"{job['label']}\n{text}")

    def _shots_idle(self) -> None:
        self._shots_building = False
        self._running_job = None
        self.cancel_shots_button.hide()
        self._update_shots_info()

    def _on_shots_done(self, scene, report, job_id: int) -> None:
        job = self._job(job_id)
        if job is not None:
            self._jobs.remove(job)
        self._shots_idle()
        self._refresh_scenes()
        if report is None:
            self._set_status("Build stopped." if job is not None else "")
            return
        session = self._current_session()
        if job is not None and session is not None and job["paths"] == {str(p) for p in session.paths}:
            self._statuses = dict(report.statuses)
            self._refresh_strip()
        summary = f"Placed {report.placed} of {report.considered} shots"
        if not report.estimated and report.agreement > 0:
            summary += f", {round(report.agreement * 100)}% agreement"
        if report.sharp:
            summary += f", {report.key_shots} rebuilt with SHARP"
        if report.fov_degrees > 0:
            summary += f". Field of view {report.fov_degrees:.0f}°, found from the shots."
        self.shots_result.setText(summary)
        self.shots_advice.setText(report.advice)
        self._set_status("Scene saved." if report.saved else "")
        if scene is None:
            return
        self._shots_scene = scene
        self._shots_rgb8 = report.root_image
        if self._mode == "shots":
            self._rgb8 = report.root_image
            self._show_scene(scene)

    def _refresh_scenes(self) -> None:
        """Queued and building jobs first, then every saved scene, newest first."""
        from . import multishot
        self.scene_list.clear()
        for job in self._jobs:
            item = QListWidgetItem(f"{job['label']}\n{job['status']}")
            item.setForeground(QColor("#EF9F27"))
            item.setData(Qt.ItemDataRole.UserRole, None)
            self.scene_list.addItem(item)
            job["row"] = item
        try:
            saved = multishot.list_saved()
        except OSError:
            saved = []
        for entry in saved:
            when = time.strftime("%d %b %H:%M", time.localtime(entry.created))
            quality = "SHARP" if entry.sharp else "depth only"
            item = QListWidgetItem(f"{entry.label.split(':')[0]}\n{when}, {entry.placed} of "
                                   f"{entry.shots} shots, {quality}")
            if entry.thumbnail.is_file():
                item.setIcon(QIcon(QPixmap(str(entry.thumbnail))))
            item.setData(Qt.ItemDataRole.UserRole, str(entry.folder))
            item.setToolTip(f"{entry.label}\n{entry.splats:,} splats\n{entry.folder}")
            self.scene_list.addItem(item)

    def _scene_clicked(self, item: QListWidgetItem) -> None:
        folder = item.data(Qt.ItemDataRole.UserRole)
        if not folder:
            return
        self._shots_gen += 1
        self._set_status("Opening scene…")
        self._request_load.emit(folder, self._shots_gen)

    def _on_saved_loaded(self, scene, thumb, token: int) -> None:
        if token != self._shots_gen:
            return
        self._set_status("")
        self._shots_scene = scene
        self._shots_rgb8 = thumb
        if self._mode == "shots":
            self._rgb8 = thumb
            self._show_scene(scene)

    def _start_build(self) -> None:
        if self._rgb8 is None or self._mode != "image":
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

    def _ensure_renderer(self, kind: str = "splat") -> bool:
        """Make `self._renderer` the renderer for this kind of scene: splats for
        a single image, per-pixel shot blending for a scene from shots."""
        if kind not in self._renderers and not self._renderer_error:
            try:
                if kind == "ibr":
                    from . import ibr3d
                    self._renderers[kind] = ibr3d.IBRRenderer()
                else:
                    self._renderers[kind] = splat3d.SplatRenderer()
            except Exception as error:  # noqa: BLE001 - no usable GPU adapter
                self._renderer_error = str(error)
        self._renderer = self._renderers.get(kind)
        if self._renderer is None:
            self.preview.setText("The 3D view needs a GPU with Vulkan or DirectX 12.\n\n"
                                 + self._renderer_error)
            return False
        return True

    def _on_scene(self, scene, gen: int) -> None:
        if gen != self._gen:
            return
        self._image_scene = self._image_base = scene
        if self._mode != "image" or not self._ensure_renderer():
            return
        self._baked = False
        self._show_scene(scene, scene)
        self._set_status("")
        self._update_bake_note()

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
        self._image_scene = scene
        if self._mode != "image":
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
        self.shots_sharp_button.setEnabled(False)
        self._request_sharp.emit()

    def _on_sharp(self, ok: bool, message: str) -> None:
        self.sharp_button.setEnabled(True)
        from . import sharp3d as _sharp
        self.shots_sharp_button.setEnabled(True)
        self.shots_sharp_button.setVisible(not _sharp.is_downloaded())
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
        if self._shots_building:
            self._shots_idle()
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
        if self.effects_track.keys:
            self.viewport.set_effects(self._effects_at(self.time))
        self.fx_panel.sync()
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

    # -- effects (the panel calls these) ----------------------------------------

    def pivot_z(self) -> float:
        return PIVOT_Z

    def ground_plane(self):
        """The detected floor as (normal, c), or None."""
        for n, c in (self._scene.planes if self._scene is not None and self._scene.planes else []):
            if abs(float(n[1])) > 0.85:
                return n, c
        return None

    def effects_at(self, t: float) -> EffectsState:
        return self._effects_at(t)

    def fx_changed(self) -> None:
        # The 3D view shows the effects as they are at the playhead, so an
        # animated effect's gizmo sits where the effect actually is.
        self.viewport.set_effects(self._effects_at(self.time))
        # Placed lightning strikes show as ticks on the FX row.
        self.timeline.set_markers([t for s in self.effects.strikes if s.enabled
                                   for t in s.strike_times])
        self._request_redraw()

    def keys_changed(self) -> None:
        self.timeline.set_effects_track(self.effects_track)
        self.fx_changed()

    def strikes_changed(self) -> None:
        self.fx_changed()

    def _effect_moved(self, item_id: str) -> None:
        self.fx_panel.moved(item_id, self.viewport.effects)
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
