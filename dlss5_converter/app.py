"""The desktop application."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from . import bootstrap, contract, evaluator, paths, pipeline, runtime
from .depth_engine import MODELS, DepthEngine
from .settings import (
    NR_COLOR_MAX,
    NR_PAPER_WHITE_MAX,
    NR_PRESETS,
    NR_STRENGTH_MAX,
    NR_STYLES,
    NR_TRANSFER_MAX,
    AppSettings,
)
from .widgets import (
    DownloadDialog,
    DropZone,
    ImageView,
    SliderRow,
    WipeView,
    first_supported,
)

STYLE = """
QMainWindow, QWidget { background: #16181d; color: #e6e8ec; }
QGroupBox { border: 1px solid #2a2e37; border-radius: 8px; margin-top: 14px; padding: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #9aa2b1; }
QLabel#hint { color: #7d8698; }
QFrame#dropZone {
    border: 2px dashed #39404e; border-radius: 12px; background: #1b1e25;
}
QFrame#dropZone[hovering="true"] { border-color: #5b8cff; background: #1f2532; }
QPushButton {
    background: #2b6cf6; border: none; border-radius: 6px; padding: 9px 16px; font-weight: 600;
}
QPushButton:disabled { background: #2a2e37; color: #6b7280; }
QPushButton#secondary { background: #262a33; }
QSlider::groove:horizontal { height: 4px; background: #2a2e37; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #5b8cff; width: 14px; margin: -6px 0; border-radius: 7px;
}
"""


class Worker(QObject):
    """Runs one conversion off the UI thread."""

    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        image_path: Path,
        settings: AppSettings,
        engine: DepthEngine,
        prepared: pipeline.Prepared | None = None,
    ) -> None:
        super().__init__()
        self._path = image_path
        self._settings = settings
        self._engine = engine
        self._prepared = prepared

    def run(self) -> None:
        try:
            result = pipeline.convert(
                self._path,
                self._settings,
                self._engine,
                progress=self.progress.emit,
                prepared=self._prepared,
            )
        except Exception as error:  # noqa: BLE001 - the UI is the error handler
            self.failed.emit(str(error))
            return
        self.finished.emit(result)


class DepthWorker(QObject):
    """Estimates depth off the UI thread, as soon as an image is opened.

    Run eagerly rather than at Convert time so the depth mask is on screen —
    and its contrast tunable — before committing to a DLSS run.
    """

    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, image_path: Path, settings: AppSettings, engine: DepthEngine) -> None:
        super().__init__()
        self._path = image_path
        self._settings = settings
        self._engine = engine

    def run(self) -> None:
        try:
            prepared = pipeline.prepare(
                self._path, self._settings, self._engine, progress=self.progress.emit
            )
        except Exception as error:  # noqa: BLE001 - the UI is the error handler
            self.failed.emit(str(error))
            return
        self.finished.emit(prepared)


class RuntimeWorker(QObject):
    """Downloads and unpacks PyTorch off the UI thread."""

    progress = Signal(str)
    # object, not int: Qt's int is 32-bit and these totals are not. The
    # unpacked torch tree is 2.87 GB, which overflows a signed 32-bit int
    # and made every progress emit raise OverflowError mid-extraction.
    bytes_progress = Signal(object, object)
    finished = Signal()
    failed = Signal(str)

    def run(self) -> None:
        try:
            bootstrap.install(
                on_bytes=lambda done, total: self.bytes_progress.emit(done, total),
                on_text=self.progress.emit,
            )
        except Exception as error:  # noqa: BLE001 - the UI is the error handler
            self.failed.emit(str(error))
            return
        self.finished.emit()


def ensure_runtime_ready(parent: QWidget | None = None) -> bool:
    """Fetch PyTorch if this build did not ship it. True if the app can run.

    Runs before the main window exists, because without torch there is no depth
    estimation and therefore nothing the window can usefully do. Blocking here
    with an explained progress bar is better than starting, looking healthy, and
    failing on the first image.
    """
    if bootstrap.is_ready():
        return True

    dialog = DownloadDialog(
        "First-run setup",
        f"PyTorch {bootstrap.TORCH_VERSION} is downloaded once, rather than "
        "shipped with the app - it is roughly 1.8 GB and would otherwise be in "
        "every copy. It is kept in the pytorch folder beside the app and "
        "survives updates.",
        parent,
    )
    dialog.setStyleSheet(STYLE)

    thread = QThread()
    worker = RuntimeWorker()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.progress.connect(dialog.set_heading)
    worker.bytes_progress.connect(dialog.update_bytes)

    state = {"ok": False, "error": ""}

    def done() -> None:
        state["ok"] = True
        dialog.mark_complete()
        dialog.accept()

    def failed(message: str) -> None:
        state["error"] = message
        dialog.reject()

    worker.finished.connect(done)
    worker.failed.connect(failed)
    thread.start()
    dialog.exec()
    thread.quit()
    thread.wait(5000)

    if not state["ok"]:
        QMessageBox.critical(
            parent,
            "Could not set up the runtime",
            (state["error"] or "The download did not finish.")
            + "\n\nThe app needs PyTorch for depth estimation. Check your "
            "connection and start it again - the download resumes from scratch "
            "but nothing is left half-installed.",
        )
    return state["ok"]


class DownloadWorker(QObject):
    """Fetches a depth model off the UI thread, reporting bytes as it goes."""

    progress = Signal(str)
    # object, not int: Qt's int is 32-bit and these totals are not. The
    # unpacked torch tree is 2.87 GB, which overflows a signed 32-bit int
    # and made every progress emit raise OverflowError mid-extraction.
    bytes_progress = Signal(object, object)
    finished = Signal()
    failed = Signal(str)

    def __init__(self, model_id: str, engine: DepthEngine) -> None:
        super().__init__()
        self._model_id = model_id
        self._engine = engine

    def run(self) -> None:
        try:
            self._engine.ensure_downloaded(
                self._model_id,
                progress=self.progress.emit,
                # Passing this is what switches ensure_downloaded into its
                # size-aware path; without it there are no byte counts to show.
                bytes_progress=lambda done, total: self.bytes_progress.emit(done, total),
            )
        except Exception as error:  # noqa: BLE001 - the UI is the error handler
            self.failed.emit(str(error))
            return
        self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DLSS 5 Image Converter")
        self.resize(1280, 820)

        self.settings = AppSettings.load(paths.settings_path())
        # One engine for the window's lifetime. Reloading Depth Anything per
        # image would add several seconds and a gigabyte of churn to every run.
        self.engine = DepthEngine()
        self.result: pipeline.Result | None = None
        self.image_path: Path | None = None
        self.prepared: pipeline.Prepared | None = None
        self._view = "photo"
        self._thread: QThread | None = None
        self._worker: Worker | None = None
        self._depth_thread: QThread | None = None
        self._depth_worker: DepthWorker | None = None
        self._previewing = False
        self._preview_pending = False
        self._download_thread: QThread | None = None
        self._download_worker: DownloadWorker | None = None
        self._download_dialog: DownloadDialog | None = None

        # Debounce. A slider drag emits a change per pixel and each one costs a
        # harness restart, so wait for the drag to settle before spending one.
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(600)
        self._preview_timer.timeout.connect(self._run_preview)

        # Dropping anywhere on the window, not just on the drop zone. The zone
        # is swapped out for the comparison view after a conversion, and when it
        # was the only drop target there was no way at all to open a second
        # image without restarting the app.
        self.setAcceptDrops(True)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        self.stack = QStackedWidget()
        self.drop = DropZone()
        self.drop.opened.connect(self.open_image)
        self.wipe = WipeView()
        self.depth_view = ImageView()
        self.stack.addWidget(self.drop)
        self.stack.addWidget(self.wipe)
        self.stack.addWidget(self.depth_view)

        canvas = QWidget()
        canvas_layout = QVBoxLayout(canvas)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(8)
        canvas_layout.addWidget(self.stack, 1)
        canvas_layout.addLayout(self._view_switch())

        layout.addWidget(canvas, 1)
        layout.addWidget(self._sidebar())

        self.setCentralWidget(central)
        self.setStyleSheet(STYLE)

        # Ctrl+V anywhere in the window. A screenshot is the most common way an
        # image reaches this app, and going via a file on disk to paste one is
        # the sort of step people give up on.
        QShortcut(QKeySequence.StandardKey.Paste, self, activated=self.paste)
        QShortcut(QKeySequence.StandardKey.Open, self, activated=self.browse)
        self.show_view("photo")
        # Status only — never a reason not to start. A frozen build has no
        # console, so an exception escaping here closes the window instantly
        # with nothing on screen and nothing in a log, which is the least
        # debuggable failure this app can have. Diagnose reports the details.
        try:
            message = runtime.describe(runtime.detect(self.settings.runtime_dir))
        except Exception as error:  # noqa: BLE001 - startup must survive anything
            message = f"Could not check the DLSS runtime: {error}"
        self.statusBar().showMessage(message)

        # After the event loop starts, so the window is painted behind the
        # dialog rather than the app appearing to hang on a bare download box.
        QTimer.singleShot(0, self.ensure_model_downloaded)

    # -- sidebar -------------------------------------------------------------

    # -- view switching ------------------------------------------------------

    def _view_switch(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.view_photo = QPushButton("Photo")
        self.view_depth = QPushButton("Depth mask")
        self.view_result = QPushButton("Result")
        for button in (self.view_photo, self.view_depth, self.view_result):
            button.setObjectName("secondary")
            button.setCheckable(True)
        self.view_photo.clicked.connect(lambda: self.show_view("photo"))
        self.view_depth.clicked.connect(lambda: self.show_view("depth"))
        self.view_result.clicked.connect(lambda: self.show_view("result"))
        row.addStretch(1)
        row.addWidget(self.view_photo)
        row.addWidget(self.view_depth)
        row.addWidget(self.view_result)
        row.addStretch(1)
        return row

    def show_view(self, which: str) -> None:
        """Switch the canvas, falling back when the requested view has no data."""
        if which == "result" and self.result is None:
            which = "depth" if self.prepared is not None else "photo"
        if which == "depth" and self.prepared is None:
            which = "photo"

        if which == "result":
            self.stack.setCurrentWidget(self.wipe)
        elif which == "depth":
            self.stack.setCurrentWidget(self.depth_view)
        elif self.prepared is not None:
            self.depth_view.set_image(self.prepared.source, "Source")
            self.stack.setCurrentWidget(self.depth_view)
            which = "photo"
        else:
            self.stack.setCurrentWidget(self.drop)
            which = "photo"

        self._view = which
        self.view_photo.setChecked(which == "photo")
        self.view_depth.setChecked(which == "depth")
        self.view_result.setChecked(which == "result")
        self.view_depth.setEnabled(self.prepared is not None)
        self.view_result.setEnabled(self.result is not None)
        self.view_photo.setEnabled(self.prepared is not None)
        if which == "depth":
            self._render_depth_preview()

    def _render_depth_preview(self) -> None:
        """Re-colour the depth mask for the current contrast.

        Cheap on purpose — a power function over one array — so the contrast
        slider can redraw live instead of re-running the depth model, which is
        the only slow part of the whole stage.
        """
        if self.prepared is None:
            return
        contrast = self.settings.depth.contrast
        shaped = contract.to_hardware_depth(self.prepared.inverse_depth, contrast)
        self.depth_view.set_image_u8(
            pipeline.depth_preview(shaped),
            f"Depth mask — contrast {contrast:.2f} (near = red)",
        )

    def _depth_settings_changed(self) -> None:
        """A change that invalidates the cached depth. Re-runs the model."""
        self.prepared = None
        self.result = None
        self.view_depth.setEnabled(False)
        self.view_result.setEnabled(False)
        if self.image_path is not None:
            self._start_depth()

    def _sidebar(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(320)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        neural = QGroupBox("Neural rendering")
        neural_layout = QVBoxLayout(neural)
        settings = self.settings.neural

        # Preset and style first: they choose *which* model runs, and the
        # sliders below only scale whatever that choice produces.
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset"))
        self.preset_box = QComboBox()
        self.preset_box.addItems(NR_PRESETS)
        self.preset_box.setCurrentIndex(min(len(NR_PRESETS) - 1, max(0, settings.preset)))
        self.preset_box.setToolTip(
            "Passed through to the add-on, which confirms receiving it — but all "
            "four presets measured bit-identical here with upscaling off, so it "
            "most likely selects a Super Resolution preset that a DLAA-only path "
            "never reaches. Style, below, does have a large effect."
        )
        self.preset_box.currentIndexChanged.connect(self._preset_changed)
        preset_row.addWidget(self.preset_box, 1)
        neural_layout.addLayout(preset_row)

        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Style"))
        self.style_box = QComboBox()
        self.style_box.addItems(NR_STYLES)
        self.style_box.setCurrentIndex(min(len(NR_STYLES) - 1, max(0, settings.style)))
        self.style_box.setToolTip("Broad look, applied on top of the preset.")
        self.style_box.currentIndexChanged.connect(self._style_changed)
        style_row.addWidget(self.style_box, 1)
        neural_layout.addLayout(style_row)

        neural_layout.addWidget(
            SliderRow(
                "Intensity",
                settings.intensity,
                self._neural_setter("intensity"),
                "Overall strength of the neural pass. At 0 this is a plain DLAA resolve. "
                "The add-on's own range is 0..2; it stops changing above 2.",
                maximum=NR_STRENGTH_MAX,
            )
        )
        neural_layout.addWidget(
            SliderRow(
                "Skin",
                settings.skin,
                self._neural_setter("skin"),
                "Subsurface scattering and pore detail on faces. Lower this first "
                "if results look waxy.",
                maximum=NR_STRENGTH_MAX,
            )
        )
        neural_layout.addWidget(
            SliderRow(
                "Local tone",
                settings.local_tone,
                self._neural_setter("local_tone"),
                "How much the model may relight the scene.",
                maximum=NR_STRENGTH_MAX,
            )
        )
        neural_layout.addWidget(
            SliderRow(
                "Structure",
                settings.structure,
                self._neural_setter("structure"),
                "Micro-contrast in fabric, hair, and surface material.",
                maximum=NR_STRENGTH_MAX,
            )
        )
        hdr = QGroupBox("HDR / display")
        hdr.setToolTip(
            "The add-on's HDR controls. This pipeline is SDR end to end, but "
            "these still change the result — the neural pass reasons about light "
            "before anything is tonemapped back. Defaults match the add-on's own."
        )
        hdr_layout = QVBoxLayout(hdr)
        hdr_layout.addWidget(
            SliderRow(
                "Paper white",
                settings.paper_white,
                self._neural_setter("paper_white"),
                "The luminance the model treats as diffuse white. On an HDR or "
                "OLED display this decides how hard highlights are pushed. The "
                "add-on defaults to 1; shipping game configs use 16, which is "
                "also where it stops changing.",
                maximum=NR_PAPER_WHITE_MAX,
            )
        )
        hdr_layout.addWidget(
            SliderRow(
                "HDR transfer",
                settings.transfer_strength,
                self._neural_setter("transfer_strength"),
                "Strength of the transfer curve the pass works through. "
                "Measured range 0..1.",
                maximum=NR_TRANSFER_MAX,
            )
        )
        hdr_layout.addWidget(
            SliderRow(
                "Colour strength",
                settings.color_strength,
                self._neural_setter("color_strength"),
                "How much of the model's colour change is kept. At 0 the source "
                "colour survives and only structure changes. Measured range 0..1.",
                maximum=NR_COLOR_MAX,
            )
        )
        self.live = QCheckBox("Live preview")
        self.live.setChecked(self.settings.evaluation.live_preview)
        self.live.setToolTip(
            "Re-run DLSS automatically when a slider changes. The add-on reads its "
            "settings only when the harness starts, so each change costs a restart "
            "— about four seconds, most of it NGX and add-on initialisation."
        )
        self.live.toggled.connect(self._live_toggled)
        neural_layout.addWidget(self.live)
        layout.addWidget(neural)
        layout.addWidget(hdr)

        depth = QGroupBox("Depth")
        depth_layout = QVBoxLayout(depth)
        self.model_box = QComboBox()
        for label, model_id in MODELS.items():
            self.model_box.addItem(label, model_id)
        index = self.model_box.findData(self.settings.depth.model_id)
        self.model_box.setCurrentIndex(max(0, index))
        self.model_box.currentIndexChanged.connect(self._model_changed)
        depth_layout.addWidget(self.model_box)
        depth_layout.addWidget(
            SliderRow(
                "Depth contrast",
                min(1.0, self.settings.depth.contrast / 3.0),
                self._contrast_changed,
                "Reshapes the near-far spread. Higher pushes more of the frame "
                "into the foreground. Updates the depth mask live.",
            )
        )
        self.tiled = QCheckBox("Tiled depth (slow, sharper silhouettes)")
        self.tiled.setChecked(self.settings.depth.tiled)
        self.tiled.toggled.connect(self._tiled_changed)
        depth_layout.addWidget(self.tiled)
        layout.addWidget(depth)

        evaluation = QGroupBox("Evaluation")
        eval_layout = QVBoxLayout(evaluation)
        row = QHBoxLayout()
        row.addWidget(QLabel("Passes"))
        self.frames = QSpinBox()
        self.frames.setRange(1, 32)
        self.frames.setValue(self.settings.evaluation.frames)
        self.frames.setToolTip(
            "DLSS is temporal. One pass leaves its accumulator empty; the result "
            "firms up over the first few and stops changing by about eight."
        )
        self.frames.valueChanged.connect(
            lambda v: setattr(self.settings.evaluation, "frames", v)
        )
        row.addStretch(1)
        row.addWidget(self.frames)
        eval_layout.addLayout(row)
        self.jitter = QCheckBox("Sub-pixel jitter")
        self.jitter.setChecked(self.settings.evaluation.jitter)
        self.jitter.toggled.connect(lambda v: setattr(self.settings.evaluation, "jitter", v))
        eval_layout.addWidget(self.jitter)
        layout.addWidget(evaluation)

        layout.addStretch(1)

        self.open_button = QPushButton("Open image…")
        self.open_button.setObjectName("secondary")
        self.open_button.clicked.connect(self.browse)
        layout.addWidget(self.open_button)

        self.convert_button = QPushButton("Convert")
        self.convert_button.setEnabled(False)
        self.convert_button.clicked.connect(self.convert)
        layout.addWidget(self.convert_button)

        self.save_button = QPushButton("Save result…")
        self.save_button.setObjectName("secondary")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save)
        layout.addWidget(self.save_button)

        diagnose = QPushButton("Check runtime")
        diagnose.setObjectName("secondary")
        diagnose.clicked.connect(self.diagnose)
        layout.addWidget(diagnose)
        return panel

    # -- first run -----------------------------------------------------------

    def ensure_model_downloaded(self, model_id: str | None = None) -> None:
        """Fetch the depth model if it is missing, behind a progress dialog.

        Nothing about the model ships with the app — it is a separate download
        under its own licence — so the first launch has to fetch it. Doing that
        here rather than inside the first conversion means the wait is explained
        and measured, instead of appearing as a stalled progress line partway
        through what the user thought was a conversion.
        """
        model_id = model_id or self.settings.depth.model_id
        if self._download_thread is not None:
            return
        if DepthEngine.is_downloaded(model_id):
            return

        label = next((k for k, v in MODELS.items() if v == model_id), model_id)
        self._download_dialog = DownloadDialog(
            "Getting the depth model",
            f"{label}\n\nThis is a one-time download, kept in the models folder "
            f"beside the app. Nothing is bundled with the release.",
            self,
        )
        self._download_dialog.setStyleSheet(STYLE)

        self._download_thread = QThread(self)
        self._download_worker = DownloadWorker(model_id, self.engine)
        self._download_worker.moveToThread(self._download_thread)
        self._download_thread.started.connect(self._download_worker.run)
        self._download_worker.progress.connect(self._download_dialog.set_status)
        self._download_worker.bytes_progress.connect(self._download_dialog.update_bytes)
        self._download_worker.finished.connect(self._download_finished)
        self._download_worker.failed.connect(self._download_failed)
        self._download_thread.start()
        # Modal: there is nothing useful to do in the app without this, and a
        # conversion started midway through would race the download.
        self._download_dialog.exec()

    def _close_download(self) -> None:
        if self._download_thread is not None:
            self._download_thread.quit()
            self._download_thread.wait(5000)
            self._download_thread = None
        self._download_worker = None
        if self._download_dialog is not None:
            self._download_dialog.accept()
            self._download_dialog = None

    def _download_finished(self) -> None:
        if self._download_dialog is not None:
            self._download_dialog.mark_complete()
        self._close_download()
        self.statusBar().showMessage("Depth model ready.")

    def _download_failed(self, message: str) -> None:
        self._close_download()
        QMessageBox.warning(
            self,
            "Could not download the depth model",
            f"{message}\n\nCheck your connection and reopen the app, or pick a "
            "different model in the sidebar.",
        )

    # -- getting an image in -------------------------------------------------

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt name
        if first_supported(url.toLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt name
        path = first_supported(url.toLocalFile() for url in event.mimeData().urls())
        if path is not None:
            self.open_image(path)
            event.acceptProposedAction()

    def browse(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Choose an image",
            str(self.image_path.parent) if self.image_path else "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.exr *.hdr)",
        )
        if chosen:
            self.open_image(Path(chosen))

    def paste(self) -> None:
        """Open whatever is on the clipboard: a copied file, or raw pixels.

        Raw pixels are written to a PNG in the scratch folder first, so the rest
        of the app keeps working in paths rather than growing a second,
        in-memory way of holding a source image.
        """
        clipboard = QApplication.clipboard()
        mime = clipboard.mimeData()

        if mime.hasUrls():
            path = first_supported(url.toLocalFile() for url in mime.urls())
            if path is not None:
                self.open_image(path)
                return

        image = clipboard.image()
        if not image.isNull():
            # A screenshot carries no filename, so stamp it. The save dialog
            # names its default after the source, and a folder of results all
            # called "pasted_dlss5.png" would be useless.
            target = paths.scratch_dir() / f"pasted_{int(time.time())}.png"
            if image.save(str(target), "PNG"):
                self.open_image(target)
            else:
                self.statusBar().showMessage("Could not read the image on the clipboard.")
            return

        self.statusBar().showMessage("Nothing on the clipboard to paste.")

    def open_image(self, path: Path) -> None:
        self.image_path = path
        self.result = None
        self.prepared = None
        self.save_button.setEnabled(False)
        self.convert_button.setEnabled(True)
        self.view_depth.setEnabled(False)
        self.view_result.setEnabled(False)
        self.wipe.clear()
        self.depth_view.clear()
        # Through show_view, not setCurrentWidget: the toggle buttons track
        # _view, and setting the stack behind their back leaves "Result" lit up
        # for a result that has just been discarded.
        self.show_view("photo")
        self.statusBar().showMessage(f"Loaded {path.name} — estimating depth…")
        self._start_depth()

    # -- depth ---------------------------------------------------------------

    def _start_depth(self) -> None:
        if self.image_path is None or self._depth_thread is not None:
            return
        self._depth_thread = QThread(self)
        self._depth_worker = DepthWorker(self.image_path, self.settings, self.engine)
        self._depth_worker.moveToThread(self._depth_thread)
        self._depth_thread.started.connect(self._depth_worker.run)
        self._depth_worker.progress.connect(self.statusBar().showMessage)
        self._depth_worker.finished.connect(self._depth_ready)
        self._depth_worker.failed.connect(self._depth_failed)
        self._depth_thread.start()

    def _depth_teardown(self) -> None:
        if self._depth_thread is not None:
            self._depth_thread.quit()
            self._depth_thread.wait()
            self._depth_thread = None
        self._depth_worker = None

    def _depth_ready(self, prepared: pipeline.Prepared) -> None:
        self.prepared = prepared
        self._depth_teardown()
        self.show_view("depth")
        self.statusBar().showMessage(
            "Depth ready — tune contrast against the mask, then Convert."
        )

    def _depth_failed(self, message: str) -> None:
        self._depth_teardown()
        # Depth is required for a conversion, but a failure here must not take
        # the window down: the user can switch model and try again.
        self.statusBar().showMessage(f"Depth estimation failed: {message}")

    def _model_changed(self, _index: int) -> None:
        self.settings.depth.model_id = self.model_box.currentData()
        # Switching model is the other way a download starts, and the same
        # dialog explains it. Fetch before re-running depth, or prepare() would
        # sit on the download with only a status line to show for it.
        self.ensure_model_downloaded(self.settings.depth.model_id)
        self._depth_settings_changed()

    def _tiled_changed(self, value: bool) -> None:
        self.settings.depth.tiled = value
        self._depth_settings_changed()

    # -- live preview --------------------------------------------------------

    def _neural_setter(self, field: str):
        """Slider callback that also schedules a live re-run."""

        def apply(value: float) -> None:
            setattr(self.settings.neural, field, value)
            self._schedule_preview()

        return apply

    def _preset_changed(self, index: int) -> None:
        self.settings.neural.preset = index
        self._schedule_preview()

    def _style_changed(self, index: int) -> None:
        self.settings.neural.style = index
        self._schedule_preview()

    def _live_toggled(self, value: bool) -> None:
        self.settings.evaluation.live_preview = value
        if value:
            self._schedule_preview()

    def _schedule_preview(self) -> None:
        """Restart the debounce timer.

        Dragging a slider emits a change per pixel, and each one is a four-second
        harness launch. Waiting for the drag to settle is the difference between
        one run and forty queued behind it.
        """
        if not self.settings.evaluation.live_preview or self.image_path is None:
            return
        self._preview_timer.start()

    def _run_preview(self) -> None:
        if not self.settings.evaluation.live_preview or self.image_path is None:
            return
        if self._thread is not None:
            # A run is already in flight and its settings are now stale. Mark it
            # so the next idle moment starts one fresh run, rather than queueing
            # a separate run for every slider movement.
            self._preview_pending = True
            return
        self._start_convert(preview=True)

    def _contrast_changed(self, value: float) -> None:
        # Contrast is applied to the finished depth array, so it never
        # invalidates the model output — only the picture drawn from it.
        self.settings.depth.contrast = max(0.2, value * 3.0)
        if self._view == "depth":
            self._render_depth_preview()

    # -- conversion ----------------------------------------------------------

    def convert(self) -> None:
        self._start_convert(preview=False)

    def _start_convert(self, preview: bool) -> None:
        if self.image_path is None or self._thread is not None:
            return
        self._preview_pending = False
        self._previewing = preview
        self.convert_button.setEnabled(False)
        if not preview:
            self.save_button.setEnabled(False)

        self._thread = QThread(self)
        self._worker = Worker(self.image_path, self.settings, self.engine, self.prepared)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.statusBar().showMessage)
        self._worker.finished.connect(self._succeeded)
        self._worker.failed.connect(self._failed)
        self._thread.start()

    def _teardown(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None
        self.convert_button.setEnabled(self.image_path is not None)
        # Settings moved on while that run was in flight; catch up now.
        if self._preview_pending:
            self._preview_pending = False
            self._schedule_preview()

    def _succeeded(self, result: pipeline.Result) -> None:
        self.result = result
        self.wipe.set_images(result.original, result.enhanced)
        self.save_button.setEnabled(True)
        self.show_view("result")
        if self._previewing:
            neural = self.settings.neural
            self.statusBar().showMessage(
                f"Preview — intensity {neural.intensity:.2f}, skin {neural.skin:.2f}, "
                f"tone {neural.local_tone:.2f}, structure {neural.structure:.2f}"
            )
        else:
            self.statusBar().showMessage(
                f"Done — {result.notes}. Drag the divider to compare, "
                "or drop/paste another image."
            )
        self.settings.save(paths.settings_path())
        self._teardown()

    def _failed(self, message: str) -> None:
        QMessageBox.warning(self, "Conversion failed", message)
        self.statusBar().showMessage("Conversion failed")
        self._teardown()

    def save(self) -> None:
        if self.result is None:
            return
        # The app's own output folder, not the home directory: a release build
        # ships one, and defaulting anywhere else scatters results.
        suggested = self.settings.last_output_dir or str(paths.output_dir())
        default = str(Path(suggested) / f"{(self.image_path or Path('image')).stem}_dlss5.png")
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Save result", default, "PNG (*.png);;TIFF (*.tif);;OpenEXR (*.exr);;JPEG (*.jpg)"
        )
        if not chosen:
            return
        try:
            pipeline.save_image(self.result.enhanced, chosen)
        except OSError as error:
            QMessageBox.warning(self, "Could not save", str(error))
            return
        self.settings.last_output_dir = str(Path(chosen).parent)
        self.settings.save(paths.settings_path())
        self.statusBar().showMessage(f"Saved {Path(chosen).name}")

    def diagnose(self) -> None:
        status = runtime.detect(self.settings.runtime_dir or None)
        lines = []
        for label, value in (
            ("Neural model", status.neural_dll),
            ("DLSS SR", status.dlss_dll),
            ("RenoDX add-on", status.addon),
            ("ReShade", status.reshade),
            ("Harness", status.harness),
        ):
            lines.append(f"{label}: {value or 'not found'}")
        if status.problems:
            lines.append("")
            lines += status.problems
        elif status.harness is not None:
            lines.append("")
            lines.append(evaluator.probe(status.harness))
        QMessageBox.information(self, "DLSS 5 runtime", "\n".join(lines))

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt name
        self.settings.save(paths.settings_path())
        self._preview_timer.stop()
        self._preview_pending = False

        # Order matters. Killing the harness first makes the worker's blocking
        # readline() return immediately, so the thread below finishes instead of
        # sitting through a whole DLSS evaluation while the window is gone.
        # Without this, closing mid-conversion leaves a harness holding the GPU
        # and the runtime DLLs open with no window left to close.
        try:
            evaluator.terminate_all()
        except Exception:  # noqa: BLE001 - closing must not fail
            pass

        for thread in (self._thread, self._depth_thread, self._download_thread):
            if thread is None:
                continue
            thread.quit()
            # Bounded: depth inference is inside torch and cannot be
            # interrupted, and hanging the close on it would be worse than
            # letting the interpreter tear it down. atexit still sweeps any
            # harness a late-finishing worker manages to start.
            if not thread.wait(5000):
                thread.terminate()
        self._thread = self._depth_thread = self._download_thread = None
        self._worker = self._depth_worker = self._download_worker = None
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("DLSS 5 Image Converter")
    # Before the window: nothing in it works without a depth model, and the
    # model needs torch. A release ships neither.
    if not ensure_runtime_ready():
        sys.exit(1)
    # High-DPI pixmaps: the wipe view draws downscaled photographs, and without
    # this they are resampled twice on a scaled display and look soft in exactly
    # the detail this tool exists to show off.
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
