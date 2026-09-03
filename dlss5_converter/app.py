"""The desktop application."""

from __future__ import annotations

import copy
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import QObject, QThread, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import (
    bootstrap,
    contract,
    discovery,
    evaluator,
    grade,
    paths,
    pipeline,
    resample,
    runtime,
    sequence,
)
from . import hdr as hdr_mod
from .depth_engine import MODELS, DepthEngine
from .settings import (
    MAX_EDGE_CHOICES,
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
    SideBySideView,
    SliderRow,
    WipeView,
    first_supported,
)

#: The sidebar's own width. The scroll area around it adds room for a bar.
SIDEBAR_WIDTH = 320

#: How many images the style comparison can show at once. Three is the useful
#: ceiling: the source plus both styles. A fourth would only repeat one of them,
#: and each pane costs a third of the width to look at.
MAX_COMPARE_PANES = 3

#: "DLSS 5 pass 3 of 8" - the only progress message carrying a count, and what
#: drives the colour sweep over the source image while a conversion runs.
_PASS_COUNT = re.compile(r"pass (\d+) of (\d+)")

#: Matches SideBySideView.GAP so the pane dropdowns line up with the panes.
GAP_BETWEEN_PANES = 12

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


#: Longest edge of the copy the colour sliders are dragged against.
#:
#: Grading a 33-megapixel 8K array on every slider tick stutters badly, and the
#: comparison view never draws more than about 1200 px across anyway, so a
#: larger preview buys nothing visible. Measured on this box: 1600 px costs
#: ~290 ms a redraw, 1200 px about 160 ms, which with the 30 ms coalescing timer
#: is the difference between a slider that drags and one that lurches.
#: The saved file is always graded at full resolution.
_PREVIEW_EDGE = 1200


def _downscale_for_preview(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, _PREVIEW_EDGE / float(max(height, width)))
    if scale >= 1.0:
        return image
    return cv2.resize(
        image, (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


#: The guide lives in the repository wiki rather than in the app so it can be
#: corrected the day a new failure is reported, instead of at the next release.
WIKI_URL = "https://github.com/criso2hd-alt/DLSS5-Image-Converter/wiki"


def open_help(page: str = "") -> None:
    """Open the wiki in the user's browser.

    Failing silently is deliberate: on a machine with no browser association
    this returns False, and an error dialog about being unable to show the help
    would be a worse experience than the missing help itself.
    """
    url = f"{WIKI_URL}/{page}" if page else WIKI_URL
    try:
        QDesktopServices.openUrl(QUrl(url))
    except Exception:  # noqa: BLE001 - opening a browser must never take the app down
        pass


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


class StyleWorker(QObject):
    """Converts the same image once per neural style, back to back.

    Sequential rather than parallel, and it has to be: the add-on reads its ini
    once when it starts, so a style change means a new harness. Two styles are
    therefore two harness launches — which is also why this exists as a
    deliberate action behind a button rather than something that happens on
    every convert.

    Depth is the expensive half and is shared: both runs get the same
    ``Prepared``, so the second one costs only its DLSS passes.
    """

    progress = Signal(str)
    finished = Signal(object)  # dict[int, pipeline.Result]
    failed = Signal(str)

    def __init__(
        self,
        image_path: Path,
        settings: AppSettings,
        engine: DepthEngine,
        styles: list[int],
        prepared: pipeline.Prepared | None = None,
    ) -> None:
        super().__init__()
        self._path = image_path
        self._settings = settings
        self._engine = engine
        self._styles = styles
        self._prepared = prepared
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        results: dict[int, pipeline.Result] = {}
        # A copy, so the sidebar keeps showing whatever the user actually chose
        # while this runs through the styles behind their back.
        settings = copy.deepcopy(self._settings)
        try:
            prepared = self._prepared
            if prepared is None:
                # Once, up front. Letting the first convert() do it internally
                # would leave the second one to estimate depth all over again.
                prepared = pipeline.prepare(
                    self._path, settings, self._engine, progress=self.progress.emit
                )

            for position, style in enumerate(self._styles, start=1):
                if self._stop:
                    return
                name = NR_STYLES[style]
                settings.neural.style = style

                def say(message: str, name=name, position=position) -> None:
                    self.progress.emit(
                        f"{name} ({position} of {len(self._styles)}) — {message}"
                    )

                results[style] = pipeline.convert(
                    self._path, settings, self._engine, progress=say, prepared=prepared
                )
        except Exception as error:  # noqa: BLE001 - the UI is the error handler
            self.failed.emit(str(error))
            return
        self.finished.emit(results)


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


class FindFilesWorker(QObject):
    """Searches the disk for the user's DLSS files, off the UI thread."""

    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, roots: list[Path]) -> None:
        super().__init__()
        self._roots = roots
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            found = discovery.scan(
                self._roots,
                on_progress=self.progress.emit,
                should_stop=lambda: self._stop,
            )
        except Exception as error:  # noqa: BLE001 - the UI is the error handler
            self.failed.emit(str(error))
            return
        self.finished.emit(found)


class FindFilesDialog(QDialog):
    """Find the four files on the user's own disk and copy them in.

    The single biggest step between "downloaded it" and "it works". Most people
    arriving here have already made DLSS 5 run in a game, so the files exist —
    they just should not have to know that `nvngx_dlssnr.dll` is the one that
    matters or which of their games has the newest add-on.
    """

    def __init__(self, window: MainWindow) -> None:
        super().__init__(window)
        self._window = window
        self._candidates: list[discovery.Candidate] = []
        self._extra_roots: list[Path] = []
        self._thread: QThread | None = None
        self._worker: FindFilesWorker | None = None

        self.setWindowTitle("Find my DLSS files")
        self.setModal(False)
        self.setMinimumWidth(720)
        self.setStyleSheet(STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)

        blurb = QLabel(
            "Searches your Steam libraries, Downloads and Documents for the four "
            "files, and copies them here.\n\n"
            "Nothing is downloaded — this only looks at files already on your "
            "machine. If DLSS 5 works in a game for you, that game's folder is "
            "what it is looking for."
        )
        blurb.setWordWrap(True)
        blurb.setObjectName("hint")
        layout.addWidget(blurb)

        self.results = QListWidget()
        self.results.setMinimumHeight(180)
        self.results.currentRowChanged.connect(self._selection_changed)
        layout.addWidget(self.results)

        self.summary = QLabel("")
        self.summary.setObjectName("hint")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.status = QLabel("Ready.")
        self.status.setObjectName("hint")
        layout.addWidget(self.status)

        buttons = QHBoxLayout()
        self.close_button = QPushButton("Close")
        self.close_button.setObjectName("secondary")
        self.add_folder = QPushButton("Search another folder…")
        self.add_folder.setObjectName("secondary")
        self.rescan = QPushButton("Search again")
        self.rescan.setObjectName("secondary")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("secondary")
        self.stop_button.setVisible(False)
        self.copy_button = QPushButton("Copy these files")
        self.copy_button.setEnabled(False)
        buttons.addWidget(self.close_button)
        buttons.addWidget(self.add_folder)
        buttons.addWidget(self.rescan)
        buttons.addStretch(1)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.copy_button)
        layout.addLayout(buttons)

        self.close_button.clicked.connect(self.close)
        self.add_folder.clicked.connect(self._add_folder)
        self.rescan.clicked.connect(self.start_scan)
        self.stop_button.clicked.connect(self._stop)
        self.copy_button.clicked.connect(self._copy)

    # -- scanning ------------------------------------------------------------

    def start_scan(self) -> None:
        if self._thread is not None:
            return
        roots = discovery.default_roots() + self._extra_roots
        if not roots:
            self.status.setText("Nowhere obvious to look. Use 'Search another folder…'.")
            return
        self.results.clear()
        self.summary.setText("")
        self.copy_button.setEnabled(False)
        self.rescan.setEnabled(False)
        self.stop_button.setVisible(True)
        self.status.setText("Searching…")

        self._thread = QThread(self)
        self._worker = FindFilesWorker(roots)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.status.setText)
        self._worker.finished.connect(self._scan_done)
        self._worker.failed.connect(self._scan_failed)
        self._thread.start()

    def _stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self.status.setText("Stopping…")

    def _teardown(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(10000)
            self._thread = None
        self._worker = None
        self.stop_button.setVisible(False)
        self.rescan.setEnabled(True)

    def _scan_done(self, candidates: list[discovery.Candidate]) -> None:
        self._teardown()
        self._candidates = candidates
        if not candidates:
            self.status.setText(
                "Nothing found. If DLSS 5 works in a game, use 'Search another "
                "folder…' and point it at that game."
            )
            return

        complete = [c for c in candidates if c.complete]
        for candidate in candidates:
            self.results.addItem(QListWidgetItem(candidate.describe()))
        self.results.setCurrentRow(0)
        self.status.setText(
            f"{len(candidates)} folder(s) found, {len(complete)} with all four. "
            "The best is selected."
        )

    def _scan_failed(self, message: str) -> None:
        self._teardown()
        self.status.setText(message)

    def _add_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Also search this folder")
        if chosen:
            self._extra_roots.append(Path(chosen))
            self.start_scan()

    # -- choosing and copying ------------------------------------------------

    def _plan(self) -> dict[str, Path]:
        """What would be copied for the current selection."""
        row = self.results.currentRow()
        if row < 0 or row >= len(self._candidates):
            return {}
        # The selected folder first, then anything else fills the gaps and the
        # newest add-on wins - same rule as the automatic choice.
        ordered = [self._candidates[row]] + [
            c for i, c in enumerate(self._candidates) if i != row
        ]
        return discovery.best_set(ordered)

    def _selection_changed(self, _row: int) -> None:
        plan = self._plan()
        self.copy_button.setEnabled(len(plan) == len(discovery.WANTED))
        if not plan:
            self.summary.setText("")
            return
        lines = []
        for name in discovery.WANTED:
            path = plan.get(name)
            if path is None:
                lines.append(f"   {name} — still missing")
            else:
                size = path.stat().st_size / 1048576
                lines.append(f"   {name}  ({size:.1f} MB)  from  {path.parent}")
        note = ""
        if len({p.parent for p in plan.values()}) > 1:
            note = (
                "\nFiles come from more than one folder. That is usually fine — the "
                "newest add-on is always preferred, because an out-of-date one is a "
                "known cause of the neural pass silently not running."
            )
        self.summary.setText("Would copy:\n" + "\n".join(lines) + note)

    def _copy(self) -> None:
        plan = self._plan()
        if not plan:
            return
        destination = paths.dlss_files_dir()
        copied, skipped = discovery.install(plan, destination)
        parts = []
        if copied:
            parts.append(f"copied {', '.join(copied)}")
        if skipped:
            parts.append(f"left alone (already there): {', '.join(skipped)}")
        self.status.setText("; ".join(parts) or "Nothing to do.")
        self._window.refresh_runtime_status()
        if copied:
            QMessageBox.information(
                self,
                "Files copied",
                f"{len(copied)} file(s) copied into:\n{destination}\n\n"
                "Use Check runtime to confirm everything is working.",
            )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt name
        if self._worker is not None:
            self._worker.stop()
        self._teardown()
        super().closeEvent(event)


class BatchWorker(QObject):
    """Runs a folder through the current settings, off the UI thread."""

    progress = Signal(str)
    item_done = Signal(object)
    finished = Signal()
    failed = Signal(str)

    def __init__(
        self,
        images: list[Path],
        settings: AppSettings,
        engine: DepthEngine,
        destination: Path,
        skip_existing: bool,
    ) -> None:
        super().__init__()
        self._images = images
        self._settings = settings
        self._engine = engine
        self._destination = destination
        self._skip = skip_existing
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            for item in pipeline.convert_batch(
                self._images,
                self._settings,
                self._engine,
                self._destination,
                grade_settings=self._settings.grade,
                skip_existing=self._skip,
                progress=self.progress.emit,
                should_stop=lambda: self._stop,
            ):
                self.item_done.emit(item)
        except Exception as error:  # noqa: BLE001 - the UI is the error handler
            self.failed.emit(str(error))
            return
        self.finished.emit()


class BatchDialog(QDialog):
    """Apply the settings you just tuned to a whole folder.

    A dialog rather than a third tab. Batch is not a different mode — it is
    "do that again, to these" — so it belongs to the page where the settings
    were chosen, and it should disappear again afterwards. A permanent panel
    would sit there empty most of the time and make the main page busier for a
    feature used occasionally.
    """

    def __init__(self, window: MainWindow) -> None:
        super().__init__(window)
        self._window = window
        self.images: list[Path] = []
        self.destination = paths.output_dir()
        self._thread: QThread | None = None
        self._worker: BatchWorker | None = None
        self._done = 0
        self._failed = 0
        self._skipped = 0

        self.setWindowTitle("Apply to folder")
        self.setModal(False)
        self.setMinimumWidth(560)
        self.setStyleSheet(STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)

        summary = QLabel(
            "Every image in the folder gets the settings currently in the "
            "sidebar — neural strengths, depth, passes, size and the colour "
            "grade. Tune them on one image first, then point this at the rest."
        )
        summary.setWordWrap(True)
        summary.setObjectName("hint")
        layout.addWidget(summary)

        source_row = QHBoxLayout()
        self.pick_source = QPushButton("Input folder…")
        self.source_label = QLabel("No folder chosen")
        self.source_label.setObjectName("hint")
        source_row.addWidget(self.pick_source)
        source_row.addWidget(self.source_label, 1)
        layout.addLayout(source_row)

        dest_row = QHBoxLayout()
        self.pick_dest = QPushButton("Output folder…")
        self.pick_dest.setObjectName("secondary")
        self.dest_label = QLabel(str(self.destination))
        self.dest_label.setObjectName("hint")
        dest_row.addWidget(self.pick_dest)
        dest_row.addWidget(self.dest_label, 1)
        layout.addLayout(dest_row)

        self.recursive = QCheckBox("Include sub-folders")
        self.skip_existing = QCheckBox("Skip images already converted")
        self.skip_existing.setChecked(True)
        self.skip_existing.setToolTip(
            "Lets an interrupted run be restarted without redoing everything."
        )
        layout.addWidget(self.recursive)
        layout.addWidget(self.skip_existing)

        # A count tells you the run is alive; the picture tells you it is doing
        # the right thing. On a folder of two hundred that is the difference
        # between watching and checking back in ten minutes.
        self.preview = ImageView()
        # Small by default - it is confirmation that the run is doing the right
        # thing, not something to judge quality on. It grows with the dialog for
        # anyone who wants a closer look.
        self.preview.setMinimumHeight(170)
        self.preview.setVisible(False)
        layout.addWidget(self.preview, 1)

        self.bar = QProgressBar()
        self.bar.setVisible(False)
        layout.addWidget(self.bar)
        self.status = QLabel("")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        buttons = QHBoxLayout()
        self.close_button = QPushButton("Close")
        self.close_button.setObjectName("secondary")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("secondary")
        self.stop_button.setVisible(False)
        self.start_button = QPushButton("Convert folder")
        self.start_button.setEnabled(False)
        buttons.addWidget(self.close_button)
        buttons.addStretch(1)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.start_button)
        layout.addLayout(buttons)

        self.pick_source.clicked.connect(self._choose_source)
        self.pick_dest.clicked.connect(self._choose_destination)
        self.recursive.toggled.connect(self._rescan)
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self._stop)
        self.close_button.clicked.connect(self.close)

    # -- choosing ------------------------------------------------------------

    def _choose_source(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Folder of images to convert")
        if chosen:
            self._source = Path(chosen)
            self._rescan()

    def _rescan(self) -> None:
        source = getattr(self, "_source", None)
        if source is None:
            return
        self.images = pipeline.list_images(source, self.recursive.isChecked())
        self.source_label.setText(f"{source}  —  {len(self.images)} image(s)")
        self.start_button.setEnabled(bool(self.images))
        if not self.images:
            self.status.setText("Nothing here this app can read.")

    def _choose_destination(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Where should the results go?", str(self.destination)
        )
        if chosen:
            self.destination = Path(chosen)
            self.dest_label.setText(chosen)

    # -- running -------------------------------------------------------------

    def _start(self) -> None:
        if not self.images or self._thread is not None:
            return
        self._done = self._failed = self._skipped = 0
        self.bar.setVisible(True)
        self.bar.setRange(0, len(self.images))
        self.bar.setValue(0)
        self.start_button.setEnabled(False)
        self.pick_source.setEnabled(False)
        self.pick_dest.setEnabled(False)
        self.stop_button.setVisible(True)

        self._thread = QThread(self)
        self._worker = BatchWorker(
            self.images,
            self._window.settings,
            self._window.engine,
            self.destination,
            self.skip_existing.isChecked(),
        )
        self.preview.setVisible(True)
        self.preview.clear()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.status.setText)
        self._worker.item_done.connect(self._item_done)
        self._worker.finished.connect(self._finished)
        self._worker.failed.connect(self._failed_run)
        self._thread.start()

    def _stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self.status.setText("Stopping after this image…")

    def _item_done(self, item: pipeline.BatchItem) -> None:
        if item.skipped:
            self._skipped += 1
        elif item.error:
            self._failed += 1
        else:
            self._done += 1
        self.bar.setValue(item.index + 1)
        self.bar.setFormat(f"%v of %m — {item.source.name}")
        if item.image is not None:
            self.preview.setVisible(True)
            self.preview.set_image(
                _downscale_for_preview(item.image),
                f"{item.index + 1} of {item.total} — {item.source.name}",
            )
        if item.error:
            # Named, not swallowed: a batch that quietly drops files is worse
            # than one that stops.
            self.status.setText(f"{item.source.name} failed — {item.error}")

    def _teardown(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(10000)
            self._thread = None
        self._worker = None
        self.stop_button.setVisible(False)
        self.start_button.setEnabled(bool(self.images))
        self.pick_source.setEnabled(True)
        self.pick_dest.setEnabled(True)

    def _finished(self) -> None:
        self._teardown()
        parts = [f"{self._done} converted"]
        if self._skipped:
            parts.append(f"{self._skipped} already done")
        if self._failed:
            parts.append(f"{self._failed} failed")
        self.status.setText(", ".join(parts) + f" — {self.destination}")

    def _failed_run(self, message: str) -> None:
        self._teardown()
        self.status.setText(message)
        QMessageBox.warning(self, "Batch failed", message)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt name
        if self._worker is not None:
            self._worker.stop()
        self._teardown()
        super().closeEvent(event)


class SequenceWorker(QObject):
    """Drives a whole sequence off the UI thread, reporting each frame."""

    progress = Signal(str)
    frame_done = Signal(object)
    finished = Signal(int)
    failed = Signal(str)

    def __init__(
        self,
        frames: list[Path],
        depth_frames: list[Path] | None,
        invert_depth: bool,
        settings: AppSettings,
        engine: DepthEngine,
        destination: Path,
    ) -> None:
        super().__init__()
        self._frames = frames
        self._depth = depth_frames
        self._invert = invert_depth
        self._settings = settings
        self._engine = engine
        self._destination = destination
        self._stop = False

    def stop(self) -> None:
        """Asked for from the UI thread; read between frames, never mid-frame."""
        self._stop = True

    def run(self) -> None:
        done = 0
        try:
            for frame in pipeline.convert_sequence(
                self._frames,
                self._settings,
                self._engine,
                self._destination,
                depth_frames=self._depth,
                invert_depth=self._invert,
                grade_settings=self._settings.grade,
                progress=self.progress.emit,
                should_stop=lambda: self._stop,
            ):
                done += 1
                self.frame_done.emit(frame)
        except Exception as error:  # noqa: BLE001 - the UI is the error handler
            self.failed.emit(str(error))
            return
        self.finished.emit(done)


class SequencePage(QWidget):
    """Page two: a rendered image sequence, frame by frame.

    Shares the sidebar with single-image mode, because the settings mean the
    same thing here — and using identical settings across every frame is most of
    what makes a sequence look consistent.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.frames: list[Path] = []
        self.depth_frames: list[Path] = []
        self.outputs: list[Path] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.preview = ImageView()
        layout.addWidget(self.preview, 1)

        self.frames_label = QLabel("No sequence chosen")
        self.frames_label.setObjectName("hint")
        self.depth_label = QLabel("Depth: estimated per frame (Depth Anything)")
        self.depth_label.setObjectName("hint")
        self.output_label = QLabel("")
        self.output_label.setObjectName("hint")

        pick_row = QHBoxLayout()
        self.pick_frames = QPushButton("Choose first frame…")
        self.pick_depth = QPushButton("Choose first depth frame…")
        self.pick_depth.setObjectName("secondary")
        self.pick_depth.setToolTip(
            "Optional, and the reason this mode can be temporally stable.\n\n"
            "Depth Anything estimates depth per frame and wobbles slightly "
            "between them, which the neural pass follows as flicker. A depth "
            "pass out of Blender or Maya is geometrically exact and does not "
            "move, so the result is steady."
        )
        self.clear_depth = QPushButton("Use estimated depth")
        self.clear_depth.setObjectName("secondary")
        self.clear_depth.setVisible(False)
        pick_row.addWidget(self.pick_frames)
        pick_row.addWidget(self.pick_depth)
        pick_row.addWidget(self.clear_depth)
        pick_row.addStretch(1)

        self.invert_depth = QCheckBox("Depth is inverted (near is dark)")
        self.invert_depth.setToolTip(
            "Renderers disagree about which way up a depth pass goes. A Blender "
            "mist pass is near-dark, so tick this. Watch the preview and pick "
            "whichever looks right - near should read as red on the depth mask."
        )
        self.invert_depth.setVisible(False)

        layout.addLayout(pick_row)
        layout.addWidget(self.frames_label)
        layout.addWidget(self.depth_label)
        layout.addWidget(self.invert_depth)

        out_row = QHBoxLayout()
        self.pick_output = QPushButton("Output folder…")
        self.pick_output.setObjectName("secondary")
        self.write_video = QCheckBox("Also write MP4")
        self.fps = QSpinBox()
        self.fps.setRange(1, 240)
        self.fps.setValue(24)
        self.fps.setSuffix(" fps")
        self.fps.setEnabled(False)
        self.write_video.toggled.connect(self.fps.setEnabled)
        self.write_video.setToolTip(
            "Encoded with mp4v, not H.264 - OpenCV ships no H.264 encoder. The "
            "PNG frames are always written too, so you can re-encode them with "
            "anything you like."
        )
        out_row.addWidget(self.pick_output)
        out_row.addWidget(self.write_video)
        out_row.addWidget(self.fps)
        out_row.addStretch(1)
        layout.addLayout(out_row)
        layout.addWidget(self.output_label)

        self.bar = QProgressBar()
        self.bar.setTextVisible(True)
        self.bar.setVisible(False)
        layout.addWidget(self.bar)

        run_row = QHBoxLayout()
        self.start = QPushButton("Convert sequence")
        self.start.setEnabled(False)
        self.stop = QPushButton("Stop")
        self.stop.setObjectName("secondary")
        self.stop.setVisible(False)
        run_row.addStretch(1)
        run_row.addWidget(self.stop)
        run_row.addWidget(self.start)
        layout.addLayout(run_row)


class ExportDialog(QDialog):
    """Choose the size of the file being written.

    This exists because "Max size" was being read as an output resolution
    picker. It is not — it is the resolution DLSS runs at, picked for VRAM and
    time — and the question people were actually asking, "how big is the image I
    get?", had nowhere to be asked. So it gets asked here, at the moment it
    means something.

    It defaults to native and does not remember a different answer between
    launches, so the fast path stays Save ▸ Enter ▸ Enter for anyone who does
    not care.
    """

    def __init__(self, parent: QWidget, size: tuple[int, int]) -> None:
        super().__init__(parent)
        self._source = size
        self._syncing = False

        self.setWindowTitle("Export size")
        self.setModal(True)
        self.setMinimumWidth(420)
        self.setStyleSheet(STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)

        width, height = size
        heading = QLabel(f"The result is {width} x {height}.")
        heading.setWordWrap(True)
        layout.addWidget(heading)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Size"))
        self.preset = QComboBox()
        self.preset.addItem("Native", 0.0)
        for multiple in (1.5, 2.0, 3.0, 4.0):
            self.preset.addItem(f"{multiple:g}x", multiple)
        # Absolute long edges as well: someone delivering to a 4K spec thinks in
        # the target, not in a multiplier of whatever DLSS happened to run at.
        for edge in (1920, 2560, 3840, 5120, 7680):
            self.preset.addItem(f"{edge} px long edge", float(-edge))
        preset_row.addStretch(1)
        preset_row.addWidget(self.preset)
        layout.addLayout(preset_row)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Width"))
        self.width_box = QSpinBox()
        self.width_box.setRange(1, resample.MAX_EDGE)
        self.width_box.setValue(width)
        size_row.addWidget(self.width_box)
        size_row.addWidget(QLabel("Height"))
        self.height_box = QSpinBox()
        self.height_box.setRange(1, resample.MAX_EDGE)
        self.height_box.setValue(height)
        size_row.addWidget(self.height_box)
        size_row.addStretch(1)
        layout.addLayout(size_row)

        self.lock = QCheckBox("Keep proportions")
        self.lock.setChecked(True)
        self.lock.setToolTip(
            "Untick only if you mean to change the aspect ratio. Everything "
            "here stretches the image; nothing crops it."
        )
        layout.addWidget(self.lock)

        self.hint = QLabel()
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

        note = QLabel(
            "Plain resampling, not a second AI pass — Lanczos when enlarging, "
            "area averaging when shrinking, both computed in linear light. "
            "Enlarging cannot add detail the neural pass did not produce; for "
            "more real detail, raise Max size instead and convert again."
        )
        note.setObjectName("hint")
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.setObjectName("secondary")
        cancel.clicked.connect(self.reject)
        save = QPushButton("Save…")
        save.setDefault(True)
        save.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        layout.addLayout(buttons)

        self.preset.currentIndexChanged.connect(self._preset_chosen)
        self.width_box.valueChanged.connect(lambda v: self._edited(v, from_width=True))
        self.height_box.valueChanged.connect(lambda v: self._edited(v, from_width=False))
        self._describe()

    # -- behaviour -----------------------------------------------------------

    def chosen(self) -> tuple[int, int]:
        return (self.width_box.value(), self.height_box.value())

    def _preset_chosen(self) -> None:
        data = self.preset.currentData()
        # Clearing the selection (index -1, when a size is typed by hand) fires
        # this too, with no data attached. It is not a choice; ignore it.
        if self._syncing or data is None:
            return
        value = float(data)
        if value == 0.0:
            target = self._source
        elif value < 0:
            target = resample.fit(self._source, int(-value))
        else:
            target = resample.proportional(self._source, value)
        self._set(target)

    def _edited(self, value: int, *, from_width: bool) -> None:
        if self._syncing:
            return
        if self.lock.isChecked():
            target = resample.match(
                self._source,
                value if from_width else None,
                None if from_width else value,
            )
        else:
            target = self.chosen()
        # Typing a size by hand means the preset no longer describes it.
        self._syncing = True
        self.preset.setCurrentIndex(-1)
        self._syncing = False
        self._set(target, keep_edited=from_width)

    def _set(self, target: tuple[int, int], keep_edited: bool | None = None) -> None:
        self._syncing = True
        if keep_edited is not True:
            self.width_box.setValue(target[0])
        if keep_edited is not False:
            self.height_box.setValue(target[1])
        self._syncing = False
        self._describe()

    def _describe(self) -> None:
        self.hint.setText(resample.describe(self._source, self.chosen()))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DLSS 5 Image Converter")
        # Sized to the screen rather than fixed: 1280x820 does not fit on a
        # 1920x1080 display at 150% scaling, which reports 1280x720 of usable
        # space, and the window came up taller than the desktop.
        available = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1280, available.width() - 40), min(820, available.height() - 60))
        self.setMinimumSize(900, 560)

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
        # Preview-sized copies of the last result, so the grade sliders stay
        # live on an 8K image. The full-resolution arrays live on self.result
        # and are what actually gets saved.
        self._preview_before: np.ndarray | None = None
        self._preview_after: np.ndarray | None = None
        self._preview_after_linear: np.ndarray | None = None
        self._preview_before_u8: np.ndarray | None = None
        self._grade_rows: dict[str, SliderRow] = {}
        self._download_thread: QThread | None = None
        self._download_worker: DownloadWorker | None = None
        self._download_dialog: DownloadDialog | None = None
        self._seq_thread: QThread | None = None
        self._seq_worker: SequenceWorker | None = None
        self.style_results: dict[int, pipeline.Result] = {}
        self._style_signature_used: tuple | None = None
        self._style_worker: StyleWorker | None = None

        # Debounce. A slider drag emits a change per pixel and each one costs a
        # harness restart, so wait for the drag to settle before spending one.
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(600)
        self._preview_timer.timeout.connect(self._run_preview)

        self._grade_timer = QTimer(self)
        self._grade_timer.setSingleShot(True)
        self._grade_timer.setInterval(30)
        self._grade_timer.timeout.connect(self._render_current)

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
        self.diff_view = ImageView()
        self.stack.addWidget(self.drop)
        self.stack.addWidget(self.wipe)
        self.stack.addWidget(self.depth_view)
        self.stack.addWidget(self.diff_view)
        self.side_by_side = SideBySideView()
        self.stack.addWidget(self.side_by_side)

        canvas = QWidget()
        canvas_layout = QVBoxLayout(canvas)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(8)
        canvas_layout.addWidget(self.stack, 1)
        canvas_layout.addLayout(self._view_switch())
        self.grade_panel = self._grade_panel()
        canvas_layout.addWidget(self.grade_panel)
        self.style_panel = self._style_panel()
        canvas_layout.addWidget(self.style_panel)

        # Two pages, one shared sidebar. The settings mean the same thing in
        # both, and using identical settings across every frame is most of what
        # makes a sequence look consistent - so they should not be duplicated.
        self.sequence_page = SequencePage()
        self.tabs = QTabWidget()
        self.tabs.addTab(canvas, "Single image")
        self.tabs.addTab(self.sequence_page, "Image sequence")
        self._wire_sequence_page()

        layout.addWidget(self.tabs, 1)
        # The sidebar scrolls rather than being squeezed. It is a fixed stack of
        # group boxes, and on a shorter window - or simply at 125% or 150%
        # display scaling, where every widget is taller - the bottom of it was
        # being clipped away silently. A user reported the neural sliders
        # missing entirely and the HDR labels cut in half.
        scroller = QScrollArea()
        scroller.setWidget(self._sidebar())
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QFrame.Shape.NoFrame)
        scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroller.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Only the settings scroll. The actions sit under them and stay put.
        sidebar = QWidget()
        column = QVBoxLayout(sidebar)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(scroller, 1)
        column.addWidget(self._actions())
        # Room for the scrollbar so it never overlaps the slider readouts.
        sidebar.setFixedWidth(SIDEBAR_WIDTH + 18)
        layout.addWidget(sidebar)

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
        self.view_diff = QPushButton("Difference")
        self.view_diff.setToolTip(
            "What the neural pass actually changed, amplified to fill the range.\n\n"
            "The honest answer to 'did anything happen?' - at conservative "
            "settings on already-realistic content the change can be real and "
            "still invisible side by side. Black means nothing changed there."
        )
        self.view_styles = QPushButton("Compare styles")
        self.view_styles.setToolTip(
            "Natural against Cinematic, on this image, at these settings.\n\n"
            "Converts once per style and wipes between them. There are only "
            "two, so this is the whole choice rather than a sample.\n\n"
            "It costs two conversions: the add-on reads its configuration once "
            "when it starts, so a style change needs a new harness. Depth is "
            "estimated once and shared."
        )
        for button in (
            self.view_photo, self.view_depth, self.view_result,
            self.view_diff, self.view_styles,
        ):
            button.setObjectName("secondary")
            button.setCheckable(True)
        self.view_photo.clicked.connect(lambda: self.show_view("photo"))
        self.view_depth.clicked.connect(lambda: self.show_view("depth"))
        self.view_result.clicked.connect(lambda: self.show_view("result"))
        self.view_diff.clicked.connect(lambda: self.show_view("difference"))
        self.view_styles.clicked.connect(lambda: self.show_view("styles"))
        self.view_grade = QPushButton("Colour")
        self.view_grade.setObjectName("secondary")
        self.view_grade.setCheckable(True)
        self.view_grade.setToolTip(
            "Exposure, contrast, saturation and vibrance, applied to the "
            "finished image.\n\n"
            "After the neural pass, not before, so it is instant - nothing is "
            "re-evaluated and nothing is lost. Grading the input instead would "
            "change what the model sees and cost a full re-run per nudge."
        )
        self.view_grade.toggled.connect(self._grade_toggled)

        row.addStretch(1)
        row.addWidget(self.view_photo)
        row.addWidget(self.view_depth)
        row.addWidget(self.view_result)
        row.addWidget(self.view_diff)
        row.addWidget(self.view_styles)
        # Set apart: it is not another view, it changes the one you are on.
        row.addSpacing(28)
        row.addWidget(self.view_grade)
        row.addStretch(1)
        return row

    def _grade_panel(self) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)
        grade = self.settings.grade

        for label, field, low, high, tip in (
            ("Exposure", "exposure", -2.0, 2.0, "Stops of light. Applied in linear."),
            ("Contrast", "contrast", -1.0, 1.0, "S-curve around middle grey."),
            ("Saturation", "saturation", -1.0, 1.0, "Uniform. -1 is greyscale."),
            ("Vibrance", "vibrance", -1.0, 1.0,
             "Lifts muted colour and leaves saturated colour alone, so skies "
             "move and skin mostly does not."),
        ):
            row = SliderRow(
                label,
                getattr(grade, field),
                self._grade_setter(field),
                tip,
                maximum=high,
                minimum=low,
            )
            self._grade_rows[field] = row
            layout.addWidget(row, 1)

        reset = QPushButton("Reset")
        reset.setObjectName("secondary")
        reset.clicked.connect(self._reset_grade)
        layout.addWidget(reset)

        panel.setVisible(False)
        return panel

    def _style_panel(self) -> QWidget:
        """What each pane shows, and which version to keep.

        A dropdown per pane rather than a fixed Natural-versus-Cinematic, because
        the first thing people asked for once they had the comparison was the
        source alongside it: "is this better" and "is this different" are
        different questions, and only the first one needs the original in view.

        The dropdowns stretch equally, so they line up under the panes they
        control - the row is the same width as the canvas and the panes are
        equal, so the alignment is structural rather than fiddled.
        """
        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        self.style_choices: list[QComboBox] = []
        picker = QHBoxLayout()
        picker.setSpacing(GAP_BETWEEN_PANES)
        for index in range(MAX_COMPARE_PANES):
            box = QComboBox()
            box.addItem("Original", -1)
            for style, name in enumerate(NR_STYLES):
                box.addItem(name, style)
            box.setToolTip("What this pane shows.")
            box.currentIndexChanged.connect(self._style_selection_changed)
            self.style_choices.append(box)
            picker.addWidget(box, 1)
        outer.addLayout(picker)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)

        self.style_caption = QLabel()
        self.style_caption.setObjectName("hint")
        row.addWidget(self.style_caption, 1)

        self.style_count_box = QComboBox()
        self.style_count_box.addItem("2 panes", 2)
        self.style_count_box.addItem("3 panes", 3)
        self.style_count_box.setToolTip(
            "Three fits the source next to both styles, which is the way to "
            "answer whether the pass is helping at all rather than only which "
            "style you prefer."
        )
        self.style_count_box.currentIndexChanged.connect(self._style_count_changed)
        row.addWidget(self.style_count_box)

        self.style_layout_box = QComboBox()
        self.style_layout_box.addItem("Side by side", "side")
        self.style_layout_box.addItem("Wipe", "wipe")
        self.style_layout_box.setToolTip(
            "Side by side shows whole frames, which is what you want when "
            "choosing between them.\n\n"
            "Wipe slides one over the other in place, which is better for "
            "spotting a small change than for judging the picture. It uses the "
            "first two panes.\n\n"
            "Either way every pane shares one zoom and one pan, so they cannot "
            "drift apart."
        )
        self.style_layout_box.currentIndexChanged.connect(self._style_layout_changed)
        row.addWidget(self.style_layout_box)

        self._style_buttons: list[QPushButton] = []
        for index, name in enumerate(NR_STYLES):
            button = QPushButton(f"Keep {name}")
            button.setObjectName("secondary")
            button.setToolTip(
                f"Make the {name} version the result, so Save exports it. "
                "Also sets the style in the sidebar, so the next conversion "
                "and any folder batch use it too."
            )
            button.clicked.connect(lambda _=False, i=index: self._adopt_style(i))
            self._style_buttons.append(button)
            row.addWidget(button)

        outer.addLayout(row)
        panel.setVisible(False)
        self._apply_pane_defaults(2)
        return panel

    # -- style comparison ----------------------------------------------------

    def _style_signature(self) -> tuple:
        """Everything a comparison depends on except the style itself.

        Held so a stale comparison is detected rather than shown. Two images
        that differ in strengths as well as style answer no question at all,
        and the difference between the styles is subtle enough that nobody
        would spot the contamination by eye.
        """
        neural = self.settings.neural
        return (
            str(self.image_path),
            neural.intensity, neural.skin, neural.local_tone, neural.structure,
            neural.preset, neural.paper_white,
            neural.transfer_strength, neural.color_strength,
            self.settings.evaluation.frames,
            self.settings.evaluation.max_edge,
            self.settings.evaluation.jitter,
            self.settings.depth.contrast,
            self.settings.depth.model_id,
            self.settings.depth.input_size,
            self.settings.depth.tiled,
        )

    def _styles_current(self) -> bool:
        return (
            len(self.style_results) == len(NR_STYLES)
            and self._style_signature_used == self._style_signature()
        )

    def _start_style_comparison(self) -> None:
        if self.image_path is None or self._thread is not None:
            return
        self.style_results = {}
        self._style_signature_used = self._style_signature()
        self.convert_button.setEnabled(False)
        self.view_styles.setEnabled(False)

        self._begin_progress()

        self._thread = QThread(self)
        self._style_worker = StyleWorker(
            self.image_path, self.settings, self.engine,
            list(range(len(NR_STYLES))), self.prepared,
        )
        self._style_worker.moveToThread(self._thread)
        self._thread.started.connect(self._style_worker.run)
        self._style_worker.progress.connect(self._report_progress)
        self._style_worker.finished.connect(self._styles_ready)
        self._style_worker.failed.connect(self._styles_failed)
        self._thread.start()

    def _styles_ready(self, results: dict) -> None:
        self.style_results = results
        self._style_teardown()
        # Adopt nothing yet. Which one wins is the user's call, and silently
        # keeping the last one converted would answer it for them.
        self.show_view("styles")
        self.statusBar().showMessage(
            "Compare styles - drag the divider, scroll to zoom, right-drag to "
            "pan. Both halves move together."
        )

    def _styles_failed(self, message: str) -> None:
        self.style_results = {}
        self._style_signature_used = None
        self._style_teardown()
        QMessageBox.warning(self, "Could not compare styles", message)
        self.statusBar().showMessage("Style comparison failed")

    def _style_teardown(self) -> None:
        self._end_progress()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._style_worker = None
        self.convert_button.setEnabled(self.image_path is not None)
        self.view_styles.setEnabled(self.image_path is not None)

    def _style_view(self):
        """Whichever comparison widget the layout box is set to."""
        side = self.style_layout_box.currentData() == "side"
        return self.side_by_side if side else self.wipe

    def _pane_count(self) -> int:
        """Panes on screen. The wipe has two sides whatever the box says."""
        if self.style_layout_box.currentData() != "side":
            return 2
        return int(self.style_count_box.currentData() or 2)

    def _apply_pane_defaults(self, count: int) -> None:
        """Sensible starting selections for a given number of panes.

        Two panes is the style question - Natural against Cinematic. Three is
        the "is it helping" question, and that reads best with the source
        first, so the eye moves from unprocessed to most processed.
        """
        defaults = [0, 1] if count == 2 else [-1, 0, 1]
        for box, value in zip(self.style_choices, defaults):
            index = box.findData(value)
            if index >= 0:
                box.setCurrentIndex(index)

    def _style_count_changed(self) -> None:
        # Changing the count re-seeds the selections: the good default for two
        # panes is not the good default for three, and a user who has picked
        # their own will pick again rather than puzzle over a stale third pane.
        self._apply_pane_defaults(int(self.style_count_box.currentData() or 2))
        if self._view == "styles":
            self.show_view("styles")

    def _style_selection_changed(self) -> None:
        if self._view == "styles":
            self._render_styles()

    def _style_layout_changed(self) -> None:
        if self._view == "styles":
            self.show_view("styles")

    def _pane_source(self, index: int) -> tuple[str, object]:
        """(label, result) for one pane. `result` is None for the source."""
        box = self.style_choices[index]
        style = box.currentData()
        if style is None or style < 0:
            return "Original", None
        return NR_STYLES[style], self.style_results.get(style)

    def _render_styles(self) -> None:
        """Draw the chosen panes, graded identically, into the chosen layout."""
        if len(self.style_results) < len(NR_STYLES):
            return
        count = self._pane_count()
        for index, box in enumerate(self.style_choices):
            box.setVisible(index < count)

        images: list[np.ndarray] = []
        labels: list[str] = []
        for index in range(count):
            label, result = self._pane_source(index)
            if result is None:
                # The source, graded like everything else. Comparing a graded
                # result against an ungraded source would show the grade rather
                # than the neural pass, which is the opposite of the point.
                any_result = next(iter(self.style_results.values()))
                images.append(self._graded_preview(any_result, original=True))
            else:
                images.append(self._graded_preview(result))
            labels.append(label)

        view = self._style_view()
        if view is self.side_by_side:
            view.set_panes_u8(images)
        else:
            view.set_images_u8(images[0], images[1])
        view.set_labels(*labels)

        how = (
            "Scroll to zoom, right-drag to pan - every pane moves together."
            if view is self.side_by_side
            else "Drag the divider; scroll to zoom, right-drag to pan."
        )
        self.style_caption.setText(f"Same image, same settings, same grade. {how}")

    def _graded_preview(
        self, result: pipeline.Result, original: bool = False
    ) -> np.ndarray:
        """One image as 8-bit, with the current grade, at preview size."""
        if original:
            small = _downscale_for_preview(result.original)
            linear = contract.srgb_to_linear(np.clip(small, 0.0, 1.0).astype(np.float32))
            return grade.apply_preview(linear, self.settings.grade)
        if result.hdr and result.enhanced_linear is not None:
            linear = _downscale_for_preview(result.enhanced_linear)
            graded = hdr_mod.tonemap(
                grade.apply_linear(linear, self.settings.grade), result.white
            )
            return (np.clip(graded, 0.0, 1.0) * 255.0).astype(np.uint8)
        small = _downscale_for_preview(result.enhanced)
        linear = contract.srgb_to_linear(np.clip(small, 0.0, 1.0).astype(np.float32))
        return grade.apply_preview(linear, self.settings.grade)

    def _adopt_style(self, index: int) -> None:
        """Make one of the compared styles the result, and the live setting."""
        result = self.style_results.get(index)
        if result is None:
            return
        self.settings.neural.style = index
        self.style_box.setCurrentIndex(index)
        self.settings.save(paths.settings_path())
        self._succeeded(result)
        self.statusBar().showMessage(
            f"{NR_STYLES[index]} kept. Save result... exports this one."
        )

    # -- colour grade --------------------------------------------------------

    def _grade_setter(self, field: str):
        def apply_value(value: float) -> None:
            setattr(self.settings.grade, field, value)
            # Coalesced: a drag emits a change per pixel of travel and each
            # redraw is tens of milliseconds of numpy over the preview.
            self._grade_timer.start()

        return apply_value

    def _render_current(self) -> None:
        """Re-draw whichever comparison is on screen after a grade change.

        The style comparison is graded too, and identically on both halves -
        the point is to isolate the style, so a grade that landed on only one
        side would defeat the whole view.
        """
        if self._view == "styles":
            self._render_styles()
        else:
            self._render_result()

    def _grade_toggled(self, shown: bool) -> None:
        self.grade_panel.setVisible(shown)
        # Grading applies to the style comparison as well, so being on that
        # view is not a reason to be dragged off it.
        if shown and self.result is not None and self._view != "styles":
            self.show_view("result")

    def _reset_grade(self) -> None:
        from .grade import GradeSettings

        self.settings.grade = GradeSettings()
        for field, row in self._grade_rows.items():
            row.set_value(getattr(self.settings.grade, field))
        self._render_result()

    def _render_difference(self) -> None:
        """Show what the neural pass changed, amplified to fill the range.

        This exists because "it doesn't work, the image is identical" is the
        most common report, and it is usually wrong: at the default strengths,
        on content that is already photographic, a real change of a few percent
        is genuinely invisible side by side. Amplifying it settles the question
        in one click, and the caption gives the number so nobody has to squint.

        Deliberately measured against the ungraded result. The colour grade is a
        separate, later step and folding it in here would flatter the neural
        pass with changes it did not make.
        """
        if self._preview_after is None or self._preview_before is None:
            return
        difference = np.abs(
            self._preview_after.astype(np.float32) - self._preview_before.astype(np.float32)
        )
        mean = float(difference.mean())
        peak = float(difference.max())
        gain = 1.0 / max(peak, 1e-4)
        self.diff_view.set_image(
            np.clip(difference * gain, 0.0, 1.0),
            f"Difference — mean {mean:.4f}, peak {peak:.4f}, amplified {gain:.0f}x"
            + ("   (nothing changed)" if peak < 1e-4 else ""),
        )

    def _render_result(self) -> None:
        """Redraw the comparison with the current grade applied.

        Grades a preview-sized copy rather than the full image: at 8K the full
        array is 33 megapixels and a slider drag would stutter. The saved file
        is graded at full resolution, so the preview is a fast stand-in and
        never the thing that gets written.
        """
        if self.result is None or self._preview_after_linear is None:
            return
        if self.result.hdr:
            # Grade the scene-referred data and tone map afterwards, which is
            # the order the export uses. Grading the tone mapped copy instead
            # would preview an exposure change behaving quite differently from
            # the one that ends up in the file.
            graded = hdr_mod.tonemap(
                grade.apply_linear(self._preview_after_linear, self.settings.grade),
                self.result.white,
            )
            graded = (np.clip(graded, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            graded = grade.apply_preview(self._preview_after_linear, self.settings.grade)
        self.wipe.set_images_u8(self._preview_before_u8, graded)

    def show_view(self, which: str) -> None:
        """Switch the canvas, falling back when the requested view has no data."""
        if which in ("result", "difference") and self.result is None:
            which = "depth" if self.prepared is not None else "photo"
        if which == "depth" and self.prepared is None:
            which = "photo"
        if which == "styles":
            # The only view that has to be computed before it can be shown.
            # If the comparison is missing or was made at other settings, this
            # starts it and comes back when it finishes.
            if not self._styles_current():
                self._start_style_comparison()
                return
            self._render_styles()
            self.stack.setCurrentWidget(self._style_view())
            self._set_view_state(which)
            return

        # Leaving the styles view puts the wipe back to source-versus-result.
        self.wipe.set_labels()
        self.side_by_side.clear()

        if which == "difference":
            self._render_difference()
            self.stack.setCurrentWidget(self.diff_view)
        elif which == "result":
            self._render_result()
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

        self._set_view_state(which)
        if which == "depth":
            self._render_depth_preview()

    def _set_view_state(self, which: str) -> None:
        self._view = which
        self.view_photo.setChecked(which == "photo")
        self.view_depth.setChecked(which == "depth")
        self.view_result.setChecked(which == "result")
        self.view_diff.setChecked(which == "difference")
        self.view_styles.setChecked(which == "styles")
        self.view_diff.setEnabled(self.result is not None)
        self.view_depth.setEnabled(self.prepared is not None)
        self.view_result.setEnabled(self.result is not None)
        self.view_photo.setEnabled(self.prepared is not None)
        self.view_styles.setEnabled(self.image_path is not None and self._thread is None)
        self.style_panel.setVisible(which == "styles")

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

    def _actions(self) -> QWidget:
        """The buttons, pinned below the sidebar rather than inside it.

        These used to scroll with the settings, which meant Convert - the
        one control used on every single image - could be off screen. That
        is the same root cause as the sliders being clipped at 150%
        scaling: 1012 px of sidebar in a 667 px viewport. Pinning the
        actions leaves only settings to scroll and removes the case
        entirely rather than making it less likely."""
        panel = QWidget()
        panel.setFixedWidth(SIDEBAR_WIDTH)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(6)

        # A rule, so the block reads as pinned rather than as the end of a list
        # that happens to have stopped scrolling.
        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.HLine)
        rule.setFrameShadow(QFrame.Shadow.Plain)
        rule.setFixedHeight(1)
        rule.setStyleSheet("background: #2a2e37; border: none;")
        layout.addWidget(rule)
        layout.addSpacing(2)

        self.open_button = QPushButton("Open image…")
        self.open_button.setObjectName("secondary")
        self.open_button.clicked.connect(self.browse)

        self.convert_button = QPushButton("Convert")
        self.convert_button.setEnabled(False)
        self.convert_button.clicked.connect(self.convert)

        self.save_button = QPushButton("Save result…")
        self.save_button.setObjectName("secondary")
        self.save_button.setToolTip("Asks for an export size first. Defaults to native.")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save)

        self.batch_button = QPushButton("Apply to folder…")
        self.batch_button.setObjectName("secondary")
        self.batch_button.setToolTip(
            "Run a whole folder with the settings above.\n\n"
            "Tune them on one image first - whatever is in this sidebar is what "
            "every image in the folder gets, colour grade included.\n\n"
            "For an animation use the Image sequence tab instead: that keeps "
            "frames consistent with each other and can take your renderer's "
            "depth pass."
        )
        self.batch_button.clicked.connect(self.open_batch)

        self.feedback_button = QPushButton("Use result as input")
        self.feedback_button.setObjectName("secondary")
        self.feedback_button.setEnabled(False)
        self.feedback_button.setToolTip(
            "Feed the result back in for another pass.\n\n"
            "The colour grade is baked in, and depth is re-estimated from the "
            "new image.\n\n"
            "Worth knowing: the pass amplifies what it already did, so a second "
            "run compounds artefacts as readily as detail. It is genuinely "
            "useful on renders that started out flat, and it is the quickest "
            "way to make a portrait look plastic. Lower the strengths for the "
            "second pass rather than repeating the first."
        )
        self.feedback_button.clicked.connect(self.use_result_as_input)

        self.find_button = QPushButton("Find my DLSS files…")
        self.find_button.setObjectName("secondary")
        self.find_button.setToolTip(
            "Search your Steam libraries, Downloads and Documents for the four "
            "files, and copy them in.\n\n"
            "Nothing is downloaded - this only looks at files already on your "
            "machine. If DLSS 5 works in a game for you, that game's folder is "
            "what it is looking for."
        )
        self.find_button.clicked.connect(self.open_find_files)

        diagnose = QPushButton("Check runtime")
        diagnose.setObjectName("secondary")
        diagnose.clicked.connect(self.diagnose)

        help_button = QPushButton("Help")
        help_button.setObjectName("secondary")
        help_button.setToolTip(
            f"Opens the guide in your browser:\n{WIKI_URL}\n\n"
            "Every issue anyone has reported, with what actually caused it."
        )
        help_button.clicked.connect(lambda: open_help())

        # One column, not two. Two-across was tried and measured: at a 320 px
        # sidebar a half cell is 156 px, and "Check runtime" needs 188, "Apply
        # to folder..." 224, "Find my DLSS files..." 260. Fitting them means
        # cutting to "Runtime", "To folder...", "Find files..." - and that last
        # one exists to say DLSS files, which is the whole reason someone
        # stuck on missing files spots it. Costing that to reclaim 72 px is a
        # bad trade, and pinning the block already took 252 px out of the
        # scrolling region, which was the actual crowding.
        layout.addWidget(self.open_button)
        layout.addWidget(self.convert_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.feedback_button)
        layout.addWidget(self.batch_button)
        layout.addWidget(self.find_button)

        # The one pair that fits: Help is a single short word.
        tools = QHBoxLayout()
        tools.setSpacing(8)
        tools.addWidget(diagnose, 1)
        tools.addWidget(help_button)
        layout.addLayout(tools)
        return panel

    def _sidebar(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(SIDEBAR_WIDTH)
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
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Max size"))
        self.max_edge = QComboBox()
        self.max_edge.setEditable(True)
        for choice in MAX_EDGE_CHOICES:
            self.max_edge.addItem(f"{choice} px", choice)
        current = self.settings.evaluation.max_edge
        index = self.max_edge.findData(current)
        if index >= 0:
            self.max_edge.setCurrentIndex(index)
        else:
            self.max_edge.setCurrentText(str(current))
        self.max_edge.setToolTip(
            "Longest edge sent to DLSS. Anything bigger is downscaled first, so "
            "this is the resolution the neural pass actually runs at.\n\n"
            "It is not an export setting. To write a file at a different size, "
            "use Save result… and pick one there — but detail comes from this "
            "number, not from that one.\n\n"
            "Raise it to keep the full resolution of large renders. Verified to "
            "8K here (25 s, 5.3 GB of VRAM) with the neural pass confirmed "
            "running at full size. 4K is the default because it is the size "
            "NVIDIA validated, not a limit of the tool."
        )
        self.max_edge.currentTextChanged.connect(self._max_edge_changed)
        size_row.addStretch(1)
        size_row.addWidget(self.max_edge)
        eval_layout.addLayout(size_row)

        self.jitter = QCheckBox("Sub-pixel jitter")
        self.jitter.setChecked(self.settings.evaluation.jitter)
        self.jitter.toggled.connect(lambda v: setattr(self.settings.evaluation, "jitter", v))
        eval_layout.addWidget(self.jitter)
        layout.addWidget(evaluation)
        layout.addStretch(1)
        return panel


    # -- sequence page -------------------------------------------------------

    def _wire_sequence_page(self) -> None:
        page = self.sequence_page
        page.pick_frames.clicked.connect(self._pick_sequence)
        page.pick_depth.clicked.connect(self._pick_depth_sequence)
        page.clear_depth.clicked.connect(self._clear_depth_sequence)
        page.pick_output.clicked.connect(self._pick_sequence_output)
        page.start.clicked.connect(self._start_sequence)
        page.stop.clicked.connect(self._stop_sequence)
        page.output_label.setText(f"Output: {paths.output_dir()}")
        self._sequence_output = paths.output_dir()

    def _pick_sequence(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Choose the first frame of the sequence",
            str(self.sequence_page.frames[0].parent) if self.sequence_page.frames else "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.exr *.hdr *.webp *.bmp *.jxr *.wdp *.hdp)",
        )
        if not chosen:
            return
        frames = sequence.find_sequence(Path(chosen))
        self.sequence_page.frames = frames
        self.sequence_page.frames_label.setText(sequence.describe(frames))
        self.sequence_page.start.setEnabled(bool(frames))
        # Show the frame they picked, so it is obvious which sequence this is.
        try:
            first = contract.fit_to_budget(contract.load_image(Path(chosen)), _PREVIEW_EDGE)
            self.sequence_page.preview.set_image(first, Path(chosen).name)
        except Exception:  # noqa: BLE001 - a preview is not worth failing over
            pass
        self._check_depth_pairing()

    def _pick_depth_sequence(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Choose the first depth frame",
            "",
            "Depth maps (*.png *.tif *.tiff *.exr *.jpg *.jpeg)",
        )
        if not chosen:
            return
        depth = sequence.find_sequence(Path(chosen))
        self.sequence_page.depth_frames = depth
        self.sequence_page.invert_depth.setVisible(True)
        self.sequence_page.clear_depth.setVisible(True)
        self.sequence_page.depth_label.setText(f"Depth from renderer: {sequence.describe(depth)}")
        self._check_depth_pairing()

    def _clear_depth_sequence(self) -> None:
        self.sequence_page.depth_frames = []
        self.sequence_page.invert_depth.setVisible(False)
        self.sequence_page.clear_depth.setVisible(False)
        self.sequence_page.depth_label.setText(
            "Depth: estimated per frame (Depth Anything)"
        )

    def _check_depth_pairing(self) -> None:
        """Warn about a count mismatch now, not 200 frames into a render."""
        page = self.sequence_page
        if not page.frames or not page.depth_frames:
            return
        if len(page.frames) != len(page.depth_frames):
            page.depth_label.setText(
                f"Mismatch: {len(page.frames)} frames but {len(page.depth_frames)} "
                "depth frames. They must pair one to one."
            )
            page.start.setEnabled(False)
        else:
            page.start.setEnabled(True)

    def _pick_sequence_output(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Where should the frames go?", str(self._sequence_output)
        )
        if chosen:
            self._sequence_output = Path(chosen)
            self.sequence_page.output_label.setText(f"Output: {chosen}")

    def _start_sequence(self) -> None:
        page = self.sequence_page
        if not page.frames or self._seq_thread is not None:
            return
        page.outputs = []
        page.start.setEnabled(False)
        page.stop.setVisible(True)
        page.bar.setVisible(True)
        page.bar.setRange(0, len(page.frames))
        page.bar.setValue(0)

        self._seq_thread = QThread(self)
        self._seq_worker = SequenceWorker(
            page.frames,
            page.depth_frames or None,
            page.invert_depth.isChecked(),
            self.settings,
            self.engine,
            self._sequence_output,
        )
        self._seq_worker.moveToThread(self._seq_thread)
        self._seq_thread.started.connect(self._seq_worker.run)
        self._seq_worker.progress.connect(self.statusBar().showMessage)
        self._seq_worker.frame_done.connect(self._sequence_frame_done)
        self._seq_worker.finished.connect(self._sequence_finished)
        self._seq_worker.failed.connect(self._sequence_failed)
        self._seq_thread.start()

    def _stop_sequence(self) -> None:
        if self._seq_worker is not None:
            self._seq_worker.stop()
            self.statusBar().showMessage("Stopping after this frame…")

    def _sequence_frame_done(self, frame: pipeline.SequenceFrame) -> None:
        page = self.sequence_page
        page.outputs.append(frame.output)
        page.bar.setValue(frame.index + 1)
        page.bar.setFormat(f"%v of %m — {frame.source.name}")
        page.preview.set_image(_downscale_for_preview(frame.image), frame.output.name)

    def _sequence_teardown(self) -> None:
        if self._seq_thread is not None:
            self._seq_thread.quit()
            self._seq_thread.wait(10000)
            self._seq_thread = None
        self._seq_worker = None
        page = self.sequence_page
        page.stop.setVisible(False)
        page.start.setEnabled(bool(page.frames))

    def _sequence_finished(self, done: int) -> None:
        page = self.sequence_page
        self._sequence_teardown()
        message = f"{done} frame(s) written to {self._sequence_output}"
        if page.write_video.isChecked() and page.outputs:
            try:
                self.statusBar().showMessage("Encoding MP4…")
                video = pipeline.write_video(
                    page.outputs,
                    self._sequence_output / f"{page.outputs[0].stem}_sequence.mp4",
                    float(page.fps.value()),
                )
                message += f" — {video.name}"
            except Exception as error:  # noqa: BLE001 - the frames are safe either way
                QMessageBox.warning(
                    self,
                    "Frames written, MP4 failed",
                    f"{error}\n\nThe PNG frames are in the output folder and can "
                    "be encoded with anything.",
                )
        self.statusBar().showMessage(message)

    def _sequence_failed(self, message: str) -> None:
        self._sequence_teardown()
        QMessageBox.warning(self, "Sequence failed", message)
        self.statusBar().showMessage("Sequence failed")

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
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.exr *.hdr *.jxr *.wdp *.hdp)",
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

    def refresh_runtime_status(self) -> None:
        try:
            message = runtime.describe(runtime.detect(self.settings.runtime_dir))
        except Exception as error:  # noqa: BLE001 - status must never raise
            message = f"Could not check the DLSS runtime: {error}"
        self.statusBar().showMessage(message)

    def open_find_files(self) -> None:
        existing = getattr(self, "_find_dialog", None)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return
        self._find_dialog = FindFilesDialog(self)
        self._find_dialog.show()
        # Straight into a search: the user clicked a button that says find.
        QTimer.singleShot(0, self._find_dialog.start_scan)

    def open_batch(self) -> None:
        """The folder dialog, created on demand and remembered while open."""
        existing = getattr(self, "_batch_dialog", None)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return
        self._batch_dialog = BatchDialog(self)
        self._batch_dialog.show()

    def use_result_as_input(self) -> None:
        """Feed the result back in, for a second pass.

        Written to disk rather than handed over in memory, so the rest of the
        app keeps working in paths and the intermediate is something the user
        can actually find and inspect. 16-bit, because a second pass reads
        gradients the first one just widened and 8 bits would band them.

        The grade is baked in deliberately: it is what is on screen, and
        grading after pass one then again after pass two is the natural way to
        work.
        """
        if self.result is None:
            return
        stem = (self.image_path or Path("image")).stem
        match = re.search(r"^(?P<base>.*)_pass(?P<n>\d+)$", stem)
        if match:
            stem = match.group("base")
            passes = int(match.group("n")) + 1
        else:
            passes = 2

        # An HDR result goes round as HDR. Writing the intermediate to PNG
        # would tone map it, and pass two would then be working on an SDR
        # image while the app still called the run HDR.
        is_hdr = self.result.hdr
        target = paths.scratch_dir() / f"{stem}_pass{passes}.{'jxr' if is_hdr else 'png'}"
        try:
            if is_hdr:
                assert self.result.enhanced_linear is not None
                payload = grade.apply_linear(self.result.enhanced_linear, self.settings.grade)
            else:
                payload = grade.apply(self.result.enhanced, self.settings.grade)
            pipeline.save_image(payload, target, linear=is_hdr)
        except (OSError, RuntimeError) as error:
            QMessageBox.warning(self, "Could not start another pass", str(error))
            return
        self.open_image(target)
        self.statusBar().showMessage(
            f"Pass {passes}: the previous result is now the input. "
            "Consider lowering the strengths."
        )

    def open_image(self, path: Path) -> None:
        self.image_path = path
        self.result = None
        self.prepared = None
        self.save_button.setEnabled(False)
        self.feedback_button.setEnabled(False)
        self.convert_button.setEnabled(True)
        self.view_depth.setEnabled(False)
        self.view_result.setEnabled(False)
        self._preview_before = self._preview_after = None
        self._preview_after_linear = None
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

    def _max_edge_changed(self, text: str) -> None:
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits:
            return
        value = max(64, int(digits))
        if value == self.settings.evaluation.max_edge:
            return
        self.settings.evaluation.max_edge = value
        # Depth is estimated on the fitted image, so a different size means the
        # cached depth is the wrong shape and has to be redone.
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
            # Deliberately not on preview re-runs: those fire on a slider drag
            # while the user is looking at the result, and yanking the view
            # back to the source every time would be worse than no feedback.
            self._begin_progress()

        self._thread = QThread(self)
        self._worker = Worker(self.image_path, self.settings, self.engine, self.prepared)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._report_progress)
        self._worker.finished.connect(self._succeeded)
        self._worker.failed.connect(self._failed)
        self._thread.start()

    def _begin_progress(self) -> None:
        """Show the source image, drained of colour, ready to refill."""
        if self.prepared is None:
            return
        self.show_view("photo")
        self.depth_view.set_progress(0.0)

    def _end_progress(self) -> None:
        self.depth_view.set_progress(None)

    def _report_progress(self, message: str) -> None:
        """Status text, and the colour sweep if this message carries a count.

        Parsed out of the message rather than plumbed through as a number: the
        progress callback is a string all the way down from the pipeline, and
        threading a second numeric channel through three call sites to drive a
        visual flourish is not worth it. If the wording ever changes the sweep
        simply stops advancing, which is a harmless way to fail.
        """
        self.statusBar().showMessage(message)
        if self.depth_view._progress is None:
            return
        match = _PASS_COUNT.search(message)
        if match:
            done, total = int(match.group(1)), int(match.group(2))
            if total > 0:
                self.depth_view.set_progress(done / total)

    def _teardown(self) -> None:
        self._end_progress()
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
        self._preview_before = _downscale_for_preview(result.original)
        self._preview_before_u8 = np.clip(self._preview_before, 0.0, 1.0).astype(np.float32)
        self._preview_before_u8 = (self._preview_before_u8 * 255.0).astype(np.uint8)
        self._preview_after = _downscale_for_preview(result.enhanced)
        if result.hdr:
            # Keep the preview scene-referred for an HDR result, so the grade
            # sliders act on the same numbers the export will.
            assert result.enhanced_linear is not None
            self._preview_after_linear = _downscale_for_preview(result.enhanced_linear)
        else:
            self._preview_after_linear = contract.srgb_to_linear(
                np.clip(self._preview_after, 0.0, 1.0).astype(np.float32)
            )
        self._render_result()
        self.save_button.setEnabled(True)
        self.feedback_button.setEnabled(True)
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
        # The moment someone actually needs the guide is the moment something
        # failed, so the way there is on the failure itself rather than only in
        # the sidebar they are no longer looking at.
        box = QMessageBox(QMessageBox.Icon.Warning, "Conversion failed", message, parent=self)
        box.addButton(QMessageBox.StandardButton.Ok)
        guide = box.addButton("Troubleshooting", QMessageBox.ButtonRole.HelpRole)
        box.exec()
        if box.clickedButton() is guide:
            open_help("Troubleshooting")
        self.statusBar().showMessage("Conversion failed")
        self._teardown()

    def save(self) -> None:
        if self.result is None:
            return
        height, width = self.result.enhanced.shape[:2]
        sizer = ExportDialog(self, (width, height))
        if sizer.exec() != QDialog.Accepted:
            return
        target = sizer.chosen()

        # The app's own output folder, not the home directory: a release build
        # ships one, and defaulting anywhere else scatters results.
        suggested = self.settings.last_output_dir or str(paths.output_dir())
        stem = (self.image_path or Path("image")).stem
        # Name the size in the file when it is not the native one, so a folder
        # of exports at three sizes is still readable a week later.
        if target != (width, height):
            stem = f"{stem}_{target[0]}x{target[1]}"
        # An HDR result defaults to a format that can hold it. Offering PNG
        # first would quietly tone map away the entire reason the source was
        # opened as HDR.
        is_hdr = self.result.hdr
        default = str(Path(suggested) / f"{stem}_dlss5.{'jxr' if is_hdr else 'png'}")
        hdr_filters = "JPEG XR (*.jxr);;OpenEXR (*.exr);;"
        sdr_filters = "PNG (*.png);;TIFF (*.tif);;JPEG (*.jpg)"
        filters = (hdr_filters + sdr_filters) if is_hdr else (sdr_filters + ";;" + hdr_filters.rstrip(";"))
        chosen, _ = QFileDialog.getSaveFileName(self, "Save result", default, filters)
        if not chosen:
            return
        try:
            # Full resolution, not the preview the sliders were dragged against.
            # Grade first, then resample: the grade is a per-pixel curve, and
            # running it after an enlargement would apply it to interpolated
            # pixels that the contrast S-curve then pushes apart again.
            if is_hdr:
                assert self.result.enhanced_linear is not None
                image = grade.apply_linear(self.result.enhanced_linear, self.settings.grade)
                image = resample.resize_linear(image, *target)
            else:
                image = grade.apply(self.result.enhanced, self.settings.grade)
                image = resample.resize(image, *target)
            pipeline.save_image(image, chosen, linear=is_hdr)
        except (OSError, ValueError, RuntimeError) as error:
            QMessageBox.warning(self, "Could not save", str(error))
            return
        self.settings.last_output_dir = str(Path(chosen).parent)
        self.settings.save(paths.settings_path())
        kept = Path(chosen).suffix.lower() in hdr_mod.SUFFIXES
        note = "" if not is_hdr else (" (HDR kept)" if kept else " (tone mapped to SDR)")
        self.statusBar().showMessage(f"Saved {Path(chosen).name}{note}")

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
        box = QMessageBox(
            QMessageBox.Icon.Information, "DLSS 5 runtime",
            "\n".join(lines), parent=self,
        )
        box.addButton(QMessageBox.StandardButton.Ok)
        # Only when something is wrong. A clean report needs no reading.
        guide = (
            box.addButton("Troubleshooting", QMessageBox.ButtonRole.HelpRole)
            if status.problems
            else None
        )
        box.exec()
        if guide is not None and box.clickedButton() is guide:
            open_help("Troubleshooting")

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

        if self._seq_worker is not None:
            self._seq_worker.stop()
        # A style comparison is two conversions; without this it would start
        # the second one after the window has gone.
        if self._style_worker is not None:
            self._style_worker.stop()
        for thread in (
            self._thread,
            self._depth_thread,
            self._download_thread,
            self._seq_thread,
        ):
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
        self._seq_thread = None
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
