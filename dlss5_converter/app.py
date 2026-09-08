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
from PySide6.QtGui import QColor, QDesktopServices, QKeySequence, QPalette, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
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
    detail,
    discovery,
    effects,
    evaluator,
    grade,
    onboarding,
    paths,
    pipeline,
    resample,
    runtime,
    sequence,
    video,
)
from . import hdr as hdr_mod
from .depth_engine import MODELS, DepthEngine
from .onnx_depth import OnnxDepthEngine
from .settings import (
    BOOST_LEVEL_LABELS,
    DETAIL_BOOST_FACTORS,
    MAX_EDGE_CHOICES,
    max_boost_factor,
    NR_COLOR_MAX,
    NR_PAPER_WHITE_MAX,
    ONBOARDING_VERSION,
    NR_PRESETS,
    NR_STRENGTH_MAX,
    NR_STYLES,
    NR_TRANSFER_MAX,
    AppSettings,
    style_slug,
)
from .widgets import (
    FONT_DISPLAY,
    FONT_MONO,
    ChipSliderGroup,
    DownloadDialog,
    GpuGauge,
    HoverBar,
    ModuleCard,
    StageHost,
    TimelineWidget,
    DropZone,
    ImageView,
    SegmentedControl,
    SideBySideView,
    SliderRow,
    WipeView,
    apply_font,
    first_supported,
    stage_placeholder,
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

#: The dark "instrument cockpit" theme, as a set of interchangeable palettes.
#: Each is a full colour world with a resting accent (`signal`) and a warm
#: action accent (`heat`, the Convert hover — "the neural pass firing"), so
#: colour reads as activity rather than decoration. The stylesheet is one
#: template; a palette fills its tokens. Glows come from a drop-shadow in code
#: (QSS has no box-shadow) and the dark canvas from a Fusion dark QPalette.
from string import Template as _Template

#: Curated, not open-ended: five tastes rather than a colour picker, so every
#: choice is one someone designed. Order is the order shown in the menu.
PALETTES: dict[str, dict[str, str]] = {
    "Neural Cyan": dict(
        ground="#0A0E15", panel_hi="#141d2c", panel_lo="#101825", base="#0b1220",
        line="#23304a", line_soft="#1a2436", ink="#E8EEF9", ink_dim="#93A2BC",
        ink_faint="#5C6B85", signal="#37E1FF", signal_deep="#0d8fb8",
        signal_light="#5eeaff", heat="#FF7A3C", heat_deep="#d9531f",
        heat_light="#ffd0a8", on_accent="#04121a",
    ),
    "Ember": dict(
        ground="#100C09", panel_hi="#20180F", panel_lo="#17110B", base="#140F0A",
        line="#3B2C1E", line_soft="#241A12", ink="#F6ECE2", ink_dim="#B79E88",
        ink_faint="#7C6857", signal="#FF9838", signal_deep="#C25E13",
        signal_light="#FFB566", heat="#FF4E67", heat_deep="#C21F38",
        heat_light="#FF9DAB", on_accent="#1A0D02",
    ),
    "Violet Flux": dict(
        ground="#0C0A17", panel_hi="#191529", panel_lo="#130F22", base="#100C1E",
        line="#2E2650", line_soft="#1C1636", ink="#ECE8F9", ink_dim="#A79BCC",
        ink_faint="#675C85", signal="#A96BFF", signal_deep="#6A2FD8",
        signal_light="#C295FF", heat="#37E1FF", heat_deep="#0d8fb8",
        heat_light="#8EF0FF", on_accent="#0D0420",
    ),
    "Emerald": dict(
        ground="#08120E", panel_hi="#10201A", panel_lo="#0C1913", base="#0A1611",
        line="#1E4436", line_soft="#142B22", ink="#E4F5EC", ink_dim="#8FBBA6",
        ink_faint="#567A68", signal="#35E0A1", signal_deep="#12A56A",
        signal_light="#6FF0C1", heat="#FFC24B", heat_deep="#D99320",
        heat_light="#FFE0A0", on_accent="#04140C",
    ),
    "Slate Mono": dict(
        ground="#0D1017", panel_hi="#171C26", panel_lo="#12161F", base="#0F131B",
        line="#2A3242", line_soft="#1D2330", ink="#E6EAF1", ink_dim="#97A0B2",
        ink_faint="#5E6675", signal="#8FB4E0", signal_deep="#567AA8",
        signal_light="#B3D0F0", heat="#E0A96B", heat_deep="#A8763C",
        heat_light="#F0D0A8", on_accent="#0A1420",
    ),
}
DEFAULT_THEME = "Neural Cyan"

_STYLE_TEMPLATE = _Template("""
* { font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif; }
QMainWindow, QDialog { background: $ground; }
QWidget { background: transparent; color: $ink; font-size: 13px; }

QGroupBox {
    border: 1px solid $line_soft; border-radius: 14px; margin-top: 16px;
    padding: 15px 14px 13px 14px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 $panel_hi, stop:1 $panel_lo);
}
QGroupBox::title {
    subcontrol-origin: margin; left: 14px; top: 2px; padding: 0 6px;
    color: $ink_dim; font-weight: 700;
}

QLabel { background: transparent; }
QLabel#hint { color: $ink_faint; }
QLabel#linkSep { color: $line; }
QLabel#onboardingEyebrow { color: $signal; }
QLabel#onboardingTitle { color: $ink; font-weight: 700; }
QListWidget#onboardingSources {
    background: $base; border: 1px solid $line_soft; border-radius: 9px;
    padding: 6px; color: $ink_dim;
}
QListWidget#onboardingSources::item { padding: 5px; }
QListWidget#onboardingSources::item:selected { background: $panel_hi; color: $ink; }

QFrame#dropZone {
    border: 2px dashed $line; border-radius: 16px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 $panel_lo, stop:1 $ground);
}
QFrame#dropZone[hovering="true"] { border-color: $signal; background: $panel_hi; }

QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 $signal, stop:1 $signal_deep);
    color: $on_accent; border: none; border-radius: 9px; padding: 9px 16px; font-weight: 700;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 $signal_light, stop:1 $signal_deep);
}
QPushButton:disabled { background: $panel_lo; color: $ink_faint; }
QPushButton#secondary {
    background: $panel_lo; color: $ink; border: 1px solid $line; font-weight: 600;
}
QPushButton#secondary:hover { border-color: $signal_deep; color: $ink; background: $panel_hi; }
QPushButton#secondary:checked {
    background: $base; color: $signal; border: 1px solid $signal_deep;
}
QPushButton#secondary:disabled { color: $ink_faint; border-color: $line_soft; background: $panel_lo; }
QPushButton#convert {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 $signal_light, stop:0.5 $signal, stop:1 $signal_deep);
    color: $on_accent; border-radius: 12px; padding: 15px; font-weight: 800; font-size: 15px;
}
QPushButton#convert:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 $heat_light, stop:0.5 $heat, stop:1 $heat_deep);
}
QPushButton#convert:disabled { background: $panel_lo; color: $ink_faint; }
QPushButton#link {
    background: transparent; color: $ink_faint; border: none; padding: 4px 2px; font-weight: 500;
}
QPushButton#link:hover { color: $signal; }
QPushButton#chip {
    background: $base; color: $ink_dim; border: 1px solid $line; border-radius: 9px;
    padding: 7px 13px; font-weight: 500;
}
QPushButton#chip:hover { color: $ink; border-color: $signal_deep; }
QPushButton#chip:checked {
    color: $on_accent; font-weight: 600; border-color: transparent;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 $signal, stop:1 $signal_deep);
}

QSlider:horizontal { min-height: 22px; }
QSlider::groove:horizontal { height: 6px; border-radius: 3px; background: $base; border: 1px solid $line_soft; margin: 0 9px; }
QSlider::sub-page:horizontal {
    height: 6px; border-radius: 3px; margin: 0 9px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 $signal_deep, stop:1 $signal);
}
QSlider::add-page:horizontal { height: 6px; border-radius: 3px; background: $base; margin: 0 9px; }
QSlider::handle:horizontal {
    background: $signal_light; border: 2px solid $signal; width: 15px; height: 15px;
    margin: -6px -9px; border-radius: 9px;
}
QSlider::handle:horizontal:hover { border-color: $signal_light; }

QComboBox { background: $base; color: $ink; border: 1px solid $line; border-radius: 9px; padding: 7px 12px; }
QComboBox:hover { border-color: $signal_deep; }
QComboBox:disabled { color: $ink_faint; border-color: $line_soft; }
QComboBox::drop-down { border: none; width: 22px; }
/* A visible chevron, drawn from borders so it needs no bundled image. Without
   it the dropdown read as a plain box and people typed values in by hand. */
QComboBox::down-arrow {
    width: 0; height: 0; margin-right: 8px;
    border-left: 5px solid transparent; border-right: 5px solid transparent;
    border-top: 6px solid $ink_dim;
}
QComboBox::down-arrow:hover { border-top-color: $signal; }
QComboBox::down-arrow:disabled { border-top-color: $line; }
QComboBox QAbstractItemView {
    background: $panel_lo; color: $ink; border: 1px solid $line;
    selection-background-color: $line; outline: none; padding: 4px;
}
QSpinBox { background: $base; color: $ink; border: 1px solid $line; border-radius: 9px; padding: 6px 8px; }
QSpinBox:hover { border-color: $signal_deep; }

QCheckBox { color: $ink; spacing: 8px; background: transparent; }
QCheckBox::indicator { width: 17px; height: 17px; border-radius: 5px; border: 1px solid $line; background: $base; }
QCheckBox::indicator:hover { border-color: $signal; }
QCheckBox::indicator:checked {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 $signal, stop:1 $signal_deep); border-color: $signal;
}

QProgressBar {
    background: $base; border: 1px solid $line_soft; border-radius: 6px; height: 12px;
    text-align: center; color: $ink_dim; font-size: 11px;
}
QProgressBar::chunk {
    border-radius: 5px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 $signal_deep, stop:1 $signal);
}

QTabWidget::pane { border: none; }
/* Full-width tab band: its own strip under the command bar, with a baseline
   that runs the whole width of the window. */
QTabWidget#mainTabs { background: #0a0f18; }
QTabWidget#mainTabs::pane { border: none; border-top: 1px solid $line_soft; top: -1px; }
QTabBar { background: #0a0f18; qproperty-drawBase: 0; padding: 3px 10px 0 10px; }
QTabBar::tab {
    background: transparent; color: $ink_dim; padding: 11px 16px; margin-right: 2px;
    border: none; border-bottom: 2px solid transparent; font-weight: 500;
}
QTabBar::tab:hover { color: $ink; }
QTabBar::tab:selected { color: $signal; border-bottom: 2px solid $signal; }
QPushButton#tabAction {
    background: transparent; color: $ink_faint; border: none; padding: 8px 18px; font-weight: 500;
}
QPushButton#tabAction:hover { color: $signal; }

QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: $line; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: $signal_deep; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle:horizontal { background: $line; border-radius: 5px; min-width: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

QStatusBar { background: $ground; color: $ink_faint; border-top: 1px solid $line_soft; }
QToolTip { background: $panel_lo; color: $ink; border: 1px solid $line; padding: 6px 8px; }

/* command bar */
QFrame#commandBar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0d1420, stop:1 $ground);
    border: none; border-bottom: 1px solid $line_soft;
}
QLabel#brandGlyph {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 $signal, stop:1 $heat);
    color: $on_accent; border-radius: 8px; font-weight: 800; font-size: 15px;
    min-width: 30px; max-width: 30px; min-height: 30px; max-height: 30px;
}
QLabel#wordmark { color: $ink; font-weight: 800; font-size: 15px; letter-spacing: 2px; }
QLabel#wordmark #accent { color: $signal; }
QLabel#wordmarkSub { color: $ink_faint; font-size: 9px; letter-spacing: 4px; }
QFrame#gauge { background: $base; border: 1px solid $line; border-radius: 11px; }
QLabel#gaugeLab { color: $ink_faint; font-size: 9px; letter-spacing: 2px; }
QLabel#gpuName { color: $ink; font-weight: 700; font-size: 12px; }
QLabel#vramRead { color: $ink_dim; font-size: 10px; }
QProgressBar#vramBar { background: $base; border: 1px solid $line_soft; border-radius: 3px; }
QProgressBar#vramBar::chunk {
    border-radius: 2px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 $signal_deep, stop:1 $signal);
}
QLabel#runtimePill {
    font-size: 10px; letter-spacing: 1px; padding: 7px 13px; border-radius: 9px;
    border: 1px solid #204a3a; background: #0c1a15; color: #54E39B;
}
QLabel#runtimePill[state="setup"] {
    border: 1px solid #4a3a20; background: #1a140c; color: $heat;
}

/* footer */
QFrame#footer { background: #0a0f18; border: none; border-top: 1px solid $line_soft; }
QLabel#footFile { color: $ink_dim; font-size: 11px; }

/* module cards (title inside a divider header, then a padded body) */
QFrame#modCard {
    border: 1px solid $line_soft; border-radius: 14px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 $panel_hi, stop:1 $panel_lo);
}
QFrame#modHead { background: transparent; border: none; border-bottom: 1px solid $line_soft; }
QLabel#modTitle { color: $ink_dim; background: transparent; }
QCheckBox#modTitle { color: $ink_dim; background: transparent; spacing: 9px; }
QLabel#modTag { color: $ink_faint; background: transparent; }

/* preview stage (Video / Sequence): the bordered panel content appears in */
QFrame#stage {
    border: 1px solid $line_soft; border-radius: 14px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 $panel_lo, stop:1 $ground);
}
QLabel#phTitle { color: $ink_dim; background: transparent; }

/* floating view bar (hover-revealed over the image, mockup's stagefoot) */
QFrame#viewBar {
    background: rgba(10, 16, 24, 0.82); border: 1px solid $line; border-radius: 12px;
}
QFrame#viewBarSep { background: $line; border: none; margin: 5px 0; }
QFrame#viewBar QPushButton#viewChip {
    background: transparent; color: $ink_dim; border: none; border-radius: 8px;
    padding: 6px 12px; font-weight: 500; font-size: 12px;
}
QFrame#viewBar QPushButton#viewChip:hover { color: $ink; background: $panel_hi; }
QFrame#viewBar QPushButton#viewChip:checked {
    color: $on_accent; font-weight: 600;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 $signal, stop:1 $signal_deep);
}
QFrame#viewBar QPushButton#viewChip:disabled { color: $ink_faint; background: transparent; }
""")


def _style_for(name: str) -> str:
    """The stylesheet for a palette name, falling back to the default."""
    return _STYLE_TEMPLATE.substitute(PALETTES.get(name, PALETTES[DEFAULT_THEME]))


def _qpalette_for(name: str) -> QPalette:
    """The Fusion QPalette for a palette, for the canvas and native controls."""
    p = PALETTES.get(name, PALETTES[DEFAULT_THEME])
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(p["ground"]))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(p["ink"]))
    pal.setColor(QPalette.ColorRole.Base, QColor(p["base"]))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(p["panel_lo"]))
    pal.setColor(QPalette.ColorRole.Text, QColor(p["ink"]))
    pal.setColor(QPalette.ColorRole.Button, QColor(p["panel_lo"]))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(p["ink"]))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(p["panel_lo"]))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(p["ink"]))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(p["ink_faint"]))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(p["signal"]))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(p["on_accent"]))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText, QPalette.ColorRole.WindowText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(p["ink_faint"]))
    return pal


#: The active stylesheet, reassigned when the theme changes. Dialogs read this
#: at creation, so a switch reaches every window opened afterwards.
STYLE = _style_for(DEFAULT_THEME)


#: Longest edge of the copy the colour sliders are dragged against.
#:
#: Grading a 33-megapixel 8K array on every slider tick stutters badly, and the
#: comparison view never draws more than about 1200 px across anyway, so a
#: larger preview buys nothing visible. Measured on this box: 1600 px costs
#: ~290 ms a redraw, 1200 px about 160 ms, which with the 30 ms coalescing timer
#: is the difference between a slider that drags and one that lurches.
#: The saved file is always graded at full resolution.
_PREVIEW_EDGE = 1200


def _format_duration(seconds: float) -> str:
    """A short, human "time left" — "45s", "2m 30s", "1h 05m".

    Rounded and coarse on purpose: an ETA on a neural conversion is an estimate
    that jitters frame to frame, and second-precision on a ten-minute job reads
    as false confidence. Hours show minutes, minutes show seconds, and under a
    minute is just seconds.
    """
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{int(round(seconds))}s"
    if seconds < 3600:
        minutes, secs = divmod(int(round(seconds)), 60)
        return f"{minutes}m {secs:02d}s"
    hours, rest = divmod(int(round(seconds)), 3600)
    minutes = rest // 60
    return f"{hours}h {minutes:02d}m"


class _EtaTracker:
    """Smoothed seconds-per-frame turned into a time-remaining estimate.

    The first inter-frame gap of a conversion is dominated by the ~3.5 s harness
    start-up — and, with per-frame depth on, the model load — which is nothing
    like steady-state throughput. So the first sample seeds nothing: the moving
    average begins at the second frame, and ``remaining`` returns None until
    there is a real rate to quote rather than a wildly wrong one.
    """

    def __init__(self, smoothing: float = 0.25) -> None:
        self._alpha = smoothing
        self._ema: float | None = None
        self._prev: float | None = None

    def start(self) -> None:
        self._ema = None
        self._prev = None

    def tick(self) -> None:
        now = time.monotonic()
        if self._prev is not None:
            dt = now - self._prev
            self._ema = dt if self._ema is None else self._alpha * dt + (1 - self._alpha) * self._ema
        self._prev = now

    def remaining(self, frames_left: int) -> float | None:
        if self._ema is None or frames_left <= 0:
            return None
        return self._ema * frames_left


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

#: Where the support banner and its button point. The same link the README
#: badge uses, kept here as the single source the app reads.
COFFEE_URL = "https://buymeacoffee.com/criso2hdj"

#: The project's home. Reachable from the actions panel so it stays available
#: after the (one-time, dismissable) support banner is gone for good.
GITHUB_URL = "https://github.com/criso2hd-alt/DLSS5-Image-Converter"


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
    #: A style is about to be converted. The pane showing it starts sweeping.
    started = Signal(int)
    #: One style is finished. Emitted as it lands rather than at the end, so
    #: each pane fills in as soon as it has something to show instead of the
    #: whole comparison appearing at once after both conversions.
    one_done = Signal(int, object)
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

                self.started.emit(style)
                result = pipeline.convert(
                    self._path, settings, self._engine, progress=say, prepared=prepared
                )
                results[style] = result
                self.one_done.emit(style, result)
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


class RuntimeProbeWorker(QObject):
    """Prove DLSS, ReShade, RenoDX and the neural module all load."""

    finished = Signal(bool, str)

    def __init__(self, harness: Path) -> None:
        super().__init__()
        self._harness = harness

    def run(self) -> None:
        try:
            report = evaluator.probe(self._harness)
        except Exception as error:  # noqa: BLE001 - reported in the setup dialog
            self.finished.emit(False, str(error))
            return
        self.finished.emit(onboarding.probe_succeeded(report), report)


class RuntimeProbe(QObject):
    """Run the native DLSS check off the UI thread, watchdog-guarded so it can
    never hang the app.

    The live check launches the harness, which initialises DLSS on the GPU. On
    some driver/runtime combinations that initialisation wedges, and the process
    holds a D3D12 device while it does - which used to freeze the whole window
    with no escape but Task Manager, because the check ran on a modal path the
    user could not leave.

    Here a worker thread runs the probe and a watchdog force-kills the child if
    it overruns. Killing that one process releases the device, exactly as
    force-quitting the app would, except the app survives. Completion, timeout
    and an explicit skip all arrive as one ``done(ok, report)`` on the UI thread,
    and nothing ever calls ``wait()`` there - so the event loop keeps running no
    matter what the runtime does.
    """

    done = Signal(bool, str)

    #: A working probe is a few seconds (NGX + add-on warm-up). Well past that is
    #: a runtime that has hung, so the watchdog steps in rather than waiting.
    TIMEOUT_MS = 30000

    def __init__(self, harness: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._settled = False
        self._thread = QThread()
        self._worker = RuntimeProbeWorker(harness)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_worker_finished)
        # Canonical fire-and-forget teardown: the worker and thread delete
        # themselves once the (possibly killed) child returns, and the UI thread
        # never blocks on it. This object itself is kept alive by its parent -
        # deleting it here, mid-signal, would pull its own connections and
        # watchdog out from under the emission and crash.
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._watchdog = QTimer(self)
        self._watchdog.setSingleShot(True)
        self._watchdog.setInterval(self.TIMEOUT_MS)
        self._watchdog.timeout.connect(self._on_timeout)

    def start(self) -> None:
        self._thread.start()
        self._watchdog.start()

    def skip(self) -> None:
        """Abandon the check now - the dialog's Skip, or app shutdown."""
        self._settle(
            False,
            "Skipped - the live check was stopped. Open Settings - Check "
            "runtime to run it again.",
        )

    def _on_timeout(self) -> None:
        self._settle(
            False,
            "The live check did not finish and was stopped. The DLSS runtime "
            "may be hanging on this GPU - open Settings - Check runtime for the "
            "full report.",
        )

    def _on_worker_finished(self, ok: bool, report: str) -> None:
        self._settle(ok, report)

    def _settle(self, ok: bool, report: str) -> None:
        # First outcome wins - the worker finishing, the watchdog and a skip all
        # route here and only the first is real. cancel_probe() releases the GPU
        # if the child is still up; the worker then returns and the thread tears
        # itself down, with no wait() on the UI thread.
        if self._settled:
            return
        self._settled = True
        self._watchdog.stop()
        evaluator.cancel_probe()
        self.done.emit(ok, report)


def ensure_runtime_ready(parent: QWidget | None = None) -> bool:
    """True if the app can run. There is no first-run download here anymore.

    Depth estimation moved from PyTorch to ONNX Runtime (see onnx_depth.py):
    ONNX Runtime is bundled in the build and the Apache-2.0 Small depth model
    ships inside the app, so the old 2.7 GB PyTorch fetch - the biggest source of
    first-run failures - is gone. A missing onnxruntime is a broken build, not a
    downloadable state, so it is reported rather than fetched.
    """
    try:
        import onnxruntime  # noqa: F401
    except Exception as error:  # noqa: BLE001 - a broken build must say so clearly
        QMessageBox.critical(
            parent,
            "Could not start",
            "ONNX Runtime is missing from this build, so depth estimation cannot "
            "run. This is a packaging problem rather than something to download - "
            f"please reinstall the app.\n\n{error}",
        )
        return False
    return True


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

    runtime_ready = Signal()

    def __init__(self, window: MainWindow, *, onboarding_mode: bool = False) -> None:
        super().__init__(window)
        self._window = window
        self._onboarding_mode = onboarding_mode
        self._candidates: list[discovery.Candidate] = []
        self._extra_roots: list[Path] = []
        self._thread: QThread | None = None
        self._worker: FindFilesWorker | None = None

        self.setWindowTitle("Find my DLSS files")
        self.setModal(onboarding_mode)
        self.setMinimumWidth(720)
        self.setStyleSheet(STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)

        if onboarding_mode:
            eyebrow = QLabel("FIRST RUN  ·  STEP 2 OF 3")
            eyebrow.setObjectName("onboardingEyebrow")
            apply_font(eyebrow, family=FONT_MONO, size=8.5, spacing=2.0, caps=True)
            layout.addWidget(eyebrow)
            title = QLabel("Connect your DLSS 5 files.")
            title.setObjectName("onboardingTitle")
            apply_font(title, family=FONT_DISPLAY, size=19)
            layout.addWidget(title)

        blurb = QLabel(
            "The automatic search checks your Steam libraries, Downloads and "
            "Documents for the four files, then copies the best matching set "
            "into the app.\n\n"
            "Nothing is downloaded — this only looks at files already on your "
            "machine. If DLSS 5 works in a game for you, that game's folder is "
            "what it is looking for."
        )
        blurb.setWordWrap(True)
        blurb.setObjectName("hint")
        layout.addWidget(blurb)

        self.results = QListWidget()
        if onboarding_mode:
            self.results.setObjectName("onboardingSources")
            self.results.addItem("Automatic search will list matching folders here.")
        self.results.setMinimumHeight(130 if onboarding_mode else 180)
        if onboarding_mode:
            self.results.setMaximumHeight(160)
        self.results.currentRowChanged.connect(self._selection_changed)
        layout.addWidget(self.results)

        self.summary = QLabel("")
        self.summary.setObjectName("hint")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        if onboarding_mode:
            self._set_plan_summary({}, searching=True)

        self.status = QLabel("Ready.")
        self.status.setObjectName("hint")
        layout.addWidget(self.status)

        buttons = QHBoxLayout()
        self.close_button = QPushButton("Skip for now" if onboarding_mode else "Close")
        self.close_button.setObjectName("secondary")
        self.add_folder = QPushButton("Search another folder…")
        self.add_folder.setObjectName("secondary")
        self.rescan = QPushButton("Search again")
        self.rescan.setObjectName("secondary")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("secondary")
        self.stop_button.setVisible(False)
        self.copy_button = QPushButton(
            "Use these files & verify" if onboarding_mode else "Copy these files"
        )
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
        if self._onboarding_mode:
            self.results.addItem("Searching likely game and download folders…")
        if self._onboarding_mode:
            self._set_plan_summary({}, searching=True)
        else:
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
        self.results.clear()
        if not candidates:
            if self._onboarding_mode:
                self._set_plan_summary({})
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
            if self._onboarding_mode:
                self._set_plan_summary({})
            else:
                self.summary.setText("")
            return
        if self._onboarding_mode:
            self._set_plan_summary(plan)
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

    def _set_plan_summary(
        self, plan: dict[str, Path], *, searching: bool = False,
    ) -> None:
        """Show the mockup's four-file checklist during first-run setup."""
        lines = []
        for name in discovery.WANTED:
            path = plan.get(name)
            if path is not None:
                size = path.stat().st_size / 1048576
                lines.append(f"FOUND       {name}  ·  {size:.1f} MB")
            elif searching:
                lines.append(f"SEARCHING   {name}")
            else:
                lines.append(f"MISSING     {name}")
        self.summary.setText("FILES REQUIRED\n\n" + "\n".join(lines))

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
            next_step = (
                "The app will now run one live check."
                if self._onboarding_mode
                else "Use Check runtime to confirm everything is working."
            )
            QMessageBox.information(
                self,
                "Files copied",
                f"{len(copied)} file(s) copied into:\n{destination}\n\n"
                + next_step,
            )
        if self._onboarding_mode:
            try:
                ready = runtime.detect(self._window.settings.runtime_dir or None).ready
            except Exception:  # noqa: BLE001 - remain in setup on a bad check
                ready = False
            if ready:
                self.runtime_ready.emit()
                self.accept()

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


#: File dialog filter for choosing videos to queue. Broad on purpose — PyAV
#: reads far more than this, and an over-strict filter only hides a user's file.
_VIDEO_FILTER = (
    "Videos (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.wmv *.flv *.mpg *.mpeg *.ts);;"
    "All files (*)"
)


class VideoQueueDialog(QDialog):
    """Queue several videos and convert them one after another.

    The "batch video convert" / "several videos in a row" ask. A dialog rather
    than a mode on the Video tab, for the same reason the image batch is one:
    the tab is where you tune the look on a single clip — scrub it, mark a
    range, watch a preview — and the queue is the separate act of pointing that
    look at a stack of whole clips and leaving. Codec, effort and depth default
    from the tab; the neural and colour settings come from the shared sidebar,
    exactly as a single conversion reads them.
    """

    def __init__(self, window: MainWindow) -> None:
        super().__init__(window)
        self._window = window
        self.jobs: list[Path] = []
        self.destination = paths.output_dir()
        self._thread: QThread | None = None
        self._worker: VideoQueueWorker | None = None
        self._eta = _EtaTracker()
        self._done = 0
        self._skipped = 0
        self._failed = 0
        self._failures: list[str] = []

        self.setWindowTitle("Convert several videos")
        self.setModal(False)
        self.setMinimumWidth(620)
        self.setStyleSheet(STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)

        blurb = QLabel(
            "Videos are converted back to back with the settings below and the "
            "neural and colour settings in the sidebar. Each clip is converted "
            "whole — to convert only part of one, use Convert video on the tab "
            "instead. You can keep using the app; leave this open while it runs."
        )
        blurb.setWordWrap(True)
        blurb.setObjectName("hint")
        layout.addWidget(blurb)

        self.list = QListWidget()
        self.list.setMinimumHeight(150)
        self.list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        layout.addWidget(self.list, 1)

        list_buttons = QHBoxLayout()
        self.add_button = QPushButton("Add videos…")
        self.remove_button = QPushButton("Remove selected")
        self.remove_button.setObjectName("secondary")
        self.clear_button = QPushButton("Clear")
        self.clear_button.setObjectName("secondary")
        list_buttons.addWidget(self.add_button)
        list_buttons.addWidget(self.remove_button)
        list_buttons.addWidget(self.clear_button)
        list_buttons.addStretch(1)
        layout.addLayout(list_buttons)

        dest_row = QHBoxLayout()
        self.pick_dest = QPushButton("Output folder…")
        self.pick_dest.setObjectName("secondary")
        self.dest_label = QLabel(str(self.destination))
        self.dest_label.setObjectName("hint")
        dest_row.addWidget(self.pick_dest)
        dest_row.addWidget(self.dest_label, 1)
        layout.addLayout(dest_row)

        # Output options, defaulted from the Video tab so the queue matches what
        # the user was just doing on a single clip.
        opts = QHBoxLayout()
        opts.addWidget(QLabel("Output"))
        self.codec_box = QComboBox()
        for codecdef in video.CODECS:
            self.codec_box.addItem(codecdef.label, codecdef.key)
        self.codec_box.setCurrentIndex(max(0, window.video_page.codec_box.currentIndex()))
        opts.addWidget(self.codec_box)
        opts.addSpacing(12)
        opts.addWidget(QLabel("Effort"))
        self.mode_box = QComboBox()
        self.mode_box.addItem("Quick (1 pass)", 1)
        self.mode_box.addItem("Quality (4 passes)", 4)
        self.mode_box.setCurrentIndex(max(0, window.video_page.mode_box.currentIndex()))
        opts.addWidget(self.mode_box)
        opts.addStretch(1)
        layout.addLayout(opts)

        self.estimate_depth = QCheckBox("Estimate depth per frame")
        self.estimate_depth.setChecked(window.video_page.estimate_depth.isChecked())
        self.estimate_depth.setToolTip(
            "Off by default: DLSS does not read depth on a still frame, so this "
            "changes nothing in the output and is by far the slowest step."
        )
        self.skip_existing = QCheckBox("Skip videos already converted")
        self.skip_existing.setChecked(True)
        self.skip_existing.setToolTip(
            "Lets an interrupted queue be restarted without redoing everything."
        )
        layout.addWidget(self.estimate_depth)
        layout.addWidget(self.skip_existing)

        self.preview = ImageView()
        self.preview.setMinimumHeight(150)
        self.preview.setVisible(False)
        layout.addWidget(self.preview, 1)

        # Two bars: the top counts clips, the bottom counts frames within the
        # current clip. A queue of long videos otherwise looks frozen for
        # minutes at a time, because the clip counter only moves once per file.
        self.overall_bar = QProgressBar()
        self.overall_bar.setTextVisible(True)
        self.overall_bar.setVisible(False)
        layout.addWidget(self.overall_bar)
        self.frame_bar = QProgressBar()
        self.frame_bar.setTextVisible(True)
        self.frame_bar.setVisible(False)
        layout.addWidget(self.frame_bar)

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
        self.start_button = QPushButton("Convert queue")
        self.start_button.setEnabled(False)
        buttons.addWidget(self.close_button)
        buttons.addStretch(1)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.start_button)
        layout.addLayout(buttons)

        self.add_button.clicked.connect(self._add)
        self.remove_button.clicked.connect(self._remove_selected)
        self.clear_button.clicked.connect(self._clear)
        self.pick_dest.clicked.connect(self._choose_destination)
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self._stop)
        self.close_button.clicked.connect(self.close)

    # -- building the queue --------------------------------------------------

    def _add(self) -> None:
        chosen, _ = QFileDialog.getOpenFileNames(
            self, "Add videos to the queue", "", _VIDEO_FILTER
        )
        for name in chosen:
            path = Path(name)
            if path not in self.jobs:
                self.jobs.append(path)
                self.list.addItem(QListWidgetItem(path.name))
        self._refresh_controls()

    def _remove_selected(self) -> None:
        for item in self.list.selectedItems():
            row = self.list.row(item)
            self.list.takeItem(row)
            del self.jobs[row]
        self._refresh_controls()

    def _clear(self) -> None:
        self.list.clear()
        self.jobs.clear()
        self._refresh_controls()

    def _refresh_controls(self) -> None:
        running = self._thread is not None
        self.start_button.setEnabled(bool(self.jobs) and not running)
        self.status.setText(
            "" if not self.jobs else f"{len(self.jobs)} video(s) queued."
        )

    def _choose_destination(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Where should the converted videos go?", str(self.destination)
        )
        if chosen:
            self.destination = Path(chosen)
            self.dest_label.setText(chosen)

    # -- running -------------------------------------------------------------

    def _output_for(self, source: Path, suffix: str) -> Path:
        # A distinct name so a queue pointed at its own source folder never
        # overwrites an input, and the H.264 default does not collide with an
        # H.264 source of the same stem.
        style = style_slug(self._window.settings.neural.style)
        return self.destination / f"{source.stem}_dlss5_{style}{suffix}"

    def _start(self) -> None:
        if not self.jobs or self._thread is not None:
            return
        settings = copy.deepcopy(self._window.settings)
        settings.evaluation.frames = int(self.mode_box.currentData() or 1)
        codec_key = self.codec_box.currentData()
        suffix = video.CODECS_BY_KEY[codec_key].suffix
        pairs = [(source, self._output_for(source, suffix)) for source in self.jobs]

        self._done = self._skipped = self._failed = 0
        self._failures = []
        self.overall_bar.setVisible(True)
        self.overall_bar.setRange(0, len(pairs))
        self.overall_bar.setValue(0)
        self.overall_bar.setFormat("%v of %m videos")
        self.frame_bar.setVisible(True)
        self.frame_bar.setValue(0)
        self.start_button.setEnabled(False)
        self.add_button.setEnabled(False)
        self.remove_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.pick_dest.setEnabled(False)
        self.stop_button.setVisible(True)

        self._thread = QThread(self)
        self._worker = VideoQueueWorker(
            pairs, settings, self._window.engine, codec_key,
            self.estimate_depth.isChecked(), self.skip_existing.isChecked(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.status.setText)
        self._worker.video_started.connect(self._video_started)
        self._worker.frame_done.connect(self._frame_done)
        self._worker.video_done.connect(self._video_done)
        self._worker.video_skipped.connect(self._video_skipped)
        self._worker.video_failed.connect(self._video_failed)
        self._worker.finished.connect(self._finished)
        self._thread.start()

    def _stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self.status.setText("Stopping after this frame…")

    def _video_started(self, index: int, total: int, source: object) -> None:
        self._eta.start()
        self.frame_bar.setValue(0)
        self.frame_bar.setMaximum(0)  # unknown until the first frame reports total
        self.frame_bar.setFormat("%p%")
        self.status.setText(f"Video {index + 1} of {total}: {Path(source).name}")

    def _frame_done(self, update: object, index: int, total: int) -> None:
        if getattr(update, "stage", "") != "converting":
            return
        if update.total and self.frame_bar.maximum() != update.total:
            self.frame_bar.setMaximum(update.total)
        self.frame_bar.setValue(update.index)
        self._eta.tick()
        left = self._eta.remaining((update.total - update.index) if update.total else 0)
        clip = f"clip {index + 1}/{total}"
        if left is not None:
            self.frame_bar.setFormat(f"%v of %m frames — ~{_format_duration(left)} left ({clip})")
        else:
            self.frame_bar.setFormat(f"%v of %m frames ({clip})")
        if update.index % 3 == 0 or update.index == update.total:
            self.preview.setVisible(True)
            self.preview.set_image(
                _downscale_for_preview(update.preview),
                f"{clip} — frame {update.index}"
                + (f" of {update.total}" if update.total else ""),
            )

    def _video_done(self, source: object, output: object) -> None:
        self._done += 1
        self.overall_bar.setValue(self._done + self._skipped + self._failed)

    def _video_skipped(self, source: object) -> None:
        self._skipped += 1
        self.overall_bar.setValue(self._done + self._skipped + self._failed)
        self.status.setText(f"{Path(source).name} — already converted, skipped.")

    def _video_failed(self, source: object, error: str) -> None:
        self._failed += 1
        self._failures.append(f"{Path(source).name}: {error}")
        self.overall_bar.setValue(self._done + self._skipped + self._failed)
        # Named, not swallowed — a queue that quietly drops a file is worse than
        # one that tells you which failed and carries on.
        self.status.setText(f"{Path(source).name} failed — {error}")

    def _teardown(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(15000)
            self._thread = None
        self._worker = None
        self.stop_button.setVisible(False)
        self.add_button.setEnabled(True)
        self.remove_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.pick_dest.setEnabled(True)
        self.start_button.setEnabled(bool(self.jobs))

    def _finished(self) -> None:
        self._teardown()
        self.frame_bar.setVisible(False)
        parts = [f"{self._done} converted"]
        if self._skipped:
            parts.append(f"{self._skipped} already done")
        if self._failed:
            parts.append(f"{self._failed} failed")
        summary = ", ".join(parts) + f" — {self.destination}"
        self.status.setText(summary)
        if self._failures:
            QMessageBox.warning(
                self, "Some videos failed",
                "These videos were not converted:\n\n" + "\n".join(self._failures),
            )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt name
        if self._worker is not None:
            self._worker.stop()
        self._teardown()
        super().closeEvent(event)


class VideoDownloadWorker(QObject):
    """Fetches PyAV the first time the Video tab is used."""

    progress = Signal(object, object)  # done, total bytes
    status = Signal(str)
    finished = Signal()
    failed = Signal(str)

    def run(self) -> None:
        try:
            bootstrap.install_av(
                on_bytes=lambda d, t: self.progress.emit(d, t),
                on_text=self.status.emit,
            )
        except Exception as error:  # noqa: BLE001 - surfaced in the UI
            self.failed.emit(str(error))
            return
        self.finished.emit()


class VideoWorker(QObject):
    """Drives a whole video off the UI thread, reporting each frame."""

    progress = Signal(str)
    frame_done = Signal(object)   # pipeline.VideoProgress
    finished = Signal(object)     # the output path
    failed = Signal(str)

    def __init__(
        self,
        source: Path,
        destination: Path,
        settings: AppSettings,
        engine: DepthEngine,
        codec_key: str,
        start: int,
        limit,
        estimate_depth: bool,
    ) -> None:
        super().__init__()
        self._source = source
        self._destination = destination
        self._settings = settings
        self._engine = engine
        self._codec = codec_key
        self._start = start
        self._limit = limit
        self._estimate = estimate_depth
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            for update in pipeline.convert_video(
                self._source, self._destination, self._settings, self._engine,
                codec_key=self._codec, start=self._start, limit=self._limit,
                estimate_depth=self._estimate,
                grade_settings=self._settings.grade,
                progress=self.progress.emit,
                should_stop=lambda: self._stop,
            ):
                self.frame_done.emit(update)
        except Exception as error:  # noqa: BLE001 - the UI is the error handler
            self.failed.emit(str(error))
            return
        self.finished.emit(self._destination)


class VideoQueueWorker(QObject):
    """Convert a list of videos back to back, off the UI thread.

    One worker for the whole queue rather than one per clip, so the thread and
    its teardown are set up once. A failure on one video is reported and the
    queue moves on — a batch that stops dead on the third of twenty files,
    hours in, is worse than one that finishes the other seventeen and tells you
    which failed.
    """

    progress = Signal(str)
    #: index (0-based), total, source Path — a clip is starting.
    video_started = Signal(int, int, object)
    #: VideoProgress, index, total — a frame of the current clip landed.
    frame_done = Signal(object, int, int)
    #: source Path, output Path — a clip finished.
    video_done = Signal(object, object)
    #: source Path — skipped because its output already existed.
    video_skipped = Signal(object)
    #: source Path, error — a clip failed; the queue continues.
    video_failed = Signal(object, str)
    finished = Signal()

    def __init__(
        self,
        jobs: list[tuple[Path, Path]],   # (source, output) pairs
        settings: AppSettings,
        engine: DepthEngine,
        codec_key: str,
        estimate_depth: bool,
        skip_existing: bool,
    ) -> None:
        super().__init__()
        self._jobs = jobs
        self._settings = settings
        self._engine = engine
        self._codec = codec_key
        self._estimate = estimate_depth
        self._skip = skip_existing
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        total = len(self._jobs)
        for index, (source, output) in enumerate(self._jobs):
            if self._stop:
                break
            if self._skip and output.exists():
                self.video_skipped.emit(source)
                continue
            self.video_started.emit(index, total, source)
            try:
                for update in pipeline.convert_video(
                    source, output, self._settings, self._engine,
                    codec_key=self._codec, start=0, limit=None,
                    estimate_depth=self._estimate,
                    grade_settings=self._settings.grade,
                    progress=self.progress.emit,
                    should_stop=lambda: self._stop,
                ):
                    self.frame_done.emit(update, index, total)
            except Exception as error:  # noqa: BLE001 - reported, not fatal to the queue
                self.video_failed.emit(source, str(error))
                continue
            if self._stop:
                break
            self.video_done.emit(source, output)
        self.finished.emit()


class VideoPage(QWidget):
    """The Video tab: inspect a clip, mark a range, convert it, keep the audio.

    Shares the sidebar with the photo tab - the neural strengths, style, depth
    and colour all mean the same thing on a frame of video - and adds a small
    player so the clip can be scrubbed and watched before committing to a
    conversion. The player uses QMediaPlayer (Windows Media Foundation / the
    bundled FFmpeg backend), which handles decode, audio and sync; the app's own
    PyAV path is for the conversion, not for playback.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.source: Path | None = None
        self.info = None

        # Same shape as the single-image tab: a bordered stage on the left with a
        # welcoming empty state, and a rail of cards with the controls on the
        # right, the convert actions pinned at its foot.
        outer = QHBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(16)

        from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
        from PySide6.QtMultimediaWidgets import QVideoWidget

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(320)
        self.preview = ImageView()  # kept: conversion progress draws here
        self.placeholder = stage_placeholder(
            "🎞", "Drop a video here, or choose one",
            "MP4 · MOV · MKV — then scrub, mark a range, and convert",
        )
        # Three faces of one area: the empty state, the source video while
        # inspecting, the converted frames while converting. A stack so the
        # layout never jumps.
        self.stack = QStackedWidget()
        self.stack.addWidget(self.placeholder)    # 0 — until a clip loads
        self.stack.addWidget(self.video_widget)   # 1
        self.stack.addWidget(self.preview)         # 2
        self.stack.setCurrentWidget(self.placeholder)

        stage = QFrame()
        stage.setObjectName("stage")
        stage_v = QVBoxLayout(stage)
        stage_v.setContentsMargins(0, 0, 0, 0)
        stage_v.addWidget(self.stack, 1)

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video_widget)

        transport = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setObjectName("secondary")
        self.play_button.setFixedWidth(80)
        self.play_button.setEnabled(False)
        transport.addWidget(self.play_button)
        self.timeline = TimelineWidget()
        transport.addWidget(self.timeline, 1)

        self.bar = QProgressBar()
        self.bar.setTextVisible(True)
        self.bar.setVisible(False)

        left = QVBoxLayout()
        left.setSpacing(10)
        left.addWidget(stage, 1)
        left.addLayout(transport)
        left.addWidget(self.bar)
        outer.addLayout(left, 1)

        # -- Source card --
        source_card = ModuleCard("Source")
        self.pick_video = QPushButton("Choose video…")
        self.source_label = QLabel("No video chosen")
        self.source_label.setObjectName("hint")
        self.source_label.setWordWrap(True)
        source_card.add(self.pick_video)
        source_card.add(self.source_label)
        self.pick_output = QPushButton("Output…")
        self.pick_output.setObjectName("secondary")
        self.output_label = QLabel("Output: choose a video first")
        self.output_label.setObjectName("hint")
        self.output_label.setWordWrap(True)
        source_card.add(self.pick_output)
        source_card.add(self.output_label)

        # -- Export card --
        export_card = ModuleCard("Export")
        self.codec_box = QComboBox()
        for codecdef in video.CODECS:
            self.codec_box.addItem(codecdef.label, codecdef.key)
        self.codec_box.setToolTip(
            "H.264/MP4 is the safe default - every editor and player takes it, "
            "and it is hardware-encoded on your GPU.\n\n"
            "H.265 is smaller for modern editors. VP9/WebM is for web upload, "
            "not for editing - editors do not import WebM cleanly."
        )
        self.mode_box = QComboBox()
        self.mode_box.addItem("Quick (1 pass)", 1)
        self.mode_box.addItem("Quality (4 passes)", 4)
        self.mode_box.setToolTip(
            "Passes let DLSS's accumulator settle. One is fast and usually "
            "plenty for video; four is steadier on tricky material."
        )
        self.range_box = QComboBox()
        self.range_box.addItem("Whole clip", "whole")
        self.range_box.addItem("Select In/Out", "range")
        self.range_box.setToolTip(
            "Whole clip converts everything. Select In/Out shows brackets on the "
            "timeline - drag them to the part you want, scroll to zoom in for a "
            "precise edit."
        )
        for label, widget in (
            ("Format", self.codec_box), ("Effort", self.mode_box), ("Range", self.range_box),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addStretch(1)
            widget.setMinimumWidth(150)
            row.addWidget(widget)
            export_card.add_layout(row)
        self.estimate_depth = QCheckBox("Estimate depth per frame")
        self.estimate_depth.setToolTip(
            "Off by default, and honestly labelled: DLSS does not read the depth "
            "plane on a still frame (there is no motion to reproject through), "
            "so this changes nothing in the output and is by far the slowest step."
        )
        export_card.add(self.estimate_depth)

        self.info_label = QLabel("")
        self.info_label.setObjectName("hint")
        self.info_label.setWordWrap(True)

        self.queue_button = QPushButton("Convert several…")
        self.queue_button.setObjectName("secondary")
        self.queue_button.setToolTip(
            "Queue several videos and convert them back to back with the current "
            "settings. Each is converted whole; use Convert video for a range "
            "within one clip."
        )
        self.start = QPushButton("Convert video")
        self.start.setEnabled(False)
        apply_font(self.start, family=FONT_DISPLAY, size=11.5, spacing=1.8, caps=True)
        self.stop = QPushButton("Stop")
        self.stop.setObjectName("secondary")
        self.stop.setVisible(False)

        rail = QWidget()
        rail.setFixedWidth(SIDEBAR_WIDTH + 18)
        rail_col = QVBoxLayout(rail)
        rail_col.setContentsMargins(0, 0, 0, 0)
        rail_col.setSpacing(12)
        rail_col.addWidget(source_card)
        rail_col.addWidget(export_card)
        rail_col.addWidget(self.info_label)
        rail_col.addStretch(1)
        rail_col.addWidget(self.queue_button)
        rail_col.addWidget(self.stop)
        rail_col.addWidget(self.start)
        outer.addWidget(rail)

    def show_video(self) -> None:
        self.stack.setCurrentWidget(self.video_widget)

    def show_preview(self) -> None:
        self.stack.setCurrentWidget(self.preview)


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

        # Same language as the single-image and video tabs: a bordered stage on
        # the left (with a welcoming empty state), a rail of cards on the right.
        outer = QHBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(16)

        self.preview = ImageView()
        self.placeholder = stage_placeholder(
            "🎬", "Choose the first frame of a sequence",
            "A folder of PNG/EXR frames — all converted with identical settings "
            "for a steady result",
        )
        self.stack = QStackedWidget()
        self.stack.addWidget(self.placeholder)  # 0 — until frames load
        self.stack.addWidget(self.preview)       # 1
        self.stack.setCurrentWidget(self.placeholder)

        stage = QFrame()
        stage.setObjectName("stage")
        stage_v = QVBoxLayout(stage)
        stage_v.setContentsMargins(0, 0, 0, 0)
        stage_v.addWidget(self.stack, 1)

        self.bar = QProgressBar()
        self.bar.setTextVisible(True)
        self.bar.setVisible(False)

        left = QVBoxLayout()
        left.setSpacing(10)
        left.addWidget(stage, 1)
        left.addWidget(self.bar)
        outer.addLayout(left, 1)

        # -- Frames card --
        frames_card = ModuleCard("Frames")
        self.pick_frames = QPushButton("Choose first frame…")
        self.frames_label = QLabel("No sequence chosen")
        self.frames_label.setObjectName("hint")
        self.frames_label.setWordWrap(True)
        frames_card.add(self.pick_frames)
        frames_card.add(self.frames_label)

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
        self.depth_label = QLabel("Depth: estimated per frame (Depth Anything)")
        self.depth_label.setObjectName("hint")
        self.depth_label.setWordWrap(True)
        self.invert_depth = QCheckBox("Depth is inverted (near is dark)")
        self.invert_depth.setToolTip(
            "Renderers disagree about which way up a depth pass goes. A Blender "
            "mist pass is near-dark, so tick this. Watch the preview and pick "
            "whichever looks right - near should read as red on the depth mask."
        )
        self.invert_depth.setVisible(False)
        frames_card.add(self.pick_depth)
        frames_card.add(self.clear_depth)
        frames_card.add(self.depth_label)
        frames_card.add(self.invert_depth)

        # -- Output card --
        out_card = ModuleCard("Output")
        self.pick_output = QPushButton("Output folder…")
        self.pick_output.setObjectName("secondary")
        self.output_label = QLabel("")
        self.output_label.setObjectName("hint")
        self.output_label.setWordWrap(True)
        self.write_video = QCheckBox("Also write MP4")
        self.fps = QSpinBox()
        self.fps.setRange(1, 240)
        self.fps.setValue(24)
        self.fps.setSuffix(" fps")
        self.fps.setEnabled(False)
        self.write_video.toggled.connect(self.fps.setEnabled)
        self.write_video.setToolTip(
            "Encoded with mp4v, not H.264 - OpenCV ships no H.264 encoder. The "
            "PNG frames are always written too, so you can re-encode them later."
        )
        out_card.add(self.pick_output)
        mp4_row = QHBoxLayout()
        mp4_row.addWidget(self.write_video)
        mp4_row.addStretch(1)
        mp4_row.addWidget(self.fps)
        out_card.add_layout(mp4_row)
        out_card.add(self.output_label)

        self.start = QPushButton("Convert sequence")
        self.start.setEnabled(False)
        apply_font(self.start, family=FONT_DISPLAY, size=11.5, spacing=1.8, caps=True)
        self.stop = QPushButton("Stop")
        self.stop.setObjectName("secondary")
        self.stop.setVisible(False)

        rail = QWidget()
        rail.setFixedWidth(SIDEBAR_WIDTH + 18)
        rail_col = QVBoxLayout(rail)
        rail_col.setContentsMargins(0, 0, 0, 0)
        rail_col.setSpacing(12)
        rail_col.addWidget(frames_card)
        rail_col.addWidget(out_card)
        rail_col.addStretch(1)
        rail_col.addWidget(self.stop)
        rail_col.addWidget(self.start)
        outer.addWidget(rail)


class EffectsPage(QWidget):
    """The Effects tab: the app's native ReShade-style post-processing stack.

    A tab of its own, as asked, but everything on it acts on the *current
    result* — the same image the photo and video tabs produce — because effects
    are a final look laid over a finished conversion, not a separate conversion.
    Turn effects on here and they show live in the preview, and are baked into
    every save, video and batch from then on.

    Live and instant, like the colour grade: the DLSS result is already
    computed, so a slider here re-runs only the cheap effect pass over a small
    preview, never the harness. See effects.apply.
    """

    #: (group title, enable field, blurb, [(label, field, min, max, tip), ...]).
    #: The LUT group is inserted separately, since it is a file picker not a
    #: slider bank.
    _GROUPS = (
        ("Sharpen", "sharpen_enabled",
         "Unsharp mask — crisp up the detail the neural pass produced.", (
            ("Amount", "sharpen_amount", 0.0, 2.0, "Strength of the sharpening."),
            ("Radius", "sharpen_radius", 0.3, 5.0, "Width of the edge halo, in pixels."),
        )),
        ("Bloom", "bloom_enabled",
         "Light bleeding out of the brightest areas, like a bright screen or lens.", (
            ("Threshold", "bloom_threshold", 0.0, 1.0, "Brightness where the glow starts."),
            ("Intensity", "bloom_intensity", 0.0, 1.0, "How much glow is added."),
            ("Radius", "bloom_radius", 1.0, 30.0, "Spread of the glow, in pixels."),
        )),
        ("Chromatic aberration", "chroma_enabled",
         "Colour fringing that grows toward the corners, the way a real lens splits light.", (
            ("Amount", "chroma_amount", 0.0, 1.0, "How far red and blue pull apart."),
        )),
        ("CRT", "crt_enabled",
         "Scanlines, an RGB phosphor mask and tube curvature — the retro-display look.", (
            ("Scanlines", "crt_scanline", 0.0, 1.0, "Darkening of alternate lines."),
            ("Mask", "crt_mask", 0.0, 1.0, "RGB phosphor-stripe strength."),
            ("Curvature", "crt_curvature", 0.0, 1.0, "Bulge of the tube face. Off by default; it is the slow knob."),
        )),
        ("Vignette", "vignette_enabled",
         "Darkened corners, to draw the eye in.", (
            ("Amount", "vignette_amount", 0.0, 1.0, "How dark the corners go."),
            ("Feather", "vignette_feather", 0.0, 1.0, "How far in the darkening reaches."),
        )),
        ("Film grain", "grain_enabled",
         "Monochrome grain laid over everything, like film stock.", (
            ("Amount", "grain_amount", 0.0, 1.0, "Strength of the grain."),
            ("Size", "grain_size", 1.0, 6.0, "Coarseness — larger is chunkier grain."),
        )),
    )

    def __init__(self, effects_settings, on_change, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = effects_settings
        self._on_change = on_change
        self._rows: dict[str, SliderRow] = {}
        self._groups: dict[str, ModuleCard] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # The live preview — the whole point of a tab rather than a menu: you
        # shape the look while looking at it.
        self.preview = ImageView()
        self.hint = QLabel(
            "Convert an image on the Single image tab, then shape the look here.\n"
            "Every effect is off until you turn it on, and applies to saves, "
            "videos and batches too."
        )
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_stack = QStackedWidget()
        self.preview_stack.addWidget(self.hint)
        self.preview_stack.addWidget(self.preview)
        layout.addWidget(self.preview_stack, 1)

        # The controls, scrolled — there are more of them than fit a short window.
        controls = QWidget()
        column = QVBoxLayout(controls)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(10)

        insert_lut_after = "Chromatic aberration"
        for title, enable_field, blurb, rows in self._GROUPS:
            column.addWidget(self._make_group(title, enable_field, blurb, rows))
            if title == insert_lut_after:
                column.addWidget(self._make_lut_group())
        column.addStretch(1)

        scroller = QScrollArea()
        scroller.setWidget(controls)
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QFrame.Shape.NoFrame)
        scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroller.setFixedWidth(SIDEBAR_WIDTH + 40)

        side = QWidget()
        side_col = QVBoxLayout(side)
        side_col.setContentsMargins(0, 0, 0, 0)
        side_col.setSpacing(8)
        side_col.addWidget(scroller, 1)
        self.reset_button = QPushButton("Reset all effects")
        self.reset_button.setObjectName("secondary")
        self.reset_button.clicked.connect(self._reset)
        side_col.addWidget(self.reset_button)
        side.setFixedWidth(SIDEBAR_WIDTH + 40)
        layout.addWidget(side)

    # -- building the controls ----------------------------------------------

    def _make_group(self, title, enable_field, blurb, rows) -> ModuleCard:
        group = ModuleCard(title, checkable=True)
        group.setToolTip(blurb)
        group.toggled.connect(lambda on, f=enable_field: self._set_enabled(f, on))
        self._groups[enable_field] = group

        note = QLabel(blurb)
        note.setObjectName("hint")
        note.setWordWrap(True)
        group.add(note)
        for label, field, low, high, tip in rows:
            row = SliderRow(
                label, getattr(self._settings, field), self._slider_setter(field),
                tip, maximum=high, minimum=low,
            )
            self._rows[field] = row
            group.add(row)
        group.setChecked(bool(getattr(self._settings, enable_field)))
        return group

    def _make_lut_group(self) -> ModuleCard:
        group = ModuleCard("LUT (.cube)", checkable=True)
        group.setToolTip(
            "Apply a 3D .cube colour LUT — film emulations, CRT looks, cinematic "
            "grades. Drop .cube files in the LUTs folder and pick one here; any "
            "pack that exports .cube works."
        )
        group.toggled.connect(lambda on: self._set_enabled("lut_enabled", on))
        self._groups["lut_enabled"] = group

        picker = QHBoxLayout()
        self.lut_combo = QComboBox()
        self.lut_combo.currentIndexChanged.connect(self._lut_chosen)
        picker.addWidget(self.lut_combo, 1)
        self.refresh_button = QPushButton("↻")
        self.refresh_button.setObjectName("secondary")
        self.refresh_button.setFixedWidth(36)
        self.refresh_button.setToolTip("Rescan the LUTs folder.")
        picker.addWidget(self.refresh_button)
        self.open_folder_button = QPushButton("Folder…")
        self.open_folder_button.setObjectName("secondary")
        self.open_folder_button.setToolTip("Open the LUTs folder to add .cube files.")
        picker.addWidget(self.open_folder_button)
        group.add_layout(picker)

        self.lut_status = QLabel("")
        self.lut_status.setObjectName("hint")
        self.lut_status.setWordWrap(True)
        group.add(self.lut_status)

        row = SliderRow(
            "Amount", self._settings.lut_amount, self._slider_setter("lut_amount"),
            "How strongly the LUT is blended in.", maximum=1.0, minimum=0.0,
        )
        self._rows["lut_amount"] = row
        group.add(row)
        group.setChecked(bool(self._settings.lut_enabled))
        return group

    # -- behaviour -----------------------------------------------------------

    def _slider_setter(self, field: str):
        def apply_value(value: float) -> None:
            setattr(self._settings, field, value)
            self._on_change()
        return apply_value

    def _set_enabled(self, field: str, on: bool) -> None:
        setattr(self._settings, field, bool(on))
        self._on_change()

    def _lut_chosen(self, _index: int) -> None:
        self._settings.lut_name = self.lut_combo.currentData() or ""
        self._on_change()

    def set_luts(self, names: list[str]) -> None:
        """Refill the LUT picker, keeping the current choice if it survives."""
        current = self._settings.lut_name
        self.lut_combo.blockSignals(True)
        self.lut_combo.clear()
        self.lut_combo.addItem("(none)", "")
        for name in names:
            self.lut_combo.addItem(name, name)
        index = self.lut_combo.findData(current)
        self.lut_combo.setCurrentIndex(index if index >= 0 else 0)
        self.lut_combo.blockSignals(False)
        if not names:
            self.lut_status.setText("No .cube files yet — add some via Folder….")
        else:
            self.lut_status.setText(f"{len(names)} LUT(s) available.")

    def set_lut_status(self, text: str) -> None:
        self.lut_status.setText(text)

    def show_preview(self, image_u8) -> None:
        if image_u8 is None:
            self.preview_stack.setCurrentWidget(self.hint)
            return
        # _graded_preview hands back 8-bit; ImageView.set_image wants 0..1 float.
        self.preview.set_image(
            image_u8.astype(np.float32) / 255.0, effects.describe(self._settings)
        )
        self.preview_stack.setCurrentWidget(self.preview)

    def _reset(self) -> None:
        from .settings import EffectsSettings

        defaults = EffectsSettings()
        for f in fields_of(self._settings):
            setattr(self._settings, f, getattr(defaults, f))
        # Push the defaults back into every widget without firing on_change per
        # control, then redraw once.
        for enable_field, group in self._groups.items():
            group.blockSignals(True)
            group.setChecked(bool(getattr(self._settings, enable_field)))
            group.blockSignals(False)
        for field, row in self._rows.items():
            row.set_value(getattr(self._settings, field))
        if hasattr(self, "lut_combo"):
            index = self.lut_combo.findData(self._settings.lut_name)
            self.lut_combo.setCurrentIndex(index if index >= 0 else 0)
        self._on_change()


def fields_of(dataclass_instance) -> list[str]:
    """Field names of a dataclass instance — a tiny import-free helper."""
    from dataclasses import fields as _fields

    return [f.name for f in _fields(dataclass_instance)]


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
        self.setWindowTitle("DLSS 5 Image & Video Converter")
        # Sized to the screen rather than fixed: 1280x820 does not fit on a
        # 1920x1080 display at 150% scaling, which reports 1280x720 of usable
        # space, and the window came up taller than the desktop.
        available = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1280, available.width() - 40), min(820, available.height() - 60))
        self.setMinimumSize(900, 560)

        self.settings = AppSettings.load(paths.settings_path())
        # One engine for the window's lifetime. Reloading Depth Anything per
        # image would add several seconds and a gigabyte of churn to every run.
        # ONNX Runtime, not PyTorch: same depth, no 2.7 GB torch download, and
        # the Apache-2.0 Small model ships in the app. See onnx_depth.py.
        self.engine = OnnxDepthEngine()
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
        #: Every ChipSliderGroup in the window, so the density toggle can flip
        #: them all between Compact and Full at once.
        self._chip_groups: list[ChipSliderGroup] = []
        self._download_thread: QThread | None = None
        self._download_worker: DownloadWorker | None = None
        self._download_dialog: DownloadDialog | None = None
        self._seq_thread: QThread | None = None
        self._seq_worker: SequenceWorker | None = None
        self._video_thread: QThread | None = None
        self._video_worker = None
        self._video_dl_thread: QThread | None = None
        self._video_dl_worker = None
        self._video_after_download = None
        self._tour_overlay: onboarding.SpotlightOverlay | None = None
        #: The background DLSS check, while it runs. Kept so shutdown can end it
        #: and so a second one is not started on top of the first.
        self._runtime_probe: RuntimeProbe | None = None
        self.style_results: dict[int, pipeline.Result] = {}
        self._style_signature_used: tuple | None = None
        self._style_worker: StyleWorker | None = None
        #: The style being converted right now, so its pane knows to sweep.
        self._style_running: int | None = None
        #: Styles queued for (re)conversion. A pane is greyed while its style is
        #: in here, whether or not a previous result is still on screen - so a
        #: settings change greys every style pane at once, rather than one at a
        #: time as each conversion reaches it.
        self._styles_pending: set[int] = set()
        #: How far that style has got, kept here because redrawing the panes
        #: resets their sweep state and it has to be put back.
        self._style_fraction = 0.0
        #: The view whose sweep is running, if any.
        self._sweeping = None

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

        # A second, slower beat. The 30 ms one keeps a slider responsive by
        # grading a small preview; this one fires once the slider settles and
        # regrades at full resolution, so what you inspect is the real output,
        # not a 1200 px stand-in. Two timers rather than one because the two
        # jobs want opposite intervals - one fast and rough, one late and sharp.
        self._grade_full_timer = QTimer(self)
        self._grade_full_timer.setSingleShot(True)
        self._grade_full_timer.setInterval(250)
        self._grade_full_timer.timeout.connect(self._render_full)

        # The effects tab's own preview, on the same coalescing idea as the
        # grade: an effect slider drag re-runs the stack over a small image, so
        # wait for the drag to settle rather than redrawing per pixel of travel.
        self._effects_preview_timer = QTimer(self)
        self._effects_preview_timer.setSingleShot(True)
        self._effects_preview_timer.setInterval(60)
        self._effects_preview_timer.timeout.connect(self._update_effects_preview)

        # Dropping anywhere on the window, not just on the drop zone. The zone
        # is swapped out for the comparison view after a conversion, and when it
        # was the only drop target there was no way at all to open a second
        # image without restarting the app.
        self.setAcceptDrops(True)

        # The window is a vertical stack now: a branded command bar across the
        # top, the working area (tabs + sidebar) in the middle, and the native
        # status bar as the footer. The command bar is what turns a form into a
        # cockpit — GPU/VRAM at a glance and the runtime state always in view.
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._command_bar())

        # Full-bleed so the tab band below can span the whole window edge to
        # edge; each page supplies its own inner padding.
        main = QWidget()
        layout = QHBoxLayout(main)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

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
        # The view controls float on the picture (bottom-left, like the mockup)
        # and are revealed only while the pointer is over it, so the image is
        # unobstructed while it is being judged. The StageHost owns that reveal.
        self.stage_host = StageHost(self.stack, self._view_bar())
        canvas_layout.addWidget(self.stage_host, 1)
        self.grade_panel = self._grade_panel()
        canvas_layout.addWidget(self.grade_panel)
        self.style_panel = self._style_panel()
        canvas_layout.addWidget(self.style_panel)

        # The four other pages, then Settings — built before the rail so the
        # theme, density, HDR and Depth controls Settings owns exist as instance
        # state the rail and the rest of the app read.
        self.video_page = VideoPage()
        self.sequence_page = SequencePage()
        self.effects_page = EffectsPage(self.settings.effects, self._effects_changed)
        self.settings_page = self._settings_page()

        # The rail is the Single-image controls, so it lives *inside* that tab —
        # not beside the tab widget, where it used to sit on top of every tab
        # including Video and Settings. It scrolls; the actions under it pin.
        scroller = QScrollArea()
        scroller.setWidget(self._sidebar())
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QFrame.Shape.NoFrame)
        scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroller.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.single_scroller = scroller
        rail = QWidget()
        rail_col = QVBoxLayout(rail)
        rail_col.setContentsMargins(0, 0, 0, 0)
        rail_col.setSpacing(0)
        rail_col.addWidget(scroller, 1)
        rail_col.addWidget(self._actions())
        rail.setFixedWidth(SIDEBAR_WIDTH + 18)

        single = QWidget()
        single_row = QHBoxLayout(single)
        single_row.setContentsMargins(16, 16, 16, 16)
        single_row.setSpacing(16)
        single_row.addWidget(canvas, 1)
        single_row.addWidget(rail)
        self.single_page = single

        # One full-width tab band across the whole window (documentMode makes it
        # flat with a baseline), with the mockup's "+ Apply to folder" shortcut
        # pinned to the right corner of the band.
        self.tabs = QTabWidget()
        self.tabs.setObjectName("mainTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(single, "Single image")
        self.tabs.addTab(self.video_page, "Video")
        self.tabs.addTab(self.sequence_page, "Image sequence")
        self.tabs.addTab(self.effects_page, "Effects")
        self.tabs.addTab(self.settings_page, "Settings")

        self.tab_apply = QPushButton("+  Apply to folder")
        self.tab_apply.setObjectName("tabAction")
        self.tab_apply.setCursor(Qt.CursorShape.PointingHandCursor)
        self.tab_apply.setToolTip(
            "Run a whole folder with the current single-image settings."
        )
        self.tab_apply.clicked.connect(self.open_batch)
        self.tabs.setCornerWidget(self.tab_apply, Qt.Corner.TopRightCorner)

        self._wire_sequence_page()
        self._wire_video_page()
        self._wire_effects_page()
        self.tabs.currentChanged.connect(self._tab_changed)

        layout.addWidget(self.tabs, 1)

        root.addWidget(main, 1)
        self.setCentralWidget(central)
        self._build_footer_links()
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
            status = runtime.detect(self.settings.runtime_dir)
            message = runtime.describe(status)
            self._set_runtime_pill(status.ready)
        except Exception as error:  # noqa: BLE001 - startup must survive anything
            message = f"Could not check the DLSS runtime: {error}"
            self._set_runtime_pill(False)
        self.statusBar().showMessage(message)

        # After the event loop starts, so the window is painted behind the
        # dialog rather than the app appearing to hang on a bare download box.
        QTimer.singleShot(0, self._start_initial_setup)

    # -- command bar ---------------------------------------------------------

    def _command_bar(self) -> QWidget:
        """The branded top strip: identity, GPU/VRAM, and the runtime state.

        This is most of what separates the cockpit look from a plain form. The
        GPU gauge answers "will this fit in VRAM" before a conversion, and the
        runtime pill keeps "am I set up" in view at all times rather than buried
        in a status line that scrolls away.
        """
        bar = QFrame()
        bar.setObjectName("commandBar")
        bar.setFixedHeight(64)
        row = QHBoxLayout(bar)
        # Vertical margins so the gauge and pill sit centred with air above and
        # below, rather than glued to the top and bottom edges.
        row.setContentsMargins(20, 12, 20, 12)
        row.setSpacing(16)

        glyph = QLabel("N")
        glyph.setObjectName("brandGlyph")
        glyph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        apply_font(glyph, family=FONT_DISPLAY, size=14)
        row.addWidget(glyph)

        # One line, the app's real name, in the display face and spaced out like
        # the mockup — no subtitle.
        self.wordmark = QLabel()
        self.wordmark.setObjectName("wordmark")
        apply_font(self.wordmark, family=FONT_DISPLAY, size=12.5, spacing=2.6)
        self._set_wordmark()
        row.addWidget(self.wordmark)

        row.addStretch(1)

        self.gpu_gauge = GpuGauge()
        row.addWidget(self.gpu_gauge)

        self.runtime_pill = QLabel("●  CHECKING")
        self.runtime_pill.setObjectName("runtimePill")
        self.runtime_pill.setProperty("state", "setup")
        apply_font(self.runtime_pill, family=FONT_MONO, size=8.5, spacing=1.4)
        row.addWidget(self.runtime_pill)
        return bar

    def _set_wordmark(self) -> None:
        """DLSS·5 IMAGE & VIDEO CONVERTER on one line, dot in the accent colour."""
        if not hasattr(self, "wordmark"):
            return
        signal = PALETTES.get(self.settings.theme, PALETTES[DEFAULT_THEME])["signal"]
        self.wordmark.setText(
            f'DLSS<span style="color:{signal};">·</span>5&nbsp;&nbsp;'
            f'IMAGE&nbsp;&amp;&nbsp;VIDEO&nbsp;CONVERTER'
        )

    def _set_runtime_pill(self, ready: bool) -> None:
        """Green READY / amber SETUP NEEDED, mirroring the status-bar message."""
        if not hasattr(self, "runtime_pill"):
            return
        self.runtime_pill.setText("●  RUNTIME READY" if ready else "●  SETUP NEEDED")
        self.runtime_pill.setProperty("state", "ready" if ready else "setup")
        # Qt does not re-evaluate a property selector on its own.
        self.runtime_pill.style().unpolish(self.runtime_pill)
        self.runtime_pill.style().polish(self.runtime_pill)

    def _build_footer_links(self) -> None:
        """Turn the status bar into the footer: GitHub · Support on the right.

        The status bar already carries the operational message on the left, which
        is exactly the mockup's footer shape, so it is reused rather than a second
        bar being stacked under it. Permanent widgets sit to the right of any
        transient message.
        """
        github = QPushButton("GitHub")
        github.setObjectName("link")
        github.setCursor(Qt.CursorShape.PointingHandCursor)
        github.setToolTip(f"Opens the project page:\n{GITHUB_URL}")
        github.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(GITHUB_URL)))

        sep = QLabel("·")
        sep.setObjectName("linkSep")

        coffee = QPushButton("Support the project ☕")
        coffee.setObjectName("link")
        coffee.setCursor(Qt.CursorShape.PointingHandCursor)
        coffee.setToolTip("Buy me a coffee — entirely optional, and thank you.")
        coffee.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(COFFEE_URL)))

        self.statusBar().addPermanentWidget(github)
        self.statusBar().addPermanentWidget(sep)
        self.statusBar().addPermanentWidget(coffee)

    # -- view switching ------------------------------------------------------

    def _view_bar(self) -> HoverBar:
        """The floating view controls, revealed on hover over the picture.

        A translucent pill pinned to the bottom-left of the stage, in the
        mockup's instrument language. The StageHost fades it in and out with the
        on-image pills so the picture is clean while it is being studied.
        """
        bar = HoverBar()
        row = QHBoxLayout(bar)
        row.setContentsMargins(6, 6, 6, 6)
        row.setSpacing(3)
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
            button.setObjectName("viewChip")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.view_photo.clicked.connect(lambda: self.show_view("photo"))
        self.view_depth.clicked.connect(lambda: self.show_view("depth"))
        self.view_result.clicked.connect(lambda: self.show_view("result"))
        self.view_diff.clicked.connect(lambda: self.show_view("difference"))
        self.view_styles.clicked.connect(lambda: self.show_view("styles"))
        self.view_grade = QPushButton("Colour")
        self.view_grade.setObjectName("viewChip")
        self.view_grade.setCheckable(True)
        self.view_grade.setCursor(Qt.CursorShape.PointingHandCursor)
        self.view_grade.setToolTip(
            "Exposure, contrast, saturation and vibrance, applied to the "
            "finished image.\n\n"
            "After the neural pass, not before, so it is instant - nothing is "
            "re-evaluated and nothing is lost. Grading the input instead would "
            "change what the model sees and cost a full re-run per nudge."
        )
        self.view_grade.toggled.connect(self._grade_toggled)

        row.addWidget(self.view_photo)
        row.addWidget(self.view_depth)
        row.addWidget(self.view_result)
        row.addWidget(self.view_diff)
        row.addWidget(self.view_styles)
        # A hairline divider, then Colour set apart: it is not another view, it
        # changes the one you are on.
        sep = QFrame()
        sep.setObjectName("viewBarSep")
        sep.setFixedWidth(1)
        row.addSpacing(6)
        row.addWidget(sep)
        row.addSpacing(6)
        row.addWidget(self.view_grade)
        return bar

    def _grade_panel(self) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        grade = self.settings.grade

        specs = (
            ("Exposure", "exposure", -2.0, 2.0, "Stops of light. Applied in linear."),
            ("Contrast", "contrast", -1.0, 1.0, "S-curve around middle grey."),
            ("Saturation", "saturation", -1.0, 1.0, "Uniform. -1 is greyscale."),
            ("Vibrance", "vibrance", -1.0, 1.0,
             "Lifts muted colour and leaves saturated colour alone, so skies "
             "move and skin mostly does not."),
        )
        # The same chip pattern as the sidebar, one row wide since the strip has
        # the whole canvas width — four chips, one slider, honouring the density
        # toggle like every other multi-slider group.
        self._grade_group = ChipSliderGroup(
            [(label, getattr(grade, field), self._grade_setter(field), tip, low, high)
             for (label, field, low, high, tip) in specs],
            mode=self.settings.density,
            columns=4,
        )
        self._chip_groups.append(self._grade_group)
        # Keep the field→row map _reset_grade relies on, sourced from the group.
        self._grade_rows = {
            field: self._grade_group.rows[label] for (label, field, *_rest) in specs
        }
        layout.addWidget(self._grade_group, 1)

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
        self._style_signature_used = self._style_signature()
        self.convert_button.setEnabled(False)
        self._style_running = None
        self._style_fraction = 0.0
        # Every style is about to be (re)computed. Marking them all pending now
        # is what greys both panes immediately: their current pixels - stale
        # results from the last run, or the source - do not represent what is
        # coming, so none of them should read as finished. style_results is
        # kept, not cleared, so a re-run shows the previous result greyed under
        # the sweep rather than flashing to the source and back.
        self._styles_pending = set(range(len(NR_STYLES)))

        # Straight to the layout it is going to end up in. Showing a full-screen
        # "working" view first and snapping to panels at the end makes the wait
        # feel like a different operation from the thing it produces.
        self._show_style_placeholders()
        # After the placeholders: showing them calls _set_view_state, which
        # would otherwise switch this straight back on.
        self.view_styles.setEnabled(False)

        self._thread = QThread(self)
        self._style_worker = StyleWorker(
            self.image_path, self.settings, self.engine,
            list(range(len(NR_STYLES))), self.prepared,
        )
        self._style_worker.moveToThread(self._thread)
        self._thread.started.connect(self._style_worker.run)
        self._style_worker.progress.connect(self._report_progress)
        self._style_worker.started.connect(self._style_started)
        self._style_worker.one_done.connect(self._style_one_done)
        self._style_worker.finished.connect(self._styles_ready)
        self._style_worker.failed.connect(self._styles_failed)
        self._thread.start()

    def _show_style_placeholders(self) -> None:
        """Switch to the comparison layout and draw its starting state.

        _render_styles greys every pending style pane, so with all styles
        pending this is what makes both panes go grey at once.
        """
        if self.prepared is None:
            return
        self._render_styles()
        self.stack.setCurrentWidget(self._style_view())
        self._set_view_state("styles")

    def _graded_source(self) -> np.ndarray:
        """The source at preview size, graded like the results will be."""
        assert self.prepared is not None
        small = _downscale_for_preview(self.prepared.source)
        linear = contract.srgb_to_linear(np.clip(small, 0.0, 1.0).astype(np.float32))
        return grade.apply_preview(linear, self.settings.grade)

    def _style_started(self, style: int) -> None:
        self._style_running = style
        self._style_fraction = 0.0
        if self._view != "styles":
            return
        for index, box in enumerate(self.style_choices):
            if box.currentData() == style and index < self._pane_count():
                self.side_by_side.set_pane_progress(index, 0.0)

    def _style_one_done(self, style: int, result: object) -> None:
        """Drop one finished style into its pane without waiting for the rest."""
        self.style_results[style] = result
        self._styles_pending.discard(style)
        self._style_running = None
        if self._view == "styles":
            self._render_styles()

    def _styles_ready(self, results: dict) -> None:
        self.style_results = results
        self._styles_pending = set()
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
        self._styles_pending = set()
        self._style_signature_used = None
        self._style_teardown()
        QMessageBox.warning(self, "Could not compare styles", message)
        self.statusBar().showMessage("Style comparison failed")

    def _style_teardown(self) -> None:
        self._end_progress()
        self._style_running = None
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._style_worker = None
        self.convert_button.setEnabled(self.image_path is not None)
        self.view_styles.setEnabled(self.image_path is not None)
        # Settings that moved while this run was in flight, same as _teardown.
        if self._preview_pending:
            self._preview_pending = False
            self._schedule_preview()

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

        Two panes is the style question - Natural against Cinematic (the two
        looks that actually differ; Default is the add-on's baseline). Three is
        the "is it helping" question, and that reads best with the source
        first, so the eye moves from unprocessed to most processed.
        """
        # NR_STYLES is (Default, Natural, Cinematic) = indices 0, 1, 2.
        defaults = [1, 2] if count == 2 else [-1, 1, 2]
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
        """Draw the panes, graded identically, into the chosen layout.

        A pane is greyed while its style is in ``_styles_pending`` - even if a
        previous result is still stored for it. That is the whole point: a
        stale result must not read as the finished one, so both panes go grey
        the instant a re-run starts and each returns to colour only when its own
        conversion lands. The stale result stays visible under the grey, so the
        view does not flash to the source and back.
        """
        if self.prepared is None:
            return
        count = self._pane_count()
        for index, box in enumerate(self.style_choices):
            box.setVisible(index < count)

        images: list[np.ndarray] = []
        labels: list[str] = []
        pending: list[int] = []
        for index in range(count):
            style = self.style_choices[index].currentData()
            if style is None or style < 0:
                # Original never converts, so it is never pending or swept.
                images.append(self._graded_source())
                labels.append("Original")
                continue
            result = self.style_results.get(style)
            if style in self._styles_pending:
                # Being (re)computed: show whatever we have, greyed.
                images.append(
                    self._graded_preview(result) if result is not None
                    else self._graded_source()
                )
                pending.append(index)
            else:
                images.append(self._graded_preview(result))
            labels.append(NR_STYLES[style])

        view = self._style_view()
        if view is self.side_by_side:
            view.set_panes_u8(images)
            view.set_labels(*labels)
        else:
            view.set_images_u8(images[0], images[1])
            # Neither style is "the result", so no accent - both pills neutral.
            view.set_labels(*labels, accent_right=False)
        # set_panes_u8 clears the sweep state, so re-arm every pending pane
        # after the redraw. The one being converted now sits at its real
        # progress; the rest sit fully grey at 0.0, waiting their turn.
        if view is self.side_by_side:
            for index in pending:
                running = self.style_choices[index].currentData() == self._style_running
                self.side_by_side.set_pane_progress(
                    index, self._style_fraction if running else 0.0
                )

        if self._styles_pending:
            self.style_caption.setText("Converting each style…")
        else:
            how = (
                "Scroll to zoom, right-drag to pan - every pane moves together."
                if view is self.side_by_side
                else "Drag the divider; scroll to zoom, right-drag to pan."
            )
            self.style_caption.setText(f"Same image, same settings, same grade. {how}")

    # -- grade + effects, the shared display stage ---------------------------
    #
    # Effects run after the grade, in display sRGB, exactly as pipeline._finish
    # does — so what the preview shows is what a save writes. These helpers are
    # the one place that pairing lives on the UI side, and every redraw and the
    # single-image save go through them.

    def _effects_active(self) -> bool:
        return not self.settings.effects.is_neutral

    def _detail_active(self) -> bool:
        # Only Preserve is a display-stage op; Boost is a conversion-time mode.
        return self.settings.detail.mode == "preserve"

    def _post_active(self) -> bool:
        """Whether any post-DLSS display stage (detail or effects) is on."""
        return self._detail_active() or self._effects_active()

    def _apply_preserve(
        self, srgb: np.ndarray, source_srgb: np.ndarray | None, preserve_range: bool = False
    ) -> np.ndarray:
        """Re-inject the source's detail, if Preserve is on and a source is given.

        Source-less callers (or a size mismatch, which happens on the HDR path
        where the scene-referred source is a different array) simply skip it.
        """
        if not self._detail_active() or source_srgb is None:
            return srgb
        if source_srgb.shape != srgb.shape:
            return srgb
        d = self.settings.detail
        return detail.preserve_detail(
            srgb, source_srgb, amount=d.amount, radius=d.radius, preserve_range=preserve_range
        )

    def _display_u8_from_linear(
        self, linear: np.ndarray, source_srgb: np.ndarray | None = None
    ) -> np.ndarray:
        """Linear RGB -> graded, detail-preserved, effected, 8-bit sRGB.

        Keeps grade.apply_preview's fast uint8 path when nothing post-DLSS is on,
        and only drops to the float route (which detail and effects need) when
        one is.
        """
        if not self._post_active():
            return grade.apply_preview(linear, self.settings.grade)
        srgb = grade.apply_to_linear(linear, self.settings.grade)  # float sRGB [0,1]
        srgb = self._apply_preserve(srgb, source_srgb)
        srgb = effects.apply(srgb, self.settings.effects, paths.luts_dir())
        return (np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _display_u8_from_srgb(
        self, srgb: np.ndarray, source_srgb: np.ndarray | None = None
    ) -> np.ndarray:
        """Already-graded sRGB float -> detail-preserved, effected, 8-bit sRGB."""
        srgb = self._apply_preserve(srgb, source_srgb)
        if self._effects_active():
            srgb = effects.apply(srgb, self.settings.effects, paths.luts_dir())
        return (np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _hdr_linear_with_effects(
        self, graded_linear: np.ndarray, source_srgb: np.ndarray | None = None
    ) -> np.ndarray:
        """Apply detail + effects to a graded HDR linear image, keeping its range.

        The same range-preserving round trip pipeline._finish uses, so an HDR
        preview and an HDR save agree: linear -> extended sRGB -> detail/effects
        -> linear, with highlights above white carried through intact.
        """
        if not self._post_active():
            return graded_linear
        srgb = contract.linear_to_srgb(graded_linear)
        srgb = self._apply_preserve(srgb, source_srgb, preserve_range=True)
        srgb = effects.apply(srgb, self.settings.effects, paths.luts_dir(), preserve_range=True)
        return contract.srgb_to_linear(srgb)

    def _graded_preview(
        self, result: pipeline.Result, original: bool = False
    ) -> np.ndarray:
        """One image as 8-bit, with the current grade and effects, at preview size."""
        if original:
            small = _downscale_for_preview(result.original)
            linear = contract.srgb_to_linear(np.clip(small, 0.0, 1.0).astype(np.float32))
            # Graded like the rest of the comparison (the caption promises "same
            # grade"), but effects are the look under test, so the source pane is
            # left free of them to compare against.
            return grade.apply_preview(linear, self.settings.grade)
        if result.hdr and result.enhanced_linear is not None:
            linear = _downscale_for_preview(result.enhanced_linear)
            graded = hdr_mod.tonemap(
                self._hdr_linear_with_effects(
                    grade.apply_linear(linear, self.settings.grade)
                ),
                result.white,
            )
            return (np.clip(graded, 0.0, 1.0) * 255.0).astype(np.uint8)
        small = _downscale_for_preview(result.enhanced)
        linear = contract.srgb_to_linear(np.clip(small, 0.0, 1.0).astype(np.float32))
        # The source at the same preview size, for Preserve to lift detail from.
        source_small = _downscale_for_preview(result.original)
        return self._display_u8_from_linear(
            linear, np.clip(source_small, 0.0, 1.0).astype(np.float32)
        )

    def _adopt_style(self, index: int) -> None:
        """Make one of the compared styles the result, and the live setting."""
        result = self.style_results.get(index)
        if result is None:
            return
        self.settings.neural.style = index
        self.style_box.set_index(index)
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
            # redraw is tens of milliseconds of numpy over the preview. The fast
            # timer keeps the drag live; the full timer sharpens once it stops.
            self._grade_timer.start()
            self._grade_full_timer.start()

        return apply_value

    def _render_current(self) -> None:
        """Fast redraw during a grade drag: small preview, stays responsive.

        The style comparison is graded too, and identically on both halves -
        the point is to isolate the style, so a grade that landed on only one
        side would defeat the whole view.
        """
        if self._view == "styles":
            self._render_styles()
        else:
            self._render_result(fast=True)

    def _render_full(self) -> None:
        """Sharp redraw once the grade settles: the real output, full size."""
        if self._view == "result":
            self._render_result(fast=False)

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
        self._render_result(fast=False)

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

    def _render_result(self, fast: bool = False, new: bool = False) -> None:
        """Redraw the before/after comparison with the current grade.

        Two resolutions for one reason: people need to *check* the result, and a
        1200 px preview cannot show the pore- and weave-level detail this tool
        exists to add - zooming it only magnifies blur. So the view holds the
        full-resolution image whenever it is idle, and drops to the fast preview
        only while a colour slider is actually moving, because grading a 4K frame
        is ~1.5 s and an 8K one ~6 s - far too slow per drag tick, fine once.

        ``new`` marks a freshly converted image, which resets the zoom; a grade
        redraw keeps it, so swapping preview<->full does not fight the zoom the
        user set to inspect something.
        """
        if self.result is None or self._preview_after_linear is None:
            return

        if fast:
            before_u8 = self._preview_before_u8
            if self.result.hdr:
                graded = hdr_mod.tonemap(
                    self._hdr_linear_with_effects(
                        grade.apply_linear(self._preview_after_linear, self.settings.grade)
                    ),
                    self.result.white,
                )
                after_u8 = (np.clip(graded, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                after_u8 = self._display_u8_from_linear(
                    self._preview_after_linear, self._preview_before
                )
        else:
            before_u8 = self._full_before_u8
            if self.result.hdr and self.result.enhanced_linear is not None:
                graded = hdr_mod.tonemap(
                    self._hdr_linear_with_effects(
                        grade.apply_linear(self.result.enhanced_linear, self.settings.grade)
                    ),
                    self.result.white,
                )
                after_u8 = (np.clip(graded, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                # Reuses the exact full-resolution grade, detail and effects the
                # export uses, so what is inspected is what gets saved.
                graded = grade.apply(self.result.enhanced, self.settings.grade)
                after_u8 = self._display_u8_from_srgb(
                    graded, np.clip(self.result.original, 0.0, 1.0).astype(np.float32)
                )

        self.wipe.set_images_u8(before_u8, after_u8, keep_view=not new)

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

        # Leaving the styles view puts the wipe back to source-versus-result,
        # which is what its labels should say. SOURCE / DLSS 5, matching the
        # mockup's on-image pills, with the result half accented so which is
        # which is clear before the divider is even moved.
        self.wipe.set_labels("SOURCE", "DLSS 5")
        self.side_by_side.clear()

        if which == "difference":
            self._render_difference()
            self.stack.setCurrentWidget(self.diff_view)
        elif which == "result":
            self._render_result(fast=False)
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
        # Not "and no thread is running": _set_view_state is called while a run
        # is in flight and not again when it ends, so keying the button on the
        # thread left it disabled for the rest of the session. Runs disable it
        # explicitly and the teardowns turn it back on.
        self.view_styles.setEnabled(self.image_path is not None)
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
        self.convert_button.setObjectName("convert")
        self.convert_button.setEnabled(False)
        # The hero action, in the display face, spaced and capitalised like the
        # mockup so it reads as the thing the whole panel points at.
        apply_font(self.convert_button, family=FONT_DISPLAY, size=12.5, spacing=2.4, caps=True)
        self.convert_button.clicked.connect(self.convert)
        # The one glow QSS cannot do: an accent halo so Convert reads as the hero
        # action, the thing the whole cockpit is pointed at. Kept on self so a
        # theme switch can retint it.
        self._convert_glow = QGraphicsDropShadowEffect(self.convert_button)
        self._convert_glow.setBlurRadius(34)
        self._convert_glow.setOffset(0, 0)
        self._tint_convert_glow()
        self.convert_button.setGraphicsEffect(self._convert_glow)

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

        # Runtime setup, Check runtime, Help and the theme picker moved to the
        # Settings tab — the action bar now carries only what you do to an image.
        # Apply to folder moved to the tab band's right corner (self.tab_apply),
        # so it is not repeated here; batch_button is kept as state for its
        # tooltip and any enable/disable, just not shown in this column.
        layout.addWidget(self.open_button)
        layout.addWidget(self.convert_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.feedback_button)

        # The GitHub · Support links moved to the footer (the status bar) so they
        # read as a footer across the whole window rather than tucked under the
        # sidebar buttons — see _build_footer_links.
        return panel

    # -- theme ---------------------------------------------------------------

    def _tint_convert_glow(self) -> None:
        """Colour the Convert halo with the active palette's signal accent."""
        signal = PALETTES.get(self.settings.theme, PALETTES[DEFAULT_THEME])["signal"]
        colour = QColor(signal)
        colour.setAlpha(150)
        self._convert_glow.setColor(colour)

    def _theme_changed(self, _index: int) -> None:
        name = self.theme_box.currentData() or DEFAULT_THEME
        self.settings.theme = name
        self.settings.save(paths.settings_path())
        self._apply_theme(name)

    def _apply_theme(self, name: str) -> None:
        """Repaint the whole app in a palette, live. STYLE is reassigned in
        apply_app_theme, so re-setting it on this window restyles every child."""
        app = QApplication.instance()
        if app is not None:
            apply_app_theme(app, name)
        self.setStyleSheet(STYLE)
        self._tint_convert_glow()
        self._set_wordmark()

    def _density_changed(self, _index: int) -> None:
        """Flip every chip/slider group between Compact and Full, live."""
        mode = self.density_toggle.current_data() or "compact"
        self.settings.density = mode
        self.settings.save(paths.settings_path())
        for group in self._chip_groups:
            group.set_mode(mode)

    def _hdr_card(self) -> QWidget:
        """The add-on's HDR / display controls, kept in the main workflow.

        They sit in the sidebar with the neural controls rather than buried in
        Settings: they change the neural result, and people went straight here
        looking for them. Defaults match the add-on's own.
        """
        s = self.settings.neural
        hdr = ModuleCard("HDR / display")
        hdr.setToolTip(
            "The add-on's HDR controls. This pipeline is SDR end to end, but "
            "these still change the result — the neural pass reasons about light "
            "before anything is tonemapped back. Defaults match the add-on's own."
        )
        hdr.add(SliderRow(
            "Paper white", s.paper_white, self._neural_setter("paper_white"),
            "The luminance the model treats as diffuse white. On an HDR or OLED "
            "display this decides how hard highlights are pushed. The add-on "
            "defaults to 1; shipping game configs use 16, where it stops changing.",
            maximum=NR_PAPER_WHITE_MAX,
        ))
        hdr.add(SliderRow(
            "HDR transfer", s.transfer_strength, self._neural_setter("transfer_strength"),
            "Strength of the transfer curve the pass works through. Range 0..1.",
            maximum=NR_TRANSFER_MAX,
        ))
        hdr.add(SliderRow(
            "Colour strength", s.color_strength, self._neural_setter("color_strength"),
            "How much of the model's colour change is kept. At 0 the source colour "
            "survives and only structure changes. Range 0..1.",
            maximum=NR_COLOR_MAX,
        ))
        return hdr

    def _settings_page(self) -> QWidget:
        """The Settings tab — a home for everything that is configured once and
        then left alone, so the sidebar can hold only per-image controls.

        Appearance, the DLSS runtime setup (previously loose buttons), and the
        advanced HDR and Depth controls all live here. Their widgets are still
        created as the same instance attributes the rest of the app reads, just
        parented into this page instead of the sidebar.
        """
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        inner = QWidget()
        cols = QHBoxLayout(inner)
        cols.setContentsMargins(18, 18, 18, 18)
        cols.setSpacing(16)
        left = QVBoxLayout()
        left.setSpacing(14)
        right = QVBoxLayout()
        right.setSpacing(14)

        # -- Appearance --
        appearance = ModuleCard("Appearance")
        theme_row = QHBoxLayout()
        theme_row.addWidget(QLabel("Colour theme"))
        self.theme_box = QComboBox()
        for palette_name in PALETTES:
            self.theme_box.addItem(palette_name, palette_name)
        current = self.theme_box.findData(self.settings.theme)
        self.theme_box.setCurrentIndex(current if current >= 0 else 0)
        self.theme_box.setToolTip("The colour palette the whole app is drawn in.")
        self.theme_box.currentIndexChanged.connect(self._theme_changed)
        theme_row.addStretch(1)
        theme_row.addWidget(self.theme_box, 1)
        appearance.add_layout(theme_row)

        density_row = QHBoxLayout()
        density_label = QLabel("Controls")
        density_label.setToolTip(
            "Compact — parameters become a row of chips over one slider, so a "
            "group of four sliders takes one slider's height.\n\n"
            "Full — every slider is shown at once, the classic stacked layout."
        )
        density_row.addWidget(density_label)
        density_row.addStretch(1)
        self.density_toggle = SegmentedControl(
            [("Compact", "compact"), ("Full", "full")],
            current=0 if self.settings.density != "full" else 1,
        )
        self.density_toggle.changed.connect(self._density_changed)
        density_row.addWidget(self.density_toggle, 1)
        appearance.add_layout(density_row)

        # -- DLSS runtime (the setup that used to be loose buttons) --
        runtime_grp = ModuleCard("DLSS runtime")
        self.settings_runtime_status = QLabel("")
        self.settings_runtime_status.setObjectName("hint")
        self.settings_runtime_status.setWordWrap(True)
        runtime_grp.add(self.settings_runtime_status)
        r_buttons = QHBoxLayout()
        find_btn = QPushButton("Find my DLSS files…")
        find_btn.setObjectName("secondary")
        find_btn.setToolTip(
            "Search your Steam libraries, Downloads and Documents for the files, "
            "and copy them in. Nothing is downloaded."
        )
        find_btn.clicked.connect(self.open_find_files)
        check_btn = QPushButton("Check runtime")
        check_btn.setObjectName("secondary")
        check_btn.clicked.connect(self.diagnose)
        r_buttons.addWidget(find_btn)
        r_buttons.addWidget(check_btn)
        r_buttons.addStretch(1)
        runtime_grp.add_layout(r_buttons)

        # HDR / display lives in the single-image sidebar now (see _hdr_card),
        # not here — people went looking for it in the workflow, not in Settings.

        # -- Advanced: Depth --
        depth = ModuleCard("Depth")
        self.model_box = QComboBox()
        for label, model_id in MODELS.items():
            self.model_box.addItem(label, model_id)
        index = self.model_box.findData(self.settings.depth.model_id)
        self.model_box.setCurrentIndex(max(0, index))
        self.model_box.currentIndexChanged.connect(self._model_changed)
        depth.add(self.model_box)
        depth.add(SliderRow(
            "Depth contrast", min(1.0, self.settings.depth.contrast / 3.0),
            self._contrast_changed,
            "Reshapes the near-far spread. Higher pushes more of the frame into "
            "the foreground. Updates the depth mask live.",
        ))
        self.tiled = QCheckBox("Tiled depth (slow, sharper silhouettes)")
        self.tiled.setChecked(self.settings.depth.tiled)
        self.tiled.toggled.connect(self._tiled_changed)
        depth.add(self.tiled)

        # -- Help --
        help_card = ModuleCard("Help")
        help_btn = QPushButton("Open the guide…")
        help_btn.setObjectName("secondary")
        help_btn.setToolTip(f"Opens the guide in your browser:\n{WIKI_URL}")
        help_btn.clicked.connect(lambda: open_help())
        help_card.add(help_btn)
        replay_btn = QPushButton("Replay introduction…")
        replay_btn.setObjectName("secondary")
        replay_btn.setToolTip(
            "Show the first-conversion introduction and the guided tour again."
        )
        replay_btn.clicked.connect(self.replay_onboarding)
        help_card.add(replay_btn)

        # Two balanced columns, so nothing — a slider especially — sprawls the
        # full width of the window.
        left.addWidget(appearance)
        left.addStretch(1)
        right.addWidget(runtime_grp)
        right.addWidget(depth)
        right.addWidget(help_card)
        right.addStretch(1)
        cols.addLayout(left, 1)
        cols.addLayout(right, 1)

        scroller = QScrollArea()
        scroller.setWidget(inner)
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroller)
        self._refresh_settings_runtime()
        return page

    def _refresh_settings_runtime(self) -> None:
        """Fill the Settings runtime line with a plain ready / not-ready verdict."""
        if not hasattr(self, "settings_runtime_status"):
            return
        try:
            status = runtime.detect(self.settings.runtime_dir or None)
            if status.ready:
                text = "✓ Runtime ready — all files found."
            else:
                text = "✗ Not ready:\n• " + "\n• ".join(status.problems)
        except Exception as error:  # noqa: BLE001 - status must never crash the tab
            text = f"Could not check the runtime: {error}"
        self.settings_runtime_status.setText(text)

    def _sidebar(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(SIDEBAR_WIDTH)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        neural = ModuleCard("Neural", tag="RenoDX · DLAA")
        self.neural_card = neural
        neural_layout = neural.body
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

        # Style is the add-on's three looks — Default, Natural, Cinematic — so it
        # reads better as a segmented control with all three visible than as a
        # dropdown that hides two. Default is what DLSS 5 gives the moment it is
        # switched on; Natural and Cinematic are the two graded looks.
        self.style_box = SegmentedControl(
            list(NR_STYLES),
            current=min(len(NR_STYLES) - 1, max(0, settings.style)),
        )
        self.style_box.setToolTip(
            "The add-on's overall look.\n\n"
            "Default — the look DLSS 5 starts with.\n"
            "Natural — subtle grade, closer to the source.\n"
            "Cinematic — stronger grade, moves furthest from the source."
        )
        self.style_box.changed.connect(self._style_changed)
        neural_layout.addWidget(self.style_box)

        # One chip per strength, one slider shown at a time. Four stacked sliders
        # were most of what made this group read as a wall; the chips carry the
        # names and the selected one drops its slider in below — the same
        # segmented pattern as the view switch under the image.
        self.neural_params = ChipSliderGroup([
            ("Intensity", settings.intensity, self._neural_setter("intensity"),
             "Overall strength of the neural pass. At 0 this is a plain DLAA "
             "resolve. The add-on's own range is 0..2; it stops changing above 2.",
             0.0, NR_STRENGTH_MAX),
            ("Skin", settings.skin, self._neural_setter("skin"),
             "Subsurface scattering and pore detail on faces. Lower this first "
             "if results look waxy.",
             0.0, NR_STRENGTH_MAX),
            ("Local tone", settings.local_tone, self._neural_setter("local_tone"),
             "How much the model may relight the scene.",
             0.0, NR_STRENGTH_MAX),
            ("Structure", settings.structure, self._neural_setter("structure"),
             "Micro-contrast in fabric, hair, and surface material.",
             0.0, NR_STRENGTH_MAX),
        ], mode=self.settings.density)
        self._chip_groups.append(self.neural_params)
        neural_layout.addWidget(self.neural_params)
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

        # HDR/display and Depth are rarely touched, so they live on the Settings
        # tab now (see _settings_page) — the sidebar keeps only what you reach for
        # on every image. The controls are still built here as instance state so
        # everything downstream that reads them is unchanged.

        evaluation = ModuleCard("Output")
        eval_layout = evaluation.body
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

        layout.addWidget(self._detail_group())
        layout.addWidget(self._hdr_card())
        layout.addStretch(1)
        return panel

    def _detail_group(self) -> QWidget:
        """Detail recovery — give back the fine texture DLAA softens.

        Preserve is instant and lives here with the neural controls because it is
        about the *quality* of the result, not an optional look. It re-injects
        the source photo's own fine detail onto the DLSS output, so brick, fabric
        and mesh stay crisp while the neural relighting is kept.
        """
        group = ModuleCard("Detail", tag="New")
        self.detail_card = group
        group.setToolTip(
            "DLAA is an anti-aliaser: on a photo it smooths genuine fine texture "
            "(brick, perforations, railings). Preserve puts that detail back by "
            "lifting the source's own high-frequency band onto the result — the "
            "real detail, not a sharpen, so it cannot halo."
        )
        d_layout = group.body

        modes = [("Off", "off"), ("Preserve", "preserve"), ("Boost", "boost")]
        current = next(
            (i for i, (_, data) in enumerate(modes) if data == self.settings.detail.mode),
            0,
        )
        self.detail_mode = SegmentedControl(modes, current=current)
        self.detail_mode.setToolTip(
            "Off — the plain DLSS result.\n\n"
            "Preserve — re-inject the source's real fine detail (instant, native "
            "resolution, no halos). The right default for renders and textured "
            "photos.\n\n"
            "Boost — supersample: run DLSS at 2×/4×/8× the size, crispened, "
            "then downscale. Sharper still on renders, but slow (many more "
            "pixels) and it takes effect on the next Convert, not live."
        )
        self.detail_mode.changed.connect(self._detail_mode_changed)
        d_layout.addWidget(self.detail_mode)

        self.detail_amount = SliderRow(
            "Amount",
            self.settings.detail.amount,
            self._detail_amount_setter,
            "Preserve: how much source detail to blend back (0 = plain DLSS, "
            "1 = full detail).\nBoost: strength of the pre-DLSS crispen.",
            maximum=1.0,
            minimum=0.0,
        )
        d_layout.addWidget(self.detail_amount)

        # Boost-only: how hard to push it. "Extra sharpness" instead of the raw
        # 2×/4×/8× multiplier - the number is an implementation detail, and the
        # levels that would blow past the hardware texture limit are disabled by
        # _sync_boost_guard rather than left to fail mid-conversion.
        ss_row = QHBoxLayout()
        self.detail_super_label = QLabel("Extra sharpness")
        ss_row.addWidget(self.detail_super_label)
        self.detail_supersample = QComboBox()
        for factor in DETAIL_BOOST_FACTORS:
            self.detail_supersample.addItem(BOOST_LEVEL_LABELS[factor], factor)
        self.detail_supersample.setToolTip(
            "How much sharper Boost pushes. Higher runs DLSS at a larger size "
            "before shrinking it back, so it is crisper but slower. Levels that "
            "would not fit at the current Max size are greyed out."
        )
        ss_idx = self.detail_supersample.findData(self.settings.detail.supersample)
        self.detail_supersample.setCurrentIndex(ss_idx if ss_idx >= 0 else 0)
        self.detail_supersample.currentIndexChanged.connect(self._detail_super_changed)
        ss_row.addStretch(1)
        ss_row.addWidget(self.detail_supersample)
        d_layout.addLayout(ss_row)

        # A second, quieter line for the guard message ("Max needs a smaller Max
        # size…"), kept separate from the mode explainer so the two do not fight
        # over one label.
        self.detail_guard = QLabel()
        self.detail_guard.setObjectName("hint")
        self.detail_guard.setWordWrap(True)
        d_layout.addWidget(self.detail_guard)

        # A one-line explainer under the controls, like the mockup's card copy —
        # updated to whichever mode is selected.
        self.detail_hint = QLabel()
        self.detail_hint.setObjectName("hint")
        self.detail_hint.setWordWrap(True)
        d_layout.addWidget(self.detail_hint)

        self._sync_detail_controls()
        return group

    _DETAIL_HINTS = {
        "off": "The plain DLSS result — no detail recovery.",
        "preserve": "Preserve keeps the source's own fine texture — brick, mesh "
                    "and fabric stay crisp — while keeping the neural relight. "
                    "Instant, and the right default.",
        "boost": "Boost runs DLSS larger than the image, then shrinks it back for "
                 "extra crispness. Slower, and it applies on the next Convert.",
    }

    def _sync_boost_guard(self) -> None:
        """Offer only the sharpness levels that fit the current Max size.

        Boost runs at (Max size × level), and a D3D12 texture cannot exceed
        D3D12_MAX_TEXTURE_DIMENSION on a side, so a high level at a high Max size
        overflows - the opaque failure a user hit as an add-on init error. Here
        the overflowing levels are simply disabled, the selection is stepped down
        to the best that fits, and a plain line explains it. Above 8192 px even
        the smallest level cannot fit, which is the "Boost needs Max size 8192 px
        or smaller" case.
        """
        if not hasattr(self, "detail_supersample"):
            return
        max_edge = int(self.settings.evaluation.max_edge)
        ceiling = max_boost_factor(max_edge)  # 0 when nothing fits
        active = self.settings.detail.mode == "boost"

        # Enable/disable each level by whether it fits, without firing the change
        # handler while we reshape the list.
        self.detail_supersample.blockSignals(True)
        model = self.detail_supersample.model()
        for i in range(self.detail_supersample.count()):
            factor = self.detail_supersample.itemData(i)
            fits = ceiling > 0 and factor <= ceiling
            model.item(i).setEnabled(fits)

        # Only clamp the stored level and speak up while Boost is the live mode.
        # Doing it when Boost is off would silently change a setting the user
        # cannot see the effect of; the guard re-runs the moment Boost is chosen.
        message = ""
        if active:
            if ceiling == 0:
                # Nothing fits: Boost can't supersample at this Max size.
                message = (
                    f"Boost needs Max size 8192 px or smaller — at {max_edge} px "
                    "it would exceed the GPU's texture limit."
                )
            else:
                chosen = int(self.settings.detail.supersample)
                if chosen > ceiling:
                    # Step the stored setting down to the best that fits, say so.
                    self.settings.detail.supersample = ceiling
                    self.settings.save(paths.settings_path())
                    idx = self.detail_supersample.findData(ceiling)
                    if idx >= 0:
                        self.detail_supersample.setCurrentIndex(idx)
                    message = (
                        f"{BOOST_LEVEL_LABELS.get(chosen, 'That level')} needs a "
                        f"smaller Max size — using {BOOST_LEVEL_LABELS[ceiling]} "
                        f"at {max_edge} px."
                    )
        self.detail_supersample.blockSignals(False)
        if hasattr(self, "detail_guard"):
            self.detail_guard.setText(message)
            self.detail_guard.setVisible(bool(message))

    def _sync_detail_controls(self) -> None:
        """Grey the sharpness row unless Boost is the active mode, set the card's
        explainer to match the selected mode, and re-run the Boost guard."""
        mode = self.settings.detail.mode
        is_boost = mode == "boost"
        self.detail_supersample.setEnabled(is_boost)
        self.detail_super_label.setEnabled(is_boost)
        if hasattr(self, "detail_hint"):
            self.detail_hint.setText(self._DETAIL_HINTS.get(mode, ""))
        self._sync_boost_guard()

    def _detail_mode_changed(self, _index: int) -> None:
        self.settings.detail.mode = self.detail_mode.current_data() or "off"
        self.settings.save(paths.settings_path())
        self._sync_detail_controls()
        self._detail_redraw()

    def _detail_super_changed(self, _index: int) -> None:
        self.settings.detail.supersample = int(self.detail_supersample.currentData() or 2)
        self.settings.save(paths.settings_path())
        # Boost is a conversion-time mode, so this takes effect on the next
        # Convert; there is nothing to redraw live.

    def _detail_amount_setter(self, value: float):
        self.settings.detail.amount = value
        self.settings.save(paths.settings_path())
        self._detail_redraw()

    def _detail_redraw(self) -> None:
        """Repaint the result and the effects preview on the grade timers.

        Preserve is a cheap post-DLSS pass like the grade, so it rides the same
        coalesced fast/sharp beat — the wipe stays responsive while the amount
        slider drags, and sharpens once it settles.
        """
        if self._view in ("result", "styles"):
            self._grade_timer.start()
            self._grade_full_timer.start()
        self._effects_preview_timer.start()


    # -- sequence page -------------------------------------------------------

    def _wire_video_page(self) -> None:
        page = self.video_page
        page.pick_video.clicked.connect(self._pick_video)
        page.pick_output.clicked.connect(self._pick_video_output)
        page.start.clicked.connect(self._start_video)
        page.stop.clicked.connect(self._stop_video)
        page.queue_button.clicked.connect(self.open_video_queue)
        page.mode_box.currentIndexChanged.connect(self._video_mode_changed)
        page.range_box.currentIndexChanged.connect(self._video_range_changed)
        page.play_button.clicked.connect(self._toggle_video_play)
        # Player <-> timeline. The player drives the playhead as it advances;
        # the timeline drives the player when the user scrubs. Both talk in
        # milliseconds to the player and frames to the timeline.
        page.player.positionChanged.connect(self._video_position_changed)
        page.player.playbackStateChanged.connect(self._video_playback_state)
        page.timeline.seeked.connect(self._video_scrub)
        page.output_path: Path | None = None

    # -- effects tab ---------------------------------------------------------

    def _wire_effects_page(self) -> None:
        page = self.effects_page
        page.refresh_button.clicked.connect(self._refresh_luts)
        page.open_folder_button.clicked.connect(self._open_luts_folder)
        self._refresh_luts()
        # A LUT chosen in a previous session needs validating on startup so its
        # status line is right before the tab is ever opened.
        self._validate_lut()

    def _refresh_luts(self) -> None:
        self.effects_page.set_luts(effects.available_luts(paths.luts_dir()))
        self._validate_lut()

    def _open_luts_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(paths.luts_dir())))

    def _validate_lut(self) -> None:
        """Say whether the chosen LUT actually loads, so a bad .cube is obvious."""
        e = self.settings.effects
        if not (e.lut_enabled and e.lut_name):
            return
        try:
            effects.load_cube(paths.luts_dir() / e.lut_name)
            self.effects_page.set_lut_status(f"Using {e.lut_name}.")
        except ValueError as error:
            self.effects_page.set_lut_status(f"Cannot use {e.lut_name}: {error}")

    def _effects_changed(self) -> None:
        """An effect toggled or a slider moved: persist, redraw, keep it live.

        Reuses the grade timers so the main result view and the effects preview
        both refresh on the same coalesced beat — a drag stays responsive and
        the sharp redraw lands once it settles.
        """
        self.settings.save(paths.settings_path())
        self._validate_lut()
        # The main comparison view, when it is showing the result.
        if self._view in ("result", "styles"):
            self._grade_timer.start()
            self._grade_full_timer.start()
        self._effects_preview_timer.start()

    def _update_effects_preview(self) -> None:
        if self.result is None:
            self.effects_page.show_preview(None)
            return
        self.effects_page.show_preview(self._graded_preview(self.result))

    def _tab_changed(self, _index: int) -> None:
        # Draw the effects preview when its tab comes forward, so it reflects the
        # latest result and settings even if nothing changed while it was hidden.
        if self.tabs.currentWidget() is self.effects_page:
            self._update_effects_preview()

    def _video_fps(self) -> float:
        info = self.video_page.info
        return info.fps if info and info.fps else 24.0

    def _frame_to_ms(self, frame: int) -> int:
        return int(round(frame / self._video_fps() * 1000))

    def _ms_to_frame(self, ms: int) -> int:
        return int(round(ms / 1000 * self._video_fps()))

    def _toggle_video_play(self) -> None:
        from PySide6.QtMultimedia import QMediaPlayer

        player = self.video_page.player
        if player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            player.pause()
        else:
            self.video_page.show_video()
            player.play()

    def _video_playback_state(self, state) -> None:
        from PySide6.QtMultimedia import QMediaPlayer

        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.video_page.play_button.setText("Pause" if playing else "Play")

    def _video_position_changed(self, ms: int) -> None:
        # Do not fight the user while they are dragging the playhead.
        if not self.video_page.timeline._drag:
            self.video_page.timeline.set_playhead(self._ms_to_frame(ms))

    def _video_scrub(self, frame: int) -> None:
        self.video_page.show_video()
        self.video_page.player.setPosition(self._frame_to_ms(frame))

    def _video_range_changed(self) -> None:
        page = self.video_page
        page.timeline.set_range_mode(page.range_box.currentData() == "range")

    def _pick_video_output(self) -> None:
        page = self.video_page
        codec = video.CODECS_BY_KEY[page.codec_box.currentData()]
        stem = page.source.stem if page.source else "video"
        style = style_slug(self.settings.neural.style)
        start_dir = str(page.output_path or (paths.output_dir() / f"{stem}_dlss5_{style}{codec.suffix}"))
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Output video", start_dir, f"{codec.label} (*{codec.suffix})"
        )
        if chosen:
            page.output_path = Path(chosen)
            page.output_label.setText(f"Output: {chosen}")

    def _pick_video(self) -> None:
        exts = " ".join(f"*{s}" for s in sorted(video.INPUT_SUFFIXES))
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Choose video", "", f"Video ({exts})"
        )
        if not chosen:
            return
        path = Path(chosen)
        # Probing needs PyAV. If it is not here yet, offer to fetch it now
        # rather than after the user has set everything up and pressed Convert.
        if not video.is_available():
            self._download_video_support(then=lambda: self._load_video(path))
            return
        self._load_video(path)

    def _load_video(self, path: Path) -> None:
        try:
            info = video.probe(path)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Could not open video", str(error))
            return
        from PySide6.QtCore import QUrl

        page = self.video_page
        page.source = path
        page.info = info
        page.source_label.setText(path.name)
        page.info_label.setText(info.describe())
        page.start.setEnabled(True)
        page.play_button.setEnabled(True)

        # Load it into the player and lay out the timeline. A frame count of 0
        # (some containers do not report one) still gives a usable scrubber via
        # the duration the player reports.
        page.timeline.set_duration(info.frames or max(1, round(info.duration * info.fps)), info.fps)
        page.timeline.set_range_mode(page.range_box.currentData() == "range")
        page.player.setSource(QUrl.fromLocalFile(str(path)))
        page.show_video()

        # A default output beside the app, so Convert works without a detour
        # through the Output picker - but the picker can override it.
        codec = video.CODECS_BY_KEY[page.codec_box.currentData()]
        style = style_slug(self.settings.neural.style)
        page.output_path = paths.output_dir() / f"{path.stem}_dlss5_{style}{codec.suffix}"
        page.output_label.setText(f"Output: {page.output_path}")

    def _video_mode_changed(self) -> None:
        # Deliberately does not touch settings. The video Effort is read at
        # convert time onto a copy - writing it to the shared settings here is
        # what let a video export silently drop the single-image sidebar to
        # 1 pass, since both read the same settings.evaluation.frames.
        pass

    def _start_video(self) -> None:
        page = self.video_page
        if page.source is None or self._video_thread is not None:
            return
        if not video.is_available():
            self._download_video_support(then=self._start_video)
            return
        if page.output_path is None:
            self._pick_video_output()
            if page.output_path is None:
                return
        output = page.output_path

        # Playback holds a read handle on the source; stop it before a run so
        # nothing contends, and free the preview area to show converted frames.
        page.player.stop()
        page.show_preview()

        # A copy, with the video's own pass count, so the shared sidebar setting
        # the single-image path reads is left exactly as the user set it. The
        # neural strengths, style and colour still come from the live settings.
        settings = copy.deepcopy(self.settings)
        settings.evaluation.frames = int(page.mode_box.currentData() or 1)

        # The range comes from the timeline's In/Out when ranging, else the
        # whole clip. Frames, inclusive of Out, which is why limit is +1.
        start = 0
        limit = None
        if page.range_box.currentData() == "range":
            in_frame, out_frame = page.timeline.in_out()
            start = in_frame
            limit = max(1, out_frame - in_frame + 1)

        page.start.setEnabled(False)
        page.pick_video.setEnabled(False)
        page.play_button.setEnabled(False)
        page.stop.setVisible(True)
        page.bar.setVisible(True)
        page.bar.setValue(0)
        total = limit if limit is not None else (page.info.frames if page.info else 0)
        page.bar.setMaximum(total or 0)
        # Fresh clock per run: the ETA is built from steady-state frame timings,
        # not from whatever the last conversion left behind.
        self._video_eta = _EtaTracker()
        self._video_eta.start()

        self._video_thread = QThread(self)
        self._video_worker = VideoWorker(
            page.source, output, settings, self.engine,
            page.codec_box.currentData(), start, limit,
            page.estimate_depth.isChecked(),
        )
        self._video_worker.moveToThread(self._video_thread)
        self._video_thread.started.connect(self._video_worker.run)
        self._video_worker.frame_done.connect(self._video_frame_done)
        self._video_worker.finished.connect(self._video_finished)
        self._video_worker.failed.connect(self._video_failed)
        self._video_thread.start()

    def _stop_video(self) -> None:
        if self._video_worker is not None:
            self._video_worker.stop()
            self.video_page.info_label.setText("Stopping after this frame...")

    def _video_frame_done(self, update) -> None:
        page = self.video_page
        if update.stage == "converting":
            page.bar.setValue(update.index)
            # ETA rides on the bar text: it is the number people watch, and a
            # long conversion the user has walked away from is exactly when
            # "how much longer" matters. Withheld for the first frame or two,
            # while the estimate is still just the harness start-up.
            self._video_eta.tick()
            left = self._video_eta.remaining((update.total - update.index) if update.total else 0)
            if left is not None:
                page.bar.setFormat(f"%v of %m frames — ~{_format_duration(left)} left")
            else:
                page.bar.setFormat("%v of %m frames")
            # Every few frames, not every frame: repainting a preview per frame
            # on a fast clip is wasted work.
            if update.index % 3 == 0 or update.index == update.total:
                page.preview.set_image(
                    _downscale_for_preview(update.preview),
                    f"frame {update.index}"
                    + (f" of {update.total}" if update.total else ""),
                )
        elif update.stage == "muxing":
            page.info_label.setText("Adding audio...")
        elif update.stage.startswith("done"):
            note = "" if update.stage == "done" else " (source had no audio)"
            page.info_label.setText(f"Done{note}.")

    def _video_finished(self, output: Path) -> None:
        self._video_teardown()
        output = Path(output)
        self.video_page.info_label.setText(f"Saved {output.name}")
        self.statusBar().showMessage(f"Video saved: {output}")

        # A clear, explicit "done" - a video is a long operation and the user
        # is often not watching by the time it finishes. Offer the folder,
        # since the next thing they want is the file.
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        box = QMessageBox(
            QMessageBox.Icon.Information,
            "Video conversion complete",
            f"Saved to:\n{output}",
            parent=self,
        )
        box.addButton(QMessageBox.StandardButton.Ok)
        reveal = box.addButton("Open folder", QMessageBox.ButtonRole.ActionRole)
        box.exec()
        if box.clickedButton() is reveal:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(output.parent)))

    def _video_failed(self, message: str) -> None:
        self._video_teardown()
        QMessageBox.warning(self, "Video conversion failed", message)
        self.video_page.info_label.setText("Conversion failed")

    def _video_teardown(self) -> None:
        if self._video_thread is not None:
            self._video_thread.quit()
            self._video_thread.wait(15000)
            self._video_thread = None
        self._video_worker = None
        page = self.video_page
        page.start.setEnabled(page.source is not None)
        page.pick_video.setEnabled(True)
        page.play_button.setEnabled(page.source is not None)
        page.stop.setVisible(False)

    def _download_video_support(self, then) -> None:
        """Fetch PyAV, then run `then`. Blocks the tab with a progress bar.

        The slots below are bound methods, not the nested closures they used to
        be, and that is a correctness fix rather than tidiness. A worker's signal
        can only be marshalled onto the UI thread when its slot is a bound method
        of a QObject that lives there; a plain closure has no thread affinity, so
        Qt connects it *directly* and runs it on the worker thread. The progress
        slot touches a QWidget, and touching a widget off the GUI thread is
        undefined - it survived here and crashed a user with an access violation
        mid-download. Bound methods of the window get a queued connection and run
        where they must.
        """
        if self._video_dl_thread is not None:
            return
        self._video_after_download = then
        page = self.video_page
        page.bar.setVisible(True)
        page.bar.setMaximum(0)  # indeterminate until byte totals arrive
        page.info_label.setText("Downloading video support (about 35 MB, once)...")
        page.pick_video.setEnabled(False)
        page.start.setEnabled(False)

        self._video_dl_thread = QThread(self)
        self._video_dl_worker = VideoDownloadWorker()
        self._video_dl_worker.moveToThread(self._video_dl_thread)
        self._video_dl_thread.started.connect(self._video_dl_worker.run)
        self._video_dl_worker.progress.connect(self._video_dl_bytes)
        self._video_dl_worker.status.connect(self.video_page.info_label.setText)
        self._video_dl_worker.finished.connect(self._video_dl_done)
        self._video_dl_worker.failed.connect(self._video_dl_failed)
        self._video_dl_thread.start()

    def _video_dl_bytes(self, done, total) -> None:
        self.video_page.bar.setMaximum(int(total) or 0)
        self.video_page.bar.setValue(int(done))

    def _end_video_download(self) -> None:
        if self._video_dl_thread is not None:
            self._video_dl_thread.quit()
            self._video_dl_thread.wait()
            self._video_dl_thread = None
        self._video_dl_worker = None
        self.video_page.bar.setVisible(False)
        self.video_page.pick_video.setEnabled(True)

    def _video_dl_done(self) -> None:
        self._end_video_download()
        then = self._video_after_download
        self._video_after_download = None
        if then is not None:
            then()

    def _video_dl_failed(self, message: str) -> None:
        self._end_video_download()
        self._video_after_download = None
        self.video_page.info_label.setText("")
        QMessageBox.warning(self, "Could not add video support", message)

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
            self.sequence_page.stack.setCurrentWidget(self.sequence_page.preview)
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
        page.stack.setCurrentWidget(page.preview)

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

    def _start_initial_setup(self) -> None:
        """Finish the required downloads, then offer onboarding on new installs.

        PyTorch is handled before the main window is constructed. The depth
        model is the last required download, so this callback is the first point
        where the complete first-run sequence can safely continue.
        """
        self.ensure_model_downloaded()
        if self.settings.onboarding_version >= ONBOARDING_VERSION:
            return
        if not OnnxDepthEngine.is_downloaded(self.settings.depth.model_id):
            # The download dialog already explained the failure. Leave the
            # version at zero so a later successful launch can resume.
            return
        self._run_first_onboarding()

    def _run_first_onboarding(self) -> None:
        """Find the user's runtime, then introduce the app.

        The live DLSS check runs in the *background* rather than gating the
        tour. It initialises DLSS on the GPU, and on some driver/runtime combos
        that wedges - which is exactly the freeze that used to trap first-run
        behind an unclosable "Checking DLSS 5" box. Decoupling it means the app
        is always usable, the tour always reachable, and onboarding always
        completes, so a hung check can never loop the user back into it.
        """
        try:
            status = runtime.detect(self.settings.runtime_dir or None)
        except Exception:  # noqa: BLE001 - the finder is the recovery path
            status = None

        if status is None or not status.ready:
            finder = FindFilesDialog(self, onboarding_mode=True)
            QTimer.singleShot(0, finder.start_scan)
            finder.exec()
            try:
                status = runtime.detect(self.settings.runtime_dir or None)
            except Exception:  # noqa: BLE001 - tutorial remains useful offline
                status = None

        if status is not None and status.ready and status.harness is not None:
            self._begin_background_probe(status.harness)

        self._show_first_conversion_intro()

    def _begin_background_probe(self, harness: Path) -> None:
        """Confirm the neural path loads, off the UI thread, without blocking.

        The tour and the whole app stay usable while this runs; the status bar
        reports the result when it lands, and the watchdog inside RuntimeProbe
        gives up on a runtime that hangs instead of freezing the window.
        """
        if self._runtime_probe is not None:
            return
        self.statusBar().showMessage("Checking DLSS 5 in the background…")
        probe = RuntimeProbe(harness, parent=self)
        self._runtime_probe = probe
        probe.done.connect(self._background_probe_done)
        probe.start()

    def _background_probe_done(self, ok: bool, report: str) -> None:
        self._runtime_probe = None
        self.statusBar().showMessage(
            "DLSS 5 verified — the neural pass is live." if ok
            else "Could not confirm the neural pass — Settings → Check runtime for details."
        )

    def _show_first_conversion_intro(self) -> None:
        palette = PALETTES.get(self.settings.theme, PALETTES[DEFAULT_THEME])
        dialog = onboarding.FirstConversionDialog(
            paths.onboarding_image(), self, palette=palette
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self._complete_onboarding()
            return

        chosen = self.browse()
        if chosen is None:
            # Cancelling the file picker is not the same as skipping. Offer the
            # introduction again next launch rather than silently completing it.
            return
        QTimer.singleShot(150, self.start_tutorial)

    def replay_onboarding(self) -> None:
        """Settings entry point for people who skipped or want a refresher."""
        self._show_first_conversion_intro()

    def start_tutorial(self) -> None:
        """Spotlight the real controls in the same order as the first workflow."""
        if self._tour_overlay is not None and self._tour_overlay.isVisible():
            return
        self.tabs.setCurrentWidget(self.single_page)
        self.show_view("photo")
        self.single_scroller.ensureWidgetVisible(self.neural_card)

        steps = [
            onboarding.TourStep(
                "Your image",
                "Your image is loaded here now — drop or paste another anytime. "
                "After a conversion, drag the divider to compare source and "
                "result; the mouse wheel zooms and right-drag pans.",
                self.stack,
            ),
            onboarding.TourStep(
                "What you convert",
                "Single image is the main workflow. Video keeps the clip's audio; "
                "Image sequence handles rendered frames and optional depth passes; "
                "Effects adds the finishing look. Apply to folder repeats your "
                "current settings across many images.",
                self.tabs.tabBar(),
            ),
            onboarding.TourStep(
                "DLSS 5 preferences",
                "Intensity sets the overall neural effect. Skin, Local tone and "
                "Structure shape faces, relighting and material detail. Start near "
                "1.0, lower Skin for portraits, and compare Natural with Cinematic.",
                self.neural_card,
            ),
            onboarding.TourStep(
                "Detail",
                "Preserve restores the source's real fine texture after DLSS. Boost "
                "instead runs DLSS at the selected 2×, 4× or 8× supersample size, "
                "then downscales — sharper, but much slower and limited by the "
                "GPU's currently available VRAM.",
                self.detail_card,
            ),
            onboarding.TourStep(
                "Convert",
                "Convert runs the neural pass with the settings above. Inspect the "
                "Result, Depth mask and Difference views, then Save result — that "
                "is the whole single-image loop.",
                self.convert_button,
            ),
        ]
        palette = PALETTES.get(self.settings.theme, PALETTES[DEFAULT_THEME])
        overlay = onboarding.SpotlightOverlay(self, steps, palette=palette)
        self._tour_overlay = overlay
        overlay.step_changed.connect(self._tutorial_step_changed)
        overlay.finished.connect(self._complete_onboarding)
        overlay.skipped.connect(self._complete_onboarding)
        overlay.start()

    def _tutorial_step_changed(self, index: int) -> None:
        """Bring sidebar targets into view before the overlay measures them."""
        if index == 2:
            self.single_scroller.ensureWidgetVisible(self.neural_card, 0, 12)
        elif index == 3:
            self.single_scroller.ensureWidgetVisible(self.detail_card, 0, 12)
        if self._tour_overlay is not None:
            QTimer.singleShot(0, self._tour_overlay.refresh_target)

    def _complete_onboarding(self) -> None:
        self.settings.onboarding_version = ONBOARDING_VERSION
        self.settings.save(paths.settings_path())
        self._tour_overlay = None
        self.statusBar().showMessage("Introduction complete — open Settings to replay it.")

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
        if OnnxDepthEngine.is_downloaded(model_id):
            return

        label = next((k for k, v in MODELS.items() if v == model_id), model_id)
        self._download_dialog = DownloadDialog(
            "Finishing setup — just this once.",
            f"FIRST RUN · STEP 1 OF 3\n\n{label}\n\nThis is a one-time download, kept in the models folder "
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

    def browse(self) -> Path | None:
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Choose an image",
            str(self.image_path.parent) if self.image_path else "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.exr *.hdr *.jxr *.wdp *.hdp)",
        )
        if chosen:
            path = Path(chosen)
            self.open_image(path)
            return path
        return None

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
            status = runtime.detect(self.settings.runtime_dir)
            message = runtime.describe(status)
            self._set_runtime_pill(status.ready)
        except Exception as error:  # noqa: BLE001 - status must never raise
            message = f"Could not check the DLSS runtime: {error}"
            self._set_runtime_pill(False)
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

    def open_video_queue(self) -> None:
        """The video queue dialog, created on demand and kept while open.

        Gated on PyAV like the single-clip path: with no video component there
        is nothing to convert, so fetch it first and reopen the queue after.
        """
        if not video.is_available():
            self._download_video_support(then=self.open_video_queue)
            return
        existing = getattr(self, "_video_queue_dialog", None)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return
        self._video_queue_dialog = VideoQueueDialog(self)
        self._video_queue_dialog.show()

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
            # The grade is baked in (it is what is on screen), but the effects
            # stack is deliberately not: feeding grain, scanlines or a LUT back
            # into another DLSS pass would have the neural model chase those
            # artefacts as if they were detail. Effects stay a final-stage look,
            # re-applied after this next pass, not fed into it.
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
        self._full_before_u8 = None
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
        # Max size is the ceiling Boost multiplies, so a change here can make the
        # current sharpness level fit or overflow. Re-run the guard.
        self._sync_boost_guard()
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

    def _rerun_for_settings(self) -> bool:
        """Handle a settings change while the style comparison is showing.

        It was falling through to an ordinary preview, which converts once and
        then switches to the result view - so changing a slider silently threw
        you out of the comparison you were in the middle of reading. In that
        view a settings change invalidates both panes, so both are redone and
        the view stays put.
        """
        if self._view != "styles" or self.image_path is None:
            return False
        if self._thread is not None:
            self._preview_pending = True
            return True
        self._start_style_comparison()
        return True

    def _run_preview(self) -> None:
        if not self.settings.evaluation.live_preview or self.image_path is None:
            return
        if self._rerun_for_settings():
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
        # Preview runs sweep too. They used to be excluded because the sweep
        # moved the view; now it stays where it is, so a slider nudge shows the
        # after half recomputing against the before half you were comparing it
        # to - which is the whole reason to be on that view.
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
        """Start the sweep on whatever is already on screen.

        Deliberately does not move the user. An earlier version pulled the view
        back to the source for every run, which is wrong for the case that
        matters most: nudging a slider while watching the result. There the
        thing you want is the previous result still in front of you, with the
        after half working - not the source image and no comparison at all.
        """
        if self._view == "result" and self.result is not None:
            self._sweeping = self.wipe
            self.wipe.set_progress(0.0)
        elif self.prepared is not None:
            self.show_view("photo")
            self._sweeping = self.depth_view
            self.depth_view.set_progress(0.0)

    def _end_progress(self) -> None:
        self.depth_view.set_progress(None)
        self.wipe.set_progress(None)
        self.side_by_side.clear_progress()
        self._sweeping = None

    def _sweep(self, fraction: float) -> None:
        """Move whichever sweep is running to `fraction`."""
        if self._view == "styles":
            # One pane per style, and only the style being converted now.
            self._style_fraction = fraction
            for index, box in enumerate(self.style_choices):
                if box.currentData() == self._style_running:
                    self.side_by_side.set_pane_progress(index, fraction)
            return
        if self._sweeping is not None:
            self._sweeping.set_progress(fraction)

    def _report_progress(self, message: str) -> None:
        """Status text, and the colour sweep if this message carries a count.

        Parsed out of the message rather than plumbed through as a number: the
        progress callback is a string all the way down from the pipeline, and
        threading a second numeric channel through three call sites to drive a
        visual flourish is not worth it. If the wording ever changes the sweep
        simply stops advancing, which is a harmless way to fail.
        """
        self.statusBar().showMessage(message)
        match = _PASS_COUNT.search(message)
        if not match:
            return
        done, total = int(match.group(1)), int(match.group(2))
        if total > 0:
            self._sweep(done / total)

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
        # Full-resolution before image, for the sharp idle view. The after side
        # is graded on demand from result.enhanced(_linear), which are already
        # held, so no second full-size copy is stored.
        self._full_before_u8 = (np.clip(result.original, 0.0, 1.0) * 255.0).astype(np.uint8)
        if result.hdr:
            # Keep the preview scene-referred for an HDR result, so the grade
            # sliders act on the same numbers the export will.
            assert result.enhanced_linear is not None
            self._preview_after_linear = _downscale_for_preview(result.enhanced_linear)
        else:
            self._preview_after_linear = contract.srgb_to_linear(
                np.clip(self._preview_after, 0.0, 1.0).astype(np.float32)
            )
        self._render_result(fast=False, new=True)
        # Keep the effects tab's preview in step with the freshly converted
        # image, whether or not that tab is the one on screen right now.
        self._update_effects_preview()
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
        style = style_slug(self.settings.neural.style)
        default = str(Path(suggested) / f"{stem}_dlss5_{style}.{'jxr' if is_hdr else 'png'}")
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
                # Effects at native, before the export resize — the same order
                # the full-resolution preview uses, so a native-size save is
                # pixel-for-pixel what was on screen. Range is kept for the .jxr.
                image = self._hdr_linear_with_effects(image)
                image = resample.resize_linear(image, *target)
            else:
                image = grade.apply(self.result.enhanced, self.settings.grade)
                # Preserve and effects at native, before the export resize — the
                # same order the full-resolution preview uses, so a native-size
                # save is pixel-for-pixel what was on screen.
                image = self._apply_preserve(
                    image, np.clip(self.result.original, 0.0, 1.0).astype(np.float32)
                )
                if self._effects_active():
                    image = effects.apply(image, self.settings.effects, paths.luts_dir())
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
            self._show_runtime_report(lines, bool(status.problems))
            return
        if status.harness is None:
            self._show_runtime_report(lines, bool(status.problems))
            return

        # The live check goes through the watchdog-guarded probe behind a
        # cancellable box, never the UI thread: a runtime that wedges DLSS must
        # not freeze this window the way a plain subprocess.run here once did.
        dialog = DownloadDialog(
            "Checking DLSS 5",
            "One live test confirms that DLSS, ReShade, RenoDX and the neural "
            "renderer all load together on this GPU.",
            self,
        )
        dialog.setStyleSheet(STYLE)
        dialog.set_status("Starting the native runtime…")
        dialog.set_busy()
        dialog.enable_cancel("Skip this check")

        probe = RuntimeProbe(status.harness, parent=self)
        outcome: dict[str, str] = {}

        def done(_ok: bool, report: str) -> None:
            outcome["report"] = report
            dialog.accept()

        probe.done.connect(done)
        dialog.cancelled.connect(probe.skip)
        probe.start()
        dialog.exec()

        report = outcome.get("report", "")
        # The probe can come back "ready" by file detection yet fail the live
        # test with every field 0 (issue #6). Interpret the fields into a plain
        # cause, show it first, and treat a failed probe as a problem so the
        # Troubleshooting button appears.
        probe_problems = evaluator.interpret_probe(report) if report else []
        extra: list[str] = []
        if probe_problems:
            extra += [""] + probe_problems
        if report:
            extra += ["", report]
        self._show_runtime_report(
            lines + extra, bool(status.problems) or bool(probe_problems)
        )

    def _show_runtime_report(self, lines: list[str], has_problems: bool) -> None:
        box = QMessageBox(
            QMessageBox.Icon.Information, "DLSS 5 runtime",
            "\n".join(lines), parent=self,
        )
        box.addButton(QMessageBox.StandardButton.Ok)
        # Only when something is wrong. A clean report needs no reading.
        guide = (
            box.addButton("Troubleshooting", QMessageBox.ButtonRole.HelpRole)
            if has_problems
            else None
        )
        box.exec()
        if guide is not None and box.clickedButton() is guide:
            open_help("Troubleshooting")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt name
        self.settings.save(paths.settings_path())
        self._preview_timer.stop()
        self._preview_pending = False

        # End a background DLSS check if one is still running, so a probe that
        # wedged the GPU is killed with the window rather than orphaned holding
        # the device. Best-effort - never let shutdown raise.
        try:
            if self._runtime_probe is not None:
                self._runtime_probe.skip()
            evaluator.cancel_probe()
        except Exception:  # noqa: BLE001 - shutdown must survive anything
            pass

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
        # A video is many conversions in a row; stop it so it does not carry on
        # feeding a harness after the window has closed.
        if self._video_worker is not None:
            self._video_worker.stop()
        # Release the media player's handle on the source file.
        try:
            self.video_page.player.stop()
        except Exception:  # noqa: BLE001 - closing must not fail
            pass
        for thread in (
            self._thread,
            self._depth_thread,
            self._download_thread,
            self._seq_thread,
            self._video_thread,
            self._video_dl_thread,
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


def apply_app_theme(app: QApplication, name: str) -> None:
    """Point the whole app at one palette: stylesheet, Fusion base, QPalette.

    Fusion is the one built-in style that honours a custom palette across
    platforms, which matters because the image canvases paint their own
    background from ``palette().window()`` — without it the app would be dark
    where the stylesheet reaches and light where it does not. ``STYLE`` is a
    module global so a dialog created after a switch picks up the new look.
    """
    global STYLE
    STYLE = _style_for(name)
    app.setStyle("Fusion")
    app.setPalette(_qpalette_for(name))


def _load_bundled_fonts(app: QApplication) -> None:
    """Register Archivo + IBM Plex so the QSS font-family names resolve.

    The whole reason the app looked like a stock Windows form was the font:
    Segoe UI where the design calls for Archivo and IBM Plex. QSS cannot fetch a
    web font, so the TTFs ship in the bundle and are registered here, before the
    stylesheet is applied. If they are missing the QSS fallbacks (system-ui) keep
    the app readable — a missing font must never be fatal.
    """
    from PySide6.QtGui import QFont, QFontDatabase

    try:
        for ttf in sorted(paths.fonts_dir().glob("*.ttf")):
            QFontDatabase.addApplicationFont(str(ttf))
        # IBM Plex Sans as the base face, so every widget that does not name a
        # family inherits it rather than Segoe UI.
        if "IBM Plex Sans" in QFontDatabase.families():
            base = QFont("IBM Plex Sans")
            base.setPointSize(app.font().pointSize())
            app.setFont(base)
    except Exception:  # noqa: BLE001 - styling is never worth a failed launch
        pass


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("DLSS 5 Image & Video Converter")
    _load_bundled_fonts(app)
    try:
        theme = AppSettings.load(paths.settings_path()).theme
    except Exception:  # noqa: BLE001 - a bad settings file must not block launch
        theme = DEFAULT_THEME
    apply_app_theme(app, theme)
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
