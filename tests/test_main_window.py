"""The main-thread half of every button.

Written after v0.1.15 shipped with Convert, Compare styles and Find my DLSS
files all raising AttributeError on the first click. A patch had inserted three
methods before "the first `_teardown`" in the file, which belonged to a dialog
rather than to MainWindow - so the methods existed, just on the wrong class, and
every one of the 142 tests passed.

Nothing here runs a conversion. Threads are stubbed, so what is exercised is
precisely the part that broke: the code a click runs on the UI thread before any
work starts. That is cheap to test and is where this class of mistake lands.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QThread  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from dlss5_converter import app as gui  # noqa: E402
from dlss5_converter import pipeline  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def window(qt_app, monkeypatch):
    # Nothing may actually start: these tests are about the UI thread.
    monkeypatch.setattr(QThread, "start", lambda self, *a, **k: None)
    win = gui.MainWindow()
    win.resize(1400, 800)
    yield win
    win.deleteLater()


def frame(value: float = 0.5) -> np.ndarray:
    return np.full((64, 96, 3), value, np.float32)


def prepare(window) -> None:
    """Give the window an image and depth, without running either."""
    window.image_path = gui.Path("photo.png")
    window.prepared = pipeline.Prepared(
        source=frame(), inverse_depth=np.zeros((64, 96), np.float32), linear=frame(0.2)
    )


def result() -> pipeline.Result:
    return pipeline.Result(
        original=frame(0.3),
        enhanced=frame(0.6),
        depth_preview=np.zeros((64, 96, 3), np.uint8),
        notes="test",
    )


# -- the mistake that shipped ------------------------------------------------


def test_the_progress_methods_are_on_the_window():
    """They were defined on a dialog, which no test noticed."""
    for name in ("_begin_progress", "_end_progress", "_report_progress"):
        assert hasattr(gui.MainWindow, name), f"MainWindow is missing {name}"
        assert not hasattr(gui.FindFilesDialog, name), f"{name} leaked onto a dialog"
        assert not hasattr(gui.BatchDialog, name), f"{name} leaked onto a dialog"


def test_convert_starts_without_raising(window):
    """The exact click that did nothing in v0.1.15."""
    prepare(window)
    window.convert()
    assert window._thread is not None, "a conversion should have been started"
    assert window._view == "photo", "the sweep needs the source on screen"
    window._teardown()
    assert window._thread is None


def test_a_preview_run_does_not_hijack_the_view(window):
    """Slider drags re-run DLSS; they must not pull the user off the result."""
    prepare(window)
    window.result = result()
    window.show_view("result")
    window._start_convert(preview=True)
    assert window._view == "result"
    window._teardown()


def test_every_teardown_survives_being_called(window):
    prepare(window)
    window._teardown()
    window._style_teardown()
    gui.FindFilesDialog(window)._teardown()


# -- the rest of the main-thread surface -------------------------------------


def test_every_view_switches_without_raising(window):
    prepare(window)
    window.result = result()
    window.style_results = {0: result(), 1: result(), 2: result()}
    window._style_signature_used = window._style_signature()
    for view in ("photo", "depth", "result", "difference", "styles", "photo"):
        window.show_view(view)
        assert window._view == view


def test_views_fall_back_when_their_data_is_missing(window):
    """Clicking Result before converting must not raise or show nothing."""
    window.show_view("result")
    assert window._view == "photo"
    prepare(window)
    window.show_view("difference")
    assert window._view == "depth"


def test_the_overlay_layer_is_hidden_until_the_pointer_is_over_the_image(window):
    """The pills, readout and view bar are hover chrome, not always-on."""
    host = window.stage_host
    # At rest: nothing floating over the picture. isHidden (the explicit flag)
    # rather than isVisible, because the test window is never actually shown.
    assert host._bar.isHidden()
    assert window.wipe._chrome_opacity == 0.0

    # Pointer over the picture reveals the whole overlay in one motion.
    host._reveal(True)
    host._fade.setCurrentTime(host._fade.duration())
    assert not host._bar.isHidden()
    assert window.wipe._chrome_opacity == pytest.approx(1.0)
    assert window.side_by_side._chrome_opacity == pytest.approx(1.0)

    # Pointer gone: it fades back out and the bar stops taking clicks.
    host._reveal(False)
    host._fade.setCurrentTime(host._fade.duration())
    assert window.wipe._chrome_opacity == pytest.approx(0.0)
    assert host._bar.isHidden()


def test_the_view_controls_live_on_the_floating_bar(window):
    """The buttons moved onto the hover bar, not a row under the stage."""
    assert window.view_result.parent() is window.stage_host._bar
    assert window.view_grade.parent() is window.stage_host._bar


def test_hdr_controls_are_in_the_single_image_workflow(window):
    """Regression: HDR / display was buried in Settings and people could not
    find it. It belongs in the single-image sidebar, next to the neural look."""
    from PySide6.QtWidgets import QLabel

    labels = {
        label.text()
        for label in window.single_page.findChildren(QLabel)
    }
    assert "Paper white" in labels
    assert "Colour strength" in labels


# -- the DLSS check can never hang the app -----------------------------------
#
# The old "Checking DLSS 5" step ran the native probe on a path the user could
# not leave, and a probe that wedged the GPU froze the whole window until it was
# force-quit. RuntimeProbe replaces it: off the UI thread, watchdog-guarded, and
# reporting through one done() that no wait() ever blocks on.


def test_runtime_probe_reports_a_failure_without_blocking(qt_app):
    """A harness that cannot even launch settles quickly as not-ok."""
    from PySide6.QtTest import QTest

    seen: list[tuple[bool, str]] = []
    probe = gui.RuntimeProbe(gui.Path("no-such-harness.exe"))
    probe.done.connect(lambda ok, report: seen.append((ok, report)))
    probe.start()
    deadline = 3000
    while not seen and deadline > 0:
        QTest.qWait(20)
        deadline -= 20
    assert len(seen) == 1
    assert seen[0][0] is False


def test_runtime_probe_watchdog_settles_when_the_check_overruns(qt_app, monkeypatch):
    """A probe that overruns is given up on by the watchdog, not waited for.

    Only the watchdog timer is armed here, not the worker thread, so the timeout
    path is exercised deterministically without a real slow subprocess whose
    teardown timing would make the test flaky.
    """
    from PySide6.QtTest import QTest

    monkeypatch.setattr(gui.RuntimeProbe, "TIMEOUT_MS", 40)
    seen: list[tuple[bool, str]] = []
    probe = gui.RuntimeProbe(gui.Path("slow-harness.exe"))
    probe.done.connect(lambda ok, report: seen.append((ok, report)))
    probe._watchdog.start()
    deadline = 1000
    while not seen and deadline > 0:
        QTest.qWait(10)
        deadline -= 10
    assert len(seen) == 1, "the watchdog must settle exactly once"
    assert seen[0][0] is False
    assert "did not finish" in seen[0][1]


def test_runtime_probe_reports_only_the_first_outcome(qt_app):
    """Watchdog, worker-finish and skip all race to done(); the first wins."""
    probe = gui.RuntimeProbe(gui.Path("x.exe"))
    seen: list[tuple[bool, str]] = []
    probe.done.connect(lambda ok, report: seen.append((ok, report)))
    probe._on_timeout()                              # watchdog fires first
    probe._on_worker_finished(True, "late success")  # arrives after — ignored
    probe.skip()                                     # also ignored
    assert len(seen) == 1
    assert seen[0][0] is False


def test_cancel_probe_is_safe_when_nothing_is_running():
    """The Skip button and shutdown call this unconditionally."""
    gui.evaluator.cancel_probe()  # must not raise with no probe in flight


def test_the_progress_sweep_follows_the_pass_count(window):
    prepare(window)
    window._begin_progress()
    assert window.depth_view._progress == 0.0
    window._report_progress("DLSS 5 pass 4 of 8…")
    assert window.depth_view._progress == pytest.approx(0.5)
    # A message with no count leaves it where it was rather than resetting.
    window._report_progress("Reading the result back…")
    assert window.depth_view._progress == pytest.approx(0.5)
    window._end_progress()
    assert window.depth_view._progress is None


def test_progress_messages_are_harmless_when_no_sweep_is_running(window):
    prepare(window)
    window._report_progress("DLSS 5 pass 4 of 8…")
    assert window.depth_view._progress is None


def test_adopting_a_style_makes_it_the_result(window):
    prepare(window)
    window.style_results = {0: result(), 1: result(), 2: result()}
    window._style_signature_used = window._style_signature()
    window._adopt_style(1)
    assert window.settings.neural.style == 1
    assert window.result is window.style_results[1]


def test_the_dialogs_construct(window):
    """Each is reachable from a button, and each one has broken before."""
    gui.FindFilesDialog(window)
    gui.BatchDialog(window)
    gui.ExportDialog(window, (1920, 1080))


def test_first_run_file_finder_has_the_four_file_checklist(window):
    dialog = gui.FindFilesDialog(window, onboarding_mode=True)
    assert dialog.isModal()
    assert dialog.close_button.text() == "Skip for now"
    assert dialog.copy_button.text() == "Use these files & verify"
    for filename in gui.discovery.WANTED:
        assert filename in dialog.summary.text()


def test_tutorial_uses_real_controls_and_skip_persists(window, monkeypatch, tmp_path):
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(gui.paths, "settings_path", lambda: settings_file)
    window.settings.onboarding_version = 0

    window.start_tutorial()
    overlay = window._tour_overlay
    assert overlay is not None
    assert [step.target for step in overlay._steps] == [
        window.stack,
        window.tabs.tabBar(),
        window.neural_card,
        window.detail_card,
        window.convert_button,
    ]
    overlay._skip()

    assert window._tour_overlay is None
    assert gui.AppSettings.load(settings_file).onboarding_version == gui.ONBOARDING_VERSION
    replay_labels = [button.text() for button in window.findChildren(gui.QPushButton)]
    assert "Replay introduction…" in replay_labels


def test_boost_offers_two_four_and_eight_without_a_fixed_8k_cap(window):
    factors = [
        window.detail_supersample.itemData(index)
        for index in range(window.detail_supersample.count())
    ]
    assert factors == [2, 4, 8]
    labels = [button.text() for button in window.detail_mode.findChildren(gui.QPushButton)]
    assert "Boost" in labels
    assert "Boost 4×" not in labels


def test_an_unavailable_depth_model_falls_back_to_small_without_crashing(window, monkeypatch):
    """Base/Large have no ONNX download source; picking one must fall back to the
    bundled Small model with a clear message, not crash the old download path."""
    from dlss5_converter import app as gui
    from dlss5_converter.onnx_depth import SMALL, OnnxDepthEngine

    monkeypatch.setattr(
        OnnxDepthEngine, "is_downloaded",
        classmethod(lambda cls, model_id: model_id == SMALL),
    )
    told: list[bool] = []
    monkeypatch.setattr(gui.QMessageBox, "information", lambda *a, **k: told.append(True))

    window.ensure_model_downloaded("depth-anything/Depth-Anything-V2-Large-hf")

    assert window.settings.depth.model_id == SMALL
    assert told, "the user must be told it fell back to Small"
    assert window.model_box.currentData() == SMALL


def test_boost_levels_read_as_plain_names(window):
    """No 2×/4×/8× jargon in the sharpness control - Standard/High/Max instead."""
    names = [
        window.detail_supersample.itemText(i)
        for i in range(window.detail_supersample.count())
    ]
    assert names == ["Standard", "High", "Max"]


def test_boost_guard_disables_levels_that_overflow_the_texture_limit(window):
    """At a high Max size the big levels are greyed and the choice steps down.

    This is jerkalerk's InvalidParameter error turned into a prevented, spoken
    state: 8192 px only leaves room for the smallest level (8192×2 = 16384).
    """
    window.settings.detail.mode = "boost"
    window.settings.detail.supersample = 8  # "Max"
    window.settings.evaluation.max_edge = 8192
    window._sync_boost_guard()

    # Stepped down to the only level that fits, and said so. (isVisible() is
    # False in an unshown test window, so assert the message text instead.)
    assert window.settings.detail.supersample == 2
    assert window.detail_guard.text() != ""
    assert not window.detail_guard.isHidden()
    model = window.detail_supersample.model()
    enabled = [model.item(i).isEnabled() for i in range(window.detail_supersample.count())]
    assert enabled == [True, False, False]  # Standard fits; High and Max do not


def test_boost_guard_reports_when_nothing_fits(window):
    """Above 8192 px even the smallest level overflows: Boost cannot supersample."""
    window.settings.detail.mode = "boost"
    window.settings.evaluation.max_edge = 12000
    window._sync_boost_guard()
    assert "8192" in window.detail_guard.text()
    assert not window.detail_guard.isHidden()


# -- what a run looks like while it is running -------------------------------


def test_a_preview_run_sweeps_the_after_half(window):
    """A slider nudge must show work happening, without moving the view.

    The previous result stays in front of you and only the after half is
    recomputed - that is the whole reason to be on this view, and an earlier
    version threw it away by switching to the source.
    """
    prepare(window)
    window.result = result()
    window._succeeded(window.result)
    window.show_view("result")

    window._start_convert(preview=True)
    assert window._view == "result", "a preview must not move the view"
    assert window.wipe._progress == 0.0
    window._report_progress("DLSS 5 pass 6 of 8")
    assert window.wipe._progress == pytest.approx(0.75)
    window._teardown()
    assert window.wipe._progress is None


def test_the_wipe_names_its_halves(window):
    prepare(window)
    window.result = result()
    window.show_view("result")
    # SOURCE / DLSS 5, matching the mockup's on-image pills, with the result
    # half accented so which side is which is clear before the divider moves.
    assert window.wipe._labels == ("SOURCE", "DLSS 5")
    assert window.wipe._accent_right is True


def test_compare_styles_shows_its_panes_before_converting(window):
    """Not a full-screen wait that snaps to panels at the end."""
    prepare(window)
    window.style_count_box.setCurrentIndex(1)  # three panes
    window.show_view("styles")

    assert window._view == "styles"
    assert window.side_by_side.count() == 3
    assert window.side_by_side._labels == ["Original", "Natural", "Cinematic"]
    assert not window.view_styles.isEnabled(), "disabled while it runs"


def test_both_style_panes_grey_the_instant_a_run_starts(window):
    """A stale pane must not read as finished while its restatement is queued.

    Reported from the live app: changing a setting greyed the panes one at a
    time, as each conversion reached them, so the pane still waiting showed its
    previous colour result - which no longer represented what was coming.
    """
    prepare(window)
    window.style_count_box.setCurrentIndex(1)  # Original / Natural / Cinematic
    window.show_view("styles")

    # Original stays colour (never converts); both style panes are grey at 0.0.
    assert window.side_by_side._progress == [None, 0.0, 0.0]


def test_each_pane_returns_to_colour_only_when_its_own_style_lands(window):
    """The styles convert one after another, so colour returns one at a time."""
    prepare(window)
    window.style_count_box.setCurrentIndex(1)
    window.show_view("styles")

    # Style indices are NR_STYLES = (Default, Natural, Cinematic); the default
    # three-pane view shows Original, Natural (1), Cinematic (2).
    window._style_started(1)
    window._report_progress("DLSS 5 pass 4 of 8")
    # Natural sweeping; Cinematic still fully grey, not blank.
    assert window.side_by_side._progress == [None, pytest.approx(0.5), 0.0]

    window._style_one_done(1, result())
    assert window.side_by_side._progress[1] is None, "Natural is done, full colour"
    assert window.side_by_side._progress[2] == 0.0, "Cinematic still waiting, grey"

    window._style_started(2)
    window._report_progress("DLSS 5 pass 2 of 8")
    assert window.side_by_side._progress == [None, None, pytest.approx(0.25)]


def test_finishing_a_comparison_clears_up(window):
    prepare(window)
    window.show_view("styles")
    window._styles_ready({0: result(), 1: result(), 2: result()})
    assert window._view == "styles"
    assert window.side_by_side._progress == [None, None]
    assert window.view_styles.isEnabled(), "the button must come back"


def test_a_settings_change_re_runs_the_comparison_in_place(window):
    """The reported bug: it fell through to a single conversion and left.

    That switched to the result view mid-comparison and, because the button
    was keyed on a thread that was never cleared, Compare styles then stayed
    disabled for the rest of the session.
    """
    prepare(window)
    window.style_results = {0: result(), 1: result(), 2: result()}
    window._style_signature_used = window._style_signature()
    window.show_view("styles")
    window._styles_ready(window.style_results)

    window.settings.evaluation.live_preview = True
    window._run_preview()

    assert window._view == "styles", "must not be thrown out of the comparison"
    assert window._thread is not None, "and it should be redoing both styles"
    window._style_teardown()
    assert window.view_styles.isEnabled()


# -- full-resolution inspection (paraDiXson's report) ------------------------


def big_result(w=1600, h=900):
    import numpy as np
    return pipeline.Result(
        original=np.zeros((h, w, 3), np.float32),
        enhanced=np.full((h, w, 3), 0.6, np.float32),
        depth_preview=np.zeros((h, w, 3), np.uint8),
        notes="big",
    )


def test_result_view_shows_full_resolution_not_a_preview(window):
    """The report: the preview was capped at 1200 px, so detail could not be
    checked even by zooming. The idle view must hold the real pixels."""
    window.image_path = gui.Path("photo.png")
    window._succeeded(big_result(1600, 900))
    assert window.wipe._after.width() == 1600, "full res, not the 1200 preview"


def test_a_grade_drag_drops_to_preview_then_sharpens(window):
    window.image_path = gui.Path("photo.png")
    window._succeeded(big_result(1600, 900))
    window.settings.grade.exposure = 0.4
    window._render_result(fast=True)
    assert window.wipe._after.width() <= 1200, "fast during a drag"
    window._render_result(fast=False)
    assert window.wipe._after.width() == 1600, "sharp once settled"


def test_swapping_resolution_keeps_the_zoom(window):
    """Zoom set to inspect something must survive a grade tweak."""
    from PySide6.QtCore import QPointF
    window.image_path = gui.Path("photo.png")
    window._succeeded(big_result(1600, 900))
    window.wipe._zoom = 4.0
    window.wipe._pan = QPointF(20.0, 10.0)
    window._render_result(fast=True)   # preview swap
    assert window.wipe._zoom == 4.0, "a grade drag must not reset the zoom"
    window._render_result(fast=False)  # full swap
    assert window.wipe._zoom == 4.0


def test_re_converting_the_same_size_keeps_the_zoom(window):
    """Live preview re-converts on a neural-slider tweak; staying zoomed to
    compare the same spot is the point, so same-size results keep the view."""
    window.image_path = gui.Path("photo.png")
    window._succeeded(big_result(1600, 900))
    window.wipe._zoom = 3.0
    window._succeeded(big_result(1600, 900))  # a re-convert at the same size
    assert window.wipe._zoom == 3.0, "a same-size re-convert holds the zoom"


def test_a_different_size_result_refits(window):
    window.image_path = gui.Path("photo.png")
    window._succeeded(big_result(1600, 900))
    window.wipe._zoom = 3.0
    window._succeeded(big_result(1280, 720))  # a genuinely different image
    assert window.wipe._zoom == 1.0, "a different size starts fitted"


def test_video_effort_does_not_touch_the_sidebar_passes(window, monkeypatch):
    """A video export used to write its pass count to the shared setting, so
    afterwards single-image conversions ran at 1 pass. The video path must use
    its own copy and leave settings.evaluation.frames alone."""
    from PySide6.QtWidgets import QFileDialog
    from PySide6.QtCore import QThread

    window.settings.evaluation.frames = 8   # what the user set in the sidebar

    page = window.video_page
    # Pretend a clip is loaded and PyAV is available.
    class _Info:
        frames, fps, duration = 100, 30.0, 3.3
    page.source = gui.Path("clip.mp4")
    page.info = _Info()
    page.output_path = gui.Path("out.mp4")
    page.mode_box.setCurrentIndex(0)  # Quick (1 pass)

    monkeypatch.setattr(gui.video, "is_available", lambda: True)
    started = {}
    real_worker = gui.VideoWorker
    def capture(*a, **k):
        started["settings"] = a[2]  # settings is the 3rd positional arg
        w = real_worker(*a, **k)
        return w
    monkeypatch.setattr(gui, "VideoWorker", capture)
    monkeypatch.setattr(QThread, "start", lambda self, *a, **k: None)

    window._start_video()

    assert window.settings.evaluation.frames == 8, "sidebar passes must be untouched"
    assert started["settings"].evaluation.frames == 1, "the video run used its own 1 pass"
    window._video_teardown()


def test_changing_video_effort_does_not_touch_the_sidebar(window):
    window.settings.evaluation.frames = 8
    window.video_page.mode_box.setCurrentIndex(1)  # Quality
    window._video_mode_changed()
    assert window.settings.evaluation.frames == 8
