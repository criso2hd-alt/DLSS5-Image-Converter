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


def test_a_preview_run_shows_the_reveal(window):
    """Live preview now plays the point-cloud reveal (by request): it moves to the
    photo view and runs the cloud, landing back on the result when it finishes."""
    prepare(window)
    window.result = result()
    window.show_view("result")
    window._start_convert(preview=True)
    assert window._view == "photo"
    assert window.depth_view.cloud_active()
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


def test_live_preview_runs_the_same_reveal(window):
    # Live preview shares the point-cloud reveal with a deliberate Convert (by
    # request), rather than the old grey sweep.
    prepare(window)
    window._previewing = True
    window._begin_progress()
    assert window.depth_view.cloud_active()
    assert window.depth_view._progress is None  # not the grey sweep
    window._end_progress()  # not finishing → stops cleanly
    assert not window.depth_view.cloud_active()


def test_the_depth_view_runs_the_point_cloud_reveal(window):
    # A full Convert runs the depth point-cloud reveal, which self-animates
    # rather than tracking the pass count (see reveal.py). It must not fall back
    # to the grey progress sweep.
    prepare(window)
    window._begin_progress()
    assert window.depth_view.cloud_active()
    assert window.depth_view._progress is None
    window._report_progress("DLSS 5 pass 4 of 8…")  # ignored by the cloud
    assert window.depth_view.cloud_active()
    window._end_progress()
    assert not window.depth_view.cloud_active()


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


def test_detail_offers_off_boost_and_ultra(window):
    """The three modes, no level control: Boost and Ultra size themselves."""
    labels = [button.text() for button in window.detail_mode.findChildren(gui.QPushButton)]
    assert "Off" in labels
    assert "Boost" in labels
    assert "Ultra Detail" in labels
    # The retired 2×/4×/8× sharpness control is gone entirely.
    assert not hasattr(window, "detail_supersample")


def test_detail_hint_tracks_the_selected_mode(window):
    """Each mode's explainer line is set when the mode is synced."""
    window.settings.detail.mode = "ultra"
    window._sync_detail_controls()
    assert "tiles" in window.detail_hint.text().lower()
    window.settings.detail.mode = "off"
    window._sync_detail_controls()
    assert "plain dlss" in window.detail_hint.text().lower()


def test_ultra_size_row_shows_only_in_ultra_and_reports_pixels(window):
    """The size multiplier + result line appear only for Ultra, and the line
    reports the real pixel size once an image is loaded."""
    prepare(window)  # 96×64 source
    window.settings.detail.mode = "boost"
    window._sync_detail_controls()
    assert not window.detail_ultra_row.isVisibleTo(window.detail_card)

    window.settings.detail.mode = "ultra"
    idx = window.detail_ultra_factor.findData(4.0)
    window.detail_ultra_factor.setCurrentIndex(idx)
    window._sync_detail_controls()
    # 96×64 at 4× → 384×256.
    assert "384×256" in window.detail_ultra_size.text()
    assert window.settings.detail.ultra_factor == 4.0


def test_ultra_save_moves_only_the_full_res_file(window, tmp_path):
    """Ultra save moves the streamed super-resolution file into place and writes
    NOTHING else — the soft downscaled native copy was dropped on purpose."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    src = scratch / "ultra_full.tiff"
    src.write_bytes(b"BIGTIFFDATA")  # stand-in for the streamed file
    window.result = pipeline.Result(
        original=frame(0.3),
        enhanced=frame(0.6),
        depth_preview=np.zeros((64, 96, 3), np.uint8),
        notes="test",
        ultra_full_path=src,
        ultra_full_size=(384, 256),
    )
    super_dest = tmp_path / "shot_ultra_2K.tiff"
    super_path = window._save_ultra(super_dest)

    assert super_path.exists() and super_path.read_bytes() == b"BIGTIFFDATA"
    assert not src.exists()                                  # moved, not copied
    assert not (tmp_path / "shot_native.png").exists()       # no native copy
    assert window.result.ultra_full_path == super_dest       # view chip still resolves


def test_ultra_save_reencodes_to_png_on_the_fly(window, tmp_path):
    """Saving Ultra as a non-TIFF re-encodes the scratch BigTIFF to that type, so
    people who can't open a TIFF still get a usable file. The scratch file stays."""
    tifffile = pytest.importorskip("tifffile")
    from dlss5_converter import bigtiff
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    src = scratch / "ultra_full.tiff"
    img = np.full((40, 60, 3), 0.5, np.float32)
    bigtiff.write_streaming(src, 40, 60, iter([(0, img)]), bits=16)
    window.result = pipeline.Result(
        original=frame(0.3), enhanced=frame(0.6),
        depth_preview=np.zeros((64, 96, 3), np.uint8), notes="test",
        ultra_full_path=src, ultra_full_size=(60, 40),
    )
    dest = tmp_path / "shot_ultra_0K.png"
    out = window._save_ultra(dest)
    assert out.exists() and out.suffix == ".png"
    assert src.exists()                                       # scratch kept for re-save
    import cv2
    decoded = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert decoded is not None and decoded.shape[:2] == (40, 60)


# -- what a run looks like while it is running -------------------------------


def test_a_preview_run_plays_the_cloud_reveal(window):
    """A slider nudge shows work happening via the point-cloud reveal on the
    photo view; progress messages no longer drive a grey sweep, and teardown of
    a run that is not finishing stops the cloud cleanly.
    """
    prepare(window)
    window.result = result()
    window._succeeded(window.result)

    window._start_convert(preview=True)
    assert window._view == "photo"
    assert window.depth_view.cloud_active()
    window._report_progress("DLSS 5 pass 6 of 8")  # ignored by the cloud
    assert window.depth_view.cloud_active()
    window._teardown()
    assert not window.depth_view.cloud_active()


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

    # Original stays colour (never converts); both style panes reveal as the cloud.
    assert window.side_by_side._reveal == [False, True, True]


def test_each_pane_returns_to_colour_only_when_its_own_style_lands(window):
    """The styles convert one after another, so colour returns one at a time."""
    prepare(window)
    window.style_count_box.setCurrentIndex(1)
    window.show_view("styles")

    # Style indices are NR_STYLES = (Default, Natural, Cinematic); the default
    # three-pane view shows Original, Natural (1), Cinematic (2).
    window._style_started(1)
    window._report_progress("DLSS 5 pass 4 of 8")
    # Natural revealing; Cinematic also still revealing (waiting its turn).
    assert window.side_by_side._reveal == [False, True, True]

    window._style_one_done(1, result())
    assert window.side_by_side._reveal[1] is False, "Natural is done, full colour"
    assert window.side_by_side._reveal[2] is True, "Cinematic still waiting, as cloud"

    window._style_started(2)
    window._report_progress("DLSS 5 pass 2 of 8")
    assert window.side_by_side._reveal == [False, False, True]


def test_finishing_a_comparison_clears_up(window):
    prepare(window)
    window.show_view("styles")
    window._styles_ready({0: result(), 1: result(), 2: result()})
    assert window._view == "styles"
    assert window.side_by_side._reveal == [False, False]
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
