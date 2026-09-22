"""Regression tests for the v0.4.1 bug fixes (GitHub issues #9-#12)."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_restoring_saved_lut_does_not_fire_during_construction(qt_app):
    """#9: a saved lut_enabled=true fired the change handler mid-construction."""
    from dlss5_converter.app import EffectsPage
    from dlss5_converter.settings import EffectsSettings

    fx = EffectsSettings()
    fx.lut_enabled = True
    calls = []
    page = EffectsPage(fx, lambda *a: calls.append(a))
    assert calls == []
    assert page._groups["lut_enabled"].isChecked()


def test_selftest_report_survives_an_unwritable_stderr(tmp_path, monkeypatch):
    """#11: a windowed exe cannot write a redirected stderr; the report must
    still be complete, in report.txt beside the exe."""
    import sys
    from dlss5_converter import paths, selftest

    class Broken:
        def write(self, _text):
            raise OSError(22, "Invalid argument")

        def flush(self):
            raise OSError(22, "Invalid argument")

    monkeypatch.setattr(paths, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(selftest, "_REPORT", None)
    monkeypatch.setattr(sys, "stderr", Broken())
    selftest._line("first line")
    selftest._line("second line")
    selftest._REPORT.close()
    assert (tmp_path / "report.txt").read_text(encoding="utf-8").splitlines() == [
        "first line", "second line"]


def test_throwaway_window_does_not_start_onboarding(qt_app, monkeypatch):
    """#12: the self-test's window ran first-run onboarding on a deleted
    window (crash) and marked onboarding done for the real first launch."""
    from dlss5_converter import app as app_mod

    started = []
    monkeypatch.setattr(app_mod.MainWindow, "_start_initial_setup",
                        lambda self: started.append(True))
    w = app_mod.MainWindow(first_run_setup=False)
    for _ in range(5):
        qt_app.processEvents()
    assert started == []
    w.deleteLater()
    qt_app.processEvents()
    assert app_mod._alive(object()) in (True, False)   # never raises


def test_standard_reshade_build_is_named_from_the_log(tmp_path):
    """#10: the non add-on ReShade build is identified from ReShade.log (the DLL
    cannot tell: both builds carry the same sentence as UI text)."""
    from dlss5_converter import runtime

    log = tmp_path / "ReShade.log"
    log.write_text("18:04 | WARN  | Skipped loading add-on from 'x.addon64' because this "
                   "build of ReShade has only limited add-on functionality.\n")
    assert runtime.reshade_log_refused_addons(log)
    ok = tmp_path / "ok.log"
    ok.write_text("18:04 | INFO  | Registered add-on \"DLSS 5 Neural Rendering\"\n"
                  "UI: This build of ReShade has only limited add-on functionality.\n")
    assert not runtime.reshade_log_refused_addons(ok)
    assert not runtime.reshade_log_refused_addons(tmp_path / "missing.log")


def test_save_completion_runs_on_the_ui_thread(qt_app, tmp_path):
    """The save's finished handler must run on the UI thread: it closes the
    progress dialog and updates the window. On the worker thread it could hang
    the app right after the file was written (large PNG save report)."""
    import threading
    import time
    from dlss5_converter import app as app_mod

    w = app_mod.MainWindow(first_run_setup=False)
    ui = threading.get_ident()
    seen = {}
    out = tmp_path / "x.png"

    def job():
        out.write_bytes(b"png")
        return out

    def message(saved):
        seen["ui"] = threading.get_ident() == ui
        seen["saved"] = saved
        return "done"

    w._start_save([("Saving…", job)], message)
    deadline = time.time() + 10
    while "ui" not in seen and time.time() < deadline:
        qt_app.processEvents()
        time.sleep(0.01)
    assert seen.get("ui") is True
    assert seen["saved"] == [out]
    assert w._save_dialog is None           # progress dialog was taken down
    assert w.save_button.isEnabled()
    w.deleteLater()
    qt_app.processEvents()
