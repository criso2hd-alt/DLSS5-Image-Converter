"""What the app says when the harness dies without explaining itself.

A handled DLSS failure arrives as an ``ERROR`` line and is surfaced verbatim.
Everything else is a crash, and for a long time all the user was told was "the
harness exited unexpectedly" - which cannot distinguish an out-of-date driver
from the GPU running out of memory, and those have opposite fixes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from dlss5_converter.evaluator import Harness, HarnessError
from dlss5_converter.settings import NeuralSettings


def harness(width: int = 3840, height: int = 2160) -> Harness:
    return Harness(
        Path("dlss5_eval.exe"),
        width=width,
        height=height,
        depth_path=Path("depth.bin"),
        motion_path=Path("motion.bin"),
        neural=NeuralSettings(),
        frames=1,
    )


def dead(code: int, stderr: str = "") -> subprocess.Popen[str]:
    """A process that has already exited with `code`."""
    script = f"import sys; sys.stderr.write({stderr!r}); sys.exit({code})"
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    process.wait(timeout=30)
    return process


def test_the_size_is_named_because_it_is_the_usual_cause():
    """A crash at 4K that would not happen at 1080p is the common shape."""
    live = harness()
    live._process = dead(1)
    message = live._died()
    assert "3840x2160" in message


def test_a_known_crash_code_is_explained():
    live = harness()
    live._process = dead(-1073741819)  # 0xC0000005, an access violation
    message = live._died()
    assert "0xC0000005" in message
    assert "Max size" in message, "should suggest something actionable"


def test_an_unknown_code_still_reports_the_number():
    """Anything unrecognised is still identifiable in a bug report."""
    live = harness()
    live._process = dead(42)
    assert "0x0000002A" in live._died()


def test_stderr_is_kept_when_the_process_managed_to_write_any():
    live = harness()
    live._process = dead(1, stderr="d3d12: something went wrong")
    assert "d3d12: something went wrong" in live._died()


def test_a_harness_that_was_never_started_says_so():
    live = harness()
    assert "not running" in live._died()


def test_read_surfaces_the_diagnosis():
    live = harness()
    live._process = dead(-1073741819)
    with pytest.raises(HarnessError) as caught:
        live._read()
    assert "0xC0000005" in str(caught.value)


def test_a_crash_before_ready_points_at_the_driver(monkeypatch):
    """Dying before READY means DLSS crashed while creating its feature (the
    Discord report: nvngx.log ended at CreateFeature_Validate). The files are
    fine; the usual cause is the driver, so say that and name the version."""
    from dlss5_converter import hardware
    monkeypatch.setattr(hardware, "query_driver_version", lambda: "999.99")
    live = harness()
    live._process = dead(-1073741819)
    live._starting = True
    message = live._died()
    assert "while starting" in message and "driver" in message
    assert "999.99" in message
    assert "0xC0000005" in message          # the raw evidence is still there


def test_a_crash_mid_conversion_does_not_blame_the_driver():
    live = harness()
    live._process = dead(-1073741819)
    assert "while starting" not in live._died()


def test_a_backend_logging_before_ready_is_skipped():
    """OptiScaler can log to the console, which is our protocol with the
    harness. Its first line used to read as "did not start cleanly"."""
    import sys
    from dlss5_converter.evaluator import Harness
    script = (
        "print('[15:21:36.145] [info] Config::Reload loading ini');"
        "print('[2026-09-22] [NGXLoadConfig:1145] [dlss]');"
        "print('READY DLSS feature created in DLAA mode');"
    )
    live = harness()
    live._command = [sys.executable, "-c", script]
    live._spawn(live._command)
    assert live.notes == "DLSS feature created in DLAA mode"
    assert len(live.startup_noise) == 2
    live.__exit__(None, None, None)


def test_a_harness_that_only_logs_still_fails():
    import sys
    live = harness()
    live._command = [sys.executable, "-c", "print('[info] nothing to say')"]
    with pytest.raises(HarnessError) as error:
        live._spawn(live._command)
    assert "no READY line" in str(error.value) or "while starting" in str(error.value)
    live.__exit__(None, None, None)


def test_the_probe_report_drops_a_backends_logging():
    from dlss5_converter.evaluator import _harness_fields
    raw = ("[15:21:36] [info] Util::DllPath E:\...\dxgi.dll\n"
           "[2026-09-22 15:21:36] [NGXLoadConfig:1151] app_E658700=310.9.0.0\n"
           "adapter: NVIDIA GeForce RTX 4080\n"
           "dlss_available: 1\n"
           "test_evaluation: ok\n")
    kept = _harness_fields(raw)
    assert kept.splitlines() == ["adapter: NVIDIA GeForce RTX 4080",
                                 "dlss_available: 1", "test_evaluation: ok"]
    # Nothing but logging: keep it rather than reporting an empty check.
    assert "[info]" in _harness_fields("[info] only noise here\n")
