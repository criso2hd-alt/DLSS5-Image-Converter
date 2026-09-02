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
