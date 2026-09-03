"""Make a crash leave evidence, because a windowed build swallows all of it.

A PyInstaller ``--windowed`` app has no console, and ``main`` redirects stdout
and stderr to the null device so nothing that writes to them dies. The cost of
that is total silence on failure: an unhandled exception, or a native crash from
a bad DLL, ends the process with no message at all. Every bug report so far has
been the words "crashes to desktop" and nothing to act on.

This writes any crash - Python *or* native - to ``crash.log`` beside the app, and
shows the user where it is. Three layers, because they catch different things:

- ``faulthandler`` dumps a C-level stack on a hard crash (an access violation,
  the shape a clashing FFmpeg DLL takes). Python's own hooks never see those.
- ``sys.excepthook`` catches an unhandled exception on the main thread, which is
  what actually terminates the app.
- ``threading.excepthook`` catches one on a worker thread, which usually only
  kills the thread - but leaves a record of why a conversion silently stopped.
"""

from __future__ import annotations

import datetime as _dt
import faulthandler
import sys
import threading
import traceback
from pathlib import Path

_log_path: Path | None = None
_fault_file = None


def log_path() -> Path:
    """Where the crash log lives: beside the app, where a user can find it."""
    from . import paths

    try:
        base = paths.data_dir()
    except Exception:  # noqa: BLE001 - paths must never be why logging fails
        base = Path.home()
    base.mkdir(parents=True, exist_ok=True)
    return base / "crash.log"


def _write(header: str, body: str) -> None:
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(log_path(), "a", encoding="utf-8") as handle:
            handle.write(f"\n===== {header} @ {stamp} =====\n{body}\n")
    except Exception:  # noqa: BLE001 - logging a crash must not raise
        pass


def install() -> None:
    """Turn on all three layers. Safe to call once, early, before anything else."""
    global _log_path, _fault_file
    _log_path = log_path()

    # faulthandler needs a real file handle kept open for the process lifetime;
    # it writes the native stack straight to it at fault time, when the Python
    # interpreter may be too broken to do anything fancier.
    try:
        _fault_file = open(_log_path, "a", encoding="utf-8")
        _fault_file.write(
            f"\n===== session started {_dt.datetime.now():%Y-%m-%d %H:%M:%S} =====\n"
        )
        _fault_file.flush()
        faulthandler.enable(file=_fault_file, all_threads=True)
    except Exception:  # noqa: BLE001
        pass

    previous = sys.excepthook

    def hook(kind, value, tb):
        _write("unhandled exception", "".join(traceback.format_exception(kind, value, tb)))
        _notify(value)
        previous(kind, value, tb)

    sys.excepthook = hook

    def thread_hook(args):
        _write(
            f"unhandled exception in thread {args.thread.name}",
            "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)),
        )

    threading.excepthook = thread_hook


def _notify(value: BaseException) -> None:
    """Tell the user where the log is, if a UI is up. Never raise doing it."""
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        if QApplication.instance() is None:
            return
        box = QMessageBox(
            QMessageBox.Icon.Critical,
            "DLSS 5 Converter crashed",
            f"{type(value).__name__}: {value}\n\n"
            f"A crash log was written to:\n{log_path()}\n\n"
            "Please attach it to a bug report - it has the detail that "
            '"crashed to desktop" does not.',
        )
        box.exec()
    except Exception:  # noqa: BLE001 - a crash while reporting a crash helps no one
        pass
