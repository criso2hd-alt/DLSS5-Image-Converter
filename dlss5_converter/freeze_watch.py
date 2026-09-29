"""Freeze detector: when the window stops responding, write down where it is stuck.

A freeze reported as "it went frozen and I had to close it" leaves nothing to
go on, and it may not happen again on the next try. So a timer on the UI
thread ticks a heartbeat, and a plain background thread watches it. If the
heartbeat stops for FREEZE_SECONDS, the watcher writes every thread's Python
stack to a log (faulthandler works from any thread, even while the UI thread
is stuck inside C code), then notes how long the stall lasted if it recovers.

It costs one timer tick every half second. Nothing is written unless the UI
actually stalls.
"""

from __future__ import annotations

import faulthandler
import threading
import time
from pathlib import Path

#: How long the UI thread may go without a heartbeat before it counts as frozen.
FREEZE_SECONDS = 10.0
#: How many freeze logs are kept; older ones are deleted.
KEEP_LOGS = 10


class FreezeWatch:
    def __init__(self, log_dir: Path, freeze_seconds: float = FREEZE_SECONDS,
                 version: str = "") -> None:
        self.log_dir = Path(log_dir)
        self.freeze_seconds = float(freeze_seconds)
        self.version = version
        self.last_log: Path | None = None
        self._beat = time.monotonic()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._timer = None

    def beat(self) -> None:
        """Called on the UI thread: proof it is still processing events."""
        self._beat = time.monotonic()

    def start(self, parent=None) -> None:
        from PySide6.QtCore import QTimer

        self._timer = QTimer(parent)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self.beat)
        self._timer.start()
        self.beat()
        self._thread = threading.Thread(target=self._watch, name="freeze-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._timer is not None:
            self._timer.stop()

    def _watch(self) -> None:
        stalled_since: float | None = None
        while not self._stop.wait(1.0):
            quiet = time.monotonic() - self._beat
            if stalled_since is None and quiet >= self.freeze_seconds:
                stalled_since = self._beat
                self._write_stacks(quiet)
            elif stalled_since is not None and quiet < self.freeze_seconds:
                self._note_recovered(time.monotonic() - stalled_since)
                stalled_since = None

    def _write_stacks(self, quiet: float) -> None:
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            path = self.log_dir / time.strftime("freeze_%Y%m%d_%H%M%S.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(f"DLSS 5 Converter {self.version}: the window stopped responding "
                             f"for {quiet:.0f} s. Stacks of every thread at that moment "
                             "(the MainThread one is where the UI is stuck):\n\n")
                handle.flush()
                faulthandler.dump_traceback(file=handle, all_threads=True)
            self.last_log = path
            self._prune()
        except Exception:  # noqa: BLE001 - a diagnostic must never add a failure
            pass

    def _note_recovered(self, seconds: float) -> None:
        if self.last_log is None:
            return
        try:
            with open(self.last_log, "a", encoding="utf-8") as handle:
                handle.write(f"\nThe window recovered after about {seconds:.0f} s.\n")
        except Exception:  # noqa: BLE001
            pass

    def _prune(self) -> None:
        logs = sorted(self.log_dir.glob("freeze_*.txt"))
        for old in logs[:-KEEP_LOGS]:
            try:
                old.unlink()
            except OSError:
                pass
