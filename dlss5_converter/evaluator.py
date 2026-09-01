"""Drive the native DLSS harness over a line protocol.

The harness is a long-lived subprocess rather than one invocation per frame, and
that is a correctness requirement rather than an optimisation: DLSS's temporal
history lives inside the NGX feature handle. Re-launching per frame would reset
the accumulator every time and make ``frames > 1`` do exactly nothing.

Protocol, one line each way, UTF-8:

    <- READY <notes>
    -> FRAME <colour.bin> <jitter_x> <jitter_y> <reset 0|1>
    <- FRAME_OK <index>
    -> WRITE <out.bin>
    <- WRITE_OK <bytes>
    -> QUIT
    <- BYE

Paths are sent unquoted and may contain spaces — a release under
``C:\\Program Files`` does, and so does anything under a user name with a space
in it. The harness parses the fixed numeric fields off the *end* of the line for
exactly this reason, so a path is whatever precedes them. Do not add quoting
here without changing ``SplitTrailingFields`` in ``main.cpp`` to strip it.

Any line beginning ``ERROR `` aborts the run and is surfaced verbatim; the
harness has far more context about an NGX failure than we do.
"""

from __future__ import annotations

import atexit
import subprocess
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType

from .settings import NeuralSettings

# Every harness process currently alive, so shutdown can end them.
#
# An orphaned harness is not a tidy-up detail: it holds a D3D12 device, keeps a
# 158 MB neural DLL mapped, and locks the files in the folder it was launched
# from — which is enough to make the next rebuild of the release fail with a
# permission error nowhere near the real cause. The GUI kills these on close and
# atexit is the backstop for every other way the interpreter can go down.
_LIVE: set[subprocess.Popen] = set()
_LIVE_LOCK = threading.Lock()


def _register(process: subprocess.Popen) -> None:
    with _LIVE_LOCK:
        _LIVE.add(process)


def _unregister(process: subprocess.Popen) -> None:
    with _LIVE_LOCK:
        _LIVE.discard(process)


def terminate_all(timeout: float = 5.0) -> int:
    """End every running harness. Returns how many were still alive.

    Kill rather than terminate: the harness is mid-DLSS-evaluation on the GPU
    and has no signal handler to run, so there is nothing to unwind gracefully
    and a polite request only delays the exit.
    """
    with _LIVE_LOCK:
        processes = list(_LIVE)
        _LIVE.clear()
    ended = 0
    for process in processes:
        try:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=timeout)
                ended += 1
        except Exception:  # noqa: BLE001 - shutdown must not raise
            pass
    return ended


atexit.register(terminate_all)


class HarnessError(RuntimeError):
    pass


class Harness(AbstractContextManager["Harness"]):
    """One live DLSS feature, fed frame by frame."""

    def __init__(
        self,
        exe: Path,
        *,
        width: int,
        height: int,
        depth_path: Path,
        motion_path: Path,
        neural: NeuralSettings,
        frames: int,
    ) -> None:
        self._exe = exe
        self._width = width
        self._height = height
        self._index = 0
        self._command = [
            str(exe),
            "--width", str(width),
            "--height", str(height),
            "--frames", str(frames),
            "--depth", str(depth_path),
            "--motion", str(motion_path),
            # Depth Anything's output is near-at-1.0, which is the reversed-Z
            # convention. See contract.to_hardware_depth.
            "--reversed-depth",
            "--intensity", f"{neural.intensity:.4f}",
            "--skin", f"{neural.skin:.4f}",
            "--local-tone", f"{neural.local_tone:.4f}",
            "--structure", f"{neural.structure:.4f}",
        ]
        self._process: subprocess.Popen[str] | None = None
        self.notes = ""

    def __enter__(self) -> Harness:
        try:
            self._process = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                # The harness owns a D3D12 device and a window; a console
                # flashing up behind the GUI on every conversion is noise.
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                cwd=str(self._exe.parent),
            )
        except OSError as error:
            raise HarnessError(f"Could not start {self._exe.name}: {error}") from error

        _register(self._process)
        line = self._read()
        if not line.startswith("READY"):
            raise HarnessError(f"Harness did not start cleanly: {line}")
        self.notes = line[len("READY") :].strip()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write("QUIT\n")
                process.stdin.flush()
                process.wait(timeout=10)
        except Exception:  # noqa: BLE001 - shutdown must not mask the real error
            pass
        finally:
            if process.poll() is None:
                process.kill()
                try:
                    process.wait(timeout=5)
                except Exception:  # noqa: BLE001 - nothing left to salvage
                    pass
            _unregister(process)

    # -- protocol ------------------------------------------------------------

    def _read(self) -> str:
        process = self._process
        if process is None or process.stdout is None:
            raise HarnessError("The harness is not running.")
        line = process.stdout.readline()
        if not line:
            stderr = ""
            if process.stderr is not None:
                stderr = process.stderr.read() or ""
            raise HarnessError(
                "The harness exited unexpectedly"
                + (f":\n{stderr.strip()}" if stderr.strip() else ".")
            )
        line = line.strip()
        if line.startswith("ERROR"):
            raise HarnessError(line[len("ERROR") :].strip() or "Unknown DLSS failure.")
        return line

    def _send(self, line: str) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise HarnessError("The harness is not running.")
        try:
            process.stdin.write(line + "\n")
            process.stdin.flush()
        except OSError as error:
            raise HarnessError(f"Lost contact with the harness: {error}") from error

    def frame(self, colour_path: Path, jitter: tuple[float, float]) -> None:
        """Evaluate one frame. The first resets DLSS's history."""
        reset = 1 if self._index == 0 else 0
        self._send(f"FRAME {colour_path} {jitter[0]:.6f} {jitter[1]:.6f} {reset}")
        response = self._read()
        if not response.startswith("FRAME_OK"):
            raise HarnessError(f"Unexpected reply to FRAME: {response}")
        self._index += 1

    def write(self, out_path: Path) -> None:
        self._send(f"WRITE {out_path}")
        response = self._read()
        if not response.startswith("WRITE_OK"):
            raise HarnessError(f"Unexpected reply to WRITE: {response}")


def probe(exe: Path) -> str:
    """Ask the harness what the installed DLSS runtime can actually do.

    Used by the settings panel and, more importantly, by hand during bring-up:
    it is the one command that answers "is the neural add-on loaded at all?"
    without going through the whole pipeline.
    """
    try:
        result = subprocess.run(
            [str(exe), "--probe"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(exe.parent),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as error:
        return f"Could not run the harness: {error}"
    except subprocess.TimeoutExpired:
        return "The harness did not respond within 60 seconds."
    return (result.stdout or result.stderr or "").strip() or "No output."


def run_frames(
    exe: Path,
    *,
    width: int,
    height: int,
    depth_path: Path,
    motion_path: Path,
    colour_path: Path,
    out_path: Path,
    neural: NeuralSettings,
    jitter: list[tuple[float, float]],
    write_colour: Callable[[Path, tuple[float, float]], None],
    progress: Callable[[str], None] | None = None,
) -> None:
    """Evaluate the whole sequence and leave the result in `out_path`.

    `write_colour` regenerates the colour plane for a given jitter offset. It is
    a callback rather than a list of pre-written files because a 4K RGBA16F
    plane is 66 MB — eight of them on disk at once is half a gigabyte for no
    benefit, since the harness only ever reads one at a time.
    """
    total = len(jitter)
    with Harness(
        exe,
        width=width,
        height=height,
        depth_path=depth_path,
        motion_path=motion_path,
        neural=neural,
        frames=total,
    ) as harness:
        if progress and harness.notes:
            progress(harness.notes)
        for index, offset in enumerate(jitter, 1):
            if progress:
                progress(f"DLSS 5 pass {index} of {total}…")
            write_colour(colour_path, offset)
            harness.frame(colour_path, offset)
        if progress:
            progress("Reading the result back…")
        harness.write(out_path)
