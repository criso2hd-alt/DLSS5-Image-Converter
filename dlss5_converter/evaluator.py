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
import mmap
import os
import subprocess
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType

import numpy as np

from .settings import NeuralSettings

#: Exit codes a crashing harness comes back with. Only the ones we have seen or
#: can give advice about - anything else is reported as a raw hex code, which is
#: still enough to identify in a bug report.
_EXIT_CODES: dict[int, str] = {
    0xC0000005: (
        "access violation. Usually a DLSS runtime mismatched with the add-on, "
        "or a driver too old for the neural pass - update the driver, and try "
        "a smaller Max size."
    ),
    0xC0000017: (
        "out of memory. Lower Max size, or use the Base depth model, which "
        "leaves the harness more VRAM."
    ),
    0xC0000409: "stack buffer overrun inside the runtime.",
    0xC0000374: "heap corruption inside the runtime.",
    0xC000001D: "illegal instruction.",
    0xC00000FD: "stack overflow.",
    0xC0000135: "a required DLL was missing.",
    0xC0000142: "a DLL failed to initialise.",
    0x8007000E: "out of memory.",
    0x887A0005: (
        "the graphics device was removed or reset. A driver timeout, a crash "
        "inside DLSS, or the GPU out of memory - lower Max size and retry."
    ),
    0x887A0006: "the graphics device hung.",
    0x887A0020: "an internal driver error.",
}


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
        use_shmem: bool = False,
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

        # Shared-memory colour transport. Only worth setting up for the
        # throughput paths (video, sequence, batch), where the same 66 MB plane
        # is otherwise written to and read from a file every pass of every
        # frame. A single-image conversion pays the harness start-up once and
        # gains nothing, so it leaves this off.
        self._want_shmem = use_shmem
        self._colour_bytes = width * height * 8  # RGBA16F, 8 bytes/pixel
        self._shmem: mmap.mmap | None = None
        #: A numpy view over the mapping. Held so the mapping is written through
        #: it in place; must be dropped before the mmap is closed, or close()
        #: raises BufferError because an export is still outstanding.
        self._shmem_np: np.ndarray | None = None
        self._shmem_active = False

    def __enter__(self) -> Harness:
        command = list(self._command)
        # Create the shared mapping and ask the harness to use it. Best-effort:
        # if the mapping cannot be made, drop straight back to the file path.
        if self._want_shmem:
            try:
                name = f"dlss5_colour_{os.getpid()}_{id(self) & 0xFFFFFFFF}"
                self._shmem = mmap.mmap(-1, self._colour_bytes, tagname=name)
                command += ["--colour-shmem", name, str(self._colour_bytes)]
            except (OSError, ValueError):
                self._shmem = None

        try:
            self._spawn(command)
        except HarnessError:
            # A harness that rejected the mapping flag is almost certainly an
            # older build that predates it (it Fails on an unknown argument). The
            # feature is meant to be invisible, so fall back to the file path and
            # try once more, rather than making an old exe unusable.
            if "--colour-shmem" in command:
                self._teardown_shmem()
                self._spawn(list(self._command))
            else:
                raise

        # The harness only advertises the token when its own mapping succeeded,
        # so this is the authority on whether shared memory is really live. If we
        # asked but it did not take, drop the unused mapping and use files.
        self._shmem_active = self._shmem is not None and "shmem:colour" in self.notes
        if not self._shmem_active:
            self._teardown_shmem()
        return self

    def _spawn(self, command: list[str]) -> None:
        """Launch one harness process and wait for its READY line, or raise."""
        try:
            self._process = subprocess.Popen(
                command,
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

    def _teardown_shmem(self) -> None:
        # The numpy view must go first: while it is alive it holds an export on
        # the mmap and close() would raise BufferError.
        self._shmem_np = None
        if self._shmem is not None:
            try:
                self._shmem.close()
            except (BufferError, OSError):
                pass
            self._shmem = None
        self._shmem_active = False

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        process = self._process
        self._process = None
        if process is None:
            # Only ever reached before a successful launch, but the mapping may
            # have been created regardless, so still release it.
            self._teardown_shmem()
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
            # Only after the child is gone: it maps the same section, and
            # unmapping while it might still read would be a use-after-free.
            self._teardown_shmem()

    # -- protocol ------------------------------------------------------------

    def _read(self) -> str:
        process = self._process
        if process is None or process.stdout is None:
            raise HarnessError("The harness is not running.")
        line = process.stdout.readline()
        if not line:
            raise HarnessError(self._died())
        line = line.strip()
        if line.startswith("ERROR"):
            raise HarnessError(line[len("ERROR") :].strip() or "Unknown DLSS failure.")
        return line

    def _died(self) -> str:
        """Explain a harness that stopped talking without saying why.

        A handled failure arrives as an ``ERROR`` line, so reaching here means
        the process died mid-sentence. Its exit code is then the only evidence
        left, and the Windows codes are specific enough to be worth naming: a
        bare "exited unexpectedly" cannot tell a crash inside NGX apart from the
        GPU running out of memory, and those have opposite fixes.
        """
        process = self._process
        if process is None:
            return "The harness is not running."

        stderr = ""
        if process.stderr is not None:
            try:
                stderr = process.stderr.read() or ""
            except Exception:  # noqa: BLE001 - a dead pipe must not mask this
                stderr = ""
        try:
            code = process.wait(timeout=5)
        except Exception:  # noqa: BLE001
            code = process.poll()

        message = f"The harness stopped at {self._width}x{self._height}"
        if code is None:
            message += "."
        else:
            unsigned = code & 0xFFFFFFFF
            detail = _EXIT_CODES.get(unsigned)
            message += f" (exit code 0x{unsigned:08X}"
            message += f" - {detail})" if detail else ")"
        if stderr.strip():
            message += "\n" + stderr.strip()
        return message

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

    def colour_buffer(self, shape: tuple[int, int, int]) -> np.ndarray:
        """A float16 buffer for the next colour plane, for throughput callers.

        When shared memory is live this is a numpy view straight over the
        mapping the harness reads, so writing into it is the transport — there
        is no separate copy step and no file. Otherwise it is a plain array the
        caller writes to its own scratch file, exactly as before. Either way the
        caller allocates once and rewrites in place across frames.
        """
        count = int(shape[0]) * int(shape[1]) * int(shape[2])
        if self._shmem_active and self._shmem is not None:
            if self._shmem_np is None:
                self._shmem_np = np.frombuffer(self._shmem, dtype=np.float16)
            return self._shmem_np[:count].reshape(shape)
        return np.empty(shape, np.float16)

    def commit_colour(
        self, plane: np.ndarray, colour_path: Path, jitter: tuple[float, float]
    ) -> None:
        """Deliver the staged colour plane and evaluate it.

        With shared memory the plane is already in the mapping (it *is* the
        mapping, via colour_buffer), so this only tells the harness to read it.
        Without it, the plane is written to the scratch file and its path is
        sent — the original path exactly.
        """
        if self._shmem_active and self._shmem is not None:
            self.frame("@shmem", jitter)
        else:
            plane.tofile(colour_path)
            self.frame(colour_path, jitter)

    def set_depth(self, depth_path: Path) -> None:
        """Replace the depth plane without restarting the harness.

        Sequence mode only. Everything expensive about a session — the device,
        the NGX feature, the add-on's warmed-up hooks — is reused across frames,
        so a hundred-frame sequence pays the ~3.5 s start-up once rather than a
        hundred times. Depth is the only per-frame input that was fixed at
        launch, so it needs its own command.
        """
        self._send(f"DEPTH {depth_path}")
        response = self._read()
        if not response.startswith("DEPTH_OK"):
            raise HarnessError(f"Unexpected reply to DEPTH: {response}")

    def reset_history(self) -> None:
        """Make the next frame start a fresh accumulation.

        Between frames of a sequence, not within one. Each frame is an
        independent still: the motion vectors are zero, so carrying temporal
        history from the previous frame would smear the last image into this one
        wherever anything moved.
        """
        self._index = 0

    def write(self, out_path: Path) -> None:
        self._send(f"WRITE {out_path}")
        response = self._read()
        if not response.startswith("WRITE_OK"):
            raise HarnessError(f"Unexpected reply to WRITE: {response}")


#: The probe currently in flight, so a stuck one can be cancelled. Guarded
#: because probe() runs on a worker thread and cancel_probe() is called from the
#: UI thread.
_PROBE_LOCK = threading.Lock()
_probe_process: subprocess.Popen | None = None

#: How long to wait for the live check before giving up. A working probe is a
#: few seconds (NGX + add-on warm-up); anything near this is a runtime that has
#: wedged, and a shorter cap means a passive user recovers on their own instead
#: of staring at a dialog that never returns.
_PROBE_TIMEOUT = 40.0


def probe(exe: Path, timeout: float = _PROBE_TIMEOUT) -> str:
    """Ask the harness what the installed DLSS runtime can actually do.

    Used by the settings panel and, more importantly, by hand during bring-up:
    it is the one command that answers "is the neural add-on loaded at all?"
    without going through the whole pipeline.

    The process is tracked like a live harness so both app shutdown and an
    explicit :func:`cancel_probe` can end it. This matters because a probe that
    hangs inside DLSS initialisation holds a D3D12 device, and a held device is
    what makes the whole app look frozen - the only way back is to kill the one
    process holding it, which must not mean killing the app.
    """
    global _probe_process
    try:
        process = subprocess.Popen(
            [str(exe), "--probe"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(exe.parent),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as error:
        return f"Could not run the harness: {error}"

    _register(process)
    with _PROBE_LOCK:
        _probe_process = process
    try:
        out, err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.communicate(timeout=5)
        except Exception:  # noqa: BLE001 - already giving up on this process
            pass
        return f"The harness did not respond within {int(timeout)} seconds."
    finally:
        _unregister(process)
        with _PROBE_LOCK:
            if _probe_process is process:
                _probe_process = None
    return (out or err or "").strip() or "No output."


def cancel_probe() -> None:
    """End a probe in flight - the setup dialog's Skip button.

    A probe stuck initialising DLSS holds the GPU device, which is what makes
    the app look hung. Killing that one process releases the device without
    losing the app: the same recovery as force-quitting, except the app lives.
    Best-effort; if the child has already gone this does nothing.
    """
    with _PROBE_LOCK:
        process = _probe_process
    if process is None:
        return
    try:
        if process.poll() is None:
            process.kill()
    except Exception:  # noqa: BLE001 - best effort; shutdown must not raise
        pass


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
