"""Fetch the heavy runtime on first launch instead of shipping it.

PyTorch with CUDA is 2.7 GB unpacked and is essentially the entire size of a
bundled release — the rest of the application is about 300 MB. It is freely
redistributable, so bundling it is *allowed*; it is just wasteful, because every
copy of the app then carries a payload that NVIDIA already require the user to
have a machine capable of running.

So the release ships without it and fetches the wheel once, with a progress bar,
into a folder that survives app updates. This is the same trade the depth model
already makes, and it is what keeps the shareable download small.

Nothing here is a package manager. The wheel is a zip; it is downloaded,
extracted, and put on ``sys.path``. No pip, no network resolution, no version
solving — a pinned URL and an unzip, so the failure modes are "no network" and
"disk full" rather than anything dependency-shaped.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from urllib.request import Request, urlopen

from . import paths

#: Pinned exactly. A wheel is tied to both the Python version (cp312) and the
#: CUDA build, and "latest" would eventually hand a frozen cp312 app a wheel it
#: cannot import. Update this deliberately, together with pyproject.
TORCH_VERSION = "2.13.0+cu130"
TORCH_WHEEL_URL = (
    "https://download.pytorch.org/whl/cu130/"
    "torch-2.13.0%2Bcu130-cp312-cp312-win_amd64.whl"
)

#: Only used to show a total before the server answers, and to sanity-check the
#: response. Not a checksum: this guards against a truncated download, not a
#: hostile one.
TORCH_APPROX_BYTES = 1_915_000_000

BytesProgress = Callable[[int, int], None]
TextProgress = Callable[[str], None]


def runtime_dir() -> Path:
    """Where the downloaded runtime lives.

    Beside the app in a release, so it is visible and deletable, and — like
    ``models`` — preserved across rebuilds. Named for what it holds rather than
    something generic: a folder called ``runtime`` next to ``dlss_files`` would
    invite people to drop their NVIDIA DLLs into it.
    """
    return paths.data_dir() / "pytorch"


def activate() -> None:
    """Put a previously downloaded runtime on the import path.

    Cheap and idempotent — safe to call on every start. Prepended rather than
    appended so the downloaded copy wins over anything stale.
    """
    target = runtime_dir()
    if not target.is_dir():
        return
    entry = str(target)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    # Torch loads its own CUDA DLLs relative to its package directory, but only
    # once the directory is a legal DLL search path on Windows.
    lib = target / "torch" / "lib"
    if lib.is_dir() and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(lib))
        except OSError:
            pass


def is_ready() -> bool:
    """Whether torch can be imported, from anywhere.

    Deliberately not "did we download it": in a source checkout torch comes from
    the virtualenv and there is nothing to fetch, and the frozen build may one
    day bundle it again. The question that matters is whether the import will
    work.
    """
    activate()
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


def _download(url: str, destination: Path, on_bytes: BytesProgress | None) -> None:
    request = Request(url, headers={"User-Agent": "dlss5-converter"})
    with urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or TORCH_APPROX_BYTES)
        done = 0
        with open(destination, "wb") as handle:
            while True:
                chunk = response.read(1024 * 512)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if on_bytes:
                    on_bytes(done, total)
    if done < total * 0.99:
        raise OSError(
            f"Download stopped early: {done} of {total} bytes. "
            "Check the connection and try again."
        )


def _extract(archive: Path, target: Path, on_bytes: BytesProgress | None) -> None:
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        total = sum(member.file_size for member in members)
        done = 0
        for member in members:
            bundle.extract(member, target)
            done += member.file_size
            if on_bytes:
                on_bytes(done, total)


def install(
    on_bytes: BytesProgress | None = None,
    on_text: TextProgress | None = None,
) -> None:
    """Download and unpack the runtime. Raises on failure.

    Staged through sibling directories and moved into place at the end, so an
    interrupted run leaves nothing that ``is_ready`` would mistake for a working
    install. A half-extracted torch imports far enough to fail confusingly.
    """
    target = runtime_dir()
    staging = target.with_name(target.name + ".partial")
    archive = target.with_name(target.name + ".whl.part")

    for leftover in (staging, archive):
        if leftover.is_dir():
            shutil.rmtree(leftover, ignore_errors=True)
        elif leftover.exists():
            leftover.unlink()

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if on_text:
            on_text(f"Downloading PyTorch {TORCH_VERSION}...")
        _download(TORCH_WHEEL_URL, archive, on_bytes)

        if on_text:
            on_text("Unpacking...")
        _extract(archive, staging, on_bytes)

        if not (staging / "torch" / "__init__.py").is_file():
            raise OSError("The downloaded runtime is missing its torch package.")

        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        staging.rename(target)
    finally:
        if archive.exists():
            try:
                archive.unlink()
            except OSError:
                pass
        if staging.is_dir():
            shutil.rmtree(staging, ignore_errors=True)

    activate()
