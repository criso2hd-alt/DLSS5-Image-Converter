"""Where the converter keeps data that must outlive an app update.

Same shape as Depth Animator's ``paths`` module, plus the runtime-locator half:
this app also has to find binaries it must never ship — the user's own copies of
``nvngx_dlssnr.dll``, ReShade, and the RenoDX add-on.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

APP_NAME = "DLSS5Converter"

#: Dropping this file beside the executable opts into keeping data with the app.
PORTABLE_MARKER = "portable.txt"

# --- Release layout ---------------------------------------------------------
#
# A built release is one folder the user can see into:
#
#     release/
#       DLSS5Converter.exe
#       dlss_files/   <- the NVIDIA + ReShade binaries the user supplies
#       models/       <- Depth Anything weights, downloaded on first run
#       output/       <- converted images land here
#       engine/       <- dlss5_eval.exe, and the staged copies of dlss_files
#
#: Where the user puts their own DLSS 5 runtime. Never shipped, never
#: downloaded — see the legal note in CLAUDE.md.
DLSS_FILES_DIR = "dlss_files"

#: Hugging Face cache root for the depth models.
MODELS_DIR = "models"

#: Default destination for converted images.
OUTPUT_DIR = "output"

#: The native harness, and at run time the staged NVIDIA binaries beside it.
#: Deliberately *not* the release root: ReShade attaches to any process that
#: finds a dxgi.dll next to it, and staging one beside the GUI executable would
#: pull ReShade into the GUI as well as into the harness that actually wants it.
ENGINE_DIR = "engine"


def app_dir() -> Path:
    """Folder containing the executable (frozen) or the project root (source)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def _user_data_dir() -> Path:
    try:
        from platformdirs import user_data_dir

        return Path(user_data_dir(APP_NAME, appauthor=False, roaming=False))
    except Exception:  # noqa: BLE001 - fall back rather than fail to start
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        return Path(base) / APP_NAME


def is_portable() -> bool:
    """Whether app data lives beside the app instead of in the user profile.

    Always true for a frozen build. The release layout puts ``dlss_files``,
    ``models`` and ``output`` next to the executable on purpose, so everything
    the app needs and everything it produced is visible in one folder.

    This is the one place the "never keep weights beside the executable" rule in
    CLAUDE.md is deliberately inverted, and the reason that rule existed is
    handled instead by ``scripts/build_release.ps1``, which preserves those
    three folders across a rebuild rather than replacing the release wholesale.
    """
    return is_frozen() or (app_dir() / PORTABLE_MARKER).exists()


def data_dir() -> Path:
    """Root for everything the app downloads and wants to keep."""
    if is_portable():
        return app_dir()
    return _user_data_dir()


def model_cache_dir() -> Path:
    """Hugging Face cache root. Honours a user-set HF_HOME above all else."""
    override = os.environ.get("HF_HOME")
    if override:
        return Path(override)
    return data_dir() / MODELS_DIR


def output_dir() -> Path:
    """Where converted images go when the user does not say otherwise."""
    path = data_dir() / OUTPUT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def dlss_files_dir() -> Path:
    """The folder the user fills with their own DLSS 5 runtime."""
    return data_dir() / DLSS_FILES_DIR


def settings_path() -> Path:
    return data_dir() / "settings.json"


def scratch_dir() -> Path:
    """Where contract planes are written for the native harness to read.

    Deliberately not the system temp folder. A 4K contract is ~100 MB across
    three planes and some machines put %TEMP% on a small or aggressively cleaned
    volume; a sweep between our write and the harness's read would surface as a
    baffling DLSS failure rather than a missing file.
    """
    # In a release build this goes beside the harness rather than in the release
    # root: the planes are transient, they are large, and the release folder is
    # something the user is meant to be able to look into and understand.
    base = native_exe().parent if is_frozen() else data_dir()
    path = base / "scratch"
    path.mkdir(parents=True, exist_ok=True)
    return path


def native_exe() -> Path:
    """The DLSS harness built by ``scripts/build_native.ps1``."""
    if is_frozen():
        return app_dir() / ENGINE_DIR / "dlss5_eval.exe"
    return app_dir() / "native" / "bin" / "dlss5_eval.exe"


# --- Locating the user's NVIDIA runtime -------------------------------------
#
# These files are not redistributable and are never bundled. The app looks in
# the places a user who followed a DLSS 5 modding guide would already have them,
# then falls back to whatever they set in the settings panel.

#: Filenames the harness needs beside itself at run time.
RUNTIME_FILES = (
    "nvngx_dlssnr.dll",  # the DLSS 5 neural-rendering model
    "nvngx_dlss.dll",  # DLSS Super Resolution; the NR pass rides on its evaluation
)

#: The ReShade add-on that hooks NGX evaluation and injects the neural pass.
ADDON_FILE = "renodx-dlss5.addon64"


def runtime_search_roots() -> list[Path]:
    """Plausible locations for a user's DLSS 5 mod files, best guess first.

    ``dlss_files`` comes first because in a release build it is the folder the
    app told the user to fill. ``NVStreamline/Production`` is listed as well as
    the folder itself: nvngx_dlss.dll ships inside a Streamline drop, and users
    reliably unpack the archive rather than flattening it.
    """
    roots = [
        dlss_files_dir(),
        dlss_files_dir() / "NVStreamline" / "Production",
        data_dir() / "runtime",
        app_dir() / "runtime",
    ]
    home = Path.home()
    roots += [
        home / "Downloads" / "dlss5",
        home / "Downloads" / "dlss5" / "NVStreamline" / "Production",
        home / "Downloads" / "DLSS5",
        home / "Downloads" / "DLSS5" / "NVStreamline" / "Production",
    ]
    return [root for root in roots if root.is_dir()]


def find_runtime_file(name: str, extra: Path | None = None) -> Path | None:
    """First readable copy of `name`, preferring an explicit user setting."""
    candidates: list[Path] = []
    if extra is not None:
        candidates += [extra / name, extra]
    candidates += [root / name for root in runtime_search_roots()]
    for candidate in candidates:
        try:
            if candidate.is_file() and candidate.name.lower() == name.lower():
                return candidate
        except OSError:
            continue
    return None


#: Windows' own dxgi.dll is around 1 MB; ReShade's is several times that. Used
#: only to tell the two apart when a user drops a proxy into the runtime folder,
#: so that copying the wrong dxgi.dll fails here rather than inside NGX.
_MIN_RESHADE_BYTES = 2 * 1024 * 1024


def find_reshade_proxy(extra: Path | None = None) -> Path | None:
    """ReShade already renamed to the DLL it stands in for.

    Anyone with ReShade in a game has it as ``dxgi.dll`` (or d3d11/opengl32),
    which is both the easiest copy to make and the name we want anyway. Only
    ``dxgi.dll`` is accepted: it is the one the harness actually imports.
    """
    candidate = find_runtime_file("dxgi.dll", extra)
    if candidate is None:
        return None
    try:
        if candidate.stat().st_size < _MIN_RESHADE_BYTES:
            return None
    except OSError:
        return None
    return candidate


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def format_bytes(count: int) -> str:
    size = float(count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def free_space(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return 0
