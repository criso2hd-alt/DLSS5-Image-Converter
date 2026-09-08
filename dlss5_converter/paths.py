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

#: Where the user drops .cube LUTs for the effects stack. A plain folder they
#: fill themselves, like dlss_files - the app ships no LUTs, and any pack from
#: the ReShade/film-emulation world works because .cube is the portable format
#: those tools export.
LUTS_DIR = "luts"

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


def resource_dir() -> Path:
    """Root the bundled read-only assets live under.

    Frozen, PyInstaller unpacks ``--add-data`` payloads under ``sys._MEIPASS``;
    from source they sit beside this file's package. Kept separate from
    ``app_dir`` because assets are read-only and travel *inside* the bundle,
    unlike the user's models/output folders which sit next to the executable.
    """
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base)
    return Path(__file__).resolve().parents[1]


def fonts_dir() -> Path:
    """The bundled UI fonts (Archivo, IBM Plex Sans/Mono)."""
    return resource_dir() / "dlss5_converter" / "assets" / "fonts"


def onboarding_image() -> Path:
    """The original architectural sample used by the first-run introduction."""
    return resource_dir() / "dlss5_converter" / "assets" / "onboarding" / "architectural-detail.jpg"


def bundled_onnx_dir() -> Path:
    """ONNX depth models shipped inside the app.

    The Apache-2.0 Small model rides here so a release runs depth out of the box
    with no download and no PyTorch. Larger models, being non-commercial, are not
    bundled; they land in the per-user cache instead. See onnx_depth.py.
    """
    return resource_dir() / "dlss5_converter" / "assets" / "onnx"


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


def luts_dir() -> Path:
    """The folder the user fills with .cube LUTs for the effects stack.

    Created on demand so the Effects tab can always offer an "open folder"
    button that lands somewhere real, even before any LUT has been added.
    """
    path = data_dir() / LUTS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


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
#:
#: RenoDX has shipped this under more than one name — ``renodx-dlss5.addon64``
#: originally, a shorter ``dlss*.addon64`` more recently — and may rename it
#: again. ReShade loads *any* file ending in ``.addon64`` that sits beside it,
#: so discovery matches the suffix (see ``find_addon``) rather than one exact
#: name. ``ADDON_FILE`` stays the name shown in guidance and preferred when more
#: than one candidate is present.
ADDON_FILE = "renodx-dlss5.addon64"
ADDON_SUFFIX = ".addon64"


def runtime_search_roots() -> list[Path]:
    """Places to look for the user's DLSS 5 files, best guess first.

    A **release build looks only in ``dlss_files``** (plus whatever folder the
    user set explicitly). The tempting Downloads guesses below are deliberately
    limited to source checkouts, because they make the developer's machine
    behave differently from everybody else's: this project's own author had a
    working app with an incomplete ``dlss_files``, purely because a stray
    ``~/Downloads/dlss5`` was satisfying the lookup, while the first outside
    tester with the identical folder got "nvngx_dlss.dll was not found".

    A convenience that hides a broken configuration from the one person able to
    fix it is not a convenience.
    """
    roots = [dlss_files_dir()]
    if not is_frozen():
        home = Path.home()
        roots += [
            data_dir() / "runtime",
            app_dir() / "runtime",
            home / "Downloads" / "dlss5",
            home / "Downloads" / "DLSS5",
        ]
    return [root for root in roots if root.is_dir()]


#: How deep to look inside a runtime folder. A Streamline drop unpacks as
#: streamline/Production/nvngx_dlss.dll, or NVStreamline/Production/..., or with
#: an extra wrapper folder depending on how it was zipped. Four levels covers
#: every shape seen so far without turning a mis-set folder into a disk crawl.
_MAX_RUNTIME_DEPTH = 4


def deep_search_roots() -> list[Path]:
    """Folders worth searching recursively, as opposed to just checking.

    Same set as ``runtime_search_roots`` — and the same reason for keeping a
    release confined to ``dlss_files``.
    """
    return runtime_search_roots()


def _find_deep(root: Path, wanted: str, depth: int = _MAX_RUNTIME_DEPTH) -> Path | None:
    """Breadth-first hunt for a filename under `root`.

    Breadth-first on purpose: a file sitting directly in the runtime folder
    should win over a copy buried inside an unpacked archive, because the loose
    one is the copy the user placed deliberately.
    """
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return None

    subdirectories: list[Path] = []
    for entry in entries:
        try:
            if entry.is_file():
                if entry.name.lower() == wanted:
                    return entry
            elif entry.is_dir():
                subdirectories.append(entry)
        except OSError:
            continue

    if depth <= 0:
        return None
    for subdirectory in subdirectories:
        found = _find_deep(subdirectory, wanted, depth - 1)
        if found is not None:
            return found
    return None


def _collect_deep(root: Path, matches, depth: int = _MAX_RUNTIME_DEPTH, out=None) -> list[Path]:
    """Every file under `root` whose lowercased name satisfies `matches`."""
    if out is None:
        out = []
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return out
    subdirectories: list[Path] = []
    for entry in entries:
        try:
            if entry.is_file():
                if matches(entry.name.lower()):
                    out.append(entry)
            elif entry.is_dir():
                subdirectories.append(entry)
        except OSError:
            continue
    if depth > 0:
        for subdirectory in subdirectories:
            _collect_deep(subdirectory, matches, depth - 1, out)
    return out


def _addon_rank(name: str) -> int:
    """How good a `.addon64` filename looks as *the* DLSS add-on. Lower wins."""
    low = name.lower()
    if low == ADDON_FILE:      # the exact name we have always shipped guidance for
        return 0
    if "dlss" in low:          # renodx-dlss.addon64, dlss.addon64, a future dlss*…
        return 1
    if "renodx" in low:        # some other RenoDX add-on, still likely right here
        return 2
    return 3                   # any other ReShade add-on — last resort


def find_addon(extra: Path | None = None) -> Path | None:
    """The RenoDX DLSS add-on, matched by its ``.addon64`` suffix.

    Deliberately not a fixed filename: RenoDX renames this file between releases,
    and hard-coding one name turns a routine add-on update into "the neural pass
    silently stopped working". Every ``.addon64`` in the runtime folders is a
    candidate; the best is chosen by name (the known name, then anything that
    looks like the DLSS add-on, then any add-on at all) and, within a rank, the
    newest — because an out-of-date add-on is itself a confirmed cause of a green
    setup that produces an unchanged image.
    """
    candidates: list[Path] = []
    if extra is not None and extra.is_file() and extra.suffix.lower() == ADDON_SUFFIX:
        candidates.append(extra)
    roots = ([extra] if extra is not None and extra.is_dir() else []) + deep_search_roots()
    seen: set[str] = {str(c).lower() for c in candidates}
    for root in roots:
        for found in _collect_deep(root, lambda n: n.endswith(ADDON_SUFFIX)):
            key = str(found).lower()
            if key not in seen:
                seen.add(key)
                candidates.append(found)
    if not candidates:
        return None

    def sort_key(path: Path) -> tuple[int, float]:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (_addon_rank(path.name), -mtime)

    return min(candidates, key=sort_key)


def find_runtime_file(name: str, extra: Path | None = None) -> Path | None:
    """First readable copy of `name`, preferring an explicit user setting.

    Falls back to searching the runtime folders recursively. People unpack
    Streamline in whatever shape the archive came in, and demanding one exact
    layout means telling a user their perfectly reasonable folder is wrong —
    which is precisely what happened to the first outside tester, twice.
    """
    wanted = name.lower()
    candidates: list[Path] = []
    if extra is not None:
        candidates += [extra / name, extra]
    candidates += [root / name for root in runtime_search_roots()]
    for candidate in candidates:
        try:
            if candidate.is_file() and candidate.name.lower() == wanted:
                return candidate
        except OSError:
            continue

    # Nothing at the expected spots, so look inside them.
    roots = ([extra] if extra is not None and extra.is_dir() else []) + deep_search_roots()
    for root in roots:
        found = _find_deep(root, wanted)
        if found is not None:
            return found
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
