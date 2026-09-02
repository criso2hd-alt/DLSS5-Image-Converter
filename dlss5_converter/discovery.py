"""Find the user's DLSS 5 files on their own disk, and copy them in.

Most people arrive here having already made DLSS 5 work in a game. The files are
sitting in that game's folder; they just do not necessarily know where, or which
four matter. Asking them to go and find `nvngx_dlssnr.dll` by hand is the single
biggest thing standing between "downloaded it" and "it works".

The important part is not finding the files — it is finding them **together**.
A runtime from one source with an add-on from another is a common way to produce
*"NR is unavailable in this session"*, so this scores whole folders rather than
individual files and prefers a folder that has a complete matched set. That set
is, by construction, one the user's own GPU and driver already accept.

Nothing here downloads anything. It only looks at files the user already has.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

#: The four files, by the name they must end up with in `dlss_files`.
NEURAL_DLL = "nvngx_dlssnr.dll"
DLSS_DLL = "nvngx_dlss.dll"
ADDON = "renodx-dlss5.addon64"
RESHADE = "dxgi.dll"
WANTED = (NEURAL_DLL, DLSS_DLL, ADDON, RESHADE)

#: The real neural model is ~158 MB. Anything far smaller is a stub, a
#: placeholder, or an error page saved with a .dll name.
_MIN_NEURAL_BYTES = 100 * 1024 * 1024

#: Windows' own dxgi.dll is around 1 MB; ReShade's is several times that. Used
#: only to avoid reading every system DLL looking for the marker below.
_MIN_RESHADE_BYTES = 2 * 1024 * 1024

#: ReShade exports this. Far more reliable than a size check for telling a
#: renamed ReShade from the system dxgi.dll it is standing in for.
_RESHADE_MARKER = b"ReShadeRegisterAddon"

#: Directory names never worth descending into. Windows and Steam both hide
#: enormous trees that cannot contain a modded game folder, and walking them
#: turns a ten-second scan into a five-minute one.
_SKIP_DIRECTORIES = {
    "$recycle.bin", "system volume information", "windows", "winsxs",
    "node_modules", ".git", "__pycache__", "appdata",
}


@dataclass
class Candidate:
    """A folder, and which of the four files it holds."""

    folder: Path
    files: dict[str, Path] = field(default_factory=dict)

    @property
    def score(self) -> int:
        return len(self.files)

    @property
    def complete(self) -> bool:
        return self.score == len(WANTED)

    @property
    def newest(self) -> float:
        """Modification time of the newest file in the set.

        Used to break ties between complete folders, and it matters more than it
        sounds: an out-of-date renodx-dlss5.addon64 is a confirmed cause of the
        neural pass silently not running. Several games on one machine commonly
        share one old add-on while the copy the user actually updated sits in
        Downloads, so "any complete set" is not good enough - it has to be the
        freshest one.
        """
        times = []
        for path in self.files.values():
            try:
                times.append(path.stat().st_mtime)
            except OSError:
                continue
        return max(times) if times else 0.0

    def describe(self) -> str:
        missing = [name for name in WANTED if name not in self.files]
        if not missing:
            return f"{self.folder}  —  all four"
        return f"{self.folder}  —  {self.score} of 4, missing {', '.join(missing)}"


def looks_like_reshade(path: Path) -> bool:
    """Whether a dxgi.dll is really ReShade under an assumed name."""
    try:
        if path.stat().st_size < _MIN_RESHADE_BYTES:
            return False
        with open(path, "rb") as handle:
            return _RESHADE_MARKER in handle.read()
    except OSError:
        return False


def looks_like_neural_runtime(path: Path) -> bool:
    try:
        return path.stat().st_size >= _MIN_NEURAL_BYTES
    except OSError:
        return False


def accepts(name: str, path: Path) -> bool:
    """Whether a file matching one of the wanted names is actually usable."""
    if name == RESHADE:
        return looks_like_reshade(path)
    if name == NEURAL_DLL:
        return looks_like_neural_runtime(path)
    return True


def steam_root() -> Path | None:
    """Where Steam is installed, from the registry.

    Guessing `Program Files` misses every machine that put Steam on another
    drive — which is most machines with a large game library, and was true of
    the first one this was tested on.
    """
    try:
        import winreg
    except ImportError:
        return None

    for hive, key in (
        (winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam"),
    ):
        for value in ("SteamPath", "InstallPath"):
            try:
                with winreg.OpenKey(hive, key) as handle:
                    path = Path(winreg.QueryValueEx(handle, value)[0])
                if path.is_dir():
                    return path
            except (OSError, FileNotFoundError):
                continue
    return None


def steam_libraries() -> list[Path]:
    """Every Steam library on this machine, from Steam's own manifest.

    People install games across several drives, and the second library is
    exactly the one they forget to mention.
    """
    manifests = []
    root = steam_root()
    if root is not None:
        manifests.append(root / "steamapps" / "libraryfolders.vdf")
        # The install itself is a library, and is not always listed in its own
        # manifest on older Steam versions.
        manifests.append(root / "steamapps" / "common")
    manifests += [
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Steam" / "steamapps" / "libraryfolders.vdf",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        / "Steam" / "steamapps" / "libraryfolders.vdf",
    ]
    found: list[Path] = []
    for manifest in list(manifests):
        # A "common" folder is already a library; keep it and move on.
        if manifest.is_dir():
            found.append(manifest)
            manifests.remove(manifest)
    for manifest in manifests:
        try:
            text = manifest.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for raw in re.findall(r'"path"\s+"([^"]+)"', text):
            library = Path(raw.replace("\\\\", "\\")) / "steamapps" / "common"
            if library.is_dir():
                found.append(library)

    unique: list[Path] = []
    for library in found:
        if not any(library == seen for seen in unique):
            unique.append(library)
    return unique


def default_roots() -> list[Path]:
    """Where to look when the user has not chosen somewhere themselves."""
    home = Path.home()
    roots = [
        *steam_libraries(),
        home / "Downloads",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Epic Games",
        home / "Documents",
    ]
    seen: list[Path] = []
    for root in roots:
        if root.is_dir() and not any(root == other for other in seen):
            seen.append(root)
    return seen


def scan(
    roots: Iterable[Path],
    max_depth: int = 8,
    on_progress: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[Candidate]:
    """Walk `roots` and return folders holding any of the four, best first.

    Sorted by how many of the four a folder has, so a complete set from one
    game outranks a stray runtime sitting on its own in Downloads.
    """
    by_folder: dict[Path, Candidate] = {}
    wanted_lower = {name.lower(): name for name in WANTED}
    scanned = 0

    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        base_depth = len(root.parts)

        for current, directories, filenames in os.walk(root, topdown=True):
            if should_stop is not None and should_stop():
                return _ranked(by_folder)

            here = Path(current)
            if len(here.parts) - base_depth >= max_depth:
                directories[:] = []
                continue
            directories[:] = [
                d for d in directories
                if d.lower() not in _SKIP_DIRECTORIES and not d.startswith(".")
            ]

            scanned += 1
            if on_progress is not None and scanned % 400 == 0:
                on_progress(f"Searching… {scanned} folders, {len(by_folder)} hit(s)")

            for filename in filenames:
                canonical = wanted_lower.get(filename.lower())
                if canonical is None:
                    continue
                path = here / filename
                if not accepts(canonical, path):
                    continue
                candidate = by_folder.setdefault(here, Candidate(here))
                candidate.files.setdefault(canonical, path)

    return _ranked(by_folder)


def _ranked(by_folder: dict[Path, Candidate]) -> list[Candidate]:
    return sorted(by_folder.values(), key=lambda c: (-c.score, -c.newest, str(c.folder)))


def best_set(candidates: list[Candidate]) -> dict[str, Path]:
    """The files to copy: the best folder, topped up from others if it is short.

    A complete folder is used on its own — that is the matched set, and mixing
    is what causes the add-on to reject a runtime. Only when nothing is complete
    does this fall back to combining, because four files from three places still
    beats telling someone to go and look themselves.
    """
    if not candidates:
        return {}
    chosen = dict(candidates[0].files)
    for candidate in candidates[1:]:
        for name, path in candidate.files.items():
            chosen.setdefault(name, path)

    # One deliberate exception to keeping a set together: always take the
    # newest add-on found anywhere. An out-of-date renodx-dlss5.addon64 makes
    # the neural pass silently not run, and games tend to keep whichever build
    # was current when they were modded while the user's updated copy sits
    # somewhere else entirely.
    newest_addon = None
    newest_time = -1.0
    for candidate in candidates:
        path = candidate.files.get(ADDON)
        if path is None:
            continue
        try:
            when = path.stat().st_mtime
        except OSError:
            continue
        if when > newest_time:
            newest_time, newest_addon = when, path
    if newest_addon is not None:
        chosen[ADDON] = newest_addon
    return chosen


def install(files: dict[str, Path], destination: Path) -> tuple[list[str], list[str]]:
    """Copy the chosen files in. Returns (copied, skipped).

    Never overwrites: a file the user placed deliberately outranks anything
    found by a search.
    """
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    skipped: list[str] = []
    for name, source in files.items():
        target = destination / name
        if target.exists():
            skipped.append(name)
            continue
        try:
            shutil.copy2(source, target)
            copied.append(name)
        except OSError as error:
            skipped.append(f"{name} ({error})")
    return copied, skipped
