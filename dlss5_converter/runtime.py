"""Locate and sanity-check the DLSS runtime the user supplies themselves.

None of these binaries ship with this app. ``nvngx_dlssnr.dll`` in particular is
a leaked pre-release NVIDIA file; the app will happily use a copy the user
already has and will not help anyone obtain one.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import paths
from .settings import (
    NR_COLOR_MAX,
    NR_PAPER_WHITE_MAX,
    NR_PRESETS,
    NR_STRENGTH_MAX,
    NR_STYLES,
    NR_TRANSFER_MAX,
    NeuralSettings,
)


@dataclass
class RuntimeStatus:
    """What we found, and what is missing, in language a user can act on."""

    neural_dll: Path | None = None
    dlss_dll: Path | None = None
    addon: Path | None = None
    reshade: Path | None = None
    harness: Path | None = None
    problems: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.problems is None:
            self.problems = []

    @property
    def ready(self) -> bool:
        return not self.problems


#: The leaked build is ~158 MB. A file dramatically smaller than this is almost
#: always a stub, a placeholder, or an HTML error page saved with a .dll name —
#: all of which otherwise fail much later with an opaque NGX error code.
_MIN_NEURAL_DLL_BYTES = 100 * 1024 * 1024


#: Filenames worth pulling out of an archive, lowercased for comparison.
_WANTED_IN_ARCHIVES = {
    "nvngx_dlssnr.dll",
    "nvngx_dlss.dll",
    paths.ADDON_FILE.lower(),
    "reshade64.dll",
    "dxgi.dll",
}


def unpack_archives(extra: Path | None = None) -> list[str]:
    """Pull the runtime files out of any zip sitting in the runtime folders.

    Streamline ships as a zip and people quite reasonably drop the whole archive
    in rather than hunting through it for one DLL — the first outside tester did
    exactly that and got "nvngx_dlss.dll was not found" while looking straight at
    a streamline.zip containing it.

    Extracted beside the archive, which is already a folder we search. Existing
    files are never overwritten: if someone has deliberately placed a particular
    build, an old copy inside a zip must not silently replace it.
    """
    import zipfile

    recovered: list[str] = []
    roots = paths.runtime_search_roots()
    if extra is not None and extra.is_dir():
        roots.insert(0, extra)

    for root in roots:
        for archive in sorted(root.glob("*.zip")):
            try:
                with zipfile.ZipFile(archive) as bundle:
                    for member in bundle.infolist():
                        if member.is_dir():
                            continue
                        name = Path(member.filename).name
                        if name.lower() not in _WANTED_IN_ARCHIVES:
                            continue
                        target = root / name
                        if target.exists():
                            continue
                        with bundle.open(member) as source, open(target, "wb") as sink:
                            shutil.copyfileobj(source, sink)
                        recovered.append(f"{name} (from {archive.name})")
            except (OSError, zipfile.BadZipFile, RuntimeError):
                # A corrupt or encrypted archive is not fatal; the missing-file
                # message downstream is still the right thing to show.
                continue
    return recovered


def detect(runtime_dir: str | Path | None = None) -> RuntimeStatus:
    """Find every piece the harness needs, without loading any of them."""
    extra = Path(runtime_dir) if runtime_dir else None
    status = RuntimeStatus()

    def locate() -> None:
        status.neural_dll = paths.find_runtime_file("nvngx_dlssnr.dll", extra)
        status.dlss_dll = paths.find_runtime_file("nvngx_dlss.dll", extra)
        status.addon = paths.find_runtime_file(paths.ADDON_FILE, extra)
        # ReShade under either name it plausibly has. An existing install has
        # already renamed it to the DLL it proxies, so a user following our own
        # advice drops a "dxgi.dll" here and would otherwise be told ReShade is
        # missing while looking straight at it.
        status.reshade = paths.find_runtime_file(
            "ReShade64.dll", extra
        ) or paths.find_reshade_proxy(extra)

    locate()
    # Only if something is missing: opening every zip in the folder on each
    # detect() would be wasted work on the common path, and detect() runs on
    # startup, on Diagnose, and before every conversion.
    if not all((status.neural_dll, status.dlss_dll, status.addon, status.reshade)):
        if unpack_archives(extra):
            locate()

    harness = paths.native_exe()
    status.harness = harness if harness.is_file() else None

    # A release build has one obvious place for these, so name it. Telling a
    # user of a packaged app to "point the runtime folder somewhere" when the
    # app shipped them an empty folder for exactly this is needless friction.
    # The single most useful instruction we can give, so it goes on every
    # missing-file message: a set already running in a game is a set this
    # machine's GPU, driver and add-on all accept.
    where = (
        f"Put it in the {paths.DLSS_FILES_DIR} folder next to the application. "
        "If DLSS 5 already works in a game for you, copy all four files out of "
        "that game's folder - they sit beside the game executable."
        if paths.is_frozen()
        else "Point the runtime folder at your own copy of the DLSS 5 files."
    )

    if status.neural_dll is None:
        status.problems.append(f"nvngx_dlssnr.dll was not found. {where}")
    else:
        try:
            if status.neural_dll.stat().st_size < _MIN_NEURAL_DLL_BYTES:
                status.problems.append(
                    f"{status.neural_dll.name} is only "
                    f"{paths.format_bytes(status.neural_dll.stat().st_size)}; the "
                    "real neural-rendering model is around 158 MB. This copy "
                    "looks incomplete."
                )
        except OSError:
            status.problems.append(f"{status.neural_dll} could not be read.")

    if status.dlss_dll is None:
        status.problems.append(
            "nvngx_dlss.dll was not found. The neural pass runs inside a DLSS "
            f"Super Resolution evaluation, so both DLLs are required. {where} "
            "It ships inside Streamline, under Production - a zipped Streamline "
            "in that folder is opened automatically, so if you are seeing this "
            "the archive does not contain it."
        )
    if status.addon is None:
        status.problems.append(
            f"{paths.ADDON_FILE} was not found. Without the add-on the harness "
            f"produces a plain DLAA resolve and no neural enhancement. {where}"
        )
    if status.harness is None:
        status.problems.append(
            f"dlss5_eval.exe is missing from the {paths.ENGINE_DIR} folder. "
            "This release is incomplete — re-extract it."
            if paths.is_frozen()
            else "dlss5_eval.exe has not been built. Run "
            "scripts\\build_native.ps1 once, then restart."
        )
    elif status.reshade is None and not (status.harness.parent / "dxgi.dll").is_file():
        status.problems.append(
            "ReShade64.dll was not found. Any existing ReShade install already "
            "has it under the name of the DLL it proxies, so a game's "
            "bin\\x64\\dxgi.dll works — otherwise extract it from the ReShade "
            f"installer. {where} It is what loads the add-on; without it the "
            "harness silently falls back to plain DLAA."
        )
    return status


def _already_staged(source: Path, destination: Path) -> bool:
    """Whether `destination` is genuinely this `source`, not merely like it.

    A hard link is the same file, so there is nothing to check and nothing that
    can drift - that is the normal case and it returns True immediately.

    Otherwise the copy is only trusted when size *and* modification time match.
    Size alone was the old test and it is not enough: replacing an add-on with a
    different build of identical size left the stale one staged, which presents
    as the neural pass silently not running.
    """
    if not destination.exists():
        return False
    try:
        here, there = destination.stat(), source.stat()
    except OSError:
        return False
    # Same inode: it *is* the file, however either path was reached.
    if here.st_ino and here.st_ino == there.st_ino and here.st_dev == there.st_dev:
        return True
    return here.st_size == there.st_size and int(here.st_mtime) == int(there.st_mtime)


def stage_runtime(status: RuntimeStatus) -> Path:
    """Copy the runtime beside the harness, which is where the loaders look.

    NGX resolves ``nvngx_*.dll`` from the executable's own directory, and
    ReShade only loads add-ons sitting next to the module that hosts it. Rather
    than ask the user to arrange that by hand, the app mirrors whatever it found
    into the harness folder.

    **The staged copy is refreshed, not merely created.** This used to skip
    whenever the destination's size matched, which quietly kept a stale file
    forever: two builds of ``renodx-dlss5.addon64`` are frequently the same
    size, and an out-of-date add-on is the single most common cause of "every
    indicator is green and the image is unchanged". Someone could follow the
    advice to update their add-on, drop the new file in ``dlss_files``, and have
    the app go on loading the old one - the exact failure the advice was meant
    to cure.

    Refreshing is free in the normal case. Source and destination sit on one
    volume, so this hard links rather than copies; relinking costs nothing and
    cannot go stale. Only when links are refused - across volumes, on FAT32 -
    does it fall back to a real copy, and only there is it worth checking
    whether the work can be skipped, because nvngx_dlssnr.dll is 158 MB.
    """
    if status.harness is None:
        raise RuntimeError("The native harness is not built.")
    target = status.harness.parent
    target.mkdir(parents=True, exist_ok=True)

    for source in (status.neural_dll, status.dlss_dll, status.addon, status.reshade):
        if source is None:
            continue
        # ReShade only hooks when it is loaded as a proxy for a DLL the process
        # already imports. dlss5_eval imports dxgi, so that is the name it takes.
        name = "dxgi.dll" if source.name.lower() == "reshade64.dll" else source.name
        destination = target / name
        try:
            if _already_staged(source, destination):
                continue
            destination.unlink(missing_ok=True)
            # Hard link rather than copy where the filesystem allows it. In a
            # release build the source is dlss_files/ and the target is engine/
            # on the same volume, and nvngx_dlssnr.dll alone is 158 MB — copying
            # means the user pays for it twice for no benefit, since nothing
            # here ever writes to these files. Falls back to a real copy across
            # volumes, on FAT32, and anywhere links are refused.
            try:
                os.link(source, destination)
            except (OSError, NotImplementedError):
                shutil.copy2(source, destination)
        except OSError as error:
            raise RuntimeError(
                f"Could not place {source.name} next to the harness: {error}"
            ) from error
    return target


#: The add-on's own ini section, read through ReShade's config API.
ADDON_SECTION = "RenoDX.DLSS5"


def write_addon_config(harness_dir: Path, neural: NeuralSettings) -> Path:
    """Point the RenoDX add-on's knobs at the user's settings.

    This is the *only* route that works. The add-on publishes no NGX parameter
    and ignores the environment; it reads ``ReShade.ini`` beside the executable,
    and it reads it **once, at startup** — measured, not assumed. So this must
    run before the harness is launched, and a settings change means a new
    harness process rather than a message to a running one.

    Strength values are written 0..2, matching the add-on's own sliders, so the
    app's controls map across unscaled.

    Merged into the existing file rather than written over it. ReShade owns this
    ini, rewrites it wholesale on exit, and stamping our four keys over its
    window layout and hotkeys would reset the user's ReShade setup every run.
    """
    import configparser

    path = harness_dir / "ReShade.ini"
    parser = configparser.ConfigParser()
    # ReShade's keys are case-sensitive; configparser lowercases by default,
    # which would silently write a second, ignored set of settings.
    parser.optionxform = str
    if path.is_file():
        try:
            parser.read(path, encoding="utf-8")
        except (OSError, configparser.Error):
            # A corrupt ini is ReShade's to regenerate. Ours still get written.
            parser = configparser.ConfigParser()
            parser.optionxform = str

    if not parser.has_section(ADDON_SECTION):
        parser.add_section(ADDON_SECTION)

    def clamp(value: float, ceiling: float = NR_STRENGTH_MAX) -> str:
        # Each knob has its own measured ceiling — the point where raising the
        # value stops changing the image. They differ (2.0 for the strengths,
        # 1.0 for colour and transfer, 16.0 for paper-white), so a single shared
        # limit would either truncate one or let another run off into a range
        # the add-on ignores.
        return f"{min(ceiling, max(0.0, float(value))):.4f}"

    # Intensity 0 turns the pass off outright rather than running it at zero
    # strength. Measured, these differ: NRIntensity=0 still moves the image,
    # NeuralUplift=0 is bit-identical to a plain DLAA resolve — which is what
    # the UI promises at 0, and the honest A/B control.
    enabled = float(neural.intensity) > 0.0
    parser.set(ADDON_SECTION, "NeuralUplift", "1" if enabled else "0")
    parser.set(ADDON_SECTION, "NRIntensity", clamp(neural.intensity))
    parser.set(ADDON_SECTION, "NRSkinStructure", clamp(neural.skin))
    parser.set(ADDON_SECTION, "NRLocalTone", clamp(neural.local_tone))
    parser.set(ADDON_SECTION, "NRLocalStructure", clamp(neural.structure))

    parser.set(ADDON_SECTION, "NRColorStrength", clamp(neural.color_strength, NR_COLOR_MAX))
    parser.set(
        ADDON_SECTION, "NRTransferStrength", clamp(neural.transfer_strength, NR_TRANSFER_MAX)
    )
    parser.set(
        ADDON_SECTION, "NRPaperWhiteScale", clamp(neural.paper_white, NR_PAPER_WHITE_MAX)
    )

    # Enum indices, clamped to the item lists the add-on actually offers. Game
    # inis in the wild carry out-of-range values (NRStyle=2 with only two
    # styles), so trusting a stored number here would pass the add-on an index
    # past the end of its own combo.
    def index(value: int, count: int) -> str:
        return str(min(count - 1, max(0, int(value))))

    parser.set(ADDON_SECTION, "NRPreset", index(neural.preset, len(NR_PRESETS)))
    parser.set(ADDON_SECTION, "NRStyle", index(neural.style, len(NR_STYLES)))
    # DLAA only — see the non-goals in ROADMAP.md.
    parser.set(ADDON_SECTION, "NREnableUpscaling", "0")

    try:
        with open(path, "w", encoding="utf-8") as handle:
            parser.write(handle, space_around_delimiters=False)
    except OSError as error:
        raise RuntimeError(f"Could not write the add-on settings: {error}") from error
    return path


def describe(status: RuntimeStatus) -> str:
    """One-line summary for the status bar."""
    if status.ready:
        return "DLSS 5 runtime ready"
    return f"{len(status.problems)} runtime problem(s) — see Settings"
