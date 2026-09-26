"""Install the DLSS5 Scene Capture add-on into a game that has ReShade.

The add-on (our own code, native/scene_capture) saves the game's depth beside
each ReShade screenshot, which "Scene from shots" needs. Installing it by
hand meant knowing two different folders: the add-on goes next to the game's
executable (where ReShade's DLL is), the effect file goes wherever ReShade
looks for shaders, which differs per game and per install. Getting the second
one wrong was the usual failure. This reads ReShade.ini to find out.

ReShade itself is not ours to ship; the game needs it installed first, the
build "with full add-on support".

Qt-free.
"""

from __future__ import annotations

import configparser
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import paths

ADDON = "dlss5_scene_capture.addon64"
EFFECT = "DLSS5Capture.fx"
#: Where ReShade puts its shaders when ReShade.ini names no search path.
DEFAULT_SHADERS = Path("reshade-shaders") / "Shaders"


@dataclass
class Result:
    ok: bool
    message: str
    installed: list[Path] = field(default_factory=list)


def bundled_dir() -> Path:
    """Where the app keeps its copy of the add-on."""
    if paths.is_frozen():
        return paths.app_dir() / paths.ENGINE_DIR / "capture"
    return paths.app_dir() / "native" / "bin"


def bundled_files() -> tuple[Path, Path]:
    d = bundled_dir()
    return d / ADDON, d / EFFECT


def game_folder(picked: str | Path) -> Path:
    """The folder of the game's executable, whether the user picked the .exe
    or the folder. (Unreal games keep it in Binaries/Win64, not the top.)"""
    p = Path(picked)
    return p.parent if p.suffix.lower() == ".exe" else p


def has_reshade(folder: Path) -> bool:
    return (folder / "ReShade.ini").is_file()


def shader_folder(folder: Path) -> Path:
    """The first shader folder ReShade.ini lists, else ReShade's default.

    EffectSearchPaths is a comma-separated list; entries may be relative to
    the game folder, absolute, and end in \\** for 'and subfolders'.
    """
    ini = folder / "ReShade.ini"
    try:
        parser = configparser.ConfigParser(interpolation=None, strict=False)
        parser.optionxform = str
        parser.read_string(ini.read_text(encoding="utf-8", errors="replace"))
        for section in parser.sections():
            raw = parser[section].get("EffectSearchPaths")
            if not raw:
                continue
            for entry in raw.split(","):
                entry = entry.strip().strip('"').rstrip("*").rstrip("\\/")
                if not entry:
                    continue
                path = Path(entry)
                if not path.is_absolute():
                    path = folder / path
                return path
    except Exception:  # noqa: BLE001 - an odd ini must not block installing
        pass
    return folder / DEFAULT_SHADERS


def is_installed(picked: str | Path) -> bool:
    return (game_folder(picked) / ADDON).is_file()


def install(picked: str | Path) -> Result:
    folder = game_folder(picked)
    addon, effect = bundled_files()
    if not addon.is_file():
        return Result(False, "This copy of the app does not include the capture add-on. "
                             "Download the full release again.")
    if not folder.is_dir():
        return Result(False, f"{folder} does not exist.")
    if not has_reshade(folder):
        return Result(False,
                      "ReShade is not installed in this game (no ReShade.ini next to its .exe). "
                      "Install ReShade with full add-on support into the game first, from "
                      "reshade.me, then try again. For Unreal games, pick the .exe inside "
                      "Binaries\\Win64.")
    shaders = shader_folder(folder)
    try:
        shaders.mkdir(parents=True, exist_ok=True)
        shutil.copy2(addon, folder / ADDON)
        done = [folder / ADDON]
        if effect.is_file():
            shutil.copy2(effect, shaders / EFFECT)
            done.append(shaders / EFFECT)
    except PermissionError:
        return Result(False, "Windows would not let the app write there (the game folder may need "
                             "administrator rights, or the game is running). Close the game and try "
                             "again, or run the app as administrator.")
    except OSError as error:
        return Result(False, f"Could not copy the add-on: {error}")
    return Result(True,
                  "Installed. Start the game, open the ReShade overlay (Home key) and check the "
                  "Add-ons tab lists DLSS5 Scene Capture. Then press F10 to take a shot with its "
                  "depth; a banner confirms each capture. Shots are saved to ReShade's screenshot "
                  "folder, which you can change in the add-on's settings.", done)


def uninstall(picked: str | Path) -> Result:
    folder = game_folder(picked)
    removed = []
    for path in (folder / ADDON, shader_folder(folder) / EFFECT):
        try:
            if path.is_file():
                path.unlink()
                removed.append(path)
        except OSError as error:
            return Result(False, f"Could not remove {path.name}: {error}. Close the game and try again.")
    if not removed:
        return Result(True, "The capture add-on was not installed in this game.")
    return Result(True, "Removed the capture add-on from this game.", removed)
