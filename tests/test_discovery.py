"""Finding the user's own DLSS files, and preferring a matched set."""

from __future__ import annotations

from pathlib import Path

from dlss5_converter import discovery
from dlss5_converter.discovery import (
    ADDON,
    DLSS_DLL,
    NEURAL_DLL,
    RESHADE,
    Candidate,
    best_set,
    install,
    scan,
)


def big(path: Path, megabytes: float, marker: bytes = b"") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(marker)
        handle.write(b"\0" * int(megabytes * 1024 * 1024))
    return path


def full_set(folder: Path) -> Path:
    big(folder / NEURAL_DLL, 101)
    big(folder / DLSS_DLL, 1)
    big(folder / ADDON, 0.5)
    big(folder / RESHADE, 3, marker=b"ReShadeRegisterAddon")
    return folder


def test_a_complete_game_folder_is_found(tmp_path):
    game = full_set(tmp_path / "SteamLibrary" / "common" / "Some Game" / "bin" / "x64")
    results = scan([tmp_path])
    assert results
    assert results[0].folder == game
    assert results[0].complete


def test_the_most_complete_folder_ranks_first(tmp_path):
    full_set(tmp_path / "game" / "bin")
    big(tmp_path / "downloads" / NEURAL_DLL, 101)  # a lone stray
    results = scan([tmp_path])
    assert results[0].complete
    assert results[0].folder.name == "bin"


def test_a_stub_runtime_is_rejected(tmp_path):
    """A 2 MB nvngx_dlssnr.dll is not the 158 MB model."""
    big(tmp_path / "fake" / NEURAL_DLL, 2)
    assert scan([tmp_path]) == []


def test_the_system_dxgi_is_not_mistaken_for_reshade(tmp_path):
    """Windows' own dxgi.dll is everywhere and is not what we want."""
    big(tmp_path / "sys" / RESHADE, 1)                      # too small
    big(tmp_path / "app" / RESHADE, 3)                      # big, but no marker
    assert scan([tmp_path]) == []

    big(tmp_path / "game" / RESHADE, 3, marker=b"ReShadeRegisterAddon")
    results = scan([tmp_path])
    assert [c.folder.name for c in results] == ["game"]


def test_a_complete_folder_is_used_without_mixing(tmp_path):
    """Mixing sources is what makes the add-on reject a runtime."""
    game = full_set(tmp_path / "game" / "bin")
    other = tmp_path / "elsewhere"
    big(other / NEURAL_DLL, 101)

    chosen = best_set(scan([tmp_path]))
    assert set(chosen) == {NEURAL_DLL, DLSS_DLL, ADDON, RESHADE}
    # Every file came from the one complete folder.
    assert all(path.parent == game for path in chosen.values())


def test_an_incomplete_set_is_topped_up_from_elsewhere(tmp_path):
    """Four files from three places still beats sending someone hunting."""
    partial = tmp_path / "game" / "bin"
    big(partial / NEURAL_DLL, 101)
    big(partial / DLSS_DLL, 1)
    big(partial / ADDON, 0.5)
    big(tmp_path / "reshade" / RESHADE, 3, marker=b"ReShadeRegisterAddon")

    chosen = best_set(scan([tmp_path]))
    assert set(chosen) == {NEURAL_DLL, DLSS_DLL, ADDON, RESHADE}
    assert chosen[RESHADE].parent.name == "reshade"


def test_install_copies_and_never_overwrites(tmp_path):
    game = full_set(tmp_path / "game")
    destination = tmp_path / "dlss_files"
    destination.mkdir()
    (destination / DLSS_DLL).write_bytes(b"the one the user chose")

    copied, skipped = install(best_set(scan([tmp_path / "game"])), destination)

    assert set(copied) == {NEURAL_DLL, ADDON, RESHADE}
    assert DLSS_DLL in skipped
    assert (destination / DLSS_DLL).read_bytes() == b"the one the user chose"
    assert (destination / NEURAL_DLL).stat().st_size > 100 * 1024 * 1024


def test_scanning_can_be_stopped(tmp_path):
    full_set(tmp_path / "game" / "bin")
    assert scan([tmp_path], should_stop=lambda: True) == []


def test_depth_limit_is_respected(tmp_path):
    deep = tmp_path
    for level in range(12):
        deep = deep / f"level{level}"
    full_set(deep)
    assert scan([tmp_path], max_depth=4) == []
    assert scan([tmp_path], max_depth=20)


def test_describe_names_what_is_missing(tmp_path):
    candidate = Candidate(tmp_path, {NEURAL_DLL: tmp_path / NEURAL_DLL})
    text = candidate.describe()
    assert "1 of 4" in text
    assert DLSS_DLL in text


def test_default_roots_are_real_directories():
    assert all(root.is_dir() for root in discovery.default_roots())


def test_the_newest_addon_wins_even_from_another_folder(tmp_path):
    """Games keep whichever add-on was current when they were modded.

    An out-of-date renodx-dlss5.addon64 makes the neural pass silently not run,
    and on a real machine five complete game folders all shared one old copy
    while the updated one sat in Downloads.
    """
    import os
    import time

    game = full_set(tmp_path / "game" / "bin")
    old = time.time() - 60 * 60 * 24 * 30
    for path in game.iterdir():
        os.utime(path, (old, old))

    fresh = tmp_path / "downloads"
    big(fresh / ADDON, 1.7)  # newer, and left with a current mtime

    chosen = best_set(scan([tmp_path]))
    assert chosen[ADDON].parent == fresh, "should prefer the newer add-on"
    # Everything else still comes from the one complete, matched folder.
    for name in (NEURAL_DLL, DLSS_DLL, RESHADE):
        assert chosen[name].parent == game


def test_a_fresher_complete_folder_outranks_an_older_one(tmp_path):
    import os
    import time

    old_game = full_set(tmp_path / "old" / "bin")
    stale = time.time() - 60 * 60 * 24 * 90
    for path in old_game.iterdir():
        os.utime(path, (stale, stale))
    new_game = full_set(tmp_path / "new" / "bin")

    results = scan([tmp_path])
    assert results[0].folder == new_game
