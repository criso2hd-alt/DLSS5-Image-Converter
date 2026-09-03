"""Keeping the staged runtime in step with the files the user chose.

The app copies the four DLSS files next to the harness, because that is where
NGX and ReShade look. If that copy is allowed to go stale, updating the add-on
does nothing — and "your add-on is out of date" is the first thing the
troubleshooting guide tells people to fix. A stale stage turns that advice into
a dead end, with every indicator still green.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from dlss5_converter import runtime
from dlss5_converter.runtime import RuntimeStatus, stage_runtime


def write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


@pytest.fixture
def staged(tmp_path):
    """A harness folder, and a source folder holding the four files."""
    harness = write(tmp_path / "engine" / "dlss5_eval.exe", b"harness")
    source = tmp_path / "dlss_files"

    def status() -> RuntimeStatus:
        return RuntimeStatus(
            harness=harness,
            neural_dll=source / "nvngx_dlssnr.dll",
            dlss_dll=source / "nvngx_dlss.dll",
            addon=source / "renodx-dlss5.addon64",
            reshade=source / "dxgi.dll",
        )

    write(source / "nvngx_dlssnr.dll", b"neural" * 100)
    write(source / "nvngx_dlss.dll", b"dlss" * 100)
    write(source / "renodx-dlss5.addon64", b"addon-v1")
    write(source / "dxgi.dll", b"reshade" * 100)
    return harness.parent, source, status


def test_the_files_are_placed_beside_the_harness(staged):
    engine, _source, status = staged
    stage_runtime(status())
    for name in (
        "nvngx_dlssnr.dll", "nvngx_dlss.dll", "renodx-dlss5.addon64", "dxgi.dll"
    ):
        assert (engine / name).exists(), name


def test_a_replacement_of_the_same_size_is_picked_up(staged):
    """The bug. Two add-on builds are frequently byte-for-byte the same size.

    Skipping on size alone left the old one staged for ever, which presents as
    the neural pass silently not running - the exact symptom the user was
    updating the add-on to cure.
    """
    engine, source, status = staged
    stage_runtime(status())
    assert (engine / "renodx-dlss5.addon64").read_bytes() == b"addon-v1"

    # A genuine replacement: unlink first, so the new file gets a new inode.
    # Writing through the old path would truncate the very inode the staged
    # copy is hard linked to, and the destination would change by itself - a
    # test that passes without the code under test doing anything at all.
    replacement = source / "renodx-dlss5.addon64"
    replacement.unlink()
    replacement.write_bytes(b"addon-v2")
    assert (engine / "renodx-dlss5.addon64").read_bytes() == b"addon-v1", (
        "the staged copy must still hold the old build, or this proves nothing"
    )
    later = time.time() + 10
    os.utime(replacement, (later, later))

    stage_runtime(status())
    assert (engine / "renodx-dlss5.addon64").read_bytes() == b"addon-v2"


def test_restaging_is_idempotent(staged):
    engine, _source, status = staged
    stage_runtime(status())
    first = (engine / "nvngx_dlssnr.dll").stat()
    stage_runtime(status())
    again = (engine / "nvngx_dlssnr.dll").stat()
    assert (first.st_ino, first.st_size) == (again.st_ino, again.st_size)


def test_a_hard_link_counts_as_already_staged(staged, tmp_path):
    """The normal case: same volume, so the staged file *is* the source."""
    engine, source, status = staged
    stage_runtime(status())
    here = (engine / "renodx-dlss5.addon64").stat()
    there = (source / "renodx-dlss5.addon64").stat()
    if here.st_ino and here.st_ino == there.st_ino:
        assert runtime._already_staged(
            source / "renodx-dlss5.addon64", engine / "renodx-dlss5.addon64"
        )
    else:  # pragma: no cover - filesystem refused links
        pytest.skip("hard links unavailable here")


def test_a_copy_that_matches_is_left_alone(staged):
    """Nothing should re-copy 158 MB on every launch."""
    engine, source, _status = staged
    target = engine / "copy.bin"
    original = source / "nvngx_dlssnr.dll"
    import shutil

    shutil.copy2(original, target)
    assert runtime._already_staged(original, target)


def test_a_missing_destination_is_never_considered_staged(staged):
    _engine, source, _status = staged
    assert not runtime._already_staged(
        source / "renodx-dlss5.addon64", source / "nope.addon64"
    )
