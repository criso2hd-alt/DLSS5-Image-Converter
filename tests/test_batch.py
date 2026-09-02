"""Folder batch: listing, and the guards that run before any GPU work."""

from __future__ import annotations

from pathlib import Path

import pytest

from dlss5_converter import pipeline
from dlss5_converter.settings import AppSettings


def make(folder: Path, name: str) -> Path:
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_only_images_are_listed(tmp_path):
    make(tmp_path, "a.png")
    make(tmp_path, "b.JPG")          # case is not significant
    make(tmp_path, "notes.txt")
    make(tmp_path, "clip.mp4")
    assert [p.name for p in pipeline.list_images(tmp_path)] == ["a.png", "b.JPG"]


def test_listing_is_flat_unless_asked(tmp_path):
    make(tmp_path, "top.png")
    make(tmp_path, "sub/nested.png")
    assert len(pipeline.list_images(tmp_path)) == 1
    assert len(pipeline.list_images(tmp_path, recursive=True)) == 2


def test_listing_is_sorted(tmp_path):
    for name in ("c.png", "a.png", "b.png"):
        make(tmp_path, name)
    assert [p.name for p in pipeline.list_images(tmp_path)] == ["a.png", "b.png", "c.png"]


def test_renderer_formats_are_included(tmp_path):
    """EXR and HDR matter here - this is aimed at 3D output."""
    for name in ("beauty.exr", "sky.hdr", "plate.tif", "shot.webp"):
        make(tmp_path, name)
    assert len(pipeline.list_images(tmp_path)) == 4


def test_an_empty_folder_is_a_no_op(tmp_path):
    """Returns before touching the DLSS runtime, so it works on any machine."""
    assert list(pipeline.convert_batch([], AppSettings(), None, tmp_path)) == []


def test_batch_is_a_generator_and_does_nothing_until_iterated(tmp_path):
    """Nothing should start merely from calling it."""
    from dlss5_converter import pipeline as module

    result = module.convert_batch([tmp_path / "a.png"], AppSettings(), None, tmp_path)
    assert hasattr(result, "__next__")
    # Not iterated, so no output folder was created and no runtime was touched.
    assert not (tmp_path / "a_dlss5.png").exists()
