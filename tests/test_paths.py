"""The release layout.

A frozen build keeps everything beside the executable — dlss_files, models,
output — which is the opposite of what a source checkout does. These tests pin
that split down, because getting it wrong is silent: the app still starts, it
just writes a 400 MB model download somewhere the user will never find.
"""

from __future__ import annotations

import sys

import pytest

from dlss5_converter import paths


@pytest.fixture
def frozen(monkeypatch, tmp_path):
    """Pretend to be a PyInstaller build living in `tmp_path`.

    HF_HOME is cleared because importing ``depth_engine`` anywhere in the suite
    sets it at module scope, and it outranks the models folder by design — so
    without this the result depends on test ordering.
    """
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "DLSS5Converter.exe"))
    return tmp_path


@pytest.fixture
def source(monkeypatch):
    """Pretend to be a source checkout, whatever the real interpreter is."""
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)
    return None


def test_frozen_build_keeps_everything_beside_the_executable(frozen):
    assert paths.is_frozen()
    assert paths.is_portable()
    assert paths.data_dir() == frozen
    assert paths.model_cache_dir() == frozen / "models"
    assert paths.output_dir() == frozen / "output"
    assert paths.dlss_files_dir() == frozen / "dlss_files"
    assert paths.native_exe() == frozen / "engine" / "dlss5_eval.exe"


def test_source_checkout_keeps_data_in_the_user_profile(source):
    assert not paths.is_frozen()
    # The project root is not where a 400 MB download belongs.
    assert paths.data_dir() != paths.app_dir()
    assert paths.model_cache_dir().name == "models"
    assert paths.native_exe().parts[-3:] == ("native", "bin", "dlss5_eval.exe")


def test_scratch_stays_out_of_the_release_root(frozen):
    """Transient 100 MB planes must not clutter the folder the user looks at."""
    scratch = paths.scratch_dir()
    assert scratch.parent == frozen / "engine"
    assert scratch.parent.parent == frozen


def test_hf_home_overrides_the_models_folder(frozen, monkeypatch):
    """A user who already has a Hugging Face cache should not pay twice."""
    monkeypatch.setenv("HF_HOME", str(frozen / "elsewhere"))
    assert paths.model_cache_dir() == frozen / "elsewhere"


def test_dlss_files_is_searched_first(frozen):
    """The folder the app told the user to fill outranks any stray copy.

    Nested layouts are handled by searching recursively rather than by listing
    the subfolder names we expect - guessing them meant a Streamline unpacked
    as "streamline/" instead of "NVStreamline/" was reported as missing.
    """
    (frozen / "dlss_files").mkdir()
    roots = paths.runtime_search_roots()
    assert roots[0] == frozen / "dlss_files"


def test_output_and_models_do_not_collide(frozen):
    assert paths.output_dir() != paths.model_cache_dir()
    assert paths.output_dir().is_dir()


def test_release_searches_only_dlss_files(frozen):
    """A release must not find files anywhere the user cannot see.

    The Downloads fallbacks made this project's author's machine work with an
    incomplete dlss_files while an outside tester with the same folder failed.
    A convenience that hides a broken setup from the person able to fix it is
    not one.
    """
    (frozen / "dlss_files").mkdir()
    assert paths.runtime_search_roots() == [frozen / "dlss_files"]


def test_source_checkout_keeps_the_developer_fallbacks(source, monkeypatch, tmp_path):
    """Convenient in a checkout, where nobody is shipping the result."""
    monkeypatch.setattr(paths, "data_dir", lambda: tmp_path)
    (tmp_path / "dlss_files").mkdir()
    (tmp_path / "runtime").mkdir()
    assert (tmp_path / "runtime") in paths.runtime_search_roots()


def test_streamline_is_found_however_it_was_unpacked(frozen):
    """streamline/, NVStreamline/, or an extra wrapper - all must work."""
    files = frozen / "dlss_files"
    for layout in (
        files / "streamline" / "Production",
        files / "NVStreamline" / "Production",
        files / "sl" / "extracted" / "bin" / "x64",
    ):
        layout.mkdir(parents=True)
        target = layout / "nvngx_dlss.dll"
        target.write_bytes(b"x")
        assert paths.find_runtime_file("nvngx_dlss.dll") == target
        target.unlink()


def test_a_loose_file_beats_one_buried_in_an_archive_folder(frozen):
    """The copy the user placed deliberately wins over an unpacked one."""
    files = frozen / "dlss_files"
    nested = files / "streamline" / "Production"
    nested.mkdir(parents=True)
    (nested / "nvngx_dlss.dll").write_bytes(b"nested")
    loose = files / "nvngx_dlss.dll"
    loose.write_bytes(b"loose")
    assert paths.find_runtime_file("nvngx_dlss.dll") == loose
