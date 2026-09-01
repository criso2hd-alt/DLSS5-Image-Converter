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
    """The folder the app told the user to fill outranks any stray copy."""
    (frozen / "dlss_files").mkdir()
    (frozen / "dlss_files" / "NVStreamline" / "Production").mkdir(parents=True)
    roots = paths.runtime_search_roots()
    assert roots[0] == frozen / "dlss_files"
    # Streamline drops are searched unflattened, because that is how they unzip.
    assert frozen / "dlss_files" / "NVStreamline" / "Production" in roots


def test_output_and_models_do_not_collide(frozen):
    assert paths.output_dir() != paths.model_cache_dir()
    assert paths.output_dir().is_dir()
