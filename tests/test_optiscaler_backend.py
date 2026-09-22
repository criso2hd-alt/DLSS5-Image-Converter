"""The OptiScaler backend: found, staged into its own folder, configured."""
from __future__ import annotations

from pathlib import Path

import pytest

from dlss5_converter import onboarding, paths, runtime
from dlss5_converter.settings import NeuralSettings


@pytest.fixture
def setup(tmp_path, monkeypatch):
    files = tmp_path / "dlss_files"
    release = files / "OptiScaler-NR-v0.8.8"          # extracted with its own folder
    (release / "OptiScaler").mkdir(parents=True)
    (release / "OptiScaler.dll").write_bytes(b"proxy")
    (release / "nvngx.dll_dlssnr.dll").write_bytes(b"forwarder")
    (release / "OptiScaler" / "backend.dll").write_bytes(b"backend")
    (release / "setup_windows.bat").write_text("echo")
    (release / "OptiScaler.ini").write_text(
        "; comment\n[Upscalers]\n; which one\nDx12Upscaler=auto\n\n"
        "[DlssNr]\nEnabled=auto\nPasses=auto\nStyle=auto\n", encoding="utf-8")
    (files / "nvngx_dlssnr.dll").write_bytes(b"model")
    (files / "nvngx_dlss.dll").write_bytes(b"dlss")
    engine = tmp_path / "engine"
    engine.mkdir()
    (engine / "dlss5_eval.exe").write_bytes(b"harness")
    monkeypatch.setattr(paths, "dlss_files_dir", lambda: files)
    monkeypatch.setattr(paths, "runtime_search_roots", lambda: [files])
    monkeypatch.setattr(paths, "deep_search_roots", lambda: [files])
    monkeypatch.setattr(paths, "native_exe", lambda: engine / "dlss5_eval.exe")
    monkeypatch.setattr(paths, "optiscaler_engine_dir", lambda: tmp_path / "engine_optiscaler")
    runtime.set_backend("optiscaler")
    yield tmp_path, release
    runtime.set_backend("renodx")


def test_found_and_ready_without_reshade_or_the_addon(setup):
    _, release = setup
    status = runtime.detect()
    assert status.backend == "optiscaler" and status.optiscaler == release
    assert status.ready, status.problems


def test_staging_renames_the_proxy_and_keeps_the_layout(setup):
    tmp_path, _ = setup
    status = runtime.detect()
    target = runtime.stage_runtime(status)
    assert target == tmp_path / "engine_optiscaler"
    assert (target / "dxgi.dll").read_bytes() == b"proxy"
    assert (target / "nvngx.dll_dlssnr.dll").is_file()
    assert (target / "OptiScaler" / "backend.dll").is_file()
    assert (target / "dlss5_eval.exe").is_file()
    assert (target / "nvngx_dlssnr.dll").is_file() and (target / "nvngx_dlss.dll").is_file()
    assert not (target / "setup_windows.bat").exists()
    assert status.harness == target / "dlss5_eval.exe"


def test_config_carries_the_settings_and_keeps_comments(setup):
    tmp_path, _ = setup
    status = runtime.detect()
    target = runtime.stage_runtime(status)
    neural = NeuralSettings(style=1, intensity=1.5, passes=2)
    ini = runtime.write_config(target, neural).read_text(encoding="utf-8")
    assert "; which one" in ini                           # release comments kept
    assert "Dx12Upscaler=dlss" in ini and "Dx12Upscaler=auto" not in ini
    assert "Enabled=true" in ini and "Passes=2" in ini and "Style=1" in ini
    assert "Intensity=1.5000" in ini
    assert "TargetProcessName=dlss5_eval.exe" in ini      # section added when missing
    assert "OverlayMenu=false" in ini


def test_a_half_extracted_release_is_named(setup):
    """Only OptiScaler.dll, without the backend folder beside it."""
    _, release = setup
    (release / "OptiScaler" / "backend.dll").unlink()
    status = runtime.detect()
    assert any("extract the whole release" in p for p in status.problems)


def test_the_obsolete_forwarder_is_not_required(setup):
    """Current releases drop nvngx.dll_dlssnr.dll and their install guide says
    to delete it. Requiring it made a correct install look broken."""
    _, release = setup
    (release / "nvngx.dll_dlssnr.dll").unlink()
    assert runtime.detect().ready


def test_probe_success_does_not_need_reshade_flags(setup):
    report = ("dlss_available: 1\nneural_addon_loaded: 0\nreshade_proxy_loaded: 0\n"
              "dlssnr_module_loaded: 1\ntest_evaluation: ok\n")
    assert onboarding.probe_succeeded(report)
    runtime.set_backend("renodx")
    assert not onboarding.probe_succeeded(report)          # RenoDX still needs them


def test_backend_setting_survives_a_restart(tmp_path):
    from dlss5_converter.settings import AppSettings
    s = AppSettings()
    s.backend = "optiscaler"
    s.neural.passes = 3
    s.save(tmp_path / "s.json")
    loaded = AppSettings.load(tmp_path / "s.json")
    assert loaded.backend == "optiscaler" and loaded.neural.passes == 3
