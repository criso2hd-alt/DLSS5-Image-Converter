"""Every top-level setting must survive save + load (the v0.4.4 GPU choice
was saved but never read back, so it reset on every restart)."""
from __future__ import annotations

from dataclasses import fields

from dlss5_converter.settings import AppSettings


def test_every_plain_setting_round_trips(tmp_path):
    path = tmp_path / "settings.json"
    s = AppSettings()
    changed = {}
    for f in fields(AppSettings):
        value = getattr(s, f.name)
        if isinstance(value, str):
            changed[f.name] = f"x-{f.name}"
        elif isinstance(value, int) and not isinstance(value, bool):
            changed[f.name] = value + 7
    for name, value in changed.items():
        setattr(s, name, value)
    s.save(path)
    loaded = AppSettings.load(path)
    for name, value in changed.items():
        assert getattr(loaded, name) == value, name


def test_gpu_choice_survives_a_restart(tmp_path):
    path = tmp_path / "settings.json"
    s = AppSettings()
    s.gpu = "NVIDIA GeForce RTX 5060"
    s.dlss_adapter = "NVIDIA GeForce RTX 5060"
    s.save(path)
    loaded = AppSettings.load(path)
    assert loaded.gpu == "NVIDIA GeForce RTX 5060"
    assert loaded.dlss_adapter == "NVIDIA GeForce RTX 5060"
