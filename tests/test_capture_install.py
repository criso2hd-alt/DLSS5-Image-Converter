"""Installing the scene capture add-on into a game with ReShade."""

from __future__ import annotations

import pytest

from dlss5_converter import capture_install as ci


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    src = tmp_path / "bundle"
    src.mkdir()
    (src / ci.ADDON).write_bytes(b"addon")
    (src / ci.EFFECT).write_text("// fx", encoding="utf-8")
    monkeypatch.setattr(ci, "bundled_dir", lambda: src)
    return src


def _game(tmp_path, ini: str | None = "[GENERAL]\n"):
    game = tmp_path / "Game" / "Binaries" / "Win64"
    game.mkdir(parents=True)
    exe = game / "Game.exe"
    exe.write_bytes(b"")
    if ini is not None:
        (game / "ReShade.ini").write_text(ini, encoding="utf-8")
    return exe


def test_installs_next_to_the_exe_and_into_reshades_default_shader_folder(tmp_path, bundle):
    exe = _game(tmp_path)
    result = ci.install(exe)
    assert result.ok
    assert (exe.parent / ci.ADDON).read_bytes() == b"addon"
    assert (exe.parent / "reshade-shaders" / "Shaders" / ci.EFFECT).is_file()
    assert ci.is_installed(exe)


def test_uses_the_shader_folder_reshade_ini_names(tmp_path, bundle):
    exe = _game(tmp_path, "[GENERAL]\nEffectSearchPaths=.\custom-shaders\**,.\other\n")
    assert ci.install(exe).ok
    assert (exe.parent / "custom-shaders" / ci.EFFECT).is_file()
    assert not (exe.parent / "reshade-shaders").exists()


def test_refuses_a_game_without_reshade(tmp_path, bundle):
    exe = _game(tmp_path, ini=None)
    result = ci.install(exe)
    assert not result.ok and "ReShade" in result.message
    assert not (exe.parent / ci.ADDON).exists()


def test_folder_or_exe_both_work(tmp_path, bundle):
    exe = _game(tmp_path)
    assert ci.install(exe.parent).ok
    assert ci.is_installed(exe)


def test_remove_takes_both_files_out(tmp_path, bundle):
    exe = _game(tmp_path)
    ci.install(exe)
    result = ci.uninstall(exe)
    assert result.ok and len(result.installed) == 2
    assert not ci.is_installed(exe)
    assert not (exe.parent / "reshade-shaders" / "Shaders" / ci.EFFECT).exists()


def test_missing_bundle_is_explained(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "bundled_dir", lambda: tmp_path / "nothing")
    result = ci.install(_game(tmp_path))
    assert not result.ok and "add-on" in result.message
