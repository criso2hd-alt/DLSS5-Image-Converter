"""The post-DLSS effects stack: neutrality, range, LUTs, and HDR safety."""

from __future__ import annotations

import numpy as np
import pytest

from dlss5_converter import effects
from dlss5_converter.settings import AppSettings, EffectsSettings


def _image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((48, 64, 3), dtype=np.float32)


def test_neutral_stack_returns_the_input_untouched():
    # The free-when-off contract: nothing enabled means the exact same array
    # back, so an idle effects stack costs nothing on every save and frame.
    img = _image()
    assert effects.apply(img, EffectsSettings()) is img


@pytest.mark.parametrize(
    "field",
    ["sharpen", "bloom", "chroma", "crt", "vignette", "grain"],
)
def test_each_effect_runs_and_stays_in_sdr_range(field):
    settings = EffectsSettings()
    setattr(settings, f"{field}_enabled", True)
    out = effects.apply(_image(), settings)
    assert out.shape == (48, 64, 3)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_hdr_preserve_range_keeps_highlights_above_white():
    # An EXR / JPEG XR source carries values above 1.0. With preserve_range the
    # stack must not crush them to white — the whole point of the HDR path.
    hdr = _image().astype(np.float32) * 3.0
    settings = EffectsSettings(
        vignette_enabled=True, sharpen_enabled=True, grain_enabled=True
    )
    out = effects.apply(hdr, settings, preserve_range=True)
    assert out.max() > 1.5
    assert out.min() >= 0.0


def test_sdr_path_clamps_but_hdr_path_does_not():
    hdr = np.full((8, 8, 3), 2.0, dtype=np.float32)
    settings = EffectsSettings(vignette_enabled=True)
    assert effects.apply(hdr, settings).max() <= 1.0
    assert effects.apply(hdr, settings, preserve_range=True).max() > 1.0


def _write_identity_cube(path, size=2):
    lines = ["LUT_3D_SIZE %d" % size, "DOMAIN_MIN 0 0 0", "DOMAIN_MAX 1 1 1"]
    for b in range(size):
        for g in range(size):
            for r in range(size):
                lines.append(f"{r/(size-1)} {g/(size-1)} {b/(size-1)}")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_identity_cube_is_near_identity(tmp_path):
    cube = tmp_path / "identity.cube"
    _write_identity_cube(cube, size=2)
    lut = effects.load_cube(cube)
    img = _image()
    assert np.abs(lut.apply(img) - img).max() < 0.02


def test_load_cube_rejects_a_1d_lut(tmp_path):
    cube = tmp_path / "one_d.cube"
    cube.write_text("LUT_1D_SIZE 2\n0 0 0\n1 1 1", encoding="utf-8")
    with pytest.raises(ValueError, match="1D"):
        effects.load_cube(cube)


def test_load_cube_rejects_wrong_entry_count(tmp_path):
    cube = tmp_path / "short.cube"
    cube.write_text("LUT_3D_SIZE 2\n0 0 0\n1 1 1", encoding="utf-8")
    with pytest.raises(ValueError):
        effects.load_cube(cube)


def test_available_luts_lists_only_cube_files(tmp_path):
    (tmp_path / "a.cube").write_text("LUT_3D_SIZE 2", encoding="utf-8")
    (tmp_path / "b.cube").write_text("LUT_3D_SIZE 2", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    assert effects.available_luts(tmp_path) == ["a.cube", "b.cube"]


def test_a_bad_lut_degrades_to_no_lut_rather_than_raising(tmp_path):
    # A broken .cube selected in the UI must never take a conversion down; the
    # cached loader swallows it and the stack simply skips the LUT.
    bad = tmp_path / "bad.cube"
    bad.write_text("this is not a cube file", encoding="utf-8")
    settings = EffectsSettings(lut_enabled=True, lut_name="bad.cube", lut_amount=1.0)
    img = _image()
    out = effects.apply(img, settings, luts_dir=tmp_path)
    # Nothing else is on, so with the LUT skipped the image is unchanged.
    assert np.array_equal(out, np.clip(img, 0.0, 1.0))


def test_settings_round_trip_restores_effects_and_grade(tmp_path):
    # Guards the load() fix: both effects and grade used to be dropped on load.
    path = tmp_path / "settings.json"
    a = AppSettings()
    a.effects.crt_enabled = True
    a.effects.crt_scanline = 0.7
    a.effects.lut_name = "film.cube"
    a.grade.exposure = 0.5
    a.save(path)

    b = AppSettings.load(path)
    assert b.effects.crt_enabled is True
    assert b.effects.crt_scanline == pytest.approx(0.7)
    assert b.effects.lut_name == "film.cube"
    assert b.grade.exposure == pytest.approx(0.5)


def test_describe_names_the_enabled_effects():
    settings = EffectsSettings(sharpen_enabled=True, crt_enabled=True)
    described = effects.describe(settings)
    assert "Sharpen" in described and "Crt" in described
    assert effects.describe(EffectsSettings()) == "None"
