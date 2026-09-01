"""The [RenoDX.DLSS5] section the harness is launched with.

This is the only route to the neural pass — the add-on publishes no NGX
parameter and ignores the environment — and it is read once at startup, so a
mistake here is silent: the conversion still runs and still produces a picture,
just not the one the user asked for.

Every ceiling below was measured by raising the value until the output stopped
changing. They are not the same, which is the whole reason these tests exist.
"""

from __future__ import annotations

import configparser

import pytest

from dlss5_converter import runtime
from dlss5_converter.settings import (
    NR_COLOR_MAX,
    NR_PAPER_WHITE_MAX,
    NR_PRESETS,
    NR_STRENGTH_MAX,
    NR_STYLES,
    NR_TRANSFER_MAX,
    NeuralSettings,
)


def read_section(path):
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(path, encoding="utf-8")
    return dict(parser[runtime.ADDON_SECTION])


def test_sliders_reach_the_addon(tmp_path):
    neural = NeuralSettings(intensity=1.5, skin=0.25, local_tone=0.75, structure=2.0)
    section = read_section(runtime.write_addon_config(tmp_path, neural))
    assert float(section["NRIntensity"]) == pytest.approx(1.5)
    assert float(section["NRSkinStructure"]) == pytest.approx(0.25)
    assert float(section["NRLocalTone"]) == pytest.approx(0.75)
    assert float(section["NRLocalStructure"]) == pytest.approx(2.0)
    assert section["NeuralUplift"] == "1"


def test_each_knob_clamps_to_its_own_ceiling(tmp_path):
    """A shared limit would truncate paper-white or overrun colour."""
    absurd = NeuralSettings(
        intensity=99, skin=99, local_tone=99, structure=99,
        color_strength=99, transfer_strength=99, paper_white=99,
    )
    section = read_section(runtime.write_addon_config(tmp_path, absurd))
    assert float(section["NRIntensity"]) == pytest.approx(NR_STRENGTH_MAX)
    assert float(section["NRColorStrength"]) == pytest.approx(NR_COLOR_MAX)
    assert float(section["NRTransferStrength"]) == pytest.approx(NR_TRANSFER_MAX)
    assert float(section["NRPaperWhiteScale"]) == pytest.approx(NR_PAPER_WHITE_MAX)
    # The ceilings genuinely differ; if they ever collapse to one value this
    # test still passes but the point of it is gone, so assert that too.
    assert len({NR_STRENGTH_MAX, NR_COLOR_MAX, NR_PAPER_WHITE_MAX}) == 3


def test_negatives_clamp_to_zero(tmp_path):
    section = read_section(runtime.write_addon_config(tmp_path, NeuralSettings(skin=-5)))
    assert float(section["NRSkinStructure"]) == pytest.approx(0.0)


def test_zero_intensity_disables_the_pass_outright(tmp_path):
    """NRIntensity=0 is not the same as off; only NeuralUplift=0 is plain DLAA."""
    section = read_section(runtime.write_addon_config(tmp_path, NeuralSettings(intensity=0.0)))
    assert section["NeuralUplift"] == "0"


def test_enum_indices_stay_inside_the_addons_own_lists(tmp_path):
    """Game inis carry NRStyle=2 with only two styles; never pass that through."""
    section = read_section(runtime.write_addon_config(tmp_path, NeuralSettings(preset=9, style=9)))
    assert int(section["NRPreset"]) == len(NR_PRESETS) - 1
    assert int(section["NRStyle"]) == len(NR_STYLES) - 1


def test_upscaling_stays_pinned_off(tmp_path):
    """DLAA only — render size equals output size. A non-goal, enforced here."""
    section = read_section(runtime.write_addon_config(tmp_path, NeuralSettings()))
    assert section["NREnableUpscaling"] == "0"


def test_reshades_own_settings_survive(tmp_path):
    """ReShade owns this file. Stamping our keys over it must not reset it."""
    path = tmp_path / "ReShade.ini"
    path.write_text(
        "[GENERAL]\nEffectSearchPaths=.\\shaders\n\n"
        "[INPUT]\nKeyOverlay=36,0,0,0\n\n"
        f"[{runtime.ADDON_SECTION}]\nNRIntensity=0.1\nNRToggleKey=117\n",
        encoding="utf-8",
    )
    runtime.write_addon_config(tmp_path, NeuralSettings(intensity=1.25))

    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(path, encoding="utf-8")
    assert parser["GENERAL"]["EffectSearchPaths"] == ".\\shaders"
    assert parser["INPUT"]["KeyOverlay"] == "36,0,0,0"
    # Keys inside our own section that we do not manage are left alone too.
    assert parser[runtime.ADDON_SECTION]["NRToggleKey"] == "117"
    assert float(parser[runtime.ADDON_SECTION]["NRIntensity"]) == pytest.approx(1.25)


def test_key_case_is_preserved(tmp_path):
    """ReShade's keys are case-sensitive; configparser lowercases by default."""
    raw = runtime.write_addon_config(tmp_path, NeuralSettings()).read_text(encoding="utf-8")
    assert "NRIntensity=" in raw
    assert "nrintensity=" not in raw


def test_runtime_files_are_recovered_from_a_zip(tmp_path, monkeypatch):
    """Streamline ships as a zip and people drop the whole archive in.

    The first outside tester did exactly that and was told
    "nvngx_dlss.dll was not found" while looking at a streamline.zip
    containing it.
    """
    import zipfile

    from dlss5_converter import paths

    folder = tmp_path / "dlss_files"
    folder.mkdir()
    monkeypatch.setattr(paths, "runtime_search_roots", lambda: [folder])

    with zipfile.ZipFile(folder / "streamline.zip", "w") as bundle:
        # Nested, as the real archive is - not at the root of the zip.
        bundle.writestr("Production/nvngx_dlss.dll", b"x" * 2048)
        bundle.writestr("Production/sl.interposer.dll", b"y" * 512)

    recovered = runtime.unpack_archives()

    assert (folder / "nvngx_dlss.dll").is_file()
    assert any("nvngx_dlss.dll" in entry for entry in recovered)
    # Only what we asked for: the rest of Streamline is not ours to scatter.
    assert not (folder / "sl.interposer.dll").exists()


def test_unpacking_never_overwrites_a_file_already_there(tmp_path, monkeypatch):
    """A deliberately placed build must win over an old copy inside a zip."""
    import zipfile

    from dlss5_converter import paths

    folder = tmp_path / "dlss_files"
    folder.mkdir()
    monkeypatch.setattr(paths, "runtime_search_roots", lambda: [folder])

    (folder / "nvngx_dlss.dll").write_bytes(b"the one the user chose")
    with zipfile.ZipFile(folder / "streamline.zip", "w") as bundle:
        bundle.writestr("Production/nvngx_dlss.dll", b"an older build")

    runtime.unpack_archives()
    assert (folder / "nvngx_dlss.dll").read_bytes() == b"the one the user chose"


def test_a_corrupt_archive_is_not_fatal(tmp_path, monkeypatch):
    from dlss5_converter import paths

    folder = tmp_path / "dlss_files"
    folder.mkdir()
    monkeypatch.setattr(paths, "runtime_search_roots", lambda: [folder])
    (folder / "broken.zip").write_bytes(b"not really a zip")

    assert runtime.unpack_archives() == []
