"""Three-way Style, the style-tagged filename, and the Boost size guard."""

from __future__ import annotations

from pathlib import Path

from dlss5_converter import pipeline
from dlss5_converter.settings import (
    D3D12_MAX_TEXTURE_DIMENSION,
    NR_STYLES,
    max_boost_factor,
    style_slug,
)


def test_style_list_is_the_addons_three():
    """Default, Natural, Cinematic - Default is index 0, the add-on's own start."""
    assert NR_STYLES == ("Default", "Natural", "Cinematic")


def test_style_slug_names_each_style_and_clamps():
    assert style_slug(0) == "default"
    assert style_slug(1) == "natural"
    assert style_slug(2) == "cinematic"
    # A stored index past the end still yields a name rather than raising.
    assert style_slug(99) == "cinematic"
    assert style_slug(-3) == "default"


def test_output_name_carries_the_style(tmp_path):
    out = pipeline.hdr_output_path(tmp_path, "shot", tmp_path / "shot.png", "cinematic")
    assert out.name == "shot_dlss5_cinematic.png"
    # HDR input keeps the format that can hold it, still tagged.
    hdr = pipeline.hdr_output_path(tmp_path, "shot", tmp_path / "shot.jxr", "natural")
    assert hdr.name == "shot_dlss5_natural.jxr"


def test_output_name_without_a_style_is_unchanged(tmp_path):
    """The default-None call still produces the old name, so nothing else breaks."""
    out = pipeline.hdr_output_path(tmp_path, "shot", tmp_path / "shot.png")
    assert out.name == "shot_dlss5.png"


def test_max_boost_factor_tracks_the_texture_limit():
    # 3840 (the 4K default): 8x would be 30720, past the limit, so 4x is the top.
    assert max_boost_factor(3840) == 4
    # 2048: even 8x (16384) exactly fits.
    assert max_boost_factor(2048) == 8
    # 8192: only 2x (16384) fits - jerkalerk's "8192 is the actual limit".
    assert max_boost_factor(8192) == 2
    # Above 8192 nothing fits: Boost cannot supersample at all.
    assert max_boost_factor(12000) == 0
    # The knee is exactly the texture limit over the smallest factor.
    assert max_boost_factor(D3D12_MAX_TEXTURE_DIMENSION // 2) == 2
    assert max_boost_factor(D3D12_MAX_TEXTURE_DIMENSION // 2 + 1) == 0
