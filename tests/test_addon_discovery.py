"""The RenoDX add-on is matched by its .addon64 suffix, not a fixed name.

RenoDX renames the file between releases (renodx-dlss5.addon64, dlss.addon64,
…). These lock in that a rename does not silently break a working setup.
"""

from __future__ import annotations

import pytest

from dlss5_converter import discovery, paths


@pytest.fixture
def only_here(monkeypatch):
    """Confine find_addon to the folder passed as `extra`, so a real dlss_files
    on the dev machine cannot leak into the result."""
    monkeypatch.setattr(paths, "deep_search_roots", lambda: [])


def test_find_addon_matches_a_renamed_addon(tmp_path, only_here):
    (tmp_path / "dlss.addon64").write_bytes(b"addon")
    found = paths.find_addon(tmp_path)
    assert found is not None and found.name == "dlss.addon64"


def test_find_addon_prefers_the_known_name(tmp_path, only_here):
    (tmp_path / "dlss.addon64").write_bytes(b"a")
    (tmp_path / "renodx-dlss5.addon64").write_bytes(b"a")
    assert paths.find_addon(tmp_path).name == "renodx-dlss5.addon64"


def test_find_addon_prefers_a_dlss_named_over_an_unrelated_addon(tmp_path, only_here):
    (tmp_path / "some-other.addon64").write_bytes(b"a")
    (tmp_path / "renodx-dlss.addon64").write_bytes(b"a")
    assert paths.find_addon(tmp_path).name == "renodx-dlss.addon64"


def test_find_addon_searches_nested_folders(tmp_path, only_here):
    nested = tmp_path / "pack" / "reshade-addons"
    nested.mkdir(parents=True)
    (nested / "dlss.addon64").write_bytes(b"a")
    assert paths.find_addon(tmp_path).name == "dlss.addon64"


def test_find_addon_returns_none_when_absent(tmp_path, only_here):
    (tmp_path / "readme.txt").write_text("no addon here", encoding="utf-8")
    assert paths.find_addon(tmp_path) is None


def test_addon_rank_orders_known_then_dlss_then_renodx_then_other():
    assert paths._addon_rank("renodx-dlss5.addon64") == 0
    assert paths._addon_rank("dlss.addon64") == 1
    assert paths._addon_rank("renodx-tonemap.addon64") == 2
    assert paths._addon_rank("crt.addon64") == 3


def test_scanner_maps_a_renamed_addon_to_the_canonical_name(tmp_path):
    # accepts() does not size-check the add-on, so a small stand-in is enough.
    (tmp_path / "my-dlss-build.addon64").write_bytes(b"addon")
    candidates = discovery.scan([tmp_path])
    assert candidates, "the folder with an .addon64 should be a candidate"
    files = candidates[0].files
    assert discovery.ADDON in files
    assert files[discovery.ADDON].name == "my-dlss-build.addon64"
