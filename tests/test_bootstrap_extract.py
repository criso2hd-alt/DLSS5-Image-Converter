"""The first-run unpack: streamed big files stay byte-identical and report."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pytest

from dlss5_converter import bootstrap


def _make_zip(path: Path) -> dict[str, bytes]:
    """A zip with a directory, small files, and one member above the stream
    threshold, so both extraction paths are exercised."""
    contents = {
        "pkg/__init__.py": b"x = 1\n",
        "pkg/small.bin": os.urandom(1024),
        # Comfortably over _STREAM_ABOVE so it takes the chunked path. Random so
        # a truncation or offset bug cannot hide behind a repeating pattern.
        "pkg/lib/big.dll": os.urandom(bootstrap._STREAM_ABOVE + 5_000_003),
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("pkg/", b"")           # explicit directory entry
        z.writestr("pkg/lib/", b"")
        for name, data in contents.items():
            z.writestr(name, data)
    return contents


def test_extraction_is_byte_identical_across_the_size_threshold(tmp_path):
    archive = tmp_path / "wheel.zip"
    expected = _make_zip(archive)
    out = tmp_path / "out"

    bootstrap._extract(archive, out, on_bytes=None, on_text=None)

    for name, data in expected.items():
        got = (out / name).read_bytes()
        assert got == data, f"{name} differs after extraction"


def test_progress_reaches_the_total_and_names_the_big_file(tmp_path):
    archive = tmp_path / "wheel.zip"
    _make_zip(archive)
    out = tmp_path / "out"

    seen_bytes: list[tuple[int, int]] = []
    seen_text: list[str] = []
    bootstrap._extract(
        archive, out,
        lambda done, total: seen_bytes.append((done, total)),
        seen_text.append,
    )

    assert seen_bytes, "progress must be reported at least once"
    done, total = seen_bytes[-1]
    assert done == total  # the forced final update lands the bar exactly full
    # The big DLL is announced by name, which is what turns a stall into a
    # visible "working on this file" rather than a frozen app.
    assert any("big.dll" in t for t in seen_text)


def test_throttle_coalesces_but_always_lets_force_through():
    calls: list[tuple[int, int]] = []
    throttle = bootstrap._Throttle(lambda d, t: calls.append((d, t)), interval=10.0)
    throttle(1, 100)          # first call: fires (last was 0)
    throttle(2, 100)          # within the 10 s window: coalesced away
    throttle(3, 100)          # still within: coalesced
    assert calls == [(1, 100)]
    throttle(100, 100, force=True)  # force always fires — the final update
    assert calls == [(1, 100), (100, 100)]


def test_throttle_with_no_callback_is_a_noop():
    throttle = bootstrap._Throttle(None)
    throttle(1, 2)  # must not raise
    throttle(1, 2, force=True)


def test_download_translates_network_errors_into_vpn_guidance(tmp_path, monkeypatch):
    # A VPN/DNS/proxy stall used to look like a frozen app; now it must surface
    # as a plain, actionable message naming the usual causes.
    from urllib.error import URLError

    def boom(*a, **k):
        raise URLError("connection reset")

    monkeypatch.setattr(bootstrap, "urlopen", boom)
    with pytest.raises(OSError) as excinfo:
        bootstrap._download("https://example/torch.whl", tmp_path / "f.part", None)
    message = str(excinfo.value)
    assert "VPN" in message and "1.1.1.1" in message


def test_remote_size_returns_zero_instead_of_raising(monkeypatch):
    from urllib.error import URLError

    def boom(*a, **k):
        raise URLError("no route to host")

    monkeypatch.setattr(bootstrap, "urlopen", boom)
    assert bootstrap._remote_size("https://example/torch.whl") == 0


def test_extract_ignores_path_traversal_entries(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escape.txt", b"nope")
        z.writestr("pkg/ok.txt", b"fine")
    out = tmp_path / "out"

    bootstrap._extract(archive, out, on_bytes=None, on_text=None)

    assert (out / "pkg" / "ok.txt").read_bytes() == b"fine"
    # The traversal segment was dropped, so the file lands inside the target
    # (as "escape.txt"), never above it.
    assert not (tmp_path / "escape.txt").exists()


def test_safe_parts_strips_traversal_and_empty_segments():
    assert bootstrap._safe_parts("a/b/c") == ["a", "b", "c"]
    assert bootstrap._safe_parts("../../etc/passwd") == ["etc", "passwd"]
    assert bootstrap._safe_parts("a//./b/") == ["a", "b"]
