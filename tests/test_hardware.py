"""Driver-level hardware queries stay optional and parseable."""

from __future__ import annotations

from types import SimpleNamespace

from dlss5_converter import hardware


def test_vram_query_parses_nvidia_smi_mebibytes(monkeypatch):
    monkeypatch.setattr(
        hardware.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="NVIDIA GeForce RTX 4080, 16376, 4096, 12280\n",
        ),
    )
    info = hardware.query_nvidia_vram()
    assert info is not None
    assert info.name == "NVIDIA GeForce RTX 4080"
    assert info.total_bytes == 16376 * 1024**2
    assert info.free_bytes == 12280 * 1024**2


def test_vram_query_failure_leaves_the_conversion_free_to_try(monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(hardware.subprocess, "run", unavailable)
    assert hardware.query_nvidia_vram() is None
