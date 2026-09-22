"""GPU selection for machines with more than one card (laptop + eGPU report)."""
from __future__ import annotations

import pytest

from dlss5_converter import gpus
from dlss5_converter.gpus import Gpu

LAPTOP = Gpu(0, "NVIDIA GeForce RTX 3070 Ti Laptop GPU", 0x10DE, 8 * 1024**3)
EGPU = Gpu(1, "NVIDIA GeForce RTX 5060", 0x10DE, 8 * 1024**3)


@pytest.fixture
def two_gpus(monkeypatch):
    monkeypatch.setattr(gpus, "_cache", [LAPTOP, EGPU])
    monkeypatch.setattr(gpus, "_preference", "")
    monkeypatch.setattr(gpus, "_harness_adapter", "")


def test_automatic_follows_the_card_dlss_runs_on(two_gpus):
    gpus.note_harness_adapter("dlss_available: 1\nadapter: NVIDIA GeForce RTX 5060\n")
    assert gpus.selected() == EGPU


def test_user_choice_wins_over_the_dlss_card(two_gpus):
    gpus.note_harness_adapter("adapter: NVIDIA GeForce RTX 5060")
    gpus.set_preference(LAPTOP.name)
    assert gpus.selected() == LAPTOP


def test_unknown_preference_falls_back_to_automatic(two_gpus):
    gpus.set_preference("A card that was unplugged")
    gpus.note_harness_adapter("adapter: NVIDIA GeForce RTX 5060")
    assert gpus.selected() == EGPU


def test_directml_is_pinned_to_the_selected_index(two_gpus):
    gpus.note_harness_adapter("adapter: NVIDIA GeForce RTX 5060")
    assert gpus.ort_providers(["DmlExecutionProvider", "CPUExecutionProvider"]) == [
        ("DmlExecutionProvider", {"device_id": "1"}), "CPUExecutionProvider"]


def test_single_gpu_providers_are_untouched(monkeypatch):
    monkeypatch.setattr(gpus, "_cache", [LAPTOP])
    assert gpus.ort_providers(["DmlExecutionProvider"]) == ["DmlExecutionProvider"]
    assert not gpus.multiple()


def test_vram_gauge_reads_the_selected_card(two_gpus):
    gpus.note_harness_adapter("adapter: NVIDIA GeForce RTX 5060")
    rows = ["NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8192, 7000",
            "NVIDIA GeForce RTX 5060, 8151, 1200"]
    assert gpus.pick_smi_line(rows) == rows[1]


def test_enumeration_runs_on_this_machine():
    # Real DXGI call: must never raise, and never list the software adapter.
    found = gpus._enumerate()
    assert all("Basic Render" not in g.name for g in found)
