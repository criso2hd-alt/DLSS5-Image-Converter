"""NVIDIA drivers 616.64 and newer block the DLSS 5 neural pass."""

from __future__ import annotations

import pytest

from dlss5_converter import evaluator, hardware
from dlss5_converter.settings import AppSettings


@pytest.mark.parametrize("version, blocked", [
    ("616.56", False), ("591.44", False), ("576.02", False),
    ("616.64", True), ("616.92", True), ("620.01", True),
    ("", False), (None, False), ("n/a", False),
])
def test_blocked_from_616_64(version, blocked):
    assert hardware.driver_blocks_neural(version) is blocked
    assert (hardware.driver_warning(version) is not None) is blocked


def test_warning_names_the_driver_and_the_one_to_roll_back_to():
    text = hardware.driver_warning("616.92")
    assert "616.92" in text and hardware.LAST_WORKING_DRIVER in text


def _probe(driver: str) -> str:
    return ("adapter: NVIDIA GeForce RTX 5070\ndlss_available: 1\nneeds_driver_update: 0\n"
            f"driver_version: {driver}\nneural_addon_loaded: 1\nreshade_proxy_loaded: 1\n"
            "dlssnr_module_loaded: 0\ntest_evaluation: ok\n")


def test_runtime_report_leads_with_the_blocked_driver():
    problems = evaluator.interpret_probe(_probe("616.92"))
    assert len(problems) == 1 and "616.92" in problems[0] and "roll" in problems[0].lower()


def test_a_working_driver_still_gets_the_usual_diagnosis():
    problems = evaluator.interpret_probe(_probe("591.44"))
    assert problems and "dlssnr_module_loaded" in problems[0]


def test_crash_advice_names_the_driver_block(monkeypatch):
    monkeypatch.setattr(hardware, "query_driver_version", lambda: "616.92")
    assert hardware.LAST_WORKING_DRIVER in evaluator.startup_crash_advice()


def test_dismissal_survives_a_restart(tmp_path):
    s = AppSettings()
    s.driver_warning_dismissed = "616.92"
    s.save(tmp_path / "s.json")
    assert AppSettings.load(tmp_path / "s.json").driver_warning_dismissed == "616.92"
