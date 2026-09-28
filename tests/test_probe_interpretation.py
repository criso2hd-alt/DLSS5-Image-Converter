"""interpret_probe: raw harness fields -> a plain, actionable cause (issue #6)."""

from __future__ import annotations

from dlss5_converter.evaluator import interpret_probe

# The report from issue #6 (RTX 5080, everything 0, feature init failed), on
# a driver before NVIDIA's block so these tests exercise the load-order
# diagnosis. (The real report was on 616.64, which the driver check now names
# as the cause; see test_issue_6_as_reported_is_the_driver.)
ISSUE_6 = """adapter: NVIDIA GeForce RTX 5080
dlss_available: 0
needs_driver_update: 0
driver_version: 591.44
neural_addon_loaded: 0
reshade_proxy_loaded: 0
dlssnr_module_loaded: 0
test_evaluation: feature creation failed: NVSDK_NGX_Result_FAIL_UnableToInitializeFeature"""

HEALTHY = """adapter: NVIDIA GeForce RTX 4080
dlss_available: 1
needs_driver_update: 0
driver_version: 560.94
neural_addon_loaded: 1
reshade_proxy_loaded: 1
dlssnr_module_loaded: 1
test_evaluation: ok"""


def test_healthy_probe_reports_nothing():
    assert interpret_probe(HEALTHY) == []


def test_issue_6_blames_reshade_first():
    """reshade_proxy_loaded: 0 is the base of the chain, so it is named first."""
    problems = interpret_probe(ISSUE_6)
    assert problems, "a failed probe must produce at least one problem"
    assert "reshade_proxy_loaded: 0" in problems[0]
    assert "dxgi.dll" in problems[0]
    # The all-0 report should NOT drown the user in the raw NGX code alone.
    assert not any("NVSDK_NGX_Result" in p for p in problems)


def test_addon_is_blamed_only_once_reshade_is_up():
    report = ISSUE_6.replace("reshade_proxy_loaded: 0", "reshade_proxy_loaded: 1")
    problems = interpret_probe(report)
    assert "neural_addon_loaded: 0" in problems[0]


def test_neural_module_is_blamed_once_reshade_and_addon_are_up():
    report = (
        ISSUE_6.replace("reshade_proxy_loaded: 0", "reshade_proxy_loaded: 1")
        .replace("neural_addon_loaded: 0", "neural_addon_loaded: 1")
        .replace("dlss_available: 0", "dlss_available: 1")
    )
    problems = interpret_probe(report)
    assert "dlssnr_module_loaded: 0" in problems[0]


def test_an_old_driver_is_called_out():
    report = HEALTHY.replace(
        "needs_driver_update: 0", "needs_driver_update: 1"
    ).replace("dlssnr_module_loaded: 1", "dlssnr_module_loaded: 0").replace(
        "test_evaluation: ok", "test_evaluation: failed"
    )
    problems = interpret_probe(report)
    assert any("driver" in p.lower() for p in problems)


def test_a_hung_or_dying_probe_is_a_startup_crash(monkeypatch):
    from dlss5_converter import evaluator, hardware
    monkeypatch.setattr(hardware, "query_driver_version", lambda: None)
    hung = evaluator.interpret_probe("The harness did not respond within 45 seconds.")
    assert len(hung) == 1 and "while starting" in hung[0]
    # Named the adapter, then died before the live test line.
    died = evaluator.interpret_probe("adapter: NVIDIA GeForce RTX 4070\ndlss_available: 1\n")
    assert len(died) == 1 and "driver" in died[0]
    # Crashed before printing anything at all.
    silent = evaluator.interpret_probe("No output.")
    assert len(silent) == 1 and "while starting" in silent[0]


def test_issue_6_as_reported_is_the_driver():
    """On 616.64 itself, NVIDIA's driver block is the cause worth naming."""
    problems = interpret_probe(ISSUE_6.replace("591.44", "616.64"))
    assert len(problems) == 1 and "616.64" in problems[0]
