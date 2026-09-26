"""A GPU that disappears mid-run must not cost the user their conversion.

Reported from an RTX 5080: DirectML suspended the device during depth
estimation (DXGI 0x887A0005) and the run failed outright. The engine now
rebuilds the session on the CPU and finishes.
"""

from __future__ import annotations

import numpy as np
import pytest

from dlss5_converter import onnx_depth


SUSPENDED = (
    "[ONNXRuntimeError] : 1 : FAIL : DmlCommandRecorder.cpp(371) "
    "Exception(4) tid(6da0) 887A0005 The GPU device instance has been "
    "suspended. Use GetDeviceRemovedReason to determine the appropriate action."
)


def test_suspended_device_is_recognised():
    assert onnx_depth._is_device_lost(RuntimeError(SUSPENDED))


def test_device_removed_is_recognised():
    assert onnx_depth._is_device_lost(RuntimeError("DXGI_ERROR_DEVICE_REMOVED"))


def test_ordinary_failure_is_not_device_loss():
    assert not onnx_depth._is_device_lost(RuntimeError("invalid input shape 3x3"))


class _Session:
    """Stands in for an onnxruntime session: fails once, then works."""

    def __init__(self, error: Exception | None):
        self.error = error
        self.runs = 0

    def get_inputs(self):
        return [type("Input", (), {"name": "pixel_values"})()]

    def run(self, _outputs, _feed):
        self.runs += 1
        if self.error is not None:
            raise self.error
        return [np.zeros((1, onnx_depth.INPUT, onnx_depth.INPUT), np.float32)]


def _engine(monkeypatch, error):
    engine = onnx_depth.OnnxDepthEngine()
    engine.session = _Session(error)
    engine.device = "directml"
    engine._path = "model.onnx"
    rebuilt = _Session(None)

    class _Ort:
        GraphOptimizationLevel = type("G", (), {"ORT_ENABLE_ALL": 99})

        @staticmethod
        def SessionOptions():
            return type("O", (), {"graph_optimization_level": 0})()

        @staticmethod
        def InferenceSession(path, options, providers):
            assert providers == ["CPUExecutionProvider"], providers
            return rebuilt

    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", _Ort)
    return engine, rebuilt


def test_lost_device_falls_back_to_cpu(monkeypatch):
    engine, rebuilt = _engine(monkeypatch, RuntimeError(SUSPENDED))
    notes: list[str] = []
    values = np.zeros((1, 3, onnx_depth.INPUT, onnx_depth.INPUT), np.float32)

    out = engine._run(values, notes.append)

    assert out.shape[-1] == onnx_depth.INPUT
    assert engine.device == "cpu"
    assert rebuilt.runs == 1
    assert any("CPU" in note for note in notes)


def test_other_errors_still_raise(monkeypatch):
    engine, rebuilt = _engine(monkeypatch, RuntimeError("bad tensor rank"))
    with pytest.raises(RuntimeError, match="bad tensor rank"):
        engine._run(np.zeros((1, 3, 4, 4), np.float32))
    assert rebuilt.runs == 0


def test_cpu_session_does_not_loop(monkeypatch):
    """Already on the CPU, a device-shaped error is a real error."""
    engine, _ = _engine(monkeypatch, RuntimeError(SUSPENDED))
    engine.device = "cpu"
    with pytest.raises(RuntimeError, match="suspended"):
        engine._run(np.zeros((1, 3, 4, 4), np.float32))
