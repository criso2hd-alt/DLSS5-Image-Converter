"""The ONNX Runtime depth backend - a torch-free drop-in for DepthEngine."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from dlss5_converter import onnx_depth  # noqa: E402
from dlss5_converter.depth_engine import MODELS  # noqa: E402
from dlss5_converter.onnx_depth import INPUT, OnnxDepthEngine  # noqa: E402


def test_onnx_files_cover_every_model():
    """Same model ids as the torch engine, so one setting selects either."""
    assert set(onnx_depth.ONNX_FILES) == set(MODELS.values())


def test_providers_always_offer_cpu_last():
    provs = onnx_depth._providers()
    assert provs, "at least one provider must be returned"
    assert provs[-1] == "CPUExecutionProvider"


def test_is_downloaded_is_false_for_unknown_model():
    assert OnnxDepthEngine.is_downloaded("not/a-real-model") is False


def test_preprocess_produces_normalised_square_nchw():
    eng = OnnxDepthEngine()
    img = (np.random.default_rng(0).random((240, 320, 3)) * 255).astype(np.uint8)
    x = eng._preprocess(img)
    assert x.shape == (1, 3, INPUT, INPUT)
    assert x.dtype == np.float32
    # ImageNet-normalised: centred near zero, not raw 0..1.
    assert -3.0 < float(x.mean()) < 3.0
    assert float(x.min()) < -0.5 and float(x.max()) > 0.5


def test_infer_matches_the_depth_contract_when_a_model_is_present():
    """If the Small ONNX is exported, a real run yields normalised inverse depth
    at the source resolution. Skipped where the model has not been exported."""
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    if not OnnxDepthEngine.is_downloaded(model_id):
        pytest.skip("Small ONNX not exported on this machine")
    eng = OnnxDepthEngine()
    eng.load(model_id)
    img = (np.random.default_rng(1).random((216, 384, 3)) * 255).astype(np.uint8)
    depth = eng.infer(img)
    assert depth.shape == (216, 384)          # back at source resolution
    assert depth.dtype == np.float32
    assert 0.0 <= float(depth.min()) and float(depth.max()) <= 1.0
