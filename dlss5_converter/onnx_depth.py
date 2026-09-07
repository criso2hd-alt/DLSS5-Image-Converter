"""Depth Anything V2 depth estimation through ONNX Runtime - no PyTorch.

A drop-in for :class:`depth_engine.DepthEngine` (same ``load`` / ``infer`` /
``is_downloaded`` surface) that runs the model exported by
``scripts/export_onnx.py``. This is what lets a release ship without the 2.7 GB
PyTorch download and without transformers: inference is ONNX Runtime, and
pre/post-processing is plain numpy replicating the DPT image processor.

Input is a fixed 518x518 square (see ``scripts/export_onnx.py`` for why the
export is not dynamic); the depth map is resized back to the source resolution
afterwards, exactly as the torch path does. The output contract is identical:
normalised inverse depth in [0, 1] where 1.0 is nearest, which is already the
reversed-Z layout DLSS expects.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from . import paths

#: Model id -> exported ONNX filename. Matches scripts/export_onnx.py output and
#: depth_engine.MODELS, so the same settings value selects either backend.
ONNX_FILES = {
    "depth-anything/Depth-Anything-V2-Small-hf": "Depth-Anything-V2-Small-hf.onnx",
    "depth-anything/Depth-Anything-V2-Base-hf": "Depth-Anything-V2-Base-hf.onnx",
    "depth-anything/Depth-Anything-V2-Large-hf": "Depth-Anything-V2-Large-hf.onnx",
}

#: The square edge the models are exported at (multiple of 14; DINOv2 patch size).
INPUT = 518

#: DPT/ImageNet normalisation, from the model's preprocessor_config.json.
_MEAN = np.asarray([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
_STD = np.asarray([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)


def onnx_models_dir() -> Path:
    """Where exported ONNX depth models live, beside the HF model cache."""
    return paths.model_cache_dir() / "onnx"


def _providers() -> list[str]:
    """GPU first, CPU as the guaranteed fallback.

    The shipped app installs onnxruntime-gpu or -directml; the plain onnxruntime
    package only offers CPU. Whatever is actually present is used, in that order
    of preference, so the same code runs on any of them.
    """
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    preferred = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    chosen = [p for p in preferred if p in available]
    return chosen or ["CPUExecutionProvider"]


class OnnxDepthEngine:
    """Depth estimation via ONNX Runtime, interchangeable with DepthEngine."""

    def __init__(self) -> None:
        self.session = None
        self.model_id: str | None = None
        self.device = "cpu"

    @classmethod
    def is_downloaded(cls, model_id: str) -> bool:
        name = ONNX_FILES.get(model_id)
        return bool(name) and (onnx_models_dir() / name).is_file()

    def load(
        self,
        model_id: str,
        progress: Callable[[str], None] | None = None,
        bytes_progress: Callable[[int, int], None] | None = None,
    ) -> str:
        """Open the ONNX session for `model_id`. Raises if the file is missing.

        The ONNX file is produced by scripts/export_onnx.py and placed in
        onnx_models_dir(); fetching it (from a bundle or a download) is the
        setup step's job, kept out of here so inference stays torch-free and
        offline.
        """
        if self.model_id == model_id and self.session is not None:
            return self.device

        name = ONNX_FILES.get(model_id)
        if name is None:
            raise RuntimeError(f"No ONNX export is known for {model_id}.")
        path = onnx_models_dir() / name
        if not path.is_file():
            raise RuntimeError(
                f"The ONNX depth model is not installed ({path.name}). Run "
                "scripts/export_onnx.py, or install the model bundle."
            )

        import onnxruntime as ort

        if progress:
            progress("Preparing the depth model…")
        providers = _providers()
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), options, providers=providers)
        active = self.session.get_providers()[0]
        self.device = "cuda" if active in (
            "TensorrtExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider",
        ) else "cpu"
        self.model_id = model_id
        return self.device

    def _preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        """Source RGB uint8 -> normalised NCHW float32 at INPUT x INPUT."""
        resized = cv2.resize(image_rgb, (INPUT, INPUT), interpolation=cv2.INTER_CUBIC)
        x = resized.astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)

    def infer(
        self,
        image_rgb: np.ndarray,
        progress: Callable[[str], None] | None = None,
        input_size: int = 518,
        tiled: bool = False,
    ) -> np.ndarray:
        """Normalised inverse depth in [0, 1]: 1.0 nearest, 0.0 furthest.

        ``input_size`` and ``tiled`` are accepted for interface parity with the
        torch engine; the ONNX export is fixed at 518, so they do not change the
        working resolution here.
        """
        if self.session is None:
            raise RuntimeError("Load a depth model before analysing an image.")
        if progress:
            progress("Estimating depth…")

        height, width = image_rgb.shape[:2]
        pixel_values = self._preprocess(image_rgb)
        name = self.session.get_inputs()[0].name
        raw = self.session.run(None, {name: pixel_values})[0]
        depth = np.asarray(raw, np.float32).reshape(INPUT, INPUT)

        # Back to the source resolution, then the same percentile normalisation
        # the torch engine uses - a blown highlight or hot pixel must not
        # compress the whole range.
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_CUBIC)
        lo, hi = np.percentile(depth, (1.0, 99.0))
        return np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)
