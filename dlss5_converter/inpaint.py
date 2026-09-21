"""LaMa background inpainting — the optional "rebuild the background" upgrade.

The classical filler (in layers.py) blurs the hidden backplate. LaMa continues
real structure (walls, shelves, sky) instead, so a slice parallaxes over a
background that looks reconstructed rather than smeared. One forward pass, fast
enough to run once per image while the user waits.

Optional and downloaded on demand: `layers.build_layered_scene` takes any
`(image, mask) -> image` callable and falls back to the classical filler when
this model is absent. Ported from Depth Animator; the only changes are the
download (Hugging Face into the app's model cache) and the ONNX providers
(DirectML, no PyTorch). Model: Carve/LaMa-ONNX, Apache-2.0.
"""

from __future__ import annotations

import threading

import numpy as np

REPO = "Carve/LaMa-ONNX"
FILENAME = "lama_fp32.onnx"           # ~207 MB; the repo has no smaller export
TILE = 512                            # the exported graph is fixed at 512x512
_SHARED = None
_LOCK = threading.RLock()


def _cache_dir() -> str:
    from . import paths
    d = paths.model_cache_dir() / "lama"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def is_downloaded() -> bool:
    from huggingface_hub import try_to_load_from_cache
    try:
        return isinstance(
            try_to_load_from_cache(REPO, FILENAME, cache_dir=_cache_dir()), str)
    except Exception:  # noqa: BLE001
        return False


def download(progress=None, bytes_progress=None) -> str:
    """Fetch the weights into the model cache (resumable)."""
    from .depth_engine import enable_system_trust_store
    enable_system_trust_store()
    from huggingface_hub import hf_hub_download
    if progress:
        progress("Downloading LaMa background model…")
    return hf_hub_download(REPO, FILENAME, cache_dir=_cache_dir())


def _bounding_boxes(mask: np.ndarray, pad: int = 24) -> list[tuple[int, int, int, int]]:
    """Native-scale context windows covering each masked region."""
    import cv2
    count, _labels, stats, _c = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    height, width = mask.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < 16:
            continue
        x0, y0 = max(0, int(x) - pad), max(0, int(y) - pad)
        x1 = min(width, int(x + w) + pad)
        y1 = min(height, int(y + h) + pad)

        def axis_starts(lo: int, hi: int, length: int) -> list[int]:
            window = min(TILE, length)
            if hi - lo <= window:
                return [max(0, min(length - window, (lo + hi - window) // 2))]
            stride = max(64, window - pad * 2)
            values = list(range(lo, max(lo + 1, hi - window + 1), stride))
            last = max(0, min(length - window, hi - window))
            if not values or values[-1] != last:
                values.append(last)
            return values

        window_w, window_h = min(TILE, width), min(TILE, height)
        for top in axis_starts(y0, y1, height):
            for left in axis_starts(x0, x1, width):
                right, bottom = left + window_w, top + window_h
                if mask[top:bottom, left:right].any():
                    box = (left, top, right, bottom)
                    if box not in seen:
                        seen.add(box)
                        boxes.append(box)
    return boxes


class LamaInpainter:
    """Callable matching the classical filler's (image, holes) -> image."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self._session = None

    def _ensure_session(self):
        if self._session is None:
            with _LOCK:
                if self._session is None:
                    self._create_session()
        return self._session

    def _create_session(self) -> None:
        import onnxruntime as ort
        from .onnx_depth import _providers
        path = self.model_path or download()
        so = ort.SessionOptions()
        so.log_severity_level = 3
        # DirectML mis-runs LaMa's spectral (FFC) ops ("MatMul parameter is
        # incorrect"), so drop DML and use CUDA (if the user installed the GPU
        # runtime) or CPU. It runs once per image, so CPU's ~2 s/tile is fine.
        provs = [p for p in _providers() if p != "DmlExecutionProvider"]
        self._session = ort.InferenceSession(
            path, sess_options=so, providers=provs or ["CPUExecutionProvider"])
        self.model_path = path

    def __call__(self, image_rgb: np.ndarray, holes: np.ndarray) -> np.ndarray:
        import cv2
        result = np.ascontiguousarray(image_rgb).copy()
        if not holes.any():
            return result
        session = self._ensure_session()
        accumulated = np.zeros_like(result, dtype=np.float32)
        weights = np.zeros(holes.shape, dtype=np.float32)
        for left, top, right, bottom in _bounding_boxes(holes):
            crop = result[top:bottom, left:right]
            crop_mask = holes[top:bottom, left:right]
            if crop.size == 0 or not crop_mask.any():
                continue
            source = cv2.resize(crop, (TILE, TILE), interpolation=cv2.INTER_AREA)
            mask_small = cv2.resize(crop_mask.astype(np.uint8) * 255, (TILE, TILE),
                                    interpolation=cv2.INTER_NEAREST)
            image_in = source.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
            mask_in = (mask_small > 127).astype(np.float32)[None, None]
            output = session.run(None, {"image": image_in, "mask": mask_in})[0][0]
            painted = output.transpose(1, 2, 0)
            if painted.max() <= 1.5:                    # exports vary 0-1 vs 0-255
                painted = painted * 255.0
            painted = np.clip(painted, 0, 255).astype(np.uint8)
            restored = cv2.resize(painted, (crop.shape[1], crop.shape[0]),
                                  interpolation=cv2.INTER_CUBIC)
            ch, cw = crop_mask.shape
            edge_y = np.minimum(np.arange(ch) + 1, np.arange(ch, 0, -1))
            edge_x = np.minimum(np.arange(cw) + 1, np.arange(cw, 0, -1))
            feather = np.minimum(edge_y[:, None], edge_x[None, :]).astype(np.float32)
            feather = np.clip(feather / 32.0, 0.05, 1.0) * crop_mask
            accumulated[top:bottom, left:right] += restored.astype(np.float32) * feather[..., None]
            weights[top:bottom, left:right] += feather
        painted_mask = holes & (weights > 1e-6)
        result[painted_mask] = np.clip(
            accumulated[painted_mask] / weights[painted_mask, None], 0, 255).astype(np.uint8)
        return result


def build_inpainter(prefer_ai: bool):
    """Shared LaMa inpainter when requested and downloaded, else None (classical)."""
    global _SHARED
    if not prefer_ai or not is_downloaded():
        return None
    with _LOCK:
        if _SHARED is None:
            _SHARED = LamaInpainter()
        return _SHARED
