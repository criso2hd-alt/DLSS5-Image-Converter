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


def _rewrite_for_dml(folded_path: str, out_path: str) -> None:
    """Make the folded LaMa graph acceptable to DirectML.

    The export implements its FFT as MatMuls of a constant [K, K] DFT matrix
    against a rank-5 [1, C, H, K, 1] tensor. DirectML rejects that broadcast at
    session creation ("The parameter is incorrect"). A @ B with B's last axis of
    size 1 equals squeeze(B) @ A^T with the axis put back, which is a plain
    rank-4 by rank-2 MatMul every backend accepts. Nothing else changes."""
    import onnx
    from onnx import helper, numpy_helper, shape_inference

    model = shape_inference.infer_shapes(onnx.load(folded_path))
    g = model.graph
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in list(g.value_info) + list(g.input)}
    inits = {i.name: i for i in g.initializer}
    for i in g.initializer:
        shapes[i.name] = list(i.dims)
    g.initializer.append(numpy_helper.from_array(np.array([-1], np.int64), "dml_axis_last"))
    transposed: dict[str, str] = {}
    nodes = []
    for n in g.node:
        if n.op_type == "MatMul":
            a, b = n.input
            sb = shapes.get(b, [])
            if a in inits and len(shapes[a]) == 2 and len(sb) == 5 and sb[-1] == 1:
                if a not in transposed:
                    at = numpy_helper.from_array(
                        numpy_helper.to_array(inits[a]).T.copy(), a + "_T")
                    g.initializer.append(at)
                    transposed[a] = at.name
                nodes += [
                    helper.make_node("Squeeze", [b, "dml_axis_last"], [n.name + "_sq"]),
                    helper.make_node("MatMul", [n.name + "_sq", transposed[a]], [n.name + "_mm"]),
                    helper.make_node("Unsqueeze", [n.name + "_mm", "dml_axis_last"], [n.output[0]]),
                ]
                continue
        nodes.append(n)
    del g.node[:]
    g.node.extend(nodes)
    onnx.save(model, out_path)


def dml_model(path: str) -> str | None:
    """A DirectML-ready copy of the LaMa graph, built once and cached beside it.

    Two steps: ONNX Runtime constant-folds the graph at the fixed 512 tile (the
    export builds its DFT matrices at run time from Range/Cos/Sin, so no shape
    is known until folded), then `_rewrite_for_dml` fixes the MatMuls. The
    result is checked once against the CPU on a random tile; a driver that gets
    it wrong is recorded and never tried again. ~40 ms a tile on the GPU
    against ~2.4 s on the CPU."""
    import os
    import onnxruntime as ort

    base = os.path.join(os.path.dirname(path), "lama_dml")
    out, ok_flag, bad_flag = base + ".onnx", base + ".verified", base + ".failed"
    if os.path.exists(bad_flag):
        return None
    if os.path.exists(out) and os.path.exists(ok_flag):
        return out
    folded = base + "_folded.onnx"
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        so.add_free_dimension_override_by_name("batch", 1)
        so.optimized_model_filepath = folded
        cpu = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        _rewrite_for_dml(folded, out)

        rng = np.random.default_rng(0)
        mask = np.zeros((1, 1, TILE, TILE), np.float32)
        mask[..., 160:352, 160:352] = 1.0
        feed = {"image": rng.random((1, 3, TILE, TILE), np.float32), "mask": mask}
        ref = cpu.run(None, feed)[0]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.enable_mem_pattern = False
        got = ort.InferenceSession(out, sess_options=so,
                                   providers=["DmlExecutionProvider"]).run(None, feed)[0]
        scale = max(float(np.abs(ref).max()), 1e-6)
        if not np.isfinite(got).all() or float(np.abs(got - ref).max()) > 0.02 * scale:
            raise RuntimeError("DirectML LaMa output does not match the CPU")
        open(ok_flag, "w").close()
        return out
    except Exception:  # noqa: BLE001 - remember, so every launch does not retry
        try:
            open(bad_flag, "w").close()
        except OSError:
            pass
        return None
    finally:
        try:
            os.remove(folded)
        except OSError:
            pass


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
        self.provider = ""

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
        self.model_path = path
        provs = _providers()
        if "CUDAExecutionProvider" not in provs and "DmlExecutionProvider" in provs:
            try:
                dml_path = dml_model(path)
                if dml_path is not None:
                    so = ort.SessionOptions()
                    so.log_severity_level = 3
                    so.enable_mem_pattern = False     # DirectML requires it off
                    self._session = ort.InferenceSession(
                        dml_path, sess_options=so, providers=["DmlExecutionProvider"])
                    self.provider = "DirectML"
                    return
            except Exception:  # noqa: BLE001 - CPU below is the floor
                pass
        so = ort.SessionOptions()
        so.log_severity_level = 3
        # The stock export only runs on CUDA or CPU; DirectML needs the
        # rewritten graph from dml_model().
        provs = [p for p in provs if p != "DmlExecutionProvider"]
        self._session = ort.InferenceSession(
            path, sess_options=so, providers=provs or ["CPUExecutionProvider"])
        self.provider = "CUDA" if "CUDAExecutionProvider" in provs else "CPU"

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
