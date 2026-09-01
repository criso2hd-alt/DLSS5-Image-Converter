"""Local Depth Anything V2 inference.

Lifted from Depth Animator and trimmed to what the contract builder needs. The
download/caching behaviour is deliberately identical — both apps share a Hugging
Face cache layout, so a user who has already pulled a model for one does not pay
for it twice.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

MODELS = {
    "Small · Fast · Apache 2.0": "depth-anything/Depth-Anything-V2-Small-hf",
    "Base · Balanced · Non-commercial": "depth-anything/Depth-Anything-V2-Base-hf",
    "Large · Detailed · Non-commercial": "depth-anything/Depth-Anything-V2-Large-hf",
}

#: Base is the default here rather than Small. The neural pass reacts to depth
#: *edges* — a soft silhouette from the Small model shows up as haloing around
#: heads and shoulders, which is exactly where people look first.
DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Base-hf"


def configure_model_cache() -> Path:
    """Point the Hugging Face cache at storage that survives an app update.

    Runs at import time because huggingface_hub reads HF_HOME into module
    constants the first time it is imported; setting it later would silently
    leave gigabytes in the user profile instead of where we intend.
    """
    from .paths import model_cache_dir

    os.environ.setdefault("HF_HOME", str(model_cache_dir()))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return Path(os.environ["HF_HOME"])


MODEL_CACHE = configure_model_cache()

_TRUST_STORE_READY = False


def enable_system_trust_store() -> None:
    """Verify TLS against the OS store instead of certifi's bundle.

    Antivirus and corporate proxies re-sign HTTPS with a private root that is in
    the Windows trust store but absent from certifi, so the model download fails
    with CERTIFICATE_VERIFY_FAILED. Certificates are still fully verified.
    """
    global _TRUST_STORE_READY
    if _TRUST_STORE_READY:
        return
    try:
        import truststore

        truststore.inject_into_ssl()
        _TRUST_STORE_READY = True
    except Exception:  # noqa: BLE001 - fall back to certifi rather than block startup
        pass


def select_device(torch) -> str:
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001 - a broken driver must not block CPU use
        pass
    return "cpu"


def device_label(torch, device: str) -> str:
    try:
        if device == "cuda":
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - the badge must never break loading
        pass
    return device.upper()


def hub_cache_dir() -> str:
    """The shared hub folder as an explicit path rather than an env var.

    huggingface_hub freezes HF_HOME on first import and ignores later changes,
    so passing ``cache_dir=`` at every call site is the only import-order-proof
    option. Relying on the env var alone means the app looks in one place while
    the download lands in another, and the weights appear to vanish.
    """
    return str(MODEL_CACHE / "hub")


def _repo_folder(model_id: str) -> Path:
    return MODEL_CACHE / "hub" / ("models--" + model_id.replace("/", "--"))


def _folder_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


class _CacheGrowthReporter:
    """Report download progress by watching the cache folder grow.

    huggingface_hub does not hand ``tqdm_class`` to individual file downloads,
    so it cannot report bytes. The folder's *absolute* size is the progress, not
    its growth since this run started: ``snapshot_download`` resumes from
    ``.incomplete`` files, and measuring the delta would show a resumed 1.5 GB
    download finishing at a quarter of the bar.
    """

    def __init__(self, folder: Path, total: int, callback: Callable[[int, int], None]) -> None:
        self._folder = folder
        self._total = total
        self._callback = callback
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.4):
            self._callback(min(_folder_bytes(self._folder), self._total), self._total)

    def __enter__(self) -> _CacheGrowthReporter:
        if self._total > 0:
            self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.5)


class DepthEngine:
    """Lazy, local-first wrapper around Hugging Face Depth Anything V2."""

    # Without the filter the hub also pulls a duplicate pytorch_model.bin
    # alongside the safetensors weights.
    ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt"]

    def __init__(self) -> None:
        self.processor = None
        self.model = None
        self.model_id: str | None = None
        self.device = "cpu"

    @classmethod
    def is_downloaded(cls, model_id: str) -> bool:
        """Whether this model is already on disk, without touching the network.

        Checked before showing a download dialog, so a normal launch does not
        stall on an HTTP round trip just to discover there is nothing to do.
        A snapshot counts only if it has both a config and weights — an
        interrupted download leaves the folder present but useless, and
        treating that as "downloaded" would fail later at load time instead of
        resuming here.
        """
        snapshots = _repo_folder(model_id) / "snapshots"
        if not snapshots.is_dir():
            return False
        for revision in snapshots.iterdir():
            if not revision.is_dir():
                continue
            files = list(revision.rglob("*"))
            has_config = any(f.name == "config.json" for f in files)
            has_weights = any(f.suffix == ".safetensors" for f in files)
            if has_config and has_weights:
                return True
        return False

    def remote_size(self, model_id: str) -> int:
        """Total download size in bytes, or 0 if it cannot be determined."""
        try:
            return self._remote_size(model_id)
        except Exception:  # noqa: BLE001 - offline or metadata failure is fine
            return 0

    def _remote_size(self, model_id: str) -> int:
        enable_system_trust_store()
        from fnmatch import fnmatch

        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id, files_metadata=True)
        total = 0
        for sibling in info.siblings or []:
            if any(fnmatch(sibling.rfilename, pattern) for pattern in self.ALLOW_PATTERNS):
                total += getattr(sibling, "size", None) or 0
        return total

    def ensure_downloaded(
        self,
        model_id: str,
        progress: Callable[[str], None] | None = None,
        bytes_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        enable_system_trust_store()
        from huggingface_hub import snapshot_download

        total = 0
        if bytes_progress is not None:
            if progress:
                progress("Checking the local model cache…")
            try:
                total = self._remote_size(model_id)
            except Exception:  # noqa: BLE001 - offline or metadata failure is fine
                total = 0

        def download() -> None:
            snapshot_download(
                model_id, allow_patterns=self.ALLOW_PATTERNS, cache_dir=hub_cache_dir()
            )

        if bytes_progress is not None and total > 0:
            with _CacheGrowthReporter(_repo_folder(model_id), total, bytes_progress):
                download()
            bytes_progress(total, total)
        else:
            download()

    def load(
        self,
        model_id: str,
        progress: Callable[[str], None] | None = None,
        bytes_progress: Callable[[int, int], None] | None = None,
    ) -> str:
        enable_system_trust_store()

        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        if self.model_id == model_id and self.model is not None:
            return self.device
        self.device = select_device(torch)

        try:
            self.ensure_downloaded(model_id, progress, bytes_progress)
        except Exception:  # noqa: BLE001 - from_pretrained raises a clearer error
            # and can still succeed from a warm cache.
            pass

        if progress:
            progress("Preparing the depth model…")
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(model_id, dtype=dtype)
            .to(self.device)
            .eval()
        )
        self.model_id = model_id
        return self.device

    def infer(
        self,
        image_rgb: np.ndarray,
        progress: Callable[[str], None] | None = None,
        input_size: int = 518,
        tiled: bool = False,
    ) -> np.ndarray:
        """Normalised inverse depth in [0, 1]: 1.0 is nearest, 0.0 is furthest.

        That orientation is not incidental — it is already the reversed-Z layout
        DLSS expects, so ``contract.py`` can pass this through with almost no
        remapping. See CLAUDE.md.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Load a depth model before analysing an image.")

        import torch

        if progress:
            progress("Estimating depth…")
        input_size = max(280, int(input_size))
        if tiled and max(image_rgb.shape[:2]) > input_size:
            depth = self._infer_tiled(image_rgb, input_size, progress, torch)
        else:
            depth = self._predict_depth(image_rgb, input_size, torch)

        # Percentile rather than min/max: a single blown-out specular highlight
        # or a sensor hot pixel would otherwise compress the whole range.
        lo, hi = np.percentile(depth, (1.0, 99.0))
        return np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)

    def _predict_depth(self, image_rgb: np.ndarray, input_size: int, torch) -> np.ndarray:
        pil_image = Image.fromarray(image_rgb)
        inputs = self.processor(
            images=pil_image,
            return_tensors="pt",
            size={"height": input_size, "width": input_size},
        )
        model_dtype = next(self.model.parameters()).dtype
        inputs = {
            key: value.to(device=self.device, dtype=model_dtype)
            if value.is_floating_point()
            else value.to(self.device)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            output = self.model(**inputs)
        processed = self.processor.post_process_depth_estimation(
            output, target_sizes=[(image_rgb.shape[0], image_rgb.shape[1])]
        )[0]["predicted_depth"]
        return processed.detach().float().cpu().numpy().astype(np.float32)

    def _infer_tiled(self, image_rgb, tile_size: int, progress, torch) -> np.ndarray:
        """Blend detailed tiles after aligning each to a global depth pass.

        Depth Anything is scale-and-shift invariant per input, so tiles disagree
        about absolute depth even where they overlap. Each tile is least-squares
        fitted to the global pass before blending, or the seams read as ledges.
        """
        height, width = image_rgb.shape[:2]
        if progress:
            progress("Estimating global depth structure…")
        anchor = self._predict_depth(image_rgb, 518, torch)
        overlap = max(64, tile_size // 6)
        step = max(1, tile_size - overlap)

        def starts(length: int) -> list[int]:
            if length <= tile_size:
                return [0]
            values = list(range(0, length - tile_size + 1, step))
            last = length - tile_size
            if values[-1] != last:
                values.append(last)
            return values

        positions = [(y, x) for y in starts(height) for x in starts(width)]
        accumulated = np.zeros((height, width), np.float32)
        weights = np.zeros((height, width), np.float32)
        for index, (y, x) in enumerate(positions, 1):
            y1, x1 = min(height, y + tile_size), min(width, x + tile_size)
            tile = image_rgb[y:y1, x:x1]
            if progress:
                progress(f"Refining depth tile {index} of {len(positions)}…")
            detailed = self._predict_depth(tile, tile_size, torch)
            reference = anchor[y:y1, x:x1]

            sample_x = detailed[::4, ::4].reshape(-1).astype(np.float64)
            sample_y = reference[::4, ::4].reshape(-1).astype(np.float64)
            finite = np.isfinite(sample_x) & np.isfinite(sample_y)
            sample_x, sample_y = sample_x[finite], sample_y[finite]
            if sample_x.size >= 16 and float(np.std(sample_x)) > 1e-8:
                matrix = np.column_stack((sample_x, np.ones_like(sample_x)))
                scale, shift = np.linalg.lstsq(matrix, sample_y, rcond=None)[0]
                detailed = detailed * float(scale) + float(shift)

            tile_h, tile_w = detailed.shape
            edge_y = np.minimum(np.arange(tile_h) + 1, np.arange(tile_h, 0, -1))
            edge_x = np.minimum(np.arange(tile_w) + 1, np.arange(tile_w, 0, -1))
            feather = np.minimum(edge_y[:, None], edge_x[None, :]).astype(np.float32)
            feather = np.clip(feather / max(overlap * 0.5, 1.0), 0.02, 1.0)
            accumulated[y:y1, x:x1] += detailed * feather
            weights[y:y1, x:x1] += feather
        return accumulated / np.maximum(weights, 1e-6)
