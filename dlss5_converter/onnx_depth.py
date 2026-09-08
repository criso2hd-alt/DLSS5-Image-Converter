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

import os
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

#: The always-available bundled model, used as a fallback when a selected
#: (non-bundled) model has not been exported or downloaded.
SMALL = "depth-anything/Depth-Anything-V2-Small-hf"

#: Where the larger depth models are fetched from. Base and Large are CC-BY-NC
#: and large (fp16: ~190 MB and ~670 MB), so bundling them would multiply the
#: release size and raise a redistribution question the Apache-2.0 Small model
#: does not — they are an opt-in one-file download instead. Small is always
#: bundled and never listed here. Kept behind a release tag so the whole set
#: moves as a unit and an older app keeps resolving the URLs it shipped with.
DOWNLOAD_BASE_URL = (
    "https://github.com/criso2hd-alt/DLSS5-Image-Converter/"
    "releases/download/depth-models-v1"
)

#: model_id -> hosted fp16 asset. ``sha256`` and ``bytes`` are filled once the
#: asset is actually uploaded (see scripts/export_onnx.py --fp16); until then
#: ``is_downloadable`` returns False and the app falls back to the bundled Small,
#: so shipping this half-populated changes nothing at run time. The downloaded
#: file is saved locally under ``ONNX_FILES[model_id]`` so ``locate`` finds it,
#: regardless of the asset's own name.
ONNX_DOWNLOADS: dict[str, dict[str, object]] = {
    "depth-anything/Depth-Anything-V2-Base-hf": {
        "asset": "Depth-Anything-V2-Base-hf.fp16.onnx",
        "sha256": "1199aedf43ce042a5dce13cee99d3ee1a77b3792fba657e628556270eff68451",
        "bytes": 194738079,
    },
    "depth-anything/Depth-Anything-V2-Large-hf": {
        "asset": "Depth-Anything-V2-Large-hf.fp16.onnx",
        "sha256": "9911617ea9c39de788ab6f2b38b9aeb811ce9b864965997cf86eb41b3276e3b1",
        "bytes": 668961707,
    },
}


def is_downloadable(model_id: str) -> bool:
    """Whether `model_id` has a hosted asset the app can fetch.

    False until the asset's ``sha256`` is filled in ONNX_DOWNLOADS, so a build
    made before the models were uploaded simply treats them as unavailable
    rather than chasing a URL that 404s.
    """
    entry = ONNX_DOWNLOADS.get(model_id)
    return bool(entry and entry.get("sha256"))


def download_model(
    model_id: str,
    dest_dir: Path,
    progress: Callable[[str], None] | None = None,
    bytes_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Fetch the hosted fp16 ONNX for `model_id` into `dest_dir`, verified.

    Streams to a ``.part`` file, checks size and SHA-256 before committing, and
    only then atomically renames into place — so an interrupted or corrupted
    download never leaves a half-file that ``locate`` would happily load and
    ONNX Runtime would then choke on. Returns the final path.
    """
    import hashlib
    import urllib.request

    entry = ONNX_DOWNLOADS.get(model_id)
    if not entry or not entry.get("sha256"):
        raise RuntimeError(f"No download is available for {model_id}.")

    # Same trust-store fix the HF download uses: verify TLS against the OS store
    # so an AV/proxy's private root does not fail the fetch (see depth_engine).
    from .depth_engine import enable_system_trust_store

    enable_system_trust_store()

    asset = str(entry["asset"])
    expected_sha = str(entry["sha256"])
    expected_bytes = int(entry["bytes"]) if entry.get("bytes") else 0
    url = f"{DOWNLOAD_BASE_URL}/{asset}"

    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / ONNX_FILES[model_id]
    part = final.with_suffix(final.suffix + ".part")

    if progress:
        progress("Downloading the depth model…")
    digest = hashlib.sha256()
    done = 0
    with urllib.request.urlopen(url) as response:  # noqa: S310 - fixed https host
        total = expected_bytes or int(response.headers.get("Content-Length") or 0)
        with open(part, "wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if bytes_progress and total:
                    bytes_progress(done, total)

    if expected_bytes and done != expected_bytes:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded {done} bytes but expected {expected_bytes}. "
            "The download was interrupted; try again."
        )
    if digest.hexdigest() != expected_sha:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            "The downloaded depth model failed its integrity check and was "
            "discarded. Try again, or pick a different model."
        )
    os.replace(part, final)
    return final

#: The square edge the models are exported at (multiple of 14; DINOv2 patch size).
INPUT = 518

#: DPT/ImageNet normalisation, from the model's preprocessor_config.json.
_MEAN = np.asarray([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
_STD = np.asarray([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)


def onnx_models_dir() -> Path:
    """Where downloaded ONNX depth models live, beside the HF model cache.

    The bundled Apache-2.0 Small model ships read-only inside the app
    (paths.bundled_onnx_dir); this cache is for larger models fetched later.
    """
    return paths.model_cache_dir() / "onnx"


def locate(model_id: str) -> Path | None:
    """The ONNX file for `model_id`, bundled copy first, else the cache.

    Returns None if it is not installed. Bundled wins so a release always has a
    working depth model with no download.
    """
    name = ONNX_FILES.get(model_id)
    if not name:
        return None
    for base in (paths.bundled_onnx_dir(), onnx_models_dir()):
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


#: The CUDA 13 / cuDNN 9 runtime DLLs onnxruntime-gpu loads by bare name at run
#: time. Windows resolves those against PATH and the executable's directory, not
#: against the folder the onnxruntime provider DLL lives in, so a bundled GPU
#: build has to put their directory on the search path itself (see
#: _prepare_gpu_libs). cudnn64_9.dll is the marker we probe for.
_CUDA_MARKER = "cudnn64_9.dll"


def _gpu_lib_dir() -> Path | None:
    """The bundled CUDA/cuDNN runtime directory, or None if this isn't a GPU build.

    The build ships the runtime DLLs beside onnxruntime's own provider DLLs
    (onnxruntime/capi), so that is where we look. A CPU-only or DirectML install
    simply has no cuDNN there and this returns None - the caller then does
    nothing and inference runs on whatever provider is available.
    """
    try:
        import onnxruntime as ort
    except Exception:  # noqa: BLE001 - onnxruntime absence is handled by the caller
        return None
    root = Path(ort.__file__).resolve().parent
    for candidate in (root / "capi", root):
        if (candidate / _CUDA_MARKER).is_file():
            return candidate
    return None


def _prepare_gpu_libs() -> None:
    """Make the bundled CUDA/cuDNN runtime DLLs loadable before a session opens.

    onnxruntime-gpu's CUDA provider dlopen()s cudnn/cublas/nvrtc by bare name at
    run time, and cuDNN 9 in turn loads its own split backend DLLs the same way.
    Windows resolves all of those against PATH and the app directory, never
    against the folder the provider DLL sits in - so co-locating them is not
    enough; the directory has to be on PATH. add_dll_directory covers the
    provider's own imports, PATH covers cuDNN's internal loads. A no-op on a
    non-Windows or non-GPU build, so it is safe to call unconditionally.
    """
    if os.name != "nt":
        return
    lib_dir = _gpu_lib_dir()
    if lib_dir is None:
        return
    d = str(lib_dir)
    if d not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(d)
    except (OSError, AttributeError):  # already added, or not on Windows
        pass


def _providers() -> list[str]:
    """GPU first, CPU as the guaranteed fallback.

    The shipped app installs onnxruntime-gpu or -directml; the plain onnxruntime
    package only offers CPU. Whatever is actually present is used, in that order
    of preference, so the same code runs on any of them.

    TensorRT is deliberately not requested: it needs the multi-hundred-MB
    TensorRT libraries (nvinfer) that we do not bundle, and listing it only makes
    onnxruntime probe for them and log a scary failure before falling back to
    CUDA. CUDA is the target GPU path; DirectML is the vendor-neutral fallback.
    """
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    preferred = [
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
        return locate(model_id) is not None

    def ensure_downloaded(
        self,
        model_id: str,
        progress: Callable[[str], None] | None = None,
        bytes_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Interface parity with depth_engine.DepthEngine: make `model_id`
        present, downloading it if there is a hosted asset for it.

        The bundled Small model is already present, so this returns at once for
        it. Base/Large are fetched from ONNX_DOWNLOADS when their asset has been
        published; if there is no source yet, it raises a clear message and the
        caller falls back to Small.

        This method existing at all is also the fix for a v0.3.0 crash: the
        download path (app.DownloadWorker) called ``engine.ensure_downloaded``
        blindly, and because this class did not define it, selecting Base/Large
        blew up with "'OnnxDepthEngine' object has no attribute
        'ensure_downloaded'".
        """
        if locate(model_id) is not None:
            return
        if is_downloadable(model_id):
            download_model(model_id, onnx_models_dir(), progress, bytes_progress)
            return
        name = ONNX_FILES.get(model_id, model_id)
        raise RuntimeError(
            f"{name} is not bundled and is not available to download in this "
            "build — only the Small model ships as ONNX."
        )

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

        if model_id not in ONNX_FILES:
            raise RuntimeError(f"No ONNX export is known for {model_id}.")
        path = locate(model_id)
        if path is None and model_id != SMALL:
            # A non-bundled model (Base/Large) that was never exported: fall back
            # to the always-present Small rather than failing the conversion.
            fallback = locate(SMALL)
            if fallback is not None:
                if progress:
                    progress("Selected depth model not installed; using Small.")
                model_id, path = SMALL, fallback
        if path is None:
            raise RuntimeError(
                f"The ONNX depth model is not installed ({ONNX_FILES[model_id]}). "
                "The Small model ships with the app; larger models must be "
                "exported with scripts/export_onnx.py or downloaded."
            )

        import onnxruntime as ort

        if progress:
            progress("Preparing the depth model…")
        # Put the bundled CUDA/cuDNN runtime on the DLL search path before we
        # ask onnxruntime which providers it can offer - get_available_providers
        # and the session both load those DLLs, and neither finds them otherwise.
        _prepare_gpu_libs()
        providers = _providers()
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), options, providers=providers)
        active = self.session.get_providers()[0]
        # Report the real backend rather than a blanket "gpu": the shipped build
        # runs on DirectML (any DX12 GPU), a user who installs onnxruntime-gpu
        # gets CUDA, and everyone else falls back to CPU.
        self.device = {
            "CUDAExecutionProvider": "cuda",
            "TensorrtExecutionProvider": "cuda",
            "DmlExecutionProvider": "directml",
        }.get(active, "cpu")
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
