"""Learned super-resolution for the tiles, on ONNX Runtime (DirectML).

Ultra and Boost enlarge before the DLSS pass so DLAA's fixed-scale softening
covers less of each real detail. Lanczos did the enlarging; it invents nothing.
This runs a real SR model instead, so the enlargement *reconstructs* texture —
the difference between "rescaled" and "more detail", which matters most for the
architectural/texture case where DLAA otherwise erases fine, regular detail.

It rides the ONNX Runtime + DirectML stack the depth engine already ships, so it
adds no new runtime: any DirectX 12 GPU, no CUDA. Models are not bundled here in
bulk — a small default may sit in assets, larger/licence-restricted ones are
downloaded on demand or pointed at by the user (see MODELS). The SR pass runs
per tile, before DLSS; the caller keeps the SR tile as a texture reference to
graft back afterwards (see detail.preserve_detail).

Everything is display-referred RGB float in ``[0, 1]``: the models are trained
in that space, and a neutral/absent model simply means the caller keeps Lanczos.
Torch/transformers are never imported — inference is ONNX Runtime only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import paths


@dataclass(frozen=True)
class SRModel:
    """One selectable super-resolution model."""

    key: str
    name: str
    scale: int
    filename: str
    #: Direct download URL, or "" when the model must be bundled or user-supplied.
    url: str
    license: str
    attribution: str
    #: DAT/transformer models overflow in fp16; they must run fp32.
    fp32_only: bool = False
    #: True to ship inside the app (assets/onnx-sr); False = download/point-to.
    bundled: bool = False


#: The models offered, in the order shown. The fast general CNN is the main
#: option; Nomos2 is the slower, heavier ESRGAN photo look, second. Both download
#: from our own release on first use. UltraSharp V2 was dropped (its licence
#: forbids commercial use, and DAT2 was far too slow here).
MODELS: dict[str, SRModel] = {
    "realesr-general-x4v3": SRModel(
        key="realesr-general-x4v3",
        name="General x4 · fast",
        scale=4,
        filename="realesr-general-x4v3.onnx",
        # ~5 MB SRVGGNet (CNN): far faster than the ESRGAN model, gentler/less
        # prone to invented texture. Exported from xinntao's release checkpoint
        # and hosted on our release. BSD-3 (commercial use fine, attribution).
        url="https://github.com/criso2hd-alt/DLSS5-Image-Converter/releases/download/"
            "sr-models-v1/realesr-general-x4v3.onnx",
        license="BSD-3-Clause",
        attribution="realesr-general-x4v3 (Real-ESRGAN, xinntao) — BSD-3-Clause.",
        fp32_only=False,
        bundled=False,
    ),
    "nomos2-otf-esrgan": SRModel(
        key="nomos2-otf-esrgan",
        name="Nomos2 photo · slow",
        scale=4,
        filename="4xNomos2_otf_esrgan_fp32_opset17.onnx",
        # Hosted on our own release (CC-BY-4.0 permits redistribution with
        # attribution — see the Credits card and the release notes), so the app
        # does not depend on the author's URL staying put.
        url="https://github.com/criso2hd-alt/DLSS5-Image-Converter/releases/download/"
            "sr-models-v1/4xNomos2_otf_esrgan_fp32_opset17.onnx",
        license="CC-BY-4.0",
        attribution="4xNomos2_otf_esrgan by Philip Hofmann — CC-BY-4.0 "
                    "(commercial use permitted with attribution).",
        fp32_only=False,
        bundled=False,
    ),
}

DEFAULT_MODEL = "realesr-general-x4v3"

#: Fixed SR inference tile side. 512 keeps every inference the same shape (so
#: DirectML compiles the graph once, not per size) and keeps each compile + run
#: cheap; a 1024 tile made compilation and per-tile cost balloon, which is what
#: made SR feel hung. plan_tiles yields uniform tiles for any image ≥ this size.
SR_TILE_MAX = 512
SR_TILE_OVERLAP = 32


def preprocess(rgb01: np.ndarray) -> np.ndarray:
    """HWC RGB float [0,1] → NCHW float32 batch of 1, contiguous for ONNX."""
    chw = np.transpose(np.clip(rgb01, 0.0, 1.0).astype(np.float32), (2, 0, 1))
    return np.ascontiguousarray(chw[None, ...])


def postprocess(nchw: np.ndarray) -> np.ndarray:
    """NCHW model output → HWC RGB float clipped to [0,1]."""
    chw = np.asarray(nchw)[0]
    return np.clip(np.transpose(chw, (1, 2, 0)), 0.0, 1.0).astype(np.float32)


def _providers(fp32_only: bool) -> list:
    """DirectML first, then CPU. Mirrors onnx_depth's selection so SR runs on the
    same GPU path the depth model already uses."""
    import onnxruntime as ort  # lazy: importing ORT at module scope slows launch

    available = set(ort.get_available_providers())
    order = ["DmlExecutionProvider", "CPUExecutionProvider"]
    return [p for p in order if p in available] or ["CPUExecutionProvider"]


def model_cache_dir() -> Path:
    """Where downloaded SR models live, beside the depth model cache."""
    return paths.model_cache_dir() / "sr"


def download_model(
    model: SRModel,
    progress=None,
    bytes_progress=None,
) -> Path:
    """Download ``model`` (which must have a URL) into the SR cache, verified by size.

    Streams to a ``.part`` file and renames on success, so an interrupted fetch
    never leaves a half-file that ONNX Runtime would choke on. Returns the path.
    """
    import urllib.request

    if not model.url:
        raise RuntimeError(
            f"{model.name} has no download URL — install it manually or pick "
            "another model."
        )
    # Same OS trust-store fix the depth download uses, so an AV/proxy root does
    # not fail TLS.
    try:
        from .depth_engine import enable_system_trust_store

        enable_system_trust_store()
    except Exception:  # noqa: BLE001 - trust store is best-effort
        pass

    dest_dir = model_cache_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / model.filename
    part = final.with_suffix(final.suffix + ".part")
    if progress:
        progress(f"Downloading {model.name}…")
    done = 0
    with urllib.request.urlopen(model.url) as response:  # noqa: S310 - fixed https host
        total = int(response.headers.get("Content-Length") or 0)
        with open(part, "wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if bytes_progress and total:
                    bytes_progress(done, total)
    if total and done != total:
        part.unlink(missing_ok=True)
        raise RuntimeError("The download was interrupted; try again.")
    part.replace(final)
    return final


def locate(model: SRModel) -> Path | None:
    """The on-disk model file, bundled or downloaded, or None if absent."""
    candidates = [
        paths.resource_dir() / "dlss5_converter" / "assets" / "onnx-sr" / model.filename,
        model_cache_dir() / model.filename,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


class UpscaleEngine:
    """Loads one SR model and upscales RGB tiles. Lazily imports ONNX Runtime."""

    def __init__(self) -> None:
        self._session = None
        self._model: SRModel | None = None
        self._input_name = ""
        self._output_name = ""

    @property
    def scale(self) -> int:
        return self._model.scale if self._model else 1

    @property
    def model_key(self) -> str:
        return self._model.key if self._model else ""

    def load(self, model_key: str, model_path: Path | None = None) -> None:
        """Create the inference session for ``model_key``.

        ``model_path`` overrides the located file, so a user can point at their
        own .onnx. Raises with a plain message if the model file is missing.
        """
        model = MODELS.get(model_key)
        if model is None:
            raise ValueError(f"Unknown super-resolution model {model_key!r}.")
        # Already loaded this model — reuse the compiled session (was comparing an
        # SRModel object to a string here, so it never actually cached).
        if self._model is not None and self._model.key == model_key and self._session is not None:
            return
        path = model_path or locate(model)
        if path is None:
            raise FileNotFoundError(
                f"The {model.name} model file ({model.filename}) is not installed. "
                "Download it or point the app at the .onnx file."
            )
        import onnxruntime as ort
        from . import gpus

        options = ort.SessionOptions()
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=gpus.ort_providers(_providers(model.fp32_only))
        )
        self._session = session
        self._model = model
        self._input_name = session.get_inputs()[0].name
        self._output_name = session.get_outputs()[0].name

    def _infer(self, rgb01: np.ndarray) -> np.ndarray:
        assert self._session is not None
        out = self._session.run([self._output_name], {self._input_name: preprocess(rgb01)})
        return postprocess(out[0])

    def upscale(self, rgb01: np.ndarray) -> np.ndarray:
        """Upscale an HWC RGB float [0,1] image by the model's scale.

        Tiles internally when the input is larger than ``SR_TILE_MAX`` so a large
        image cannot exhaust GPU memory in one inference; small inputs (the usual
        per-tile case) run in a single pass.
        """
        if self._session is None:
            raise RuntimeError("No super-resolution model is loaded.")
        h, w = rgb01.shape[:2]
        if max(h, w) <= SR_TILE_MAX:
            return self._infer(rgb01)

        # Big input: SR each sub-tile, then feather-merge at the upscaled size.
        from . import tiling

        tiles = tiling.plan_tiles(w, h, SR_TILE_MAX, SR_TILE_OVERLAP)
        scale = self.scale
        out_tiles = [
            tiling.Tile(t.x * scale, t.y * scale, t.w * scale, t.h * scale) for t in tiles
        ]
        results = [
            self._infer(np.ascontiguousarray(rgb01[t.y:t.bottom, t.x:t.right]))
            for t in tiles
        ]
        return tiling.merge_tiles((h * scale, w * scale), out_tiles, results, SR_TILE_OVERLAP * scale)
