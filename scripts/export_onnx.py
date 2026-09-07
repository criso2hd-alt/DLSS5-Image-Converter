"""Export Depth Anything V2 (the models the app uses) to ONNX.

Run once, on a machine with torch + transformers, to produce the ONNX files the
shipped app runs through ONNX Runtime - so the release itself never needs
PyTorch (a 2.7 GB download) or transformers.

    python scripts/export_onnx.py --all --out onnx_models

**Fixed 518x518 square input, on purpose.** Depth Anything's DINOv2 backbone
interpolates its positional grid from the concrete input size, which both
torch.onnx and torch.export specialise to the export shape - a genuinely
dynamic export needs surgery on the model code (what fabio-sim's repo does).
The app already resizes to 518 for depth, and a plain square resize scores 0.989
correlation against the aspect-preserving path on a 16:9 screenshot - depth here
is a guide for the neural pass, not the output, so that is well within tolerance.
A dynamic export is a possible later quality refinement.

This file is dev-time tooling and is never imported by the app.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}

#: The size the app feeds Depth Anything. Must be a multiple of 14 (patch size).
INPUT = 518


def export(model_id: str, out_dir: Path, fp16: bool) -> Path:
    from dlss5_converter import paths

    os.environ.setdefault("HF_HOME", str(paths.model_cache_dir()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # reuse the downloaded weights

    import torch
    from transformers import AutoModelForDepthEstimation

    model = AutoModelForDepthEstimation.from_pretrained(model_id, dtype=torch.float32).eval()

    class Wrap(torch.nn.Module):
        """Return just the depth map - all the pipeline consumes."""

        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values: "torch.Tensor") -> "torch.Tensor":
            return self.inner(pixel_values=pixel_values).predicted_depth

    out_dir.mkdir(parents=True, exist_ok=True)
    name = model_id.split("/")[-1]
    path = out_dir / f"{name}.onnx"

    dummy = torch.randn(1, 3, INPUT, INPUT, dtype=torch.float32)
    torch.onnx.export(
        Wrap(model), (dummy,), str(path),
        input_names=["pixel_values"], output_names=["predicted_depth"],
        opset_version=17, dynamo=False,
        # Batch stays dynamic; H and W are fixed at INPUT (see the module note).
        dynamic_axes={"pixel_values": {0: "batch"}, "predicted_depth": {0: "batch"}},
    )

    if fp16:
        import onnx
        from onnxconverter_common import float16

        fp16_model = float16.convert_float_to_float16(onnx.load(str(path)), keep_io_types=True)
        path = out_dir / f"{name}.fp16.onnx"
        onnx.save(fp16_model, str(path))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), help="Which size to export.")
    parser.add_argument("--all", action="store_true", help="Export small, base and large.")
    parser.add_argument("--out", type=Path, default=Path("onnx_models"))
    parser.add_argument("--fp16", action="store_true", help="Emit half-precision copies.")
    args = parser.parse_args()

    targets = list(MODELS) if args.all else [args.model or "small"]
    for key in targets:
        path = export(MODELS[key], args.out, args.fp16)
        print(f"{key:5} -> {path}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
