"""Try to build one 3D scene from a folder of photo-mode shots.

    python scripts/multiview_probe.py "C:\\shots" --fov 70 --out scene.ply

Prints what each consecutive pair contributed and writes a coloured point
cloud. This is the honesty check before any of it reaches the app: if the
parallax column is near zero or pairs start failing, the capture is the
problem, and no amount of work downstream will fix it.

Shots can arrive in any order: every pair is scored and the scene grows from
the strongest overlap, so jumps in the capture only orphan the shots nothing
else overlaps.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dlss5_converter import game_depth  # noqa: E402
from dlss5_converter import multiview as mv  # noqa: E402


SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def load(folder: Path, longest: int) -> list[mv.Shot]:
    shots = []
    for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in SUFFIXES):
        raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if raw is None:
            print(f"  skipped (unreadable): {path.name}")
            continue
        image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        # Matching and depth both cost time quadratically in pixels, and neither
        # gains anything above ~1600px for this purpose.
        height, width = image.shape[:2]
        if max(width, height) > longest:
            factor = longest / max(width, height)
            image = cv2.resize(image, (round(width * factor), round(height * factor)),
                               interpolation=cv2.INTER_AREA)
        shot = mv.Shot(path.name, image)
        # Real depth from the DLSS5 Scene Capture ReShade add-on, when present.
        # It is a measurement rather than a per-photo guess, so it wins.
        shot.disparity = game_depth.load(path, shot.size)
        shots.append(shot)
    return shots


def add_depth(shots: list[mv.Shot]) -> None:
    from dlss5_converter.onnx_depth import OnnxDepthEngine
    from dlss5_converter.settings import AppSettings

    measured = [shot for shot in shots if shot.disparity is not None]
    if measured:
        print(f"game depth for {len(measured)} of {len(shots)} shots")
    guessing = [shot for shot in shots if shot.disparity is None]
    if not guessing:
        return
    engine = OnnxDepthEngine()
    device = engine.load(AppSettings().depth.model_id)
    print(f"Depth Anything on {device} for {len(guessing)} shots")
    for shot in guessing:
        started = time.perf_counter()
        shot.disparity = engine.infer(shot.image)
        print(f"  {shot.name}: depth in {time.perf_counter() - started:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--fov", type=float, default=70.0,
                        help="horizontal field of view in degrees (game setting)")
    parser.add_argument("--vertical", action="store_true",
                        help="the --fov value is vertical, as some games report it")
    parser.add_argument("--longest", type=int, default=1600)
    parser.add_argument("--stride", type=int, default=2,
                        help="use every Nth pixel when building the cloud")
    parser.add_argument("--voxel", type=float, default=0.0,
                        help="merge points onto a grid of this size (world units)")
    parser.add_argument("--out", type=Path, default=Path("scene.ply"))
    args = parser.parse_args()

    shots = load(args.folder, args.longest)
    print(f"{len(shots)} shots from {args.folder}")
    if len(shots) < 2:
        print("Need at least two shots.")
        return 1

    sizes = {shot.size for shot in shots}
    if len(sizes) > 1:
        print(f"WARNING: mixed resolutions {sizes}; one camera model cannot fit all.")

    add_depth(shots)
    camera = mv.intrinsics(*shots[0].size, args.fov, horizontal=not args.vertical)

    started = time.perf_counter()
    reports, _ = mv.solve_graph(shots, camera)
    print(f"\nsolved in {time.perf_counter() - started:.1f}s\n")
    print(f"{'pair':<34}{'matches':>9}{'inliers':>9}{'parallax':>10}{'scale':>9}  note")
    for report in reports:
        pair = f"{report.first} -> {report.second}"
        print(f"{pair[:33]:<34}{report.matches:>9}{report.inliers:>9}"
              f"{report.parallax:>9.2f}°{report.scale:>9.3f}  "
              f"{'' if report.ok else 'FAILED: ' + report.note}")

    placed = [shot for shot in shots if shot.depth is not None]
    orphans = [shot.name for shot in shots if shot.depth is None]
    print(f"\nplaced {len(placed)} of {len(shots)} shots")
    for shot in placed:
        print(f"  {shot.name:<28} at {np.round(shot.centre, 3)}")

    if orphans:
        print(f"  not placed: {', '.join(orphans)}")
    for note in mv.diagnose(reports, shots):
        print(f"  ! {note}")

    points, colours = mv.fuse(shots, camera, args.stride, args.voxel)
    print(f"\nfused cloud: {len(points):,} points")
    if len(points):
        extent = points.max(axis=0) - points.min(axis=0)
        print(f"extent: {np.round(extent, 2)} (world units, 1 = first baseline)")
        mv.write_ply(args.out, points, colours)
        print(f"wrote {args.out}")
    return 0 if not orphans else 1


if __name__ == "__main__":
    raise SystemExit(main())
