"""Turn a depth map into real 3D geometry.

Depth Anything V2 emits *relative inverse depth* — bright means near. The old
2.5D renderer used that directly as a pixel displacement, which cannot represent
occlusion and produced doubling at depth edges. Here the map is unprojected into
view space through a pinhole camera so the scene can be rasterised properly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import retopology, silhouette


@dataclass(slots=True)
class Mesh:
    """Triangle mesh in view space, textured by the source image."""

    positions: np.ndarray  # (N, 3) float32, metres, -Z forward
    uvs: np.ndarray  # (N, 2) float32
    indices: np.ndarray  # (M, 3) uint32
    width: int
    height: int
    wall_start: int = -1  # first triangle of the bridged walls, -1 if none

    @property
    def triangle_count(self) -> int:
        return int(self.indices.shape[0])

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def wall_triangle_count(self) -> int:
        if self.wall_start < 0:
            return 0
        return int(self.indices.shape[0]) - self.wall_start


def disparity_to_depth(
    disparity: np.ndarray, near: float = 1.0, far: float = 12.0
) -> np.ndarray:
    """Map normalised inverse depth in [0, 1] to view-space Z in [near, far].

    Interpolating in *disparity* rather than depth is what keeps parallax
    physically sensible: screen-space motion is proportional to 1/Z, so a linear
    ramp in disparity gives even foreground/background separation instead of
    crushing everything into the distance.
    """
    disparity = np.clip(disparity.astype(np.float32), 0.0, 1.0)
    inv_near, inv_far = 1.0 / float(near), 1.0 / float(far)
    inv_z = inv_far + (inv_near - inv_far) * disparity
    return (1.0 / np.maximum(inv_z, 1e-6)).astype(np.float32)


def focal_from_fov(height: int, fov_degrees: float) -> float:
    """Focal length in pixels for a vertical field of view."""
    return float(height) / (2.0 * np.tan(np.radians(fov_degrees) * 0.5))


def unproject(
    depth_z: np.ndarray, fov_degrees: float = 55.0, focal_height: int | None = None
) -> np.ndarray:
    """Lift a Z map to (H, W, 3) view-space points through a pinhole camera.

    `focal_height` keeps the focal length tied to the original photograph when
    the map has been padded for overscan. Without it, padding would silently
    widen the lens instead of extending the scene past the frame.
    """
    height, width = depth_z.shape[:2]
    focal = focal_from_fov(focal_height or height, fov_degrees)
    cx, cy = (width - 1) * 0.5, (height - 1) * 0.5
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    # Right-handed, -Z forward: X right, Y up, so image rows invert.
    x = (xs - cx) * depth_z / focal
    y = -(ys - cy) * depth_z / focal
    return np.stack([x, y, -depth_z], axis=-1).astype(np.float32)


def build_mesh(
    depth: np.ndarray,
    fov_degrees: float = 55.0,
    near: float = 1.0,
    far: float = 12.0,
    stride: int = 1,
    discontinuity: float = 0.06,
    valid: np.ndarray | None = None,
    focal_height: int | None = None,
    no_cut: np.ndarray | None = None,
    bridge: bool = True,
    wall_extent: float = 1.0,
    detail: float = 1.0,
) -> Mesh:
    """Build a displaced grid mesh, cut where the depth map jumps and bridged.

    `stride` subsamples the grid for preview-rate rendering; `discontinuity` is
    the fractional depth ratio above which a quad is dropped rather than
    stretched. Setting it to 0 or less keeps every quad, which reproduces the
    old rubber-sheet behaviour and is useful for comparison.

    Cutting alone leaves the scene full of holes, so by default every cut is
    immediately `bridge`d — see `silhouette`, which rebuilds the near side of a
    cut quad up to the true sub-quad edge and extrudes that edge backwards.
    `wall_extent` is the fraction of the near-to-far gap a wall spans; 1.0
    closes the opening completely, and less leaves the reconstructed backdrop
    to cover the remainder.

    `no_cut` marks pixels whose depth edges must be left connected. It exists
    for structures too thin to have a reconstructable volume — a tram wire has
    no side to bridge to, and cutting round it only shreds it into fragments.

    `detail` below 1.0 turns on adaptive quad retopology: flat regions collapse
    into larger quads while silhouettes keep every triangle. Unlike `stride`,
    which throws away resolution everywhere, this spends it where the geometry
    actually is. See `retopology`.
    """
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2D, got shape {depth.shape}")
    stride = max(1, int(stride))
    sampled = depth[::stride, ::stride]
    height, width = sampled.shape
    if height < 2 or width < 2:
        raise ValueError("depth map is too small to triangulate")

    depth_z = disparity_to_depth(sampled, near, far)
    reference = None if focal_height is None else max(1, focal_height // stride)
    points = unproject(depth_z, fov_degrees, reference).reshape(-1, 3)

    ys, xs = np.mgrid[0:height, 0:width]
    # UVs address the full-resolution texture regardless of stride.
    uvs = np.stack(
        [
            (xs * stride) / max(depth.shape[1] - 1, 1),
            (ys * stride) / max(depth.shape[0] - 1, 1),
        ],
        axis=-1,
    ).reshape(-1, 2).astype(np.float32)

    idx = np.arange(height * width, dtype=np.uint32).reshape(height, width)
    tl = idx[:-1, :-1].ravel()
    tr = idx[:-1, 1:].ravel()
    bl = idx[1:, :-1].ravel()
    br = idx[1:, 1:].ravel()
    # Counter-clockwise winding when viewed down -Z.
    first = np.stack([tl, bl, tr], axis=-1)
    second = np.stack([tr, bl, br], axis=-1)

    keep_first = np.ones(first.shape[0], dtype=bool)
    keep_second = np.ones(second.shape[0], dtype=bool)
    cut = np.zeros(first.shape[0], dtype=bool)
    silhouette_result = None
    coverage_trim = None
    if discontinuity > 0:
        # Sub-quad silhouette reconstruction. A cut quad's near side is rebuilt
        # up to the true edge and walled back, rather than dropped whole and
        # bridged only when its split happened to follow the grid. See
        # `silhouette` for why the grid-aligned version staircased.
        silhouette_result = silhouette.build(
            depth_z,
            sampled,
            points,
            uvs,
            discontinuity,
            wall_extent=wall_extent if bridge else 0.0,
            spare=None if no_cut is None else no_cut[::stride, ::stride].astype(bool),
            owned=None if valid is None else valid[::stride, ::stride].astype(bool),
        )
        cut = silhouette_result.cut.ravel()
        keep_first &= ~cut
        keep_second &= ~cut
    if valid is not None:
        # Test triangles independently. Requiring all four quad corners removed
        # both triangles when only one corner crossed a silhouette, producing a
        # visible one-quad crack around every layer.
        v = valid[::stride, ::stride].astype(bool)
        keep_first &= (v[:-1, :-1] & v[1:, :-1] & v[:-1, 1:]).ravel()
        keep_second &= (v[:-1, 1:] & v[1:, :-1] & v[1:, 1:]).ravel()
        # Whole-quad culling leaves a layer's own outline as a grid staircase —
        # the comb edge left over once the depth silhouettes were smoothed.
        # Rebuild the covered part of each straddling quad instead.
        coverage_trim = silhouette.trim_coverage(
            v,
            depth_z,
            points,
            uvs,
            skip=None if silhouette_result is None else silhouette_result.cut,
            base=points.shape[0]
            + (0 if silhouette_result is None else silhouette_result.positions.shape[0]),
        )
    tolerance = retopology.tolerance_for_detail(detail)
    merged = np.zeros((0, 3), np.uint32)
    if tolerance > 0:
        # A cell may only merge when neither of its triangles was dropped, so
        # every cut and every layer boundary keeps its full-resolution shape.
        blocked = (~keep_first | ~keep_second).reshape(height - 1, width - 1)
        levels = retopology.leaf_levels(depth_z, blocked, tolerance)
        merged = retopology.leaf_triangles(levels, width)
        swallowed = (levels > 0).ravel()
        keep_first &= ~swallowed
        keep_second &= ~swallowed

    first, second = first[keep_first], second[keep_second]

    indices = np.concatenate([first, second, merged], axis=0).astype(np.uint32)
    # Vertex order must match the bases the two stages allocated against:
    # grid, then depth silhouette, then coverage trim.
    wall_start = -1
    if silhouette_result is not None and not silhouette_result.is_empty:
        points = np.concatenate([points, silhouette_result.positions], axis=0)
        uvs = np.concatenate([uvs, silhouette_result.uvs], axis=0)
        indices = np.concatenate([indices, silhouette_result.surface], axis=0)
    if coverage_trim is not None and coverage_trim[2].shape[0]:
        trim_points, trim_uvs, trim_surface, _trimmed = coverage_trim
        points = np.concatenate([points, trim_points], axis=0)
        uvs = np.concatenate([uvs, trim_uvs], axis=0)
        indices = np.concatenate([indices, trim_surface], axis=0)
    if silhouette_result is not None and bridge and silhouette_result.walls.shape[0]:
        # Walls last so `wall_start` stays a clean split point in the index list.
        wall_start = int(indices.shape[0])
        indices = np.concatenate([indices, silhouette_result.walls], axis=0)
    if tolerance > 0:
        # Merging strands most of the grid, so compact once at the end — after
        # the silhouette geometry is in, so a single remap covers all of it.
        points, uvs, indices, _remap = retopology.compact(
            points, uvs, indices, np.zeros(0, np.uint32)
        )
    return Mesh(points, uvs, indices, width, height, wall_start)


def depth_discontinuity_mask(
    depth: np.ndarray, near: float = 1.0, far: float = 12.0, threshold: float = 0.06
) -> np.ndarray:
    """Pixel mask of foreground edges, for driving background inpainting.

    Marks the *near* side of each depth jump — the surface that will move and
    expose whatever sits behind it.
    """
    depth_z = disparity_to_depth(depth, near, far)
    disparity = np.clip(depth.astype(np.float32), 0.0, 1.0)
    mask = np.zeros(depth.shape, dtype=bool)
    ratio = 1.0 + threshold
    for axis in (0, 1):
        a = np.take(depth_z, np.arange(depth_z.shape[axis] - 1), axis=axis)
        b = np.take(depth_z, np.arange(1, depth_z.shape[axis]), axis=axis)
        da = np.take(disparity, np.arange(disparity.shape[axis] - 1), axis=axis)
        db = np.take(disparity, np.arange(1, disparity.shape[axis]), axis=axis)
        # Same two-part test the mesh cut uses, so the holes this marks are
        # exactly the holes the geometry opens. See `silhouette`.
        jump = (np.maximum(a, b) / np.maximum(np.minimum(a, b), 1e-6) > ratio) & (
            np.abs(da - db) > threshold
        )
        nearer_first = a < b
        lo = [slice(None), slice(None)]
        hi = [slice(None), slice(None)]
        lo[axis] = slice(0, depth_z.shape[axis] - 1)
        hi[axis] = slice(1, depth_z.shape[axis])
        mask[tuple(lo)] |= jump & nearer_first
        mask[tuple(hi)] |= jump & ~nearer_first
    return mask
