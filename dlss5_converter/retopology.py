"""Adaptive quad retopology for the displaced grid mesh.

A Full HD depth map triangulated at full density is ~11.8M triangles, and almost
all of them describe surfaces that are locally flat — a road, a wall, a sky. The
detail that matters is concentrated at silhouettes, which is exactly where the
mesh is cut and bridged.

So rather than remesh afterwards, merge *before* triangulating: walk a quadtree
over the grid and keep any block whose samples already lie on the plane its four
corners describe. The grid makes this unusually cheap, and it preserves
everything the general remeshers would destroy — the UVs stay exact, because a
merged block's corners are still grid vertices with their original texture
coordinates, and no new vertex is ever invented.

**Why there are no cracks.** A block merges only when *every* sample inside it,
including the midpoints of its edges, sits within `tolerance` of the two
triangles that will replace it. Along a shared edge that surface is a straight
line between the two endpoints, so a finer neighbour's extra vertices can be off
by at most `tolerance` — the T-junction gap is bounded by the merge test itself.
That is what makes the usual 2:1 balancing and midpoint stitching unnecessary.
"""

from __future__ import annotations

import numpy as np

# 2**5 = 32 cells. Past this the blocks are large enough that a single outlier
# sample keeps rejecting them, so the extra levels cost more than they save.
MAX_LEVEL = 5


def tolerance_for_detail(detail: float) -> float:
    """Map a 0-1 detail control to a merge tolerance in disparity units.

    1.0 disables merging outright, so the default path is byte-for-byte the
    uniform grid it always was. The curve is quadratic because the interesting
    range is all at the fine end: the first few thousandths of tolerance remove
    most of the triangles, and everything after that is just flattening.
    """
    detail = float(np.clip(detail, 0.0, 1.0))
    if detail >= 1.0:
        return 0.0
    return float((1.0 - detail) ** 2 * 0.04)


def _windows(values: np.ndarray, size: int, ny: int, nx: int) -> np.ndarray:
    """Aligned, overlapping (ny, nx, size+1, size+1) view of a vertex grid."""
    view = np.lib.stride_tricks.sliding_window_view(values, (size + 1, size + 1))
    return view[:: size, :: size][:ny, :nx]


def _planar(values: np.ndarray, size: int, ny: int, nx: int, tolerance: float):
    """Blocks whose every sample lies within `tolerance` of its two triangles.

    Measured against the triangle pair that will actually be emitted, not a
    bilinear patch, so the bound is the real reconstruction error.
    """
    block = _windows(values, size, ny, nx)
    top_left = block[..., :1, :1]
    top_right = block[..., :1, -1:]
    bottom_left = block[..., -1:, :1]
    bottom_right = block[..., -1:, -1:]

    axis = np.linspace(0.0, 1.0, size + 1, dtype=np.float32)
    u = axis[None, :]  # across columns
    v = axis[:, None]  # down rows
    # build_mesh splits each quad along bottom-left -> top-right, which is the
    # line u + v = 1 in block coordinates.
    lower = top_left + u * (top_right - top_left) + v * (bottom_left - top_left)
    upper = (
        bottom_right
        + (1.0 - u) * (bottom_left - bottom_right)
        + (1.0 - v) * (top_right - bottom_right)
    )
    predicted = np.where((u + v) <= 1.0, lower, upper)
    deviation = np.abs(block - predicted).reshape(ny, nx, -1).max(axis=-1)
    return deviation <= tolerance


def _any(cells: np.ndarray, size: int, ny: int, nx: int) -> np.ndarray:
    """True where an aligned block of cells contains at least one flagged cell."""
    view = np.lib.stride_tricks.sliding_window_view(cells, (size, size))
    return view[::size, ::size][:ny, :nx].reshape(ny, nx, -1).any(axis=-1)


def leaf_levels(
    values: np.ndarray,
    blocked: np.ndarray,
    tolerance: float,
    max_level: int = MAX_LEVEL,
) -> np.ndarray:
    """Quadtree level per grid cell; 0 keeps the cell at full resolution.

    `blocked` marks cells that must not be merged — a cut quad, or one whose
    corners are not all part of this layer. Silhouettes therefore keep every
    triangle they had, which is the whole point: the density is spent where the
    reconstruction is, not on the flat road in front of it.
    """
    cells_y, cells_x = values.shape[0] - 1, values.shape[1] - 1
    level = np.zeros((cells_y, cells_x), np.int8)
    if tolerance <= 0 or cells_y < 2 or cells_x < 2:
        return level

    taken = np.zeros((cells_y, cells_x), bool)
    offsets = None
    for exponent in range(int(max_level), 0, -1):
        size = 1 << exponent
        ny, nx = cells_y // size, cells_x // size
        if ny < 1 or nx < 1:
            continue
        free = ~_any(taken, size, ny, nx) & ~_any(blocked, size, ny, nx)
        if not free.any():
            continue
        merge = free & _planar(values, size, ny, nx, tolerance)
        block_y, block_x = np.nonzero(merge)
        if block_y.size == 0:
            continue
        offsets = np.arange(size)
        rows = (block_y[:, None] * size + offsets)[:, :, None]
        cols = (block_x[:, None] * size + offsets)[:, None, :]
        level[rows, cols] = exponent
        taken[rows, cols] = True
    return level


def leaf_triangles(
    level: np.ndarray, grid_width: int, max_level: int = MAX_LEVEL
) -> np.ndarray:
    """Two triangles per merged leaf, indexed into the original vertex grid.

    Cells left at level 0 are not emitted here: they still carry the per-triangle
    cut and coverage decisions that `build_mesh` made, and only it knows which of
    them survived.
    """
    cells_y, cells_x = level.shape
    parts: list[np.ndarray] = []
    for exponent in range(1, int(max_level) + 1):
        size = 1 << exponent
        if size > cells_y or size > cells_x:
            break
        # A leaf is identified by its top-left cell, so only look at origins.
        origins = level[::size, ::size] == exponent
        block_y, block_x = np.nonzero(origins)
        if block_y.size == 0:
            continue
        top_left = (block_y * size * grid_width + block_x * size).astype(np.uint32)
        top_right = top_left + size
        bottom_left = top_left + size * grid_width
        bottom_right = bottom_left + size
        # Same winding and diagonal as the full-resolution quads.
        parts.append(np.stack([top_left, bottom_left, top_right], axis=-1))
        parts.append(np.stack([top_right, bottom_left, bottom_right], axis=-1))
    if not parts:
        return np.zeros((0, 3), np.uint32)
    return np.concatenate(parts, axis=0).astype(np.uint32)


def compact(
    positions: np.ndarray,
    uvs: np.ndarray,
    indices: np.ndarray,
    keep: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Drop vertices no triangle references any more, and return the remap.

    Merging leaves most of the grid unused. Uploading it anyway would keep the
    vertex buffer at full size and give back only the index-buffer savings,
    which is the smaller half.

    `keep` names vertices to retain even when unreferenced, so that indices held
    outside the triangle list — a `CutBoundary`, for one — stay meaningful after
    the remap rather than silently pointing at the wrong vertex.
    """
    referenced = [indices.ravel()]
    if keep is not None and keep.size:
        referenced.append(keep.ravel())
    used = np.unique(np.concatenate(referenced).astype(np.uint32))
    remap = np.zeros(positions.shape[0], np.uint32)
    remap[used] = np.arange(used.size, dtype=np.uint32)
    if used.size == positions.shape[0]:
        return positions, uvs, indices, remap
    return positions[used], uvs[used], remap[indices], remap
