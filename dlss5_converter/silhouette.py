"""Sub-quad silhouette reconstruction at depth discontinuities.

The previous approach classified each culled quad and only bridged the four
patterns whose near/far split happened to follow a grid row or column. On a
clean rectangular test object nearly every split is axis aligned, so it looked
correct. On a photograph almost every silhouette is a diagonal roofline, a pole
edge or a wire, and those patterns produced *no wall at all* — measured at 8,256
wall triangles against 7.26 million, or 0.11%. What survived was a grid-aligned
staircase, which is the sawtooth edge visible in every off-axis render.

This module replaces that with a marching-squares boundary:

* **Crossings live on edges, not quads.** Whether a grid edge spans a
  discontinuity is a property of its two samples alone, so two quads sharing an
  edge always agree, and the crossing vertex is shared between them. That is
  what makes a silhouette come out as one continuous curve instead of a row of
  disconnected panels.
* **Every configuration is handled.** The near-side polygon is derived by
  walking the quad's corner cycle and inserting a crossing wherever the
  near/far label changes, so all fourteen non-trivial cases fall out of one
  rule rather than a lookup table of the convenient ones. The two diagonal
  saddles yield two components, which is correct — those really are two
  separate corners of surface.
* **The near polygon is real geometry.** A cut quad is not simply dropped: the
  part of it on the near side is rebuilt up to the true silhouette, which is
  what removes the staircase and recovers the half-quad of surface the old code
  threw away at every edge.
* **Crossing vertices sit at the near surface's own depth**, not interpolated
  towards the far sample. Interpolating in depth is precisely the smear along
  the view ray this whole rewrite exists to remove.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Quad corners, indexed as (top-left, top-right, bottom-left, bottom-right).
TL, TR, BL, BR = 0, 1, 2, 3
#: Corners walked in cyclic order around the quad.
_CORNER_CYCLE = (TL, TR, BR, BL)
#: Edge leaving _CORNER_CYCLE[i] towards _CORNER_CYCLE[i + 1].
#: 0 top, 1 right, 2 bottom, 3 left.
_CYCLE_EDGES = (0, 1, 2, 3)


def _build_templates() -> dict[int, list[list[tuple[str, int]]]]:
    """Near-side polygons for all sixteen corner labellings.

    Generated rather than written out, because a hand-maintained table is how
    ten of the fourteen real cases came to be missing in the first place.

    Each component starts and ends with a crossing; the chord closing it back
    from the last crossing to the first is the silhouette segment that gets
    extruded into a wall.
    """
    templates: dict[int, list[list[tuple[str, int]]]] = {}
    for code in range(16):
        near = [(code >> corner) & 1 for corner in range(4)]
        if all(near) or not any(near):
            continue  # nothing crosses this quad
        sequence: list[tuple[str, int]] = []
        for step in range(4):
            corner = _CORNER_CYCLE[step]
            following = _CORNER_CYCLE[(step + 1) % 4]
            if near[corner]:
                sequence.append(("c", corner))
            if near[corner] != near[following]:
                sequence.append(("e", _CYCLE_EDGES[step]))
        length = len(sequence)
        # Two crossings in a row mean the walk left the near region and
        # re-entered it: that is where one component ends and the next begins.
        breaks = [
            i
            for i in range(length)
            if sequence[i][0] == "e" and sequence[(i + 1) % length][0] == "e"
        ]
        components = []
        for position, start_break in enumerate(breaks):
            end = breaks[(position + 1) % len(breaks)]
            component = []
            index = (start_break + 1) % length
            while True:
                component.append(sequence[index])
                if index == end:
                    break
                index = (index + 1) % length
            components.append(component)
        templates[code] = components
    return templates


TEMPLATES = _build_templates()


def edge_crossings(
    depth_z: np.ndarray, disparity: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Flag grid edges that span a depth discontinuity.

    Both readings must agree, for the reason the old per-triangle test needed
    them: depth is 1/disparity, so out near the far plane a few thousandths of
    estimator noise reads as a large depth ratio and would shred a noisy sky.
    """

    def spans(near_z, far_z, near_d, far_d):
        low = np.minimum(near_z, far_z)
        high = np.maximum(near_z, far_z)
        ratio = (high / np.maximum(low, 1e-6)) > (1.0 + threshold)
        return ratio & (np.abs(near_d - far_d) > threshold)

    horizontal = spans(
        depth_z[:, :-1], depth_z[:, 1:], disparity[:, :-1], disparity[:, 1:]
    )
    vertical = spans(
        depth_z[:-1, :], depth_z[1:, :], disparity[:-1, :], disparity[1:, :]
    )
    return horizontal, vertical


@dataclass(slots=True)
class Silhouette:
    """Rebuilt near-side geometry plus the walls closing what it cut away."""

    #: Extra vertices: two per crossing edge, at the near and far depths.
    positions: np.ndarray  # (V, 3) float32
    uvs: np.ndarray  # (V, 2) float32
    #: Triangles rebuilding the near side of each cut quad.
    surface: np.ndarray  # (A, 3) uint32
    #: Triangles walling the openings, appended after the surface.
    walls: np.ndarray  # (B, 3) uint32
    #: Quads whose original two triangles must be dropped.
    cut: np.ndarray  # (H-1, W-1) bool

    @property
    def is_empty(self) -> bool:
        return self.surface.shape[0] == 0 and self.walls.shape[0] == 0

    @staticmethod
    def empty(shape: tuple[int, int]) -> "Silhouette":
        return Silhouette(
            np.zeros((0, 3), np.float32),
            np.zeros((0, 2), np.float32),
            np.zeros((0, 3), np.uint32),
            np.zeros((0, 3), np.uint32),
            np.zeros(shape, bool),
        )


def build(
    depth_z: np.ndarray,
    disparity: np.ndarray,
    points: np.ndarray,
    uvs: np.ndarray,
    threshold: float,
    wall_extent: float = 1.0,
    spare: np.ndarray | None = None,
    owned: np.ndarray | None = None,
) -> Silhouette:
    """Cut at discontinuities and rebuild the near side up to the silhouette.

    `spare` protects structures too thin to have a reconstructable volume — a
    wire has no side to bridge to, and cutting round it only shreds it.
    `owned` restricts rebuilt geometry to one layer's own pixels.
    """
    height, width = depth_z.shape
    if height < 2 or width < 2 or threshold <= 0:
        return Silhouette.empty((max(height - 1, 0), max(width - 1, 0)))

    cross_h, cross_v = edge_crossings(depth_z, disparity, threshold)
    # Quad (r, c) edges: top and bottom are horizontal, left and right vertical.
    top = cross_h[:-1, :]
    bottom = cross_h[1:, :]
    left = cross_v[:, :-1]
    right = cross_v[:, 1:]
    cut = top | bottom | left | right

    if spare is not None:
        # One protected corner spares the whole quad: a thin structure is only
        # worth protecting if it stays attached along its length.
        quad_spared = (
            spare[:-1, :-1] | spare[:-1, 1:] | spare[1:, :-1] | spare[1:, 1:]
        )
        cut &= ~quad_spared
    if not cut.any():
        return Silhouette.empty((height - 1, width - 1))

    z = depth_z
    corner_z = np.stack(
        [z[:-1, :-1], z[:-1, 1:], z[1:, :-1], z[1:, 1:]], axis=-1
    )  # (H-1, W-1, 4) in TL, TR, BL, BR order
    lowest = corner_z.min(axis=-1)
    highest = corner_z.max(axis=-1)
    # Geometric mean keeps the split scale invariant, matching the ratio test
    # that decided the edge crossed at all.
    split = np.sqrt(lowest * highest)[..., None]
    near_label = corner_z < split

    # The labelling and the edge flags must tell the same story, otherwise the
    # polygon would not close. They disagree only for borderline thresholds;
    # those quads are left as a plain cut rather than guessed at.
    consistent = (
        ((near_label[..., TL] != near_label[..., TR]) == top)
        & ((near_label[..., BL] != near_label[..., BR]) == bottom)
        & ((near_label[..., TL] != near_label[..., BL]) == left)
        & ((near_label[..., TR] != near_label[..., BR]) == right)
    )
    buildable = cut & consistent

    grid = np.arange(height * width, dtype=np.uint32).reshape(height, width)
    corner_index = np.stack(
        [grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1], grid[1:, 1:]], axis=-1
    )
    if owned is not None:
        # A rebuilt near polygon belongs to the layer that owns its near corners.
        owns = np.stack(
            [owned[:-1, :-1], owned[:-1, 1:], owned[1:, :-1], owned[1:, 1:]], axis=-1
        )
        buildable &= np.all(owns | ~near_label, axis=-1)

    # Crossing vertices are appended after the grid, so allocate their ids in
    # the final mesh's index space and the two kinds mix freely in a triangle.
    crossing = _allocate_crossings(
        cross_h, cross_v, depth_z, points, uvs, wall_extent, height, width,
        base=height * width,
    )
    surface, walls = _emit(
        buildable, near_label, corner_index, crossing, top, bottom, left, right
    )
    return Silhouette(
        crossing.positions, crossing.uvs, surface, walls, cut
    )


def trim_coverage(
    covered: np.ndarray,
    depth_z: np.ndarray,
    points: np.ndarray,
    uvs: np.ndarray,
    skip: np.ndarray | None = None,
    base: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild partly covered quads up to the coverage boundary.

    A layer's own outline was culled a whole quad at a time, so every layer
    ended in the same grid staircase the depth cuts used to have — the comb
    edge left over after the depth silhouettes were fixed.

    The same marching-squares walk applies, with two differences: "near" means
    "this layer owns the pixel" rather than "closer", and nothing is walled. A
    coverage boundary is not a depth edge; the layer behind it is already
    complete, so giving it a wall would rebuild a silhouette that does not
    exist. `skip` excludes quads a depth cut has already rebuilt.

    Returns (positions, uvs, surface triangles, trimmed quads).
    """
    height, width = covered.shape
    empty = (
        np.zeros((0, 3), np.float32),
        np.zeros((0, 2), np.float32),
        np.zeros((0, 3), np.uint32),
        np.zeros((max(height - 1, 0), max(width - 1, 0)), bool),
    )
    if height < 2 or width < 2:
        return empty

    corner = np.stack(
        [covered[:-1, :-1], covered[:-1, 1:], covered[1:, :-1], covered[1:, 1:]],
        axis=-1,
    )
    partial = corner.any(axis=-1) & ~corner.all(axis=-1)
    if skip is not None:
        partial &= ~skip
    if not partial.any():
        return empty

    # Coverage is binary, so an edge crosses exactly when its ends disagree.
    cross_h = covered[:, :-1] != covered[:, 1:]
    cross_v = covered[:-1, :] != covered[1:, :]
    crossing = _allocate_crossings(
        cross_h, cross_v, depth_z, points, uvs,
        # extent 0: no wall is built, so the far vertex is never referenced and
        # collapsing it onto the near one keeps the buffer half the size.
        0.0, height, width,
        # These vertices are appended *after* any the depth silhouette already
        # allocated, so the caller supplies the offset. Sharing a base with
        # `build` silently repointed every wall triangle at a coverage vertex.
        base=points.shape[0] if base is None else int(base),
        keep_covered=covered, with_far=False,
    )
    grid = np.arange(height * width, dtype=np.uint32).reshape(height, width)
    corner_index = np.stack(
        [grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1], grid[1:, 1:]], axis=-1
    )
    surface, _walls = _emit(
        partial,
        corner,
        corner_index,
        crossing,
        cross_h[:-1, :],
        cross_v[:, 1:],
        cross_h[1:, :],
        cross_v[:, :-1],
        walls=False,
    )
    return crossing.positions, crossing.uvs, surface, partial


@dataclass(slots=True)
class _Crossings:
    positions: np.ndarray
    uvs: np.ndarray
    near_id: np.ndarray  # (2, H, W) int64 per orientation, -1 when absent
    far_id: np.ndarray


def _allocate_crossings(
    cross_h, cross_v, depth_z, points, uvs, wall_extent, height, width, base,
    keep_covered=None, with_far: bool = True,
) -> _Crossings:
    """One near and one far vertex per crossing edge, shared by both quads.

    The near vertex sits at the *midpoint of the two samples in the image* but
    at the nearer sample's own depth. A depth map has a hard step at a
    silhouette and no sub-pixel information, so the midpoint is the honest
    estimate, and holding the near depth keeps the surface flat right out to
    its edge instead of ramping towards the background.
    """
    grid_points = points.reshape(height, width, 3)
    grid_uvs = uvs.reshape(height, width, 2)

    near_id = np.full((2, height, width), -1, np.int64)
    far_id = np.full((2, height, width), -1, np.int64)
    position_parts, uv_parts = [], []
    total = 0

    for orientation, (flags, a_slice, b_slice) in enumerate(
        (
            (cross_h, (slice(None), slice(0, -1)), (slice(None), slice(1, None))),
            (cross_v, (slice(0, -1), slice(None)), (slice(1, None), slice(None))),
        )
    ):
        rows, cols = np.nonzero(flags)
        if rows.size == 0:
            continue
        a_pts = grid_points[a_slice][rows, cols]
        b_pts = grid_points[b_slice][rows, cols]
        a_uv = grid_uvs[a_slice][rows, cols]
        b_uv = grid_uvs[b_slice][rows, cols]
        a_z = depth_z[a_slice][rows, cols]
        b_z = depth_z[b_slice][rows, cols]

        if keep_covered is None:
            a_is_near = a_z <= b_z
        else:
            # Coverage decides the surviving side, not depth: the rebuilt
            # polygon belongs to the layer that owns the pixel, whichever of
            # the two happens to be closer.
            a_is_near = keep_covered[a_slice][rows, cols]
        near_z = np.where(a_is_near, a_z, b_z)
        far_z = np.where(a_is_near, b_z, a_z)
        # X and Y are the image midpoint; only Z is chosen, so the vertex never
        # slides along the view ray.
        mid = (a_pts + b_pts) * 0.5
        scale_near = near_z / np.maximum(-mid[:, 2], 1e-6)
        wall_z = near_z + (far_z - near_z) * float(wall_extent)
        scale_far = wall_z / np.maximum(-mid[:, 2], 1e-6)

        near_pts = np.empty_like(mid)
        near_pts[:, 0] = mid[:, 0] * scale_near
        near_pts[:, 1] = mid[:, 1] * scale_near
        near_pts[:, 2] = -near_z
        far_pts = np.empty_like(mid)
        far_pts[:, 0] = mid[:, 0] * scale_far
        far_pts[:, 1] = mid[:, 1] * scale_far
        far_pts[:, 2] = -wall_z

        # The near rim keeps the near surface's own texel rather than the
        # midpoint's, which would blend background across the silhouette.
        near_uv = np.where(a_is_near[:, None], a_uv, b_uv)
        far_uv = np.where(a_is_near[:, None], b_uv, a_uv)

        count = rows.size
        stride = 2 if with_far else 1
        ids = base + total + np.arange(count, dtype=np.int64) * stride
        near_id[orientation][rows, cols] = ids
        if not with_far:
            # Coverage trimming builds no walls, so a far vertex would be
            # allocated and never referenced. Skipping it halves the buffer.
            position_parts.append(near_pts.astype(np.float32))
            uv_parts.append(near_uv.astype(np.float32))
            total += count
            continue
        far_id[orientation][rows, cols] = ids + 1
        interleaved_pts = np.empty((count * 2, 3), np.float32)
        interleaved_pts[0::2] = near_pts
        interleaved_pts[1::2] = far_pts
        interleaved_uv = np.empty((count * 2, 2), np.float32)
        interleaved_uv[0::2] = near_uv
        interleaved_uv[1::2] = far_uv
        position_parts.append(interleaved_pts)
        uv_parts.append(interleaved_uv)
        total += count * 2

    positions = (
        np.concatenate(position_parts) if position_parts else np.zeros((0, 3), np.float32)
    )
    crossing_uvs = np.concatenate(uv_parts) if uv_parts else np.zeros((0, 2), np.float32)
    return _Crossings(positions, crossing_uvs, near_id, far_id)


def _emit(
    buildable, near_label, corner_index, crossing, top, bottom, left, right,
    walls: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Fan the near polygons and extrude their chords, grouped by code.

    Fourteen small vectorised groups rather than a loop over quads: a real
    photograph cuts tens of thousands of them.
    """
    code = (
        near_label[..., TL].astype(np.uint8)
        | (near_label[..., TR].astype(np.uint8) << 1)
        | (near_label[..., BL].astype(np.uint8) << 2)
        | (near_label[..., BR].astype(np.uint8) << 3)
    )
    rows, cols = np.nonzero(buildable)
    if rows.size == 0:
        return np.zeros((0, 3), np.uint32), np.zeros((0, 3), np.uint32)
    codes = code[rows, cols]

    # Edge lookup per quad: (orientation, row, col) for top/right/bottom/left.
    def edge_ids(table, rows, cols):
        return (
            table[0][rows, cols],          # top    : horizontal at row
            table[1][rows, cols + 1],      # right  : vertical at col + 1
            table[0][rows + 1, cols],      # bottom : horizontal at row + 1
            table[1][rows, cols],          # left   : vertical at col
        )

    surface_parts, wall_parts = [], []
    for value, components in TEMPLATES.items():
        hit = codes == value
        if not hit.any():
            continue
        r, c = rows[hit], cols[hit]
        corners = corner_index[r, c]  # (n, 4)
        near_edges = edge_ids(crossing.near_id, r, c)
        far_edges = edge_ids(crossing.far_id, r, c)

        for component in components:
            resolved = [
                corners[:, item] if kind == "c" else near_edges[item]
                for kind, item in component
            ]
            # Any unresolved crossing means the labelling and the edge flags
            # disagreed after all; drop those quads rather than index -1.
            valid = np.ones(r.shape[0], bool)
            for kind, item in component:
                if kind == "e":
                    valid &= near_edges[item] >= 0
            if not valid.any():
                continue
            resolved = [part[valid] for part in resolved]
            for pivot in range(1, len(resolved) - 1):
                surface_parts.append(
                    np.stack(
                        [resolved[0], resolved[pivot], resolved[pivot + 1]], axis=-1
                    )
                )
            if not walls:
                continue
            first_edge = component[0][1]
            last_edge = component[-1][1]
            near_a = near_edges[last_edge][valid]
            near_b = near_edges[first_edge][valid]
            far_a = far_edges[last_edge][valid]
            far_b = far_edges[first_edge][valid]
            wall_parts.append(np.stack([near_a, near_b, far_a], axis=-1))
            wall_parts.append(np.stack([far_a, near_b, far_b], axis=-1))

    surface = (
        np.concatenate(surface_parts).astype(np.uint32)
        if surface_parts
        else np.zeros((0, 3), np.uint32)
    )
    walls = (
        np.concatenate(wall_parts).astype(np.uint32)
        if wall_parts
        else np.zeros((0, 3), np.uint32)
    )
    return surface, walls
