"""Spatial tiling for Ultra Detail, and the auto sizing behind silent Boost.

Boost supersamples the whole image and runs the neural pass **once**. That is
capped twice over: a D3D12 2D texture stops at 16384 px a side, and the DLSS
runtime itself refuses a feature above a working resolution well below that
(reference testing succeeds at 7680 and rejects 10240). So single-pass Boost
cannot process a long edge past roughly 7680 px no matter how much VRAM is free.

Ultra Detail lifts that ceiling. It supersamples further, then cuts the large
image into **overlapping tiles** each small enough to be one legal evaluation,
runs the neural pass per tile, and feather-merges them back. The overlap gives
each tile the neighbouring context the semantic pass needs and hides the seam;
the feather weights the overlap so no hard edge shows where two tiles meet.

Everything here is pure geometry and blending on plain arrays. It calls no
runtime and imports no Qt, so the whole plan and merge can be exercised without
a GPU — see ``tests/test_tiling.py``. The one place hardware enters is the auto
sizing, which reads free VRAM through :mod:`hardware` and otherwise falls back
to the D3D12 limit alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: A D3D12 2D texture cannot exceed this on a side. Also mirrored in settings for
#: the historical Boost guard; kept here so tiling's own clamps do not depend on
#: importing settings (which would drag the whole config surface into a module
#: that is deliberately math-only).
D3D12_MAX_TEXTURE_DIMENSION = 16384

#: Practical per-evaluation side ceiling. The DLSS neural feature rejects working
#: resolutions above this with InvalidParameter even when D3D12 and VRAM allow
#: the textures (measured: 7680 ok, 10240 refused). Boost targets it as its
#: single-pass ceiling.
SAFE_EVAL_DIM = 7680

#: Ultra sizes its tiles to this, well under SAFE_EVAL_DIM. Tiles right at 7680
#: created the DLSS feature but the *neural* pass did not take on them, so a
#: multi-tile Ultra came out as the plain upscale with no DLSS effect (reported
#: on 15K/30K, which is exactly where more than one tile appears). 6144 keeps
#: every tile comfortably inside the range where the neural pass applies; it just
#: means a few more, smaller tiles.
ULTRA_TILE_CEILING = 6144

#: Fraction of the tile spanned by the overlap band, per side that has a
#: neighbour. 12.5% of a ~2K tile is ~256 px of feather, wide enough that the
#: semantic pass's local relighting blends invisibly without wasting a large
#: share of every tile on redundant work.
DEFAULT_OVERLAP_FRACTION = 0.125

#: Smallest overlap worth having. Below this the feather is too narrow to hide a
#: seam, so a small tile still gets a usable band.
MIN_OVERLAP = 32

#: Smallest tile side Ultra will ever plan. Free VRAM is a moving target (a
#: browser, the 3D tab or another app can take most of it for a moment), and
#: the affordable side went to 0 when it was low, which planned one-pixel
#: tiles: 80 million of them for a 12k image, and minutes of a frozen window.
#: Below this the machine cannot do Ultra at all, and saying so beats planning
#: something absurd; the runtime is the authority on what it can allocate.
MIN_TILE_SIDE = 512

#: VRAM model for one evaluation at a given working size, mirroring pipeline's
#: Boost preflight so a tile is sized against the same estimate the whole-image
#: path uses. Fixed cost plus a per-pixel term above the harness's 24 B/px floor.
_FIXED_VRAM = int(1.25 * 1024**3)
_VRAM_PER_PIXEL = 32
_USABLE_FREE_FRACTION = 0.90

#: Host-RAM model for the Ultra merge. During tiling we hold the supersampled
#: colour (RGB f32, 12 B) and depth (f32, 4 B), plus the merge accumulator
#: (RGB f32, 12 B) and its weight (f32, 4 B), then a downscaled copy and a save
#: buffer. 40 B/px covers the simultaneous peak with headroom; only 60% of free
#: RAM is offered so the machine is not driven to swap.
_MERGE_BYTES_PER_PIXEL = 40
_RAM_USABLE_FRACTION = 0.60


@dataclass(frozen=True)
class Tile:
    """One tile's rectangle inside the (supersampled) working image.

    ``x, y`` is the top-left corner and ``w, h`` the size, all in working
    pixels. Neighbouring tiles overlap, so these rectangles are not a partition
    — they cover the image with ``overlap`` px of shared border.
    """

    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h


def eval_vram_estimate(width: int, height: int) -> int:
    """Estimated bytes the native harness needs at one working size."""
    return _FIXED_VRAM + width * height * _VRAM_PER_PIXEL


def _vram_side_limit(free_bytes: int | None) -> int:
    """Largest square side one evaluation can afford in the free VRAM.

    Returns the D3D12 side limit when VRAM is unknown — the caller then relies
    on the runtime to make the authoritative decision, exactly as the legacy
    Boost path did on hardware without an NVIDIA query.
    """
    if free_bytes is None:
        return D3D12_MAX_TEXTURE_DIMENSION
    usable = int(free_bytes * _USABLE_FREE_FRACTION) - _FIXED_VRAM
    if usable <= 0:
        return 0
    side = int((usable / _VRAM_PER_PIXEL) ** 0.5)
    return max(0, side)


def auto_boost_factor(
    width: int,
    height: int,
    *,
    free_bytes: int | None = None,
    ceiling: int = SAFE_EVAL_DIM,
) -> float:
    """Pick the single-pass Boost factor. No level, no slider — this is the calc.

    Boost's whole benefit is running DLAA at a larger size so its fixed-size
    softening covers less of each real detail. We take the long edge as high as
    one legal evaluation allows: the smaller of the runtime ceiling and what
    free VRAM affords. Returns a factor ``>= 1`` (1.0 meaning the image already
    fills the budget and Boost cannot do more — the UI then points at Ultra).
    """
    long_edge = max(1, int(max(width, height)))
    side_budget = min(int(ceiling), _vram_side_limit(free_bytes))
    if side_budget <= long_edge:
        return 1.0
    return side_budget / long_edge


def ram_max_factor(width: int, height: int, ram_free: int | None) -> float | None:
    """Largest enlargement whose merge fits in free host RAM, or None if unknown.

    The merged image lives in ordinary memory, so this — not VRAM — is what
    bounds "Max". Returns None when free RAM cannot be measured, so the caller
    falls back to the fixed ``max_factor`` ceiling alone.
    """
    if ram_free is None:
        return None
    usable = ram_free * _RAM_USABLE_FRACTION
    max_pixels = usable / _MERGE_BYTES_PER_PIXEL
    source_pixels = max(1, int(width) * int(height))
    if max_pixels <= source_pixels:
        return 1.0
    return max(1.0, (max_pixels / source_pixels) ** 0.5)


def auto_ultra(
    width: int,
    height: int,
    *,
    requested_factor: float = 0.0,
    ram_free: int | None = None,
    vram_free: int | None = None,
    max_factor: float = 16.0,
    tile_ceiling: int = ULTRA_TILE_CEILING,
) -> tuple[float, int, int]:
    """Plan Ultra Detail: how far to supersample, and how big a tile can be.

    The supersample factor is **not** bounded by the D3D12 texture limit — only
    the tiles become textures, and each is kept under ``tile_ceiling``. The
    merged output is a host array, so RAM bounds it instead. Returns
    ``(factor, tile_max, overlap)``:

    - ``factor``   how much to enlarge the source before tiling (>= 1),
    - ``tile_max`` the largest tile side one evaluation can afford (VRAM/runtime),
    - ``overlap``  the feather band width to plan tiles with.

    ``requested_factor`` is the user's multiplier; ``0`` (or below) means "Max" —
    as far as RAM and ``max_factor`` allow. Any request is clamped to the RAM and
    ``max_factor`` ceilings so a choice can never plan a merge the machine cannot
    hold.
    """
    ram_cap = ram_max_factor(width, height, ram_free)
    ceiling = float(max_factor)
    if ram_cap is not None:
        ceiling = min(ceiling, ram_cap)

    if requested_factor and requested_factor > 0:
        factor = min(float(requested_factor), ceiling)
    else:  # "Max"
        factor = ceiling
    factor = max(1.0, factor)

    tile_max = max(MIN_TILE_SIDE, min(int(tile_ceiling), _vram_side_limit(vram_free)))
    overlap = max(MIN_OVERLAP, int(tile_max * DEFAULT_OVERLAP_FRACTION))
    # Overlap must leave forward progress: a tile has to advance by at least one
    # pixel past the overlap, or plan_tiles would never reach the far edge.
    overlap = min(overlap, max(0, tile_max - 1))
    return factor, tile_max, overlap


def count_tiles(width: int, height: int, tile_max: int, overlap: int) -> int:
    """How many tiles plan_tiles would make, without making them.

    The size label only ever wanted the number, and building the list to count
    it is what froze the window when a tile came out tiny.
    """
    if tile_max <= 0:
        raise ValueError("tile_max must be positive")
    overlap = max(0, min(int(overlap), tile_max - 1))
    step = max(1, tile_max - overlap)

    def count(extent: int) -> int:
        if extent <= tile_max:
            return 1
        whole = (extent - tile_max) // step + 1
        # plan_tiles adds one more flush with the far edge when the last step
        # does not land on it.
        return whole + (1 if (extent - tile_max) % step else 0)

    return count(width) * count(height)


def plan_tiles(width: int, height: int, tile_max: int, overlap: int) -> list[Tile]:
    """Cover ``width×height`` with overlapping tiles of at most ``tile_max`` a side.

    Tiles step by ``tile_max - overlap`` and the last one on each axis is pulled
    back flush with the far edge, so the whole image is covered and every tile is
    a legal size. An image that already fits in one tile returns a single tile.
    """
    if tile_max <= 0:
        raise ValueError("tile_max must be positive")
    overlap = max(0, min(int(overlap), tile_max - 1))

    def starts(extent: int) -> list[int]:
        if extent <= tile_max:
            return [0]
        step = tile_max - overlap
        positions = list(range(0, extent - tile_max + 1, step))
        # range stops short of the exact edge whenever (extent - tile_max) is not
        # a multiple of step; add a final tile flush to the edge so nothing is
        # left uncovered. It overlaps its predecessor by more than `overlap`,
        # which the feather handles — a wider blend, never a gap.
        if positions[-1] != extent - tile_max:
            positions.append(extent - tile_max)
        return positions

    tiles: list[Tile] = []
    for y in starts(height):
        for x in starts(width):
            tiles.append(Tile(x, y, min(tile_max, width), min(tile_max, height)))
    return tiles


def _ramp(length: int, band: int, rising: bool) -> np.ndarray:
    """A 0→1 (or 1→0) linear ramp `band` px wide, flat 1 across the rest."""
    weight = np.ones(length, dtype=np.float32)
    if band <= 0 or length <= 0:
        return weight
    band = min(band, length)
    # Endpoints kept just above 0 so a pixel covered by only one tile at the very
    # edge of the overlap still contributes; a true 0 there would divide-by-zero
    # in merge where two feathers meet exactly.
    edge = np.linspace(1.0 / (band + 1), 1.0, band, dtype=np.float32)
    if rising:
        weight[:band] = edge
    else:
        weight[length - band:] = edge[::-1]
    return weight


def feather_weight(tile: Tile, full_w: int, full_h: int, overlap: int) -> np.ndarray:
    """Per-pixel blend weight for a tile, ramped only on sides with a neighbour.

    A side flush against the image border is left at full weight — there is no
    neighbour to blend with there, and feathering it would darken the frame edge.
    An interior side ramps from near-zero at the outer overlap edge to one past
    it, so where two tiles overlap their weights sum smoothly to full coverage.
    Returns an ``(h, w)`` float32 array in ``(0, 1]``.
    """
    wx = np.ones(tile.w, dtype=np.float32)
    wy = np.ones(tile.h, dtype=np.float32)
    if tile.x > 0:
        wx *= _ramp(tile.w, overlap, rising=True)
    if tile.right < full_w:
        wx *= _ramp(tile.w, overlap, rising=False)
    if tile.y > 0:
        wy *= _ramp(tile.h, overlap, rising=True)
    if tile.bottom < full_h:
        wy *= _ramp(tile.h, overlap, rising=False)
    return np.outer(wy, wx)


def merge_tiles(
    full_shape: tuple[int, int],
    tiles: list[Tile],
    results: list[np.ndarray],
    overlap: int,
) -> np.ndarray:
    """Feather-merge per-tile results into one image.

    ``results[i]`` is the neural output for ``tiles[i]``, an ``(h, w, C)`` array.
    Each is laid down weighted by its feather and normalised by the summed
    weight, so overlaps blend and the seams disappear. Works in whatever dtype
    the results carry, accumulating in float32.
    """
    if len(tiles) != len(results):
        raise ValueError(f"{len(tiles)} tiles but {len(results)} results")
    if not results:
        raise ValueError("nothing to merge")
    full_h, full_w = full_shape
    channels = results[0].shape[2] if results[0].ndim == 3 else 1
    accum = np.zeros((full_h, full_w, channels), dtype=np.float32)
    weight = np.zeros((full_h, full_w, 1), dtype=np.float32)

    for tile, patch in zip(tiles, results):
        patch = np.asarray(patch, dtype=np.float32)
        if patch.ndim == 2:
            patch = patch[:, :, None]
        w = feather_weight(tile, full_w, full_h, overlap)[:, :, None]
        accum[tile.y:tile.bottom, tile.x:tile.right] += patch * w
        weight[tile.y:tile.bottom, tile.x:tile.right] += w

    # No pixel can have zero weight: plan_tiles covers the whole image and every
    # feather keeps its border sides at full weight. Guard anyway so a future
    # planning change degrades to a visible artefact instead of NaNs.
    np.maximum(weight, 1e-6, out=weight)
    merged = accum / weight
    return merged[:, :, 0] if channels == 1 and results[0].ndim == 2 else merged


class StreamMerger:
    """Feather-merge tiles into an on-disk accumulator, for images too large for RAM.

    ``merge_tiles`` builds the whole accumulator in memory — fine for a few
    hundred megapixels, fatal at 600 MP where the accumulator, weight, source and
    encode buffers together exhaust RAM and the machine swaps to a standstill
    (the Ultra freeze). This does the identical feathered accumulation, but the
    accumulator and weight are ``np.memmap`` files on disk, so peak RAM is one
    tile: ``add`` each tile as it is produced, then stream normalised row blocks
    out with :meth:`rows` (e.g. straight into a BigTIFF). Same maths as
    ``merge_tiles`` — a test asserts they agree.

    Costs ~16 bytes/pixel of scratch disk (a 600 MP merge ≈ 10 GB) and expects
    an SSD; that is a disk the app has, versus RAM it does not.
    """

    def __init__(
        self, full_h: int, full_w: int, channels: int, overlap: int, scratch_dir: Path
    ) -> None:
        self.full_h = int(full_h)
        self.full_w = int(full_w)
        self.channels = int(channels)
        self.overlap = int(overlap)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        self._accum_path = scratch_dir / "ultra_accum.dat"
        self._weight_path = scratch_dir / "ultra_weight.dat"
        # mode "w+" creates and zero-fills, which is exactly the starting state.
        self.accum = np.memmap(
            self._accum_path, dtype=np.float32, mode="w+",
            shape=(self.full_h, self.full_w, self.channels),
        )
        self.weight = np.memmap(
            self._weight_path, dtype=np.float32, mode="w+",
            shape=(self.full_h, self.full_w, 1),
        )

    def add(self, tile: Tile, patch: np.ndarray) -> None:
        """Accumulate one tile's feathered contribution into the on-disk buffers."""
        patch = np.asarray(patch, dtype=np.float32)
        if patch.ndim == 2:
            patch = patch[:, :, None]
        w = feather_weight(tile, self.full_w, self.full_h, self.overlap)[:, :, None]
        # Read-modify-write against the memmap slice: bounded to the tile region,
        # so RAM holds one tile, not the image.
        self.accum[tile.y:tile.bottom, tile.x:tile.right] += patch * w
        self.weight[tile.y:tile.bottom, tile.x:tile.right] += w

    def rows(self, block: int = 256):
        """Yield ``(y0, normalised_block)`` in row bands, for streaming to a writer.

        Each block is ``accum / weight`` for those rows — the finished pixels —
        in float32, RAM bounded to ``block`` rows. The weight guard mirrors
        ``merge_tiles`` so a coverage gap degrades to an artefact, not NaNs.
        """
        for y0 in range(0, self.full_h, block):
            y1 = min(self.full_h, y0 + block)
            accum_block = np.array(self.accum[y0:y1], dtype=np.float32)
            weight_block = np.array(self.weight[y0:y1], dtype=np.float32)
            np.maximum(weight_block, 1e-6, out=weight_block)
            yield y0, accum_block / weight_block

    def close(self) -> None:
        """Release the memmaps and delete the scratch files."""
        # Drop the references so the files can be removed on Windows, where an
        # open memmap keeps a lock.
        self.accum = None  # type: ignore[assignment]
        self.weight = None  # type: ignore[assignment]
        for path in (self._accum_path, self._weight_path):
            try:
                path.unlink()
            except OSError:
                pass
