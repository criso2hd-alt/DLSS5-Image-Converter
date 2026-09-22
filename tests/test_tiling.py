"""Ultra Detail tiling: the geometry, the feather merge, and the auto sizing.

Pure math, no GPU: this is the half of Ultra Detail that can be proven on any
machine. The neural pass per tile is exercised on hardware; the guarantee here
is that whatever it returns, the plan covers the image and the merge is seamless.
"""

from __future__ import annotations

import numpy as np
import pytest

from dlss5_converter import tiling
from dlss5_converter.tiling import Tile


# -- plan_tiles ------------------------------------------------------------------


def test_small_image_is_one_tile():
    tiles = tiling.plan_tiles(1000, 800, tile_max=2048, overlap=128)
    assert tiles == [Tile(0, 0, 1000, 800)]


def test_tiles_cover_every_pixel():
    w, h, tile_max, overlap = 5000, 3000, 2048, 256
    tiles = tiling.plan_tiles(w, h, tile_max, overlap)
    covered = np.zeros((h, w), dtype=bool)
    for t in tiles:
        covered[t.y:t.bottom, t.x:t.right] = True
    assert covered.all(), "tiling left a gap"


def test_tiles_stay_within_the_size_limit():
    tiles = tiling.plan_tiles(9000, 4000, tile_max=2048, overlap=200)
    assert all(t.w <= 2048 and t.h <= 2048 for t in tiles)


def test_last_tile_is_flush_with_the_edge():
    tiles = tiling.plan_tiles(5000, 2048, tile_max=2048, overlap=256)
    xs = sorted({t.x for t in tiles})
    assert xs[-1] == 5000 - 2048  # pulled back so the far edge is covered


def test_adjacent_tiles_actually_overlap():
    tiles = tiling.plan_tiles(5000, 2048, tile_max=2048, overlap=256)
    xs = sorted({t.x for t in tiles})
    # Each step advances by no more than tile_max - overlap, so neighbours share
    # at least `overlap` columns.
    assert all(nxt - cur <= 2048 - 256 for cur, nxt in zip(xs, xs[1:]))


# -- feather + merge -------------------------------------------------------------


def test_uniform_field_survives_the_merge_unchanged():
    """A constant image, tiled and merged, must come back constant — the feather
    weights have to sum to 1 everywhere, seams included."""
    w, h = 4000, 2500
    tiles = tiling.plan_tiles(w, h, tile_max=1500, overlap=192)
    const = np.full((h, w, 3), 0.37, dtype=np.float32)
    patches = [const[t.y:t.bottom, t.x:t.right].copy() for t in tiles]
    merged = tiling.merge_tiles((h, w), tiles, patches, overlap=192)
    assert np.allclose(merged, 0.37, atol=1e-4)


def test_merge_reconstructs_a_gradient():
    """A smooth ramp split across tiles and feathered back must match the
    original to within blend error — proof the overlap blends, not just abuts."""
    w, h = 3000, 1200
    ramp = np.linspace(0, 1, w, dtype=np.float32)
    img = np.repeat(ramp[None, :], h, axis=0)[:, :, None].repeat(3, axis=2)
    tiles = tiling.plan_tiles(w, h, tile_max=1024, overlap=160)
    patches = [img[t.y:t.bottom, t.x:t.right].copy() for t in tiles]
    merged = tiling.merge_tiles((h, w), tiles, patches, overlap=160)
    assert np.abs(merged - img).max() < 1e-3


def test_feather_keeps_border_sides_at_full_weight():
    # A tile flush against the top-left corner must not fade its outer edges.
    tile = Tile(0, 0, 500, 500)
    w = tiling.feather_weight(tile, full_w=2000, full_h=2000, overlap=128)
    assert w[0, 0] == pytest.approx(1.0)      # image corner, no neighbour
    assert w[-1, -1] < 1.0                     # interior corner, fades


def test_grayscale_result_merges_back_to_2d():
    w, h = 2500, 1500
    tiles = tiling.plan_tiles(w, h, tile_max=1024, overlap=128)
    img = np.random.default_rng(0).random((h, w), dtype=np.float32)
    patches = [img[t.y:t.bottom, t.x:t.right].copy() for t in tiles]
    merged = tiling.merge_tiles((h, w), tiles, patches, overlap=128)
    assert merged.shape == (h, w)


# -- auto sizing -----------------------------------------------------------------


def test_boost_factor_targets_the_runtime_ceiling():
    # A 1920-wide image with VRAM out of the way boosts up to the 7680 ceiling.
    factor = tiling.auto_boost_factor(1920, 1080, free_bytes=None, ceiling=7680)
    assert factor == pytest.approx(7680 / 1920)  # exactly 4x


def test_boost_factor_is_one_when_already_large():
    factor = tiling.auto_boost_factor(8000, 4000, free_bytes=None, ceiling=7680)
    assert factor == 1.0


def test_boost_factor_backs_off_when_vram_is_tight():
    # 512 MB free leaves almost nothing after the 1.25 GB fixed cost: no boost.
    factor = tiling.auto_boost_factor(1920, 1080, free_bytes=512 * 1024**2)
    assert factor == 1.0


def test_ultra_honours_the_requested_multiplier():
    factor, tile_max, overlap = tiling.auto_ultra(
        8000, 4000, requested_factor=4.0, ram_free=None, vram_free=None
    )
    assert factor == pytest.approx(4.0)
    # 8000 * 4 = 32000 — far past the 16384 texture limit, which is fine because
    # only the tiles become textures.
    assert tile_max <= tiling.SAFE_EVAL_DIM
    assert 0 < overlap < tile_max


def test_ultra_max_is_bounded_by_ram_not_the_texture_limit():
    # 8 GB free RAM: with 40 B/px and 60% usable, ~4.8 GB / 40 = 120 MP of merge.
    # For a 2000×1000 (2 MP) source that is ~sqrt(120/2) ≈ 7.7× — the bound is
    # RAM, and it is allowed to exceed the D3D12 side limit in total pixels.
    factor, _, _ = tiling.auto_ultra(
        2000, 1000, requested_factor=0.0, ram_free=8 * 1024**3, max_factor=64.0
    )
    expected = tiling.ram_max_factor(2000, 1000, 8 * 1024**3)
    assert factor == pytest.approx(expected)
    assert factor > 4.0  # RAM allows a lot for a small source


def test_ultra_request_is_clamped_to_the_ram_ceiling():
    # A greedy 16× request on a big source with little RAM is pulled back to fit.
    ram = 4 * 1024**3
    factor, _, _ = tiling.auto_ultra(
        8000, 4000, requested_factor=16.0, ram_free=ram, max_factor=64.0
    )
    assert factor <= tiling.ram_max_factor(8000, 4000, ram) + 1e-6


def test_ultra_falls_back_to_max_factor_without_ram_info():
    factor, _, _ = tiling.auto_ultra(
        500, 500, requested_factor=0.0, ram_free=None, max_factor=8.0
    )
    assert factor == pytest.approx(8.0)


def test_ultra_plan_tiles_are_all_legal_evaluations():
    factor, tile_max, overlap = tiling.auto_ultra(
        6000, 4000, requested_factor=4.0, ram_free=None, vram_free=None
    )
    big_w, big_h = int(6000 * factor), int(4000 * factor)
    tiles = tiling.plan_tiles(big_w, big_h, tile_max, overlap)
    assert all(t.w <= tiling.SAFE_EVAL_DIM and t.h <= tiling.SAFE_EVAL_DIM for t in tiles)


def test_ram_max_factor_is_none_without_a_reading():
    assert tiling.ram_max_factor(4000, 2000, None) is None


# -- streaming merger ------------------------------------------------------------


def test_stream_merger_matches_in_memory_merge(tmp_path):
    """The disk-backed streaming merge must produce the same pixels as the
    in-RAM merge_tiles — it is the same maths, just spilled to a memmap."""
    w, h = 2600, 1400
    rng = np.random.default_rng(1)
    img = rng.random((h, w, 3), dtype=np.float32)
    tiles = tiling.plan_tiles(w, h, tile_max=1024, overlap=160)
    patches = [img[t.y:t.bottom, t.x:t.right].copy() for t in tiles]

    in_ram = tiling.merge_tiles((h, w), tiles, patches, overlap=160)

    merger = tiling.StreamMerger(h, w, channels=3, overlap=160, scratch_dir=tmp_path)
    for tile, patch in zip(tiles, patches):
        merger.add(tile, patch)
    streamed = np.zeros((h, w, 3), np.float32)
    for y0, block in merger.rows(block=200):
        streamed[y0:y0 + block.shape[0]] = block
    merger.close()

    assert np.allclose(streamed, in_ram, atol=1e-4)


def test_stream_merger_cleans_up_its_scratch(tmp_path):
    merger = tiling.StreamMerger(64, 64, channels=3, overlap=8, scratch_dir=tmp_path)
    merger.add(Tile(0, 0, 64, 64), np.zeros((64, 64, 3), np.float32))
    list(merger.rows())
    merger.close()
    assert not list(tmp_path.glob("*.dat"))  # memmap files removed


def test_a_tile_is_never_planned_absurdly_small():
    """With little free VRAM the affordable side went to 0 and Ultra planned
    one-pixel tiles: 80 million for a 12k image, and a frozen window while the
    size label built them all."""
    from dlss5_converter import tiling
    factor, tile_max, overlap = tiling.auto_ultra(
        1920, 1040, requested_factor=0.0, ram_free=32 * 1024**3,
        vram_free=300 * 1024**2)          # 300 MB free: not enough for a tile
    assert tile_max >= tiling.MIN_TILE_SIDE
    assert overlap < tile_max
    assert tiling.count_tiles(int(1920 * factor), int(1040 * factor), tile_max, overlap) < 10_000


def test_count_tiles_matches_plan_tiles():
    from dlss5_converter import tiling
    for w, h, tile, overlap in ((1920, 1080, 512, 64), (4000, 2000, 1024, 128),
                                (900, 400, 1024, 128), (5000, 5000, 2048, 256)):
        assert tiling.count_tiles(w, h, tile, overlap) == len(
            tiling.plan_tiles(w, h, tile, overlap)), (w, h, tile, overlap)
