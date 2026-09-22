"""Photo in, DLSS 5 frame out.

The whole conversion in one place so it can be exercised without the GUI:

    python -m dlss5_converter.pipeline portrait.jpg out.png
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import (
    bigtiff,
    contract,
    detail,
    effects,
    evaluator,
    grade,
    hardware,
    hdr,
    imaging,
    paths,
    runtime,
    sequence,
    tiling,
    wic,
)
from .depth_engine import DepthEngine
from .onnx_depth import OnnxDepthEngine
from .settings import (
    AppSettings,
    style_slug,
)

Progress = Callable[[str], None]


@dataclass
class Result:
    """Everything the UI wants to show after a run."""

    original: np.ndarray  # 0..1 float32 RGB, at the processed size
    enhanced: np.ndarray  # 0..1 float32 RGB
    depth_preview: np.ndarray  # uint8 RGB, turbo-mapped
    notes: str = ""
    #: The enhanced image before tone mapping, in linear light, when the source
    #: was HDR. This is the real output in that case - `enhanced` is a version
    #: of it made fit for an 8-bit screen - so an HDR export must come from
    #: here or the highlights it exists to preserve are already gone.
    enhanced_linear: np.ndarray | None = None
    #: The tone mapping white point, shared with `original` so the wipe does
    #: not change exposure halfway across.
    white: float = 1.0
    #: Ultra Detail only: the full-resolution image is streamed to this BigTIFF on
    #: scratch during the convert (it is far too large to hold in RAM), and
    #: `ultra_full_size` is its (width, height). Save moves this file to the user's
    #: location; the toggle by the size readout views it. None in every other mode.
    ultra_full_path: Path | None = None
    ultra_full_size: tuple[int, int] | None = None
    #: Ultra only: the grade + effects look was already baked into the source
    #: before the pipeline (the full output is too large to grade afterwards), so
    #: `enhanced` and the saved file already carry it and the app must NOT apply
    #: grade/effects again on top.
    look_baked: bool = False

    @property
    def hdr(self) -> bool:
        return self.enhanced_linear is not None

    @property
    def has_full(self) -> bool:
        """Whether a distinct full-resolution Ultra image is available to save."""
        return self.ultra_full_path is not None


@dataclass
class Prepared:
    """The expensive, image-only half of a conversion.

    Depth estimation is by far the slowest step and depends on nothing the user
    tunes afterwards — depth *contrast* is applied later, to this array, for
    pennies. Splitting it out lets the UI run it once when an image is opened
    and then re-convert repeatedly without paying for it again.
    """

    source: np.ndarray  # 0..1 float32 sRGB RGB, already fitted to the size budget
    inverse_depth: np.ndarray  # 0..1, near at 1.0 — see contract.to_hardware_depth
    #: The same image in linear light, which is what DLSS is fed. Above 1.0 for
    #: an HDR source. Pre-computed here rather than in convert() because the
    #: sRGB decode is the most expensive per-pixel step in the pipeline and
    #: nothing the user tunes afterwards changes it.
    linear: np.ndarray | None = None
    hdr: bool = False
    white: float = 1.0


def prepare(
    image_path: str | Path,
    settings: AppSettings,
    engine: DepthEngine,
    progress: Progress | None = None,
) -> Prepared:
    """Load an image and estimate its depth. No DLSS runtime needed."""

    def say(message: str) -> None:
        if progress:
            progress(message)

    say("Loading image…")
    loaded = contract.load_source(image_path)
    # Fit the linear copy, then re-derive the display copy from it. Resizing
    # sRGB-encoded values averages the wrong quantity and darkens edges;
    # resizing the linear one and tone mapping afterwards does not.
    linear = contract.fit_to_budget(loaded.linear, settings.evaluation.max_edge)
    if loaded.hdr:
        source = hdr.tonemap(linear, loaded.white)
    else:
        source = np.clip(contract.linear_to_srgb(linear), 0.0, 1.0)

    engine.load(settings.depth.model_id, progress=progress)
    # Depth Anything wants an ordinary 8-bit picture. The tone mapped copy is
    # exactly that, and gives the model the same scene an SDR capture would.
    inverse_depth = engine.infer(
        (np.clip(source, 0.0, 1.0) * 255).astype(np.uint8),
        progress=progress,
        input_size=settings.depth.input_size,
        tiled=settings.depth.tiled,
    )
    return Prepared(
        source=source,
        inverse_depth=inverse_depth,
        linear=linear,
        hdr=loaded.hdr,
        white=loaded.white,
    )


# Boost and Ultra both crispen the enlarged source before DLSS, so DLAA has a
# sharper starting point to anti-alias rather than a bicubic-soft one. With the
# level and slider retired (Boost is automatic now), these are fixed: a moderate
# unsharp that measurably helps brick/mesh without haloing high-contrast edges.
BOOST_CRISPEN_AMOUNT = 1.5
BOOST_CRISPEN_RADIUS = 2.0

#: Below this enlargement Boost is not worth the cost — the image is already
#: near the single-evaluation ceiling — so Boost quietly runs at native size and
#: says so, rather than paying for a 5% supersample.
BOOST_MIN_FACTOR = 1.15

#: Boost supersamples then shrinks back to native, and the area-average shrink
#: softens the result a little. A gentle unsharp on the finished image restores
#: the bite without haloing — deliberately light. (Ultra keeps its full-res
#: output and never shrinks, so it does not need this.)
BOOST_POST_SHARPEN_AMOUNT = 0.4
BOOST_POST_SHARPEN_RADIUS = 1.2

#: Ultra sharpens each tile right after DLSS, before the merge — the neural pass
#: softens, and doing it per tile gets the "sharpen it in Photoshop afterwards"
#: look without ever touching the full 30k image as one array. A touch stronger
#: than Boost's since Ultra keeps full resolution; still gentle. The overlap
#: feather hides any per-tile difference at the seams.
ULTRA_TILE_SHARPEN_AMOUNT = 0.55
ULTRA_TILE_SHARPEN_RADIUS = 1.0


def _free_vram_bytes() -> int | None:
    """Free VRAM for the auto sizing, or None when there is no NVIDIA query."""
    info = hardware.query_nvidia_vram()
    return info.free_bytes if info is not None else None


def _evaluate_linear(
    harness_exe: Path,
    linear: np.ndarray,
    inverse_depth: np.ndarray,
    *,
    settings: AppSettings,
    scratch: Path,
    out_path: Path,
    colour_path: Path,
    label: str,
    progress: Progress | None,
) -> np.ndarray:
    """Run one DLSS evaluation over a linear image + its depth; return the result.

    The single evaluation shared by every Detail mode: Off and Boost run it once
    on the whole (native or supersampled) image, Ultra runs it once per tile.
    ``label`` names the working size in the runtime-limit message so a tile
    failure and a whole-image failure read differently.
    """
    height, width = linear.shape[:2]
    plan = contract.build(
        linear,
        inverse_depth,
        depth_contrast=settings.depth.contrast,
        frames=settings.evaluation.frames,
        jitter=settings.evaluation.jitter,
        already_linear=True,
    )
    plane_paths = contract.write_planes(plan, scratch)

    def write_colour(path: Path, offset: tuple[float, float]) -> None:
        shifted = contract.shift_subpixel(linear, offset[0], offset[1])
        plane = np.empty((height, width, 4), np.float16)
        plane[..., :3] = shifted.astype(np.float16)
        plane[..., 3] = np.float16(1.0)
        plane.tofile(path)

    try:
        evaluator.run_frames(
            harness_exe,
            width=width,
            height=height,
            depth_path=plane_paths["depth"],
            motion_path=plane_paths["motion"],
            colour_path=colour_path,
            out_path=out_path,
            neural=settings.neural,
            jitter=plan.jitter,
            write_colour=write_colour,
            progress=progress,
        )
    except evaluator.HarnessError as error:
        # Current DLSS builds can reject a feature above their supported working
        # resolution with InvalidParameter even when D3D12 and VRAM both allow
        # the textures. Name the actual attempted size; a future runtime with a
        # higher limit can try the same request unchanged.
        if "CREATE_DLSS" in str(error) and "InvalidParameter" in str(error):
            raise RuntimeError(
                f"DLSS rejected the {width}×{height} {label} working size even "
                "though it passed the VRAM and D3D12 checks. This is a runtime "
                "feature limit, not out-of-memory (reference testing succeeds at "
                "7680 px per side and rejects 10240). Reduce Max size, or use "
                "Boost instead of Ultra."
            ) from error
        raise
    return contract.read_output(out_path, width, height)


#: The SR engine is cached across conversions: creating the DirectML session and
#: compiling the model graph costs ~13 s, so rebuilding it every convert was most
#: of the "hang". One engine per process, swapped only when the model changes.
_SR_ENGINE = None


def _make_upscale_engine(settings: AppSettings, say: Progress | None):
    """An SR engine when enabled and its model loads, else None (Lanczos path).

    Never raises: a missing/broken model must degrade to Lanczos, not stop a
    conversion. The engine (and its compiled session) is cached across converts.
    """
    global _SR_ENGINE
    if not settings.detail.sr_enabled:
        return None
    try:
        from . import upscale

        model = upscale.MODELS.get(settings.detail.sr_model)
        if model is None:
            raise ValueError(f"Unknown SR model {settings.detail.sr_model!r}.")
        if _SR_ENGINE is not None and _SR_ENGINE.model_key == model.key:
            return _SR_ENGINE  # reuse the already-compiled session
        # Fetch on first use if it is not bundled/downloaded yet and has a URL.
        if upscale.locate(model) is None and model.url:
            upscale.download_model(model, progress=say)
        if say:
            say(f"Preparing the AI upscaler ({model.name})… first run compiles the model.")
        engine = upscale.UpscaleEngine()
        engine.load(model.key)
        _SR_ENGINE = engine
        if say:
            say(f"AI upscaler ready: {model.name}")
        return engine
    except Exception as error:  # noqa: BLE001 - optional; degrade to Lanczos
        if say:
            say(f"AI upscaler unavailable ({error}); enlarging with Lanczos instead.")
        return None


def _enlarge_for_dlss(
    source_srgb: np.ndarray,
    inverse_depth: np.ndarray,
    size: tuple[int, int],
    *,
    engine=None,
    regraft: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Enlarge the display source (+depth) to a working size for evaluation.

    With an SR ``engine`` the enlarge is a learned upscale (reconstructs texture)
    resampled to ``size``; the SR result is returned as the regraft reference so
    DLAA-erased texture can be restored afterwards. Without one it is Lanczos plus
    a crispen (which invents nothing), and the reference is None. Returns
    ``(linear, inverse_depth, reference_srgb_or_None)``.
    """
    big_w, big_h = size
    depth = cv2.resize(inverse_depth, (big_w, big_h), interpolation=cv2.INTER_LINEAR)

    if engine is not None:
        sr = engine.upscale(np.clip(source_srgb, 0.0, 1.0).astype(np.float32))
        if sr.shape[1] != big_w or sr.shape[0] != big_h:
            # Model scale (×4) rarely equals the requested factor exactly; resize
            # to the target. Detail came from the SR pass; this only sets size.
            interp = cv2.INTER_AREA if sr.shape[1] > big_w else cv2.INTER_LANCZOS4
            sr = cv2.resize(sr, (big_w, big_h), interpolation=interp)
        sr = np.clip(sr, 0.0, 1.0).astype(np.float32)
        linear = contract.srgb_to_linear(sr)
        return linear, depth, (sr if regraft > 0.0 else None)

    big_srgb = cv2.resize(source_srgb, (big_w, big_h), interpolation=cv2.INTER_LANCZOS4)
    big_srgb = detail.sharpen(
        np.clip(big_srgb, 0.0, 1.0).astype(np.float32),
        amount=BOOST_CRISPEN_AMOUNT,
        radius=BOOST_CRISPEN_RADIUS,
    )
    linear = contract.srgb_to_linear(np.clip(big_srgb, 0.0, 1.0))
    return linear, depth, None


def _regraft(dlss_linear: np.ndarray, reference_srgb: np.ndarray | None, amount: float) -> np.ndarray:
    """Restore the SR reference's high-frequency texture onto the DLSS result.

    DLSS's neural pass anti-aliases, which on fine regular texture (brick,
    façades, mesh) reads as aliasing and smooths it away — unacceptable for
    architectural work. Because the SR tile (before DLSS) and the DLSS result are
    the same size, we graft the SR tile's real high-frequency band back on. It is
    a frequency graft of genuine detail (see detail.preserve_detail), not a
    sharpen, so it cannot halo. Done in display space, returned linear.
    """
    if reference_srgb is None or amount <= 0.0:
        return dlss_linear
    dlss_srgb = np.clip(contract.linear_to_srgb(dlss_linear), 0.0, 1.0)
    if reference_srgb.shape != dlss_srgb.shape:
        return dlss_linear
    grafted = detail.preserve_detail(
        dlss_srgb, reference_srgb, amount=float(amount), radius=detail.DEFAULT_RADIUS
    )
    return contract.srgb_to_linear(np.clip(grafted, 0.0, 1.0))


#: Frequency split for the tone transfer. Small enough that only fine detail is
#: kept from the boosted image; everything broader (lighting, contrast, colour)
#: comes from the native neural pass, so the cinematic look is not diluted.
BOOST_TONE_RADIUS = 4.0


def _transfer_tone(detail_linear: np.ndarray, tone_linear: np.ndarray, radius: float) -> np.ndarray:
    """Keep ``detail_linear``'s fine detail but take its tone from ``tone_linear``.

    The DLSS neural pass's look (contrast, saturation, relighting) weakens as the
    working resolution rises, so a boosted or tiled result comes back sharper but
    flatter than a native pass. This lays the native pass's low/mid band (the
    look) under the high-frequency detail of the boosted result, restoring the
    cinematic contrast without giving up the detail. Both are native-size linear.
    """
    if detail_linear.shape != tone_linear.shape:
        return detail_linear
    radius = max(0.5, float(radius))
    tone_low = cv2.GaussianBlur(tone_linear, (0, 0), radius)
    detail_low = cv2.GaussianBlur(detail_linear, (0, 0), radius)
    return np.maximum(tone_low + (detail_linear - detail_low), 0.0)


def _run_ultra_stream(
    source_srgb: np.ndarray,
    inverse_depth: np.ndarray,
    *,
    settings: AppSettings,
    scratch: Path,
    dest_tiff: Path,
    evaluate,
    engine=None,
    say: Progress | None = None,
) -> tuple[tuple[int, int], np.ndarray]:
    """Tiled SR→DLSS→regraft, streamed to a BigTIFF at ``dest_tiff``.

    Returns ``((full_w, full_h), native_srgb)`` — the on-disk full size and a
    native-resolution downscale for the preview and the native copy. The full
    image never exists in RAM: tiles are fed into a disk-backed
    :class:`tiling.StreamMerger` and written out as a BigTIFF strip by strip, so
    peak memory is one tile.

    ``evaluate(linear, inverse_depth, tile) -> linear`` runs the DLSS pass; it is
    injected so the whole tiling/SR/merge/stream path can be tested without a GPU.
    """
    h, w = source_srgb.shape[:2]
    factor, tile_max, overlap = tiling.auto_ultra(
        w, h,
        requested_factor=settings.detail.ultra_factor,
        ram_free=hardware.query_system_ram(),
        vram_free=_free_vram_bytes(),
        max_factor=settings.detail.ultra_max_factor,
    )
    big_w = max(w, int(round(w * factor)))
    big_h = max(h, int(round(h * factor)))
    tiles = tiling.plan_tiles(big_w, big_h, tile_max, overlap)
    regraft = settings.detail.sr_regraft if engine is not None else 0.0

    # One native-resolution DLSS pass, up front, purely for its look. The neural
    # pass's contrast/colour/relighting weakens as the working resolution rises,
    # so every big tile comes back sharp but flat. This native pass carries the
    # true cinematic tone; per tile we lay its low/mid band under the tile's fine
    # detail (see _transfer_tone), exactly as Boost does. Native is small, so this
    # is one cheap extra evaluation, and it fixes both the full image and the
    # preview downscale at once.
    if say:
        say("Ultra Detail: capturing the neural look at native size…")
    native_look = evaluate(
        np.ascontiguousarray(contract.srgb_to_linear(np.clip(source_srgb, 0.0, 1.0))),
        np.ascontiguousarray(inverse_depth),
        tiling.Tile(0, 0, w, h),
    )

    merger = tiling.StreamMerger(big_h, big_w, 3, overlap, scratch)
    # Native downscale is small, so it is merged in RAM from per-tile downscales.
    native_overlap = max(1, int(round(overlap * w / big_w)))
    native_tiles: list[tiling.Tile] = []
    native_patches: list[np.ndarray] = []
    try:
        for index, tile in enumerate(tiles, 1):
            if say:
                say(f"Ultra Detail: tile {index} of {len(tiles)} ({tile.w}×{tile.h})…")
            # The native region this big tile came from (map back by the factor).
            nx0 = min(w - 1, int(tile.x * w / big_w))
            ny0 = min(h - 1, int(tile.y * h / big_h))
            nx1 = min(w, int(round(tile.right * w / big_w)))
            ny1 = min(h, int(round(tile.bottom * h / big_h)))
            src_region = np.ascontiguousarray(source_srgb[ny0:ny1, nx0:nx1])
            depth_region = np.ascontiguousarray(inverse_depth[ny0:ny1, nx0:nx1])

            linear, depth, reference = _enlarge_for_dlss(
                src_region, depth_region, (tile.w, tile.h), engine=engine, regraft=regraft
            )
            dlss_linear = evaluate(
                np.ascontiguousarray(linear), np.ascontiguousarray(depth), tile
            )
            merged_linear = _regraft(dlss_linear, reference, regraft)
            # Sharpen each tile after DLSS (which softens), before the merge —
            # cheap per tile, and equivalent to sharpening the whole huge image.
            merged_linear = detail.sharpen(
                merged_linear, amount=ULTRA_TILE_SHARPEN_AMOUNT,
                radius=ULTRA_TILE_SHARPEN_RADIUS, preserve_range=True,
            )

            # Transfer the native pass's tone onto this tile. The tone correction
            # is a smooth low-frequency delta computed at native size (cheap): take
            # the tile's own native downscale, lay the native look's low/mid band
            # under it (Boost-style, BOOST_TONE_RADIUS), and add the difference
            # back to the full-size tile. This restores the cinematic contrast and
            # colour without touching the tile's genuinely new fine detail.
            n_tile = tiling.Tile(nx0, ny0, nx1 - nx0, ny1 - ny0)
            native_patch = cv2.resize(
                merged_linear, (n_tile.w, n_tile.h), interpolation=cv2.INTER_AREA
            )
            tone_region = np.ascontiguousarray(native_look[ny0:ny1, nx0:nx1])
            corrected_patch = _transfer_tone(native_patch, tone_region, BOOST_TONE_RADIUS)
            tone_delta = cv2.resize(
                corrected_patch - native_patch, (tile.w, tile.h),
                interpolation=cv2.INTER_LINEAR,
            )
            merged_linear = np.maximum(merged_linear + tone_delta, 0.0)
            merger.add(tile, merged_linear)

            # A native-sized downscale of this tile, for the native copy/preview.
            # Use the tone-corrected patch so the preview matches the full image.
            native_tiles.append(n_tile)
            native_patches.append(corrected_patch)

        if say:
            say("Ultra Detail: writing the full-resolution image to disk…")
        rows = (
            (y0, np.clip(contract.linear_to_srgb(block), 0.0, 1.0))
            for y0, block in merger.rows()
        )
        bigtiff.write_streaming(dest_tiff, big_h, big_w, rows, bits=16)
    finally:
        merger.close()

    native_linear = tiling.merge_tiles((h, w), native_tiles, native_patches, native_overlap)
    native_srgb = np.clip(contract.linear_to_srgb(native_linear), 0.0, 1.0).astype(np.float32)
    return (big_w, big_h), native_srgb


def depth_preview(inverse_depth: np.ndarray) -> np.ndarray:
    depth_u8 = np.round(np.clip(inverse_depth, 0.0, 1.0) * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)


def convert(
    image_path: str | Path,
    settings: AppSettings,
    engine: DepthEngine,
    progress: Progress | None = None,
    prepared: Prepared | None = None,
    grade_settings=None,
    effects_settings=None,
    luts_dir: Path | None = None,
) -> Result:
    """Run the full pipeline on one image.

    `prepared` skips loading and depth estimation when the caller already has
    them for this image and these depth settings. Nothing here validates that
    claim — the UI owns invalidating its cache when the model or tiling changes.

    `grade_settings`/`effects_settings` are used only by **Ultra**: its output is
    too large to grade/effect afterwards, so the look is baked into the source
    *before* the SR+DLSS pipeline (Result.look_baked is then True). Every other
    mode ignores them — the app applies the grade/effects live on the preview and
    at save time, exactly as before.
    """

    def say(message: str) -> None:
        if progress:
            progress(message)

    status = runtime.detect(settings.runtime_dir or None)
    if not status.ready:
        raise RuntimeError("\n".join(status.problems))
    staged = runtime.stage_runtime(status)
    assert status.harness is not None
    # Before launching, never after: the add-on reads this once at startup.
    runtime.write_config(staged, settings.neural)

    if prepared is None:
        prepared = prepare(image_path, settings, engine, progress)
    source = prepared.source
    inverse_depth = prepared.inverse_depth
    height, width = source.shape[:2]

    # Older Prepared values, and any caller building one by hand, may not carry
    # the linear copy. Deriving it is cheap next to depth estimation.
    linear = prepared.linear
    if linear is None:
        linear = contract.srgb_to_linear(np.clip(source, 0.0, 1.0))

    # Detail decides how large the neural pass runs. `source` and the native
    # depth are kept untouched for the Result, so the before/after and the depth
    # preview stay native-sized whatever the working resolution was.
    scratch = paths.scratch_dir()
    colour_path = scratch / "colour.bin"
    out_path = scratch / "out.bin"
    native_wh = (width, height)
    free_bytes = _free_vram_bytes()
    mode = settings.detail.mode
    # AI upscale is Ultra-only. In Boost the SR model reconstructs structure but
    # smooths true micro-texture (skin pores), which the shrink-back then loses,
    # so Boost stays pure Lanczos supersample + sharpen (measured better). None
    # here means the Lanczos path.
    sr_engine = _make_upscale_engine(settings, say) if mode == "ultra" else None
    regraft = settings.detail.sr_regraft if sr_engine is not None else 0.0

    if mode == "ultra":
        # Ultra Detail streams the full-resolution result to a scratch BigTIFF as
        # it merges, so the giant image never lives in RAM (the freeze). Save later
        # just moves/re-encodes that file. Only a native downscale comes back for
        # the preview.
        ultra_tiff = scratch / "ultra_full.tiff"

        # The grade/effects look cannot be applied to the giant output afterwards,
        # so bake it into the source here (cheap, native size) and let the SR+DLSS
        # pipeline carry it into the full-resolution result. The app then shows and
        # saves the result as-is (Result.look_baked), applying nothing more.
        look_baked = False
        ultra_source = source
        effects_on = effects_settings is not None and not effects_settings.is_neutral
        grade_on = grade_settings is not None and not grade_settings.is_neutral
        if grade_on or effects_on:
            say("Ultra Detail: baking the colour and effects into the source…")
            look = np.clip(source, 0.0, 1.0).astype(np.float32)
            if grade_on:
                look = grade.apply(look, grade_settings)
            if effects_on:
                look = effects.apply(look, effects_settings, luts_dir or paths.luts_dir())
            ultra_source = np.clip(look, 0.0, 1.0).astype(np.float32)
            look_baked = True

        def _eval_tile(lin: np.ndarray, dep: np.ndarray, _tile) -> np.ndarray:
            return _evaluate_linear(
                status.harness, lin, dep, settings=settings, scratch=scratch,
                out_path=out_path, colour_path=colour_path, label="Ultra tile",
                progress=progress,
            )

        (big_w, big_h), native_srgb = _run_ultra_stream(
            ultra_source, inverse_depth, settings=settings, scratch=scratch,
            dest_tiff=ultra_tiff, evaluate=_eval_tile, engine=sr_engine, say=say,
        )
        sr_note = " (AI upscaled)" if sr_engine is not None else " (Lanczos)"
        return Result(
            original=np.clip(source, 0.0, 1.0),
            enhanced=native_srgb,
            depth_preview=depth_preview(inverse_depth),
            notes=(
                f"{width}x{height} → {big_w}x{big_h} full, "
                f"{settings.evaluation.frames} DLSS passes, ultra{sr_note}"
            ),
            ultra_full_path=ultra_tiff,
            ultra_full_size=(big_w, big_h),
            look_baked=look_baked,
        )

    detail_note = ""
    if mode == "boost":
        # Boost: one enlarged evaluation, sized automatically to the largest a
        # single pass legally allows. The enlarge is SR (reconstructs detail) when
        # available, else Lanczos; the SR reference is grafted back after DLSS so
        # DLAA cannot erase texture.
        factor = tiling.auto_boost_factor(width, height, free_bytes=free_bytes)
        if factor >= BOOST_MIN_FACTOR:
            big_w, big_h = int(round(width * factor)), int(round(height * factor))
            how = "AI upscale" if sr_engine is not None else "supersample"
            say(f"Detail boost: {how} to {big_w}×{big_h}…")
            big_linear, big_depth, reference = _enlarge_for_dlss(
                source, inverse_depth, (big_w, big_h), engine=sr_engine, regraft=regraft
            )
            enhanced_big = _evaluate_linear(
                status.harness, big_linear, big_depth,
                settings=settings, scratch=scratch, out_path=out_path,
                colour_path=colour_path, label="Boost", progress=progress,
            )
            enhanced_big = _regraft(enhanced_big, reference, regraft)
            enhanced_linear = cv2.resize(enhanced_big, native_wh, interpolation=cv2.INTER_AREA)
            # Restore the bite the area-average shrink cost. Gentle, range-kept so
            # an HDR boost is not clipped.
            enhanced_linear = detail.sharpen(
                enhanced_linear, amount=BOOST_POST_SHARPEN_AMOUNT,
                radius=BOOST_POST_SHARPEN_RADIUS, preserve_range=True,
            )
            # The big pass gives detail but a diluted neural look. Run a native
            # pass for the real cinematic tone and lay it under the boost detail.
            say("Detail boost: restoring the neural look at native size…")
            native_look = _evaluate_linear(
                status.harness, linear, inverse_depth,
                settings=settings, scratch=scratch, out_path=out_path,
                colour_path=colour_path, label="native", progress=progress,
            )
            enhanced_linear = _transfer_tone(enhanced_linear, native_look, BOOST_TONE_RADIUS)
            detail_note = f", boost ×{factor:.1f}" + (" (AI)" if sr_engine is not None else "")
        else:
            # Already near the ceiling: supersampling would gain a few percent for
            # a full extra evaluation. Run native and say so rather than pretend.
            say("Boost: already near the size ceiling — running at native size…")
            enhanced_linear = _evaluate_linear(
                status.harness, linear, inverse_depth,
                settings=settings, scratch=scratch, out_path=out_path,
                colour_path=colour_path, label="native", progress=progress,
            )
            detail_note = ", boost (native — no headroom)"

    else:  # off
        say("Building the DLAA contract…")
        enhanced_linear = _evaluate_linear(
            status.harness, linear, inverse_depth,
            settings=settings, scratch=scratch, out_path=out_path,
            colour_path=colour_path, label="native", progress=progress,
        )

    width, height = native_wh
    notes = f"{width}x{height}, {settings.evaluation.frames} DLSS passes{detail_note}"
    if prepared.hdr:
        # Tone map with the source's white point, not one measured on this
        # image: the two are shown side by side under a wipe, and a different
        # mapping on each half would read as an exposure change the neural pass
        # did not make.
        return Result(
            original=source,
            enhanced=hdr.tonemap(enhanced_linear, prepared.white),
            depth_preview=depth_preview(inverse_depth),
            notes=f"{notes}, {hdr.describe(enhanced_linear)}",
            enhanced_linear=enhanced_linear,
            white=prepared.white,
        )

    return Result(
        original=np.clip(source, 0.0, 1.0),
        enhanced=np.clip(contract.linear_to_srgb(enhanced_linear), 0.0, 1.0),
        depth_preview=depth_preview(inverse_depth),
        notes=notes,
    )


@dataclass
class SequenceFrame:
    """One finished frame, handed back as the sequence runs."""

    index: int
    total: int
    source: Path
    output: Path
    image: np.ndarray  # 0..1 float RGB, graded


def hdr_output_path(
    destination: Path, stem: str, source: Path, style: str | None = None,
    fmt: str | None = None,
) -> Path:
    """Where a converted frame goes, in a format that can hold what it holds.

    Decided from the *input* extension rather than from the decoded pixels,
    because the batch has to know the output name before it loads anything -
    that is what lets it skip files it has already done.

    ``style`` (default/natural/cinematic) is written into the name when given, so
    a folder of results says which look each was made with. ``fmt`` (an extension
    without the dot, e.g. "png"/"jpg"/"tif"/"jxr") forces the output type; when
    None the type matches the source (JPEG XR for HDR, PNG otherwise).
    """
    if fmt:
        suffix = f".{fmt.lstrip('.')}"
    else:
        suffix = ".jxr" if hdr.is_hdr_source(source) else ".png"
    tag = f"_{style}" if style else ""
    return destination / f"{stem}_dlss5{tag}{suffix}"


def _finish(
    enhanced_linear: np.ndarray,
    *,
    is_hdr: bool,
    grade_settings,
    white: float,
    effects_settings=None,
    luts_dir: Path | None = None,
) -> tuple[np.ndarray, bool, np.ndarray]:
    """Grade, apply effects, and encode one result. Returns (payload, linear, preview).

    `payload` is what gets written and `linear` says which space it is in;
    `preview` is always display-referred, because the UI shows a thumbnail of
    every frame and cannot show linear light.

    Effects run after the grade, in display-referred sRGB — the space they are
    defined in. On the HDR path that means a round trip through the (range-
    preserving) sRGB transfer either side, so an HDR export keeps its highlights
    instead of having them clamped by the effect stack; see effects.apply.
    """
    active = effects_settings is not None and not effects_settings.is_neutral

    if is_hdr:
        graded = (
            enhanced_linear
            if grade_settings is None
            else grade.apply_linear(enhanced_linear, grade_settings)
        )
        if active:
            # linear -> extended sRGB -> effects (range kept) -> linear. The
            # transfer curve is monotonic above 1.0, so highlights survive the
            # round trip; effects.apply(preserve_range) never clamps them.
            srgb = contract.linear_to_srgb(graded)
            srgb = effects.apply(srgb, effects_settings, luts_dir, preserve_range=True)
            graded = contract.srgb_to_linear(srgb)
        return graded, True, hdr.tonemap(graded, white)

    enhanced = np.clip(contract.linear_to_srgb(enhanced_linear), 0.0, 1.0)
    if grade_settings is not None:
        enhanced = grade.apply(enhanced, grade_settings)
    if active:
        enhanced = effects.apply(enhanced, effects_settings, luts_dir)
    return enhanced, False, enhanced


def _load_for_evaluation(path: Path, max_edge: int):
    """Load one frame as (display sRGB, linear, hdr flag, white point)."""
    loaded = contract.load_source(path)
    linear = contract.fit_to_budget(loaded.linear, max_edge)
    if loaded.hdr:
        return hdr.tonemap(linear, loaded.white), linear, True, loaded.white
    return np.clip(contract.linear_to_srgb(linear), 0.0, 1.0), linear, False, 1.0


def convert_sequence(
    frames: list[Path],
    settings: AppSettings,
    engine: DepthEngine,
    destination: Path,
    depth_frames: list[Path] | None = None,
    invert_depth: bool = False,
    grade_settings=None,
    progress: Progress | None = None,
    should_stop: Callable[[], bool] | None = None,
):
    """Convert a whole sequence, yielding each frame as it finishes.

    One harness for the entire run. Start-up is ~3.5 s and dominates a single
    conversion, so paying it per frame would make a 200-frame sequence mostly
    idle time; here it is paid once and each frame costs only its evaluations.

    Every frame resets DLSS's temporal history. Motion vectors are zero — the
    contract says nothing moved — so carrying accumulation between two genuinely
    different frames would drag the previous image into this one wherever the
    scene changed. Consistency between frames comes from feeding identical
    settings and stable depth, not from shared history.

    `depth_frames`, when given, replaces depth estimation entirely with the
    renderer's own depth pass. That is the reason this mode can be temporally
    stable: an estimated depth map wobbles slightly frame to frame and the
    neural pass follows it, while a rendered depth pass does not move at all.
    """

    def say(message: str) -> None:
        if progress:
            progress(message)

    if not frames:
        return
    if depth_frames and len(depth_frames) != len(frames):
        raise ValueError(
            f"{len(frames)} image frames but {len(depth_frames)} depth frames. "
            "They have to correspond one to one."
        )

    status = runtime.detect(settings.runtime_dir or None)
    if not status.ready:
        raise RuntimeError("\n".join(status.problems))
    staged = runtime.stage_runtime(status)
    assert status.harness is not None
    runtime.write_config(staged, settings.neural)

    destination.mkdir(parents=True, exist_ok=True)
    scratch = paths.scratch_dir()
    luts_dir = paths.luts_dir()  # resolved once; _finish uses it only if a LUT is on
    colour_path = scratch / "seq_colour.bin"
    depth_path = scratch / "seq_depth.bin"
    motion_path = scratch / "seq_motion.bin"
    out_path = scratch / "seq_out.bin"

    # The first frame fixes the size for the whole run: one harness means one
    # set of NGX buffers, and DLSS cannot be handed a different resolution
    # halfway through without recreating the feature.
    say("Loading the first frame…")
    first = contract.fit_to_budget(contract.load_image(frames[0]), settings.evaluation.max_edge)
    height, width = first.shape[:2]

    if depth_frames is None:
        engine.load(settings.depth.model_id, progress=progress)

    np.zeros((height, width, 2), np.float16).tofile(motion_path)
    # The harness reads both planes at launch, before any DEPTH command can
    # arrive, so a placeholder has to exist. It is overwritten for real by the
    # first frame of the loop below.
    np.zeros((height, width), np.float32).tofile(depth_path)
    offsets = contract.jitter_sequence(settings.evaluation.frames)
    if not settings.evaluation.jitter:
        offsets = [(0.0, 0.0)] * len(offsets)

    with evaluator.Harness(
        status.harness,
        width=width,
        height=height,
        depth_path=depth_path,
        motion_path=motion_path,
        neural=settings.neural,
        frames=settings.evaluation.frames,
        use_shmem=True,  # throughput path; falls back to files on an old harness
    ) as harness:
        # One colour buffer for the whole sequence — the mapping itself when
        # shared memory is live — rewritten in place each pass.
        colour_plane = harness.colour_buffer((height, width, 4))
        colour_plane[..., 3] = np.float16(1.0)
        for index, frame_path in enumerate(frames):
            if should_stop is not None and should_stop():
                say("Stopped.")
                return
            say(f"Frame {index + 1} of {len(frames)} — {frame_path.name}")

            source, linear, is_hdr, white = _load_for_evaluation(
                frame_path, settings.evaluation.max_edge
            )
            if source.shape[:2] != (height, width):
                raise RuntimeError(
                    f"{frame_path.name} is {source.shape[1]}x{source.shape[0]}, but the "
                    f"sequence started at {width}x{height}. Frames must all be one size."
                )

            if depth_frames is not None:
                inverse_depth = sequence.load_depth_map(depth_frames[index], invert_depth)
                if inverse_depth.shape != (height, width):
                    inverse_depth = cv2.resize(
                        inverse_depth, (width, height), interpolation=cv2.INTER_NEAREST
                    )
            else:
                inverse_depth = engine.infer(
                    (np.clip(source, 0.0, 1.0) * 255).astype(np.uint8),
                    input_size=settings.depth.input_size,
                    tiled=settings.depth.tiled,
                )

            shaped = contract.to_hardware_depth(inverse_depth, settings.depth.contrast)
            np.ascontiguousarray(shaped).tofile(depth_path)
            harness.set_depth(depth_path)

            harness.reset_history()
            for offset in offsets:
                shifted = contract.shift_subpixel(linear, offset[0], offset[1])
                colour_plane[..., :3] = shifted.astype(np.float16)
                harness.commit_colour(colour_plane, colour_path, offset)

            harness.write(out_path)
            payload, is_linear, preview = _finish(
                contract.read_output(out_path, width, height),
                is_hdr=is_hdr,
                grade_settings=grade_settings,
                white=white,
                effects_settings=settings.effects,
                luts_dir=luts_dir,
            )
            output = hdr_output_path(
                destination, frame_path.stem, frame_path, style_slug(settings.neural.style)
            )
            save_image(payload, output, linear=is_linear)
            yield SequenceFrame(index, len(frames), frame_path, output, preview)


@dataclass
class BatchItem:
    """One file's outcome, handed back as the batch runs."""

    index: int
    total: int
    source: Path
    output: Path | None  # None when skipped
    skipped: bool = False
    error: str = ""
    #: The finished frame, display-referred, for the dialog to show. Handed
    #: over rather than re-read from disk: it is already in memory here, and a
    #: batch of 8K files would otherwise pay a decode per item purely to draw a
    #: thumbnail.
    image: np.ndarray | None = None


@dataclass
class VideoProgress:
    """One video conversion tick, for the UI to render."""

    index: int          # frames done
    total: int          # frames planned, 0 if unknown
    preview: np.ndarray  # the frame just converted, display-referred
    stage: str = "converting"  # converting | encoding | muxing | done


def convert_video(
    source: Path,
    destination: Path,
    settings: AppSettings,
    engine: DepthEngine,
    codec_key: str = "h264",
    start: int = 0,
    limit: int | None = None,
    estimate_depth: bool = False,
    grade_settings=None,
    progress: Progress | None = None,
    should_stop: Callable[[], bool] | None = None,
):
    """Convert a video frame by frame and mux the source audio back in.

    Each frame is an independent single-image conversion - DLSS history is reset
    every frame, exactly as in convert_sequence - so nothing smears between two
    genuinely different frames. That independence is why the result is stable.

    ``estimate_depth`` runs Depth Anything per frame. It is off by default
    because the depth plane is not read by DLSS on a still frame (there is no
    motion to reproject through), so estimating it changes nothing in the output
    and is by far the slowest step. It is offered only for parity with the photo
    path, and labelled honestly in the UI.

    Yields VideoProgress as each frame lands, then once more for muxing.
    """
    from . import video

    def say(message: str) -> None:
        if progress:
            progress(message)

    if not video.is_available():
        raise RuntimeError(
            "Video support needs the PyAV component, which has not been "
            "downloaded yet."
        )
    codec = video.CODECS_BY_KEY.get(codec_key)
    if codec is None:
        raise ValueError(f"Unknown codec {codec_key!r}.")

    info = video.probe(source)
    total = info.frames
    if limit is not None:
        total = min(limit, total - start) if total else limit

    status = runtime.detect(settings.runtime_dir or None)
    if not status.ready:
        raise RuntimeError("\n".join(status.problems))
    staged = runtime.stage_runtime(status)
    assert status.harness is not None
    runtime.write_config(staged, settings.neural)

    if estimate_depth:
        engine.load(settings.depth.model_id, progress=progress)
    offsets = contract.jitter_sequence(settings.evaluation.frames)
    if not settings.evaluation.jitter:
        offsets = [(0.0, 0.0)] * len(offsets)

    scratch = paths.scratch_dir()
    luts_dir = paths.luts_dir()
    colour_path = scratch / "vid_colour.bin"
    depth_path = scratch / "vid_depth.bin"
    motion_path = scratch / "vid_motion.bin"
    out_path = scratch / "vid_out.bin"
    video_only = scratch / f"vid_video_only{codec.suffix}"

    harness: evaluator.Harness | None = None
    writer: video.VideoWriter | None = None
    size: tuple[int, int] | None = None
    done = 0

    try:
        for source_rgb in video.frames(source, start=start, limit=limit,
                                        should_stop=should_stop):
            if should_stop is not None and should_stop():
                say("Stopped.")
                return
            fitted = contract.fit_to_budget(source_rgb, settings.evaluation.max_edge)
            height, width = fitted.shape[:2]

            if harness is None:
                # The first frame fixes the size for the whole clip: one harness,
                # one set of NGX buffers, and one output stream.
                np.zeros((height, width, 2), np.float16).tofile(motion_path)
                np.zeros((height, width), np.float32).tofile(depth_path)
                harness = evaluator.Harness(
                    status.harness, width=width, height=height,
                    depth_path=depth_path, motion_path=motion_path,
                    neural=settings.neural, frames=settings.evaluation.frames,
                    # Video is the throughput case: shared memory skips a 66 MB
                    # file write and read on every pass of every frame. Falls
                    # back to the file path automatically on an older harness.
                    use_shmem=True,
                )
                harness.__enter__()
                size = (width, height)
                writer = video.VideoWriter(video_only, codec, info.fps, size)
                # The colour plane is 66 MB at 4K and identical in shape every
                # pass of every frame, so it is allocated once and rewritten in
                # place. With shared memory this buffer *is* the mapping the
                # harness reads, so writing into it is the whole transport; the
                # alpha row, always 1.0, is filled here and never touched again.
                colour_plane = harness.colour_buffer((height, width, 4))
                colour_plane[..., 3] = np.float16(1.0)
            elif (width, height) != size:
                # A source whose frames change size mid-stream is degenerate;
                # refuse rather than silently rescaling to the first frame.
                raise RuntimeError(
                    f"Frame {done + 1} is {width}x{height}, but the video "
                    f"started at {size[0]}x{size[1]}."
                )

            if estimate_depth:
                inverse = engine.infer(
                    (np.clip(fitted, 0, 1) * 255).astype(np.uint8),
                    input_size=settings.depth.input_size, tiled=settings.depth.tiled,
                )
                shaped = contract.to_hardware_depth(inverse, settings.depth.contrast)
                np.ascontiguousarray(shaped).tofile(depth_path)
                harness.set_depth(depth_path)

            linear = contract.srgb_to_linear(np.clip(fitted, 0, 1))
            harness.reset_history()
            for offset in offsets:
                shifted = contract.shift_subpixel(linear, offset[0], offset[1])
                # Reused buffer: only the colour channels change per pass; alpha
                # was set to 1.0 when it was allocated. commit_colour sends it
                # through shared memory or the file, whichever is live.
                colour_plane[..., :3] = shifted.astype(np.float16)
                harness.commit_colour(colour_plane, colour_path, offset)
            harness.write(out_path)

            enhanced = np.clip(
                contract.linear_to_srgb(contract.read_output(out_path, width, height)),
                0.0, 1.0,
            )
            if grade_settings is not None:
                enhanced = grade.apply(enhanced, grade_settings)
            # Detail (Boost/Ultra) is a single-image, supersampled operation and
            # is not applied per video frame — it would multiply an already
            # frame-by-frame conversion by the tile count. Video keeps the neural
            # pass at native size; the grade and effects still apply.
            # Video frames are display-referred (the codecs are 8-bit SDR), so
            # the plain sRGB effect path — the same one the photo save uses.
            enhanced = effects.apply(enhanced, settings.effects, luts_dir)
            assert writer is not None
            writer.write(enhanced)
            done += 1
            say(f"pass {done} of {total}" if total else f"frame {done}")
            yield VideoProgress(done, total, enhanced, "converting")

        if writer is not None:
            writer.close()
            writer = None
        if harness is not None:
            harness.__exit__(None, None, None)
            harness = None

        if done == 0:
            raise RuntimeError("No frames were converted.")

        say("Adding audio…")
        # The audio window must match the frames actually converted, or a
        # trimmed clip plays its picture and then sits on black over the rest of
        # the full soundtrack. `start` and `done` are in frames; the source fps
        # turns them into the seconds mux_audio wants.
        audio_start = start / info.fps if info.fps else 0.0
        audio_duration = None if (start == 0 and limit is None) else done / (info.fps or 1.0)
        yield VideoProgress(done, total, np.zeros((1, 1, 3), np.float32), "muxing")
        had_audio = video.mux_audio(
            Path(source), video_only, destination,
            start=audio_start, duration=audio_duration,
        )
        yield VideoProgress(done, total, np.zeros((1, 1, 3), np.float32),
                            "done" if had_audio else "done-no-audio")
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass
        if harness is not None:
            harness.__exit__(None, None, None)
        video_only.unlink(missing_ok=True)


def convert_batch(
    images: list[Path],
    settings: AppSettings,
    engine: DepthEngine,
    destination: Path,
    grade_settings=None,
    skip_existing: bool = True,
    save_format: str | None = None,
    progress: Progress | None = None,
    should_stop: Callable[[], bool] | None = None,
):
    """Apply the current settings to a folder of unrelated images.

    Distinct from `convert_sequence`, and deliberately so. A sequence is one
    shot: same size throughout, a shared depth pass, and frames that have to
    look consistent with each other. A batch is a pile of images that happen to
    want the same treatment, so the sizes vary and each is judged on its own.

    The harness is kept alive across files and restarted only when the frame
    size changes. A folder of renders straight out of one scene is all one size,
    which is the common case and gets the whole batch on a single ~3.5 s
    start-up; a mixed folder pays it once per run of matching sizes.

    One file failing does not stop the batch. An unreadable image in the middle
    of two hundred should cost that file, not the afternoon — the failure is
    reported on the item and the run carries on.
    """

    def say(message: str) -> None:
        if progress:
            progress(message)

    if not images:
        return

    status = runtime.detect(settings.runtime_dir or None)
    if not status.ready:
        raise RuntimeError("\n".join(status.problems))
    staged = runtime.stage_runtime(status)
    assert status.harness is not None
    runtime.write_config(staged, settings.neural)

    destination.mkdir(parents=True, exist_ok=True)
    scratch = paths.scratch_dir()
    luts_dir = paths.luts_dir()
    colour_path = scratch / "batch_colour.bin"
    depth_path = scratch / "batch_depth.bin"
    motion_path = scratch / "batch_motion.bin"
    out_path = scratch / "batch_out.bin"

    engine.load(settings.depth.model_id, progress=progress)
    offsets = contract.jitter_sequence(settings.evaluation.frames)
    if not settings.evaluation.jitter:
        offsets = [(0.0, 0.0)] * len(offsets)

    harness: evaluator.Harness | None = None
    harness_size: tuple[int, int] | None = None

    def close_harness() -> None:
        nonlocal harness, harness_size
        if harness is not None:
            harness.__exit__(None, None, None)
            harness = None
            harness_size = None

    try:
        for index, path in enumerate(images):
            if should_stop is not None and should_stop():
                say("Stopped.")
                return

            output = hdr_output_path(
                destination, path.stem, path, style_slug(settings.neural.style),
                fmt=save_format,
            )
            if skip_existing and output.exists():
                yield BatchItem(index, len(images), path, output, skipped=True)
                continue

            say(f"{index + 1} of {len(images)} — {path.name}")
            try:
                source, linear, is_hdr, white = _load_for_evaluation(
                    path, settings.evaluation.max_edge
                )
                height, width = source.shape[:2]

                if harness is None or harness_size != (width, height):
                    close_harness()
                    np.zeros((height, width, 2), np.float16).tofile(motion_path)
                    np.zeros((height, width), np.float32).tofile(depth_path)
                    harness = evaluator.Harness(
                        status.harness,
                        width=width,
                        height=height,
                        depth_path=depth_path,
                        motion_path=motion_path,
                        neural=settings.neural,
                        frames=settings.evaluation.frames,
                        use_shmem=True,  # falls back to files on an old harness
                    )
                    harness.__enter__()
                    harness_size = (width, height)
                    # Re-fetched with every (re)created harness, since the buffer
                    # is sized to the frame and the mapping changes with it.
                    colour_plane = harness.colour_buffer((height, width, 4))
                    colour_plane[..., 3] = np.float16(1.0)

                inverse_depth = engine.infer(
                    (np.clip(source, 0.0, 1.0) * 255).astype(np.uint8),
                    input_size=settings.depth.input_size,
                    tiled=settings.depth.tiled,
                )
                shaped = contract.to_hardware_depth(inverse_depth, settings.depth.contrast)
                np.ascontiguousarray(shaped).tofile(depth_path)
                harness.set_depth(depth_path)

                harness.reset_history()
                for offset in offsets:
                    shifted = contract.shift_subpixel(linear, offset[0], offset[1])
                    colour_plane[..., :3] = shifted.astype(np.float16)
                    harness.commit_colour(colour_plane, colour_path, offset)

                harness.write(out_path)
                payload, is_linear, preview = _finish(
                    contract.read_output(out_path, width, height),
                    is_hdr=is_hdr,
                    grade_settings=grade_settings,
                    white=white,
                    effects_settings=settings.effects,
                    luts_dir=luts_dir,
                )
                save_image(payload, output, linear=is_linear)
                yield BatchItem(index, len(images), path, output, image=preview)

            except Exception as error:  # noqa: BLE001 - one bad file, not the batch
                # The harness may be in an unknown state after a failure, so
                # drop it; the next file starts a clean one.
                close_harness()
                yield BatchItem(
                    index, len(images), path, None, error=f"{type(error).__name__}: {error}"
                )
    finally:
        close_harness()


def list_images(folder: Path, recursive: bool = False) -> list[Path]:
    """Every image in `folder`, sorted.

    Suffixes come from `sequence`, not from the widgets module: this file has to
    stay importable without Qt so the pipeline can be driven headlessly.
    """
    walker = folder.rglob("*") if recursive else folder.glob("*")
    found = [
        p for p in walker if p.is_file() and p.suffix.lower() in sequence.SEQUENCE_SUFFIXES
    ]
    return sorted(found)


def write_video(images: list[Path], destination: Path, fps: float) -> Path:
    """Encode finished frames to an MP4.

    mp4v rather than H.264: OpenCV's shipped builds carry no H.264 encoder for
    licensing reasons, so asking for one silently produces an empty file. mp4v
    is larger at the same quality but it plays everywhere, and the PNG sequence
    is written regardless, so anyone who wants H.264 has the frames to encode.
    """
    if not images:
        raise ValueError("No frames to encode.")
    first = imaging.imread(images[0], cv2.IMREAD_UNCHANGED)
    if first is None:
        raise OSError(f"Could not read {images[0]}")
    height, width = first.shape[:2]

    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise OSError(f"Could not open {destination.name} for writing.")
    try:
        for path in images:
            frame = imaging.imread(path, cv2.IMREAD_UNCHANGED)
            if frame is None:
                continue
            if frame.dtype == np.uint16:
                frame = (frame // 257).astype(np.uint8)
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            writer.write(frame[:, :, :3])
    finally:
        writer.release()
    return destination


def save_image(image_rgb: np.ndarray, path: str | Path, *, linear: bool = False) -> None:
    """Write an image, choosing bit depth and encoding from the extension.

    `image_rgb` is 0..1 sRGB float by default. With ``linear=True`` it is
    scene-referred linear light and may exceed 1.0 — that is the form an HDR
    result arrives in, and it is preserved for the formats that can hold it and
    tone mapped for the ones that cannot.

    16-bit for PNG and TIFF because the neural pass genuinely widens tonal
    range in skin and shadows, and 8 bits puts visible banding into exactly the
    gradients this tool exists to improve.
    """
    target = Path(path)
    suffix = target.suffix.lower()

    if suffix in wic.SUFFIXES:
        # JPEG XR is stored in linear scRGB, so an SDR image has to be decoded
        # into that space rather than written as-is.
        wic.write(target, image_rgb if linear else contract.srgb_to_linear(
            np.clip(image_rgb, 0.0, 1.0)
        ))
        return

    if suffix in {".exr", ".hdr"} and linear:
        # The one path where values above 1.0 survive into an OpenCV format.
        # Written linear, which is what both formats mean by convention.
        data = np.maximum(image_rgb, 0.0).astype(np.float32)
        if not imaging.imwrite(target, data[:, :, ::-1]):
            raise OSError(f"Could not write {target}")
        return

    # Everything below is display-referred and bounded. An HDR image reaching
    # here is being asked for in a format that cannot hold it, so it is tone
    # mapped rather than clipped - clipping is what turns a bright sky white.
    rgb = hdr.tonemap(image_rgb) if linear else np.clip(image_rgb, 0.0, 1.0)
    if suffix in {".png", ".tif", ".tiff"}:
        data = np.round(rgb * 65535.0).astype(np.uint16)
    elif suffix in {".exr", ".hdr"}:
        data = rgb.astype(np.float32)
    else:
        data = np.round(rgb * 255.0).astype(np.uint8)
    if not imaging.imwrite(target, data[:, :, ::-1]):
        raise OSError(f"Could not write {target}")


def main() -> None:
    """Headless entry point, mostly for bring-up and batch scripting."""
    import argparse

    parser = argparse.ArgumentParser(description="Run DLSS 5 over a still image.")
    parser.add_argument("input")
    # Optional: a release build has an output folder of its own, and making the
    # user name a destination for a batch of conversions is friction with no
    # payoff. An explicit path still wins.
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Destination image. Defaults to the output folder beside the app.",
    )
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--intensity", type=float, default=None)
    parser.add_argument("--skin", type=float, default=None)
    parser.add_argument("--tiled-depth", action="store_true")
    parser.add_argument("--runtime-dir", default=None)
    args = parser.parse_args()

    settings = AppSettings.load(paths.settings_path())
    if args.frames is not None:
        settings.evaluation.frames = args.frames
    if args.intensity is not None:
        settings.neural.intensity = args.intensity
    if args.skin is not None:
        settings.neural.skin = args.skin
    if args.tiled_depth:
        settings.depth.tiled = True
    if args.runtime_dir:
        settings.runtime_dir = args.runtime_dir

    output = (
        Path(args.output) if args.output
        else _default_output(args.input, style_slug(settings.neural.style))
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    result = convert(args.input, settings, OnnxDepthEngine(), progress=print)
    # Ultra saves only the full-resolution BigTIFF (a downscaled copy loses
    # detail): move the streamed scratch file next to the output.
    if result.has_full and result.ultra_full_path is not None:
        import shutil

        fw, fh = result.ultra_full_size or (0, 0)
        kk = f"{round(max(fw, fh) / 1000)}K"
        super_out = output.with_name(f"{output.stem}_ultra_{kk}.tiff")
        shutil.move(str(result.ultra_full_path), str(super_out))
        print(f"Wrote {super_out} (Ultra Detail {fw}x{fh})")
    else:
        save_image(result.enhanced, output)
        print(f"Wrote {output} ({result.notes})")


def _default_output(input_path: str | Path, style: str = "") -> Path:
    """``output/<name>_dlss5_<style>.png``, without overwriting an earlier run."""
    stem = Path(input_path).stem
    folder = paths.output_dir()
    tag = f"_{style}" if style else ""
    candidate = folder / f"{stem}_dlss5{tag}.png"
    index = 2
    while candidate.exists():
        candidate = folder / f"{stem}_dlss5{tag}_{index}.png"
        index += 1
    return candidate


if __name__ == "__main__":
    main()
