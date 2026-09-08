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
    contract,
    detail,
    effects,
    evaluator,
    grade,
    hardware,
    hdr,
    paths,
    runtime,
    sequence,
    wic,
)
from .depth_engine import DepthEngine
from .onnx_depth import OnnxDepthEngine
from .settings import (
    D3D12_MAX_TEXTURE_DIMENSION,
    DETAIL_BOOST_FACTORS,
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

    @property
    def hdr(self) -> bool:
        return self.enhanced_linear is not None


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


# D3D12_MAX_TEXTURE_DIMENSION is imported from settings at the top of this file:
# it is defined there so the sidebar's Boost guard and this conversion-time
# check share one number and cannot drift apart. (An API limit on a 2D texture
# side, independent of VRAM capacity.)

#: Conservative working-set estimate for the four harness textures, NGX feature
#: state and driver overhead. This is a preflight, not an allocator: D3D12 still
#: makes the authoritative decision. The fixed part reflects measured NGX/add-on
#: startup cost; the per-pixel part is deliberately above the harness's visible
#: 24 B/px texture floor so hidden feature resources have room.
_BOOST_FIXED_VRAM = int(1.25 * 1024**3)
_BOOST_VRAM_PER_PIXEL = 32
_BOOST_USABLE_FREE_FRACTION = 0.90


def _boost_target(factor: int, width: int, height: int) -> tuple[int, int]:
    """Return the requested Boost dimensions, refusing only hard API limits."""
    factor = max(1, int(factor))
    target = (width * factor, height * factor)
    if max(target) > D3D12_MAX_TEXTURE_DIMENSION:
        largest = max(
            candidate for candidate in (1, *DETAIL_BOOST_FACTORS)
            if width * candidate <= D3D12_MAX_TEXTURE_DIMENSION
            and height * candidate <= D3D12_MAX_TEXTURE_DIMENSION
        )
        raise RuntimeError(
            f"Boost {factor}× would process at {target[0]}×{target[1]}, but "
            f"D3D12 textures stop at {D3D12_MAX_TEXTURE_DIMENSION} pixels per "
            f"side. Use {largest}× or lower, or reduce Max size first."
        )
    return target


def _boost_vram_estimate(width: int, height: int) -> int:
    """Estimated bytes needed by the native harness at one working size."""
    return _BOOST_FIXED_VRAM + width * height * _BOOST_VRAM_PER_PIXEL


def _preflight_boost_vram(width: int, height: int, say: Progress | None = None) -> None:
    """Refuse a likely OOM from current free VRAM; unknown hardware may try."""
    info = hardware.query_nvidia_vram()
    if info is None:
        if say:
            say("VRAM availability unavailable — letting D3D12 decide…")
        return
    estimated = _boost_vram_estimate(width, height)
    usable = int(info.free_bytes * _BOOST_USABLE_FREE_FRACTION)
    if say:
        say(
            f"VRAM preflight: about {estimated / 1024**3:.1f} GB needed, "
            f"{usable / 1024**3:.1f} GB currently usable…"
        )
    if estimated > usable:
        raise RuntimeError(
            f"Boost at {width}×{height} is estimated to need about "
            f"{estimated / 1024**3:.1f} GB of VRAM, but {info.name} has "
            f"{info.free_bytes / 1024**3:.1f} GB free right now. Choose a lower "
            "Boost factor or Max size, or close other GPU applications."
        )


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
) -> Result:
    """Run the full pipeline on one image.

    `prepared` skips loading and depth estimation when the caller already has
    them for this image and these depth settings. Nothing here validates that
    claim — the UI owns invalidating its cache when the model or tiling changes.
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
    runtime.write_addon_config(staged, settings.neural)

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

    # Boost: supersample the input so DLAA's fixed-size softening covers far less
    # of each real detail, then deliver at the native size. Proven to keep brick
    # and mesh crisp on renders. The crispen is done in display space (where an
    # unsharp mask is defined) and taken back to linear for DLSS; depth rides
    # along at the same scale. `source` and the native depth are kept for the
    # Result, so the before/after and the depth mask stay native-sized.
    boost_factor = 1
    target_wh = (width, height)
    native_inverse_depth = inverse_depth
    if settings.detail.mode == "boost":
        boost_factor = max(1, int(settings.detail.supersample))
        if boost_factor > 1:
            say(f"Detail boost: supersampling ×{boost_factor}…")
            big_w, big_h = _boost_target(boost_factor, width, height)
            _preflight_boost_vram(big_w, big_h, say)
            big_srgb = cv2.resize(source, (big_w, big_h), interpolation=cv2.INTER_LANCZOS4)
            big_srgb = detail.sharpen(
                np.clip(big_srgb, 0.0, 1.0).astype(np.float32),
                amount=settings.detail.amount * 2.0,
                radius=settings.detail.radius,
            )
            linear = contract.srgb_to_linear(np.clip(big_srgb, 0.0, 1.0))
            inverse_depth = cv2.resize(
                inverse_depth, (big_w, big_h), interpolation=cv2.INTER_LINEAR
            )
            height, width = big_h, big_w

    say("Building the DLAA contract…")
    plan = contract.build(
        linear,
        inverse_depth,
        depth_contrast=settings.depth.contrast,
        frames=settings.evaluation.frames,
        jitter=settings.evaluation.jitter,
        already_linear=True,
    )
    scratch = paths.scratch_dir()
    plane_paths = contract.write_planes(plan, scratch)
    colour_path = plane_paths["colour"]
    out_path = scratch / "out.bin"

    def write_colour(path: Path, offset: tuple[float, float]) -> None:
        shifted = contract.shift_subpixel(linear, offset[0], offset[1])
        plane = np.empty((height, width, 4), np.float16)
        plane[..., :3] = shifted.astype(np.float16)
        plane[..., 3] = np.float16(1.0)
        plane.tofile(path)

    try:
        evaluator.run_frames(
            status.harness,
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
        # the textures. Do not silently reduce Boost; name the actual attempted
        # size and let a future runtime with a higher limit try the same request.
        if (
            boost_factor > 1
            and "CREATE_DLSS" in str(error)
            and "InvalidParameter" in str(error)
        ):
            raise RuntimeError(
                f"DLSS rejected the requested {width}×{height} Boost working "
                "size even though it passed the VRAM and D3D12 checks. This "
                "is a runtime feature limit rather than an out-of-memory error "
                "(reference testing succeeds at 7680 pixels and rejects 10240). "
                "Choose a lower Boost factor or Max size. The app did not "
                "silently substitute a smaller factor."
            ) from error
        raise

    say("Encoding…")
    enhanced_linear = contract.read_output(out_path, width, height)

    if boost_factor > 1:
        # Concentrate the supersampled result back to the native size. Area
        # averaging in linear light is the clean downsample — this is the step
        # that turns "processed at 4x" into crisp native detail.
        enhanced_linear = cv2.resize(enhanced_linear, target_wh, interpolation=cv2.INTER_AREA)
        width, height = target_wh
        inverse_depth = native_inverse_depth

    boost_note = f", boost ×{boost_factor}" if boost_factor > 1 else ""
    notes = f"{width}x{height}, {settings.evaluation.frames} DLSS passes{boost_note}"
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
    destination: Path, stem: str, source: Path, style: str | None = None
) -> Path:
    """Where a converted frame goes, in a format that can hold what it holds.

    Decided from the *input* extension rather than from the decoded pixels,
    because the batch has to know the output name before it loads anything -
    that is what lets it skip files it has already done.

    ``style`` (default/natural/cinematic) is written into the name when given, so
    a folder of results says which look each was made with.
    """
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
    detail_settings=None,
    source_srgb: np.ndarray | None = None,
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
    # Detail (Preserve) before effects, matching the app's display order, and
    # only when a same-size source is on hand to lift the real detail from.
    if (
        detail_settings is not None
        and detail_settings.mode == "preserve"
        and source_srgb is not None
        and source_srgb.shape == enhanced.shape
    ):
        enhanced = detail.preserve_detail(
            enhanced, source_srgb, amount=detail_settings.amount, radius=detail_settings.radius
        )
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
    runtime.write_addon_config(staged, settings.neural)

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
                detail_settings=settings.detail,
                source_srgb=source,
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
    runtime.write_addon_config(staged, settings.neural)

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
            # Detail (Preserve) before effects, lifting the fine texture back
            # from this frame's own source (fitted is display sRGB at this size).
            if settings.detail.mode == "preserve" and fitted.shape == enhanced.shape:
                enhanced = detail.preserve_detail(
                    enhanced, np.clip(fitted, 0.0, 1.0).astype(np.float32),
                    amount=settings.detail.amount, radius=settings.detail.radius,
                )
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
    runtime.write_addon_config(staged, settings.neural)

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
                destination, path.stem, path, style_slug(settings.neural.style)
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
                    detail_settings=settings.detail,
                    source_srgb=source,
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
    first = cv2.imread(str(images[0]), cv2.IMREAD_UNCHANGED)
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
            frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
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
        if not cv2.imwrite(str(target), data[:, :, ::-1]):
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
    if not cv2.imwrite(str(target), data[:, :, ::-1]):
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
