"""Convert a video, one frame at a time, and carry its audio across.

A video is not treated as a video here. It is a stack of stills: each frame is
decoded, run through exactly the same single-image DLSS path as the photo tab,
and written back in order. DLSS's temporal history is reset every frame, so
nothing from one frame bleeds into the next - which is what keeps the result
stable rather than smeared, because the "motion" between two real frames is not
motion DLSS should try to reproject.

Audio is copied, not touched. The source's audio stream is muxed into the output
unchanged, so a converted clip still has its sound and stays in sync.

PyAV rather than OpenCV for the container work: OpenCV decodes frames but cannot
read or write an audio stream at all, and its encoder list on Windows is
whatever its prebuilt FFmpeg happened to include (no H.264). PyAV bundles a full
FFmpeg, which is where H.264, H.265, NVENC and the AAC muxing all come from. It
is downloaded on first use of this tab, like PyTorch, rather than shipped.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

Progress = Callable[[str], None]


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int          # 0 when the container does not report a count
    has_audio: bool
    duration: float      # seconds, 0.0 when unknown

    def describe(self) -> str:
        count = f"{self.frames}" if self.frames else "?"
        audio = "with audio" if self.has_audio else "no audio"
        return (
            f"{self.width}x{self.height}, {self.fps:.3g} fps, "
            f"{count} frames, {audio}"
        )


@dataclass
class Codec:
    """One output choice: how to encode, and what editors do with it."""

    key: str
    label: str
    suffix: str
    encoder: str          # the encoder PyAV should try first
    fallback: str         # a software encoder when the first is unavailable
    pix_fmt: str
    note: str


#: Offered in the UI in this order. H.264 first, because it is the one format
#: every editor and player ingests; VP9 is last because WebM is a web-delivery
#: format that Premiere, Resolve and Final Cut do not import cleanly.
CODECS: tuple[Codec, ...] = (
    Codec("h264", "H.264 / MP4", ".mp4", "h264_nvenc", "libx264", "yuv420p",
          "Universal - every editor and player. Hardware-encoded on your GPU."),
    Codec("h265", "H.265 / MP4", ".mp4", "hevc_nvenc", "libx265", "yuv420p",
          "Smaller files, modern editors. Also hardware-encoded."),
    Codec("vp9", "VP9 / WebM", ".webm", "libvpx-vp9", "libvpx-vp9", "yuv420p",
          "For web upload. Not for editors - WebM does not import cleanly."),
)

CODECS_BY_KEY = {c.key: c for c in CODECS}

#: Extensions we will offer to open. PyAV reads far more than this; the list is
#: the common cases, kept short so the file dialog is not a wall of formats.
INPUT_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"})


def is_available() -> bool:
    """Whether PyAV can be imported. False means it needs downloading first."""
    from . import bootstrap

    bootstrap.activate_av()  # pick up a copy downloaded on a previous run
    try:
        import av  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means unavailable
        return False
    return True


def probe(path: str | Path) -> VideoInfo:
    """Read a video's shape without decoding it."""
    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("That file has no video stream.")
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.base_rate or Fraction(24, 1)
        duration = 0.0
        if stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        elif container.duration:
            duration = container.duration / 1_000_000
        return VideoInfo(
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            fps=float(rate),
            frames=stream.frames or (round(duration * float(rate)) if duration else 0),
            has_audio=bool(container.streams.audio),
            duration=duration,
        )


def _encoder_opens(name: str, codec: Codec, fps: float, size: tuple[int, int]) -> bool:
    """Whether this encoder will actually open and take a frame, here and now.

    This is the fix for a real failure: ``add_stream`` succeeds for NVENC even
    when NVENC cannot run - the codec is only opened on the first frame, so a
    driver refusing it, a consumer GPU's concurrent-session cap, or a frame
    below NVENC's minimum size all surface as
    ``avcodec_open2("h264_nvenc") returned 22`` at encode time, far from the
    stream setup. Probing with a standalone context here, before committing the
    output stream, is what lets the libx264 fallback actually happen instead of
    the whole conversion dying.
    """
    import av

    width, height = size
    try:
        ctx = av.CodecContext.create(name, "w")
        ctx.width, ctx.height = width, height
        ctx.pix_fmt = codec.pix_fmt
        ctx.framerate = Fraction(fps).limit_denominator(90000)
        ctx.open()
        probe = av.VideoFrame(width, height, codec.pix_fmt)
        ctx.encode(probe)  # forces avcodec_open2 and one real encode
        # No explicit close: PyAV's codec context has none and frees on GC.
        # Flushing here is unnecessary - the point was only to prove it opens.
        return True
    except Exception:  # noqa: BLE001 - a probe failure just means "try the next"
        return False


def _pick_encoder(container, codec: Codec, fps: float, size: tuple[int, int]):
    """Add a video stream with the first encoder that actually works.

    NVENC is the default because DLSS already requires an RTX card, so hardware
    encoding is normally free - but "present" is not "usable": another app may
    hold every NVENC session, a laptop may gate it, a driver may refuse it. Each
    candidate is opened for real before it is chosen, and libx264 (always
    available, CPU) is the guaranteed floor.
    """
    width, height = size
    for name in (codec.encoder, codec.fallback):
        if name != codec.fallback and not _encoder_opens(name, codec, fps, size):
            continue
        try:
            stream = container.add_stream(name, rate=Fraction(fps).limit_denominator(90000))
            stream.width = width
            stream.height = height
            stream.pix_fmt = codec.pix_fmt
            return stream, name
        except Exception:  # noqa: BLE001 - fall through to the software floor
            continue
    raise RuntimeError(
        f"No usable encoder for {codec.label}. Even the software encoder "
        f"({codec.fallback}) would not open at {width}x{height}."
    )


def frames(
    path: str | Path,
    start: int = 0,
    limit: int | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[np.ndarray]:
    """Yield decoded frames as 0..1 float32 RGB, the form the pipeline wants.

    ``start`` and ``limit`` cut a range out of the middle, so someone can test
    five seconds before committing to the whole clip.
    """
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        index = 0
        emitted = 0
        for frame in container.decode(stream):
            if should_stop is not None and should_stop():
                return
            if index < start:
                index += 1
                continue
            if limit is not None and emitted >= limit:
                return
            rgb = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
            yield np.ascontiguousarray(rgb)
            index += 1
            emitted += 1


def mux_audio(
    source: Path,
    video_only: Path,
    destination: Path,
    start: float = 0.0,
    duration: float | None = None,
) -> bool:
    """Put the source's audio onto the converted video, trimmed to match.

    Returns True if audio was carried across, False if the source had none.

    ``start`` and ``duration`` (seconds) are the window the video covers. This
    is the fix for a real bug: a converted In/Out range got the *whole* audio
    track, so a 1.5 s clip played its picture and then sat on black for the rest
    of a two-minute soundtrack. When a window is given the audio is trimmed to
    it and re-timestamped to zero; the whole-clip case (no window) still remuxes
    the stream untouched, which is faster and lossless.
    """
    import av

    with av.open(str(source)) as src:
        if not src.streams.audio:
            shutil.copy2(video_only, destination)
            return False
        audio_in = src.streams.audio[0]
        trimming = start > 0 or duration is not None

        with av.open(str(video_only)) as vid, av.open(str(destination), mode="w") as out:
            video_in = vid.streams.video[0]
            video_out = out.add_stream_from_template(video_in)

            # Every output stream must be added before the first mux writes the
            # container header - add one afterwards and the mux dies with
            # "cannot rebase to zero time". So the audio stream is decided and
            # created here, up front, and only fed later.
            remux = False
            if not trimming:
                try:
                    audio_out = out.add_stream_from_template(audio_in)
                    remux = True
                except Exception:  # noqa: BLE001 - container will not take it raw
                    audio_out = out.add_stream("aac", rate=audio_in.rate)
                    audio_out.codec_context.time_base = Fraction(1, audio_in.rate)
            else:
                audio_out = out.add_stream("aac", rate=audio_in.rate)
                audio_out.codec_context.time_base = Fraction(1, audio_in.rate)

            for packet in vid.demux(video_in):
                if packet.dts is None:
                    continue
                packet.stream = video_out
                out.mux(packet)

            if remux:
                # Whole clip, container-compatible: copy the stream untouched.
                for packet in src.demux(audio_in):
                    if packet.dts is None:
                        continue
                    packet.stream = audio_out
                    out.mux(packet)
            else:
                # Trimmed, or a container that needs a re-encode: encode the
                # window (whole clip when not trimming), retimed to zero.
                end = None if duration is None else start + duration
                _reencode_audio_window(
                    src, out, audio_in, audio_out, start if trimming else 0.0, end
                )
    return True


def _reencode_audio_window(src, out, audio_in, audio_out, start: float, end: float | None) -> None:
    """Encode a time window of audio into an already-created AAC stream.

    ``audio_out`` is created by the caller before any muxing, because a stream
    cannot be added after the header is written. Two more things are
    load-bearing, each learned from a failure: each frame needs ``sample_rate``
    as well as ``pts``/``time_base``, and a FIFO repacketises to the encoder's
    fixed frame size, since decoded frames do not arrive that size and AAC will
    not take an odd one.
    """
    import av
    from fractions import Fraction

    rate = audio_in.rate
    time_base = Fraction(1, rate)
    resampler = av.AudioResampler(format="fltp", layout=audio_in.layout, rate=rate)
    fifo = av.AudioFifo()
    frame_size = audio_out.codec_context.frame_size or 1024
    counter = 0

    if start > 0:
        try:
            src.seek(int(start / audio_in.time_base), stream=audio_in, backward=True)
        except Exception:  # noqa: BLE001 - decode from the top if seek is refused
            pass

    def emit(flush: bool) -> None:
        nonlocal counter
        while True:
            frame = fifo.read() if flush else fifo.read(frame_size)
            if frame is None:
                return
            frame.pts = counter
            frame.time_base = time_base
            frame.sample_rate = rate
            counter += frame.samples
            for packet in audio_out.encode(frame):
                out.mux(packet)
            if flush:
                return

    for frame in src.decode(audio_in):
        if frame.pts is None:
            continue
        t = float(frame.pts * audio_in.time_base)
        if t < start:
            continue
        if end is not None and t >= end:
            break
        for resampled in resampler.resample(frame):
            resampled.pts = None
            fifo.write(resampled)
        emit(flush=False)
    emit(flush=True)
    for packet in audio_out.encode():
        out.mux(packet)


class VideoWriter:
    """An open output video, fed converted frames one at a time."""

    def __init__(self, path: Path, codec: Codec, fps: float, size: tuple[int, int]):
        import av

        self._container = av.open(str(path), mode="w")
        self._stream, self.encoder_used = _pick_encoder(self._container, codec, fps, size)
        self._av = av

    def write(self, image_rgb: np.ndarray) -> None:
        data = (np.clip(image_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        frame = self._av.VideoFrame.from_ndarray(np.ascontiguousarray(data), format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
