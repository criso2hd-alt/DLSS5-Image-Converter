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


def _pick_encoder(container, codec: Codec, fps: float, size: tuple[int, int]):
    """Add a video stream, preferring the hardware encoder, then software.

    NVENC is guaranteed present on the machines this app runs on - DLSS needs an
    RTX card - but a driver or a locked-down system can still refuse it, and a
    silent fall back to libx264 is far better than failing the whole export.
    """
    width, height = size
    last_error: Exception | None = None
    for name in (codec.encoder, codec.fallback):
        try:
            stream = container.add_stream(name, rate=Fraction(fps).limit_denominator(90000))
            stream.width = width
            stream.height = height
            stream.pix_fmt = codec.pix_fmt
            return stream, name
        except Exception as error:  # noqa: BLE001 - try the next encoder
            last_error = error
    raise RuntimeError(
        f"No usable encoder for {codec.label}: {last_error}"
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


def mux_audio(source: Path, video_only: Path, destination: Path) -> bool:
    """Copy the source's audio stream onto the converted video.

    Returns True if audio was carried across, False if the source had none.
    Both files are read and a third written, rather than editing in place,
    because a container cannot be safely appended to while it is being read.
    """
    import av

    with av.open(str(source)) as src:
        if not src.streams.audio:
            shutil.copy2(video_only, destination)
            return False
        audio_in = src.streams.audio[0]

        with av.open(str(video_only)) as vid, av.open(str(destination), mode="w") as out:
            video_in = vid.streams.video[0]
            video_out = out.add_stream_from_template(video_in)
            # Remux the audio without re-encoding when the container can hold it
            # as-is (MP4 takes AAC directly); re-encode to AAC only if not.
            try:
                audio_out = out.add_stream_from_template(audio_in)
                reencode = False
            except Exception:  # noqa: BLE001 - container will not take it raw
                audio_out = out.add_stream("aac", rate=audio_in.rate)
                reencode = False if audio_out is None else True

            for packet in vid.demux(video_in):
                if packet.dts is None:
                    continue
                packet.stream = video_out
                out.mux(packet)

            if reencode:
                resampler = None
                for frame in src.decode(audio_in):
                    for out_frame in _resample(frame, audio_out, resampler):
                        for packet in audio_out.encode(out_frame):
                            out.mux(packet)
                for packet in audio_out.encode():
                    out.mux(packet)
            else:
                for packet in src.demux(audio_in):
                    if packet.dts is None:
                        continue
                    packet.stream = audio_out
                    out.mux(packet)
    return True


def _resample(frame, stream, resampler):
    frame.pts = None
    yield frame


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
