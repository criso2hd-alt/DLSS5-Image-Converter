"""Video conversion: decode, per-frame convert, encode, carry audio across.

The pipeline itself is exercised elsewhere; these pin the video-specific parts -
codec selection, range cutting, and the promise that audio survives - plus the
UI wiring, because a video tab that raises on the first click is the same class
of miss that shipped in v0.1.15.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dlss5_converter import video

pyav = pytest.mark.skipif(not video.is_available(), reason="PyAV not installed")


def make_clip(path, frames=12, size=(160, 90), fps=24, audio=True):
    """A tiny H.264 clip, optionally with an AAC tone."""
    import av

    with av.open(str(path), mode="w") as container:
        vs = container.add_stream("libx264", rate=fps)
        vs.width, vs.height, vs.pix_fmt = size[0], size[1], "yuv420p"
        astream = container.add_stream("aac", rate=48000) if audio else None
        for i in range(frames):
            img = np.zeros((size[1], size[0], 3), np.uint8)
            img[:, (i * 5) % size[0]:] = 120
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for p in vs.encode(frame):
                container.mux(p)
        for p in vs.encode():
            container.mux(p)
        if astream is not None:
            sr = 48000
            t = np.arange(int(sr * frames / fps)) / sr
            tone = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            af = av.AudioFrame.from_ndarray(tone.reshape(1, -1), format="fltp", layout="mono")
            af.sample_rate = sr
            for p in astream.encode(af):
                container.mux(p)
            for p in astream.encode():
                container.mux(p)
    return path


# -- codec catalogue ---------------------------------------------------------


def test_h264_is_the_default_and_first_offered():
    """Universal editor compatibility, so it leads."""
    assert video.CODECS[0].key == "h264"
    assert video.CODECS_BY_KEY["h264"].suffix == ".mp4"


def test_webm_is_last_and_flagged_for_web_not_editing():
    vp9 = video.CODECS[-1]
    assert vp9.key == "vp9"
    assert "web" in vp9.note.lower()


def test_every_codec_has_a_software_fallback():
    """NVENC can be refused; the export must still complete."""
    for codec in video.CODECS:
        assert codec.fallback


# -- decode / probe ----------------------------------------------------------


@pyav
def test_probe_reads_shape_without_decoding(tmp_path):
    make_clip(tmp_path / "c.mp4", frames=20, size=(160, 90), fps=24)
    info = video.probe(tmp_path / "c.mp4")
    assert (info.width, info.height) == (160, 90)
    assert info.fps == pytest.approx(24, abs=0.1)
    assert info.frames == 20
    assert info.has_audio


@pyav
def test_a_file_with_no_video_is_rejected(tmp_path):
    # A .mp4 that is audio-only.
    import av

    path = tmp_path / "audio.mp4"
    with av.open(str(path), mode="w") as c:
        astream = c.add_stream("aac", rate=48000)
        tone = np.zeros((1, 48000), np.float32)
        af = av.AudioFrame.from_ndarray(tone, format="fltp", layout="mono")
        af.sample_rate = 48000
        for p in astream.encode(af):
            c.mux(p)
        for p in astream.encode():
            c.mux(p)
    with pytest.raises(ValueError, match="no video"):
        video.probe(path)


@pyav
def test_frames_yield_float_rgb(tmp_path):
    make_clip(tmp_path / "c.mp4", frames=6)
    got = list(video.frames(tmp_path / "c.mp4"))
    assert len(got) == 6
    assert got[0].dtype == np.float32
    assert 0.0 <= got[0].min() and got[0].max() <= 1.0


@pyav
def test_the_range_limiter_cuts_the_middle(tmp_path):
    make_clip(tmp_path / "c.mp4", frames=20)
    assert len(list(video.frames(tmp_path / "c.mp4", start=5, limit=8))) == 8


@pyav
def test_frames_stop_when_asked(tmp_path):
    make_clip(tmp_path / "c.mp4", frames=20)
    n = 0
    for _ in video.frames(tmp_path / "c.mp4", should_stop=lambda: n >= 3):
        n += 1
    assert n <= 4  # the check is between frames, so one may slip through


# -- encode / mux ------------------------------------------------------------


@pyav
def test_a_written_video_reads_back_at_the_right_size(tmp_path):
    out = tmp_path / "out.mp4"
    writer = video.VideoWriter(out, video.CODECS_BY_KEY["h264"], 24, (160, 90))
    for _ in range(10):
        writer.write(np.random.default_rng(0).random((90, 160, 3)).astype(np.float32))
    writer.close()
    info = video.probe(out)
    assert (info.width, info.height) == (160, 90)
    assert info.frames == 10


@pyav
def test_audio_is_carried_across(tmp_path):
    """The whole point of muxing rather than just re-encoding video."""
    import av

    src = make_clip(tmp_path / "src.mp4", frames=10, audio=True)
    video_only = tmp_path / "vonly.mp4"
    writer = video.VideoWriter(video_only, video.CODECS_BY_KEY["h264"], 24, (160, 90))
    for _ in range(10):
        writer.write(np.zeros((90, 160, 3), np.float32))
    writer.close()

    final = tmp_path / "final.mp4"
    had_audio = video.mux_audio(src, video_only, final)
    assert had_audio
    with av.open(str(final)) as c:
        assert c.streams.audio, "the muxed file must have an audio stream"
        assert c.streams.video


@pyav
def test_a_silent_source_yields_a_silent_output(tmp_path):
    src = make_clip(tmp_path / "src.mp4", frames=6, audio=False)
    video_only = tmp_path / "v.mp4"
    writer = video.VideoWriter(video_only, video.CODECS_BY_KEY["h264"], 24, (160, 90))
    for _ in range(6):
        writer.write(np.zeros((90, 160, 3), np.float32))
    writer.close()
    final = tmp_path / "final.mp4"
    assert video.mux_audio(src, video_only, final) is False
    assert final.exists()


# -- UI wiring (the v0.1.15 guard) -------------------------------------------


@pytest.fixture(scope="module")
def qt_app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_the_video_tab_sits_between_single_image_and_sequence(qt_app):
    from dlss5_converter.app import MainWindow

    w = MainWindow()
    order = [w.tabs.tabText(i) for i in range(w.tabs.count())]
    assert order == ["Single image", "Video", "Image sequence"]
    w.deleteLater()


def test_the_video_tab_handlers_exist_on_the_window(qt_app):
    """A tab whose button raises on click is the v0.1.15 failure again."""
    from dlss5_converter.app import MainWindow

    for name in ("_pick_video", "_start_video", "_stop_video",
                 "_video_frame_done", "_video_finished", "_video_teardown",
                 "_download_video_support"):
        assert hasattr(MainWindow, name), name
