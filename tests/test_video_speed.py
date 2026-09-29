"""The fast video path: exact colour tables, integer 3D, HDR in and out."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dlss5_converter import contract, hdrvideo, stereo, video
from dlss5_converter.settings import AppSettings

av = pytest.importorskip("av")


def test_srgb_table_is_exactly_the_float_curve():
    values = np.arange(256, dtype=np.float32) / 255.0
    expected = contract.srgb_to_linear(values).astype(np.float16)
    assert np.array_equal(contract.srgb8_to_half_lut().view(np.float16), expected)
    assert contract.srgb8_to_half_lut().view(np.float16)[255] == 1.0     # alpha 255 -> 1.0


def test_output_table_matches_the_old_float_path():
    rng = np.random.default_rng(1)
    halves = rng.uniform(-0.5, 3.0, 4096).astype(np.float16)
    halves[:3] = [np.nan, np.inf, -np.inf]
    old = np.nan_to_num(halves.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    old = (np.clip(contract.linear_to_srgb(old), 0, 1) * 255.0 + 0.5).astype(np.uint8)
    assert np.array_equal(contract.half_to_srgb8_lut()[halves.view(np.uint16)], old)


def test_threaded_lookup_equals_indexing():
    rng = np.random.default_rng(2)
    table = rng.integers(0, 255, 65536).astype(np.uint8)
    index = rng.integers(0, 65536, (300, 97, 3)).astype(np.uint16)
    assert np.array_equal(contract.table_lookup(table, index), table[index])


def test_pq_curve_known_points_and_round_trip():
    assert hdrvideo.nits_to_pq(np.array([10000.0]))[0] == pytest.approx(1.0, abs=1e-6)
    assert hdrvideo.nits_to_pq(np.array([100.0]))[0] == pytest.approx(0.5081, abs=2e-3)
    signal = np.linspace(0.05, 1.0, 50)
    assert np.allclose(hdrvideo.nits_to_pq(hdrvideo.pq_to_nits(signal)), signal, atol=1e-6)


def test_reference_white_lands_on_one():
    code = int(hdrvideo.nits_to_pq(np.array([hdrvideo.SDR_WHITE_NITS]))[0] * 65535 + 0.5)
    assert hdrvideo.decode_lut("pq")[code] == pytest.approx(1.0, rel=2e-3)
    one = np.array([1.0], np.float16).view(np.uint16)
    assert abs(int(hdrvideo.half_to_pq_lut()[one][0]) - code) <= 2


def test_hlg_is_monotonic_and_peaks_at_display_peak():
    nits = hdrvideo.hlg_to_nits(np.linspace(0, 1, 200))
    assert (np.diff(nits) >= 0).all()
    assert nits[-1] == pytest.approx(hdrvideo.HLG_PEAK_NITS, rel=1e-3)


def test_tone_map_keeps_midtones_and_rolls_highlights_off():
    x = np.array([0.0, 0.25, 0.5, 0.8, 1.0, 2.0, 10.0])
    y = hdrvideo.tone_map(x)
    assert np.allclose(y[:4], x[:4])
    assert (np.diff(y) > 0).all() and y[-1] <= 1.0 and y[4] < 1.0


def test_integer_eyes_match_float_eyes():
    rng = np.random.default_rng(3)
    rgb8 = rng.integers(0, 256, (72, 128, 3)).astype(np.uint8)
    depth = np.tile(np.linspace(0, 1, 128, dtype=np.float32), (72, 1))
    left8, right8 = stereo.views(rgb8, depth, 0.8, 0.3)
    leftf, rightf = stereo.views(rgb8.astype(np.float32) / 255.0, depth, 0.8, 0.3)
    assert left8.dtype == np.uint8
    assert np.abs(left8.astype(float) - leftf * 255).max() <= 1.0
    assert np.abs(right8.astype(float) - rightf * 255).max() <= 1.0


def test_depth_from_a_small_frame_scales_up_in_compose():
    settings = AppSettings().stereo
    rgb = np.zeros((200, 320, 3), np.uint8)
    depth = np.full((50, 80), 0.5, np.float32)
    for fmt in ("sbs_half", "depth", "rgbd"):
        settings.format = fmt
        out = stereo.compose(rgb, depth, settings)
        assert out.shape[:2] == stereo.output_size(fmt, (320, 200))[::-1]


def _hdr_clip(path: Path, frames: int = 6, size=(256, 144)) -> None:
    w, h = size
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx265", rate=24)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p10le"
        stream.options = {"preset": "ultrafast", "crf": "4", "x265-params": "log-level=none"}
        cc = stream.codec_context
        cc.color_primaries, cc.color_trc, cc.colorspace, cc.color_range = 9, 16, 9, 1
        ramp = np.linspace(0.0, 4.0, w, dtype=np.float32)
        for _ in range(frames):
            lin = np.repeat(np.tile(ramp, (h, 1))[..., None], 3, axis=2)
            pq = hdrvideo.to_pq16(lin)
            frame = av.VideoFrame.from_ndarray(pq, format="rgb48le")
            frame.color_trc, frame.color_primaries = 16, 9
            frame = frame.reformat(format="yuv420p10le", src_colorspace="bt2020", dst_colorspace="bt2020",
                                   src_color_range="JPEG", dst_color_range="MPEG")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_hdr_source_is_detected_and_decodes_to_linear_light(tmp_path):
    clip = tmp_path / "hdr.mp4"
    _hdr_clip(clip)
    assert video.probe(clip).hdr == "pq"
    frame = next(video.frames(clip, hdr="pq"))
    assert frame.dtype == np.float32
    row = frame[72, :, 1]
    # The ramp ran 0..4x reference white: highlights survive above 1.0.
    assert row.max() > 3.0 and row[:4].max() < 0.06
    assert np.allclose(row[::32], np.linspace(0, 4, 256)[::32], atol=0.08 * row[::32] + 0.02)


def test_hdr10_writer_round_trips(tmp_path):
    clip = tmp_path / "in.mp4"
    _hdr_clip(clip)
    source = next(video.frames(clip, hdr="pq"))
    out = tmp_path / "out.mp4"
    codec = video.hdr_codec(video.CODECS_BY_KEY["h265"])
    writer = video.VideoWriter(out, codec, 24, (source.shape[1], source.shape[0]), hdr=True)
    for _ in range(4):
        writer.write(hdrvideo.to_pq16(source))
    writer.close()
    assert video.probe(out).hdr == "pq"
    back = next(video.frames(out, hdr="pq"))
    # Tagged, not converted: the light comes back as it went in.
    assert np.allclose(back[72], source[72], atol=0.05 * source[72] + 0.02)


def test_codecs_that_cannot_carry_hdr_are_tone_mapped():
    assert video.hdr_codec(video.CODECS_BY_KEY["h265"]).pix_fmt == "yuv420p10le"
    assert video.hdr_codec(video.CODECS_BY_KEY["prores"]) is video.CODECS_BY_KEY["prores"]
    assert video.hdr_codec(video.CODECS_BY_KEY["h264"]) is None
    assert video.hdr_codec(video.CODECS_BY_KEY["vp9"]) is None


class _FlatDepth:
    """A stand-in depth model: nearer towards the right edge."""

    def load(self, *_a, **_k):
        pass

    def infer(self, image, **_k):
        h, w = image.shape[:2]
        return np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))


@pytest.mark.parametrize("codec_key, expect_hdr", [("h265", "pq"), ("h264", "")])
def test_hdr_3d_conversion_end_to_end(tmp_path, codec_key, expect_hdr):
    from dlss5_converter import pipeline

    clip = tmp_path / "hdr.mp4"
    _hdr_clip(clip)
    settings = AppSettings()
    settings.stereo.enabled = True
    settings.stereo.run_dlss = False        # 3D only: no DLSS runtime needed here
    settings.stereo.format = "sbs_half"
    out = tmp_path / f"out_{codec_key}.mp4"
    stages = [u.stage for u in pipeline.convert_video(clip, out, settings, _FlatDepth(), codec_key=codec_key)]
    assert stages.count("converting") == 6 and stages[-1].startswith("done")
    info = video.probe(out)
    assert info.hdr == expect_hdr and (info.width, info.height) == (256, 144)
