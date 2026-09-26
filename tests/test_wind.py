"""Scene wind: one wind carrying particles and volumes."""

from __future__ import annotations

import numpy as np
import pytest

from dlss5_converter import effects3d as fx


def test_still_air_by_default():
    now, avg, _ = fx.wind_vector(fx.LightingSettings(), 3.0)
    assert np.allclose(now, 0.0) and np.allclose(avg, 0.0)


@pytest.mark.parametrize("degrees, expected", [(0.0, (0, 0, -1)), (90.0, (1, 0, 0)),
                                               (180.0, (0, 0, 1)), (270.0, (-1, 0, 0))])
def test_direction_is_a_compass_heading(degrees, expected):
    light = fx.LightingSettings(wind_direction=degrees, wind_strength=2.0, wind_gusts=0.0)
    _now, avg, _ = fx.wind_vector(light, 0.0)
    assert np.allclose(avg / 2.0, expected, atol=1e-9)


def test_gusts_swell_but_never_reverse():
    light = fx.LightingSettings(wind_direction=90.0, wind_strength=1.0, wind_gusts=1.0)
    speeds = [fx.wind_vector(light, t)[0][0] for t in np.linspace(0, 60, 600)]
    assert min(speeds) >= 0.0 and max(speeds) > 1.3 and min(speeds) < 0.7


def test_wind_is_keyframable_like_every_other_setting():
    paths = fx.animatable_paths(fx.EffectsState())
    for name in ("wind_direction", "wind_strength", "wind_gusts", "wind_turbulence"):
        assert f"{fx.ENVIRONMENT}.{name}" in paths


def test_each_effect_has_its_own_wind_response():
    assert fx.emitter_preset("rain").wind_response == 1.0
    assert fx.emitter_preset("fire").wind_response < 1.0
    assert fx.volume_preset("fog").wind_response == 1.0


def test_snow_builds_up_over_time_with_preroll():
    light = fx.LightingSettings(snow_cover=0.8)
    assert fx.snow_amount(light, 0.0) == pytest.approx(0.8)          # no build-up: already there
    light.snow_build = 10.0
    assert fx.snow_amount(light, 0.0) == pytest.approx(0.0)
    assert fx.snow_amount(light, 5.0) == pytest.approx(0.4)
    assert fx.snow_amount(light, 30.0) == pytest.approx(0.8)
    light.snow_preroll = 5.0
    assert fx.snow_amount(light, 0.0) == pytest.approx(0.4)          # half built at frame 0


def test_duplicate_copies_every_value_and_keyframe():
    state = fx.EffectsState()
    smoke = fx.emitter_preset("smoke")
    smoke.count, smoke.colour = 999, (0.1, 0.2, 0.3)
    state.emitters.append(smoke)
    state.volumes.append(fx.volume_preset("fog"))
    track = fx.EffectsTrack()
    track.key_property(0.0, f"{smoke.id}.count", state)
    later = state.clone()
    later.emitters[0].count = 50
    track.key_property(2.0, f"{smoke.id}.count", later)

    copy_ = fx.duplicate_effect(state, track, smoke.id)
    assert copy_.id != smoke.id and copy_.name == "Smoke emitter copy"
    assert copy_.count == 999 and copy_.colour == (0.1, 0.2, 0.3)
    assert copy_.position[0] == pytest.approx(smoke.position[0] + fx.DUPLICATE_NUDGE)
    assert [e.id for e in state.emitters] == [smoke.id, copy_.id]     # right after it
    # The animation came along: halfway between the keys, both are at the same count.
    mid = track.evaluate(1.0, state)
    assert mid.item(copy_.id).count == mid.item(smoke.id).count < 999
    assert fx.duplicate_effect(state, track, "nope") is None
