"""Lightning: the storm is repeatable, and a strike lights the frame."""
from __future__ import annotations

import numpy as np

from dlss5_converter import fx3d
from dlss5_converter.effects3d import EffectsState, LightningStrike


def test_strikes_are_the_same_on_every_playback():
    item = LightningStrike(rate=20.0, seed=4.0)
    times = np.arange(0.0, 30.0, 1.0 / 30.0)
    first = [fx3d.strike_at(item, t) for t in times]
    again = [fx3d.strike_at(item, t) for t in times]
    assert first == again
    assert any(b > 0.5 for b, _ in first), "20 strikes a minute must strike within 30 s"
    assert sum(b > 0.02 for b, _ in first) < len(times) * 0.3, "strikes are brief"


def test_the_bolt_reaches_its_target():
    item = LightningStrike(position=(1.0, -0.5, -4.0), height=5.0)
    main, width, _ = fx3d.bolt_paths(item, 3)[0]
    assert width == 1.0
    assert np.allclose(main[-1], item.position, atol=1e-5)
    assert main[0][1] > item.position[1] + 4.0


def test_flash_follows_the_strike_and_respects_hidden_bolts():
    item = LightningStrike(rate=30.0, flash=1.0, show_bolt=False)
    fx = EffectsState(strikes=[item])
    peak = max(fx3d.scene_flash(fx, t) for t in np.arange(0.0, 20.0, 1.0 / 60.0))
    assert peak > 0.8                       # a flash with no bolt drawn
    item.enabled = False
    assert max(fx3d.scene_flash(fx, t) for t in np.arange(0.0, 20.0, 0.05)) == 0.0
