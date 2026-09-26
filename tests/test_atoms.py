"""Atoms: items followed across shots, trusted only at 90% agreement."""

from __future__ import annotations

import numpy as np
import pytest

from dlss5_converter import atoms as at
from dlss5_converter import multiview as mv


class _Key:
    def __init__(self, x, y):
        self.pt = (float(x), float(y))


CAMERA = mv.intrinsics(960, 540, 70.0)


def _placed_shots(centres):
    shots = []
    for i, centre in enumerate(centres):
        shot = mv.Shot(f"s{i}", np.zeros((540, 960, 3), np.uint8))
        shot.centre = np.asarray(centre, float)
        shots.append(shot)
    return shots


def _see(shots, point):
    """Keypoints where each shot sees a world point."""
    keys = []
    for shot in shots:
        c = shot.rotation @ (point - shot.centre)
        keys.append(_Key(CAMERA[0, 0] * c[0] / c[2] + CAMERA[0, 2],
                         CAMERA[1, 1] * c[1] / c[2] + CAMERA[1, 2]))
    return keys


def test_matches_chain_into_one_atom_across_shots():
    described = [([_Key(0, 0)] * 3, None) for _ in range(3)]
    pairs = {(0, 1): np.array([[0, 1]]), (1, 2): np.array([[1, 2]])}
    atoms = at.build_atoms(described, pairs)
    assert len(atoms) == 1
    assert sorted(atoms[0].sightings) == [(0, 0), (1, 1), (2, 2)]


def test_an_atom_with_two_keypoints_in_one_shot_is_dropped():
    """A chain of wrong matches glued two items together: do not guess."""
    described = [([_Key(0, 0)] * 3, None) for _ in range(3)]
    pairs = {(0, 1): np.array([[0, 0]]), (1, 2): np.array([[0, 0]]),
             (0, 2): np.array([[0, 1]])}
    assert at.build_atoms(described, pairs) == []


def test_atom_seen_by_three_agreeing_shots_is_trusted():
    shots = _placed_shots([(0, 0, 0), (0.5, 0, 0), (1.0, 0, 0)])
    point = np.array([0.3, -0.2, 6.0])
    described = [([k], None) for k in _see(shots, point)]
    atom = at.Atom([(0, 0), (1, 0), (2, 0)])
    at.locate(atom, shots, described, CAMERA, {0, 1, 2})
    assert atom.position == pytest.approx(point, abs=1e-6)
    assert atom.confidence == 1.0 and atom.trusted


def test_two_sightings_are_usable_but_never_trusted():
    shots = _placed_shots([(0, 0, 0), (0.5, 0, 0)])
    described = [([k], None) for k in _see(shots, np.array([0.0, 0.0, 5.0]))]
    atom = at.Atom([(0, 0), (1, 0)])
    at.locate(atom, shots, described, CAMERA, {0, 1})
    assert atom.usable and not atom.trusted


def test_one_disagreeing_shot_in_three_fails_the_90_percent_rule():
    shots = _placed_shots([(0, 0, 0), (0.5, 0, 0), (1.0, 0, 0)])
    keys = _see(shots, np.array([0.2, 0.1, 6.0]))
    keys[2] = _Key(keys[2].pt[0] + 40, keys[2].pt[1])     # a wrong match
    described = [([k], None) for k in keys]
    atom = at.Atom([(0, 0), (1, 0), (2, 0)])
    at.locate(atom, shots, described, CAMERA, {0, 1, 2})
    assert atom.confidence < at.MIN_CONFIDENCE
    assert not atom.trusted


def test_atom_from_one_direction_is_not_trusted():
    """Sightings that agree but all look the same way leave depth undetermined."""
    shots = _placed_shots([(0, 0, 0), (0.001, 0, 0), (0.002, 0, 0)])
    described = [([k], None) for k in _see(shots, np.array([0.0, 0.0, 20.0]))]
    atom = at.Atom([(0, 0), (1, 0), (2, 0)])
    at.locate(atom, shots, described, CAMERA, {0, 1, 2})
    assert not atom.trusted


def test_bend_field_is_smooth_and_passes_near_the_samples():
    rng = np.random.default_rng(0)
    pixels = rng.uniform([0, 0], [960, 540], (400, 2))
    truth = 0.3 * pixels[:, 0] / 960 - 0.1          # a gentle left-right tilt
    field = at.bend_field((540, 960), pixels, truth)
    assert field.shape == (540, 960)
    assert field[270, 900] > field[270, 60]
    assert np.abs(np.diff(field, axis=1)).max() < 0.01


def test_bend_field_ignores_a_few_wild_samples():
    """Medians: atoms on a reflection must not drag a region of wall."""
    rng = np.random.default_rng(1)
    pixels = rng.uniform([0, 0], [960, 540], (600, 2))
    values = np.zeros(600)
    values[:30] = 5.0
    field = at.bend_field((540, 960), pixels, values)
    assert np.abs(np.median(field)) < 0.05


def test_bend_field_with_no_samples_changes_nothing():
    field = at.bend_field((54, 96), np.zeros((0, 2)), np.zeros(0))
    assert not field.any()


# --- trust tiers ------------------------------------------------------------

def _tier_scene(confidences, spacing=40.0, depths=None):
    """A row of atoms seen by one shot, each `spacing` px from the next."""
    shots = _placed_shots([(0, 0, 0)])
    atoms, keys = [], []
    for i, confidence in enumerate(confidences):
        depth = 6.0 if depths is None else depths[i]
        x = (100 + i * spacing - CAMERA[0, 2]) / CAMERA[0, 0] * depth
        atom = at.Atom([(0, i)], position=np.array([x, 0.0, depth]),
                       confidence=confidence, views=3 if i == 0 else 2)
        atoms.append(atom)
        keys.append(_Key(100 + i * spacing, CAMERA[1, 2]))
    return atoms, shots, [(keys, None)]


def test_trust_spreads_outward_with_decaying_bars():
    atoms, shots, described = _tier_scene([1.0, 0.85, 0.75, 0.65, 0.55, 0.45])
    counts = at.assign_tiers(atoms, shots, described, {0})
    assert [a.tier for a in atoms] == [0, 1, 2, 3, 4, -1]
    assert counts == {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}


def test_a_weak_link_breaks_the_chain():
    """Tier 1 needs 80%: a 70% atom next to the core stops the spread there."""
    atoms, shots, described = _tier_scene([1.0, 0.7, 1.0])
    at.assign_tiers(atoms, shots, described, {0})
    assert [a.tier for a in atoms] == [0, -1, -1]


def test_trust_does_not_jump_a_depth_edge():
    """A face in front of a far wall does not vouch for the wall."""
    atoms, shots, described = _tier_scene([1.0, 1.0], depths=[3.0, 12.0])
    at.assign_tiers(atoms, shots, described, {0})
    assert atoms[1].tier == -1


def test_trust_does_not_jump_a_gap():
    atoms, shots, described = _tier_scene([1.0, 1.0], spacing=at.NEIGHBOUR_PX * 2)
    at.assign_tiers(atoms, shots, described, {0})
    assert atoms[1].tier == -1


def test_hard_cut_is_adjustable():
    atoms, shots, described = _tier_scene([1.0, 0.85, 0.75, 0.65, 0.55, 0.45])
    at.assign_tiers(atoms, shots, described, {0}, hard_cut=0.7)
    assert [a.tier for a in atoms] == [0, 1, 2, -1, -1, -1]


def test_requirement_map_decays_with_distance_then_cuts():
    atoms, shots, described = _tier_scene([1.0])
    at.assign_tiers(atoms, shots, described, {0})
    bar = at.requirement_map(0, atoms, shots, described)
    y = int(CAMERA[1, 2])
    assert bar[y, 100] == pytest.approx(0.9)
    assert bar[y, 100 + int(at.NEIGHBOUR_PX) + 5] == pytest.approx(0.8)
    assert np.isnan(bar[y, 100 + int(at.NEIGHBOUR_PX * 6)])
