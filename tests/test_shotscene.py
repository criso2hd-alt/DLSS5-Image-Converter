"""Shot scenes: each screenshot kept as a layer placed by its game depth."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from dlss5_converter import measured, multishot, shotscene


def _shot(tmp_path, name, distance, colour=(90, 140, 200), fraction_at_zero=0.0):
    """A screenshot and its depth sidecars (reversed Z, near 0.1)."""
    h, w = distance.shape
    image = tmp_path / f"{name}.png"
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = colour
    cv2.imwrite(str(image), img)
    raw = np.where(np.isfinite(distance), 0.1 / distance, 0.0).astype("<f4")
    raw.tofile(tmp_path / f"{name}.depth.f32")
    info = {"width": w, "height": h, "fraction_at_zero": fraction_at_zero, "fraction_at_one": 0.0}
    (tmp_path / f"{name}.depth.json").write_text(json.dumps(info), encoding="utf-8")
    return str(image)


def _scene(names, size, **kw):
    n = len(names)
    w, h = size
    K = np.array([[w * 0.8, 0, w / 2], [0, w * 0.8, h / 2], [0, 0, 1.0]])
    base = dict(names=names, rotations=np.array([np.eye(3)] * n), centres=np.zeros((n, 3)),
                cameras=np.array([K] * n), size=size,
                # raw d = 0.1 / z  ->  1/z = 10 d
                depth_a=np.full(n, 10.0), depth_b=np.zeros(n), gain=np.ones((n, 3)), trust=np.ones(n),
                used=np.ones(n, bool))
    base.update(kw)
    return shotscene.ShotScene(**base)


def test_depth_uses_each_shots_own_conversion(tmp_path):
    dist = np.full((40, 64), 3.0)
    path = _shot(tmp_path, "a", dist)
    scene = _scene([path], (64, 40))
    assert np.allclose(scene.depth(0, (64, 40)), 3.0, rtol=1e-4)
    # A shot whose game changed its near plane: a different a, same pixels.
    scene.depth_a[0] = 20.0
    assert np.allclose(scene.depth(0, (64, 40)), 1.5, rtol=1e-4)


def test_empty_depth_becomes_sky(tmp_path):
    dist = np.full((40, 64), 3.0)
    dist[:10] = np.inf
    path = _shot(tmp_path, "a", dist, fraction_at_zero=0.25)
    z = _scene([path], (64, 40)).depth(0, (64, 40))
    assert (z[:10] >= measured.SKY / 2).all() and np.allclose(z[20:], 3.0, rtol=1e-4)


def test_layer_does_not_bridge_depth_jumps(tmp_path):
    """A mesh stretched from a foreground edge to the wall behind it would
    smear across the gap; those triangles must be dropped."""
    dist = np.full((40, 64), 4.0)
    dist[:, :32] = 1.0                      # a near box on the left, a wall on the right
    path = _shot(tmp_path, "a", dist)
    scene = _scene([path], (64, 40))
    scene_layer = scene.layer(0)
    w, h = scene_layer["size"]
    z = scene.depth(0, (w, h))
    tri = scene_layer["indices"]
    zs = z.ravel()[tri]
    assert len(tri) > 0
    assert (zs.max(1) < zs.min(1) * (1 + shotscene.EDGE_JUMP)).all()


def test_layer_positions_follow_the_camera_ray(tmp_path):
    dist = np.full((40, 64), 2.0)
    path = _shot(tmp_path, "a", dist)
    scene = _scene([path], (64, 40))
    layer = scene.layer(0)
    w, h = layer["size"]
    centre = (h // 2) * w + w // 2
    # Identity camera and identity renderer transform: straight ahead is +z.
    assert layer["positions"][centre] == pytest.approx([0.0, 0.0, 2.0], abs=0.05)


def test_gain_undoes_exposure(tmp_path):
    dist = np.full((40, 64), 2.0)
    path = _shot(tmp_path, "a", dist, colour=(60, 60, 60))
    scene = _scene([path], (64, 40), gain=np.full((1, 3), 2.0))
    brighter = scene.colour(0, (32, 20)).astype(float).mean()
    plain = _scene([path], (64, 40)).colour(0, (32, 20)).astype(float).mean()
    assert brighter > plain * 1.3


def test_best_shots_prefers_near_and_same_direction(tmp_path):
    names = ["a", "b", "c"]
    turn = cv2.Rodrigues(np.array([0.0, np.pi, 0.0]))[0]       # looking the other way
    scene = _scene(names, (64, 40), rotations=np.array([np.eye(3), np.eye(3), turn]),
                   centres=np.array([[0, 0, 0], [3.0, 0, 0], [0.1, 0, 0]]))
    assert scene.best_shots(np.zeros(3), np.array([0, 0, 1.0]), count=2) == [0, 1]
    scene.used[0] = False
    assert scene.best_shots(np.zeros(3), np.array([0, 0, 1.0]), count=1) == [1]


def test_save_and_load_round_trip(tmp_path):
    dist = np.full((40, 64), 2.0)
    path = _shot(tmp_path, "a", dist)
    scene = _scene([path], (64, 40), planes=[(np.array([0, 1.0, 0]), 0.5)])
    moved = {0: np.zeros((20, 32), bool)}
    moved[0][5:10, 5:10] = True
    folder = scene.save(tmp_path / "saved", moved)
    back = shotscene.load(folder)
    assert shotscene.is_shot_scene(folder)
    assert back.names == scene.names and back.size == scene.size
    assert np.allclose(back.depth_a, scene.depth_a) and back.used.all()
    assert back.planes[0][1] == pytest.approx(0.5)
    assert back.moved_mask(0, (32, 20))[7, 7] and not back.moved_mask(0, (32, 20))[15, 25]


def test_multishot_saves_lists_and_loads_shot_scenes(tmp_path):
    dist = np.full((40, 64), 2.0)
    path = _shot(tmp_path, "Game 2026-01-01 10-00-00_1", dist)
    scene = _scene([path], (64, 40))
    report = multishot.Report()
    report.statuses[path] = multishot.PLACED
    report.root_image = np.zeros((40, 64, 3), np.uint8)
    folder = multishot.save(scene, report, "Game, 1 Jan 10:00: 1 shots", root=tmp_path / "scenes")
    listed = multishot.list_saved(tmp_path / "scenes")
    assert len(listed) == 1 and listed[0].folder == folder
    assert isinstance(multishot.load(folder), shotscene.ShotScene)


def test_saved_scene_draws_from_its_own_layer_cache(tmp_path):
    """Opening a saved scene must not depend on re-reading the 4K screenshots:
    the layers are cached at layer resolution when it is saved."""
    dist = np.full((40, 64), 2.5)
    path = _shot(tmp_path, "a", dist, colour=(30, 120, 220))
    scene = _scene([path], (64, 40), gain=np.full((1, 3), 1.5))
    folder = scene.save(tmp_path / "saved")
    size = scene.layer_size()
    assert (folder / "layer_000.npy").is_file() and (folder / "layer_000.jpg").is_file()
    back = shotscene.load(folder)
    (tmp_path / "a.png").unlink()                    # the original is gone
    assert np.allclose(back.depth(0, size), 2.5, rtol=2e-3)
    assert np.abs(back.colour(0, size).astype(int) - scene.colour(0, size).astype(int)).max() <= 6
