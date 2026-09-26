"""A scene built from game screenshots: every shot kept as itself.

Fusing shots into one cloud of splats (SHARP, Brush, carving) always lost the
screenshots' detail: independent reconstructions never agree exactly. This
keeps each screenshot whole instead. A shot is a layer: its own pixels, placed
by the game's depth, seen from its solved camera. The renderer (`ibr3d`)
blends the few best layers per pixel for any view.

What is stored per shot (all solved when the scene is built):
- camera: rotation, centre, intrinsics (a shot may have its own focal length,
  photo modes can zoom);
- depth conversion 1/z = a*d + b from the raw buffer value d (the game can
  change its near plane between screenshots);
- colour gain (undoes the game's auto-exposure);
- trust (how well the shot agreed with its neighbours) and whether it is used;
- a "moved" mask: pixels whose surface a neighbour sees straight through,
  i.e. something that moved between shots. Those come from one shot only.

Qt-free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import game_depth, measured

#: Layers are built at half the screenshot's resolution: a 4K shot is 1920
#: wide on the GPU, sharper than the viewport and a quarter of the memory.
LAYER_SUB = 2

#: Pixels with no depth (sky, a street far outside a window) go on a distant
#: backdrop instead of being dropped, so windows do not render black. Inside
#: the viewport camera's far plane (60).
BACKDROP = 50.0

#: A mesh must not bridge a depth jump: a triangle whose corners differ in
#: depth by more than this fraction would stretch a rubber sheet from a
#: foreground edge to the wall behind it.
EDGE_JUMP = 0.03

#: Width of the soft border (fraction of the frame) where a shot fades out,
#: so a neighbour takes over gradually instead of leaving a seam.
FEATHER = 0.08


@dataclass
class ShotScene:
    names: list[str]
    rotations: np.ndarray          # N x 3 x 3, world -> camera (capture world)
    centres: np.ndarray            # N x 3
    cameras: np.ndarray            # N x 3 x 3, at full screenshot resolution
    size: tuple[int, int]          # full screenshot (w, h)
    depth_a: np.ndarray            # N
    depth_b: np.ndarray            # N
    gain: np.ndarray               # N x 3
    trust: np.ndarray              # N
    used: np.ndarray               # N bool
    # Levelled renderer world: p_r = (p_c - target) @ axes.T * scale + offset
    axes: np.ndarray = field(default_factory=lambda: np.eye(3))
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: float = 1.0
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    planes: list = field(default_factory=list)       # [(normal, c)] renderer world
    folder: Path | None = None     # where moved masks live, once saved
    pending_moved: dict = field(default_factory=dict, repr=False)   # masks not yet saved

    def __len__(self) -> int:
        return len(self.names)

    # --- cameras in the renderer world ---------------------------------------
    def to_renderer(self, p: np.ndarray) -> np.ndarray:
        return (p - self.target) @ self.axes.T * self.scale + self.offset

    def eye(self, k: int) -> np.ndarray:
        return self.to_renderer(self.centres[k][None])[0]

    def forward(self, k: int) -> np.ndarray:
        return self.axes @ self.rotations[k][2]

    def up(self, k: int) -> np.ndarray:
        return self.axes @ -self.rotations[k][1]

    def best_shots(self, eye: np.ndarray, forward: np.ndarray, count: int = 4) -> list[int]:
        """The used shots nearest to a view, by position and looking direction."""
        use = np.flatnonzero(self.used)
        if len(use) == 0:
            return []
        eyes = np.array([self.eye(k) for k in use])
        fwds = np.array([self.forward(k) for k in use])
        spread = np.median(np.linalg.norm(eyes - eyes.mean(0), axis=1)) + 1e-9
        score = np.linalg.norm(eyes - eye, axis=1) / spread + np.arccos(np.clip(fwds @ forward, -1, 1))
        return [int(use[i]) for i in np.argsort(score)[:count]]

    # --- per-shot pixels -----------------------------------------------------
    def _cached(self, k: int, kind: str, size) -> Path | None:
        if self.folder is None or tuple(size) != self.layer_size():
            return None
        path = self.folder / f"layer_{k:03d}.{kind}"
        return path if path.is_file() else None

    def layer_size(self) -> tuple[int, int]:
        return self.size[0] // LAYER_SUB, self.size[1] // LAYER_SUB

    def depth(self, k: int, size: tuple[int, int]) -> np.ndarray:
        """Camera-space z at `size`, with this shot's own conversion; SKY where
        the buffer holds nothing."""
        cached = self._cached(k, "npy", size)
        if cached is not None:
            z = np.load(cached).astype(np.float64)
            return np.where(np.isfinite(z), z, measured.SKY)
        raw = game_depth.load(self.names[k], size)
        if raw is None:
            return np.full((size[1], size[0]), measured.SKY)
        d = raw.astype(np.float64)
        z = 1.0 / np.maximum(self.depth_a[k] * d + self.depth_b[k], 1e-9)
        return np.where(d > 1e-6, z, measured.SKY)

    def colour(self, k: int, size: tuple[int, int]) -> np.ndarray:
        """RGB uint8 at `size`, with this shot's exposure undone."""
        cached = self._cached(k, "jpg", size)
        if cached is not None:
            img = cv2.imdecode(np.fromfile(str(cached), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.imdecode(np.fromfile(self.names[k], np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(cv2.resize(img, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        c = img.astype(np.float32) / 255.0
        lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4) * self.gain[k][None, None, :]
        out = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(np.maximum(lin, 0), 1 / 2.4) - 0.055)
        return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)

    def moved_mask(self, k: int, size: tuple[int, int]) -> np.ndarray:
        if self.folder is not None:
            path = self.folder / f"moved_{k:03d}.png"
            if path.is_file():
                m = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
                return cv2.resize(m, size, interpolation=cv2.INTER_NEAREST) > 127
        return np.zeros((size[1], size[0]), bool)

    def layer(self, k: int):
        """Mesh for shot k at layer resolution, in the renderer world.

        Returns dict: positions (H*W x 3 f32), uv (H*W x 2), feather (H*W),
        moved (H*W), indices (M x 3 uint32), colour (H x W x 4 uint8).
        """
        fw, fh = self.size
        w, h = fw // LAYER_SUB, fh // LAYER_SUB
        z = self.depth(k, (w, h))
        K = self.cameras[k] / np.array([[LAYER_SUB], [LAYER_SUB], [1.0]])
        sky = z >= measured.SKY / 2
        # Backdrop: a far shell at a fixed renderer distance, so it sits
        # behind everything but inside the camera's far plane.
        z = np.where(sky, BACKDROP / self.scale, z)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        cam = np.stack([(xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z], -1).reshape(-1, 3)
        world = self.centres[k] + cam @ self.rotations[k]
        positions = self.to_renderer(world).astype(np.float32)
        uv = np.stack([(xs + 0.5) / w, (ys + 0.5) / h], -1).reshape(-1, 2).astype(np.float32)
        edge = np.minimum(np.minimum(xs, w - 1 - xs) / (FEATHER * w), np.minimum(ys, h - 1 - ys) / (FEATHER * h))
        f = np.clip(edge, 0, 1)
        feather = (f * f * (3 - 2 * f)).ravel().astype(np.float32)
        moved = self.moved_mask(k, (w, h)).ravel().astype(np.float32)
        # Two triangles per quad, dropped where the depth jumps or where a
        # real surface meets the backdrop.
        idx = np.arange(h * w, dtype=np.uint32).reshape(h, w)
        a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
        zz = z.astype(np.float32)
        za, zb, zc, zd = zz[:-1, :-1], zz[:-1, 1:], zz[1:, :-1], zz[1:, 1:]
        sk = sky.astype(np.int8)
        same_sky = (sk[:-1, :-1] + sk[:-1, 1:] + sk[1:, :-1] + sk[1:, 1:]) % 4 == 0

        def flat(p, q, r):
            lo = np.minimum(np.minimum(p, q), r)
            hi = np.maximum(np.maximum(p, q), r)
            return hi < lo * (1 + EDGE_JUMP)

        t1 = flat(za, zb, zc) & same_sky
        t2 = flat(zb, zd, zc) & same_sky
        indices = np.concatenate([np.stack([a[t1], b[t1], c[t1]], -1),
                                  np.stack([b[t2], d[t2], c[t2]], -1)]).astype(np.uint32)
        rgb = self.colour(k, (w, h))
        colour = np.concatenate([rgb, np.full((h, w, 1), 255, np.uint8)], -1)
        return {"positions": positions, "uv": uv, "feather": feather, "moved": moved,
                "indices": indices, "colour": colour, "size": (w, h)}

    # --- saving --------------------------------------------------------------
    def save(self, folder: Path, moved: dict[int, np.ndarray] | None = None, progress=None) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        planes = np.array([list(n) + [c] for n, c in self.planes], np.float32).reshape(-1, 4)
        np.savez(folder / "shots.npz", rotations=self.rotations, centres=self.centres, cameras=self.cameras,
                 depth_a=self.depth_a, depth_b=self.depth_b, gain=self.gain, trust=self.trust,
                 used=self.used, axes=self.axes, target=self.target, scale=np.float64(self.scale),
                 offset=self.offset, planes=planes, size=np.array(self.size))
        (folder / "shots.json").write_text(json.dumps({"kind": "shots", "names": self.names}, indent=1),
                                           encoding="utf-8")
        for k, m in (moved or {}).items():
            ok, buf = cv2.imencode(".png", (m.astype(np.uint8) * 255))
            if ok:
                buf.tofile(str(folder / f"moved_{k:03d}.png"))
        # Each used shot at layer resolution, ready to draw: decoding a 4K PNG
        # and its 24 MB depth file when a shot first comes into view stalled
        # the viewport for seconds.
        size = self.layer_size()
        for k in np.flatnonzero(self.used):
            if progress is not None:
                progress(f"Saving shots… {int(k) + 1}/{len(self)}")
            z = self.depth(int(k), size)
            np.save(folder / f"layer_{int(k):03d}.npy",
                    np.where(z < measured.SKY / 2, z, np.inf).astype(np.float16))
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(self.colour(int(k), size), cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                buf.tofile(str(folder / f"layer_{int(k):03d}.jpg"))
        self.folder = folder
        return folder


def is_shot_scene(folder: str | Path) -> bool:
    return (Path(folder) / "shots.json").is_file()


def load(folder: str | Path) -> ShotScene:
    folder = Path(folder)
    meta = json.loads((folder / "shots.json").read_text(encoding="utf-8"))
    d = np.load(folder / "shots.npz")
    planes = [(row[:3].astype(np.float64), float(row[3])) for row in d["planes"]]
    return ShotScene(names=list(meta["names"]), rotations=d["rotations"], centres=d["centres"],
                     cameras=d["cameras"], size=tuple(int(v) for v in d["size"]), depth_a=d["depth_a"],
                     depth_b=d["depth_b"], gain=d["gain"], trust=d["trust"], used=d["used"].astype(bool),
                     axes=d["axes"], target=d["target"], scale=float(d["scale"]), offset=d["offset"],
                     planes=planes, folder=folder)
