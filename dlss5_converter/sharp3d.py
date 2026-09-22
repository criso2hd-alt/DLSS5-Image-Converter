"""Apple SHARP as the optional "High quality" source for the 3D tab.

SHARP (Mescheder et al., Apple, 2025) predicts a whole 3D Gaussian scene from
one image in a single network pass: positions, sizes, orientations, colour and
opacity for ~1.2M Gaussians, on a two-layer 768x768 grid. Against our
heuristic splats (one per pixel from Depth Anything) it is far cleaner at
silhouettes, with no stair-step fringe, and it already covers a little of what
sits just behind each edge; seen from the side its geometry holds together
instead of showing slanted sheets. It is softer in fine texture, and it uses
its own depth (a Depth Pro backbone), not ours.

Weights: a community ONNX export of Apple's release (pearsonkyle/Sharp-onnx,
FP16, 1.32 GB). Apple's model licence is research / non-commercial, with
redistribution allowed with attribution, so it is never bundled: the user
downloads it on demand, like LaMa. Runs through ONNX Runtime (DirectML first),
~20 s per image on an RTX 4080.

Output conversion: SHARP's means are in an OpenCV camera (x right, y down,
z forward) with x/z spanning -1..1 across the image, which is squashed to a
1536 square on input. Depth is remapped the way our own scenes are (inverse
depth normalised into NEAR..FAR), and every covariance is carried through the
Jacobian of that mapping, so a SHARP scene drops into the same renderer,
camera, effects and fill as ours.
"""

from __future__ import annotations

import threading

import numpy as np

from . import splat3d

REPO = "pearsonkyle/Sharp-onnx"
FILENAME = "sharp_fp16.onnx"
SIZE_LABEL = "1.3 GB"
INPUT = 1536
_LOCK = threading.RLock()
_SESSION = None


def _cache_dir() -> str:
    from . import paths
    d = paths.model_cache_dir() / "sharp"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def is_downloaded() -> bool:
    from huggingface_hub import try_to_load_from_cache
    try:
        return isinstance(try_to_load_from_cache(REPO, FILENAME, cache_dir=_cache_dir()), str)
    except Exception:  # noqa: BLE001
        return False


def download(progress=None) -> str:
    from .depth_engine import enable_system_trust_store
    enable_system_trust_store()
    from huggingface_hub import hf_hub_download
    if progress:
        progress(f"Downloading the SHARP model ({SIZE_LABEL})…")
    return hf_hub_download(REPO, FILENAME, cache_dir=_cache_dir())


def _session():
    """One shared session: 1.3 GB of weights is not something to load twice."""
    global _SESSION
    with _LOCK:
        if _SESSION is None:
            import onnxruntime as ort
            from .onnx_depth import _providers
            from . import gpus
            so = ort.SessionOptions()
            so.log_severity_level = 3
            path = download()
            last = None
            for prov in _providers():
                try:
                    _SESSION = ort.InferenceSession(path, sess_options=so, providers=gpus.ort_providers([prov]))
                    break
                except Exception as error:  # noqa: BLE001 - try the next provider
                    last = error
            if _SESSION is None:
                raise RuntimeError(f"SHARP could not start: {last}")
        return _SESSION


def predict(rgb8: np.ndarray):
    """Run SHARP: (means, scales, quaternions wxyz, linear colours, opacity)."""
    import cv2
    x = cv2.resize(rgb8, (INPUT, INPUT), interpolation=cv2.INTER_AREA)
    x = x.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    out = _session().run(None, {"image": x, "disparity_factor": np.array([1.0], np.float32)})
    return [np.asarray(o[0], np.float32) for o in out]


def to_scene(pred, width: int, height: int, depth_contrast: float = 1.0) -> splat3d.SplatScene:
    """Carry SHARP's Gaussians into our camera frame and depth range."""
    m, sc, q, col, a = pred
    focal = splat3d.focal_px(height)
    ax, by = (width / 2) / focal, (height / 2) / focal
    z = m[:, 2]
    znear, zfar = np.percentile(z, 1), np.percentile(z, 99)
    zc = np.clip(z, znear, zfar)
    # Normalised disparity d = p/z + r, contrast around 0.5 like our slider,
    # then 1/Z' = 1/FAR + D*d. Everything stays linear in 1/z, which keeps the
    # Jacobian simple: Z' = z / (alpha*z + beta).
    p = 1.0 / (1.0 / znear - 1.0 / zfar)
    r = -p / zfar
    k = float(depth_contrast)
    dd = 1.0 / splat3d.NEAR - 1.0 / splat3d.FAR
    beta = dd * k * p
    alpha = 1.0 / splat3d.FAR + dd * (k * r + 0.5 * (1.0 - k))
    den = alpha * zc + beta
    g = 1.0 / den
    gp = -alpha / den ** 2
    zp = zc * g
    dz = beta / den ** 2
    pos = np.stack([ax * m[:, 0] * g, -by * m[:, 1] * g, -zp], -1).astype(np.float32)

    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rot = np.stack([
        np.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)], -1),
        np.stack([2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)], -1),
        np.stack([2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)], -1)], 1)
    cov = rot @ (sc[:, :, None] ** 2 * np.transpose(rot, (0, 2, 1)))
    jac = np.zeros((len(m), 3, 3), np.float32)
    jac[:, 0, 0] = ax * g
    jac[:, 0, 2] = ax * m[:, 0] * gp
    jac[:, 1, 1] = -by * g
    jac[:, 1, 2] = -by * m[:, 1] * gp
    jac[:, 2, 2] = -dz
    cov = jac @ cov @ np.transpose(jac, (0, 2, 1))
    cov6 = np.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2],
                     cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], -1).astype(np.float32)
    srgb = np.where(col <= 0.0031308, col * 12.92,
                    1.055 * np.power(np.maximum(col, 0.0), 1 / 2.4) - 0.055)
    return splat3d.SplatScene(pos, np.clip(srgb, 0, 1).astype(np.float32),
                              np.clip(a, 0, 1).astype(np.float32), cov6, len(pos), focal=focal)


def build_scene(rgb8: np.ndarray, renderer: splat3d.SplatRenderer,
                depth_contrast: float = 1.0) -> splat3d.SplatScene:
    """Predict, convert, then read SHARP's own depth back from the photo camera.

    The fill needs the photo's depth map (to keep fills behind what the photo
    shows) and the scene planes; for a SHARP scene both come from rendering it
    from the original viewpoint."""
    from .camera3d import Camera
    h, w = rgb8.shape[:2]
    scene = to_scene(predict(rgb8), w, h, depth_contrast)
    renderer.set_scene(scene)
    cam = Camera(position=(0, 0, 0), target=(0, 0, -1), fov_degrees=splat3d.FOV)
    cam.near, cam.far = 0.05, 60.0
    view, proj = cam.view_matrix(), cam.projection_matrix(w / max(h, 1))
    _rgb, _alpha, dist = renderer.render_coverage(view, proj, (w, h))
    photo_z = np.where(dist > 0, dist, splat3d.FAR).astype(np.float32)
    scene.photo_z = photo_z
    scene.planes = splat3d.fit_planes(splat3d._unproject(photo_z, scene.focal),
                                      photo_z < splat3d.FAR * 0.98)
    return scene
