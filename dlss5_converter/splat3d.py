"""Gaussian-splat renderer and a heuristic single-image splat builder.

Replaces the depth mesh as the 3D tab's draw path. A mesh has to connect every
pixel to its neighbour, so at a depth edge it either stretches a rubber sheet
from the face to the wall behind it, or gets cut and leaves a hole. Splats never
connect: each pixel becomes a small soft disc at its own depth, a silhouette
simply separates, and the gap is covered by a second set of splats that stand in
for the background the photo never saw.

The renderer is standard 3D Gaussian splatting (project each 3-D covariance to a
2-D ellipse, sort back to front, alpha-blend), so a scene predicted by a real
model such as Apple's SHARP can be drawn by the same code later. Only the
builder here is heuristic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

FOV = 55.0
NEAR, FAR = 1.0, 6.0

SHADER = """
struct U {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    viewport: vec4<f32>,   // x w, y h, z fx (px), w fy (px)
    extra: vec4<f32>,      // x view mode (0 colour, 1 depth, 2 points), y near, z far
};
struct Splat {
    pos: vec4<f32>,        // xyz, w opacity
    color: vec4<f32>,      // rgb, w unused
    cov_a: vec4<f32>,      // xx, xy, xz, yy
    cov_b: vec4<f32>,      // yz, zz, -, -
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
@group(0) @binding(2) var<storage, read> order: array<u32>;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) offset: vec2<f32>,     // pixels from the splat centre, y up
    @location(1) conic: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) depth: f32,
};

fn turbo(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    let r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234
            + x * (-152.94239396 + x * 59.28637943))));
    let g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333
            + x * (4.27729857 + x * 2.82956604))));
    let b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771
            + x * (-89.90310912 + x * 27.34824973))));
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

struct FOut {
    @location(0) colour: vec4<f32>,
    // Coverage-weighted view depth, (depth * alpha, 0, 0, alpha), blended like
    // colour. Dividing r by a gives the visible surface's depth, which is what
    // the fog raymarch, particle occlusion and the hole finder all read.
    @location(1) depth: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    var o: VOut;
    let s = splats[order[ii]];
    let t = u.view * vec4<f32>(s.pos.xyz, 1.0);
    let tz = -t.z;
    // Behind the camera or too close: emit a degenerate quad.
    if (tz < 0.02) {
        o.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return o;
    }
    let fx = u.viewport.z;
    let fy = u.viewport.w;
    let V = mat3x3<f32>(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);
    let S = mat3x3<f32>(
        vec3<f32>(s.cov_a.x, s.cov_a.y, s.cov_a.z),
        vec3<f32>(s.cov_a.y, s.cov_a.w, s.cov_b.x),
        vec3<f32>(s.cov_a.z, s.cov_b.x, s.cov_b.y));
    let C = V * S * transpose(V);
    // Jacobian of (fx*x/tz, fy*y/tz) with tz = -z, as columns.
    let J = mat3x3<f32>(
        vec3<f32>(fx / tz, 0.0, 0.0),
        vec3<f32>(0.0, fy / tz, 0.0),
        vec3<f32>(fx * t.x / (tz * tz), fy * t.y / (tz * tz), 0.0));
    let P = J * C * transpose(J);
    // The 0.3 px^2 floor is the usual anti-alias low-pass: without it a splat
    // seen edge-on collapses below a pixel and flickers between frames.
    let a = P[0][0] + 0.3;
    let b = P[0][1];
    let c = P[1][1] + 0.3;
    let det = a * c - b * b;
    if (det <= 0.0) {
        o.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return o;
    }
    let mid = 0.5 * (a + c);
    let lam = mid + sqrt(max(0.1, mid * mid - det));
    var radius = min(ceil(3.0 * sqrt(lam)), 512.0);
    var conic = vec3<f32>(c, -b, a) / det;
    if (u.extra.x > 1.5) {
        // Point cloud: every splat as a fixed dot, so the structure of the
        // scene (what is real, what was filled) can be read at a glance.
        radius = 1.5;
        conic = vec3<f32>(1.2, 0.0, 1.2);
    }
    var corner = vec2<f32>(-1.0, -1.0);
    if (vi == 1u) { corner = vec2<f32>(1.0, -1.0); }
    if (vi == 2u) { corner = vec2<f32>(-1.0, 1.0); }
    if (vi == 3u) { corner = vec2<f32>(1.0, 1.0); }
    let off = corner * radius;
    let clip = u.proj * t;
    o.clip = vec4<f32>(clip.xy + off * 2.0 / u.viewport.xy * clip.w, clip.zw);
    o.offset = off;
    o.conic = conic;
    o.color = vec4<f32>(s.color.rgb, s.pos.w);
    o.depth = tz;
    if (u.extra.x > 0.5 && u.extra.x < 1.5) {
        let near = u.extra.y;
        let far = u.extra.z;
        let t = (1.0 / tz - 1.0 / far) / max(1.0 / near - 1.0 / far, 1e-6);
        o.color = vec4<f32>(turbo(t), s.pos.w);
    }
    return o;
}

@fragment
fn fs_main(i: VOut) -> FOut {
    let d = i.offset;
    let power = -0.5 * (i.conic.x * d.x * d.x + i.conic.z * d.y * d.y) - i.conic.y * d.x * d.y;
    let alpha = min(0.99, i.color.a * exp(power));
    if (alpha < 1.0 / 255.0) { discard; }
    var o: FOut;
    o.colour = vec4<f32>(i.color.rgb * alpha, alpha);
    o.depth = vec4<f32>(i.depth * alpha, 0.0, 0.0, alpha);
    return o;
}
"""

# Back-to-front order on the GPU: a counting sort on 16-bit quantised view
# distance. A CPU argsort of 2.4M floats took 140 ms a frame, which is the whole
# difference between a slideshow and a live viewport. Splats that share a bucket
# (1/65536 of the scene depth) blend in arbitrary order, which is invisible.
SORT_SHADER = """
struct SU {
    view_z: vec4<f32>,     // row 2 of the view matrix
    range: vec4<f32>,      // x nearest distance, y 1/(far-near), z count
};
@group(0) @binding(0) var<uniform> su: SU;
@group(0) @binding(1) var<storage, read> splats: array<vec4<f32>>;   // stride 4 vec4
@group(0) @binding(2) var<storage, read_write> keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> hist: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> order: array<u32>;

const BINS: u32 = 65536u;

@compute @workgroup_size(256)
fn cs_keys(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x + g.y * 65535u * 256u;
    if (i >= u32(su.range.z)) { return; }
    let p = splats[i * 4u].xyz;
    let dist = -(dot(su.view_z.xyz, p) + su.view_z.w);
    let f = clamp((dist - su.range.x) * su.range.y, 0.0, 1.0);
    let k = (BINS - 1u) - u32(f * f32(BINS - 1u));      // farthest -> key 0
    keys[i] = k;
    atomicAdd(&hist[k], 1u);
}

var<workgroup> partial: array<u32, 256>;

@compute @workgroup_size(256)
fn cs_scan(@builtin(local_invocation_id) l: vec3<u32>) {
    let t = l.x;
    var sum = 0u;
    for (var j = 0u; j < 256u; j++) { sum += atomicLoad(&hist[t * 256u + j]); }
    partial[t] = sum;
    workgroupBarrier();
    // Hillis-Steele inclusive scan over the 256 per-thread sums.
    for (var off = 1u; off < 256u; off = off * 2u) {
        var v = 0u;
        if (t >= off) { v = partial[t - off]; }
        workgroupBarrier();
        partial[t] += v;
        workgroupBarrier();
    }
    var run = partial[t] - sum;                        // exclusive start of this chunk
    for (var j = 0u; j < 256u; j++) {
        let b = t * 256u + j;
        let c = atomicLoad(&hist[b]);
        atomicStore(&hist[b], run);
        run += c;
    }
}

@compute @workgroup_size(256)
fn cs_scatter(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x + g.y * 65535u * 256u;
    if (i >= u32(su.range.z)) { return; }
    let slot = atomicAdd(&hist[keys[i]], 1u);
    order[slot] = i;
}
"""


# -- scene -------------------------------------------------------------------

@dataclass
class SplatScene:
    positions: np.ndarray    # N x 3
    colors: np.ndarray       # N x 3, linear-ish 0..1 (display space, as the photo)
    opacity: np.ndarray      # N
    cov: np.ndarray          # N x 6: xx xy xz yy yz zz
    n_front: int             # splats from the visible photo; the rest is backfill
    photo_z: np.ndarray | None = None   # the photo's view-Z map, for culling fills
    focal: float = 0.0
    planes: list = None                  # [(normal, offset)] from fit_planes

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    def keep(self, mask: np.ndarray) -> None:
        self.n_front = int(mask[: self.n_front].sum())
        self.positions = self.positions[mask]
        self.colors = self.colors[mask]
        self.opacity = self.opacity[mask]
        self.cov = self.cov[mask]

    def packed(self) -> np.ndarray:
        n = len(self)
        out = np.zeros((n, 16), np.float32)
        out[:, 0:3] = self.positions
        out[:, 3] = self.opacity
        out[:, 4:7] = self.colors
        out[:, 8:12] = self.cov[:, 0:4]
        out[:, 12:14] = self.cov[:, 4:6]
        return out

    def save_ply(self, path: str) -> None:
        """Write a minimal xyz/rgb point file, for inspecting the scene in other
        tools. Not the 3DGS training format (that stores SH and log-scales)."""
        rgb = (np.clip(self.colors, 0, 1) * 255).astype(np.uint8)
        head = ("ply\nformat binary_little_endian 1.0\n"
                f"element vertex {len(self)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n").encode()
        rec = np.zeros(len(self), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        rec["p"] = self.positions
        rec["c"] = rgb
        with open(path, "wb") as fh:
            fh.write(head)
            fh.write(rec.tobytes())


def disparity_to_depth(disp: np.ndarray, near: float = NEAR, far: float = FAR) -> np.ndarray:
    """Normalised inverse depth -> view Z, interpolated in disparity so parallax
    stays even (screen motion goes as 1/Z)."""
    disp = np.clip(disp.astype(np.float32), 0.0, 1.0)
    inv = (1.0 / far) + ((1.0 / near) - (1.0 / far)) * disp
    return (1.0 / np.maximum(inv, 1e-6)).astype(np.float32)


def focal_px(height: int, fov_deg: float = FOV) -> float:
    return height / (2.0 * math.tan(math.radians(fov_deg) * 0.5))


def _unproject(z: np.ndarray, focal: float) -> np.ndarray:
    h, w = z.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (xs - (w - 1) * 0.5) * z / focal
    y = -(ys - (h - 1) * 0.5) * z / focal
    return np.stack([x, y, -z], -1)


def _snap_flying_pixels(disp: np.ndarray, radius: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Depth models blur silhouettes: the pixels on an edge get a depth halfway
    between the face and the wall. As splats those float in mid-air as a
    visible fringe. Snap each edge pixel to whichever side it is closer to."""
    k = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    hi = cv2.dilate(disp, k)
    lo = cv2.erode(disp, k)
    edge = (hi - lo) > 0.04
    snapped = np.where(disp - lo < hi - disp, lo, hi)
    return np.where(edge, snapped, disp).astype(np.float32), edge


def _surface_cov(pos: np.ndarray, footprint: np.ndarray, edge: np.ndarray,
                 thin: float = 0.12) -> np.ndarray:
    """Flat discs lying on the surface. Normal from the position gradient; on
    depth edges the gradient is meaningless, so those discs face the camera."""
    dx = np.gradient(pos, axis=1)
    dy = np.gradient(pos, axis=0)
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    n[edge] = (0.0, 0.0, 1.0)
    # Keep discs from turning fully edge-on to the original lens: a grazing
    # surface would otherwise shrink to slivers and open cracks.
    view = -pos / (np.linalg.norm(pos, axis=-1, keepdims=True) + 1e-9)
    facing = np.abs((n * view).sum(-1, keepdims=True))
    n = np.where(facing < 0.25, view, n)
    n = n.reshape(-1, 3)
    st = footprint.reshape(-1)
    sn = st * thin
    nn = n[:, :, None] * n[:, None, :]
    eye = np.eye(3, dtype=np.float32)[None]
    cov = (st[:, None, None] ** 2) * (eye - nn) + (sn[:, None, None] ** 2) * nn
    return np.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2],
                     cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], -1).astype(np.float32)


def background_plate(rgb8: np.ndarray, disp: np.ndarray, inpainter=None,
                     reach: float = 0.06, step: float = 0.05):
    """Estimate what sits behind the foreground: its depth and colour.

    Depth: a wide min-filter on disparity drops every object narrower than
    `reach` and keeps the surface behind it, which is then smoothed so it reads
    as a continuous wall rather than a staircase.

    Colour: the hole handed to the inpainter is the WHOLE occluder, not just a
    band around its edge. A band leaves foreground pixels on the hole's inner
    border, and the inpainter then extends the face or shirt into the wall;
    that is the "stretched" look. With the full object masked, the only
    context left is background.
    """
    h, w = disp.shape
    r = max(3, int(reach * max(h, w)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    bg = cv2.erode(disp, k)
    bg = cv2.dilate(bg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r | 1, r | 1)))
    bg = np.minimum(cv2.GaussianBlur(bg, (0, 0), r * 0.5), disp)
    occluder = (disp - bg) > step
    occluder = cv2.morphologyEx(occluder.astype(np.uint8), cv2.MORPH_CLOSE,
                                np.ones((5, 5), np.uint8)) > 0
    # Thin things (wires, hair, twigs) come out of the depth model blurred into
    # the background, so they miss the occluder test yet still sit in the
    # photo. Left as inpaint context they get copied into the fill as ghosts.
    # Any strong depth edge is therefore masked too.
    gx = cv2.Sobel(disp, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(disp, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.hypot(gx, gy) > 0.06
    pad = max(3, int(0.004 * max(h, w))) | 1
    hole = cv2.dilate((occluder | edges).astype(np.uint8), np.ones((pad, pad), np.uint8)) > 0
    if inpainter is not None and hole.any():
        try:
            colour = inpainter(rgb8, hole)
        except Exception:  # noqa: BLE001 - the classical fill is the floor
            colour = cv2.inpaint(rgb8, hole.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
    else:
        colour = cv2.inpaint(rgb8, hole.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
    return bg.astype(np.float32), colour, occluder


def build_splats(rgb8: np.ndarray, disp: np.ndarray, inpainter=None,
                 depth_contrast: float = 1.0, backfill: bool = True) -> SplatScene:
    """One splat per pixel of the photo, plus backfill splats behind every
    occluder. The backfill is only drawn where the photo has something in
    front of it, so from the original viewpoint it is invisible."""
    h, w = disp.shape
    if rgb8.shape[:2] != (h, w):
        rgb8 = cv2.resize(rgb8, (w, h), interpolation=cv2.INTER_AREA)
    d = disp.astype(np.float32)
    d = (d - d.min()) / (np.ptp(d) + 1e-6)
    if depth_contrast != 1.0:
        d = np.clip((d - 0.5) * depth_contrast + 0.5, 0.0, 1.0)
    d, edge = _snap_flying_pixels(d, radius=max(2, int(0.0015 * max(h, w))))

    focal = focal_px(h)
    z = disparity_to_depth(d)
    pos = _unproject(z, focal)
    # Sigma of ~0.6 px keeps neighbours overlapping enough to leave no cracks
    # while staying as sharp as the source.
    foot = (z / focal) * 0.62
    rgb = rgb8.astype(np.float32) / 255.0

    positions = [pos.reshape(-1, 3)]
    colors = [rgb.reshape(-1, 3)]
    opac = [np.ones(h * w, np.float32)]
    covs = [_surface_cov(pos, foot, edge)]
    n_front = h * w

    if backfill:
        bg, colour, occluder = background_plate(rgb8, d, inpainter)
        if occluder.any():
            zb = disparity_to_depth(bg)
            pb = _unproject(zb, focal)
            fb = (zb / focal) * 0.7
            covb = _surface_cov(pb, fb, np.zeros_like(occluder))
            sel = occluder.reshape(-1)
            positions.append(pb.reshape(-1, 3)[sel])
            colors.append((colour.astype(np.float32) / 255.0).reshape(-1, 3)[sel])
            opac.append(np.ones(int(sel.sum()), np.float32))
            covs.append(covb[sel])

    return SplatScene(np.concatenate(positions).astype(np.float32),
                      np.concatenate(colors).astype(np.float32),
                      np.concatenate(opac), np.concatenate(covs), n_front,
                      photo_z=z, focal=focal,
                      planes=fit_planes(pos, ~edge & (z < FAR * 0.98)))


# -- fill as the camera moves ---------------------------------------------------

# -- scene layout: planes ---------------------------------------------------------

def _pixel_normals(pos: np.ndarray) -> np.ndarray:
    dx = np.gradient(pos, axis=1)
    dy = np.gradient(pos, axis=0)
    n = np.cross(dx, dy)
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)


def fit_planes(pos: np.ndarray, valid: np.ndarray, max_planes: int = 4,
               min_share: float = 0.06, seed: int = 0) -> list[tuple[np.ndarray, float]]:
    """Find the big flat surfaces of the scene: the ground, walls, a backdrop.

    Sequential RANSAC on a sample of the photo's 3-D points. Each round tries
    random three-point planes, keeps the one most points agree with, refines
    it by least squares, removes its points and looks again. A plane must hold
    at least `min_share` of the scene to count, so clutter never becomes one.

    Agreement is RELATIVE to depth (2% of the distance): depth from a single
    image is only good to a proportion, so a fixed tolerance would accept
    everything far away and nothing near. Points must also face the same way
    as the plane, so a wall is not swallowed by the floor it touches.

    Returns planes as (unit normal n, offset c) with n . p + c = 0, in the
    photo camera's frame (the scene's world frame)."""
    rng = np.random.default_rng(seed)
    normals = _pixel_normals(pos)
    idx = np.flatnonzero(valid.reshape(-1))
    if idx.size < 500:
        return []
    take = rng.choice(idx, size=min(idx.size, 30000), replace=False)
    P = pos.reshape(-1, 3)[take].astype(np.float64)
    N = normals.reshape(-1, 3)[take].astype(np.float64)
    scale = np.abs(P[:, 2]) + 1e-6
    alive = np.ones(len(P), bool)
    planes: list[tuple[np.ndarray, float]] = []
    far_cut = float(np.percentile(P[:, 2], 30))    # z is negative: 30th pct = far 70%
    for _ in range(max_planes + 3):
        live = np.flatnonzero(alive)
        if live.size < min_share * len(P):
            break
        best, best_count = None, 0
        for _ in range(400):
            a, b, c = P[rng.choice(live, 3, replace=False)]
            n = np.cross(b - a, c - a)
            ln = np.linalg.norm(n)
            if ln < 1e-9:
                continue
            n /= ln
            d = -n @ a
            res = np.abs(P[live] @ n + d) / scale[live]
            agree = (res < 0.02) & (np.abs(N[live] @ n) > 0.8)
            count = int(agree.sum())
            if count > best_count:
                best, best_count = (n, d), count
        if best is None or best_count < min_share * len(P):
            break
        n, d = best
        res = np.abs(P @ n + d) / scale
        inl = alive & (res < 0.02) & (np.abs(N @ n) > 0.8)
        # Least-squares refit on the inliers: centroid plus smallest singular
        # vector, which is the plane closest to all of them at once.
        Q = P[inl]
        cen = Q.mean(0)
        _u, _s, vt = np.linalg.svd(Q - cen, full_matrices=False)
        n = vt[-1]
        if n @ cen > 0:
            n = -n          # normal faces the camera, which sits at the origin
        alive &= ~inl
        # Only STRUCTURE counts as a plane: the ground, or a wall standing
        # behind things. Random alignments pass RANSAC too (a torso and a car
        # bumper at the same depth, a slanted band across trees and sky), and
        # treating those as planes stops anything being filled behind them.
        # (Normals are in camera space, so "horizontal" assumes a roughly
        # level camera, which is what photos almost always are.)
        ground = abs(n[1]) > 0.85
        wall = (abs(n[1]) < 0.35 and float(np.median(Q[:, 2])) <= far_cut
                and len(Q) >= 0.10 * len(P))
        if ground or wall:
            planes.append((n.astype(np.float32), float(-n @ cen)))
            if len(planes) >= max_planes:
                break
    return planes


def _push_pull(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Fill every pixel from the weighted known ones, smoothly and fast.

    Weighted image pyramid: average known values down to a tiny image where
    everything is covered, then walk back up, keeping real values where they
    exist and taking the coarser estimate where they do not."""
    levels = [(values * weight, weight)]
    while min(levels[-1][1].shape) > 4:
        vw, w = levels[-1]
        levels.append((cv2.pyrDown(vw), cv2.pyrDown(w)))
    vw, w = levels[-1]
    filled = vw / np.maximum(w, 1e-6)
    if (w <= 1e-6).any():
        filled[w <= 1e-6] = float(filled[w > 1e-6].mean()) if (w > 1e-6).any() else FAR
    for vw, w in reversed(levels[:-1]):
        up = cv2.resize(filled, (w.shape[1], w.shape[0]), interpolation=cv2.INTER_LINEAR)
        own = vw / np.maximum(w, 1e-6)
        a = np.clip(w * 4.0, 0.0, 1.0)
        filled = own * a + up * (1.0 - a)
    return filled.astype(np.float32)


def _far_side_depth(dist: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Give every hole pixel the depth of the surface BEHIND it.

    A disocclusion is by definition something behind the foreground. Its border
    has two sides: the subject it slipped out from behind, and the wall it
    reveals. Only the wall is the right depth. Growing from the whole border
    gave the pixels next to the subject the subject's depth, and those fills
    stuck out backwards from it as long streaks when seen from the side.

    So the sources are only pixels within 10% of the deepest surface nearby,
    and the hole is filled smoothly from those alone."""
    back = _wall_depth(dist, ~hole)
    return np.where(hole, back, dist).astype(np.float32)


def _wall_depth(dist: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Estimated depth of the backmost surface under EVERY pixel, from the
    known pixels within 10% of the deepest surface near them."""
    h, w = dist.shape
    r = max(8, int(0.03 * max(h, w)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    local_far = cv2.dilate(np.where(known, dist, 0.0).astype(np.float32), kernel)
    far_src = known & (dist >= local_far * 0.9)
    if not far_src.any():
        return np.full(dist.shape, float(dist[known].max()) if known.any() else FAR, np.float32)
    back = _push_pull(np.where(far_src, dist, 0.0).astype(np.float32),
                      far_src.astype(np.float32))
    # Never nearer than the far sources around it: a hole can only reveal
    # what is behind, so the estimate is held at the local far surface.
    return np.maximum(back, np.where(known, 0.0, local_far * 0.9)).astype(np.float32)


def _disc_cov(normals: np.ndarray, sigma: np.ndarray, thin: float = 0.12) -> np.ndarray:
    """Flat discs of radius `sigma` lying in the plane of each normal."""
    nn = normals[:, :, None] * normals[:, None, :]
    eye = np.eye(3, dtype=np.float32)[None]
    cov3 = (sigma[:, None, None] ** 2) * ((eye - nn) + (thin ** 2) * nn)
    return np.stack([cov3[:, 0, 0], cov3[:, 0, 1], cov3[:, 0, 2],
                     cov3[:, 1, 1], cov3[:, 1, 2], cov3[:, 2, 2]], -1).astype(np.float32)


def _on_planes(pos: np.ndarray, planes, tol: float = 0.025) -> np.ndarray:
    """Which photo pixels lie on one of the scene planes (relative tolerance)."""
    on = np.zeros(pos.shape[:2], bool)
    scale = np.abs(pos[..., 2]) + 1e-6
    for n, c in planes or []:
        on |= np.abs(pos @ n + c) / scale < tol
    return on


def _planes_behind(origin, dirs, planes, t_min):
    """(depth, normal) of the first plane behind t_min per ray; depth 0 = none."""
    best = np.zeros(len(dirs), np.float32)
    normal = np.zeros((len(dirs), 3), np.float32)
    for n, c in planes or []:
        denom = dirs @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -(origin @ n + c) / denom
        ok = np.isfinite(t) & (t > t_min) & (t < FAR * 2.5)
        take = ok & ((best == 0) | (t < best))
        best[take] = t[take]
        normal[take] = n
    return best, normal


def fill_backplate(scene: SplatScene, renderer: "SplatRenderer", inpainter=None,
                   reach: float = 0.12) -> int:
    """Paint the wall behind every subject, from the original viewpoint.

    The path fill only covers what the camera move actually reveals; a gentle
    move never looks far behind a subject, so seen from any other angle the
    back wall shows a black subject-shaped shadow. This pass fills a band
    behind each silhouette up front: every photo pixel that has something at
    least 10% deeper within `reach` of the image is treated as covering a
    hidden wall there. Depth comes from the far side only (`_far_side_depth`),
    and the inpainter sees nothing nearer than the wall, so it continues the
    background instead of copying the subject into it."""
    if scene.photo_z is None:
        return 0
    z = scene.photo_z
    h, w = z.shape
    if scene.n_front != h * w:
        return 0          # already baked or cleaned: the photo splats are not a grid now
    rgb8 = (np.clip(scene.colors[: h * w].reshape(h, w, 3), 0, 1) * 255 + 0.5).astype(np.uint8)
    r = max(8, int(reach * max(h, w)))
    # A max filter this wide is slow at full size; the band is smooth, so it is
    # found at quarter resolution and scaled back up.
    small = cv2.resize(z, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    rs = max(2, r // 4)
    far_ref = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rs + 1, 2 * rs + 1)))
    far_ref = cv2.resize(far_ref, (w, h), interpolation=cv2.INTER_LINEAR)
    hole = z < far_ref * 0.9

    # Scene layout: every pixel that is not on one of the scene's planes is an
    # object, and wherever a plane continues behind an object, that is what
    # the object hides. This is what fills the road under a car: the old rule
    # only knew "the far wall", so the floor behind objects was left open.
    f = scene.focal
    ys_all, xs_all = np.mgrid[0:h, 0:w].astype(np.float32)
    dirs = np.stack([(xs_all - (w - 1) * 0.5) / f, -(ys_all - (h - 1) * 0.5) / f,
                     -np.ones_like(xs_all)], -1).reshape(-1, 3)
    plane_t = np.zeros(h * w, np.float32)
    plane_n = np.zeros((h * w, 3), np.float32)
    if scene.planes:
        pos = _unproject(z, f)
        obj = ~_on_planes(pos, scene.planes)
        plane_t, plane_n = _planes_behind(np.zeros(3, np.float32), dirs, scene.planes,
                                          z.reshape(-1) * 1.05)
        plane_t[~obj.reshape(-1)] = 0.0
        hole |= plane_t.reshape(h, w) > 0

    hole = cv2.morphologyEx(hole.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    if int(hole.sum()) < 64:
        return 0
    depth = _far_side_depth(z, hole)
    use_plane = (plane_t.reshape(h, w) > 0) & hole
    depth = np.where(use_plane, plane_t.reshape(h, w), depth)
    # Hide every subject from the inpainter, not just the band behind its
    # edge. With only the band masked it could see the rest of the person and
    # continued their skin and clothes into the wall: a ghost of the subject
    # that peeks out from behind them as the camera moves.
    wall = _wall_depth(z, ~hole)
    # 0.95 and a margin of ~1% of the image: hair and the soft depth halo
    # round a silhouette sit just outside a tight mask, and a dark rim left in
    # the context is extended across the wall as a shadow of the subject.
    paint = hole | (z < wall * 0.95) | use_plane
    m = max(9, int(0.012 * max(h, w))) | 1
    paint = cv2.dilate(paint.astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (m, m))) > 0
    if inpainter is not None:
        try:
            filled = inpainter(rgb8, paint)
        except Exception:  # noqa: BLE001 - the classical fill is the floor
            filled = cv2.inpaint(rgb8, paint.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
    else:
        filled = cv2.inpaint(rgb8, paint.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
    ys, xs = np.nonzero(hole)
    tz = depth[ys, xs]
    world = np.stack([(xs - (w - 1) * 0.5) * tz / f, -(ys - (h - 1) * 0.5) * tz / f, -tz],
                     -1).astype(np.float32)
    cov = _fill_cov(world, tz / f, plane_n.reshape(h, w, 3)[ys, xs], np.array([0.0, 0.0, 1.0]))
    scene.positions = np.concatenate([scene.positions, world])
    scene.colors = np.concatenate([scene.colors, filled[ys, xs].astype(np.float32) / 255.0])
    scene.opacity = np.concatenate([scene.opacity, np.ones(len(ys), np.float32)])
    scene.cov = np.concatenate([scene.cov, cov])
    renderer.set_scene(scene)
    return int(len(ys))


def _fill_cov(world: np.ndarray, pixel: np.ndarray, plane_n: np.ndarray,
              facing: np.ndarray, cam=None) -> np.ndarray:
    """Covariances for fill splats.

    On a scene plane the disc lies IN the plane, sized to the pixel's
    footprint on that plane (larger when the plane is seen at a grazing angle,
    so a far stretch of road is still covered). Elsewhere it faces the camera
    that made it. Camera-facing discs are what looked like stripes of cards
    from the side; plane-aligned ones stay a continuous floor or wall."""
    on_plane = np.abs(plane_n).sum(-1) > 0.5
    normals = np.where(on_plane[:, None], plane_n, facing[None, :]).astype(np.float32)
    origin = np.zeros(3, np.float32) if cam is None else np.asarray(cam, np.float32)
    ray = world - origin
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True) + 1e-9
    cos = np.abs((ray * normals).sum(-1))
    sigma = pixel * 0.8 / np.clip(np.where(on_plane, cos, 1.0), 0.3, 1.0)
    return _disc_cov(normals, sigma.astype(np.float32))


def _occludes_photo(scene: SplatScene, world: np.ndarray, margin: float = 0.03) -> np.ndarray:
    """True for points that would sit in FRONT of the photo's own surface when
    seen from the original camera. A fill only ever stands in for something
    hidden, so such a point is wrong by construction: it is the floating slab
    that appears over a face or a chest."""
    if scene.photo_z is None:
        return np.zeros(len(world), bool)
    h, w = scene.photo_z.shape
    z = -world[:, 2]
    ok = z > 1e-3
    zs = np.where(ok, z, 1.0)
    u = np.round(world[:, 0] * scene.focal / zs + (w - 1) * 0.5).astype(np.int64)
    v = np.round(-world[:, 1] * scene.focal / zs + (h - 1) * 0.5).astype(np.int64)
    inside = ok & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    bad = np.zeros(len(world), bool)
    idx = np.nonzero(inside)[0]
    bad[idx] = z[idx] < scene.photo_z[v[idx], u[idx]] * (1.0 - margin)
    return bad


def fill_view(scene: SplatScene, renderer: "SplatRenderer", view: np.ndarray,
              proj: np.ndarray, size, inpainter=None, min_hole: int = 64) -> int:
    """Render one camera pose, inpaint what it reveals, and add the fill to the
    scene as splats standing where the hidden surface would be. Returns how
    many splats were added (0 when this pose showed no holes)."""
    w, h = int(size[0]), int(size[1])
    rgb, alpha, dist = renderer.render_coverage(view, proj, (w, h))
    # 0.85, not 0.5: a pixel only partly covered shows the dark background
    # through it as a speck, so it counts as a hole to fill.
    hole = alpha < (0.85 if _is_grid(scene) else 0.5)
    # Past the photo's own frame edges is a hole too; the far-side depth
    # handles it the same way, which extends the picture as the camera swings.
    hole = cv2.morphologyEx(hole.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    if int(hole.sum()) < min_hole:
        return 0
    # Grow slightly so the half-covered fringe on the border is repainted too,
    # instead of leaving a dark seam around each fill.
    hole = cv2.dilate(hole.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    # Depth may only be borrowed from solidly covered pixels: a half-covered
    # edge pixel blends the wire's depth with the wall's and lands in between.
    solid = (alpha > 0.95) & ~hole
    depth = _far_side_depth(np.where(solid, dist, 0.0), ~solid)

    # The first scene plane behind the foreground next to the hole wins over
    # the far-wall estimate: a gap opening beside a car shows more road, not
    # the backdrop. Rays run through THIS camera in world space.
    fx, fy = proj[0, 0] * w * 0.5, proj[1, 1] * h * 0.5
    inv = np.linalg.inv(view)
    plane_n_map = np.zeros((h, w, 3), np.float32)
    if scene.planes:
        near_src = np.where(solid, dist, 1e9).astype(np.float32)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        local_near = cv2.erode(near_src, k)
        local_near = np.where(local_near > 1e8, 0.0, local_near)
        hy, hx = np.nonzero(hole)
        d_cam = np.stack([(hx - (w - 1) * 0.5) / fx, -(hy - (h - 1) * 0.5) / fy,
                          -np.ones(len(hx), np.float32)], -1)
        d_world = (inv[:3, :3] @ d_cam.T).T.astype(np.float32)
        t, nrm = _planes_behind(inv[:3, 3].astype(np.float32), d_world, scene.planes,
                                local_near[hy, hx] * 1.03)
        ok = (t > 0) & (t <= depth[hy, hx] * 1.1)
        depth[hy[ok], hx[ok]] = t[ok]
        plane_n_map[hy[ok], hx[ok]] = nrm[ok]

    # What LaMa may look at: only surfaces at least as far as the hole. Anything
    # nearer (a fence wire crossing the gap, the edge of the face) is masked out
    # as well, otherwise it is copied into the background as a streak.
    band = cv2.dilate(hole.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    nearer = band & ~hole & (dist < depth * 0.92)
    paint = hole | nearer
    rgb8 = (rgb * 255 + 0.5).astype(np.uint8)
    if inpainter is not None:
        try:
            filled = inpainter(rgb8, paint)
        except Exception:  # noqa: BLE001 - the classical fill is the floor
            filled = cv2.inpaint(rgb8, paint.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
    else:
        filled = cv2.inpaint(rgb8, paint.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)

    # Unproject the hole pixels through THIS camera into world space.
    ys, xs = np.nonzero(hole)
    tz = depth[ys, xs]
    cam_pts = np.stack([(xs - (w - 1) * 0.5) * tz / fx,
                        -(ys - (h - 1) * 0.5) * tz / fy,
                        -tz, np.ones_like(tz)], -1)
    world = (inv @ cam_pts.T).T[:, :3].astype(np.float32)
    keep = ~_occludes_photo(scene, world)
    ys, xs, tz, world = ys[keep], xs[keep], tz[keep], world[keep]
    if len(ys) == 0:
        return 0
    cov = _fill_cov(world, tz / fx * 1.4, plane_n_map[ys, xs],
                    view[2, :3] / (np.linalg.norm(view[2, :3]) + 1e-9), cam=inv[:3, 3])
    scene.positions = np.concatenate([scene.positions, world])
    scene.colors = np.concatenate([scene.colors, filled[ys, xs].astype(np.float32) / 255.0])
    scene.opacity = np.concatenate([scene.opacity, np.ones(len(ys), np.float32)])
    scene.cov = np.concatenate([scene.cov, cov])
    renderer.set_scene(scene)
    return int(len(ys))


def _is_grid(scene: SplatScene) -> bool:
    """True for our own one-splat-per-pixel scenes. A model-predicted scene
    (SHARP) has soft, semi-transparent edges by design; treating those as
    holes painted dark smears along every silhouette."""
    return scene.photo_z is not None and scene.n_front == scene.photo_z.size


def remove_floaters(scene: SplatScene, cell_px: float = 3.0, depth_step: float = 0.015,
                    min_neighbours: int = 12) -> int:
    """Drop splats that have almost nothing around them in 3-D.

    The grid is built in the original camera's frame: screen x/y in pixels and
    log-depth. That makes every cell hold roughly the same number of splats at
    any distance, so one threshold works for the near face and the far wall.
    A splat whose 3x3x3 block of cells is nearly empty is a stray: a flying
    edge pixel, or a lone fill that landed at a depth nothing else agrees with.
    Returns how many were removed."""
    if scene.photo_z is None or len(scene) == 0:
        return 0
    h, w = scene.photo_z.shape
    p = scene.positions
    z = np.maximum(-p[:, 2], 1e-3)
    u = p[:, 0] * scene.focal / z / cell_px
    v = -p[:, 1] * scene.focal / z / cell_px
    d = np.log(z) / depth_step
    cells = np.stack([np.floor(u), np.floor(v), np.floor(d)], -1).astype(np.int64)
    cells -= cells.min(0)
    dims = cells.max(0) + 3
    key = ((cells[:, 0] + 1) * dims[1] + (cells[:, 1] + 1)) * dims[2] + (cells[:, 2] + 1)
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    total = np.zeros(len(uniq), np.int64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                nk = uniq + (dx * dims[1] + dy) * dims[2] + dz
                pos = np.searchsorted(uniq, nk)
                pos = np.minimum(pos, len(uniq) - 1)
                hit = uniq[pos] == nk
                total += np.where(hit, counts[pos], 0)
    keep = total[inv] >= min_neighbours
    if not _is_grid(scene):
        # A model-predicted scene (SHARP) is sparser and uneven by design, so
        # density says nothing about its own splats; only fills are judged.
        keep[: scene.n_front] = True
    removed = int((~keep).sum())
    if removed:
        scene.keep(keep)
    return removed


def fill_along_path(scene: SplatScene, renderer: "SplatRenderer", poses, size,
                    inpainter=None, progress=None) -> int:
    """Fill every hole a camera move reveals. `poses` is a list of
    (view, proj). The poses farthest from the original viewpoint go first:
    they reveal the most, and filling them first leaves the in-between poses
    with little or nothing left to do."""
    def reach(vp):
        v = vp[0]
        cam = -v[:3, :3].T @ v[:3, 3]
        return float(np.linalg.norm(cam)) + float(np.linalg.norm(v[2, :3] - (0, 0, 1)))
    ordered = sorted(poses, key=reach, reverse=True)
    if progress is not None:
        progress(0, 2 * len(ordered))
    fill_backplate(scene, renderer, inpainter)
    # Two rounds. The first fills everything, then strays are removed; that
    # cleanup also takes out a few good fills that happened to be isolated, so
    # the second round (cheap: little is left) patches exactly those gaps.
    added = 0
    steps = 2 * len(ordered)
    for rnd in range(2):
        for i, (view, proj) in enumerate(ordered):
            added += fill_view(scene, renderer, view, proj, size, inpainter)
            if progress is not None:
                progress(rnd * len(ordered) + i + 1, steps)
        if rnd == 0:
            remove_floaters(scene)
            renderer.set_scene(scene)
    return added


#: Final pass: un-premultiply, fill low-coverage pixels from a
#: coverage-weighted 5x5 neighbourhood, composite over the background, write
#: 8-bit. Doing this on the CPU cost ~80 ms a frame at 720p.
FINISH_SHADER = """
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> bg: vec4<f32>;
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    return vec4(p[i], 0.0, 1.0);
}
@fragment fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(src));
    let c = vec2<i32>(pos.xy);
    let here = textureLoad(src, c, 0);
    var rgb = here.rgb / max(here.a, 1e-4);
    var alpha = here.a;
    if (here.a < 0.35) {
        var pre = vec3<f32>(0.0);
        var cov = 0.0;
        var wsum = 0.0;
        for (var dy = -2; dy <= 2; dy++) {
            for (var dx = -2; dx <= 2; dx++) {
                let q = clamp(c + vec2<i32>(dx, dy), vec2<i32>(0), dims - 1);
                let s = textureLoad(src, q, 0);
                let w = exp(-f32(dx * dx + dy * dy) / 4.5);
                pre += s.rgb * w;
                cov += s.a * w;
                wsum += w;
            }
        }
        if (cov > 1e-4) { rgb = pre / cov; }
        alpha = max(alpha, clamp(cov / wsum * 1.6, 0.0, 1.0));
    }
    let a = clamp(alpha / 0.35, 0.0, 1.0);
    // bg.w is the lightning flash: everything brightens and cools for an
    // instant, the empty sky most of all.
    let flash = bg.w;
    let sky = bg.rgb * (1.0 + flash * 3.0) + vec3<f32>(0.5, 0.55, 0.7) * flash * 0.15;
    let lit = rgb * (1.0 + flash * 1.6) + vec3<f32>(0.55, 0.6, 0.75) * flash * 0.1;
    return vec4<f32>(clamp(lit * a + sky * (1.0 - a), vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
"""


# -- renderer ----------------------------------------------------------------

class SplatRenderer:
    """Holds a wgpu device and renders a SplatScene to an RGB float array."""

    def __init__(self, device=None) -> None:
        import wgpu
        self._wgpu = wgpu
        if device is None:
            from . import gpus
            adapter = gpus.wgpu_adapter()
            limits = adapter.limits
            device = adapter.request_device_sync(required_limits={
                "max-storage-buffer-binding-size": limits["max-storage-buffer-binding-size"],
                "max-buffer-size": limits["max-buffer-size"]})
            self.adapter = adapter
        self.device = device
        self._shader = device.create_shader_module(code=SHADER)
        self._uniform = device.create_buffer(
            size=176, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._layout = device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 1, "visibility": wgpu.ShaderStage.VERTEX,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 2, "visibility": wgpu.ShaderStage.VERTEX,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}}])
        blend = {"color": {"src_factor": "one", "dst_factor": "one-minus-src-alpha",
                           "operation": "add"},
                 "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha",
                           "operation": "add"}}
        self._pipe = device.create_render_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._layout]),
            vertex={"module": self._shader, "entry_point": "vs_main", "buffers": []},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_strip,
                       "cull_mode": wgpu.CullMode.none},
            # Premultiplied, back to front, into colour and depth together.
            fragment={"module": self._shader, "entry_point": "fs_main", "targets": [
                {"format": "rgba16float", "blend": blend},
                {"format": "rgba16float", "blend": blend}]})
        from .fx3d import FxPass
        self._fx = FxPass(device)
        fin = device.create_shader_module(code=FINISH_SHADER)
        self._fin_layout = device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": wgpu.TextureSampleType.unfilterable_float}},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": wgpu.BufferBindingType.uniform}}])
        self._fin_pipe = device.create_render_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._fin_layout]),
            vertex={"module": fin, "entry_point": "vs_main", "buffers": []},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            fragment={"module": fin, "entry_point": "fs_main",
                      "targets": [{"format": "rgba8unorm"}]})
        self._fin_uniform = device.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._targets = None
        self._n = 0
        self._planes: list = []
        sort_mod = device.create_shader_module(code=SORT_SHADER)
        st = wgpu.BufferBindingType.storage
        self._sort_layout = device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": st}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": st}},
            {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": st}}])
        pl = device.create_pipeline_layout(bind_group_layouts=[self._sort_layout])
        self._sort_pipes = {
            name: device.create_compute_pipeline(
                layout=pl, compute={"module": sort_mod, "entry_point": name})
            for name in ("cs_keys", "cs_scan", "cs_scatter")}
        self._sort_uniform = device.create_buffer(
            size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._hist = device.create_buffer(
            size=65536 * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)

    def set_scene(self, scene: SplatScene) -> None:
        wgpu = self._wgpu
        n = max(1, len(scene))
        self._splats = self.device.create_buffer_with_data(
            data=scene.packed(), usage=wgpu.BufferUsage.STORAGE)
        self._order = self.device.create_buffer(size=n * 4, usage=wgpu.BufferUsage.STORAGE)
        self._keys = self.device.create_buffer(size=n * 4, usage=wgpu.BufferUsage.STORAGE)
        self._bind = self.device.create_bind_group(layout=self._layout, entries=[
            {"binding": 0, "resource": {"buffer": self._uniform}},
            {"binding": 1, "resource": {"buffer": self._splats}},
            {"binding": 2, "resource": {"buffer": self._order}}])
        self._sort_bind = self.device.create_bind_group(layout=self._sort_layout, entries=[
            {"binding": 0, "resource": {"buffer": self._sort_uniform}},
            {"binding": 1, "resource": {"buffer": self._splats}},
            {"binding": 2, "resource": {"buffer": self._keys}},
            {"binding": 3, "resource": {"buffer": self._hist}},
            {"binding": 4, "resource": {"buffer": self._order}}])
        # Bounding sphere, so any camera gets a tight distance range for the
        # 16-bit keys without reading positions back.
        pos = scene.positions
        self._centre = pos.mean(0) if len(scene) else np.zeros(3, np.float32)
        self._radius = float(np.linalg.norm(pos - self._centre, axis=1).max()) if len(scene) else 1.0
        self._n = len(scene)
        self._planes = list(scene.planes or [])

    def _encode_sort(self, enc, view: np.ndarray) -> None:
        cam = -view[:3, :3].T @ view[:3, 3]
        d = float(np.linalg.norm(self._centre - cam))
        lo = max(0.0, d - self._radius)
        su = np.zeros(8, np.float32)
        su[0:4] = view[2]
        su[4:7] = (lo, 1.0 / max(2.0 * self._radius, 1e-6), self._n)
        self.device.queue.write_buffer(self._sort_uniform, 0, su.tobytes())
        enc.clear_buffer(self._hist)
        groups = (self._n + 255) // 256
        gx, gy = min(groups, 65535), (groups + 65534) // 65535
        for name, dims in (("cs_keys", (gx, gy)), ("cs_scan", (1, 1)), ("cs_scatter", (gx, gy))):
            cp = enc.begin_compute_pass()
            cp.set_pipeline(self._sort_pipes[name])
            cp.set_bind_group(0, self._sort_bind)
            cp.dispatch_workgroups(dims[0], dims[1], 1)
            cp.end()

    def _ensure_targets(self, w: int, h: int) -> None:
        if self._targets == (w, h):
            return
        wgpu = self._wgpu
        usage = (wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC
                 | wgpu.TextureUsage.TEXTURE_BINDING)
        self._color = self.device.create_texture(size=(w, h, 1), format="rgba16float", usage=usage)
        self._depth = self.device.create_texture(size=(w, h, 1), format="rgba16float", usage=usage)
        self._final = self.device.create_texture(
            size=(w, h, 1), format="rgba8unorm",
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
        self._fin_bind = self.device.create_bind_group(layout=self._fin_layout, entries=[
            {"binding": 0, "resource": self._color.create_view()},
            {"binding": 1, "resource": {"buffer": self._fin_uniform}}])
        self._targets = (w, h)

    def _read(self, tex, w: int, h: int) -> np.ndarray:
        raw = self.device.queue.read_texture(
            {"texture": tex}, {"bytes_per_row": w * 8, "rows_per_image": h}, (w, h, 1))
        return np.frombuffer(raw, np.float16).reshape(h, w, 4).astype(np.float32)

    def _draw(self, view, proj, size, mode: int = 0, effects=None, time_seconds: float = 0.0,
              depth_range=(NEAR, FAR)) -> None:
        wgpu = self._wgpu
        w, h = int(size[0]), int(size[1])
        self._ensure_targets(w, h)
        u = np.zeros(44, np.float32)
        u[0:16] = np.ascontiguousarray(view.T, np.float32).ravel()
        u[16:32] = np.ascontiguousarray(proj.T, np.float32).ravel()
        u[32:36] = (w, h, proj[0, 0] * w * 0.5, proj[1, 1] * h * 0.5)
        u[36:39] = (mode, depth_range[0], depth_range[1])
        self.device.queue.write_buffer(self._uniform, 0, u.tobytes())
        enc = self.device.create_command_encoder()
        self._encode_sort(enc, view)
        colour_view = self._color.create_view()
        rp = enc.begin_render_pass(color_attachments=[
            {"view": colour_view, "clear_value": (0.0, 0.0, 0.0, 0.0),
             "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store},
            {"view": self._depth.create_view(), "clear_value": (0.0, 0.0, 0.0, 0.0),
             "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store}])
        rp.set_pipeline(self._pipe)
        rp.set_bind_group(0, self._bind)
        rp.draw(4, self._n, 0, 0)
        rp.end()
        if effects is not None and mode == 0:
            cam = -view[:3, :3].T @ view[:3, 3]
            self._fx.encode(enc, colour_view, self._depth.create_view(), view, proj,
                            cam, effects, time_seconds, self._planes)
        self.device.queue.submit([enc.finish()])

    def render(self, view: np.ndarray, proj: np.ndarray, size,
               background=(0.02, 0.021, 0.026), effects=None, time_seconds: float = 0.0,
               mode: int = 0) -> np.ndarray:
        """Final frame, RGB float 0..1."""
        return self.render_u8(view, proj, size, background, effects, time_seconds,
                              mode).astype(np.float32) / 255.0

    def render_u8(self, view: np.ndarray, proj: np.ndarray, size,
                  background=(0.02, 0.021, 0.026), effects=None, time_seconds: float = 0.0,
                  mode: int = 0) -> np.ndarray:
        """Final frame as uint8 RGB, finished on the GPU (see FINISH_SHADER)."""
        wgpu = self._wgpu
        w, h = int(size[0]), int(size[1])
        self._draw(view, proj, (w, h), mode, effects, time_seconds)
        flash = 0.0
        if effects is not None and mode == 0:
            from .fx3d import scene_flash
            flash = scene_flash(effects, time_seconds)
        self.device.queue.write_buffer(
            self._fin_uniform, 0, np.array([*background, flash], np.float32).tobytes())
        enc = self.device.create_command_encoder()
        rp = enc.begin_render_pass(color_attachments=[{
            "view": self._final.create_view(), "clear_value": (0, 0, 0, 1),
            "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store}])
        rp.set_pipeline(self._fin_pipe)
        rp.set_bind_group(0, self._fin_bind)
        rp.draw(3, 1, 0, 0)
        rp.end()
        self.device.queue.submit([enc.finish()])
        raw = self.device.queue.read_texture(
            {"texture": self._final}, {"bytes_per_row": w * 4, "rows_per_image": h}, (w, h, 1))
        return np.ascontiguousarray(np.frombuffer(raw, np.uint8).reshape(h, w, 4)[:, :, :3])

    def render_coverage(self, view, proj, size):
        """(colour, alpha, distance) with a transparent background, for finding
        the holes a camera pose reveals. Distance is 0 where nothing was drawn."""
        w, h = int(size[0]), int(size[1])
        self._draw(view, proj, (w, h))
        rgba = np.clip(self._read(self._color, w, h), 0.0, 1.0)
        alpha = rgba[:, :, 3]
        rgb = rgba[:, :, :3] / np.maximum(alpha, 1e-4)[..., None]
        d = self._read(self._depth, w, h)
        dist = np.where(d[:, :, 3] > 1e-3, d[:, :, 0] / np.maximum(d[:, :, 3], 1e-4), 0.0)
        return np.clip(rgb, 0, 1), alpha, dist.astype(np.float32)


def is_available() -> bool:
    try:
        import wgpu
        return wgpu.gpu.request_adapter_sync() is not None
    except Exception:  # noqa: BLE001
        return False
