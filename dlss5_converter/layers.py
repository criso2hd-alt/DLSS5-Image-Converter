"""Layered depth reconstruction — completing the scene behind the foreground.

Cutting the mesh at depth edges removes doubling, but opens holes where the
camera sees background that was never photographed. This module reconstructs
that hidden background **once**, independent of any camera path, so the move can
be changed freely afterwards without rebuilding anything.

The reconstruction deliberately fills the *entire* occluded region rather than
only what a particular move would reveal. Fitting the fill to the animation
would couple the two, and every edit to the camera would invalidate the scene.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry3d import (
    Mesh,
    build_mesh,
    depth_discontinuity_mask,
    disparity_to_depth,
    focal_from_fov,
)

# A layer's rendered silhouette is extended this far past its coverage so the
# base layer's seam stays hidden behind it. Wall texture has to begin outside
# that lip, or painting the band would repaint the lip as well.
LIP_PIXELS = 2


@dataclass(slots=True)
class Layer:
    """One depth slice: its colour, its disparity, and where it has geometry."""

    colour: np.ndarray  # (H, W, 3) uint8
    disparity: np.ndarray  # (H, W) float32, 1 = near
    coverage: np.ndarray  # (H, W) bool
    synthetic: np.ndarray  # (H, W) bool — pixels that were inpainted
    index: int  # 0 is the farthest layer

    @property
    def synthetic_fraction(self) -> float:
        total = int(self.coverage.sum())
        return float(self.synthetic.sum()) / total if total else 0.0


@dataclass(slots=True)
class LayeredScene:
    layers: list[Layer]
    meshes: list[Mesh]

    @property
    def triangle_count(self) -> int:
        return sum(m.triangle_count for m in self.meshes)


def _smooth_histogram(disparity: np.ndarray, bins: int = 128) -> np.ndarray:
    hist, _ = np.histogram(disparity, bins=bins, range=(0.0, 1.0))
    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    kernel /= kernel.sum()
    return np.convolve(hist.astype(np.float32), kernel, mode="same")


def find_layer_thresholds(disparity: np.ndarray, layer_count: int) -> list[float]:
    """Split disparity where the scene naturally separates.

    Photographs of a subject against a background produce a bimodal disparity
    histogram; the valleys between modes are the depth planes worth cutting at.
    Picking the deepest valleys is more robust than a fixed threshold, which
    would either merge the subject into the background or slice through it.
    """
    if layer_count < 2:
        return []
    bins = 128
    smoothed = _smooth_histogram(disparity, bins)
    centres = (np.arange(bins) + 0.5) / bins

    # Interior local minima, ranked by how deep the valley is relative to the
    # peaks flanking it.
    candidates: list[tuple[float, float]] = []
    for i in range(2, bins - 2):
        window = smoothed[i - 2 : i + 3]
        if smoothed[i] > window.min():
            continue
        left = smoothed[:i].max() if i else 0.0
        right = smoothed[i + 1 :].max() if i + 1 < bins else 0.0
        prominence = min(left, right) - smoothed[i]
        if prominence > 0:
            candidates.append((prominence, float(centres[i])))

    if not candidates:
        # No clear separation: fall back to even splits so behaviour stays sane.
        return [float(q) for q in np.linspace(0, 1, layer_count + 1)[1:-1]]

    candidates.sort(reverse=True)
    chosen = sorted(value for _p, value in candidates[: layer_count - 1])
    return chosen


def segment_layers(
    disparity: np.ndarray,
    layer_count: int = 2,
    discontinuity: float = 0.06,
    min_area: int = 64,
) -> np.ndarray:
    """Label pixels by layer; 0 is farthest.

    Occluders are found from **depth discontinuities**, not a global disparity
    threshold. A receding ground plane spans the whole disparity range without
    ever occluding anything; thresholding slices straight through it and asks
    the backdrop to invent an enormous region that was never hidden, which is
    what produced white smears across the lower half of real photographs.

    A region is only promoted to a nearer layer when it is bounded by a genuine
    depth jump *and* is closer than the surface immediately surrounding it.
    """
    labels = np.zeros(disparity.shape, dtype=np.int32)
    if layer_count < 2:
        return labels

    edges = depth_discontinuity_mask(disparity, threshold=discontinuity)
    if not edges.any():
        return labels

    height, width = disparity.shape[:2]
    kernel = np.ones((3, 3), np.uint8)
    walls = cv2.dilate(edges.astype(np.uint8), kernel, iterations=1) > 0
    free = (~walls).astype(np.uint8)
    # Stats give area and bounding box in a single pass, so small regions are
    # rejected without ever touching the image. Working inside each bounding
    # box then keeps the whole loop proportional to the picture rather than to
    # regions x pixels, which took minutes on a detailed photograph.
    count, components, stats, _centroids = cv2.connectedComponentsWithStats(
        free, connectivity=8
    )

    candidates: list[tuple[float, tuple[int, int, int, int], np.ndarray]] = []
    # The ring must clear the dilated wall band, otherwise it lands entirely
    # inside it and the region is discarded for having no neighbourhood.
    ring_reach = 5
    margin = ring_reach + 3
    for label in range(1, count):
        x, y, box_w, box_h, area = (int(v) for v in stats[label])
        if area < min_area:
            continue
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(width, x + box_w + margin), min(height, y + box_h + margin)
        sub_components = components[y0:y1, x0:x1]
        region = sub_components == label
        # Compare the region against the surface just outside it. A true
        # occluder stands in front of its surroundings; part of a continuous
        # slope does not.
        grown = cv2.dilate(region.astype(np.uint8), kernel, iterations=ring_reach) > 0
        ring = grown & ~region & ~walls[y0:y1, x0:x1]
        if int(ring.sum()) < max(8, min_area // 4):
            continue
        sub_disparity = disparity[y0:y1, x0:x1]
        inside = float(np.median(sub_disparity[region]))
        outside = float(np.median(sub_disparity[ring]))
        if inside - outside > discontinuity:
            candidates.append((inside, (x0, y0, x1, y1), region))

    if not candidates:
        return labels

    # Nearest regions occupy the highest layer indices.
    candidates.sort(key=lambda item: item[0])
    tiers = min(layer_count - 1, len(candidates))
    for position, (_depth, (x0, y0, x1, y1), region) in enumerate(candidates):
        tier = 1 + (position * tiers) // max(len(candidates), 1)
        window = labels[y0:y1, x0:x1]
        window[region] = min(tier, layer_count - 1)
    return labels


def background_hole_mask(
    disparity: np.ndarray,
    labels: np.ndarray,
    discontinuity: float = 0.06,
    min_area: int = 16,
    reach: float = 0.12,
) -> np.ndarray:
    """Pixels the farthest layer must reconstruct behind occluders.

    Promoted depth layers already identify complete foreground objects, so
    their full silhouettes are removed. Smaller or ambiguous objects may not
    survive layer segmentation, but the mesh is still cut at their depth edge.
    For those, closed near-side contours are completed when the enclosed region
    is demonstrably closer than its surroundings. Any edge that cannot be
    completed safely remains in the mask as a conservative band, guaranteeing
    that every geometric cut has painted background immediately behind it.

    `reach` bounds the result to a band that many image widths behind each
    silhouette. Removing a foreground layer's *entire* footprint sounds safer,
    but a receding surface such as a street gets thresholded into the near
    layer along with real occluders, and then the backdrop is asked to invent
    ground nothing was ever hiding. Measured on the reference alley photograph
    that left the bottom third of the backdrop 100% synthetic, which no
    inpainter recovers from — it came back as flat olive. A camera move only
    ever reveals a band, so only a band needs inventing, and a smaller hole is
    one the inpainter can actually reason about.
    """
    if disparity.shape != labels.shape:
        raise ValueError(
            f"disparity {disparity.shape} and labels {labels.shape} differ"
        )

    complete = labels > 0
    edges = depth_discontinuity_mask(disparity, threshold=discontinuity)
    if not edges.any():
        return complete

    kernel = np.ones((3, 3), np.uint8)
    # Remove edges already explained by a promoted silhouette. The remaining
    # contours are the small/thin objects and ambiguous cuts that used to leave
    # transparent cracks in the reconstructed backdrop.
    claimed = cv2.dilate(complete.astype(np.uint8), kernel, iterations=2) > 0
    unresolved = edges & ~claimed
    sealed = cv2.morphologyEx(
        unresolved.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2
    )
    contours, _hierarchy = cv2.findContours(
        sealed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    height, width = disparity.shape
    for contour in contours:
        x, y, box_w, box_h = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        # A contour touching the frame is not a closed photographed object; it
        # is commonly a horizon or foreground plane. Keep only its edge band.
        if (
            area < min_area
            or box_w < 3
            or box_h < 3
            or x <= 0
            or y <= 0
            or x + box_w >= width
            or y + box_h >= height
            or area > disparity.size * 0.65
        ):
            continue

        candidate = np.zeros(disparity.shape, np.uint8)
        cv2.drawContours(candidate, [contour], -1, 1, thickness=cv2.FILLED)
        candidate_bool = candidate.astype(bool)
        ring = (
            cv2.dilate(candidate, kernel, iterations=4).astype(bool)
            & ~candidate_bool
        )
        if int(ring.sum()) < 8:
            continue
        inside = float(np.median(disparity[candidate_bool]))
        outside = float(np.median(disparity[ring]))
        if inside - outside > discontinuity:
            complete |= candidate_bool

    # `edges` is the conservative fallback. build_layered_scene dilates the
    # result by its seam width before inpainting, turning each line into the
    # narrow background band needed immediately behind the cut.
    mask = complete | edges
    if reach > 0:
        # Keep only what a camera move can actually expose: a band behind each
        # silhouette. A promoted region smaller than the band is covered
        # entirely by it, so compact objects still lose their whole footprint.
        span = max(1, round(float(reach) * max(disparity.shape)))
        near_side = cv2.dilate(
            edges.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=span
        ).astype(bool)
        mask &= near_side
    return mask


def sky_mask(
    disparity: np.ndarray,
    image_rgb: np.ndarray | None = None,
    percentile: float = 12.0,
) -> np.ndarray:
    """Far-plane regions with no texture — the sky.

    Distinct from `visible_background_mask`, which *protects* sky so it is not
    mistaken for an occluder. This finds it in order to remove it: a sky plane
    is a flat sheet at the far plane spanning the whole frustum cross section,
    and in a pinhole unprojection the far plane is `far / near` times wider than
    the nearest surface. Keeping it means every camera move drags an enormous
    washed-out sheet through the frame, and an HDRI cannot replace a sky that is
    still geometry.

    Depth alone is not enough, and neither is openness to the frame border: a
    photograph looking up a light well has its sky as a *hole in the middle*,
    enclosed by buildings on every side, and a border rule scores that at 0%.
    What separates sky from a distant wall is texture — the wall has some. So
    the test is "far, and locally flat", which holds wherever the sky sits.
    """
    if disparity.ndim != 2 or disparity.size == 0:
        raise ValueError("disparity must be a non-empty 2D array")
    # Two depth conditions, whichever is stricter. The percentile alone fails
    # when the far region is small: with most of a frame at one depth, the
    # twelfth percentile lands *on* that depth and selects the whole image.
    spread = float(np.ptp(disparity))
    limit = min(
        float(np.percentile(disparity, percentile)),
        float(disparity.min()) + 0.10 * spread,
    )
    far = disparity <= limit
    if not far.any():
        return np.zeros(disparity.shape, dtype=bool)

    if image_rgb is not None and image_rgb.shape[:2] == disparity.shape:
        grey = cv2.cvtColor(
            np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2GRAY
        ).astype(np.float32)
        mean = cv2.blur(grey, (15, 15))
        variance = np.maximum(cv2.blur(grey * grey, (15, 15)) - mean * mean, 0.0)
        deviation = np.sqrt(variance)
        # Relative to the picture's own contrast, so a flat overcast sky and a
        # crisp blue one are judged the same way.
        flat = deviation < max(4.0, 0.35 * float(np.median(deviation)))
        far &= flat

    candidates = cv2.morphologyEx(
        far.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
    )
    count, components = cv2.connectedComponents(candidates, connectivity=8)
    sky = np.zeros(disparity.shape, dtype=bool)
    minimum = max(64, disparity.size // 500)
    for label in range(1, count):
        region = components == label
        if int(region.sum()) >= minimum:
            sky |= region
    return sky


def visible_background_mask(disparity: np.ndarray) -> np.ndarray:
    """Protect far scenery connected to the upper frame, especially sky.

    Monocular estimators occasionally place clouds, haze, or a smooth sky in a
    separate histogram island. Treating that island as an occluder removes the
    photographed sky and asks the inpainter to invent a replacement. A real
    background is normally both relatively far and connected to the top/upper
    side boundary, so flood-fill that conservative subset and keep it visible.
    """
    if disparity.ndim != 2 or disparity.size == 0:
        raise ValueError("disparity must be a non-empty 2D array")
    height, width = disparity.shape
    far_limit = float(np.percentile(disparity, 58.0))
    thresholds = find_layer_thresholds(disparity, 2)
    if thresholds:
        far_limit = min(far_limit, float(thresholds[0]))
    candidates = (disparity <= far_limit).astype(np.uint8)
    # Close thin cloud/wire interruptions without allowing the protection to
    # jump across a large foreground object.
    kernel = np.ones((5, 5), np.uint8)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, kernel, iterations=1)
    count, components = cv2.connectedComponents(candidates, connectivity=8)
    protected = np.zeros(disparity.shape, dtype=bool)
    upper_side = max(1, height // 3)
    for label in range(1, count):
        region = components == label
        touches_upper_frame = (
            region[0].any()
            or region[:upper_side, 0].any()
            or region[:upper_side, -1].any()
        )
        if not touches_upper_frame:
            continue
        # A tiny dark corner is not a sky/background component.
        if int(region.sum()) < max(16, disparity.size // 1000):
            continue
        protected |= region
    return protected


_PLANE_MIN_AREA = 96
_RING_SAMPLES_MIN = 24


def _propagate_min_behind(
    disparity: np.ndarray, holes: np.ndarray, iterations: int = 0
) -> np.ndarray:
    """The conservative envelope: deepest-rim depth propagated across the hole.

    Never nearer than any real pixel around the hole, which makes it both the
    seam-safe base of the plane fill and the fallback wherever no surface can
    be fitted with confidence.
    """
    work = disparity.astype(np.float32).copy()
    mask = holes.astype(bool)
    # Unknown pixels start "very near" so that erosion (a local minimum filter)
    # always prefers a real neighbour over them.
    work[mask] = 1e6
    kernel = np.ones((3, 3), np.uint8)
    if iterations <= 0:
        # Enough passes for the fill to cross the widest hole.
        distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        iterations = int(np.ceil(float(distance.max()))) + 2
    for _ in range(max(1, iterations)):
        eroded = cv2.erode(work, kernel)
        work = np.where(mask, eroded, disparity.astype(np.float32))
        if work.max() < 1e5:
            break
    work[work > 1e5] = float(np.min(disparity))
    return work


def _weighted_plane(xs, ys, zs, w) -> tuple[float, float, float]:
    """Least-squares plane z = alpha + beta*x + gamma*y through weighted samples.

    Under this pinhole model a 3D plane is exactly linear in normalised image
    coordinates, so one fit reproduces a wall, a ground plane, or any other flat
    surface at any orientation — including full perspective recession.
    """
    sw = w.sum()
    sx = float((w * xs).sum())
    sy = float((w * ys).sum())
    sxx = float((w * xs * xs).sum())
    sxy = float((w * xs * ys).sum())
    syy = float((w * ys * ys).sum())
    sz = float((w * zs).sum())
    sxz = float((w * xs * zs).sum())
    syz = float((w * ys * zs).sum())
    matrix = np.array(
        [[sw, sx, sy], [sx, sxx, sxy], [sy, sxy, syy]], np.float64
    )
    rhs = np.array([sz, sxz, syz], np.float64)
    alpha, beta, gamma = (float(v) for v in np.linalg.solve(matrix, rhs))
    return alpha, beta, gamma


def _plane_depth_at(alpha, beta, gamma, x_n, y_n, near, far):
    """Disparity of a plane along each pixel's viewing ray.

    The ray hits the plane where `z = alpha / (1 - beta*x_n - gamma*y_n)`; a
    vanishing denominator means the plane runs parallel to the ray and recedes
    to infinity, which maps to the far plane. Depths in front of the camera are
    reported as invalid rather than clamped, so callers can fall back.
    """
    denom = 1.0 - beta * x_n - gamma * y_n
    valid = denom > 0.05
    z = np.where(valid, alpha / np.where(valid, denom, 1.0), far)
    valid &= z >= near * 0.9
    inv_near, inv_far = 1.0 / float(near), 1.0 / float(far)
    disp = (1.0 / np.maximum(z, 1e-6) - inv_far) / (inv_near - inv_far)
    return np.clip(disp, 0.0, 1.0).astype(np.float32), valid


def _fit_component_planes(
    disparity: np.ndarray,
    component: np.ndarray,
    holes: np.ndarray,
    edges: np.ndarray,
    focal: float,
    cx: float,
    cy: float,
    near: float,
    far: float,
):
    """Fit up to two background planes behind one connected hole region.

    Returns the filled disparity for the component's pixels, or `None` when the
    ring cannot support a fit. Two planes handle the common corner case — ground
    meeting a wall — chosen per pixel by which fit has the nearer supporting
    evidence.
    """
    height, width = disparity.shape
    ys, xs = np.nonzero(component)
    area = int(ys.size)

    ring_k = int(np.clip(round(area**0.5 * 0.12), 3, 10))
    grown = cv2.dilate(
        component.astype(np.uint8),
        np.ones((ring_k, ring_k), np.uint8),
    ).astype(bool)
    # Samples on the near side of a depth cliff are the occluder's own rim —
    # the very surface this hole exists because of. Letting them vote would
    # continue the occluder *into* the hole instead of reconstructing the
    # background behind it, so the cut test that opened the hole also decides
    # who gets to fill it.
    suspect = cv2.dilate(edges.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(
        bool
    )
    ring = grown & ~holes & ~suspect
    sample_ys, sample_xs = np.nonzero(ring)
    if sample_ys.size < _RING_SAMPLES_MIN:
        return None

    d_ring = disparity[sample_ys, sample_xs].astype(np.float64)
    z_ring = disparity_to_depth(d_ring.astype(np.float32), near, far).astype(np.float64)
    x_n = (sample_xs - cx) / focal
    y_n = -(sample_ys - cy) / focal
    world_x = x_n * z_ring
    world_y = y_n * z_ring

    # Mild far bias only tilts the initial fit away from any occluder rim that
    # leaks into the ring; robust reweighting then lets whichever surface has
    # consensus win, so genuine recession is never flattened by the bias.
    reference = float(np.percentile(d_ring, 5.0))
    weights = np.exp(-(d_ring - reference) / 0.30)
    try:
        plane_a = _weighted_plane(world_x, world_y, z_ring, weights)
        residuals = np.abs(z_ring - (plane_a[0] + plane_a[1] * world_x + plane_a[2] * world_y))
        weights *= np.exp(-residuals / 0.02)
        plane_a = _weighted_plane(world_x, world_y, z_ring, weights)
    except np.linalg.LinAlgError:
        return None

    tol_z = max(0.02, 0.04 * abs(far - near))
    residuals = np.abs(
        z_ring - (plane_a[0] + plane_a[1] * world_x + plane_a[2] * world_y)
    )
    inliers = residuals < tol_z
    outlier_mask = ~inliers
    planes = [(plane_a, inliers)]
    if outlier_mask.sum() >= max(_RING_SAMPLES_MIN, 0.15 * sample_ys.size):
        try:
            plane_b = _weighted_plane(
                world_x[outlier_mask],
                world_y[outlier_mask],
                z_ring[outlier_mask],
                np.ones(int(outlier_mask.sum())),
            )
            planes.append((plane_b, outlier_mask))
        except np.linalg.LinAlgError:
            pass

    local = np.zeros((height, width), np.uint8)
    local[ys, xs] = 1
    inside_distance = cv2.distanceTransform(local, cv2.DIST_L2, 3)

    def support_distance(mask_values):
        field = np.zeros((height, width), np.uint8)
        field[sample_ys[mask_values], sample_xs[mask_values]] = 1
        return cv2.distanceTransform((field == 0).astype(np.uint8), cv2.DIST_L2, 3)

    x_pix = (xs - cx) / focal
    y_pix = -(ys - cy) / focal
    envelope = _propagate_min_behind(disparity, component)[component].astype(np.float32)
    depth_a, valid_a = _plane_depth_at(*planes[0][0], x_pix, y_pix, near, far)
    filled = np.where(valid_a, depth_a, envelope)
    if len(planes) == 2:
        depth_b, valid_b = _plane_depth_at(*planes[1][0], x_pix, y_pix, near, far)
        dist_a = support_distance(planes[0][1])
        dist_b = support_distance(planes[1][1])[ys, xs].astype(np.float32)
        dist_a = dist_a[ys, xs].astype(np.float32)
        # Where both surfaces have support nearby, blend by which fit's own
        # evidence is closer; where only B has any, take B outright.
        weight_b = np.clip((dist_a - dist_b) / 3.0 + 0.5, 0.0, 1.0) * valid_b
        weight_b = np.maximum(weight_b, (valid_b & ~valid_a).astype(np.float32))
        filled = (1.0 - weight_b) * filled + weight_b * np.where(
            valid_b, depth_b, envelope
        )

    # Seam surface: the plane capped by the disparity of the nearest real
    # pixel. Where the plane meets the rim it is exactly continuous with the
    # photograph; where it would sit in front of its neighbour it yields, so
    # the fill can crease away from the camera but never bulge into real
    # pixels. A narrow interior feather hands over from this guarded surface
    # to the raw plane, and the ring clamp backs everything off against a
    # pathological fit.
    _distance, nearest = cv2.distanceTransformWithLabels(
        (~ring).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    lookup = np.zeros(int(nearest.max()) + 1, np.float64)
    lookup[nearest[sample_ys, sample_xs]] = d_ring
    seam = np.minimum(filled, lookup[nearest[ys, xs]].astype(np.float32))
    ring_max = float(disparity[sample_ys, sample_xs].max())
    feather = float(np.clip(0.15 * area**0.5, 3.0, 10.0))
    t = np.clip(inside_distance[ys, xs] / feather, 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    blended = (1.0 - t) * seam + t * filled
    return np.minimum(blended, ring_max).astype(np.float32)


def fill_disparity_behind(
    disparity: np.ndarray,
    holes: np.ndarray,
    iterations: int = 0,
    fov_degrees: float = 55.0,
    near: float = 1.0,
    far: float = 12.0,
    discontinuity: float = 0.06,
) -> np.ndarray:
    """Reconstruct the depth behind occluders as surfaces, not as a sagging sheet.

    Min-propagating the deepest rim value everywhere fills every hole with a
    shelf at the farthest rim depth: a street behind a person stops receding, a
    wall stops being a wall. A modeller would continue the surrounding surfaces,
    so each connected hole gets a robust plane fit through the real pixels
    ringing it, evaluated along each hole pixel's own viewing ray. Any 3D plane
    is linear in normalised image coordinates here, so walls, ground and full
    perspective recession are reproduced exactly.

    Safety is preserved deliberately:

    - Ring samples on the near side of a depth cliff (`discontinuity`, the same
      test that cut the mesh) are the occluder's own rim. They are excluded
      from the fit, from the seam guard and from the clamp, so the occluder is
      never continued into the hole it opened.
    - At the seam the plane yields to the nearest trusted pixel's disparity, so
      the fill meets the photograph continuously and can crease away from the
      camera but never bulge in front of it; a narrow interior feather hands
      over to the raw plane.
    - The min-propagation envelope stands in wherever no surface fits, so a
      hole with no coherent rim keeps exactly the old conservative behaviour.
    - The result is clamped to the trusted ring's maximum disparity:
      synthesised pixels may never sit nearer than any real background pixel
      adjacent to the hole.
    """
    if not holes.any():
        return disparity.astype(np.float32).copy()
    mask = holes.astype(bool)
    result = _propagate_min_behind(disparity, mask, iterations)
    edges = depth_discontinuity_mask(
        disparity, near=near, far=far, threshold=discontinuity
    )

    height, width = disparity.shape
    focal = focal_from_fov(height, fov_degrees)
    cx, cy = (width - 1) * 0.5, (height - 1) * 0.5
    count, labels_cc, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    for index in range(1, count):
        x, y, box_w, box_h, area = (int(v) for v in stats[index])
        if area < _PLANE_MIN_AREA:
            continue
        margin = 12
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(width, x + box_w + margin), min(height, y + box_h + margin)
        component = labels_cc[y0:y1, x0:x1] == index
        values = _fit_component_planes(
            disparity[y0:y1, x0:x1],
            component,
            mask[y0:y1, x0:x1],
            edges[y0:y1, x0:x1],
            focal,
            cx - x0,
            cy - y0,
            near,
            far,
        )
        if values is None:
            continue
        region = result[y0:y1, x0:x1]
        region[component] = values
        result[y0:y1, x0:x1] = region

    # Soften only inside the fill; the real surface must stay untouched.
    weight = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2.0)
    blurred = cv2.GaussianBlur(result * mask.astype(np.float32), (0, 0), 2.0)
    smoothed = np.divide(blurred, np.maximum(weight, 1e-5))
    return np.where(mask, smoothed, disparity.astype(np.float32)).astype(np.float32)


def inpaint_colour(
    image_rgb: np.ndarray, holes: np.ndarray, radius: int = 5
) -> np.ndarray:
    """Fill masked colour. Classical by default so no download is required."""
    if not holes.any():
        return np.ascontiguousarray(image_rgb)
    bgr = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2BGR)
    mask = (holes.astype(np.uint8)) * 255
    filled = cv2.inpaint(bgr, mask, radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)


def thin_structure_mask(
    disparity: np.ndarray, discontinuity: float = 0.06, width: int = 5
) -> np.ndarray:
    """Near structures too narrow to reconstruct a volume for.

    Tram wires, railings and sign edges are the case where near/far pairing
    across a cut is genuinely ambiguous: a one-pixel wire has no side to bridge
    to, so cutting round it fragments it into single-triangle holes instead of
    reconstructing anything. Leaving those edges connected costs a smear one
    pixel wide, which is by far the better failure.

    A grey-scale opening erases any bright structure narrower than its kernel,
    so what the opening removed *is* the set of thin near structures. The
    threshold is compared in disparity, which only gates whether a structure is
    worth protecting — the cut itself still uses the scale-invariant depth ratio.
    """
    width = max(3, int(width) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
    opened = cv2.morphologyEx(disparity.astype(np.float32), cv2.MORPH_OPEN, kernel)
    return (disparity.astype(np.float32) - opened) > float(discontinuity)


def extend_edge_colour(
    image_rgb: np.ndarray, interior: np.ndarray, band: np.ndarray
) -> np.ndarray:
    """Continue the colour of `interior` outward into `band`.

    Nearest-valid-pixel replication rather than inpainting, and deliberately so:
    this band is the *side* of a photographed object, and the only evidence for
    what it looks like is the object's own rim. A general inpainter is trained
    to agree with everything around a hole, so here it would blend the
    background straight back in — the exact surface the wall exists to hide.
    """
    if not band.any() or not interior.any():
        return np.ascontiguousarray(image_rgb)
    # Each interior pixel gets its own label; every other pixel inherits the
    # label of the nearest one, which is the lookup we want in a single pass.
    _distance, labels = cv2.distanceTransformWithLabels(
        (~interior).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    ys, xs = np.nonzero(interior)
    source_y = np.zeros(int(labels.max()) + 1, np.int32)
    source_x = np.zeros_like(source_y)
    own = labels[ys, xs]
    source_y[own], source_x[own] = ys, xs

    result = np.ascontiguousarray(image_rgb).copy()
    band_y, band_x = np.nonzero(band)
    nearest = labels[band_y, band_x]
    result[band_y, band_x] = image_rgb[source_y[nearest], source_x[nearest]]
    return result


def paint_wall_band(
    image_rgb: np.ndarray,
    coverage: np.ndarray,
    reach: int,
    lip: int = LIP_PIXELS,
    inpainter=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Give a layer's silhouette the strip of texture its walls will sample.

    Bridging a cut produces faces that were never photographed, so they need
    texels of their own. Those texels have to live somewhere in the layer's one
    texture; the band immediately outside the silhouette is free — no base
    geometry reaches it — and a wall's far rim samples its own grid texel,
    which lands squarely inside that band.

    `inpainter(image, mask) -> image` is optional and refines the band after the
    edge extension has filled it, so the model continues real structure instead
    of inventing content from the background it can also see.
    """
    reach = max(0, int(reach))
    if reach <= 0 or not coverage.any() or coverage.all():
        return np.ascontiguousarray(image_rgb), np.zeros(coverage.shape, bool)
    kernel = np.ones((3, 3), np.uint8)
    solid = coverage.astype(np.uint8)
    inner = (
        cv2.dilate(solid, kernel, iterations=lip).astype(bool) if lip > 0 else coverage
    )
    outer = cv2.dilate(solid, kernel, iterations=lip + reach).astype(bool)
    band = outer & ~inner
    if not band.any():
        return np.ascontiguousarray(image_rgb), band

    painted = extend_edge_colour(image_rgb, coverage, band)
    if inpainter is not None:
        try:
            refined = inpainter(painted, band)
            if refined is not None and refined.shape == painted.shape:
                painted = refined
        except Exception:  # noqa: BLE001 - the extension is already usable
            pass
    return painted, band


def add_edge_thickness(
    disparity: np.ndarray,
    coverage: np.ndarray,
    thickness: float = 0.06,
    falloff: int = 6,
) -> np.ndarray:
    """Bevel a layer's silhouette backwards so it reads as a solid object.

    Cutting the mesh at depth edges stops foreground objects smearing into the
    distance, but leaves them infinitely thin — a cardboard cutout that vanishes
    edge-on. Rolling the disparity back near the boundary gives the silhouette
    an implied thickness and turns the cut into a rim facing away from camera.
    """
    if thickness <= 0 or not coverage.any():
        return disparity.astype(np.float32)
    inside = cv2.distanceTransform(coverage.astype(np.uint8), cv2.DIST_L2, 3)
    # 0 at the silhouette, 1 once we are `falloff` pixels inside it.
    ramp = np.clip(inside / max(float(falloff), 1e-3), 0.0, 1.0)
    result = disparity.astype(np.float32).copy()
    # Stabilise depth *along* the silhouette. Monocular maps can alternate
    # sharply from row to row at a roof/person edge; that turns into comb-like
    # horizontal shards when the camera moves sideways. A coverage-normalised
    # blur keeps the average on the photographed object instead of mixing in
    # the far sky outside it, and affects only the narrow bevel band.
    sigma = max(1.0, float(falloff) * 0.45)
    weights = cv2.GaussianBlur(coverage.astype(np.float32), (0, 0), sigma)
    smoothed = cv2.GaussianBlur(
        disparity.astype(np.float32) * coverage.astype(np.float32), (0, 0), sigma
    ) / np.maximum(weights, 1e-5)
    edge_mix = np.clip(1.0 - ramp, 0.0, 1.0)
    result[coverage] = (
        result[coverage] * (1.0 - edge_mix[coverage])
        + smoothed[coverage] * edge_mix[coverage]
    )
    result[coverage] -= thickness * (1.0 - ramp[coverage])
    return np.clip(result, 0.0, 1.0)


def _to_shape(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Resample a working-size mask onto the texture it has to line up with."""
    if mask.shape[:2] == tuple(shape[:2]):
        return mask.astype(bool)
    return cv2.resize(
        mask.astype(np.uint8),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def build_layered_scene(
    image_rgb: np.ndarray,
    disparity: np.ndarray,
    layer_count: int = 2,
    fov_degrees: float = 55.0,
    near: float = 1.0,
    far: float = 12.0,
    stride: int = 1,
    discontinuity: float = 0.06,
    dilate: int = 6,
    overscan: float = 0.18,
    thickness: float = 0.06,
    wall_extent: float = 0.12,
    wall_band: int = 6,
    thin_width: int = 5,
    trim_sky: bool = False,
    detail: float = 1.0,
    texture_rgb: np.ndarray | None = None,
    colour_inpainter=None,
    progress=None,
) -> LayeredScene:
    """Reconstruct the scene as depth-ordered layers with hidden areas filled.

    `colour_inpainter(image, mask) -> image` allows swapping in a learned model
    later; the classical filler is used when none is supplied.

    Depth edges are *cut* and then bridged with oriented walls. The earlier
    softening kept the mesh watertight by smearing every cliff into a short
    ramp, which reads as an estimate rather than a reconstruction. `wall_band`
    is how many pixels of rim texture those walls are given to sample.

    `wall_extent` is the fraction of the near-to-far gap a wall runs back, and
    the default is deliberately small. Bridging the whole way closes the opening
    on its own, which is the right answer for a lone mesh — but here the
    backdrop layer behind is already complete and watertight, so a full bridge
    buys nothing and extrudes every object into a beam running to the horizon.
    Enough depth to read as a solid side is all the wall has to supply.
    `thin_width` is the width below which a near structure is left connected
    instead of cut — see `thin_structure_mask`. `detail` below 1.0 turns on
    adaptive quad retopology, which trades flat-surface triangles for build
    speed and GPU memory without touching silhouettes.
    """
    if image_rgb.shape[:2] != disparity.shape[:2]:
        raise ValueError(
            f"image {image_rgb.shape[:2]} and depth {disparity.shape[:2]} differ"
        )
    layer_count = max(1, int(layer_count))
    if texture_rgb is not None and texture_rgb.shape[:2] != image_rgb.shape[:2]:
        # Mesh density follows the working-size depth map, but the colour can be
        # sampled from the original photograph. UVs are normalised, so a
        # higher-resolution texture costs nothing but memory.
        scale_y = texture_rgb.shape[0] / image_rgb.shape[0]
        scale_x = texture_rgb.shape[1] / image_rgb.shape[1]
    else:
        texture_rgb, scale_y, scale_x = None, 1.0, 1.0
    removed_sky = sky_mask(disparity, image_rgb) if trim_sky else None
    labels = segment_layers(disparity, layer_count)
    protected_background = visible_background_mask(disparity)
    labels[protected_background] = 0
    backdrop_holes = background_hole_mask(disparity, labels, discontinuity)
    spare_thin = (
        thin_structure_mask(disparity, discontinuity, thin_width)
        if thin_width > 0
        else None
    )
    inpaint = colour_inpainter or inpaint_colour
    kernel = np.ones((3, 3), np.uint8)

    layers: list[Layer] = []
    meshes: list[Mesh] = []
    for index in range(layer_count):
        own = labels == index
        if index == 0:
            # The farthest layer must span the whole frame: it is the backdrop
            # every other layer can reveal.
            coverage = np.ones_like(own, dtype=bool)
            if removed_sky is not None:
                # Leaving a hole where the sky was is the point: the dome or
                # the clear colour shows through it.
                coverage &= ~removed_sky
        else:
            coverage = labels >= index
            if not coverage.any():
                continue
        occluded = backdrop_holes if index == 0 else coverage & ~own
        if occluded.any() and dilate > 0:
            # Grow slightly past the silhouette so the seam is hidden behind the
            # occluder rather than landing exactly on its edge.
            occluded = cv2.dilate(
                occluded.astype(np.uint8), kernel, iterations=int(dilate)
            ).astype(bool) & coverage

        source = image_rgb if texture_rgb is None else texture_rgb
        if occluded.any():
            if progress:
                progress(f"Reconstructing hidden areas · layer {index + 1} of {layer_count}…")
            mask = occluded
            if texture_rgb is not None:
                mask = cv2.resize(
                    occluded.astype(np.uint8),
                    (texture_rgb.shape[1], texture_rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            colour = inpaint(source, mask)
            filled = fill_disparity_behind(
                disparity,
                occluded,
                fov_degrees=fov_degrees,
                near=near,
                far=far,
                discontinuity=discontinuity,
            )
        else:
            colour = np.ascontiguousarray(source)
            filled = disparity.astype(np.float32).copy()
        _ = scale_y, scale_x

        if index > 0:
            # Only occluders get a bevel; the backdrop has no free silhouette.
            # The bevel is no longer load-bearing now that cuts are bridged, but
            # it still stabilises depth *along* a silhouette, which is what stops
            # a noisy roofline shearing into comb-like shards.
            filled = add_edge_thickness(filled, own, thickness)

        if index > 0 and wall_band > 0:
            # Walls need texels of their own, and the strip just outside the
            # silhouette is free: no base geometry reaches it, so painting there
            # is invisible except through the walls that sample it.
            band_scale = colour.shape[1] / max(disparity.shape[1], 1)
            painted, band = paint_wall_band(
                colour,
                _to_shape(coverage, colour.shape),
                reach=max(1, round(wall_band * band_scale)),
                lip=max(1, round(LIP_PIXELS * band_scale)),
            )
            colour = painted
            if band.shape == occluded.shape:
                occluded = occluded | band

        layers.append(Layer(colour, filled, coverage, occluded, index))

        if progress:
            progress(f"Building 3D geometry · layer {index + 1} of {layer_count}…")

        mesh_colour, mesh_disparity, mesh_valid = colour, filled, coverage
        if index > 0:
            # Segmentation walls are deliberately removed before connected
            # components are found, leaving a narrow unlabelled band around a
            # foreground object. Extend only the rendered silhouette into that
            # band; the photographed colour is still present there and hides
            # base-layer seams without changing layer ownership/inpaint masks.
            mesh_valid = cv2.dilate(
                coverage.astype(np.uint8), kernel, iterations=2
            ).astype(bool)
            # The added lip must inherit foreground depth as well as coverage.
            # Leaving its original (usually far/sky) depth connected a near
            # roof pixel to a distant neighbour and recreated the very spikes
            # the overlap is intended to hide. Max disparity propagates the
            # nearest photographed surface outward by only those two pixels.
            extended_depth = cv2.dilate(
                filled.astype(np.float32), kernel, iterations=2
            )
            lip = mesh_valid & ~coverage
            mesh_disparity = np.where(lip, extended_depth, filled).astype(np.float32)
        focal_height: int | None = None
        if index == 0 and overscan > 0:
            # Extend the backdrop past the photograph. Turning the camera
            # otherwise reveals the edge of the world, which is a different
            # problem from disocclusion and cannot be inpainted away.
            mesh_colour, mesh_disparity, mesh_valid, focal_height = _extend_backdrop(
                colour, filled, overscan, coverage
            )

        meshes.append(
            build_mesh(
                mesh_disparity,
                fov_degrees=fov_degrees,
                near=near,
                far=far,
                stride=stride,
                # Cut at every real depth cliff, then bridge the openings with
                # walls that stand perpendicular to the image plane. Leaving the
                # quads connected is the rubber sheet; cutting without bridging
                # is a hole; this is the reconstruction.
                #
                # The farthest layer is the exception and stays watertight. It
                # is the safety surface every cut in front of it falls back to,
                # and there is nothing behind it to show through a hole its own
                # cuts failed to bridge.
                discontinuity=0.0 if index == 0 else discontinuity,
                valid=mesh_valid,
                focal_height=focal_height,
                no_cut=None if index == 0 else spare_thin,
                bridge=True,
                wall_extent=wall_extent,
                detail=detail,
            )
        )
        if index == 0 and overscan > 0:
            # Every array on a Layer must share one shape, so the synthetic mask
            # is padded alongside the colour and disparity it describes.
            pad_y = (mesh_disparity.shape[0] - occluded.shape[0]) // 2
            pad_x = (mesh_disparity.shape[1] - occluded.shape[1]) // 2
            # value=1: the ring beyond the photograph is entirely invented, so
            # the reconstruction overlay must flag it. It previously came back
            # 0% synthetic while being 46% of the texture, which made the
            # overlay quietly understate how far a camera move had strayed.
            padded_synthetic = cv2.copyMakeBorder(
                occluded.astype(np.uint8), pad_y, pad_y, pad_x, pad_x,
                cv2.BORDER_CONSTANT, value=1,
            ).astype(bool)
            layers[-1] = Layer(
                mesh_colour, mesh_disparity, mesh_valid, padded_synthetic, index
            )

    return LayeredScene(layers, meshes)


def _extend_backdrop(
    colour: np.ndarray,
    disparity: np.ndarray,
    overscan: float,
    coverage: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Pad the backdrop outward by replicating its border, then defocusing it.

    Replication rather than inpainting: beyond the frame there is no evidence at
    all, and a learned model would invent detail that draws the eye to exactly
    the region the viewer should not be looking at.

    Raw replication, though, produces hard streaks radiating from every border
    pixel, which draws the eye just as badly. Blurring the ring progressively
    with distance from the frame keeps it reading as out-of-focus surround
    instead of broken texture. It is still invented, and `build_layered_scene`
    marks it synthetic so the reconstruction overlay says so.
    """
    height, width = disparity.shape[:2]
    pad_y = int(round(height * overscan))
    pad_x = int(round(width * overscan))
    if pad_y <= 0 and pad_x <= 0:
        base = np.ones_like(disparity, bool) if coverage is None else coverage
        return colour, disparity, base, height

    # The colour texture may be higher resolution than the depth map, so it is
    # padded by the same *fraction* rather than the same pixel count.
    colour_pad_y = int(round(colour.shape[0] * overscan))
    colour_pad_x = int(round(colour.shape[1] * overscan))
    padded_colour = cv2.copyMakeBorder(
        np.ascontiguousarray(colour),
        colour_pad_y, colour_pad_y, colour_pad_x, colour_pad_x,
        cv2.BORDER_REPLICATE,
    )
    padded_disparity = cv2.copyMakeBorder(
        disparity.astype(np.float32), pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE
    )
    padded_colour = _defocus_border(
        padded_colour, colour_pad_y, colour_pad_x
    )
    # Pad the caller's coverage rather than replacing it. Returning a solid
    # mask here silently discarded any region the backdrop was told to skip —
    # which made sky trimming a no-op, since the trim is expressed as coverage.
    if coverage is None:
        valid = np.ones(padded_disparity.shape, dtype=bool)
    else:
        valid = cv2.copyMakeBorder(
            coverage.astype(np.uint8), pad_y, pad_y, pad_x, pad_x,
            cv2.BORDER_CONSTANT, value=1,
        ).astype(bool)
    return padded_colour, padded_disparity, valid, height


def _defocus_border(padded: np.ndarray, pad_y: int, pad_x: int) -> np.ndarray:
    """Blend a blurred copy in, ramping from the frame edge outward."""
    if pad_y <= 0 and pad_x <= 0:
        return padded
    height, width = padded.shape[:2]
    radius = max(3, (max(pad_y, pad_x) // 4) * 2 + 1)
    blurred = cv2.GaussianBlur(padded, (radius, radius), 0)
    rows = np.zeros(height, np.float32)
    cols = np.zeros(width, np.float32)
    if pad_y > 0:
        ramp = np.linspace(1.0, 0.0, pad_y, dtype=np.float32)
        rows[:pad_y] = ramp
        rows[height - pad_y :] = ramp[::-1]
    if pad_x > 0:
        ramp = np.linspace(1.0, 0.0, pad_x, dtype=np.float32)
        cols[:pad_x] = ramp
        cols[width - pad_x :] = ramp[::-1]
    weight = np.maximum(rows[:, None], cols[None, :])[..., None]
    mixed = padded.astype(np.float32) * (1.0 - weight) + blurred.astype(np.float32) * weight
    return np.clip(mixed, 0, 255).astype(np.uint8)
