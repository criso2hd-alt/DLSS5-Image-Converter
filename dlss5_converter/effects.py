"""A curated post-processing stack applied to the finished image.

This is the app's answer to "let me use ReShade shaders". Rather than host
ReShade's ``.fx`` runtime — which the harness cannot feed, because the DLSS
result is read back from the NGX output texture, not the swapchain ReShade
draws on — the popular effects are reimplemented natively and run *after* the
neural pass, exactly where the colour grade runs.

That placement is the whole point. Like the grade, every effect here is instant
and non-destructive: the DLSS result is computed once and cached, and moving an
effect slider re-runs only this cheap CPU pass, never the ~3.5 s harness. It is
also why a settings change costs nothing when nothing is enabled — a neutral
stack returns the input array untouched.

Everything operates on **display-referred sRGB float**, HxWx3 in ``[0, 1]`` —
the space ``grade.apply`` outputs. Effects like scanlines, LUTs and grain are
defined in that space (a film LUT is authored against sRGB, a scanline is a
display artefact), so this is both the correct space and the one the rest of
the display path already speaks.

The order is fixed and deliberate: tonal and detail work first (bloom, sharpen),
then the look (chromatic aberration, LUT), then the display simulation (CRT
curvature, scanlines, mask), then framing (vignette), and grain last so it sits
on top of everything as real film grain does.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np

from .settings import EffectsSettings

#: Rec.709 luma, matching grade.py so "brightness" means one thing everywhere.
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


# --- individual effects -----------------------------------------------------
#
# Each takes and returns display sRGB float32 in [0, 1]. They assume their
# caller has already decided they are enabled; the neutral short-circuit lives
# in apply().


def _sharpen(img: np.ndarray, amount: float, radius: float) -> np.ndarray:
    """Unsharp mask. amount 0..2 is the weight of the high-pass, radius its px."""
    radius = max(0.3, float(radius))
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    return img + np.float32(amount) * (img - blurred)


def _bloom(img: np.ndarray, threshold: float, intensity: float, radius: float) -> np.ndarray:
    """Bleed light out of the brightest areas, the way a bright CRT or a lens does."""
    threshold = float(np.clip(threshold, 0.0, 0.999))
    luma = (img * _LUMA).sum(axis=-1, keepdims=True)
    # Everything above the threshold, renormalised so the knee is soft rather
    # than a hard cut that shows as a visible edge around highlights.
    mask = np.clip((luma - threshold) / (1.0 - threshold), 0.0, 1.0)
    bright = img * mask
    blurred = cv2.GaussianBlur(bright, (0, 0), max(1.0, float(radius)))
    return img + np.float32(intensity) * blurred


def _chromatic_aberration(img: np.ndarray, amount: float) -> np.ndarray:
    """Split the red and blue channels radially, a lens/retro-display tell.

    Scales the red channel very slightly out and the blue slightly in about the
    centre — at most ~1% — which reads as colour fringing that grows toward the
    corners, exactly like the real thing.
    """
    height, width = img.shape[:2]
    k = float(amount) * 0.01
    if k <= 0.0:
        return img
    centre = (width / 2.0, height / 2.0)

    def scaled(channel: np.ndarray, scale: float) -> np.ndarray:
        matrix = cv2.getRotationMatrix2D(centre, 0.0, scale)
        return cv2.warpAffine(
            channel, matrix, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )

    out = img.copy()
    out[..., 0] = scaled(img[..., 0], 1.0 + k)   # red pushed out
    out[..., 2] = scaled(img[..., 2], 1.0 - k)   # blue pulled in
    return out


def _apply_cube(img: np.ndarray, lut: "CubeLUT", amount: float, preserve_range: bool) -> np.ndarray:
    """Trilinearly interpolate the image through a 3D .cube LUT and blend.

    A .cube LUT is defined only over its domain (almost always ``[0, 1]``), so an
    HDR image has to be clamped into that domain before the lookup. To honour
    "keep the range" we do not throw the excess away: the over-domain part is
    carried around the LUT and added back, so a highlight above white keeps its
    energy instead of being pinned to the brightest LUT entry.
    """
    if preserve_range:
        lo, hi = lut.domain_min, lut.domain_max
        clamped = np.clip(img, lo, hi)
        residual = img - clamped          # the HDR highlights the LUT cannot see
        graded = lut.apply(clamped) + residual
    else:
        graded = lut.apply(img)
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount >= 1.0:
        return graded
    return img * (1.0 - amount) + graded * amount


def _crt_curvature(img: np.ndarray, amount: float) -> np.ndarray:
    """Barrel-distort the frame so it bulges like a CRT tube face.

    Cached maps would be faster, but curvature is the one CRT knob people leave
    off, so paying for the remap only when it is on is the right trade.
    """
    k = float(amount) * 0.28
    if k <= 0.0:
        return img
    height, width = img.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    # Normalised to [-1, 1] with the centre at 0, so the distortion is radial.
    nx = (xs / (width - 1)) * 2.0 - 1.0
    ny = (ys / (height - 1)) * 2.0 - 1.0
    r2 = nx * nx + ny * ny
    factor = 1.0 + k * r2
    map_x = ((nx * factor + 1.0) * 0.5) * (width - 1)
    map_y = ((ny * factor + 1.0) * 0.5) * (height - 1)
    return cv2.remap(
        img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0.0, 0.0, 0.0),
    )


def _crt_scanlines(img: np.ndarray, strength: float) -> np.ndarray:
    """Darken alternate horizontal lines, the defining CRT artefact."""
    height = img.shape[0]
    # A soft sine between full and dimmed, period two pixels, rather than a hard
    # on/off which aliases badly when the image is later resized.
    phase = np.arange(height, dtype=np.float32) * np.pi
    line = 1.0 - float(strength) * 0.5 * (1.0 - np.cos(phase))
    return img * line[:, None, None]


def _crt_mask(img: np.ndarray, strength: float) -> np.ndarray:
    """An aperture-grille phosphor mask: tint successive columns R, G, B."""
    width = img.shape[1]
    mask = np.ones((width, 3), dtype=np.float32)
    dim = 1.0 - float(strength)
    # Each column favours one phosphor and dims the other two.
    mask[0::3] = (1.0, dim, dim)
    mask[1::3] = (dim, 1.0, dim)
    mask[2::3] = (dim, dim, 1.0)
    return img * mask[None, :, :]


def _vignette(img: np.ndarray, amount: float, feather: float) -> np.ndarray:
    """Darken the corners toward the frame edge."""
    height, width = img.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    nx = (xs / (width - 1)) * 2.0 - 1.0
    ny = (ys / (height - 1)) * 2.0 - 1.0
    dist = np.sqrt(nx * nx + ny * ny) / np.sqrt(2.0)  # 0 centre, 1 corner
    inner = float(np.clip(feather, 0.0, 0.95)) * 0.9
    falloff = np.clip((dist - inner) / max(1e-3, 1.0 - inner), 0.0, 1.0)
    # Smoothstep, so the darkening eases in rather than starting with a ring.
    falloff = falloff * falloff * (3.0 - 2.0 * falloff)
    shade = 1.0 - float(amount) * falloff
    return img * shade[:, :, None]


def _grain(img: np.ndarray, amount: float, size: float) -> np.ndarray:
    """Add monochrome film grain, coarsened by size."""
    height, width = img.shape[:2]
    size = max(1.0, float(size))
    small_h = max(1, int(height / size))
    small_w = max(1, int(width / size))
    # Generated small and scaled up, so `size` gives genuinely coarser grain
    # rather than just more of the same fine noise.
    noise = np.random.default_rng().standard_normal((small_h, small_w)).astype(np.float32)
    if (small_h, small_w) != (height, width):
        noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_LINEAR)
    return img + (float(amount) * 0.15) * noise[:, :, None]


# --- LUT loading ------------------------------------------------------------


class CubeLUT:
    """A parsed 3D ``.cube`` lookup table, ready to apply.

    Only 3D LUTs are supported — the format most film and CRT packs ship — and
    a 1D LUT or a malformed file raises rather than silently doing nothing, so
    the UI can say why a chosen file did not take.
    """

    def __init__(self, size: int, table: np.ndarray, domain_min: np.ndarray, domain_max: np.ndarray) -> None:
        self.size = size
        # Stored as [size, size, size, 3] indexed [b, g, r] — .cube runs red
        # fastest, so red is the last spatial axis.
        self.table = table
        self.domain_min = domain_min
        self.domain_max = domain_max

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Trilinearly sample display sRGB float through the table."""
        size = self.size
        lo, hi = self.domain_min, self.domain_max
        normalised = (np.clip(img, lo, hi) - lo) / np.maximum(hi - lo, 1e-6)
        coords = normalised * (size - 1)

        base = np.floor(coords).astype(np.int32)
        base = np.clip(base, 0, size - 1)
        nxt = np.clip(base + 1, 0, size - 1)
        frac = coords - base

        r0, g0, b0 = base[..., 0], base[..., 1], base[..., 2]
        r1, g1, b1 = nxt[..., 0], nxt[..., 1], nxt[..., 2]
        fr = frac[..., 0:1]
        fg = frac[..., 1:2]
        fb = frac[..., 2:3]

        t = self.table  # [b, g, r, 3]

        def corner(bi, gi, ri):
            return t[bi, gi, ri]

        # Standard trilinear blend across the eight surrounding table entries.
        c00 = corner(b0, g0, r0) * (1 - fr) + corner(b0, g0, r1) * fr
        c01 = corner(b0, g1, r0) * (1 - fr) + corner(b0, g1, r1) * fr
        c10 = corner(b1, g0, r0) * (1 - fr) + corner(b1, g0, r1) * fr
        c11 = corner(b1, g1, r0) * (1 - fr) + corner(b1, g1, r1) * fr
        c0 = c00 * (1 - fg) + c01 * fg
        c1 = c10 * (1 - fg) + c11 * fg
        return (c0 * (1 - fb) + c1 * fb).astype(np.float32)


def load_cube(path: Path) -> CubeLUT:
    """Parse a ``.cube`` file into a CubeLUT, or raise with a plain reason."""
    size: int | None = None
    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    values: list[tuple[float, float, float]] = []

    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise ValueError(f"Could not read the LUT: {error}") from error

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        if upper.startswith("TITLE"):
            continue
        if upper.startswith("LUT_1D_SIZE"):
            raise ValueError("This is a 1D LUT; only 3D .cube LUTs are supported.")
        if upper.startswith("LUT_3D_SIZE"):
            size = int(line.split()[-1])
            continue
        if upper.startswith("DOMAIN_MIN"):
            domain_min = np.array([float(v) for v in line.split()[1:4]], dtype=np.float32)
            continue
        if upper.startswith("DOMAIN_MAX"):
            domain_max = np.array([float(v) for v in line.split()[1:4]], dtype=np.float32)
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                values.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue

    if size is None:
        raise ValueError("No LUT_3D_SIZE found — not a 3D .cube LUT.")
    if len(values) != size * size * size:
        raise ValueError(
            f"LUT is {size}x{size}x{size} but has {len(values)} entries, not "
            f"{size ** 3}."
        )

    # .cube lists red fastest, then green, then blue, so the natural reshape is
    # [b, g, r, 3] and no transpose is needed.
    table = np.array(values, dtype=np.float32).reshape(size, size, size, 3)
    return CubeLUT(size, table, domain_min, domain_max)


# --- the stack --------------------------------------------------------------


def apply(
    image_rgb: np.ndarray,
    settings: EffectsSettings,
    luts_dir: Path | None = None,
    preserve_range: bool = False,
) -> np.ndarray:
    """Run the enabled effects over display sRGB float, returning the same.

    A neutral stack returns the input untouched — the same free-when-off
    contract the grade honours, so nothing here costs anything until a user
    turns an effect on.

    ``preserve_range`` is the HDR path. Off (the default, SDR), the image is
    clamped to ``[0, 1]`` in and out, which is what an 8-bit screen or PNG
    wants. On, the clamps are removed: an HDR source — EXR, JPEG XR — keeps
    highlights above white through the whole stack instead of having them
    crushed to 1.0. Every effect here is a multiply or an add (the LUT carries
    its over-domain residual, see _apply_cube), so all of them survive that
    change with their range intact.
    """
    if settings.is_neutral:
        return image_rgb

    if preserve_range:
        img = np.asarray(image_rgb, dtype=np.float32).copy()
    else:
        img = np.clip(image_rgb, 0.0, 1.0).astype(np.float32, copy=True)

    if settings.bloom_enabled:
        img = _bloom(img, settings.bloom_threshold, settings.bloom_intensity, settings.bloom_radius)
    if settings.sharpen_enabled:
        img = _sharpen(img, settings.sharpen_amount, settings.sharpen_radius)
    if settings.chroma_enabled:
        img = _chromatic_aberration(img, settings.chroma_amount)
    if settings.lut_enabled and settings.lut_name and luts_dir is not None:
        lut = _load_lut_cached(Path(luts_dir) / settings.lut_name)
        if lut is not None:
            img = _apply_cube(img, lut, settings.lut_amount, preserve_range)
    if settings.crt_enabled:
        if settings.crt_curvature:
            img = _crt_curvature(img, settings.crt_curvature)
        if settings.crt_scanline:
            img = _crt_scanlines(img, settings.crt_scanline)
        if settings.crt_mask:
            img = _crt_mask(img, settings.crt_mask)
    if settings.vignette_enabled:
        img = _vignette(img, settings.vignette_amount, settings.vignette_feather)
    if settings.grain_enabled:
        img = _grain(img, settings.grain_amount, settings.grain_size)

    # HDR keeps its highlights; only the floor is held at zero, since negative
    # light is meaningless and grain or sharpen can dip a dark pixel below it.
    return np.maximum(img, 0.0) if preserve_range else np.clip(img, 0.0, 1.0)


#: One-entry LUT cache: a slider drag re-runs apply() many times a second and
#: reparsing a 33^3 .cube each tick would stutter. Keyed by path and mtime so an
#: edited LUT is picked up, not served stale.
_LUT_CACHE: dict[tuple[str, float], CubeLUT] = {}


def _load_lut_cached(path: Path) -> CubeLUT | None:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    key = (str(path), mtime)
    cached = _LUT_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        lut = load_cube(path)
    except ValueError:
        # A bad LUT must degrade to "no LUT", never take down a conversion. The
        # Effects tab validates on selection and is where the reason is shown.
        return None
    _LUT_CACHE.clear()  # only the current LUT is worth keeping resident
    _LUT_CACHE[key] = lut
    return lut


def available_luts(luts_dir: Path) -> list[str]:
    """The .cube files the user has dropped in, by name, for the picker."""
    try:
        return sorted(p.name for p in Path(luts_dir).glob("*.cube"))
    except OSError:
        return []


def describe(settings: EffectsSettings) -> str:
    """A short "Sharpen, CRT, Vignette" summary of what is on, for tooltips."""
    on = []
    for f in fields(settings):
        if f.name.endswith("_enabled") and getattr(settings, f.name):
            on.append(f.name[: -len("_enabled")].replace("_", " ").title())
    return ", ".join(on) if on else "None"
