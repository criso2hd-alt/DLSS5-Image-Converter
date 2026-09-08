"""Everything the user can turn, in one serialisable place."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .depth_engine import DEFAULT_MODEL
from .grade import GradeSettings


#: The add-on's own combo items, in its order. The ini stores the *index*, so
#: these lists are the mapping and their order is not ours to change. Recovered
#: from the add-on binary and confirmed by measuring each value's output.
NR_PRESETS = ("Default", "Preset #1", "Preset #2", "Preset #3")
#: The add-on's NRStyle combo. It has THREE entries and Default is index 0 - the
#: look you get the moment DLSS 5 is switched on, before choosing Natural or
#: Cinematic. An earlier revision listed only ("Natural", "Cinematic"), which
#: put our index 0 ("Natural") on the add-on's Default and made Cinematic
#: unreachable; confirmed against the RenoDX add-on's own UI. The ini stores the
#: index, so this order must match the add-on's.
NR_STYLES = ("Default", "Natural", "Cinematic")


def style_slug(style_index: int) -> str:
    """The style's lowercase name for a filename, e.g. 'natural'.

    Written into the output name (…_dlss5_natural.png) so a folder of results
    says which look each was made with - a request from users comparing styles.
    Clamped, so a stored index past the end of NR_STYLES still yields a name.
    """
    clamped = min(len(NR_STYLES) - 1, max(0, int(style_index)))
    return NR_STYLES[clamped].lower()

#: Top of the add-on's own strength sliders, and measured to be real: on a
#: photograph the output keeps changing from 1.0 through 2.0 and then stops dead
#: at exactly 2.0. An earlier version of this file said the ceiling was 1.0,
#: which came from testing on a synthetic checkerboard that happened not to
#: respond above 1 — do not trust a saturation claim measured on synthetic
#: input.
NR_STRENGTH_MAX = 2.0

#: The HDR group's ceilings, each measured the same way — raise the value until
#: the output stops changing. They are not all the same and not all 2.0:
#: colour and transfer stop dead at 1.0, paper-white keeps going to 16 (which is
#: also the value real game configs carry) and is flat from there.
NR_COLOR_MAX = 1.0
NR_TRANSFER_MAX = 1.0
NR_PAPER_WHITE_MAX = 16.0

#: Increment only when an existing user should be offered a substantially new
#: tour. Existing settings without this key predate onboarding and are migrated
#: as complete; a genuinely new install starts at zero.
ONBOARDING_VERSION = 1


@dataclass
class NeuralSettings:
    """The RenoDX DLSS 5 add-on's exposed controls.

    Names mirror the add-on's own UI labels so a user who followed a modding
    guide finds what they expect, and so do the ranges: the strengths run
    0..``NR_STRENGTH_MAX`` (2.0), matching the add-on's own sliders, and the two
    enums are indices into the lists above.
    """

    #: One of the add-on's four presets. Exposed for completeness and confirmed
    #: to reach the add-on (it echoes the value back in its log), but all four
    #: measured bit-identical with upscaling off — it most likely picks a Super
    #: Resolution preset, which a DLAA-only path never exercises.
    preset: int = 0
    #: Default, Natural or Cinematic (index into NR_STYLES). Unlike the preset
    #: this is very much live: on a portrait, Cinematic moves the image about 50%
    #: further from the source than Natural does at the same strengths. Default
    #: (0) is the add-on's own starting look.
    style: int = 0
    # Defaults are the maximum (2.0). The whole point of the tool is the neural
    # effect, so it opens fully on and obviously working — the commonest first
    # report on gentler defaults was "it does nothing" — and anyone who finds a
    # face waxy or the relight too strong pulls the sliders down from there.
    #: Overall strength of the neural pass. 0 is a plain DLAA resolve.
    intensity: float = NR_STRENGTH_MAX
    #: Subsurface-scattering and pore-level work on faces. The reason most
    #: people want this tool, and the first thing to lower when output looks
    #: waxy or "yassified".
    skin: float = NR_STRENGTH_MAX
    #: Local tone response — how much the model is allowed to relight.
    local_tone: float = NR_STRENGTH_MAX
    #: Micro-contrast and material structure (fabric weave, hair strands).
    structure: float = NR_STRENGTH_MAX

    # --- HDR group ---------------------------------------------------------
    #
    # The add-on's HDR controls. This pipeline is SDR end to end, and these were
    # left out at first for that reason — but they measurably change an SDR
    # result too, because the neural pass reasons about light transport before
    # anything is tonemapped back. They default to the add-on's own defaults, so
    # leaving them alone reproduces previous behaviour exactly.

    #: How much of the model's colour change is kept. 0 keeps the source colour.
    color_strength: float = 1.0
    #: Strength of the HDR transfer curve the pass works through.
    transfer_strength: float = 1.0
    #: Scene paper-white, the anchor the model treats as diffuse white. Games in
    #: the wild ship 16 here; the add-on's own default is 1. On an HDR/OLED
    #: display this is the control that decides how bright "white" is assumed to
    #: be, and therefore how hard the pass pushes highlights.
    paper_white: float = 1.0


@dataclass
class DepthSettings:
    model_id: str = DEFAULT_MODEL
    #: Native resolution of the depth pass. Higher catches finer silhouettes at
    #: a roughly quadratic cost.
    input_size: int = 518
    #: Tile the depth pass for large images. Slow, but the only way to get
    #: hair-level depth detail out of a 4K portrait.
    tiled: bool = False
    #: Compresses or expands the near-far spread before it becomes hardware
    #: depth. Above 1.0 pushes the scene towards the near plane, which makes the
    #: model treat more of the frame as foreground.
    contrast: float = 1.0


@dataclass
class EvaluationSettings:
    #: How many times the same contract is evaluated. DLSS is temporal and a
    #: single pass leaves the accumulator empty; the neural result visibly firms
    #: up over the first few frames and stops changing by roughly eight.
    frames: int = 8
    #: Halton sub-pixel offsets, resampling the source each frame. This is the
    #: only way a still image gives DLSS the sample diversity it was built
    #: around. It cannot invent information the photo lacks, but it does stop
    #: the accumulator from locking onto one sample grid.
    jitter: bool = True
    #: Cap on the longest edge sent to DLSS. Anything larger is downscaled
    #: first, so this is also the resolution the result comes back at.
    #:
    #: 3840 was chosen as "the ceiling NVIDIA quotes for real-time evaluation",
    #: on the assumption that beyond it VRAM would climb sharply. Measured on a
    #: 16 GB RTX 4080 that assumption was wrong: 8K completes in 25 s using
    #: 5.3 GB, barely more than 4K's 5.2 GB, and the add-on confirms the neural
    #: pass running at full 7680x4320 rather than quietly degrading. The cap
    #: stays at 4K as a *default* because it is the validated size and a sane
    #: first run, not because larger does not work — people doing architectural
    #: renders at 5-6K should raise it.
    max_edge: int = 3840
    #: Re-run DLSS automatically when a neural slider moves.
    #:
    #: Not free, and not a live renderer: the add-on reads its configuration
    #: once when the harness starts, so every change is a fresh process. Measured
    #: on an RTX 4080, that start-up is ~3.5 s and dominates everything else —
    #: the eight evaluations at 4K add 0.6 s and the readback 0.1 s. Previewing
    #: at a lower resolution therefore saves almost nothing, which is why there
    #: is no separate preview size.
    live_preview: bool = False


#: Offered in the sidebar. 8192 is the top because it is the largest verified
#: here; the field accepts anything, so an unusual workflow is not blocked.
MAX_EDGE_CHOICES = (1920, 2560, 3840, 5120, 6144, 7680, 8192)

#: Explicit Boost choices. Centralised so the UI and D3D12-limit guidance can
#: never disagree about a multiplier the person can actually select.
DETAIL_BOOST_FACTORS = (2, 4, 8)

#: Plain names for the Boost factors, shown to users in place of "2×/4×/8×" -
#: the multiplier is an implementation detail nobody outside the code needs.
BOOST_LEVEL_LABELS = {2: "Standard", 4: "High", 8: "Max"}

#: A D3D12 2D texture cannot exceed this on a side. Boost runs DLSS at
#: (working size × factor), so the working size × factor must stay under it -
#: this is the hard limit behind "Boost needs Max size 8192 px or smaller"
#: (8192 × 2 = 16384). Mirrored in pipeline for the conversion-time check.
D3D12_MAX_TEXTURE_DIMENSION = 16384


def max_boost_factor(max_edge: int) -> int:
    """Largest Boost factor whose working size fits the texture limit.

    Returns 0 when even the smallest factor overflows (Max size above 8192),
    which is the signal that Boost cannot run at this Max size at all.
    """
    allowed = [f for f in DETAIL_BOOST_FACTORS if max_edge * f <= D3D12_MAX_TEXTURE_DIMENSION]
    return max(allowed) if allowed else 0


@dataclass
class DetailSettings:
    """How fine detail is recovered after the neural pass.

    DLAA softens genuine photographic texture; this decides how it is given
    back. Flat and neutral-by-default (mode "off"), matching the other settings
    groups. See detail.py for the maths and detail_engine.py for the AI step.
    """

    #: "off"      — leave the DLSS result as-is.
    #: "preserve" — re-inject the source's real high-frequency band (native
    #:              resolution; the faithful, free default).
    #: "boost"    — supersample: upscale the source, crispen, run DLSS at that
    #:              size, then downscale. Slow, punchy, for high-end renders.
    mode: str = "off"
    #: Preserve: how much source detail to blend back, 0..1.
    #: Boost: strength of the pre-DLSS crispen.
    amount: float = 0.75
    #: Gaussian radius of the high/low frequency split, in pixels.
    radius: float = 2.0
    #: Boost only: how far to supersample (2, 4 or 8). Higher processes the
    #: square of the factor in pixels; it is not guaranteed to be sharper on
    #: every source and is limited by live VRAM and the active DLSS runtime.
    supersample: int = 4

    @property
    def is_neutral(self) -> bool:
        return self.mode == "off"


@dataclass
class EffectsSettings:
    """The post-DLSS effects stack — the app's native ReShade-style library.

    Flat rather than a dict of sub-objects on purpose: it mirrors NeuralSettings,
    and the flat shape round-trips through AppSettings.load's field filter with
    no special handling. Every effect is off by default and every default is a
    sensible "on" value, so ticking one is immediately visible without hunting
    for a strength. Applied after the grade; see effects.apply.
    """

    # Sharpen (unsharp mask).
    sharpen_enabled: bool = False
    sharpen_amount: float = 0.6       # 0..2, weight of the high-pass
    sharpen_radius: float = 1.5       # px

    # Bloom (light bleed from highlights).
    bloom_enabled: bool = False
    bloom_threshold: float = 0.75     # 0..1 luma where the glow starts
    bloom_intensity: float = 0.4      # 0..1
    bloom_radius: float = 8.0         # px

    # Chromatic aberration (radial R/B split).
    chroma_enabled: bool = False
    chroma_amount: float = 0.4        # 0..1

    # LUT (.cube from the luts folder).
    lut_enabled: bool = False
    lut_name: str = ""                # filename in paths.luts_dir()
    lut_amount: float = 1.0           # 0..1 blend

    # CRT (scanlines / phosphor mask / tube curvature).
    crt_enabled: bool = False
    crt_scanline: float = 0.4         # 0..1
    crt_mask: float = 0.3             # 0..1
    crt_curvature: float = 0.0        # 0..1

    # Vignette.
    vignette_enabled: bool = False
    vignette_amount: float = 0.4      # 0..1 corner darkening
    vignette_feather: float = 0.5     # 0..1 how far in it reaches

    # Film grain.
    grain_enabled: bool = False
    grain_amount: float = 0.25        # 0..1
    grain_size: float = 1.5           # >=1, coarseness

    @property
    def is_neutral(self) -> bool:
        """True when nothing is on, so effects.apply can return the input as-is."""
        return not any((
            self.sharpen_enabled, self.bloom_enabled, self.chroma_enabled,
            self.lut_enabled, self.crt_enabled, self.vignette_enabled,
            self.grain_enabled,
        ))


@dataclass
class AppSettings:
    neural: NeuralSettings = field(default_factory=NeuralSettings)
    depth: DepthSettings = field(default_factory=DepthSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    #: Applied to the finished image, after the neural pass. Neutral by default,
    #: so it costs nothing until someone touches it.
    grade: GradeSettings = field(default_factory=GradeSettings)
    #: The post-DLSS effects stack, applied after the grade. Also neutral by
    #: default — nothing runs until an effect is turned on.
    effects: EffectsSettings = field(default_factory=EffectsSettings)
    #: Detail recovery (Preserve / Boost / AI sharpen). Neutral by default.
    detail: DetailSettings = field(default_factory=DetailSettings)
    #: Folder holding the user's own nvngx_dlssnr.dll and the RenoDX add-on.
    #: Empty means "search the usual places" (see paths.runtime_search_roots).
    runtime_dir: str = ""
    last_output_dir: str = ""
    #: The colour palette the UI is drawn in, by name (see app.PALETTES). An
    #: unknown value falls back to the default at apply time.
    theme: str = "Neural Cyan"
    #: How multi-slider groups are laid out. "compact" shows a row of parameter
    #: chips over a single slider; "full" stacks every slider at once. Compact by
    #: default because it is what keeps the sidebar from reading as a wall.
    density: str = "compact"
    #: Zero only for a fresh install. Completing or skipping the introduction
    #: writes the current version so normal launches go straight to work.
    onboarding_version: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def load(cls, path: Path) -> AppSettings:
        """Read settings, ignoring anything this version does not understand.

        A settings file written by a newer build must not stop an older one from
        starting, and a key we removed must not raise. Unknown keys are dropped
        and missing ones keep their defaults.
        """
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a corrupt file is not worth a crash
            return cls()
        if not isinstance(raw, dict):
            return cls()

        def build(target, payload):
            if not isinstance(payload, dict):
                return target()
            known = {f.name for f in fields(target)}
            return target(**{k: v for k, v in payload.items() if k in known})

        try:
            onboarding_version = int(raw.get("onboarding_version", ONBOARDING_VERSION))
        except (TypeError, ValueError):
            onboarding_version = ONBOARDING_VERSION

        return cls(
            neural=build(NeuralSettings, raw.get("neural")),
            depth=build(DepthSettings, raw.get("depth")),
            evaluation=build(EvaluationSettings, raw.get("evaluation")),
            # grade and effects are both written by to_json but were not read
            # back here; without these two lines a saved colour grade (and now
            # an effects stack) silently resets to neutral on every restart.
            grade=build(GradeSettings, raw.get("grade")),
            effects=build(EffectsSettings, raw.get("effects")),
            detail=build(DetailSettings, raw.get("detail")),
            runtime_dir=str(raw.get("runtime_dir") or ""),
            last_output_dir=str(raw.get("last_output_dir") or ""),
            theme=str(raw.get("theme") or "Neural Cyan"),
            density=str(raw.get("density") or "compact"),
            # Do not surprise established users with a first-run flow after an
            # update. A settings file with no key is proof this is not a fresh
            # install, so migrate it as already introduced.
            onboarding_version=onboarding_version,
        )

    def save(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.to_json(), encoding="utf-8")
        except OSError:
            # Settings are a convenience. Losing them must never interrupt work.
            pass
