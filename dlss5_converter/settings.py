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
NR_STYLES = ("Natural", "Cinematic")

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
    #: Natural or Cinematic. Unlike the preset this is very much live: on a
    #: portrait, Cinematic moves the image about 50% further from the source
    #: than Natural does at the same strengths.
    style: int = 0
    #: Overall strength of the neural pass. 0 is a plain DLAA resolve.
    intensity: float = 0.65
    #: Subsurface-scattering and pore-level work on faces. The reason most
    #: people want this tool, and the first thing to lower when output looks
    #: waxy or "yassified".
    skin: float = 0.45
    #: Local tone response — how much the model is allowed to relight.
    local_tone: float = 0.40
    #: Micro-contrast and material structure (fabric weave, hair strands).
    structure: float = 0.50

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


@dataclass
class AppSettings:
    neural: NeuralSettings = field(default_factory=NeuralSettings)
    depth: DepthSettings = field(default_factory=DepthSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    #: Applied to the finished image, after the neural pass. Neutral by default,
    #: so it costs nothing until someone touches it.
    grade: GradeSettings = field(default_factory=GradeSettings)
    #: Folder holding the user's own nvngx_dlssnr.dll and the RenoDX add-on.
    #: Empty means "search the usual places" (see paths.runtime_search_roots).
    runtime_dir: str = ""
    last_output_dir: str = ""

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

        return cls(
            neural=build(NeuralSettings, raw.get("neural")),
            depth=build(DepthSettings, raw.get("depth")),
            evaluation=build(EvaluationSettings, raw.get("evaluation")),
            runtime_dir=str(raw.get("runtime_dir") or ""),
            last_output_dir=str(raw.get("last_output_dir") or ""),
        )

    def save(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.to_json(), encoding="utf-8")
        except OSError:
            # Settings are a convenience. Losing them must never interrupt work.
            pass
