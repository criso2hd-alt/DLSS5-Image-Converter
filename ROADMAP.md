# Roadmap

## Where this stands

**Phase 1 is done.** The harness compiles, DLSS evaluates, and the RenoDX add-on
runs the neural pass on a still image in a headless process. Everything
Python-side is tested (`pytest`, 22 cases covering the depth mapping, the sRGB
round trip, the jitter sequence, and the plane formats).

Bring-up results on the reference machine (RTX 4080, driver 616.56, VS 2026,
ReShade 6.8.0.2155, RenoDX DLSS5 Generic v4.1.5):

```
adapter: NVIDIA GeForce RTX 4080
dlss_available: 1
needs_driver_update: 0
neural_addon_loaded: 1
reshade_proxy_loaded: 1
dlssnr_module_loaded: 1
```

The next real work is Phase 2 — the knobs are not connected to anything.

---

## Phase 1 — prove the contract reaches DLSS ✅

**Goal: a picture comes out, and it is not just the picture that went in.**

All four steps pass. What actually happened, since none of it went as predicted:

1. `.\scripts\build_native.ps1` compiled **clean first time**. No NGX macro
   drift; `NGX_D3D12_CREATE_DLSS_EXT` and `NGX_D3D12_EVALUATE_DLSS_EXT` match
   the current public headers as written. Three things around it were wrong
   instead, all now fixed: the SDK repo had moved (`NVIDIA-RTX/DLSS` 404s, it is
   `NVIDIA/DLSS`), the static helper lives in `lib/Windows_x86_64/x64` and not
   `x86_64`, and the script demanded `cmake` on PATH when Visual Studio's bundled
   copy is right there.
2. `--probe` reports `dlss_available: 1` once `nvngx_dlss.dll` sits beside the
   executable. It could not report anything at all before: the availability
   check ran *before* the probe's early return and killed the process with an
   `ERROR` line, so the one command meant to diagnose a broken runtime was the
   one command that could not survive one.
3. Pure DLAA resolve at `intensity 0`, ReShade absent: output correlates with
   input at **1.000000**, mean |Δ| 0.00018. The plane layout and the 256-byte
   row padding in `UploadTexture` are correct.
4. Accumulation is live. Same contract, Halton jitter, rising frame counts:

   | frames | mean abs Δ vs 1 frame | mean abs Δ vs previous |
   | ------ | --------------------- | ---------------------- |
   | 1      | 0.000000              | —                      |
   | 2      | 0.004215              | 0.004215               |
   | 4      | 0.006362              | 0.003038               |
   | 8      | 0.006448              | 0.001498               |
   | 16     | 0.006328              | 0.000647               |

   The per-step delta halves each time and the total plateaus by frame 4–8, which
   is a converging temporal history — so the feature handle really is persisting
   across passes and `--frames` is doing what it claims.

**The known unknown is answered: the add-on hooks a headless process.** A hidden
64×64 swapchain plus a `Present()` per evaluation is enough. ReShade loads from
`native\bin\dxgi.dll`, finds `renodx-dlss5.addon64` beside it, and logs:

```
signed DLSSNR 310.8.0 D3D12 runtime initialized
created inline NR resources 512x512 -> 512x512 (native) format=10
NGX feature create intercepted: feature=18 (DLSSNR/reserved-18), slot=0
inline feature 18 evaluation succeeded
```

With the add-on live the same contract moves 169× further from its input
(mean |Δ| 0.03042 against 0.00018 for plain DLAA, correlation 0.996). None of the
three fallbacks were needed; they are struck from this plan.

### `--frames 1` silently produces no neural pass at all

Not a wart — the sharpest edge found so far, and it is invisible.

The add-on installs its NGX hooks **lazily, on the first NGX call**, and that
first call is therefore not intercepted. Measured on the reference machine, same
3840×1678 image each time:

| `--frames` | NR evaluations | what you get |
| ---------- | -------------- | ------------ |
| 1          | **0**          | plain DLAA, no neural pass, no warning |
| 2          | 1              | neural pass runs |
| 8          | ≥1             | neural pass runs |

At `--frames 1` the log ends `... inline DLSS contract capture armed` and then
unloads 143 ms later having never logged `first NGX evaluate intercepted`. The
one evaluation is spent arming the hooks. The image still comes back, still looks
plausible, and nothing anywhere says the point of the program did not happen —
the exact failure the README warns about, reachable from the GUI, which lets the
frame spinbox go to 1.

Two things to do:

- **Warm up before the real passes.** Issue one throwaway evaluate (or present a
  frame) before the first counted frame, so the hooks are armed by the time
  frame 1 runs. This also removes the `CreateFeature was not intercepted ...
  registering lazily from evaluate contract` warning, since the feature would be
  created behind live hooks.
- Until that exists, **clamp the GUI spinbox to a minimum of 2**, or say plainly
  in the tooltip that 1 disables the neural pass. `settings.frames` defaults to
  8, so only a user who deliberately lowers it is exposed.

Note the add-on logs `evaluation succeeded (count=1)` once and not per frame, so
the log confirms *that* NR ran, never *how many times*.

## Phase 2 — the knobs ✅

**The knobs are not wired to anything.** Neither guessed route was live: the NGX
parameters (`DLSS5.NeuralUplift.*`) are names nothing reads, and the environment
variables are ignored too. The add-on ran at its own defaults during Phase 1 —
`intensity=1.000000 color_strength=1.000000 transfer=1.000000 paper_white=1.000000
preset=0 style=0` — regardless of what the harness was told.

The real interface is the third guess: a ReShade `.ini` beside the executable.
The add-on reads it through ReShade's own `ReShadeGetConfigValue`, under
`[RenoDX.DLSS5]`. A working config, lifted from a game install:

```ini
[RenoDX.DLSS5]
NeuralUplift=1
NRAutoMask=0
NRColorStrength=1
NRDepthMode=0
NREnableUpscaling=0
NRIntensity=2
NRLocalStructure=2
NRLocalTone=2
NRPaperWhiteScale=16
NRPreset=0
NRSkinStructure=2
NRStyle=2
NRTransferStrength=1
NRUICorrection=0
```

### Measured: the ini knobs work, and the range is 0..1

Same 512×512 contract, four passes, only `[RenoDX.DLSS5]` in
`native\bin\ReShade.ini` changed between runs:

| setting | mean abs Δ vs `NeuralUplift=0` |
| ------- | ------------------------------ |
| `NeuralUplift=0` | 0.000000 — bit-identical to plain DLAA |
| `NRIntensity=0`  | 0.004022 |
| `NRIntensity=1`  | 0.030329 |
| `NRIntensity=4`  | 0.030329 — identical to 1 |

Three things follow, and they settle Phase 2:

- **The add-on reads the ini**, at process start. Writing the section before
  launching the harness is all the wiring that is needed.
- **`NeuralUplift=0` is a clean off switch**, reproducing plain DLAA exactly.
  That is the A/B control Phase 3 wants, and a far better "intensity 0" than
  asking the model for a weak pass.
- **`NRIntensity` saturates at 1.0.** The `2` seen in a game's ini is above the
  effective ceiling. The app's sliders are already 0..1, so they map straight
  across with no rescaling. (`NRIntensity=0` still differs slightly from
  `NeuralUplift=0`, so some part of the pass — probably the colour transfer —
  runs regardless of intensity.)

The UI labels, recovered from the add-on binary, name the rest:

| ini key | label | kind |
| ------- | ----- | ---- |
| `NeuralUplift` | Enable DLSS Neural Rendering | bool |
| `NRIntensity` | NR Intensity | float, `%.2f` |
| `NRSkinStructure` | Skin Structure Strength | float |
| `NRLocalTone` | Local Tone Strength | float |
| `NRLocalStructure` | Local Structure Strength | float |
| `NRColorStrength` | Color Strength | float |
| `NRTransferStrength` | HDR Transfer Strength | float |
| `NRPaperWhiteScale` | Scene Paper-White Scale | float, `%.3f` |
| `NRPreset` | Default / Preset #1 / #2 / #3 | enum |
| `NRStyle` | Natural / Cinematic | enum |
| `NRAutoMask`, `NRUICorrection` | masking | bool |
| `NRDepthMode` | Use game NGX flag / Force normal / **Force inverted** | enum |
| `NRMVecScaleX`, `NRMVecScaleY` | motion scale multipliers | float |

`NRDepthMode` is worth Phase 3's attention: it overrides the depth convention
directly, which is the assumption the whole project rests on.

So Phase 2 is now concrete:

- Map the app's four settings onto `NRIntensity`, `NRSkinStructure`,
  `NRLocalTone`, `NRLocalStructure`. Note the values are **small integers**, not
  the 0..1 floats the GUI currently carries — find the range before scaling.
- Write `[RenoDX.DLSS5]` into `native\bin\ReShade.ini` before launching the
  harness. ReShade rewrites that file on exit, so merge into it rather than
  overwriting it.
- Delete both dead paths from `main.cpp`: the `params_->Set("DLSS5.NeuralUplift.*")`
  calls and the `SetEnvironmentVariableA` block in `main()`, plus the comment
  apologising for setting them twice.
- `NREnableUpscaling=0` matches the DLAA-only non-goal below. `NRAutoMask` and
  `NRUICorrection` are the masking controls Phase 4 wants.

### Done

`runtime.write_addon_config` merges `[RenoDX.DLSS5]` into the harness folder's
`ReShade.ini` before every launch, mapping the four sliders straight across at
0..1. Measured through the app, 1280×560, four passes:

| sliders | mean abs Δ vs intensity 0 |
| ------- | ------------------------- |
| intensity 0 | 0.000000 |
| 0.25 | 0.001892 |
| 0.65 (defaults) | 0.008900 |
| 1.0 | 0.029264 |

Monotonic, and the individual knobs separate: skin 1.0 with tone and structure
at 0.1 lands at 0.002032, well away from the 0.0089 the same intensity gives
with balanced settings.

Both dead paths are deleted. The `--intensity` family of CLI flags stays so the
harness is still drivable by hand with the arguments the pipeline has always
passed, but nothing reads them any more.

Intensity 0 writes `NeuralUplift=0` rather than `NRIntensity=0`, because those
are not the same thing — the former is bit-identical to plain DLAA, which is what
the UI promises at zero, and the latter still moves the image slightly.

### Correction: the strength range is 0..2, not 0..1

The saturation figure above was measured on a synthetic checkerboard, and it was
wrong. On a photograph the strengths keep working well past 1.0:

| all four strengths | mean abs Δ vs source (1920×1080 portrait) |
| ------------------ | ----------------------------------------- |
| 0.5 | 0.008668 |
| 1.0 | 0.030466 |
| 1.5 | 0.035824 |
| 2.0 | 0.040560 |
| 3.0 | identical to 2.0 |

So the ceiling is exactly **2.0**, which is also where the add-on's own sliders
stop. The app's sliders now run 0..2 (`NR_STRENGTH_MAX`) and
`write_addon_config` clamps there. Clamping at 1.0, as it briefly did, silently
discarded half the usable range.

The lesson worth keeping: **do not measure a saturation point on synthetic
input.** A gradient-and-checker test card stops responding above 1.0 while a
photograph does not, and the test card is what made 1.0 look like a ceiling.

### Preset does nothing here; Style does a lot

Both are written to the ini and both reach the add-on — it echoes them back as
`preset=3 style=1` in its log. Measured on a portrait at matched strengths:

| setting | vs source | vs preset 0 / Natural |
| ------- | --------- | --------------------- |
| preset 0 / Natural   | 0.030466 | 0.000000 |
| preset 1 / Natural   | 0.030466 | **0.000000** |
| preset 2 / Natural   | 0.030466 | **0.000000** |
| preset 3 / Natural   | 0.030466 | **0.000000** |
| preset 0 / Cinematic | 0.048301 | 0.044842 |
| preset 3 / Cinematic | 0.048301 | 0.044842 |

`NRStyle` is a large, real effect — Cinematic lands ~50% further from the source
than Natural at the same strengths. `NRPreset` is inert: all four presets are
bit-identical, and the style delta is unchanged across them. The likely reason is
that it selects a **Super Resolution** preset, which a DLAA-only path with
`NREnableUpscaling=0` never exercises. It is still exposed, with a tooltip that
says so, because it costs nothing and would matter if upscaling ever landed.

### The HDR group, and its three different ceilings

Surfaced for HDR and OLED users. They were nearly left out on the grounds that
this pipeline is SDR end to end — that reasoning was wrong. The neural pass
reasons about light transport *before* anything is tonemapped back to sRGB, so
these change an SDR result too, measurably:

| change from defaults | mean abs Δ (960×540 portrait, full pipeline) |
| -------------------- | -------------------------------------------- |
| `NRPaperWhiteScale` 1 → 16 | 0.041252 |
| `NRTransferStrength` 1 → 0 | 0.033952 |
| `NRColorStrength` 1 → 0 | 0.013685 |

Each ceiling was found the same way — raise the value until the output stops
changing — and **they are all different**:

| key | ceiling | evidence |
| --- | ------- | -------- |
| `NRColorStrength` | **1.0** | 1.1 and 1.5 bit-identical to 1.0 |
| `NRTransferStrength` | **1.0** | 1.1 and 1.5 bit-identical to 1.0 |
| `NRPaperWhiteScale` | **16.0** | rises through 6/8/10/12; 18 identical to 16 |

16 is also what shipping game configs carry for paper-white, which is a useful
cross-check on a number that would otherwise look arbitrary. The add-on's own
defaults are 1.0 for all three, and so are ours, so an existing settings file
reproduces its previous output exactly.

`write_addon_config` clamps each knob to its own ceiling rather than a shared
one, and `tests/test_addon_config.py` pins that down — a single shared limit
would either truncate paper-white at 2 or let colour run into a range the add-on
ignores, and both failures are silent.

### Live preview, and why it is ~4 s rather than real time

The add-on reads its configuration **once, at startup** — measured by flipping
`NeuralUplift` mid-run and getting a result bit-identical to the unflipped
control. So a settings change means a new harness process; there is no message
to send to a running one.

That start-up dominates everything, and barely depends on resolution:

| resolution | frames | planes | startup | evaluations | readback | total |
| ---------- | ------ | ------ | ------- | ----------- | -------- | ----- |
| 1024×448   | 4 | 0.01 | 3.19 | 0.04 | 0.02 | 3.25 |
| 1280×560   | 8 | 0.01 | 3.65 | 0.10 | 0.02 | 3.79 |
| 3840×1678  | 8 | 0.13 | 3.46 | 0.64 | 0.10 | 4.32 |

Which kills the obvious optimisation: a reduced-resolution preview saves about
one second in four, not the order of magnitude it would need to feel live. So
there is deliberately **no separate preview resolution** — the preview is the
real thing at the real size, and the honest budget is ~4 s per adjustment.

What makes it usable is caching and debouncing instead: depth is estimated once
per image and reused, and a slider drag is collapsed by a 600 ms timer into a
single run at the value the user settled on (verified: twenty changes in 200 ms
produce one harness launch). A change arriving mid-run sets a pending flag and
starts exactly one fresh run on completion, rather than queueing per movement.

If this ever needs to be genuinely interactive, the target is that 3.5 s, and the
only real lever is not paying it: a resident harness that can be told to re-read
its settings. That needs the add-on to support reconfiguration, which it does not.

### Paths with spaces broke the whole protocol

Fixed. Worth recording because it was invisible until the release layout made it
unavoidable, and because the symptom pointed at the wrong thing entirely.

`FRAME <path> <jx> <jy> <reset>` was parsed with `>>`, which stops at the first
space. In a source checkout the scratch folder is `%LOCALAPPDATA%\DLSS5Converter`
and nothing ever noticed. A release keeps its scratch beside the executable, so
the path became `...\DLSS5 IMAGE Converter\release\engine\scratch\colour.bin`,
the harness opened `...\Claude` instead, and reported

    ERROR Could not open a contract plane for reading.

which reads as a corrupt or missing contract rather than as a quoting bug. Every
user under `C:\Program Files`, or with a space in their user name, would have hit
this on the first conversion.

`main.cpp` now peels the fixed numeric fields off the **end** of the line
(`SplitTrailingFields`) and treats everything before them as the path, so no
quoting is needed on either side. `WRITE` takes the whole remainder. The failure
message now names the path it actually tried to open.

## Phase 3 — depth quality

The reversed-Z mapping is the load-bearing assumption of the whole project and it
is currently unvalidated against the model's actual expectations. Things to try
once Phase 1 is green:

- Compare `contrast` 0.5 / 1.0 / 2.0 on the same portrait. If the neural pass is
  genuinely reading geometry, these should differ visibly around silhouettes.
- Feed a **flat** depth plane as a control. If the output is identical to real
  depth, the model is ignoring the depth buffer and the whole Depth Anything
  stage can be dropped — a large simplification worth knowing about early.
- Tiled depth on a 4K portrait: hair should gain silhouette detail. If it gains
  haloes instead, the tile alignment fit is drifting at the edges.

## Phase 4 — product

- Batch mode / watch folder. The pipeline is already headless; this is a loop.
- Presets: "Portrait" (low skin, moderate structure), "Game screenshot" (high
  everything), "Subtle".
- Per-region masking. NVIDIA's own announcement mentions masking controls, and
  the obvious use here is *keep the face, enhance everything else* — the inverse
  of what the model wants to do, and probably what a photographer wants.
- Side-by-side export for sharing comparisons.

## Deliberate non-goals

- **No downloader for the NVIDIA binaries.** The user brings their own.
- **No diffusion fallback in this app.** If a photo needs a generative pass to
  look better, that is a different tool with different failure modes, and mixing
  them makes it impossible to tell which one produced an artefact.
- **No upscaling.** DLAA mode only. Render size equals output size; if you want
  more pixels, upscale before or after, not inside the neural pass.

## Notes worth not re-deriving

- Depth Anything V2's normalised inverse depth is *already* the reversed-Z curve
  a game writes (`near / z_view`). No reprojection, no metric depth, no camera.
  This is why `to_hardware_depth` is nine lines.
- The harness must be one long-lived process. DLSS's temporal history lives in
  the NGX feature handle; one process per frame silently disables `--frames`.
- Motion vectors are exactly zero, not approximately. Anything else tells DLSS
  its history is stale and it discards the accumulation.
- Colour goes in **linear**, not sRGB. The neural pass reasons about light
  transport; gamma values make it read shadows as mid-grey and over-lift them.
- A missing add-on produces a *plausible* image, not an error. Any debugging
  session that starts "the output looks a bit flat" should check `--probe`
  before touching anything else.
