# DLSS 5 Image Converter

A desktop app that runs NVIDIA's DLSS 5 neural-rendering model over a **still
image** instead of a game frame.

## The core idea

DLSS 5's neural renderer (`nvngx_dlssnr.dll`) is not a standalone image model.
It is an NGX snippet that piggybacks on a DLSS Super Resolution evaluation: the
RenoDX ReShade add-on hooks `NGX_D3D12_EVALUATE_DLSS` and injects the neural
pass into that evaluation. So to run it on a photo we do not "call DLSS 5" — we
**fabricate a convincing DLSS DLAA frame** out of a single image and let the
add-on do its thing.

A DLAA evaluation needs four things. For a still image they are:

| DLSS input     | Game source              | Our source                          |
| -------------- | ------------------------ | ----------------------------------- |
| Colour         | backbuffer               | the photo, linearised to RGBA16F    |
| Depth          | hardware depth buffer    | Depth Anything V2, reversed-Z       |
| Motion vectors | velocity buffer          | zeros (nothing moved)               |
| Jitter         | sub-pixel projection off | optional Halton sub-pixel resample  |

The depth mapping is the load-bearing trick and it is a lucky one. Games almost
universally use **reversed-Z with an infinite far plane**, where the value in the
depth buffer is `near / z_view` — near objects at 1.0, far at 0.0. Depth Anything
V2 emits **normalised inverse relative depth** — near at 1.0, far at 0.0. These
are the same curve. `contract.py` therefore hands the model DA-V2's output almost
directly and flags the range as reversed, rather than trying to reconstruct
metric depth and re-project it.

`--frames N` re-evaluates the same contract N times. DLSS is temporal; a single
evaluation gives it no history to work with and the neural pass is visibly
weaker. Repeating a static frame lets the accumulator settle.

## Layout

- `dlss5_converter/` — Python: GUI, depth estimation, contract construction.
- `native/dlss5_eval/` — C++: the D3D12 + NGX harness. The only part that
  touches NVIDIA APIs. Built separately, invoked as a subprocess.
- `scripts/` — PowerShell setup/run/build, mirroring Depth Animator.

Python never links against NGX. The native harness is a plain CLI that reads
raw binary planes and writes one back, so it can be run and debugged by hand
without the GUI, and a crash in DLSS cannot take the app down with it.

## Conventions (inherited from Depth Animator)

- Python 3.12, PySide6, `from __future__ import annotations` everywhere.
- Torch/transformers are imported **lazily inside functions**, never at module
  scope — importing torch at startup adds seconds to launch.
- Model weights live in the per-user data dir via `paths.py`, never beside the
  executable, so a portable rebuild does not wipe a 1 GB download.
- Comments explain **why**, not what. If a line looks odd, the comment says what
  breaks without it.
- Defensive `except Exception:  # noqa: BLE001` around anything optional —
  a broken driver or a missing model must degrade, never block startup.

## Legal note

`nvngx_dlssnr.dll` is a leaked pre-release NVIDIA binary. This repo does not
ship it, reference it by hash, or help acquire it — the user points the app at
their own copy. Do not add a downloader.
