DLSS 5 Image Converter — run the neural renderer on a still image

A small desktop app that feeds a **photo** to DLSS 5's neural renderer instead of a game frame.
Drag and drop, paste (Ctrl+V), or browse. Free, open source, bring your own DLSS files.

# DLSS 5 Image Converter — run the neural renderer on a still image

A small desktop app that feeds a **photo** to DLSS 5's neural renderer instead of a game frame.
Drag and drop, paste (Ctrl+V), or browse. Free, open source, bring your own DLSS files.

**Source:** <your GitHub link>

---

## How it works

`nvngx_dlssnr.dll` isn't a standalone image model — it's an NGX snippet that the RenoDX
add-on injects into a DLSS Super Resolution evaluation. So the app doesn't "call DLSS 5".
It **fabricates a convincing DLAA frame** out of one still image and lets the add-on do its thing:

| DLSS input | Game source | Here |
|---|---|---|
| Colour | backbuffer | your image, linearised to RGBA16F |
| Depth | depth buffer | Depth Anything V2, reversed-Z |
| Motion vectors | velocity buffer | zeros — nothing moved |
| Jitter | sub-pixel projection | Halton sub-pixel resample |

The depth mapping is the load-bearing trick, and it's a lucky one. Games use reversed-Z
(near = 1.0, far = 0.0). Depth Anything V2 outputs normalised inverse depth — near = 1.0,
far = 0.0. Same curve. No reprojection needed.

It runs a hidden 64×64 swapchain and presents per evaluation, which turns out to be
enough for ReShade to attach and load the add-on in a **headless** process.

## What you need

Nothing NVIDIA ships with this and it will not help you get it.

- `nvngx_dlssnr.dll` — your own copy (RTX 40-series needs the patched build)
- `nvngx_dlss.dll` — from a Streamline `Production` folder
- `renodx-dlss5.addon64`
- `dxgi.dll` — ReShade. **Already have ReShade in a game? Just copy that game's
  `bin\x64\dxgi.dll`.** No need to touch the installer.

Drop all four in the `dlss_files` folder next to the exe. Hit **Check runtime** and you should get:

```
adapter: NVIDIA GeForce RTX 4080
dlss_available: 1
needs_driver_update: 0
neural_addon_loaded: 1
reshade_proxy_loaded: 1
dlssnr_module_loaded: 1
```

If the first three are 1 and the last is 0, DLSS works and the neural pass doesn't —
you still get a picture, just a plain DLAA resolve. That's the most confusing failure
this thing has, so check it first.

## Findings that might be useful to you

I measured these rather than guessed, and some contradict what's floating around:

- **`NRIntensity` and friends really do go to 2.0**, not 1.0. Output keeps changing all
  the way up and is identical at 3.0. (I initially measured a ceiling of 1.0 on a synthetic
  test card — don't measure saturation on synthetic input, photos behave differently.)
- **`NRStyle` is a big effect.** Cinematic lands ~50% further from the source than Natural
  at matched strengths.
- **`NRPreset` appears inert with upscaling off.** All four presets came back *bit-identical*
  on my portrait tests. The add-on echoes `preset=3` back in its log, so it receives the
  value — it just doesn't change the native/DLAA path. My guess is it selects an SR preset.
  Happy to be proven wrong if someone sees otherwise.
- **`NeuralUplift=0` is a clean off switch** — bit-identical to a plain DLAA resolve.
  Good A/B control.
- **`--frames 1` silently does nothing.** The add-on installs its NGX hooks on the first
  evaluate, so that first one can't be intercepted. Use 2 or more. Default is 8, and the
  result stops changing by about 8.
- Settings are read **once, at add-on load** — flipping the ini mid-run does nothing, so
  every settings change means a new process (~3.5 s, and it barely depends on resolution).

## Using it

Depth runs as soon as you open an image, so you can look at the **depth mask** and tune
its contrast live before spending a DLSS pass on it.

**Live preview** re-runs DLSS when a slider settles — budget ~4 s per change. That's not
render cost, it's add-on init: settings are startup-only, so each change is a fresh
process. Depth is cached across runs so you don't pay for it twice.

Sliders map straight onto the add-on's own: Intensity, Skin Structure, Local Tone,
Local Structure, plus Preset/Style and an HDR group (Paper White 0–16, HDR Transfer 0–1,
Colour Strength 0–1) for the HDR/OLED crowd.

## What it's good at

Game screenshots, 3D renders, CG stills. DLSS 5 was trained to push *rendered* images
toward photoreal, so it has the most to say about images that started out rendered.

On real photographs it does less, and what it does is more likely to read as uncanny —
the model adds cues it expects a render to be missing, and a photo already has them.
Drop **Skin** first when faces go waxy. That's the model, not the harness.

## On trust

Fair question for a random exe, so:

- **Source is public.** Build it yourself: one PowerShell script for the Python side,
  one for the C++ harness.
- **No NVIDIA binaries are bundled**, and there's no downloader for them. The build
  literally refuses to finish if any `nvngx_*.dll`, `*.addon64` or `dxgi.dll` ends up
  inside the app.
- The download is ~400 MB. On first launch it fetches **PyTorch** (from
  `download.pytorch.org`) and **Depth Anything V2** (from `huggingface.co`), with a
  progress bar. Those two hosts are the only things it talks to. Nothing else phones home.
- Python side is ~2500 lines; the only NVIDIA-facing code is one ~600-line C++ file.

MIT, except nothing here grants any rights to NVIDIA's binaries.

## Known issues

- `--frames 1` skips the neural pass entirely (see above). Leave it at 8.
- `NRPreset` does nothing that I can measure.
- First launch needs ~2.2 GB of downloads before it's usable.
- Tested on one machine: RTX 4080, driver 616.56, ReShade 6.8.0.2155,
  RenoDX DLSS5 Generic v4.1.5. Reports from other setups very welcome.
