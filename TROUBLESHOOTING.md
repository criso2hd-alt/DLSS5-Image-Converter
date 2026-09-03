# Troubleshooting

Start here, always:

```powershell
DLSS5Converter.exe --selftest 2> report.txt
```

That runs a real conversion end to end — imports PyTorch, checks CUDA, estimates
depth, evaluates DLSS, and prints what the RenoDX add-on said, including its
version. Paste `report.txt` into any bug report and most of the questions below
answer themselves.

---

## The result looks identical to the input

By far the most common report, and it has two completely different causes that
produce the same symptom.

**First, update your RenoDX add-on.** An out-of-date `renodx-dlss5.addon64` was
the cause on an RTX 5070 — every indicator green, no error, image unchanged.
Updating it fixed it. `--selftest` prints the version it loaded.

**Second, it may have worked and you cannot see it.** The defaults sit at 1.0
of a possible 2.00 — visible on most content, but on a render that is already
photographic even a real change can be hard to spot side by side.

Click **Difference**. It shows what the neural pass changed, amplified, with
numbers:

```
Difference — mean 0.0079, peak 0.0488, amplified 21x     it worked, subtly
Difference — mean 0.0000, peak 0.0000  (nothing changed)  it did not
```

If it worked but you want to *see* it, push Intensity, Skin, Local tone and
Structure to 2.00.

---

## `NVSDK_NGX_D3D12_Init failed: FeatureNotSupported`

NGX refused the GPU. Two causes, and the error message names which one by
reporting the adapter it actually ran on.

**Not an NVIDIA GPU.** The app is on your machine's default graphics adapter,
which on a laptop is usually the integrated one. CUDA finds the NVIDIA card by
itself, which is why depth estimation works and DLSS then fails. Force it:

> Settings ▸ System ▸ Display ▸ Graphics ▸ Add desktop app.
> Add **both** `DLSS5Converter.exe` and `engine\dlss5_eval.exe`, set each to
> **High performance**, restart the app.

`engine\dlss5_eval.exe` is the one that matters — it is a separate process and
it creates the Direct3D device. Setting only the main executable will not help.

If Windows will not cooperate, NVIDIA Control Panel ▸ Manage 3D settings ▸
Program Settings ▸ add `dlss5_eval.exe` ▸ High-performance NVIDIA processor.
Some laptops also have a MUX or "discrete only" mode in Armoury Crate, Lenovo
Vantage or Omen Gaming Hub.

**NVIDIA but not RTX.** GTX 10 and 16-series have CUDA but no tensor cores.
Depth estimation runs, DLSS cannot, and no driver update changes that.

---

## `dlssnr_module_loaded: 0` with everything else `1`

The file is present and found — the add-on **refused** it. Its own message:

> NR is unavailable in this session. Update your NVIDIA driver, or replace
> nvngx_dlssnr.dll with the reference build (sha256 in the addon README) and
> restart.

**Easiest fix: click "Find my DLSS files…"** in the app. It searches your Steam
libraries, Downloads and Documents and copies a matched set in — including the
newest add-on it can find anywhere, which is usually the fix for this exact
symptom.

**By hand: if DLSS 5 already works in a game for you, copy all four files out of
that game's folder** into `dlss_files\`. They sit beside the game
executable, usually in `bin\x64`. A set already running on your card is a set
your GPU, your driver and the add-on all accept — no version guessing.

Keep the four together. Mixing a runtime from one source with an add-on from
another is a common way to produce exactly this.

Otherwise: check your driver version, and compare your DLL's hash
(`Get-FileHash nvngx_dlssnr.dll`) against the reference in the add-on's README.
`--selftest` prints the hash the add-on computed for the copy you gave it.

---

## `nvngx_dlss.dll was not found` (but it is right there)

Put the files in `dlss_files\` in whatever shape they arrived — all of these
work:

- `streamline.zip` **left zipped** — the app opens it and takes what it needs
- unzipped to `streamline\Production\`, `NVStreamline\Production\`, or with an
  extra wrapper folder
- everything loose in `dlss_files\`

The search is recursive. A file placed loose wins over one buried in an unpacked
archive, and nothing you put there is ever overwritten.

---

## The sidebar is cut off, sliders missing

Fixed in v0.1.5. The sidebar needs 1000 px of height; a 1080p screen at 150%
scaling gives a 720 px window, so the bottom third was being clipped — taking
the neural sliders and half the HDR group with it. Update, and it scrolls.

---

## `'NoneType' object has no attribute 'write'` on first launch

Fixed in v0.1.1. A windowed build has no console, so `sys.stdout` and
`sys.stderr` do not exist, and the Hugging Face downloader wrote its progress
bar to a stream that was not there. Update.

---

## My renders come out at 3840 px / how do I choose the export resolution?

Two different settings, and conflating them is the most common confusion there
is.

**Max size**, under Evaluation, is the longest edge sent to DLSS — the
resolution the neural pass actually runs at. Anything larger is downscaled
first. This is where detail comes from. Raise it: 8K is verified here at 25 s
and 5.3 GB of VRAM, barely more than 4K, because the cost is dominated by fixed
NGX allocations rather than by the image. The field is editable if you want an
exact number.

**Save result…** now asks for an export size before writing the file. Native by
default; presets for 1.5x/2x/3x/4x and for a fixed long edge (1920 … 7680); or
type a width and the height follows. Proportions are kept unless you untick.

Which one you want:

| you want | change |
| -------- | ------ |
| more actual detail | **Max size**, then convert again |
| a file at a delivery spec | **Save result…** |

Enlarging on export is plain resampling — Lanczos, computed in linear light. It
cannot invent detail the neural pass did not produce. It is deliberately not a
second AI upscaler: stacking one on the neural pass compounds the waxiness
people already hit on a second pass.

---

## Sequence mode: the depth looks wrong

Renderers disagree about which way up a depth pass goes, and it cannot be
inferred — so the app asks.

Blender's **mist pass** is 0 at the camera and 1 in the distance, so it needs
**"Depth is inverted (near is dark)"** ticked. Check the **Depth mask** view:
near objects should read **red**, far ones **blue**. If it is reversed, flip the
toggle.

Getting it backwards does not error. It produces a plausible image with the
depth cues inverted, which is easier to see than to describe — so look once.

There is a ready-made test scene in [`blender/`](blender/) that renders matched
beauty and depth sequences.

---

## Sequence mode: frames must all be the same size

One harness means one set of NGX buffers, fixed when the run starts. Mixed sizes
are refused rather than silently resized.

For unrelated images of different sizes, use **Apply to folder…** on the Single
image page instead — that restarts the harness when the size changes.

---

## The MP4 will not play / looks blocky

It is encoded with **mp4v**, not H.264 — OpenCV ships no H.264 encoder for
licensing reasons. The PNG frames are always written alongside it, so re-encode
them with ffmpeg, Resolve, or anything else:

```powershell
ffmpeg -framerate 24 -i beauty_%04d_dlss5.png -c:v libx264 -crf 16 out.mp4
```

---

## First launch wants 2.2 GB

PyTorch (~1.8 GB, from `download.pytorch.org`) and a Depth Anything V2 model
(~400 MB, from `huggingface.co`). Those two hosts are the only things the app
talks to.

Nothing that can be downloaded is bundled, which is why the app itself is ~400 MB
instead of 3 GB. Both land in folders beside the executable and survive updates —
unpack a new version over the old folder and you will not pay for them twice.

An interrupted PyTorch download resumes rather than restarting.

---

## Faces look waxy or plastic

That is the model, not the harness. DLSS 5 was trained to push *rendered* images
towards photoreal, so it adds cues it expects a render to be missing — and a
photograph already has them.

Lower **Skin** first. If you ran a second pass, lower everything: it compounds,
roughly as much again on the second run as the first.

---

## What driver version do I need?

There is no certified answer, because DLSS 5 is not released. `nvngx_dlssnr.dll`
is a pre-release binary and NVIDIA has published no minimum for it.

**Confirmed case: 572.83 fails.** An RTX 4070 Ti SUPER, all four files correct,
every indicator in Check runtime green including `test_evaluation: ok` — and
every conversion died with the harness crashing. Updating the driver, and
changing nothing else, fixed it. 616.56 is verified working here.

What is known beyond that:

- **Run the latest Game Ready driver.** Every confirmed-working report is on a
  recent one; the one confirmed failure was on a driver about eighteen months
  old. There is not enough data to name a cutoff, and guessing one would give
  false warnings to people whose driver is fine.
- **Ignore `min_driver_version` in the runtime check.** That number comes from
  NGX and is the minimum for *Super Resolution* — it reads 470.0, which DLSS 5
  will not honour. It is printed because NGX offers it, not because it answers
  this question.
- **`driver_version` is yours**, decoded the way NVIDIA Control Panel spells it,
  so you can compare it against nvidia.com directly.

An old driver does not fail cleanly. It can pass every indicator in the runtime
check — including `test_evaluation: ok`, which is a 64x64 evaluation — and then
take the harness down on a real image.

---

## `The harness stopped at 3840x2160 (exit code 0x…)`

The harness crashed rather than reporting a failure, so all there is to go on is
the exit code, which the message now names. In order of likelihood:

1. **Update your NVIDIA driver.** This is the confirmed cause of the only report
   of this so far — green runtime check, crash on every real conversion, fixed
   by a driver update alone. Try it first.
2. **Lower Max size** to 1920 and convert again. If 1920 works and 3840 does
   not, it is a size limit — VRAM, or the driver refusing a buffer that large.
3. **Switch the depth model to Base.** The Large model stays resident on the GPU
   while the harness allocates its own full-size buffers; on a 12 GB card at 4K
   they compete.
4. **Update `renodx-dlss5.addon64`**, or click **Find my DLSS files…** which
   takes the newest one on your disk.

`0x887A0005` is the graphics device being removed or reset, and `0xC0000005` is
a crash inside the runtime; both are usually one of the four above rather than
anything specific to that code.

---

## HDR: highlights come back dimmer than they should

**Update `renodx-dlss5.addon64`.** The add-on version changes this measurably.
Measured here on one machine, same code, same input, same GPU — an HDR plate
with a highlight at 8.0x diffuse white:

```
newer add-on (1,732,608 bytes)   peak 7.77x   highlight preserved
older add-on (1,703,424 bytes)   peak 4.59x   highlight crushed to 57%
```

Ordinary SDR photos measured identical between the two, so this is specifically
an HDR difference — and it is invisible unless you go looking, because both
results are plausible pictures.

`--selftest` prints the peak its own HDR round trip came back with, and the
add-on's version string, so you can see which build you are on.

**Click "Find my DLSS files…"** to pick up the newest add-on on your disk; it
always prefers the freshest one it can find, wherever it lives.

---

## HDR: my .jxr will not open / the result looks flat

`.jxr` is decoded by Windows' own imaging codecs. `--selftest` proves they work
on your machine by writing an HDR file and reading it back:

```
jpeg xr          : ok (peak 8.00, expected 8.00)
```

If that line says `unavailable` or `FAILED`, the codec is missing or blocked —
unusual, and not something this app installs.

**A result that looks flat or washed out is the tone mapping, not the pipeline.**
Your monitor gets 0..1, so the preview has to compress a highlight at 12x diffuse
white into it. That is a display step only: the exported `.jxr` or `.exr` still
has the full range, and looks right in something that can show HDR.

**A `.jxr` that is not actually HDR is treated as SDR.** If nothing in the file
exceeds diffuse white, tone mapping it would only cost quality, so it is skipped.
The status bar after a conversion says which happened.

**PNG from an HDR result is tone mapped, not clipped.** Clipping is what turns a
bright sky into a flat white shape. If you want the range, save `.jxr` or `.exr`
— those are offered first when the result is HDR.

---

## Still stuck

Open an issue with `report.txt` from `--selftest` attached, plus your GPU, driver
version and RenoDX add-on version. The self test prints all three.
