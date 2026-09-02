# Blender test scene

A script rather than a `.blend`, because it is readable, works on any Blender
version, and you can see exactly what it sets up.

## Run it

Headless, which renders straight away:

```powershell
blender --background --python dlss5_test_scene.py
```

Or open Blender, paste the script into the **Scripting** tab, press **Run**, then
**Render ▸ Render Animation** (Ctrl+F12).

Options, after a bare `--`:

```powershell
blender --background --python dlss5_test_scene.py -- --frames 24 --samples 64 --width 1920 --height 1080
```

| option | default | |
| ------ | ------- | - |
| `--frames` | 12 | frames to render |
| `--samples` | 32 | render samples |
| `--width` / `--height` | 1280 / 720 | resolution |
| `--output` | `./renders` | where the two sequences go |

## What it makes

```
renders/beauty_0001.png …   the frames to convert
renders/depth_0001.png  …   Blender's mist pass, 16-bit greyscale
```

Five objects at clearly different distances, varied roughness and metallic so
the neural pass has material response to work with, soft key and a cool rim
light, and a camera that moves across the sequence — a static camera would make
temporal stability look better than it really is.

## Then, in the app

**Image sequence** tab:

1. **Choose first frame…** → `beauty_0001.png`
2. **Choose first depth frame…** → `depth_0001.png`
3. **Tick "Depth is inverted (near is dark)"**
4. Convert

That toggle is the thing worth checking. Blender's mist pass is 0 at the camera
and 1 in the distance; the converter wants near = 1.0, matching reversed-Z and
Depth Anything. So it needs inverting — and with it ticked, the **Depth mask**
view should show the near sphere in **red** and the far cylinder in **blue**. If
it is the other way round, untick it.

Worth trying both ways once. Getting it backwards does not error — it produces a
plausible image with the depth cues inverted, which is exactly the kind of
failure that is easier to see than to describe.

## Why the mist settings matter

`start = 4.0`, `depth = 22.0` bracket the actual extent of the scene. Left at
Blender's defaults, everything lands in a narrow band of grey — a flat depth map
with extra steps. If you build your own scene, set these to roughly the nearest
and furthest thing the camera can see.
