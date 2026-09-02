"""Build and render a test sequence for DLSS 5 Image Converter's sequence mode.

Makes a small scene with real depth variation — objects at clearly different
distances, a camera that moves — and renders two matched sequences:

    renders/beauty_0001.png …   the frames to convert
    renders/depth_0001.png  …   Blender's mist pass, 16-bit

Run it either way:

    blender --background --python dlss5_test_scene.py
    blender --background --python dlss5_test_scene.py -- --frames 24 --samples 64

or open Blender, paste it into the Scripting tab, and press Run.

**In the app, tick "Depth is inverted (near is dark)".** Blender's mist pass is
0 at the camera and 1 in the distance; the converter wants near = 1.0, matching
reversed-Z and Depth Anything. That toggle is precisely what this scene is for —
if you tick it and the depth mask shows the near objects in red, the pairing is
right.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy

# --- arguments --------------------------------------------------------------
# Blender passes script arguments after a bare "--".
argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def option(name: str, default):
    if name in argv:
        return type(default)(argv[argv.index(name) + 1])
    return default


FRAMES = option("--frames", 12)
SAMPLES = option("--samples", 32)
WIDTH = option("--width", 1280)
HEIGHT = option("--height", 720)
OUTPUT = Path(option("--output", str(Path(bpy.data.filepath or __file__).parent / "renders")))


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def material(name: str, colour, roughness: float, metallic: float = 0.0):
    """A Principled BSDF. Varied roughness on purpose.

    The neural pass has the most to say about material response — sheen,
    micro-contrast, how a surface catches light — so a test scene of uniformly
    matte grey would under-report what it does.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*colour, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def build_scene() -> None:
    clear_scene()
    scene = bpy.context.scene

    # Floor.
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.data.materials.append(material("floor", (0.22, 0.22, 0.24), 0.65))

    # Objects at clearly different distances, so the depth pass has range and
    # the invert toggle is obvious to judge by eye.
    specs = [
        ("near_sphere", "sphere", (-1.6, -3.0, 1.0), 1.0, (0.75, 0.28, 0.22), 0.35, 0.0),
        ("mid_suzanne", "suzanne", (0.6, 1.0, 1.2), 1.3, (0.80, 0.72, 0.55), 0.45, 0.0),
        ("mid_torus", "torus", (-2.6, 3.5, 1.1), 1.0, (0.30, 0.55, 0.72), 0.25, 0.9),
        ("far_cube", "cube", (3.2, 9.0, 1.6), 1.6, (0.62, 0.62, 0.66), 0.55, 0.4),
        ("far_cylinder", "cylinder", (-4.5, 14.0, 2.0), 2.0, (0.45, 0.38, 0.55), 0.5, 0.0),
    ]
    for name, kind, location, size, colour, roughness, metallic in specs:
        if kind == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=size / 2, location=location)
            bpy.ops.object.shade_smooth()
        elif kind == "suzanne":
            bpy.ops.mesh.primitive_monkey_add(size=size, location=location)
            bpy.ops.object.shade_smooth()
        elif kind == "torus":
            bpy.ops.mesh.primitive_torus_add(
                major_radius=size / 2, minor_radius=size / 6, location=location
            )
            bpy.ops.object.shade_smooth()
        elif kind == "cylinder":
            bpy.ops.mesh.primitive_cylinder_add(radius=size / 3, depth=size, location=location)
        else:
            bpy.ops.mesh.primitive_cube_add(size=size, location=location)
        obj = bpy.context.active_object
        obj.name = name
        obj.data.materials.append(material(name, colour, roughness, metallic))

    # Lighting: one key with some size so shadows are soft, plus a rim.
    bpy.ops.object.light_add(type="AREA", location=(4, -6, 8))
    key = bpy.context.active_object
    key.data.energy = 2000
    key.data.size = 5
    bpy.ops.object.light_add(type="AREA", location=(-6, 8, 6))
    rim = bpy.context.active_object
    rim.data.energy = 900
    rim.data.size = 4
    rim.data.color = (0.6, 0.7, 1.0)

    # Camera, animated so the depth genuinely changes between frames. A static
    # camera would make temporal stability look better than it is.
    bpy.ops.object.camera_add(location=(0, -9, 3.2), rotation=(math.radians(78), 0, 0))
    camera = bpy.context.active_object
    scene.camera = camera
    camera.data.lens = 40

    scene.frame_start = 1
    scene.frame_end = FRAMES
    for frame in (1, FRAMES):
        scene.frame_set(frame)
        travel = (frame - 1) / max(1, FRAMES - 1)
        camera.location = (-2.5 + 5.0 * travel, -9.0 + 2.0 * travel, 3.2 + 0.6 * travel)
        camera.rotation_euler = (math.radians(78), 0, math.radians(-8 + 16 * travel))
        camera.keyframe_insert("location", frame=frame)
        camera.keyframe_insert("rotation_euler", frame=frame)
    scene.frame_set(1)


def configure_render() -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = WIDTH
    scene.render.resolution_y = HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False

    # EEVEE for speed; the renamed identifier in 4.2+ is handled either way.
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = SAMPLES
    elif hasattr(scene, "eevee"):
        try:
            scene.eevee.taa_render_samples = SAMPLES
        except AttributeError:
            pass

    # The mist pass is the depth we hand to the converter.
    layer = scene.view_layers[0]
    layer.use_pass_mist = True

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.05, 0.06, 0.08, 1)
    world.mist_settings.use_mist = True
    # Bracket the actual scene depth. Too wide and every object lands in a
    # narrow band of grey, which is a flat depth map with extra steps.
    world.mist_settings.start = 4.0
    world.mist_settings.depth = 22.0
    world.mist_settings.falloff = "LINEAR"


def configure_output() -> None:
    """Write beauty and mist as two matched, numbered sequences."""
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)

    layers = tree.nodes.new("CompositorNodeRLayers")
    layers.location = (0, 0)

    beauty = tree.nodes.new("CompositorNodeOutputFile")
    beauty.location = (400, 150)
    beauty.base_path = str(OUTPUT)
    beauty.format.file_format = "PNG"
    beauty.format.color_mode = "RGB"
    beauty.format.color_depth = "8"
    beauty.file_slots[0].path = "beauty_"

    depth = tree.nodes.new("CompositorNodeOutputFile")
    depth.location = (400, -150)
    depth.base_path = str(OUTPUT)
    depth.format.file_format = "PNG"
    depth.format.color_mode = "BW"
    # 16-bit: the converter normalises whatever it is given, and 8 bits over a
    # 20-metre scene quantises the depth into visible steps.
    depth.format.color_depth = "16"
    depth.file_slots[0].path = "depth_"

    tree.links.new(layers.outputs["Image"], beauty.inputs[0])
    tree.links.new(layers.outputs["Mist"], depth.inputs[0])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    build_scene()
    configure_render()
    configure_output()

    print(f"[dlss5] rendering {FRAMES} frames at {WIDTH}x{HEIGHT} into {OUTPUT}")
    if bpy.app.background:
        bpy.ops.render.render(animation=True)
        print("[dlss5] done.")
        print("[dlss5] In the app: Image sequence tab ->")
        print("[dlss5]   first frame  = beauty_0001.png")
        print("[dlss5]   first depth  = depth_0001.png")
        print("[dlss5]   tick 'Depth is inverted (near is dark)'")
    else:
        print("[dlss5] scene ready. Render > Render Animation (Ctrl+F12).")


main()
