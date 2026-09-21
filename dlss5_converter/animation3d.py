"""Keyframe animation for the 3D camera.

Ported from Depth Animator (animation.py); uses camera3d.

Replaces the old `Keyframe(time, x, y, zoom, depth)`, which described a 2D shift
and could not express a camera at all. Keys now hold a real rig pose, and each
key carries its own outgoing easing so timing can be shaped per segment the way
it is in Blender or Premiere.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, field, replace
from enum import Enum

from .camera3d import Camera, OrbitRig  # noqa: F401 - OrbitRig kept for presets


class Easing(str, Enum):
    LINEAR = "Linear"
    EASE_IN = "Ease In"
    EASE_OUT = "Ease Out"
    EASE_IN_OUT = "Ease In Out"
    STEP = "Hold"

    @property
    def label(self) -> str:
        return self.value


def apply_easing(easing: Easing, t: float) -> float:
    """Reshape normalised progress within one segment."""
    t = max(0.0, min(1.0, float(t)))
    if easing is Easing.STEP:
        return 0.0
    if easing is Easing.EASE_IN:
        return t * t
    if easing is Easing.EASE_OUT:
        return 1.0 - (1.0 - t) * (1.0 - t)
    if easing is Easing.EASE_IN_OUT:
        return t * t * (3.0 - 2.0 * t)
    return t


@dataclass(slots=True)
class CameraKey:
    """A free camera pose pinned to a point in time, in seconds.

    Position and orientation are independent, as in any 3D application. The
    earlier orbit-rig form (yaw/pitch/dolly around a fixed pivot) could not
    represent a camera dragged to an arbitrary point, which made a transform
    gizmo impossible to build honestly.
    """

    time: float
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0  # radians, + turns right
    pitch: float = 0.0  # radians, + looks up
    roll: float = 0.0
    fov_degrees: float = 55.0
    easing: Easing = Easing.EASE_IN_OUT

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def lerp(self, other: "CameraKey", f: float) -> "CameraKey":
        def mix(a: float, b: float) -> float:
            return a + (b - a) * f

        def mix_angle(a: float, b: float) -> float:
            # Take the short way round, so a move never spins the long way.
            delta = (b - a + math.pi) % math.tau - math.pi
            return a + delta * f

        return CameraKey(
            time=mix(self.time, other.time),
            x=mix(self.x, other.x),
            y=mix(self.y, other.y),
            z=mix(self.z, other.z),
            yaw=mix_angle(self.yaw, other.yaw),
            pitch=mix_angle(self.pitch, other.pitch),
            roll=mix_angle(self.roll, other.roll),
            fov_degrees=mix(self.fov_degrees, other.fov_degrees),
            easing=self.easing,
        )


def _catmull_rom(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    """Position along a Catmull-Rom segment between p1 and p2.

    Passes exactly through every key, so the user's keyframes stay
    authoritative, while the tangent at each key is inferred from its
    neighbours — which is what turns a chain of straight segments into a flowing
    path.
    """
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


@dataclass(slots=True)
class CameraTrack:
    """An ordered set of camera keys, evaluated at any time in seconds."""

    keys: list[CameraKey] = field(default_factory=list)
    #: Smooth curves position through the keys; linear keeps straight segments.
    smooth: bool = True

    def sorted_keys(self) -> list[CameraKey]:
        return sorted(self.keys, key=lambda k: k.time)

    def add(self, key: CameraKey, tolerance: float = 0.02) -> CameraKey:
        """Insert a key, replacing any existing key at the same instant."""
        for existing in self.keys:
            if abs(existing.time - key.time) <= tolerance:
                self.keys.remove(existing)
                break
        self.keys.append(key)
        self.keys.sort(key=lambda k: k.time)
        return key

    def remove_at(self, time: float, tolerance: float = 0.05) -> bool:
        nearest = self.nearest(time, tolerance)
        if nearest is None:
            return False
        self.keys.remove(nearest)
        return True

    def nearest(self, time: float, tolerance: float = 0.05) -> CameraKey | None:
        if not self.keys:
            return None
        best = min(self.keys, key=lambda k: abs(k.time - time))
        return best if abs(best.time - time) <= tolerance else None

    def evaluate(self, time: float) -> CameraKey | None:
        """Pose at `time`; None when the track is empty."""
        keys = self.sorted_keys()
        if not keys:
            return None
        if len(keys) == 1 or time <= keys[0].time:
            return replace(keys[0], time=time)
        if time >= keys[-1].time:
            return replace(keys[-1], time=time)

        times = [k.time for k in keys]
        index = max(1, bisect_left(times, time))
        left, right = keys[index - 1], keys[index]
        span = max(right.time - left.time, 1e-6)
        f = apply_easing(left.easing, (time - left.time) / span)
        result = replace(left.lerp(right, f), time=time)

        if self.smooth and len(keys) > 2 and left.easing is not Easing.STEP:
            # Curve only the position; rotation and lens stay eased-linear so a
            # smooth path never introduces unasked-for swinging or lens pumping.
            before = keys[max(index - 2, 0)]
            after = keys[min(index + 1, len(keys) - 1)]
            result.x = _catmull_rom(before.x, left.x, right.x, after.x, f)
            result.y = _catmull_rom(before.y, left.y, right.y, after.y, f)
            result.z = _catmull_rom(before.z, left.z, right.z, after.z, f)
        return result

    @property
    def duration(self) -> float:
        keys = self.sorted_keys()
        return keys[-1].time if keys else 0.0


def key_to_camera(key: CameraKey, _pivot_z: float = 0.0) -> Camera:
    """Build a renderer camera from a free pose.

    Yaw and pitch define a look direction; -Z is forward at zero rotation, which
    matches `geometry.unproject`.
    """
    cos_pitch = math.cos(key.pitch)
    forward = (
        math.sin(key.yaw) * cos_pitch,
        math.sin(key.pitch),
        -math.cos(key.yaw) * cos_pitch,
    )
    target = (key.x + forward[0], key.y + forward[1], key.z + forward[2])
    up = (math.sin(key.roll), math.cos(key.roll), 0.0)
    return Camera(
        position=key.position,
        target=target,
        up=up,
        fov_degrees=key.fov_degrees,
    )


# Kept so existing call sites keep working while the rig form is retired.
def key_to_rig(key: CameraKey, pivot_z: float = 0.0) -> Camera:
    return key_to_camera(key, pivot_z)


def track_from_preset(
    name: str, duration: float, pivot_z: float, strength: float = 1.0, samples: int = 5
) -> CameraTrack:
    """Bake a preset into editable keys.

    Presets are a starting point, not a fixed path — turning them into real keys
    means anything the presets can do can then be adjusted by hand.
    """
    from .camera3d import preset_rig

    track = CameraTrack()
    if name == "Static":
        camera = preset_rig(name, 0.0, pivot_z, strength=strength).to_camera()
        track.add(
            CameraKey(
                time=0.0,
                x=float(camera.position[0]),
                y=float(camera.position[1]),
                z=float(camera.position[2]),
                fov_degrees=camera.fov_degrees,
            ),
            tolerance=0.0,
        )
        return track
    for i in range(max(2, samples)):
        t = i / (max(2, samples) - 1)
        camera = preset_rig(name, t, pivot_z, strength=strength).to_camera()
        eye = camera.position
        target = camera.target
        dx = target[0] - eye[0]
        dy = target[1] - eye[1]
        dz = target[2] - eye[2]
        horizontal = math.hypot(dx, dz)
        track.add(
            CameraKey(
                time=t * duration,
                x=float(eye[0]),
                y=float(eye[1]),
                z=float(eye[2]),
                # Bake the rig's look direction into free yaw/pitch.
                yaw=math.atan2(dx, -dz),
                pitch=math.atan2(dy, horizontal if horizontal > 1e-6 else 1e-6),
                fov_degrees=camera.fov_degrees,
                easing=Easing.EASE_IN_OUT if i else Easing.EASE_OUT,
            ),
            tolerance=0.0,
        )
    return track


# -- per-key easing, After Effects style ------------------------------------
#
# The data model stores one easing per SEGMENT (on the key that starts it):
# whether that segment starts slow and whether it ends slow. Animators think
# per KEY instead: Ease In slows the arrival into this key, Ease Out slows the
# departure from it. These helpers translate, so the dropdown and the timeline
# glyphs both speak per key while evaluation stays unchanged.

KEY_EASES = ("Linear", "Ease In", "Ease Out", "Easy Ease", "Hold")

_SEGMENT = {
    Easing.LINEAR: (False, False),
    Easing.EASE_IN: (True, False),     # t^2: the segment starts slow
    Easing.EASE_OUT: (False, True),    # the segment ends slow
    Easing.EASE_IN_OUT: (True, True),
}


def _segment(slow_start: bool, slow_end: bool) -> Easing:
    for easing, flags in _SEGMENT.items():
        if flags == (slow_start, slow_end):
            return easing
    return Easing.LINEAR


def _neighbours(track: CameraTrack, key: CameraKey):
    keys = track.sorted_keys()
    index = next((i for i, k in enumerate(keys) if k is key), None)
    prev = keys[index - 1] if index else None
    return prev


def key_ease_label(track: CameraTrack, key: CameraKey) -> str:
    if key.easing is Easing.STEP:
        return "Hold"
    prev = _neighbours(track, key)
    slow_in = prev is not None and prev.easing is not Easing.STEP and _SEGMENT[prev.easing][1]
    slow_out = _SEGMENT[key.easing][0]
    return {(False, False): "Linear", (True, False): "Ease In",
            (False, True): "Ease Out", (True, True): "Easy Ease"}[(slow_in, slow_out)]


def set_key_ease(track: CameraTrack, key: CameraKey, label: str) -> None:
    """Apply a per-key easing by editing the segments either side of it."""
    if label == "Hold":
        key.easing = Easing.STEP
        return
    slow_in = label in ("Ease In", "Easy Ease")
    slow_out = label in ("Ease Out", "Easy Ease")
    prev = _neighbours(track, key)
    if prev is not None and prev.easing is not Easing.STEP:
        prev.easing = _segment(_SEGMENT[prev.easing][0], slow_in)
    end = False if key.easing is Easing.STEP else _SEGMENT[key.easing][1]
    key.easing = _segment(slow_out, end)
