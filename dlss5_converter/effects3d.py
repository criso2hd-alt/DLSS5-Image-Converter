"""Lighting, volumetric FX, particles, and their animation tracks.

Ported from Depth Animator (effects.py).

The effect model is deliberately renderer-agnostic.  A project can therefore
preview and export the exact same snapshot, while the editor is free to expose
the values as ordinary transform widgets and keyframes.
"""

from __future__ import annotations

import copy
import math
import uuid
from bisect import bisect_left
from dataclasses import dataclass, field, fields


def _id() -> str:
    return uuid.uuid4().hex[:10]


@dataclass(slots=True)
class LightingSettings:
    hdri_path: str = ""
    dome_enabled: bool = False
    enabled: bool = True
    strength: float = 1.0
    exposure: float = 0.0
    rotation: float = 0.0
    relight: float = 0.0
    ambient_colour: tuple[float, float, float] = (0.55, 0.60, 0.72)
    key_colour: tuple[float, float, float] = (1.0, 0.94, 0.84)
    key_direction: tuple[float, float, float] = (-0.4, 0.6, 0.7)


@dataclass(slots=True)
class VolumeEffect:
    name: str = "Fog volume"
    kind: str = "fog"  # fog, smoke, fire, cloud, godrays
    id: str = field(default_factory=_id)
    enabled: bool = True
    position: tuple[float, float, float] = (0.0, 0.0, -4.0)
    size: tuple[float, float, float] = (3.0, 2.0, 2.0)
    #: Euler XYZ in radians. Without this a volume could only ever axis-align,
    #: so it could not be tilted to follow a scene that is not square to camera.
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    density: float = 0.35
    noise_scale: float = 1.8
    detail: float = 0.55
    speed: float = 0.20
    colour: tuple[float, float, float] = (0.68, 0.76, 0.86)
    emission: float = 0.0
    light_response: float = 0.8
    seed: float = 1.0


@dataclass(slots=True)
class ParticleEmitter:
    name: str = "Smoke emitter"
    kind: str = "smoke"  # smoke, fire, embers, dust, snow, clouds
    id: str = field(default_factory=_id)
    enabled: bool = True
    position: tuple[float, float, float] = (0.0, -1.0, -4.0)
    size: tuple[float, float, float] = (1.2, 1.2, 1.2)
    #: Euler XYZ in radians, matching VolumeEffect.
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    count: int = 384
    rate: float = 1.0
    lifetime: float = 3.0
    particle_size: float = 0.10
    speed: float = 0.65
    spread: float = 0.65
    gravity: float = -0.05
    turbulence: float = 0.55
    colour: tuple[float, float, float] = (0.72, 0.76, 0.80)
    emission: float = 0.0
    light_response: float = 0.8
    seed: float = 2.0


@dataclass(slots=True)
class EffectsState:
    lighting: LightingSettings = field(default_factory=LightingSettings)
    volumes: list[VolumeEffect] = field(default_factory=list)
    emitters: list[ParticleEmitter] = field(default_factory=list)

    def clone(self) -> "EffectsState":
        return copy.deepcopy(self)

    def item(self, item_id: str):
        return next(
            (item for item in [*self.volumes, *self.emitters] if item.id == item_id),
            None,
        )


@dataclass(slots=True)
class EffectsKey:
    time: float
    state: EffectsState
    #: Property paths this key pins. Empty means a whole-state snapshot, which
    #: is what the "Key FX snapshot" button writes and what every project saved
    #: before per-property keying existed contains.
    paths: set[str] = field(default_factory=set)

    def covers(self, path: str) -> bool:
        return not self.paths or path in self.paths


def _mix(a: float, b: float, f: float) -> float:
    return float(a) + (float(b) - float(a)) * f


def _mix_tuple(a, b, f: float):
    return tuple(_mix(x, y, f) for x, y in zip(a, b))


def _mix_angles(a, b, f: float):
    """Interpolate Euler angles the short way, so a turn never unwinds."""
    out = []
    for x, y in zip(a, b):
        delta = (float(y) - float(x) + math.pi) % math.tau - math.pi
        out.append(float(x) + delta * f)
    return tuple(out)


#: Identity, not animatable state. Everything else on an effect is keyframable.
_STATIC_FIELDS = frozenset({"id", "kind", "name", "hdri_path"})
#: Interpolated as angles so a turn takes the short way round.
_ANGLE_FIELDS = frozenset({"rotation"})


def blend_field(name: str, left, right, f: float):
    """Interpolate one field according to its type.

    Driven by the dataclass rather than a hand-written list, so a property
    added to an effect is animatable immediately instead of silently ignored —
    which is exactly how `rotation` was missed when it was introduced.
    """
    if name in _STATIC_FIELDS or isinstance(left, str):
        return copy.deepcopy(left if f < 0.5 else right)
    if isinstance(left, bool):
        # Booleans cannot be half-on; they switch at the midpoint.
        return left if f < 0.5 else right
    if isinstance(left, (tuple, list)):
        if name in _ANGLE_FIELDS:
            return _mix_angles(left, right, f)
        return _mix_tuple(left, right, f)
    if isinstance(left, int):
        return int(round(_mix(left, right, f)))
    if isinstance(left, float):
        return _mix(left, right, f)
    return copy.deepcopy(left if f < 0.5 else right)


def _interpolate_item(a, b, f: float):
    result = copy.deepcopy(a)
    for field_info in fields(a):
        name = field_info.name
        if not hasattr(b, name):
            continue
        setattr(result, name, blend_field(name, getattr(a, name), getattr(b, name), f))
    return result


def interpolate_effects(a: EffectsState, b: EffectsState, f: float) -> EffectsState:
    """Interpolate matching effect IDs; additions/removals use the nearer key."""
    f = max(0.0, min(1.0, float(f)))
    result = a.clone()
    la, lb = a.lighting, b.lighting
    for field_info in fields(la):
        name = field_info.name
        if name == "rotation":
            # Degrees around the equirectangular dome; take the short way.
            delta = (lb.rotation - la.rotation + 180.0) % 360.0 - 180.0
            result.lighting.rotation = la.rotation + delta * f
            continue
        setattr(
            result.lighting,
            name,
            blend_field(name, getattr(la, name), getattr(lb, name), f),
        )

    def blend_lists(left, right):
        right_by_id = {item.id: item for item in right}
        output = [
            _interpolate_item(item, right_by_id[item.id], f)
            if item.id in right_by_id else copy.deepcopy(item)
            for item in left
        ]
        if f >= 0.5:
            known = {item.id for item in output}
            output.extend(copy.deepcopy(item) for item in right if item.id not in known)
        return output

    result.volumes = blend_lists(a.volumes, b.volumes)
    result.emitters = blend_lists(a.emitters, b.emitters)
    return result


#: Owner segment naming the lighting block in a property path.
ENVIRONMENT = "environment"


def _owner(state: EffectsState, owner_id: str):
    return state.lighting if owner_id == ENVIRONMENT else state.item(owner_id)


def animatable_paths(state: EffectsState) -> list[str]:
    """Every scalar the effects expose, as `owner.field[.component]` paths.

    Enumerated from `dataclasses.fields` for the same reason `blend_field` is:
    a property added to an effect becomes keyframable without a second edit,
    which is exactly how `rotation` was missed when it was introduced.
    """
    paths: list[str] = []
    owners = [(ENVIRONMENT, state.lighting)]
    owners += [(item.id, item) for item in [*state.volumes, *state.emitters]]
    for owner_id, target in owners:
        for info in fields(target):
            name = info.name
            value = getattr(target, name)
            if name in _STATIC_FIELDS or isinstance(value, str):
                continue
            if isinstance(value, (tuple, list)):
                paths.extend(f"{owner_id}.{name}.{index}" for index in range(len(value)))
            else:
                paths.append(f"{owner_id}.{name}")
    return paths


def get_path(state: EffectsState, path: str) -> float | None:
    """Read one scalar. Returns None when the path names nothing on `state`."""
    owner_id, _, rest = path.partition(".")
    target = _owner(state, owner_id)
    if target is None or not rest:
        return None
    name, _, component = rest.partition(".")
    if not hasattr(target, name):
        return None
    value = getattr(target, name)
    if component:
        sequence = list(value)
        index = int(component)
        return float(sequence[index]) if index < len(sequence) else None
    return float(value)


def set_path(state: EffectsState, path: str, value: float) -> bool:
    """Write one scalar, coercing to the field's own type."""
    owner_id, _, rest = path.partition(".")
    target = _owner(state, owner_id)
    if target is None or not rest:
        return False
    name, _, component = rest.partition(".")
    if not hasattr(target, name):
        return False
    current = getattr(target, name)
    if component:
        sequence = list(current)
        index = int(component)
        if index >= len(sequence):
            return False
        sequence[index] = float(value)
        setattr(target, name, tuple(sequence))
    elif isinstance(current, bool):
        # Matches blend_field: booleans switch at the midpoint, never half-on.
        setattr(target, name, bool(value >= 0.5))
    elif isinstance(current, int):
        setattr(target, name, int(round(value)))
    else:
        setattr(target, name, float(value))
    return True


def _blend_path(path: str, a: float, b: float, f: float) -> float:
    owner_id, _, rest = path.partition(".")
    name = rest.partition(".")[0]
    if name in _ANGLE_FIELDS:
        # Lighting keeps its dome rotation in degrees; effects use radians.
        period = 360.0 if owner_id == ENVIRONMENT else math.tau
        delta = (b - a + period / 2.0) % period - period / 2.0
        return a + delta * f
    return _mix(a, b, f)


def _sample_path(samples: list[tuple[float, float]], time: float, path: str) -> float:
    """Interpolate one property across only the keys that pin it."""
    if time <= samples[0][0]:
        return samples[0][1]
    if time >= samples[-1][0]:
        return samples[-1][1]
    times = [sample[0] for sample in samples]
    index = max(1, bisect_left(times, time))
    (left_time, left_value), (right_time, right_value) = samples[index - 1], samples[index]
    f = (time - left_time) / max(right_time - left_time, 1e-6)
    f = f * f * (3.0 - 2.0 * f)
    return _blend_path(path, left_value, right_value, f)


@dataclass(slots=True)
class EffectsTrack:
    keys: list[EffectsKey] = field(default_factory=list)

    def add(self, time: float, state: EffectsState, tolerance: float = 0.02) -> EffectsKey:
        existing = self.nearest(time, tolerance)
        if existing is not None:
            self.keys.remove(existing)
        key = EffectsKey(float(time), state.clone())
        self.keys.append(key)
        self.keys.sort(key=lambda item: item.time)
        return key

    def nearest(self, time: float, tolerance: float = 0.05) -> EffectsKey | None:
        if not self.keys:
            return None
        key = min(self.keys, key=lambda item: abs(item.time - time))
        return key if abs(key.time - time) <= tolerance else None

    def remove_at(self, time: float, tolerance: float = 0.05) -> bool:
        key = self.nearest(time, tolerance)
        if key is None:
            return False
        self.keys.remove(key)
        return True

    # -- per-property keying --------------------------------------------------

    def key_property(
        self, time: float, path: str, state: EffectsState, tolerance: float = 0.02
    ) -> EffectsKey:
        """Pin one property at `time`, leaving every other property free."""
        key = self.nearest(time, tolerance)
        if key is None:
            key = EffectsKey(float(time), state.clone(), {path})
            self.keys.append(key)
            self.keys.sort(key=lambda item: item.time)
            return key
        value = get_path(state, path)
        if value is not None:
            set_path(key.state, path, value)
        # An existing snapshot key already covers every path, so leave it alone.
        if key.paths:
            key.paths.add(path)
        return key

    def unkey_property(self, time: float, path: str, tolerance: float = 0.02) -> bool:
        key = self.nearest(time, tolerance)
        if key is None or not key.covers(path):
            return False
        if not key.paths:
            # Expand the snapshot into explicit paths so a single property can
            # be lifted out of it without discarding the rest of the key.
            key.paths = set(animatable_paths(key.state))
        key.paths.discard(path)
        if not key.paths:
            self.keys.remove(key)
        return True

    def is_keyed(self, time: float, path: str, tolerance: float = 0.02) -> bool:
        key = self.nearest(time, tolerance)
        return key is not None and key.covers(path)

    def is_animated(self, path: str) -> bool:
        """True when any key pins this property, keyed at the playhead or not."""
        return any(key.covers(path) for key in self.keys)

    # -- evaluation -----------------------------------------------------------

    def evaluate(self, time: float, fallback: EffectsState) -> EffectsState:
        keys = sorted(self.keys, key=lambda item: item.time)
        if not keys:
            return fallback.clone()
        if all(not key.paths for key in keys):
            return self._evaluate_snapshots(keys, time)
        return self._evaluate_paths(keys, time, fallback)

    @staticmethod
    def _evaluate_snapshots(keys: list[EffectsKey], time: float) -> EffectsState:
        if time <= keys[0].time:
            return keys[0].state.clone()
        if time >= keys[-1].time:
            return keys[-1].state.clone()
        times = [key.time for key in keys]
        index = max(1, bisect_left(times, time))
        left, right = keys[index - 1], keys[index]
        f = (time - left.time) / max(right.time - left.time, 1e-6)
        # Smoothstep prevents visible velocity changes at effect keys.
        f = f * f * (3.0 - 2.0 * f)
        return interpolate_effects(left.state, right.state, f)

    def _evaluate_paths(
        self, keys: list[EffectsKey], time: float, fallback: EffectsState
    ) -> EffectsState:
        """Evaluate each property against only the keys that pin it.

        A property nobody keyed holds its live value instead of being dragged
        along by a neighbouring key, which is what makes a single diamond on a
        single field mean what it does in Blender or After Effects.
        """
        snapshots = [key for key in keys if not key.paths]
        result = (
            self._evaluate_snapshots(snapshots, time) if snapshots else fallback.clone()
        )
        touched: set[str] = set()
        for key in keys:
            touched.update(key.paths or animatable_paths(key.state))
        for path in touched:
            samples = [
                (key.time, get_path(key.state, path))
                for key in keys
                if key.covers(path)
            ]
            samples = [(t, v) for t, v in samples if v is not None]
            if samples:
                set_path(result, path, _sample_path(samples, time, path))
        return result


def volume_preset(kind: str, pivot_z: float = -4.0) -> VolumeEffect:
    kind = kind.lower()
    presets = {
        "fog": dict(name="Fog volume", density=0.22, noise_scale=1.2, speed=0.08),
        "smoke": dict(name="Smoke volume", density=0.55, noise_scale=2.4, speed=0.28),
        "fire": dict(name="Fire volume", density=0.75, noise_scale=3.0, speed=0.70,
                     colour=(1.0, 0.22, 0.035), emission=3.2),
        "cloud": dict(name="Cloud volume", density=0.48, noise_scale=1.7, speed=0.05,
                      colour=(0.9, 0.93, 1.0)),
        "godrays": dict(name="Godray volume", density=0.34, noise_scale=2.1, speed=0.04,
                        colour=(1.0, 0.82, 0.52), light_response=1.5),
    }
    values = presets.get(kind, presets["fog"])
    return VolumeEffect(kind=kind if kind in presets else "fog", position=(0.0, 0.0, pivot_z), **values)


def emitter_preset(kind: str, pivot_z: float = -4.0) -> ParticleEmitter:
    kind = kind.lower()
    presets = {
        "smoke": dict(name="Smoke emitter", count=420, speed=0.5, gravity=-0.03),
        "fire": dict(name="Fire emitter", count=520, speed=0.9, gravity=0.08,
                     colour=(1.0, 0.24, 0.03), emission=4.0, particle_size=0.075),
        "embers": dict(name="Ember emitter", count=300, speed=1.2, gravity=0.12,
                       colour=(1.0, 0.32, 0.04), emission=5.0, particle_size=0.035),
        "dust": dict(name="Dust emitter", count=500, speed=0.12, gravity=-0.01,
                     colour=(0.76, 0.66, 0.48), particle_size=0.025),
        "snow": dict(name="Snow emitter", count=650, speed=-0.38, gravity=-0.22,
                     colour=(0.92, 0.96, 1.0), particle_size=0.035),
        "clouds": dict(name="Cloud particles", count=260, speed=0.08, gravity=0.0,
                       colour=(0.9, 0.93, 1.0), particle_size=0.22),
    }
    values = presets.get(kind, presets["smoke"])
    return ParticleEmitter(kind=kind if kind in presets else "smoke", position=(0.0, -0.8, pivot_z), **values)


def normalise_direction(value) -> tuple[float, float, float]:
    length = math.sqrt(sum(float(v) ** 2 for v in value))
    if length <= 1e-8:
        return (0.0, 1.0, 0.0)
    return tuple(float(v) / length for v in value)
