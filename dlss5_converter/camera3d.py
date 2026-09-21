"""Perspective camera for the 3D tab.

Adapted from Depth Animator: a real camera (position, target, FOV) so occlusion,
fog and dolly-zoom are meaningful. Right-handed, -Z forward, matching the mesh
built by mesh3d (which unprojects the same way). Projection targets wgpu clip
space (Z in [0, 1]).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Camera:
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target: tuple[float, float, float] = (0.0, 0.0, -4.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov_degrees: float = 55.0
    near: float = 0.05
    far: float = 200.0

    def view_matrix(self) -> np.ndarray:
        return look_at(self.position, self.target, self.up)

    def projection_matrix(self, aspect: float) -> np.ndarray:
        return perspective(self.fov_degrees, aspect, self.near, self.far)

    def view_projection(self, aspect: float) -> np.ndarray:
        return self.projection_matrix(aspect) @ self.view_matrix()


def _normalise(v: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(v))
    if length < 1e-9:
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return (v / length).astype(np.float32)


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    forward = _normalise(target - eye)
    right = np.cross(forward, up)
    if float(np.linalg.norm(right)) < 1e-6:
        right = np.cross(forward, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    right = _normalise(right)
    true_up = np.cross(right, forward)
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, :3] = right
    matrix[1, :3] = true_up
    matrix[2, :3] = -forward
    matrix[0, 3] = -float(np.dot(right, eye))
    matrix[1, 3] = -float(np.dot(true_up, eye))
    matrix[2, 3] = float(np.dot(forward, eye))
    return matrix


def perspective(fov_degrees: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_degrees) * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = f / max(aspect, 1e-6)
    matrix[1, 1] = f
    matrix[2, 2] = far / (near - far)
    matrix[2, 3] = (far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


PRESET_NAMES = ["Static", "Orbit", "Drift", "Push in", "Pull out", "Vertigo"]


def _ease(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def preset_rig(name: str, t: float, pivot_z: float, strength: float = 1.0,
               base_fov: float = 55.0) -> "OrbitRig":
    """Camera pose for a named preset at normalised loop time t (loop-safe:
    returns to the start pose at t = 1). `strength` scales the whole move."""
    t = max(0.0, min(1.0, float(t)))
    phase = t * math.tau
    distance = abs(pivot_z)
    pivot = (0.0, 0.0, -distance)
    # Deliberately gentle: a single depth map only holds up under small moves,
    # and big orbits read as dizzy. `strength` still lets the user push it.
    yaw_amp = 0.05 * strength
    dolly_amp = distance * 0.07 * strength

    def rig(**kwargs) -> "OrbitRig":
        params = {"pivot": pivot, "distance": distance, "fov_degrees": base_fov}
        params.update(kwargs)
        return OrbitRig(**params)

    if name == "Static":
        return rig()
    if name == "Push in":
        return rig(dolly=dolly_amp * _ease(t))
    if name == "Pull out":
        return rig(dolly=dolly_amp * (1.0 - _ease(t)))
    if name == "Drift":
        return rig(yaw=math.sin(phase) * yaw_amp * 0.7,
                   pitch=math.sin(phase * 2.0) * yaw_amp * 0.3,
                   dolly=dolly_amp * 0.2 * math.sin(phase))
    if name == "Vertigo":
        push = math.sin(math.pi * t)
        travel = dolly_amp * 1.4 * push
        remaining = max(distance - travel, 0.15 * distance)
        fov = math.degrees(2.0 * math.atan(
            math.tan(math.radians(base_fov) * 0.5) * distance / remaining))
        return rig(dolly=travel, fov_degrees=float(np.clip(fov, 12.0, 120.0)))
    # Orbit (default).
    return rig(yaw=math.sin(phase) * yaw_amp,
               pitch=math.cos(phase) * yaw_amp * 0.42 - yaw_amp * 0.42)


@dataclass(slots=True)
class OrbitRig:
    pivot: tuple[float, float, float] = (0.0, 0.0, -4.0)
    distance: float = 4.0
    yaw: float = 0.0
    pitch: float = 0.0
    dolly: float = 0.0
    fov_degrees: float = 55.0
    roll: float = 0.0
    offset: tuple[float, float, float] = field(default=(0.0, 0.0, 0.0))

    def to_camera(self) -> Camera:
        radius = max(self.distance - self.dolly, 0.05)
        cos_pitch = math.cos(self.pitch)
        eye = np.array([
            self.pivot[0] + radius * math.sin(self.yaw) * cos_pitch + self.offset[0],
            self.pivot[1] + radius * math.sin(self.pitch) + self.offset[1],
            self.pivot[2] + radius * math.cos(self.yaw) * cos_pitch + self.offset[2],
        ], dtype=np.float32)
        up = (math.sin(self.roll), math.cos(self.roll), 0.0)
        return Camera(position=(float(eye[0]), float(eye[1]), float(eye[2])),
                      target=tuple(float(v) for v in self.pivot), up=up,
                      fov_degrees=self.fov_degrees)
