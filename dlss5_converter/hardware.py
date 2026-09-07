"""Small, dependency-free hardware queries used by conversion preflights."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess


MIB = 1024 * 1024


@dataclass(frozen=True)
class VramInfo:
    """The first NVIDIA GPU reported by the installed driver."""

    name: str
    total_bytes: int
    used_bytes: int
    free_bytes: int


def query_nvidia_vram() -> VramInfo | None:
    """Return current NVIDIA VRAM availability without importing PyTorch.

    ``nvidia-smi`` ships with the driver and reports the memory left *after*
    the depth model and other applications have taken their share. Failure is
    deliberately non-fatal: an unusual driver setup should still be allowed to
    try the conversion and let D3D12 report the authoritative allocation error.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None

    line = next((line for line in result.stdout.splitlines() if line.strip()), "")
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 4:
        return None
    try:
        total, used, free = (int(float(value) * MIB) for value in parts[1:4])
    except ValueError:
        return None
    if total <= 0 or free < 0:
        return None
    return VramInfo(parts[0], total, used, free)
