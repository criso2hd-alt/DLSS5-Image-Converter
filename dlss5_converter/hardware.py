"""Small, dependency-free hardware queries used by conversion preflights."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess


MIB = 1024 * 1024


@dataclass(frozen=True)
class VramInfo:
    """The NVIDIA GPU the app runs on (see gpus.selected), per the driver."""

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

    from . import gpus
    line = gpus.pick_smi_line(result.stdout.splitlines())
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


def query_system_ram() -> int | None:
    """Free system RAM in bytes, or None when it cannot be determined.

    Ultra Detail merges its tiles into an ordinary host array, so its real limit
    is system memory, not VRAM — a 32K×32K merge needs ~16 GB of accumulator
    before the save buffer. This uses the Win32 API directly (via ctypes) so it
    adds no dependency; on a non-Windows host or any failure it returns None and
    the caller falls back to a conservative fixed ceiling.
    """
    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullAvailPhys)
    except Exception:  # noqa: BLE001 - a missing API must degrade, never block
        return None
