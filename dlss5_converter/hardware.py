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


#: NVIDIA drivers from 616.64 fault inside NVIDIA's own neural runtime on
#: every evaluate, so the DLSS 5 neural pass cannot run on them (the same limit
#: DLSS 5 Swapper warns about, and what every "worked after rolling the driver
#: back" report in the community comes down to). 616.56 is the last driver
#: measured to complete one. Update these two if NVIDIA changes course.
LAST_WORKING_DRIVER = "616.56"
FIRST_BLOCKED_DRIVER = "616.64"
DRIVER_DOWNLOADS_URL = "https://www.nvidia.com/en-us/drivers/"


def parse_driver(version: str | None) -> tuple[int, int] | None:
    """'616.92' -> (616, 92); None for anything that is not a driver number."""
    try:
        major, minor = str(version).strip().split(".")[:2]
        return int(major), int(minor)
    except (ValueError, AttributeError):
        return None


def driver_blocks_neural(version: str | None) -> bool:
    parsed = parse_driver(version)
    return parsed is not None and parsed >= parse_driver(FIRST_BLOCKED_DRIVER)


def driver_warning(version: str | None) -> str | None:
    """What to tell someone on a driver that cannot run the neural pass, or None."""
    if not driver_blocks_neural(version):
        return None
    return (f"Your NVIDIA driver ({version}) cannot run the DLSS 5 neural pass. NVIDIA "
            f"drivers {FIRST_BLOCKED_DRIVER} and newer fault inside NVIDIA's own neural "
            f"runtime, so conversions fail or come out unchanged, whatever files you use. "
            f"Roll the driver back to {LAST_WORKING_DRIVER} or older (NVIDIA's driver "
            f"page lets you pick an older version). Everything else in the app, depth, "
            f"the 3D tab, effects and export, still works on this driver.")


def query_driver_version() -> str | None:
    """The NVIDIA driver version as people quote it (e.g. 576.02), or None."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = next((ln.strip() for ln in result.stdout.splitlines() if ln.strip()), "")
    return line if result.returncode == 0 and line else None


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
