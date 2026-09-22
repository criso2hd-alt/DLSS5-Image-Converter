"""Which GPU the app's own GPU work runs on, for machines with more than one.

The DLSS pass is not chosen here. The harness has to create its device on the
Windows default adapter (NGX rejects a device made on an explicitly chosen
one, see native/dlss5_eval/main.cpp), so that card is decided by Windows'
per-app graphics setting. Everything else the app runs on the GPU is ours to
place: the ONNX models (depth, SHARP, LaMa, AI upscale) through DirectML, the
3D renderer through wgpu, and the VRAM gauge.

Left alone, each of those picks its own "first" GPU, and on a laptop with an
external card they disagree: the harness lands on the eGPU while SHARP and the
3D view load onto the laptop GPU, run out of its VRAM and render black. So the
automatic choice follows whatever adapter the harness reported, which keeps
all of the app's work on one card, and the user can override it in Settings.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading

_NVIDIA = 0x10DE
_SOFTWARE_FLAG = 0x2          # DXGI_ADAPTER_FLAG_SOFTWARE (Microsoft Basic Render)


@dataclass(frozen=True)
class Gpu:
    #: Position among the hardware adapters in DXGI order, duplicates
    #: included. DirectML's device_id counts the same list.
    index: int
    name: str
    vendor_id: int
    vram_bytes: int

    @property
    def label(self) -> str:
        return f"{self.name}  ({self.vram_bytes / 1024**3:.0f} GB)"


_lock = threading.Lock()
_cache: list[Gpu] | None = None
_preference = ""        # a GPU name from Settings; empty means automatic
_harness_adapter = ""   # what the last --probe said DLSS runs on


def _enumerate() -> list[Gpu]:
    """Hardware adapters via DXGI, through ctypes so nothing new is bundled."""
    import ctypes
    from ctypes import wintypes

    class Luid(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.LONG)]

    class Desc1(ctypes.Structure):
        _fields_ = [
            ("description", ctypes.c_wchar * 128),
            ("vendor_id", ctypes.c_uint), ("device_id", ctypes.c_uint),
            ("subsys_id", ctypes.c_uint), ("revision", ctypes.c_uint),
            ("dedicated_video", ctypes.c_size_t), ("dedicated_system", ctypes.c_size_t),
            ("shared_system", ctypes.c_size_t), ("luid", Luid), ("flags", ctypes.c_uint),
        ]

    class Guid(ctypes.Structure):
        _fields_ = [("d1", ctypes.c_ulong), ("d2", ctypes.c_ushort),
                    ("d3", ctypes.c_ushort), ("d4", ctypes.c_ubyte * 8)]

    # IID_IDXGIFactory1 {770aae78-f26f-4dba-a829-253c83d1b387}
    iid = Guid(0x770AAE78, 0xF26F, 0x4DBA,
               (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
    factory = ctypes.c_void_p()
    if ctypes.windll.dxgi.CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(factory)) != 0:
        return []

    def method(obj, slot, *argtypes):
        vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtable[slot])

    found: list[Gpu] = []
    seen: set[tuple] = set()
    hardware = 0
    try:
        i = 0
        while True:
            adapter = ctypes.c_void_p()
            # IDXGIFactory1::EnumAdapters1 is vtable slot 12.
            if method(factory, 12, ctypes.c_uint, ctypes.c_void_p)(
                    factory, i, ctypes.byref(adapter)) != 0:
                break                                 # DXGI_ERROR_NOT_FOUND: done
            try:
                desc = Desc1()
                # IDXGIAdapter1::GetDesc1 is slot 10.
                if method(adapter, 10, ctypes.c_void_p)(adapter, ctypes.byref(desc)) == 0 \
                        and not desc.flags & _SOFTWARE_FLAG:
                    # Windows can list one card twice under two LUIDs (seen on
                    # a single RTX 4080). Counting it twice would show a
                    # picker on a one-GPU machine, so identical hardware is
                    # listed once, at its first position.
                    key = (desc.vendor_id, desc.device_id, desc.subsys_id, desc.revision)
                    if key not in seen:
                        seen.add(key)
                        found.append(Gpu(hardware, desc.description.strip(),
                                         desc.vendor_id, int(desc.dedicated_video)))
                    hardware += 1
            finally:
                method(adapter, 2)(adapter)           # Release
            i += 1
    finally:
        method(factory, 2)(factory)
    return found


def list_gpus() -> list[Gpu]:
    """Every hardware GPU, cached (adapters do not come and go mid-session
    often enough to justify re-asking DXGI on every model load)."""
    global _cache
    with _lock:
        if _cache is None:
            try:
                _cache = _enumerate()
            except Exception:  # noqa: BLE001 - no DXGI means no choice to make
                _cache = []
        return list(_cache)


def set_preference(name: str) -> None:
    global _preference
    _preference = name or ""


def preference() -> str:
    return _preference


def note_harness_adapter(report: str) -> None:
    """Remember the adapter a --probe report says DLSS runs on."""
    global _harness_adapter
    for line in report.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "adapter" and value.strip():
            _harness_adapter = value.strip()
            return


def harness_adapter() -> str:
    return _harness_adapter


def _by_name(gpus: list[Gpu], name: str) -> Gpu | None:
    name = name.strip().lower()
    return next((g for g in gpus if g.name.lower() == name), None) if name else None


def selected() -> Gpu | None:
    """The GPU the app's own work should run on."""
    gpus = list_gpus()
    if not gpus:
        return None
    chosen = _by_name(gpus, _preference) or _by_name(gpus, _harness_adapter)
    if chosen:
        return chosen
    # Nothing to go on yet: the NVIDIA card with the most memory, since that is
    # where DLSS has to run anyway.
    nvidia = [g for g in gpus if g.vendor_id == _NVIDIA]
    return max(nvidia or gpus, key=lambda g: g.vram_bytes)


def multiple() -> bool:
    return len(list_gpus()) > 1


def ort_providers(names: list[str]) -> list:
    """onnxruntime provider list with DirectML pinned to the selected GPU.

    With one GPU this returns the names unchanged, so single-GPU machines take
    exactly the path they always have."""
    gpu = selected() if multiple() else None
    if gpu is None:
        return list(names)
    return [("DmlExecutionProvider", {"device_id": str(gpu.index)})
            if n == "DmlExecutionProvider" else n for n in names]


def wgpu_adapter():
    """A wgpu adapter on the selected GPU, on the backend wgpu picks by
    default (Vulkan, then D3D12), so only the card changes, not the API."""
    import wgpu
    gpu = selected() if multiple() else None
    if gpu is not None:
        try:
            matches = [a for a in wgpu.gpu.enumerate_adapters_sync()
                       if str(a.info.get("device", "")).lower() == gpu.name.lower()]
            order = {"Vulkan": 0, "D3D12": 1}
            matches.sort(key=lambda a: order.get(a.info.get("backend_type"), 2))
            if matches:
                return matches[0]
        except Exception:  # noqa: BLE001 - fall back to wgpu's own choice
            pass
    return wgpu.gpu.request_adapter_sync(power_preference="high-performance")


def pick_smi_line(lines: list[str]) -> str:
    """The nvidia-smi CSV row for the selected GPU (name is the first column)."""
    rows = [ln for ln in lines if ln.strip()]
    gpu = selected()
    if gpu is not None:
        for row in rows:
            if row.split(",")[0].strip().lower() == gpu.name.lower():
                return row
    return rows[0] if rows else ""
