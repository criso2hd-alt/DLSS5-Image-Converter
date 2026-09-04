# Security & transparency

This app is open source and does far less under the hood than an antivirus
heuristic suggests. This page lays out exactly what it touches, why, which
versions of everything it uses, and how to verify every claim yourself — because
"trust me, it's a false positive" is not good enough, and the people asking for
detail are right to.

- Repository: <https://github.com/criso2hd-alt/DLSS5-Image-Converter>
- This page in the wiki: [Security](https://github.com/criso2hd-alt/DLSS5-Image-Converter/wiki/Security)
- Antivirus specifics: [Troubleshooting → Windows Security](https://github.com/criso2hd-alt/DLSS5-Image-Converter/wiki/Troubleshooting)

## In one paragraph

The code **never writes to the registry**. The **only** registry value it reads
is your Steam install path, read-only, to find your games for *Find my DLSS
files*. The **only** servers it contacts are three, all HTTPS, all one-time
first-run downloads of its own runtime. It bundles and downloads **no NVIDIA
binaries** — you supply those. The two registry keys a scanner flagged
(`Tcpip\Parameters`, `SystemCertificates\Root`) are read by **Windows itself**
validating those downloads and by **NVIDIA's own DLSS updater**, not by our
code. Everything below shows this in the actual source.

## Under the hood — the actual code

There is no hidden logic. These are the *only* two files that touch the registry
or the network, quoted verbatim.

### Every registry access, in full

The whole of our registry code — `steam_root()` in
[`dlss5_converter/discovery.py`](dlss5_converter/discovery.py):

```python
def steam_root() -> Path | None:
    """Where Steam is installed, from the registry."""
    import winreg
    for hive, key in (
        (winreg.HKEY_CURRENT_USER,  r"Software\Valve\Steam"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam"),
    ):
        for value in ("SteamPath", "InstallPath"):
            with winreg.OpenKey(hive, key) as handle:      # OpenKey = read
                path = Path(winreg.QueryValueEx(handle, value)[0])  # QueryValue = read
            if path.is_dir():
                return path
```

`OpenKey` + `QueryValueEx` are **reads**. There is no `SetValueEx`,
`CreateKey`, or `DeleteKey` anywhere in the project — grep it. This runs only
when you press *Find my DLSS files*, and only to locate your Steam library so it
can copy your own DLSS files in.

### Every network access, in full

The whole of our network code lives in
[`dlss5_converter/bootstrap.py`](dlss5_converter/bootstrap.py). It uses Python's
standard `urllib` and contacts exactly three hosts, once, to download the app's
own runtime:

```python
from urllib.request import Request, urlopen

TORCH_VERSION   = "2.13.0+cu130"
TORCH_WHEEL_URL = "https://download.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp312-cp312-win_amd64.whl"
AV_VERSION      = "18.1.0"           # resolved from https://pypi.org/pypi/av/18.1.0/json
# Depth model: "depth-anything/Depth-Anything-V2-Base-hf" from https://huggingface.co
```

No sockets are opened anywhere else, there is no telemetry, no analytics, and
nothing is sent *out* — these are downloads only. After first-run setup the app
makes no network calls at all.

## The registry keys a scanner flagged

A security tool reported this chain:

```
DLSS5Converter.exe > dlss5_eval.exe > NGX Updater
  reads  HKLM\SYSTEM\...\Services\Tcpip\Parameters
  reads  HKLM\Software\Microsoft\SystemCertificates\Root
```

Neither key is touched by our code. `dlss5_eval.exe` (the DLSS harness) contains
**no registry code at all** — confirm in
[`native/dlss5_eval/main.cpp`](native/dlss5_eval/main.cpp). Both keys are
**read, never written**, by two things that are not our logic:

1. **Windows, during the HTTPS downloads above.** Validating a TLS certificate
   makes Windows read the root certificate store (`SystemCertificates\Root`);
   opening a connection reads `Tcpip\Parameters` for DNS. *Every* program that
   downloads over HTTPS causes these reads — your browser does it on every page.

2. **NVIDIA's NGX runtime — the "NGX Updater" in that chain.** When DLSS
   initialises, NVIDIA's `nvngx*.dll` starts its own update/telemetry component,
   which contacts NVIDIA and triggers the same certificate/network reads. This is
   NVIDIA's code running inside our process, and it is the **identical** behaviour
   any DLSS game triggers — the same updater touches the same keys when Cyberpunk
   starts. There is no public NGX switch to disable it; it is part of the DLSS
   runtime you supply, not something we can turn off.

## Why the first versions weren't flagged and newer ones are

A fair and important question. The honest answer is that the *binary* changed,
not the intent, and the detection is a moving target on Microsoft's side. In
plain terms:

- **Every release is a brand-new, unsigned file with a new hash and zero
  reputation.** Defender's machine-learning classifier judges each new binary
  from scratch. It does not "remember" that the last version was fine — it has
  never seen this exact file before.
- **The app got bigger and more complex.** Newer versions added the video
  feature, which bundles Qt Multimedia and downloads PyAV (FFmpeg). More code,
  more bundled DLLs, more imports — the executable's size, structure and byte
  patterns all shifted. ML malware classifiers score exactly those surface
  features, so a leaner early build can sit under a threshold that a fatter
  later build crosses, with no change in what the program *does*.
- **More behaviours that heuristics weight.** Newer builds download an extra
  component and spawn the harness, and the harness now surfaces NVIDIA's NGX
  updater. "Downloads a file, launches a process, touches the cert store" is a
  generic pattern heuristics add up — legitimate here, but it accumulates score.
- **Defender's models update daily.** A model change on Microsoft's side can
  start flagging a PyInstaller pattern that passed last week, entirely
  independent of us. This is the classic "it was fine before" cause.

The labels themselves give it away: `Wacatac.C!ml` and `Dropper.Agent` are
*generic families*, the `!ml` means "machine-learning guess", and **no two
engines agree** on what it supposedly is. Real malware gets consistent,
specific names across vendors. A VirusTotal scan shows this file clean across
the large majority of engines.

The durable fix is **code signing** — a certificate gives each release a stable,
trusted identity so it is not re-judged as an anonymous new binary every time.
This is a free solo project without a signing certificate yet; that is the real
reason unsigned rebuilds keep tripping heuristics, and it is on the roadmap.

## Reported to Microsoft

The current build has been submitted to Microsoft as an incorrect detection so
the flag is corrected for everyone:

- **Submission ID:** `9d149265-5aa7-4060-b401-f3de55c26d22`
- **Type:** Incorrect detection (false positive)
- **Detection disputed:** `Trojan:Win32/Wacatac.C!ml`
- **Status:** submitted, analyst determination pending
- Submit your own or check status: <https://www.microsoft.com/en-us/wdsi/filesubmission>

## Versions of everything it uses

So you know exactly what code is running, not just "some libraries":

| Component | Version | Source |
| --------- | ------- | ------ |
| Python runtime | 3.12.10 | bundled |
| PyTorch | 2.13.0+cu130 | download.pytorch.org (first run) |
| Depth Anything V2 (Base) | `depth-anything/Depth-Anything-V2-Base-hf` | huggingface.co (first run) |
| transformers | 4.57.6 | bundled |
| tokenizers | 0.22.2 | bundled |
| safetensors | 0.8.0 | bundled |
| huggingface-hub | 0.36.2 | bundled |
| NumPy | 2.5.2 | bundled |
| Pillow | 11.3.0 | bundled |
| OpenCV | 4.14.0 | bundled |
| PySide6 (Qt) | 6.11.2 | bundled |
| PyAV | 18.1.0 | pypi.org (first use of Video) |
| FFmpeg (inside PyAV) | 8.x — libavcodec 62.28, libavformat 62.12, libavutil 60.26 | inside the PyAV wheel |
| NVIDIA NGX SDK (API) | 1.5.0 | headers only; **you** supply the runtime DLLs |
| ReShade | your own install (6.x) | **you** supply it |

The NVIDIA and ReShade binaries are never shipped or downloaded by this app —
you point it at copies you already have.

## Verify all of it yourself

- **Read the two files.** `discovery.py` (registry) and `bootstrap.py` (network)
  are the only ones that matter here, and they are short.
- **Check the hash** on each release against your download — no tampered copy can
  match. v0.1.23:
  ```
  exe  SHA-256: 8852b8f3bc0f720fbf557b97467c19ebe591ce4d9e5c7f054c7c522c47cc5acc
  zip  SHA-256: 42a148cf4a55e67e5b6f5feb276a4b97093d37313adf8865acb1ab64f85a9e4a
  ```
- **Watch it live** with Sysinternals Process Monitor or Wireshark: you will see
  the three hosts above plus NVIDIA's NGX, registry *reads*, and **no writes** to
  protected keys.
- **Build it from source** yourself — the whole thing is here.

## Reporting a real issue

If something above does not explain what you see, open an issue with the tool,
the exact key or endpoint, and read-vs-write. Genuine security reports are
welcome and answered directly.
