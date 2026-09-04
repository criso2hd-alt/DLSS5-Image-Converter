# Security & transparency

This app is open source and does far less under the hood than an antivirus
heuristic suggests. This page lays out exactly what it touches, why, which
versions of everything it uses, and how to check every claim yourself. "Trust
me, it's a false positive" is not good enough, and the people asking for detail
are right to.

- Repository: <https://github.com/criso2hd-alt/DLSS5-Image-Converter>
- This page in the wiki: [Security](https://github.com/criso2hd-alt/DLSS5-Image-Converter/wiki/Security)
- Antivirus specifics: [Troubleshooting](https://github.com/criso2hd-alt/DLSS5-Image-Converter/wiki/Troubleshooting)

## In one paragraph

The code never writes to the registry. The only registry value it reads is your
Steam install path, read-only, so *Find my DLSS files* can locate your games.
The only servers it contacts are three, all over HTTPS, all one-time first-run
downloads of its own runtime. It bundles and downloads no NVIDIA binaries; you
supply those yourself. The two registry keys a scanner flagged
(`Tcpip\Parameters` and `SystemCertificates\Root`) are read by Windows itself
while it validates those downloads, and by NVIDIA's own DLSS updater. Our code
does not touch them. Everything below shows this in the actual source.

## Under the hood: the actual code

There is no hidden logic. These are the only two files that touch the registry
or the network, quoted straight from the source.

### Every registry access, in full

The whole of our registry code is `steam_root()` in
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

`OpenKey` and `QueryValueEx` are reads. There is no `SetValueEx`, `CreateKey`,
or `DeleteKey` anywhere in the project; grep it and see. This runs only when you
press *Find my DLSS files*, and only to find your Steam library so it can copy
your own DLSS files into place.

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

No sockets are opened anywhere else. There is no telemetry, no analytics, and
nothing is sent out. These are downloads only, and after first-run setup the app
makes no network calls at all.

## The registry keys a scanner flagged

A security tool reported this chain:

```
DLSS5Converter.exe > dlss5_eval.exe > NGX Updater
  reads  HKLM\SYSTEM\...\Services\Tcpip\Parameters
  reads  HKLM\Software\Microsoft\SystemCertificates\Root
```

Neither key is touched by our code. `dlss5_eval.exe` (the DLSS harness) has no
registry code at all, which you can confirm in
[`native/dlss5_eval/main.cpp`](native/dlss5_eval/main.cpp). Both keys are read,
never written, by two things that are not our logic:

1. Windows, during the HTTPS downloads above. Validating a TLS certificate makes
   Windows read the root certificate store (`SystemCertificates\Root`), and
   opening a connection reads `Tcpip\Parameters` for DNS. Every program that
   downloads over HTTPS causes these reads. Your browser does it on every page.

2. NVIDIA's NGX runtime, shown as "NGX Updater" in that chain. When DLSS starts,
   NVIDIA's `nvngx*.dll` runs its own update and telemetry component, which
   contacts NVIDIA and triggers the same certificate and network reads. That is
   NVIDIA's code running inside our process, and it is the same behavior any DLSS
   game triggers. The same updater touches the same keys when Cyberpunk starts.
   There is no public NGX setting to disable it, since it is part of the DLSS
   runtime you supply rather than anything we can turn off.

## Why the first versions were not flagged and newer ones are

A fair question, and an important one. The honest answer is that the binary
changed, not the intent, and the detection moves around on Microsoft's side.

Every release is a new, unsigned file with a new hash and no reputation.
Defender's machine learning judges each new binary from scratch. It does not
remember that the previous version was fine, because it has never seen this exact
file before.

The app also got bigger. Newer versions added the video feature, which bundles
Qt Multimedia and downloads PyAV (FFmpeg). More code, more bundled DLLs, more
imports, so the file's size and structure shifted. Machine-learning scanners
score those surface features rather than what the program actually does, so a
heavier build can cross a threshold a leaner one stayed under, with no change in
behavior. Newer builds also do a couple more things heuristics add up: download
an extra component, spawn the harness, and surface NVIDIA's NGX updater. And
Defender's models update on Microsoft's side almost daily, so a pattern that
passed last week can start getting flagged with nothing changed on our end.

The labels give it away. `Wacatac.C!ml` and `Dropper.Agent` are generic
families, the `!ml` means it is a machine-learning guess, and no two scanners
agree on what it supposedly is. Real malware gets consistent, specific names
across vendors. A VirusTotal scan shows this file clean across the large
majority of engines.

The proper fix is code signing. A certificate gives each release a stable,
trusted identity so it is not re-judged as an anonymous new file every time.
This is a free solo project without a signing certificate yet, which is the real
reason unsigned rebuilds keep tripping heuristics. It is on the roadmap.

## Reported to Microsoft

The current build has been submitted to Microsoft as an incorrect detection so
the flag gets corrected for everyone:

- Submission ID: `9d149265-5aa7-4060-b401-f3de55c26d22`
- Type: Incorrect detection (false positive)
- Detection disputed: `Trojan:Win32/Wacatac.C!ml`
- Status: submitted, analyst determination pending
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
| FFmpeg (inside PyAV) | 8.x (libavcodec 62.28, libavformat 62.12, libavutil 60.26) | inside the PyAV wheel |
| NVIDIA NGX SDK (API) | 1.5.0 | headers only; you supply the runtime DLLs |
| ReShade | your own install (6.x) | you supply it |

The NVIDIA and ReShade binaries are never shipped or downloaded by this app. You
point it at copies you already have.

## Check all of it yourself

- Read the two files. `discovery.py` (registry) and `bootstrap.py` (network) are
  the only ones that matter here, and both are short.
- Check the hash on each release against your download. No tampered copy can
  match. For v0.1.23:
  ```
  exe  SHA-256: 8852b8f3bc0f720fbf557b97467c19ebe591ce4d9e5c7f054c7c522c47cc5acc
  zip  SHA-256: 42a148cf4a55e67e5b6f5feb276a4b97093d37313adf8865acb1ab64f85a9e4a
  ```
- Watch it live with Sysinternals Process Monitor or Wireshark. You will see the
  three hosts above plus NVIDIA's NGX, registry reads, and no writes to protected
  keys.
- Build it from source yourself. The whole thing is here.

## Reporting a real issue

If something above does not explain what you see, open an issue with the tool,
the exact key or endpoint, and whether it was a read or a write. Genuine security
reports are welcome and get answered directly.
