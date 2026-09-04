# Security & transparency

This app is open source and does far less under the hood than the antivirus
heuristics suggest. This page says exactly what it touches, why, and how to
verify every claim yourself — because "trust me, it's a false positive" is not
good enough, and the people asking for detail are right to.

## Short version

- The code **never writes to the registry**.
- The **only** registry value our code reads is your **Steam install path**,
  read-only, and only when you click *Find my DLSS files* — to locate your
  games. It is in [`discovery.py`](dlss5_converter/discovery.py).
- The **only** servers our code contacts are three, all HTTPS, all first-run
  downloads: `download.pytorch.org`, `huggingface.co`, `pypi.org`. They are in
  [`bootstrap.py`](dlss5_converter/bootstrap.py). After the one-time setup, our
  code makes no network calls at all.
- It bundles, references, and downloads **no NVIDIA binaries**. You supply your
  own DLSS files.

Everything below is the long version of those four points.

## The registry accesses people have flagged

A security tool reported this chain:

```
DLSS5Converter.exe > dlss5_eval.exe > NGX Updater
  reads HKLM\...\Services\Tcpip\Parameters
  reads HKLM\Software\Microsoft\SystemCertificates\Root
```

Neither key is touched by our code. `dlss5_eval.exe` (the DLSS harness) contains
**no registry code whatsoever** — you can confirm this in
[`native/dlss5_eval/main.cpp`](native/dlss5_eval/main.cpp). Both keys are **read,
never written**, by two things that are not our logic:

1. **Windows itself, during the first-run HTTPS downloads.** Validating a TLS
   certificate makes Windows read the root certificate store
   (`SystemCertificates\Root`); opening a network connection reads
   `Tcpip\Parameters` for DNS and network configuration. *Every* program that
   downloads anything over HTTPS causes these reads. They are the operating
   system doing its job, not the app inspecting your certificates or network.

2. **NVIDIA's NGX runtime — the "NGX Updater" in that chain.** When DLSS
   initialises, NVIDIA's `nvngx*.dll` starts its own update/telemetry component,
   which contacts NVIDIA and therefore triggers the same OS-level certificate
   and network reads. This is NVIDIA's code running inside our harness process,
   and it is the **identical** behaviour you get when any DLSS game starts — the
   same updater touches the same keys when you launch Cyberpunk. There is no
   public NGX setting to disable it; it is part of the DLSS runtime you supply.

So: no registry writes, no certificate tampering. Read-only accesses caused by
downloading over HTTPS and by NVIDIA's own DLSS updater.

## The network endpoints, in full

Our code talks to exactly these, only during first-run setup, only over HTTPS:

| Host | What | Why |
| ---- | ---- | --- |
| `download.pytorch.org` | PyTorch wheel (~1.8 GB) | depth estimation runs on it |
| `huggingface.co` | Depth Anything V2 weights (~400 MB) | the depth model |
| `pypi.org` | PyAV wheel (~28 MB) | video decode/encode, first time you use the Video tab |

Nothing else. No analytics, no telemetry, no "phone home" from our code. NVIDIA's
NGX may contact NVIDIA independently, as described above.

## "Earlier versions didn't trigger this — what changed?"

A fair question, and the honest answer is: the *binary* changed, not the
intent. These are heuristic detections on an **unsigned** executable, and they
are unstable across builds — the machine-learning models re-evaluate every new
binary, so a rebuild can flip a verdict with no change in behaviour. Concretely,
newer versions added:

- the **video feature**, which downloads PyAV (one more HTTPS download — more of
  the same certificate/network reads);
- driver and adapter reporting in the harness.

None of that is a new capability to reach into your system. The labels the
scanners use — `Wacatac.C!ml`, `Dropper.Agent` — are *generic* families, and no
two engines agree on what it supposedly is, which is the signature of a false
positive rather than real malware. A VirusTotal scan shows it clean across the
overwhelming majority of engines.

Why unsigned matters: signing an executable with a certificate is what lets
Windows and scanners trust its origin. This is a free, solo project without a
code-signing certificate yet, and unsigned PyInstaller apps are the single most
common source of exactly these heuristic flags. Signing is the real long-term
fix and is on the list.

## How to verify all of this yourself

- **Read the two files named above.** `discovery.py` is the only place with
  registry code; `bootstrap.py` is the only place with network code. They are
  short.
- **Check the hashes** published on each release against your download
  (`Get-FileHash`), so you know you have the genuine, unmodified file.
- **Watch it with a network/registry monitor** (Wireshark, Sysinternals
  Process Monitor). You will see the three hosts above and NVIDIA's NGX, and
  registry reads — no writes to protected keys.
- **It's the whole source.** Nothing is hidden in a binary blob of ours; the
  only binaries we did not write are the NVIDIA/ReShade files you supply
  yourself.

## Reporting a real issue

If you find something the above does not explain, open an issue with what you
observed (the tool, the exact key or endpoint, read vs write). Genuine security
reports are welcome and will be answered directly.
