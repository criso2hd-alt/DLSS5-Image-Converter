param(
    # Wipe the frozen application before rebuilding. The user's own folders -
    # dlss_files, models, output - are preserved either way; this only discards
    # PyInstaller's own output and caches.
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$Release = Join-Path $ProjectRoot "release"
$Engine = Join-Path $Release "engine"
$Staging = Join-Path $ProjectRoot "build\pyinstaller"

# Folders that belong to the user, not to the build. A rebuild must never take
# out a 400 MB model download or a folder of converted images, which is the
# hazard that made CLAUDE.md keep weights out of the app directory in the first
# place. Keeping them across rebuilds is what buys back that guarantee.
$UserFolders = @("dlss_files", "models", "output", "pytorch")

$Python = Join-Path $ProjectRoot ".venv-cuda\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Dependencies are not installed. Run .\scripts\setup.ps1 first."
}

$Harness = Join-Path $ProjectRoot "native\bin\dlss5_eval.exe"
if (-not (Test-Path -LiteralPath $Harness)) {
    throw "native\bin\dlss5_eval.exe is missing. Run .\scripts\build_native.ps1 first."
}

# Not `2>$null`: redirecting a native command's stderr in Windows PowerShell
# wraps each line in an ErrorRecord and trips $ErrorActionPreference = "Stop"
# even when the exit code is 0. Same dance as Test-Interpreter in setup.ps1.
$Previous = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
& $Python -c "import PyInstaller" | Out-Null
$ErrorActionPreference = $Previous
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing PyInstaller..." -ForegroundColor Cyan
    & $Python -m pip install "pyinstaller>=6.11,<7"
    if ($LASTEXITCODE -ne 0) { throw "Could not install PyInstaller." }
}

if ($Clean -and (Test-Path -LiteralPath $Staging)) {
    Remove-Item -LiteralPath $Staging -Recurse -Force
}

Write-Host "Freezing the application (lean: PyTorch is fetched on first run)." -ForegroundColor Cyan

# PyInstaller writes its whole progress log to stderr, and under
# $ErrorActionPreference = "Stop" Windows PowerShell treats a native command's
# stderr as a terminating error - the build would abort on its own banner. Drop
# to Continue for the call and judge it by $LASTEXITCODE, which is the only
# honest signal here anyway.
$Previous = $ErrorActionPreference
$ErrorActionPreference = "Continue"

# torch is excluded on purpose: it is 2.7 GB of the ~3 GB a bundled build used
# to be, and dlss5_converter.bootstrap downloads the wheel on first launch
# instead. Its own metadata comes from the wheel's dist-info, which lands on
# sys.path with it.
#
# Its dependencies do have to stay bundled, though, and there are two kinds.
#
# Third-party (sympy, networkx, jinja2...) are separate packages the torch wheel
# does not carry. Standard library (pickletools, dis, tarfile...) are modules
# that only torch reaches for - with torch out of the analysis, PyInstaller sees
# nothing referencing them and leaves them out of the bundle entirely.
#
# Both failures look identical and both are invisible until the first
# conversion, because bootstrap.is_ready only locates torch rather than
# importing it. `DLSS5Converter.exe --selftest` is what catches them; it found
# "No module named 'pickletools'" in the first lean build.
& $Python -m PyInstaller `
    --noconfirm `
    --windowed `
    --name DLSS5Converter `
    --distpath $Staging `
    --workpath (Join-Path $ProjectRoot "build\pyinstaller-work") `
    --specpath (Join-Path $ProjectRoot "build") `
    --exclude-module torch `
    --collect-all sympy `
    --collect-all networkx `
    --hidden-import jinja2 `
    --hidden-import fsspec `
    --hidden-import mpmath `
    --hidden-import filelock `
    --hidden-import typing_extensions `
    --hidden-import pickletools `
    --hidden-import dis `
    --hidden-import ast `
    --hidden-import tokenize `
    --hidden-import inspect `
    --hidden-import linecache `
    --hidden-import difflib `
    --hidden-import textwrap `
    --hidden-import pprint `
    --hidden-import copy `
    --hidden-import copyreg `
    --hidden-import weakref `
    --hidden-import dataclasses `
    --hidden-import contextlib `
    --hidden-import sysconfig `
    --hidden-import platform `
    --hidden-import glob `
    --hidden-import fnmatch `
    --hidden-import tarfile `
    --hidden-import gzip `
    --hidden-import bz2 `
    --hidden-import lzma `
    --hidden-import zipfile `
    --hidden-import shutil `
    --hidden-import tempfile `
    --hidden-import subprocess `
    --hidden-import multiprocessing `
    --hidden-import queue `
    --hidden-import ctypes.util `
    --hidden-import decimal `
    --hidden-import fractions `
    --hidden-import numbers `
    --hidden-import statistics `
    --hidden-import bisect `
    --hidden-import heapq `
    --hidden-import array `
    --hidden-import mmap `
    --hidden-import struct `
    --hidden-import uuid `
    --hidden-import string `
    --hidden-import unicodedata `
    --hidden-import csv `
    --hidden-import logging.config `
    --hidden-import logging.handlers `
    --hidden-import unittest `
    --hidden-import unittest.mock `
    --hidden-import doctest `
    --hidden-import argparse `
    --hidden-import importlib.metadata `
    --hidden-import importlib.machinery `
    --collect-all transformers `
    --collect-all tokenizers `
    --collect-data safetensors `
    --copy-metadata transformers `
    --copy-metadata tokenizers `
    --copy-metadata safetensors `
    --copy-metadata huggingface-hub `
    --copy-metadata numpy `
    --copy-metadata packaging `
    --copy-metadata pyyaml `
    --copy-metadata regex `
    --copy-metadata requests `
    --copy-metadata filelock `
    --copy-metadata tqdm `
    --exclude-module tkinter `
    --exclude-module matplotlib `
    --exclude-module pytest `
    main.py
$FrozenExit = $LASTEXITCODE
$ErrorActionPreference = $Previous
if ($FrozenExit -ne 0) { throw "PyInstaller failed (exit $FrozenExit)." }

$Frozen = Join-Path $Staging "DLSS5Converter"
if (-not (Test-Path -LiteralPath (Join-Path $Frozen "DLSS5Converter.exe"))) {
    throw "PyInstaller reported success but produced no DLSS5Converter.exe."
}

# A running copy holds its own DLLs open, and the replace step below would then
# delete half the release before hitting the locked file and failing with an
# "Access to the path is denied" from somewhere deep in _internal. Checking up
# front turns that into one clear sentence, before anything is removed.
$Running = Get-Process -Name "DLSS5Converter" -ErrorAction SilentlyContinue
if ($Running) {
    throw ("DLSS5Converter.exe is running (PID $($Running.Id -join ', ')). " +
           "Close it before rebuilding - a running copy locks files in release\_internal.")
}

# Replace only the frozen application, leaving the user's folders alone.
New-Item -ItemType Directory -Force -Path $Release | Out-Null
Get-ChildItem -LiteralPath $Release -Force | Where-Object {
    $UserFolders -notcontains $_.Name
} | Remove-Item -Recurse -Force

Write-Host "Copying the frozen application into release\ ..." -ForegroundColor Cyan
Copy-Item -Path (Join-Path $Frozen "*") -Destination $Release -Recurse -Force

foreach ($Folder in $UserFolders) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Release $Folder) | Out-Null
}
New-Item -ItemType Directory -Force -Path $Engine | Out-Null
Copy-Item -LiteralPath $Harness -Destination $Engine -Force

# The one thing a new user has to do, written where they will look for it.
$Readme = @'
Put your own DLSS 5 files in this folder.

None of these ship with the app and it will not help you obtain them.

  nvngx_dlssnr.dll        the DLSS 5 neural renderer (about 158 MB)
                          on an RTX 40-series card this must be the
                          RTX-40-patched build, not the raw one
  nvngx_dlss.dll          DLSS Super Resolution, from a Streamline
                          Production folder - the neural pass runs
                          inside a DLSS evaluation, so it is required
  renodx-dlss5.addon64    the RenoDX DLSS 5 ReShade add-on
  dxgi.dll                ReShade, renamed. If you already have ReShade
                          in a game, copy that game's bin\x64\dxgi.dll
                          here. Otherwise extract ReShade64.dll from the
                          ReShade installer and rename it to dxgi.dll.

You can drop a Streamline folder in here unflattened; the app looks inside
NVStreamline\Production too.

The app copies these next to engine\dlss5_eval.exe on first run, because that
is where NGX and ReShade look. Use Diagnose in the app to check what it found.
'@
Set-Content -Path (Join-Path $Release "dlss_files\READ ME FIRST.txt") -Value $Readme -Encoding utf8

Set-Content -Path (Join-Path $Release "models\READ ME.txt") -Encoding utf8 -Value @'
Depth Anything V2 weights are downloaded here on first launch, about 400 MB for
the default model. Switching model in the app downloads that one too.

Nothing here ships with the release. Deleting this folder only costs you the
download; rebuilding the app does not touch it.
'@

Set-Content -Path (Join-Path $Release "output\READ ME.txt") -Encoding utf8 -Value @'
Converted images are saved here by default.
'@

Set-Content -Path (Join-Path $Release "pytorch\READ ME.txt") -Encoding utf8 -Value @'
PyTorch is downloaded here on first launch, about 1.8 GB.

It is not bundled with the release on purpose: it is by far the largest thing
the app needs, and shipping it would put 2.7 GB into every copy. It is freely
redistributable, so this is a size decision rather than a licensing one.

Deleting this folder only costs you the download. Rebuilding the app does not
touch it.
'@

# Belt and braces: nothing in the NVIDIA runtime may ever end up in the part of
# the release we actually built. This is what makes "bring your own files" a
# property of the build rather than something to remember.
#
# Filtered with Where-Object rather than -Include: -Include is silently ignored
# alongside -LiteralPath and matches every file instead of none, so the check
# would "fail" on the whole release and teach you to ignore it.
#
# dlss_files\ and engine\ are excluded on purpose. They are runtime state, not
# build output: the user is *told* to put their binaries in the first, and the
# app stages copies into the second on first run. Failing the build over them
# would mean nobody who has actually run the app could ever rebuild it.
function Test-Contraband {
    param([string]$Root)
    if (-not (Test-Path -LiteralPath $Root)) { return @() }
    Get-ChildItem -LiteralPath $Root -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -like "nvngx_*.dll" -or $_.Name -like "*.addon64" -or $_.Name -eq "dxgi.dll"
        }
}

$Shipped = @(Get-ChildItem -LiteralPath $Release -File -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -like "nvngx_*.dll" -or $_.Name -like "*.addon64" -or $_.Name -eq "dxgi.dll"
    })
$Shipped += @(Test-Contraband (Join-Path $Release "_internal"))
if ($Shipped.Count -gt 0) {
    Write-Host ""
    Write-Host "REFUSING TO FINISH: NVIDIA/ReShade runtime files are inside the built app:" -ForegroundColor Red
    $Shipped | ForEach-Object { Write-Host "  $($_.FullName)" -ForegroundColor Red }
    throw "These are not ours to distribute. Remove them before sharing."
}

# Present-but-legitimate: yours to use, never yours to send.
$Local = @(Test-Contraband (Join-Path $Release "dlss_files")) +
         @(Test-Contraband $Engine)
if ($Local.Count -gt 0) {
    Write-Host ""
    Write-Host "NOTE: $($Local.Count) NVIDIA/ReShade file(s) are in dlss_files\ and engine\." -ForegroundColor Yellow
    Write-Host "      Fine for running it here. Delete both folders' contents before" -ForegroundColor Yellow
    Write-Host "      zipping this release for anyone else." -ForegroundColor Yellow
}

# Reported separately, because only the first number is what gets shared. The
# other three are the user's own data and downloads, and rolling them into one
# total makes a 3 GB app look like a 5 GB one.
function Measure-Tree {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return 0 }
    $sum = (Get-ChildItem -LiteralPath $Path -Recurse -File -ErrorAction SilentlyContinue |
        Measure-Object -Property Length -Sum).Sum
    if ($null -eq $sum) { return 0 }
    return $sum
}

$AppSize = (Get-ChildItem -LiteralPath $Release -File | Measure-Object -Property Length -Sum).Sum
$AppSize += Measure-Tree (Join-Path $Release "_internal")
$AppSize += Measure-Tree $Engine

Write-Host ""
Write-Host ("Built {0}" -f (Join-Path $Release "DLSS5Converter.exe")) -ForegroundColor Green
Write-Host ("Application: {0:N1} GB   <- this is what you share" -f ($AppSize / 1GB))
foreach ($Folder in $UserFolders) {
    $Bytes = Measure-Tree (Join-Path $Release $Folder)
    if ($Bytes -gt 0) {
        Write-Host ("  {0,-12} {1,7:N1} GB   (yours, kept across rebuilds)" -f $Folder, ($Bytes / 1GB))
    }
}
Write-Host ""
Write-Host "release\"
Write-Host "  DLSS5Converter.exe"
Write-Host "  dlss_files\   <- the user drops their own DLSS 5 binaries here"
Write-Host "  models\       <- depth weights, downloaded on first run"
Write-Host "  pytorch\      <- PyTorch, downloaded on first run"
Write-Host "  output\       <- converted images"
Write-Host "  engine\       <- dlss5_eval.exe"
