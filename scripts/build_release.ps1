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

# Absolute so --add-data resolves against the project, not the --specpath dir
# (PyInstaller resolves a relative data source relative to the spec folder,
# which is build\, where dlss5_converter\assets does not exist).
$AssetsSrc = Join-Path $ProjectRoot "dlss5_converter\assets"

# Folders that belong to the user, not to the build. A rebuild must never take
# out a 400 MB model download or a folder of converted images, which is the
# hazard that made CLAUDE.md keep weights out of the app directory in the first
# place. Keeping them across rebuilds is what buys back that guarantee.
# No "pytorch" folder any more - depth is ONNX Runtime and the model is bundled,
# so nothing large is downloaded on first run.
$UserFolders = @("dlss_files", "models", "output")

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

Write-Host "Freezing the application (ONNX Runtime; depth model bundled, no downloads)." -ForegroundColor Cyan

# PyInstaller writes its whole progress log to stderr, and under
# $ErrorActionPreference = "Stop" Windows PowerShell treats a native command's
# stderr as a terminating error - the build would abort on its own banner. Drop
# to Continue for the call and judge it by $LASTEXITCODE, which is the only
# honest signal here anyway.
$Previous = $ErrorActionPreference
$ErrorActionPreference = "Continue"

# Codex desktop adds its document/PDF runtime (including Poppler's private ICU
# build) to PATH. PyInstaller follows PATH while resolving Qt6Core's dependency
# on Windows' system ICU shim and can therefore bundle Poppler's incompatible
# `icuuc.dll` instead. The names match, the exports do not, and the frozen app
# then fails at import with "specified procedure could not be found". External
# Codex runtimes are build-host tooling, never application dependencies.
$PreviousPath = $env:Path
$env:Path = (($env:Path -split ";") | Where-Object {
    $_ -and $_ -notmatch "[\\/]\.cache[\\/]codex-runtimes[\\/]"
}) -join ";"

# Bundle the Apache-2.0 Small ONNX depth model. Exported from the Depth Anything
# V2 weights (needs the `export` extra: torch + transformers, dev-only) into
# assets\onnx, so the --add-data of the assets tree carries it into the release.
# Reused if already present, so a normal rebuild does not re-export.
$OnnxDir = Join-Path $AssetsSrc "onnx"
$SmallOnnx = Join-Path $OnnxDir "Depth-Anything-V2-Small-hf.onnx"
if (-not (Test-Path -LiteralPath $SmallOnnx)) {
    Write-Host "Exporting the Small ONNX depth model (one-time)..." -ForegroundColor Cyan
    & $Python (Join-Path $ProjectRoot "scripts\export_onnx.py") --model small --out $OnnxDir
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $SmallOnnx)) {
        throw "Could not export the Small ONNX depth model. Install the export extra (pip install -e .[export]) and retry."
    }
}

# Depth runs on ONNX Runtime now, not PyTorch (see dlss5_converter/onnx_depth.py),
# which is what removed the old 2.7 GB torch download. So torch and the whole HF
# transformers stack are EXCLUDED from the bundle - they are dev-only tooling for
# scripts/export_onnx.py now. onnxruntime is collected whole so its execution
# provider DLLs (CUDA/TensorRT + CPU) ship with the app.
#
# CUDA runtime note: onnxruntime-gpu needs the CUDA + cuDNN runtime DLLs at run
# time. `--collect-all onnxruntime` bundles onnxruntime's own DLLs; when those
# runtime DLLs come from the nvidia-*-cu12 pip packages they must be collected
# too (add --collect-all nvidia_cudnn / nvidia_cublas / nvidia_cuda_runtime in
# the build venv that has them). This build must be made in a venv with
# onnxruntime-gpu installed, and verified on a real RTX GPU - the CUDA execution
# provider cannot be exercised on the CI/build box alone.
#
# The bundled ONNX depth model rides inside assets\ (see the export step above),
# so the existing --add-data of the assets tree carries it into the release.
& $Python -m PyInstaller `
    --noconfirm `
    --windowed `
    --name DLSS5Converter `
    --distpath $Staging `
    --workpath (Join-Path $ProjectRoot "build\pyinstaller-work") `
    --specpath (Join-Path $ProjectRoot "build") `
    --exclude-module torch `
    --exclude-module transformers `
    --exclude-module tokenizers `
    --exclude-module safetensors `
    --exclude-module av `
    --exclude-module tkinter `
    --exclude-module matplotlib `
    --exclude-module pytest `
    --collect-all onnxruntime `
    --hidden-import PySide6.QtMultimedia `
    --hidden-import PySide6.QtMultimediaWidgets `
    --collect-all PySide6.QtMultimedia `
    --collect-all PySide6.QtMultimediaWidgets `
    --copy-metadata numpy `
    --copy-metadata packaging `
    --copy-metadata requests `
    --copy-metadata filelock `
    --copy-metadata huggingface-hub `
    --add-data "$AssetsSrc;dlss5_converter/assets" `
    main.py
$FrozenExit = $LASTEXITCODE
$env:Path = $PreviousPath
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

# PyInstaller brings Python's older VC runtime into _internal, while current
# PySide6 ships the newer runtime it was linked against inside PySide6\. Windows
# loads the root copy first, so QtCore then fails with "specified procedure could
# not be found" even though every Qt DLL is present. Put PySide's matching pair
# at the shared search root; they are backward-compatible with Python and keep
# the frozen app self-contained on machines without the latest VC redistributable.
$Internal = Join-Path $Release "_internal"
$PySideInternal = Join-Path $Internal "PySide6"
foreach ($RuntimeDll in @("VCRUNTIME140.dll", "VCRUNTIME140_1.dll")) {
    $MatchingRuntime = Join-Path $PySideInternal $RuntimeDll
    if (-not (Test-Path -LiteralPath $MatchingRuntime)) {
        throw "PySide6 did not supply $RuntimeDll; cannot guarantee a compatible Qt runtime."
    }
    Copy-Item -LiteralPath $MatchingRuntime -Destination (Join-Path $Internal $RuntimeDll) -Force
}

# Qt on Windows intentionally resolves `icuuc.dll` from System32. A copy at the
# application root can only shadow that contract; in our build environment it
# means Poppler leaked in from PATH. Fail before presenting a broken release.
$ForeignIcu = Join-Path $Internal "icuuc.dll"
if (Test-Path -LiteralPath $ForeignIcu) {
    throw "A foreign icuuc.dll was bundled at $ForeignIcu. Check PATH contamination."
}

foreach ($Folder in $UserFolders) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Release $Folder) | Out-Null
}
New-Item -ItemType Directory -Force -Path $Engine | Out-Null
Copy-Item -LiteralPath $Harness -Destination $Engine -Force

# The one thing a new user has to do, written where they will look for it.
$Readme = @'
Put your own DLSS 5 files in this folder.

THIS IS THE ONLY FOLDER YOU TOUCH. You do not copy anything into engine\.
The app copies what it needs from here into engine\ (next to dlss5_eval.exe)
automatically on first run, because that is where NVIDIA's NGX and ReShade load
their DLLs from. On the same drive the copy is a hard link, so it costs no extra
space. Never put dlss5_eval.exe in this folder - it ships in engine\ and stays
there. Anyone telling you to place files in both folders by hand is mistaken.

None of these ship with the app and it will not help you obtain them.

EASIEST WAY, if DLSS 5 already works in a game for you:
copy all four files out of that game's folder. They sit next to the game
executable, usually in bin\x64. A set that is already running on your card
is a set the add-on has accepted on your GPU and driver, which saves you
guessing about versions - and keeping them together matters, because mixing
a runtime from one source with an add-on from another is a common way to
get "NR is unavailable in this session".

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

You do not need to tidy any of this up:

  - a Streamline folder can stay unflattened; the app looks inside
    NVStreamline\Production too
  - a .zip can stay zipped. If nvngx_dlss.dll is missing the app opens any
    zip in this folder, pulls out what it needs, and leaves the rest alone.
    It never overwrites a file you put here yourself.

The app copies these next to engine\dlss5_eval.exe on first run, because that
is where NGX and ReShade look. Use Diagnose in the app to check what it found.
'@
Set-Content -Path (Join-Path $Release "dlss_files\READ ME FIRST.txt") -Value $Readme -Encoding utf8

Set-Content -Path (Join-Path $Release "models\READ ME.txt") -Encoding utf8 -Value @'
The default depth model (Depth Anything V2 Small, ONNX) ships inside the app, so
it works out of the box with no download. Larger models (Base/Large) would be
downloaded here if selected.

Nothing here is required for the app to run. Rebuilding the app does not touch it.
'@

Set-Content -Path (Join-Path $Release "output\READ ME.txt") -Encoding utf8 -Value @'
Converted images are saved here by default.
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
Write-Host "  models\       <- optional larger depth models (Small ships bundled)"
Write-Host "  output\       <- converted images"
Write-Host "  engine\       <- dlss5_eval.exe"
