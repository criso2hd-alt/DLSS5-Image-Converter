param(
    # Re-clone the SDK even if it is already present.
    [switch]$Refresh
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Native = Join-Path $ProjectRoot "native"
$SdkRoot = Join-Path $Native "_sdk"
$Sdk = Join-Path $SdkRoot "DLSS"

if (-not (Get-Command "git" -ErrorAction SilentlyContinue)) {
    throw "git is required to build the harness. Install it and run this script again."
}

# Visual Studio bundles CMake but does not put it on PATH, and the VS installer
# is the likeliest way a Windows machine got a C++ toolchain at all. Preferring
# PATH but falling back to the bundled copy means "install VS with the C++
# workload" is sufficient, with no second install and no PATH surgery.
$CMake = (Get-Command "cmake" -ErrorAction SilentlyContinue).Source
if (-not $CMake) {
    $VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $VsWhere) {
        $VsRoot = & $VsWhere -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath 2>$null | Select-Object -First 1
        if ($VsRoot) {
            $Candidate = Join-Path $VsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
            if (Test-Path -LiteralPath $Candidate) { $CMake = $Candidate }
        }
    }
}
if (-not $CMake) {
    throw "cmake is required to build the harness. Install CMake, or Visual Studio with the 'Desktop development with C++' workload, and run this script again."
}

if ($Refresh -and (Test-Path -LiteralPath $Sdk)) {
    Remove-Item -LiteralPath $Sdk -Recurse -Force
}

if (-not (Test-Path -LiteralPath (Join-Path $Sdk "include\nvsdk_ngx.h"))) {
    # Public SDK. It supplies the NGX headers and the static helper library the
    # harness links; it does NOT contain the DLSS 5 neural model, which the user
    # supplies themselves and this project never distributes.
    New-Item -ItemType Directory -Force -Path $SdkRoot | Out-Null
    Write-Host "Cloning the NVIDIA DLSS SDK (headers + nvsdk_ngx_s.lib)..." -ForegroundColor Cyan
    # Blobless and sparse. The full repo is ~1 GB, nearly all of it the shipping
    # nvngx_dlss*.dll runtimes and sample assets we never touch: we link the
    # static helper and the user brings their own DLLs. This pulls ~85 MB.
    git clone --depth 1 --filter=blob:none --sparse https://github.com/NVIDIA/DLSS.git $Sdk
    if ($LASTEXITCODE -ne 0) { throw "Cloning the DLSS SDK failed." }
    git -C $Sdk sparse-checkout set include lib/Windows_x86_64/x64
    if ($LASTEXITCODE -ne 0) { throw "Sparse checkout of the DLSS SDK failed." }
}

$Build = Join-Path $Native "build"
& $CMake -S (Join-Path $Native "dlss5_eval") -B $Build -A x64 -DDLSS_SDK="$Sdk"
if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed." }

& $CMake --build $Build --config Release
if ($LASTEXITCODE -ne 0) { throw "Build failed." }

$Exe = Join-Path $Native "bin\dlss5_eval.exe"
if (-not (Test-Path -LiteralPath $Exe)) {
    throw "The build reported success but $Exe is missing."
}

Write-Host ""
Write-Host "Built $Exe" -ForegroundColor Green
Write-Host ""
Write-Host "ReShade is loaded as a proxy DLL and must sit beside the harness:" -ForegroundColor Yellow
Write-Host "  copy your ReShade64.dll to  native\bin\dxgi.dll"
Write-Host "The app copies nvngx_dlssnr.dll, nvngx_dlss.dll and the RenoDX add-on"
Write-Host "there for you on first run."
Write-Host ""
Write-Host "Check what the runtime can do:  .\native\bin\dlss5_eval.exe --probe"
