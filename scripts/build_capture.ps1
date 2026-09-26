param(
    # ReShade release whose add-on headers to build against. Must not be newer
    # than the ReShade players run: ReShade refuses add-ons built for a newer
    # API than its own.
    [string]$ReShadeTag = "v6.8.0",
    # The ImGui commit that ReShade release pins as a submodule. The overlay
    # header refuses any other version, so it moves with $ReShadeTag.
    [string]$ImGuiCommit = "3912b3d9a9c1b3f17431aebafd86d2f40ee6e59c",
    # Re-fetch the headers even if they are already present.
    [switch]$Refresh
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Native = Join-Path $ProjectRoot "native"
$Source = Join-Path $Native "scene_capture"
$Headers = Join-Path $Native "_sdk\reshade-$ReShadeTag"
$ImGui = Join-Path $Native "_sdk\imgui-$($ImGuiCommit.Substring(0, 7))"
$Build = Join-Path $Native "build\scene_capture"
$Out = Join-Path $Native "bin"

if (-not (Get-Command "git" -ErrorAction SilentlyContinue)) {
    throw "git is required to fetch the ReShade headers. Install it and run this script again."
}

# Same toolchain lookup as build_native.ps1: prefer cmake on PATH, else the copy
# Visual Studio bundles, so "install VS with the C++ workload" is enough.
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
    throw "cmake is required. Install CMake, or Visual Studio with the 'Desktop development with C++' workload."
}

if ($Refresh -and (Test-Path -LiteralPath $Headers)) {
    Remove-Item -LiteralPath $Headers -Recurse -Force
}
if (-not (Test-Path -LiteralPath (Join-Path $Headers "include\reshade.hpp"))) {
    Write-Host "Fetching ReShade $ReShadeTag add-on headers..."
    # Only include/ is needed, so a sparse, blob-filtered clone keeps this to a
    # few hundred KB instead of the whole ReShade history.
    git -c advice.detachedHead=false clone --quiet --depth 1 --branch $ReShadeTag --filter=blob:none --sparse `
        https://github.com/crosire/reshade.git $Headers
    if ($LASTEXITCODE -ne 0) { throw "Could not fetch ReShade $ReShadeTag." }
    git -C $Headers sparse-checkout set include
    if ($LASTEXITCODE -ne 0) { throw "Could not check out the ReShade headers." }
}

if (-not (Test-Path -LiteralPath (Join-Path $ImGui "imgui.h"))) {
    Write-Host "Fetching the matching ImGui header..."
    git init -q $ImGui
    git -C $ImGui remote add origin https://github.com/ocornut/imgui.git
    git -C $ImGui -c advice.detachedHead=false fetch -q --depth 1 --filter=blob:none origin $ImGuiCommit
    if ($LASTEXITCODE -ne 0) { throw "Could not fetch ImGui $ImGuiCommit." }
    git -C $ImGui sparse-checkout set --no-cone imgui.h imconfig.h
    git -C $ImGui -c advice.detachedHead=false checkout -q FETCH_HEAD
    if ($LASTEXITCODE -ne 0) { throw "Could not check out the ImGui header." }
}

& $CMake -S $Source -B $Build -A x64 "-DRESHADE_INCLUDE=$(Join-Path $Headers 'include')" "-DIMGUI_INCLUDE=$ImGui"
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed." }
& $CMake --build $Build --config Release
if ($LASTEXITCODE -ne 0) { throw "Build failed." }

New-Item -ItemType Directory -Force -Path $Out | Out-Null
Copy-Item -LiteralPath (Join-Path $Build "Release\dlss5_scene_capture.addon64") -Destination $Out -Force
Copy-Item -LiteralPath (Join-Path $Source "DLSS5Capture.fx") -Destination $Out -Force

Write-Host ""
Write-Host "Built $(Join-Path $Out 'dlss5_scene_capture.addon64')"
Write-Host "Install: copy dlss5_scene_capture.addon64 next to the game's ReShade dxgi.dll,"
Write-Host "and DLSS5Capture.fx into a folder listed in ReShade.ini EffectSearchPaths"
Write-Host "(by default .\ , the same folder; subfolders are only searched if the path ends in \**)."
