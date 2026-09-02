# Build the portable zip that goes on the GitHub release.
#
# Separate from build_release.ps1 on purpose. That script builds a release
# folder you can actually run, which means it leaves your own NVIDIA and
# ReShade files staged in dlss_files\ and engine\. Those must never leave this
# machine, so packaging is its own step with its own refusal.
#
# Staged from a clean copy rather than by deleting out of release\, so a
# mistake here cannot destroy the working install you just tested.
#
# Pure ASCII. An em-dash in a .ps1 has broken the parser here before.

param(
    [Parameter(Mandatory = $true)][string] $Version,
    [string] $OutputDir
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Release = Join-Path $ProjectRoot "release"

if (-not (Test-Path -LiteralPath (Join-Path $Release "DLSS5Converter.exe"))) {
    throw "No release build found. Run .\scripts\build_release.ps1 first."
}
if (-not $OutputDir) { $OutputDir = Join-Path $ProjectRoot "dist" }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Anything matching these never goes in the zip, whatever folder it turns up
# in. nvngx_dlssnr.dll is a leaked pre-release NVIDIA binary; the others are
# not ours to redistribute either.
$Contraband = @("nvngx_dlssnr.dll", "nvngx_dlss.dll", "renodx-dlss5.addon64", "dxgi.dll",
                "ReShade64.dll", "ReShade.ini", "ReShade.log")

$Name = "DLSS5-Image-Converter"
$Staging = Join-Path ([System.IO.Path]::GetTempPath()) ("dlss5-package-" + [guid]::NewGuid().ToString("N"))
$Root = Join-Path $Staging $Name
New-Item -ItemType Directory -Force -Path $Root | Out-Null

Write-Host "Staging the portable copy..." -ForegroundColor Cyan

Copy-Item -LiteralPath (Join-Path $Release "DLSS5Converter.exe") -Destination $Root
Copy-Item -LiteralPath (Join-Path $Release "_internal") -Destination $Root -Recurse

# The harness, and only the harness. Everything else in engine\ is staged
# there at runtime from the user's own dlss_files.
New-Item -ItemType Directory -Force -Path (Join-Path $Root "engine") | Out-Null
Copy-Item -LiteralPath (Join-Path $Release "engine\dlss5_eval.exe") `
          -Destination (Join-Path $Root "engine")

# The four folders the app expects, each carrying only its placeholder. These
# are what tell a new user where their own files go.
foreach ($folder in @("dlss_files", "models", "output", "pytorch")) {
    $target = Join-Path $Root $folder
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    Get-ChildItem -LiteralPath (Join-Path $Release $folder) -Filter "READ*.txt" -File |
        ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $target }
}

Copy-Item -LiteralPath (Join-Path $ProjectRoot "LICENSE") -Destination (Join-Path $Root "LICENSE.txt")
Copy-Item -LiteralPath (Join-Path $ProjectRoot "TROUBLESHOOTING.md") `
          -Destination (Join-Path $Root "TROUBLESHOOTING.txt")

# The refusal. Checked against what is actually staged rather than against
# what was intended to be staged, because those are different things and only
# one of them is about to be uploaded.
$found = Get-ChildItem -LiteralPath $Root -Recurse -File |
         Where-Object { $Contraband -contains $_.Name }
if ($found) {
    Remove-Item -LiteralPath $Staging -Recurse -Force
    $found | ForEach-Object { Write-Host "  $($_.FullName)" -ForegroundColor Red }
    throw "Refusing to package: files that must not be redistributed were staged."
}

$Zip = Join-Path $OutputDir "$Name-v$Version-portable.zip"
if (Test-Path -LiteralPath $Zip) { Remove-Item -LiteralPath $Zip -Force }
Write-Host "Compressing..." -ForegroundColor Cyan
Compress-Archive -Path $Root -DestinationPath $Zip -CompressionLevel Optimal

Remove-Item -LiteralPath $Staging -Recurse -Force

$size = (Get-Item -LiteralPath $Zip).Length / 1GB
Write-Host ""
Write-Host ("Wrote {0}" -f $Zip) -ForegroundColor Green
Write-Host ("{0:N2} GB" -f $size)
