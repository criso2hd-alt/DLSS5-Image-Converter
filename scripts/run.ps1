$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

# Prefer the CUDA environment when it exists: depth estimation on CPU turns a
# two-second step into most of a minute.
$Python = Join-Path $ProjectRoot ".venv-cuda\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Dependencies are not installed. Run .\scripts\setup.ps1 first."
}

Set-Location -LiteralPath $ProjectRoot
& $Python main.py
