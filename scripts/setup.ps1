param(
    # The accelerated environment is built in its own venv rather than modifying
    # .venv, so both flavours stay reproducible side by side.
    [switch]$Cuda,
    [string]$CudaTag = "cu130"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

if ($Cuda) { $VenvName = ".venv-cuda" } else { $VenvName = ".venv" }
$VenvPython = "$VenvName\Scripts\python.exe"

# A missing interpreter makes the launcher write to stderr, which PowerShell
# would otherwise promote to a terminating error under $ErrorActionPreference.
function Test-Interpreter {
    param([string[]]$Command)
    $Probe = "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)"
    try {
        $Previous = $ErrorActionPreference
        $ErrorActionPreference = "SilentlyContinue"
        & $Command[0] @($Command[1..($Command.Length - 1)]) -c $Probe 2>&1 | Out-Null
        $ErrorActionPreference = $Previous
        return $LASTEXITCODE -eq 0
    }
    catch {
        $ErrorActionPreference = $Previous
        return $false
    }
}

$PythonCommand = $null
if (Test-Interpreter @("py", "-3.12")) { $PythonCommand = @("py", "-3.12") }
elseif (Test-Interpreter @("python")) { $PythonCommand = @("python") }

if ($null -eq $PythonCommand) {
    throw "This app requires 64-bit Python 3.12. Install it from python.org, then run this script again."
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    & $PythonCommand[0] @($PythonCommand[1..($PythonCommand.Length - 1)]) -m venv $VenvName
}

& $VenvPython -m pip install --upgrade pip wheel setuptools
& $VenvPython -m pip install -e ".[test]"

if ($Cuda) {
    # Installed after the project so the accelerated wheel replaces the CPU
    # torch the resolver pulls from PyPI. The retry/timeout flags matter: these
    # wheels are ~2 GB and the stream stalls behind antivirus HTTPS inspection,
    # which pip would otherwise wait on forever.
    Write-Host ""
    Write-Host "Installing accelerated PyTorch ($CudaTag). This download is ~2 GB." -ForegroundColor Yellow
    & $VenvPython -m pip install --upgrade --force-reinstall `
        --retries 20 --timeout 60 `
        --index-url "https://download.pytorch.org/whl/$CudaTag" torch
    if ($LASTEXITCODE -ne 0) {
        throw "Accelerated PyTorch install failed. Re-run this script; pip resumes from its cache."
    }
    & $VenvPython -c "from dlss5_converter.depth_engine import select_device, device_label; import torch; d = select_device(torch); print(f'torch {torch.__version__} -> device {d} ({device_label(torch, d)})')"
}

Write-Host ""
Write-Host "Python side ready." -ForegroundColor Green
Write-Host "Next: .\scripts\build_native.ps1   (builds the DLSS harness)"
Write-Host "Then: .\scripts\run.ps1"
