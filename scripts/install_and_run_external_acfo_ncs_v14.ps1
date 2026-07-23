[CmdletBinding()]
param(
    [ValidateSet("quick", "full")]
    [string]$Mode = "full",
    [string]$PythonTag = "auto",
    [string]$TorchVersion = "2.12.1",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu126",
    [string]$CuPyVersion = "14.1.1",
    [string]$RunDir = "",
    [switch]$Resume,
    [switch]$AllowReferenceMachine,
    [switch]$DryRun,
    [switch]$RefreshEnvironment
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $Root
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not $env:CUPY_CACHE_DIR) {
    $env:CUPY_CACHE_DIR = Join-Path $Root ".cupy_cache"
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "Python launcher 'py' was not found. Install Python 3.11, 3.12, or 3.13."
    }
    $ResolvedPythonTag = $PythonTag
    if ($PythonTag -eq "auto") {
        $ResolvedPythonTag = $null
        foreach ($Candidate in @("3.12", "3.11", "3.13")) {
            & py "-$Candidate" -c "import sys; raise SystemExit(0)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                $ResolvedPythonTag = $Candidate
                break
            }
        }
        if (-not $ResolvedPythonTag) {
            throw "No supported Python runtime found. Install Python 3.11, 3.12, or 3.13."
        }
    }
    Write-Host "[setup 1/3] Creating Python $ResolvedPythonTag virtual environment"
    & py "-$ResolvedPythonTag" -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed: $LASTEXITCODE" }
    $RefreshEnvironment = $true
}

if ($RefreshEnvironment -or (-not $Resume)) {
    Write-Host "[setup 2/3] Installing frozen release extras"
    & $VenvPython -m pip install -U pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed: $LASTEXITCODE" }
    Write-Host "[setup 2/3] Installing CUDA-enabled PyTorch $TorchVersion from $TorchIndexUrl"
    & $VenvPython -m pip install --upgrade "torch==$TorchVersion" --index-url $TorchIndexUrl
    if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch install failed: $LASTEXITCODE" }
    Write-Host "[setup 2/3] Installing CuPy $CuPyVersion with CUDA toolkit component wheels"
    & $VenvPython -m pip install --upgrade "cupy-cuda12x[ctk]==$CuPyVersion"
    if ($LASTEXITCODE -ne 0) { throw "CuPy CUDA toolkit component install failed: $LASTEXITCODE" }
    & $VenvPython -m pip install -e ".[reproduce,gpu-baseline]"
    if ($LASTEXITCODE -ne 0) { throw "release dependency install failed: $LASTEXITCODE" }
    & $VenvPython -c "import torch; print(f'torch={torch.__version__} built_cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}'); raise SystemExit(0 if torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA-enabled PyTorch verification failed. Check nvidia-smi and the selected Python/Torch wheel."
    }
    & $VenvPython -c "import cupy as cp; x=cp.arange(4, dtype=cp.float32); y=x*x; cp.cuda.runtime.deviceSynchronize(); print(f'cupy={cp.__version__} cuda_runtime={cp.cuda.runtime.runtimeGetVersion()} jit_probe={y.get().tolist()}')"
    if ($LASTEXITCODE -ne 0) {
        throw "CuPy CUDA JIT verification failed. CUDA toolkit headers may be missing."
    }
}
else {
    Write-Host "[setup 2/3] Reusing the existing environment for resume"
}

Write-Host "[setup 3/3] Starting the recorded validation runner"
$RunnerArguments = @(
    "scripts\run_external_acfo_ncs_validation_v14.py",
    "--mode", $Mode,
    "--python", $VenvPython
)
if ($RunDir) { $RunnerArguments += @("--run-dir", $RunDir) }
if ($Resume) { $RunnerArguments += "--resume" }
if ($RefreshEnvironment) { $RunnerArguments += "--refresh-receipts" }
if ($AllowReferenceMachine) { $RunnerArguments += "--allow-reference-machine" }
if ($DryRun) { $RunnerArguments += "--dry-run" }

& $VenvPython @RunnerArguments
exit $LASTEXITCODE
