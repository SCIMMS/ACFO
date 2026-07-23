[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OriginalRunDir,
    [string]$ReleaseRoot = "",
    [string]$OutputDir = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
if (-not $ReleaseRoot) {
    $ReleaseRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
else {
    $ReleaseRoot = (Resolve-Path -LiteralPath $ReleaseRoot).Path
}
$VenvPython = Join-Path $ReleaseRoot ".venv\Scripts\python.exe"
$Runner = Join-Path $PSScriptRoot "run_external_acfo_ncs_v14_waxs_detector_only.py"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Release virtual environment was not found: $VenvPython"
}
if (-not $env:CUPY_CACHE_DIR) {
    $env:CUPY_CACHE_DIR = Join-Path $ReleaseRoot ".cupy_cache"
}

$Arguments = @(
    $Runner,
    "--release-root", $ReleaseRoot,
    "--original-run-dir", $OriginalRunDir,
    "--python", $VenvPython
)
if ($OutputDir) { $Arguments += @("--output-dir", $OutputDir) }
if ($DryRun) { $Arguments += "--dry-run" }

& $VenvPython @Arguments
exit $LASTEXITCODE
