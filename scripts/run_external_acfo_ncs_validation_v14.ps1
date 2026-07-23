[CmdletBinding()]
param(
    [ValidateSet("quick", "full")]
    [string]$Mode = "full",
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$RunDir = "",
    [switch]$Resume,
    [switch]$AllowReferenceMachine,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $Root

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}

$Arguments = @(
    "scripts\run_external_acfo_ncs_validation_v14.py",
    "--mode", $Mode,
    "--python", $Python
)
if ($RunDir) { $Arguments += @("--run-dir", $RunDir) }
if ($Resume) { $Arguments += "--resume" }
if ($AllowReferenceMachine) { $Arguments += "--allow-reference-machine" }
if ($DryRun) { $Arguments += "--dry-run" }

& $Python @Arguments
exit $LASTEXITCODE
