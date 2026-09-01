$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$wslScript = $PSScriptRoot -replace '\\', '/'
$drive = $wslScript.Substring(0, 1).ToLowerInvariant()
$wslScript = '/mnt/' + $drive + $wslScript.Substring(2) + '/setup_pymeep_wsl.sh'

Write-Host 'Checking Ubuntu WSL...'
& wsl.exe -d Ubuntu -u root -- bash -lc 'uname -a && cat /etc/os-release | head -n 6'
if ($LASTEXITCODE -ne 0) {
    throw "Ubuntu WSL did not start (exit $LASTEXITCODE)"
}

Write-Host 'Installing the pinned PyMeep environment and running its smoke test...'
& wsl.exe -d Ubuntu -u root -- bash $wslScript
if ($LASTEXITCODE -ne 0) {
    throw "PyMeep setup failed (exit $LASTEXITCODE)"
}

Get-Content -LiteralPath (Join-Path $root 'benchmark_results\pymeep_wsl_environment.json') -Raw -Encoding utf8
