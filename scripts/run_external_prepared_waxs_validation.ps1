[CmdletBinding()]
param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [switch]$Resume,
    [switch]$SkipReducedSuite
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $Root

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}

$ResultDir = Join-Path $Root "benchmark_results"
New-Item -ItemType Directory -Force -Path $ResultDir | Out-Null

$EnvironmentJson = Join-Path $ResultDir "external_prepared_waxs_machine_environment.json"
$ResultJson = Join-Path $ResultDir "external_protein_lattice_prepared_finufft_512_abba.json"
$ResultMarkdown = Join-Path $ResultDir "external_protein_lattice_prepared_finufft_512_abba.md"
$ValidationJson = Join-Path $ResultDir "external_prepared_waxs_abba_validation.json"
$ValidationMarkdown = Join-Path $ResultDir "external_prepared_waxs_abba_validation.md"
$PipFreeze = Join-Path $ResultDir "external_prepared_waxs_pip_freeze.txt"
$Transcript = Join-Path $ResultDir "external_prepared_waxs_transcript.txt"
$ReturnZip = Join-Path $ResultDir "external_prepared_waxs_return_package.zip"
$ValidationExitCode = 0

Start-Transcript -LiteralPath $Transcript -Force | Out-Null
try {
    Write-Host "[1/4] Collecting the independent-machine environment receipt"
    & $Python scripts\collect_prepared_waxs_machine_environment.py --output benchmark_results\external_prepared_waxs_machine_environment.json
    if ($LASTEXITCODE -ne 0) { throw "Environment receipt failed with exit code $LASTEXITCODE" }

    & $Python -m pip freeze | Out-File -LiteralPath $PipFreeze -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw "pip freeze failed with exit code $LASTEXITCODE" }

    if (-not $SkipReducedSuite) {
        Write-Host "[2/4] Running the reduced release suite"
        & $Python scripts\run_acfo_ncs_reduced_release_suite.py
        if ($LASTEXITCODE -ne 0) { throw "Reduced release suite failed with exit code $LASTEXITCODE" }
    }
    else {
        Write-Host "[2/4] Reduced release suite skipped by request"
    }

    Write-Host "[3/4] Running the prepared fused 1M WAXS 10/30 AB/BA benchmark"
    $BenchmarkArguments = @(
        "scripts\benchmark_protein_lattice_finufft_512_abba.py",
        "--finufft-threads", "4",
        "--factorized-backend", "prepared_fused",
        "--lattice-backend", "separable",
        "--output", "benchmark_results\external_protein_lattice_prepared_finufft_512_abba.json",
        "--summary-md", "benchmark_results\external_protein_lattice_prepared_finufft_512_abba.md"
    )
    if ($Resume) { $BenchmarkArguments += "--resume" }
    & $Python @BenchmarkArguments
    if ($LASTEXITCODE -ne 0) { throw "Prepared WAXS benchmark failed with exit code $LASTEXITCODE" }

    Write-Host "[4/4] Applying the independent-machine, source, accuracy, and timing gates"
    & $Python scripts\validate_external_prepared_waxs_abba.py `
        benchmark_results\external_protein_lattice_prepared_finufft_512_abba.json `
        benchmark_results\external_prepared_waxs_machine_environment.json `
        --output benchmark_results\external_prepared_waxs_abba_validation.json `
        --summary-md benchmark_results\external_prepared_waxs_abba_validation.md
    $ValidationExitCode = $LASTEXITCODE
    if ($ValidationExitCode -ne 0) {
        Write-Warning "External validation did not pass (exit code $ValidationExitCode); packaging the evidence for review."
    }
}
finally {
    Stop-Transcript | Out-Null
}

$ReturnFiles = @(
    $EnvironmentJson,
    $ResultJson,
    $ResultMarkdown,
    $ValidationJson,
    $ValidationMarkdown,
    $PipFreeze,
    $Transcript
)
$ReducedReceipt = Join-Path $ResultDir "acfo_ncs_reduced_release_suite.json"
if ((-not $SkipReducedSuite) -and (Test-Path -LiteralPath $ReducedReceipt)) {
    $ReturnFiles += $ReducedReceipt
}

foreach ($Path in $ReturnFiles) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Return file is missing: $Path" }
}
Compress-Archive -LiteralPath $ReturnFiles -DestinationPath $ReturnZip -Force
$ZipHash = (Get-FileHash -LiteralPath $ReturnZip -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ValidationExitCode -eq 0) {
    Write-Host "PASS: independent prepared-WAXS replication gates passed."
}
else {
    Write-Host "FAIL: one or more independent-replication gates did not pass. Return the package for review."
}
Write-Host "Return package: $ReturnZip"
Write-Host "SHA-256: $ZipHash"
exit $ValidationExitCode
