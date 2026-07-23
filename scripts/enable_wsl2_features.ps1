$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$log = Join-Path $root 'benchmark_results\wsl2_feature_enable.log'
$state = Join-Path $root 'benchmark_results\wsl2_feature_state.json'

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $log) | Out-Null
"WSL2 feature enable started: $(Get-Date -Format o)" | Set-Content -LiteralPath $log -Encoding utf8

foreach ($feature in @('Microsoft-Windows-Subsystem-Linux', 'VirtualMachinePlatform')) {
    "Enabling $feature" | Add-Content -LiteralPath $log -Encoding utf8
    & dism.exe /online /enable-feature "/featurename:$feature" /all /norestart 2>&1 |
        Add-Content -LiteralPath $log -Encoding utf8
    if ($LASTEXITCODE -notin @(0, 3010)) {
        throw "DISM failed for $feature with exit code $LASTEXITCODE"
    }
}

$features = foreach ($feature in @('Microsoft-Windows-Subsystem-Linux', 'VirtualMachinePlatform')) {
    $item = Get-WindowsOptionalFeature -Online -FeatureName $feature
    [ordered]@{
        feature = $item.FeatureName
        state = [string]$item.State
        restart_required = [string]$item.RestartRequired
    }
}

[ordered]@{
    generated_at = (Get-Date -Format o)
    features = @($features)
    reboot_was_not_issued = $true
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $state -Encoding utf8

"WSL2 feature enable completed: $(Get-Date -Format o)" | Add-Content -LiteralPath $log -Encoding utf8
