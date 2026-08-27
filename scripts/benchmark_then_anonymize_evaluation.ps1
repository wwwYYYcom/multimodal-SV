param(
    [string]$PythonExe = 'D:\codeAPP\anaconda3\envs\pytorch\python.exe',
    [int]$BenchmarkCount = 100,
    [double]$MaximumDiskFraction = 0.80
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$plan = 'artifacts/anonymization/evaluation_plan.csv'
$manifest = 'artifacts/metadata/fisher_anonymized_evaluation_manifest.csv'
$streamVoiceRoot = 'third_party/StreamVoiceAnon'
$runDir = 'results/runs/anonymization_evaluation'
$fullCount = 86222
$fullSourceHours = 94.14482750000104
New-Item -ItemType Directory -Path $runDir -Force | Out-Null

Write-Output "supervisor_started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Write-Output "benchmark_count=$BenchmarkCount"
$benchmarkStarted = Get-Date
& $PythonExe -u -m mmsv.cli anonymize-streamvoice `
    --plan $plan `
    --output-manifest $manifest `
    --streamvoice-root $streamVoiceRoot `
    --delay 2 `
    --alpha 1.0 `
    --limit $BenchmarkCount
if ($LASTEXITCODE -ne 0) {
    throw "StreamVoiceAnon benchmark failed with exit code $LASTEXITCODE"
}
$benchmarkSeconds = ((Get-Date) - $benchmarkStarted).TotalSeconds

$manifestPath = Resolve-Path -LiteralPath $manifest
$auditPath = [System.IO.Path]::ChangeExtension($manifestPath.Path, '.audit.json')
$progressPath = [System.IO.Path]::ChangeExtension($manifestPath.Path, '.progress.jsonl')
Copy-Item -LiteralPath $manifestPath.Path -Destination (Join-Path $runDir 'benchmark.manifest.csv') -Force
Copy-Item -LiteralPath $auditPath -Destination (Join-Path $runDir 'benchmark.audit.json') -Force
Copy-Item -LiteralPath $progressPath -Destination (Join-Path $runDir 'benchmark.progress.jsonl') -Force

& $PythonExe scripts/validate_anonymization_outputs.py `
    --plan $plan `
    --manifest $manifest `
    --expected $BenchmarkCount `
    --wall-seconds $benchmarkSeconds `
    --full-plan-source-hours $fullSourceHours `
    --output (Join-Path $runDir 'benchmark.validation.json')
if ($LASTEXITCODE -ne 0) {
    throw 'Anonymization benchmark validation failed; full run was not started'
}
$validation = Get-Content -LiteralPath (Join-Path $runDir 'benchmark.validation.json') -Raw | ConvertFrom-Json
$freeBytes = (Get-PSDrive -Name D).Free
$safeBytes = [double]$freeBytes * $MaximumDiskFraction
Write-Output "benchmark_completed=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Write-Output "benchmark_seconds=$benchmarkSeconds"
Write-Output "projected_full_wall_hours=$($validation.projected_full_wall_hours)"
Write-Output "projected_full_bytes=$($validation.projected_full_bytes)"
Write-Output "free_bytes=$freeBytes"
if ([double]$validation.projected_full_bytes -gt $safeBytes) {
    throw "Projected output exceeds disk safety threshold: $($validation.projected_full_bytes) > $safeBytes"
}

Write-Output "full_run_started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
& $PythonExe -u -m mmsv.cli anonymize-streamvoice `
    --plan $plan `
    --output-manifest $manifest `
    --streamvoice-root $streamVoiceRoot `
    --delay 2 `
    --alpha 1.0
if ($LASTEXITCODE -ne 0) {
    throw "Full evaluation anonymization failed with exit code $LASTEXITCODE"
}

& $PythonExe scripts/validate_anonymization_outputs.py `
    --plan $plan `
    --manifest $manifest `
    --expected $fullCount `
    --finite-check-limit 100 `
    --output (Join-Path $runDir 'final.validation.json')
if ($LASTEXITCODE -ne 0) {
    throw 'Full evaluation anonymization validation failed'
}
Write-Output "full_run_completed=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
