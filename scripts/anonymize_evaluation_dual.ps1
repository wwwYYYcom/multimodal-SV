param(
    [string]$PythonExe = 'D:\codeAPP\anaconda3\envs\pytorch\python.exe',
    [int]$SplitIndex = 44950,
    [int]$MonitorSeconds = 60
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$plan = 'artifacts/anonymization/evaluation_plan.csv'
$finalManifest = 'artifacts/metadata/fisher_anonymized_evaluation_manifest.csv'
$streamVoiceRoot = 'third_party/StreamVoiceAnon'
$sharedRunDir = 'results/runs/anonymization_evaluation'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$runDir = Join-Path $sharedRunDir "dual_$stamp"
$fullCount = 86222
$worker1Count = $SplitIndex
$worker2Count = $fullCount - $SplitIndex
New-Item -ItemType Directory -Path $runDir -Force | Out-Null

$worker1Manifest = Join-Path $runDir 'worker1.manifest.csv'
$worker2Manifest = Join-Path $runDir 'worker2.manifest.csv'
$worker1Out = Join-Path $runDir 'worker1.stdout.log'
$worker1Err = Join-Path $runDir 'worker1.stderr.log'
$worker2Out = Join-Path $runDir 'worker2.stdout.log'
$worker2Err = Join-Path $runDir 'worker2.stderr.log'
$common = @(
    '-u', '-m', 'mmsv.cli', 'anonymize-streamvoice',
    '--plan', $plan,
    '--streamvoice-root', $streamVoiceRoot,
    '--delay', '2',
    '--alpha', '1.0'
)

Write-Output "dual_supervisor_started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Write-Output "run_dir=$((Resolve-Path -LiteralPath $runDir).Path)"
Write-Output "split_index=$SplitIndex"
Write-Output "worker1_count=$worker1Count"
Write-Output "worker2_count=$worker2Count"
Write-Output 'precision=fp32_weights_with_upstream_cuda_autocast'
$started = Get-Date
$worker1 = Start-Process -FilePath $PythonExe `
    -ArgumentList ($common + @('--output-manifest', $worker1Manifest, '--limit', "$worker1Count")) `
    -WorkingDirectory $projectRoot -RedirectStandardOutput $worker1Out -RedirectStandardError $worker1Err `
    -WindowStyle Hidden -PassThru
$worker2 = Start-Process -FilePath $PythonExe `
    -ArgumentList ($common + @('--output-manifest', $worker2Manifest, '--start-index', "$SplitIndex")) `
    -WorkingDirectory $projectRoot -RedirectStandardOutput $worker2Out -RedirectStandardError $worker2Err `
    -WindowStyle Hidden -PassThru
Write-Output "worker1_pid=$($worker1.Id)"
Write-Output "worker2_pid=$($worker2.Id)"

$peakGpuMemoryMiB = 0
$nextReport = Get-Date
while (-not ($worker1.HasExited -and $worker2.HasExited)) {
    $memoryText = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    $memoryMiB = [int](($memoryText | Select-Object -First 1).Trim())
    if ($memoryMiB -gt $peakGpuMemoryMiB) {
        $peakGpuMemoryMiB = $memoryMiB
    }
    if ((Get-Date) -ge $nextReport) {
        $progress1 = [System.IO.Path]::ChangeExtension($worker1Manifest, '.progress.jsonl')
        $progress2 = [System.IO.Path]::ChangeExtension($worker2Manifest, '.progress.jsonl')
        $count1 = if (Test-Path -LiteralPath $progress1) { (Get-Content -LiteralPath $progress1).Count } else { 0 }
        $count2 = if (Test-Path -LiteralPath $progress2) { (Get-Content -LiteralPath $progress2).Count } else { 0 }
        Write-Output "progress_time=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') worker1=$count1/$worker1Count worker2=$count2/$worker2Count gpu_memory_mib=$memoryMiB"
        $nextReport = (Get-Date).AddSeconds($MonitorSeconds)
    }
    Start-Sleep -Seconds 5
    $worker1.Refresh()
    $worker2.Refresh()
}
$worker1.WaitForExit()
$worker2.WaitForExit()
$wallSeconds = ((Get-Date) - $started).TotalSeconds

$worker1Audit = [System.IO.Path]::ChangeExtension($worker1Manifest, '.audit.json')
$worker2Audit = [System.IO.Path]::ChangeExtension($worker2Manifest, '.audit.json')
if (-not (Test-Path -LiteralPath $worker1Audit) -or -not (Test-Path -LiteralPath $worker2Audit)) {
    throw 'At least one anonymization worker did not produce a final audit; rerun is safe and resumes existing FLAC files'
}
$audit1 = Get-Content -LiteralPath $worker1Audit -Raw | ConvertFrom-Json
$audit2 = Get-Content -LiteralPath $worker2Audit -Raw | ConvertFrom-Json
if ($audit1.processed -ne $worker1Count -or $audit2.processed -ne $worker2Count) {
    throw "Worker audit counts are incomplete: worker1=$($audit1.processed), worker2=$($audit2.processed)"
}

& $PythonExe scripts/merge_anonymization_manifests.py `
    --plan $plan `
    --manifests $worker1Manifest $worker2Manifest `
    --output $finalManifest
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to merge anonymization worker manifests'
}
& $PythonExe scripts/validate_anonymization_outputs.py `
    --plan $plan `
    --manifest $finalManifest `
    --expected $fullCount `
    --finite-check-limit 100 `
    --wall-seconds $wallSeconds `
    --output (Join-Path $sharedRunDir 'final.validation.json')
if ($LASTEXITCODE -ne 0) {
    throw 'Full dual-process evaluation anonymization validation failed'
}
Write-Output "wall_seconds=$wallSeconds"
Write-Output "peak_gpu_memory_mib=$peakGpuMemoryMiB"
Write-Output "dual_supervisor_completed=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
