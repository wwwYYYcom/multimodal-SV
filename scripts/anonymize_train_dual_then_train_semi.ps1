param(
    [string]$PythonExe = 'D:\codeAPP\anaconda3\envs\pytorch\python.exe',
    [int]$SplitIndex = 3867,
    [int]$MonitorSeconds = 60
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$plan = 'artifacts/anonymization/train_one_per_call_side_plan.csv'
$finalManifest = 'artifacts/metadata/fisher_anonymized_train_corrected_manifest.csv'
$streamVoiceRoot = 'third_party/StreamVoiceAnon'
$sharedRunDir = 'results/runs/anonymization_train'
$semiRunDir = 'results/runs/audio_semi_corrected'
$lazyCheckpoint = 'results/runs/audio_corrected_p1/last.pt'
$semiConfig = 'configs/semi_local_corrected.yaml'
$splits = 'artifacts/metadata/speaker_splits.csv'
$trials = 'artifacts/trials/evaluation.jsonl'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$runDir = Join-Path $sharedRunDir "dual_$stamp"
$fullCount = 7272
$worker1Count = $SplitIndex
$worker2Count = $fullCount - $SplitIndex
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
New-Item -ItemType Directory -Path $runDir -Force | Out-Null
New-Item -ItemType Directory -Path $semiRunDir -Force | Out-Null

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

Write-Output "pipeline_started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Write-Output "run_dir=$((Resolve-Path -LiteralPath $runDir).Path)"
Write-Output "split_index=$SplitIndex"
Write-Output "worker1_count=$worker1Count"
Write-Output "worker2_count=$worker2Count"
Write-Output 'precision=fp32_weights_with_upstream_cuda_autocast'
Write-Output 'huggingface_offline=true'
$anonymizationStarted = Get-Date
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
        Write-Output "anonymization_progress=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') worker1=$count1/$worker1Count worker2=$count2/$worker2Count gpu_memory_mib=$memoryMiB"
        $nextReport = (Get-Date).AddSeconds($MonitorSeconds)
    }
    Start-Sleep -Seconds 5
    $worker1.Refresh()
    $worker2.Refresh()
}
$worker1.WaitForExit()
$worker2.WaitForExit()
$anonymizationWallSeconds = ((Get-Date) - $anonymizationStarted).TotalSeconds

$worker1Audit = [System.IO.Path]::ChangeExtension($worker1Manifest, '.audit.json')
$worker2Audit = [System.IO.Path]::ChangeExtension($worker2Manifest, '.audit.json')
if (-not (Test-Path -LiteralPath $worker1Audit) -or -not (Test-Path -LiteralPath $worker2Audit)) {
    throw 'At least one train anonymization worker did not produce a final audit; rerun safely resumes existing FLAC files'
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
    throw 'Failed to merge train anonymization worker manifests'
}
& $PythonExe scripts/validate_anonymization_outputs.py `
    --plan $plan `
    --manifest $finalManifest `
    --expected $fullCount `
    --finite-check-limit 100 `
    --wall-seconds $anonymizationWallSeconds `
    --output (Join-Path $sharedRunDir 'final.validation.json')
if ($LASTEXITCODE -ne 0) {
    throw 'Full dual-process train anonymization validation failed'
}
Write-Output "anonymization_wall_seconds=$anonymizationWallSeconds"
Write-Output "anonymization_peak_gpu_memory_mib=$peakGpuMemoryMiB"
Write-Output "anonymization_completed=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"

$semiCheckpoint = Join-Path $semiRunDir 'last.pt'
$trainOut = Join-Path $semiRunDir 'process.stdout.log'
$trainErr = Join-Path $semiRunDir 'process.stderr.log'
$trainCommon = @(
    '-u', '-m', 'mmsv.cli', 'train-audio',
    '--config', $semiConfig,
    '--manifest', $finalManifest,
    '--splits', $splits,
    '--output-dir', $semiRunDir
)
if (Test-Path -LiteralPath $semiCheckpoint) {
    $trainArgs = $trainCommon + @('--resume', $semiCheckpoint)
    Write-Output 'training_mode=resume'
} else {
    $trainArgs = $trainCommon + @('--init-from', $lazyCheckpoint)
    Write-Output 'training_mode=init_from_corrected_lazy_reset_optimizer'
}
$training = Start-Process -FilePath $PythonExe -ArgumentList $trainArgs `
    -WorkingDirectory $projectRoot -RedirectStandardOutput $trainOut -RedirectStandardError $trainErr `
    -WindowStyle Hidden -PassThru
Write-Output "training_started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Write-Output "training_pid=$($training.Id)"
while (-not $training.HasExited) {
    Start-Sleep -Seconds $MonitorSeconds
    $training.Refresh()
    $memoryText = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    $memoryMiB = [int](($memoryText | Select-Object -First 1).Trim())
    $trainLines = if (Test-Path -LiteralPath (Join-Path $semiRunDir 'train.jsonl')) {
        (Get-Content -LiteralPath (Join-Path $semiRunDir 'train.jsonl')).Count
    } else { 0 }
    Write-Output "training_progress=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') completed_epochs=$trainLines/15 gpu_memory_mib=$memoryMiB"
}
$training.WaitForExit()
if (-not (Test-Path -LiteralPath $semiCheckpoint)) {
    throw 'Semi-informed training ended without last.pt'
}
& $PythonExe scripts/validate_training_checkpoint.py `
    --checkpoint $semiCheckpoint `
    --expected-last-epoch 14
if ($LASTEXITCODE -ne 0) {
    throw 'Semi-informed checkpoint completeness validation failed'
}
Write-Output "training_completed=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"

$originalEmbeddings = 'artifacts/embeddings/original_evaluation_semi_corrected.npz'
$anonymizedEmbeddings = 'artifacts/embeddings/anonymized_evaluation_semi_corrected.npz'
& $PythonExe -m mmsv.cli extract-embeddings `
    --checkpoint $semiCheckpoint `
    --manifest artifacts/metadata/fisher_manifest.csv `
    --trials $trials `
    --output $originalEmbeddings
if ($LASTEXITCODE -ne 0) {
    throw 'Semi-informed original embedding extraction failed'
}
& $PythonExe -m mmsv.cli extract-embeddings `
    --checkpoint $semiCheckpoint `
    --manifest artifacts/metadata/fisher_anonymized_evaluation_manifest.csv `
    --trials $trials `
    --output $anonymizedEmbeddings
if ($LASTEXITCODE -ne 0) {
    throw 'Semi-informed anonymized embedding extraction failed'
}

foreach ($condition in @('O-A', 'A-A')) {
    $resultDir = if ($condition -eq 'O-A') { 'results/o_a_semi_corrected' } else { 'results/a_a_semi_corrected' }
    New-Item -ItemType Directory -Path $resultDir -Force | Out-Null
    foreach ($n in @(1, 5, 10, 15)) {
        & $PythonExe -m mmsv.cli score-mean `
            --trials $trials `
            --original-embeddings $originalEmbeddings `
            --anonymized-embeddings $anonymizedEmbeddings `
            --condition $condition `
            --n $n `
            --output (Join-Path $resultDir "mean_n$n.csv")
        if ($LASTEXITCODE -ne 0) {
            throw "$condition semi-informed mean scoring failed for N=$n"
        }
    }
}

Write-Output "pipeline_completed=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Write-Output 'next_action=append semi-informed metrics and hashes to EXPERIMENT_RESULTS.md'
