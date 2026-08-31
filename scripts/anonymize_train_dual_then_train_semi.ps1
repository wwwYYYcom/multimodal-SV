param(
    [string]$PythonExe = 'D:\codeAPP\anaconda3\envs\pytorch\python.exe',
    [int]$SplitIndex = 301378,
    [int]$MonitorSeconds = 60,
    [int]$MaxWorkerAttempts = 20,
    [int]$MaxTrainingAttempts = 10
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$plan = 'artifacts/anonymization/train_all_utterances_plan.csv'
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
$fullCount = 572951
$worker1Count = $SplitIndex
$worker2Count = $fullCount - $SplitIndex
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
New-Item -ItemType Directory -Path $runDir -Force | Out-Null
New-Item -ItemType Directory -Path $semiRunDir -Force | Out-Null

$planAuditPath = [System.IO.Path]::ChangeExtension((Resolve-Path -LiteralPath $plan).Path, '.audit.json')
$planAudit = Get-Content -LiteralPath $planAuditPath -Raw | ConvertFrom-Json
if ($planAudit.source_utterances -ne $fullCount -or $planAudit.one_per_call_side) {
    throw "Full-utterance train plan audit is invalid: $($planAudit | ConvertTo-Json -Compress)"
}
if ($SplitIndex -le 0 -or $SplitIndex -ge $fullCount) {
    throw "SplitIndex must be within (0, $fullCount): $SplitIndex"
}
$projectedOutputBytes = 35828004345L
$existingOutputBytes = if (Test-Path -LiteralPath 'artifacts/anonymized/train') {
    [long]((Get-ChildItem -LiteralPath 'artifacts/anonymized/train' -File -Recurse | Measure-Object Length -Sum).Sum)
} else { 0L }
$remainingOutputBytes = [Math]::Max(0L, $projectedOutputBytes - $existingOutputBytes)
$reserveBytes = 6L * 1024L * 1024L * 1024L
$freeBytes = [long](Get-PSDrive -Name D).Free
if ($freeBytes -lt ($remainingOutputBytes + $reserveBytes)) {
    throw "Insufficient D: space: free=$freeBytes required=$($remainingOutputBytes + $reserveBytes)"
}

$worker1Manifest = Join-Path $runDir 'worker1.manifest.csv'
$worker2Manifest = Join-Path $runDir 'worker2.manifest.csv'
$common = @(
    '-u', '-m', 'mmsv.cli', 'anonymize-streamvoice',
    '--plan', $plan,
    '--streamvoice-root', $streamVoiceRoot,
    '--delay', '2',
    '--alpha', '1.0',
    '--max-source-chunk-seconds', '30.0'
)

function Start-AnonymizationWorker {
    param(
        [int]$WorkerNumber,
        [int]$Attempt,
        [string]$Manifest,
        [string[]]$SliceArguments
    )
    if ($Attempt -gt 1) {
        $progress = [System.IO.Path]::ChangeExtension($Manifest, '.progress.jsonl')
        if (Test-Path -LiteralPath $progress) {
            $archivedProgress = Join-Path $runDir "worker$WorkerNumber.attempt$($Attempt - 1).progress.jsonl"
            Move-Item -LiteralPath $progress -Destination $archivedProgress -Force
        }
    }
    $stdout = Join-Path $runDir "worker$WorkerNumber.attempt$Attempt.stdout.log"
    $stderr = Join-Path $runDir "worker$WorkerNumber.attempt$Attempt.stderr.log"
    return Start-Process -FilePath $PythonExe `
        -ArgumentList ($common + @('--output-manifest', $Manifest) + $SliceArguments) `
        -WorkingDirectory $projectRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
        -WindowStyle Hidden -PassThru
}

Write-Output "pipeline_started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Write-Output "run_dir=$((Resolve-Path -LiteralPath $runDir).Path)"
Write-Output "split_index=$SplitIndex"
Write-Output "worker1_count=$worker1Count"
Write-Output "worker2_count=$worker2Count"
Write-Output 'precision=fp32_weights_with_upstream_cuda_autocast'
Write-Output 'huggingface_offline=true'
Write-Output "existing_output_bytes=$existingOutputBytes"
Write-Output "projected_output_bytes=$projectedOutputBytes"
Write-Output "startup_free_bytes=$freeBytes"
$anonymizationStarted = Get-Date
$workerStates = @(
    @{
        Number = 1; Expected = $worker1Count; Manifest = $worker1Manifest
        SliceArguments = @('--limit', "$worker1Count"); Attempt = 1
        Process = $null; Completed = $false
    },
    @{
        Number = 2; Expected = $worker2Count; Manifest = $worker2Manifest
        SliceArguments = @('--start-index', "$SplitIndex"); Attempt = 1
        Process = $null; Completed = $false
    }
)
foreach ($state in $workerStates) {
    $state.Process = Start-AnonymizationWorker `
        -WorkerNumber $state.Number -Attempt $state.Attempt `
        -Manifest $state.Manifest -SliceArguments $state.SliceArguments
    Write-Output "worker_started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') worker=$($state.Number) attempt=$($state.Attempt) pid=$($state.Process.Id)"
}

$peakGpuMemoryMiB = 0
$nextReport = Get-Date
while (@($workerStates | Where-Object { -not $_.Completed }).Count -gt 0) {
    foreach ($state in $workerStates) {
        if ($state.Completed) {
            continue
        }
        $state.Process.Refresh()
        if (-not $state.Process.HasExited) {
            continue
        }
        $state.Process.WaitForExit()
        $auditPath = [System.IO.Path]::ChangeExtension($state.Manifest, '.audit.json')
        $auditComplete = $false
        if (Test-Path -LiteralPath $auditPath) {
            $audit = Get-Content -LiteralPath $auditPath -Raw | ConvertFrom-Json
            $auditComplete = $audit.processed -eq $state.Expected
        }
        if ($auditComplete) {
            $state.Completed = $true
            Write-Output "worker_completed=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') worker=$($state.Number) attempt=$($state.Attempt)"
            continue
        }
        $exitCode = $state.Process.ExitCode
        Write-Output "worker_failed=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') worker=$($state.Number) attempt=$($state.Attempt) exit_code=$exitCode"
        if ($state.Attempt -ge $MaxWorkerAttempts) {
            throw "Worker $($state.Number) exhausted $MaxWorkerAttempts attempts"
        }
        $state.Attempt += 1
        Start-Sleep -Seconds 10
        $state.Process = Start-AnonymizationWorker `
            -WorkerNumber $state.Number -Attempt $state.Attempt `
            -Manifest $state.Manifest -SliceArguments $state.SliceArguments
        Write-Output "worker_restarted=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') worker=$($state.Number) attempt=$($state.Attempt) pid=$($state.Process.Id)"
    }
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
        Write-Output "anonymization_progress=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') worker1=$count1/$worker1Count attempt1=$($workerStates[0].Attempt) worker2=$count2/$worker2Count attempt2=$($workerStates[1].Attempt) gpu_memory_mib=$memoryMiB"
        $nextReport = (Get-Date).AddSeconds($MonitorSeconds)
    }
    Start-Sleep -Seconds 5
}
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
$trainCommon = @(
    '-u', '-m', 'mmsv.cli', 'train-audio',
    '--config', $semiConfig,
    '--manifest', $finalManifest,
    '--splits', $splits,
    '--output-dir', $semiRunDir
)
$trainingComplete = $false
$trainingAttempt = 0
while (-not $trainingComplete) {
    $trainingAttempt += 1
    if (Test-Path -LiteralPath $semiCheckpoint) {
        $trainArgs = $trainCommon + @('--resume', $semiCheckpoint)
        $trainingMode = 'resume'
    } else {
        $trainArgs = $trainCommon + @('--init-from', $lazyCheckpoint)
        $trainingMode = 'init_from_corrected_lazy_reset_optimizer'
    }
    $trainOut = Join-Path $semiRunDir "process.attempt$trainingAttempt.stdout.log"
    $trainErr = Join-Path $semiRunDir "process.attempt$trainingAttempt.stderr.log"
    $training = Start-Process -FilePath $PythonExe -ArgumentList $trainArgs `
        -WorkingDirectory $projectRoot -RedirectStandardOutput $trainOut -RedirectStandardError $trainErr `
        -WindowStyle Hidden -PassThru
    Write-Output "training_started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') attempt=$trainingAttempt pid=$($training.Id) mode=$trainingMode"
    while (-not $training.HasExited) {
        Start-Sleep -Seconds $MonitorSeconds
        $training.Refresh()
        $memoryText = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
        $memoryMiB = [int](($memoryText | Select-Object -First 1).Trim())
        $trainLines = if (Test-Path -LiteralPath (Join-Path $semiRunDir 'train.jsonl')) {
            (Get-Content -LiteralPath (Join-Path $semiRunDir 'train.jsonl')).Count
        } else { 0 }
        Write-Output "training_progress=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') attempt=$trainingAttempt completed_epochs=$trainLines/15 gpu_memory_mib=$memoryMiB"
    }
    $training.WaitForExit()
    if (Test-Path -LiteralPath $semiCheckpoint) {
        & $PythonExe scripts/validate_training_checkpoint.py `
            --checkpoint $semiCheckpoint `
            --expected-last-epoch 14
        $trainingComplete = $LASTEXITCODE -eq 0
    }
    if (-not $trainingComplete) {
        Write-Output "training_failed=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') attempt=$trainingAttempt exit_code=$($training.ExitCode)"
        if ($trainingAttempt -ge $MaxTrainingAttempts) {
            throw "Semi-informed training exhausted $MaxTrainingAttempts attempts"
        }
        Start-Sleep -Seconds 30
    }
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
