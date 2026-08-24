param(
    [string]$PythonExe = 'D:\codeAPP\anaconda3\envs\pytorch\python.exe',
    [string]$RunDir = 'results/runs/audio_corrected_p1',
    [string]$CacheDir = 'artifacts/cache/fisher_train_all_p1',
    [string]$ReuseCacheDir = 'artifacts/cache/fisher_train_selected_30e'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

Write-Output "supervisor_started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Write-Output "cache_dir=$CacheDir"

& $PythonExe -u scripts/build_fisher_full_training_cache.py `
    --config configs/local_fisher_p1_corrected.yaml `
    --manifest artifacts/metadata/fisher_manifest.csv `
    --splits artifacts/metadata/speaker_splits.csv `
    --output-dir $CacheDir `
    --reuse-cache-dir $ReuseCacheDir
if ($LASTEXITCODE -ne 0) {
    throw "Full Fisher cache failed with exit code $LASTEXITCODE"
}

$audit = Get-Content -LiteralPath (Join-Path $CacheDir 'audit.json') -Raw | ConvertFrom-Json
$accounted = $audit.generated + $audit.hardlinked + $audit.skipped_existing
if ($audit.train_utterances -ne 572951 -or $audit.target_utterances -ne 572951 -or $accounted -ne 572951) {
    throw "Full Fisher cache audit is incomplete: $($audit | ConvertTo-Json -Compress)"
}
Write-Output "cache_completed=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Write-Output "train_utterances=$($audit.train_utterances)"

New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
$trainingArgs = @(
    '-u', '-m', 'mmsv.cli', 'train-audio',
    '--config', 'configs/local_fisher_p1_corrected.yaml',
    '--manifest', 'artifacts/metadata/fisher_manifest.csv',
    '--splits', 'artifacts/metadata/speaker_splits.csv',
    '--output-dir', $RunDir
)
$training = Start-Process -FilePath $PythonExe -ArgumentList $trainingArgs `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput (Join-Path $RunDir 'process.stdout.log') `
    -RedirectStandardError (Join-Path $RunDir 'process.stderr.log') `
    -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 15
if ($training.HasExited) {
    throw "Corrected training exited early with code $($training.ExitCode)"
}

$watchArgs = @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass',
    '-File', (Join-Path $PSScriptRoot 'run_o_o_after_training.ps1'),
    '-TrainingPid', $training.Id,
    '-RunDir', $RunDir,
    '-EmbeddingPath', 'artifacts/embeddings/original_evaluation_corrected.npz',
    '-ResultDir', 'results/o_o_corrected'
)
$watcher = Start-Process -FilePath 'powershell.exe' -ArgumentList $watchArgs `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput (Join-Path $RunDir 'post_pipeline.stdout.log') `
    -RedirectStandardError (Join-Path $RunDir 'post_pipeline.stderr.log') `
    -WindowStyle Hidden -PassThru

Write-Output "training_pid=$($training.Id)"
Write-Output "watcher_pid=$($watcher.Id)"
Write-Output "supervisor_completed=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
