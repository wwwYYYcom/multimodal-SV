param(
    [string]$PythonExe = 'D:\codeAPP\anaconda3\envs\pytorch\python.exe',
    [string]$RunDir = 'results/runs/audio_lazy_p1_v2',
    [string]$CacheDir = 'artifacts/cache/fisher_train_selected_30e'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

Write-Output "supervisor_started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Write-Output "cache_dir=$CacheDir"

& $PythonExe -u scripts/build_fisher_training_cache.py `
    --config configs/local_fisher_p1.yaml `
    --manifest artifacts/metadata/fisher_manifest.csv `
    --splits artifacts/metadata/speaker_splits.csv `
    --output-dir $CacheDir
if ($LASTEXITCODE -ne 0) {
    throw "Fisher segment cache failed with exit code $LASTEXITCODE"
}

$audit = Get-Content -LiteralPath (Join-Path $CacheDir 'audit.json') -Raw | ConvertFrom-Json
if ($audit.unique_utterances -le 0 -or ($audit.generated + $audit.skipped_existing) -ne $audit.unique_utterances) {
    throw "Fisher segment cache audit is incomplete: $($audit | ConvertTo-Json -Compress)"
}
Write-Output "cache_completed=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Write-Output "unique_utterances=$($audit.unique_utterances)"

$trainingArgs = @(
    '-u', '-m', 'mmsv.cli', 'train-audio',
    '--config', 'configs/local_fisher_p1.yaml',
    '--manifest', 'artifacts/metadata/fisher_manifest.csv',
    '--splits', 'artifacts/metadata/speaker_splits.csv',
    '--output-dir', $RunDir,
    '--resume', (Join-Path $RunDir 'last.pt')
)
$training = Start-Process -FilePath $PythonExe -ArgumentList $trainingArgs `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput (Join-Path $RunDir 'process_cached.stdout.log') `
    -RedirectStandardError (Join-Path $RunDir 'process_cached.stderr.log') `
    -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 12
if ($training.HasExited) {
    throw "Cached training exited early with code $($training.ExitCode)"
}

$watchArgs = @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass',
    '-File', (Join-Path $PSScriptRoot 'run_o_o_after_training.ps1'),
    '-TrainingPid', $training.Id,
    '-RunDir', $RunDir
)
$watcher = Start-Process -FilePath 'powershell.exe' -ArgumentList $watchArgs `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput (Join-Path $RunDir 'post_pipeline_cached.stdout.log') `
    -RedirectStandardError (Join-Path $RunDir 'post_pipeline_cached.stderr.log') `
    -WindowStyle Hidden -PassThru

Write-Output "training_pid=$($training.Id)"
Write-Output "watcher_pid=$($watcher.Id)"
Write-Output "supervisor_completed=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
