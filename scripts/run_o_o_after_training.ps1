param(
    [int]$TrainingPid = 0,
    [string]$PythonExe = 'D:\codeAPP\anaconda3\envs\pytorch\python.exe',
    [string]$RunDir = 'results/runs/audio_lazy_p1_v2',
    [string]$EmbeddingPath = 'artifacts/embeddings/original_evaluation.npz',
    [string]$ResultDir = 'results/o_o'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

Write-Output "pipeline_started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Write-Output "project_root=$projectRoot"
Write-Output "training_pid=$TrainingPid"

if ($TrainingPid -gt 0 -and $null -ne (Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue)) {
    Write-Output 'waiting_for_training=true'
    Wait-Process -Id $TrainingPid
}

$checkpoint = Join-Path $RunDir 'last.pt'
if (-not (Test-Path -LiteralPath $checkpoint)) {
    throw "Training ended without checkpoint: $checkpoint"
}

& $PythonExe scripts/validate_training_checkpoint.py `
    --checkpoint $checkpoint `
    --expected-last-epoch 29
if ($LASTEXITCODE -ne 0) {
    throw 'Checkpoint completeness validation failed'
}

& $PythonExe -m mmsv.cli extract-embeddings `
    --checkpoint $checkpoint `
    --manifest artifacts/metadata/fisher_manifest.csv `
    --trials artifacts/trials/evaluation.jsonl `
    --output $EmbeddingPath
if ($LASTEXITCODE -ne 0) {
    throw 'Embedding extraction failed'
}

New-Item -ItemType Directory -Path $ResultDir -Force | Out-Null
foreach ($n in @(5, 10, 15)) {
    & $PythonExe -m mmsv.cli score-mean `
        --trials artifacts/trials/evaluation.jsonl `
        --original-embeddings $EmbeddingPath `
        --condition O-O `
        --n $n `
        --output (Join-Path $ResultDir "mean_n$n.csv")
    if ($LASTEXITCODE -ne 0) {
        throw "O-O mean scoring failed for N=$n"
    }
}

Write-Output "pipeline_completed=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Write-Output 'next_action=append generated metrics and hashes to EXPERIMENT_RESULTS.md'
