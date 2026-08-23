param(
    [int]$TrainingPid = 0,
    [string]$PythonExe = 'D:\codeAPP\anaconda3\envs\pytorch\python.exe',
    [string]$RunDir = 'results/runs/audio_lazy_p1_v2'
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

& $PythonExe -c @'
import json
import sys
import torch

path = sys.argv[1]
state = torch.load(path, map_location="cpu", weights_only=False)
metadata = {
    "epoch": int(state["epoch"]),
    "epoch_complete": bool(state.get("epoch_complete", True)),
    "batch_in_epoch": int(state.get("batch_in_epoch", 0)),
    "global_step": int(state["global_step"]),
}
print(json.dumps(metadata))
if metadata["epoch"] != 29 or not metadata["epoch_complete"]:
    raise SystemExit(
        f"Refusing evaluation: training is incomplete ({metadata})"
    )
'@ $checkpoint
if ($LASTEXITCODE -ne 0) {
    throw 'Checkpoint completeness validation failed'
}

$embeddingPath = 'artifacts/embeddings/original_evaluation.npz'
& $PythonExe -m mmsv.cli extract-embeddings `
    --checkpoint $checkpoint `
    --manifest artifacts/metadata/fisher_manifest.csv `
    --trials artifacts/trials/evaluation.jsonl `
    --output $embeddingPath
if ($LASTEXITCODE -ne 0) {
    throw 'Embedding extraction failed'
}

New-Item -ItemType Directory -Path 'results/o_o' -Force | Out-Null
foreach ($n in @(5, 10, 15)) {
    & $PythonExe -m mmsv.cli score-mean `
        --trials artifacts/trials/evaluation.jsonl `
        --original-embeddings $embeddingPath `
        --condition O-O `
        --n $n `
        --output "results/o_o/mean_n$n.csv"
    if ($LASTEXITCODE -ne 0) {
        throw "O-O mean scoring failed for N=$n"
    }
}

Write-Output "pipeline_completed=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Write-Output 'next_action=append generated metrics and hashes to EXPERIMENT_RESULTS.md'
