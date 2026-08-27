param(
    [int]$GeneratorPid = 0,
    [string]$PythonExe = 'D:\codeAPP\anaconda3\envs\pytorch\python.exe',
    [string]$Checkpoint = 'results/runs/audio_corrected_p1/last.pt',
    [string]$Manifest = 'artifacts/metadata/fisher_anonymized_evaluation_manifest.csv',
    [string]$OriginalEmbeddings = 'artifacts/embeddings/original_evaluation_corrected.npz',
    [string]$AnonymizedEmbeddings = 'artifacts/embeddings/anonymized_evaluation_corrected.npz'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$runDir = 'results/runs/anonymization_evaluation'
$trialPath = 'artifacts/trials/evaluation.jsonl'

Write-Output "scoring_watcher_started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Write-Output "generator_pid=$GeneratorPid"
if ($GeneratorPid -gt 0 -and $null -ne (Get-Process -Id $GeneratorPid -ErrorAction SilentlyContinue)) {
    Write-Output 'waiting_for_generation=true'
    Wait-Process -Id $GeneratorPid
}

$auditPath = [System.IO.Path]::ChangeExtension((Resolve-Path -LiteralPath $Manifest).Path, '.audit.json')
$audit = Get-Content -LiteralPath $auditPath -Raw | ConvertFrom-Json
$accounted = $audit.generated + $audit.skipped_existing
if ($audit.processed -ne 86222 -or $accounted -ne 86222) {
    throw "Evaluation anonymization audit is incomplete: $($audit | ConvertTo-Json -Compress)"
}
$validation = Get-Content -LiteralPath (Join-Path $runDir 'final.validation.json') -Raw | ConvertFrom-Json
if (-not $validation.valid -or $validation.manifest_rows -ne 86222) {
    throw "Evaluation anonymization validation is incomplete: $($validation | ConvertTo-Json -Compress)"
}
Write-Output "generation_validated=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"

& $PythonExe -m mmsv.cli extract-embeddings `
    --checkpoint $Checkpoint `
    --manifest $Manifest `
    --trials $trialPath `
    --output $AnonymizedEmbeddings
if ($LASTEXITCODE -ne 0) {
    throw 'Anonymized embedding extraction failed'
}

foreach ($condition in @('O-A', 'A-A')) {
    $resultDir = if ($condition -eq 'O-A') { 'results/o_a_corrected' } else { 'results/a_a_corrected' }
    New-Item -ItemType Directory -Path $resultDir -Force | Out-Null
    foreach ($n in @(1, 5, 10, 15)) {
        & $PythonExe -m mmsv.cli score-mean `
            --trials $trialPath `
            --original-embeddings $OriginalEmbeddings `
            --anonymized-embeddings $AnonymizedEmbeddings `
            --condition $condition `
            --n $n `
            --output (Join-Path $resultDir "mean_n$n.csv")
        if ($LASTEXITCODE -ne 0) {
            throw "$condition mean scoring failed for N=$n"
        }
    }
}

Write-Output "scoring_pipeline_completed=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Write-Output 'next_action=append anonymized metrics and hashes to EXPERIMENT_RESULTS.md'
