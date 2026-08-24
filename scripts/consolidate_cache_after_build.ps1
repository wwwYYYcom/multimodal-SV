param(
    [int]$BuilderPid = 0,
    [string]$PythonExe = 'D:\codeAPP\anaconda3\envs\pytorch\python.exe',
    [string]$SourceDir = 'artifacts/cache/fisher_train_selected_30e',
    [string]$DestinationDir = 'artifacts/cache/fisher_train_all_p1'
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $projectRoot

Write-Output "consolidation_started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
Write-Output "builder_pid=$BuilderPid"
if ($BuilderPid -gt 0 -and $null -ne (Get-Process -Id $BuilderPid -ErrorAction SilentlyContinue)) {
    Write-Output 'waiting_for_builder=true'
    Wait-Process -Id $BuilderPid
}

$rootResolved = (Resolve-Path -LiteralPath $projectRoot).Path.TrimEnd('\')
$sourceResolved = (Resolve-Path -LiteralPath $SourceDir).Path.TrimEnd('\')
$destinationResolved = (Resolve-Path -LiteralPath $DestinationDir).Path.TrimEnd('\')
$expectedSource = Join-Path $rootResolved 'artifacts\cache\fisher_train_selected_30e'
$expectedDestination = Join-Path $rootResolved 'artifacts\cache\fisher_train_all_p1'
if ($sourceResolved -ne $expectedSource -or $destinationResolved -ne $expectedDestination) {
    throw "Refusing unexpected cache paths: source=$sourceResolved destination=$destinationResolved"
}
if ($sourceResolved -eq $destinationResolved) {
    throw 'Source and destination cache paths must differ'
}

$destinationAuditPath = Join-Path $destinationResolved 'audit.json'
if (-not (Test-Path -LiteralPath $destinationAuditPath)) {
    throw "Full cache audit is missing: $destinationAuditPath"
}
$destinationAudit = Get-Content -LiteralPath $destinationAuditPath -Raw | ConvertFrom-Json
if ($destinationAudit.train_utterances -ne 572951 -or $destinationAudit.target_utterances -ne 572951) {
    throw "Full cache audit is incomplete: $($destinationAudit | ConvertTo-Json -Compress)"
}

$verificationOutput = & $PythonExe scripts/verify_cache_hardlinks.py `
    --source $sourceResolved `
    --destination $destinationResolved `
    --expected 180311
if ($LASTEXITCODE -ne 0) {
    throw 'Hardlink verification failed; preserving the source cache'
}
$verificationJson = $verificationOutput -join "`n"
$verification = $verificationJson | ConvertFrom-Json
if (-not $verification.valid -or $verification.verified_hardlinks -ne 180311) {
    throw "Hardlink verification is incomplete: $($verification | ConvertTo-Json -Compress)"
}
Write-Output ($verification | ConvertTo-Json -Compress)

$sourceAuditPath = Join-Path $sourceResolved 'audit.json'
$preservedAuditPath = Join-Path $destinationResolved 'selected_30e.audit.json'
Copy-Item -LiteralPath $sourceAuditPath -Destination $preservedAuditPath -Force

# Removing the old names keeps the audio available under the full-cache directory.
Remove-Item -LiteralPath $sourceResolved -Recurse -Force
if (Test-Path -LiteralPath $sourceResolved) {
    throw "Source cache directory still exists after consolidation: $sourceResolved"
}

Write-Output "preserved_audit=$preservedAuditPath"
Write-Output "removed_directory=$sourceResolved"
Write-Output 'recoverability=audio_data_remains_in_destination;old_directory_name_removed'
Write-Output "consolidation_completed=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
