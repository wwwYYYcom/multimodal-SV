param(
    [Parameter(Mandatory = $true)]
    [string]$Archive,
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"
$resolvedArchive = (Resolve-Path -LiteralPath $Archive).Path
$shaManifest = "$resolvedArchive.sha256"
if (-not (Test-Path -LiteralPath $shaManifest)) {
    throw "Missing SHA-256 manifest: $shaManifest"
}

$firstLine = (Get-Content -LiteralPath $shaManifest -TotalCount 1).Trim()
if ($firstLine -notmatch '^([0-9a-fA-F]{64})\s+') {
    throw "Malformed SHA-256 manifest: $shaManifest"
}
$expected = $Matches[1].ToLowerInvariant()
$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedArchive).Hash.ToLowerInvariant()
if ($actual -ne $expected) {
    throw "SHA-256 mismatch: expected=$expected actual=$actual"
}

$members = @(& tar.exe -tf $resolvedArchive)
if ($LASTEXITCODE -ne 0 -or $members.Count -eq 0) {
    throw "Unable to list archive: $resolvedArchive"
}
$unsafe = @($members | Where-Object {
    $_ -match '^[A-Za-z]:[/\\]' -or $_ -match '^/' -or $_ -match '(^|/)\.\.(/|$)'
})
if ($unsafe.Count -gt 0) {
    throw "Archive contains unsafe paths: $($unsafe -join ', ')"
}

& tar.exe -xf $resolvedArchive -C $ProjectRoot
if ($LASTEXITCODE -ne 0) {
    throw "tar.exe extraction failed with exit code $LASTEXITCODE"
}

$summary = Join-Path $ProjectRoot "artifacts/saar/session_baseline/evaluation_summary.json"
$gate = Join-Path $ProjectRoot "artifacts/saar/session_baseline/metrics/gate_1.json"
if (-not (Test-Path -LiteralPath $summary) -or -not (Test-Path -LiteralPath $gate)) {
    throw "Gate 1 summary files are missing after extraction"
}

[ordered]@{
    imported = $true
    archive = $resolvedArchive
    sha256 = $actual
    members = $members.Count
    evaluation_summary = $summary
    gate_1 = $gate
} | ConvertTo-Json -Depth 3
