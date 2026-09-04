param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$OutputDirectory = "D:\download4browser"
)

$ErrorActionPreference = "Stop"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$name = "mmsv_saar_phase12_inputs_$stamp.tar"
$output = Join-Path $OutputDirectory $name
$manifest = Join-Path $OutputDirectory "$name.sha256"

$relativeInputs = @(
    "artifacts/saar/session_baseline/manifests",
    "artifacts/saar/utterance_random_control/metrics",
    "artifacts/saar/utterance_random_control/figures",
    "artifacts/embeddings/original_evaluation_corrected.npz"
)
foreach ($relative in $relativeInputs) {
    $absolute = Join-Path $ProjectRoot $relative
    if (-not (Test-Path -LiteralPath $absolute)) {
        throw "Missing transfer input: $absolute"
    }
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
& tar.exe -cf $output -C $ProjectRoot @relativeInputs
if ($LASTEXITCODE -ne 0) {
    throw "tar.exe failed with exit code $LASTEXITCODE"
}

$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $output).Hash.ToLowerInvariant()
"$hash  $name" | Set-Content -LiteralPath $manifest -Encoding ascii
$bytes = (Get-Item -LiteralPath $output).Length

[ordered]@{
    archive = $output
    sha256_manifest = $manifest
    sha256 = $hash
    bytes = $bytes
    inputs = $relativeInputs
} | ConvertTo-Json -Depth 3
