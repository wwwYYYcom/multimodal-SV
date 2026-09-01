[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Name,

    [Parameter(Mandatory = $true)]
    [string]$BaseDirectory,

    [Parameter(Mandatory = $true)]
    [string[]]$InputPath,

    [string[]]$ExcludeRelativePath = @(),

    [Parameter(Mandatory = $true)]
    [string]$DestinationRoot,

    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 1000000)]
    [int]$PartNumber,

    [string]$OutputDirectory = 'C:\mmsv_transfer',

    [ValidateRange(1048576, 2140000000)]
    [Int64]$TargetInputBytes = 1900000000
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Get-RelativeArchivePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Base,

        [Parameter(Mandatory = $true)]
        [string]$FullName
    )

    $basePrefix = $Base.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if (-not $FullName.StartsWith($basePrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Input is outside BaseDirectory: $FullName"
    }
    return $FullName.Substring($basePrefix.Length).Replace('\', '/')
}

$base = (Resolve-Path -LiteralPath $BaseDirectory).Path.TrimEnd('\', '/')
$allFiles = New-Object System.Collections.Generic.List[object]

foreach ($input in $InputPath) {
    $resolved = (Resolve-Path -LiteralPath $input).Path
    $item = Get-Item -LiteralPath $resolved
    $files = if ($item.PSIsContainer) {
        Get-ChildItem -LiteralPath $resolved -File -Recurse
    }
    else {
        @($item)
    }

    foreach ($file in $files) {
        $relative = Get-RelativeArchivePath -Base $base -FullName $file.FullName
        $allFiles.Add([pscustomobject]@{
            RelativePath = $relative
            Bytes = [Int64]$file.Length
        })
    }
}

$normalizedExclusions = @(
    $ExcludeRelativePath |
        ForEach-Object { $_.Replace('\', '/').TrimStart('/') } |
        Sort-Object -Unique
)
$exclusionSet = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
foreach ($excludedPath in $normalizedExclusions) {
    $null = $exclusionSet.Add($excludedPath)
}

$orderedFiles = @(
    $allFiles |
        Where-Object { -not $exclusionSet.Contains($_.RelativePath) } |
        Sort-Object RelativePath -Unique
)
if ($orderedFiles.Count -eq 0) {
    throw 'No input files were found.'
}

$parts = New-Object System.Collections.Generic.List[object]
$current = New-Object System.Collections.Generic.List[object]
[Int64]$currentBytes = 0

foreach ($file in $orderedFiles) {
    if ($current.Count -gt 0 -and ($currentBytes + $file.Bytes) -gt $TargetInputBytes) {
        $completedPart = New-Object object[] $current.Count
        $current.CopyTo($completedPart)
        $parts.Add($completedPart)
        $current = New-Object System.Collections.Generic.List[object]
        $currentBytes = 0
    }
    $current.Add($file)
    $currentBytes += $file.Bytes
}
if ($current.Count -gt 0) {
    $completedPart = New-Object object[] $current.Count
    $current.CopyTo($completedPart)
    $parts.Add($completedPart)
}

if ($PartNumber -gt $parts.Count) {
    throw "PartNumber $PartNumber exceeds the computed part count $($parts.Count)."
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$safeName = $Name -replace '[^A-Za-z0-9._-]', '_'
$stem = '{0}.part{1:D3}-of-{2:D3}' -f $safeName, $PartNumber, $parts.Count
$archivePath = Join-Path $OutputDirectory "$stem.tar"
$listPath = Join-Path $OutputDirectory "$stem.files.txt"
$auditPath = Join-Path $OutputDirectory "$stem.audit.json"
$checksumPath = Join-Path $OutputDirectory "$stem.sha256"

foreach ($output in @($archivePath, $listPath, $auditPath, $checksumPath)) {
    if (Test-Path -LiteralPath $output) {
        throw "Refusing to overwrite existing output: $output"
    }
}

$selected = @($parts[$PartNumber - 1])
$utf8NoBom = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllLines($listPath, [string[]]$selected.RelativePath, $utf8NoBom)

$tar = Join-Path $env:SystemRoot 'System32\tar.exe'
if (-not (Test-Path -LiteralPath $tar)) {
    throw "tar.exe was not found at $tar"
}

& $tar -C $base -cf $archivePath -T $listPath
if ($LASTEXITCODE -ne 0) {
    throw "tar.exe failed with exit code $LASTEXITCODE"
}

$archive = Get-Item -LiteralPath $archivePath
$archiveHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant()
$inputBytes = [Int64](($selected | Measure-Object -Property Bytes -Sum).Sum)
$allInputBytes = [Int64](($orderedFiles | Measure-Object -Property Bytes -Sum).Sum)

$audit = [ordered]@{
    schema_version = 1
    completed_at = (Get-Date).ToString('o')
    name = $Name
    base_directory = $base
    destination_root = $DestinationRoot
    target_input_bytes = $TargetInputBytes
    part_number = $PartNumber
    part_count = $parts.Count
    archive_path = $archive.FullName
    archive_bytes = [Int64]$archive.Length
    archive_sha256 = $archiveHash
    file_list_path = $listPath
    selected_file_count = $selected.Count
    selected_input_bytes = $inputBytes
    total_file_count = $orderedFiles.Count
    total_input_bytes = $allInputBytes
    excluded_file_count = $normalizedExclusions.Count
    excluded_relative_paths = $normalizedExclusions
    first_member = $selected[0].RelativePath
    last_member = $selected[-1].RelativePath
}

[IO.File]::WriteAllText(
    $auditPath,
    (($audit | ConvertTo-Json -Depth 4) + [Environment]::NewLine),
    $utf8NoBom
)
[IO.File]::WriteAllText(
    $checksumPath,
    "$archiveHash  $($archive.Name)$([Environment]::NewLine)",
    $utf8NoBom
)

$audit | ConvertTo-Json -Depth 4
