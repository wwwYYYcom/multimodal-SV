param(
    [string]$PythonExe = 'D:\codeAPP\anaconda3\envs\pytorch\python.exe',
    [int]$StartIndex = 303,
    [int]$CountPerWorker = 20,
    [int]$Warmup = 3,
    [ValidateSet('baseline', 'fp16')]
    [string]$Variant = 'fp16'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$runDir = Join-Path 'results/runs/anonymization_dual_fp16_benchmark' $stamp
New-Item -ItemType Directory -Path $runDir -Force | Out-Null

$worker1Out = Join-Path $runDir 'worker1.stdout.log'
$worker1Err = Join-Path $runDir 'worker1.stderr.log'
$worker2Out = Join-Path $runDir 'worker2.stdout.log'
$worker2Err = Join-Path $runDir 'worker2.stderr.log'
$secondStart = $StartIndex + $CountPerWorker
$common = @(
    '-u', 'scripts/benchmark_streamvoice_compile.py',
    '--count', "$CountPerWorker",
    '--warmup', "$Warmup",
    '--variants', $Variant
)

$started = Get-Date
$worker1 = Start-Process -FilePath $PythonExe -ArgumentList ($common + @('--start-index', "$StartIndex")) `
    -WorkingDirectory $projectRoot -RedirectStandardOutput $worker1Out -RedirectStandardError $worker1Err `
    -WindowStyle Hidden -PassThru
$worker2 = Start-Process -FilePath $PythonExe -ArgumentList ($common + @('--start-index', "$secondStart")) `
    -WorkingDirectory $projectRoot -RedirectStandardOutput $worker2Out -RedirectStandardError $worker2Err `
    -WindowStyle Hidden -PassThru

$peakGpuMemoryMiB = 0
while (-not ($worker1.HasExited -and $worker2.HasExited)) {
    $memoryText = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    $memoryMiB = [int](($memoryText | Select-Object -First 1).Trim())
    if ($memoryMiB -gt $peakGpuMemoryMiB) {
        $peakGpuMemoryMiB = $memoryMiB
    }
    Start-Sleep -Seconds 1
    $worker1.Refresh()
    $worker2.Refresh()
}
$wallSeconds = ((Get-Date) - $started).TotalSeconds
$worker1.WaitForExit()
$worker2.WaitForExit()

function Read-WorkerResult([string]$stdoutPath, [System.Diagnostics.Process]$process) {
    $stdout = Get-Content -LiteralPath $stdoutPath -Raw
    $match = [regex]::Match($stdout, '(?m)^result=(.+)$')
    $resultPath = if ($match.Success) { $match.Groups[1].Value.Trim() } else { $null }
    $benchmark = if ($resultPath -and (Test-Path -LiteralPath $resultPath)) {
        Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
    } else {
        $null
    }
    return [ordered]@{
        process_id = $process.Id
        exit_code = $process.ExitCode
        stdout = (Resolve-Path -LiteralPath $stdoutPath).Path
        result = $resultPath
        benchmark = $benchmark
    }
}

$result1 = Read-WorkerResult $worker1Out $worker1
$result2 = Read-WorkerResult $worker2Out $worker2
$valid = (
    $null -ne $result1.benchmark -and
    $null -ne $result2.benchmark -and
    $result1.benchmark.variants.$Variant.return_code -eq 0 -and
    $result2.benchmark.variants.$Variant.return_code -eq 0 -and
    $result1.benchmark.variants.$Variant.validation.valid -and
    $result2.benchmark.variants.$Variant.validation.valid
)
$baselineWallPer20 = 93.04274310008623
$sequentialBaselineEstimate = 2.0 * $baselineWallPer20
$summary = [ordered]@{
    started = $started.ToString('o')
    completed = (Get-Date).ToString('o')
    start_index = $StartIndex
    count_per_worker = $CountPerWorker
    total_items = 2 * $CountPerWorker
    variant = $Variant
    wall_seconds = $wallSeconds
    peak_gpu_memory_mib = $peakGpuMemoryMiB
    baseline_sequential_wall_seconds_for_40 = $sequentialBaselineEstimate
    speedup_vs_sequential_baseline = $sequentialBaselineEstimate / $wallSeconds
    switch_threshold = 1.0 / 0.85
    recommend_dual = $valid -and (($sequentialBaselineEstimate / $wallSeconds) -ge (1.0 / 0.85))
    valid = $valid
    workers = @($result1, $result2)
}
$output = Join-Path $runDir 'benchmark.json'
$summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $output -Encoding UTF8
Write-Output "result=$((Resolve-Path -LiteralPath $output).Path)"
Write-Output "wall_seconds=$wallSeconds"
Write-Output "peak_gpu_memory_mib=$peakGpuMemoryMiB"
Write-Output "speedup=$($summary.speedup_vs_sequential_baseline)"
Write-Output "recommend_dual=$($summary.recommend_dual)"
