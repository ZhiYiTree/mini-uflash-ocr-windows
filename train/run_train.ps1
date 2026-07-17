# Continue-train Stage 11B on Windows 8GB.
# Usage:
#   .\train\run_train.ps1 -DryRun
#   .\train\run_train.ps1 -Steps 500
#   .\train\run_train.ps1

param(
    [int]$Steps = 0,
    [int]$MicroBatch = 0,
    [int]$GradAccum = 0,
    [switch]$DryRun,
    [string]$Teacher = "",
    [string]$OutputDir = "",
    [string]$Resume = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Missing $Py" }

$argsList = @("train\train_continue_8g.py")
if ($Teacher) { $argsList += @("--teacher", $Teacher) }
if ($OutputDir) { $argsList += @("--output-dir", $OutputDir) }
if ($Resume) { $argsList += @("--resume-checkpoint", $Resume) }
if ($Steps -gt 0) { $argsList += @("--steps", "$Steps") }
if ($MicroBatch -gt 0) { $argsList += @("--micro-batch-size", "$MicroBatch") }
if ($GradAccum -gt 0) { $argsList += @("--grad-accum", "$GradAccum") }
if ($DryRun) { $argsList += "--dry-run" }

Write-Host "Running: $Py $($argsList -join ' ')" -ForegroundColor Cyan
& $Py @argsList
exit $LASTEXITCODE
