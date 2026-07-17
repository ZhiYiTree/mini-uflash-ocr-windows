# Extract teacher features. Default is safe: you must pass -Limit or omit DryRun carefully.
# Usage:
#   .\train\run_extract.ps1 -DryRun
#   .\train\run_extract.ps1 -Limit 10
#   .\train\run_extract.ps1 -Limit 100 -Recursive

param(
    [int]$Limit = 0,
    [int]$Offset = 0,
    [switch]$DryRun,
    [switch]$Recursive,
    [switch]$Overwrite,
    [string]$ImageDir = "",
    [string]$TeacherDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Missing $Py" }

$argsList = @("train\extract_teachers.py")
if ($ImageDir) { $argsList += @("--image-dir", $ImageDir) }
if ($TeacherDir) { $argsList += @("--teacher-dir", $TeacherDir) }
if ($Limit -gt 0) { $argsList += @("--limit", "$Limit") }
if ($Offset -gt 0) { $argsList += @("--offset", "$Offset") }
if ($DryRun) { $argsList += "--dry-run" }
if ($Recursive) { $argsList += "--recursive" }
if ($Overwrite) { $argsList += "--overwrite" }

Write-Host "Running: $Py $($argsList -join ' ')" -ForegroundColor Cyan
& $Py @argsList
exit $LASTEXITCODE
