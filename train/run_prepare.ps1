# Prepare training directories and validate environment. Does NOT extract or train.
# Usage:  .\train\run_prepare.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "train\config.py"))) {
    $Root = $PSScriptRoot
    if (-not (Test-Path (Join-Path $Root "config.py"))) {
        throw "Cannot locate project root (expected train\config.py)"
    }
    $Root = Split-Path -Parent $Root
}

Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    throw "Missing venv python: $Py  (run install_windows.ps1 first)"
}

Write-Host "=== Mini UFlash train prepare (no GPU training) ===" -ForegroundColor Cyan
Write-Host "Root: $Root"

& $Py -c @"
from pathlib import Path
import train.config as c
dirs = c.ensure_data_dirs()
print('Data dirs ready:')
for k, v in dirs.items():
    print(f'  {k}: {v}')
print('Model path :', c.model_path(), 'exists=', c.model_path().is_dir())
print('Resume wt  :', c.resume_weight(), 'exists=', c.resume_weight().is_file())
"@

Write-Host ""
Write-Host "Next steps (when you are ready):" -ForegroundColor Yellow
Write-Host "  1. Put page images into train\data\pages\pool  (or run collect_pages)"
Write-Host "  2. .\train\run_extract.ps1 -Limit 10 -DryRun"
Write-Host "  3. .\train\run_extract.ps1 -Limit 10"
Write-Host "  4. .\train\run_train.ps1 -DryRun"
Write-Host "  5. .\train\run_train.ps1"
Write-Host ""
Write-Host "STATUS: scaffolding ready; extraction/training NOT started." -ForegroundColor Green
