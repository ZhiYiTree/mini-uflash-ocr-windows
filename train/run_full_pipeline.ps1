# Full local pipeline: prepare → collect pages → extract → split → train.
# NOT executed by scaffolding. Run only when ready for overnight training.
#
# Usage:
#   .\train\run_full_pipeline.ps1 -WhatIf
#   .\train\run_full_pipeline.ps1 -ExtractLimit 120 -TrainSteps 3000

param(
    [int]$ExtractLimit = 120,
    [int]$TrainSteps = 3000,
    [switch]$SkipCollect,
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Missing $Py" }

Write-Host "=== Full Mini UFlash Windows pipeline ===" -ForegroundColor Cyan
Write-Host "ExtractLimit=$ExtractLimit TrainSteps=$TrainSteps WhatIf=$WhatIf"

if ($WhatIf) {
    Write-Host "[WhatIf] Would: prepare dirs, optional collect, extract $ExtractLimit pages, make split, train $TrainSteps steps"
    exit 0
}

& (Join-Path $PSScriptRoot "run_prepare.ps1")
if (-not $SkipCollect) {
    & $Py "train\collect_pages.py"
}
& $Py "train\extract_teachers.py" "--limit" "$ExtractLimit"
if ($LASTEXITCODE -ne 0) { throw "extract_teachers failed with $LASTEXITCODE" }
& $Py "train\make_split.py"
& $Py "train\train_continue_8g.py" "--steps" "$TrainSteps" "--split-manifest" "train\data\splits\page_split.json"
exit $LASTEXITCODE
