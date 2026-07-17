# Mini UFlash OCR - download Unlimited-OCR + production weight
# Usage:
#   .\download_models.ps1
#   .\download_models.ps1 -CheckOnly
#   .\download_models.ps1 -SkipWeight
param(
    [switch]$CheckOnly,
    [switch]$SkipWeight,
    [switch]$SkipOcr,
    [switch]$AllowLargeDownload
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = (Get-Location).Path }

$ModelPath = Join-Path $ProjectRoot "models\PaddlePaddle\Unlimited-OCR"
$WeightsDir = Join-Path $ProjectRoot "weights"
$WeightProd = Join-Path $WeightsDir "mini-uflash-win-domain-continue-best.pt"
$WeightUpstream = Join-Path $WeightsDir "mini-uflash-v2.0.0-alpha.1-drafter-stage11b-best.pt"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

$WeightReleaseTag = "v1.0.0"
$WeightReleaseUrl = "https://github.com/ZhiYiTree/mini-uflash-ocr-windows/releases/download/$WeightReleaseTag/mini-uflash-win-domain-continue-best.pt"
$WeightSha256 = "5D9DAA2749B5C9C770724ECBA7BEB2627FC398CB47BE67337FDBD5B5FFC4B079"
$OcrRepoId = if ($env:UNLIMITED_OCR_HF_REPO) { $env:UNLIMITED_OCR_HF_REPO } else { "baidu/Unlimited-OCR" }

function Test-OcrReady {
    param([string]$Dir)
    if (-not (Test-Path -LiteralPath $Dir)) { return $false }
    $cfg = Join-Path $Dir "config.json"
    if (-not (Test-Path -LiteralPath $cfg)) { return $false }
    $weights = Get-ChildItem -LiteralPath $Dir -File -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -match '\.(safetensors|bin|pt)$' -or
            $_.Name -eq 'model.safetensors.index.json'
        }
    return ($null -ne $weights -and @($weights).Count -gt 0)
}

function Test-AnyWeight {
    if (Test-Path -LiteralPath $WeightProd) { return $true }
    if (Test-Path -LiteralPath $WeightUpstream) { return $true }
    if (-not (Test-Path -LiteralPath $WeightsDir)) { return $false }
    $any = Get-ChildItem -LiteralPath $WeightsDir -Filter "*.pt" -ErrorAction SilentlyContinue
    return ($null -ne $any -and @($any).Count -gt 0)
}

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host " Mini UFlash - model / weight setup" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host " Unlimited-OCR : $ModelPath"
Write-Host " Prod weight   : $WeightProd"
Write-Host ""

$ocrOk = Test-OcrReady $ModelPath
$weightOk = Test-AnyWeight

if ($ocrOk) {
    Write-Host "OCR: ready" -ForegroundColor Green
} else {
    Write-Host "OCR: missing (~6GB+, first download required)" -ForegroundColor Yellow
}
if ($weightOk) {
    Write-Host "Weight: ready" -ForegroundColor Green
} else {
    Write-Host "Weight: missing (~32MB; stable OCR still works without it)" -ForegroundColor Yellow
}

if ($CheckOnly) {
    # Exit 0 if OCR is ready. Weight is optional for launching the web UI.
    if ($ocrOk) { exit 0 }
    exit 2
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python venv not found: $Python. Run install_windows.ps1 first."
}

# ---- Unlimited-OCR ----
if ((-not $SkipOcr) -and (-not $ocrOk)) {
    Write-Host ""
    Write-Host "Downloading Unlimited-OCR from Hugging Face: $OcrRepoId" -ForegroundColor Yellow
    Write-Host "This is ~6GB+ and may take a while." -ForegroundColor DarkGray
    New-Item -ItemType Directory -Path (Split-Path $ModelPath -Parent) -Force | Out-Null

    $helper = Join-Path $env:TEMP "mini_uflash_dl_ocr.py"
    $helperBody = @"
from huggingface_hub import snapshot_download
import os
repo = os.environ.get("UNLIMITED_OCR_HF_REPO", "$OcrRepoId")
local = r"$ModelPath"
print("snapshot_download", repo, "->", local, flush=True)
snapshot_download(repo_id=repo, local_dir=local, local_dir_use_symlinks=False)
print("done", flush=True)
"@
    Set-Content -LiteralPath $helper -Value $helperBody -Encoding UTF8
    $env:UNLIMITED_OCR_HF_REPO = $OcrRepoId
    & $Python $helper
    $code = $LASTEXITCODE
    Remove-Item -LiteralPath $helper -Force -ErrorAction SilentlyContinue
    if ($code -ne 0) {
        throw "Unlimited-OCR download failed. Check network / HF access, or copy model to: $ModelPath"
    }
    if (-not (Test-OcrReady $ModelPath)) {
        throw "Download finished but model folder looks incomplete: $ModelPath"
    }
    Write-Host "OCR ready." -ForegroundColor Green
} elseif ($SkipOcr) {
    Write-Host "Skip OCR download (-SkipOcr)." -ForegroundColor DarkGray
}

# ---- Production weight ----
if ((-not $SkipWeight) -and (-not (Test-Path -LiteralPath $WeightProd))) {
    Write-Host ""
    Write-Host "Downloading production weight (Release $WeightReleaseTag) ..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $WeightsDir -Force | Out-Null
    $tmp = Join-Path $WeightsDir "_download_weight.tmp"
    try {
        Invoke-WebRequest -Uri $WeightReleaseUrl -OutFile $tmp -UseBasicParsing
        $hash = (Get-FileHash -LiteralPath $tmp -Algorithm SHA256).Hash.ToUpperInvariant()
        if ($WeightSha256 -and ($hash -ne $WeightSha256.ToUpperInvariant())) {
            throw "Weight SHA256 mismatch. expected=$WeightSha256 actual=$hash"
        }
        Move-Item -LiteralPath $tmp -Destination $WeightProd -Force
        Write-Host "Weight ready: $WeightProd" -ForegroundColor Green
        Write-Host "SHA256 $hash" -ForegroundColor DarkGray
    } catch {
        if (Test-Path -LiteralPath $tmp) {
            Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
        }
        Write-Host "Release weight download failed: $($_.Exception.Message)" -ForegroundColor Yellow
        Write-Host "Manual URL: $WeightReleaseUrl" -ForegroundColor Yellow
        if (-not (Test-AnyWeight)) {
            Write-Host "No drafter weight; accel mode unavailable, stable OCR still works." -ForegroundColor Yellow
        }
    }
} elseif ($SkipWeight) {
    Write-Host "Skip weight download (-SkipWeight)." -ForegroundColor DarkGray
} else {
    Write-Host "Production weight already present." -ForegroundColor Green
}

Write-Host ""
Write-Host "Done. Start UI: double-click 启动前端.bat  or  .\launch_webapp.ps1" -ForegroundColor Cyan
exit 0
