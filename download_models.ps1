<#
.SYNOPSIS
    下载 Unlimited-OCR 与生产 drafter 权重，使前端可直接使用。
.DESCRIPTION
    Unlimited-OCR 约 6GB+，无法放入 Git 仓库；默认从 Hugging Face 拉取。
    生产 Mini UFlash 权重（~32MB）从本仓库 GitHub Release 拉取。

    用法：
        .\download_models.ps1              # 完整下载（推荐）
        .\download_models.ps1 -SkipWeight  # 只下 Unlimited-OCR
        .\download_models.ps1 -CheckOnly   # 仅检查，不下载
#>
param(
    [switch]$CheckOnly,
    [switch]$SkipWeight,
    [switch]$SkipOcr,
    # 兼容旧参数名：历史上默认不下载，现已改为默认完整下载
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

# Production weight on GitHub Releases (update tag when re-publishing)
$WeightReleaseTag = "v1.0.0"
$WeightReleaseUrl = "https://github.com/ZhiYiTree/mini-uflash-ocr-windows/releases/download/$WeightReleaseTag/mini-uflash-win-domain-continue-best.pt"
$WeightSha256 = "5D9DAA2749B5C9C770724ECBA7BEB2627FC398CB47BE67337FDBD5B5FFC4B079"

# Prefer known-good HF / ModelScope style id used by Paddle Unlimited-OCR packaging
$OcrRepoId = if ($env:UNLIMITED_OCR_HF_REPO) { $env:UNLIMITED_OCR_HF_REPO } else { "baidu/Unlimited-OCR" }

function Test-OcrReady {
    param([string]$Dir)
    if (-not (Test-Path -LiteralPath $Dir)) { return $false }
    $cfg = Join-Path $Dir "config.json"
    $weights = Get-ChildItem -LiteralPath $Dir -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '\.(safetensors|bin|pt)$' -or $_.Name -eq 'model.safetensors.index.json' }
    return (Test-Path -LiteralPath $cfg) -and ($null -ne $weights -and $weights.Count -gt 0)
}

function Test-AnyWeight {
    if (Test-Path -LiteralPath $WeightProd) { return $true }
    if (Test-Path -LiteralPath $WeightUpstream) { return $true }
    $any = Get-ChildItem -LiteralPath $WeightsDir -Filter "*.pt" -ErrorAction SilentlyContinue
    return ($null -ne $any -and $any.Count -gt 0)
}

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host " Mini UFlash — 模型 / 权重下载" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host " Unlimited-OCR : $ModelPath"
Write-Host " 生产权重      : $WeightProd"
Write-Host ""

$ocrOk = Test-OcrReady $ModelPath
$weightOk = Test-AnyWeight

if ($ocrOk) {
    Write-Host "[OCR]  已就绪" -ForegroundColor Green
} else {
    Write-Host "[OCR]  未就绪（约 6GB+，首次需下载）" -ForegroundColor Yellow
}
if ($weightOk) {
    Write-Host "[权重] 已就绪" -ForegroundColor Green
} else {
    Write-Host "[权重] 未就绪（约 32MB；无权重时仅普通稳定模式可用）" -ForegroundColor Yellow
}

if ($CheckOnly) {
    if ($ocrOk -and $weightOk) { exit 0 }
    exit 2
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "未找到 .venv\Scripts\python.exe。请先运行 .\install_windows.ps1"
}

# ---- Unlimited-OCR -------------------------------------------------------
if (-not $SkipOcr -and -not $ocrOk) {
    Write-Host ""
    Write-Host "[1/2] 正在下载 Unlimited-OCR（$OcrRepoId）…" -ForegroundColor Yellow
    Write-Host "      体积约 6GB+，需可访问 Hugging Face；可能需要较长时间。" -ForegroundColor DarkGray
    New-Item -ItemType Directory -Path (Split-Path $ModelPath -Parent) -Force | Out-Null
    $env:UNLIMITED_OCR_HF_REPO = $OcrRepoId
    # One-liner keeps PowerShell quoting simple on Windows.
    & $Python -c "from huggingface_hub import snapshot_download; import os; r=os.environ.get('UNLIMITED_OCR_HF_REPO', r'$OcrRepoId'); d=r'$ModelPath'; print('snapshot_download', r, '->', d, flush=True); snapshot_download(repo_id=r, local_dir=d, local_dir_use_symlinks=False); print('done', flush=True)"
    if ($LASTEXITCODE -ne 0) {
        throw "Unlimited-OCR 下载失败。可设置 HF_ENDPOINT / 代理后重试，或手动放到：$ModelPath"
    }
    if (-not (Test-OcrReady $ModelPath)) {
        throw "下载完成但目录不完整：$ModelPath"
    }
    Write-Host "      Unlimited-OCR 就绪" -ForegroundColor Green
} elseif ($SkipOcr) {
    Write-Host "[1/2] 跳过 Unlimited-OCR（-SkipOcr）" -ForegroundColor DarkGray
}

# ---- Production drafter weight -------------------------------------------
if (-not $SkipWeight -and -not (Test-Path -LiteralPath $WeightProd)) {
    Write-Host ""
    Write-Host "[2/2] 正在下载生产 drafter 权重（Release $WeightReleaseTag）…" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $WeightsDir -Force | Out-Null
    $tmp = Join-Path $WeightsDir "_download_weight.tmp"
    try {
        Invoke-WebRequest -Uri $WeightReleaseUrl -OutFile $tmp -UseBasicParsing
        $hash = (Get-FileHash -LiteralPath $tmp -Algorithm SHA256).Hash.ToUpperInvariant()
        if ($WeightSha256 -and ($hash -ne $WeightSha256.ToUpperInvariant())) {
            throw "权重 SHA256 不匹配。期望 $WeightSha256，实际 $hash"
        }
        Move-Item -LiteralPath $tmp -Destination $WeightProd -Force
        Write-Host "      权重就绪：$WeightProd" -ForegroundColor Green
        Write-Host "      SHA256 $hash" -ForegroundColor DarkGray
    } catch {
        if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }
        Write-Host "      Release 下载失败：$($_.Exception.Message)" -ForegroundColor Yellow
        Write-Host "      请手动将 mini-uflash-win-domain-continue-best.pt 放到 weights\" -ForegroundColor Yellow
        Write-Host "      或从： $WeightReleaseUrl" -ForegroundColor Yellow
        if (-not (Test-AnyWeight)) {
            Write-Host "      警告：无 drafter 时加速版不可用，普通稳定模式仍可 OCR。" -ForegroundColor Yellow
        }
    }
} elseif ($SkipWeight) {
    Write-Host "[2/2] 跳过权重（-SkipWeight）" -ForegroundColor DarkGray
} else {
    Write-Host "[2/2] 生产权重已存在，跳过" -ForegroundColor Green
}

Write-Host ""
Write-Host "完成。启动前端：双击 启动前端.bat  或  .\launch_webapp.ps1" -ForegroundColor Cyan
exit 0
