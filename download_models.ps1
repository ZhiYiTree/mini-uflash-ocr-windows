param([switch]$AllowLargeDownload)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$ModelPath = Join-Path $ProjectRoot "models\PaddlePaddle\Unlimited-OCR"
$WeightPath = Join-Path $ProjectRoot "weights\mini-uflash-v2.0.0-alpha.1-drafter-stage11b-best.pt"

Write-Host "Unlimited-OCR：$ModelPath"
Write-Host "Stage 11B：$WeightPath"
if (Test-Path -LiteralPath $ModelPath) { Write-Host "Unlimited-OCR 已存在。" -ForegroundColor Green }
elseif (-not $AllowLargeDownload) {
    Write-Host "未下载 6GB+ 模型。请把本地 Unlimited-OCR 放到上述目录，或显式使用 -AllowLargeDownload。" -ForegroundColor Yellow
} else {
    $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $Python)) { throw "请先运行 install_windows.ps1" }
    & $Python -c "from huggingface_hub import snapshot_download; snapshot_download('baidu/Unlimited-OCR', local_dir=r'$ModelPath')"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
if (Test-Path -LiteralPath $WeightPath) { Write-Host "Stage 11B 权重已存在。" -ForegroundColor Green }
else { Write-Host "Stage 11B 权重缺失；稳定模式仍可使用。" -ForegroundColor Yellow }
