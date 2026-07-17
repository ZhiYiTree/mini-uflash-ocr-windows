<#
.SYNOPSIS
    一键完整安装：Python 环境 + Unlimited-OCR + 生产权重。
.DESCRIPTION
    给新用户用。跑完后即可启动前端网页识别。

        Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
        .\setup_full.ps1
        .\launch_webapp.ps1
#>
$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = (Get-Location).Path }
Set-Location $ProjectRoot

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host " Mini UFlash OCR — 完整安装" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host " 将安装依赖、下载 Unlimited-OCR（~6GB）与加速权重（~32MB）"
Write-Host ""

& (Join-Path $ProjectRoot "install_windows.ps1")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& (Join-Path $ProjectRoot "download_models.ps1")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "=====================================" -ForegroundColor Green
Write-Host " 完整安装成功" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host " 下一步：双击「启动前端.bat」或执行 .\launch_webapp.ps1"
Write-Host " 地址：http://127.0.0.1:7860"
Write-Host ""
exit 0
