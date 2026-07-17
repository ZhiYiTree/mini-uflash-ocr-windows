@echo off
setlocal
chcp 65001 >nul
title Mini UFlash OCR

cd /d "%~dp0"

echo.
echo =====================================
echo  Mini UFlash OCR
echo =====================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] 未检测到运行环境，开始完整安装（含依赖）...
    echo.
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_windows.ps1"
    if errorlevel 1 (
        echo.
        echo 环境安装失败。也可手动运行 setup_full.ps1
        pause
        exit /b 1
    )
) else (
    echo [1/3] 运行环境已就绪
)

echo.
echo [2/3] 检查 Unlimited-OCR 与加速权重...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download_models.ps1" -CheckOnly
if errorlevel 1 (
    echo 模型未齐全，开始自动下载（Unlimited-OCR 约 6GB，首次较慢）...
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download_models.ps1"
    if errorlevel 1 (
        echo.
        echo 模型下载失败。请检查网络 / Hugging Face 访问，或手动运行：
        echo   .\download_models.ps1
        pause
        exit /b 1
    )
) else (
    echo 模型与权重已就绪
)

echo.
echo [3/3] 正在启动网页...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch_webapp.ps1"
if errorlevel 1 (
    echo.
    echo 启动失败，请查看 logs 文件夹中的日志。
    pause
    exit /b 1
)

endlocal
