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
    echo [1/3] No venv found. Running install_windows.ps1 ...
    echo.
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_windows.ps1"
    if errorlevel 1 (
        echo.
        echo Install failed. Try: .\setup_full.ps1
        pause
        exit /b 1
    )
) else (
    echo [1/3] Python venv OK
)

echo.
echo [2/3] Checking Unlimited-OCR model ...
if exist "models\PaddlePaddle\Unlimited-OCR\config.json" (
    echo Model folder OK
) else (
    echo Model missing. Downloading via download_models.ps1 ...
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download_models.ps1"
    if errorlevel 1 (
        echo.
        echo Model download failed. If you already have the model, place it at:
        echo   models\PaddlePaddle\Unlimited-OCR
        echo Then run this bat again.
        pause
        exit /b 1
    )
)

echo.
echo [3/3] Starting web UI ...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch_webapp.ps1"
if errorlevel 1 (
    echo.
    echo Launch failed. See logs\webapp.stderr.log and logs\webapp.stdout.log
    pause
    exit /b 1
)

endlocal
