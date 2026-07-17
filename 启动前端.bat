@echo off
setlocal
chcp 65001 >nul
title Mini UFlash OCR

cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [提示] 尚未安装运行环境，请先运行 install_windows.ps1。
    echo.
    pause
    exit /b 1
)

echo 正在启动 Mini UFlash OCR...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch_webapp.ps1"
if errorlevel 1 (
    echo.
    echo 启动失败，请查看 logs 文件夹中的日志。
    pause
    exit /b 1
)

endlocal
