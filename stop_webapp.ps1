$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PidFile = Join-Path $ProjectRoot "webapp.pid"

try {
    if (-not (Test-Path -LiteralPath $PidFile -PathType Leaf)) {
        Write-Host "未找到 webapp.pid；当前没有由本项目记录的网页进程。" -ForegroundColor Yellow
        exit 0
    }
    $stored = (Get-Content -LiteralPath $PidFile -Raw).Trim()
    $pidValue = 0
    if (-not [int]::TryParse($stored, [ref]$pidValue) -or $pidValue -le 0) {
        throw "PID 文件内容无效：$stored"
    }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue
    if (-not $process) {
        Remove-Item -LiteralPath $PidFile -Force
        Write-Host "进程 $pidValue 已结束；已清理 PID 文件。" -ForegroundColor Yellow
        exit 0
    }
    $ours = ($process.CommandLine -match "webapp\.app") -and
            (($process.ExecutablePath -eq $Python) -or ($process.CommandLine -like "*$ProjectRoot*"))
    if (-not $ours) {
        throw "拒绝终止 PID $pidValue：命令行不属于当前 Mini UFlash 项目。"
    }
    Stop-Process -Id $pidValue -Force
    Remove-Item -LiteralPath $PidFile -Force
    Write-Host "已停止 Mini UFlash OCR（PID $pidValue）。" -ForegroundColor Green
    exit 0
} catch {
    Write-Error $_
    exit 1
}
