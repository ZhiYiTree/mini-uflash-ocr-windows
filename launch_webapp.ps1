param(
    [switch]$NoBrowser,
    [switch]$Background
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PidFile = Join-Path $ProjectRoot "webapp.pid"
$LogDir = Join-Path $ProjectRoot "logs"
$Address = "http://127.0.0.1:7860"
$env:NO_PROXY = "localhost,127.0.0.1,::1"
$env:no_proxy = $env:NO_PROXY
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

try {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "未找到项目 Python：$Python。请先运行 .\install_windows.ps1"
    }
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

    if (Test-Path -LiteralPath $PidFile) {
        $oldPid = 0
        [void][int]::TryParse((Get-Content -LiteralPath $PidFile -Raw).Trim(), [ref]$oldPid)
        if ($oldPid -gt 0) {
            $old = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
            if ($old -and ($old.CommandLine -match "webapp\.app") -and
                (($old.ExecutablePath -eq $Python) -or ($old.CommandLine -like "*$ProjectRoot*"))) {
                Write-Host "网页已在运行（PID $oldPid）：$Address" -ForegroundColor Yellow
                exit 0
            }
        }
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    }

    $stdout = Join-Path $LogDir "webapp.stdout.log"
    $stderr = Join-Path $LogDir "webapp.stderr.log"
    $start = @{
        FilePath = $Python
        ArgumentList = @("-m", "webapp.app")
        WorkingDirectory = $ProjectRoot
        PassThru = $true
        RedirectStandardOutput = $stdout
        RedirectStandardError = $stderr
    }
    if ($Background) { $start.WindowStyle = "Hidden" }
    $process = Start-Process @start
    Set-Content -LiteralPath $PidFile -Value $process.Id -Encoding ascii

    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        if ($process.HasExited) { break }
        try {
            $response = Invoke-WebRequest -Uri $Address -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) { $ready = $true; break }
        } catch {
            $null = $_
        }
        Start-Sleep -Seconds 1
    }
    if (-not $ready) {
        if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force }
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
        $tail = if (Test-Path $stderr) { (Get-Content $stderr -Tail 30) -join [Environment]::NewLine } else { "" }
        throw "网页未能在 60 秒内启动。日志：$stderr`n$tail"
    }

    Write-Host "Mini UFlash OCR 已启动（PID $($process.Id)）" -ForegroundColor Green
    Write-Host "访问地址：$Address" -ForegroundColor Cyan
    if (-not $NoBrowser) { Start-Process $Address }
    if (-not $Background) {
        Write-Host "按 Ctrl+C 停止前台服务。" -ForegroundColor DarkGray
        Wait-Process -Id $process.Id
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    }
    exit 0
} catch {
    Write-Error $_
    exit 1
}
