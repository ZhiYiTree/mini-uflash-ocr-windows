$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ReportDir = Join-Path $ProjectRoot "reports"
$Report = Join-Path $ReportDir "environment_windows.txt"

try {
    New-Item -ItemType Directory -Path $ReportDir -Force | Out-Null
    $lines = [System.Collections.Generic.List[string]]::new()
    $os = Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion, OsBuildNumber, OsArchitecture
    $lines.Add(($os | Format-List | Out-String).Trim())
    $lines.Add("nvidia-smi:")
    $lines.Add(((& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader) | Out-String).Trim())
    $lines.Add("py --list:")
    $lines.Add(((& py --list) | Out-String).Trim())
    if (-not (Test-Path -LiteralPath $Python)) { throw "未找到项目解释器：$Python" }
    $lines.Add("Project Python:")
    $lines.Add(((& $Python -c "import sys,torch; print(sys.version); print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')") | Out-String).Trim())
    Set-Content -LiteralPath $Report -Value $lines -Encoding utf8
    Get-Content -LiteralPath $Report
    Write-Host "环境报告：$Report" -ForegroundColor Green
    exit 0
} catch {
    Write-Error $_
    exit 1
}
