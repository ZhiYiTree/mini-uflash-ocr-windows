<#
.SYNOPSIS
    Mini UFlash OCR — Windows installation script.
.DESCRIPTION
    Creates a Python virtual environment (if missing) and installs all
    dependencies. The script never replaces a working CUDA PyTorch with a
    CPU build. All commands use the project venv exclusively.
.NOTES
    Run this once after cloning the repository:
        Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
        .\install_windows.ps1
#>

$ErrorActionPreference = "Stop"
trap {
    Write-Host "安装失败：$($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = Get-Location }

$VenvDir  = Join-Path $ProjectRoot ".venv"
$Python   = Join-Path $VenvDir "Scripts\python.exe"
$Pip      = Join-Path $VenvDir "Scripts\pip.exe"

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host " Mini UFlash OCR — Windows Install"  -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host " Project : $ProjectRoot"
Write-Host ""

# ---- Step 1: Create / verify venv ----------------------------------------
if (-not (Test-Path $Python)) {
    Write-Host "[1/6] Creating virtual environment ..." -ForegroundColor Yellow
    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if (-not $pyCmd) {
        Write-Error "py launcher not found. Install Python 3.12+ from python.org"
        exit 1
    }
    py -3 -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create venv"
        exit 1
    }
    Write-Host "        OK" -ForegroundColor Green
} else {
    Write-Host "[1/6] Virtual environment exists: $VenvDir" -ForegroundColor Green
}

# ---- Step 2: Upgrade pip -------------------------------------------------
Write-Host "[2/6] Upgrading pip ..." -ForegroundColor Yellow
& $Python -m pip install --upgrade pip --quiet 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Error "pip upgrade failed"
    exit 1
}
Write-Host "        OK" -ForegroundColor Green

# ---- Step 3: Check NVIDIA GPU --------------------------------------------
Write-Host "[3/6] Checking NVIDIA GPU ..." -ForegroundColor Yellow
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    $gpuInfo = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
    if ($gpuInfo) {
        Write-Host "        GPU: $($gpuInfo.Trim())" -ForegroundColor Green
    } else {
        Write-Host "        WARNING: nvidia-smi found but no GPU detected" -ForegroundColor Yellow
    }
} else {
    Write-Host "        WARNING: nvidia-smi not found (CUDA may not be available)" -ForegroundColor Yellow
}

# ---- Step 4: Install PyTorch (CUDA) --------------------------------------
Write-Host "[4/6] Installing PyTorch (CUDA 12.8) ..." -ForegroundColor Yellow
# Check if torch is already installed with CUDA support
$torchCuda = & $Python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>$null
if ($torchCuda -eq "cuda") {
    $torchVer = & $Python -c "import torch; print(torch.__version__)" 2>$null
    Write-Host "        Torch $torchVer with CUDA already installed — skipping" -ForegroundColor Green
} else {
    & $Python -m pip install torch --index-url https://download.pytorch.org/whl/cu128 --quiet 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "PyTorch CUDA install failed"
        exit 1
    }
    $torchVer = & $Python -c "import torch; print(torch.__version__)" 2>$null
    Write-Host "        OK (torch $torchVer)" -ForegroundColor Green
}

# ---- Step 5: Check CUDA works -------------------------------------------
Write-Host "[5/6] Verifying CUDA ..." -ForegroundColor Yellow
$cudaCheck = & $Python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>$null
Write-Host "        $cudaCheck" -ForegroundColor Green

# torchvision must come from the same CUDA wheel index as torch. Keeping it
# outside requirements-windows.txt prevents a PyPI dependency resolution from
# replacing the working CUDA torch build.
$visionOk = & $Python -c "import torchvision; print(torchvision.__version__)" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "        Installing matching CUDA torchvision ..." -ForegroundColor Yellow
    & $Python -m pip install torchvision --index-url https://download.pytorch.org/whl/cu128 --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Error "CUDA torchvision install failed"
        exit 1
    }
}

# ---- Step 6: Install project dependencies ---------------------------------
Write-Host "[6/6] Installing project requirements ..." -ForegroundColor Yellow
$reqFile = Join-Path $ProjectRoot "requirements-windows.txt"
if (-not (Test-Path $reqFile)) {
    Write-Error "requirements-windows.txt not found at $reqFile"
    exit 1
}
& $Python -m pip install -r $reqFile --quiet 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Requirements install failed"
    exit 1
}
Write-Host "        OK" -ForegroundColor Green

# ---- Audioop fix for Python 3.13 -----------------------------------------
$audioopCheck = & $Python -c "import audioop" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[fix]  Installing audioop-lts for Python 3.13 ..." -ForegroundColor Yellow
    & $Python -m pip install audioop-lts --quiet 2>&1 | Out-Null
    Write-Host "        OK" -ForegroundColor Green
}

# ---- Summary -------------------------------------------------------------
Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host " Installation complete!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  .\launch_webapp.ps1" -ForegroundColor Yellow
Write-Host ""
exit 0
