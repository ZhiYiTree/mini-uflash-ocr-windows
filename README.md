# Mini UFlash OCR（Windows 原生）

本项目是 Windows 11 + NVIDIA GPU 的本地 OCR 网页工作台。普通模式调用本地 Unlimited-OCR 官方 `model.infer()`；加速版加载域适配 Stage 11B drafter，走 **稳定 DFlash**：目标模型验证草稿前缀 → 可裁剪提交 → **周期 prefill 重同步** + 退化熔断 + 低接受率时回退 B1。支持图片和逐页 PDF。

不使用 WSL、Docker、Bash、Flash Attention、Triton 或 xformers。模型 attention 按 SDPA → eager 尝试。旧版无界 Direct 仍保留在代码中供研究对比；网页加速入口已切换为稳定 DFlash。墙钟加速取决于页面接受率，长文优先正确收敛而非硬冲吞吐。

## 快速开始

普通使用：双击根目录的 `启动前端.bat`，浏览器会自动打开。

也可以使用 PowerShell：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\install_windows.ps1
.\launch_webapp.ps1
```

Python 前端入口：`.\.venv\Scripts\python.exe frontend.py`。各文件用途见 `文件说明.md`。

后台启动：`.\launch_webapp.ps1 -Background -NoBrowser`。停止：`.\stop_webapp.ps1`。地址固定为 <http://127.0.0.1:7860>，不创建公网分享链接。

## 本机路径

- Unlimited-OCR：`models\PaddlePaddle\Unlimited-OCR`，也可设置进程环境变量 `UNLIMITED_OCR_PATH` 或在网页填写。
- Stage 11B：`weights\mini-uflash-v2.0.0-alpha.1-drafter-stage11b-best.pt`，也可设置 `MINI_UFLASH_WEIGHT`。
- 输出：`webapp\outputs\YYYYMMDD_HHMMSS_random\`。

应用不会自动下载 6GB+ 模型。`download_models.ps1` 默认只检查路径；只有显式传入 `-AllowLargeDownload` 才会下载 Unlimited-OCR。

## 兼容性说明

本机只有 Python 3.13，因此当前 `.venv` 使用 Python 3.13；CUDA PyTorch 2.11.0 + cu128 已验证可用。`transformers==4.46.3` 依据本地模型 `config.json` 的 `transformers_version` 锁定。`torchvision` 由安装脚本从与 PyTorch 相同的 CUDA 12.8 wheel 源安装，防止 CPU 包覆盖 CUDA PyTorch。
Gradio 使用支持 Python 3.13 与新版 FastAPI/Starlette 的 5.x，并锁定 Pydantic 2.10.x。

加速实验版会对 PDF 逐页运行，模型和 Drafter 只加载一次；每页完成后立即保存，单页发生异常时自动回退普通 Unlimited-OCR。Direct Block 是非无损实验路径，可能出现漏字、错字或缓存漂移，请抽查重要内容。页面显示的 token/forward 是目标模型调用折算，不等同于端到端加速；实际收益取决于页面和草稿接受率。

## 验证

```powershell
.\.venv\Scripts\python.exe -m compileall webapp
.\.venv\Scripts\python.exe -m pytest webapp\tests -v
.\check_environment.ps1
.\launch_webapp.ps1 -Background -NoBrowser
Invoke-WebRequest http://127.0.0.1:7860 -UseBasicParsing
.\stop_webapp.ps1
```

## 本机续训（Windows 8GB，可选）

域适配 / Stage 11B 续训脚手架在 `train/`。默认**不启动**任何 GPU 训练；说明见 `train/README.md` 与 `train/STATUS.md`。准备环境：`.\train\run_prepare.ps1`。

## GitHub 发布（本仓库约定）

**会提交**

- `webapp/` 源码与测试、`train/` 训练脚手架与 manifest/文档
- PowerShell 安装/启动脚本、`requirements-windows.txt`、`README.md` / `文件说明.md`
- `weights/README.md`、`train/conf_calibration.json`（小配置，无权重）
- `reports/` 中不含隐私的环境/测速说明（若有）

**不会提交（见 `.gitignore`）**

| 内容 | 原因 |
| --- | --- |
| `models/` | Unlimited-OCR ≈6GB+ |
| `weights/*.pt` | drafter 权重，请用 Release 或本地训练 |
| `train/data/`、`train/runs/` | 页图/teacher/checkpoint 可达数十 GB |
| `webapp/outputs/`、`logs/`、`.venv/` | 运行产物与环境 |

克隆后：

```powershell
.\install_windows.ps1
# 准备 models\ 与 weights\（见 weights\README.md）
.\launch_webapp.ps1
```

发布 drafter 权重时：单独上传 **GitHub Release** 资产，并附 SHA-256；不要把 `.pt` 打进主仓库。
