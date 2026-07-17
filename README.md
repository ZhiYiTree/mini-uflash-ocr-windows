# Mini UFlash OCR（Windows 原生 · 完整可用）

本机 OCR 网页工作台：**官方 Unlimited-OCR 稳定识别** + 可选 **Stable DFlash 加速**（快速 / 均衡 / 无损三档）。

不依赖 WSL、Docker、Flash Attention。适合 Windows 11 + NVIDIA GPU（约 8GB 显存可跑）。

> **仓库不含 6GB+ 模型文件**（GitHub 单文件限制）。克隆后运行安装脚本会**自动下载** Unlimited-OCR（Hugging Face）与生产加速权重（GitHub Release），装完即可用网页。

## 给新用户：三步直接用

```powershell
# 1. 克隆
git clone https://github.com/ZhiYiTree/mini-uflash-ocr-windows.git
cd mini-uflash-ocr-windows

# 2. 完整安装（环境 + Unlimited-OCR ≈6GB + 加速权重 ≈32MB）
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_full.ps1

# 3. 启动网页
.\launch_webapp.ps1
# 或双击「启动前端.bat」（缺环境/模型时会自动补齐）
```

浏览器打开 <http://127.0.0.1:7860>，上传图片或 PDF 即可。

| 处理方式 | 说明 |
| --- | --- |
| **普通版 · 稳定** | 官方 Unlimited-OCR，质量基准 |
| **加速版 · 稳定 DFlash** | 验证前缀投机解码；档位见下 |

加速档位（加速版可见）：

| 档位 | 含义 |
| --- | --- |
| **快速**（默认） | 墙钟优先，可 soft 截断；金标准约 1.7× |
| **均衡** | 更完整一点；约 1.5× |
| **无损** | 无 soft 截断；当前 live 接受率下未必更快 |

## 本机路径

| 内容 | 路径 | 如何获得 |
| --- | --- | --- |
| Unlimited-OCR | `models\PaddlePaddle\Unlimited-OCR` | `download_models.ps1` / HF `baidu/Unlimited-OCR` |
| 生产 drafter | `weights\mini-uflash-win-domain-continue-best.pt` | GitHub Release `v1.0.0` |
| 识别输出 | `webapp\outputs\...` | 运行时生成 |

环境变量（可选）：

- `UNLIMITED_OCR_PATH` — 模型目录
- `MINI_UFLASH_WEIGHT` — drafter 权重文件
- `UNLIMITED_OCR_HF_REPO` — 覆盖 HF 仓库 id
- `MINI_UFLASH_SKIP_MODELS=1` — `install_windows.ps1` 跳过自动下模型

## 分步安装（可选）

```powershell
.\install_windows.ps1          # Python venv + CUDA torch + 依赖（默认也会尝试下模型）
.\download_models.ps1          # 仅下载 / 补齐模型与权重
.\download_models.ps1 -CheckOnly
.\check_environment.ps1
```

后台启动：`.\launch_webapp.ps1 -Background -NoBrowser`。停止：`.\stop_webapp.ps1`。

## 为什么不把 Unlimited-OCR 放进 Git？

| 资源 | 大约体积 | 处理方式 |
| --- | ---: | --- |
| Unlimited-OCR `model-*.safetensors` | **~6.3 GB** | Hugging Face 自动下载（Git / Release 单文件上限远小于此） |
| 生产 drafter `.pt` | **~32 MB** | **GitHub Release** 资产 + 安装脚本拉取 |
| 训练页图 / teacher | 可达数十 GB | 不发布；见 `train/` 自建 |

对使用者等价于「完整版」：克隆 → `setup_full.ps1` → 启动前端，无需手找模型。

## 兼容性

- Python 3.12/3.13；本机验证过 3.13 + CUDA 12.8 PyTorch。
- `transformers==4.46.3`（匹配 Unlimited-OCR remote code）。
- Attention：SDPA → eager。

## 验证

```powershell
.\.venv\Scripts\python.exe -m compileall webapp
.\.venv\Scripts\python.exe -m pytest webapp\tests -v
.\check_environment.ps1
.\download_models.ps1 -CheckOnly
.\launch_webapp.ps1 -Background -NoBrowser
Invoke-WebRequest http://127.0.0.1:7860 -UseBasicParsing
.\stop_webapp.ps1
```

## 本机续训（可选）

`train/` 为 Windows 8GB 域适配脚手架。说明见 `train/README.md`、`train/STATUS.md`、`train/GOLD_BASELINE.md`。

## 许可证与上游

- Unlimited-OCR：遵循其 Hugging Face / 官方许可证。
- 本仓库应用与训练脚手架：见仓库声明；提交 issue 前请说明 GPU / 驱动 / 是否完成 `setup_full.ps1`。
