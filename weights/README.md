# Mini UFlash weights

Binary checkpoints (`.pt` / `.pth`) are **not** tracked by Git. Place them here after clone.

## Recommended production weight (this project)

| File | Role |
| --- | --- |
| `mini-uflash-win-domain-continue-best.pt` | Production drafter (R5 domain-continue; used by webapp auto-discovery) |

Optional backups you may keep locally:

- `mini-uflash-win-domain-continue-r5-pos0-best.pt`
- `mini-uflash-win-domain-continue-r4-p012-best.pt`
- earlier r2/r3 checkpoints

Upstream research baseline (optional):

- `mini-uflash-v2.0.0-alpha.1-drafter-stage11b-best.pt`

`drafter_last.pt` style files are training snapshots; prefer `*-best.pt`.

## How to obtain

1. **推荐**：项目根目录运行 `.\download_models.ps1` 或 `.\setup_full.ps1`  
   - 自动从 GitHub Release `v1.0.0` 拉取  
     `mini-uflash-win-domain-continue-best.pt`  
   - SHA256：`5D9DAA2749B5C9C770724ECBA7BEB2627FC398CB47BE67337FDBD5B5FFC4B079`
2. **手动**：打开仓库 Releases 页下载同名文件到本目录。
3. **Train yourself**：见 `train/README.md`（Windows 8GB continue）。
4. **Env override**：`MINI_UFLASH_WEIGHT` = 任意 `.pt` 绝对路径。

Unlimited-OCR（≈6GB+）不在 Git 中，由 `download_models.ps1` 从 Hugging Face `baidu/Unlimited-OCR` 装到  
`models/PaddlePaddle/Unlimited-OCR`。
