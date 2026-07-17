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

1. **GitHub Release** (if published): download the production `.pt` and put it in this folder.
2. **Train yourself**: see `train/README.md` (Windows 8GB continue pipeline).
3. **Env override**: set `MINI_UFLASH_WEIGHT` to the absolute path of any checkpoint.

Unlimited-OCR (≈6GB+) goes under `models/PaddlePaddle/Unlimited-OCR` and is also not in Git — use `download_models.ps1 -AllowLargeDownload` or set `UNLIMITED_OCR_PATH`.
