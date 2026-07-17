from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch

from .model import DFlashOCRConfig, MaskBlockDrafter


def save_checkpoint(
    path: str | Path,
    model: MaskBlockDrafter,
    step: int,
    metrics: Dict[str, object],
    extra: Optional[Dict[str, object]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "mini_uflash_v2_dflash_ocr",
            "config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "step": int(step),
            "metrics": metrics,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[MaskBlockDrafter, Dict[str, object]]:
    value = torch.load(path, map_location=device, weights_only=False)
    if value.get("format") != "mini_uflash_v2_dflash_ocr":
        raise ValueError("Unsupported checkpoint format")
    raw_cfg = dict(value.get("config") or {})
    # Newer heads (Markov / expanded conf) are additive; default them on for
    # 8GB DSpark-style refine even when loading R3-era checkpoints.
    raw_cfg.setdefault("use_markov", True)
    raw_cfg.setdefault("markov_rank", 64)
    config = DFlashOCRConfig.from_dict(raw_cfg)
    model = MaskBlockDrafter(config)
    missing, unexpected = model.load_state_dict(value["state_dict"], strict=False)
    # Ignore brand-new Markov / head keys when loading older weights.
    if unexpected:
        raise ValueError(f"Unexpected keys in checkpoint: {unexpected[:8]}")
    value = dict(value)
    value["load_missing_keys"] = list(missing)
    return model, value
