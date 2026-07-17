#!/usr/bin/env python3
"""Create a train/validation split manifest from teacher .pt files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train import config as cfg  # noqa: E402
from train.lib.teacher_data import discover_teacher_files, split_pages  # noqa: E402
from train.lib.utils import atomic_write_json  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Write page-isolated train/val split JSON")
    p.add_argument("--teacher", type=str, default=str(cfg.TEACHERS_TRAIN))
    p.add_argument("--output", type=str, default=str(cfg.SPLITS_DIR / "page_split.json"))
    p.add_argument("--val-ratio", type=float, default=cfg.TRAIN_VAL_RATIO)
    p.add_argument("--seed", type=int, default=cfg.TRAIN_SEED)
    args = p.parse_args()

    paths = discover_teacher_files(args.teacher)
    train, val = split_pages(paths, args.val_ratio, args.seed)
    payload = {
        "format": "mini_uflash_v2_page_split",
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "train": [p.name for p in train],
        "validation": [p.name for p in val],
        "train_count": len(train),
        "validation_count": len(val),
    }
    out = Path(args.output).expanduser().resolve()
    atomic_write_json(out, payload)
    print(f"Wrote {out}")
    print(f"train={len(train)} val={len(val)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
