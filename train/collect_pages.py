#!/usr/bin/env python3
"""Collect existing PNG pages from webapp outputs into train/data/pages/pool.

Does not run OCR or training. Safe to run anytime.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train import config as cfg  # noqa: E402
from train.lib.utils import safe_stem  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Copy page PNGs into the training pool")
    p.add_argument(
        "--source",
        type=str,
        default=str(cfg.PROJECT_ROOT / "webapp" / "outputs"),
        help="Root to scan for pages/*.png (default: webapp/outputs)",
    )
    p.add_argument("--dest", type=str, default=str(cfg.PAGES_POOL))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    source = Path(args.source).expanduser().resolve()
    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    images = sorted(source.rglob("p*.png")) + sorted(source.rglob("pages/*.png"))
    # Deduplicate by resolve
    seen = set()
    unique = []
    for img in images:
        key = str(img.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(img)
    if args.limit is not None:
        unique = unique[: args.limit]

    print(f"Found {len(unique)} page images under {source}")
    copied = 0
    for img in unique:
        digest = hashlib.md5(str(img.resolve()).encode("utf-8")).hexdigest()[:8]
        name = f"{safe_stem(img.parent.parent)}_{safe_stem(img)}_{digest}.png"
        # Prefer run folder name if present
        try:
            run_id = img.parents[1].name  # .../run_id/pages/p00000.png
            name = f"{run_id}_{img.stem}_{digest}.png"
        except Exception:
            pass
        target = dest / name
        if target.exists():
            continue
        print(f"  {'would copy' if args.dry_run else 'copy'}: {img} -> {target.name}")
        if not args.dry_run:
            shutil.copy2(img, target)
        copied += 1

    print(f"Done. new={copied} dest={dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
