"""Small utilities for Windows-friendly training scripts."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def safe_stem(path: Path) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", path.stem).strip("._-")
    return text or "page"


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def weighted_choice(
    values: Sequence[int], probabilities: Sequence[float], rng: random.Random
) -> int:
    if len(values) != len(probabilities):
        raise ValueError("values/probabilities length mismatch")
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("block probabilities must sum to a positive value")
    point = rng.random() * total
    cumulative = 0.0
    for value, probability in zip(values, probabilities):
        if probability < 0:
            raise ValueError("block probabilities cannot be negative")
        cumulative += probability
        if point <= cumulative:
            return value
    return values[-1]


def list_images(image_dir: Path, recursive: bool = False, limit: int | None = None) -> list[Path]:
    from train.config import IMAGE_EXTENSIONS

    image_dir = Path(image_dir)
    iterator: Iterable[Path] = image_dir.rglob("*") if recursive else image_dir.glob("*")
    images = sorted(
        p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit is not None:
        images = images[:limit]
    return images


def empty_cuda() -> None:
    """Best-effort VRAM reclaim; never raise (OOM-poisoned contexts included)."""
    import gc

    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
