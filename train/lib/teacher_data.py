"""Teacher page loading with optional RAM-limited cache (16GB-friendly)."""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class TeacherPage:
    path: Path
    page_id: str
    target_features: torch.Tensor  # [T, F, H]
    generated_ids: torch.Tensor  # [T + 1]
    predictive_hidden: torch.Tensor  # [T, H]
    layer_indices: Tuple[int, ...]

    @property
    def valid_anchor_count(self) -> int:
        return int(self.target_features.shape[0])


@dataclass(frozen=True)
class TeacherIndexEntry:
    path: Path
    page_id: str
    valid_anchor_count: int
    feature_count: int
    hidden_size: int
    layer_indices: Tuple[int, ...]


def _as_cpu_contiguous(x: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    x = x.detach().cpu()
    if dtype is not None:
        x = x.to(dtype=dtype)
    return x.contiguous()


def load_teacher_page(path: Path) -> TeacherPage:
    path = Path(path)
    value = torch.load(path, map_location="cpu", weights_only=False)
    required = {"target_features", "generated_ids", "predictive_hidden"}
    missing = sorted(required - set(value))
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")

    # Keep float32 in RAM for stable training math; disk may be float16.
    features = _as_cpu_contiguous(value["target_features"], torch.float32)
    ids = _as_cpu_contiguous(value["generated_ids"], torch.long).flatten()
    predictive = _as_cpu_contiguous(value["predictive_hidden"], torch.float32)
    if predictive.ndim == 3 and predictive.shape[0] == 1:
        predictive = predictive[0]
    if features.ndim == 4 and features.shape[0] == 1:
        features = features[0]

    if features.ndim != 3:
        raise ValueError(f"{path}: target_features must be [T,F,H]")
    if predictive.ndim != 2:
        raise ValueError(f"{path}: predictive_hidden must be [T,H]")
    if len(ids) != features.shape[0] + 1:
        raise ValueError(
            f"{path}: generated_ids should have T+1 entries; "
            f"got ids={len(ids)}, T={features.shape[0]}"
        )
    if predictive.shape[0] != features.shape[0]:
        raise ValueError(f"{path}: predictive_hidden length mismatch")

    page_id = str(value.get("page_id") or path.stem)
    layer_indices = tuple(int(x) for x in value.get("layer_indices", range(features.shape[1])))
    return TeacherPage(path, page_id, features, ids, predictive, layer_indices)


def discover_teacher_files(path: str | Path) -> List[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(path.glob("*_v2_teacher.pt"))
    if not files:
        files = sorted(
            p
            for p in path.glob("*.pt")
            if not p.name.startswith("checkpoint") and not p.name.startswith("drafter")
        )
    if not files:
        raise FileNotFoundError(f"No teacher .pt files found in {path}")
    return files


def split_pages(
    paths: Sequence[Path],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Path], List[Path]]:
    if len(paths) == 1:
        return list(paths), list(paths)
    order = list(paths)
    random.Random(seed).shuffle(order)
    val_count = max(1, round(len(order) * val_ratio))
    val_count = min(val_count, len(order) - 1)
    val = sorted(order[:val_count])
    train = sorted(order[val_count:])
    return train, val


def load_split_manifest(
    teacher_paths: Sequence[Path],
    split_manifest: Optional[str],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Path], List[Path]]:
    if not split_manifest:
        return split_pages(teacher_paths, val_ratio, seed)
    data = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    by_name = {p.name: p for p in teacher_paths}
    by_stem = {p.stem: p for p in teacher_paths}

    def resolve(items: Sequence[str]) -> List[Path]:
        result = []
        for item in items:
            raw = Path(item)
            candidate = raw if raw.exists() else by_name.get(raw.name) or by_stem.get(raw.stem)
            if candidate is None:
                raise FileNotFoundError(f"Split entry not found: {item}")
            result.append(candidate)
        return result

    train = resolve(data["train"])
    validation = resolve(data["validation"])
    if not train or not validation:
        raise ValueError("Split manifest contains an empty split")
    return train, validation


def build_index(paths: Sequence[Path]) -> List[TeacherIndexEntry]:
    """Scan teacher files once; prefers sidecar .json when present."""
    entries: List[TeacherIndexEntry] = []
    for path in paths:
        side = path.with_suffix(".json")
        if side.is_file():
            meta = json.loads(side.read_text(encoding="utf-8"))
            shape = meta.get("target_features")
            if shape and len(shape) >= 3:
                t, f, h = int(shape[0]), int(shape[1]), int(shape[2])
                layers = tuple(int(x) for x in meta.get("layer_indices", range(f)))
                entries.append(
                    TeacherIndexEntry(
                        path=path,
                        page_id=str(meta.get("page_id") or path.stem),
                        valid_anchor_count=t,
                        feature_count=f,
                        hidden_size=h,
                        layer_indices=layers,
                    )
                )
                continue
        page = load_teacher_page(path)
        entries.append(
            TeacherIndexEntry(
                path=page.path,
                page_id=page.page_id,
                valid_anchor_count=page.valid_anchor_count,
                feature_count=int(page.target_features.shape[1]),
                hidden_size=int(page.target_features.shape[2]),
                layer_indices=page.layer_indices,
            )
        )
    return entries


class LazyTeacherStore:
    """LRU page cache so 16GB hosts do not load every teacher at once."""

    def __init__(self, entries: Sequence[TeacherIndexEntry], cache_size: int = 12) -> None:
        if cache_size < 1:
            raise ValueError("cache_size must be >= 1")
        self.entries = list(entries)
        self.cache_size = cache_size
        self._cache: "OrderedDict[str, TeacherPage]" = OrderedDict()
        if not self.entries:
            raise ValueError("LazyTeacherStore requires at least one teacher entry")

    def __len__(self) -> int:
        return len(self.entries)

    def validate_shapes(self) -> Tuple[int, int, Tuple[int, ...]]:
        counts = {e.feature_count for e in self.entries}
        hiddens = {e.hidden_size for e in self.entries}
        layers = {e.layer_indices for e in self.entries}
        if len(counts) != 1 or len(hiddens) != 1 or len(layers) != 1:
            raise ValueError("Teacher pages have inconsistent feature shapes or layers")
        return counts.pop(), hiddens.pop(), layers.pop()

    def eligible_indices(self, draft_len: int) -> List[int]:
        return [i for i, e in enumerate(self.entries) if e.valid_anchor_count >= draft_len]

    def get(self, index: int) -> TeacherPage:
        entry = self.entries[index]
        key = str(entry.path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        page = load_teacher_page(entry.path)
        self._cache[key] = page
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return page

    def get_by_path(self, path: Path) -> TeacherPage:
        key = str(Path(path).resolve())
        # Fall back to path string as stored.
        for i, entry in enumerate(self.entries):
            if entry.path == path or entry.path.resolve() == Path(path).resolve():
                return self.get(i)
        return load_teacher_page(path)
