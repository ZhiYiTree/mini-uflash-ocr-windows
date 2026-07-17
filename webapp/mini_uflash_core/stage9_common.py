from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch


def ensure_batch(value: torch.Tensor) -> torch.Tensor:
    return value.unsqueeze(0) if value.ndim == 1 else value


def decoder_layers(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "model", None), "layers", None),
        getattr(model, "layers", None),
    ]
    for layers in candidates:
        if layers is not None and hasattr(layers, "__len__") and len(layers) > 0:
            return layers
    raise RuntimeError("Could not locate target decoder layers")


def hidden_from_output(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    if hasattr(value, "last_hidden_state"):
        return value.last_hidden_state
    raise TypeError(f"Unsupported decoder layer output: {type(value)!r}")


class FeatureTap:
    def __init__(self, target: torch.nn.Module, layer_indices: Sequence[int]) -> None:
        self.layer_indices = tuple(int(x) for x in layer_indices)
        layers = decoder_layers(target)
        if len(self.layer_indices) != 5:
            raise ValueError("Exactly five target layers are required")
        if min(self.layer_indices) < 0 or max(self.layer_indices) >= len(layers):
            raise ValueError(
                f"Layer indices {self.layer_indices} outside decoder range [0, {len(layers)-1}]"
            )
        self.captured: Dict[int, torch.Tensor] = {}
        self.handles = []
        for index in self.layer_indices:
            def make_hook(layer_index: int):
                def hook(_module, _inputs, output):
                    self.captured[layer_index] = hidden_from_output(output).detach()
                return hook
            self.handles.append(layers[index].register_forward_hook(make_hook(index)))

    def clear(self) -> None:
        self.captured.clear()

    def last_features(self) -> torch.Tensor:
        missing = [x for x in self.layer_indices if x not in self.captured]
        if missing:
            raise RuntimeError(f"Missing captured target layers: {missing}")
        return torch.stack(
            [self.captured[x][:, -1, :] for x in self.layer_indices], dim=1
        )

    def position_features(self, position: int) -> torch.Tensor:
        """Return the five tapped target features at one block position."""
        missing = [x for x in self.layer_indices if x not in self.captured]
        if missing:
            raise RuntimeError(f"Missing captured target layers: {missing}")
        return torch.stack(
            [self.captured[x][:, int(position), :] for x in self.layer_indices],
            dim=1,
        )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.captured.clear()


def get_input_embeddings(model: torch.nn.Module) -> torch.nn.Module:
    value = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if value is None:
        candidates = [
            getattr(model, "embed_tokens", None),
            getattr(getattr(model, "model", None), "embed_tokens", None),
            getattr(getattr(getattr(model, "model", None), "model", None), "embed_tokens", None),
        ]
        value = next((x for x in candidates if x is not None), None)
    if value is None:
        raise RuntimeError("Could not locate target input embeddings")
    return value


def get_output_embeddings(model: torch.nn.Module) -> torch.nn.Module:
    value = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if value is None:
        value = getattr(model, "lm_head", None)
    if value is None:
        raise RuntimeError("Could not locate target LM head")
    return value


def extend_image_mask(base_mask: torch.Tensor, target_length: int) -> torch.Tensor:
    mask = ensure_batch(base_mask).bool()
    if mask.shape[1] < target_length:
        mask = torch.cat(
            [
                mask,
                torch.zeros(
                    (mask.shape[0], target_length - mask.shape[1]),
                    dtype=torch.bool,
                ),
            ],
            dim=1,
        )
    return mask[:, :target_length]


def clone_cache(value: Any) -> Any:
    """Clone a Transformers cache while preserving Unlimited-OCR ring metadata.

    Unlimited-OCR stores ring-buffer state directly on the Cache object via
    ``_prefill_length`` and ``_ring_pos``. Converting that Cache to a legacy
    tuple silently drops those attributes and changes the subsequent decode
    path. Prefer a full deepcopy of Cache-like objects so both GPU tensors and
    custom metadata are independent. Only use legacy reconstruction as a
    compatibility fallback.
    """
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(clone_cache(x) for x in value)
    if isinstance(value, list):
        return [clone_cache(x) for x in value]

    if hasattr(value, "to_legacy_cache"):
        try:
            return copy.deepcopy(value)
        except Exception:
            legacy = clone_cache(value.to_legacy_cache())
            factory = getattr(type(value), "from_legacy_cache", None)
            if callable(factory):
                restored = factory(legacy)
                for name in (
                    "_prefill_length",
                    "_ring_pos",
                    "_seen_tokens",
                    "seen_tokens",
                ):
                    if hasattr(value, name):
                        setattr(restored, name, copy.deepcopy(getattr(value, name)))
                return restored
            return legacy

    try:
        return copy.deepcopy(value)
    except Exception as exc:  # pragma: no cover - target-version dependent
        raise TypeError(f"Unsupported past_key_values type: {type(value)!r}") from exc



def iter_cache_tensors(value: Any, _seen: set[int] | None = None):
    """Yield tensors reachable from a Cache-like object without looping."""
    if _seen is None:
        _seen = set()
    object_id = id(value)
    if object_id in _seen:
        return
    _seen.add(object_id)
    if torch.is_tensor(value):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from iter_cache_tensors(item, _seen)
        return
    if isinstance(value, (tuple, list, set)):
        for item in value:
            yield from iter_cache_tensors(item, _seen)
        return
    namespace = getattr(value, "__dict__", None)
    if isinstance(namespace, dict):
        for item in namespace.values():
            yield from iter_cache_tensors(item, _seen)


def cache_tensors_are_disjoint(left: Any, right: Any) -> bool:
    """Return True when two Cache-like objects share no tensor storage."""
    left_ptrs = {
        (tensor.device.type, tensor.device.index, tensor.untyped_storage().data_ptr())
        for tensor in iter_cache_tensors(left)
        if tensor.numel()
    }
    right_ptrs = {
        (tensor.device.type, tensor.device.index, tensor.untyped_storage().data_ptr())
        for tensor in iter_cache_tensors(right)
        if tensor.numel()
    }
    return left_ptrs.isdisjoint(right_ptrs)


def cache_seq_length(value: Any) -> int:
    """Read the physical KV length from modern or legacy cache formats."""
    if value is None:
        return 0
    getter = getattr(value, "get_seq_length", None)
    if callable(getter):
        return int(getter())
    if isinstance(value, (tuple, list)) and value:
        first = value[0]
        if isinstance(first, (tuple, list)) and first and torch.is_tensor(first[0]):
            return int(first[0].shape[-2])
    raise TypeError(f"Cannot determine cache sequence length for {type(value)!r}")

def leading_prefix_length(matches: Iterable[bool]) -> int:
    total = 0
    for value in matches:
        if not bool(value):
            break
        total += 1
    return total


def evenly_spaced(items: Sequence[Path], limit: int) -> List[Path]:
    if limit <= 0 or len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[len(items) // 2]]
    indices = [round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)]
    return [items[i] for i in indices]


def discover_payloads(
    explicit: Sequence[str],
    payload_dir: str | None,
    pattern: str,
    limit: int,
) -> List[Path]:
    paths = [Path(x).expanduser().resolve() for x in explicit]
    if payload_dir:
        root = Path(payload_dir).expanduser().resolve()
        paths.extend(sorted(root.glob(pattern)))
    unique = sorted({x for x in paths if x.is_file()})
    if not unique:
        raise FileNotFoundError("No oracle payload files were found")
    return evenly_spaced(unique, limit)


def payload_tensors(payload: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    required = (
        "input_ids",
        "output_ids",
        "prompt_length",
        "images_seq_mask",
        "images_crop",
        "images_ori",
        "images_spatial_crop",
    )
    missing = [x for x in required if x not in payload]
    if missing:
        raise KeyError(f"Payload missing keys: {missing}")
    prompt = ensure_batch(payload["input_ids"]).long().to(device)
    output = ensure_batch(payload["output_ids"]).long().to(device)
    prompt_length = int(payload["prompt_length"])
    if output.shape[1] < prompt_length:
        raise ValueError("output_ids shorter than prompt_length")
    return {
        "prompt_ids": prompt,
        "official_generated": output[:, prompt_length:],
        "base_image_mask": ensure_batch(payload["images_seq_mask"]).bool(),
        "images_crop": payload["images_crop"].to(device),
        "images_ori": payload["images_ori"].to(device),
        "spatial_crop": payload["images_spatial_crop"].to(device),
    }



def generation_policy(payload: Dict[str, Any]) -> Dict[str, int]:
    """Read the exact greedy-generation policy captured in the oracle payload."""
    config = payload.get("generation") or {}
    return {
        "no_repeat_ngram_size": int(config.get("no_repeat_ngram_size", 0) or 0),
        "ngram_window": int(config.get("ngram_window", 0) or 0),
    }


def apply_sliding_window_no_repeat(
    logits: torch.Tensor,
    prefix_ids: Sequence[int],
    ngram_size: int,
    window: int,
) -> torch.Tensor:
    """Match Unlimited-OCR's SlidingWindowNoRepeatNgramProcessor exactly.

    ``prefix_ids`` must be the complete sequence visible to generation at this
    position, including the prompt/image placeholder tokens and all committed
    output tokens. The returned scores are float32 and safe to mutate.
    """
    scores = logits.float().clone()
    if scores.ndim == 1:
        scores = scores.unsqueeze(0)
    if ngram_size <= 0 or window <= 0 or len(prefix_ids) < ngram_size:
        return scores

    sequence = [int(x) for x in prefix_ids]
    search_start = max(0, len(sequence) - int(window))
    search_end = len(sequence) - int(ngram_size) + 1
    if search_end <= search_start:
        return scores

    if ngram_size > 1:
        current_prefix = tuple(sequence[-(ngram_size - 1):])
    else:
        current_prefix = tuple()

    banned = set()
    for index in range(search_start, search_end):
        ngram = sequence[index:index + ngram_size]
        if ngram_size == 1 or tuple(ngram[:-1]) == current_prefix:
            banned.add(int(ngram[-1]))
    if banned:
        valid = [x for x in banned if 0 <= x < scores.shape[-1]]
        if valid:
            scores[:, valid] = float("-inf")
    return scores


def policy_argmax(
    logits: torch.Tensor,
    prefix_ids: Sequence[int],
    policy: Dict[str, int],
) -> int:
    scores = apply_sliding_window_no_repeat(
        logits,
        prefix_ids,
        int(policy.get("no_repeat_ngram_size", 0)),
        int(policy.get("ngram_window", 0)),
    )
    return int(scores.argmax(dim=-1).item())

def first_mismatch(left: Sequence[int], right: Sequence[int]) -> int | None:
    for i, (a, b) in enumerate(zip(left, right)):
        if int(a) != int(b):
            return i
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(x) for x in values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[index]


def atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
