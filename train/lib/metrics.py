"""Offline B4/B6/B8 metrics (ported from Mini UFlash V2)."""

from __future__ import annotations

from collections import Counter
from typing import Dict

import torch

from .losses import prefix_length


def block_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, object]:
    matches = predictions.eq(targets)
    accepted = prefix_length(matches)
    histogram = Counter(int(x) for x in accepted.cpu().tolist())
    positions = matches.shape[1]
    return {
        "position_accuracy": matches.float().mean(dim=0).cpu().tolist(),
        "full_draft_accuracy": float(matches.all(dim=1).float().mean().item()),
        "average_accepted_draft": float(accepted.float().mean().item()),
        "average_effective_emitted": float(1.0 + accepted.float().mean().item()),
        "draft_length": positions,
        "acceptance_histogram": {str(k): int(v) for k, v in sorted(histogram.items())},
        "samples": int(matches.shape[0]),
    }
