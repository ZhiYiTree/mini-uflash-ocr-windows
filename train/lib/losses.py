"""Training losses (Stage 11B prefix-survival + DSpark-style TV / conf)."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def position_weights(
    length: int,
    decay: float,
    device: torch.device,
    tail_boost: float = 0.0,
    *,
    exp_gamma: Optional[float] = None,
    pos0_boost: float = 0.0,
) -> torch.Tensor:
    """Per-position weights.

    ``decay**pos`` is the legacy Stage-11B schedule.
    When ``exp_gamma`` is set (DSpark-style), use ``exp(-(k)/gamma)`` instead.
    ``pos0_boost`` further amplifies the first draft position (Stage-2 pos0 push).
    """
    if length <= 0:
        raise ValueError("length must be positive")
    positions = torch.arange(length, device=device, dtype=torch.float32)
    if exp_gamma is not None and exp_gamma > 0:
        weights = torch.exp(-(positions) / float(exp_gamma))
    else:
        if decay <= 0:
            raise ValueError("decay must be positive")
        weights = decay ** positions
    if length > 1 and tail_boost > 0:
        tail = positions / float(length - 1)
        weights = weights * (1.0 + tail_boost * tail)
    if pos0_boost > 0:
        weights = weights.clone()
        weights[0] = weights[0] * (1.0 + float(pos0_boost))
    return weights / weights.mean().clamp_min(1e-8)


def prefix_length(matches: torch.Tensor) -> torch.Tensor:
    if matches.ndim != 2:
        raise ValueError("matches must be [batch, positions]")
    return matches.to(torch.long).cumprod(dim=-1).sum(dim=-1)


def acceptance_support(matches: torch.Tensor) -> torch.Tensor:
    prefix = prefix_length(matches)
    positions = torch.arange(matches.shape[1], device=matches.device)[None, :]
    return positions <= prefix[:, None]


def expected_prefix_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError("logits/targets shape mismatch")
    gold_log_prob = F.log_softmax(logits.float(), dim=-1).gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)
    cumulative_log_survival = gold_log_prob.cumsum(dim=-1).clamp_min(-30.0)
    return cumulative_log_survival.exp().sum(dim=-1)


def total_variation(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
) -> torch.Tensor:
    """0.5 * ||p_d - p_t||_1 per position → [batch, positions]."""
    p_d = F.softmax(draft_logits.float(), dim=-1)
    p_t = F.softmax(target_logits.float(), dim=-1)
    return 0.5 * (p_d - p_t).abs().sum(dim=-1)


def compute_training_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    predicted_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    acceptance_logits: torch.Tensor,
    position_decay: float = 0.8,
    tail_boost: float = 0.0,
    hidden_weight: float = 0.05,
    acceptance_weight: float = 0.1,
    prefix_survival_weight: float = 0.0,
    use_auf: bool = False,
    post_acceptance_logits: Optional[torch.Tensor] = None,
    *,
    target_logits: Optional[torch.Tensor] = None,
    tv_weight: float = 0.0,
    ce_weight: float = 1.0,
    exp_position_gamma: Optional[float] = None,
    conf_label_from_tv: bool = False,
    pos0_boost: float = 0.0,
) -> Dict[str, torch.Tensor]:
    batch, positions, vocab = logits.shape
    if targets.shape != (batch, positions):
        raise ValueError("targets shape mismatch")

    token_ce = F.cross_entropy(
        logits.float().reshape(-1, vocab),
        targets.reshape(-1),
        reduction="none",
    ).view(batch, positions)
    weights = position_weights(
        positions,
        position_decay,
        logits.device,
        tail_boost=tail_boost,
        exp_gamma=exp_position_gamma,
        pos0_boost=pos0_boost,
    )[None, :]

    predictions = logits.detach().argmax(dim=-1)
    matches = predictions.eq(targets)
    support = torch.ones_like(matches, dtype=token_ce.dtype)
    if use_auf:
        support = acceptance_support(matches).to(token_ce.dtype)

    weighted = token_ce * weights * support
    ce = weighted.sum() / (weights * support).sum().clamp_min(1.0)

    pred_norm = F.normalize(predicted_hidden.float(), dim=-1)
    target_norm = F.normalize(target_hidden.float(), dim=-1)
    cosine = 1.0 - (pred_norm * target_norm).sum(dim=-1)
    smooth = F.smooth_l1_loss(pred_norm, target_norm, reduction="none").mean(dim=-1)
    hidden = ((cosine + smooth) * weights * support).sum() / (
        weights * support
    ).sum().clamp_min(1.0)

    # DSpark: soft acceptance label from TV when target logits available.
    if conf_label_from_tv and target_logits is not None:
        with torch.no_grad():
            tv_pos = total_variation(logits.detach(), target_logits.detach())
            correctness_labels = (1.0 - tv_pos).clamp(0.0, 1.0).to(
                dtype=acceptance_logits.dtype
            )
    else:
        correctness_labels = matches.to(dtype=acceptance_logits.dtype)

    gate_raw = F.binary_cross_entropy_with_logits(
        acceptance_logits.float(), correctness_labels.float(), reduction="none"
    )
    gate_acceptance = (gate_raw * weights).mean()
    if post_acceptance_logits is not None:
        post_raw = F.binary_cross_entropy_with_logits(
            post_acceptance_logits.float(), correctness_labels.float(), reduction="none"
        )
        post_acceptance = (post_raw * weights).mean()
        acceptance = 0.5 * (gate_acceptance + post_acceptance)
    else:
        post_acceptance = gate_acceptance.detach()
        acceptance = gate_acceptance

    expected_prefix = expected_prefix_from_logits(logits, targets)
    prefix_survival = 1.0 - expected_prefix.mean() / float(positions)

    if tv_weight > 0.0 and target_logits is not None:
        tv_pos = total_variation(logits, target_logits)
        tv = (tv_pos * weights * support).sum() / (weights * support).sum().clamp_min(1.0)
    else:
        tv = logits.new_zeros(())

    total = (
        ce_weight * ce
        + tv_weight * tv
        + hidden_weight * hidden
        + acceptance_weight * acceptance
        + prefix_survival_weight * prefix_survival
    )
    accepted = prefix_length(matches)
    return {
        "loss": total,
        "ce": ce.detach(),
        "tv": tv.detach() if torch.is_tensor(tv) else torch.tensor(0.0),
        "hidden": hidden.detach(),
        "acceptance_bce": acceptance.detach(),
        "gate_acceptance_bce": gate_acceptance.detach(),
        "post_acceptance_bce": post_acceptance.detach(),
        "prefix_survival": prefix_survival.detach(),
        "expected_soft_prefix": expected_prefix.detach().mean(),
        "matches": matches,
        "accepted_prefix": accepted,
        "position_weights": weights.detach().squeeze(0),
    }
