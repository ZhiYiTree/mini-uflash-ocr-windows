"""Mini UFlash Stage 11B exactness probe engine.

This module orchestrates the precise mode pipeline:

1. **Live payload capture** (``payload_capture.capture_payload``) — wraps
   ``model.generate`` during one official ``.infer()`` call to record the
   oracle token trajectory.
2. **Drafter load** — loads the Stage 11B ``drafter_v2_stage11b_best.pt``
   checkpoint (7.96 M params, pure PyTorch SDPA).
3. **B8 exactness probe** — the strict-B1-replay loop from
   ``stage9_v2_b8_exactness_probe.py:run_case()``. For every speculative
   round the drafter predicts 7 draft tokens (B8 = 1 anchor + 7 draft).
   A diagnostic block verifier runs on a cloned cache; the *authoritative*
   trajectory advances through q_len=1 strict replay. Commit happens only
   when the draft matches the strict replay token-for-token.
4. **Official text stays official** — the decoded text from step 1 is the
   user-visible OCR result regardless of exactness outcome. Mini UFlash
   metrics are diagnostic.

Windows compatibility:
* Pure-PyTorch drafter (SDPA, no flash-attn/triton/xformers).
* ``torch.cuda.synchronize()`` for accurate latency on Windows.
* ``copy.deepcopy`` for cache cloning (preserves Unlimited-OCR ring metadata).
* ``if __name__ == "__main__":`` guard pattern (Windows spawn-safe).
"""

from __future__ import annotations

import gc
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from . import config
from .model_manager import ModelHandles
from .payload_capture import capture_payload, prepare_input_payload
from .unlimited_ocr_engine import clean_markdown

from .mini_uflash_core import load_checkpoint
from .mini_uflash_core.online_target import OnlineTarget
from .mini_uflash_core.stage9_common import (
    FeatureTap,
    clone_cache,
    cache_seq_length,
    cache_tensors_are_disjoint,
    payload_tensors,
    policy_argmax,
    generation_policy,
    first_mismatch,
    synchronize,
    atomic_json,
)


# ---------------------------------------------------------------------------
# Data classes for the probe results
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """Complete output of a Mini UFlash precise-mode run."""

    # Official OCR text (always from the official infer path).
    official_markdown: str = ""
    official_plain_text: str = ""
    official_raw: str = ""

    # Mini UFlash exactness probe metrics.
    probe_ok: bool = False
    checkpoint_path: str = ""
    checkpoint_step: int = -1
    parameter_count: int = 0
    block_size: int = 8
    dtype_name: str = ""
    generated_tokens: int = 0
    tokens_checked: int = 0
    speculative_rounds: int = 0
    mean_accepted_draft: float = 0.0
    effective_tokens_per_round: float = 0.0
    full_block_count: int = 0
    full_block_ratio: float = 0.0
    acceptance_histogram: Dict[str, int] = field(default_factory=dict)
    per_position_accuracy: List[float] = field(default_factory=list)
    drafter_latency_ms: float = 0.0
    block_verifier_latency_ms: float = 0.0
    strict_replay_latency_ms: float = 0.0
    block_vs_b1_disagreement_rate: float = 0.0
    final_token_exactness: bool = False
    first_mismatch: Optional[int] = None
    official_token_ids: List[int] = field(default_factory=list)
    strict_replay_token_ids: List[int] = field(default_factory=list)
    theoretical_speedup: float = 0.0

    # Detailed info.
    failure: Optional[Dict[str, Any]] = None
    cycles_detail: List[Dict[str, Any]] = field(default_factory=list)
    layer_indices: List[int] = field(default_factory=list)
    generation_policy: Dict[str, int] = field(default_factory=dict)

    # Timing.
    payload_capture_seconds: float = 0.0
    probe_seconds: float = 0.0
    total_seconds: float = 0.0

    # VRAM at end.
    vram_gb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": "mini_uflash_precise",
            "probe_ok": self.probe_ok,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_step": self.checkpoint_step,
            "parameter_count": self.parameter_count,
            "block_size": self.block_size,
            "dtype": self.dtype_name,
            "generated_tokens": self.generated_tokens,
            "tokens_checked": self.tokens_checked,
            "speculative_rounds": self.speculative_rounds,
            "mean_accepted_draft": round(self.mean_accepted_draft, 4),
            "effective_tokens_per_round": round(self.effective_tokens_per_round, 4),
            "full_block_count": self.full_block_count,
            "full_block_ratio": round(self.full_block_ratio, 4),
            "acceptance_histogram": self.acceptance_histogram,
            "per_position_accuracy": [round(x, 4) for x in self.per_position_accuracy],
            "drafter_latency_ms": round(self.drafter_latency_ms, 2),
            "block_verifier_latency_ms": round(self.block_verifier_latency_ms, 2),
            "strict_replay_latency_ms": round(self.strict_replay_latency_ms, 2),
            "block_vs_b1_disagreement_rate": round(self.block_vs_b1_disagreement_rate, 4),
            "final_token_exactness": self.final_token_exactness,
            "first_mismatch": self.first_mismatch,
            "official_token_ids": self.official_token_ids,
            "strict_replay_token_ids": self.strict_replay_token_ids,
            "theoretical_speedup": round(self.theoretical_speedup, 3),
            "failure": self.failure,
            "layer_indices": self.layer_indices,
            "payload_capture_seconds": round(self.payload_capture_seconds, 2),
            "probe_seconds": round(self.probe_seconds, 2),
            "total_seconds": round(self.total_seconds, 2),
            "vram_gb": round(self.vram_gb, 2),
            "note": config.INFERENCE.theoretical_speedup_note,
        }


@dataclass
class DirectResult:
    """Output from the explicitly non-lossless Direct Block decoder."""

    markdown: str = ""
    plain_text: str = ""
    raw_output: str = ""
    generated_tokens: int = 0
    total_seconds: float = 0.0
    preprocessing_seconds: float = 0.0
    speculative_rounds: int = 0
    direct_block_commits: int = 0
    full_block_commits: int = 0
    fallback_rounds: int = 0
    direct_committed_tokens: int = 0
    target_decode_forwards: int = 0
    mean_accepted_draft: float = 0.0
    full_block_ratio: float = 0.0
    acceptance_histogram: Dict[str, int] = field(default_factory=dict)
    drafter_latency_ms: float = 0.0
    block_verifier_latency_ms: float = 0.0
    strict_fallback_latency_ms: float = 0.0
    target_forward_reduction: float = 0.0
    checkpoint_path: str = ""
    checkpoint_step: int = -1
    block_size: int = 8
    stopped_on_eos: bool = False
    warning: str = (
        "Direct Block KV-cache commit 是非无损实验路径；结果可能与普通 Unlimited-OCR 不同。"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": "mini_uflash_direct",
            "non_lossless": True,
            "commit_policy": "direct_accepted_prefix_cache_crop",
            "generated_tokens": self.generated_tokens,
            "total_seconds": round(self.total_seconds, 2),
            "preprocessing_seconds": round(self.preprocessing_seconds, 2),
            "speculative_rounds": self.speculative_rounds,
            "direct_block_commits": self.direct_block_commits,
            "full_block_commits": self.full_block_commits,
            "fallback_rounds": self.fallback_rounds,
            "direct_committed_tokens": self.direct_committed_tokens,
            "target_decode_forwards": self.target_decode_forwards,
            "mean_accepted_draft": round(self.mean_accepted_draft, 4),
            "full_block_ratio": round(self.full_block_ratio, 4),
            "acceptance_histogram": self.acceptance_histogram,
            "drafter_latency_ms": round(self.drafter_latency_ms, 2),
            "block_verifier_latency_ms": round(self.block_verifier_latency_ms, 2),
            "strict_fallback_latency_ms": round(self.strict_fallback_latency_ms, 2),
            "target_forward_reduction": round(self.target_forward_reduction, 3),
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_step": self.checkpoint_step,
            "block_size": self.block_size,
            "stopped_on_eos": self.stopped_on_eos,
            "warning": self.warning,
        }


# ---------------------------------------------------------------------------
# Drafter lifecycle
# ---------------------------------------------------------------------------

_drafter_cache: Optional[Dict[str, Any]] = None
_drafter_model: Optional[Any] = None
_checkpoint_data: Optional[Dict[str, Any]] = None


def load_drafter(weight_path: Optional[Path] = None, device: Optional[torch.device] = None) -> tuple:
    """Load the Stage 11B drafter checkpoint. Returns (drafter, checkpoint_data)."""
    global _drafter_model, _checkpoint_data
    if _drafter_model is not None and _checkpoint_data is not None:
        return _drafter_model, _checkpoint_data

    path = weight_path or config.discover_weight()
    if path is None:
        raise FileNotFoundError(
            "未找到 Mini UFlash drafter 权重文件。\n"
            "稳定模式仍可运行，精确模式已禁用。\n"
            f"搜索路径：{[str(r) for r in config.weight_search_roots()]}"
        )

    dev = device or torch.device("cuda")
    drafter, ckpt = load_checkpoint(path, device=dev)
    # map_location controls checkpoint tensor deserialization; it does not move
    # the freshly constructed nn.Module parameters. Move the module explicitly.
    drafter = drafter.to(dev).eval()
    _drafter_model = drafter
    _checkpoint_data = ckpt
    return drafter, ckpt


def unload_drafter() -> None:
    """Free the drafter from GPU memory."""
    global _drafter_model, _checkpoint_data
    _drafter_model = None
    _checkpoint_data = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_drafter_loaded() -> bool:
    return _drafter_model is not None


# ---------------------------------------------------------------------------
# Direct Block experiment (actual output path, explicitly non-lossless)
# ---------------------------------------------------------------------------

def run_direct_mode(
    handles: ModelHandles,
    image_path: Path,
    *,
    preset_name: str = "gundam",
    max_length: int = config.INFERENCE.max_length,
    block_size: int = config.INFERENCE.block_size,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> DirectResult:
    """Decode one image with B8 drafting and direct full-block cache commits.

    Each round commits the anchor plus the target-verified draft prefix, then
    crops rejected suffix entries from the q_len=Block cache. Because
    Unlimited-OCR's custom cache can drift after direct prefix commits, this
    path is intentionally marked non-lossless and is never used by ordinary
    mode.
    """
    total_started = time.perf_counter()
    prep_started = time.perf_counter()
    prepared = prepare_input_payload(
        handles.model,
        handles.tokenizer,
        Path(image_path),
        preset_name=preset_name,
        max_length=max_length,
    )
    preprocessing_seconds = time.perf_counter() - prep_started
    drafter, checkpoint = load_drafter(device=torch.device("cuda"))

    device = torch.device("cuda")
    tensors = payload_tensors(prepared.payload, device)
    policy = generation_policy(prepared.payload)
    prompt_ids = [int(x) for x in tensors["prompt_ids"][0].detach().cpu().tolist()]
    max_generated = max(1, int(max_length) - len(prompt_ids))
    eos_id = getattr(handles.tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = getattr(handles.model.config, "eos_token_id", 1)
    eos_id = int(eos_id)

    layer_indices = checkpoint.get("extra", {}).get(
        "layer_indices", list(config.INFERENCE.layer_indices)
    )
    tap = FeatureTap(handles.model, layer_indices)
    online = OnlineTarget(handles.model, tap, tensors, device)

    generated: List[int] = []
    cycles: List[Dict[str, Any]] = []
    accepted_histogram: Counter[int] = Counter()
    direct_commits = 0
    full_block_commits = 0
    fallback_rounds = 0
    target_forwards = 0
    drafter_times: List[float] = []
    verifier_times: List[float] = []
    fallback_times: List[float] = []
    stopped_on_eos = False
    original_sw = getattr(handles.model.config, "sliding_window", None)
    ring_window = getattr(handles.model.config, "sliding_window_size", None) or original_sw
    handles.model.config._ring_window = ring_window
    handles.model.config.sliding_window = None

    try:
        state = online.prefill([])
        while len(generated) < max_generated:
            prefix = [*prompt_ids, *generated]
            anchor_id = policy_argmax(state["next_logits"], prefix, policy)
            remaining = max_generated - len(generated)
            if anchor_id == eos_id:
                generated.append(anchor_id)
                stopped_on_eos = True
                break

            # A short final tail cannot form B8; finish it through strict B1.
            if remaining < block_size:
                generated.append(anchor_id)
                if len(generated) < max_generated:
                    step = online.step(anchor_id, state["cache"], state["absolute_length"])
                    target_forwards += 1
                    state = step
                continue

            features = state["features"]
            anchor_tensor = torch.tensor([anchor_id], dtype=torch.long, device=device)
            anchor_embedding = handles.input_embeddings(anchor_tensor)
            synchronize(device)
            draft_started = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                draft_output = drafter.draft(
                    features,
                    handles.output_embeddings,
                    block_size=block_size,
                    anchor_embedding=anchor_embedding,
                )
            synchronize(device)
            draft_ms = (time.perf_counter() - draft_started) * 1000.0
            draft_ids = [int(x) for x in draft_output["tokens"][0].tolist()]
            drafter_times.append(draft_ms)

            block_cache = clone_cache(state["cache"])
            if not cache_tensors_are_disjoint(state["cache"], block_cache):
                raise RuntimeError("Direct Block cache clone aliases the live cache")
            block_result = online.verify_block(
                anchor_id,
                draft_ids,
                block_cache,
                state["absolute_length"],
                prefix,
                policy,
            )
            target_forwards += 1
            verifier_ms = float(block_result["seconds"] * 1000.0)
            verifier_times.append(verifier_ms)
            verifier_predictions = [int(x) for x in block_result["predictions"]]
            verifier_accepted = 0
            for candidate, target_id in zip(draft_ids, verifier_predictions):
                if candidate != target_id:
                    break
                verifier_accepted += 1

            # Commit the target-verified prefix, not only complete B8 blocks.
            # The block cache contains rejected suffix tokens, so crop it back
            # to anchor + accepted draft positions before carrying it forward.
            accepted = verifier_accepted
            appended = [anchor_id, *draft_ids[:accepted]]
            if len(appended) > remaining:
                appended = appended[:remaining]
                accepted = max(0, len(appended) - 1)
            generated.extend(appended)
            direct_commits += 1
            if accepted == block_size - 1:
                full_block_commits += 1

            if eos_id in appended:
                eos_at = appended.index(eos_id)
                trailing = len(appended) - eos_at - 1
                if trailing > 0:
                    del generated[-trailing:]
                stopped_on_eos = True
                strict_ms = 0.0
            else:
                committed_cache = block_result["cache"]
                physical_target = (
                    int(block_result["physical_cache_length_before"])
                    + len(appended)
                )
                crop = getattr(committed_cache, "crop", None)
                if not callable(crop):
                    raise RuntimeError("Current Transformers cache cannot crop an accepted prefix")
                crop(physical_target)
                state = {
                    "cache": committed_cache,
                    "next_logits": block_result["logits"][:, accepted, :],
                    "features": online.tap.position_features(accepted),
                    "absolute_length": int(state["absolute_length"]) + len(appended),
                    "seconds": 0.0,
                }
                strict_ms = 0.0

            accepted_histogram[accepted] += 1
            cycles.append({
                "round": len(cycles) + 1,
                "accepted_draft": accepted,
                "direct_commit": True,
                "full_block_commit": accepted == block_size - 1,
                "appended": len(appended),
                "draft_ms": draft_ms,
                "block_ms": verifier_ms,
                "fallback_ms": strict_ms,
            })
            if progress_callback is not None:
                progress_callback({
                    "generated": len(generated),
                    "target": max_generated,
                    "round": len(cycles),
                    "accepted": accepted,
                    "direct_commits": direct_commits,
                    "full_b8_ratio": full_block_commits / len(cycles),
                })
            if stopped_on_eos:
                break
    finally:
        tap.close()
        handles.model.config.sliding_window = original_sw
        if prepared.scratch_dir.exists():
            shutil.rmtree(prepared.scratch_dir, ignore_errors=True)

    raw = handles.tokenizer.decode(generated, skip_special_tokens=False)
    markdown = clean_markdown(raw)
    accepted_values = [int(x["accepted_draft"]) for x in cycles]

    def _average(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    result = DirectResult(
        markdown=markdown,
        plain_text=_strip_markdown(markdown),
        raw_output=raw,
        generated_tokens=len(generated),
        total_seconds=time.perf_counter() - total_started,
        preprocessing_seconds=preprocessing_seconds,
        speculative_rounds=len(cycles),
        direct_block_commits=direct_commits,
        full_block_commits=full_block_commits,
        fallback_rounds=fallback_rounds,
        direct_committed_tokens=sum(1 + int(x["accepted_draft"]) for x in cycles),
        target_decode_forwards=target_forwards,
        mean_accepted_draft=_average(accepted_values),
        full_block_ratio=full_block_commits / len(cycles) if cycles else 0.0,
        acceptance_histogram={
            str(k): int(v) for k, v in sorted(accepted_histogram.items())
        },
        drafter_latency_ms=_average(drafter_times),
        block_verifier_latency_ms=_average(verifier_times),
        strict_fallback_latency_ms=_average(fallback_times),
        target_forward_reduction=(len(generated) / target_forwards) if target_forwards else 0.0,
        checkpoint_path=str(config.discover_weight() or "N/A"),
        checkpoint_step=int(checkpoint.get("step", -1)),
        block_size=block_size,
        stopped_on_eos=stopped_on_eos,
    )
    return result


# ---------------------------------------------------------------------------
# Stable DFlash — product-oriented speculative path for long documents
# ---------------------------------------------------------------------------

@dataclass
class StableDFlashResult:
    """Speculative OCR with periodic resync + degeneration guards.

    Unlike Direct Block, this path rebuilds the target KV from the true token
    prefix on a schedule so long-document cache drift cannot run away. Draft
    verification still uses one block forward; accepted prefixes may be
    committed via cache crop between resync points for speed.
    """

    markdown: str = ""
    plain_text: str = ""
    raw_output: str = ""
    generated_tokens: int = 0
    total_seconds: float = 0.0
    preprocessing_seconds: float = 0.0
    speculative_rounds: int = 0
    direct_block_commits: int = 0
    full_block_commits: int = 0
    fallback_rounds: int = 0
    resync_count: int = 0
    pure_b1_rounds: int = 0
    direct_committed_tokens: int = 0
    target_decode_forwards: int = 0
    mean_accepted_draft: float = 0.0
    full_block_ratio: float = 0.0
    acceptance_histogram: Dict[str, int] = field(default_factory=dict)
    drafter_latency_ms: float = 0.0
    block_verifier_latency_ms: float = 0.0
    strict_fallback_latency_ms: float = 0.0
    resync_latency_ms: float = 0.0
    # Gold-bench cost breakdown (sum of timed sections, ms)
    total_draft_ms: float = 0.0
    total_verify_ms: float = 0.0
    total_resync_ms: float = 0.0
    total_b1_ms: float = 0.0
    cost_share_draft: float = 0.0
    cost_share_verify: float = 0.0
    cost_share_resync: float = 0.0
    cost_share_b1: float = 0.0
    page_degraded_to_b1: bool = False
    length_hard_cap_used: int = 0
    target_forward_reduction: float = 0.0
    checkpoint_path: str = ""
    checkpoint_step: int = -1
    block_size: int = 8
    stopped_on_eos: bool = False
    tier: str = ""
    warning: str = (
        "稳定 DFlash（sys_v2）：锁定 B4、软/硬长度帽、低τ停写与整页降级、"
        "低期望跳过 verify；成本拆分 draft/verify/resync/B1。"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": "mini_uflash_stable_dflash",
            "tier": self.tier or None,
            "non_lossless": bool(self.tier and self.tier != "lossless"),
            "commit_policy": "verified_prefix_crop_with_periodic_prefill_resync",
            "generated_tokens": self.generated_tokens,
            "total_seconds": round(self.total_seconds, 2),
            "preprocessing_seconds": round(self.preprocessing_seconds, 2),
            "speculative_rounds": self.speculative_rounds,
            "direct_block_commits": self.direct_block_commits,
            "full_block_commits": self.full_block_commits,
            "fallback_rounds": self.fallback_rounds,
            "resync_count": self.resync_count,
            "pure_b1_rounds": self.pure_b1_rounds,
            "direct_committed_tokens": self.direct_committed_tokens,
            "target_decode_forwards": self.target_decode_forwards,
            "mean_accepted_draft": round(self.mean_accepted_draft, 4),
            "full_block_ratio": round(self.full_block_ratio, 4),
            "acceptance_histogram": self.acceptance_histogram,
            "drafter_latency_ms": round(self.drafter_latency_ms, 2),
            "block_verifier_latency_ms": round(self.block_verifier_latency_ms, 2),
            "strict_fallback_latency_ms": round(self.strict_fallback_latency_ms, 2),
            "resync_latency_ms": round(self.resync_latency_ms, 2),
            "total_draft_ms": round(self.total_draft_ms, 1),
            "total_verify_ms": round(self.total_verify_ms, 1),
            "total_resync_ms": round(self.total_resync_ms, 1),
            "total_b1_ms": round(self.total_b1_ms, 1),
            "cost_share_draft": round(self.cost_share_draft, 3),
            "cost_share_verify": round(self.cost_share_verify, 3),
            "cost_share_resync": round(self.cost_share_resync, 3),
            "cost_share_b1": round(self.cost_share_b1, 3),
            "page_degraded_to_b1": self.page_degraded_to_b1,
            "length_hard_cap_used": self.length_hard_cap_used,
            "target_forward_reduction": round(self.target_forward_reduction, 3),
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_step": self.checkpoint_step,
            "block_size": self.block_size,
            "stopped_on_eos": self.stopped_on_eos,
            "warning": self.warning,
        }


def _is_degenerate_tail(generated: Sequence[int], window: int = 48) -> bool:
    """Detect short-cycle / low-diversity tails that burn long-doc time."""
    if len(generated) < window:
        return False
    tail = [int(x) for x in generated[-window:]]
    if len(set(tail)) < max(4, window // 4):
        return True
    for cycle in range(1, 9):
        if window < cycle * 4:
            continue
        unit = tail[-cycle:]
        if tail[-cycle * 4 :] == (unit * 4)[: cycle * 4]:
            return True
    return False


def _confidence_verify_len(
    correctness_probability: Sequence[float],
    *,
    min_len: int = 1,
    conf_floor: float = 0.22,
    survival_floor: float = 0.08,
) -> int:
    """DSpark-style prefix truncation: only verify tokens worth sending to target.

    Truncates the *tail* when conditional conf / cumulative survival drops.
    Always keeps at least one token when the first conf is not catastrophic —
    never forces a permanent pure-B1 lock-in from soft confidence alone.
    """
    if not correctness_probability:
        return 0
    n = len(correctness_probability)
    survival = 1.0
    keep = n
    for i, c in enumerate(correctness_probability):
        c = float(c)
        survival *= max(1e-6, min(1.0, c))
        if i == 0 and c < 0.12:
            # First token looks doomed — skip verify, caller does B1.
            return 0
        if i > 0 and (c < conf_floor or survival < survival_floor):
            keep = i  # drop from this position onward
            break
    # Soft floor: if expected mass is high, keep at least 2 when available.
    expected = sum(float(x) for x in correctness_probability)
    if keep < 2 and n >= 2 and expected >= 1.2:
        keep = 2
    if keep < min_len:
        keep = min_len if float(correctness_probability[0]) >= 0.12 else 0
    return int(min(keep, n))


def _choose_active_block(
    max_block: int,
    recent_accept: Sequence[int],
    *,
    default_block: int = 4,
) -> int:
    """Stage-1 dynamic γ: default B4; promote only when rolling accept is strong.

    Hunyuan-style NUM_SPEC_TOKENS: large K only pays when τ is high. On 8GB
    eager verify, starting at B8 with mean accept ~1–2 is pure overhead.
    """
    max_block = max(4, int(max_block))
    default_block = min(max(4, int(default_block)), max_block)
    if not recent_accept:
        return default_block
    window = recent_accept[-8:]
    mean_recent = sum(window) / len(window)
    # Promote conservatively; demote fast.
    if mean_recent >= 2.5 and max_block >= 8:
        return 8
    if mean_recent >= 1.6 and max_block >= 6:
        return min(6, max_block)
    if mean_recent < 0.9:
        return 4
    return default_block


def run_stable_dflash_mode(
    handles: ModelHandles,
    image_path: Path,
    *,
    preset_name: str = "gundam",
    max_length: int = config.INFERENCE.max_length,
    tier: Optional[str] = None,
    block_size: int = 4,
    max_block_size: int = 4,
    resync_every: int = 256,
    conf_floor: float = 0.22,
    survival_floor: float = 0.08,
    length_hard_cap: int = 384,
    length_soft_cap: int = 280,
    low_tau_b1_threshold: float = 1.0,
    low_tau_window: int = 6,
    low_tau_b1_steps: int = 8,
    zero_accept_page_b1_after: int = 6,
    skip_verify_expected_below: float = 0.45,
    use_conf_schedule: bool = True,
    conf_schedule_min_expected: float = 0.0,
    page_degrade_enabled: bool = True,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> StableDFlashResult:
    """Stable speculative OCR — wall-clock focused (8GB).

    Optional ``tier`` in {fast, balanced, lossless} applies Hunyuan-style product
    presets (see ``dflash_tiers``). Lossless disables soft truncation and uses
    DSpark-style calibrated verify_len scheduling.
    """
    if tier:
        from .dflash_tiers import resolve_tier

        preset = resolve_tier(tier)
        block_size = int(preset.get("block_size", block_size))
        max_block_size = int(preset.get("max_block_size", max_block_size))
        resync_every = int(preset.get("resync_every", resync_every))
        length_hard_cap = int(preset.get("length_hard_cap", length_hard_cap))
        length_soft_cap = int(preset.get("length_soft_cap", length_soft_cap))
        low_tau_b1_threshold = float(
            preset.get("low_tau_b1_threshold", low_tau_b1_threshold)
        )
        low_tau_window = int(preset.get("low_tau_window", low_tau_window))
        low_tau_b1_steps = int(preset.get("low_tau_b1_steps", low_tau_b1_steps))
        zero_accept_page_b1_after = int(
            preset.get("zero_accept_page_b1_after", zero_accept_page_b1_after)
        )
        skip_verify_expected_below = float(
            preset.get("skip_verify_expected_below", skip_verify_expected_below)
        )
        use_conf_schedule = bool(preset.get("use_conf_schedule", use_conf_schedule))
        conf_schedule_min_expected = float(
            preset.get("conf_schedule_min_expected", conf_schedule_min_expected)
        )
        page_degrade_enabled = bool(
            preset.get("page_degrade_enabled", page_degrade_enabled)
        )

    total_started = time.perf_counter()
    prep_started = time.perf_counter()
    prepared = prepare_input_payload(
        handles.model,
        handles.tokenizer,
        Path(image_path),
        preset_name=preset_name,
        max_length=max_length,
    )
    preprocessing_seconds = time.perf_counter() - prep_started
    drafter, checkpoint = load_drafter(device=torch.device("cuda"))

    device = torch.device("cuda")
    tensors = payload_tensors(prepared.payload, device)
    policy = generation_policy(prepared.payload)
    # Ensure no-repeat is on even if payload capture omitted it.
    if int(policy.get("no_repeat_ngram_size", 0) or 0) <= 0:
        policy = {
            "no_repeat_ngram_size": int(config.INFERENCE.no_repeat_ngram_size),
            "ngram_window": int(config.INFERENCE.ngram_window),
        }
    prompt_ids = [int(x) for x in tensors["prompt_ids"][0].detach().cpu().tolist()]
    # Budget for *new* tokens after the multimodal prompt.
    max_generated = max(1, int(max_length) - len(prompt_ids))
    # Stage-1 hard cap: bounds runaway generation that kills wall-clock
    # (bench worst case grew 366→629 tokens). 0 disables.
    if length_hard_cap and length_hard_cap > 0:
        max_generated = min(max_generated, int(length_hard_cap))
    # Configured max block (promote ceiling); default active starts at B4.
    max_block = max(4, min(int(max_block_size), int(getattr(drafter.config, "max_block_size", 8))))
    default_block = max(4, min(int(block_size), max_block))
    eos_id = getattr(handles.tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = getattr(handles.model.config, "eos_token_id", 1)
    eos_id = int(eos_id)

    layer_indices = checkpoint.get("extra", {}).get(
        "layer_indices", list(config.INFERENCE.layer_indices)
    )
    tap = FeatureTap(handles.model, layer_indices)
    online = OnlineTarget(handles.model, tap, tensors, device)

    generated: List[int] = []
    cycles: List[Dict[str, Any]] = []
    accepted_histogram: Counter[int] = Counter()
    direct_commits = 0
    full_block_commits = 0
    fallback_rounds = 0
    pure_b1_rounds = 0
    resync_count = 0
    target_forwards = 0
    drafter_times: List[float] = []
    verifier_times: List[float] = []
    fallback_times: List[float] = []
    resync_times: List[float] = []
    recent_accept: List[int] = []
    pure_b1_cooldown = 0
    tokens_since_resync = 0
    stopped_on_eos = False
    active_block = int(default_block)
    degeneracy_hits = 0
    draft_accept_samples: List[int] = []
    skip_draft_rounds = 0  # consecutive pure-B1 from low-τ policy
    consecutive_zero_accept = 0
    page_degraded_to_b1 = False
    # Adaptive length: start at hard cap; if rolling τ strong, allow a bit more.
    effective_length_cap = int(max_generated)
    if length_hard_cap and length_hard_cap > 0:
        effective_length_cap = min(max_generated, int(length_hard_cap))
    max_generated = effective_length_cap

    original_sw = getattr(handles.model.config, "sliding_window", None)
    ring_window = getattr(handles.model.config, "sliding_window_size", None) or original_sw
    handles.model.config._ring_window = ring_window
    handles.model.config.sliding_window = None

    def _average(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _resync(reason: str) -> Dict[str, Any]:
        nonlocal resync_count, tokens_since_resync, target_forwards
        synchronize(device)
        started = time.perf_counter()
        new_state = online.prefill(generated)
        synchronize(device)
        ms = (time.perf_counter() - started) * 1000.0
        resync_times.append(ms)
        resync_count += 1
        tokens_since_resync = 0
        target_forwards += 1  # count full prefill as heavy target work
        cycles.append({
            "round": len(cycles) + 1,
            "event": "resync",
            "reason": reason,
            "generated": len(generated),
            "resync_ms": ms,
        })
        return new_state

    def _b1_commit(state: Dict[str, Any], token_id: int) -> Dict[str, Any]:
        nonlocal target_forwards, pure_b1_rounds
        generated.append(int(token_id))
        step = online.step(int(token_id), state["cache"], state["absolute_length"])
        target_forwards += 1
        pure_b1_rounds += 1
        return step

    try:
        state = online.prefill([])
        while len(generated) < max_generated:
            # Periodic correctness anchor for long documents.
            if (
                resync_every > 0
                and tokens_since_resync >= resync_every
                and len(generated) > 0
            ):
                state = _resync("interval")

            if _is_degenerate_tail(generated):
                degeneracy_hits += 1
                if degeneracy_hits >= 2:
                    # Bound long-doc runaway: stop rather than grinding to max_length.
                    cycles.append({
                        "round": len(cycles) + 1,
                        "event": "stop_degenerate",
                        "generated": len(generated),
                    })
                    break
                state = _resync("degenerate_tail")
                pure_b1_cooldown = max(pure_b1_cooldown, 12)
            else:
                degeneracy_hits = 0

            prefix = [*prompt_ids, *generated]
            anchor_id = policy_argmax(state["next_logits"], prefix, policy)
            remaining = max_generated - len(generated)
            if anchor_id == eos_id:
                generated.append(anchor_id)
                stopped_on_eos = True
                break

            # Length hard stop (cap already applied to max_generated).
            if len(generated) >= max_generated:
                cycles.append({
                    "round": len(cycles) + 1,
                    "event": "stop_length_cap",
                    "generated": len(generated),
                    "cap": max_generated,
                })
                break

            # Soft length stop: under weak τ, do not grind pure-B1 out to hard cap
            # (main cause of len_ratio 2×+ wall-clock damage).
            soft = int(length_soft_cap) if length_soft_cap and length_soft_cap > 0 else 0
            if soft > 0 and len(generated) >= soft and draft_accept_samples:
                w = draft_accept_samples[-max(4, int(low_tau_window)) :]
                if sum(w) / len(w) < float(low_tau_b1_threshold):
                    cycles.append({
                        "round": len(cycles) + 1,
                        "event": "stop_soft_length_low_tau",
                        "generated": len(generated),
                        "soft_cap": soft,
                        "mean_tau": sum(w) / len(w),
                    })
                    break
            # Also stop if page already degraded and we are well past soft floor.
            if (
                page_degraded_to_b1
                and soft > 0
                and len(generated) >= max(soft, int(0.75 * max_generated))
            ):
                cycles.append({
                    "round": len(cycles) + 1,
                    "event": "stop_degraded_length",
                    "generated": len(generated),
                })
                break

            # 8GB: shrink promote ceiling when free VRAM is critical (never lock B1).
            force_block_cap = int(max_block)
            if device.type == "cuda":
                try:
                    free_b, total_b = torch.cuda.mem_get_info()
                    free_ratio = float(free_b) / max(1.0, float(total_b))
                    if free_b < int(0.12 * 1024**3) or free_ratio < 0.03:
                        force_block_cap = min(force_block_cap, 4)
                except Exception:
                    pass

            # Dynamic γ: default B4, promote only on strong rolling accept.
            active_block = min(
                force_block_cap,
                _choose_active_block(
                    max_block, recent_accept, default_block=default_block
                ),
            )

            # Low-τ policy: weak recent accepts → temporary B1 (skip draft+verify).
            if (
                not page_degraded_to_b1
                and pure_b1_cooldown <= 0
                and len(draft_accept_samples) >= int(low_tau_window)
            ):
                window = draft_accept_samples[-int(low_tau_window) :]
                mean_tau = sum(window) / len(window)
                if mean_tau < float(low_tau_b1_threshold):
                    pure_b1_cooldown = max(pure_b1_cooldown, int(low_tau_b1_steps))
                    skip_draft_rounds += 1
                # If τ is strong, slightly relax remaining length budget once.
                if (
                    mean_tau >= 2.0
                    and length_hard_cap
                    and length_hard_cap > 0
                    and max_generated < min(
                        max(1, int(max_length) - len(prompt_ids)),
                        int(length_hard_cap) + 128,
                    )
                ):
                    max_generated = min(
                        max(1, int(max_length) - len(prompt_ids)),
                        int(length_hard_cap) + 128,
                    )

            # Whole-page degrade (disabled in lossless tier).
            if (
                page_degrade_enabled
                and not page_degraded_to_b1
                and int(zero_accept_page_b1_after) > 0
                and consecutive_zero_accept >= int(zero_accept_page_b1_after)
            ):
                page_degraded_to_b1 = True
                pure_b1_cooldown = 10**9  # rest of page
                cycles.append({
                    "round": len(cycles) + 1,
                    "event": "page_degrade_b1",
                    "consecutive_zero_accept": consecutive_zero_accept,
                    "generated": len(generated),
                })

            use_pure_b1 = pure_b1_cooldown > 0 or remaining < 3 or page_degraded_to_b1
            if use_pure_b1:
                if pure_b1_cooldown > 0 and not page_degraded_to_b1:
                    pure_b1_cooldown -= 1
                fb_started = time.perf_counter()
                state = _b1_commit(state, anchor_id)
                fallback_ms = (time.perf_counter() - fb_started) * 1000.0
                fallback_times.append(fallback_ms)
                fallback_rounds += 1
                tokens_since_resync += 1
                accepted_histogram[0] += 1
                cycles.append({
                    "round": len(cycles) + 1,
                    "accepted_draft": 0,
                    "direct_commit": False,
                    "pure_b1": True,
                    "page_degraded": page_degraded_to_b1,
                    "appended": 1,
                    "fallback_ms": fallback_ms,
                    "block_size": 1,
                })
                if anchor_id == eos_id:
                    stopped_on_eos = True
                    break
                if progress_callback is not None:
                    progress_callback({
                        "generated": len(generated),
                        "target": max_generated,
                        "round": len(cycles),
                        "accepted": 0,
                        "direct_commits": direct_commits,
                        "resync_count": resync_count,
                        "full_b8_ratio": (
                            full_block_commits / max(1, direct_commits)
                        ),
                    })
                continue

            draft_len = active_block - 1
            features = state["features"]
            anchor_tensor = torch.tensor([anchor_id], dtype=torch.long, device=device)
            anchor_embedding = handles.input_embeddings(anchor_tensor)
            synchronize(device)
            draft_started = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                draft_output = drafter.draft(
                    features,
                    handles.output_embeddings,
                    block_size=active_block,
                    anchor_embedding=anchor_embedding,
                    input_embeddings=handles.input_embeddings,
                )
            synchronize(device)
            draft_ms = (time.perf_counter() - draft_started) * 1000.0
            draft_ids = [int(x) for x in draft_output["tokens"][0].tolist()][:draft_len]
            drafter_times.append(draft_ms)

            # P0 confidence-scheduled verify length (trim doomed *suffix* only).
            conf_list: List[float] = []
            if "correctness_probability" in draft_output:
                conf_list = [
                    float(x)
                    for x in draft_output["correctness_probability"][0][:draft_len]
                    .detach()
                    .cpu()
                    .tolist()
                ]
            expected_acc = float(
                draft_output.get("expected_accepted", torch.tensor([0.0]))[0]
                .detach()
                .cpu()
                .item()
            ) if "expected_accepted" in draft_output else 0.0
            # DSpark-style verify_len from (optionally calibrated) conf.
            if conf_list and use_conf_schedule:
                from .dflash_tiers import load_calibration, schedule_verify_len

                cal = load_calibration()
                if expected_acc >= float(conf_schedule_min_expected):
                    vlen = schedule_verify_len(conf_list, calibration=cal, min_len=1)
                    if vlen <= 0:
                        # pos0 doomed under calibrated conf → B1 without verify
                        skip_verify = True
                    elif vlen < len(draft_ids):
                        draft_ids = draft_ids[:vlen]
                        skip_verify = False
                    else:
                        skip_verify = False
                else:
                    skip_verify = False
            elif conf_list and expected_acc >= 0.9:
                verify_len = _confidence_verify_len(
                    conf_list,
                    min_len=1,
                    conf_floor=conf_floor,
                    survival_floor=survival_floor,
                )
                if 0 < verify_len < len(draft_ids):
                    draft_ids = draft_ids[:verify_len]
                skip_verify = False
            else:
                skip_verify = False

            # Fast/balanced: also skip verify when expected accept is clearly doomed.
            if (
                not skip_verify
                and float(skip_verify_expected_below) > 0
                and expected_acc > 0
                and expected_acc < float(skip_verify_expected_below)
                and (not conf_list or float(conf_list[0]) < 0.35)
            ):
                skip_verify = True

            if skip_verify:
                fb_started = time.perf_counter()
                state = _b1_commit(state, anchor_id)
                fallback_ms = (time.perf_counter() - fb_started) * 1000.0
                fallback_times.append(fallback_ms)
                fallback_rounds += 1
                tokens_since_resync += 1
                recent_accept.append(0)
                recent_accept = recent_accept[-16:]
                draft_accept_samples.append(0)
                consecutive_zero_accept += 1
                pure_b1_cooldown = max(pure_b1_cooldown, 1)
                accepted_histogram[0] += 1
                cycles.append({
                    "round": len(cycles) + 1,
                    "accepted_draft": 0,
                    "direct_commit": False,
                    "pure_b1": True,
                    "skip_verify": True,
                    "expected_acc": expected_acc,
                    "appended": 1,
                    "draft_ms": draft_ms,
                    "fallback_ms": fallback_ms,
                    "block_size": active_block,
                })
                continue

            try:
                block_cache = clone_cache(state["cache"])
                if not cache_tensors_are_disjoint(state["cache"], block_cache):
                    raise RuntimeError("Stable DFlash cache clone aliases the live cache")
                block_result = online.verify_block(
                    anchor_id,
                    draft_ids,
                    block_cache,
                    state["absolute_length"],
                    prefix,
                    policy,
                )
            except RuntimeError as exc:
                # OOM / cache clone failure on 8GB → pure B1 + resync recovery.
                msg = str(exc).lower()
                if "out of memory" in msg or "cuda" in msg:
                    try:
                        del block_cache  # type: ignore[name-defined]
                    except Exception:
                        pass
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    state = _resync("oom_recover")
                    pure_b1_cooldown = max(pure_b1_cooldown, 24)
                    fb_started = time.perf_counter()
                    state = _b1_commit(state, anchor_id)
                    fallback_ms = (time.perf_counter() - fb_started) * 1000.0
                    fallback_times.append(fallback_ms)
                    fallback_rounds += 1
                    tokens_since_resync += 1
                    recent_accept.append(0)
                    recent_accept = recent_accept[-16:]
                    draft_accept_samples.append(0)
                    accepted_histogram[0] += 1
                    cycles.append({
                        "round": len(cycles) + 1,
                        "event": "oom_fallback_b1",
                        "accepted_draft": 0,
                        "appended": 1,
                        "draft_ms": draft_ms,
                        "block_size": active_block,
                    })
                    continue
                raise
            target_forwards += 1
            verifier_ms = float(block_result["seconds"] * 1000.0)
            verifier_times.append(verifier_ms)
            verifier_predictions = [int(x) for x in block_result["predictions"]]
            accepted = 0
            for candidate, target_id in zip(draft_ids, verifier_predictions):
                if candidate != target_id:
                    break
                accepted += 1

            # Always keep the target-verified anchor; add matching draft prefix.
            appended = [anchor_id, *draft_ids[:accepted]]
            if len(appended) > remaining:
                appended = appended[:remaining]
                accepted = max(0, len(appended) - 1)

            # If nothing beyond anchor matched, prefer strict B1 for the anchor
            # only (do not import a partially wrong block cache).
            if accepted == 0:
                try:
                    del block_cache
                except Exception:
                    pass
                if device.type == "cuda" and (len(cycles) % 8 == 0):
                    torch.cuda.empty_cache()
                fb_started = time.perf_counter()
                state = _b1_commit(state, anchor_id)
                fallback_ms = (time.perf_counter() - fb_started) * 1000.0
                fallback_times.append(fallback_ms)
                fallback_rounds += 1
                tokens_since_resync += 1
                recent_accept.append(0)
                recent_accept = recent_accept[-16:]
                draft_accept_samples.append(0)
                consecutive_zero_accept += 1
                accepted_histogram[0] += 1
                cycles.append({
                    "round": len(cycles) + 1,
                    "accepted_draft": 0,
                    "direct_commit": False,
                    "pure_b1": True,
                    "appended": 1,
                    "draft_ms": draft_ms,
                    "block_ms": verifier_ms,
                    "fallback_ms": fallback_ms,
                    "block_size": active_block,
                })
                if progress_callback is not None:
                    progress_callback({
                        "generated": len(generated),
                        "target": max_generated,
                        "round": len(cycles),
                        "accepted": 0,
                        "direct_commits": direct_commits,
                        "resync_count": resync_count,
                    })
                continue

            # Commit verified prefix via cropped block cache (speed path).
            generated.extend(appended)
            direct_commits += 1
            # full-block relative to the *configured* draft, not the truncated one
            if accepted >= (active_block - 1) and accepted > 0:
                full_block_commits += 1
            recent_accept.append(accepted)
            recent_accept = recent_accept[-16:]
            draft_accept_samples.append(accepted)
            if accepted == 0:
                consecutive_zero_accept += 1
            else:
                consecutive_zero_accept = 0
            accepted_histogram[accepted] += 1
            tokens_since_resync += len(appended)
            # High accept clears B1 cooldown so we stay in draft mode.
            if accepted >= 2:
                pure_b1_cooldown = 0
            elif accepted == 0:
                pure_b1_cooldown = max(pure_b1_cooldown, 2)

            if eos_id in appended:
                eos_at = appended.index(eos_id)
                trailing = len(appended) - eos_at - 1
                if trailing > 0:
                    del generated[-trailing:]
                stopped_on_eos = True
                strict_ms = 0.0
            else:
                committed_cache = block_result["cache"]
                physical_target = (
                    int(block_result["physical_cache_length_before"]) + len(appended)
                )
                crop = getattr(committed_cache, "crop", None)
                if not callable(crop):
                    # Fallback: rebuild from true prefix if crop unavailable.
                    state = _resync("crop_unavailable")
                    strict_ms = 0.0
                else:
                    crop(physical_target)
                    # logits index: position `accepted` is the logit after the
                    # last accepted draft (0-based over draft slots).
                    next_logits = block_result["logits"][:, accepted, :]
                    try:
                        features_next = online.tap.position_features(accepted)
                    except Exception:
                        features_next = block_result.get("features")
                    state = {
                        "cache": committed_cache,
                        "next_logits": next_logits,
                        "features": features_next,
                        "absolute_length": int(state["absolute_length"]) + len(appended),
                        "seconds": 0.0,
                    }
                    strict_ms = 0.0

            cycles.append({
                "round": len(cycles) + 1,
                "accepted_draft": accepted,
                "direct_commit": True,
                "full_block_commit": accepted == draft_len,
                "appended": len(appended),
                "draft_ms": draft_ms,
                "block_ms": verifier_ms,
                "fallback_ms": strict_ms,
                "block_size": active_block,
            })
            if progress_callback is not None:
                progress_callback({
                    "generated": len(generated),
                    "target": max_generated,
                    "round": len(cycles),
                    "accepted": accepted,
                    "direct_commits": direct_commits,
                    "resync_count": resync_count,
                    "full_b8_ratio": full_block_commits / max(1, direct_commits),
                })
            if stopped_on_eos:
                break
            # Stage-1: no weak-accept resync — full prefill is expensive on 8GB.
            # Interval / degenerate / OOM resync only (see loop head + OOM path).
    finally:
        tap.close()
        handles.model.config.sliding_window = original_sw
        if prepared.scratch_dir.exists():
            shutil.rmtree(prepared.scratch_dir, ignore_errors=True)

    raw = handles.tokenizer.decode(generated, skip_special_tokens=False)
    markdown = clean_markdown(raw)
    # Mean acceptance only over actual draft attempts (exclude pure-B1 filler).
    accepted_values = [int(x) for x in draft_accept_samples]
    sum_draft = float(sum(drafter_times))
    sum_verify = float(sum(verifier_times))
    sum_resync = float(sum(resync_times))
    sum_b1 = float(sum(fallback_times))
    cost_denom = max(1e-6, sum_draft + sum_verify + sum_resync + sum_b1)
    result = StableDFlashResult(
        markdown=markdown,
        plain_text=_strip_markdown(markdown),
        raw_output=raw,
        generated_tokens=len(generated),
        total_seconds=time.perf_counter() - total_started,
        preprocessing_seconds=preprocessing_seconds,
        speculative_rounds=len(draft_accept_samples),
        direct_block_commits=direct_commits,
        full_block_commits=full_block_commits,
        fallback_rounds=fallback_rounds,
        resync_count=resync_count,
        pure_b1_rounds=pure_b1_rounds,
        direct_committed_tokens=sum(
            int(x.get("appended", 0)) for x in cycles if x.get("direct_commit")
        ),
        target_decode_forwards=target_forwards,
        mean_accepted_draft=_average(accepted_values),
        full_block_ratio=(
            full_block_commits / direct_commits if direct_commits else 0.0
        ),
        acceptance_histogram={
            str(k): int(v) for k, v in sorted(accepted_histogram.items())
        },
        drafter_latency_ms=_average(drafter_times),
        block_verifier_latency_ms=_average(verifier_times),
        strict_fallback_latency_ms=_average(fallback_times),
        resync_latency_ms=_average(resync_times),
        total_draft_ms=sum_draft,
        total_verify_ms=sum_verify,
        total_resync_ms=sum_resync,
        total_b1_ms=sum_b1,
        cost_share_draft=sum_draft / cost_denom,
        cost_share_verify=sum_verify / cost_denom,
        cost_share_resync=sum_resync / cost_denom,
        cost_share_b1=sum_b1 / cost_denom,
        page_degraded_to_b1=page_degraded_to_b1,
        length_hard_cap_used=int(max_generated),
        target_forward_reduction=(
            (len(generated) / target_forwards) if target_forwards else 0.0
        ),
        checkpoint_path=str(config.discover_weight() or "N/A"),
        checkpoint_step=int(checkpoint.get("step", -1)),
        block_size=default_block,
        stopped_on_eos=stopped_on_eos,
        tier=str(tier or ""),
        warning=(
            {
                "fast": (
                    "稳定 DFlash · 快速档：墙钟优先，软/硬长度帽与低τ停写；"
                    "可能比官方 stable 输出更短。"
                ),
                "balanced": (
                    "稳定 DFlash · 均衡档：中等软帽，完整度与速度折中。"
                ),
                "lossless": (
                    "稳定 DFlash · 无损档：无 soft 截断；仅验证前缀 + conf 调度。"
                    "当前 live 接受率下墙钟未必快于普通版。"
                ),
            }.get(str(tier or "").lower(), "")
            or (
                "稳定 DFlash（sys_v2）：锁定 B4、软/硬长度帽、低τ停写与整页降级、"
                "低期望跳过 verify；成本拆分 draft/verify/resync/B1。"
            )
        ),
    )
    return result


# ---------------------------------------------------------------------------
# Exactness probe
# ---------------------------------------------------------------------------

def _run_exactness_probe(
    handles: ModelHandles,
    capture: Any,  # CaptureResult from payload_capture
    drafter: Any,
    checkpoint: Dict[str, Any],
    *,
    block_size: int = config.INFERENCE.block_size,
    max_new_tokens: int = config.INFERENCE.probe_max_new_tokens,
    warmup_cycles: int = config.INFERENCE.warmup_cycles,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> ProbeResult:
    """Run the Stage 9A B8 strict-B1-replay exactness probe on a live payload.

    This is a Windows-native port of ``stage9_v2_b8_exactness_probe.py:run_case()``.
    """
    device = torch.device("cuda")
    payload = capture.payload
    tensors = payload_tensors(payload, device)
    policy = generation_policy(payload)
    prompt_token_ids = tensors["prompt_ids"][0].detach().cpu().tolist()
    official = tensors["official_generated"][0].detach().cpu().tolist()
    target_tokens = min(len(official), max_new_tokens)

    layer_indices = checkpoint.get("extra", {}).get(
        "layer_indices", list(config.INFERENCE.layer_indices)
    )

    tap = FeatureTap(handles.model, layer_indices)
    online = OnlineTarget(handles.model, tap, tensors, device)
    result = ProbeResult(
        checkpoint_path=str(config.discover_weight() or "N/A"),
        checkpoint_step=int(checkpoint.get("step", -1)),
        parameter_count=sum(p.numel() for p in drafter.parameters()),
        block_size=block_size,
        dtype_name=str(next(drafter.parameters()).dtype).removeprefix("torch."),
        layer_indices=list(layer_indices),
        generation_policy=policy,
    )

    try:
        # --- Strict B1 oracle preflight ---------------------------------------
        preflight = _strict_oracle_preflight(
            online, prompt_token_ids, official, target_tokens, policy
        )
        result.tokens_checked = int(preflight.get("tokens_checked", 0))
        result.official_token_ids = [int(x) for x in official[:target_tokens]]
        result.strict_replay_token_ids = [int(x) for x in preflight.get("generated_tokens", [])]
        if not preflight.get("exact", False):
            result.failure = {
                "kind": "strict_b1_preflight_vs_official",
                "position": preflight.get("first_mismatch"),
                "generated": preflight.get("generated"),
                "official": preflight.get("official"),
            }
            return result

        # --- Main speculative loop -------------------------------------------
        state = online.prefill([])
        initial_prefill_ms = float(state["seconds"] * 1000.0)

        generated: List[int] = []
        cycles: List[Dict[str, Any]] = []
        accepted_histogram: Counter[int] = Counter()
        position_total = [0] * (block_size - 1)
        position_correct = [0] * (block_size - 1)
        block_strict_disagreements = 0
        block_positions_compared = 0
        official_match = True

        while len(generated) < target_tokens:
            cycle_idx = len(cycles)
            anchor_prefix = [*prompt_token_ids, *generated]
            anchor_id = policy_argmax(state["next_logits"], anchor_prefix, policy)
            official_anchor = int(official[len(generated)])
            if anchor_id != official_anchor:
                result.failure = {
                    "kind": "anchor_vs_official",
                    "position": len(generated),
                    "generated": anchor_id,
                    "official": official_anchor,
                }
                official_match = False
                break

            features = state["features"]
            anchor_tensor = torch.tensor([anchor_id], dtype=torch.long, device=device)
            anchor_embedding = handles.input_embeddings(anchor_tensor)
            synchronize(device)
            draft_started = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                 enabled=device.type == "cuda"):
                draft_output = drafter.draft(
                    features, handles.output_embeddings,
                    block_size=block_size, anchor_embedding=anchor_embedding,
                )
            synchronize(device)
            draft_seconds = time.perf_counter() - draft_started
            draft_ids = [int(x) for x in draft_output["tokens"][0].tolist()]

            # Diagnostic block verifier on a cloned cache.
            block_cache = clone_cache(state["cache"])
            if not cache_tensors_are_disjoint(state["cache"], block_cache):
                raise RuntimeError(
                    "Block cache clone shares tensor storage with authoritative B1 cache"
                )
            block_result = online.verify_block(
                anchor_id, draft_ids, block_cache,
                state["absolute_length"],
                [*prompt_token_ids, *generated], policy,
            )

            # Strict B1 replay — the ONLY commit authority.
            replay_cache = state["cache"]
            replay_length = state["absolute_length"]
            replay_step_seconds = 0.0
            appended = [anchor_id]
            accepted = 0
            fallback_id: Optional[int] = None
            strict_predictions: List[int] = []

            first_step = online.step(anchor_id, replay_cache, replay_length)
            replay_cache = first_step["cache"]
            replay_length = first_step["absolute_length"]
            next_logits = first_step["next_logits"]
            next_features = first_step["features"]
            replay_step_seconds += float(first_step["seconds"])

            remaining = target_tokens - (len(generated) + 1)
            for position, candidate in enumerate(draft_ids[:remaining]):
                strict_prefix = [*prompt_token_ids, *generated, *appended]
                strict_id = policy_argmax(next_logits, strict_prefix, policy)
                strict_predictions.append(strict_id)
                block_id = int(block_result["predictions"][position])
                block_positions_compared += 1
                if block_id != strict_id:
                    block_strict_disagreements += 1
                position_total[position] += 1
                if candidate == strict_id:
                    position_correct[position] += 1
                    appended.append(candidate)
                    accepted += 1
                    replay = online.step(candidate, replay_cache, replay_length)
                    replay_cache = replay["cache"]
                    replay_length = replay["absolute_length"]
                    next_logits = replay["next_logits"]
                    next_features = replay["features"]
                    replay_step_seconds += float(replay["seconds"])
                    continue
                # First mismatch: emit strict token, stop drafting.
                fallback_id = strict_id
                appended.append(strict_id)
                replay = online.step(strict_id, replay_cache, replay_length)
                replay_cache = replay["cache"]
                replay_length = replay["absolute_length"]
                next_logits = replay["next_logits"]
                next_features = replay["features"]
                replay_step_seconds += float(replay["seconds"])
                break

            appended = appended[: target_tokens - len(generated)]
            generated.extend(appended)
            accepted_histogram[accepted] += 1

            ok, mismatch, got, expected = _check_against_official(generated, official)
            if not ok:
                official_match = False
                result.failure = {
                    "kind": "committed_output_vs_official",
                    "position": mismatch,
                    "generated": got,
                    "official": expected,
                }

            cycles.append({
                "cycle": cycle_idx,
                "anchor": anchor_id,
                "draft": draft_ids,
                "accepted_draft": accepted,
                "fallback": fallback_id,
                "appended": appended,
                "draft_ms": float(draft_seconds * 1000.0),
                "block_verify_ms": float(block_result["seconds"] * 1000.0),
                "strict_replay_ms": float(replay_step_seconds * 1000.0),
            })
            if progress_callback is not None:
                accepted_so_far = [int(x["accepted_draft"]) for x in cycles]
                progress_callback({
                    "generated": len(generated),
                    "target": target_tokens,
                    "round": len(cycles),
                    "accepted": accepted,
                    "mean_accepted": sum(accepted_so_far) / len(accepted_so_far),
                    "full_b8_ratio": (
                        sum(x == block_size - 1 for x in accepted_so_far)
                        / len(accepted_so_far)
                    ),
                    "draft_ms": float(draft_seconds * 1000.0),
                    "block_ms": float(block_result["seconds"] * 1000.0),
                    "replay_ms": float(replay_step_seconds * 1000.0),
                })
            if result.failure is not None:
                break

            state = {
                "cache": replay_cache,
                "next_logits": next_logits,
                "features": next_features,
                "absolute_length": replay_length,
                "seconds": 0.0,
            }

        # --- Aggregate metrics -------------------------------------------------
        result.generated_tokens = len(generated)
        result.speculative_rounds = len(cycles)
        result.cycles_detail = cycles

        generated_trimmed = generated[:target_tokens]
        result.strict_replay_token_ids = [int(x) for x in generated_trimmed]
        official_trimmed = official[: len(generated_trimmed)]
        mismatch = first_mismatch(generated_trimmed, official_trimmed)

        measured = [x for x in cycles if not x.get("warmup")]
        accepted_values = [int(x["accepted_draft"]) for x in measured]
        full_blocks = sum(x == block_size - 1 for x in accepted_values)

        def _avg(fld: str) -> float:
            vals = [float(x[fld]) for x in measured]
            return sum(vals) / len(vals) if vals else 0.0

        result.mean_accepted_draft = (
            sum(accepted_values) / len(accepted_values) if accepted_values else 0.0
        )
        result.effective_tokens_per_round = 1.0 + result.mean_accepted_draft
        result.full_block_count = full_blocks
        result.full_block_ratio = full_blocks / len(cycles) if cycles else 0.0
        result.acceptance_histogram = {
            str(k): int(v) for k, v in sorted(accepted_histogram.items())
        }
        result.per_position_accuracy = [
            (position_correct[i] / position_total[i]) if position_total[i] else 0.0
            for i in range(block_size - 1)
        ]
        result.drafter_latency_ms = _avg("draft_ms")
        result.block_verifier_latency_ms = _avg("block_verify_ms")
        result.strict_replay_latency_ms = _avg("strict_replay_ms")
        result.block_vs_b1_disagreement_rate = (
            block_strict_disagreements / block_positions_compared
            if block_positions_compared else 0.0
        )
        result.final_token_exactness = (mismatch is None)
        result.first_mismatch = mismatch
        result.probe_ok = (
            official_match
            and mismatch is None
            and len(generated_trimmed) == target_tokens
        )
        # Theoretical speedup = effective_tokens_per_round (1 accepted +
        # mean_accepted_draft) vs. 1 token per round baseline.
        result.theoretical_speedup = result.effective_tokens_per_round

    finally:
        tap.close()
        # Clean up probe scratch dir.
        if capture.scratch_dir.exists():
            shutil.rmtree(capture.scratch_dir, ignore_errors=True)

    return result


def _strict_oracle_preflight(
    online: OnlineTarget,
    prompt_token_ids: Sequence[int],
    official: Sequence[int],
    target_tokens: int,
    policy: Dict[str, int],
) -> Dict[str, Any]:
    """Prove that manual q_len=1 replay matches the captured oracle first."""
    state = online.prefill([])
    generated: List[int] = []
    started = time.perf_counter()
    while len(generated) < target_tokens:
        prefix = [*[int(x) for x in prompt_token_ids], *generated]
        token_id = policy_argmax(state["next_logits"], prefix, policy)
        expected = int(official[len(generated)])
        if token_id != expected:
            return {
                "exact": False,
                "tokens_checked": len(generated),
                "first_mismatch": len(generated),
                "generated": token_id,
                "official": expected,
                "generated_tokens": generated,
            }
        generated.append(token_id)
        if len(generated) < target_tokens:
            state = online.step(token_id, state["cache"], state["absolute_length"])
    return {
        "exact": True,
        "tokens_checked": len(generated),
        "first_mismatch": None,
        "generated_tokens": generated,
    }


def _check_against_official(
    generated: Sequence[int], official: Sequence[int]
) -> tuple:
    mismatch = first_mismatch(generated, official[: len(generated)])
    if mismatch is None:
        return True, None, None, None
    got = int(generated[mismatch]) if mismatch < len(generated) else None
    expected = int(official[mismatch]) if mismatch < len(official) else None
    return False, mismatch, got, expected


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_precise_mode(
    handles: ModelHandles,
    image_path: Path,
    *,
    preset_name: str = "gundam",
    max_length: int = config.INFERENCE.max_length,
    max_new_tokens: int = config.INFERENCE.probe_max_new_tokens,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> ProbeResult:
    """Run Mini UFlash precise mode on a single image.

    This performs:
    1. Live payload capture (one official .infer() call).
    2. Drafter load (lazy, cached).
    3. B8 strict-B1-replay exactness probe.

    The official OCR text is always from step 1 (the official path).
    """
    total_started = time.perf_counter()

    # 1. Live payload capture.
    cap_started = time.perf_counter()
    capture = capture_payload(
        handles.model, handles.tokenizer, image_path,
        preset_name=preset_name, max_length=max_length,
    )
    cap_seconds = time.perf_counter() - cap_started

    # 2. Load drafter (no-op if already loaded).
    drafter, ckpt = load_drafter(device=torch.device("cuda"))

    # 3. Wrap sliding window for the probe.
    original_sw = getattr(handles.model.config, "sliding_window", None)
    ring_window = (
        getattr(handles.model.config, "sliding_window_size", None)
        or original_sw
    )
    handles.model.config._ring_window = ring_window
    handles.model.config.sliding_window = None

    try:
        probe_started = time.perf_counter()
        result = _run_exactness_probe(
            handles, capture, drafter, ckpt,
            max_new_tokens=max_new_tokens,
            progress_callback=progress_callback,
        )
        result.probe_seconds = time.perf_counter() - probe_started
    finally:
        handles.model.config.sliding_window = original_sw

    result.official_markdown = clean_markdown(capture.decoded_text)
    result.official_raw = capture.decoded_text
    # Plain text: strip markdown formatting.
    result.official_plain_text = _strip_markdown(result.official_markdown)
    result.payload_capture_seconds = cap_seconds
    result.total_seconds = time.perf_counter() - total_started
    try:
        result.vram_gb = torch.cuda.memory_allocated() / 1024**3
    except Exception:
        pass
    return result


def _strip_markdown(text: str) -> str:
    """Light markdown → plain-text."""
    import re
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    return text.strip()


if __name__ == "__main__":
    print("mini_uflash_engine.py — import this module, do not run it directly.")
