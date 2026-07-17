#!/usr/bin/env python3
"""Continue-train Mini UFlash Stage 11B drafter on Windows 8GB.

Designed for RTX 5060 Laptop ~8GB VRAM + ~16GB host RAM:
  * micro-batch + gradient accumulation (effective batch ≈ 32)
  * lazy teacher page cache (does not load all pages into RAM)
  * freezes Unlimited-OCR embeddings + LM head only (full OCR not kept)

Does nothing until you run this script. Safe to import.

Usage (from project root)::

    .\\.venv\\Scripts\\python.exe train\\train_continue_8g.py --teacher train\\data\\teachers\\train
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train import config as cfg  # noqa: E402
from train.lib.losses import compute_training_loss  # noqa: E402
from train.lib.metrics import block_metrics  # noqa: E402
from train.lib.teacher_data import (  # noqa: E402
    LazyTeacherStore,
    TeacherIndexEntry,
    TeacherPage,
    build_index,
    discover_teacher_files,
    load_split_manifest,
)
from train.lib.utils import (  # noqa: E402
    atomic_write_json,
    empty_cuda,
    parse_floats,
    parse_ints,
    seed_everything,
    weighted_choice,
)
from webapp.mini_uflash_core.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402


def resolve_embedding_modules(target: nn.Module) -> Tuple[nn.Module, nn.Module]:
    lm_head = target.get_output_embeddings()
    input_embeddings = target.get_input_embeddings()
    if lm_head is None:
        lm_head = getattr(target, "lm_head", None)
    if input_embeddings is None:
        candidates = [
            getattr(getattr(target, "model", None), "embed_tokens", None),
            getattr(getattr(getattr(target, "model", None), "model", None), "embed_tokens", None),
        ]
        input_embeddings = next((x for x in candidates if x is not None), None)
    if lm_head is None or input_embeddings is None:
        raise RuntimeError("Could not resolve target input embeddings and LM head")
    return input_embeddings, lm_head


def load_frozen_target_heads(model_path: str, dtype: torch.dtype) -> Tuple[nn.Module, nn.Module]:
    print("Loading frozen Unlimited-OCR embeddings + LM head (temporary full model)...", flush=True)
    last_exc: Exception | None = None
    target = None
    for backend in ("sdpa", "eager"):
        try:
            target = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
                use_safetensors=True,
                torch_dtype=dtype,
                attn_implementation=backend,
            ).eval()
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    if target is None:
        raise RuntimeError(f"Failed to load target model: {last_exc}")

    # Move only what we need; free the rest ASAP for 8GB cards.
    input_embeddings, lm_head = resolve_embedding_modules(target)
    input_embeddings = input_embeddings.to("cuda")
    lm_head = lm_head.to("cuda")
    for module in (input_embeddings, lm_head):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    # Drop parent model graph while keeping submodule references alive.
    del target
    empty_cuda()
    print("Frozen heads ready; full Unlimited-OCR released from VRAM.", flush=True)
    return input_embeddings, lm_head


def _safe_anchor_window(
    page: TeacherPage, anchor: int, draft_len: int
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return feature/anchor/target/hidden tensors only if the full window fits."""
    end_id = anchor + 1 + draft_len
    end_hid = anchor + draft_len
    if anchor < 0:
        return None
    if end_id > int(page.generated_ids.numel()):
        return None
    if end_hid > int(page.predictive_hidden.shape[0]):
        return None
    if anchor >= int(page.target_features.shape[0]):
        return None
    targets = page.generated_ids[anchor + 1 : end_id]
    hidden = page.predictive_hidden[anchor:end_hid]
    if int(targets.numel()) != draft_len or int(hidden.shape[0]) != draft_len:
        return None
    return (
        page.target_features[anchor],
        page.generated_ids[anchor],
        targets,
        hidden,
    )


def sample_batch(
    store: LazyTeacherStore,
    block_size: int,
    batch_size: int,
    rng: random.Random,
    *,
    anchors_per_page: int = 1,
) -> Dict[str, torch.Tensor]:
    """Sample a micro-batch; optionally take multiple anchors from one page (P2)."""
    draft_len = block_size - 1
    eligible = store.eligible_indices(draft_len)
    if not eligible:
        raise ValueError(f"No teacher page supports draft length {draft_len}")

    features, anchor_ids, targets, target_hidden = [], [], [], []
    attempts = 0
    anchors_per_page = max(1, int(anchors_per_page))
    while len(features) < batch_size and attempts < batch_size * 40:
        attempts += 1
        page_index = rng.choice(eligible)
        page = store.get(page_index)
        max_anchor = min(
            page.valid_anchor_count - draft_len,
            int(page.generated_ids.numel()) - 1 - draft_len,
            int(page.predictive_hidden.shape[0]) - draft_len,
        )
        if max_anchor < 0:
            continue
        # Multi-anchor denser sampling from the same teacher page (still one page
        # load in the lazy cache — 8GB-host friendly).
        n_take = min(anchors_per_page, batch_size - len(features), max_anchor + 1)
        chosen = set()
        for _ in range(n_take * 3):
            if len(chosen) >= n_take:
                break
            chosen.add(rng.randint(0, max_anchor))
        for anchor in chosen:
            if len(features) >= batch_size:
                break
            window = _safe_anchor_window(page, anchor, draft_len)
            if window is None:
                continue
            feat, aid, tgt, hid = window
            features.append(feat)
            anchor_ids.append(aid)
            targets.append(tgt)
            target_hidden.append(hid)

    if len(features) < batch_size:
        raise RuntimeError(
            f"Could not sample a full micro-batch for B{block_size} "
            f"(got {len(features)}/{batch_size})"
        )

    return {
        "target_features": torch.stack(features),
        "anchor_ids": torch.stack(anchor_ids).long(),
        "targets": torch.stack(targets).long(),
        "target_hidden": torch.stack(target_hidden),
    }


@torch.inference_mode()
def evaluate_block(
    model: nn.Module,
    input_embeddings: nn.Module,
    lm_head: nn.Module,
    store: LazyTeacherStore,
    block_size: int,
    batch_size: int,
    max_samples_per_page: int,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    draft_len = block_size - 1
    rows: List[Tuple[int, int]] = []
    for page_index in store.eligible_indices(draft_len):
        page = store.get(page_index)
        max_anchor = min(
            page.valid_anchor_count - draft_len,
            int(page.generated_ids.numel()) - 1 - draft_len,
            int(page.predictive_hidden.shape[0]) - draft_len,
        )
        if max_anchor < 0:
            continue
        count = max_anchor + 1
        if max_samples_per_page > 0 and count > max_samples_per_page:
            indices = (
                torch.linspace(0, count - 1, max_samples_per_page).round().long().tolist()
            )
        else:
            indices = list(range(count))
        for anchor in indices:
            if _safe_anchor_window(page, int(anchor), draft_len) is not None:
                rows.append((page_index, int(anchor)))

    if not rows:
        return {
            "position_accuracy": [0.0] * draft_len,
            "full_draft_accuracy": 0.0,
            "average_accepted_draft": 0.0,
            "average_effective_emitted": 1.0,
            "draft_length": draft_len,
            "acceptance_histogram": {},
            "samples": 0,
        }

    predictions, targets_out = [], []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        windows = []
        for page_index, anchor in batch_rows:
            page = store.get(page_index)
            window = _safe_anchor_window(page, anchor, draft_len)
            if window is not None:
                windows.append(window)
        if not windows:
            continue
        features = torch.stack([w[0] for w in windows]).to(device)
        anchor_ids = torch.stack([w[1] for w in windows]).long().to(device)
        targets = torch.stack([w[2] for w in windows]).long()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            anchor_emb = input_embeddings(anchor_ids)
            # Free-running draft (Markov sequential sample when enabled) —
            # matches live stable_dflash path more closely than teacher-force.
            if hasattr(model, "draft"):
                draft_out = model.draft(
                    features,
                    lm_head,
                    block_size=block_size,
                    anchor_embedding=anchor_emb,
                    input_embeddings=input_embeddings,
                )
                pred = draft_out["tokens"].cpu()
            else:
                output = model(
                    features,
                    block_size=block_size,
                    anchor_embedding=anchor_emb,
                )
                pred = lm_head(output["hidden"]).argmax(dim=-1).cpu()
        predictions.append(pred)
        targets_out.append(targets)

    prediction = torch.cat(predictions)
    target = torch.cat(targets_out)
    metrics = block_metrics(prediction, target)
    metrics["block_size"] = block_size
    return metrics


def evaluate_all(
    model: nn.Module,
    input_embeddings: nn.Module,
    lm_head: nn.Module,
    store: LazyTeacherStore,
    block_sizes: Sequence[int],
    batch_size: int,
    max_samples_per_page: int,
    device: torch.device,
) -> Dict[str, object]:
    return {
        str(block): evaluate_block(
            model,
            input_embeddings,
            lm_head,
            store,
            block_size=block,
            batch_size=batch_size,
            max_samples_per_page=max_samples_per_page,
            device=device,
        )
        for block in block_sizes
    }


def model_score(metrics: Mapping[str, object], primary_block: int) -> float:
    """Selection score: prefer longer accepted prefixes and high pos0 accuracy."""
    primary = metrics[str(primary_block)]
    score = float(primary["average_accepted_draft"])
    score += 0.20 * float(primary["full_draft_accuracy"])
    # Stage-2: pos0 is highest-leverage (whole block dies if first token fails).
    pos = primary.get("position_accuracy") or []
    if pos:
        score += 0.35 * float(pos[0])
        if len(pos) > 1:
            score += 0.10 * float(pos[1])
    if "6" in metrics and primary_block != 6:
        score += 0.03 * float(metrics["6"]["average_accepted_draft"])
    if "4" in metrics and primary_block != 4:
        # B4 matches Stage-1 default inference γ — weight it more.
        score += 0.08 * float(metrics["4"]["average_accepted_draft"])
        b4_pos = metrics["4"].get("position_accuracy") or []
        if b4_pos:
            score += 0.12 * float(b4_pos[0])
    return score


def format_metrics(metrics: Mapping[str, object], block: int) -> str:
    item = metrics[str(block)]
    pos = "/".join(f"{x:.1%}" for x in item["position_accuracy"])
    return (
        f"B{block}={item['average_accepted_draft']:.3f}/{block - 1} "
        f"full={item['full_draft_accuracy']:.2%} pos={pos}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Continue Mini UFlash Stage 11B on Windows 8GB")
    p.add_argument("--teacher", type=str, default=str(cfg.TEACHERS_TRAIN))
    p.add_argument("--val-teacher", type=str, default=str(cfg.TEACHERS_VAL))
    p.add_argument("--resume-checkpoint", type=str, default=str(cfg.resume_weight()))
    p.add_argument("--model-path", type=str, default=str(cfg.model_path()))
    p.add_argument("--output-dir", type=str, default=str(cfg.RUNS_DIR / "stage11b_win_continue"))
    p.add_argument("--split-manifest", type=str, default=None)
    p.add_argument("--block-sizes", type=str, default=",".join(str(x) for x in cfg.TRAIN_BLOCK_SIZES))
    p.add_argument(
        "--block-probs",
        type=str,
        default=",".join(str(x) for x in cfg.TRAIN_BLOCK_PROBS),
    )
    p.add_argument("--steps", type=int, default=cfg.TRAIN_STEPS)
    p.add_argument("--micro-batch-size", type=int, default=cfg.TRAIN_MICRO_BATCH)
    p.add_argument("--grad-accum", type=int, default=cfg.TRAIN_GRAD_ACCUM)
    p.add_argument("--eval-batch-size", type=int, default=cfg.TRAIN_EVAL_BATCH)
    p.add_argument("--eval-interval", type=int, default=cfg.TRAIN_EVAL_INTERVAL)
    p.add_argument("--max-val-samples-per-page", type=int, default=cfg.TRAIN_MAX_VAL_SAMPLES_PER_PAGE)
    p.add_argument("--page-cache-size", type=int, default=cfg.TRAIN_PAGE_CACHE_SIZE)
    p.add_argument("--learning-rate", type=float, default=cfg.TRAIN_LR)
    p.add_argument("--weight-decay", type=float, default=cfg.TRAIN_WEIGHT_DECAY)
    p.add_argument("--warmup-steps", type=int, default=cfg.TRAIN_WARMUP)
    p.add_argument("--position-decay", type=float, default=cfg.TRAIN_POSITION_DECAY)
    p.add_argument("--tail-boost", type=float, default=cfg.TRAIN_TAIL_BOOST)
    p.add_argument("--hidden-weight", type=float, default=cfg.TRAIN_HIDDEN_WEIGHT)
    p.add_argument("--ce-weight", type=float, default=cfg.TRAIN_CE_WEIGHT)
    p.add_argument("--tv-weight", type=float, default=cfg.TRAIN_TV_WEIGHT)
    p.add_argument("--acceptance-weight", type=float, default=cfg.TRAIN_ACCEPTANCE_WEIGHT)
    p.add_argument(
        "--prefix-survival-weight",
        type=float,
        default=cfg.TRAIN_PREFIX_SURVIVAL_WEIGHT,
    )
    p.add_argument(
        "--exp-position-gamma",
        type=float,
        default=cfg.TRAIN_EXP_POSITION_GAMMA,
        help="DSpark exp(-(k)/gamma) position weights; 0 disables",
    )
    p.add_argument(
        "--conf-label-from-tv",
        action=argparse.BooleanOptionalAction,
        default=cfg.TRAIN_CONF_LABEL_FROM_TV,
    )
    p.add_argument(
        "--use-markov",
        action=argparse.BooleanOptionalAction,
        default=cfg.TRAIN_USE_MARKOV,
    )
    p.add_argument(
        "--anchors-per-page",
        type=int,
        default=cfg.TRAIN_ANCHORS_PER_PAGE_ATTEMPT,
        help="Multi-anchor samples per page load (P2 denser training)",
    )
    p.add_argument(
        "--pos0-boost",
        type=float,
        default=1.0,
        help="Extra loss weight on first draft position (Stage-2 pos0 push)",
    )
    p.add_argument("--use-auf", action=argparse.BooleanOptionalAction, default=cfg.TRAIN_USE_AUF)
    p.add_argument("--val-ratio", type=float, default=cfg.TRAIN_VAL_RATIO)
    p.add_argument("--seed", type=int, default=cfg.TRAIN_SEED)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate teachers/checkpoint paths only; no GPU training",
    )
    p.add_argument(
        "--cuda-memory-fraction",
        type=float,
        default=cfg.CUDA_MEMORY_FRACTION,
        help="Cap process VRAM fraction (leave headroom for desktop)",
    )
    return p.parse_args()


def _build_stores(args: argparse.Namespace) -> Tuple[LazyTeacherStore, LazyTeacherStore, List[Path], List[Path]]:
    teacher_root = Path(args.teacher).expanduser().resolve()
    val_root = Path(args.val_teacher).expanduser().resolve()

    if val_root.is_dir() and any(val_root.glob("*.pt")) and teacher_root.resolve() != val_root.resolve():
        train_paths = discover_teacher_files(teacher_root)
        val_paths = discover_teacher_files(val_root)
    else:
        all_paths = discover_teacher_files(teacher_root)
        train_paths, val_paths = load_split_manifest(
            all_paths, args.split_manifest, args.val_ratio, args.seed
        )

    train_index = build_index(train_paths)
    val_index = build_index(val_paths)
    train_store = LazyTeacherStore(train_index, cache_size=args.page_cache_size)
    val_store = LazyTeacherStore(val_index, cache_size=max(4, args.page_cache_size // 2))
    return train_store, val_store, train_paths, val_paths


def main() -> int:
    args = parse_args()
    if not args.dry_run and not torch.cuda.is_available():
        print("ERROR: CUDA is required", file=sys.stderr)
        return 2

    if (
        not args.dry_run
        and torch.cuda.is_available()
        and 0.1 < float(args.cuda_memory_fraction) < 1.0
    ):
        torch.cuda.set_per_process_memory_fraction(float(args.cuda_memory_fraction))
        print(
            f"CUDA memory fraction capped at {args.cuda_memory_fraction:.2f}",
            flush=True,
        )

    seed_everything(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    block_sizes = parse_ints(args.block_sizes)
    block_probs = parse_floats(args.block_probs)
    if len(block_sizes) != len(block_probs):
        raise ValueError("block-sizes and block-probs length mismatch")
    if args.micro_batch_size < 1 or args.grad_accum < 1:
        raise ValueError("micro-batch-size and grad-accum must be >= 1")

    resume_path = Path(args.resume_checkpoint).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()

    print("=" * 78)
    print("Mini UFlash V2 — Windows 8GB Stage 11B continuation")
    print(f"Resume     : {resume_path}")
    print(f"Model path : {model_path}")
    print(f"Teacher    : {args.teacher}")
    print(f"Output     : {output_dir}")
    print(
        f"Micro-batch: {args.micro_batch_size} x accum {args.grad_accum} "
        f"= effective {args.micro_batch_size * args.grad_accum}"
    )
    print(f"Steps      : {args.steps}  eval every {args.eval_interval}")
    print(f"Dry-run    : {args.dry_run}")
    print("=" * 78)

    if not resume_path.is_file():
        print(f"ERROR: resume checkpoint missing: {resume_path}", file=sys.stderr)
        return 1
    if not model_path.is_dir():
        print(f"ERROR: Unlimited-OCR path missing: {model_path}", file=sys.stderr)
        return 1

    try:
        train_store, val_store, train_paths, val_paths = _build_stores(args)
    except FileNotFoundError as exc:
        print(f"No teachers yet — {exc}")
        print(
            "Place page images under train/data/pages/pool then run extract_teachers.py"
        )
        if args.dry_run:
            print("Dry-run: scaffold OK; waiting for teacher data. No training started.")
            return 0
        return 1

    feature_count, hidden_size, layers = train_store.validate_shapes()
    val_store.validate_shapes()
    print(f"Train/val pages: {len(train_store)}/{len(val_store)}")
    print(f"Features F={feature_count} H={hidden_size} layers={layers}")

    if args.dry_run:
        print("Dry-run OK: teachers + checkpoint paths look valid. No training started.")
        atomic_write_json(
            output_dir / "dry_run_report.json",
            {
                "resume_checkpoint": str(resume_path),
                "model_path": str(model_path),
                "train_pages": len(train_paths),
                "val_pages": len(val_paths),
                "feature_count": feature_count,
                "hidden_size": hidden_size,
                "layer_indices": list(layers),
                "effective_batch": args.micro_batch_size * args.grad_accum,
            },
        )
        return 0

    model, resume_meta = load_checkpoint(resume_path, device="cpu")
    if model.config.num_target_features != feature_count:
        raise ValueError("Checkpoint and teacher feature counts differ")
    if model.config.target_hidden_size != hidden_size:
        raise ValueError("Checkpoint and teacher hidden sizes differ")
    if max(block_sizes) > model.config.max_block_size:
        raise ValueError("Block size exceeds checkpoint architecture")
    model.config.use_markov = bool(args.use_markov)
    missing = resume_meta.get("load_missing_keys") or []
    if missing:
        print(f"Loaded with new/untrained keys ({len(missing)}): e.g. {missing[:4]}", flush=True)
    model = model.to(device)

    input_embeddings, lm_head = load_frozen_target_heads(str(model_path), torch.bfloat16)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return max(1e-3, (step + 1) / max(1, args.warmup_steps))
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    baseline_metrics = evaluate_all(
        model,
        input_embeddings,
        lm_head,
        val_store,
        block_sizes=block_sizes,
        batch_size=args.eval_batch_size,
        max_samples_per_page=args.max_val_samples_per_page,
        device=device,
    )
    primary_block = max(block_sizes)
    baseline_score = model_score(baseline_metrics, primary_block)
    best_score = baseline_score
    best_metrics = baseline_metrics
    base_step = int(resume_meta.get("step", 0))
    atomic_write_json(output_dir / "baseline_metrics.json", baseline_metrics)
    print("Baseline:", format_metrics(baseline_metrics, primary_block), flush=True)

    history: List[Dict[str, object]] = []
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for local_step in range(1, args.steps + 1):
        model.train()
        block = weighted_choice(block_sizes, block_probs, rng)
        micro_losses = []
        for _accum in range(args.grad_accum):
            batch = sample_batch(
                train_store,
                block,
                args.micro_batch_size,
                rng,
                anchors_per_page=args.anchors_per_page,
            )
            features = batch["target_features"].to(device)
            anchor_ids = batch["anchor_ids"].to(device)
            targets = batch["targets"].to(device)
            target_hidden = batch["target_hidden"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                anchor_emb = input_embeddings(anchor_ids)
                output = model(
                    features,
                    block_size=block,
                    anchor_embedding=anchor_emb,
                )
                hidden = output["hidden"]
                # Teacher-forced Markov refine (DSpark semi-AR) — 8GB-safe hidden residual.
                if args.use_markov and hasattr(model, "refine_with_markov"):
                    # prev gold tokens for positions 1..L-1; build [B, L] pad for API
                    hidden = model.refine_with_markov(
                        hidden,
                        prev_token_ids=targets,
                        input_embeddings=input_embeddings,
                        anchor_embedding=anchor_emb,
                        teacher_force=True,
                    )
                    # refine_with_markov uses targets[:, k-1] via prev_token_ids path
                    # Fix: our refine uses prev_token_ids[:, k-1] when provided as full targets
                    # but the method expects prev_token_ids for each prev — using targets works
                    # if we pass targets as the gold sequence (index k-1 is correct).
                logits = lm_head(hidden)
                # Target distribution for TV (no grad through frozen head path beyond hidden).
                with torch.no_grad():
                    target_logits = lm_head(target_hidden)
            losses = compute_training_loss(
                logits=logits,
                targets=targets,
                predicted_hidden=hidden,
                target_hidden=target_hidden,
                acceptance_logits=output["acceptance_logits"],
                position_decay=args.position_decay,
                tail_boost=args.tail_boost,
                hidden_weight=args.hidden_weight,
                acceptance_weight=args.acceptance_weight,
                prefix_survival_weight=args.prefix_survival_weight,
                use_auf=args.use_auf,
                post_acceptance_logits=output["post_acceptance_logits"],
                target_logits=target_logits,
                tv_weight=float(args.tv_weight),
                ce_weight=float(args.ce_weight),
                exp_position_gamma=(
                    float(args.exp_position_gamma)
                    if float(args.exp_position_gamma) > 0
                    else None
                ),
                conf_label_from_tv=bool(args.conf_label_from_tv),
                pos0_boost=float(getattr(args, "pos0_boost", 0.0) or 0.0),
            )
            (losses["loss"] / args.grad_accum).backward()
            micro_losses.append(float(losses["loss"].detach().item()))

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        should_eval = (
            local_step == 1
            or local_step % args.eval_interval == 0
            or local_step == args.steps
        )
        if should_eval:
            metrics = evaluate_all(
                model,
                input_embeddings,
                lm_head,
                val_store,
                block_sizes=block_sizes,
                batch_size=args.eval_batch_size,
                max_samples_per_page=args.max_val_samples_per_page,
                device=device,
            )
            score = model_score(metrics, primary_block)
            global_step = base_step + local_step
            elapsed = time.perf_counter() - started
            mean_loss = sum(micro_losses) / max(1, len(micro_losses))
            line = (
                f"local={local_step:05d} global={global_step:05d} B{block} "
                f"loss={mean_loss:.4f} | {format_metrics(metrics, primary_block)} "
                f"score={score:.4f} elapsed={elapsed:.1f}s"
            )
            print(line, flush=True)
            history.append(
                {
                    "local_step": local_step,
                    "global_step": global_step,
                    "loss": mean_loss,
                    "score": score,
                    "metrics": metrics,
                }
            )
            save_checkpoint(
                output_dir / "drafter_last.pt",
                model,
                global_step,
                metrics,
                extra={
                    "resume_checkpoint": str(resume_path),
                    "local_step": local_step,
                    "block_sizes": list(block_sizes),
                    "block_probs": list(block_probs),
                    "micro_batch_size": args.micro_batch_size,
                    "grad_accum": args.grad_accum,
                    "prefix_survival_weight": args.prefix_survival_weight,
                    "platform": "windows_8g",
                },
            )
            if score > best_score:
                best_score = score
                best_metrics = metrics
                save_checkpoint(
                    output_dir / "drafter_best.pt",
                    model,
                    global_step,
                    metrics,
                    extra={
                        "resume_checkpoint": str(resume_path),
                        "local_step": local_step,
                        "baseline_score": baseline_score,
                        "best_score": best_score,
                        "platform": "windows_8g",
                    },
                )
                print(f"  * new best score={best_score:.4f}", flush=True)

    final_metrics = evaluate_all(
        model,
        input_embeddings,
        lm_head,
        val_store,
        block_sizes=block_sizes,
        batch_size=args.eval_batch_size,
        max_samples_per_page=args.max_val_samples_per_page,
        device=device,
    )
    final_score = model_score(final_metrics, primary_block)
    report = {
        "format": "mini_uflash_v2_windows8g_training_report",
        "resume_checkpoint": str(resume_path),
        "base_step": base_step,
        "local_steps": args.steps,
        "baseline_score": baseline_score,
        "best_score": best_score,
        "final_score": final_score,
        "baseline_metrics": baseline_metrics,
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
        "train_pages": len(train_paths),
        "validation_pages": len(val_paths),
        "block_sizes": list(block_sizes),
        "block_probs": list(block_probs),
        "micro_batch_size": args.micro_batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.micro_batch_size * args.grad_accum,
        "history": history,
        "platform": "windows_8g",
    }
    atomic_write_json(output_dir / "training_report.json", report)
    # Human-readable summary
    summary = (
        f"Mini UFlash Windows 8GB continue-train\n"
        f"Resume: {resume_path}\n"
        f"Train/val pages: {len(train_paths)}/{len(val_paths)}\n"
        f"Baseline score: {baseline_score:.4f}\n"
        f"Best score: {best_score:.4f}\n"
        f"Final score: {final_score:.4f}\n"
        f"Best B8: {format_metrics(best_metrics, primary_block)}\n"
        f"Checkpoints: {output_dir / 'drafter_best.pt'}\n"
    )
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print("=" * 78)
    print(summary)
    empty_cuda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
