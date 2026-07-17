#!/usr/bin/env python3
"""Fit position-wise conf temperatures for DSpark-style verify scheduling.

Runs short stable_dflash-like draft/verify loops on gold pages, records
(predicted conf, actual accept) per draft position, then grid-searches
temperature per position so mean calibrated conf ≈ empirical accept rate.

8GB-safe: uses incremental step/crop (same as engine), never full-prefill
every round. OOM on a page → skip and continue.

Usage (project root)::

    .\\.venv\\Scripts\\python.exe train\\calibrate_conf.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train.lib.utils import empty_cuda  # noqa: E402

# Keep calib light on 8GB VRAM.
_MAX_GEN = 96
_MAX_STEPS = 36
_BLOCK = 4
_PAGES_CAP = 6


def _logit(p: float, eps: float = 1e-4) -> float:
    p = min(1.0 - eps, max(eps, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def fit_temp(preds: List[float], labels: List[float]) -> float:
    """Temperature scaling via binary NLL (standard for conf calibration).

    T<1 sharpens (raises high conf); T>1 flattens toward 0.5.
    Prefer mean-bias secondary: if underconfident (mean_p < emp), prefer T<=1.
    """
    if not preds:
        return 1.0
    emp = sum(labels) / len(labels)
    mean_p = sum(preds) / len(preds)
    best_t, best_nll = 1.0, 1e18
    for t in [0.45, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8, 2.2, 2.8]:
        nll = 0.0
        for p, y in zip(preds, labels):
            pc = _sigmoid(_logit(p) / t)
            pc = min(1.0 - 1e-6, max(1e-6, pc))
            nll += -(y * math.log(pc) + (1.0 - y) * math.log(1.0 - pc))
        nll /= len(preds)
        # Soft prior: underconfident → slight penalty on T>1, and vice versa.
        if mean_p + 0.02 < emp and t > 1.0:
            nll += 0.02 * (t - 1.0)
        if mean_p > emp + 0.02 and t < 1.0:
            nll += 0.02 * (1.0 - t)
        if nll < best_nll:
            best_nll, best_t = nll, t
    return best_t


def main() -> int:
    from webapp import config
    from webapp.dflash_tiers import DEFAULT_CALIBRATION_PATH
    from webapp.mini_uflash_engine import load_drafter, unload_drafter
    from webapp.model_manager import MODEL_MANAGER
    from webapp.payload_capture import prepare_input_payload
    from webapp.mini_uflash_core.online_target import OnlineTarget
    from webapp.mini_uflash_core.stage9_common import (
        FeatureTap,
        clone_cache,
        cache_tensors_are_disjoint,
        generation_policy,
        payload_tensors,
        policy_argmax,
        synchronize,
    )

    man = json.loads((_ROOT / "train" / "bench_manifest.json").read_text(encoding="utf-8"))
    arch = _ROOT / "train" / "bench_manifest_v1_archive.json"
    if arch.is_file():
        man = json.loads(arch.read_text(encoding="utf-8"))
    pool = _ROOT / man.get("pool_dir", "train/data/pages/pool")
    names = list(man.get("warmup") or []) + list(man.get("scored_pages") or [])
    pages = [pool / n for n in names if (pool / n).is_file()][:_PAGES_CAP]
    if not pages:
        print("No calibration pages", file=sys.stderr)
        return 2

    print("=" * 72)
    print("Conf calibration (DSpark-style temperature fit, 8GB-safe)")
    print(f"Pages: {len(pages)}  max_gen={_MAX_GEN}  max_steps={_MAX_STEPS}")
    print(f"Weight: {config.discover_weight()}")
    print("=" * 72)

    handles = MODEL_MANAGER.load()
    drafter, checkpoint = load_drafter(device=torch.device("cuda"))
    device = torch.device("cuda")
    layer_indices = checkpoint.get("extra", {}).get(
        "layer_indices", list(config.INFERENCE.layer_indices)
    )

    buckets: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(7)}
    rounds = 0
    pages_ok = 0

    for page in pages:
        print(f"-- {page.name}", flush=True)
        empty_cuda()
        prepared = None
        tap = None
        try:
            prepared = prepare_input_payload(
                handles.model,
                handles.tokenizer,
                page,
                preset_name="gundam",
                max_length=1536,  # lighter than 2048 for calib
            )
            tensors = payload_tensors(prepared.payload, device)
            policy = generation_policy(prepared.payload)
            if int(policy.get("no_repeat_ngram_size", 0) or 0) <= 0:
                policy = {
                    "no_repeat_ngram_size": int(config.INFERENCE.no_repeat_ngram_size),
                    "ngram_window": int(config.INFERENCE.ngram_window),
                }
            prompt_ids = [int(x) for x in tensors["prompt_ids"][0].tolist()]
            max_gen = min(_MAX_GEN, max(1, 1536 - len(prompt_ids)))
            eos_id = getattr(handles.tokenizer, "eos_token_id", None) or 1
            eos_id = int(eos_id)

            tap = FeatureTap(handles.model, layer_indices)
            online = OnlineTarget(handles.model, tap, tensors, device)
            generated: List[int] = []
            state = online.prefill([])
            steps = 0
            page_rounds = 0

            while len(generated) < max_gen and steps < _MAX_STEPS:
                steps += 1
                prefix = [*prompt_ids, *generated]
                anchor_id = policy_argmax(state["next_logits"], prefix, policy)
                if anchor_id == eos_id:
                    generated.append(anchor_id)
                    break

                draft_len = _BLOCK - 1
                anchor_t = torch.tensor([anchor_id], dtype=torch.long, device=device)
                anchor_emb = handles.input_embeddings(anchor_t)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    draft_out = drafter.draft(
                        state["features"],
                        handles.output_embeddings,
                        block_size=_BLOCK,
                        anchor_embedding=anchor_emb,
                        input_embeddings=handles.input_embeddings,
                    )
                draft_ids = [int(x) for x in draft_out["tokens"][0].tolist()][:draft_len]
                conf = [
                    float(x)
                    for x in draft_out["correctness_probability"][0][:draft_len]
                    .detach()
                    .cpu()
                    .tolist()
                ]

                try:
                    block_cache = clone_cache(state["cache"])
                    if not cache_tensors_are_disjoint(state["cache"], block_cache):
                        raise RuntimeError("cache alias")
                    block_result = online.verify_block(
                        anchor_id,
                        draft_ids,
                        block_cache,
                        state["absolute_length"],
                        prefix,
                        policy,
                    )
                except RuntimeError as exc:
                    msg = str(exc).lower()
                    if "out of memory" in msg or "cuda" in msg:
                        print(f"   OOM on verify step={steps}; stop page", flush=True)
                        empty_cuda()
                        break
                    raise

                preds = [int(x) for x in block_result["predictions"]]
                accepted = 0
                for cand, tgt in zip(draft_ids, preds):
                    if cand != tgt:
                        break
                    accepted += 1

                # Record conditional accept labels (positions evaluated in order).
                for k, c in enumerate(conf):
                    if k < accepted:
                        buckets[k].append((c, 1.0))
                    elif k == accepted:
                        buckets[k].append((c, 0.0))
                        break
                    else:
                        break

                rounds += 1
                page_rounds += 1

                # Incremental commit (same as engine) — no full prefill each step.
                if accepted == 0:
                    try:
                        del block_cache
                    except Exception:
                        pass
                    generated.append(anchor_id)
                    state = online.step(
                        anchor_id, state["cache"], state["absolute_length"]
                    )
                else:
                    appended = [anchor_id, *draft_ids[:accepted]]
                    generated.extend(appended)
                    if eos_id in appended:
                        break
                    committed = block_result["cache"]
                    physical_target = (
                        int(block_result["physical_cache_length_before"]) + len(appended)
                    )
                    crop = getattr(committed, "crop", None)
                    if not callable(crop):
                        # Rare fallback: full rebuild once.
                        del committed
                        empty_cuda()
                        state = online.prefill(generated)
                    else:
                        crop(physical_target)
                        next_logits = block_result["logits"][:, accepted, :]
                        try:
                            features_next = online.tap.position_features(accepted)
                        except Exception:
                            features_next = block_result.get("features")
                        state = {
                            "cache": committed,
                            "next_logits": next_logits,
                            "features": features_next,
                            "absolute_length": int(state["absolute_length"])
                            + len(appended),
                            "seconds": 0.0,
                        }

                if steps % 12 == 0:
                    empty_cuda()

            pages_ok += 1
            print(f"   rounds={page_rounds} gen={len(generated)}", flush=True)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg or "cuda" in msg:
                print(f"   OOM on page; skip: {exc}", flush=True)
            else:
                print(f"   error: {exc}", flush=True)
                raise
        except Exception as exc:
            print(f"   error: {exc}", flush=True)
            raise
        finally:
            if tap is not None:
                try:
                    tap.close()
                except Exception:
                    pass
            if prepared is not None and getattr(prepared, "scratch_dir", None):
                if prepared.scratch_dir.exists():
                    import shutil

                    shutil.rmtree(prepared.scratch_dir, ignore_errors=True)
            empty_cuda()

    if rounds < 8:
        print(
            f"Too few rounds ({rounds}) for a reliable fit; abort",
            file=sys.stderr,
        )
        try:
            unload_drafter()
        except Exception:
            pass
        MODEL_MANAGER.unload()
        empty_cuda()
        return 1

    temps: List[float] = []
    stats = []
    for k in range(7):
        pairs = buckets[k]
        if not pairs:
            temps.append(1.0)
            stats.append({"pos": k, "n": 0, "emp_accept": None, "temp": 1.0})
            continue
        preds = [p for p, _ in pairs]
        labels = [y for _, y in pairs]
        emp = sum(labels) / len(labels)
        t = fit_temp(preds, labels)
        temps.append(t)
        mean_raw = sum(preds) / len(preds)
        stats.append(
            {
                "pos": k,
                "n": len(pairs),
                "emp_accept": round(emp, 4),
                "mean_raw_conf": round(mean_raw, 4),
                "temp": t,
            }
        )
        print(
            f"pos{k}: n={len(pairs)} emp={emp:.3f} raw_conf={mean_raw:.3f} T={t:.2f}",
            flush=True,
        )

    pos0_emp = stats[0].get("emp_accept") or 0.5
    out = {
        "format": "mini_uflash_conf_calibration_v1",
        "fitted": True,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "weight": str(config.discover_weight()),
        "rounds": rounds,
        "pages_ok": pages_ok,
        "temps": temps,
        "conf_floor": 0.20,
        "survival_floor": 0.10,
        "pos0_skip_below": round(max(0.12, min(0.28, float(pos0_emp) * 0.35)), 3),
        "stats": stats,
    }
    DEFAULT_CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_CALIBRATION_PATH.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {DEFAULT_CALIBRATION_PATH}", flush=True)

    try:
        unload_drafter()
    except Exception:
        pass
    MODEL_MANAGER.unload()
    empty_cuda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
