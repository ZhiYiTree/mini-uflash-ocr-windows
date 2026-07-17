#!/usr/bin/env python3
"""Head-to-head wall-clock: Unlimited-OCR stable vs Mini UFlash Direct.

Uses the currently discovered Stage-11B / domain-continue weight.
Writes JSON + Markdown under train/runs/bench_stable_vs_direct_*.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train.lib.utils import atomic_write_json, empty_cuda, list_images  # noqa: E402


def pick_pages(pool: Path, limit: int) -> list[Path]:
    images = list_images(pool, recursive=False)
    if not images:
        raise FileNotFoundError(f"No images in {pool}")
    # Prefer diversity: first writing-course pages + mao pages + smoke-like short names.
    mao = [p for p in images if p.name.startswith("mao")]
    course = [p for p in images if "153718" in p.name]
    other = [p for p in images if p not in mao and p not in course]
    selected: list[Path] = []
    # Spread course pages
    if course:
        step = max(1, len(course) // max(1, limit // 2))
        selected.extend(course[::step][: max(1, limit // 2 + 1)])
    if mao:
        step = max(1, len(mao) // max(1, limit // 3))
        selected.extend(mao[::step][: max(1, limit // 3 + 1)])
    for p in other:
        if len(selected) >= limit:
            break
        if p not in selected:
            selected.append(p)
    # fill
    for p in images:
        if len(selected) >= limit:
            break
        if p not in selected:
            selected.append(p)
    return selected[:limit]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, default=str(_ROOT / "train" / "data" / "pages" / "pool"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--preset", type=str, default="gundam")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=1, help="Warmup pages (not scored)")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(_ROOT / "train" / "bench_manifest.json"),
        help="Gold bench manifest (fixed pages + scheduler). Empty string disables.",
    )
    parser.add_argument(
        "--accel",
        choices=("stable_dflash", "direct"),
        default="stable_dflash",
        help="Accelerated decoder to compare against official stable",
    )
    args = parser.parse_args()

    from webapp import config
    from webapp.model_manager import MODEL_MANAGER
    from webapp.mini_uflash_engine import (
        load_drafter,
        run_direct_mode,
        run_stable_dflash_mode,
        unload_drafter,
    )
    from webapp.unlimited_ocr_engine import recognize_image

    manifest: dict[str, Any] | None = None
    dflash_cfg: dict[str, Any] = {
        "tier": None,
        "block_size": 4,
        "max_block_size": 8,
        "length_hard_cap": 512,
        "length_soft_cap": 0,
        "resync_every": 192,
        "low_tau_b1_threshold": 1.0,
        "low_tau_window": 8,
        "low_tau_b1_steps": 6,
        "zero_accept_page_b1_after": 12,
        "skip_verify_expected_below": 0.0,
        "use_conf_schedule": True,
        "page_degrade_enabled": True,
    }
    pool = Path(args.image_dir)
    n_warmup = max(0, int(args.warmup))

    if args.manifest:
        man_path = Path(args.manifest).expanduser()
        if not man_path.is_file():
            man_path = _ROOT / args.manifest
        if man_path.is_file():
            manifest = json.loads(man_path.read_text(encoding="utf-8"))
            pool = _ROOT / manifest.get("pool_dir", "train/data/pages/pool")
            if not pool.is_dir():
                pool = Path(args.image_dir)
            warm_names = list(manifest.get("warmup") or [])
            scored_names = list(manifest.get("scored_pages") or [])
            pages = []
            for name in warm_names + scored_names:
                p = pool / name
                if not p.is_file():
                    raise FileNotFoundError(f"Manifest page missing: {p}")
                pages.append(p)
            n_warmup = len(warm_names)
            args.limit = len(scored_names)
            if manifest.get("stable"):
                args.preset = str(manifest["stable"].get("preset", args.preset))
                args.max_length = int(manifest["stable"].get("max_length", args.max_length))
            if manifest.get("stable_dflash"):
                dflash_cfg.update(manifest["stable_dflash"])
            print(f"Gold manifest: {man_path}")
        else:
            print(f"WARNING: manifest not found ({args.manifest}), using pick_pages")
            pages = pick_pages(pool, args.limit + n_warmup)
    else:
        pages = pick_pages(pool, args.limit + n_warmup)

    weight = config.discover_weight()
    out_dir = (
        config.PROJECT_ROOT
        / "train"
        / "runs"
        / f"bench_stable_vs_direct_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "_scratch"
    scratch.mkdir(exist_ok=True)

    print("=" * 72)
    print(f"Stable Unlimited-OCR vs Mini UFlash ({args.accel})")
    print(f"Weight : {weight}")
    print(f"Model  : {config.unlimited_ocr_path()}")
    print(f"Pages  : {args.limit} (+{n_warmup} warmup)")
    print(f"Preset : {args.preset}  max_length={args.max_length}")
    print(f"Accel  : {args.accel}")
    print(f"DFlash : {dflash_cfg}")
    print(f"Output : {out_dir}")
    print("=" * 72)

    t0 = time.perf_counter()
    handles = MODEL_MANAGER.load()
    load_s = time.perf_counter() - t0
    print(f"Target loaded in {load_s:.1f}s backend={handles.attention_backend}")

    t1 = time.perf_counter()
    load_drafter(device=__import__("torch").device("cuda"))
    draft_s = time.perf_counter() - t1
    print(f"Drafter loaded in {draft_s:.1f}s")

    results: list[dict[str, Any]] = []
    scored = 0
    for i, image in enumerate(pages):
        is_warmup = i < n_warmup
        tag = "warmup" if is_warmup else f"page{scored+1}"
        print("-" * 72)
        print(f"[{tag}] {image.name}")

        # Stable
        empty_cuda()
        st = recognize_image(
            handles,
            image,
            scratch / f"{tag}_stable",
            preset_name=args.preset,
            max_length=args.max_length,
            page_index=i,
        )
        stable_row = {
            "mode": "stable",
            "seconds": round(st.elapsed_seconds, 3),
            "tokens": st.generated_tokens,
            "error": st.error,
            "chars": len(st.markdown or ""),
            "tok_per_s": (
                round(st.generated_tokens / st.elapsed_seconds, 2)
                if st.elapsed_seconds > 0 and st.generated_tokens
                else None
            ),
        }
        print(
            f"  stable: {stable_row['seconds']:.2f}s  tokens={stable_row['tokens']}  "
            f"tok/s={stable_row['tok_per_s']}  err={stable_row['error']}"
        )

        # Accelerated path (stable DFlash by default)
        empty_cuda()
        try:
            if args.accel == "direct":
                dr = run_direct_mode(
                    handles,
                    image,
                    preset_name=args.preset,
                    max_length=args.max_length,
                )
            else:
                tier_name = dflash_cfg.get("tier")
                if tier_name:
                    dr = run_stable_dflash_mode(
                        handles,
                        image,
                        preset_name=args.preset,
                        max_length=int(args.max_length),
                        tier=str(tier_name),
                    )
                else:
                    dr = run_stable_dflash_mode(
                        handles,
                        image,
                        preset_name=args.preset,
                        max_length=int(args.max_length),
                        block_size=int(dflash_cfg.get("block_size", 4)),
                        max_block_size=int(dflash_cfg.get("max_block_size", 8)),
                        length_hard_cap=int(dflash_cfg.get("length_hard_cap", 512)),
                        length_soft_cap=int(dflash_cfg.get("length_soft_cap", 0) or 0),
                        resync_every=int(dflash_cfg.get("resync_every", 192)),
                        low_tau_b1_threshold=float(
                            dflash_cfg.get("low_tau_b1_threshold", 1.0)
                        ),
                        low_tau_window=int(dflash_cfg.get("low_tau_window", 8)),
                        low_tau_b1_steps=int(dflash_cfg.get("low_tau_b1_steps", 6)),
                        zero_accept_page_b1_after=int(
                            dflash_cfg.get("zero_accept_page_b1_after", 12)
                        ),
                        skip_verify_expected_below=float(
                            dflash_cfg.get("skip_verify_expected_below", 0.0) or 0.0
                        ),
                        use_conf_schedule=bool(
                            dflash_cfg.get("use_conf_schedule", True)
                        ),
                        page_degrade_enabled=bool(
                            dflash_cfg.get("page_degrade_enabled", True)
                        ),
                    )
            length_ratio = None
            if stable_row.get("tokens") and dr.generated_tokens:
                length_ratio = round(
                    float(dr.generated_tokens) / max(1, int(stable_row["tokens"])), 3
                )
            direct_row = {
                "mode": args.accel,
                "seconds": round(dr.total_seconds, 3),
                "prep_seconds": round(dr.preprocessing_seconds, 3),
                "tokens": dr.generated_tokens,
                "mean_accepted_draft": round(dr.mean_accepted_draft, 4),
                "full_block_ratio": round(dr.full_block_ratio, 4),
                "target_forward_reduction": round(dr.target_forward_reduction, 3),
                "target_decode_forwards": dr.target_decode_forwards,
                "drafter_latency_ms": round(dr.drafter_latency_ms, 2),
                "block_verifier_latency_ms": round(dr.block_verifier_latency_ms, 2),
                "resync_count": getattr(dr, "resync_count", None),
                "total_draft_ms": round(getattr(dr, "total_draft_ms", 0.0), 1),
                "total_verify_ms": round(getattr(dr, "total_verify_ms", 0.0), 1),
                "total_resync_ms": round(getattr(dr, "total_resync_ms", 0.0), 1),
                "total_b1_ms": round(getattr(dr, "total_b1_ms", 0.0), 1),
                "cost_share_draft": round(getattr(dr, "cost_share_draft", 0.0), 3),
                "cost_share_verify": round(getattr(dr, "cost_share_verify", 0.0), 3),
                "cost_share_resync": round(getattr(dr, "cost_share_resync", 0.0), 3),
                "cost_share_b1": round(getattr(dr, "cost_share_b1", 0.0), 3),
                "page_degraded_to_b1": getattr(dr, "page_degraded_to_b1", False),
                "length_ratio_direct_over_stable": length_ratio,
                "chars": len(dr.markdown or ""),
                "error": None,
                "tok_per_s": (
                    round(dr.generated_tokens / dr.total_seconds, 2)
                    if dr.total_seconds > 0 and dr.generated_tokens
                    else None
                ),
            }
        except Exception as exc:  # noqa: BLE001
            direct_row = {
                "mode": args.accel,
                "seconds": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        cost = ""
        if direct_row.get("cost_share_verify") is not None and direct_row.get("error") is None:
            cost = (
                f"  cost[d/v/r/b1]="
                f"{direct_row.get('cost_share_draft'):.0%}/"
                f"{direct_row.get('cost_share_verify'):.0%}/"
                f"{direct_row.get('cost_share_resync'):.0%}/"
                f"{direct_row.get('cost_share_b1'):.0%}"
            )
        print(
            f"  {args.accel}: {direct_row.get('seconds')}s  tokens={direct_row.get('tokens')}  "
            f"acc={direct_row.get('mean_accepted_draft')}  "
            f"fwd_x={direct_row.get('target_forward_reduction')}  "
            f"len_x={direct_row.get('length_ratio_direct_over_stable')}  "
            f"resync={direct_row.get('resync_count')}  "
            f"degrade={direct_row.get('page_degraded_to_b1')}  "
            f"err={direct_row.get('error')}{cost}"
        )

        speedup = None
        if (
            not is_warmup
            and stable_row.get("seconds")
            and direct_row.get("seconds")
            and not stable_row.get("error")
            and not direct_row.get("error")
        ):
            speedup = round(float(stable_row["seconds"]) / float(direct_row["seconds"]), 3)

        row = {
            "tag": tag,
            "image": str(image),
            "name": image.name,
            "warmup": is_warmup,
            "stable": stable_row,
            "direct": direct_row,
            "wall_speedup_stable_over_direct": speedup,
        }
        # Save short text samples for quality glance
        if not is_warmup:
            (out_dir / f"{tag}_stable.md").write_text(st.markdown or "", encoding="utf-8")
            if direct_row.get("error") is None:
                # re-get markdown only stored in dr when success — write from last success
                try:
                    (out_dir / f"{tag}_direct.md").write_text(dr.markdown or "", encoding="utf-8")
                except Exception:
                    pass
            scored += 1
            results.append(row)
        else:
            # still record warmup separately
            pass

        atomic_write_json(out_dir / "partial_results.json", {"pages": results})

    scored_rows = results
    stable_secs = [r["stable"]["seconds"] for r in scored_rows if r["stable"].get("seconds") and not r["stable"].get("error")]
    direct_secs = [r["direct"]["seconds"] for r in scored_rows if r["direct"].get("seconds") and not r["direct"].get("error")]
    speedups = [r["wall_speedup_stable_over_direct"] for r in scored_rows if r.get("wall_speedup_stable_over_direct")]
    accs = [
        r["direct"]["mean_accepted_draft"]
        for r in scored_rows
        if r["direct"].get("mean_accepted_draft") is not None
    ]
    fwds = [
        r["direct"]["target_forward_reduction"]
        for r in scored_rows
        if r["direct"].get("target_forward_reduction") is not None
    ]

    def avg(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 3) if xs else None

    len_ratios = [
        r["direct"]["length_ratio_direct_over_stable"]
        for r in scored_rows
        if r["direct"].get("length_ratio_direct_over_stable") is not None
    ]
    cost_verify = [
        r["direct"]["cost_share_verify"]
        for r in scored_rows
        if r["direct"].get("cost_share_verify") is not None
    ]
    worst = min(speedups) if speedups else None
    summary = {
        "format": "mini_uflash_stable_vs_direct_bench",
        "weight": str(weight),
        "model": str(config.unlimited_ocr_path()),
        "preset": args.preset,
        "max_length": args.max_length,
        "manifest": str(args.manifest) if args.manifest else None,
        "dflash_cfg": dflash_cfg,
        "pages_scored": len(scored_rows),
        "mean_stable_seconds": avg(stable_secs),
        "mean_direct_seconds": avg(direct_secs),
        "mean_wall_speedup": avg(speedups),
        "median_wall_speedup": (
            round(sorted(speedups)[len(speedups) // 2], 3) if speedups else None
        ),
        "worst_wall_speedup": worst,
        "mean_accepted_draft": avg(accs),
        "mean_length_ratio": avg(len_ratios),
        "mean_cost_share_verify": avg(cost_verify),
        "mean_target_forward_reduction": avg(fwds),
        "target_load_seconds": round(load_s, 2),
        "drafter_load_seconds": round(draft_s, 2),
        "pages": scored_rows,
        "note": (
            "wall_speedup = stable_seconds / direct_seconds (>1 means Direct/accel faster). "
            "target_forward_reduction is decode-step estimate, not full end-to-end. "
            "cost_share_* is fraction of timed draft+verify+resync+b1 ms."
        ),
    }
    atomic_write_json(out_dir / "summary.json", summary)

    # Markdown table
    lines = [
        "# Stable vs Direct 对照测时",
        "",
        f"- 权重: `{weight}`",
        f"- 预设: `{args.preset}`  max_length={args.max_length}",
        f"- 评分页数: {len(scored_rows)}",
        "",
        f"| 页 | stable 秒 | accel 秒 | 墙钟比 | tok(s/d) | mean_acc | len比 | verify% |",
        f"| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in scored_rows:
        s, d = r["stable"], r["direct"]
        vshare = d.get("cost_share_verify")
        vshare_s = f"{100 * vshare:.0f}%" if isinstance(vshare, (int, float)) else ""
        lines.append(
            f"| {r['name'][:40]} | {s.get('seconds')} | {d.get('seconds')} | "
            f"{r.get('wall_speedup_stable_over_direct')} | "
            f"{s.get('tokens')}/{d.get('tokens')} | "
            f"{d.get('mean_accepted_draft')} | {d.get('length_ratio_direct_over_stable')} | "
            f"{vshare_s} |"
        )
    lines += [
        "",
        "## 汇总",
        "",
        f"- 平均 stable: **{summary['mean_stable_seconds']} s/页**",
        f"- 平均 accel: **{summary['mean_direct_seconds']} s/页**",
        f"- 平均墙钟比 (stable/accel): **{summary['mean_wall_speedup']}×** "
        f"(中位 {summary['median_wall_speedup']}×，最差 {summary['worst_wall_speedup']}×)",
        f"- 平均草稿接受: **{summary['mean_accepted_draft']}**",
        f"- 平均长度比 accel/stable: **{summary['mean_length_ratio']}**",
        f"- 平均 verify 成本占比: **{summary['mean_cost_share_verify']}**",
        f"- 平均 target forward 折算: **{summary['mean_target_forward_reduction']}×**",
        "",
        "> 墙钟比 >1 表示加速路径更快。成本占比为 draft+verify+resync+b1 计时之和的份额。",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("=" * 72)
    print("\n".join(lines))
    print(f"Wrote {out_dir / 'summary.md'}")

    unload_drafter()
    MODEL_MANAGER.unload()
    empty_cuda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
