"""One-shot snapshot of stable vs Mini UFlash speed metrics from local runs."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    outs = ROOT / "webapp" / "outputs"
    rows = []
    for d in sorted(outs.iterdir()):
        m = d / "metrics.json"
        if not m.is_file():
            continue
        try:
            data = json.loads(m.read_text(encoding="utf-8"))
        except Exception:
            continue
        mode = data.get("mode")
        pages = data.get("pages")
        if pages is None and "page_metrics" in data:
            pages = len(data["page_metrics"])
        if isinstance(pages, list):
            pages = len(pages)
        elapsed = data.get("elapsed_seconds") or data.get("total_seconds")
        tokens = data.get("generated_tokens")
        tfr = data.get("target_forward_reduction")
        mean_acc = data.get("mean_accepted_draft")
        if mean_acc is None and data.get("page_metrics"):
            pms = [
                p
                for p in data["page_metrics"]
                if str(p.get("mode", "")).startswith("mini_uflash")
            ]
            if pms:
                mean_acc = sum(float(p.get("mean_accepted_draft") or 0) for p in pms) / len(
                    pms
                )
                tfr = sum(
                    float(p.get("target_forward_reduction") or 0) for p in pms
                ) / len(pms)
                if tokens is None:
                    tokens = sum(int(p.get("generated_tokens") or 0) for p in pms)
        pps = None
        tps = None
        if elapsed and pages:
            try:
                pps = float(elapsed) / float(pages)
            except Exception:
                pass
        if elapsed and tokens:
            try:
                tps = float(tokens) / float(elapsed)
            except Exception:
                pass
        rows.append((d.name, mode, pages, elapsed, tokens, mean_acc, tfr, pps, tps))

    print("=== Local webapp runs (metrics.json) ===")
    print(
        f"{'run':<22} {'mode':<24} {'pg':>3} {'sec':>7} {'s/pg':>6} {'tok/s':>6} {'acc':>6} {'fwd_x':>5}"
    )
    for name, mode, pages, elapsed, tokens, mean_acc, tfr, pps, tps in rows:
        print(
            f"{name:<22} {str(mode)[:24]:<24} {str(pages or '-'):>3} "
            f"{elapsed if elapsed is not None else '-':>7} "
            f"{(f'{pps:.1f}' if pps else '-'):>6} "
            f"{(f'{tps:.1f}' if tps else '-'):>6} "
            f"{(f'{mean_acc:.2f}' if mean_acc is not None else '-'):>6} "
            f"{(f'{tfr:.2f}' if tfr is not None else '-'):>5}"
        )

    rep = ROOT / "train" / "runs" / "stage11b_win_continue_r2" / "training_report.json"
    if rep.exists():
        data = json.loads(rep.read_text(encoding="utf-8"))
        b0 = data["baseline_metrics"]["8"]
        b1 = data["best_metrics"]["8"]
        print()
        print("=== Offline drafter acceptance (R2 val, 30 pages) ===")
        print(
            "R2 start mean_acc",
            round(b0["average_accepted_draft"], 3),
            "tokens/round",
            round(b0["average_effective_emitted"], 3),
            "full",
            f"{b0['full_draft_accuracy']*100:.2f}%",
        )
        print(
            "R2 best  mean_acc",
            round(b1["average_accepted_draft"], 3),
            "tokens/round",
            round(b1["average_effective_emitted"], 3),
            "full",
            f"{b1['full_draft_accuracy']*100:.2f}%",
        )
        # Component-level rough ratio if each round ≈ one target block verify
        # and baseline AR spends 1 forward per token:
        print()
        print("Rough decode-only theoretical ratio (tokens per target-forward):")
        print("  AR baseline: ~1.0 token / target step")
        print(
            "  R2 best draft path: ~{:.2f} tokens / block-verify round".format(
                1.0 + b1["average_accepted_draft"]
            )
        )
        print(
            "  => upper-bound decode speedup ~{:.2f}x if verify cost == one B1 "
            "(usually optimistic on Windows eager)".format(
                1.0 + b1["average_accepted_draft"]
            )
        )
        print(
            "  Real wall-clock usually lower: prefill/image crop + verifier overhead "
            "+ non-lossless Direct path quality issues."
        )

    print()
    print("=== Comparable image smoke (same bilingual page if present) ===")
    for d in sorted((ROOT / "webapp" / "outputs").iterdir()):
        m = d / "metrics.json"
        if not m.is_file():
            continue
        data = json.loads(m.read_text(encoding="utf-8"))
        # single-page-ish runs
        pages = data.get("pages", 1)
        if isinstance(pages, list):
            pages = len(pages)
        if pages not in (1, None) and pages != 1:
            continue
        mode = data.get("mode")
        elapsed = data.get("elapsed_seconds") or data.get("total_seconds")
        tokens = data.get("generated_tokens")
        if elapsed:
            print(f"  {d.name}: mode={mode} sec={elapsed} tokens={tokens}")


if __name__ == "__main__":
    main()
