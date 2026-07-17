"""Metrics formatting for the Mini UFlash OCR webapp.

Converts raw probe/engine metrics into human-readable strings for the UI's
"Mini UFlash 指标" tab. The spec requires a rich set of metrics; this module
maps the :class:`~webapp.mini_uflash_engine.ProbeResult` dict into structured
Markdown / JSON / plain text.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def format_metrics_report(
    result: Dict[str, Any], *, mode: str = "stable"
) -> str:
    """Return a Markdown-formatted metrics summary for the UI."""
    lines: list[str] = []
    lines.append("# 运行指标\n")

    if mode in ("mini_uflash_stable_dflash", "mini_uflash_direct"):
        lines.append(_format_dflash_metrics(result, mode=mode))
    elif mode == "mini_uflash_precise":
        lines.append(_format_probe_metrics(result))
    else:
        lines.append(_format_stable_metrics(result))

    return "\n".join(lines)


_TIER_LABELS = {
    "fast": "快速（墙钟优先，可软截断）",
    "balanced": "均衡（中等软帽）",
    "lossless": "无损（无 soft 截断）",
}


def _format_dflash_metrics(r: Dict[str, Any], *, mode: str = "mini_uflash_stable_dflash") -> str:
    pages = int(r.get("pages", 1))
    fallback_pages = r.get("fallback_pages", [])
    tier = str(r.get("tier") or "").strip().lower()
    tier_label = _TIER_LABELS.get(tier, tier or "默认/sys_v2")
    title = (
        "## 稳定 DFlash（验证前缀 + 周期重同步）\n"
        if "stable_dflash" in mode
        else "## 加速实验版（Direct Block）\n"
    )
    lines = [
        title,
        f"- 加速档位: **{tier_label}**" if tier or "stable_dflash" in mode else None,
        f"- 处理页数: {pages}",
        f"- 总耗时: {r.get('elapsed_seconds', r.get('total_seconds', 0)):.1f} 秒",
        f"- 生成 token: {r.get('generated_tokens', 0)}",
        f"- 推测轮数: {r.get('speculative_rounds', 0)}",
        f"- 验证前缀提交: {r.get('direct_block_commits', 0)} 次",
        f"- 整块提交: {r.get('full_block_commits', 0)} 次",
        f"- 提交 token: {r.get('direct_committed_tokens', 0)}",
        f"- Full Block 比率: {r.get('full_block_ratio', 0):.2%}",
        f"- 平均接受 draft: {r.get('mean_accepted_draft', 0):.3f}",
        f"- 周期重同步: {r.get('resync_count', 0)} 次",
        f"- 纯 B1 轮次: {r.get('pure_b1_rounds', 0)}",
        f"- 目标模型前向折算: {r.get('target_forward_reduction', 0):.2f} token/forward",
        f"- 回退普通模式页: {fallback_pages if fallback_pages else '无'}",
        "",
    ]
    lines = [x for x in lines if x is not None]
    if "stable_dflash" in mode:
        if tier == "lossless":
            lines.append(
                "✅ **无损档**：仅提交目标验证通过的前缀，无 soft 截断；"
                "墙钟收益取决于 live 接受率（当前金标准约 0.95×）。"
            )
        elif tier == "balanced":
            lines.append(
                "✅ **均衡档**：中等长度软帽；比快速更完整，墙钟约 1.5× 量级。"
            )
        else:
            lines.append(
                "✅ **快速档 / 稳定 DFlash**：验证前缀 + 周期重同步；"
                "软截断换墙钟（金标准约 1.7×）。"
            )
        warn = r.get("warning")
        if warn:
            lines.append(f"\n> {warn}")
    else:
        lines.append(
            "⚠️ **这是非无损实验输出，可能出现漏字、错字或缓存漂移，请抽查重要内容。**"
        )
    return "\n".join(lines)


def _format_stable_metrics(r: Dict[str, Any]) -> str:
    lines = [
        "## 稳定模式\n",
        f"- 耗时: {r.get('elapsed_seconds', 0):.1f} 秒",
        f"- 生成 token 数: {r.get('generated_tokens', 0)}",
        f"- 页数: {len(r.get('pages', []))}",
    ]
    return "\n".join(lines)


def _format_probe_metrics(r: Dict[str, Any]) -> str:
    lines = [
        "## Mini UFlash 精确模式\n",
        "",
        "### 基本信息",
        f"- Checkpoint 路径: `{r.get('checkpoint_path', 'N/A')}`",
        f"- Checkpoint step: {r.get('checkpoint_step', -1)}",
        f"- 参数量: {r.get('parameter_count', 0):,}",
        f"- Target Layers: {r.get('layer_indices', [])}",
        f"- Block Size: B{r.get('block_size', 8)} (1 anchor + 7 draft)",
        f"- dtype: {r.get('dtype', 'N/A')}",
        "",
        "### 推理结果",
        f"- 生成的 token 总数: {r.get('generated_tokens', 0)}",
        f"- 推测轮数: {r.get('speculative_rounds', 0)}",
        f"- 平均接受 draft: {r.get('mean_accepted_draft', 0):.3f} / 7",
        f"- 每轮有效 token: {r.get('effective_tokens_per_round', 0):.3f}",
        f"- Full Block 次数: {r.get('full_block_count', 0)}",
        f"- Full Block 比率: {r.get('full_block_ratio', 0):.2%}",
        "",
        "### 接受前缀直方图",
    ]
    hist = r.get("acceptance_histogram", {})
    if hist:
        for k, v in sorted(hist.items()):
            lines.append(f"  - 接受 {k} 个 draft: {v} 次")
    else:
        lines.append("  （无数据）")

    lines.append("")
    lines.append("### 各位置准确率")
    pos_acc = r.get("per_position_accuracy", [])
    for i, acc in enumerate(pos_acc):
        lines.append(f"  - 位置 {i + 1}: {acc:.2%}")

    lines.append("")
    lines.append("### 延迟 (ms)")
    lines.append(f"- Drafter 平均: {r.get('drafter_latency_ms', 0):.2f}")
    lines.append(f"- Block Verifier 平均: {r.get('block_verifier_latency_ms', 0):.2f}")
    lines.append(f"- Strict Replay 平均: {r.get('strict_replay_latency_ms', 0):.2f}")

    lines.append("")
    lines.append("### 精确度")
    lines.append(f"- Final Token Exactness: {'✅ 通过' if r.get('final_token_exactness') else '❌ 失败'}")
    if r.get("first_mismatch") is not None:
        lines.append(f"- 首次不匹配位置: {r['first_mismatch']}")
    else:
        lines.append("- 首次不匹配位置: 无")

    lines.append(f"- Block vs B1 不一致率: {r.get('block_vs_b1_disagreement_rate', 0):.4%}")

    lines.append("")
    lines.append("### 理论速度")
    lines.append(f"- 理论有效 token/轮: {r.get('effective_tokens_per_round', 0):.3f}")
    lines.append(f"- 理论解码加速: {r.get('theoretical_speedup', 0):.2f}×")
    lines.append(f"- ⚠️ **{r.get('note', '')}**")

    lines.append("")
    lines.append("### 显存与耗时")
    lines.append(f"- Payload 捕获: {r.get('payload_capture_seconds', 0):.2f}s")
    lines.append(f"- 探测耗时: {r.get('probe_seconds', 0):.2f}s")
    lines.append(f"- 总耗时: {r.get('total_seconds', 0):.2f}s")
    lines.append(f"- 显存占用: {r.get('vram_gb', 0):.2f} GB")

    if r.get("failure"):
        lines.append("")
        lines.append("### ⚠️ 失败详情")
        lines.append(f"```json")
        import json
        lines.append(json.dumps(r["failure"], ensure_ascii=False, indent=2))
        lines.append("```")

    return "\n".join(lines)


def format_run_log(
    mode: str,
    elapsed: float,
    tokens: int,
    pages: int,
    gpu: str = "",
    attention: str = "",
    extra: str = "",
) -> str:
    """Return a plain-text run log."""
    lines = [
        f"Mini UFlash OCR — Run Log",
        f"=" * 40,
        f"Mode: {mode}",
        f"Elapsed: {elapsed:.1f}s",
        f"Tokens: {tokens}",
        f"Pages: {pages}",
    ]
    if gpu:
        lines.append(f"GPU: {gpu}")
    if attention:
        lines.append(f"Attention: {attention}")
    if extra:
        lines.append("")
        lines.append(extra)
    return "\n".join(lines)
