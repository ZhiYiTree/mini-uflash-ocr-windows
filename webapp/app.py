"""Main entry point for the Mini UFlash OCR webapp.

Run with::

    D:\\Python\\OCR\\.venv\\Scripts\\python.exe -m webapp.app

Or via the PowerShell launcher::

    .\\launch_webapp.ps1

The Gradio server binds to ``127.0.0.1:7860``. Development hot-reload is
disabled so that the model is never loaded twice.
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import gradio as gr

from . import config
from . import path_utils
from . import process_manager
from .model_manager import MODEL_MANAGER
from .unlimited_ocr_engine import (
    recognize_image,
    assemble_result,
    StablePageResult,
    StableResult,
    clean_markdown,
)
from .mini_uflash_engine import (
    run_precise_mode,
    run_direct_mode,
    run_stable_dflash_mode,
    ProbeResult,
)
from .pdf_utils import page_count, render_pages, iter_render_pages, cleanup_dir, is_pdf
from .export_utils import (
    create_run_dir, save_input, save_result, save_log, build_result_json,
)
from .metrics import format_metrics_report, format_run_log
from .ui import build_ui, render_status_html
from .vram_utils import (
    clamp_max_length,
    empty_cuda,
    is_cuda_oom,
    mem_info_gb,
    oom_user_message,
)

_log = logging.getLogger(__name__)
_PAUSE_EVENT = threading.Event()
_CANCEL_EVENT = threading.Event()


class JobCancelled(Exception):
    """Raised when the user clicks 停止 during an OCR job."""


def _setup_logging() -> None:
    log_dir = config.LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_dir / "webapp.log"), encoding="utf-8"),
        ],
    )


def _refresh_env_report() -> config.EnvReport:
    report = config.build_static_env_report()
    if MODEL_MANAGER.is_loaded:
        handles = MODEL_MANAGER.handles()
        if handles is not None:
            report.unlimited_ocr = "已加载"
            report.attention_backend = handles.attention_backend
    else:
        report.attention_backend = "unloaded"
    from .mini_uflash_engine import is_drafter_loaded
    if is_drafter_loaded():
        report.mini_uflash = "Stage 11B 已加载"
    return report


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def on_file_change(file_path):
    """Handle file upload: show preview image and detect type."""
    if file_path is None:
        return gr.update(value=None, visible=False), "选择文件后，点击“识别文档”即可。"
    try:
        p = Path(file_path)
        if is_pdf(p):
            return gr.update(value=None, visible=False), f"已选择 {p.name}，共 {page_count(p)} 页。"
        return gr.update(value=file_path, visible=True), f"已选择 {p.name}。"
    except Exception as e:
        return gr.update(value=None, visible=False), f"无法读取这个文件：{e}"


def on_mode_change(mode):
    """Toggle accel-only controls (tier dropdown)."""
    from .ui import _toggle_mode_controls
    return _toggle_mode_controls(mode)


def on_load_model(model_path_str, _status, progress=gr.Progress()):
    """Load the Unlimited-OCR model."""
    if progress is not None:
        progress(0, desc="正在验证本地模型")
    yield "⏳ 正在加载 Unlimited-OCR…", render_status_html(_refresh_env_report()), gr.update(interactive=False)
    try:
        if progress is not None:
            progress(0.15, desc="正在读取 tokenizer 与权重")
        handles = MODEL_MANAGER.load(model_path_str)
        if progress is not None:
            progress(1, desc="模型加载完成")
        report = _refresh_env_report()
        status = (
            f"✅ 模型已加载\n"
            f"- 路径: {handles.model_path}\n"
            f"- Attention: {handles.attention_backend}\n"
            f"- dtype: {handles.dtype_name}\n"
            f"- VRAM: {handles.vram_gb:.1f} GB"
        )
        yield status, render_status_html(report), gr.update(interactive=True)
    except Exception as e:
        _log.exception("Model load failed")
        yield f"❌ 加载失败：{e}", render_status_html(_refresh_env_report()), gr.update(interactive=True)


def on_unload_model():
    """Unload the model and free VRAM."""
    MODEL_MANAGER.unload()
    empty_cuda()
    free_gb, total_gb = mem_info_gb()
    report = _refresh_env_report()
    msg = (
        f"模型已卸载并释放缓存。显存约空闲 {free_gb:.1f}/{total_gb:.1f} GB"
        if total_gb > 0
        else "模型已卸载"
    )
    return msg, render_status_html(report)


def on_toggle_pause():
    """Pause or resume the active OCR loop without discarding completed pages."""
    from .ui import _progress_markup

    if _CANCEL_EVENT.is_set():
        return gr.update(value="暂停", interactive=False), _progress_markup(False)
    if _PAUSE_EVENT.is_set():
        _PAUSE_EVENT.clear()
        return gr.update(value="暂停"), _progress_markup(True)
    _PAUSE_EVENT.set()
    return gr.update(value="继续"), _progress_markup(True, paused=True)


def on_stop_job():
    """Request cooperative cancel; loops check _CANCEL_EVENT between pages/steps."""
    from .ui import _progress_markup

    _CANCEL_EVENT.set()
    _PAUSE_EVENT.clear()  # unblock a paused wait so cancel can be observed
    return (
        gr.update(value="暂停", interactive=False),
        gr.update(interactive=False),
        _progress_markup(False),
        "⏹ 正在停止…已完成的页会保留并可导出。",
    )


def _wait_if_paused() -> None:
    while _PAUSE_EVENT.is_set() and not _CANCEL_EVENT.is_set():
        time.sleep(0.12)
    if _CANCEL_EVENT.is_set():
        raise JobCancelled()


def _raise_if_cancelled() -> None:
    if _CANCEL_EVENT.is_set():
        raise JobCancelled()


def _file_update(path: Optional[Path]):
    """Gradio DownloadButton value; keep previous if path missing."""
    if path is None:
        return gr.update()
    try:
        p = Path(path).resolve()
        if p.is_file():
            return gr.update(value=str(p))
    except Exception:
        pass
    return gr.update()


def _save_bundle(
    run_dir: Path,
    *,
    mode: str,
    input_type: str,
    result: StableResult,
    metrics: Dict[str, Any],
    report: config.EnvReport,
    log_extra: str = "",
) -> Dict[str, Path]:
    """Write current result set so export works mid-run and after cancel."""
    rj = build_result_json(
        mode,
        input_type,
        result.markdown,
        result.plain_text,
        result.elapsed_seconds,
        result.generated_tokens,
        pages=[p.__dict__ for p in result.pages],
        mini_uflash=metrics if "dflash" in mode or "mini_uflash" in mode else None,
    )
    files = save_result(
        run_dir,
        markdown=result.markdown or "",
        plain_text=result.plain_text or "",
        raw_output=result.raw_output or "",
        result_json=rj,
        metrics_json=metrics,
    )
    log = _build_log(
        mode,
        result.elapsed_seconds,
        result.generated_tokens,
        len(result.pages),
        report,
        extra=log_extra,
    )
    save_log(run_dir, log)
    return files


def on_run(
    file_path,
    model_path_str,
    mode,
    preset,
    max_tokens,
    probe_tokens,
    tier="balanced",
    progress=gr.Progress(),
):
    """Run OCR on the uploaded file.

    Yields partial PDF text page by page, followed by the final downloads.
    """
    _PAUSE_EVENT.clear()
    _CANCEL_EVENT.clear()
    # Validate input.
    if file_path is None:
        yield _run_error("⚠️ 请先上传图片或 PDF")
        return

    if not MODEL_MANAGER.is_loaded:
        try:
            if progress is not None:
                progress(0, desc="首次使用，正在准备本地模型")
            MODEL_MANAGER.load(model_path_str or config.unlimited_ocr_path())
        except Exception as e:
            yield _run_error(f"模型未能加载：{e}")
            return

    handles = MODEL_MANAGER.handles()
    if handles is None:
        yield _run_error("❌ 模型句柄无效")
        return

    file_path = Path(file_path)
    is_pdf_input = is_pdf(file_path)
    is_precise = mode == "Mini UFlash 精确模式"
    tier_key = str(tier or "balanced").strip().lower()
    if tier_key not in ("fast", "balanced", "lossless"):
        tier_key = "balanced"

    # 8GB safety: UI default 4096 will OOM with Unlimited-OCR + drafter + KV.
    max_tokens = clamp_max_length(int(max_tokens or 2048))
    free_gb, total_gb = mem_info_gb()
    _log.info(
        "OCR start max_length=%s free_vram=%.2fGB total=%.2fGB tier=%s mode=%s",
        max_tokens,
        free_gb,
        total_gb,
        tier_key if is_precise else "-",
        mode,
    )
    if free_gb > 0 and free_gb < 1.2:
        empty_cuda()
        free_gb, _ = mem_info_gb()
        if free_gb < 0.9:
            yield _run_error(
                oom_user_message(
                    "启动前空闲显存过低，已拒绝开始。请先「释放显存」或关闭其它 GPU 程序。"
                )
            )
            return

    report = _refresh_env_report()
    report.mode = f"{mode} · {tier_key}" if is_precise else mode

    run_dir = create_run_dir()
    save_input(file_path, run_dir)
    cancelled = False
    result: Optional[StableResult] = None
    metrics_payload: Dict[str, Any] = {}

    try:
        empty_cuda()
        if not is_precise:
            for partial, metrics_payload, complete in _iter_stable(
                handles, file_path, run_dir, preset, max_tokens, progress,
            ):
                result = partial
                files = _save_bundle(
                    run_dir,
                    mode="stable",
                    input_type="pdf" if is_pdf_input else "image",
                    result=partial,
                    metrics={
                        "mode": "stable",
                        "elapsed_seconds": partial.elapsed_seconds,
                        "generated_tokens": partial.generated_tokens,
                        "pages": len(partial.pages),
                        "vram_gb": getattr(handles, "vram_gb", 0),
                        "attention_backend": getattr(handles, "attention_backend", ""),
                        "complete": complete,
                    },
                    report=report,
                )
                status = (
                    f"已识别 {len(partial.pages)} 页"
                    + ("，完成。" if complete else "，文字已显示；正在继续…")
                )
                yield _stream_response(partial, report, status, files, metrics_mode="stable")
                if complete:
                    break
            if result is None:
                raise RuntimeError("没有生成任何识别页面")
            met = format_metrics_report(
                {
                    "mode": "stable",
                    "elapsed_seconds": result.elapsed_seconds,
                    "generated_tokens": result.generated_tokens,
                    "pages": [p.__dict__ for p in result.pages],
                },
                mode="stable",
            )
            log = _build_log(
                "stable",
                result.elapsed_seconds,
                result.generated_tokens,
                len(result.pages),
                report,
            )
            files = _save_bundle(
                run_dir,
                mode="stable",
                input_type="pdf" if is_pdf_input else "image",
                result=result,
                metrics={
                    "mode": "stable",
                    "elapsed_seconds": result.elapsed_seconds,
                    "generated_tokens": result.generated_tokens,
                    "pages": len(result.pages),
                    "vram_gb": getattr(handles, "vram_gb", 0),
                    "attention_backend": getattr(handles, "attention_backend", ""),
                },
                report=report,
            )
            status_parts = [
                "识别完成",
                f"耗时 {result.elapsed_seconds:.1f}s",
                f"{result.generated_tokens} tokens",
                f"{len(result.pages)} 页" if is_pdf_input else "",
                f"导出目录 {run_dir.name}",
            ]
            status = " | ".join(p for p in status_parts if p)
            yield _final_response(result, report, status, files, met, log)
            return

        # ---- Stable DFlash ----
        for partial, metrics_payload, complete in _iter_stable_dflash(
            handles,
            file_path,
            run_dir,
            preset,
            max_tokens,
            progress,
            tier=tier_key,
        ):
            result = partial
            files = _save_bundle(
                run_dir,
                mode="mini_uflash_stable_dflash",
                input_type="pdf" if is_pdf_input else "image",
                result=partial,
                metrics=metrics_payload,
                report=report,
                log_extra=f"Tier: {tier_key}",
            )
            status = (
                f"稳定 DFlash（{tier_key}）已完成 {len(partial.pages)} 页"
                + (
                    "，完成。"
                    if complete
                    else (
                        f"；投机提交 {metrics_payload.get('direct_block_commits', 0)} 次 · "
                        f"重同步 {metrics_payload.get('resync_count', 0)} 次"
                    )
                )
            )
            yield _stream_response(
                partial,
                report,
                status,
                files,
                metrics_mode="mini_uflash_stable_dflash",
                metrics=metrics_payload,
            )
            if complete:
                break
        if result is None:
            raise RuntimeError("没有生成任何识别页面")
        met = format_metrics_report(
            metrics_payload, mode="mini_uflash_stable_dflash"
        )
        page_total = len(result.pages)
        log = _build_log(
            "mini_uflash_stable_dflash",
            result.elapsed_seconds,
            result.generated_tokens,
            page_total,
            report,
            extra=(
                f"Tier: {tier_key}\n"
                f"Verified prefix commits: {metrics_payload.get('direct_block_commits', 0)}\n"
                f"Full-block commits: {metrics_payload.get('full_block_commits', 0)}\n"
                f"Resync count: {metrics_payload.get('resync_count', 0)}\n"
                f"Pure B1 rounds: {metrics_payload.get('pure_b1_rounds', 0)}\n"
                f"Fallback pages: {len(metrics_payload.get('fallback_pages', []) or [])}\n"
                f"Output dir: {run_dir}"
            ),
        )
        files = _save_bundle(
            run_dir,
            mode="mini_uflash_stable_dflash",
            input_type="pdf" if is_pdf_input else "image",
            result=result,
            metrics=metrics_payload,
            report=report,
            log_extra=f"Tier: {tier_key}",
        )
        status = (
            f"稳定 DFlash（{tier_key}）完成 | {page_total} 页 | "
            f"耗时 {result.elapsed_seconds:.1f}s | "
            f"投机提交 {metrics_payload.get('direct_block_commits', 0)} 次 | "
            f"导出 {run_dir.name}"
        )
        if metrics_payload.get("fallback_pages"):
            status += f" | {len(metrics_payload['fallback_pages'])} 页回退普通模式"
        yield _final_response(result, report, status, files, met, log)
        return

    except JobCancelled:
        cancelled = True
        _log.info("OCR job cancelled by user; pages=%s dir=%s",
                  0 if result is None else len(result.pages), run_dir)
        empty_cuda()
        if result is not None and result.pages:
            files = _save_bundle(
                run_dir,
                mode="cancelled",
                input_type="pdf" if is_pdf_input else "image",
                result=result,
                metrics={**(metrics_payload or {}), "cancelled": True},
                report=report,
                log_extra="User cancelled",
            )
            met = format_metrics_report(
                metrics_payload or {"mode": "cancelled"},
                mode="mini_uflash_stable_dflash"
                if is_precise
                else "stable",
            )
            log = _build_log(
                "cancelled",
                result.elapsed_seconds,
                result.generated_tokens,
                len(result.pages),
                report,
                extra=f"Cancelled. Partial output: {run_dir}",
            )
            status = (
                f"⏹ 已停止 | 保留 {len(result.pages)} 页 | "
                f"可导出 {run_dir.name}"
            )
            yield _final_response(result, report, status, files, met, log)
        else:
            yield _run_error("⏹ 已停止（尚无完成任何页）。")
    except Exception as e:
        if is_cuda_oom(e):
            _log.exception("CUDA OOM during OCR")
            empty_cuda()
            # Prefer partial export if any page finished.
            if result is not None and result.pages:
                try:
                    files = _save_bundle(
                        run_dir,
                        mode="oom_partial",
                        input_type="pdf" if is_pdf_input else "image",
                        result=result,
                        metrics={**(metrics_payload or {}), "oom": True},
                        report=report,
                    )
                    status = (
                        oom_user_message(
                            f"已保留 **{len(result.pages)}** 页，可导出目录：`{run_dir.name}`"
                        )
                    )
                    yield _final_response(
                        result,
                        report,
                        status,
                        files,
                        format_metrics_report(
                            metrics_payload or {}, mode="stable"
                        ),
                        _build_log(
                            "oom",
                            result.elapsed_seconds,
                            result.generated_tokens,
                            len(result.pages),
                            report,
                            extra=str(e),
                        ),
                    )
                    return
                except Exception:
                    pass
            yield _run_error(oom_user_message())
            return
        _log.exception("OCR run failed")
        tb = traceback.format_exc()
        empty_cuda()
        # Keep partial exports if any pages finished.
        if result is not None and result.pages:
            try:
                files = _save_bundle(
                    run_dir,
                    mode="error_partial",
                    input_type="pdf" if is_pdf_input else "image",
                    result=result,
                    metrics={**(metrics_payload or {}), "error": repr(e)},
                    report=report,
                )
                status = (
                    f"❌ 运行中断：{e}\n\n"
                    f"已保留 {len(result.pages)} 页，可从导出按钮或目录下载：\n`{run_dir}`"
                )
                yield _final_response(
                    result,
                    report,
                    status,
                    files,
                    format_metrics_report(metrics_payload or {}, mode="stable"),
                    _build_log("error", result.elapsed_seconds, result.generated_tokens,
                               len(result.pages), report, extra=tb),
                )
                return
            except Exception:
                pass
        yield _run_error(f"❌ 运行失败：{e}\n\n```\n{tb}\n```")
    finally:
        _PAUSE_EVENT.clear()
        if not cancelled:
            _CANCEL_EVENT.clear()


def _run_error(message: str):
    """Keep callback errors in the action status slot, not the page header."""
    return (
        "",
        "",
        "",
        "",
        "",
        message,
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        render_status_html(_refresh_env_report()),
    )


def _stream_response(
    result: StableResult,
    report: config.EnvReport,
    status: str,
    files: Dict[str, Path],
    *,
    metrics_mode: str = "stable",
    metrics: Optional[Dict[str, Any]] = None,
):
    """Page-by-page UI update; downloads point at progressively saved files."""
    met = ""
    if metrics:
        try:
            met = format_metrics_report(metrics, mode=metrics_mode)
        except Exception:
            met = ""
    return (
        result.markdown or "",
        result.plain_text or "",
        result.raw_output or "",
        met,
        "",
        status,
        _file_update(files.get("result.md")),
        _file_update(files.get("result.txt")),
        _file_update(files.get("result.json")),
        _file_update(files.get("metrics.json")),
        render_status_html(report),
    )


def _final_response(
    result: StableResult,
    report: config.EnvReport,
    status: str,
    files: Dict[str, Path],
    metrics_md: str,
    log: str,
):
    return (
        result.markdown or "",
        result.plain_text or "",
        result.raw_output or "",
        metrics_md or "",
        log or "",
        status,
        _file_update(files.get("result.md")),
        _file_update(files.get("result.txt")),
        _file_update(files.get("result.json")),
        _file_update(files.get("metrics.json")),
        render_status_html(report),
    )


def _iter_stable(handles, file_path, run_dir, preset, max_tokens, progress=None):
    """Yield stable OCR after each completed page."""
    pages_dir = run_dir / "pages"
    scratch = config.WEBAPP_DIR / "_stable_scratch"

    if is_pdf(file_path):
        total_pages = page_count(file_path)
        pages_dir.mkdir(parents=True, exist_ok=True)
        rendered = render_pages(file_path, pages_dir)
        page_results = []
        started = time.perf_counter()
        for rp in rendered:
            _raise_if_cancelled()
            _wait_if_paused()
            _raise_if_cancelled()
            if progress is not None:
                progress(
                    rp.page_index / max(1, total_pages),
                    desc=f"正在识别第 {rp.page_index + 1}/{total_pages} 页",
                )
            page_scratch = scratch / f"page_{rp.page_index}"
            try:
                page_results.append(
                    recognize_image(
                        handles, rp.image_path, page_scratch,
                        preset_name=preset, max_length=max_tokens,
                        page_index=rp.page_index,
                    )
                )
            except Exception as exc:
                if is_cuda_oom(exc):
                    empty_cuda()
                    # Retry once at shorter budget.
                    short = min(int(max_tokens), 1024)
                    _log.warning("OOM on stable page %s; retry max_length=%s",
                                 rp.page_index + 1, short)
                    page_results.append(
                        recognize_image(
                            handles, rp.image_path, page_scratch,
                            preset_name=preset, max_length=short,
                            page_index=rp.page_index,
                        )
                    )
                else:
                    raise
            empty_cuda()
            elapsed = time.perf_counter() - started
            done = len(page_results) == total_pages
            yield assemble_result(page_results, elapsed), {}, done
            if done:
                break
        if progress is not None and not _CANCEL_EVENT.is_set():
            progress(1, desc="PDF 识别完成")
        cleanup_dir(scratch)
        empty_cuda()
    else:
        _raise_if_cancelled()
        started = time.perf_counter()
        try:
            page_result = recognize_image(
                handles, file_path, scratch,
                preset_name=preset, max_length=max_tokens,
            )
        except Exception as exc:
            if is_cuda_oom(exc):
                empty_cuda()
                short = min(int(max_tokens), 1024)
                page_result = recognize_image(
                    handles, file_path, scratch,
                    preset_name=preset, max_length=short,
                )
            else:
                raise
        elapsed = time.perf_counter() - started
        result = assemble_result([page_result], elapsed)
        cleanup_dir(scratch)
        empty_cuda()
        yield result, {}, True


def _aggregate_dflash_metrics(
    page_metrics: list[dict],
    elapsed_seconds: float,
    fallback_pages: list[int],
    *,
    tier: str = "",
) -> dict:
    """Combine per-page Stable DFlash metrics without hiding fallback pages."""
    dflash = [
        m
        for m in page_metrics
        if m.get("mode") in ("mini_uflash_stable_dflash", "mini_uflash_direct")
    ]
    rounds = sum(int(m.get("speculative_rounds", 0)) for m in dflash)
    commits = sum(int(m.get("full_block_commits", 0)) for m in dflash)
    target_forwards = sum(int(m.get("target_decode_forwards", 0)) for m in dflash)
    generated = sum(int(m.get("generated_tokens", 0)) for m in page_metrics)
    tier_key = str(tier or "").strip().lower()
    if not tier_key:
        for m in dflash:
            if m.get("tier"):
                tier_key = str(m["tier"])
                break
    return {
        "mode": (
            "mini_uflash_stable_dflash_pdf"
            if len(page_metrics) > 1
            else "mini_uflash_stable_dflash"
        ),
        "tier": tier_key or None,
        "non_lossless": bool(tier_key and tier_key != "lossless"),
        "pages": len(page_metrics),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "generated_tokens": generated,
        "speculative_rounds": rounds,
        "direct_block_commits": sum(
            int(m.get("direct_block_commits", 0)) for m in dflash
        ),
        "full_block_commits": commits,
        "fallback_rounds": sum(int(m.get("fallback_rounds", 0)) for m in dflash),
        "resync_count": sum(int(m.get("resync_count", 0)) for m in dflash),
        "pure_b1_rounds": sum(int(m.get("pure_b1_rounds", 0)) for m in dflash),
        "direct_committed_tokens": sum(
            int(m.get("direct_committed_tokens", 0)) for m in dflash
        ),
        "target_decode_forwards": target_forwards,
        "target_forward_reduction": (
            round(generated / target_forwards, 3) if target_forwards else 0.0
        ),
        "full_block_ratio": round(commits / rounds, 4) if rounds else 0.0,
        "fallback_pages": fallback_pages,
        "page_metrics": page_metrics,
        "warning": (
            "稳定 DFlash：验证前缀提交 + 周期 prefill 重同步；"
            "长文优先抑制 cache 漂移与退化。"
        ),
    }


def _iter_stable_dflash(
    handles,
    file_path: Path,
    run_dir: Path,
    preset: str,
    max_tokens: int,
    progress=None,
    *,
    tier: str = "fast",
):
    """Yield Stable DFlash output after every recoverably saved page."""
    started = time.perf_counter()
    total_pages = page_count(file_path) if is_pdf(file_path) else 1
    rendered_iter = (
        iter_render_pages(file_path, run_dir / "pages")
        if is_pdf(file_path)
        else [type("Page", (), {"page_index": 0, "image_path": file_path})()]
    )
    page_results: list[StablePageResult] = []
    page_metrics: list[dict] = []
    fallback_pages: list[int] = []
    tier_key = str(tier or "fast").strip().lower() or "fast"

    for rendered in rendered_iter:
        _raise_if_cancelled()
        page_index = int(rendered.page_index)

        def _page_progress(info, index=page_index):
            _wait_if_paused()
            _raise_if_cancelled()
            if progress is None:
                return
            inner = info.get("generated", 0) / max(1, info.get("target", 1))
            overall = (index + min(0.98, inner)) / max(1, total_pages)
            progress(
                overall,
                desc=(
                    f"稳定 DFlash（{tier_key}）第 {index + 1}/{total_pages} 页 · "
                    f"提交 {info.get('direct_commits', 0)} · "
                    f"重同步 {info.get('resync_count', 0)}"
                ),
            )

        try:
            dflash = run_stable_dflash_mode(
                handles,
                Path(rendered.image_path),
                preset_name=preset,
                max_length=max_tokens,
                tier=tier_key,
                progress_callback=_page_progress,
            )
            page_result = StablePageResult(
                page_index=page_index,
                raw_markdown=dflash.raw_output,
                markdown=dflash.markdown,
                elapsed_seconds=dflash.total_seconds,
                generated_tokens=dflash.generated_tokens,
            )
            metric = dflash.to_dict()
        except JobCancelled:
            raise
        except Exception as dflash_error:
            _log.exception(
                "Stable DFlash failed on page %d; using stable fallback",
                page_index + 1,
            )
            fallback_pages.append(page_index + 1)
            try:
                live = MODEL_MANAGER.handles() or handles
                if live is None or getattr(live, "model", None) is None:
                    live = MODEL_MANAGER.load()
                page_result = recognize_image(
                    live,
                    Path(rendered.image_path),
                    config.WEBAPP_DIR / "_direct_fallback" / f"page_{page_index}",
                    preset_name=preset,
                    max_length=max_tokens,
                    page_index=page_index,
                )
                metric = {
                    "mode": "stable_fallback",
                    "page": page_index + 1,
                    "generated_tokens": page_result.generated_tokens,
                    "elapsed_seconds": page_result.elapsed_seconds,
                    "dflash_error": repr(dflash_error),
                }
            except Exception as fallback_error:
                page_result = StablePageResult(
                    page_index=page_index,
                    raw_markdown="",
                    markdown="",
                    elapsed_seconds=0.0,
                    error=f"稳定 DFlash 与普通模式均失败：{fallback_error}",
                )
                metric = {
                    "mode": "failed",
                    "page": page_index + 1,
                    "generated_tokens": 0,
                    "dflash_error": repr(dflash_error),
                    "fallback_error": repr(fallback_error),
                }

        page_results.append(page_result)
        page_metrics.append(metric)
        # Drop per-page GPU residuum before the next prefill (critical on 8GB).
        empty_cuda()
        elapsed = time.perf_counter() - started
        combined = assemble_result(page_results, elapsed)
        aggregate = _aggregate_dflash_metrics(
            page_metrics, elapsed, fallback_pages, tier=tier_key
        )
        aggregate["complete"] = len(page_results) == total_pages

        page_dir = run_dir / "page_results" / f"page_{page_index + 1:05d}"
        page_dir.mkdir(parents=True, exist_ok=True)
        page_json = build_result_json(
            metric.get("mode", "mini_uflash_stable_dflash"), "image",
            page_result.markdown, "", page_result.elapsed_seconds,
            page_result.generated_tokens,
            pages=[page_result.__dict__], mini_uflash=metric,
        )
        save_result(
            page_dir, markdown=page_result.markdown,
            raw_output=page_result.raw_markdown,
            result_json=page_json, metrics_json=metric,
        )
        partial_json = build_result_json(
            "mini_uflash_stable_dflash", "pdf" if total_pages > 1 else "image",
            combined.markdown, combined.plain_text, elapsed,
            combined.generated_tokens,
            pages=[p.__dict__ for p in page_results], mini_uflash=aggregate,
        )
        save_result(
            run_dir, markdown=combined.markdown, plain_text=combined.plain_text,
            raw_output=combined.raw_output, result_json=partial_json,
            metrics_json=aggregate,
        )
        if progress is not None:
            progress(
                len(page_results) / max(1, total_pages),
                desc=f"已保存第 {page_index + 1}/{total_pages} 页",
            )
        yield combined, aggregate, aggregate["complete"]

    cleanup_dir(config.WEBAPP_DIR / "_direct_fallback")
    empty_cuda()


def _build_log(mode, elapsed, tokens, pages, report, extra=""):
    parts = [
        f"Mode: {mode}",
        f"Elapsed: {elapsed:.1f}s",
        f"Tokens: {tokens}",
        f"Pages: {pages}",
        f"GPU: {report.gpu_name}" if report.cuda_available else "GPU: CPU",
        f"Attention: {report.attention_backend}",
    ]
    if extra:
        parts.append("")
        parts.append(extra)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Gradio performs a localhost reachability check. Never route it through a
    # user-configured HTTP proxy or it may incorrectly demand share=True.
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
    os.environ.setdefault("no_proxy", os.environ["NO_PROXY"])
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    _setup_logging()
    _log.info("Mini UFlash OCR starting ...")
    _log.info("Project root: %s", config.PROJECT_ROOT)
    _log.info("Unlimited-OCR path: %s", config.unlimited_ocr_path())
    _log.info("Drafter weight: %s", config.discover_weight())

    report = config.build_static_env_report()
    _log.info("Platform: %s", report.platform)
    _log.info("Python: %s (venv: %s)", report.python_version, report.venv_python)
    _log.info("Torch: %s / CUDA %s / %s", report.torch_version,
              report.cuda_version, report.gpu_name)
    _log.info("Model loaded: %s", MODEL_MANAGER.is_loaded)

    callbacks = {
        "on_file_change": on_file_change,
        "on_mode_change": on_mode_change,
        "on_load_model": on_load_model,
        "on_unload_model": on_unload_model,
        "on_toggle_pause": on_toggle_pause,
        "on_stop_job": on_stop_job,
        "on_run": on_run,
        "on_clear": None,  # use ui._clear_all
    }
    ui = build_ui(callbacks)

    _log.info("Launching Gradio on %s:%d ...", config.HOST, config.PORT)
    favicon = config.ASSETS_DIR / "logo-64.png"
    if not favicon.is_file():
        favicon = config.ASSETS_DIR / "logo.png"
    # Downloads live under webapp/outputs — must be allowed for DownloadButton.
    allowed = [
        str(config.ASSETS_DIR.resolve()),
        str(config.OUTPUTS_DIR.resolve()),
        str(config.WEBAPP_DIR.resolve()),
        str(config.PROJECT_ROOT.resolve()),
    ]
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    launch_kwargs = {
        "server_name": config.HOST,
        "server_port": config.PORT,
        "share": False,
        "prevent_thread_lock": False,
        "show_error": True,
        "allowed_paths": allowed,
        "max_threads": 8,
    }
    if favicon.is_file():
        launch_kwargs["favicon_path"] = str(favicon)
    ui.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
