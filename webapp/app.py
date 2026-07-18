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

_log = logging.getLogger(__name__)
_PAUSE_EVENT = threading.Event()


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
    report = _refresh_env_report()
    return "模型已卸载", render_status_html(report)


def on_toggle_pause():
    """Pause or resume the active OCR loop without discarding completed pages."""
    from .ui import _progress_markup

    if _PAUSE_EVENT.is_set():
        _PAUSE_EVENT.clear()
        return gr.update(value="暂停"), _progress_markup(True)
    _PAUSE_EVENT.set()
    return gr.update(value="继续"), _progress_markup(True, paused=True)


def _wait_if_paused() -> None:
    while _PAUSE_EVENT.is_set():
        time.sleep(0.15)


def on_run(
    file_path,
    model_path_str,
    mode,
    preset,
    max_tokens,
    probe_tokens,
    tier="fast",
    progress=gr.Progress(),
):
    """Run OCR on the uploaded file.

    Yields partial PDF text page by page, followed by the final downloads.
    """
    _PAUSE_EVENT.clear()
    # Validate input.
    if file_path is None:
        yield _run_error("⚠️ 请先上传图片或 PDF")
        return

    if not MODEL_MANAGER.is_loaded:
        try:
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
    # Normalize tier; default product path is wall-clock first (fast).
    tier_key = str(tier or "fast").strip().lower()
    if tier_key not in ("fast", "balanced", "lossless"):
        tier_key = "fast"

    report = _refresh_env_report()
    report.mode = f"{mode} · {tier_key}" if is_precise else mode

    # Create output directory.
    run_dir = create_run_dir()
    save_input(file_path, run_dir)

    started = time.perf_counter()
    log_lines: list[str] = []

    try:
        if not is_precise:
            # ---- Stable mode (image or PDF) ----
            result = None
            metrics_dict = {}
            for partial, metrics_dict, complete in _iter_stable(
                handles, file_path, run_dir, preset, max_tokens, progress,
            ):
                result = partial
                if not complete:
                    yield _partial_response(
                        partial, report,
                        f"已识别 {len(partial.pages)} 页，文字已显示；正在继续…",
                    )
            if result is None:
                raise RuntimeError("没有生成任何识别页面")
            md = result.markdown
            txt = result.plain_text
            raw = result.raw_output
            met = format_metrics_report(result.to_dict(), mode="stable")
            log = _build_log("stable", result.elapsed_seconds, result.generated_tokens,
                             len(result.pages), report)

            # Save files.
            rj = build_result_json(
                "stable", "pdf" if is_pdf_input else "image",
                md, txt, result.elapsed_seconds, result.generated_tokens,
                pages=[p.__dict__ for p in result.pages],
            )
            stable_metrics = {
                "mode": "stable",
                "elapsed_seconds": result.elapsed_seconds,
                "generated_tokens": result.generated_tokens,
                "pages": len(result.pages),
                "vram_gb": handles.vram_gb,
                "attention_backend": handles.attention_backend,
            }
            files = save_result(run_dir, markdown=md, plain_text=txt,
                                raw_output=raw, result_json=rj,
                                metrics_json=stable_metrics)
            save_log(run_dir, log)

            # Downloads.
            dl_md = str(files.get("result.md", ""))
            dl_txt = str(files.get("result.txt", ""))
            dl_json = str(files.get("result.json", ""))
            dl_met = str(files.get("metrics.json", ""))

            status_parts = [
                "识别完成",
                f"耗时 {result.elapsed_seconds:.1f}s",
                f"{result.generated_tokens} tokens",
                f"{len(result.pages)} 页" if is_pdf_input else "",
            ]
            status = " | ".join(p for p in status_parts if p)
            yield (md, txt, raw, met, log, status,
                   dl_md, dl_txt, dl_json, dl_met,
                   render_status_html(report))
            return

        else:
            # ---- Stable DFlash (verified prefix + periodic resync) ----
            result = None
            dflash_metrics = {}
            for partial, dflash_metrics, complete in _iter_stable_dflash(
                handles,
                file_path,
                run_dir,
                preset,
                max_tokens,
                progress,
                tier=tier_key,
            ):
                result = partial
                if not complete:
                    yield _partial_response(
                        partial, report,
                        (
                            f"稳定 DFlash（{tier_key}）已完成 {len(partial.pages)} 页，文字已显示；"
                            f"投机提交 {dflash_metrics.get('direct_block_commits', 0)} 次 · "
                            f"重同步 {dflash_metrics.get('resync_count', 0)} 次"
                        ),
                    )
            if result is None:
                raise RuntimeError("没有生成任何识别页面")
            md = result.markdown
            txt = result.plain_text
            raw = result.raw_output
            met = format_metrics_report(
                dflash_metrics, mode="mini_uflash_stable_dflash"
            )
            page_total = len(result.pages)
            log = _build_log(
                "mini_uflash_stable_dflash", result.elapsed_seconds,
                result.generated_tokens, page_total, report,
                extra=(
                    f"Tier: {tier_key}\n"
                    f"Verified prefix commits: {dflash_metrics.get('direct_block_commits', 0)}\n"
                    f"Full-block commits: {dflash_metrics.get('full_block_commits', 0)}\n"
                    f"Resync count: {dflash_metrics.get('resync_count', 0)}\n"
                    f"Pure B1 rounds: {dflash_metrics.get('pure_b1_rounds', 0)}\n"
                    f"Fallback pages: {len(dflash_metrics.get('fallback_pages', []))}\n"
                    "Policy: verified-prefix crop + periodic prefill resync + tier schedule"
                ),
            )
            rj = build_result_json(
                "mini_uflash_stable_dflash",
                "pdf" if is_pdf_input else "image",
                md, txt, result.elapsed_seconds, result.generated_tokens,
                pages=[p.__dict__ for p in result.pages],
                mini_uflash=dflash_metrics,
            )
            files = save_result(
                run_dir, markdown=md, plain_text=txt, raw_output=raw,
                result_json=rj, metrics_json=dflash_metrics,
            )
            save_log(run_dir, log)
            dl_md = str(files.get("result.md", ""))
            dl_txt = str(files.get("result.txt", ""))
            dl_json = str(files.get("result.json", ""))
            dl_met = str(files.get("metrics.json", ""))
            status = (
                f"稳定 DFlash（{tier_key}）完成 | {page_total} 页 | "
                f"耗时 {result.elapsed_seconds:.1f}s | "
                f"投机提交 {dflash_metrics.get('direct_block_commits', 0)} 次 | "
                f"重同步 {dflash_metrics.get('resync_count', 0)} 次"
            )
            if dflash_metrics.get("fallback_pages"):
                status += f" | {len(dflash_metrics['fallback_pages'])} 页回退普通模式"
            yield (md, txt, raw, met, log, status,
                   dl_md, dl_txt, dl_json, dl_met,
                   render_status_html(report))
            return

    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        yield _run_error("❌ 显存不足。请关闭其他 GPU 程序、降低最大生成长度或重新加载模型。")
    except Exception as e:
        _log.exception("OCR run failed")
        tb = traceback.format_exc()
        yield _run_error(f"❌ 运行失败：{e}\n\n```\n{tb}\n```")
    finally:
        _PAUSE_EVENT.clear()


def _run_error(message: str):
    """Keep callback errors in the action status slot, not the page header."""
    return (
        "", "", "", "", "", message,
        None, None, None, None, render_status_html(_refresh_env_report()),
    )


def _partial_response(result: StableResult, report: config.EnvReport, status: str):
    """Expose completed pages immediately while keeping downloads final-only."""
    return (
        result.markdown, result.plain_text, result.raw_output, "", "", status,
        None, None, None, None, render_status_html(report),
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
            _wait_if_paused()
            if progress is not None:
                progress(rp.page_index / max(1, total_pages), desc=f"正在识别第 {rp.page_index + 1}/{total_pages} 页")
            page_scratch = scratch / f"page_{rp.page_index}"
            page_results.append(
                recognize_image(
                    handles, rp.image_path, page_scratch,
                    preset_name=preset, max_length=max_tokens,
                    page_index=rp.page_index,
                )
            )
            elapsed = time.perf_counter() - started
            yield assemble_result(page_results, elapsed), {}, len(page_results) == total_pages
        if progress is not None:
            progress(1, desc="PDF 识别完成")
        # Cleanup temp images.
        cleanup_dir(scratch)
    else:
        started = time.perf_counter()
        page_result = recognize_image(
            handles, file_path, scratch,
            preset_name=preset, max_length=max_tokens,
        )
        elapsed = time.perf_counter() - started
        result = assemble_result([page_result], elapsed)
        cleanup_dir(scratch)
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
        page_index = int(rendered.page_index)

        def _page_progress(info, index=page_index):
            _wait_if_paused()
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
        "on_run": on_run,
        "on_clear": lambda: (
            None, gr.update(value=None, visible=False),
            "*识别结果会显示在这里。*", "", "", "", "",
            "选择文件后，点击“识别文档”即可。",
            None, None, None, None, None,
        ),
    }
    ui = build_ui(callbacks)

    _log.info("Launching Gradio on %s:%d ...", config.HOST, config.PORT)
    favicon = config.ASSETS_DIR / "logo-64.png"
    if not favicon.is_file():
        favicon = config.ASSETS_DIR / "logo.png"
    launch_kwargs = {
        "server_name": config.HOST,
        "server_port": config.PORT,
        "share": False,
        "prevent_thread_lock": False,
        "show_error": True,
        "allowed_paths": [str(config.ASSETS_DIR)],
    }
    if favicon.is_file():
        launch_kwargs["favicon_path"] = str(favicon)
    ui.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
