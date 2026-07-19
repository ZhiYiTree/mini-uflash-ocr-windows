"""Simple, inclusive Gradio interface for Mini UFlash OCR."""

from __future__ import annotations

import base64
from html import escape
from pathlib import Path
from typing import Any, Dict

import gradio as gr  # type: ignore

from . import config


def _logo_data_uri(name: str = "logo-128.png") -> str:
    """Embed project logo for Gradio HTML (no extra static mount)."""
    path = Path(config.ASSETS_DIR) / name
    if not path.is_file():
        path = Path(config.ASSETS_DIR) / "logo.png"
    if not path.is_file():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


CSS = r"""
:root {
    font: 100%/1.5 "Segoe UI Variable", "Microsoft YaHei UI", sans-serif;
    color-scheme: light dark;
    --canvas: #f4f7f5;
    --surface: #ffffff;
    --surface-solid: #fafcfb;
    --text: #1a211e;
    --muted: #63706b;
    --line: #d9e3dd;
    --accent: #21816a;
    --accent-hover: #1b6d59;
    --accent-soft: #e6f4ee;
    --accent-bright: #65bfa5;
    --focus: rgba(33, 129, 106, .25);
    --warning: #a95039;
    --warning-soft: #fcefe9;
}

body, .gradio-container { background: var(--canvas) !important; color: var(--text) !important; }
.gradio-container {
    max-width: 1440px !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 1.25rem 2rem 4rem !important;
}
.gradio-container .contain, .contain { max-width: 1440px !important; width: 100% !important; }

#product-header { padding: .35rem .25rem 1.55rem; }
.topbar { display: flex; align-items: center; justify-content: space-between; min-height: 2.4rem; }
.brand { display: flex; align-items: center; gap: .55rem; }
.brand-logo {
    width: 36px; height: 36px; border-radius: 10px; object-fit: contain;
    background: #fff; box-shadow: 0 2px 8px rgba(31,62,49,.08);
}
.wordmark { color: var(--text); font-size: 1.02rem; font-weight: 680; letter-spacing: -.02em; }
.hero-logo-wrap { display: flex; align-items: center; gap: .85rem; margin: 0 0 .35rem; }
.hero-logo {
    width: 72px; height: 72px; border-radius: 18px; object-fit: contain;
    background: #fff7ed; box-shadow: 0 8px 20px rgba(31,62,49,.1);
    flex: none;
}
.privacy-note { display: flex; align-items: center; gap: .52rem; padding: .48rem .72rem; border-radius: 14px; background: var(--accent-soft); color: var(--accent-hover); font-size: .8rem; font-weight: 650; }
.privacy-dot { width: .5rem; height: .5rem; border-radius: 4px; background: var(--accent); animation: status-breathe 2.2s ease-in-out infinite; }
.hero { max-width: 800px; padding-top: 2.4rem; }
.eyebrow { margin: 0 0 .45rem; color: var(--accent); font-size: .84rem; font-weight: 700; }
.hero h1 { margin: 0; color: var(--text); font-size: clamp(2.25rem, 4.5vw, 4.25rem); line-height: 1.02; letter-spacing: -.055em; font-weight: 720; text-wrap: balance; }
.mobile-break { display: none; }
.hero > p:last-child { max-width: 650px; margin: .8rem 0 0; color: var(--muted); font-size: 1.05rem; }
.status-bar { width: fit-content; margin-top: 1.1rem; padding: .52rem .76rem; border: 1px solid var(--line); border-radius: 14px; background: var(--surface); color: var(--muted); font-size: .78rem; }
.status-bar strong { color: var(--text); font-weight: 620; }

#workspace-row { align-items: stretch; gap: 1.1rem; flex-wrap: nowrap !important; }
#input-card, #result-card {
    padding: 1.45rem !important;
    border: 1px solid var(--line) !important;
    border-radius: 26px !important;
    background: var(--surface) !important;
    box-shadow: 0 14px 34px rgba(31,62,49,.07) !important;
}
#result-card { min-height: 700px; }
.section-heading { padding: .1rem .15rem 1rem; }
.section-heading h2 { margin: 0; color: var(--text); font-size: 1.24rem; line-height: 1.25; letter-spacing: -.02em; font-weight: 670; }
.section-heading p { margin: .28rem 0 0; color: var(--muted); font-size: .86rem; }
.result-heading { display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem; }
.live-badge { display: inline-flex; align-items: center; gap: .42rem; flex: none; padding: .44rem .66rem; border-radius: 12px; background: var(--accent-soft); color: var(--accent-hover); font-size: .75rem; font-weight: 700; }
.live-badge::before { content: ""; width: 7px; height: 7px; border-radius: 3px; background: var(--accent); animation: status-breathe 1.8s ease-in-out infinite; }
.field-heading { margin: 1rem .15rem .55rem; }
.field-heading h3 { margin: 0; color: var(--text); font-size: .92rem; font-weight: 650; }
.field-heading p { margin: .18rem 0 0; color: var(--muted); font-size: .78rem; }

#input-upload, #input-preview, #result-tabs, #advanced-settings, #technical-details, #mode-choice {
    border-color: var(--line) !important;
    border-radius: 16px !important;
    box-shadow: none !important;
}
#input-upload { min-height: 158px; background: rgba(118,118,128,.055) !important; border-style: dashed !important; }
#input-preview { overflow: hidden; }
#mode-choice { padding: 0 !important; border: 0 !important; background: transparent !important; }
#mode-choice > .wrap:not(.default) { display: grid !important; grid-template-columns: 1fr 1fr; gap: .62rem !important; padding: 0 !important; background: transparent !important; }
#mode-choice > .wrap:not(.default) label {
    display: flex !important; justify-content: center !important; min-height: 3.45rem;
    margin: 0 !important; padding: .8rem .9rem !important;
    border: 1px solid var(--line) !important; border-radius: 16px !important;
    background: var(--surface-solid) !important; color: var(--text) !important;
    font-size: .86rem !important; font-weight: 680 !important; cursor: pointer;
    box-shadow: 0 4px 10px rgba(31,62,49,.05) !important;
    transition: border-color 160ms ease, background 160ms ease, color 160ms ease, transform 100ms ease;
}
#mode-choice input[type="radio"] { position: absolute !important; opacity: 0 !important; pointer-events: none !important; }
#mode-choice > .wrap:not(.default) label:has(input:checked) { border-color: var(--accent) !important; background: var(--accent) !important; color: #fff !important; box-shadow: 0 7px 16px rgba(22,122,102,.18) !important; }
#mode-choice > .wrap:not(.default) label:has(input:focus-visible) { outline: 3px solid var(--focus) !important; outline-offset: 2px; }
#mode-choice > .wrap:not(.default) label:active { transform: scale(.985); }
.experiment-note { margin: .68rem 0 .95rem; padding: .72rem .8rem; border-radius: 14px; background: var(--warning-soft); color: var(--warning); font-size: .76rem; line-height: 1.45; }

#primary-action, #pause-action, #clear-action { min-height: 3.25rem !important; border-radius: 14px !important; }
#primary-action {
    border: 0 !important; background: var(--accent) !important; color: #fff !important;
    font-size: 1rem !important; font-weight: 700 !important; box-shadow: 0 7px 16px rgba(22,122,102,.2) !important;
    transition: transform 100ms ease-out, background 160ms ease-out !important;
}
#primary-action:hover { background: var(--accent-hover) !important; }
#primary-action:active, #pause-action:active, #clear-action:active, .download-button:active { transform: scale(.975); }
#pause-action { border: 1px solid #bad8cd !important; background: var(--accent-soft) !important; color: var(--accent-hover) !important; font-weight: 680 !important; }
#pause-action:disabled { border-color: var(--line) !important; background: #f1f4f2 !important; color: #9aa49f !important; opacity: 1 !important; }
#stop-action { border: 1px solid #e7b4a8 !important; background: #fcefe9 !important; color: #a95039 !important; font-weight: 680 !important; }
#stop-action:disabled { border-color: var(--line) !important; background: #f1f4f2 !important; color: #9aa49f !important; opacity: 1 !important; }
#clear-action { background: rgba(118,118,128,.08) !important; color: var(--text) !important; font-weight: 580 !important; }
button:focus-visible, input:focus-visible, textarea:focus-visible, [tabindex]:focus-visible { outline: 3px solid var(--focus) !important; outline-offset: 2px !important; }

#run-status { min-height: 2.25rem; padding: .15rem .15rem 0; color: var(--muted); font-size: .82rem; }
#recognition-progress { margin: .55rem 0 .2rem; }
.ocr-progress[data-state="idle"] { display: none; }
.ocr-progress[data-state="running"] {
    display: grid; grid-template-columns: 58px 1fr; gap: .85rem; align-items: center;
    padding: .8rem; border: 1px solid #c7e2d8; border-radius: 16px; background: var(--accent-soft);
}
.ocr-progress[data-state="paused"] { border-color: #e0c6a9; background: #fbf3e8; }
.ocr-progress[data-state="paused"] .scan-beam,
.ocr-progress[data-state="paused"] .scan-page span,
.ocr-progress[data-state="paused"] .live-track i { animation-play-state: paused !important; }
.ocr-progress[data-state="paused"] .live-track i { width: 100%; background: #b77a37; }
.scan-page { position: relative; width: 44px; height: 54px; overflow: hidden; border: 1px solid #b9d8cd; border-radius: 10px; background: var(--surface-solid); box-shadow: 0 5px 12px rgba(31,62,49,.08); }
.scan-page span { display: block; width: 25px; height: 2px; margin: 9px 0 -3px 9px; border-radius: 2px; background: #9bcdbd; animation: ink-pulse 1.8s ease-in-out infinite; }
.scan-page span:nth-child(2) { width: 20px; animation-delay: .15s; }
.scan-page span:nth-child(3) { width: 28px; animation-delay: .3s; }
.scan-beam { position: absolute; left: 4px; right: 4px; top: 5px; height: 2px; border-radius: 2px; background: var(--accent); box-shadow: 0 0 7px rgba(22,122,102,.45); animation: scan-page 1.65s cubic-bezier(.37,0,.63,1) infinite alternate; }
.progress-copy strong { display: block; color: var(--text); font-size: .88rem; }
.progress-copy span { display: block; margin-top: .12rem; color: var(--muted); font-size: .76rem; }
.live-track { height: 5px; margin-top: .55rem; overflow: hidden; border-radius: 4px; background: #c4dfd5; }
.live-track i { display: block; width: 36%; height: 100%; border-radius: 4px; background: var(--accent); animation: progress-shuttle 1.25s cubic-bezier(.4,0,.2,1) infinite; }
@keyframes scan-page { from { transform: translateY(0); } to { transform: translateY(40px); } }
@keyframes ink-pulse { 0%,100% { opacity: .28; transform: scaleX(.8); transform-origin: left; } 50% { opacity: 1; transform: scaleX(1); } }
@keyframes progress-shuttle { from { transform: translateX(-105%); } to { transform: translateX(285%); } }

#result-tabs { min-height: 500px; overflow: hidden; background: var(--surface-solid) !important; }
#result-tabs [role="tablist"] { display: grid !important; grid-template-columns: 1fr 1fr; gap: .6rem !important; width: calc(100% - 1.4rem) !important; margin: .7rem !important; padding: 0 !important; border: 0 !important; background: transparent !important; }
#result-tabs [role="tab"] { min-height: 2.65rem !important; padding: .52rem 1rem !important; border: 1px solid var(--line) !important; border-radius: 14px !important; background: var(--surface-solid) !important; color: var(--muted) !important; font-weight: 650 !important; }
#result-tabs [role="tab"][aria-selected="true"] { border-color: var(--accent) !important; background: var(--accent) !important; color: #fff !important; box-shadow: 0 6px 14px rgba(22,122,102,.16) !important; }
#result-markdown {
    min-height: 435px !important; max-height: 62vh; overflow: auto !important;
    padding: 1.35rem 1.45rem !important; background: var(--surface-solid) !important;
    color: var(--text) !important; font-size: 1rem; line-height: 1.72;
}
#result-markdown, #result-markdown * { color: var(--text); opacity: 1; }
#result-markdown > div { animation: result-arrive 260ms ease-out; }
.result-empty { display: flex !important; align-items: center !important; justify-content: center !important; width: 100% !important; min-height: 390px; margin: 0 auto !important; text-align: center !important; color: var(--muted) !important; }
.result-empty > div { display: flex !important; flex-direction: column; align-items: center !important; justify-content: center !important; width: 100% !important; margin: 0 auto !important; color: var(--muted) !important; text-align: center !important; }
.result-empty strong { display: block; margin-bottom: .3rem; color: var(--text) !important; font-size: 1rem; }
.result-empty svg { display: block; width: 42px; height: 42px; margin: 0 auto .8rem !important; color: #78847e !important; }
#plain-result textarea { min-height: 430px !important; color: var(--text) !important; background: var(--surface-solid) !important; font-size: .96rem !important; line-height: 1.65 !important; }
.export-label { margin: .75rem .15rem .45rem; color: var(--muted); font-size: .76rem; font-weight: 600; }
.download-button { min-height: 2.55rem !important; border: 1px solid var(--line) !important; border-radius: 14px !important; background: var(--surface-solid) !important; color: var(--text) !important; font-size: .8rem !important; transition: transform 100ms ease-out, border-color 160ms ease !important; }
.download-button:hover { border-color: var(--accent) !important; }
#advanced-settings, #technical-details { margin-top: .7rem; overflow: hidden !important; padding: 0 !important; border: 1px solid var(--line) !important; border-radius: 18px !important; background: var(--surface-solid) !important; }
#advanced-settings > button, #technical-details > button { width: 100% !important; min-height: 3.15rem !important; padding: .75rem .9rem !important; border-radius: 17px !important; background: var(--accent-soft) !important; color: var(--text) !important; font-weight: 680 !important; }
#advanced-settings > button .icon, #technical-details > button .icon { display: grid !important; place-items: center; width: 1.7rem; height: 1.7rem; border-radius: 8px; background: #fff; color: var(--accent); }
#advanced-settings > div:last-child, #technical-details > div:last-child { padding: .75rem !important; }
#advanced-settings input, #advanced-settings textarea, #advanced-settings [role="combobox"] { border: 1px solid var(--line) !important; border-radius: 13px !important; background: #fff !important; color: var(--text) !important; box-shadow: none !important; }
#advanced-settings [role="listbox"] { overflow: hidden !important; border: 1px solid var(--line) !important; border-radius: 16px !important; background: #fff !important; box-shadow: 0 14px 28px rgba(31,62,49,.12) !important; }
#advanced-settings [role="option"] { margin: 4px !important; border-radius: 11px !important; color: var(--text) !important; }
#advanced-settings [role="option"]:hover, #advanced-settings [role="option"][aria-selected="true"] { background: var(--accent-soft) !important; color: var(--accent-hover) !important; }
.research-note { margin-top: .65rem; color: var(--muted); font-size: .76rem; }

/* Gradio's real progress tracker: compact, solid-color, and non-blocking. */
#result-card .wrap.generating, #input-card .wrap.generating {
    justify-content: flex-end !important;
    border: 0 !important;
    background: transparent !important;
    padding: 1rem !important;
    animation: none !important;
}
#result-card .wrap.generating .progress-text,
#input-card .wrap.generating .progress-text,
#result-card .wrap.generating .meta-text-center,
#input-card .wrap.generating .meta-text-center { display: none !important; }
#result-card .progress-level, #input-card .progress-level {
    position: relative !important;
    align-items: stretch !important;
    width: min(560px, calc(100% - 1rem)) !important;
    margin: 0 auto !important;
    padding: .8rem .9rem .85rem 3.4rem !important;
    border: 1px solid #c7ddd4 !important;
    border-radius: 18px !important;
    background: #f7fbf9 !important;
    box-shadow: 0 12px 26px rgba(31,62,49,.11) !important;
}
#result-card .progress-level::before, #input-card .progress-level::before {
    content: "▤";
    position: absolute;
    left: 1rem;
    top: 50%;
    width: 1.65rem;
    height: 1.9rem;
    transform: translateY(-50%);
    border: 1px solid #afd3c5;
    border-radius: 7px;
    background: var(--accent-soft);
    color: var(--accent);
    font-size: 1.05rem;
    line-height: 1.85rem;
    text-align: center;
    animation: page-bob 1.4s ease-in-out infinite;
}
#result-card .progress-level-inner, #input-card .progress-level-inner {
    margin: 0 0 .55rem !important;
    color: var(--text) !important;
    font-family: "Cascadia Mono", "Microsoft YaHei UI", monospace !important;
    font-size: .8rem !important;
    font-weight: 600 !important;
    text-align: left !important;
}
#result-card .progress-bar-wrap, #input-card .progress-bar-wrap {
    width: 100% !important;
    height: 9px !important;
    overflow: visible !important;
    border: 0 !important;
    border-radius: 6px !important;
    background: #dcebe5 !important;
}
#result-card .progress-bar, #input-card .progress-bar {
    position: relative;
    min-width: 8px;
    border-radius: 6px !important;
    background: var(--accent) !important;
    transition: width 220ms ease-out !important;
}
#result-card .progress-bar::after, #input-card .progress-bar::after {
    content: "";
    position: absolute;
    top: -3px;
    right: -5px;
    width: 14px;
    height: 14px;
    border: 3px solid #f7fbf9;
    border-radius: 6px;
    background: var(--accent-bright);
    animation: progress-pulse 1s ease-in-out infinite;
}
#input-card .progress-text, #result-card .progress-text,
#input-card .progress-level, #result-card .progress-level { display: none !important; }
#result-markdown .progress-level { display: flex !important; }
@keyframes page-bob { 0%,100% { transform: translateY(-50%); } 50% { transform: translateY(calc(-50% - 4px)); } }
@keyframes progress-pulse { 0%,100% { transform: scale(.8); opacity: .65; } 50% { transform: scale(1); opacity: 1; } }
@keyframes status-breathe { 0%,100% { transform: scale(.85); opacity: .65; } 50% { transform: scale(1); opacity: 1; } }
@keyframes result-arrive { from { opacity: .35; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }

@media (max-width: 820px) {
    .gradio-container { padding: .65rem .65rem 2rem !important; }
    .hero { padding-top: 1.65rem; }
    .hero h1 { font-size: 2.1rem; word-break: keep-all; }
    .mobile-break { display: block; }
    #workspace-row { flex-direction: column; flex-wrap: wrap !important; }
    #input-card, #result-card { min-width: 0 !important; width: 100% !important; border-radius: 21px !important; }
    #action-row { flex-direction: column !important; flex-wrap: wrap !important; }
    #mode-choice > .wrap:not(.default) { grid-template-columns: 1fr; }
    #primary-action, #clear-action { width: 100% !important; flex: none !important; }
    #result-card { min-height: 520px; }
    #result-markdown { min-height: 330px !important; }
    .result-empty { min-height: 286px; }
    .result-heading { align-items: center; }
    .live-badge { padding: .38rem .52rem; font-size: .7rem; }
}
@media (prefers-color-scheme: dark) {
    :root { --canvas:#111814; --surface:#18221d; --surface-solid:#1d2923; --text:#f1f7f3; --muted:#a9b8b0; --line:#34473d; --accent:#58b99f; --accent-hover:#70cbb2; --accent-soft:#233d34; --accent-bright:#8bd7c2; --focus:rgba(88,185,159,.35); --warning:#ef9a82; --warning-soft:#3d2924; }
    #clear-action { background: rgba(118,118,128,.18) !important; }
    #advanced-settings input, #advanced-settings textarea, #advanced-settings [role="combobox"], #advanced-settings [role="listbox"] { background: #1d2923 !important; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; animation: none !important; }
    .scan-beam { transform: translateY(20px); }
    .live-track i { width: 100%; }
}
@media (prefers-reduced-transparency: reduce) { #input-card, #result-card { backdrop-filter: none; -webkit-backdrop-filter: none; background: var(--surface-solid) !important; } }
@media (prefers-contrast: more) { #input-card, #result-card, #input-upload, #result-tabs { border-width: 2px !important; } }
"""


def build_ui(app_callbacks: Dict[str, Any]) -> gr.Blocks:
    """Build a common-task-first UI; research controls stay one level deeper."""
    theme = gr.themes.Soft(
        primary_hue="green",
        secondary_hue="slate",
        radius_size="lg",
        font=["-apple-system", "BlinkMacSystemFont", "Segoe UI", "Microsoft YaHei UI", "sans-serif"],
    )
    logo_uri = _logo_data_uri("logo-128.png")
    logo_img = (
        f'<img class="brand-logo" src="{logo_uri}" alt="Mini UFlash" width="36" height="36" />'
        if logo_uri
        else ""
    )
    hero_logo = (
        f'<img class="hero-logo" src="{logo_uri}" alt="Mini UFlash logo" width="72" height="72" />'
        if logo_uri
        else ""
    )
    with gr.Blocks(title="Mini UFlash OCR", css=CSS, theme=theme) as blocks:
        gr.HTML(
            '<header id="product-header">'
            f'<div class="topbar"><div class="brand">{logo_img}'
            '<div class="wordmark">Mini UFlash</div></div>'
            '<div class="privacy-note"><span class="privacy-dot"></span>仅在本机处理</div></div>'
            f'<div class="hero"><div class="hero-logo-wrap">{hero_logo}'
            '<div><p class="eyebrow">本地文档识别</p>'
            '<h1>从页面到文字，<br class="mobile-break">一步完成。</h1></div></div>'
            '<p>上传图片或 PDF，选择稳定版或加速实验版，结果会直接显示在右侧。</p></div>'
            '</header>'
        )
        status_html = gr.HTML(value=_initial_status())

        with gr.Row(elem_id="workspace-row"):
            with gr.Column(scale=4, min_width=0, elem_id="input-card"):
                gr.HTML('<div class="section-heading"><h2>选择文档</h2><p>支持常见图片格式与 PDF</p></div>')
                input_file = gr.File(
                    label="图片或 PDF",
                    file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".pdf"],
                    type="filepath",
                    elem_id="input-upload",
                )
                input_preview = gr.Image(
                    label="原件预览",
                    height=250,
                    interactive=False,
                    visible=False,
                    elem_id="input-preview",
                )
                gr.HTML(
                    '<div class="field-heading"><h3>处理方式</h3>'
                    '<p>普通版走官方 Unlimited-OCR；加速版为稳定 DFlash（验证前缀 + 周期重同步）</p></div>'
                )
                mode_radio = gr.Radio(
                    choices=[
                        ("普通版 · 稳定", "稳定模式"),
                        ("加速版 · 稳定 DFlash", "Mini UFlash 精确模式"),
                    ],
                    value="稳定模式",
                    label="处理方式",
                    show_label=False,
                    elem_id="mode-choice",
                )
                tier_dropdown = gr.Dropdown(
                    choices=[
                        ("快速 · 墙钟优先（可软截断）", "fast"),
                        ("均衡 · 更完整一点", "balanced"),
                        ("无损 · 不软截断（未必更快）", "lossless"),
                    ],
                    value="balanced",
                    label="加速档位",
                    visible=False,
                    elem_id="tier-choice",
                )
                gr.HTML(
                    '<div class="experiment-note">'
                    "加速版使用目标模型验证的草稿前缀；默认「均衡」减少截断。"
                    "「停止」会结束当前任务并保留已识别页，可立即导出。"
                    "</div>"
                )
                with gr.Row(elem_id="action-row"):
                    btn_run = gr.Button(
                        "开始识别", variant="primary", scale=3, min_width=0, elem_id="primary-action"
                    )
                    btn_pause = gr.Button(
                        "暂停", scale=1, min_width=0, interactive=False, elem_id="pause-action"
                    )
                    btn_stop = gr.Button(
                        "停止", scale=1, min_width=0, interactive=False, elem_id="stop-action"
                    )
                    btn_clear = gr.Button("清空", scale=1, min_width=0, elem_id="clear-action")
                progress_visual = gr.HTML(value=_progress_markup(False), elem_id="recognition-progress")
                run_status = gr.Markdown("选择文件后即可开始。", elem_id="run-status")

                with gr.Accordion("高级设置", open=False, elem_id="advanced-settings"):
                    preset_dropdown = gr.Dropdown(
                        choices=[("复杂排版（推荐）", "gundam"), ("简单页面", "base")],
                        value="gundam",
                        label="识别版式",
                    )
                    max_tokens_slider = gr.Slider(
                        256,
                        4096,
                        value=1536,
                        step=128,
                        label="最长输出（8GB 建议 ≤1536，过大易 OOM）",
                    )

                    probe_tokens_slider = gr.Slider(
                        32, 512, value=256, step=16, label="研究探针长度", visible=False
                    )
                    model_path_input = gr.Textbox(
                        label="本地模型位置",
                        value=str(config.unlimited_ocr_path()),
                        interactive=True,
                    )
                    with gr.Row():
                        btn_load = gr.Button("提前加载模型", size="sm")
                        btn_unload = gr.Button("释放显存", size="sm")
                    load_status = gr.Markdown("")
                    gr.HTML('<div class="research-note">PDF 会逐页保存；实验失败的页面会自动改用普通版。</div>')

            with gr.Column(scale=8, min_width=0, elem_id="result-card"):
                gr.HTML(
                    '<div class="section-heading result-heading"><div><h2>文字结果</h2>'
                    '<p>识别完一页，就在这里追加一页</p></div>'
                    '<span class="live-badge">实时追加</span></div>'
                )
                with gr.Tabs(elem_id="result-tabs"):
                    with gr.Tab("排版视图"):
                        out_md = gr.Markdown(value=_empty_result_markup(), elem_id="result-markdown")
                    with gr.Tab("纯文本"):
                        out_txt = gr.Textbox(
                            lines=17, value="", show_copy_button=True,
                            show_label=False, elem_id="plain-result",
                        )

                gr.HTML(
                    '<div class="export-label">导出'
                    '<span style="font-weight:500;color:var(--muted);margin-left:.4rem">'
                    '识别中也可下载已完成部分</span></div>'
                )
                with gr.Row():
                    dl_md = gr.DownloadButton(
                        "Markdown", value=None, size="sm", elem_classes="download-button"
                    )
                    dl_txt = gr.DownloadButton(
                        "纯文本", value=None, size="sm", elem_classes="download-button"
                    )
                    dl_json = gr.DownloadButton(
                        "完整数据", value=None, size="sm", elem_classes="download-button"
                    )
                    dl_metrics = gr.DownloadButton(
                        "运行指标", value=None, size="sm", elem_classes="download-button"
                    )

                with gr.Accordion("技术详情", open=False, elem_id="technical-details"):
                    with gr.Tabs():
                        with gr.Tab("原始输出"):
                            out_raw = gr.Textbox(lines=12, value="", show_label=False)
                        with gr.Tab("Mini UFlash 指标"):
                            out_metrics = gr.Markdown(value="")
                        with gr.Tab("运行日志"):
                            out_log = gr.Textbox(lines=12, value="", show_label=False)

        input_file.change(
            fn=app_callbacks.get("on_file_change", lambda x: x),
            inputs=[input_file],
            outputs=[input_preview, run_status],
        )
        mode_radio.change(
            fn=app_callbacks.get("on_mode_change", _toggle_mode_controls),
            inputs=[mode_radio],
            outputs=[probe_tokens_slider, tier_dropdown],
        )
        btn_load.click(
            fn=app_callbacks.get("on_load_model", lambda p, s: (p, s)),
            inputs=[model_path_input, load_status],
            outputs=[load_status, status_html, btn_load],
            concurrency_limit=1,
            trigger_mode="once",
        )
        btn_unload.click(
            fn=app_callbacks.get("on_unload_model", lambda: ""),
            outputs=[load_status, status_html],
        )
        btn_pause.click(
            fn=app_callbacks.get("on_toggle_pause", lambda: (gr.update(), _progress_markup(True))),
            outputs=[btn_pause, progress_visual],
            queue=False,
        )
        btn_stop.click(
            fn=app_callbacks.get(
                "on_stop_job",
                lambda: (
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    _progress_markup(False),
                    "⏹ 停止请求已发送",
                ),
            ),
            outputs=[btn_pause, btn_stop, progress_visual, run_status],
            queue=False,
        )
        start_event = btn_run.click(
            fn=_show_progress,
            outputs=[progress_visual, btn_pause, btn_stop],
            queue=False,
        )
        run_event = start_event.then(
            fn=app_callbacks.get("on_run", lambda *a: ""),
            inputs=[
                input_file, model_path_input, mode_radio, preset_dropdown,
                max_tokens_slider, probe_tokens_slider, tier_dropdown,
            ],
            outputs=[
                out_md, out_txt, out_raw, out_metrics, out_log, run_status,
                dl_md, dl_txt, dl_json, dl_metrics, status_html,
            ],
            concurrency_limit=1,
            show_progress="full",
        )
        run_event.then(
            fn=_hide_progress,
            outputs=[progress_visual, btn_pause, btn_stop],
            queue=False,
        )
        clear_fn = app_callbacks.get("on_clear") or _clear_all
        btn_clear.click(
            fn=clear_fn,
            outputs=[
                input_file, input_preview, out_md, out_txt, out_raw,
                out_metrics, out_log, run_status, load_status,
                dl_md, dl_txt, dl_json, dl_metrics,
            ],
        )

    return blocks


def _status_markup(report: config.EnvReport) -> str:
    model = "模型已就绪" if "已加载" in report.unlimited_ocr else "模型将在首次识别时加载"
    return f'<div class="status-bar"><strong>本机处理</strong> · {escape(model)}</div>'


def _initial_status() -> str:
    return _status_markup(config.build_static_env_report())


def _toggle_probe_visibility(mode: str):
    # Direct Block now decodes the complete page; the old bounded probe length
    # remains as an internal compatibility input and should not clutter the UI.
    return gr.update(visible=False)


def _toggle_mode_controls(mode: str):
    """Show accel tier dropdown only for Mini UFlash / Stable DFlash path."""
    is_accel = mode == "Mini UFlash 精确模式"
    return gr.update(visible=False), gr.update(visible=is_accel)


def _clear_all():
    return (
        None, None, _empty_result_markup(), "", "", "", "",
        "选择文件后即可开始。", "",
        None, None, None, None,
    )


def _empty_result_markup() -> str:
    return (
        '<div class="result-empty"><div>'
        '<svg viewBox="0 0 44 44" fill="none" aria-hidden="true">'
        '<rect x="8" y="5" width="28" height="34" rx="5" stroke="currentColor" stroke-width="1.8"/>'
        '<path d="M14 15h16M14 21h12M14 27h16" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>'
        '</svg><strong>文字会显示在这里</strong>上传文档并开始识别</div></div>'
    )


def _progress_markup(running: bool, paused: bool = False) -> str:
    state = "paused" if paused else ("running" if running else "idle")
    title = "识别已暂停" if paused else "正在识别文档"
    detail = "点击“继续”恢复，已完成的文字不会丢失" if paused else "扫描、解码并逐页保存"
    return (
        f'<div class="ocr-progress" data-state="{state}" role="status" aria-live="polite">'
        '<div class="scan-page" aria-hidden="true"><span></span><span></span><span></span><i class="scan-beam"></i></div>'
        f'<div class="progress-copy"><strong>{title}</strong>'
        f'<span>{detail}</span><div class="live-track"><i></i></div></div></div>'
    )


def _show_progress():
    return (
        _progress_markup(True),
        gr.update(value="暂停", interactive=True),
        gr.update(interactive=True),
    )


def _hide_progress():
    return (
        _progress_markup(False),
        gr.update(value="暂停", interactive=False),
        gr.update(interactive=False),
    )


def render_status_html(report: config.EnvReport) -> str:
    return _status_markup(report)
