"""Unlimited-OCR stable inference engine.

This is the default, trustworthy OCR path. It drives the model's official
``.infer()`` method with the same parameters already proven out by the existing
``batch_ocr.py`` script, so the output is byte-for-byte the official
Unlimited-OCR result.

Two presets:

* **Gundam / Crop** — ``crop_mode=True``, ``base_size=1024``,
  ``image_size=640``. Best for complex layouts, multi-column and long
  documents. (This is the preset used by the reference batch script.)
* **Base** — a single global view at ``image_size``. Good for clean
  single-page recognition.
"""

from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config
from .model_manager import ModelHandles

# Pattern matching the <image>document parsing.</image> prompt the reference
# script uses, and the Unlimited-OCR README's recommended document-parsing task.
DEFAULT_PROMPT = "<image>document parsing."

# Official decode leaves <｜end▁of▁sentence｜> behind; infer() strips it in
# eval_mode/save_results. We also drop residual detection tags here.
_DET_TAG_RE = re.compile(r"<\|det\|>[^<]*<\|/det\|>")
_EOS_RE = re.compile(r"<｜end▁of▁sentence｜>")


def clean_markdown(text: str) -> str:
    """Strip residual detection tags / EOS markers from a raw OCR result."""
    if not text:
        return ""
    text = _DET_TAG_RE.sub("", text)
    text = _EOS_RE.sub("", text)
    return text.strip()


@dataclass
class StablePageResult:
    page_index: int
    raw_markdown: str
    markdown: str
    elapsed_seconds: float
    generated_tokens: int = 0
    error: Optional[str] = None


@dataclass
class StableResult:
    pages: list[StablePageResult] = field(default_factory=list)
    markdown: str = ""
    plain_text: str = ""
    raw_output: str = ""
    elapsed_seconds: float = 0.0
    generated_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "mode": "stable",
            "pages": [p.__dict__ for p in self.pages],
            "markdown": self.markdown,
            "plain_text": self.plain_text,
            "raw_output": self.raw_output,
            "elapsed_seconds": self.elapsed_seconds,
            "generated_tokens": self.generated_tokens,
        }


@dataclass(frozen=True)
class Preset:
    name: str
    crop_mode: bool
    base_size: int
    image_size: int


def presets() -> dict[str, Preset]:
    inf = config.INFERENCE
    return {
        "gundam": Preset(
            name="Gundam / Crop",
            crop_mode=True,
            base_size=inf.crop_base_size,
            image_size=inf.crop_image_size,
        ),
        "base": Preset(
            name="Base",
            crop_mode=False,
            base_size=inf.crop_base_size,
            image_size=inf.base_image_size,
        ),
    }


def _run_infer(
    handles: ModelHandles,
    image_path: Path,
    scratch_dir: Path,
    preset: Preset,
    max_length: int,
) -> tuple[str, int]:
    """Call the official ``.infer()`` with ``save_results`` and read result.md.

    Returns ``(markdown, generated_tokens)``. Mirrors the proven path from
    ``batch_ocr.py``.
    """
    inf = config.INFERENCE
    # Start from a clean scratch dir so stale results never leak in.
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir, ignore_errors=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    handles.model.infer(
        handles.tokenizer,
        prompt=DEFAULT_PROMPT,
        image_file=str(image_path),
        output_path=str(scratch_dir),
        base_size=preset.base_size,
        image_size=preset.image_size,
        crop_mode=preset.crop_mode,
        max_length=max_length,
        no_repeat_ngram_size=inf.no_repeat_ngram_size,
        ngram_window=inf.ngram_window,
        save_results=True,
    )

    result_md = scratch_dir / "result.md"
    markdown = ""
    if result_md.is_file():
        markdown = result_md.read_text(encoding="utf-8")

    # Best-effort token estimate from the decoded text length.
    generated_tokens = max(0, len(handles.tokenizer.encode(markdown)) - 1) if markdown else 0
    return markdown, generated_tokens


def recognize_image(
    handles: ModelHandles,
    image_path: Path,
    scratch_dir: Path,
    preset_name: str = "gundam",
    max_length: int = config.INFERENCE.max_length,
    page_index: int = 0,
) -> StablePageResult:
    """Recognize a single image and return a :class:`StablePageResult`."""
    preset = presets().get(preset_name, presets()["gundam"])
    started = time.perf_counter()
    try:
        raw, tokens = _run_infer(handles, image_path, scratch_dir, preset, max_length)
        cleaned = clean_markdown(raw)
        return StablePageResult(
            page_index=page_index,
            raw_markdown=raw,
            markdown=cleaned,
            elapsed_seconds=time.perf_counter() - started,
            generated_tokens=tokens,
        )
    except Exception as exc:  # noqa: BLE001 - per-page resilience
        return StablePageResult(
            page_index=page_index,
            raw_markdown="",
            markdown="",
            elapsed_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def assemble_result(pages: list[StablePageResult], elapsed: float) -> StableResult:
    """Combine per-page results into a single :class:`StableResult`."""
    md_parts: list[str] = []
    raw_parts: list[str] = []
    total_tokens = 0
    for page in pages:
        if page.error:
            md_parts.append(f"**[第 {page.page_index + 1} 页 — 失败]** {page.error}\n")
            continue
        header = f"**[第 {page.page_index + 1} 页]**\n\n" if len(pages) > 1 else ""
        md_parts.append(f"{header}{page.markdown}\n")
        raw_parts.append(page.raw_markdown)
        total_tokens += page.generated_tokens

    markdown = "\n---\n\n".join(p for p in md_parts if p.strip()).strip()
    if len(pages) > 1:
        markdown = f"> 共 {len(pages)} 页\n\n{markdown}"
    plain_text = _markdown_to_plain(markdown)
    return StableResult(
        pages=pages,
        markdown=markdown,
        plain_text=plain_text,
        raw_output="\n\n---\n\n".join(raw_parts),
        elapsed_seconds=elapsed,
        generated_tokens=total_tokens,
    )


def _markdown_to_plain(text: str) -> str:
    """A light Markdown → plain-text conversion for the “纯文本” tab."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # links
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)  # headings
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # bold
    text = re.sub(r"\*([^*]+)\*", r"\1", text)  # italic
    text = re.sub(r"`([^`]+)`", r"\1", text)  # code
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)  # quotes
    return text.strip()
