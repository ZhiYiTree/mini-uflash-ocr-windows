"""Live oracle payload capture for the Mini UFlash exactness probe.

The Mini UFlash B8 exactness probe (``stage9_v2_b8_exactness_probe.py``) operates
on pre-captured "oracle payloads" — tensors that contain the official
Unlimited-OCR generation trajectory (input_ids, output_ids, images_seq_mask,
images_crop/ori/spatial_crop, generation policy).

The research scripts built these offline from pre-captured runs on a training
machine. This module reproduces that capture **live from any uploaded image** by
wrapping ``model.generate`` during a single official ``model.infer`` call — the
exact technique used by ``build_v2_teachers_from_images.py``.

Flow:
    1. Monkey-patch ``model.generate`` with a wrapper that captures all tensors.
    2. Call ``model.infer(tokenizer, ...)`` with ``eval_mode=True``.
    3. The wrapper records ``input_ids``, ``output_ids``, image tensors, and
       the generation config (no_repeat_ngram_size, ngram_window).
    4. Restore ``model.generate`` and return the payload dict.

The payload is then fed into the OnlineTarget / run_exactness_probe pipeline
in ``mini_uflash_engine.py``.
"""

from __future__ import annotations

import copy
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from . import config
from .unlimited_ocr_engine import DEFAULT_PROMPT


@dataclass(frozen=True)
class CaptureResult:
    """The output of a live payload capture.

    ``payload`` contains all tensors needed by the exactness probe.
    ``decoded_text`` is the official OCR text returned by ``.infer()``.
    ``scratch_dir`` is the temp directory used by ``.infer()``; the caller
    should clean it up after the probe finishes.
    """

    payload: Dict[str, Any]
    decoded_text: str
    scratch_dir: Path
    generated_length: int


@dataclass(frozen=True)
class PreparedInput:
    """Preprocessed model inputs captured without running official decoding."""

    payload: Dict[str, Any]
    scratch_dir: Path


def capture_payload(
    model: Any,
    tokenizer: Any,
    image_path: Path,
    *,
    prompt: str = DEFAULT_PROMPT,
    preset_name: str = "gundam",
    max_length: int = config.INFERENCE.max_length,
) -> CaptureResult:
    """Run one official ``.infer()`` while capturing the oracle payload.

    Returns a :class:`CaptureResult` whose ``payload`` is compatible with
    ``payload_tensors()`` and ``generation_policy()`` from ``stage9_common``.
    """
    preset = _resolve_preset(preset_name)
    inf = config.INFERENCE

    scratch = config.WEBAPP_DIR / "_probe_scratch"
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)

    # --- Monkey-patch model.generate to capture tensors during infer(). ---
    generation_capture: Dict[str, Any] = {}
    original_generate = model.generate

    def _ensure_batch(x: Any) -> Any:
        if hasattr(x, "unsqueeze") and x.ndim == 1:
            return x.unsqueeze(0)
        return x

    def _cpu_tensor(x: Any) -> Any:
        if hasattr(x, "detach"):
            return x.detach().cpu().contiguous()
        import torch
        return torch.as_tensor(x).cpu().contiguous()

    def capturing_generate(*gen_args, **gen_kwargs):
        output_ids = original_generate(*gen_args, **gen_kwargs)
        generation_capture["input_ids"] = _cpu_tensor(gen_kwargs["input_ids"])
        generation_capture["output_ids"] = _cpu_tensor(output_ids)
        generation_capture["images_seq_mask"] = _cpu_tensor(
            gen_kwargs["images_seq_mask"]
        ).bool()
        generation_capture["images_crop"] = _cpu_tensor(gen_kwargs["images"][0][0])
        generation_capture["images_ori"] = _cpu_tensor(gen_kwargs["images"][0][1])
        generation_capture["images_spatial_crop"] = _cpu_tensor(
            gen_kwargs["images_spatial_crop"]
        ).long()
        return output_ids

    model.generate = capturing_generate
    try:
        decoded = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=str(image_path),
            output_path=str(scratch),
            base_size=preset["base_size"],
            image_size=preset["image_size"],
            crop_mode=preset["crop_mode"],
            save_results=False,
            eval_mode=True,
            max_length=max_length,
            no_repeat_ngram_size=inf.no_repeat_ngram_size,
            ngram_window=inf.ngram_window,
            temperature=0.0,
        )
    finally:
        model.generate = original_generate

    if not generation_capture:
        raise RuntimeError(
            "model.infer() 没有调用 model.generate()，无法捕获 payload。"
        )

    input_ids = generation_capture["input_ids"].long()
    output_ids = generation_capture["output_ids"].long()
    prompt_length = int(input_ids.shape[1])
    generated_length = int(output_ids.shape[1] - prompt_length)

    if generated_length < 9:
        raise RuntimeError(
            f"仅生成了 {generated_length} 个 token；B8 至少需要 9 个。"
            "请尝试 max_length 更大的值或更复杂的图片。"
        )

    payload = {
        "format": "mini_uflash_v1_oracle_payload",
        "page_id": image_path.stem,
        "image_path": str(image_path),
        "prompt": prompt,
        "input_ids": input_ids,
        "output_ids": output_ids,
        "prompt_length": prompt_length,
        "images_seq_mask": generation_capture["images_seq_mask"],
        "images_crop": generation_capture["images_crop"],
        "images_ori": generation_capture["images_ori"],
        "images_spatial_crop": generation_capture["images_spatial_crop"],
        "decoded_text": decoded,
        "generation": {
            "max_length": max_length,
            "no_repeat_ngram_size": inf.no_repeat_ngram_size,
            "ngram_window": inf.ngram_window,
            "temperature": 0.0,
        },
    }

    return CaptureResult(
        payload=payload,
        decoded_text=decoded,
        scratch_dir=scratch,
        generated_length=generated_length,
    )


def prepare_input_payload(
    model: Any,
    tokenizer: Any,
    image_path: Path,
    *,
    prompt: str = DEFAULT_PROMPT,
    preset_name: str = "gundam",
    max_length: int = config.INFERENCE.max_length,
) -> PreparedInput:
    """Capture Unlimited-OCR preprocessing tensors without AR generation.

    ``model.infer`` remains the source of truth for image resizing, crop layout,
    prompt construction and masks. Its ``generate`` call is temporarily replaced
    by a capture-only function that returns the prompt unchanged. The Direct
    Block decoder can therefore start from exactly the official model inputs
    without first paying for a complete official OCR pass.
    """
    preset = _resolve_preset(preset_name)
    inf = config.INFERENCE
    scratch = config.WEBAPP_DIR / "_direct_scratch"
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)

    captured: Dict[str, Any] = {}
    original_generate = model.generate

    def _cpu_tensor(value: Any) -> Any:
        if hasattr(value, "detach"):
            return value.detach().cpu().contiguous()
        import torch
        return torch.as_tensor(value).cpu().contiguous()

    def capture_only_generate(*gen_args, **gen_kwargs):
        input_ids = gen_kwargs["input_ids"]
        captured["input_ids"] = _cpu_tensor(input_ids)
        captured["images_seq_mask"] = _cpu_tensor(
            gen_kwargs["images_seq_mask"]
        ).bool()
        captured["images_crop"] = _cpu_tensor(gen_kwargs["images"][0][0])
        captured["images_ori"] = _cpu_tensor(gen_kwargs["images"][0][1])
        captured["images_spatial_crop"] = _cpu_tensor(
            gen_kwargs["images_spatial_crop"]
        ).long()
        # ``infer(eval_mode=True)`` decodes only the suffix. Returning the prompt
        # unchanged makes that suffix empty and avoids any target decoding.
        return input_ids

    model.generate = capture_only_generate
    try:
        model.infer(
            tokenizer,
            prompt=prompt,
            image_file=str(image_path),
            output_path=str(scratch),
            base_size=preset["base_size"],
            image_size=preset["image_size"],
            crop_mode=preset["crop_mode"],
            save_results=False,
            eval_mode=True,
            max_length=max_length,
            no_repeat_ngram_size=inf.no_repeat_ngram_size,
            ngram_window=inf.ngram_window,
            temperature=0.0,
        )
    finally:
        model.generate = original_generate

    if not captured:
        raise RuntimeError("Unlimited-OCR 预处理未提供 generate 输入。")

    input_ids = captured["input_ids"].long()
    payload = {
        "format": "mini_uflash_direct_input_v1",
        "page_id": image_path.stem,
        "image_path": str(image_path),
        "prompt": prompt,
        "input_ids": input_ids,
        "output_ids": input_ids.clone(),
        "prompt_length": int(input_ids.shape[1]),
        "images_seq_mask": captured["images_seq_mask"],
        "images_crop": captured["images_crop"],
        "images_ori": captured["images_ori"],
        "images_spatial_crop": captured["images_spatial_crop"],
        "generation": {
            "max_length": max_length,
            "no_repeat_ngram_size": inf.no_repeat_ngram_size,
            "ngram_window": inf.ngram_window,
            "temperature": 0.0,
        },
    }
    return PreparedInput(payload=payload, scratch_dir=scratch)


def _resolve_preset(name: str) -> Dict[str, Any]:
    """Map preset name to the parameters used by .infer()."""
    inf = config.INFERENCE
    if name == "base":
        return {"crop_mode": False, "base_size": inf.crop_base_size,
                "image_size": inf.base_image_size}
    # Default: gundam
    return {"crop_mode": True, "base_size": inf.crop_base_size,
            "image_size": inf.crop_image_size}
