"""Windows-native real inference smoke test for the local models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from webapp import config
from webapp.mini_uflash_engine import run_precise_mode, run_direct_mode
from webapp.model_manager import MODEL_MANAGER
from webapp.unlimited_ocr_engine import assemble_result, recognize_image


def make_test_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1280, 520), "white")
    draw = ImageDraw.Draw(image)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = ImageFont.truetype(str(font_path), 42) if font_path.is_file() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 30) if font_path.is_file() else font
    draw.text((60, 55), "Mini UFlash OCR 真实测试", fill="#111827", font=font)
    draw.text((60, 145), "Windows 原生推理 · Unlimited-OCR", fill="#334155", font=small)
    draw.text((60, 220), "Invoice No: MUF-2026-0716", fill="#111827", font=small)
    draw.text((60, 290), "金额 Amount: ¥ 1,280.50", fill="#111827", font=small)
    draw.text((60, 360), "准确、可验证、无静态演示数据。", fill="#111827", font=small)
    image.save(path)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--precise", action="store_true")
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--probe-tokens", type=int, default=32)
    args = parser.parse_args()

    report_dir = config.PROJECT_ROOT / "reports" / "real_inference"
    image_path = report_dir / "bilingual_smoke.png"
    make_test_image(image_path)
    print(f"IMAGE={image_path}", flush=True)

    started = time.perf_counter()
    handles = MODEL_MANAGER.load(config.unlimited_ocr_path())
    print(f"MODEL_LOADED_SECONDS={time.perf_counter() - started:.3f}", flush=True)
    print(f"ATTENTION={handles.attention_backend}", flush=True)
    print(f"DTYPE={handles.dtype_name}", flush=True)
    print(f"VRAM_GB={handles.vram_gb:.3f}", flush=True)

    page = recognize_image(
        handles,
        image_path,
        report_dir / "stable_scratch",
        preset_name="base",
        max_length=args.max_length,
    )
    stable = assemble_result([page], page.elapsed_seconds)
    stable_path = report_dir / "stable_result.json"
    stable_path.write_text(json.dumps(stable.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"STABLE_ERROR={page.error}", flush=True)
    print(f"STABLE_SECONDS={page.elapsed_seconds:.3f}", flush=True)
    print(f"STABLE_TOKENS={page.generated_tokens}", flush=True)
    print("STABLE_MARKDOWN_BEGIN", flush=True)
    print(stable.markdown, flush=True)
    print("STABLE_MARKDOWN_END", flush=True)

    if args.precise and page.error is None:
        precise = run_precise_mode(
            handles,
            image_path,
            preset_name="base",
            max_length=args.max_length,
            max_new_tokens=args.probe_tokens,
        )
        precise_path = report_dir / "precise_metrics.json"
        precise_path.write_text(json.dumps(precise.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"PRECISE_OK={precise.probe_ok}", flush=True)
        print(f"PRECISE_STEP={precise.checkpoint_step}", flush=True)
        print(f"PRECISE_FIRST_MISMATCH={precise.first_mismatch}", flush=True)
        print(f"PRECISE_MEAN_ACCEPTED={precise.mean_accepted_draft:.4f}", flush=True)

    if args.direct and page.error is None:
        direct = run_direct_mode(
            handles,
            image_path,
            preset_name="base",
            max_length=args.max_length,
        )
        direct_path = report_dir / "direct_metrics.json"
        direct_path.write_text(
            json.dumps(direct.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"DIRECT_SECONDS={direct.total_seconds:.3f}", flush=True)
        print(f"DIRECT_TOKENS={direct.generated_tokens}", flush=True)
        print(f"DIRECT_PREFIX_COMMITS={direct.direct_block_commits}", flush=True)
        print(f"DIRECT_FULL_B8_COMMITS={direct.full_block_commits}", flush=True)
        print(f"DIRECT_FALLBACK_ROUNDS={direct.fallback_rounds}", flush=True)
        print(f"DIRECT_FORWARD_REDUCTION={direct.target_forward_reduction:.3f}", flush=True)
        print("DIRECT_MARKDOWN_BEGIN", flush=True)
        print(direct.markdown, flush=True)
        print("DIRECT_MARKDOWN_END", flush=True)

    MODEL_MANAGER.unload()
    return 0 if page.error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
