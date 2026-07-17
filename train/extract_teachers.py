#!/usr/bin/env python3
"""Extract Mini UFlash V2 teacher features from page images (Windows 8GB).

One page at a time: official Unlimited-OCR infer → oracle payload → five-layer
teacher.pt. Skips existing outputs unless --overwrite. Does not start training.

Usage (from project root)::

    .\\.venv\\Scripts\\python.exe train\\extract_teachers.py --image-dir train\\data\\pages\\pool --limit 10
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from transformers import AutoModel, AutoTokenizer

# Project imports
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train import config as cfg  # noqa: E402
from train.lib.utils import empty_cuda, list_images, safe_stem  # noqa: E402


def choose_layers(num_layers: int, explicit: str | None) -> List[int]:
    if explicit:
        values = [int(x.strip()) for x in explicit.split(",") if x.strip()]
        if len(values) != 5 or len(set(values)) != 5:
            raise ValueError("--layers must contain five distinct decoder layer indices")
        if min(values) < 0 or max(values) >= num_layers:
            raise ValueError(f"Layer indices must be in [0, {num_layers - 1}]")
        return values
    return list(cfg.LAYER_INDICES)


def find_decoder_layers(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    candidates = [
        lambda m: getattr(getattr(m, "model", None), "layers", None),
        lambda m: getattr(getattr(getattr(m, "model", None), "model", None), "layers", None),
        lambda m: getattr(m, "layers", None),
    ]
    for getter in candidates:
        layers = getter(model)
        if layers is not None and hasattr(layers, "__len__") and len(layers) > 0:
            return layers
    raise RuntimeError("Could not locate decoder layers on the target model")


def tensor_from_layer_output(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    if hasattr(value, "last_hidden_state"):
        return value.last_hidden_state
    raise TypeError(f"Unsupported decoder layer output type: {type(value)!r}")


def cpu_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().cpu().contiguous()
    return torch.as_tensor(value).cpu().contiguous()


def load_target_model(model_path: Path, dtype: torch.dtype) -> tuple[str, Any]:
    last_exc: Exception | None = None
    for backend in ("sdpa", "eager"):
        try:
            model = AutoModel.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                local_files_only=True,
                use_safetensors=True,
                torch_dtype=dtype,
                attn_implementation=backend,
            )
            return backend, model
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    raise RuntimeError(f"Failed to load Unlimited-OCR with sdpa/eager: {last_exc}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract Mini UFlash V2 teachers (Windows 8GB)")
    p.add_argument("--image-dir", type=str, default=str(cfg.PAGES_POOL))
    p.add_argument("--payload-dir", type=str, default=str(cfg.PAYLOADS_DIR))
    p.add_argument("--teacher-dir", type=str, default=str(cfg.TEACHERS_TRAIN))
    p.add_argument("--model-path", type=str, default=str(cfg.model_path()))
    p.add_argument("--prompt", type=str, default=cfg.DEFAULT_PROMPT)
    p.add_argument("--layers", type=str, default=",".join(str(x) for x in cfg.LAYER_INDICES))
    p.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default=cfg.EXTRACT_DTYPE)
    p.add_argument(
        "--storage-dtype",
        choices=("float16", "bfloat16", "float32"),
        default=cfg.STORAGE_DTYPE,
    )
    p.add_argument("--base-size", type=int, default=cfg.BASE_SIZE)
    p.add_argument("--image-size", type=int, default=cfg.IMAGE_SIZE)
    p.add_argument("--crop-mode", action=argparse.BooleanOptionalAction, default=cfg.CROP_MODE)
    p.add_argument(
        "--max-length",
        type=int,
        default=cfg.EXTRACT_MAX_LENGTH_SAFE,
        help="Cap generation length to reduce 8GB peak VRAM",
    )
    p.add_argument(
        "--cuda-memory-fraction",
        type=float,
        default=cfg.CUDA_MEMORY_FRACTION,
        help="Cap process VRAM fraction (leave headroom for desktop)",
    )
    p.add_argument("--no-repeat-ngram-size", type=int, default=cfg.NO_REPEAT_NGRAM_SIZE)
    p.add_argument("--ngram-window", type=int, default=cfg.NGRAM_WINDOW)
    p.add_argument("--limit", type=int, default=None, help="Max images to process")
    p.add_argument("--offset", type=int, default=0, help="Skip first N images after sort")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned work only; load nothing on GPU",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dry_run and not torch.cuda.is_available():
        print("ERROR: CUDA is required for teacher extraction", file=sys.stderr)
        return 2

    image_dir = Path(args.image_dir).expanduser().resolve()
    payload_dir = Path(args.payload_dir).expanduser().resolve()
    teacher_dir = Path(args.teacher_dir).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    payload_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(image_dir, recursive=args.recursive, limit=None)
    if args.offset:
        images = images[args.offset :]
    if args.limit is not None:
        images = images[: args.limit]

    print("=" * 72)
    print("Mini UFlash V2 teacher extraction (Windows 8GB)")
    print(f"Images dir : {image_dir}")
    print(f"Images     : {len(images)}")
    print(f"Model      : {model_path}")
    print(f"Payload dir: {payload_dir}")
    print(f"Teacher dir: {teacher_dir}")
    print(f"Prompt     : {args.prompt!r}")
    print(f"Crop       : base={args.base_size} image={args.image_size} crop={args.crop_mode}")
    print(f"Dry-run    : {args.dry_run}")
    print("=" * 72)

    if not images:
        print(
            "No images found under the image dir yet.\n"
            f"  Put PNGs in: {image_dir}\n"
            "  Or run: .\\.venv\\Scripts\\python.exe train\\collect_pages.py"
        )
        return 0 if args.dry_run else 1
    if args.dry_run:
        for i, path in enumerate(images, 1):
            stem = f"{i + args.offset:04d}_{safe_stem(path)}"
            print(f"  [{i}/{len(images)}] {path.name} -> {stem}_v2_teacher.pt")
        print("Dry-run complete. No GPU work performed.")
        return 0

    if not model_path.is_dir():
        print(f"ERROR: model path missing: {model_path}", file=sys.stderr)
        return 1

    dtype = getattr(torch, args.dtype)
    storage_dtype = getattr(torch, args.storage_dtype)

    if 0.1 < float(args.cuda_memory_fraction) < 1.0:
        torch.cuda.set_per_process_memory_fraction(float(args.cuda_memory_fraction))
        print(
            f"CUDA memory fraction capped at {args.cuda_memory_fraction:.2f} "
            f"(leaving headroom for desktop)",
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    backend, model = load_target_model(model_path, dtype)
    model = model.eval().cuda()
    print(f"Attention backend: {backend}")

    decoder_layers = find_decoder_layers(model)
    layer_indices = choose_layers(len(decoder_layers), args.layers)
    print(f"Decoder layers: {len(decoder_layers)} selected={layer_indices}")

    # Hooks are ONLY registered for the teacher-forced feature pass.
    # Leaving them on during generate() wastes VRAM on every decode step.
    captured: Dict[int, torch.Tensor] = {}
    hooks: list = []

    def install_hooks() -> None:
        hooks.clear()
        for index in layer_indices:

            def make_hook(layer_index: int):
                def hook(_module, _inputs, output):
                    # Move to CPU immediately so GPU activations can free.
                    captured[layer_index] = (
                        tensor_from_layer_output(output).detach().to("cpu")
                    )

                return hook

            hooks.append(decoder_layers[index].register_forward_hook(make_hook(index)))

    def remove_hooks() -> None:
        for handle in hooks:
            handle.remove()
        hooks.clear()

    original_generate = model.generate
    generation_capture: Dict[str, Any] = {}

    def capturing_generate(*gen_args, **gen_kwargs):
        output_ids = original_generate(*gen_args, **gen_kwargs)
        generation_capture.clear()
        generation_capture["input_ids"] = cpu_tensor(gen_kwargs["input_ids"])
        generation_capture["output_ids"] = cpu_tensor(output_ids)
        generation_capture["images_seq_mask"] = cpu_tensor(gen_kwargs["images_seq_mask"]).bool()
        generation_capture["images_crop"] = cpu_tensor(gen_kwargs["images"][0][0])
        generation_capture["images_ori"] = cpu_tensor(gen_kwargs["images"][0][1])
        generation_capture["images_spatial_crop"] = cpu_tensor(
            gen_kwargs["images_spatial_crop"]
        ).long()
        return output_ids

    model.generate = capturing_generate
    infer_scratch = teacher_dir / "_infer_scratch"
    infer_scratch.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, Any]] = []
    manifest_path = teacher_dir / "build_manifest.json"
    if manifest_path.is_file() and not args.overwrite:
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(old.get("records"), list):
                manifest = list(old["records"])
        except Exception:
            pass

    succeeded = 0
    failed = 0
    skipped = 0

    try:
        for ordinal, image_path in enumerate(images, 1):
            stem = f"{ordinal + args.offset:04d}_{safe_stem(image_path)}"
            payload_path = payload_dir / f"{stem}_oracle_payload.pt"
            teacher_path = teacher_dir / f"{stem}_v2_teacher.pt"

            if teacher_path.exists() and not args.overwrite:
                print(f"[{ordinal}/{len(images)}] skip existing: {image_path.name}")
                skipped += 1
                succeeded += 1
                continue

            print("-" * 72)
            print(f"[{ordinal}/{len(images)}] {image_path}")
            generation_capture.clear()
            captured.clear()
            started = time.perf_counter()
            used_crop = bool(args.crop_mode)

            try:
                def _run_infer(crop_mode: bool):
                    generation_capture.clear()
                    empty_cuda()
                    return model.infer(
                        tokenizer,
                        prompt=args.prompt,
                        image_file=str(image_path),
                        output_path=str(infer_scratch / stem),
                        base_size=args.base_size,
                        image_size=args.image_size if crop_mode else cfg.BASE_SIZE,
                        crop_mode=crop_mode,
                        save_results=False,
                        eval_mode=True,
                        max_length=args.max_length,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        ngram_window=args.ngram_window,
                        temperature=0.0,
                    )

                try:
                    decoded = _run_infer(used_crop)
                except torch.cuda.OutOfMemoryError:
                    if not used_crop:
                        raise
                    print("  OOM under crop; retrying Base mode (single view)...", flush=True)
                    empty_cuda()
                    used_crop = False
                    decoded = _run_infer(False)

                if not generation_capture:
                    raise RuntimeError("model.infer did not call model.generate")

                input_ids = generation_capture["input_ids"].long()
                output_ids = generation_capture["output_ids"].long()
                prompt_length = int(input_ids.shape[1])
                generated_length = int(output_ids.shape[1] - prompt_length)
                if generated_length < cfg.MIN_GENERATED_TOKENS:
                    raise RuntimeError(
                        f"Only {generated_length} generated tokens; "
                        f"need >= {cfg.MIN_GENERATED_TOKENS} for B8"
                    )

                payload = {
                    "format": "mini_uflash_v1_oracle_payload",
                    "page_id": stem,
                    "image_path": str(image_path),
                    "prompt": args.prompt,
                    "input_ids": input_ids,
                    "output_ids": output_ids,
                    "prompt_length": prompt_length,
                    "images_seq_mask": generation_capture["images_seq_mask"],
                    "images_crop": generation_capture["images_crop"],
                    "images_ori": generation_capture["images_ori"],
                    "images_spatial_crop": generation_capture["images_spatial_crop"],
                    "decoded_text": decoded,
                    "generation": {
                        "max_length": args.max_length,
                        "no_repeat_ngram_size": args.no_repeat_ngram_size,
                        "ngram_window": args.ngram_window,
                        "temperature": 0.0,
                    },
                }
                torch.save(payload, payload_path)

                # Free generate leftovers before the teacher-forced full pass.
                empty_cuda()
                sequence = output_ids.cuda()
                image_mask = generation_capture["images_seq_mask"]
                if image_mask.shape[1] < sequence.shape[1]:
                    pad = torch.zeros(
                        (image_mask.shape[0], sequence.shape[1] - image_mask.shape[1]),
                        dtype=torch.bool,
                    )
                    image_mask = torch.cat([image_mask, pad], dim=1)

                captured.clear()
                install_hooks()
                try:
                    with torch.inference_mode():
                        model(
                            input_ids=sequence,
                            attention_mask=torch.ones_like(sequence, dtype=torch.long),
                            images=[
                                (
                                    generation_capture["images_crop"].cuda(),
                                    generation_capture["images_ori"].cuda(),
                                )
                            ],
                            images_seq_mask=image_mask[:, : sequence.shape[1]].cuda(),
                            images_spatial_crop=generation_capture[
                                "images_spatial_crop"
                            ].cuda(),
                            use_cache=False,
                            return_dict=True,
                        )
                finally:
                    remove_hooks()
                    del sequence
                    empty_cuda()

                missing = [index for index in layer_indices if index not in captured]
                if missing:
                    raise RuntimeError(f"Hooks missed layers: {missing}")

                generated_ids = output_ids[:, prompt_length:]
                context_start = prompt_length - 1
                context_end = context_start + generated_length - 1
                pred_start = prompt_length
                pred_end = pred_start + generated_length - 1

                selected = []
                for index in layer_indices:
                    hidden = captured[index]
                    if hidden.shape[1] < pred_end:
                        raise RuntimeError(
                            f"Layer {index} length {hidden.shape[1]} < required {pred_end}"
                        )
                    selected.append(hidden[:, context_start:context_end, :])

                target_features = torch.stack(selected, dim=2)[0]
                predictive_hidden = captured[layer_indices[-1]][0, pred_start:pred_end, :]
                teacher = {
                    "format": "mini_uflash_v2_teacher",
                    "page_id": stem,
                    "source_payload": str(payload_path),
                    "source_image": str(image_path),
                    "model_path": str(model_path),
                    "layer_indices": layer_indices,
                    "target_features": target_features.to(storage_dtype).cpu().contiguous(),
                    "predictive_hidden": predictive_hidden.to(storage_dtype).cpu().contiguous(),
                    "generated_ids": generated_ids[0].cpu().long().contiguous(),
                    "prompt_length": prompt_length,
                    "alignment": {
                        "target_features[a]": "hidden at clean anchor a",
                        "predictive_hidden[a]": "deep hidden predicting generated_ids[a+1]",
                        "targets_for_anchor_a": "generated_ids[a+1:a+1+draft_len]",
                    },
                }
                torch.save(teacher, teacher_path)

                elapsed = time.perf_counter() - started
                record = {
                    "page_id": stem,
                    "image": str(image_path),
                    "payload": str(payload_path),
                    "teacher": str(teacher_path),
                    "prompt_length": prompt_length,
                    "generated_length": generated_length,
                    "target_features": list(teacher["target_features"].shape),
                    "layer_indices": layer_indices,
                    "crop_mode": used_crop,
                    "seconds": round(elapsed, 2),
                }
                teacher_path.with_suffix(".json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                manifest = [r for r in manifest if r.get("page_id") != stem]
                manifest.append(record)
                succeeded += 1
                print(
                    f"  OK gen_tokens={generated_length} "
                    f"features={record['target_features']} {elapsed:.1f}s"
                )

            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
                if payload_path.exists() and not teacher_path.exists():
                    payload_path.unlink(missing_ok=True)
            finally:
                generation_capture.clear()
                captured.clear()
                empty_cuda()
                # Persist progress after every page for resume safety.
                manifest_path.write_text(
                    json.dumps(
                        {
                            "format": "mini_uflash_v2_teacher_manifest",
                            "requested_images": len(images),
                            "succeeded": succeeded,
                            "failed": failed,
                            "skipped": skipped,
                            "layer_indices": layer_indices,
                            "model_path": str(model_path),
                            "records": manifest,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
    finally:
        model.generate = original_generate
        remove_hooks()
        del model
        empty_cuda()

    print("=" * 72)
    print(f"Done: ok={succeeded} failed={failed} skipped={skipped} total={len(images)}")
    print(f"Manifest: {manifest_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
