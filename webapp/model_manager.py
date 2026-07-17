"""Thread-safe model manager for the Mini UFlash OCR webapp.

Loads the Unlimited-OCR target model and (optionally) the Mini UFlash Stage 11B
drafter as singletons shared by stable mode and precise mode. Loading is lazy:
nothing touches the GPU until the user clicks "加载模型", and unloading frees
the VRAM with ``gc.collect()`` + ``torch.cuda.empty_cache()``.

Windows compatibility notes baked in here:

* Attention backend: the local Unlimited-OCR model only registers ``eager`` and
  ``flash_attention_2`` (no ``sdpa`` class). ``flash_attention_2`` requires the
  flash-attn package, which is Linux-only and therefore never installed here.
  We therefore force ``attn_implementation="eager"`` — a pure-PyTorch matmul /
  softmax path that needs no extra kernels. This is recorded and surfaced in
  the UI. (The spec's SDPA→eager ladder resolves to eager for this model.)
* BF16 is the primary dtype. If the GPU cannot represent BF16 we report it
  loudly rather than silently downgrading.
* ``trust_remote_code=True`` with the local model directory.
* No automatic download of multi-GB weights.
"""

from __future__ import annotations

import gc
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from . import config
from . import path_utils


@dataclass
class ModelHandles:
    """The currently loaded target model + tokenizer + metadata."""

    model: Any
    tokenizer: Any
    model_path: Path
    attention_backend: str
    dtype_name: str
    input_embeddings: Any
    output_embeddings: Any

    @property
    def vram_gb(self) -> float:
        try:
            import torch  # type: ignore

            return torch.cuda.memory_allocated() / 1024**3
        except Exception:
            return 0.0


class ModelManager:
    """A thread-safe singleton wrapper around the two models."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._handles: Optional[ModelHandles] = None
        self._loading = False
        self._model_path_override: Optional[Path] = None
        self._last_error: str = ""

    # -- state accessors ----------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        with self._condition:
            return self._handles is not None

    @property
    def is_loading(self) -> bool:
        with self._lock:
            return self._loading

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    def effective_model_path(self) -> Path:
        """The model path that will be / was used for loading."""
        if self._model_path_override is not None:
            return self._model_path_override
        return config.unlimited_ocr_path()

    def handles(self) -> Optional[ModelHandles]:
        with self._lock:
            return self._handles

    # -- loading / unloading ------------------------------------------------
    def set_model_path(self, path: str | Path) -> Path:
        """Set the model directory to load from (validated, resolved)."""
        resolved = path_utils.normalize(path)
        with self._lock:
            if self._handles is not None:
                raise RuntimeError("请先卸载当前模型再更改模型路径")
            self._model_path_override = resolved
        return resolved

    def load(self, model_path: Optional[str | Path] = None,
             dtype_name: Optional[str] = None) -> ModelHandles:
        """Load the target model + tokenizer. Raises on failure.

        Safe to call from any thread; concurrent callers block on the same load.
        Re-entry returns the existing handles.
        """
        import torch  # type: ignore
        from transformers import AutoTokenizer  # type: ignore

        with self._condition:
            if self._handles is not None:
                return self._handles
            if self._loading:
                while self._loading:
                    self._condition.wait()
                if self._handles is not None:
                    return self._handles
                if self._last_error:
                    raise RuntimeError(self._last_error)

            self._loading = True
            self._last_error = ""
            target_path = (
                path_utils.normalize(model_path)
                if model_path
                else self.effective_model_path()
            )
            if model_path:
                self._model_path_override = target_path

        try:
            self._validate_model_dir(target_path)
            dt_name = (dtype_name or config.INFERENCE.dtype).lower()
            dtype = _resolve_dtype(dt_name, torch)

            tokenizer = AutoTokenizer.from_pretrained(
                str(target_path), trust_remote_code=True, local_files_only=True
            )

            backend, model = _load_target_model(target_path, dtype, torch)

            model = model.eval()
            if torch.cuda.is_available():
                model = model.to("cuda")

            input_emb = _get_input_embeddings(model)
            output_emb = _get_output_embeddings(model)
            for module in (input_emb, output_emb):
                for p in module.parameters():
                    p.requires_grad_(False)

            handles = ModelHandles(
                model=model,
                tokenizer=tokenizer,
                model_path=target_path,
                attention_backend=backend,
                dtype_name=dt_name,
                input_embeddings=input_emb,
                output_embeddings=output_emb,
            )
            with self._lock:
                self._handles = handles
            return handles
        except Exception as exc:  # noqa: BLE001 - surface to UI
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            self._cleanup_gpu()
            raise
        finally:
            with self._condition:
                self._loading = False
                self._condition.notify_all()

    def unload(self) -> None:
        """Drop the loaded model and release VRAM."""
        with self._lock:
            handles = self._handles
            self._handles = None
        if handles is None:
            return
        # Drop references first.
        handles.model = None
        handles.tokenizer = None
        handles.input_embeddings = None
        handles.output_embeddings = None
        try:
            from .mini_uflash_engine import unload_drafter
            unload_drafter()
        except Exception:
            pass
        self._cleanup_gpu()

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _validate_model_dir(path: Path) -> None:
        if not path.is_dir():
            raise FileNotFoundError(
                f"未找到 Unlimited-OCR 模型目录：\n{path}"
            )
        has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
        has_code = (path / "modeling_unlimitedocr.py").is_file() or any(
            path.glob("modeling_*.py")
        )
        if not has_weights or not has_code:
            raise FileNotFoundError(
                f"目录存在但缺少模型权重或远程代码：\n{path}\n"
                "需要 *.safetensors 与 modeling_*.py。"
            )

    @staticmethod
    def _cleanup_gpu() -> None:
        gc.collect()
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _resolve_dtype(name: str, torch: Any) -> Any:
    table = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = name.lower()
    if key not in table:
        raise ValueError(f"不支持的 dtype: {name}")
    if key in ("bfloat16", "bf16") and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        # Ampere (sm_80) and newer fully support BF16. Older cards do not.
        if cap[0] < 8:
            raise RuntimeError(
                "当前 GPU 不支持 BF16（需要 sm_80 / Ampere 或更新）。"
                "请在界面中选择 float16 或 float32，不要静默降级。"
            )
    return table[key]


def _load_target_model(path: Path, dtype: Any, torch: Any) -> tuple[str, Any]:
    """Load the model, forcing a Windows-native attention backend.

    Tries ``sdpa`` first per the spec's preference order, but the local
    Unlimited-OCR model does not register an SDPA class, so transformers
    raises; we then fall back to ``eager`` (pure PyTorch). flash_attention_2 is
    never attempted.
    """
    last_exc: Optional[Exception] = None
    from transformers import AutoModel  # type: ignore

    for backend in ("sdpa", "eager"):
        try:
            model = AutoModel.from_pretrained(  # type: ignore[name-defined]
                str(path),
                trust_remote_code=True,
                local_files_only=True,
                use_safetensors=True,
                torch_dtype=dtype,
                attn_implementation=backend,
            )
            return backend, model
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
    raise RuntimeError(
        "无法以 sdpa/eager 后端加载模型（均失败）。最后错误：\n"
        f"{last_exc}"
    )


def _get_input_embeddings(model: Any) -> Any:
    emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if emb is None:
        for attr in ("embed_tokens",):
            emb = getattr(model, attr, None)
            if emb is not None:
                break
    if emb is None:
        raise RuntimeError("无法定位模型的 input embedding 层")
    return emb


def _get_output_embeddings(model: Any) -> Any:
    emb = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if emb is None:
        emb = getattr(model, "lm_head", None)
    if emb is None:
        raise RuntimeError("无法定位模型的 LM head (output embedding)")
    return emb


# Module-level singleton used across the app.
MODEL_MANAGER = ModelManager()
