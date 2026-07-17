"""Central configuration for the Mini UFlash OCR webapp.

All paths and tunables live here. Nothing else in the package hard-codes a
location, so the app can be relocated by editing this one file (or by setting
the documented environment variables). Paths are resolved with
:mod:`webapp.path_utils` and therefore tolerate spaces, Chinese characters and
non-C: drive letters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import path_utils


def _project_root() -> Path:
    """Return the project root (the parent of this package directory)."""
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT: Path = _project_root()

# --- Models & weights -------------------------------------------------------
# The Unlimited-OCR model is large (~6 GB). We reference it in place rather
# than copying. Override with the UNLIMITED_OCR_PATH env var if it lives
# somewhere other than the in-tree default.
_DEFAULT_UNLIMITED_OCR = PROJECT_ROOT / "models" / "PaddlePaddle" / "Unlimited-OCR"


def unlimited_ocr_path() -> Path:
    return path_utils.normalize(
        os.environ.get("UNLIMITED_OCR_PATH", str(_DEFAULT_UNLIMITED_OCR))
    )


# --- Stage 11B drafter weight auto-discovery --------------------------------
# Priority: Stage 11B best  >  Stage 11A best  >  30k baseline.
# We never use ``drafter_last.pt``. Candidates are searched in several likely
# locations; the first existing match wins. Override with MINI_UFLASH_WEIGHT.
_WEIGHT_CANDIDATES: list[str] = [
    # Local Windows domain-continue (prefer when present)
    "mini-uflash-win-domain-continue-best.pt",
    "weights/mini-uflash-win-domain-continue-best.pt",
    # Stage 11B best (upstream research checkpoint)
    "mini-uflash-v2.0.0-alpha.1-drafter-stage11b-best.pt",
    "drafter-stage11b-best.pt",
    "drafter_v2_stage11b_best.pt",
    "stage11b_prefix_survival/drafter_best.pt",
    "weights/mini-uflash-v2.0.0-alpha.1-drafter-stage11b-best.pt",
    "weights/drafter_v2_stage11b_best.pt",
    # Stage 11A best (fallback)
    "drafter-stage11a-best.pt",
    "drafter_v2_stage11a_best.pt",
    "stage11a_tail_finetune/drafter_best.pt",
    # 30k baseline (last resort)
    "drafter-30k.pt",
    "drafter_v2_30k.pt",
]

# Search roots, in order: explicit override > project weights/ dir > the
# archived preview bundle kept for reference > project root.
_WEIGHT_SEARCH_ROOTS: list[Path] = [
    PROJECT_ROOT / "weights",
    PROJECT_ROOT
    / "archive"
    / "reference-sources"
    / "mini-uflash-preview"
    / "mini-uflash-v2.0.0-alpha.1"
    / "weights",
    PROJECT_ROOT,
]


def discover_weight() -> Optional[Path]:
    """Return the best-available drafter checkpoint, or ``None`` if absent.

    Stable mode still runs without a weight; only Mini UFlash precise mode is
    disabled when this returns ``None``.
    """
    explicit = os.environ.get("MINI_UFLASH_WEIGHT")
    if explicit:
        resolved = path_utils.normalize(explicit)
        if resolved.is_file():
            return resolved

    for root in _WEIGHT_SEARCH_ROOTS:
        for name in _WEIGHT_CANDIDATES:
            candidate = root / name
            if candidate.is_file():
                return candidate.resolve()
    return None


def weight_search_roots() -> list[Path]:
    """Expose the search roots (used by the UI's "where did you look" message)."""
    return list(_WEIGHT_SEARCH_ROOTS)


# --- Runtime directories ----------------------------------------------------
WEBAPP_DIR: Path = PROJECT_ROOT / "webapp"
OUTPUTS_DIR: Path = WEBAPP_DIR / "outputs"
ASSETS_DIR: Path = WEBAPP_DIR / "assets"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
TESTS_DIR: Path = WEBAPP_DIR / "tests"

# --- Network ----------------------------------------------------------------
HOST: str = "127.0.0.1"
PORT: int = 7860


# --- Inference defaults -----------------------------------------------------
# These mirror the values proven out by the existing batch_ocr.py / the
# Unlimited-OCR README and the Mini UFlash teacher-builder defaults.
@dataclass(frozen=True)
class InferenceDefaults:
    # Gundam / Crop (complex layouts, multi-column, long documents)
    crop_base_size: int = 1024
    crop_image_size: int = 640
    crop_mode: bool = True
    # Shared
    max_length: int = 4096
    no_repeat_ngram_size: int = 35
    ngram_window: int = 128
    # Base mode (single global view)
    # The local official README specifies 1024 for the single-view Base path.
    base_image_size: int = 1024
    # Mini UFlash exactness probe
    block_size: int = 8
    layer_indices: tuple[int, ...] = (1, 4, 6, 9, 11)
    probe_max_new_tokens: int = 256  # tokens compared per image
    warmup_cycles: int = 1
    dtype: str = "bfloat16"
    # Theoretical decode speedup ceiling reported by the drafter gate. The UI
    # always labels this as a theoretical estimate, not end-to-end speedup.
    theoretical_speedup_note: str = "理论估算，不是当前端到端实际加速"


INFERENCE = InferenceDefaults()

# Supported upload extensions.
IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
)
PDF_EXTENSIONS: tuple[str, ...] = (".pdf",)


@dataclass
class EnvReport:
    """A snapshot of the runtime environment, shown in the UI header."""

    platform: str = "Windows"
    python_version: str = ""
    torch_version: str = ""
    cuda_available: bool = False
    cuda_version: str = ""
    gpu_name: str = "CPU"
    attention_backend: str = "unloaded"
    unlimited_ocr: str = "未加载"
    mini_uflash: str = "未加载"
    mode: str = "stable"
    venv_python: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "platform": self.platform,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "gpu_name": self.gpu_name,
            "attention_backend": self.attention_backend,
            "unlimited_ocr": self.unlimited_ocr,
            "mini_uflash": self.mini_uflash,
            "mode": self.mode,
            "venv_python": self.venv_python,
        }


def build_static_env_report() -> EnvReport:
    """Build an :class:`EnvReport` with the static fields filled in.

    Engine state (``unlimited_ocr`` / ``mini_uflash`` / ``attention_backend``)
    is left at its default "未加载"/"unloaded" and refreshed by the UI as the
    model manager loads.
    """
    import platform
    import sys

    report = EnvReport()
    report.platform = f"Windows {platform.uname().release}"
    report.python_version = sys.version.split()[0]
    report.venv_python = sys.executable
    try:
        import torch  # type: ignore

        report.torch_version = torch.__version__
        report.cuda_available = bool(torch.cuda.is_available())
        report.cuda_version = str(torch.version.cuda or "")
        if report.cuda_available:
            try:
                report.gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                report.gpu_name = "GPU"
    except Exception:
        report.torch_version = "torch 未安装"
    if unlimited_ocr_path().is_dir():
        report.unlimited_ocr = "本地模型可用（未加载）"
    if discover_weight() is not None:
        report.mini_uflash = "Stage 11B 可用（按需加载）"
    else:
        report.mini_uflash = "权重缺失"
    return report
