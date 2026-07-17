"""Windows 8GB Mini UFlash training defaults.

All paths live under the project root. Override with env vars or CLI flags.
Training is NOT started by importing this module.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Project root = parent of train/
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
TRAIN_ROOT: Path = PROJECT_ROOT / "train"

# Ensure project packages (webapp.mini_uflash_core) are importable.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Models & weights -------------------------------------------------------
DEFAULT_MODEL_PATH: Path = PROJECT_ROOT / "models" / "PaddlePaddle" / "Unlimited-OCR"
DEFAULT_RESUME_WEIGHT: Path = (
    PROJECT_ROOT / "weights" / "mini-uflash-v2.0.0-alpha.1-drafter-stage11b-best.pt"
)

# --- Data layout ------------------------------------------------------------
DATA_ROOT: Path = TRAIN_ROOT / "data"
PAGES_TRAIN: Path = DATA_ROOT / "pages" / "train"
PAGES_VAL: Path = DATA_ROOT / "pages" / "val"
PAGES_POOL: Path = DATA_ROOT / "pages" / "pool"  # unsorted inbox
TEACHERS_TRAIN: Path = DATA_ROOT / "teachers" / "train"
TEACHERS_VAL: Path = DATA_ROOT / "teachers" / "val"
PAYLOADS_DIR: Path = DATA_ROOT / "payloads"
SPLITS_DIR: Path = DATA_ROOT / "splits"
RUNS_DIR: Path = TRAIN_ROOT / "runs"

# --- Extraction (must match production Gundam / Crop when possible) ---------
DEFAULT_PROMPT: str = "<image>document parsing."
LAYER_INDICES: tuple[int, ...] = (1, 4, 6, 9, 11)
BASE_SIZE: int = 1024
IMAGE_SIZE: int = 640
CROP_MODE: bool = True
MAX_LENGTH: int = 4096
NO_REPEAT_NGRAM_SIZE: int = 35
NGRAM_WINDOW: int = 128
EXTRACT_DTYPE: str = "bfloat16"
STORAGE_DTYPE: str = "float16"  # teacher tensors on disk
MIN_GENERATED_TOKENS: int = 9  # B8 needs anchor + 7 + room

IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
)

# --- 8GB training defaults --------------------------------------------------
# Official Stage 11B used batch=32 on 4090D. Here we use micro-batch + accum.
TRAIN_MICRO_BATCH: int = 2
TRAIN_GRAD_ACCUM: int = 12  # effective batch ≈ 24 (safer than 32 on 8GB)
TRAIN_EVAL_BATCH: int = 2
TRAIN_STEPS: int = 2500  # start smaller than cloud 8k; resume later
TRAIN_EVAL_INTERVAL: int = 250
TRAIN_LR: float = 2e-5
TRAIN_WARMUP: int = 150
TRAIN_WEIGHT_DECAY: float = 0.01
TRAIN_BLOCK_SIZES: tuple[int, ...] = (4, 6, 8)
TRAIN_BLOCK_PROBS: tuple[float, ...] = (0.05, 0.20, 0.75)
TRAIN_POSITION_DECAY: float = 1.0
TRAIN_TAIL_BOOST: float = 0.0
TRAIN_HIDDEN_WEIGHT: float = 0.01
# DSpark-style: TV (distribution match) dominates; CE is light.
TRAIN_CE_WEIGHT: float = 0.1
TRAIN_TV_WEIGHT: float = 0.9
TRAIN_ACCEPTANCE_WEIGHT: float = 0.35
TRAIN_PREFIX_SURVIVAL_WEIGHT: float = 0.15
TRAIN_USE_AUF: bool = True
# exp(-(k-1)/gamma) position schedule (None → legacy decay**k)
TRAIN_EXP_POSITION_GAMMA: float = 7.0
TRAIN_CONF_LABEL_FROM_TV: bool = True
TRAIN_USE_MARKOV: bool = True
# Multi-anchor denser sampling per micro-step (P2, still 8GB-safe).
TRAIN_ANCHORS_PER_PAGE_ATTEMPT: int = 2
TRAIN_MAX_VAL_SAMPLES_PER_PAGE: int = 24  # slightly lower for 8GB eval VRAM
TRAIN_PAGE_CACHE_SIZE: int = 3  # keep host RAM free on ~16GB machines
TRAIN_SEED: int = 43
TRAIN_VAL_RATIO: float = 0.15
TRAIN_HARD_SAMPLE_PROB: float = 0.0  # hard mining off for first Windows run

# Leave headroom so Windows desktop does not hard-lock under OCR load.
# 0.0 = disabled. Extraction needs nearly full 8GB for Unlimited-OCR+crop;
# host-RAM headroom is handled via TRAIN_PAGE_CACHE_SIZE instead.
CUDA_MEMORY_FRACTION: float = 0.0
EXTRACT_MAX_LENGTH_SAFE: int = 2048  # lower peak activation vs 4096 on 8GB


def model_path() -> Path:
    return Path(os.environ.get("UNLIMITED_OCR_PATH", str(DEFAULT_MODEL_PATH))).expanduser().resolve()


def resume_weight() -> Path:
    return Path(
        os.environ.get("MINI_UFLASH_WEIGHT", str(DEFAULT_RESUME_WEIGHT))
    ).expanduser().resolve()


def ensure_data_dirs() -> dict[str, Path]:
    """Create the standard data/run directories. Idempotent."""
    dirs = {
        "pages_train": PAGES_TRAIN,
        "pages_val": PAGES_VAL,
        "pages_pool": PAGES_POOL,
        "teachers_train": TEACHERS_TRAIN,
        "teachers_val": TEACHERS_VAL,
        "payloads": PAYLOADS_DIR,
        "splits": SPLITS_DIR,
        "runs": RUNS_DIR,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
