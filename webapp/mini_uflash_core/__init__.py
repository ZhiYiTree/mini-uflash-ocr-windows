"""Vendored, Windows-clean Mini UFlash Stage 11B research core.

This package contains the exact drafter architecture (``model.py``) and
checkpoint loader (``checkpoint.py``) from the Stage 11B prefix-survival run,
plus the Stage 9 common helpers (``stage9_common.py``) and the online target
wrapper (``online_target.py``) used by the strict-B1-replay exactness probe.

Everything here is pure PyTorch and uses ``F.scaled_dot_product_attention``
(SDPA) for the drafter's self/cross attention. There are **no** flash-attn,
triton or xformers imports anywhere in this package, so it runs natively on
Windows without a compiler toolchain.
"""

from .checkpoint import load_checkpoint, save_checkpoint  # noqa: F401
from .model import DFlashOCRConfig, MaskBlockDrafter  # noqa: F401

__all__ = [
    "DFlashOCRConfig",
    "MaskBlockDrafter",
    "load_checkpoint",
    "save_checkpoint",
]
