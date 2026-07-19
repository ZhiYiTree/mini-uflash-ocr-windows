"""VRAM helpers for 8GB-class Windows laptops."""

from __future__ import annotations

import gc
from typing import Optional, Tuple


def empty_cuda() -> None:
    """Best-effort free of unused CUDA caching allocator memory."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Release IPC handles when possible (no-op on some drivers).
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def mem_info_gb() -> Tuple[float, float]:
    """Return (free_gb, total_gb); zeros if CUDA unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0, 0.0
        free_b, total_b = torch.cuda.mem_get_info()
        return free_b / (1024**3), total_b / (1024**3)
    except Exception:
        return 0.0, 0.0


def is_cuda_oom(exc: BaseException) -> bool:
    """True for torch OOM and PyTorch 2.x AcceleratorError CUDA OOM."""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    name = type(exc).__name__
    msg = str(exc).lower()
    if "out of memory" in msg:
        return True
    if name in ("AcceleratorError", "RuntimeError") and "cuda" in msg and (
        "memory" in msg or "alloc" in msg
    ):
        return True
    return False


def clamp_max_length(requested: int, *, total_gb: Optional[float] = None) -> int:
    """Cap generation budget for small GPUs to reduce KV / image peaks."""
    req = max(256, int(requested or 2048))
    if total_gb is None:
        _, total_gb = mem_info_gb()
    if total_gb <= 0:
        return min(req, 2048)
    # Laptop 8GB class: Unlimited-OCR already ~5–6GB weights in bf16.
    if total_gb <= 8.5:
        return min(req, 1536)
    if total_gb <= 12.5:
        return min(req, 2048)
    return min(req, 4096)


def oom_user_message(extra: str = "") -> str:
    free_gb, total_gb = mem_info_gb()
    lines = [
        "❌ **显存不足 (CUDA OOM)**",
        "",
        f"- 当前 GPU 总显存约 **{total_gb:.1f} GB**，空闲约 **{free_gb:.1f} GB**",
        "- Unlimited-OCR 本身约 5–6GB；再叠加草稿模型、图像 crop、KV cache 很容易爆 8GB",
        "",
        "**请依次尝试：**",
        "1. 点「停止」后点「释放显存」，再重新识别",
        "2. 高级设置里把 **最长输出** 降到 **1024～1536**",
        "3. 用 **普通版** 或加速 **快速/均衡**（勿用无损 + 超长输出）",
        "4. 关闭其他占 GPU 的程序（浏览器硬件加速、其它 Python/训练）",
        "5. PDF 很长时分页跑，或降低 DPI / 先裁剪页面",
    ]
    if extra:
        lines.extend(["", extra])
    return "\n".join(lines)
