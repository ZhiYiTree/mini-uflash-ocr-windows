"""DFlash acceleration tiers (Hunyuan-style product split + DSpark verify schedule).

Tiers
-----
fast      : wall-clock first (sys_v2 soft length caps)
balanced  : moderate soft cap
lossless  : no soft truncation; DSpark-style calibrated verify_len only
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Project root: webapp/..
_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CALIBRATION_PATH = _ROOT / "train" / "conf_calibration.json"

TIER_PRESETS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "block_size": 4,
        "max_block_size": 4,
        "resync_every": 256,
        "length_hard_cap": 384,
        "length_soft_cap": 280,
        "low_tau_b1_threshold": 1.0,
        "low_tau_window": 6,
        "low_tau_b1_steps": 8,
        "zero_accept_page_b1_after": 6,
        "skip_verify_expected_below": 0.45,
        "use_conf_schedule": True,
        "conf_schedule_min_expected": 0.0,
        "page_degrade_enabled": True,
    },
    "balanced": {
        "block_size": 4,
        "max_block_size": 4,
        "resync_every": 224,
        "length_hard_cap": 480,
        "length_soft_cap": 360,
        "low_tau_b1_threshold": 0.9,
        "low_tau_window": 8,
        "low_tau_b1_steps": 6,
        "zero_accept_page_b1_after": 10,
        "skip_verify_expected_below": 0.35,
        "use_conf_schedule": True,
        "conf_schedule_min_expected": 0.0,
        "page_degrade_enabled": True,
    },
    "lossless": {
        # Honest acceleration: no soft truncation; length only hard-capped by max_length.
        "block_size": 4,
        "max_block_size": 6,
        "resync_every": 192,
        "length_hard_cap": 0,  # disabled → use full max_length budget
        "length_soft_cap": 0,
        "low_tau_b1_threshold": 0.75,
        "low_tau_window": 10,
        "low_tau_b1_steps": 4,
        "zero_accept_page_b1_after": 0,  # never whole-page abandon
        "skip_verify_expected_below": 0.0,  # use conf schedule instead
        "use_conf_schedule": True,
        "conf_schedule_min_expected": 0.0,
        "page_degrade_enabled": False,
    },
}


def resolve_tier(name: Optional[str]) -> Dict[str, Any]:
    key = (name or "fast").strip().lower()
    if key not in TIER_PRESETS:
        raise ValueError(f"Unknown dflash tier {name!r}; choose {list(TIER_PRESETS)}")
    return dict(TIER_PRESETS[key])


def load_calibration(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path) if path else DEFAULT_CALIBRATION_PATH
    if not p.is_file():
        return {
            "format": "mini_uflash_conf_calibration_v1",
            "temps": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "conf_floor": 0.22,
            "survival_floor": 0.10,
            "pos0_skip_below": 0.18,
            "fitted": False,
        }
    data = json.loads(p.read_text(encoding="utf-8"))
    data.setdefault("temps", [1.0] * 7)
    data.setdefault("conf_floor", 0.22)
    data.setdefault("survival_floor", 0.10)
    data.setdefault("pos0_skip_below", 0.18)
    return data


def _logit(p: float, eps: float = 1e-4) -> float:
    p = min(1.0 - eps, max(eps, float(p)))
    import math

    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    import math

    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def apply_temperature(conf: Sequence[float], temps: Sequence[float]) -> List[float]:
    """Position-wise temperature scaling on Bernoulli confidences."""
    out: List[float] = []
    for i, c in enumerate(conf):
        t = float(temps[i]) if i < len(temps) else float(temps[-1] if temps else 1.0)
        t = max(0.25, min(4.0, t))
        out.append(_sigmoid(_logit(c) / t))
    return out


def schedule_verify_len(
    conf: Sequence[float],
    *,
    calibration: Optional[Dict[str, Any]] = None,
    min_len: int = 1,
) -> int:
    """DSpark-style: keep longest prefix whose calibrated survival stays healthy."""
    if not conf:
        return 0
    cal = calibration or load_calibration()
    temps = list(cal.get("temps") or [1.0])
    calibrated = apply_temperature(conf, temps)
    conf_floor = float(cal.get("conf_floor", 0.22))
    survival_floor = float(cal.get("survival_floor", 0.10))
    pos0_skip = float(cal.get("pos0_skip_below", 0.18))

    if float(calibrated[0]) < pos0_skip:
        return 0

    survival = 1.0
    keep = 0
    for i, c in enumerate(calibrated):
        survival *= max(1e-6, min(1.0, float(c)))
        if float(c) < conf_floor or survival < survival_floor:
            break
        keep = i + 1

    if keep < min_len and float(calibrated[0]) >= pos0_skip:
        keep = min_len
    # Prefer at least 2 when prefix looks strong (amortize clone cost).
    if keep == 1 and len(calibrated) >= 2 and float(calibrated[0]) >= 0.55:
        if float(calibrated[1]) >= conf_floor * 0.9:
            keep = 2
    return int(min(keep, len(conf)))
