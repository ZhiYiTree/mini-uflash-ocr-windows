"""Output export and file I/O for the Mini UFlash OCR webapp.

Every OCR run creates a timestamped directory under ``webapp/outputs/`` and
writes structured results there (Markdown, plain text, JSON, metrics, log).
The UI provides download links for each file.

All writes use ``with`` blocks (the spec explicitly requires prompt file-handle
release on Windows). JSON is written with ``ensure_ascii=False, indent=2``.
Paths are resolved via ``pathlib.Path.resolve()``.
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from . import config, path_utils

_log = logging.getLogger(__name__)


def create_run_dir() -> Path:
    """Create ``webapp/outputs/YYYYMMDD_HHMMSS_<random>`` and return it.

    The random suffix prevents collisions when two runs start in the same
    second (unlikely but possible in testing).
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"{random.randint(1000, 9999)}"
    run_dir = config.OUTPUTS_DIR / f"{ts}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir.resolve()


def save_input(image_path: Optional[Path], run_dir: Path) -> None:
    """Copy the uploaded file into ``run_dir/input/`` for reproducibility."""
    if image_path is None or not image_path.is_file():
        return
    dest = run_dir / "input"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(image_path), str(dest / image_path.name))


def save_result(
    run_dir: Path,
    *,
    markdown: str = "",
    plain_text: str = "",
    raw_output: str = "",
    result_json: Dict[str, Any],
    metrics_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Write all output files and return a name→Path mapping.

    Files written:
    * ``result.md``
    * ``result.txt``
    * ``result.json``
    * ``metrics.json`` (if provided)
    """
    files: Dict[str, Path] = {}

    # result.md
    md_path = run_dir / "result.md"
    with open(str(md_path), "w", encoding="utf-8") as f:
        f.write(markdown)
    files["result.md"] = md_path

    # result.txt
    txt_path = run_dir / "result.txt"
    with open(str(txt_path), "w", encoding="utf-8") as f:
        f.write(plain_text)
    files["result.txt"] = txt_path

    # result.json
    json_path = run_dir / "result.json"
    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    files["result.json"] = json_path

    # metrics.json
    if metrics_json is not None:
        met_path = run_dir / "metrics.json"
        with open(str(met_path), "w", encoding="utf-8") as f:
            json.dump(metrics_json, f, ensure_ascii=False, indent=2)
        files["metrics.json"] = met_path

    return files


def save_log(run_dir: Path, text: str) -> Path:
    """Write the run log to ``run_dir/run.log``."""
    log_path = run_dir / "run.log"
    with open(str(log_path), "w", encoding="utf-8") as f:
        f.write(text)
    return log_path


def build_result_json(
    mode: str,
    input_type: str,
    markdown: str,
    plain_text: str,
    elapsed_seconds: float,
    generated_tokens: int,
    pages: Optional[list] = None,
    mini_uflash: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the output result.json schema per the spec."""
    return {
        "mode": mode,
        "platform": "Windows",
        "input_type": input_type,
        "pages": pages or [],
        "markdown": markdown,
        "plain_text": plain_text,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "generated_tokens": generated_tokens,
        "mini_uflash": mini_uflash or {},
    }
