"""Recover completed per-page Markdown from an interrupted stable PDF run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from webapp.unlimited_ocr_engine import StablePageResult, assemble_result, clean_markdown


def recover(scratch: Path, run_dir: Path) -> int:
    page_dirs = []
    for candidate in scratch.glob("page_*"):
        match = re.fullmatch(r"page_(\d+)", candidate.name)
        result_file = candidate / "result.md"
        if match and result_file.is_file():
            page_dirs.append((int(match.group(1)), result_file))
    page_dirs.sort()

    pages = []
    for page_index, result_file in page_dirs:
        raw = result_file.read_text(encoding="utf-8")
        pages.append(StablePageResult(
            page_index=page_index,
            raw_markdown=raw,
            markdown=clean_markdown(raw),
            elapsed_seconds=0.0,
        ))

    result = assemble_result(pages, 0.0)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "recovered_partial_result.md").write_text(result.markdown, encoding="utf-8")
    (run_dir / "recovered_partial_result.txt").write_text(result.plain_text, encoding="utf-8")
    payload = {
        "status": "recovered_partial",
        "completed_pages": len(pages),
        "last_completed_page": pages[-1].page_index + 1 if pages else 0,
        "pages": [p.__dict__ for p in pages],
    }
    (run_dir / "recovered_partial_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return len(pages)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scratch", type=Path)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    count = recover(args.scratch.resolve(), args.run_dir.resolve())
    print(f"Recovered {count} completed pages.")


if __name__ == "__main__":
    main()
