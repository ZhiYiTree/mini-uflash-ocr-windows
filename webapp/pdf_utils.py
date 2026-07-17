"""PDF support for the Mini UFlash OCR webapp.

Uses PyMuPDF (``fitz``) to render each PDF page to an RGB PNG. Every file
handle is opened with a ``with`` block so Windows releases it promptly — the
spec calls this out explicitly, since Windows will refuse to delete a file that
is still open. Temporary page images are cleaned up at the end of each task.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from . import path_utils


@dataclass(frozen=True)
class RenderedPage:
    """A single rendered PDF page ready for OCR."""

    page_index: int  # zero-based
    image_path: Path


class PdfError(Exception):
    """Raised when a PDF cannot be opened or rendered."""


def page_count(pdf_path: str | Path) -> int:
    """Return the number of pages in ``pdf_path`` (0 if unreadable)."""
    import fitz  # type: ignore

    path = path_utils.normalize(pdf_path)
    try:
        with fitz.open(str(path)) as doc:
            return int(doc.page_count)
    except Exception as exc:  # noqa: BLE001
        raise PdfError(f"无法打开 PDF：{path}\n{exc}") from exc


def render_pages(
    pdf_path: str | Path,
    out_dir: str | Path,
    dpi: int = 200,
    *,
    start_page: int = 0,
    end_page: Optional[int] = None,
) -> list[RenderedPage]:
    """Render ``pdf_path`` pages to PNGs in ``out_dir`` and return them.

    Pages are rendered one at a time inside a ``with fitz.open(...)`` block so
    the document handle is closed as soon as rendering finishes. The pixmap
    for each page is saved and the pixmap/pix references dropped before moving
    on, so the file handle backing each PNG is released and can be deleted.
    """
    import fitz  # type: ignore

    path = path_utils.normalize(pdf_path)
    out = path_utils.ensure_dir(out_dir)
    rendered: list[RenderedPage] = []

    with fitz.open(str(path)) as doc:
        total = int(doc.page_count)
        last = total - 1 if end_page is None else min(int(end_page), total - 1)
        first = max(0, int(start_page))
        if first > last:
            return rendered
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for index in range(first, last + 1):
            page = doc.load_page(index)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out_path = out / f"p{index:05d}.png"
            pix.save(str(out_path))
            # Explicitly drop pixmap reference so its underlying data is freed.
            del pix
            rendered.append(RenderedPage(page_index=index, image_path=out_path))
        # Leaving the ``with`` block closes the document handle.
    return rendered


def iter_render_pages(
    pdf_path: str | Path,
    out_dir: str | Path,
    dpi: int = 200,
) -> Iterator[RenderedPage]:
    """Yield rendered pages one at a time (for streaming progress to the UI).

    The document is held open for the duration of iteration; callers must fully
    consume the generator (or close it) so the handle is released.
    """
    import fitz  # type: ignore

    path = path_utils.normalize(pdf_path)
    out = path_utils.ensure_dir(out_dir)
    with fitz.open(str(path)) as doc:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for index in range(int(doc.page_count)):
            page = doc.load_page(index)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out_path = out / f"p{index:05d}.png"
            pix.save(str(out_path))
            del pix
            yield RenderedPage(page_index=index, image_path=out_path)


def cleanup_dir(directory: str | Path) -> None:
    """Delete a temporary directory tree if it exists (best effort)."""
    path = path_utils.normalize(directory)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def is_pdf(path: str | Path) -> bool:
    return Path(str(path)).suffix.lower() == ".pdf"
