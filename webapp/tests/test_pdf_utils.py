from pathlib import Path

import fitz

from webapp.pdf_utils import cleanup_dir, page_count, render_pages


def _make_pdf(path: Path, pages: int = 2):
    with fitz.open() as doc:
        for index in range(pages):
            page = doc.new_page()
            page.insert_text((72, 72), f"page {index + 1}")
        doc.save(path)


def test_pdf_render_and_windows_handle_release(tmp_path):
    pdf = tmp_path / "中文 文档.pdf"
    _make_pdf(pdf)
    assert page_count(pdf) == 2
    pages_dir = tmp_path / "pages"
    rendered = render_pages(pdf, pages_dir, dpi=72)
    assert [p.page_index for p in rendered] == [0, 1]
    assert all(p.image_path.is_file() for p in rendered)
    pdf.unlink()
    assert not pdf.exists()
    cleanup_dir(pages_dir)
    assert not pages_dir.exists()

