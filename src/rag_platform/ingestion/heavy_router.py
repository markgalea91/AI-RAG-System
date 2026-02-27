from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple


def file_size_mb(p: Path) -> float:
    return p.stat().st_size / (1024 * 1024)


def pdf_page_count(p: Path) -> int:
    """
    Requires: pip install pypdf
    If it can't read the PDF, treat as very heavy to be safe.
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(p))
        return len(reader.pages)
    except Exception:
        return 10**9


def classify_files_heavy(
    files: List[str],
    *,
    heavy_size_mb: float = 50.0,
    pdf_heavy_pages: int = 80,
) -> Tuple[List[str], List[str], List[Dict]]:
    """
    Split into (normal_files, heavy_files, reasons).
    """
    normal: List[str] = []
    heavy: List[str] = []
    reasons: List[Dict] = []

    for f in files:
        p = Path(f)
        ext = p.suffix.lower().lstrip(".")
        size = file_size_mb(p)

        # Size threshold
        if size >= heavy_size_mb:
            heavy.append(f)
            reasons.append(
                {"file": f, "reason": f"size_mb>={heavy_size_mb}", "size_mb": round(size, 2)}
            )
            continue

        # PDF page-count threshold
        if ext == "pdf":
            pages = pdf_page_count(p)
            if pages >= pdf_heavy_pages:
                heavy.append(f)
                reasons.append(
                    {
                        "file": f,
                        "reason": f"pdf_pages>={pdf_heavy_pages}",
                        "size_mb": round(size, 2),
                        "pages": pages,
                    }
                )
                continue

        normal.append(f)

    return normal, heavy, reasons