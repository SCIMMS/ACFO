"""Verify and render the WAXS/aIDT/ODT progress-report PDF."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "docs" / "ACFO_WAXS_aIDT_ODT_progress_report_2026-07-13_ko.pdf"
SUPPORT = ROOT / "docs" / "waxs_aidt_odt_progress_report_20260713_support"
TEXT_OUTPUT = SUPPORT / "extracted_text.txt"
QA_OUTPUT = SUPPORT / "pdf_qa.json"
PAGE_DIR = SUPPORT / "pdf_pages"

REQUIRED_ANCHORS = {
    "title": "ACFO WAXS·aIDT·ODT",
    "waxs_operator_accuracy": "2.5e-15",
    "waxs_source_failure": "77.65%",
    "waxs_detector_speedup": "1.976",
    "aidt_hot_rate": "10.31",
    "odt_hot_pair": "0.444",
    "odt_resident_oom": "9.12 GiB",
    "odt_physical_gate": "17.11%",
    "oriented_protein_question": "protein crystal",
}

FORBIDDEN_VISIBLE_TEXT = [
    "snapshot status",
    "widget type",
    "manifest path",
    "package path",
    "validation status",
    "C:\\Users\\",
    "file:///",
]


def main() -> int:
    if not PDF.is_file():
        raise FileNotFoundError(PDF)

    SUPPORT.mkdir(parents=True, exist_ok=True)
    if PAGE_DIR.exists():
        shutil.rmtree(PAGE_DIR)
    PAGE_DIR.mkdir(parents=True)

    document = fitz.open(PDF)
    page_records: list[dict[str, object]] = []
    page_texts: list[str] = []
    blank_pages: list[int] = []

    zoom = 1.35
    matrix = fitz.Matrix(zoom, zoom)
    for page_index, page in enumerate(document):
        text = page.get_text("text")
        page_texts.append(text)
        drawing_count = len(page.get_drawings())
        image_count = len(page.get_images(full=True))
        text_chars = len(text.strip())
        if text_chars == 0 and drawing_count == 0 and image_count == 0:
            blank_pages.append(page_index + 1)

        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        output_path = PAGE_DIR / f"page_{page_index + 1:02d}.png"
        pixmap.save(output_path)
        page_records.append(
            {
                "page": page_index + 1,
                "width_pt": round(page.rect.width, 2),
                "height_pt": round(page.rect.height, 2),
                "text_chars": text_chars,
                "drawing_count": drawing_count,
                "image_count": image_count,
                "preview": str(output_path.relative_to(ROOT)),
            }
        )

    full_text = "\n\n".join(page_texts)
    TEXT_OUTPUT.write_text(full_text, encoding="utf-8")

    anchors = {
        key: {"text": value, "found": value in full_text}
        for key, value in REQUIRED_ANCHORS.items()
    }
    missing_anchors = [key for key, result in anchors.items() if not result["found"]]
    forbidden_hits = [value for value in FORBIDDEN_VISIBLE_TEXT if value in full_text]

    qa = {
        "pdf": str(PDF.relative_to(ROOT)),
        "file_size_bytes": PDF.stat().st_size,
        "page_count": document.page_count,
        "metadata": document.metadata,
        "anchors": anchors,
        "missing_anchors": missing_anchors,
        "forbidden_visible_text_hits": forbidden_hits,
        "blank_pages": blank_pages,
        "pages": page_records,
        "passed": not missing_anchors and not forbidden_hits and not blank_pages,
    }
    QA_OUTPUT.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    document.close()
    return 0 if qa["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
