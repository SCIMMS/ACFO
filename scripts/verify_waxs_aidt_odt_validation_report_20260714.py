from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfReader, PdfWriter


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "docs" / "ACFO_WAXS_aIDT_ODT_validation_progress_report_2026-07-14_ko.pdf"
SUPPORT = ROOT / "docs" / "waxs_aidt_odt_validation_report_20260714_support"
PAGE_DIR = SUPPORT / "pdf_pages"
SPLIT_DIR = SUPPORT / "split_pdf_pages"
TEXT_OUTPUT = SUPPORT / "extracted_text.txt"
QA_OUTPUT = SUPPORT / "pdf_qa.json"

ANCHORS = {
    "title": "ACFO WAXS · aIDT · ODT",
    "waxs_detector": "1.976×",
    "waxs_oriented_object": "protein crystal, single crystal",
    "aidt_rate": "10.31 Hz",
    "odt_pair": "111.366 ms",
    "odt_speedup": "81.24×",
    "odt_full": "8.46 Hz",
    "odt_slab": "10.64 Hz",
    "odt_banded": "12.52 Hz",
    "rtx5090_projection": "10.42 Hz",
    "claim_boundary": "10 completed reconstructions/s",
}

FORBIDDEN = [
    "file://",
    "C:\\Users\\",
    "sourceId",
    "artifact.json",
    "localhost",
]


def main() -> None:
    if not PDF.exists():
        raise FileNotFoundError(PDF)
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for old in SPLIT_DIR.glob("page-*.pdf"):
        old.unlink()

    document = PdfReader(str(PDF))
    pages: list[dict[str, object]] = []
    text_parts: list[str] = []
    for index, page in enumerate(document.pages):
        text = page.extract_text() or ""
        text_parts.append(text)
        split_path = SPLIT_DIR / f"page-{index + 1:02d}.pdf"
        writer = PdfWriter()
        writer.add_page(page)
        with split_path.open("wb") as handle:
            writer.write(handle)
        pages.append({
            "page": index + 1,
            "text_chars": len(text.strip()),
            "split_pdf": str(split_path),
        })

    full_text = "\n".join(text_parts)
    normalized = " ".join(full_text.split())
    anchors = {name: value in normalized for name, value in ANCHORS.items()}
    forbidden = {value: value in full_text for value in FORBIDDEN}
    blank_pages = [row["page"] for row in pages if row["text_chars"] < 80]
    qa = {
        "pdf": str(PDF),
        "bytes": PDF.stat().st_size,
        "page_count": len(pages),
        "pages": pages,
        "blank_pages": blank_pages,
        "anchors": anchors,
        "forbidden": forbidden,
        "passed": len(pages) == 10 and not blank_pages and all(anchors.values()) and not any(forbidden.values()),
    }
    TEXT_OUTPUT.write_text(full_text, encoding="utf-8")
    QA_OUTPUT.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    if not qa["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
