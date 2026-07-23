from __future__ import annotations

import json
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "acfo_ncs_validation_closure_20260714"
PDF = REPORT_DIR / "ACFO_NCS_validation_closure_report_ko.pdf"
INVENTORY = REPORT_DIR / "report_source_inventory.json"
QA_DIR = REPORT_DIR / "qa"


ANCHORS = [
    "ACFO NCS 검증 종결 및 주장 경계",
    "matched-error 비교",
    "한 번의 warm update는 15.91 Hz",
    "WAXS는 새로운 범용 가속 주장",
    "aIDT는 10 Hz급 core",
    "측정 정의와 검증 설계",
    "제한과 robustness",
    "다음 단계는 더 많은 로컬 최적화",
    "재현 명령과 증거 파일",
]


def main() -> None:
    if not PDF.is_file() or not INVENTORY.is_file():
        raise FileNotFoundError("report PDF or source inventory is missing")
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    if inventory["schema"] != "acfo-ncs-validation-closure-report-sources-v1":
        raise AssertionError("unexpected source inventory schema")
    doc = fitz.open(PDF)
    text_by_page = [page.get_text("text") for page in doc]
    all_text = "\n".join(text_by_page)
    missing = [anchor for anchor in ANCHORS if anchor not in all_text]
    empty_pages = [index + 1 for index, text in enumerate(text_by_page) if len(text.strip()) < 80]
    if missing:
        raise AssertionError(f"missing PDF anchors: {missing}")
    if empty_pages:
        raise AssertionError(f"nearly empty PDF pages: {empty_pages}")
    if len(doc) < 8:
        raise AssertionError("report is unexpectedly short")
    QA_DIR.mkdir(parents=True, exist_ok=True)
    render_pages = sorted({0, 2, 3, 4, len(doc) - 1})
    rendered = []
    for index in render_pages:
        pix = doc[index].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        path = QA_DIR / f"page_{index + 1:02d}.png"
        pix.save(path)
        rendered.append(str(path))
    result = {
        "passed": True,
        "page_count": len(doc),
        "pdf_bytes": PDF.stat().st_size,
        "inventory_source_count": len(inventory["sources"]),
        "chart_count": len(inventory["chart_map"]),
        "anchors_checked": ANCHORS,
        "rendered_pages": rendered,
    }
    (REPORT_DIR / "verification_receipt.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
