"""Create a print-safe HTML variant of the portable progress report."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "docs" / "waxs_aidt_odt_progress_report_20260713_support"
SOURCE = SUPPORT / "report.html"
OUTPUT = SUPPORT / "report_print.html"

PRINT_STYLE = r"""
<style id="acfo-print-overrides">
@media print {
  @page { size: A4 portrait; margin: 11mm 10mm 12mm; }
  html, body { print-color-adjust: exact; -webkit-print-color-adjust: exact; }
  .portable-fallback { width: 100% !important; max-width: none !important; padding: 0 !important; }
  .portable-page-header { gap: 12px !important; padding-bottom: 14px !important; }
  .portable-page-meta { min-width: 115px; }
  .portable-block-stack { gap: 15px !important; margin-top: 16px !important; }
  .portable-markdown { max-width: none !important; }
  .portable-markdown h2 { margin: 16px 0 8px; }
  .portable-content-card,
  .portable-chart-summary,
  .portable-table-card { break-inside: auto !important; }
  .portable-metric-card { break-inside: avoid !important; }
  .portable-table-scroll {
    width: 100% !important;
    max-width: 100% !important;
    overflow: visible !important;
  }
  .portable-table-scroll table {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    table-layout: fixed !important;
    border-collapse: collapse !important;
  }
  .portable-table-scroll thead { display: table-header-group; }
  .portable-table-scroll tr { break-inside: avoid !important; }
  .portable-table-scroll th,
  .portable-table-scroll td {
    max-width: none !important;
    overflow: visible !important;
    padding: 5px 7px 5px 0 !important;
    font-size: 8.2pt !important;
    line-height: 1.3 !important;
    white-space: normal !important;
    overflow-wrap: anywhere !important;
    word-break: normal !important;
    text-overflow: clip !important;
  }
  .portable-table-scroll th:last-child,
  .portable-table-scroll td:last-child { padding-right: 0 !important; }
  .portable-table-number { text-align: right !important; }
  .portable-inline-source,
  .portable-table-note { font-size: 7.4pt !important; line-height: 1.3 !important; }
  .portable-static-chart { overflow: visible !important; }
}
</style>
"""


def main() -> None:
    html = SOURCE.read_text(encoding="utf-8")
    if "</head>" not in html:
        raise RuntimeError("Portable report HTML is missing </head>.")
    if 'id="acfo-print-overrides"' in html:
        raise RuntimeError("Canonical report already contains the print override.")
    OUTPUT.write_text(html.replace("</head>", f"{PRINT_STYLE}</head>", 1), encoding="utf-8")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
