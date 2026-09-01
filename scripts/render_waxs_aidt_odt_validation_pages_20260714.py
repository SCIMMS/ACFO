from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "docs" / "waxs_aidt_odt_validation_report_20260714_support"
HTML = SUPPORT / "report_print.html"
PAGE_DIR = SUPPORT / "pdf_pages"
CHROME = Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe")


def main() -> None:
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    for old in PAGE_DIR.glob("page-*.png"):
        old.unlink()

    outputs: list[dict[str, object]] = []
    base_uri = HTML.resolve().as_uri()
    for page_number in range(1, 11):
        output = PAGE_DIR / f"page-{page_number:02d}.png"
        completed = subprocess.run(
            [
                str(CHROME),
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
                "--hide-scrollbars",
                "--allow-file-access-from-files",
                "--virtual-time-budget=1500",
                "--window-size=794,1123",
                "--force-device-scale-factor=1",
                f"--screenshot={output}",
                f"{base_uri}#page-{page_number:02d}",
            ],
            check=True,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        if not output.exists():
            raise RuntimeError(f"Screenshot missing: {output}")
        outputs.append({
            "page": page_number,
            "path": str(output),
            "bytes": output.stat().st_size,
            "chrome_stderr": completed.stderr.strip(),
        })
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
