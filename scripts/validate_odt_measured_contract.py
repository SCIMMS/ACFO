from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from waxs_cake.odt_measured_contract import (  # noqa: E402
    build_prepared_operator_from_contract,
    load_odt_measured_contract,
    validate_odt_measured_contract,
)


def write_markdown(path: Path, payload: dict) -> None:
    report = payload["validation"]
    descriptor = payload.get("prepared_operator_descriptor")
    lines = [
        "# ODT measured-data contract validation",
        "",
        f"- path: `{payload['path']}`",
        f"- valid: `{report['ok']}`",
        "",
        "## Summary",
        "",
        "| key | value |",
        "| --- | --- |",
    ]
    for key, value in report.get("summary", {}).items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Errors", ""])
    if report["errors"]:
        for issue in report["errors"]:
            lines.append(f"- `{issue['field']}`: {issue['message']}")
    else:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        for issue in report["warnings"]:
            lines.append(f"- `{issue['field']}`: {issue['message']}")
    else:
        lines.append("- none")
    if descriptor is not None:
        lines.extend(["", "## Prepared Operator Descriptor", "", "| key | value |", "| --- | --- |"])
        for key, value in descriptor.items():
            lines.append(f"| `{key}` | `{value}` |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an ODT measured-data contract NPZ file.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when validation has errors.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    measured = load_odt_measured_contract(args.path)
    report = validate_odt_measured_contract(measured)
    descriptor = None
    if report.ok:
        descriptor = build_prepared_operator_from_contract(measured).__dict__
    payload = {
        "path": str(args.path),
        "validation": report.to_dict(),
        "prepared_operator_descriptor": descriptor,
    }
    text = json.dumps(payload, indent=2, default=str)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if args.summary_md is not None:
        write_markdown(args.summary_md, payload)
    print(text)
    if args.strict and not report.ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
