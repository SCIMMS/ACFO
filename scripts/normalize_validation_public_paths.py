"""Normalize machine-local paths in curated validation JSON files.

The scientific result fields are left unchanged. Each normalized document gets
an ``_publication`` record containing the SHA-256 of the original receipt and a
description of the publication-only normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
PUBLIC_MARKERS = (
    "benchmark_results/",
    "scripts/",
    "structures/",
    "src/",
    "tests/",
    "validation/",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_path(value: str) -> str:
    candidate = value.replace("\\", "/")
    if not WINDOWS_ABSOLUTE.match(value):
        return value

    lowered = candidate.lower()
    if lowered.endswith("/.venv/scripts/python.exe"):
        return "python"

    for marker in PUBLIC_MARKERS:
        index = lowered.find(marker.lower())
        if index >= 0:
            return candidate[index:]

    return f"<local-path-redacted>/{Path(candidate).name}"


def normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return normalize_path(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_value(item) for key, item in value.items()}
    return value


def normalize_document(path: Path) -> bool:
    raw = path.read_bytes()
    document = json.loads(raw)
    if not isinstance(document, dict):
        return False

    normalized = normalize_value(document)
    if normalized == document:
        return False

    publication = normalized.get("_publication", {})
    publication.setdefault("source_sha256", hashlib.sha256(raw).hexdigest())
    publication["normalization"] = (
        "Machine-local absolute paths were replaced with repository-relative "
        "paths or a redacted placeholder. Non-finite uncollected output-statistic "
        "fields were represented as JSON null. Scientific timing and accuracy "
        "values are unchanged."
    )
    normalized["_publication"] = publication
    path.write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("validation"),
        help="Validation directory to normalize.",
    )
    args = parser.parse_args()

    changed = []
    for path in sorted(args.root.rglob("*.json")):
        if normalize_document(path):
            changed.append(path.as_posix())

    print(json.dumps({"changed": changed, "count": len(changed)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
