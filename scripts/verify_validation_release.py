"""Audit the compact ACFO manuscript validation release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "acfo-validation-release-manifest-v1"
WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
SECRET_PATTERN = re.compile(
    r"(?i)(github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+|"
    r"(?:api[_-]?key|password|secret)\s*[:=]\s*[^\s,;]+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from iter_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_strings(item)


def close(actual: float, expected: float, *, rel: float = 1e-12) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=rel, abs_tol=0.0)


def inventory(root: Path) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "MANIFEST.json":
            continue
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return entries


def build_manifest(root: Path) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": "https://github.com/SCIMMS/ACFO",
        "license": "ACFO Citation-Required License 1.0",
        "files": inventory(root),
    }


def check_manifest(root: Path, errors: list[str]) -> None:
    path = root / "MANIFEST.json"
    if not path.exists():
        errors.append("validation/MANIFEST.json is missing")
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("schema") != SCHEMA:
        errors.append("manifest schema mismatch")
    if saved.get("files") != inventory(root):
        errors.append("manifest inventory or SHA-256 values do not match")


def check_json_and_paths(root: Path, errors: list[str]) -> None:
    def reject_nonstandard_constant(value: str):
        raise ValueError(f"non-standard JSON constant: {value}")

    for path in sorted(root.rglob("*.json")):
        try:
            document = json.loads(
                path.read_text(encoding="utf-8-sig"),
                parse_constant=reject_nonstandard_constant,
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"invalid JSON: {path}: {exc}")
            continue
        for value in iter_strings(document):
            if WINDOWS_PATH.match(value):
                errors.append(f"machine-local absolute path remains in {path}: {value}")


def check_secrets(root: Path, errors: list[str]) -> None:
    text_suffixes = {".json", ".csv", ".md", ".txt", ".cff", ""}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if SECRET_PATTERN.search(text):
            errors.append(f"possible credential-like value in {path}")


def check_package(root: Path, errors: list[str]) -> None:
    package_dir = root / "general_curvature" / "replication_package_v1_1"
    archive = package_dir / "general_curvature_cpu_replication_v1_1_20260722.zip"
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    expected = checksum.read_text(encoding="utf-8").split()[0].lower()
    if sha256(archive) != expected:
        errors.append("general-curvature replication ZIP checksum mismatch")
    with zipfile.ZipFile(archive) as package:
        bad = package.testzip()
    if bad is not None:
        errors.append(f"general-curvature replication ZIP CRC failure: {bad}")


def check_headline_metrics(root: Path, errors: list[str]) -> None:
    headline = json.loads((root / "headline_metrics.json").read_text(encoding="utf-8"))
    curation = json.loads(
        (root / "provenance" / "external_rtx3090_curation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    independent = json.loads(
        (
            root
            / "general_curvature"
            / "independent_cpu_return_manifest.json"
        ).read_text(encoding="utf-8")
    )
    aidt = json.loads(
        (root / "odt_aidt" / "aidt_10hz_full700_opt_repeat.json").read_text(
            encoding="utf-8"
        )
    )

    checks = (
        (
            curation["headline_metrics"]["waxs_prepared"]["paired_speedup_median"],
            headline["waxs"]["prepared_1001000_atoms"]["paired_speedup_median"],
            "prepared WAXS speedup",
        ),
        (
            curation["headline_metrics"]["waxs_detector"][
                "speedup_finufft_over_acfo"
            ],
            headline["waxs"]["detector_aware"]["speedup"],
            "detector WAXS speedup",
        ),
        (
            curation["headline_metrics"]["odt"]["same_dtype_pair_speedup"],
            headline["odt_aidt"]["odt_same_dtype_pair_speedup"],
            "ODT same-dtype speedup",
        ),
        (
            independent["headline_metrics"]["max_curve_operator_relative_l2"],
            headline["general_curvature"]["max_curve_operator_relative_l2"],
            "general-curvature operator error",
        ),
        (
            1.0 / aidt["summary"]["gpu_run_median_s"],
            headline["odt_aidt"]["aidt_public_core_hz"],
            "aIDT processing-core rate",
        ),
    )
    for actual, expected, label in checks:
        if not close(actual, expected):
            errors.append(f"headline mismatch: {label}: {actual} != {expected}")

    high_na_rows = list(
        csv.DictReader(
            (root / "high_na_si" / "high_na_si_source_data.csv").open(
                encoding="utf-8", newline=""
            )
        )
    )
    if len(high_na_rows) != 8:
        errors.append(f"expected 8 High-NA source rows, found {len(high_na_rows)}")
    high_na_values = {row["metric"]: [] for row in high_na_rows}
    for row in high_na_rows:
        high_na_values[row["metric"]].append(float(row["value"]))
    adaptive = high_na_values.get("adaptive_sparse_field_l2_vs_direct", [])
    expected_adaptive = sorted(
        [
            headline["high_na_si"]["adaptive_sparse_field_l2_small_vortex"],
            headline["high_na_si"][
                "adaptive_sparse_field_l2_representative_vortex"
            ],
        ]
    )
    if len(adaptive) != 2 or any(
        not close(actual, expected)
        for actual, expected in zip(sorted(adaptive), expected_adaptive)
    ):
        errors.append("High-NA adaptive-error rows do not match the headline ledger")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("validation"),
        help="Validation directory to audit.",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write a fresh SHA-256 manifest before auditing.",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    if args.write_manifest:
        manifest = build_manifest(root)
        (root / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    errors: list[str] = []
    check_manifest(root, errors)
    check_json_and_paths(root, errors)
    check_secrets(root, errors)
    check_package(root, errors)
    check_headline_metrics(root, errors)

    result = {
        "passed": not errors,
        "file_count": len(inventory(root)),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
