#!/usr/bin/env python3
"""Audit a returned UoB-100(Dy) n_d estimator sensitivity bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    summary = json.loads((args.evidence / "summary.json").read_text(encoding="utf-8"))
    estimators = summary.get("estimators", {})
    checks = {
        "summary_schema": summary.get("schema")
        == "uob100dy-nd-estimator-sensitivity-summary-v1",
        "summary_pass": summary.get("status") == "PASS",
        "all_technical_gates": all(summary.get("technical_gates", {}).values()),
        "prediction_file_present": (args.evidence / "predictions.npz").is_file(),
        "parent_hashes": summary.get("parent_hashes")
        == {
            name: protocol["frozen_parent"][f"{name}_sha256"]
            for name in (
                "atomic_predictions",
                "atomic_summary",
                "native_predictions",
                "native_summary",
            )
        },
        "all_estimators_reported": set(estimators) == set(protocol["estimators"]),
        "all_outcomes_boolean": all(
            isinstance(value.get("primary_source_interval_recovered"), bool)
            and isinstance(value.get("nearest_source_interval_recovered"), bool)
            for value in estimators.values()
        ),
        "claim_boundary_preserved": summary.get("claim_boundary", {}).get(
            "positive_physical_claim_allowed"
        )
        is False,
    }
    failures = [name for name, passed in checks.items() if not passed]
    result = {
        "schema": "uob100dy-nd-estimator-sensitivity-return-audit-v1",
        "checks": checks,
        "failures": failures,
        "passed": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
