#!/usr/bin/env python3
"""Audit evidence returned by the frozen UoB-100(Dy) native-grid package."""

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
    parent = summary.get("parent", {})
    ranking = summary.get("ranking", {})
    adapters = ranking.get("adapters", {})
    checks = {
        "summary_schema": summary.get("schema")
        == "uob100dy-scatty-native-grid-audit-summary-v1",
        "summary_pass": summary.get("status") == "PASS",
        "all_technical_gates": all(summary.get("technical_gates", {}).values()),
        "prediction_file_present": (args.evidence / "predictions.npz").is_file(),
        "parent_predictions_hash": parent.get("file_hashes", {}).get("predictions")
        == protocol["frozen_parent"]["predictions_sha256"],
        "parent_summary_hash": parent.get("file_hashes", {}).get("summary")
        == protocol["frozen_parent"]["summary_sha256"],
        "monte_carlo_not_rerun": parent.get("monte_carlo_rerun") is False,
        "primary_adapter_frozen": ranking.get("primary_adapter")
        == protocol["measurement_adapters"]["primary"],
        "all_adapters_reported": set(adapters)
        == {
            protocol["measurement_adapters"]["primary"],
            *protocol["measurement_adapters"]["sensitivity"],
        },
        "claim_boundary_preserved": summary.get("claim_boundary", {}).get(
            "positive_physical_claim_allowed"
        )
        is False,
    }
    failures = [name for name, passed in checks.items() if not passed]
    result = {
        "schema": "uob100dy-scatty-native-grid-return-audit-v1",
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
