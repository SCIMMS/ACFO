#!/usr/bin/env python3
"""Audit evidence returned by the frozen UoB-100(Dy) atomic package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_HASHES = {
    "spin_up": "a9979b91f8cbfdf1f75b6ed6bb97dea157212acb153a2cd2c04c5dbff15b4c7a",
    "spin_down": "3df554117f96a3ebb5c22a484fe154a3ca8e9df56d58ab182576e0596294966b",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "score"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    checks: dict[str, bool]
    if args.mode == "preflight":
        inventory = json.loads(
            (args.evidence / "zip_inventory.json").read_text(encoding="utf-8")
        )
        receipt = json.loads(
            (args.evidence / "files_description_receipt.json").read_text(encoding="utf-8")
        )
        source = protocol["source"]
        checks = {
            "archive_bytes": int(inventory["total_remote_bytes"])
            == int(source["archive_bytes"]),
            "all_member_paths_safe": int(inventory["unsafe_entry_count"]) == 0,
            "frozen_nexus_member_present_once": sum(
                entry["name"] == source["member"] for entry in inventory["entries"]
            )
            == 1,
            "description_crc32": receipt["crc32"] == "0f9ae259",
            "description_bytes": int(receipt["uncompressed_bytes"]) == 1207,
        }
    else:
        summary = json.loads((args.evidence / "summary.json").read_text(encoding="utf-8"))
        geometry = summary.get("geometry", {})
        monte_carlo = summary.get("monte_carlo", {})
        checks = {
            "summary_schema": summary.get("schema")
            == "uob100dy-atomic-decoration-validation-summary-v1",
            "summary_pass": summary.get("status") == "PASS",
            "all_technical_gates": all(summary.get("technical_gates", {}).values()),
            "prediction_file_present": (args.evidence / "predictions.npz").is_file(),
            "geometry_hashes": geometry.get("canonical_hashes") == EXPECTED_HASHES,
            "atoms_per_state": int(geometry.get("atoms_per_state", -1)) == 59,
            "schedule_points": len(monte_carlo.get("schedule_J_over_T", [])) == 23,
            "samples_per_temperature": int(
                monte_carlo.get("samples_per_temperature", -1)
            )
            == 80,
            "claim_boundary_preserved": summary.get("claim_boundary", {}).get(
                "positive_physical_claim_allowed"
            )
            is False,
        }
    failures = [name for name, passed in checks.items() if not passed]
    result = {
        "schema": "uob100dy-atomic-decoration-return-audit-v1",
        "mode": args.mode,
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
