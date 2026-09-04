"""Verify exact return bytes and coverage; do not confer publication eligibility."""
import sys
sys.dont_write_bytecode = True
import argparse
from pathlib import Path
import tempfile
from acfo_storyline_package_io import audit, extract, load, sha, validate_request, write
from run_acfo_operator_storyline_external import plan


def verify(archive, request):
    archive, request = Path(archive), Path(request)
    sidecar = Path(str(archive)+".sha256")
    checks = {"sidecar": sidecar.is_file() and sidecar.read_text().split()[0] == sha(archive),
              "request_valid": validate_request(request)["passed"]}
    with tempfile.TemporaryDirectory(prefix="acfo_storyline_return_") as tmp:
        root = Path(tmp)/"return"
        extract(archive, root)
        inv = audit(root, "RETURN_MANIFEST.json")
        checks["exact_inventory"] = inv["passed"]
        summary = load(root/"outputs/summary.json")
        for name, key in (("PACKAGE_MANIFEST.json", "request_manifest_sha256"), ("PROTOCOL.json", "protocol_sha256"), ("CLAIM_FREEZE.json", "claim_freeze_sha256")):
            checks[key] = sha(root/name) == sha(request/name) == summary[key]
        profile = summary["profile"]
        checks["profile_valid"] = profile in ("local_check", "server36_full", "server59_replication")
        dummy = {k: root/k for k in ("extension", "waxs", "odt", "composite")}
        expected = [s["id"] for s in plan(profile, dummy, root/"outputs")]
        checks["exact_stage_coverage"] = expected == summary["expected_stage_ids"] == [r["id"] for r in summary["steps"]]
        checks["no_automatic_claim_promotion"] = summary["main_text_eligible"] is False
        checks["no_false_full_support"] = summary["full_original_grid_support_numerically_validated"] is False
        return {"schema": "acfo-storyline-return-verification-v2", "passed": all(checks.values()), "checks": checks,
            "archive_sha256": sha(archive), "inventory_errors": inv["errors"], "profile": profile,
            "technical_execution_completed": summary["technical_execution_completed"],
            "all_stage_acceptance_passed": summary["all_stage_acceptance_passed"],
            "system_specific_science_and_performance_review_required": profile != "local_check",
            "publication_eligible": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--request-root", type=Path, required=True)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    r = verify(a.archive, a.request_root)
    if a.output: write(a.output, r)
    print(__import__("json").dumps(r, indent=2))
    raise SystemExit(0 if r["passed"] else 2)
