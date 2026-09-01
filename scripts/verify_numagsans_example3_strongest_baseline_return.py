"""Fail-closed local verifier for a NuMagSANS Example 3 return archive."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RETURN_NAME = "numagsans_example3_strongest_baseline_external_return"
RETURN_MANIFEST_SCHEMA = (
    "numagsans-example3-strongest-baseline-return-manifest-v1"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}")
REQUIRED_AUDIT_CHECKS = {
    "schema",
    "prospective_protocol_status",
    "full_workload",
    "rtx3090",
    "source_archive_md5",
    "output_archive_md5",
    "source_topology",
    "affine_shape",
    "affine_counts",
    "affine_spacing",
    "affine_residual",
    "proper_lattice_invariance",
    "gpu_miller_margin",
    "packing_centers_excluded_from_qR",
    "accuracy_finished_before_timing",
    "production_gpu_arms_accuracy_before_timing",
    "type2_largest_passing_eps",
    "type3_largest_passing_eps",
    "complete_fft_frontier",
    "fft_qualification_accuracy_only",
    "screen_on_rtx3090",
    "screen_only_qualified_fft",
    "screen_fixed_samples",
    "fft_selected_by_frozen_screen_rule",
    "confirmation_arms",
    "confirmation_10_warmup_30_abba",
    "screen_samples_not_reused",
    "strongest_frozen_before_orientation2",
    "heldout_orientation_count",
    "heldout_accuracy",
    "full_ensemble_method_agreement",
    "archive_oracle_performed",
    "archive_accuracy",
    "all_prefixes_reported",
    "runner_gates_exclude_performance_sign",
    "runner_gates",
    "result_claim_eligibility_separate",
    "result_verdict_matches_signed_outcome",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_member(name: str) -> bool:
    if not name or "\\" in name:
        return False
    value = PurePosixPath(name)
    return bool(
        not value.is_absolute()
        and all(part not in ("", ".", "..") for part in value.parts)
    )


def _load_json(raw: bytes, label: str, failures: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        failures.append({"path": label, "reason": "invalid_json", "detail": repr(exc)})
        return {}
    if not isinstance(value, dict):
        failures.append({"path": label, "reason": "json_root_not_object"})
        return {}
    return value


def _read_sidecar(sidecar: Path, archive_name: str) -> tuple[str | None, bool]:
    lines = [
        line.strip()
        for line in sidecar.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != 1:
        return None, False
    match = re.fullmatch(r"([0-9A-Fa-f]{64})\s+\*?(\S+)", lines[0])
    if match is None or match.group(2) != archive_name:
        return None, False
    return match.group(1).lower(), True


def audit_return_archive(
    *,
    archive_path: Path,
    sidecar_path: Path,
    protocol_path: Path,
    request_manifest_path: Path,
    expected_request_sha256: str,
    expected_run_id: str,
    expected_mode: str,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    gates: dict[str, bool] = {}
    expected_request_sha256 = expected_request_sha256.strip().lower()
    gates["expected_request_sha256_format"] = bool(
        SHA256_PATTERN.fullmatch(expected_request_sha256)
    )
    gates["expected_run_id_format"] = bool(
        RUN_ID_PATTERN.fullmatch(expected_run_id)
    )
    gates["expected_mode"] = expected_mode == "full"
    gates["archive_exists"] = archive_path.is_file()
    gates["sidecar_exists"] = sidecar_path.is_file()
    gates["protocol_exists"] = protocol_path.is_file()
    gates["request_manifest_exists"] = request_manifest_path.is_file()

    archive_sha256: str | None = None
    sidecar_sha256: str | None = None
    if gates["archive_exists"]:
        archive_sha256 = sha256(archive_path)
    if gates["sidecar_exists"]:
        try:
            sidecar_sha256, sidecar_format = _read_sidecar(
                sidecar_path, archive_path.name
            )
        except Exception as exc:
            sidecar_format = False
            failures.append(
                {"path": str(sidecar_path), "reason": "sidecar_read", "detail": repr(exc)}
            )
        gates["sidecar_format_and_name"] = sidecar_format
    else:
        gates["sidecar_format_and_name"] = False
    gates["sidecar_matches_archive"] = bool(
        archive_sha256 is not None
        and sidecar_sha256 is not None
        and archive_sha256 == sidecar_sha256
    )

    protocol_raw = protocol_path.read_bytes() if protocol_path.is_file() else b""
    request_manifest_raw = (
        request_manifest_path.read_bytes() if request_manifest_path.is_file() else b""
    )
    protocol_sha256 = sha256_bytes(protocol_raw) if protocol_raw else None
    request_manifest_sha256 = (
        sha256_bytes(request_manifest_raw) if request_manifest_raw else None
    )
    manifest: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    embedded_protocol_raw = b""
    embedded_request_manifest_raw = b""
    manifest_raw = b""

    if archive_path.is_file():
        try:
            with zipfile.ZipFile(archive_path) as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                gates["zip_test"] = archive.testzip() is None
                gates["no_duplicate_members"] = len(names) == len(set(names))
                gates["members_are_safe_files"] = all(
                    not info.is_dir() and _safe_member(info.filename)
                    for info in infos
                )
                prefix = f"{RETURN_NAME}/"
                gates["single_expected_archive_root"] = all(
                    name.startswith(prefix) for name in names
                )
                manifest_member = f"{prefix}RETURN_MANIFEST.json"
                gates["return_manifest_present_once"] = (
                    names.count(manifest_member) == 1
                )
                if gates["return_manifest_present_once"]:
                    manifest_raw = archive.read(manifest_member)
                    manifest = _load_json(
                        manifest_raw, "RETURN_MANIFEST.json", failures
                    )
                declared = manifest.get("files")
                gates["manifest_files_object"] = isinstance(declared, dict)
                declared = declared if isinstance(declared, dict) else {}
                gates["declared_paths_safe"] = all(
                    isinstance(relative, str)
                    and _safe_member(relative)
                    and relative != "RETURN_MANIFEST.json"
                    for relative in declared
                )
                actual_payload = {
                    name.removeprefix(prefix)
                    for name in names
                    if name.startswith(prefix) and name != manifest_member
                }
                gates["exact_member_set"] = bool(
                    gates["single_expected_archive_root"]
                    and actual_payload == set(declared)
                )
                for relative, record in declared.items():
                    if not isinstance(relative, str) or not _safe_member(relative):
                        continue
                    if not isinstance(record, dict):
                        failures.append(
                            {"path": relative, "reason": "manifest_record_not_object"}
                        )
                        continue
                    member = f"{prefix}{relative}"
                    try:
                        raw = archive.read(member)
                    except KeyError:
                        failures.append({"path": relative, "reason": "missing"})
                        continue
                    if len(raw) != record.get("bytes"):
                        failures.append({"path": relative, "reason": "bytes"})
                    if sha256_bytes(raw) != record.get("sha256"):
                        failures.append({"path": relative, "reason": "sha256"})
                gates["declared_file_hashes"] = not failures
                required = {
                    "evidence/SUMMARY.json",
                    "evidence/AUDIT.json",
                    protocol_path.name,
                    "PROSPECTIVE_PROTOCOL_AUDIT.json",
                    "REQUEST_MANIFEST.json",
                }
                gates["required_files_declared"] = required.issubset(declared)
                if gates["required_files_declared"]:
                    summary = _load_json(
                        archive.read(f"{prefix}evidence/SUMMARY.json"),
                        "evidence/SUMMARY.json",
                        failures,
                    )
                    audit = _load_json(
                        archive.read(f"{prefix}evidence/AUDIT.json"),
                        "evidence/AUDIT.json",
                        failures,
                    )
                    embedded_protocol_raw = archive.read(
                        f"{prefix}{protocol_path.name}"
                    )
                    embedded_request_manifest_raw = archive.read(
                        f"{prefix}REQUEST_MANIFEST.json"
                    )
        except Exception as exc:
            gates["zip_test"] = False
            failures.append(
                {"path": str(archive_path), "reason": "zip_read", "detail": repr(exc)}
            )
    else:
        gates["zip_test"] = False

    gates["return_manifest_schema"] = (
        manifest.get("schema") == RETURN_MANIFEST_SCHEMA
    )
    gates["request_archive_binding"] = (
        manifest.get("request_archive_sha256") == expected_request_sha256
    )
    gates["request_manifest_binding"] = bool(
        request_manifest_sha256 is not None
        and manifest.get("request_manifest_sha256") == request_manifest_sha256
        and embedded_request_manifest_raw == request_manifest_raw
    )
    gates["protocol_binding"] = bool(
        protocol_sha256 is not None
        and manifest.get("protocol_sha256") == protocol_sha256
        and embedded_protocol_raw == protocol_raw
    )
    gates["run_id_binding"] = manifest.get("run_id") == expected_run_id
    gates["mode_binding"] = manifest.get("mode") == expected_mode
    gates["summary_schema"] = (
        summary.get("schema")
        == "numagsans-example3-strongest-baseline-external-summary-v1"
    )
    gates["summary_full_pass"] = bool(
        summary.get("mode") == "full" and summary.get("verdict") == "PASS"
    )
    summary_binding = summary.get("run_binding", {})
    gates["summary_binding"] = bool(
        summary_binding.get("request_archive_sha256") == expected_request_sha256
        and summary_binding.get("request_manifest_sha256")
        == request_manifest_sha256
        and summary_binding.get("protocol_sha256") == protocol_sha256
        and summary_binding.get("run_id") == expected_run_id
    )
    gates["manifest_summary_verdict_match"] = (
        manifest.get("verdict") == summary.get("verdict") == "PASS"
    )
    steps = summary.get("steps")
    gates["all_external_steps_passed"] = bool(
        isinstance(steps, list)
        and steps
        and all(isinstance(step, dict) and step.get("pass") is True for step in steps)
    )
    gates["audit_schema"] = (
        audit.get("schema") == "numagsans-example3-strongest-baseline-audit-v1"
    )
    audit_checks = audit.get("checks")
    gates["audit_required_checks"] = bool(
        isinstance(audit_checks, dict)
        and REQUIRED_AUDIT_CHECKS.issubset(audit_checks)
        and all(value is True for value in audit_checks.values())
    )
    positive_eligible = audit.get("acfo_positive_claim_eligible")
    gates["audit_signed_outcome"] = bool(
        audit.get("integrity_passed") is True
        and isinstance(positive_eligible, bool)
        and audit.get("verdict")
        == (
            "PASS_STRONGEST_BASELINE_VALIDATED"
            if positive_eligible
            else "PASS_STRONGEST_BASELINE_VALIDATED_NO_GO"
        )
    )
    gates["summary_embeds_same_audit"] = summary.get("audit") == audit
    integrity_gate_names = {
        name
        for name in gates
        if name
        not in {
            "summary_full_pass",
            "manifest_summary_verdict_match",
            "all_external_steps_passed",
            "audit_required_checks",
            "audit_signed_outcome",
            "summary_embeds_same_audit",
        }
    }
    integrity_passed = bool(
        not failures and all(gates[name] for name in integrity_gate_names)
    )
    scientific_closure_passed = bool(
        integrity_passed and not failures and all(gates.values())
    )
    return {
        "schema": "numagsans-example3-strongest-baseline-local-return-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS_RETURN_VALIDATED" if scientific_closure_passed else "FAIL",
        "archive": str(archive_path),
        "archive_sha256": archive_sha256,
        "return_manifest_sha256": (
            sha256_bytes(manifest_raw) if manifest_raw else None
        ),
        "expected": {
            "request_archive_sha256": expected_request_sha256,
            "request_manifest_sha256": request_manifest_sha256,
            "protocol_sha256": protocol_sha256,
            "run_id": expected_run_id,
            "mode": expected_mode,
        },
        "bindings": {
            "request_archive_sha256": manifest.get("request_archive_sha256"),
            "request_manifest_sha256": manifest.get("request_manifest_sha256"),
            "protocol_sha256": manifest.get("protocol_sha256"),
            "run_id": manifest.get("run_id"),
            "mode": manifest.get("mode"),
            "verdict": manifest.get("verdict"),
        },
        "integrity_passed": integrity_passed,
        "scientific_closure_passed": scientific_closure_passed,
        "acfo_positive_claim_eligible": (
            positive_eligible if isinstance(positive_eligible, bool) else None
        ),
        "gates": gates,
        "failures": failures,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--archive", type=Path, required=True)
    value.add_argument("--sidecar", type=Path, required=True)
    value.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "NUMAGSANS_EXAMPLE3_STRONGEST_BASELINE_PROTOCOL.json",
    )
    value.add_argument("--request-manifest", type=Path, required=True)
    value.add_argument("--expected-request-sha256", required=True)
    value.add_argument("--expected-run-id", required=True)
    value.add_argument("--expected-mode", choices=("full",), default="full")
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    receipt = audit_return_archive(
        archive_path=args.archive,
        sidecar_path=args.sidecar,
        protocol_path=args.protocol,
        request_manifest_path=args.request_manifest,
        expected_request_sha256=args.expected_request_sha256,
        expected_run_id=args.expected_run_id,
        expected_mode=args.expected_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["scientific_closure_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
