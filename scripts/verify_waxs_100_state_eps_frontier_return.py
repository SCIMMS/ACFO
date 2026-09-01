"""Fail-closed local verifier for the WAXS epsilon-frontier return."""

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
RETURN_NAME = "waxs_100_state_eps_frontier_external_return"
RETURN_ARCHIVE_NAME = f"{RETURN_NAME}.zip"
RETURN_MANIFEST_SCHEMA = "waxs-affine-library-eps-frontier-return-manifest-v2"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}")
SAFE_PART_PATTERN = re.compile(r"[A-Za-z0-9._-]+")
REQUIRED_EXTERNAL_AUDIT_GATES = {
    "schema",
    "mode",
    "protocol_sha256",
    "independent_machine",
    "epsilon_order",
    "no_frontier_timing_fields",
    "selected_is_first_passing",
    "accuracy_pass",
    "harmonic_accuracy_pass",
    "fused_type3_n_trans_8",
    "selected_eps_retimed",
    "abba",
    "warmups",
    "raw_samples",
    "full_external_contract",
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
        and all(
            part not in ("", ".", "..")
            and SAFE_PART_PATTERN.fullmatch(part) is not None
            for part in value.parts
        )
    )


def _load_json(
    raw: bytes, label: str, failures: list[dict[str, Any]]
) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        failures.append(
            {"path": label, "reason": "invalid_json", "detail": repr(exc)}
        )
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


def _read_remote_exit(path: Path) -> tuple[int | None, bool]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != 1 or re.fullmatch(r"[0-9]+", lines[0]) is None:
        return None, False
    value = int(lines[0])
    return value, value in range(0, 256)


def audit_return_archive(
    *,
    archive_path: Path,
    sidecar_path: Path,
    remote_exit_path: Path,
    protocol_path: Path,
    request_manifest_path: Path,
    expected_request_sha256: str,
    expected_request_manifest_sha256: str,
    expected_run_id: str,
    expected_mode: str,
    expected_remote_exit_status: int,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    gates: dict[str, bool] = {}
    expected_request_sha256 = expected_request_sha256.strip().lower()
    expected_request_manifest_sha256 = (
        expected_request_manifest_sha256.strip().lower()
    )
    gates["expected_request_sha256_format"] = bool(
        SHA256_PATTERN.fullmatch(expected_request_sha256)
    )
    gates["expected_manifest_sha256_format"] = bool(
        SHA256_PATTERN.fullmatch(expected_request_manifest_sha256)
    )
    gates["expected_run_id_format"] = bool(
        RUN_ID_PATTERN.fullmatch(expected_run_id)
    )
    gates["expected_mode"] = expected_mode == "full"
    gates["expected_remote_exit_status"] = expected_remote_exit_status in range(
        0, 256
    )
    for name, path in {
        "archive_exists": archive_path,
        "sidecar_exists": sidecar_path,
        "remote_exit_exists": remote_exit_path,
        "protocol_exists": protocol_path,
        "request_manifest_exists": request_manifest_path,
    }.items():
        gates[name] = path.is_file()

    archive_sha256 = sha256(archive_path) if archive_path.is_file() else None
    sidecar_sha256: str | None = None
    if sidecar_path.is_file():
        try:
            sidecar_sha256, sidecar_valid = _read_sidecar(
                sidecar_path, archive_path.name
            )
        except Exception as exc:
            sidecar_valid = False
            failures.append(
                {"path": str(sidecar_path), "reason": "sidecar", "detail": repr(exc)}
            )
    else:
        sidecar_valid = False
    gates["sidecar_format_and_name"] = sidecar_valid
    gates["sidecar_matches_archive"] = bool(
        archive_sha256 is not None
        and sidecar_sha256 is not None
        and archive_sha256 == sidecar_sha256
    )

    remote_exit_status: int | None = None
    if remote_exit_path.is_file():
        try:
            remote_exit_status, remote_exit_valid = _read_remote_exit(
                remote_exit_path
            )
        except Exception as exc:
            remote_exit_valid = False
            failures.append(
                {
                    "path": str(remote_exit_path),
                    "reason": "remote_exit",
                    "detail": repr(exc),
                }
            )
    else:
        remote_exit_valid = False
    gates["remote_exit_format"] = remote_exit_valid
    gates["remote_exit_matches_ssh"] = (
        remote_exit_status == expected_remote_exit_status
    )

    protocol_raw = protocol_path.read_bytes() if protocol_path.is_file() else b""
    request_manifest_raw = (
        request_manifest_path.read_bytes() if request_manifest_path.is_file() else b""
    )
    protocol_sha256 = sha256_bytes(protocol_raw) if protocol_raw else None
    request_manifest_sha256 = (
        sha256_bytes(request_manifest_raw) if request_manifest_raw else None
    )
    gates["local_manifest_matches_expected"] = bool(
        request_manifest_sha256 is not None
        and request_manifest_sha256 == expected_request_manifest_sha256
    )
    local_request_manifest = (
        _load_json(request_manifest_raw, str(request_manifest_path), failures)
        if request_manifest_raw
        else {}
    )
    gates["local_request_manifest_schema"] = (
        local_request_manifest.get("schema")
        == "waxs-affine-library-eps-frontier-request-manifest-v1"
    )
    gates["local_request_manifest_protocol"] = (
        local_request_manifest.get("protocol_sha256") == protocol_sha256
    )

    manifest: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    result: dict[str, Any] = {}
    manifest_raw = b""
    audit_raw = b""
    result_raw = b""
    embedded_protocol_raw = b""
    embedded_request_manifest_raw = b""
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
                file_failures_before = len(failures)
                for relative, record in declared.items():
                    if not isinstance(relative, str) or not _safe_member(relative):
                        continue
                    if not isinstance(record, dict):
                        failures.append(
                            {"path": relative, "reason": "manifest_record_not_object"}
                        )
                        continue
                    try:
                        raw = archive.read(f"{prefix}{relative}")
                    except KeyError:
                        failures.append({"path": relative, "reason": "missing"})
                        continue
                    if len(raw) != record.get("bytes"):
                        failures.append({"path": relative, "reason": "bytes"})
                    if sha256_bytes(raw) != record.get("sha256"):
                        failures.append({"path": relative, "reason": "sha256"})
                gates["declared_file_hashes"] = (
                    len(failures) == file_failures_before
                )
                required = {
                    "evidence/SUMMARY.json",
                    "evidence/EXTERNAL_RETURN_AUDIT.json",
                    "evidence/waxs_100_state_eps_frontier.json",
                    protocol_path.name,
                    "REQUEST_PACKAGE_MANIFEST.json",
                }
                gates["required_files_declared"] = required.issubset(declared)
                if gates["required_files_declared"]:
                    summary = _load_json(
                        archive.read(f"{prefix}evidence/SUMMARY.json"),
                        "evidence/SUMMARY.json",
                        failures,
                    )
                    audit_raw = archive.read(
                        f"{prefix}evidence/EXTERNAL_RETURN_AUDIT.json"
                    )
                    audit = _load_json(
                        audit_raw, "evidence/EXTERNAL_RETURN_AUDIT.json", failures
                    )
                    result_raw = archive.read(
                        f"{prefix}evidence/waxs_100_state_eps_frontier.json"
                    )
                    result = _load_json(
                        result_raw,
                        "evidence/waxs_100_state_eps_frontier.json",
                        failures,
                    )
                    embedded_protocol_raw = archive.read(
                        f"{prefix}{protocol_path.name}"
                    )
                    embedded_request_manifest_raw = archive.read(
                        f"{prefix}REQUEST_PACKAGE_MANIFEST.json"
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
    gates["verdict_binding"] = manifest.get("verdict") == "PASS"
    gates["manifest_remote_exit_binding"] = bool(
        manifest.get("runner_exit_status")
        == remote_exit_status
        == expected_remote_exit_status
        == 0
    )

    gates["summary_schema"] = (
        summary.get("schema")
        == "waxs-affine-library-eps-frontier-external-summary-v2"
    )
    gates["summary_full_pass"] = bool(
        summary.get("mode") == "full"
        and summary.get("verdict") == "PASS"
        and summary.get("passed") is True
    )
    summary_binding = summary.get("run_binding", {})
    gates["summary_binding"] = bool(
        summary_binding.get("request_archive_sha256") == expected_request_sha256
        and summary_binding.get("request_manifest_sha256")
        == request_manifest_sha256
        and summary_binding.get("protocol_sha256") == protocol_sha256
        and summary_binding.get("run_id") == expected_run_id
    )
    gates["summary_remote_exit_binding"] = (
        summary.get("runner_exit_status") == remote_exit_status == 0
    )
    gates["summary_evidence_hashes"] = bool(
        audit_raw
        and result_raw
        and summary.get("external_audit_sha256") == sha256_bytes(audit_raw)
        and summary.get("result_sha256") == sha256_bytes(result_raw)
    )
    gates["summary_return_names"] = bool(
        summary.get("return_archive") == RETURN_ARCHIVE_NAME
        and summary.get("return_sidecar") == f"{RETURN_ARCHIVE_NAME}.sha256"
    )
    gates["manifest_summary_verdict_match"] = (
        manifest.get("verdict") == summary.get("verdict") == "PASS"
    )

    external_gates = audit.get("gates")
    gates["external_audit_schema"] = (
        audit.get("schema")
        == "waxs-affine-library-eps-frontier-external-audit-v1"
    )
    gates["external_audit_contract"] = bool(
        audit.get("passed") is True
        and audit.get("performance_sign_is_gate") is False
        and isinstance(external_gates, dict)
        and REQUIRED_EXTERNAL_AUDIT_GATES.issubset(external_gates)
        and all(value is True for value in external_gates.values())
        and audit.get("package_audit_passed") is True
        and audit.get("process_status")
        == {"build": 0, "tests": 0, "benchmark": 0}
    )

    timing = result.get("final_timing") or {}
    gates["result_schema"] = (
        result.get("schema") == "waxs-affine-library-eps-frontier-result-v1"
    )
    gates["result_full_mode"] = result.get("mode") == "full"
    gates["result_protocol_sha256"] = (
        result.get("protocol", {}).get("sha256") == protocol_sha256
    )
    gates["accuracy_first"] = (
        result.get("epsilon_selection", {}).get("timing_used") is False
    )
    gates["frontier_pass"] = (
        result.get("epsilon_selection", {}).get("accuracy_pass") is True
    )
    gates["harmonic_accuracy_pass"] = (
        result.get("acfo_harmonic_accuracy", {}).get("accuracy_pass") is True
    )
    gates["fused_type3_n_trans_8"] = bool(
        timing.get("finufft_plan_type") == 3
        and timing.get("finufft_n_trans") == 8
    )
    gates["ten_warmups_thirty_samples"] = bool(
        timing.get("warmup_count") == 10
        and timing.get("samples_per_arm") == 30
        and len(timing.get("acfo_seconds", {}).get("samples_s", [])) == 30
        and len(timing.get("baseline_seconds", {}).get("samples_s", [])) == 30
    )
    gates["external_contract_complete"] = (
        result.get("decision", {}).get("external_contract_complete") is True
    )
    gates["performance_sign_not_integrity_gate"] = (
        result.get("decision", {}).get("speed_sign_is_integrity_gate") is False
    )

    passed = bool(not failures and all(gates.values()))
    return {
        "schema": "waxs-affine-library-eps-frontier-local-return-audit-v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS_RETURN_VALIDATED" if passed else "FAIL",
        "archive": str(archive_path),
        "archive_sha256": archive_sha256,
        "return_manifest_sha256": (
            sha256_bytes(manifest_raw) if manifest_raw else None
        ),
        "external_audit_sha256": sha256_bytes(audit_raw) if audit_raw else None,
        "result_sha256": sha256_bytes(result_raw) if result_raw else None,
        "remote_exit_status": remote_exit_status,
        "bindings": {
            "request_archive_sha256": manifest.get("request_archive_sha256"),
            "request_manifest_sha256": manifest.get("request_manifest_sha256"),
            "protocol_sha256": manifest.get("protocol_sha256"),
            "run_id": manifest.get("run_id"),
            "mode": manifest.get("mode"),
            "verdict": manifest.get("verdict"),
            "runner_exit_status": manifest.get("runner_exit_status"),
        },
        "gates": gates,
        "file_failures": failures,
        "passed": passed,
        "selected_eps": result.get("epsilon_selection", {}).get("selected_eps"),
        "speedup": timing.get("baseline_over_acfo_speedup"),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--archive", type=Path, required=True)
    value.add_argument("--sidecar", type=Path, required=True)
    value.add_argument("--remote-exit-code-file", type=Path, required=True)
    value.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "WAXS_100_STATE_EPS_FRONTIER_PROTOCOL.json",
    )
    value.add_argument("--request-manifest", type=Path, required=True)
    value.add_argument("--expected-request-sha256", required=True)
    value.add_argument("--expected-request-manifest-sha256", required=True)
    value.add_argument("--expected-run-id", required=True)
    value.add_argument("--expected-mode", choices=("full",), default="full")
    value.add_argument("--expected-remote-exit-status", type=int, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    receipt = audit_return_archive(
        archive_path=args.archive,
        sidecar_path=args.sidecar,
        remote_exit_path=args.remote_exit_code_file,
        protocol_path=args.protocol,
        request_manifest_path=args.request_manifest,
        expected_request_sha256=args.expected_request_sha256,
        expected_request_manifest_sha256=(
            args.expected_request_manifest_sha256
        ),
        expected_run_id=args.expected_run_id,
        expected_mode=args.expected_mode,
        expected_remote_exit_status=args.expected_remote_exit_status,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
