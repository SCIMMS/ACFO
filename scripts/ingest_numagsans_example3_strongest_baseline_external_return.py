"""Build the canonical NuMagSANS Example 3 external-return evidence record.

This ingestion step is intentionally narrower than the return verifier.  It
first re-runs the fail-closed archive audit, then projects the verified return
onto the small set of identities, accuracy gates, and timing quantities that
publication-facing evidence consumers need.  The output is deterministic: it
uses timestamps embedded in the frozen return and never records local absolute
paths.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_numagsans_example3_strongest_baseline_return import (
    RETURN_NAME,
    audit_return_archive,
)
from scripts.verify_request_package_manifest import audit_manifest


RUN_ID = (
    "numagsans_example3_strongest_baseline_438e2555ef28_"
    "20260825_200600_bd7dfd67307e42ea832c037c4628ded2"
)
RUN_ROOT = (
    ROOT
    / "benchmark_results"
    / "external_validation"
    / RUN_ID
)
DEFAULT_OUTPUT = (
    ROOT
    / "benchmark_results"
    / "numagsans_example3_strongest_baseline_external_evidence_20260825.json"
)

EXPECTED_SHA256 = {
    "request_archive": (
        "438e2555ef28f0f2132522b94601e2005053ba26ed14463e3f9c595fd72b013d"
    ),
    "request_build_receipt": (
        "d2d344fd2d5103dd8455823b438cdd48552e6a02c5788883b62c49e049c3c16f"
    ),
    "request_sidecar": (
        "879a78b8e1662b34853055e5602b82ff8c3d06f4d6e27482ebe3851843bcdb9f"
    ),
    "request_manifest": (
        "9c365f4c09d514e23d1f41ab5bac622e52c7941d9f6dc8771659e78a7098f062"
    ),
    "protocol": (
        "41bda823c1e2b6316511aba3bfc14486a5130a74c39d5857e235cc2bd90fcc48"
    ),
    "return_archive": (
        "657fce873703669e4702946e130b2c9eabcb9dd6241ebee6c7d596a661a3face"
    ),
    "return_sidecar": (
        "c90e56cb0327c8786d4b3ee99f3708a03ac8a42167346a140e027d77fac47973"
    ),
    "local_return_audit": (
        "48b43b5d3822196909856f5af0a512fd405d06840b24d153773f295174ed1f26"
    ),
    "remote_exit_code": (
        "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa"
    ),
    "return_manifest": (
        "e8b62da96bbbc269c72c9878d169963f02a14be81f004ccea2b7b0eb55bb85af"
    ),
    "summary": (
        "027ba0cdde3ce0f1e79aa182a198316a0bdcbfbf349d9c4e8bb9f6125629c57a"
    ),
    "audit": (
        "e4af46d4df090a731186001b28d746a12b82bad69c91e6a9d4f571a0eb8e66fa"
    ),
    "benchmark": (
        "43678e6d159046e4c1bb5c8f20943b42e1f274ac42187c73704fec58cc1b05f7"
    ),
}

RETURN_MEMBERS = {
    "return_manifest": "RETURN_MANIFEST.json",
    "summary": "evidence/SUMMARY.json",
    "audit": "evidence/AUDIT.json",
    "benchmark": "evidence/numagsans_example3_strongest_baseline.json",
}


@dataclass(frozen=True)
class IngestionInputs:
    request_archive: Path
    request_build_receipt: Path
    request_sidecar: Path
    request_manifest: Path
    request_package_root: Path
    protocol: Path
    return_archive: Path
    return_sidecar: Path
    local_return_audit: Path
    remote_exit_code: Path


def default_inputs() -> IngestionInputs:
    request_root = (
        ROOT / "reports" / "numagsans_example3_strongest_baseline_external_request_v1"
    )
    return_root = RUN_ROOT / "return"
    return IngestionInputs(
        request_archive=request_root.with_suffix(".zip"),
        request_build_receipt=(
            ROOT
            / "reports"
            / "numagsans_example3_strongest_baseline_external_request_v1_build_receipt.json"
        ),
        request_sidecar=request_root.with_suffix(".zip.sha256"),
        request_manifest=request_root / "MANIFEST.json",
        request_package_root=request_root,
        protocol=ROOT / "NUMAGSANS_EXAMPLE3_STRONGEST_BASELINE_PROTOCOL.json",
        return_archive=(
            return_root / "numagsans_example3_strongest_baseline_external_return.zip"
        ),
        return_sidecar=(
            return_root
            / "numagsans_example3_strongest_baseline_external_return.zip.sha256"
        ),
        local_return_audit=RUN_ROOT / "LOCAL_EXTERNAL_RETURN_AUDIT.json",
        remote_exit_code=return_root / "REMOTE_RUN_EXIT_CODE.txt",
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def _require_hash(path: Path, key: str) -> str:
    _require(path.is_file(), f"missing frozen input: {path}")
    actual = sha256(path)
    expected = EXPECTED_SHA256[key]
    _require(actual == expected, f"{key} sha256 mismatch: {actual} != {expected}")
    return actual


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _read_sidecar(path: Path, archive: Path) -> str:
    line = path.read_text(encoding="utf-8").strip()
    match = re.fullmatch(r"([0-9A-Fa-f]{64})\s+\*?(\S+)", line)
    _require(match is not None, f"invalid sidecar format: {path}")
    assert match is not None
    _require(match.group(2) == archive.name, f"sidecar name mismatch: {path}")
    value = match.group(1).lower()
    _require(value == sha256(archive), f"sidecar digest mismatch: {path}")
    return value


def _source_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _zip_json(
    archive: zipfile.ZipFile,
    relative: str,
) -> tuple[dict[str, Any], str]:
    member = f"{RETURN_NAME}/{relative}"
    raw = archive.read(member)
    value = json.loads(raw.decode("utf-8"))
    _require(isinstance(value, dict), f"ZIP JSON root is not an object: {relative}")
    return value, sha256_bytes(raw)


def _semantic_local_audit(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "schema",
            "verdict",
            "archive_sha256",
            "return_manifest_sha256",
            "expected",
            "bindings",
            "integrity_passed",
            "scientific_closure_passed",
            "acfo_positive_claim_eligible",
            "gates",
            "failures",
        )
    }


def build_canonical_evidence(inputs: IngestionInputs) -> dict[str, Any]:
    input_hashes = {
        key: _require_hash(getattr(inputs, key), key)
        for key in (
            "request_archive",
            "request_build_receipt",
            "request_sidecar",
            "request_manifest",
            "protocol",
            "return_archive",
            "return_sidecar",
            "local_return_audit",
            "remote_exit_code",
        )
    }
    _require(
        _read_sidecar(inputs.request_sidecar, inputs.request_archive)
        == EXPECTED_SHA256["request_archive"],
        "request sidecar does not bind the frozen request",
    )
    _require(
        _read_sidecar(inputs.return_sidecar, inputs.return_archive)
        == EXPECTED_SHA256["return_archive"],
        "return sidecar does not bind the frozen return",
    )

    package_audit = audit_manifest(
        inputs.request_package_root,
        inputs.request_manifest,
        require_exact=True,
    )
    _require(package_audit["passed"] is True, "request package manifest audit failed")
    _require(package_audit["file_count"] == 78, "request manifest file count changed")

    build_receipt = _load_json(inputs.request_build_receipt)
    _require(
        build_receipt["archive_sha256"] == EXPECTED_SHA256["request_archive"],
        "request build receipt archive binding failed",
    )
    _require(
        build_receipt["manifest_sha256"] == EXPECTED_SHA256["request_manifest"],
        "request build receipt manifest binding failed",
    )
    _require(
        build_receipt["protocol_sha256"] == EXPECTED_SHA256["protocol"],
        "request build receipt protocol binding failed",
    )
    _require(build_receipt["manifest_files"] == 78, "build receipt file count changed")

    request_manifest = _load_json(inputs.request_manifest)
    _require(
        request_manifest["schema"]
        == "numagsans-example3-strongest-baseline-request-manifest-v1",
        "request manifest schema changed",
    )
    _require(len(request_manifest["files"]) == 78, "request manifest changed")
    _require(
        request_manifest["files"][inputs.protocol.name]["sha256"]
        == EXPECTED_SHA256["protocol"],
        "request manifest protocol binding failed",
    )

    remote_exit_code = inputs.remote_exit_code.read_text(encoding="utf-8").strip()
    _require(remote_exit_code == "0", "remote runner exit code was not zero")

    recomputed_audit = audit_return_archive(
        archive_path=inputs.return_archive,
        sidecar_path=inputs.return_sidecar,
        protocol_path=inputs.protocol,
        request_manifest_path=inputs.request_manifest,
        expected_request_sha256=EXPECTED_SHA256["request_archive"],
        expected_run_id=RUN_ID,
        expected_mode="full",
    )
    _require(
        recomputed_audit["scientific_closure_passed"] is True,
        "independent return audit did not close scientifically",
    )
    saved_local_audit = _load_json(inputs.local_return_audit)
    _require(
        _semantic_local_audit(saved_local_audit)
        == _semantic_local_audit(recomputed_audit),
        "saved and recomputed local-return audits differ",
    )

    embedded: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(inputs.return_archive) as archive:
        _require(archive.testzip() is None, "return ZIP CRC failed")
        for key, relative in RETURN_MEMBERS.items():
            payload, digest = _zip_json(archive, relative)
            _require(
                digest == EXPECTED_SHA256[key],
                f"embedded {key} sha256 mismatch",
            )
            embedded[key] = payload

    return_manifest = embedded["return_manifest"]
    summary = embedded["summary"]
    audit = embedded["audit"]
    benchmark = embedded["benchmark"]

    _require(return_manifest["verdict"] == "PASS", "return manifest is not PASS")
    _require(summary["verdict"] == "PASS", "external summary is not PASS")
    _require(
        audit["verdict"] == "PASS_STRONGEST_BASELINE_VALIDATED",
        "scientific audit is not a positive PASS",
    )
    _require(
        benchmark["verdict"] == "PASS_STRONGEST_BASELINE_CLOSURE",
        "benchmark result did not close",
    )
    _require(
        benchmark["acfo_positive_claim_eligible"] is True,
        "ACFO positive claim is not eligible",
    )
    _require(
        all(benchmark["gates"].values()),
        "one or more benchmark contract gates failed",
    )

    workload = benchmark["workload"]
    _require(workload["orientations"] == 800, "orientation count changed")
    _require(workload["packing_cases"] == 5, "packing case count changed")
    _require(workload["targets"] == 999000, "target count changed")
    qualification = benchmark["accuracy_only_qualification"]
    _require(
        qualification["selection_completed_before_timing"] is True,
        "accuracy selection was not completed before timing",
    )
    _require(
        qualification["tested_fft_pad_factors"]
        == qualification["declared_fft_pad_factors"]
        == [4, 8, 12, 16, 20, 24, 28, 32],
        "full frozen FFT pad frontier was not completed",
    )
    _require(
        qualification["qualified_fft_pad_factors"] == [],
        "unexpected FFT arm qualified",
    )
    _require(
        qualification["resource_limited_frontier"] is False,
        "full frontier was resource-limited",
    )

    confirmation = benchmark["independent_confirmation"]
    selected = confirmation["strongest_baseline_selected_before_orientation_2"]
    _require(selected == "affine_type2", "strongest baseline selection changed")
    _require(
        confirmation["screen_sample_reuse_count"] == 0,
        "screen samples leaked into confirmation",
    )
    selected_confirmation = confirmation["comparisons"][selected]
    _require(
        selected_confirmation["warmups_per_arm"] == 10
        and selected_confirmation["samples_per_arm"] == 30
        and selected_confirmation["order_contract"] == "ABBA",
        "independent timing contract changed",
    )

    full_ratio = benchmark["cold_total_speedup_selected_baseline_over_acfo"]
    _require(full_ratio == audit["full_speed_ratio"], "full speed ratio audit mismatch")
    _require(full_ratio["lower_95"] > 1.0, "positive speed CI did not clear unity")

    worst_errors = audit["worst_errors"]
    _require(
        worst_errors["heldout_amplitude"]
        == benchmark["heldout_accuracy"]["worst_amplitude_relative_l2"],
        "held-out amplitude error mismatch",
    )
    _require(
        worst_errors["heldout_output"]
        == benchmark["heldout_accuracy"]["worst_output_relative_l2"],
        "held-out output error mismatch",
    )
    _require(
        worst_errors["full_ensemble"]
        == benchmark["full_ensemble_acfo_vs_selected_baseline"]["worst_relative_l2"],
        "full-ensemble error mismatch",
    )

    source_records = {
        key: {
            "path": _source_path(getattr(inputs, key)),
            "bytes": getattr(inputs, key).stat().st_size,
            "sha256": digest,
        }
        for key, digest in input_hashes.items()
    }
    embedded_records = {
        key: {
            "member": f"{RETURN_NAME}/{RETURN_MEMBERS[key]}",
            "sha256": EXPECTED_SHA256[key],
        }
        for key in RETURN_MEMBERS
    }

    return {
        "schema": (
            "numagsans-example3-strongest-baseline-canonical-external-evidence-v1"
        ),
        "source_return_created_utc": return_manifest["created_utc"],
        "evidence_id": "NUMAGSANS_EXAMPLE3_ROTATED_LATTICE_TYPE2",
        "system": "NuMagSANS Example 3",
        "validation": {
            "status": "PASS_EXTERNAL_STRONGEST_BASELINE_VALIDATED",
            "integrity_passed": True,
            "scientific_closure_passed": True,
            "acfo_positive_claim_eligible": True,
            "remote_runner_exit_code": 0,
            "all_return_gates_passed": all(recomputed_audit["gates"].values()),
            "request_package_manifest_exact": package_audit["passed"],
        },
        "bindings": {
            "run_id": RUN_ID,
            "mode": "full",
            "request_archive_sha256": EXPECTED_SHA256["request_archive"],
            "request_manifest_sha256": EXPECTED_SHA256["request_manifest"],
            "protocol_sha256": EXPECTED_SHA256["protocol"],
            "return_archive_sha256": EXPECTED_SHA256["return_archive"],
            "return_manifest_sha256": EXPECTED_SHA256["return_manifest"],
        },
        "source_artifacts": source_records,
        "embedded_artifacts": embedded_records,
        "workload": {
            "orientations": workload["orientations"],
            "packing_cases": workload["packing_cases"],
            "sources_per_orientation": workload["sources_per_orientation"],
            "q_nodes": workload["q_nodes"],
            "unique_theta": workload["unique_theta"],
            "targets": workload["targets"],
            "streaming": workload["streaming"],
            "machine": benchmark["environment"]["gpu"],
            "dtype": benchmark["environment"]["dtype"],
        },
        "baseline_selection": {
            "strongest_eligible_baseline": selected,
            "selected_type2_eps": qualification["selected_type2_eps"],
            "selected_type3_eps": qualification["selected_type3_eps"],
            "declared_fft_pad_factors": qualification["declared_fft_pad_factors"],
            "tested_fft_pad_factors": qualification["tested_fft_pad_factors"],
            "qualified_fft_pad_factors": qualification["qualified_fft_pad_factors"],
            "accuracy_selection_completed_before_timing": True,
            "screen_confirmation_sample_overlap": confirmation[
                "screen_confirmation_sample_overlap"
            ],
            "screen_sample_reuse_count": confirmation["screen_sample_reuse_count"],
            "frozen_before_orientation_2": (
                benchmark["strongest_baseline_frozen_before_orientation_2"]
            ),
        },
        "timing": {
            "metric": "selected-baseline/ACFO cold-total ratio",
            "full_800_orientation_speed_ratio": full_ratio,
            "independent_confirmation_speed_ratio": selected_confirmation[
                "baseline_over_acfo_speed_ratio"
            ],
            "independent_confirmation_contract": {
                "order": selected_confirmation["order_contract"],
                "warmups_per_arm": selected_confirmation["warmups_per_arm"],
                "samples_per_arm": selected_confirmation["samples_per_arm"],
            },
            "measured_crossover_orientations": benchmark["measured_crossover"],
        },
        "accuracy": {
            "heldout_orientation_count": len(benchmark["heldout_accuracy"]["rows"]),
            "worst_heldout_amplitude_relative_l2": worst_errors[
                "heldout_amplitude"
            ],
            "worst_heldout_output_relative_l2": worst_errors["heldout_output"],
            "worst_full_ensemble_relative_l2": worst_errors["full_ensemble"],
            "archive_oracle_performed": benchmark["archived_output_validation"][
                "performed"
            ],
            "worst_acfo_archive_relative_l2": worst_errors["acfo_archive"],
            "worst_selected_baseline_archive_relative_l2": worst_errors[
                "selected_archive"
            ],
        },
        "harmonic_support": benchmark["harmonic_support"],
        "claim": {
            "status": "READY_EXTERNAL_POSITIVE_CLAIM",
            "headline_eligible": True,
            "point_estimate": full_ratio["point"],
            "ci_lower_95": full_ratio["lower_95"],
            "ci_upper_95": full_ratio["upper_95"],
            "comparator": "accuracy-qualified reusable affine 3-D Type-2",
            "supersedes": "NUMAGSANS_EXAMPLE3_PROJECTED_TYPE3_800X5",
            "claim_boundary": (
                "hardware-, dtype-, implementation-, and frozen-workload-specific; "
                "not a universal speedup claim"
            ),
        },
    }


def write_evidence(output: Path, payload: dict[str, Any]) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    digest = sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n",
        encoding="utf-8",
    )
    return digest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--check-only", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    payload = build_canonical_evidence(default_inputs())
    if args.check_only:
        _require(args.output.is_file(), f"missing canonical evidence: {args.output}")
        current = json.loads(args.output.read_text(encoding="utf-8"))
        _require(current == payload, "canonical evidence is stale")
        sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
        _require(sidecar.is_file(), f"missing canonical evidence sidecar: {sidecar}")
        _require(
            _read_sidecar(sidecar, args.output) == sha256(args.output),
            "canonical evidence sidecar mismatch",
        )
        digest = sha256(args.output)
    else:
        digest = write_evidence(args.output, payload)
    print(
        json.dumps(
            {
                "status": "PASS",
                "check_only": args.check_only,
                "output": str(args.output),
                "sha256": digest,
                "claim_status": payload["claim"]["status"],
                "speed_ratio": payload["timing"][
                    "full_800_orientation_speed_ratio"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
