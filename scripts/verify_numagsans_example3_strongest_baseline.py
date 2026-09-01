"""Fail-closed audit for the Example 3 strongest-baseline closure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _first_passing(rows: list[dict[str, Any]], key: str) -> Any:
    return next((row[key] for row in rows if row.get("pass") is True), None)


def audit_decision(
    checks: dict[str, bool],
    *,
    speed_lower_95: float,
    required_speed_lower_95: float,
) -> dict[str, Any]:
    """Keep signed performance eligibility out of scientific integrity."""

    integrity_passed = all(checks.values())
    acfo_positive_claim_eligible = bool(
        integrity_passed and speed_lower_95 >= required_speed_lower_95
    )
    if not integrity_passed:
        verdict = "FAIL"
    elif acfo_positive_claim_eligible:
        verdict = "PASS_STRONGEST_BASELINE_VALIDATED"
    else:
        verdict = "PASS_STRONGEST_BASELINE_VALIDATED_NO_GO"
    return {
        "integrity_passed": integrity_passed,
        "acfo_positive_claim_eligible": acfo_positive_claim_eligible,
        "verdict": verdict,
    }


def run(result_path: Path, protocol_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    checks["schema"] = (
        result.get("schema")
        == "numagsans-example3-strongest-baseline-result-v1"
    )
    checks["prospective_protocol_status"] = (
        result.get("protocol_status") == protocol["status"]
        and protocol["status"].startswith("PROSPECTIVE_FROZEN_BEFORE_RTX3090")
    )
    checks["full_workload"] = bool(
        result.get("mode") == "full"
        and result["workload"]["orientations"]
        == protocol["dataset"]["orientation_count"]
        and result["workload"]["packing_cases"] == 5
        and result["workload"]["targets"] == protocol["workload"]["targets"]
    )
    checks["rtx3090"] = (
        result["environment"]["gpu"]
        == protocol["external_machine"]["gpu_required"]
    )
    checks["source_archive_md5"] = (
        result["provenance"].get("source_archive_md5")
        == protocol["dataset"]["source_archive"]["md5"]
    )
    checks["output_archive_md5"] = (
        result["provenance"].get("output_archive_md5")
        == protocol["dataset"]["output_archive"]["md5"]
    )
    checks["source_topology"] = (
        result["provenance"]["source_topology"]["pass"] is True
    )

    lattice = result["affine_lattice_certificate_orientation_1"]
    lattice_spec = protocol["affine_lattice_contract"]
    checks["affine_shape"] = lattice["shape"] == lattice_spec["expected_shape"]
    checks["affine_counts"] = bool(
        lattice["active_sites"] == lattice_spec["expected_active_sites"]
        and lattice["dense_sites"] == lattice_spec["expected_dense_sites"]
    )
    checks["affine_spacing"] = bool(
        np.allclose(
            lattice["spacing"], lattice_spec["expected_spacing_nm"], atol=1e-10
        )
    )
    checks["affine_residual"] = (
        lattice["maximum_coordinate_residual"]
        <= lattice_spec["coordinate_reconstruction_tolerance_nm"]
    )
    checks["proper_lattice_invariance"] = all(
        row["pass"] is True
        and row["indices_equal"] is True
        and row["rigid_map_determinant"] > 0.0
        for row in result["accuracy_only_qualification"][
            "orientation_lattice_invariance"
        ]
    )

    support = result["harmonic_support"]
    checks["gpu_miller_margin"] = bool(
        support["kernel_backend"]
        == protocol["harmonic_support"]["required_kernel_backend"]
        and support["miller_recurrence_margin"]
        == protocol["harmonic_support"]["miller_recurrence_margin"]
    )
    checks["packing_centers_excluded_from_qR"] = (
        support["packing_centers_entered_qR"] is False
    )

    qualification = result["accuracy_only_qualification"]
    rows = qualification["rows"]
    checks["accuracy_finished_before_timing"] = (
        qualification["selection_completed_before_timing"] is True
    )
    checks["production_gpu_arms_accuracy_before_timing"] = all(
        row["pass"] is True
        for row in result[
            "production_gpu_implementation_accuracy_before_timing"
        ].values()
    )
    checks["type2_largest_passing_eps"] = (
        qualification["selected_type2_eps"]
        == _first_passing(rows["affine_type2"], "eps")
        and qualification["selected_type2_eps"] is not None
    )
    checks["type3_largest_passing_eps"] = (
        qualification["selected_type3_eps"]
        == _first_passing(rows["projected_type3"], "eps")
        and qualification["selected_type3_eps"] is not None
    )
    declared_pads = protocol["baseline_candidates"]["dense_periodic_fft"][
        "pad_factors"
    ]
    checks["complete_fft_frontier"] = bool(
        qualification["resource_limited_frontier"] is False
        and qualification["tested_fft_pad_factors"] == declared_pads
        and [row["pad_factor"] for row in rows["dense_periodic_fft"]]
        == declared_pads
    )
    expected_qualified = [
        row["pad_factor"] for row in rows["dense_periodic_fft"] if row["pass"]
    ]
    checks["fft_qualification_accuracy_only"] = (
        qualification["qualified_fft_pad_factors"] == expected_qualified
    )

    screen = result["fft_rtx3090_timing_screen"]
    checks["screen_on_rtx3090"] = screen["machine_eligible"] is True
    checks["screen_only_qualified_fft"] = (
        [row["pad_factor"] for row in screen["rows"]] == expected_qualified
    )
    checks["screen_fixed_samples"] = all(
        len(row["samples_seconds"])
        == protocol["timing_screen"]["samples_per_fft_candidate"]
        and len(row["sample_ids"])
        == protocol["timing_screen"]["samples_per_fft_candidate"]
        for row in screen["rows"]
    )
    expected_selected_fft = (
        min(screen["rows"], key=lambda row: row["median_seconds"])["pad_factor"]
        if screen["rows"]
        else None
    )
    checks["fft_selected_by_frozen_screen_rule"] = (
        screen["selected_fft_pad_factor"] == expected_selected_fft
    )

    confirmation = result["independent_confirmation"]
    comparisons = confirmation["comparisons"]
    expected_baselines = {"affine_type2", "projected_type3"}
    if expected_selected_fft is not None:
        expected_baselines.add("dense_periodic_fft")
    checks["confirmation_arms"] = set(comparisons) == expected_baselines
    checks["confirmation_10_warmup_30_abba"] = all(
        row["order_contract"] == "ABBA"
        and row["warmups_per_arm"]
        == protocol["independent_confirmation"]["warmups_per_arm_per_pair"]
        and row["samples_per_arm"]
        == protocol["independent_confirmation"]["paired_samples_per_arm"]
        and len(row["samples_seconds"]["acfo"])
        == protocol["independent_confirmation"]["paired_samples_per_arm"]
        and len(row["samples_seconds"][name])
        == protocol["independent_confirmation"]["paired_samples_per_arm"]
        for name, row in comparisons.items()
    )
    checks["screen_samples_not_reused"] = bool(
        confirmation["screen_sample_reuse_count"]
        == protocol["independent_confirmation"]["screen_sample_reuse_must_equal"]
        == 0
        and not confirmation["screen_confirmation_sample_overlap"]
    )
    expected_strongest = min(
        comparisons,
        key=lambda name: comparisons[name]["median_seconds"][name],
    )
    checks["strongest_frozen_before_orientation2"] = bool(
        confirmation["strongest_baseline_selected_before_orientation_2"]
        == expected_strongest
        == result["strongest_baseline_frozen_before_orientation_2"]
    )

    gate = protocol["accuracy_qualification"]
    checks["heldout_orientation_count"] = (
        len(result["heldout_accuracy"]["rows"])
        == len(gate["held_out_orientation_indices"])
    )
    checks["heldout_accuracy"] = bool(
        result["heldout_accuracy"]["worst_amplitude_relative_l2"]
        <= gate["amplitude_relative_l2_max"]
        and result["heldout_accuracy"]["worst_output_relative_l2"]
        <= gate["worst_twelve_output_relative_l2_max"]
    )
    checks["full_ensemble_method_agreement"] = (
        result["full_ensemble_acfo_vs_selected_baseline"]["worst_relative_l2"]
        <= gate["full_ensemble_selected_baseline_worst_output_relative_l2_max"]
    )
    archive = result["archived_output_validation"]
    checks["archive_oracle_performed"] = archive["performed"] is True
    selected = result["strongest_baseline_frozen_before_orientation_2"]
    checks["archive_accuracy"] = bool(
        archive["performed"]
        and archive["acfo"]["worst_relative_l2"]
        <= gate["archived_output_worst_relative_l2_max"]
        and archive[selected]["worst_relative_l2"]
        <= gate["archived_output_worst_relative_l2_max"]
    )
    checks["all_prefixes_reported"] = (
        [row["orientations"] for row in result["prefix_results"]]
        == protocol["workload"]["prefix_orientation_counts"]
    )
    runner_gates = result["gates"]
    checks["runner_gates_exclude_performance_sign"] = bool(
        "timing_pass" not in runner_gates
        and "acfo_positive_claim_eligible" not in runner_gates
    )
    checks["runner_gates"] = all(runner_gates.values())
    full_speed_ratio = result[
        "cold_total_speedup_selected_baseline_over_acfo"
    ]
    required_speed_lower_95 = float(
        protocol["full_ensemble"]["required_lower_95_speed_ratio"]
    )
    preliminary = audit_decision(
        checks,
        speed_lower_95=float(full_speed_ratio["lower_95"]),
        required_speed_lower_95=required_speed_lower_95,
    )
    expected_result_verdict = (
        "PASS_STRONGEST_BASELINE_CLOSURE"
        if preliminary["acfo_positive_claim_eligible"]
        else (
            "PASS_STRONGEST_BASELINE_CLOSURE_NO_GO"
            if preliminary["integrity_passed"]
            else "FAIL_PRESERVE_STRONGEST_BASELINE_RESULT"
        )
    )
    checks["result_claim_eligibility_separate"] = bool(
        result.get("acfo_positive_claim_eligible")
        is preliminary["acfo_positive_claim_eligible"]
    )
    checks["result_verdict_matches_signed_outcome"] = bool(
        result.get("verdict") == expected_result_verdict
    )
    decision = audit_decision(
        checks,
        speed_lower_95=float(full_speed_ratio["lower_95"]),
        required_speed_lower_95=required_speed_lower_95,
    )
    return {
        "schema": "numagsans-example3-strongest-baseline-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": decision["verdict"],
        "integrity_passed": decision["integrity_passed"],
        "acfo_positive_claim_eligible": decision[
            "acfo_positive_claim_eligible"
        ],
        "required_speed_lower_95": required_speed_lower_95,
        "checks": checks,
        "selected": {
            "type2_eps": qualification["selected_type2_eps"],
            "type3_eps": qualification["selected_type3_eps"],
            "fft_pad": expected_selected_fft,
            "strongest_baseline": selected,
        },
        "confirmation_speed_ratios": {
            name: row["baseline_over_acfo_speed_ratio"]
            for name, row in comparisons.items()
        },
        "full_speed_ratio": full_speed_ratio,
        "worst_errors": {
            "heldout_amplitude": result["heldout_accuracy"][
                "worst_amplitude_relative_l2"
            ],
            "heldout_output": result["heldout_accuracy"][
                "worst_output_relative_l2"
            ],
            "full_ensemble": result[
                "full_ensemble_acfo_vs_selected_baseline"
            ]["worst_relative_l2"],
            "acfo_archive": archive.get("acfo", {}).get("worst_relative_l2"),
            "selected_archive": archive.get(selected, {}).get(
                "worst_relative_l2"
            ),
        },
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("result", type=Path)
    p.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "NUMAGSANS_EXAMPLE3_STRONGEST_BASELINE_PROTOCOL.json",
    )
    p.add_argument("--output", type=Path)
    return p


def main() -> int:
    args = parser().parse_args()
    audit = run(args.result, args.protocol)
    output = args.output or args.result.with_name("AUDIT.json")
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0 if audit["verdict"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
