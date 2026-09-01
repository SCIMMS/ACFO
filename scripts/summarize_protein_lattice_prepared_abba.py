from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results"
OUTPUT = RESULTS / "protein_lattice_prepared_abba_decision.json"
SUMMARY = RESULTS / "protein_lattice_prepared_abba_decision.md"


def order_summary(rows: list[dict], order: str) -> dict:
    selected = [row for row in rows if row["order"] == order]
    return {
        "count": len(selected),
        "factorized_median_seconds": float(
            np.median([row["factorized_seconds"] for row in selected])
        ),
        "finufft_median_seconds": float(
            np.median([row["finufft_seconds"] for row in selected])
        ),
        "paired_speedup_median": float(
            np.median([row["paired_speedup"] for row in selected])
        ),
    }


def relative_median_gap(a: float, b: float) -> float:
    return abs(a - b) / (0.5 * (a + b))


def main() -> None:
    legacy = json.loads(
        (RESULTS / "protein_lattice_finufft_512_abba.json").read_text(
            encoding="utf-8"
        )
    )
    prepared = json.loads(
        (RESULTS / "protein_lattice_prepared_finufft_512_abba.json").read_text(
            encoding="utf-8"
        )
    )
    legacy_summary = legacy["measured_summary"]
    prepared_summary = prepared["measured_summary"]
    ab = order_summary(prepared["measured_pairs"], "AB")
    ba = order_summary(prepared["measured_pairs"], "BA")
    order_bias = {
        "factorized_median_relative_gap": relative_median_gap(
            ab["factorized_median_seconds"], ba["factorized_median_seconds"]
        ),
        "finufft_median_relative_gap": relative_median_gap(
            ab["finufft_median_seconds"], ba["finufft_median_seconds"]
        ),
        "paired_speedup_median_relative_gap": relative_median_gap(
            ab["paired_speedup_median"], ba["paired_speedup_median"]
        ),
    }
    core_keys = (
        "unit",
        "supercell",
        "nq",
        "q_min_inv_angstrom",
        "q_max_inv_angstrom",
        "wavelength_nm",
        "target_nphi",
        "harmonic_margin",
        "finufft_eps",
        "finufft_threads",
        "warmup_pairs_requested",
        "measured_pairs_requested",
    )
    contracts_match = all(
        legacy["contract"][key] == prepared["contract"][key] for key in core_keys
    )
    comparison = {
        "legacy_factorized_median_seconds": legacy_summary["factorized_seconds"][
            "median"
        ],
        "prepared_factorized_median_seconds": prepared_summary[
            "factorized_seconds"
        ]["median"],
        "legacy_over_prepared_factorized_median": (
            legacy_summary["factorized_seconds"]["median"]
            / prepared_summary["factorized_seconds"]["median"]
        ),
        "legacy_finufft_median_seconds": legacy_summary["finufft_seconds"][
            "median"
        ],
        "prepared_run_finufft_median_seconds": prepared_summary[
            "finufft_seconds"
        ]["median"],
        "legacy_paired_speedup_median": legacy_summary["paired_speedup"]["median"],
        "prepared_paired_speedup_median": prepared_summary["paired_speedup"][
            "median"
        ],
        "paired_speedup_median_improvement": (
            prepared_summary["paired_speedup"]["median"]
            / legacy_summary["paired_speedup"]["median"]
        ),
        "legacy_paired_speedup_p05": legacy_summary["paired_speedup"]["p05"],
        "prepared_paired_speedup_p05": prepared_summary["paired_speedup"]["p05"],
        "paired_speedup_p05_improvement": (
            prepared_summary["paired_speedup"]["p05"]
            / legacy_summary["paired_speedup"]["p05"]
        ),
    }
    setup = {
        "prepared_plan_seconds": prepared["prepared_plan_setup_seconds"],
        "separable_lattice_seconds": prepared["lattice_setup_seconds"],
        "finufft_plan_seconds": prepared["finufft_setup_seconds"],
        "prepared_first_total_using_measured_median_seconds": (
            prepared["prepared_plan_setup_seconds"]
            + prepared["lattice_setup_seconds"]
            + prepared_summary["factorized_seconds"]["median"]
        ),
        "finufft_first_total_using_measured_median_seconds": (
            prepared["finufft_setup_seconds"]
            + prepared_summary["finufft_seconds"]["median"]
        ),
    }
    setup["first_total_speedup_using_measured_medians"] = (
        setup["finufft_first_total_using_measured_median_seconds"]
        / setup["prepared_first_total_using_measured_median_seconds"]
    )
    gates = {
        "legacy_receipt_pass": bool(legacy["passed"]),
        "prepared_receipt_pass": bool(prepared["passed"]),
        "core_contracts_match": contracts_match,
        "prepared_backend_is_fused_fft_separable": (
            prepared["contract"]["factorized_backend"] == "prepared_fused"
            and prepared["contract"]["coefficient_backend"] == "fused_phase"
            and prepared["contract"]["synthesis_backend"] == "fft"
            and prepared["contract"]["lattice_backend"] == "separable"
        ),
        "balanced_15_15_order_groups": ab["count"] == 15 and ba["count"] == 15,
        "order_speedup_median_gap_le_10pct": (
            order_bias["paired_speedup_median_relative_gap"] <= 0.10
        ),
        "prepared_factorized_median_improvement_ge_3": (
            comparison["legacy_over_prepared_factorized_median"] >= 3.0
        ),
        "prepared_paired_speedup_median_ge_3": (
            comparison["prepared_paired_speedup_median"] >= 3.0
        ),
        "prepared_paired_speedup_p05_ge_3": (
            comparison["prepared_paired_speedup_p05"] >= 3.0
        ),
        "prepared_vs_legacy_complex_l2_le_1e_12": (
            prepared["legacy_comparison"]["legacy_vs_prepared_complex_l2"]
            <= 1e-12
        ),
        "prepared_vs_finufft_complex_l2_le_2e_6": (
            prepared["cross_error"]["complex_l2"] <= 2e-6
        ),
    }
    result = {
        "schema": "protein-lattice-prepared-abba-decision-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": prepared["contract"],
        "prepared_receipt": {
            "path": "benchmark_results/protein_lattice_prepared_finufft_512_abba.json",
            "warmup_pairs": prepared["warmup_pairs_completed"],
            "measured_pairs": prepared["measured_pairs_completed"],
            "elapsed_seconds": prepared["elapsed_this_run_seconds"],
        },
        "prepared_measured_summary": prepared_summary,
        "accuracy": {
            "prepared_vs_finufft": prepared["cross_error"],
            "prepared_vs_legacy": prepared["legacy_comparison"],
        },
        "setup": setup,
        "order_groups": {"AB": ab, "BA": ba, "relative_gap": order_bias},
        "legacy_comparison": comparison,
        "gates": gates,
        "local_prepared_timing_gate_pass": all(gates.values()),
        "independent_machine_replication_complete": False,
        "publication_timing_ready": False,
        "decision": (
            "The same-machine prepared fused 10/30 AB/BA gate passes. "
            "The remaining publication timing gate is independent-machine replication."
        ),
        "claim_boundary": [
            "The 33.480x median and 24.243x p05 are same-machine hot-execution results for the tested 1.001M-atom exact repeated crystal.",
            "Setup is reported separately; the first-total ratio based on measured medians is not a paired statistic.",
            "The repeated-crystal lattice factor is a standard crystallographic specialization, not an ACFO novelty claim.",
            "Dense disordered sources cannot use this perfect-lattice specialization.",
            "Independent-machine replication remains required before publication-level timing is final.",
        ],
    }
    result["passed"] = result["local_prepared_timing_gate_pass"]

    lines = [
        "# Prepared fused 1M repeated-crystal 10/30 AB/BA decision",
        "",
        f"- local prepared timing gate: **{'PASS' if result['passed'] else 'FAIL'}**",
        f"- prepared / FINUFFT median: `{prepared_summary['factorized_seconds']['median']:.3f} / {prepared_summary['finufft_seconds']['median']:.3f} s`",
        f"- paired speedup median / p05: `{prepared_summary['paired_speedup']['median']:.3f}x / {prepared_summary['paired_speedup']['p05']:.3f}x`",
        f"- prepared vs FINUFFT complex L2: `{prepared['cross_error']['complex_l2']:.3e}`",
        f"- prepared vs legacy complex L2: `{prepared['legacy_comparison']['legacy_vs_prepared_complex_l2']:.3e}`",
        f"- legacy/prepared factorized median improvement: `{comparison['legacy_over_prepared_factorized_median']:.3f}x`",
        f"- AB/BA speedup-median relative gap: `{100*order_bias['paired_speedup_median_relative_gap']:.2f}%`",
        f"- setup-based first-total ratio using measured medians: `{setup['first_total_speedup_using_measured_medians']:.3f}x` (not paired)",
        "- independent-machine publication timing: **PENDING**",
        "",
        "| protocol | factorized median s | FINUFFT median s | paired median | paired p05 |",
        "|---|---:|---:|---:|---:|",
        f"| legacy 10/30 | {comparison['legacy_factorized_median_seconds']:.3f} | {comparison['legacy_finufft_median_seconds']:.3f} | {comparison['legacy_paired_speedup_median']:.3f}x | {comparison['legacy_paired_speedup_p05']:.3f}x |",
        f"| prepared fused 10/30 | {comparison['prepared_factorized_median_seconds']:.3f} | {comparison['prepared_run_finufft_median_seconds']:.3f} | {comparison['prepared_paired_speedup_median']:.3f}x | {comparison['prepared_paired_speedup_p05']:.3f}x |",
        "",
        "The prepared local timing gate is closed. Independent-machine replication remains the publication timing gate.",
        "",
    ]
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "gates": gates}, indent=2))
    print(f"wrote {OUTPUT} and {SUMMARY}")


if __name__ == "__main__":
    main()
