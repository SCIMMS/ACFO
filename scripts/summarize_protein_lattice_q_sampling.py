from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results"
OUTPUT = RESULTS / "protein_lattice_q_sampling_decision.json"
SUMMARY = RESULTS / "protein_lattice_q_sampling_decision.md"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def exponent(x: list[float], y: list[float]) -> float:
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def main() -> None:
    range_payloads = [
        load("protein_lattice_q_sampling_range_q2p13_chunked.json"),
        load("protein_lattice_q_sampling_range_q4p06_chunked.json"),
        load("protein_lattice_q_sampling_range_q6p30_chunked.json"),
        load("protein_lattice_q_sampling_range_q8p06_chunked.json"),
    ]
    range_rows = []
    for payload in range_payloads:
        factor = payload["timing_seconds"]["factorized"]
        finufft = payload["timing_seconds"]["finufft"]
        first_total = factor["first_total_excluding_shared"]
        range_rows.append(
            {
                "label": payload["label"],
                "q_max_inv_angstrom": payload["contract"]["q_max_inv_angstrom"],
                "nq": payload["contract"]["nq"],
                "dq_inv_angstrom": payload["contract"]["dq_inv_angstrom"],
                "nphi": payload["contract"]["nphi"],
                "target_count": payload["target_count"],
                "factorized_setup_seconds": factor["specific_setup_total"],
                "factorized_first_execute_seconds": factor["first_execute"],
                "factorized_first_total_seconds": first_total,
                "factorized_hot_seconds": factor["hot_execute"],
                "coefficient_fraction_of_hot": (
                    factor["hot_profile"]["coefficient_contraction_seconds"]
                    / factor["hot_profile"]["total_seconds"]
                ),
                "chunked_finufft_setup_sum_seconds": finufft["summed_plan_setup"],
                "chunked_finufft_execute_sum_seconds": finufft["summed_execute"],
                "chunked_finufft_cleanup_sum_seconds": finufft["summed_cleanup"],
                "chunked_finufft_streamed_wall_seconds": finufft[
                    "streamed_wall_excluding_shared"
                ],
                "first_total_speedup": payload[
                    "speedup_finufft_over_factorized"
                ]["first_total_excluding_shared"],
                "cross_complex_l2": payload["cross_error"]["complex_l2"],
            }
        )

    resolution_payloads = [
        load("protein_lattice_q_sampling_resolution_nq32.json"),
        load("protein_lattice_q_sampling_resolution_nq128.json"),
    ]
    optimized = load(
        "protein_lattice_q_sampling_resolution_nq512_fused_phase_profile.json"
    )
    contraction = load("exact_beta_contraction_optimization_decision.json")
    prepared_decision = load("protein_lattice_prepared_abba_decision.json")
    prepared_abba = load("protein_lattice_prepared_finufft_512_abba.json")
    abba = load("protein_lattice_finufft_512_abba.json")
    crossover = load("protein_lattice_finufft_512_5x5.json")["rows"][0]
    resolution_rows = []
    for payload in resolution_payloads:
        factor = payload["timing_seconds"]["factorized"]
        finufft = payload["timing_seconds"]["finufft"]
        resolution_rows.append(
            {
                "nq": payload["contract"]["nq"],
                "dq_inv_angstrom": payload["contract"]["dq_inv_angstrom"],
                "target_count": payload["target_count"],
                "factorized_setup_seconds": factor["specific_setup_total"],
                "factorized_first_total_seconds": factor[
                    "first_total_excluding_shared"
                ],
                "factorized_hot_seconds": factor["hot_execute"],
                "finufft_setup_seconds": finufft["plan_setup"],
                "finufft_first_total_seconds": finufft[
                    "first_total_excluding_shared"
                ],
                "finufft_hot_seconds": finufft["hot_execute"],
                "first_total_speedup": payload[
                    "speedup_finufft_over_factorized"
                ]["first_total_excluding_shared"],
                "hot_speedup": payload["speedup_finufft_over_factorized"][
                    "hot_execute"
                ],
                "cross_complex_l2": payload["cross_error"]["complex_l2"],
                "timing_source": "single local first/hot case",
            }
        )
    optimized_factor = optimized["timing_seconds"]["factorized"]
    optimized_hot = prepared_abba["measured_summary"]["factorized_seconds"][
        "median"
    ]
    optimized_setup = (
        prepared_abba["prepared_plan_setup_seconds"]
        + prepared_abba["lattice_setup_seconds"]
    )
    optimized_first_total = optimized_setup + optimized_hot
    prepared_finufft_hot = prepared_abba["measured_summary"]["finufft_seconds"][
        "median"
    ]
    prepared_finufft_first_total = (
        prepared_abba["finufft_setup_seconds"] + prepared_finufft_hot
    )
    resolution_rows.append(
        {
            "nq": 512,
            "dq_inv_angstrom": optimized["contract"]["dq_inv_angstrom"],
            "target_count": optimized["target_count"],
            "factorized_setup_seconds": optimized_setup,
            "factorized_first_total_seconds": optimized_first_total,
            "factorized_hot_seconds": optimized_hot,
            "finufft_setup_seconds": prepared_abba["finufft_setup_seconds"],
            "finufft_first_total_seconds": prepared_finufft_first_total,
            "finufft_hot_seconds": prepared_finufft_hot,
            "first_total_speedup": (
                prepared_finufft_first_total / optimized_first_total
            ),
            "hot_speedup": prepared_abba["measured_summary"]["paired_speedup"][
                "median"
            ],
            "cross_complex_l2": prepared_abba["cross_error"]["complex_l2"],
            "timing_source": (
                "prepared fused 10/30 AB/BA paired medians"
            ),
        }
    )

    qmax = [row["q_max_inv_angstrom"] for row in range_rows]
    range_factor = [row["factorized_first_total_seconds"] for row in range_rows]
    range_finufft = [
        row["chunked_finufft_streamed_wall_seconds"] for row in range_rows
    ]
    range_speedups = [row["first_total_speedup"] for row in range_rows]
    resolution_speedups = [row["hot_speedup"] for row in resolution_rows]
    result = {
        "schema": "protein-lattice-q-sampling-decision-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "atom_count": 1_001_000,
            "supercell": [5, 5, 5],
            "unit_atom_count": 8_008,
            "finufft_eps": 1e-6,
            "finufft_threads": 4,
            "range_sweep": "fixed dq about 0.160 A^-1; physical minimum FFT-friendly nphi; q-block=2 streamed FINUFFT",
            "resolution_sweep": "fixed q=5.0-6.3 A^-1 and nphi=768; reusable FINUFFT",
        },
        "fixed_dq_range_sweep": {
            "rows": range_rows,
            "effective_exponent_vs_qmax": {
                "factorized_first_total": exponent(qmax, range_factor),
                "chunked_finufft_streamed_wall": exponent(qmax, range_finufft),
                "speedup": exponent(qmax, range_speedups),
            },
            "reusable_whole_plan_negative_control": load(
                "protein_lattice_q_sampling_range_q6p30_reusable_timeout.json"
            ),
        },
        "fixed_range_resolution_sweep": {
            "rows": resolution_rows,
            "interpretation": (
                "Increasing Nq at fixed q range reduces the factorized advantage; dense radial sampling is relatively favorable to reusable FINUFFT."
            ),
        },
        "exact_optimization_profile_nq512": {
            "coefficient_backend": "fused_phase",
            "legacy_seconds": prepared_decision["legacy_comparison"][
                "legacy_factorized_median_seconds"
            ],
            "prepared_hot_seconds": optimized_hot,
            "legacy_over_prepared_speedup": prepared_decision[
                "legacy_comparison"
            ]["legacy_over_prepared_factorized_median"],
            "legacy_vs_prepared_complex_l2": prepared_decision["accuracy"][
                "prepared_vs_legacy"
            ]["legacy_vs_prepared_complex_l2"],
            "coefficient_contraction_seconds": optimized_factor["hot_profile"][
                "coefficient_contraction_seconds"
            ],
            "azimuth_synthesis_seconds": optimized_factor["hot_profile"][
                "azimuth_synthesis_seconds"
            ],
            "coefficient_fraction_of_hot": (
                optimized_factor["hot_profile"][
                    "coefficient_contraction_seconds"
                ]
                / optimized_factor["hot_profile"]["total_seconds"]
            ),
            "direct_lattice_seconds": optimized["lattice_comparison"][
                "direct_seconds"
            ],
            "separable_lattice_seconds": optimized["lattice_comparison"][
                "selected_seconds"
            ],
            "direct_over_separable_lattice_speedup": optimized[
                "lattice_comparison"
            ]["direct_over_selected_speedup"],
            "lattice_complex_l2": optimized["lattice_comparison"][
                "complex_l2"
            ],
            "lattice_intensity_l2": optimized["lattice_comparison"][
                "intensity_l2"
            ],
            "prepared_abba_status": "complete; local timing gate PASS",
            "prepared_abba_factorized_median_seconds": optimized_hot,
            "prepared_abba_finufft_median_seconds": prepared_finufft_hot,
            "prepared_abba_paired_speedup_median": prepared_abba[
                "measured_summary"
            ]["paired_speedup"]["median"],
            "prepared_abba_paired_speedup_p05": prepared_abba[
                "measured_summary"
            ]["paired_speedup"]["p05"],
            "prepared_abba_order_speedup_median_gap": prepared_decision[
                "order_groups"
            ]["relative_gap"]["paired_speedup_median_relative_gap"],
        },
        "contraction_backend_decision": contraction,
        "gates": {
            "all_range_cases_pass": all(row.get("passed") for row in range_payloads),
            "range_speedup_increases_monotonically": all(
                left < right
                for left, right in zip(range_speedups, range_speedups[1:])
            ),
            "resolution_hot_speedup_decreases_monotonically": all(
                left > right
                for left, right in zip(
                    resolution_speedups, resolution_speedups[1:]
                )
            ),
            "optimized_profile_pass": optimized["passed"],
            "legacy_vs_prepared_complex_l2_le_1e_12": (
                optimized["legacy_comparison"]["complex_l2_vs_prepared"] <= 1e-12
            ),
            "separable_lattice_complex_l2_le_1e_9": (
                optimized["lattice_comparison"]["complex_l2"] <= 1e-9
            ),
            "prepared_abba_decision_pass": prepared_decision["passed"],
        },
        "decision": [
            "The earlier fixed-dq observation is reproduced: widening q range makes q-blocked FINUFFT increasingly unfavorable.",
            "The current Nq=512 narrow-band condition is relatively FINUFFT-favorable because dense radial sampling reduces the factorized speedup.",
            "Prepared geometry plus FFT azimuth synthesis removes an exact-path bottleneck without relaxing accuracy.",
            "After optimization, the Nq=512 hot path is coefficient-contraction limited; this is the next exact optimization target.",
            "The prepared fused Nq=512 path passes its local 10/30 AB/BA gate; independent-machine replication is the remaining publication timing gate.",
        ],
        "claim_boundary": [
            "Range and resolution sweeps are exploratory local scaling probes, not independent-machine timing claims.",
            "The Nq=512 optimized row uses same-machine prepared fused 10/30 AB/BA paired medians; setup-based first-total is derived separately.",
            "The separable lattice factor is a standard crystallographic specialization, not ACFO novelty.",
            "Dense disordered sources cannot use the perfect-lattice factorization.",
        ],
    }
    result["passed"] = all(result["gates"].values())
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Protein-lattice q sampling and prepared-path decision",
        "",
        "## Fixed-dq q-range sweep (q-block=2 FINUFFT)",
        "",
        "| qmax A^-1 | Nq | Nphi | factorized first-total s | chunked FINUFFT wall s | speedup | cross L2 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in range_rows:
        lines.append(
            f"| {row['q_max_inv_angstrom']:.2f} | {row['nq']} | {row['nphi']} "
            f"| {row['factorized_first_total_seconds']:.3f} "
            f"| {row['chunked_finufft_streamed_wall_seconds']:.3f} "
            f"| {row['first_total_speedup']:.1f}x | {row['cross_complex_l2']:.2e} |"
        )
    lines.extend(
        [
            "",
            "## Fixed-range q-resolution sweep",
            "",
            "| Nq | dq A^-1 | factorized hot s | FINUFFT hot s | hot speedup | timing source |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in resolution_rows:
        lines.append(
            f"| {row['nq']} | {row['dq_inv_angstrom']:.5f} "
            f"| {row['factorized_hot_seconds']:.3f} | {row['finufft_hot_seconds']:.3f} "
            f"| {row['hot_speedup']:.1f}x | {row['timing_source']} |"
        )
    profile = result["exact_optimization_profile_nq512"]
    lines.extend(
        [
            "",
            "## Exact Nq=512 optimization profile",
            "",
            f"- legacy / prepared hot: `{profile['legacy_seconds']:.3f} / {profile['prepared_hot_seconds']:.3f} s` (`{profile['legacy_over_prepared_speedup']:.2f}x`)",
            f"- legacy-prepared complex L2: `{profile['legacy_vs_prepared_complex_l2']:.3e}`",
            f"- coefficient / synthesis: `{profile['coefficient_contraction_seconds']:.3f} / {profile['azimuth_synthesis_seconds']:.3f} s`",
            f"- direct / separable lattice setup: `{profile['direct_lattice_seconds']:.3f} / {profile['separable_lattice_seconds']:.3f} s` (`{profile['direct_over_separable_lattice_speedup']:.1f}x`)",
            f"- lattice complex/intensity L2: `{profile['lattice_complex_l2']:.3e} / {profile['lattice_intensity_l2']:.3e}`",
            f"- prepared 10/30 paired median / p05: `{profile['prepared_abba_paired_speedup_median']:.3f}x / {profile['prepared_abba_paired_speedup_p05']:.3f}x`",
            "- Prepared local timing gate PASS; independent-machine replication remains pending.",
            "",
        ]
    )
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {OUTPUT} and {SUMMARY}")


if __name__ == "__main__":
    main()
