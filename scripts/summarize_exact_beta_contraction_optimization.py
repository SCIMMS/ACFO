from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results"
OUTPUT = RESULTS / "exact_beta_contraction_optimization_decision.json"
SUMMARY = RESULTS / "exact_beta_contraction_optimization_decision.md"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main() -> None:
    backend = load("exact_beta_contraction_backends_nq512.json")
    full = load("protein_lattice_q_sampling_resolution_nq512_fused_phase_profile.json")
    abba = load("protein_lattice_finufft_512_abba.json")
    crossover = load("protein_lattice_finufft_512_5x5.json")["rows"][0]
    summary = backend["measured_summary"]
    factor = full["timing_seconds"]["factorized"]
    first_total = factor["first_total_excluding_shared"]
    hot = factor["hot_execute"]
    result = {
        "schema": "exact-beta-contraction-optimization-decision-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": backend["contract"],
        "backend_comparison": {
            "baseline_coefficient_median_seconds": summary[
                "baseline_coefficient_seconds"
            ]["median"],
            "fused_phase_coefficient_median_seconds": summary[
                "fused_coefficient_seconds"
            ]["median"],
            "cached_phase_coefficient_median_seconds": summary[
                "cached_coefficient_seconds"
            ]["median"],
            "baseline_over_fused_paired_median": summary[
                "coefficient_speedup"
            ]["median"],
            "baseline_over_fused_paired_p05": summary[
                "coefficient_speedup"
            ]["p05"],
            "fused_over_cached_paired_median": summary[
                "fused_over_cached_coefficient_speedup"
            ]["median"],
            "fused_over_cached_paired_p05": summary[
                "fused_over_cached_coefficient_speedup"
            ]["p05"],
            "cached_phase_setup_seconds": backend["setup_seconds"][
                "cached_phase"
            ],
            "cached_phase_mib": backend["cached_phase_mib"],
            "fused_vs_baseline_complex_l2": backend["cross_error"][
                "fused_vs_baseline_complex_l2"
            ],
            "cached_vs_fused_complex_l2": backend["cross_error"][
                "cached_vs_fused_complex_l2"
            ],
        },
        "selected_low_memory_full_path": {
            "backend": "fused_phase",
            "specific_setup_seconds": factor["specific_setup_total"],
            "first_execute_seconds": factor["first_execute"],
            "first_total_seconds": first_total,
            "hot_execute_seconds": hot,
            "coefficient_seconds": factor["hot_profile"][
                "coefficient_contraction_seconds"
            ],
            "synthesis_seconds": factor["hot_profile"][
                "azimuth_synthesis_seconds"
            ],
            "legacy_unprepared_seconds": full["legacy_comparison"]["seconds"],
            "legacy_over_selected_speedup": full["legacy_comparison"][
                "legacy_over_prepared_hot"
            ],
            "legacy_vs_selected_complex_l2": full["legacy_comparison"][
                "complex_l2_vs_prepared"
            ],
            "direct_lattice_seconds": full["lattice_comparison"][
                "direct_seconds"
            ],
            "separable_lattice_seconds": full["lattice_comparison"][
                "selected_seconds"
            ],
            "lattice_complex_l2": full["lattice_comparison"]["complex_l2"],
        },
        "unpaired_finufft_projection": {
            "first_total_finufft_over_selected": (
                crossover["finufft"]["first_total_seconds"] / first_total
            ),
            "abba_finufft_median_over_selected_hot": (
                abba["measured_summary"]["finufft_seconds"]["median"] / hot
            ),
            "status": "exploratory; optimized factorized and saved FINUFFT receipts are not paired",
        },
        "decision": {
            "recommended_backend": "fused_phase",
            "reason": (
                "It gives a robust exact speedup without a source-by-harmonic cache; cached_phase is optional only when 42.89 MiB extra setup memory is acceptable for repeated hot execution."
            ),
            "default_policy": (
                "Keep the legacy baseline as the API default and expose fused_phase/cached_phase through explicit prepared-plan flags until optimized 10/30 AB/BA and independent reruns are complete."
            ),
            "remaining_exact_bottleneck": (
                "Miller Bessel recurrence plus source/q axial phase and harmonic accumulation. Further cache-based acceleration increases working memory; approximation-based tables require a separate error sweep."
            ),
        },
        "gates": {
            "backend_benchmark_pass": backend["passed"],
            "full_profile_pass": full["passed"],
            "baseline_over_fused_p05_ge_1_1": summary["coefficient_speedup"][
                "p05"
            ]
            >= 1.1,
            "fused_vs_baseline_complex_l2_le_1e_12": backend["cross_error"][
                "fused_vs_baseline_complex_l2"
            ]
            <= 1e-12,
            "cached_vs_fused_complex_l2_le_1e_12": backend["cross_error"][
                "cached_vs_fused_complex_l2"
            ]
            <= 1e-12,
            "legacy_vs_selected_complex_l2_le_1e_12": full[
                "legacy_comparison"
            ]["complex_l2_vs_prepared"]
            <= 1e-12,
        },
        "claim_boundary": [
            "The backend A/B is a local same-machine 2-warmup/10-measured ABC/CBA comparison.",
            "The optimized factorized-to-FINUFFT ratios combine separate local receipts and are not a paired timing claim.",
            "The current publication timing receipt remains the legacy 10/30 AB/BA result until the optimized protocol is rerun.",
            "The repeated-lattice path remains a standard crystallographic specialization and does not apply to dense disorder.",
        ],
    }
    result["passed"] = all(result["gates"].values())
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    b = result["backend_comparison"]
    f = result["selected_low_memory_full_path"]
    lines = [
        "# Exact-beta contraction optimization decision",
        "",
        f"- baseline / fused / cached coefficient medians: `{b['baseline_coefficient_median_seconds']:.3f} / {b['fused_phase_coefficient_median_seconds']:.3f} / {b['cached_phase_coefficient_median_seconds']:.3f} s`",
        f"- baseline/fused paired median and p05: `{b['baseline_over_fused_paired_median']:.3f}x / {b['baseline_over_fused_paired_p05']:.3f}x`",
        f"- fused/cached paired median and p05: `{b['fused_over_cached_paired_median']:.3f}x / {b['fused_over_cached_paired_p05']:.3f}x`",
        f"- cached phase table: `{b['cached_phase_mib']:.2f} MiB`; setup `{b['cached_phase_setup_seconds']:.3f} s`",
        f"- selected full path legacy / fused hot: `{f['legacy_unprepared_seconds']:.3f} / {f['hot_execute_seconds']:.3f} s` (`{f['legacy_over_selected_speedup']:.3f}x`)",
        f"- legacy-selected complex L2: `{f['legacy_vs_selected_complex_l2']:.3e}`",
        "- selected backend: **fused_phase**; cached_phase remains an explicit repeated-hot option.",
        "- optimized 10/30 AB/BA and independent-machine rerun remain pending.",
        "",
    ]
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
