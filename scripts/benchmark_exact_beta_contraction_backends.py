from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from validate_public_waxs_structures import build_form_factors, load_structure  # noqa: E402
from waxs_cake import PreparedExactCoordinateHarmonicPlan, encode_elements  # noqa: E402
from waxs_cake.geometry import ewald_ring  # noqa: E402
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402
from waxs_cake.physical_scaling import q_to_inv_nm  # noqa: E402
from waxs_cake.solvers import normalize_form_factors  # noqa: E402


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matched baseline/fused-phase exact-beta contraction timing."
    )
    parser.add_argument(
        "--unit",
        type=Path,
        default=Path(
            "structures/processed/protein_nanocrystal_lysozyme_1iee_1x1x1_fixed.npz"
        ),
    )
    parser.add_argument("--q-min", type=float, default=5.0)
    parser.add_argument("--q-max", type=float, default=6.3)
    parser.add_argument("--nq", type=int, default=512)
    parser.add_argument("--nphi", type=int, default=768)
    parser.add_argument("--wavelength-nm", type=float, default=0.08)
    parser.add_argument("--harmonic-margin", type=int, default=48)
    parser.add_argument("--warmup-pairs", type=int, default=2)
    parser.add_argument("--measured-pairs", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark_results/exact_beta_contraction_backends_nq512.json"
        ),
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path(
            "benchmark_results/exact_beta_contraction_backends_nq512.md"
        ),
    )
    args = parser.parse_args()

    coords, elements, metadata = load_structure(args.unit)
    element_indices, element_order = encode_elements(elements)
    q_report = np.linspace(args.q_min, args.q_max, args.nq, dtype=np.float64)
    q_solver = np.asarray(
        [q_to_inv_nm(value, "inv_angstrom") for value in q_report]
    )
    q_perp, q_z = ewald_ring(q_solver, args.wavelength_nm)
    form_factors = normalize_form_factors(
        element_order,
        q_solver,
        build_form_factors(elements, q_solver, "xray_f0"),
    ).astype(np.complex128, copy=False)
    phi = (np.arange(args.nphi) + 0.5) * (2.0 * np.pi / args.nphi)
    weights = np.ones(coords.shape[0], dtype=np.complex128)

    setup_start = time.perf_counter()
    baseline = PreparedExactCoordinateHarmonicPlan(
        coords,
        q_perp,
        q_z,
        phi,
        element_indices=element_indices,
        form_factors=form_factors,
        harmonic_margin=args.harmonic_margin,
        prepare_direct_basis=False,
        coefficient_backend="baseline",
    )
    baseline_setup_seconds = time.perf_counter() - setup_start
    setup_start = time.perf_counter()
    fused = PreparedExactCoordinateHarmonicPlan(
        coords,
        q_perp,
        q_z,
        phi,
        element_indices=element_indices,
        form_factors=form_factors,
        harmonic_margin=args.harmonic_margin,
        prepare_direct_basis=False,
        coefficient_backend="fused_phase",
    )
    fused_setup_seconds = time.perf_counter() - setup_start
    setup_start = time.perf_counter()
    cached = PreparedExactCoordinateHarmonicPlan(
        coords,
        q_perp,
        q_z,
        phi,
        element_indices=element_indices,
        form_factors=form_factors,
        harmonic_margin=args.harmonic_margin,
        prepare_direct_basis=False,
        coefficient_backend="cached_phase",
    )
    cached_setup_seconds = time.perf_counter() - setup_start

    def execute(plan: PreparedExactCoordinateHarmonicPlan):
        value, _ = plan.execute(
            atom_weights=weights, synthesis_backend="fft"
        )
        return value, dict(plan.last_profile)

    plans = {"A": baseline, "B": fused, "C": cached}
    names = {"A": "baseline", "B": "fused", "C": "cached"}
    last_values = {"baseline": None, "fused": None, "cached": None}
    warmup_rows = []
    for pair_index in range(args.warmup_pairs):
        order = "ABC" if pair_index % 2 == 0 else "CBA"
        row = {"pair_index": pair_index + 1, "order": order}
        for label in order:
            gc.collect()
            name = names[label]
            last_values[name], profile = execute(plans[label])
            row[f"{name}_total_seconds"] = profile["total_seconds"]
            row[f"{name}_coefficient_seconds"] = profile[
                "coefficient_contraction_seconds"
            ]
        warmup_rows.append(row)
        print(
            f"warmup {pair_index + 1}/{args.warmup_pairs}: "
            f"baseline {row['baseline_coefficient_seconds']:.3f} s; "
            f"fused {row['fused_coefficient_seconds']:.3f} s; "
            f"cached {row['cached_coefficient_seconds']:.3f} s",
            flush=True,
        )

    measured_rows = []
    for pair_index in range(args.measured_pairs):
        order = "ABC" if pair_index % 2 == 0 else "CBA"
        row = {"pair_index": pair_index + 1, "order": order}
        for label in order:
            gc.collect()
            name = names[label]
            last_values[name], profile = execute(plans[label])
            row[f"{name}_total_seconds"] = profile["total_seconds"]
            row[f"{name}_coefficient_seconds"] = profile[
                "coefficient_contraction_seconds"
            ]
            row[f"{name}_synthesis_seconds"] = profile[
                "azimuth_synthesis_seconds"
            ]
        row["coefficient_speedup"] = (
            row["baseline_coefficient_seconds"] / row["fused_coefficient_seconds"]
        )
        row["total_speedup"] = (
            row["baseline_total_seconds"] / row["fused_total_seconds"]
        )
        row["fused_over_cached_coefficient_speedup"] = (
            row["fused_coefficient_seconds"] / row["cached_coefficient_seconds"]
        )
        row["baseline_over_cached_coefficient_speedup"] = (
            row["baseline_coefficient_seconds"] / row["cached_coefficient_seconds"]
        )
        measured_rows.append(row)
        print(
            f"measured {pair_index + 1}/{args.measured_pairs}: "
            f"baseline {row['baseline_coefficient_seconds']:.3f} s; "
            f"fused {row['fused_coefficient_seconds']:.3f} s; "
            f"cached {row['cached_coefficient_seconds']:.3f} s; "
            f"speedups {row['coefficient_speedup']:.3f}x / "
            f"{row['fused_over_cached_coefficient_speedup']:.3f}x",
            flush=True,
        )

    cross_error = {
        "fused_vs_baseline_complex_l2": relative_l2(
            last_values["fused"], last_values["baseline"]
        ),
        "fused_vs_baseline_intensity_l2": relative_l2(
            intensity(last_values["fused"]), intensity(last_values["baseline"])
        ),
        "cached_vs_fused_complex_l2": relative_l2(
            last_values["cached"], last_values["fused"]
        ),
        "cached_vs_fused_intensity_l2": relative_l2(
            intensity(last_values["cached"]), intensity(last_values["fused"])
        ),
    }
    summary = {
        key: distribution([row[key] for row in measured_rows])
        for key in (
            "baseline_total_seconds",
            "baseline_coefficient_seconds",
            "baseline_synthesis_seconds",
            "fused_total_seconds",
            "fused_coefficient_seconds",
            "fused_synthesis_seconds",
            "cached_total_seconds",
            "cached_coefficient_seconds",
            "cached_synthesis_seconds",
            "coefficient_speedup",
            "total_speedup",
            "fused_over_cached_coefficient_speedup",
            "baseline_over_cached_coefficient_speedup",
        )
    }
    gates = {
        "warmup_count_matches_request": len(warmup_rows) == args.warmup_pairs,
        "measured_count_matches_request": len(measured_rows) == args.measured_pairs,
        "fused_vs_baseline_complex_l2_le_1e_12": (
            cross_error["fused_vs_baseline_complex_l2"] <= 1e-12
        ),
        "cached_vs_fused_complex_l2_le_1e_12": (
            cross_error["cached_vs_fused_complex_l2"] <= 1e-12
        ),
        "median_coefficient_speedup_ge_1": summary["coefficient_speedup"][
            "median"
        ]
        >= 1.0,
    }
    result = {
        "schema": "exact-beta-contraction-backends-v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "unit": args.unit.as_posix(),
            "unit_structure": metadata.get("structure_id", args.unit.stem),
            "atom_count": int(coords.shape[0]),
            "q_min_inv_angstrom": args.q_min,
            "q_max_inv_angstrom": args.q_max,
            "nq": args.nq,
            "nphi": args.nphi,
            "harmonic_margin": args.harmonic_margin,
            "warmup_pairs": args.warmup_pairs,
            "measured_pairs": args.measured_pairs,
            "hardware_threads": os.cpu_count(),
        },
        "setup_seconds": {
            "baseline": baseline_setup_seconds,
            "fused_phase": fused_setup_seconds,
            "cached_phase": cached_setup_seconds,
        },
        "cached_phase_mib": cached.last_profile["coefficient_cache_mib"] if cached.last_profile else None,
        "maximum_harmonic": baseline.max_cutoff,
        "warmup_rows": warmup_rows,
        "measured_rows": measured_rows,
        "measured_summary": summary,
        "cross_error": cross_error,
        "gates": gates,
        "passed": all(gates.values()),
        "claim_boundary": [
            "Local same-machine coefficient-kernel A/B only; FINUFFT is not part of this benchmark.",
            "Both paths use the same Miller order, cutoffs, form factors and FFT synthesis.",
            "The fused path precomputes one angular step per source and does not materialize a source-by-harmonic cache.",
            "The cached path materializes the exact source-by-harmonic angular phase table and is an explicit memory/speed tradeoff.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Exact-beta coefficient contraction backend comparison",
        "",
        f"- baseline coefficient median: `{summary['baseline_coefficient_seconds']['median']:.3f} s`",
        f"- fused-phase coefficient median: `{summary['fused_coefficient_seconds']['median']:.3f} s`",
        f"- cached-phase coefficient median: `{summary['cached_coefficient_seconds']['median']:.3f} s`",
        f"- paired coefficient speedup median / p05: `{summary['coefficient_speedup']['median']:.3f}x / {summary['coefficient_speedup']['p05']:.3f}x`",
        f"- fused/cached coefficient speedup median: `{summary['fused_over_cached_coefficient_speedup']['median']:.3f}x`",
        f"- fused-baseline / cached-fused complex L2: `{cross_error['fused_vs_baseline_complex_l2']:.3e} / {cross_error['cached_vs_fused_complex_l2']:.3e}`",
        f"- gate: **{'PASS' if result['passed'] else 'FAIL'}**",
        "",
    ]
    args.summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
