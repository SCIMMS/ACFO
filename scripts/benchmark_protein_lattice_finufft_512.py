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

from benchmark_physical_scaling import timed_call  # noqa: E402
from benchmark_protein_nanocrystal_finufft_fair import build_finufft_plans  # noqa: E402
from validate_public_waxs_structures import build_form_factors, load_structure  # noqa: E402
from waxs_cake import (  # noqa: E402
    encode_elements,
    exact_coordinate_harmonic_amplitude_factorized,
    repeated_block_translations,
    translation_lattice_factor,
)
from waxs_cake.geometry import ewald_ring  # noqa: E402
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402
from waxs_cake.physical_scaling import q_to_inv_nm  # noqa: E402
from waxs_cake.solvers import normalize_form_factors  # noqa: E402


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def write_markdown(result: dict, path: Path) -> None:
    lines = [
        "# Perfect protein crystal lattice factorization vs FINUFFT, Nq=512",
        "",
        f"- unit atoms: `{result['unit_atom_count']:,}`",
        f"- targets: `{result['target_count']:,}`",
        f"- FINUFFT eps / threads: `{result['finufft_eps']:.0e} / {result['finufft_threads']}`",
        "",
        "| case | atoms | cells | factorized first/hot s | FINUFFT first-total/hot s | hot speedup | complex L2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['label']} | {row['atom_count']:,} | {row['cell_count']} "
            f"| {row['factorized']['first_total_seconds']:.3f}/{row['factorized']['hot_seconds']['median']:.3f} "
            f"| {row['finufft']['first_total_seconds']:.3f}/{row['finufft']['hot_seconds']['median']:.3f} "
            f"| {row['speedup']['hot_finufft_over_factorized']:.3f}x "
            f"| {row['cross_error']['complex_l2']:.3e} |"
        )
    lines.extend(
        [
            "",
            f"- structured-regime performance gate: **{'PASS' if result['comparative_performance_pass'] else 'FAIL'}**",
            "- Direct NDFT correctness is carried by the separate q=3/subset lattice control.",
            "- The lattice-factor path is a standard crystallographic specialization, not an ACFO novelty claim.",
            "- Timings are local CPU measurements; first-total and hot paths are reported separately.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matched exact lattice-factorization versus reusable FINUFFT at Nq=512."
    )
    parser.add_argument(
        "--unit",
        type=Path,
        default=Path("structures/processed/protein_nanocrystal_lysozyme_1iee_1x1x1_fixed.npz"),
    )
    parser.add_argument(
        "--supercells",
        default=(
            "structures/processed/protein_nanocrystal_lysozyme_1iee_3x3x3_fixed.npz,"
            "structures/processed/protein_nanocrystal_lysozyme_1iee_5x5x5_fixed.npz"
        ),
    )
    parser.add_argument("--nq", type=int, default=512)
    parser.add_argument("--q-min", type=float, default=5.0)
    parser.add_argument("--q-max", type=float, default=6.3)
    parser.add_argument("--wavelength-nm", type=float, default=0.08)
    parser.add_argument("--target-nphi", type=int, default=768)
    parser.add_argument("--harmonic-margin", type=int, default=48)
    parser.add_argument("--factorized-hot-repeats", type=int, default=3)
    parser.add_argument("--finufft-eps", type=float, default=1e-6)
    parser.add_argument("--finufft-threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--finufft-hot-repeats", type=int, default=3)
    parser.add_argument("--memory-sample-interval-s", type=float, default=0.02)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/protein_lattice_finufft_512.json"),
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path("benchmark_results/protein_lattice_finufft_512.md"),
    )
    args = parser.parse_args()

    unit_coords, unit_elements, unit_metadata = load_structure(args.unit)
    unit_e, element_order = encode_elements(unit_elements)
    q_report = np.linspace(args.q_min, args.q_max, args.nq, dtype=np.float64)
    q_solver = np.asarray(
        [q_to_inv_nm(value, "inv_angstrom") for value in q_report]
    )
    q_perp, q_z_rows = ewald_ring(q_solver, args.wavelength_nm)
    ff_mapping = build_form_factors(unit_elements, q_solver, "xray_f0")
    form_factors = normalize_form_factors(
        element_order, q_solver, ff_mapping
    ).astype(np.complex128, copy=False)
    phi = (np.arange(args.target_nphi) + 0.5) * (
        2.0 * np.pi / args.target_nphi
    )
    qx = np.ascontiguousarray(
        (q_perp[:, None] * np.cos(phi)[None, :]).ravel()
    )
    qy = np.ascontiguousarray(
        (q_perp[:, None] * np.sin(phi)[None, :]).ravel()
    )
    qz = np.ascontiguousarray(
        np.broadcast_to(q_z_rows[:, None], (args.nq, args.target_nphi)).ravel()
    )
    target_q_indices = np.repeat(np.arange(args.nq), args.target_nphi)
    unit_weights = np.ones(unit_coords.shape[0], dtype=np.complex128)

    def unit_execute():
        return exact_coordinate_harmonic_amplitude_factorized(
            unit_coords,
            q_perp,
            q_z_rows,
            phi,
            element_indices=unit_e,
            form_factors=form_factors,
            atom_weights=unit_weights,
            harmonic_margin=args.harmonic_margin,
        )

    (unit_first, cutoffs), unit_first_seconds, unit_memory = timed_call(
        unit_execute,
        measure_memory=True,
        sample_interval_s=args.memory_sample_interval_s,
    )
    print(f"unit exact-beta first: {unit_first_seconds:.3f} s", flush=True)
    rows = []
    for supercell_text in args.supercells.split(","):
        path = Path(supercell_text.strip())
        coords, elements, metadata = load_structure(path)
        blocks = elements.reshape(-1, unit_elements.size)
        if not np.array_equal(
            blocks,
            np.broadcast_to(unit_elements, blocks.shape),
        ):
            raise RuntimeError(f"{path} does not repeat the unit element ordering")
        translations, repetition_residual = repeated_block_translations(
            unit_coords, coords, atol=1e-9
        )
        lattice, lattice_seconds, lattice_memory = timed_call(
            lambda: translation_lattice_factor(qx, qy, qz, translations),
            measure_memory=True,
            sample_interval_s=args.memory_sample_interval_s,
        )
        lattice_2d = lattice.reshape(args.nq, args.target_nphi)
        factorized_first = unit_first * lattice_2d
        print(
            f"{path.stem}: lattice factor {lattice_seconds:.3f} s; starting factorized hot",
            flush=True,
        )

        factorized_hot_times = []
        factorized_hot = factorized_first
        for _ in range(args.factorized_hot_repeats):
            gc.collect()
            start = time.perf_counter()
            unit_hot, _ = unit_execute()
            factorized_hot = unit_hot * lattice_2d
            factorized_hot_times.append(time.perf_counter() - start)
        print(
            f"{path.stem}: factorized hot median {np.median(factorized_hot_times):.3f} s; starting FINUFFT setup",
            flush=True,
        )

        element_indices, _ = encode_elements(elements, element_order=element_order)
        source_weights = np.ones(coords.shape[0], dtype=np.complex128)
        (plans, masks), setup_seconds, setup_memory = timed_call(
            lambda: build_finufft_plans(
                coords,
                element_indices,
                qx,
                qy,
                qz,
                n_elements=len(element_order),
                eps=args.finufft_eps,
                threads=args.finufft_threads,
            ),
            measure_memory=True,
            sample_interval_s=args.memory_sample_interval_s,
        )
        print(
            f"{path.stem}: FINUFFT setup {setup_seconds:.3f} s; starting first execute",
            flush=True,
        )

        def finufft_execute() -> np.ndarray:
            active = np.zeros(qx.size, dtype=np.complex128)
            for element_index, (plan, mask) in enumerate(zip(plans, masks)):
                values = plan.execute(np.ascontiguousarray(source_weights[mask]))
                active += values * form_factors[element_index, target_q_indices]
            return active.reshape(args.nq, args.target_nphi)

        finufft_first, finufft_first_seconds, first_memory = timed_call(
            finufft_execute,
            measure_memory=True,
            sample_interval_s=args.memory_sample_interval_s,
        )
        print(
            f"{path.stem}: FINUFFT first {finufft_first_seconds:.3f} s; starting hot repeats",
            flush=True,
        )
        finufft_hot_times = []
        finufft_hot = finufft_first
        for _ in range(args.finufft_hot_repeats):
            gc.collect()
            start = time.perf_counter()
            finufft_hot = finufft_execute()
            finufft_hot_times.append(time.perf_counter() - start)
        print(
            f"{path.stem}: FINUFFT hot median {np.median(finufft_hot_times):.3f} s",
            flush=True,
        )

        factorized_hot_summary = summary(factorized_hot_times)
        finufft_hot_summary = summary(finufft_hot_times)
        factorized_first_total = unit_first_seconds + lattice_seconds
        finufft_first_total = setup_seconds + finufft_first_seconds
        cross_error = {
            "complex_l2": relative_l2(finufft_hot, factorized_hot),
            "intensity_l2": relative_l2(
                intensity(finufft_hot), intensity(factorized_hot)
            ),
        }
        rows.append(
            {
                "label": "x".join(str(value) for value in metadata["supercell"]),
                "structure_path": path.as_posix(),
                "atom_count": int(coords.shape[0]),
                "cell_count": int(translations.shape[0]),
                "repetition_residual_nm": repetition_residual,
                "factorized": {
                    "unit_first_seconds": unit_first_seconds,
                    "lattice_factor_seconds": lattice_seconds,
                    "first_total_seconds": factorized_first_total,
                    "hot_repeats": args.factorized_hot_repeats,
                    "hot_seconds": factorized_hot_summary,
                    "unit_peak_rss_delta_mib": unit_memory["peak_rss_delta_mib"],
                    "lattice_peak_rss_delta_mib": lattice_memory["peak_rss_delta_mib"],
                },
                "finufft": {
                    "plan_count": len(plans),
                    "setup_seconds": setup_seconds,
                    "first_execute_seconds": finufft_first_seconds,
                    "first_total_seconds": finufft_first_total,
                    "hot_repeats": args.finufft_hot_repeats,
                    "hot_seconds": finufft_hot_summary,
                    "setup_peak_rss_delta_mib": setup_memory["peak_rss_delta_mib"],
                    "execute_peak_rss_delta_mib": first_memory["peak_rss_delta_mib"],
                },
                "speedup": {
                    "first_total_finufft_over_factorized": (
                        finufft_first_total / factorized_first_total
                    ),
                    "hot_finufft_over_factorized": (
                        finufft_hot_summary["median"]
                        / factorized_hot_summary["median"]
                    ),
                },
                "cross_error": cross_error,
            }
        )
        del plans, masks, finufft_first, finufft_hot
        gc.collect()

    gates = {
        "maximum_harmonic_below_target_nyquist": (
            int(np.max(cutoffs)) < args.target_nphi // 2
        ),
        "all_repetition_residual_le_1e_9_nm": all(
            row["repetition_residual_nm"] <= 1e-9 for row in rows
        ),
        "all_cross_complex_l2_le_2e_6": all(
            row["cross_error"]["complex_l2"] <= 2e-6 for row in rows
        ),
        "all_cross_intensity_l2_le_5e_6": all(
            row["cross_error"]["intensity_l2"] <= 5e-6 for row in rows
        ),
    }
    comparative_performance_pass = all(
        row["speedup"]["first_total_finufft_over_factorized"] >= 1.0
        and row["speedup"]["hot_finufft_over_factorized"] >= 1.0
        for row in rows
    )
    result = {
        "schema": "protein-lattice-finufft-512-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "unit_structure": unit_metadata.get("structure_id", args.unit.stem),
        "unit_atom_count": int(unit_coords.shape[0]),
        "q_inv_angstrom": [args.q_min, args.q_max],
        "nq": args.nq,
        "target_nphi": args.target_nphi,
        "target_count": int(qx.size),
        "maximum_harmonic": int(np.max(cutoffs)),
        "finufft_eps": args.finufft_eps,
        "finufft_threads": args.finufft_threads,
        "rows": rows,
        "gates": gates,
        "validity_pass": all(gates.values()),
        "comparative_performance_pass": comparative_performance_pass,
        "passed": all(gates.values()),
        "decision": (
            "Keep the structured repeated-crystal performance claim."
            if comparative_performance_pass
            else "Do not claim a general repeated-crystal speed advantage from this workload."
        ),
        "claim_boundary": [
            "The lattice factor is the standard exact crystallographic specialization and is not an ACFO novelty claim.",
            "Direct NDFT correctness is established by the separate q=3/subset control; Nq=512 is an optimized-method cross-check.",
            "FINUFFT eps=1e-6 is a practical timing baseline, not a converged correctness oracle.",
            "Timings and sampled RSS are local to this machine, build, and thread policy.",
            "The hot factorized path recomputes the unit-cell amplitude while reusing the finite lattice factor; FINUFFT reuses its plan.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(result, args.summary_md)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
