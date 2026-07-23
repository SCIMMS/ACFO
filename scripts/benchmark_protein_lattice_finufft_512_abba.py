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

from benchmark_protein_nanocrystal_finufft_fair import build_finufft_plans  # noqa: E402
from validate_public_waxs_structures import build_form_factors, load_structure  # noqa: E402
from waxs_cake import (  # noqa: E402
    PreparedExactCoordinateHarmonicPlan,
    encode_elements,
    exact_coordinate_harmonic_amplitude_factorized,
    repeated_block_translations,
    translation_lattice_factor,
    translation_lattice_factor_separable,
)
from waxs_cake.geometry import ewald_ring  # noqa: E402
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402
from waxs_cake.physical_scaling import q_to_inv_nm  # noqa: E402
from waxs_cake.solvers import normalize_form_factors  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "cv": None,
            "p05": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return {
        "count": int(array.size),
        "mean": mean,
        "median": float(np.median(array)),
        "std": std,
        "cv": 0.0 if mean == 0.0 else std / mean,
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def write_markdown(result: dict, path: Path) -> None:
    measured = result["measured_summary"]
    cross = result["cross_error"]
    lines = [
        f"# Million-atom repeated-crystal 10/30 AB/BA timing ({result['factorized_backend']})",
        "",
        f"- status: `{result['status']}`",
        f"- atoms / targets: `{result['atom_count']:,} / {result['target_count']:,}`",
        f"- warm-up pairs / measured pairs: `{result['warmup_pairs_completed']} / {result['measured_pairs_completed']}`",
        f"- factorized median: `{measured['factorized_seconds']['median']:.3f} s`",
        f"- FINUFFT median: `{measured['finufft_seconds']['median']:.3f} s`",
        f"- paired speedup median / p05: `{measured['paired_speedup']['median']:.3f}x / {measured['paired_speedup']['p05']:.3f}x`",
        f"- cross complex/intensity L2: `{cross['complex_l2']:.3e} / {cross['intensity_l2']:.3e}`",
        f"- local timing gate: **{'PASS' if result['local_timing_gate_pass'] else 'FAIL'}**",
        "",
        "| pair | order | factorized s | FINUFFT s | paired speedup |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in result["measured_pairs"]:
        lines.append(
            f"| {row['pair_index']} | {row['order']} "
            f"| {row['factorized_seconds']:.3f} | {row['finufft_seconds']:.3f} "
            f"| {row['paired_speedup']:.3f}x |"
        )
    lines.extend(
        [
            "",
            f"- Factorized backend: `{result['factorized_backend']}`; lattice backend: `{result['lattice_backend']}`.",
            "- The factorized hot path recomputes the unit-cell amplitude while reusing its explicit prepared state and finite lattice factor when selected.",
            "- FINUFFT reuses four element-specific type-3 plans at eps=1e-6.",
            "- Direct NDFT correctness is established by the separate q=3/subset control.",
            "- This remains a same-machine result until independently repeated.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resumable million-atom exact lattice-factor/FINUFFT AB/BA timing."
    )
    parser.add_argument(
        "--unit",
        type=Path,
        default=Path("structures/processed/protein_nanocrystal_lysozyme_1iee_1x1x1_fixed.npz"),
    )
    parser.add_argument(
        "--supercell",
        type=Path,
        default=Path("structures/processed/protein_nanocrystal_lysozyme_1iee_5x5x5_fixed.npz"),
    )
    parser.add_argument("--nq", type=int, default=512)
    parser.add_argument("--q-min", type=float, default=5.0)
    parser.add_argument("--q-max", type=float, default=6.3)
    parser.add_argument("--wavelength-nm", type=float, default=0.08)
    parser.add_argument("--target-nphi", type=int, default=768)
    parser.add_argument("--harmonic-margin", type=int, default=48)
    parser.add_argument("--finufft-eps", type=float, default=1e-6)
    parser.add_argument("--finufft-threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--warmup-pairs", type=int, default=10)
    parser.add_argument("--measured-pairs", type=int, default=30)
    parser.add_argument(
        "--factorized-backend",
        choices=("legacy", "prepared_fused"),
        default="legacy",
    )
    parser.add_argument(
        "--lattice-backend", choices=("direct", "separable"), default="direct"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/protein_lattice_finufft_512_abba.json"),
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path("benchmark_results/protein_lattice_finufft_512_abba.md"),
    )
    args = parser.parse_args()
    if args.warmup_pairs < 0 or args.measured_pairs <= 0:
        raise ValueError("warmup-pairs must be non-negative and measured-pairs positive")

    contract = {
        "unit": args.unit.as_posix(),
        "supercell": args.supercell.as_posix(),
        "nq": args.nq,
        "q_min_inv_angstrom": args.q_min,
        "q_max_inv_angstrom": args.q_max,
        "wavelength_nm": args.wavelength_nm,
        "target_nphi": args.target_nphi,
        "harmonic_margin": args.harmonic_margin,
        "finufft_eps": args.finufft_eps,
        "finufft_threads": args.finufft_threads,
        "warmup_pairs_requested": args.warmup_pairs,
        "measured_pairs_requested": args.measured_pairs,
    }
    if args.factorized_backend != "legacy" or args.lattice_backend != "direct":
        contract.update(
            {
                "factorized_backend": args.factorized_backend,
                "lattice_backend": args.lattice_backend,
                "coefficient_backend": (
                    "fused_phase"
                    if args.factorized_backend == "prepared_fused"
                    else "legacy"
                ),
                "synthesis_backend": (
                    "fft" if args.factorized_backend == "prepared_fused" else "legacy"
                ),
            }
        )
    schema = (
        "protein-lattice-prepared-finufft-512-abba-v1"
        if args.factorized_backend == "prepared_fused"
        else "protein-lattice-finufft-512-abba-v1"
    )
    previous_pairs: list[dict] = []
    previous_created = utc_now()
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous.get("contract") != contract:
            raise RuntimeError("resume contract does not match the existing checkpoint")
        previous_pairs = list(previous.get("measured_pairs", []))
        previous_created = previous.get("created_at_utc", previous_created)
        if len(previous_pairs) >= args.measured_pairs:
            print("requested measured pairs are already complete", flush=True)
            return

    unit_coords, unit_elements, unit_metadata = load_structure(args.unit)
    coords, elements, metadata = load_structure(args.supercell)
    blocks = elements.reshape(-1, unit_elements.size)
    if not np.array_equal(blocks, np.broadcast_to(unit_elements, blocks.shape)):
        raise RuntimeError("supercell does not repeat the unit element ordering")
    translations, repetition_residual = repeated_block_translations(
        unit_coords, coords, atol=1e-9
    )
    unit_e, element_order = encode_elements(unit_elements)
    element_indices, _ = encode_elements(elements, element_order=element_order)
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
    source_weights = np.ones(coords.shape[0], dtype=np.complex128)

    setup_start = time.perf_counter()
    if args.lattice_backend == "separable":
        lattice = translation_lattice_factor_separable(
            qx, qy, qz, translations, metadata["supercell"]
        ).reshape(args.nq, args.target_nphi)
    else:
        lattice = translation_lattice_factor(qx, qy, qz, translations).reshape(
            args.nq, args.target_nphi
        )
    lattice_setup_seconds = time.perf_counter() - setup_start
    prepared_plan = None
    prepared_plan_setup_seconds = 0.0
    if args.factorized_backend == "prepared_fused":
        setup_start = time.perf_counter()
        prepared_plan = PreparedExactCoordinateHarmonicPlan(
            unit_coords,
            q_perp,
            q_z_rows,
            phi,
            element_indices=unit_e,
            form_factors=form_factors,
            harmonic_margin=args.harmonic_margin,
            prepare_direct_basis=False,
            coefficient_backend="fused_phase",
        )
        if not prepared_plan.fft_supported:
            raise RuntimeError("prepared fused plan does not support FFT synthesis")
        prepared_plan_setup_seconds = time.perf_counter() - setup_start
    setup_start = time.perf_counter()
    plans, masks = build_finufft_plans(
        coords,
        element_indices,
        qx,
        qy,
        qz,
        n_elements=len(element_order),
        eps=args.finufft_eps,
        threads=args.finufft_threads,
    )
    finufft_setup_seconds = time.perf_counter() - setup_start
    print(
        f"setup: prepared {prepared_plan_setup_seconds:.3f} s; "
        f"lattice {lattice_setup_seconds:.3f} s; FINUFFT {finufft_setup_seconds:.3f} s",
        flush=True,
    )

    def factorized_execute() -> np.ndarray:
        if prepared_plan is None:
            unit, _ = exact_coordinate_harmonic_amplitude_factorized(
                unit_coords,
                q_perp,
                q_z_rows,
                phi,
                element_indices=unit_e,
                form_factors=form_factors,
                atom_weights=unit_weights,
                harmonic_margin=args.harmonic_margin,
            )
        else:
            unit, _ = prepared_plan.execute(
                atom_weights=unit_weights, synthesis_backend="fft"
            )
        return unit * lattice

    def finufft_execute() -> np.ndarray:
        active = np.zeros(qx.size, dtype=np.complex128)
        for element_index, (plan, mask) in enumerate(zip(plans, masks)):
            values = plan.execute(np.ascontiguousarray(source_weights[mask]))
            active += values * form_factors[element_index, target_q_indices]
        return active.reshape(args.nq, args.target_nphi)

    last_factorized = None
    last_finufft = None
    legacy_comparison = None
    if prepared_plan is not None:
        comparison_start = time.perf_counter()
        legacy_unit, _ = exact_coordinate_harmonic_amplitude_factorized(
            unit_coords,
            q_perp,
            q_z_rows,
            phi,
            element_indices=unit_e,
            form_factors=form_factors,
            atom_weights=unit_weights,
            harmonic_margin=args.harmonic_margin,
        )
        legacy_value = legacy_unit * lattice
        legacy_seconds = time.perf_counter() - comparison_start
        comparison_start = time.perf_counter()
        prepared_value = factorized_execute()
        prepared_seconds = time.perf_counter() - comparison_start
        legacy_comparison = {
            "legacy_seconds": legacy_seconds,
            "prepared_seconds": prepared_seconds,
            "legacy_vs_prepared_complex_l2": relative_l2(
                legacy_value, prepared_value
            ),
            "legacy_vs_prepared_intensity_l2": relative_l2(
                intensity(legacy_value), intensity(prepared_value)
            ),
        }
        del legacy_unit, legacy_value, prepared_value
        gc.collect()
    warmup_rows = []
    run_start = time.perf_counter()
    for pair_index in range(args.warmup_pairs):
        order = "AB" if pair_index % 2 == 0 else "BA"
        row = {"pair_index": pair_index + 1, "order": order}
        for label in order:
            gc.collect()
            start = time.perf_counter()
            if label == "A":
                last_factorized = factorized_execute()
                row["factorized_seconds"] = time.perf_counter() - start
            else:
                last_finufft = finufft_execute()
                row["finufft_seconds"] = time.perf_counter() - start
        warmup_rows.append(row)
        print(
            f"warmup {pair_index + 1}/{args.warmup_pairs}: A {row['factorized_seconds']:.3f} s; B {row['finufft_seconds']:.3f} s",
            flush=True,
        )

    measured_rows = previous_pairs
    for pair_index in range(len(measured_rows), args.measured_pairs):
        order = "AB" if pair_index % 2 == 0 else "BA"
        row = {"pair_index": pair_index + 1, "order": order}
        for label in order:
            gc.collect()
            start = time.perf_counter()
            if label == "A":
                last_factorized = factorized_execute()
                row["factorized_seconds"] = time.perf_counter() - start
            else:
                last_finufft = finufft_execute()
                row["finufft_seconds"] = time.perf_counter() - start
        row["paired_speedup"] = (
            row["finufft_seconds"] / row["factorized_seconds"]
        )
        measured_rows.append(row)
        payload = {
            "schema": schema,
            "status": "running",
            "created_at_utc": previous_created,
            "updated_at_utc": utc_now(),
            "contract": contract,
            "unit_structure": unit_metadata.get("structure_id", args.unit.stem),
            "supercell": metadata.get("supercell"),
            "atom_count": int(coords.shape[0]),
            "cell_count": int(translations.shape[0]),
            "repetition_residual_nm": repetition_residual,
            "target_count": int(qx.size),
            "lattice_setup_seconds": lattice_setup_seconds,
            "finufft_setup_seconds": finufft_setup_seconds,
            "prepared_plan_setup_seconds": prepared_plan_setup_seconds,
            "factorized_backend": args.factorized_backend,
            "lattice_backend": args.lattice_backend,
            "legacy_comparison": legacy_comparison,
            "warmup_pairs_completed": len(warmup_rows),
            "warmup_pairs": warmup_rows,
            "measured_pairs_completed": len(measured_rows),
            "measured_pairs": measured_rows,
            "elapsed_this_run_seconds": time.perf_counter() - run_start,
            "passed": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.output, payload)
        print(
            f"measured {pair_index + 1}/{args.measured_pairs}: A {row['factorized_seconds']:.3f} s; B {row['finufft_seconds']:.3f} s; speedup {row['paired_speedup']:.3f}x",
            flush=True,
        )

    if last_factorized is None:
        last_factorized = factorized_execute()
    if last_finufft is None:
        last_finufft = finufft_execute()
    cross_error = {
        "complex_l2": relative_l2(last_finufft, last_factorized),
        "intensity_l2": relative_l2(
            intensity(last_finufft), intensity(last_factorized)
        ),
    }
    factorized_times = [row["factorized_seconds"] for row in measured_rows]
    finufft_times = [row["finufft_seconds"] for row in measured_rows]
    speedups = [row["paired_speedup"] for row in measured_rows]
    measured_summary = {
        "factorized_seconds": distribution(factorized_times),
        "finufft_seconds": distribution(finufft_times),
        "paired_speedup": distribution(speedups),
    }
    local_timing_gate = (
        len(measured_rows) == args.measured_pairs
        and measured_summary["paired_speedup"]["median"] >= 3.0
        and measured_summary["paired_speedup"]["p05"] >= 3.0
    )
    gates = {
        "warmup_count_matches_request": len(warmup_rows) == args.warmup_pairs,
        "measured_count_matches_request": len(measured_rows) == args.measured_pairs,
        "cross_complex_l2_le_2e_6": cross_error["complex_l2"] <= 2e-6,
        "cross_intensity_l2_le_5e_6": cross_error["intensity_l2"] <= 5e-6,
        "paired_speedup_median_ge_3": (
            measured_summary["paired_speedup"]["median"] >= 3.0
        ),
        "paired_speedup_p05_ge_3": (
            measured_summary["paired_speedup"]["p05"] >= 3.0
        ),
        "legacy_comparison_l2_le_1e_12": (
            legacy_comparison is None
            or legacy_comparison["legacy_vs_prepared_complex_l2"] <= 1e-12
        ),
    }
    result = {
        "schema": schema,
        "status": "complete",
        "created_at_utc": previous_created,
        "updated_at_utc": utc_now(),
        "contract": contract,
        "unit_structure": unit_metadata.get("structure_id", args.unit.stem),
        "supercell": metadata.get("supercell"),
        "atom_count": int(coords.shape[0]),
        "cell_count": int(translations.shape[0]),
        "repetition_residual_nm": repetition_residual,
        "target_count": int(qx.size),
        "lattice_setup_seconds": lattice_setup_seconds,
        "finufft_setup_seconds": finufft_setup_seconds,
        "prepared_plan_setup_seconds": prepared_plan_setup_seconds,
        "factorized_backend": args.factorized_backend,
        "lattice_backend": args.lattice_backend,
        "legacy_comparison": legacy_comparison,
        "warmup_pairs_completed": len(warmup_rows),
        "warmup_pairs": warmup_rows,
        "measured_pairs_completed": len(measured_rows),
        "measured_pairs": measured_rows,
        "measured_summary": measured_summary,
        "cross_error": cross_error,
        "elapsed_this_run_seconds": time.perf_counter() - run_start,
        "gates": gates,
        "local_timing_gate_pass": local_timing_gate,
        "passed": all(gates.values()),
        "claim_boundary": [
            f"This is a same-machine {args.warmup_pairs}-warmup/{args.measured_pairs}-pair AB/BA result, not an independent-machine replication.",
            "The finite lattice factor is a standard crystallographic specialization and not an ACFO novelty claim.",
            f"The timing claim applies only to the tested {coords.shape[0]:,}-atom exact repeated crystal.",
            "FINUFFT eps=1e-6 is a practical timing baseline; direct NDFT correctness is established separately.",
            "Prepared and FINUFFT plan setup are excluded from paired hot execution and reported separately.",
        ],
    }
    atomic_write_json(args.output, result)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(result, args.summary_md)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
