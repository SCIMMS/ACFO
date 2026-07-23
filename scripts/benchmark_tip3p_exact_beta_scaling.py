from __future__ import annotations

import argparse
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

from prepare_openmm_water_waxs_inputs import (  # noqa: E402
    center_from_box,
    iter_dcd_frames,
    load_npz,
)
from validate_public_waxs_structures import build_form_factors  # noqa: E402
from waxs_cake import (  # noqa: E402
    encode_elements,
    exact_coordinate_harmonic_amplitude,
    exact_coordinate_harmonic_amplitude_factorized,
)
from waxs_cake.geometry import ewald_ring  # noqa: E402
from waxs_cake.metrics import relative_l2  # noqa: E402
from waxs_cake.physical_scaling import q_to_inv_nm  # noqa: E402
from waxs_cake.solvers import normalize_form_factors  # noqa: E402


def timed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def write_markdown(result: dict, path: Path) -> None:
    lines = [
        "# TIP3P factorized exact-beta q-scaling",
        "",
        f"- atoms: `{result['atom_count']:,}`",
        f"- q range: `{result['q_range_inv_angstrom']}` inverse angstrom",
        f"- detector Nphi: `{result['target_nphi']}`",
        f"- factorized vs expanded q=2 complex L2: `{result['factorized_vs_expanded_q2_complex_l2']:.3e}`",
        "",
        "| Nq | targets | max h | median s | min-max s | atom-by-q matrix MiB avoided | accounted arrays MiB |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        timing = row["seconds"]
        memory = row["memory_accounting_mib"]
        lines.append(
            f"| {row['nq']} | {row['output_targets']:,} | {row['maximum_harmonic']} "
            f"| {timing['median']:.3f} | {timing['min']:.3f}-{timing['max']:.3f} "
            f"| {memory['avoided_atom_q_complex_matrix']:.1f} "
            f"| {memory['accounted_total']:.1f} |"
        )
    lines.extend(
        [
            "",
            f"- q=512 runtime gate <= 120 s: **{'PASS' if result['gates']['nq512_seconds_le_120'] else 'FAIL'}**",
            f"- q=512 accounted arrays <= 64 MiB: **{'PASS' if result['gates']['nq512_accounted_arrays_le_64_mib'] else 'FAIL'}**",
            "- Timings include coordinate preprocessing, fused contraction, and harmonic evaluation, but exclude file loading and form-factor construction.",
            "- Accounted arrays are not peak RSS; allocator, interpreter, extension, and input-file overhead are excluded.",
            "- This is a local CPU measurement for one TIP3P frame, not a hardware-independent speed claim.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure factorized fused exact-beta scaling without an atom-by-q coefficient matrix."
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=Path("runs/water_tip3p_8nm/water_tip3p_8nm_final.npz"),
    )
    parser.add_argument(
        "--trajectory-dcd",
        type=Path,
        default=Path("runs/water_tip3p_8nm/water_tip3p_8nm_trajectory.dcd"),
    )
    parser.add_argument("--q-counts", default="2,8,32,128,512")
    parser.add_argument("--q-min", type=float, default=5.0)
    parser.add_argument("--q-max", type=float, default=6.3)
    parser.add_argument("--wavelength-nm", type=float, default=0.08)
    parser.add_argument("--target-nphi", type=int, default=768)
    parser.add_argument("--harmonic-margin", type=int, default=48)
    parser.add_argument("--repeats-small", type=int, default=3)
    parser.add_argument("--repeats-large", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/tip3p_exact_beta_factorized_q_scaling.json"),
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path("benchmark_results/tip3p_exact_beta_factorized_q_scaling.md"),
    )
    args = parser.parse_args()

    q_counts = [int(value) for value in args.q_counts.split(",")]
    if any(value <= 0 for value in q_counts):
        raise ValueError("q counts must be positive")
    _, elements, _, source_metadata = load_npz(args.input_npz)
    element_indices, element_order = encode_elements(elements)
    frame_rows = list(iter_dcd_frames(args.trajectory_dcd, frame_indices={0}))
    if len(frame_rows) != 1:
        raise RuntimeError("failed to read trajectory frame 0")
    _, frame_coords, frame_box = frame_rows[0]
    coords = np.ascontiguousarray(
        frame_coords - center_from_box(frame_coords, frame_box)
    )
    atom_weights = np.ones(coords.shape[0], dtype=np.complex128)
    phi = (np.arange(args.target_nphi) + 0.5) * (
        2.0 * np.pi / args.target_nphi
    )

    rows = []
    factorized_vs_expanded_q2 = None
    for nq in q_counts:
        q_report = np.linspace(args.q_min, args.q_max, nq, dtype=np.float64)
        q_solver = np.asarray(
            [q_to_inv_nm(value, "inv_angstrom") for value in q_report]
        )
        q_perp, q_z = ewald_ring(q_solver, args.wavelength_nm)
        ff_mapping = build_form_factors(elements, q_solver, "xray_f0")
        form_factors = normalize_form_factors(
            element_order, q_solver, ff_mapping
        ).astype(np.complex128, copy=False)

        def evaluate_factorized():
            return exact_coordinate_harmonic_amplitude_factorized(
                coords,
                q_perp,
                q_z,
                phi,
                element_indices=element_indices,
                form_factors=form_factors,
                atom_weights=atom_weights,
                harmonic_margin=args.harmonic_margin,
            )

        (amplitude, cutoffs), _ = timed(evaluate_factorized)
        repeats = args.repeats_small if nq <= 32 else args.repeats_large
        times = []
        for _ in range(repeats):
            (amplitude, cutoffs), elapsed = timed(evaluate_factorized)
            times.append(elapsed)

        if nq == 2:
            expanded, _ = exact_coordinate_harmonic_amplitude(
                coords,
                q_perp,
                q_z,
                phi,
                atom_coefficients=form_factors[element_indices],
                harmonic_margin=args.harmonic_margin,
                bessel_backend="cpp_fused",
            )
            factorized_vs_expanded_q2 = relative_l2(amplitude, expanded)

        maximum_harmonic = int(np.max(cutoffs))
        n_threads = min(nq, os.cpu_count() or 1)
        memory = {
            "coordinates": coords.nbytes / 1024**2,
            "element_indices": element_indices.nbytes / 1024**2,
            "atom_weights": atom_weights.nbytes / 1024**2,
            "form_factors": form_factors.nbytes / 1024**2,
            "harmonic_coefficients": (
                nq * 2 * (maximum_harmonic + 1) * 16 / 1024**2
            ),
            "amplitude_output": nq * args.target_nphi * 16 / 1024**2,
            "per_q_basis": (
                (maximum_harmonic + 1) * args.target_nphi * 16 / 1024**2
            ),
            "thread_scratch_upper_bound": (
                n_threads
                * (
                    (maximum_harmonic + 1) * 16
                    + (maximum_harmonic + 1 + 32) * 8
                )
                / 1024**2
            ),
            "avoided_atom_q_complex_matrix": (
                coords.shape[0] * nq * 16 / 1024**2
            ),
        }
        memory["accounted_total"] = sum(
            memory[key]
            for key in (
                "coordinates",
                "element_indices",
                "atom_weights",
                "form_factors",
                "harmonic_coefficients",
                "amplitude_output",
                "per_q_basis",
                "thread_scratch_upper_bound",
            )
        )
        rows.append(
            {
                "nq": nq,
                "output_targets": nq * args.target_nphi,
                "maximum_harmonic": maximum_harmonic,
                "repeats": repeats,
                "seconds": summary(times),
                "memory_accounting_mib": memory,
            }
        )

    if factorized_vs_expanded_q2 is None:
        raise ValueError("q-counts must include 2 for the factorization check")
    row512 = next((row for row in rows if row["nq"] == 512), None)
    if row512 is None:
        raise ValueError("q-counts must include 512 for the production-size gate")
    gates = {
        "factorized_vs_expanded_q2_complex_l2_le_1e_12": (
            factorized_vs_expanded_q2 <= 1e-12
        ),
        "all_harmonics_below_target_nyquist": all(
            row["maximum_harmonic"] < args.target_nphi // 2 for row in rows
        ),
        "nq512_seconds_le_120": row512["seconds"]["median"] <= 120.0,
        "nq512_accounted_arrays_le_64_mib": (
            row512["memory_accounting_mib"]["accounted_total"] <= 64.0
        ),
    }
    result = {
        "schema": "tip3p-exact-beta-factorized-q-scaling-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "input_npz": args.input_npz.as_posix(),
            "trajectory_dcd": args.trajectory_dcd.as_posix(),
            "frame_index": 0,
            "source_metadata": source_metadata,
        },
        "atom_count": int(coords.shape[0]),
        "q_range_inv_angstrom": [args.q_min, args.q_max],
        "target_nphi": args.target_nphi,
        "target_nyquist": args.target_nphi // 2,
        "harmonic_margin": args.harmonic_margin,
        "timing_scope": (
            "local CPU wall time including coordinate preprocessing, fused contraction, "
            "and harmonic evaluation; excludes file loading and form-factor construction"
        ),
        "factorized_vs_expanded_q2_complex_l2": factorized_vs_expanded_q2,
        "rows": rows,
        "gates": gates,
        "passed": all(gates.values()),
        "claim_boundary": [
            "Nq=512 timing is a one-frame local CPU measurement, not an independent-machine result.",
            "Accounted arrays are an analytic inventory, not measured peak RSS.",
            "The factorized API applies when per-atom coefficients separate into atom weights and element form factors.",
            "Direct NDFT correctness is established separately on small cases and 20 TIP3P frames.",
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
