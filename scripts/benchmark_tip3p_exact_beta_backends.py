from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_protein_nanocrystal_finufft_fair import detector_rectangle_mask  # noqa: E402
from prepare_openmm_water_waxs_inputs import (  # noqa: E402
    center_from_box,
    iter_dcd_frames,
    load_npz,
)
from validate_public_waxs_structures import build_form_factors  # noqa: E402
from validate_waxs_direct_finufft_triad import direct_active_amplitude  # noqa: E402
from waxs_cake import encode_elements, exact_coordinate_harmonic_amplitude  # noqa: E402
from waxs_cake.geometry import ewald_ring  # noqa: E402
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402
from waxs_cake.physical_scaling import q_to_inv_nm  # noqa: E402
from waxs_cake.solvers import normalize_form_factors  # noqa: E402


def timed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def error_metrics(model: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    return {
        "complex_l2": relative_l2(model, reference),
        "intensity_l2": relative_l2(intensity(model), intensity(reference)),
    }


def write_markdown(result: dict, path: Path) -> None:
    lines = [
        "# TIP3P exact-beta backend comparison",
        "",
        f"- atoms: `{result['atom_count']:,}`",
        f"- q rows / active detector targets: `{result['nq']} / {result['active_targets']}`",
        f"- maximum harmonic / detector Nyquist: `{result['maximum_harmonic']} / {result['target_nyquist']}`",
        "",
        "| backend | time s | direct/backend | complex L2 vs direct | intensity L2 vs direct |",
        "|---|---:|---:|---:|---:|",
        f"| direct NDFT | {result['direct_ndft_seconds']:.3f} | 1.000x | oracle | oracle |",
    ]
    for name in ("cpp_miller", "cpp_fused"):
        row = result["backends"][name]
        lines.append(
            f"| {name} | {row['seconds']:.3f} | {row['direct_over_backend_speedup']:.3f}x "
            f"| {row['complex_l2_vs_direct']:.3e} | {row['intensity_l2_vs_direct']:.3e} |"
        )
    lines.extend(
        [
            "",
            f"- fused speedup vs cpp_miller: `{result['fused_over_cpp_miller_speedup']:.3f}x`",
            f"- comparison gate: **{'PASS' if result['passed'] else 'FAIL'}**",
            "- Both optimized rows use the same cutoff, form factors, coordinates, and detector samples.",
            "- Direct NDFT is the correctness oracle; this is a local CPU wall-time snapshot.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matched direct/cpp_miller/cpp_fused TIP3P comparison."
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
    parser.add_argument("--nq", type=int, default=8)
    parser.add_argument("--q-min", type=float, default=5.0)
    parser.add_argument("--q-max", type=float, default=6.3)
    parser.add_argument("--wavelength-nm", type=float, default=0.08)
    parser.add_argument("--target-nphi", type=int, default=768)
    parser.add_argument("--harmonic-margin", type=int, default=48)
    parser.add_argument("--atom-chunk-size", type=int, default=256)
    parser.add_argument("--direct-target-chunk", type=int, default=128)
    parser.add_argument("--detector-active-width-mm", type=float, default=155.1)
    parser.add_argument("--detector-active-height-mm", type=float, default=162.15)
    parser.add_argument("--detector-distance-mm", type=float, default=100.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/tip3p_exact_beta_backend_comparison.json"),
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path("benchmark_results/tip3p_exact_beta_backend_comparison.md"),
    )
    args = parser.parse_args()

    _, elements, _, source_metadata = load_npz(args.input_npz)
    element_indices, element_order = encode_elements(elements)
    frame_rows = list(iter_dcd_frames(args.trajectory_dcd, frame_indices={0}))
    if len(frame_rows) != 1:
        raise RuntimeError("failed to read trajectory frame 0")
    _, frame_coords, frame_box = frame_rows[0]
    coords = np.ascontiguousarray(
        frame_coords - center_from_box(frame_coords, frame_box)
    )
    q_report = np.linspace(args.q_min, args.q_max, args.nq, dtype=np.float64)
    q_solver = np.asarray(
        [q_to_inv_nm(value, "inv_angstrom") for value in q_report]
    )
    q_perp, q_z_rows = ewald_ring(q_solver, args.wavelength_nm)
    ff_mapping = build_form_factors(elements, q_solver, "xray_f0")
    form_factors = normalize_form_factors(element_order, q_solver, ff_mapping)
    atom_coefficients = form_factors[element_indices]
    phi = (np.arange(args.target_nphi) + 0.5) * (
        2.0 * np.pi / args.target_nphi
    )
    mask = detector_rectangle_mask(
        q_report,
        phi,
        wavelength_nm=args.wavelength_nm,
        active_width_mm=args.detector_active_width_mm,
        active_height_mm=args.detector_active_height_mm,
        distance_mm=args.detector_distance_mm,
    )
    target_q_indices = np.broadcast_to(
        np.arange(args.nq)[:, None], mask.shape
    )[mask]
    qx = np.ascontiguousarray((q_perp[:, None] * np.cos(phi)[None, :])[mask])
    qy = np.ascontiguousarray((q_perp[:, None] * np.sin(phi)[None, :])[mask])
    qz_targets = np.ascontiguousarray(
        np.broadcast_to(q_z_rows[:, None], mask.shape)[mask]
    )

    direct, direct_seconds = timed(
        lambda: direct_active_amplitude(
            coords,
            element_indices,
            np.ones(coords.shape[0]),
            form_factors,
            qx,
            qy,
            qz_targets,
            target_q_indices,
            target_chunk=args.direct_target_chunk,
        )
    )
    backend_rows = {}
    maximum_harmonic = 0
    backend_amplitudes = {}
    for backend in ("cpp_miller", "cpp_fused"):
        (amplitude, cutoffs), seconds = timed(
            lambda backend=backend: exact_coordinate_harmonic_amplitude(
                coords,
                q_perp,
                q_z_rows,
                phi,
                atom_coefficients=atom_coefficients,
                harmonic_margin=args.harmonic_margin,
                atom_chunk_size=args.atom_chunk_size,
                bessel_backend=backend,
            )
        )
        active = amplitude[mask]
        backend_amplitudes[backend] = active
        metrics = error_metrics(active, direct)
        maximum_harmonic = max(maximum_harmonic, int(np.max(cutoffs)))
        backend_rows[backend] = {
            "seconds": seconds,
            "direct_over_backend_speedup": direct_seconds / seconds,
            "complex_l2_vs_direct": metrics["complex_l2"],
            "intensity_l2_vs_direct": metrics["intensity_l2"],
        }

    fused_over_old = (
        backend_rows["cpp_miller"]["seconds"]
        / backend_rows["cpp_fused"]["seconds"]
    )
    gates = {
        "cpp_fused_complex_l2_vs_direct_le_1e_9": (
            backend_rows["cpp_fused"]["complex_l2_vs_direct"] <= 1e-9
        ),
        "cpp_fused_vs_cpp_miller_complex_l2_le_1e_12": (
            relative_l2(
                backend_amplitudes["cpp_fused"],
                backend_amplitudes["cpp_miller"],
            )
            <= 1e-12
        ),
        "maximum_harmonic_below_target_nyquist": (
            maximum_harmonic < args.target_nphi // 2
        ),
        "cpp_fused_speedup_vs_cpp_miller_ge_10": fused_over_old >= 10.0,
        "cpp_fused_speedup_vs_direct_ge_10": (
            backend_rows["cpp_fused"]["direct_over_backend_speedup"] >= 10.0
        ),
    }
    result = {
        "schema": "tip3p-exact-beta-backend-comparison-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "input_npz": args.input_npz.as_posix(),
            "trajectory_dcd": args.trajectory_dcd.as_posix(),
            "frame_index": 0,
            "source_metadata": source_metadata,
        },
        "atom_count": int(coords.shape[0]),
        "q_inv_angstrom": q_report.tolist(),
        "nq": args.nq,
        "target_nphi": args.target_nphi,
        "target_nyquist": args.target_nphi // 2,
        "active_targets": int(mask.sum()),
        "maximum_harmonic": maximum_harmonic,
        "direct_ndft_seconds": direct_seconds,
        "backends": backend_rows,
        "fused_over_cpp_miller_speedup": fused_over_old,
        "gates": gates,
        "passed": all(gates.values()),
        "claim_boundary": [
            "Direct NDFT is the correctness oracle; neither optimized backend is treated as truth.",
            "Timings are one local CPU run for frame 0 and are not hardware-independent.",
            "The comparison measures eight high-q rows, not the Nq=512 production workload.",
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
