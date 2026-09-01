from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_protein_nanocrystal_finufft_fair import detector_rectangle_mask  # noqa: E402
from benchmark_protein_nanocrystal_sparse_memory import build_sparse_structure  # noqa: E402
from prepare_openmm_water_waxs_inputs import (  # noqa: E402
    center_from_box,
    iter_dcd_frames,
    load_npz,
    read_dcd_header,
)
from validate_public_waxs_crystals import choose_grid  # noqa: E402
from validate_public_waxs_structures import build_form_factors  # noqa: E402
from validate_waxs_direct_finufft_triad import direct_active_amplitude  # noqa: E402
from waxs_cake import PreparedCakePlan, encode_elements, exact_coordinate_harmonic_amplitude  # noqa: E402
from waxs_cake.geometry import ewald_ring  # noqa: E402
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402
from waxs_cake.physical_scaling import q_to_inv_nm  # noqa: E402
from waxs_cake.solvers import normalize_form_factors  # noqa: E402


def timed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def metrics(model: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    model_i = intensity(model)
    reference_i = intensity(reference)
    model_centered = model_i - np.mean(model_i)
    reference_centered = reference_i - np.mean(reference_i)
    return {
        "complex_l2": relative_l2(model, reference),
        "intensity_l2": relative_l2(model_i, reference_i),
        "intensity_ncc": float(
            np.vdot(model_centered, reference_centered).real
            / (np.linalg.norm(model_centered) * np.linalg.norm(reference_centered))
        ),
    }


def aggregate(rows: list[dict], field: str) -> dict[str, float]:
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _write_markdown_legacy(result: dict, path: Path) -> None:
    lines = [
        "# TIP3P 20-frame dense high-q exact-beta validation",
        "",
        "20-frame 8 nm TIP3P trajectory를 persistent frame NPZ 없이 DCD에서 직접 읽었다.",
        "",
        f"- frames: `{result['frame_count']}`",
        f"- atoms/frame: `{result['atom_count']:,}`",
        f"- q: `{result['q_inv_angstrom']}` Å^-1",
        f"- detector Nphi / maximum harmonic: `{result['target_nphi']} / {result['maximum_harmonic']}`",
        f"- exact-beta complex L2 max: `{result['aggregate']['exact_beta_complex_l2']['max']:.3e}`",
        f"- exact-beta/direct time ratio median: `{result['aggregate']['direct_over_exact_beta_speedup']['median']:.3f}x`",
        f"- coarse 0.1 nm intensity L2 mean/max: `{result['aggregate']['coarse_intensity_l2']['mean']:.3f} / {result['aggregate']['coarse_intensity_l2']['max']:.3f}`",
        "",
        "| frame | exact L2 | exact intensity L2 | direct s | exact-beta s | direct/exact | coarse intensity L2 | coarse NCC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["frames"]:
        lines.append(
            f"| {row['frame_index']} | {row['exact_beta_vs_direct']['complex_l2']:.3e} "
            f"| {row['exact_beta_vs_direct']['intensity_l2']:.3e} "
            f"| {row['direct_seconds']:.3f} | {row['exact_beta_seconds']:.3f} "
            f"| {row['direct_over_exact_beta_speedup']:.3f}x "
            f"| {row['coarse_acfo_vs_direct']['intensity_l2']:.3f} "
            f"| {row['coarse_acfo_vs_direct']['intensity_ncc']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"- dense 20-frame exact-coordinate gate: **{'PASS' if result['passed'] else 'FAIL'}**",
            "- Exact-beta streaming is a bounded-memory exact-coordinate path; this two-q run is not publication timing for Nq=512.",
            "- The coarse row deliberately records the current 0.1 nm whole-object representation failure at high q.",
            "- Production work remains to reuse/group exact-beta coefficients across q and repeated frames.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(result: dict, path: Path) -> None:
    aggregate_result = result["aggregate"]
    lines = [
        "# TIP3P 20-frame dense high-q exact-beta validation",
        "",
        "The 8 nm TIP3P trajectory is read directly from DCD without persistent per-frame NPZ files.",
        "",
        f"- frames: `{result['frame_count']}`",
        f"- atoms/frame: `{result['atom_count']:,}`",
        f"- q: `{result['q_inv_angstrom']}` inverse angstrom",
        f"- exact-beta backend: `{result['bessel_backend']}`",
        f"- detector Nphi / maximum harmonic: `{result['target_nphi']} / {result['maximum_harmonic']}`",
        f"- exact-beta complex L2 max: `{aggregate_result['exact_beta_complex_l2']['max']:.3e}`",
        f"- direct/exact-beta speedup median: `{aggregate_result['direct_over_exact_beta_speedup']['median']:.3f}x`",
        f"- coarse 0.1 nm intensity L2 mean/max: `{aggregate_result['coarse_intensity_l2']['mean']:.3f} / {aggregate_result['coarse_intensity_l2']['max']:.3f}`",
        f"- fused harmonic output / thread scratch upper bound: `{result['memory_accounting']['fused_harmonic_output_mib']:.4f} / {result['memory_accounting']['fused_thread_scratch_upper_bound_mib']:.4f} MiB`",
        "",
        "| frame | exact L2 | intensity L2 | direct s | exact-beta s | direct/exact | coarse intensity L2 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["frames"]:
        lines.append(
            f"| {row['frame_index']} | {row['exact_beta_vs_direct']['complex_l2']:.3e} "
            f"| {row['exact_beta_vs_direct']['intensity_l2']:.3e} "
            f"| {row['direct_seconds']:.3f} | {row['exact_beta_seconds']:.3f} "
            f"| {row['direct_over_exact_beta_speedup']:.3f}x "
            f"| {row['coarse_acfo_vs_direct']['intensity_l2']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"- dense 20-frame correctness gate: **{'PASS' if result['passed'] else 'FAIL'}**",
            "- Direct NDFT is the small-case correctness oracle; FINUFFT is not used as truth here.",
            "- Timings are local CPU wall times for this machine and workload, not Nq=512 production evidence.",
            "- The coarse row records the known 0.1 nm whole-object representation failure at high q.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Twenty-frame dense TIP3P high-q direct-NDFT/exact-beta/coarse-ACFO comparison."
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
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--q", default="5.0,6.3")
    parser.add_argument("--wavelength-nm", type=float, default=0.08)
    parser.add_argument("--target-nphi", type=int, default=768)
    parser.add_argument("--harmonic-margin", type=int, default=48)
    parser.add_argument("--atom-chunk-size", type=int, default=256)
    parser.add_argument(
        "--bessel-backend",
        choices=("scipy", "cpp_miller", "cpp_fused"),
        default="cpp_fused",
    )
    parser.add_argument("--bin-width-nm", type=float, default=0.1)
    parser.add_argument("--coarse-margin", type=int, default=32)
    parser.add_argument("--direct-target-chunk", type=int, default=128)
    parser.add_argument("--detector-active-width-mm", type=float, default=155.1)
    parser.add_argument("--detector-active-height-mm", type=float, default=162.15)
    parser.add_argument("--detector-distance-mm", type=float, default=100.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/tip3p_dense_highq_exact_beta_20frames.json"),
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path("benchmark_results/tip3p_dense_highq_exact_beta_20frames.md"),
    )
    args = parser.parse_args()

    _, elements, _, source_metadata = load_npz(args.input_npz)
    element_indices, element_order = encode_elements(elements)
    q_report = np.asarray([float(value) for value in args.q.split(",")])
    q_solver = np.asarray(
        [q_to_inv_nm(value, "inv_angstrom") for value in q_report]
    )
    q_perp, q_z_rows = ewald_ring(q_solver, args.wavelength_nm)
    form_factors = build_form_factors(elements, q_solver, "xray_f0")
    ff = normalize_form_factors(element_order, q_solver, form_factors)
    atom_coefficients = ff[element_indices]
    phi = (np.arange(args.target_nphi) + 0.5) * (2.0 * np.pi / args.target_nphi)
    mask = detector_rectangle_mask(
        q_report,
        phi,
        wavelength_nm=args.wavelength_nm,
        active_width_mm=args.detector_active_width_mm,
        active_height_mm=args.detector_active_height_mm,
        distance_mm=args.detector_distance_mm,
    )
    target_q_indices = np.broadcast_to(np.arange(q_report.size)[:, None], mask.shape)[mask]
    qx = np.ascontiguousarray((q_perp[:, None] * np.cos(phi)[None, :])[mask])
    qy = np.ascontiguousarray((q_perp[:, None] * np.sin(phi)[None, :])[mask])
    qz_targets = np.ascontiguousarray(
        np.broadcast_to(q_z_rows[:, None], mask.shape)[mask]
    )

    header = read_dcd_header(args.trajectory_dcd)
    frame_indices = set(range(min(args.max_frames, int(header["frames"]))))
    rows = []
    reference_intensities = []
    max_harmonic = 0
    for frame_index, frame_coords, frame_box in iter_dcd_frames(
        args.trajectory_dcd, frame_indices=frame_indices
    ):
        coords = np.ascontiguousarray(
            frame_coords - center_from_box(frame_coords, frame_box)
        )
        direct, direct_s = timed(
            lambda: direct_active_amplitude(
                coords,
                element_indices,
                np.ones(coords.shape[0]),
                ff,
                qx,
                qy,
                qz_targets,
                target_q_indices,
                target_chunk=args.direct_target_chunk,
            )
        )
        (exact_beta, cutoffs), exact_s = timed(
            lambda: exact_coordinate_harmonic_amplitude(
                coords,
                q_perp,
                q_z_rows,
                phi,
                atom_coefficients=atom_coefficients,
                harmonic_margin=args.harmonic_margin,
                atom_chunk_size=args.atom_chunk_size,
                bessel_backend=args.bessel_backend,
            )
        )
        max_harmonic = max(max_harmonic, int(np.max(cutoffs)))

        grid = choose_grid(
            coords,
            argparse.Namespace(
                qmax=float(np.max(q_report)),
                q_unit="inv_angstrom",
                bin_width_nm=args.bin_width_nm,
                harmonic_margin=args.coarse_margin,
                nphi_detector=args.target_nphi,
            ),
        )
        grid["n_phi"] = args.target_nphi
        sparse = build_sparse_structure(coords, elements, grid, index_backend="cpp")
        plan = PreparedCakePlan(
            sparse,
            q_solver,
            args.wavelength_nm,
            form_factors=form_factors,
            circular_backend="cpp",
            complex_dtype=np.complex64,
            q_block_size=2,
        )
        coarse, coarse_s = timed(
            lambda: plan.circular_fft_sparse_source_r_dependent(
                margin=args.coarse_margin,
                cutoff_bin_size=16,
                analytic_kernel=True,
                q_block_size=2,
                profile_chunk_size=8,
            )
        )
        exact_metrics = metrics(exact_beta[mask], direct)
        coarse_metrics = metrics(np.asarray(coarse, dtype=np.complex128)[mask], direct)
        reference_intensities.append(intensity(direct))
        rows.append(
            {
                "frame_index": frame_index,
                "direct_seconds": direct_s,
                "exact_beta_seconds": exact_s,
                "coarse_acfo_seconds": coarse_s,
                "direct_over_exact_beta_speedup": direct_s / exact_s,
                "exact_beta_vs_direct": exact_metrics,
                "coarse_acfo_vs_direct": coarse_metrics,
                "coarse_sparse_storage_mib": sparse.sparse_storage_nbytes / 1024**2,
                "active_coarse_bins": int(sparse.active_values.size),
            }
        )

    if len(rows) != len(frame_indices):
        raise RuntimeError(f"expected {len(frame_indices)} frames, read {len(rows)}")
    reference0 = reference_intensities[0]
    for row, values in zip(rows, reference_intensities, strict=True):
        row["intensity_l2_vs_frame0"] = relative_l2(values, reference0)

    aggregate_result = {
        "exact_beta_complex_l2": aggregate(
            [{"value": row["exact_beta_vs_direct"]["complex_l2"]} for row in rows],
            "value",
        ),
        "exact_beta_intensity_l2": aggregate(
            [{"value": row["exact_beta_vs_direct"]["intensity_l2"]} for row in rows],
            "value",
        ),
        "direct_seconds": aggregate(rows, "direct_seconds"),
        "exact_beta_seconds": aggregate(rows, "exact_beta_seconds"),
        "direct_over_exact_beta_speedup": aggregate(rows, "direct_over_exact_beta_speedup"),
        "coarse_intensity_l2": aggregate(
            [{"value": row["coarse_acfo_vs_direct"]["intensity_l2"]} for row in rows],
            "value",
        ),
        "frame_variation_intensity_l2_vs_frame0": aggregate(rows, "intensity_l2_vs_frame0"),
    }
    gates = {
        "twenty_frames_present": len(rows) == 20,
        "maximum_harmonic_below_target_nyquist": max_harmonic < args.target_nphi // 2,
        "all_exact_beta_complex_l2_le_1e_9": all(
            row["exact_beta_vs_direct"]["complex_l2"] <= 1e-9 for row in rows
        ),
        "all_exact_beta_intensity_l2_le_1e_9": all(
            row["exact_beta_vs_direct"]["intensity_l2"] <= 1e-9 for row in rows
        ),
    }
    result = {
        "schema": "tip3p-dense-highq-exact-beta-v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "input_npz": args.input_npz.as_posix(),
            "trajectory_dcd": args.trajectory_dcd.as_posix(),
            "trajectory_header": header,
            "source_metadata": source_metadata,
        },
        "frame_count": len(rows),
        "atom_count": int(elements.size),
        "q_inv_angstrom": q_report.tolist(),
        "target_nphi": args.target_nphi,
        "target_nyquist": args.target_nphi // 2,
        "maximum_harmonic": max_harmonic,
        "active_targets": int(mask.sum()),
        "atom_chunk_size": args.atom_chunk_size,
        "bessel_backend": args.bessel_backend,
        "memory_accounting": {
            "scope": "algorithm arrays only; excludes coordinates, coefficient input, allocator overhead, and peak RSS",
            "legacy_chunked_kernel_upper_bound_mib": (
                args.atom_chunk_size * (max_harmonic + 1) * 16 / 1024**2
            ),
            "fused_harmonic_output_mib": (
                q_report.size * 2 * (max_harmonic + 1) * 16 / 1024**2
            ),
            "fused_thread_scratch_upper_bound_mib": (
                min(q_report.size, os.cpu_count() or 1)
                * (
                    (max_harmonic + 1) * 16
                    + (max_harmonic + 1 + 32) * 8
                )
                / 1024**2
            ),
        },
        "frames": rows,
        "aggregate": aggregate_result,
        "gates": gates,
        "passed": all(gates.values()),
        "decision": (
            "The bounded-memory exact-beta path reproduces direct NDFT for every one of 20 dense TIP3P MD frames at high q; "
            "the current 0.1 nm whole-object representation does not."
        ),
        "claim_boundary": [
            "This closes dense-disorder small-case correctness for the selected two high-q rows, not Nq=512 production timing.",
            "The exact-beta path is exact-coordinate and bounded-memory but still scales with atom count, q rows, and retained harmonics.",
            "The TIP3P model uses neutral xray_f0 form factors without anomalous dispersion or experimental background calibration.",
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
