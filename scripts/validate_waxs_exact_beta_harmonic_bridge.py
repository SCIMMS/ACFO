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
from validate_public_waxs_crystals import choose_grid  # noqa: E402
from validate_public_waxs_structures import build_form_factors, load_structure  # noqa: E402
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


def ring_mean(values: np.ndarray, q_indices: np.ndarray, nq: int) -> np.ndarray:
    values_i = intensity(values)
    return np.asarray([np.mean(values_i[q_indices == iq]) for iq in range(nq)])


def metrics(model, reference, q_indices, nq):
    model_i = intensity(model)
    reference_i = intensity(reference)
    centered_model = model_i - np.mean(model_i)
    centered_reference = reference_i - np.mean(reference_i)
    return {
        "complex_l2": relative_l2(model, reference),
        "intensity_l2": relative_l2(model_i, reference_i),
        "ring_intensity_l2": relative_l2(
            ring_mean(model, q_indices, nq),
            ring_mean(reference, q_indices, nq),
        ),
        "intensity_ncc": float(
            np.vdot(centered_model, centered_reference).real
            / (np.linalg.norm(centered_model) * np.linalg.norm(centered_reference))
        ),
    }


def quantize_rz_keep_exact_beta(
    coords: np.ndarray,
    *,
    n_r: int,
    n_z: int,
    r_max: float,
    z_range: tuple[float, float],
) -> np.ndarray:
    """Move R and z to bin centers while preserving each atom's exact beta."""

    radius = np.hypot(coords[:, 0], coords[:, 1])
    beta = np.arctan2(coords[:, 1], coords[:, 0])
    z_min, z_max = z_range
    r_idx = (radius * (n_r / r_max)).astype(np.int64)
    z_idx = ((coords[:, 2] - z_min) * (n_z / (z_max - z_min))).astype(np.int64)
    np.clip(r_idx, 0, n_r - 1, out=r_idx)
    np.clip(z_idx, 0, n_z - 1, out=z_idx)
    r_center = (r_idx + 0.5) * (r_max / n_r)
    z_center = z_min + (z_idx + 0.5) * ((z_max - z_min) / n_z)
    return np.column_stack(
        (r_center * np.cos(beta), r_center * np.sin(beta), z_center)
    )


def write_markdown(result: dict, path: Path) -> None:
    exact = result["exact_coordinate_harmonic"]
    lines = [
        "# WAXS exact-beta harmonic bridge",
        "",
        "Detector azimuth 수는 720으로 고정하고 per-atom beta를 직접 harmonic phase에 넣었다.",
        "",
        f"- exact-coordinate harmonic vs direct NDFT complex L2: `{exact['complex_l2']:.3e}`",
        f"- maximum retained harmonic: `{result['maximum_harmonic']}` / target Nyquist `{result['target_azimuth_nyquist']}`",
        f"- exact-coordinate harmonic time: `{exact['seconds']:.3f} s`",
        f"- direct NDFT time: `{result['direct_ndft_seconds']:.3f} s`",
        "",
        "| R/z bin (nm) | complex L2 | pixel intensity L2 | ring L2 | NCC | time s |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rz_quantization_sweep"]:
        lines.append(
            f"| {row['bin_width_nm']:.7g} | {row['complex_l2']:.3e} "
            f"| {row['intensity_l2']:.3e} | {row['ring_intensity_l2']:.3e} "
            f"| {row['intensity_ncc']:.6f} | {row['seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"- exact-coordinate bridge: **{'PASS' if result['gates']['exact_harmonic_complex_l2_le_1e_10'] else 'FAIL'}**",
            f"- fine R/z with exact beta, pixel <=1%: **{('PASS' if result['gates'].get('fine_rz_exact_beta_intensity_l2_le_1pct') else 'FAIL') if result['rz_quantization_sweep'] else 'SKIPPED'}**",
            "- 이는 source-coordinate와 detector-Nphi 분리가 수학적으로 가능하다는 proof-of-concept다.",
            "- 현재 Python 경로는 O(Nsource x Nq x Nharmonic) reference이며 production 성능 주장이 아니다.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exact per-atom beta Jacobi-Anger bridge on fixed detector targets."
    )
    parser.add_argument(
        "structure",
        type=Path,
        nargs="?",
        default=Path("structures/processed/protein_nanocrystal_lysozyme_1iee_1x1x1_fixed.npz"),
    )
    parser.add_argument("--wavelength-nm", type=float, default=0.08)
    parser.add_argument("--target-nphi", type=int, default=720)
    parser.add_argument("--harmonic-margin", type=int, default=48)
    parser.add_argument("--atom-chunk-size", type=int, default=256)
    parser.add_argument(
        "--bessel-backend",
        choices=("auto", "scipy", "cpp_miller", "cpp_fused"),
        default="auto",
    )
    parser.add_argument("--skip-rz-sweep", action="store_true")
    parser.add_argument("--direct-target-chunk", type=int, default=256)
    parser.add_argument("--detector-active-width-mm", type=float, default=155.1)
    parser.add_argument("--detector-active-height-mm", type=float, default=162.15)
    parser.add_argument("--detector-distance-mm", type=float, default=100.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/waxs_exact_beta_harmonic_bridge.json"),
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path("benchmark_results/waxs_exact_beta_harmonic_bridge.md"),
    )
    args = parser.parse_args()

    coords, elements, metadata = load_structure(args.structure)
    element_indices, element_order = encode_elements(elements)
    q_report = np.asarray([5.0, 5.65, 6.3], dtype=np.float64)
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
    direct, direct_s = timed(
        lambda: direct_active_amplitude(
            coords,
            element_indices,
            np.ones(coords.shape[0], dtype=np.float64),
            ff,
            qx,
            qy,
            qz_targets,
            target_q_indices,
            target_chunk=args.direct_target_chunk,
        )
    )

    (harmonic_result, cutoffs), harmonic_s = timed(
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
    harmonic_active = harmonic_result[mask]
    exact_metrics = metrics(
        harmonic_active, direct, target_q_indices, q_report.size
    )
    exact_metrics["seconds"] = harmonic_s

    quantized_rows = []
    bin_widths = () if args.skip_rz_sweep else (0.1, 0.00625, 0.00078125)
    for bin_width_nm in bin_widths:
        grid = choose_grid(
            coords,
            argparse.Namespace(
                qmax=6.3,
                q_unit="inv_angstrom",
                bin_width_nm=bin_width_nm,
                harmonic_margin=args.harmonic_margin,
                nphi_detector=args.target_nphi,
            ),
        )
        quantized = quantize_rz_keep_exact_beta(
            coords,
            n_r=grid["n_r"],
            n_z=grid["n_z"],
            r_max=grid["r_max_nm"],
            z_range=grid["z_range_nm"],
        )
        (quantized_result, quantized_cutoffs), elapsed = timed(
            lambda quantized=quantized: exact_coordinate_harmonic_amplitude(
                quantized,
                q_perp,
                q_z_rows,
                phi,
                atom_coefficients=atom_coefficients,
                harmonic_margin=args.harmonic_margin,
                atom_chunk_size=args.atom_chunk_size,
                bessel_backend=args.bessel_backend,
            )
        )
        row = {
            "bin_width_nm": bin_width_nm,
            "n_r": grid["n_r"],
            "n_z": grid["n_z"],
            "maximum_harmonic": int(np.max(quantized_cutoffs)),
            "seconds": elapsed,
            **metrics(
                quantized_result[mask], direct, target_q_indices, q_report.size
            ),
        }
        quantized_rows.append(row)

    gates = {
        "exact_harmonic_complex_l2_le_1e_10": exact_metrics["complex_l2"] <= 1e-10,
        "maximum_harmonic_below_target_nyquist": int(np.max(cutoffs)) < args.target_nphi // 2,
    }
    if quantized_rows:
        fine = quantized_rows[-1]
        gates.update(
            {
                "fine_rz_exact_beta_intensity_l2_le_1pct": (
                    fine["intensity_l2"] <= 0.01
                ),
                "fine_rz_exact_beta_ring_l2_le_0p5pct": (
                    fine["ring_intensity_l2"] <= 0.005
                ),
            }
        )
    result = {
        "schema": "waxs-exact-beta-harmonic-bridge-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "structure": metadata.get("structure_id", args.structure.stem),
        "structure_path": args.structure.as_posix(),
        "atom_count": int(coords.shape[0]),
        "q_inv_angstrom": q_report.tolist(),
        "target_nphi": args.target_nphi,
        "target_azimuth_nyquist": args.target_nphi // 2,
        "active_targets": int(mask.sum()),
        "harmonic_margin": args.harmonic_margin,
        "bessel_backend": args.bessel_backend,
        "atom_chunk_size": args.atom_chunk_size,
        "streamed_kernel_upper_bound_mib": (
            args.atom_chunk_size * (int(np.max(cutoffs)) + 1) * 16 / 1024**2
        ),
        "maximum_harmonic": int(np.max(cutoffs)),
        "direct_ndft_seconds": direct_s,
        "exact_coordinate_harmonic": exact_metrics,
        "rz_quantization_sweep": quantized_rows,
        "gates": gates,
        "passed": all(gates.values()),
        "decision": (
            "Exact per-atom beta and coordinates recover the direct NDFT on a 720-azimuth detector grid; "
            "source-coordinate precision therefore need not be represented by a 96,000-point detector/output FFT."
        ),
        "claim_boundary": [
            "This is a small-case exact-coordinate harmonic reference, not a scalable production implementation.",
            "The fine R/z threshold is exploratory and uses fixed detector targets.",
            "Production work remains to accelerate exact-beta/sub-bin source contraction without scaling the output FFT to source-coordinate resolution.",
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
