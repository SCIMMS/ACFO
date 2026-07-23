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

from benchmark_protein_nanocrystal_finufft_fair import (  # noqa: E402
    detector_rectangle_mask,
    source_arrays,
)
from benchmark_protein_nanocrystal_sparse_memory import build_sparse_structure  # noqa: E402
from validate_public_waxs_crystals import (  # noqa: E402
    choose_grid,
    next_fft_friendly_even,
)
from validate_public_waxs_structures import build_form_factors, load_structure  # noqa: E402
from validate_waxs_direct_finufft_triad import (  # noqa: E402
    direct_active_amplitude,
    finufft_active_amplitude,
)
from waxs_cake import PreparedCakePlan, encode_elements  # noqa: E402
from waxs_cake.geometry import ewald_ring  # noqa: E402
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402
from waxs_cake.physical_scaling import q_to_inv_nm  # noqa: E402
from waxs_cake.solvers import normalize_form_factors  # noqa: E402


def timed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def metric_pair(model: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    return {
        "complex_l2": relative_l2(model, reference),
        "intensity_l2": relative_l2(intensity(model), intensity(reference)),
    }


def row_errors(
    model: np.ndarray,
    reference: np.ndarray,
    target_q_indices: np.ndarray,
    nq: int,
) -> list[dict[str, float | int]]:
    rows = []
    for iq in range(nq):
        selected = target_q_indices == iq
        if not np.any(selected):
            continue
        rows.append({"q_index": iq, **metric_pair(model[selected], reference[selected])})
    return rows


def ring_intensity_l2(
    model: np.ndarray,
    reference: np.ndarray,
    target_q_indices: np.ndarray,
    nq: int,
) -> float:
    model_i = intensity(model)
    reference_i = intensity(reference)
    model_mean = np.asarray(
        [np.mean(model_i[target_q_indices == iq]) for iq in range(nq)]
    )
    reference_mean = np.asarray(
        [np.mean(reference_i[target_q_indices == iq]) for iq in range(nq)]
    )
    return relative_l2(model_mean, reference_mean)


def geometry_for_scale(
    q_solver: np.ndarray,
    wavelength_nm: float,
    curvature_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    _, physical_qz = ewald_ring(q_solver, wavelength_nm)
    q_z = curvature_scale * physical_qz
    q_perp = np.sqrt(np.maximum(q_solver**2 - q_z**2, 0.0))
    return q_perp, q_z


def run_case(
    *,
    case: dict,
    coords: np.ndarray,
    elements: np.ndarray,
    atom_element_indices: np.ndarray,
    element_order: list[str],
    args: argparse.Namespace,
) -> dict:
    bin_width_nm = float(case.get("bin_width_nm", args.bin_width_nm))
    q_report = np.linspace(case["qmin"], case["qmax"], args.nq)
    q_solver = np.asarray(
        [q_to_inv_nm(value, "inv_angstrom") for value in q_report], dtype=np.float64
    )
    grid = choose_grid(
        coords,
        argparse.Namespace(
            qmax=case["qmax"],
            q_unit="inv_angstrom",
            bin_width_nm=bin_width_nm,
            harmonic_margin=args.harmonic_margin,
            nphi_detector=args.nphi_min,
        ),
    )
    base_nphi = int(grid["n_phi"])
    requested_nphi = max(32, int(round(base_nphi * case["angular_scale"])))
    grid["n_phi"] = next_fft_friendly_even(requested_nphi)

    form_factors = build_form_factors(elements, q_solver, "xray_f0")
    ff = normalize_form_factors(element_order, q_solver, form_factors)
    sparse = build_sparse_structure(coords, elements, grid, index_backend="cpp")
    sources, source_e, source_weights = source_arrays(
        "same_binned", coords, atom_element_indices, sparse
    )
    q_perp, q_z = geometry_for_scale(
        q_solver, args.wavelength_nm, case["curvature_scale"]
    )

    plan = PreparedCakePlan(
        sparse,
        q_solver,
        args.wavelength_nm,
        form_factors=form_factors,
        circular_backend="cpp",
        complex_dtype=np.complex64,
        q_block_size=args.q_block_size,
        q_perp=q_perp,
        q_z=q_z,
    )
    acfo, acfo_s = timed(
        lambda: plan.circular_fft_sparse_source_r_dependent(
            margin=args.r_dependent_margin,
            cutoff_bin_size=args.cutoff_bin_size,
            analytic_kernel=True,
            q_block_size=args.q_block_size,
            profile_chunk_size=args.profile_chunk_size,
        )
    )
    accuracy_plan = PreparedCakePlan(
        sparse,
        q_solver,
        args.wavelength_nm,
        form_factors=form_factors,
        circular_backend="cpp",
        complex_dtype=np.complex128,
        q_block_size=args.q_block_size,
        q_perp=q_perp,
        q_z=q_z,
    )
    acfo_full_harmonic, acfo_full_harmonic_s = timed(
        lambda: accuracy_plan.circular_fft_sparse_source_projection(
            q_block_size=args.q_block_size,
            profile_chunk_size=args.profile_chunk_size,
        )
    )

    phi = sparse.beta_centers
    mask = detector_rectangle_mask(
        q_report,
        phi,
        wavelength_nm=args.wavelength_nm,
        active_width_mm=args.detector_active_width_mm,
        active_height_mm=args.detector_active_height_mm,
        distance_mm=args.detector_distance_mm,
    )
    target_q_indices = np.broadcast_to(np.arange(args.nq)[:, None], mask.shape)[mask]
    qx = np.ascontiguousarray((q_perp[:, None] * np.cos(phi)[None, :])[mask])
    qy = np.ascontiguousarray((q_perp[:, None] * np.sin(phi)[None, :])[mask])
    qz_targets = np.ascontiguousarray(np.broadcast_to(q_z[:, None], mask.shape)[mask])

    direct_binned, direct_binned_s = timed(
        lambda: direct_active_amplitude(
            sources,
            source_e,
            source_weights,
            ff,
            qx,
            qy,
            qz_targets,
            target_q_indices,
            target_chunk=args.direct_target_chunk,
        )
    )
    acfo_active = np.asarray(acfo, dtype=np.complex128)[mask]
    acfo_full_harmonic_active = np.asarray(acfo_full_harmonic, dtype=np.complex128)[mask]
    acfo_metrics = metric_pair(acfo_active, direct_binned)
    acfo_full_harmonic_metrics = metric_pair(
        acfo_full_harmonic_active, direct_binned
    )
    acfo_rows = row_errors(acfo_active, direct_binned, target_q_indices, args.nq)

    finufft_rows = []
    for eps in args.eps_values:
        finufft, elapsed = timed(
            lambda eps=eps: finufft_active_amplitude(
                sources,
                source_e,
                source_weights,
                ff,
                qx,
                qy,
                qz_targets,
                target_q_indices,
                eps=eps,
                threads=args.finufft_threads,
            )
        )
        metrics = metric_pair(finufft, direct_binned)
        finufft_rows.append(
            {
                "eps": eps,
                "seconds": elapsed,
                **metrics,
                "complex_l2_vs_acfo": relative_l2(finufft, acfo_active),
                "per_q_row": row_errors(
                    finufft, direct_binned, target_q_indices, args.nq
                ),
            }
        )

    atom_reference = None
    if case["exact_atom_reference"]:
        direct_atom, direct_atom_s = timed(
            lambda: direct_active_amplitude(
                coords,
                atom_element_indices,
                np.ones(coords.shape[0], dtype=np.float64),
                ff,
                qx,
                qy,
                qz_targets,
                target_q_indices,
                target_chunk=args.direct_target_chunk,
            )
        )
        atom_reference = {
            "seconds": direct_atom_s,
            "binned_direct_vs_exact_atom": {
                **metric_pair(direct_binned, direct_atom),
                "ring_intensity_l2": ring_intensity_l2(
                    direct_binned, direct_atom, target_q_indices, args.nq
                ),
            },
            "acfo_vs_exact_atom": {
                **metric_pair(acfo_active, direct_atom),
                "ring_intensity_l2": ring_intensity_l2(
                    acfo_active, direct_atom, target_q_indices, args.nq
                ),
            },
        }

    eps_1e6 = next(row for row in finufft_rows if row["eps"] == 1e-6)
    eps_tight = finufft_rows[-1]
    return {
        "name": case["name"],
        "purpose": case["purpose"],
        "q_range_inv_angstrom": [case["qmin"], case["qmax"]],
        "nq": args.nq,
        "curvature_scale": case["curvature_scale"],
        "curvature_definition": "qz=scale*qz_Ewald; qperp=sqrt(q^2-qz^2)",
        "bin_width_nm": bin_width_nm,
        "angular_scale_vs_bandlimit_grid": case["angular_scale"],
        "base_nphi": base_nphi,
        "nphi": int(phi.size),
        "active_targets": int(mask.sum()),
        "same_binned_source_count": int(sources.shape[0]),
        "timings_s": {
            "acfo_production": acfo_s,
            "acfo_full_harmonic_complex128": acfo_full_harmonic_s,
            "direct_binned": direct_binned_s,
        },
        "acfo_vs_direct_binned": {**acfo_metrics, "per_q_row": acfo_rows},
        "acfo_full_harmonic_vs_direct_binned": acfo_full_harmonic_metrics,
        "finufft_vs_direct_binned": finufft_rows,
        "exact_atom_reference": atom_reference,
        "gates": {
            "acfo_complex_l2_le_2e_6": acfo_metrics["complex_l2"] <= 2e-6,
            "acfo_full_harmonic_complex_l2_le_1e_10": (
                acfo_full_harmonic_metrics["complex_l2"] <= 1e-10
            ),
            "finufft_eps_1e_6_complex_l2_le_2e_6": eps_1e6["complex_l2"] <= 2e-6,
            "finufft_tight_complex_l2_le_1e_9": eps_tight["complex_l2"] <= 1e-9,
            "finufft_endpoint_error_decreases": (
                eps_tight["complex_l2"] < finufft_rows[0]["complex_l2"]
            ),
        },
    }


def write_markdown(result: dict, path: Path) -> None:
    lines = [
        "# WAXS direct-NDFT reference sweep",
        "",
        "ACFO와 FINUFFT를 동일한 binned source 및 curved-manifold target의 complex128 direct NDFT와 각각 비교했다.",
        "",
        "| case | q (Å⁻¹) | curvature | bin (nm) | Nphi | ACFO prod | ACFO full | FINUFFT 1e-6 | FINUFFT tight | binned-vs-atom intensity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in result["cases"]:
        finufft_1e6 = next(
            row for row in case["finufft_vs_direct_binned"] if row["eps"] == 1e-6
        )
        tight = case["finufft_vs_direct_binned"][-1]
        atom = case["exact_atom_reference"]
        atom_text = (
            f"{atom['binned_direct_vs_exact_atom']['intensity_l2']:.3e}"
            if atom is not None
            else "—"
        )
        lines.append(
            f"| {case['name']} | {case['q_range_inv_angstrom'][0]:g}–{case['q_range_inv_angstrom'][1]:g} "
            f"| {case['curvature_scale']:.2f} | {case['bin_width_nm']:.5g} | {case['nphi']} "
            f"| {case['acfo_vs_direct_binned']['complex_l2']:.3e} "
            f"| {case['acfo_full_harmonic_vs_direct_binned']['complex_l2']:.3e} "
            f"| {finufft_1e6['complex_l2']:.3e} | {tight['complex_l2']:.3e} | {atom_text} |"
        )
    lines.extend(
        [
            "",
            f"- operator/NUFFT gates: **{'PASS' if result['passed'] else 'FAIL'}**",
            "- Direct FFT는 curved nonuniform target의 동일 연산자가 아니므로 oracle로 사용하지 않았다.",
            "- exact-atom 열은 operator 오차가 아니라 cylindrical source discretization까지 포함한 end-to-end 오차다.",
            "- 시간은 이 소규모 correctness 실행의 부수 기록이며 production 성능 비교에 사용하지 않는다.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Small WAXS direct-NDFT sweep over q band, curvature, angular sampling, and FINUFFT tolerance."
    )
    parser.add_argument(
        "structure",
        type=Path,
        nargs="?",
        default=Path("structures/processed/protein_nanocrystal_lysozyme_1iee_1x1x1_fixed.npz"),
    )
    parser.add_argument("--nq", type=int, default=6)
    parser.add_argument("--wavelength-nm", type=float, default=0.08)
    parser.add_argument("--bin-width-nm", type=float, default=0.1)
    parser.add_argument("--nphi-min", type=int, default=128)
    parser.add_argument("--harmonic-margin", type=int, default=32)
    parser.add_argument("--r-dependent-margin", type=int, default=32)
    parser.add_argument("--cutoff-bin-size", type=int, default=16)
    parser.add_argument("--q-block-size", type=int, default=2)
    parser.add_argument("--profile-chunk-size", type=int, default=8)
    parser.add_argument("--finufft-eps", default="1e-4,1e-6,1e-8,1e-10")
    parser.add_argument("--finufft-threads", type=int, default=4)
    parser.add_argument("--direct-target-chunk", type=int, default=256)
    parser.add_argument("--detector-active-width-mm", type=float, default=155.1)
    parser.add_argument("--detector-active-height-mm", type=float, default=162.15)
    parser.add_argument("--detector-distance-mm", type=float, default=100.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/waxs_direct_reference_sweep.json"),
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path("benchmark_results/waxs_direct_reference_sweep.md"),
    )
    args = parser.parse_args()
    args.eps_values = [float(value) for value in args.finufft_eps.split(",")]
    if args.eps_values != sorted(args.eps_values, reverse=True):
        raise ValueError("FINUFFT eps values must be ordered from loose to tight")
    for required in (1e-6,):
        if required not in args.eps_values:
            raise ValueError(f"FINUFFT eps sweep must include {required:g}")

    coords, elements, metadata = load_structure(args.structure)
    atom_element_indices, element_order = encode_elements(elements)
    cases = [
        {"name": "low_q_physical", "purpose": "q-band", "qmin": 0.05, "qmax": 1.0, "curvature_scale": 1.0, "angular_scale": 1.0, "exact_atom_reference": True},
        {"name": "mid_q_physical", "purpose": "q-band", "qmin": 2.0, "qmax": 4.0, "curvature_scale": 1.0, "angular_scale": 1.0, "exact_atom_reference": True},
        {"name": "high_q_physical", "purpose": "q-band/anchor", "qmin": 5.0, "qmax": 6.3, "curvature_scale": 1.0, "angular_scale": 1.0, "exact_atom_reference": True},
        {"name": "high_q_planar", "purpose": "curvature", "qmin": 5.0, "qmax": 6.3, "curvature_scale": 0.0, "angular_scale": 1.0, "exact_atom_reference": False},
        {"name": "high_q_half_curvature", "purpose": "curvature", "qmin": 5.0, "qmax": 6.3, "curvature_scale": 0.5, "angular_scale": 1.0, "exact_atom_reference": False},
        {"name": "high_q_angular_half", "purpose": "angular-resolution", "qmin": 5.0, "qmax": 6.3, "curvature_scale": 1.0, "angular_scale": 0.5, "exact_atom_reference": True},
        {"name": "high_q_angular_double", "purpose": "angular-resolution", "qmin": 5.0, "qmax": 6.3, "curvature_scale": 1.0, "angular_scale": 2.0, "exact_atom_reference": True},
        {"name": "high_q_bin_0p05nm", "purpose": "bin-width", "qmin": 5.0, "qmax": 6.3, "curvature_scale": 1.0, "angular_scale": 1.0, "bin_width_nm": 0.05, "exact_atom_reference": True},
        {"name": "high_q_bin_0p025nm", "purpose": "bin-width", "qmin": 5.0, "qmax": 6.3, "curvature_scale": 1.0, "angular_scale": 1.0, "bin_width_nm": 0.025, "exact_atom_reference": True},
        {"name": "high_q_bin_0p0125nm", "purpose": "bin-width", "qmin": 5.0, "qmax": 6.3, "curvature_scale": 1.0, "angular_scale": 1.0, "bin_width_nm": 0.0125, "exact_atom_reference": True},
        {"name": "high_q_bin_0p00625nm", "purpose": "bin-width", "qmin": 5.0, "qmax": 6.3, "curvature_scale": 1.0, "angular_scale": 1.0, "bin_width_nm": 0.00625, "exact_atom_reference": True},
        {"name": "high_q_bin_0p003125nm", "purpose": "bin-width", "qmin": 5.0, "qmax": 6.3, "curvature_scale": 1.0, "angular_scale": 1.0, "bin_width_nm": 0.003125, "exact_atom_reference": True},
        {"name": "high_q_bin_0p0015625nm", "purpose": "bin-width", "qmin": 5.0, "qmax": 6.3, "curvature_scale": 1.0, "angular_scale": 1.0, "bin_width_nm": 0.0015625, "exact_atom_reference": True},
        {"name": "high_q_bin_0p00078125nm", "purpose": "bin-width", "qmin": 5.0, "qmax": 6.3, "curvature_scale": 1.0, "angular_scale": 1.0, "bin_width_nm": 0.00078125, "exact_atom_reference": True},
    ]
    rows = [
        run_case(
            case=case,
            coords=coords,
            elements=elements,
            atom_element_indices=atom_element_indices,
            element_order=element_order,
            args=args,
        )
        for case in cases
    ]
    all_case_gates = {
        f"{case['name']}:{gate}": passed
        for case in rows
        for gate, passed in case["gates"].items()
    }
    production_rows = [
        case for case in rows if case["name"] != "high_q_angular_half"
    ]
    operator_reference_pass = all(
        passed
        for case in production_rows
        for passed in case["gates"].values()
    )
    underresolved = next(case for case in rows if case["name"] == "high_q_angular_half")
    finest_bin = next(case for case in rows if case["name"] == "high_q_bin_0p00078125nm")
    fixed_nphi_750_finest_bin_intensity_l2 = finest_bin["exact_atom_reference"][
        "binned_direct_vs_exact_atom"
    ]["intensity_l2"]
    fixed_nphi_750_representation_pass = (
        fixed_nphi_750_finest_bin_intensity_l2 <= 0.01
    )
    result = {
        "schema": "waxs-direct-reference-sweep-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "structure": metadata.get("structure_id", args.structure.stem),
        "structure_path": args.structure.as_posix(),
        "atom_count": int(coords.shape[0]),
        "direct_oracle": "complex128 explicit phase sum at each nonuniform target",
        "comparison_contract": "ACFO and FINUFFT use identical binned sources, form factors, and targets within every case",
        "dimensions": {
            "q_bands_inv_angstrom": [[0.05, 1.0], [2.0, 4.0], [5.0, 6.3]],
            "high_q_curvature_scales": [0.0, 0.5, 1.0],
            "high_q_angular_scales": [0.5, 1.0, 2.0],
            "high_q_bin_widths_nm": [
                0.1, 0.05, 0.025, 0.0125, 0.00625,
                0.003125, 0.0015625, 0.00078125,
            ],
            "finufft_eps": args.eps_values,
        },
        "cases": rows,
        "case_gates": all_case_gates,
        "gates": {
            "operator_reference_pass": operator_reference_pass,
            "underresolved_angular_negative_control_detected": (
                underresolved["acfo_vs_direct_binned"]["complex_l2"] > 2e-6
            ),
            "fixed_nphi_750_finest_bin_intensity_l2_le_1pct": (
                fixed_nphi_750_representation_pass
            ),
        },
        "operator_reference_pass": operator_reference_pass,
        "fixed_nphi_750_finest_bin_intensity_l2": (
            fixed_nphi_750_finest_bin_intensity_l2
        ),
        "fixed_nphi_750_representation_pass": fixed_nphi_750_representation_pass,
        "publication_full_pass": (
            operator_reference_pass and fixed_nphi_750_representation_pass
        ),
        "passed": operator_reference_pass,
        "claim_boundary": [
            "Direct NDFT is the correctness oracle; FINUFFT is a tolerance-controlled practical baseline.",
            "Direct FFT is not used as an oracle because the curved targets are not a uniform Cartesian reciprocal grid.",
            "Same-binned-source comparisons isolate operator error; exact-atom comparisons also include source discretization.",
            "Small-case timings are not production performance evidence.",
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
