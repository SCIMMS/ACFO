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
    build_finufft_plans,
    detector_rectangle_mask,
    masked_ring_mean,
    source_arrays,
)
from benchmark_protein_nanocrystal_sparse_memory import build_sparse_structure  # noqa: E402
from validate_public_waxs_crystals import choose_grid  # noqa: E402
from validate_public_waxs_structures import build_form_factors, load_structure  # noqa: E402
from waxs_cake import PreparedCakePlan, encode_elements  # noqa: E402
from waxs_cake.geometry import ewald_ring  # noqa: E402
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402
from waxs_cake.physical_scaling import q_to_inv_nm  # noqa: E402
from waxs_cake.solvers import normalize_form_factors  # noqa: E402


def timed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def direct_active_amplitude(
    sources: np.ndarray,
    source_element_indices: np.ndarray,
    source_weights: np.ndarray,
    ff: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    target_q_indices: np.ndarray,
    *,
    target_chunk: int,
) -> np.ndarray:
    """Direct nonuniform DFT on selected curved-Ewald targets."""
    out = np.empty(qx.size, dtype=np.complex128)
    for start in range(0, qx.size, target_chunk):
        stop = min(start + target_chunk, qx.size)
        phase = (
            sources[:, 0, None] * qx[None, start:stop]
            + sources[:, 1, None] * qy[None, start:stop]
            + sources[:, 2, None] * qz[None, start:stop]
        )
        coeff = source_weights[:, None] * ff[
            source_element_indices[:, None], target_q_indices[None, start:stop]
        ]
        out[start:stop] = np.sum(coeff * np.exp(1j * phase), axis=0)
    return out


def finufft_active_amplitude(
    sources: np.ndarray,
    source_element_indices: np.ndarray,
    source_weights: np.ndarray,
    ff: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    target_q_indices: np.ndarray,
    *,
    eps: float,
    threads: int,
) -> np.ndarray:
    plans, masks = build_finufft_plans(
        sources,
        source_element_indices,
        qx,
        qy,
        qz,
        n_elements=ff.shape[0],
        eps=eps,
        threads=threads,
    )
    out = np.zeros(qx.size, dtype=np.complex128)
    for element_index, (plan, mask) in enumerate(zip(plans, masks)):
        out += plan.execute(np.ascontiguousarray(source_weights[mask])) * ff[
            element_index, target_q_indices
        ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Small WAXS triad: ACFO and FINUFFT independently compared with a direct NDFT oracle."
    )
    parser.add_argument(
        "structure",
        type=Path,
        nargs="?",
        default=Path("structures/processed/protein_nanocrystal_lysozyme_1iee_1x1x1_fixed.npz"),
    )
    parser.add_argument("--qmin", type=float, default=0.05)
    parser.add_argument("--qmax", type=float, default=6.3)
    parser.add_argument("--nq", type=int, default=8)
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
        default=Path("benchmark_results/waxs_direct_finufft_triad.json"),
    )
    args = parser.parse_args()
    eps_values = [float(value) for value in args.finufft_eps.split(",")]
    if any(value <= 0 for value in eps_values):
        raise ValueError("FINUFFT eps values must be positive")

    coords, elements, metadata = load_structure(args.structure)
    element_indices, element_order = encode_elements(elements)
    grid = choose_grid(
        coords,
        argparse.Namespace(
            qmax=args.qmax,
            q_unit="inv_angstrom",
            bin_width_nm=args.bin_width_nm,
            harmonic_margin=args.harmonic_margin,
            nphi_detector=args.nphi_min,
        ),
    )
    q_report = np.linspace(args.qmin, args.qmax, args.nq)
    q_solver = np.asarray([q_to_inv_nm(value, "inv_angstrom") for value in q_report])
    form_factors = build_form_factors(elements, q_solver, "xray_f0")
    ff = normalize_form_factors(element_order, q_solver, form_factors)
    sparse = build_sparse_structure(coords, elements, grid, index_backend="cpp")
    sources, source_e, source_weights = source_arrays(
        "same_binned", coords, element_indices, sparse
    )

    plan = PreparedCakePlan(
        sparse,
        q_solver,
        args.wavelength_nm,
        form_factors=form_factors,
        circular_backend="cpp",
        complex_dtype=np.complex64,
        q_block_size=args.q_block_size,
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
    q_perp, q_z = ewald_ring(q_solver, args.wavelength_nm)
    qx = np.ascontiguousarray((q_perp[:, None] * np.cos(phi)[None, :])[mask])
    qy = np.ascontiguousarray((q_perp[:, None] * np.sin(phi)[None, :])[mask])
    qz_targets = np.ascontiguousarray(np.broadcast_to(q_z[:, None], mask.shape)[mask])

    direct, direct_s = timed(
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
    direct_intensity = intensity(direct)
    acfo_intensity = intensity(acfo_active)
    acfo_metrics = {
        "complex_l2_vs_direct": relative_l2(acfo_active, direct),
        "intensity_l2_vs_direct": relative_l2(acfo_intensity, direct_intensity),
    }

    finufft_rows = []
    for eps in eps_values:
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
        finufft_i = intensity(finufft)
        finufft_rows.append(
            {
                "eps": eps,
                "seconds": elapsed,
                "complex_l2_vs_direct": relative_l2(finufft, direct),
                "intensity_l2_vs_direct": relative_l2(finufft_i, direct_intensity),
                "complex_l2_vs_acfo": relative_l2(finufft, acfo_active),
            }
        )

    acfo_full_i = intensity(acfo)
    direct_grid = np.zeros_like(acfo, dtype=np.complex128)
    direct_grid[mask] = direct
    direct_grid_i = intensity(direct_grid)
    result = {
        "schema": "waxs-direct-finufft-triad-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "structure": metadata.get("structure_id", args.structure.stem),
        "structure_path": args.structure.as_posix(),
        "atom_count": int(coords.shape[0]),
        "same_binned_source_count": int(sources.shape[0]),
        "q_range_inv_angstrom": [args.qmin, args.qmax],
        "nq": args.nq,
        "nphi": int(phi.size),
        "active_targets": int(mask.sum()),
        "direct_oracle": "complex128 explicit nonuniform phase sum on active curved-Ewald targets",
        "comparison_contract": "ACFO and FINUFFT use the same binned sources, form factors, and active detector targets",
        "timings_s": {"acfo": acfo_s, "direct": direct_s},
        "acfo": {
            **acfo_metrics,
            "masked_ring_intensity_l2_vs_direct": relative_l2(
                masked_ring_mean(acfo_full_i, mask),
                masked_ring_mean(direct_grid_i, mask),
            ),
        },
        "finufft_sweep": finufft_rows,
        "gates": {
            "acfo_complex_l2_le_2e_6": acfo_metrics["complex_l2_vs_direct"] <= 2e-6,
            "finufft_eps_1e_6_complex_l2_le_2e_6": next(
                row["complex_l2_vs_direct"] for row in finufft_rows if row["eps"] == 1e-6
            )
            <= 2e-6,
            "finufft_error_decreases_from_1e_4_to_1e_10": (
                finufft_rows[-1]["complex_l2_vs_direct"]
                < finufft_rows[0]["complex_l2_vs_direct"]
            ),
        },
    }
    result["passed"] = all(result["gates"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
