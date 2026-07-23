from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_physical_scaling import memory_fields, timed_call  # noqa: E402
from benchmark_protein_nanocrystal_finufft_fair import (  # noqa: E402
    build_finufft_plans,
    source_arrays,
)
from benchmark_protein_nanocrystal_sparse_memory import build_sparse_structure  # noqa: E402
from validate_public_waxs_crystals import choose_grid  # noqa: E402
from validate_public_waxs_structures import build_form_factors, load_structure  # noqa: E402
from waxs_cake import PreparedCakePlan, encode_elements  # noqa: E402
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402
from waxs_cake.physical_scaling import q_to_inv_nm  # noqa: E402
from waxs_cake.solvers import normalize_form_factors  # noqa: E402


def median_calls(func, repeats: int) -> tuple[np.ndarray, float, list[float]]:
    value = None
    timings: list[float] = []
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        value = func()
        timings.append(time.perf_counter() - started)
    if value is None:
        raise ValueError("repeats must be positive")
    return value, float(median(timings)), timings


def curvature_manifold(
    q: np.ndarray,
    *,
    wavelength_nm: float,
    curvature_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=np.float64)
    scale = float(curvature_scale)
    if scale < 0.0:
        raise ValueError("curvature_scale must be nonnegative")
    k = 2.0 * np.pi / float(wavelength_nm)
    q_z = -scale * q * q / (2.0 * k)
    inside = q * q - q_z * q_z
    if np.any(inside < -1e-10 * np.max(q * q)):
        raise ValueError("curvature scale pushes the manifold outside the elastic sphere")
    q_perp = np.sqrt(np.maximum(inside, 0.0))
    return q_perp, q_z


def curvature_metrics(
    qmax_inv_nm: float,
    *,
    wavelength_nm: float,
    curvature_scale: float,
) -> dict[str, float]:
    k = 2.0 * np.pi / float(wavelength_nm)
    axial_fraction = float(curvature_scale) * qmax_inv_nm / (2.0 * k)
    denominator = math.sqrt(max(1e-30, 1.0 - axial_fraction * axial_fraction))
    radial_derivative = (1.0 - 2.0 * axial_fraction * axial_fraction) / denominator
    return {
        "max_abs_qz_over_q": axial_fraction,
        "dqperp_dq_at_qmax": radial_derivative,
        "normalized_radial_compression": 1.0 - radial_derivative,
        "effective_wavelength_nm": float(curvature_scale) * float(wavelength_nm),
    }


def make_finufft_execute(
    *,
    sources: np.ndarray,
    source_e: np.ndarray,
    source_weights: np.ndarray,
    q_perp: np.ndarray,
    q_z: np.ndarray,
    phi: np.ndarray,
    ff: np.ndarray,
    n_elements: int,
    eps: float,
    threads: int,
    q_block_size: int,
):
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    nq = q_perp.size

    def execute() -> np.ndarray:
        out = np.zeros((nq, phi.size), dtype=np.complex128)
        for start in range(0, nq, q_block_size):
            stop = min(start + q_block_size, nq)
            qx_block = np.ascontiguousarray(
                (q_perp[start:stop, None] * cos_phi[None, :]).ravel()
            )
            qy_block = np.ascontiguousarray(
                (q_perp[start:stop, None] * sin_phi[None, :]).ravel()
            )
            qz_block = np.ascontiguousarray(
                np.broadcast_to(q_z[start:stop, None], (stop - start, phi.size)).ravel()
            )
            plans, masks = build_finufft_plans(
                sources,
                source_e,
                qx_block,
                qy_block,
                qz_block,
                n_elements=n_elements,
                eps=eps,
                threads=threads,
            )
            for element_index, (plan, mask) in enumerate(zip(plans, masks)):
                values = plan.execute(np.ascontiguousarray(source_weights[mask]))
                out[start:stop] += values.reshape(stop - start, phi.size) * ff[
                    element_index,
                    start:stop,
                    None,
                ]
            del plans, masks
        return out

    return execute


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolate curvature while holding protein object, q grid, target count, and accuracy fixed."
    )
    parser.add_argument(
        "structure",
        type=Path,
        nargs="?",
        default=Path("structures/processed/protein_nanocrystal_lysozyme_1iee_3x3x3_fixed.npz"),
    )
    parser.add_argument("--curvature-scales", default="0,0.25,0.5,0.75,1,1.25")
    parser.add_argument("--qmin", type=float, default=0.05)
    parser.add_argument("--qmax", type=float, default=6.3)
    parser.add_argument("--q-unit", choices=["inv_angstrom", "inv_nm"], default="inv_angstrom")
    parser.add_argument("--nq", type=int, default=256)
    parser.add_argument("--wavelength-nm", type=float, default=0.1)
    parser.add_argument("--bin-width-nm", type=float, default=0.1)
    parser.add_argument("--nphi-min", type=int, default=1024)
    parser.add_argument("--harmonic-margin", type=int, default=16)
    parser.add_argument("--r-dependent-margin", type=int, default=32)
    parser.add_argument("--cutoff-bin-size", type=int, default=16)
    parser.add_argument("--acfo-q-block-size", type=int, default=2)
    parser.add_argument("--profile-chunk-size", type=int, default=8)
    parser.add_argument("--form-factor-model", default="xray_f0")
    parser.add_argument("--finufft-eps", type=float, default=1e-6)
    parser.add_argument("--finufft-threads", type=int, default=4)
    parser.add_argument("--finufft-q-block-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--memory-sample-interval-s", type=float, default=0.002)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/waxs_curvature_isolated_216k_nq256.json"),
    )
    args = parser.parse_args()
    curvature_scales = [float(value) for value in args.curvature_scales.split(",")]
    if args.nq <= 1 or args.repeats <= 0:
        raise ValueError("nq must exceed one and repeats must be positive")
    if args.finufft_q_block_size <= 0 or args.finufft_threads <= 0:
        raise ValueError("FINUFFT block size and thread count must be positive")

    import finufft

    coords, elements, metadata = load_structure(args.structure)
    element_indices, element_order = encode_elements(elements)
    grid_args = argparse.Namespace(
        qmax=args.qmax,
        q_unit=args.q_unit,
        bin_width_nm=args.bin_width_nm,
        harmonic_margin=args.harmonic_margin,
        nphi_detector=args.nphi_min,
    )
    grid = choose_grid(coords, grid_args)
    q_report = np.linspace(args.qmin, args.qmax, args.nq)
    q_solver = np.asarray(
        [q_to_inv_nm(value, args.q_unit) for value in q_report], dtype=np.float64
    )
    form_factors = build_form_factors(elements, q_solver, args.form_factor_model)
    ff = normalize_form_factors(element_order, q_solver, form_factors)
    sparse, sparse_build_s, sparse_build_memory = timed_call(
        lambda: build_sparse_structure(coords, elements, grid, index_backend="cpp"),
        measure_memory=True,
        sample_interval_s=args.memory_sample_interval_s,
    )
    sources, source_e, source_weights = source_arrays(
        "same_binned", coords, element_indices, sparse
    )
    phi = sparse.beta_centers

    rows: list[dict[str, object]] = []
    for index, curvature_scale in enumerate(curvature_scales, start=1):
        print(
            f"[{index}/{len(curvature_scales)}] curvature_scale={curvature_scale:g} ACFO",
            flush=True,
        )
        q_perp, q_z = curvature_manifold(
            q_solver,
            wavelength_nm=args.wavelength_nm,
            curvature_scale=curvature_scale,
        )
        plan, plan_s, plan_memory = timed_call(
            lambda: PreparedCakePlan(
                sparse,
                q_solver,
                args.wavelength_nm,
                form_factors=form_factors,
                circular_backend="cpp",
                complex_dtype=np.complex64,
                q_block_size=args.acfo_q_block_size,
                q_perp=q_perp,
                q_z=q_z,
            ),
            measure_memory=True,
            sample_interval_s=args.memory_sample_interval_s,
        )
        acfo_execute = lambda: plan.circular_fft_sparse_source_r_dependent(
            margin=args.r_dependent_margin,
            cutoff_bin_size=args.cutoff_bin_size,
            analytic_kernel=True,
            q_block_size=args.acfo_q_block_size,
            profile_chunk_size=args.profile_chunk_size,
        )
        acfo_first, acfo_first_s, acfo_first_memory = timed_call(
            acfo_execute,
            measure_memory=True,
            sample_interval_s=args.memory_sample_interval_s,
        )
        acfo_cached, acfo_cached_s, acfo_cached_times = median_calls(
            acfo_execute, args.repeats
        )

        print(
            f"[{index}/{len(curvature_scales)}] curvature_scale={curvature_scale:g} FINUFFT",
            flush=True,
        )
        finufft_execute = make_finufft_execute(
            sources=sources,
            source_e=source_e,
            source_weights=source_weights,
            q_perp=q_perp,
            q_z=q_z,
            phi=phi,
            ff=ff,
            n_elements=len(element_order),
            eps=args.finufft_eps,
            threads=args.finufft_threads,
            q_block_size=args.finufft_q_block_size,
        )
        finufft_first, finufft_first_s, finufft_first_memory = timed_call(
            finufft_execute,
            measure_memory=True,
            sample_interval_s=args.memory_sample_interval_s,
        )
        finufft_cached, finufft_cached_s, finufft_cached_times = median_calls(
            finufft_execute, args.repeats
        )
        acfo_intensity = intensity(acfo_cached)
        finufft_intensity = intensity(finufft_cached)
        curvature = curvature_metrics(
            float(q_solver[-1]),
            wavelength_nm=args.wavelength_nm,
            curvature_scale=curvature_scale,
        )
        row = {
            "curvature_scale": curvature_scale,
            **curvature,
            "qperp_max_inv_nm": float(np.max(q_perp)),
            "abs_qz_max_inv_nm": float(np.max(np.abs(q_z))),
            "acfo_plan_s": plan_s,
            **memory_fields("acfo_plan", plan_memory),
            "acfo_first_s": acfo_first_s,
            **memory_fields("acfo_first", acfo_first_memory),
            "acfo_cached_median_s": acfo_cached_s,
            "acfo_cached_times": acfo_cached_times,
            "finufft_first_s": finufft_first_s,
            **memory_fields("finufft_first", finufft_first_memory),
            "finufft_cached_median_s": finufft_cached_s,
            "finufft_cached_times": finufft_cached_times,
            "complex_l2_acfo_vs_finufft": relative_l2(acfo_cached, finufft_cached),
            "intensity_l2_acfo_vs_finufft": relative_l2(
                acfo_intensity, finufft_intensity
            ),
            "ring_l2_acfo_vs_finufft": relative_l2(
                np.mean(acfo_intensity, axis=1),
                np.mean(finufft_intensity, axis=1),
            ),
            "warm_speedup_finufft_over_acfo": finufft_cached_s / acfo_cached_s,
            "first_speedup_finufft_over_acfo": finufft_first_s / acfo_first_s,
            "geometry_first_total_speedup": finufft_first_s / (plan_s + acfo_first_s),
            "all_finite": bool(
                np.all(np.isfinite(acfo_cached))
                and np.all(np.isfinite(finufft_cached))
            ),
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "curvature_scale": curvature_scale,
                    "complex_l2": row["complex_l2_acfo_vs_finufft"],
                    "acfo_cached_s": acfo_cached_s,
                    "finufft_cached_s": finufft_cached_s,
                    "warm_speedup": row["warm_speedup_finufft_over_acfo"],
                },
                indent=2,
            ),
            flush=True,
        )
        del plan, acfo_first, acfo_cached, finufft_first, finufft_cached
        gc.collect()

    ordered = sorted(rows, key=lambda row: float(row["curvature_scale"]))
    curvature_values = np.asarray(
        [row["max_abs_qz_over_q"] for row in ordered], dtype=np.float64
    )
    speedups = np.asarray(
        [row["warm_speedup_finufft_over_acfo"] for row in ordered], dtype=np.float64
    )
    pearson = float(np.corrcoef(curvature_values, speedups)[0, 1])
    monotonic = bool(np.all(np.diff(speedups) >= 0.0))
    result = {
        "schema": "waxs-curvature-isolated-benchmark-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "curvature-only sweep at fixed protein object, q samples, azimuth samples, "
            "target count, form factors, accuracy, and FINUFFT blocking"
        ),
        "configuration": {
            "structure_path": args.structure.as_posix(),
            "structure_id": metadata.get("structure_id", args.structure.stem),
            "atoms": int(coords.shape[0]),
            "elements": list(element_order),
            "source_mode": "same_binned",
            "source_count": int(sources.shape[0]),
            "qmin": args.qmin,
            "qmax": args.qmax,
            "q_unit": args.q_unit,
            "nq": args.nq,
            "n_phi": int(phi.size),
            "targets": int(args.nq * phi.size),
            "reference_wavelength_nm": args.wavelength_nm,
            "curvature_scales": curvature_scales,
            "curvature_definition": (
                "q_z=-scale*q^2/(2*k_ref), q_perp=sqrt(q^2-q_z^2); "
                "scale=0 planar and scale=1 physical Ewald at reference wavelength"
            ),
            "r_dependent_margin": args.r_dependent_margin,
            "finufft_eps": args.finufft_eps,
            "finufft_threads": args.finufft_threads,
            "finufft_q_block_size": args.finufft_q_block_size,
            "repeats": args.repeats,
            "grid": {"n_r": grid["n_r"], "n_z": grid["n_z"], "n_phi": grid["n_phi"]},
        },
        "shared_sparse_build_s": sparse_build_s,
        **memory_fields("shared_sparse_build", sparse_build_memory),
        "rows": ordered,
        "summary": {
            "speedup_monotonic_non_decreasing": monotonic,
            "pearson_curvature_speedup": pearson,
            "planar_warm_speedup": ordered[0]["warm_speedup_finufft_over_acfo"],
            "physical_ewald_warm_speedup": next(
                row["warm_speedup_finufft_over_acfo"]
                for row in ordered
                if abs(float(row["curvature_scale"]) - 1.0) < 1e-12
            ),
            "strongest_warm_speedup": ordered[-1]["warm_speedup_finufft_over_acfo"],
            "interpretation": (
                "A monotonic increase supports curvature-amplified relative advantage; "
                "non-monotonic behavior indicates target-range or implementation effects dominate."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "finufft": finufft.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
