from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_physical_scaling import memory_fields, timed_call  # noqa: E402
from benchmark_protein_nanocrystal_sparse_memory import build_sparse_structure  # noqa: E402
from validate_public_waxs_crystals import choose_grid  # noqa: E402
from validate_public_waxs_structures import build_form_factors, load_structure  # noqa: E402
from waxs_cake import PreparedCakePlan, encode_elements  # noqa: E402
from waxs_cake.geometry import ewald_ring  # noqa: E402
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402
from waxs_cake.physical_scaling import q_to_inv_nm  # noqa: E402
from waxs_cake.solvers import normalize_form_factors  # noqa: E402


def median_calls(func, repeats: int) -> tuple[np.ndarray, float, list[float]]:
    value = None
    times: list[float] = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        value = func()
        times.append(time.perf_counter() - start)
    assert value is not None
    return value, float(median(times)), times


def timing_summary(times: list[float]) -> dict[str, float | int | None]:
    if not times:
        return {"count": 0, "median_s": None, "p05_s": None, "p95_s": None}
    values = np.asarray(times, dtype=np.float64)
    return {
        "count": int(values.size),
        "median_s": float(np.median(values)),
        "p05_s": float(np.percentile(values, 5)),
        "p95_s": float(np.percentile(values, 95)),
    }


def alternating_calls(
    func_a,
    func_b,
    repeats: int,
    *,
    progress_label: str,
) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    value_a = None
    value_b = None
    times_a: list[float] = []
    times_b: list[float] = []
    for repeat in range(repeats):
        ordered = (("a", func_a), ("b", func_b))
        if repeat % 2:
            ordered = tuple(reversed(ordered))
        for label, func in ordered:
            gc.collect()
            start = time.perf_counter()
            value = func()
            elapsed = time.perf_counter() - start
            if label == "a":
                value_a = value
                times_a.append(elapsed)
            else:
                value_b = value
                times_b.append(elapsed)
        if (repeat + 1) % 5 == 0 or repeat + 1 == repeats:
            print(
                f"{progress_label}: completed {repeat + 1}/{repeats} AB/BA pairs",
                flush=True,
            )
    assert value_a is not None and value_b is not None
    return value_a, value_b, times_a, times_b


def source_arrays(
    source_mode: str,
    coords: np.ndarray,
    element_indices: np.ndarray,
    sparse,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if source_mode == "exact_atoms":
        return (
            np.ascontiguousarray(coords, dtype=np.float64),
            np.ascontiguousarray(element_indices, dtype=np.intp),
            np.ones(coords.shape[0], dtype=np.complex128),
        )
    radius = sparse.r_centers[sparse.active_r]
    beta = sparse.beta_centers[sparse.active_beta]
    xyz = np.column_stack(
        (
            radius * np.cos(beta),
            radius * np.sin(beta),
            sparse.z_centers[sparse.active_z],
        )
    )
    return (
        np.ascontiguousarray(xyz, dtype=np.float64),
        np.ascontiguousarray(sparse.active_e, dtype=np.intp),
        np.ascontiguousarray(sparse.active_values, dtype=np.complex128),
    )


def detector_rectangle_mask(
    q_inv_angstrom: np.ndarray,
    phi: np.ndarray,
    *,
    wavelength_nm: float,
    active_width_mm: float,
    active_height_mm: float,
    distance_mm: float,
) -> np.ndarray:
    """Return polar nodes whose rays intersect a rectangular flat detector."""
    if wavelength_nm <= 0.0:
        raise ValueError("wavelength-nm must be positive")
    if active_width_mm <= 0.0 or active_height_mm <= 0.0 or distance_mm <= 0.0:
        raise ValueError("detector dimensions and distance must be positive")
    wavelength_angstrom = 10.0 * wavelength_nm
    sine_half_angle = np.asarray(q_inv_angstrom, dtype=np.float64) * wavelength_angstrom / (
        4.0 * np.pi
    )
    if np.any(sine_half_angle < 0.0) or np.any(sine_half_angle > 1.0):
        raise ValueError("q range lies outside the elastic scattering sphere")
    two_theta = 2.0 * np.arcsin(sine_half_angle)
    radius_mm = distance_mm * np.tan(two_theta)
    x_mm = radius_mm[:, None] * np.cos(phi)[None, :]
    y_mm = radius_mm[:, None] * np.sin(phi)[None, :]
    return np.ascontiguousarray(
        (np.abs(x_mm) <= 0.5 * active_width_mm)
        & (np.abs(y_mm) <= 0.5 * active_height_mm)
    )


def masked_ring_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    counts = np.sum(mask, axis=1)
    safe_counts = np.maximum(counts, 1)
    return np.sum(np.where(mask, values, 0.0), axis=1) / safe_counts


def masked_row_relative_l2(
    values: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    rows = []
    for index in range(values.shape[0]):
        active = mask[index]
        if not np.any(active):
            continue
        denominator = np.linalg.norm(reference[index, active])
        numerator = np.linalg.norm(values[index, active] - reference[index, active])
        rows.append(0.0 if denominator == 0.0 and numerator == 0.0 else numerator / denominator)
    return np.asarray(rows, dtype=np.float64)


def build_finufft_plans(
    sources: np.ndarray,
    source_e: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    *,
    n_elements: int,
    eps: float,
    threads: int,
):
    import finufft

    plans = []
    masks = []
    for element_index in range(n_elements):
        mask = np.flatnonzero(source_e == element_index)
        plan = finufft.Plan(
            3,
            3,
            eps=eps,
            isign=1,
            dtype="complex128",
            nthreads=threads,
        )
        plan.setpts(
            np.ascontiguousarray(sources[mask, 0]),
            np.ascontiguousarray(sources[mask, 1]),
            np.ascontiguousarray(sources[mask, 2]),
            qx,
            qy,
            qz,
        )
        plans.append(plan)
        masks.append(mask)
    return plans, masks


def break_even_repeat(
    acfo_setup: float,
    acfo_first: float,
    acfo_cached: float,
    baseline_setup: float,
    baseline_first: float,
    baseline_cached: float,
    *,
    limit: int = 1000,
) -> int | None:
    for repeat in range(1, limit + 1):
        acfo = acfo_setup + acfo_first + (repeat - 1) * acfo_cached
        baseline = baseline_setup + baseline_first + (repeat - 1) * baseline_cached
        if baseline >= acfo:
            return repeat
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fair reusable-plan FINUFFT baseline for a true-sparse protein nanocrystal."
    )
    parser.add_argument("structure", type=Path)
    parser.add_argument("--source-mode", choices=["exact_atoms", "same_binned"], default="same_binned")
    parser.add_argument("--qmin", type=float, default=0.05)
    parser.add_argument("--qmax", type=float, default=6.3)
    parser.add_argument("--q-unit", choices=["inv_angstrom", "inv_nm"], default="inv_angstrom")
    parser.add_argument("--nq", type=int, default=40)
    parser.add_argument("--wavelength-nm", type=float, default=0.1)
    parser.add_argument("--bin-width-nm", type=float, default=0.1)
    parser.add_argument("--nphi-min", type=int, default=1024)
    parser.add_argument("--harmonic-margin", type=int, default=16)
    parser.add_argument("--r-dependent-margin", type=int, default=16)
    parser.add_argument("--cutoff-bin-size", type=int, default=16)
    parser.add_argument("--acfo-q-block-size", type=int, default=2)
    parser.add_argument("--profile-chunk-size", type=int, default=8)
    parser.add_argument("--form-factor-model", default="xray_f0")
    parser.add_argument("--finufft-eps", type=float, default=1e-6)
    parser.add_argument("--finufft-threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--detector-label", default="full_polar_grid")
    parser.add_argument("--detector-active-width-mm", type=float)
    parser.add_argument("--detector-active-height-mm", type=float)
    parser.add_argument("--detector-distance-mm", type=float)
    parser.add_argument(
        "--finufft-q-block-size",
        type=int,
        help="Memory-safe q rows per FINUFFT plan; setup is then included in every blocked evaluation.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--warmups",
        type=int,
        default=0,
        help="Unmeasured hot-cache calls per method after the separately recorded first call.",
    )
    parser.add_argument(
        "--timing-order",
        choices=["sequential", "alternating"],
        default="sequential",
        help="Run all ACFO calls first, or alternate AB/BA pairs to reduce time/order bias.",
    )
    parser.add_argument("--memory-sample-interval-s", type=float, default=0.002)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/protein_nanocrystal_finufft_fair.json"),
    )
    parser.add_argument(
        "--amplitude-output",
        type=Path,
        help="Optional NPZ path for ACFO and FINUFFT complex amplitudes used by follow-up sweeps.",
    )
    args = parser.parse_args()
    if args.nq <= 0 or args.repeats < 0 or args.warmups < 0 or args.finufft_threads <= 0:
        raise ValueError(
            "nq and finufft-threads must be positive; repeats and warmups must be nonnegative"
        )
    detector_values = (
        args.detector_active_width_mm,
        args.detector_active_height_mm,
        args.detector_distance_mm,
    )
    if any(value is not None for value in detector_values) and not all(
        value is not None for value in detector_values
    ):
        raise ValueError(
            "detector-active-width-mm, detector-active-height-mm, and detector-distance-mm "
            "must be supplied together"
        )

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
        [q_to_inv_nm(value, args.q_unit) for value in q_report],
        dtype=np.float64,
    )
    form_factors = build_form_factors(elements, q_solver, args.form_factor_model)
    ff = normalize_form_factors(element_order, q_solver, form_factors)

    sparse, sparse_build_s, sparse_build_memory = timed_call(
        lambda: build_sparse_structure(coords, elements, grid, index_backend="cpp"),
        measure_memory=True,
        sample_interval_s=args.memory_sample_interval_s,
    )
    acfo_plan, acfo_plan_s, acfo_plan_memory = timed_call(
        lambda: PreparedCakePlan(
            sparse,
            q_solver,
            args.wavelength_nm,
            form_factors=form_factors,
            circular_backend="cpp",
            complex_dtype=np.complex64,
            q_block_size=args.acfo_q_block_size,
        ),
        measure_memory=True,
        sample_interval_s=args.memory_sample_interval_s,
    )
    acfo_solve = lambda: acfo_plan.circular_fft_sparse_source_r_dependent(
        margin=args.r_dependent_margin,
        cutoff_bin_size=args.cutoff_bin_size,
        analytic_kernel=True,
        q_block_size=args.acfo_q_block_size,
        profile_chunk_size=args.profile_chunk_size,
    )
    acfo_first, acfo_first_s, acfo_first_memory = timed_call(
        acfo_solve,
        measure_memory=True,
        sample_interval_s=args.memory_sample_interval_s,
    )
    acfo_warmup_times: list[float] = []
    acfo_cached, acfo_cached_s, acfo_cached_times = acfo_first, acfo_first_s, []
    if args.timing_order == "sequential":
        if args.warmups:
            _, _, acfo_warmup_times = median_calls(acfo_solve, args.warmups)
        if args.repeats:
            acfo_cached, acfo_cached_s, acfo_cached_times = median_calls(
                acfo_solve,
                args.repeats,
            )

    sources, source_e, source_weights = source_arrays(
        args.source_mode,
        coords,
        element_indices,
        sparse,
    )
    q_perp, q_z = ewald_ring(q_solver, args.wavelength_nm)
    phi = sparse.beta_centers
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    if all(value is not None for value in detector_values):
        detector_mask = detector_rectangle_mask(
            q_report,
            phi,
            wavelength_nm=args.wavelength_nm,
            active_width_mm=float(args.detector_active_width_mm),
            active_height_mm=float(args.detector_active_height_mm),
            distance_mm=float(args.detector_distance_mm),
        )
        detector_mode = "rectangular_flat_detector"
    else:
        detector_mask = np.ones((args.nq, phi.size), dtype=bool)
        detector_mode = "full_polar_grid"
    target_q_indices = np.broadcast_to(
        np.arange(args.nq, dtype=np.intp)[:, None], detector_mask.shape
    )[detector_mask]

    if args.finufft_q_block_size is None:
        qx = np.ascontiguousarray(
            (q_perp[:, None] * cos_phi[None, :])[detector_mask]
        )
        qy = np.ascontiguousarray(
            (q_perp[:, None] * sin_phi[None, :])[detector_mask]
        )
        qz_targets = np.ascontiguousarray(
            np.broadcast_to(q_z[:, None], detector_mask.shape)[detector_mask]
        )
        if qx.size == 0:
            raise ValueError("detector mask contains no active targets")
        (finufft_plans, finufft_masks), finufft_setup_s, finufft_setup_memory = timed_call(
            lambda: build_finufft_plans(
                sources,
                source_e,
                qx,
                qy,
                qz_targets,
                n_elements=len(element_order),
                eps=args.finufft_eps,
                threads=args.finufft_threads,
            ),
            measure_memory=True,
            sample_interval_s=args.memory_sample_interval_s,
        )

        def finufft_execute() -> np.ndarray:
            out = np.zeros((args.nq, phi.size), dtype=np.complex128)
            active_out = np.zeros(qx.size, dtype=np.complex128)
            for element_index, (plan, mask) in enumerate(zip(finufft_plans, finufft_masks)):
                values = plan.execute(np.ascontiguousarray(source_weights[mask]))
                active_out += values * ff[element_index, target_q_indices]
            out[detector_mask] = active_out
            return out

        finufft_plan_mode = "reusable_full_target"
    else:
        block_size = int(args.finufft_q_block_size)
        if block_size <= 0:
            raise ValueError("finufft-q-block-size must be positive")
        finufft_setup_s = 0.0
        finufft_setup_memory = None

        def finufft_execute() -> np.ndarray:
            out = np.zeros((args.nq, phi.size), dtype=np.complex128)
            for start in range(0, args.nq, block_size):
                stop = min(start + block_size, args.nq)
                block_mask = detector_mask[start:stop]
                if not np.any(block_mask):
                    continue
                qx_block = np.ascontiguousarray(
                    (q_perp[start:stop, None] * cos_phi[None, :])[block_mask]
                )
                qy_block = np.ascontiguousarray(
                    (q_perp[start:stop, None] * sin_phi[None, :])[block_mask]
                )
                qz_block = np.ascontiguousarray(
                    np.broadcast_to(q_z[start:stop, None], block_mask.shape)[block_mask]
                )
                local_rows, local_columns = np.nonzero(block_mask)
                global_rows = local_rows + start
                plans, masks = build_finufft_plans(
                    sources,
                    source_e,
                    qx_block,
                    qy_block,
                    qz_block,
                    n_elements=len(element_order),
                    eps=args.finufft_eps,
                    threads=args.finufft_threads,
                )
                for element_index, (plan, mask) in enumerate(zip(plans, masks)):
                    values = plan.execute(np.ascontiguousarray(source_weights[mask]))
                    out[global_rows, local_columns] += values * ff[
                        element_index, global_rows
                    ]
                del plans, masks
            return out

        finufft_plan_mode = "memory_safe_q_blocked_setup_per_evaluation"

    finufft_first, finufft_first_s, finufft_first_memory = timed_call(
        finufft_execute,
        measure_memory=True,
        sample_interval_s=args.memory_sample_interval_s,
    )
    finufft_warmup_times: list[float] = []
    finufft_cached, finufft_cached_s, finufft_cached_times = (
        finufft_first,
        finufft_first_s,
        [],
    )
    if args.timing_order == "sequential":
        if args.warmups:
            _, _, finufft_warmup_times = median_calls(finufft_execute, args.warmups)
        if args.repeats:
            finufft_cached, finufft_cached_s, finufft_cached_times = median_calls(
                finufft_execute,
                args.repeats,
            )
    else:
        if args.warmups:
            _, _, acfo_warmup_times, finufft_warmup_times = alternating_calls(
                acfo_solve,
                finufft_execute,
                args.warmups,
                progress_label="warmup",
            )
        if args.repeats:
            acfo_cached, finufft_cached, acfo_cached_times, finufft_cached_times = (
                alternating_calls(
                    acfo_solve,
                    finufft_execute,
                    args.repeats,
                    progress_label="measured",
                )
            )
            acfo_cached_s = float(median(acfo_cached_times))
            finufft_cached_s = float(median(finufft_cached_times))

    acfo_i = intensity(acfo_cached)
    finufft_i = intensity(finufft_cached)
    intensity_row_error = masked_row_relative_l2(acfo_i, finufft_i, detector_mask)
    active_per_q = np.sum(detector_mask, axis=1)
    totals = []
    for repeat in (1, 10, 100):
        acfo_total = sparse_build_s + acfo_plan_s + acfo_first_s + (repeat - 1) * acfo_cached_s
        baseline_total = finufft_setup_s + finufft_first_s + (repeat - 1) * finufft_cached_s
        totals.append(
            {
                "T": repeat,
                "acfo_total_s": acfo_total,
                "finufft_total_s": baseline_total,
                "speedup_finufft_over_acfo": baseline_total / acfo_total,
                "modeled_from_measured_first_and_cached": repeat > 1,
            }
        )

    row = {
        "structure_path": args.structure.as_posix(),
        "structure_id": metadata.get("structure_id", args.structure.stem),
        "atoms": int(coords.shape[0]),
        "source_mode": args.source_mode,
        "source_count": int(sources.shape[0]),
        "elements": list(element_order),
        "qmin": args.qmin,
        "qmax": args.qmax,
        "q_unit": args.q_unit,
        "nq": args.nq,
        "n_phi": int(phi.size),
        "targets": int(args.nq * phi.size),
        "active_detector_targets": int(np.count_nonzero(detector_mask)),
        "active_detector_fraction": float(np.mean(detector_mask)),
        "active_fraction_at_qmax": float(np.mean(detector_mask[-1])),
        "active_phi_count_min": int(np.min(active_per_q)),
        "active_phi_count_max": int(np.max(active_per_q)),
        "detector_label": args.detector_label,
        "detector_mode": detector_mode,
        "detector_active_width_mm": args.detector_active_width_mm,
        "detector_active_height_mm": args.detector_active_height_mm,
        "detector_distance_mm": args.detector_distance_mm,
        "wavelength_nm": args.wavelength_nm,
        "bin_width_nm": args.bin_width_nm,
        "grid": {"n_r": grid["n_r"], "n_z": grid["n_z"], "n_phi": grid["n_phi"]},
        "form_factor_model": args.form_factor_model,
        "acfo_sparse_build_s": sparse_build_s,
        **memory_fields("acfo_sparse_build", sparse_build_memory),
        "acfo_plan_s": acfo_plan_s,
        **memory_fields("acfo_plan", acfo_plan_memory),
        "acfo_first_s": acfo_first_s,
        **memory_fields("acfo_first", acfo_first_memory),
        "acfo_cached_median_s": acfo_cached_s,
        "acfo_cached_times": acfo_cached_times,
        "acfo_warmup_times": acfo_warmup_times,
        "acfo_timing_summary": timing_summary(acfo_cached_times),
        "finufft_version": finufft.__version__,
        "finufft_eps": args.finufft_eps,
        "finufft_threads": args.finufft_threads,
        "finufft_plan_mode": finufft_plan_mode,
        "finufft_q_block_size": args.finufft_q_block_size,
        "finufft_setup_s": finufft_setup_s,
        **memory_fields("finufft_setup", finufft_setup_memory),
        "finufft_first_s": finufft_first_s,
        **memory_fields("finufft_first", finufft_first_memory),
        "finufft_cached_median_s": finufft_cached_s,
        "finufft_cached_times": finufft_cached_times,
        "finufft_warmup_times": finufft_warmup_times,
        "finufft_timing_summary": timing_summary(finufft_cached_times),
        "timing_protocol": {
            "first_calls_recorded_separately": True,
            "warmups_per_method": args.warmups,
            "measured_repeats_per_method": args.repeats,
            "method_order": (
                "all_acfo_then_all_finufft"
                if args.timing_order == "sequential"
                else "alternating_ab_ba"
            ),
            "garbage_collection_before_each_warmup_and_measured_call": True,
        },
        "metric_domain": "active_detector_nodes",
        "complex_l2_acfo_vs_finufft": relative_l2(
            acfo_cached[detector_mask], finufft_cached[detector_mask]
        ),
        "intensity_l2_acfo_vs_finufft": relative_l2(
            acfo_i[detector_mask], finufft_i[detector_mask]
        ),
        "ring_l2_acfo_vs_finufft": relative_l2(
            masked_ring_mean(acfo_i, detector_mask),
            masked_ring_mean(finufft_i, detector_mask),
        ),
        "intensity_row_relative_l2_median": float(np.median(intensity_row_error)),
        "intensity_row_relative_l2_p99": float(np.quantile(intensity_row_error, 0.99)),
        "warm_speedup_finufft_over_acfo": finufft_cached_s / acfo_cached_s,
        "break_even_repeat": break_even_repeat(
            sparse_build_s + acfo_plan_s,
            acfo_first_s,
            acfo_cached_s,
            finufft_setup_s,
            finufft_first_s,
            finufft_cached_s,
        ),
        "total_time_model": totals,
        "all_finite": bool(
            np.all(np.isfinite(acfo_cached[detector_mask]))
            and np.all(np.isfinite(finufft_cached[detector_mask]))
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "gpu_baseline_status": "unavailable: cufinufft 2.5.1 DLL dependency load failure",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    if args.amplitude_output is not None:
        args.amplitude_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.amplitude_output,
            acfo=np.asarray(acfo_cached),
            finufft=np.asarray(finufft_cached),
            q_report=q_report,
            q_solver=q_solver,
            phi=phi,
            detector_mask=detector_mask,
            metadata_json=np.asarray(json.dumps(row)),
        )
    md = [
        f"# Protein nanocrystal ACFO vs FINUFFT: {row['structure_id']}",
        "",
        f"- source mode: `{row['source_mode']}`; sources `{row['source_count']:,}`; full/active targets `{row['targets']:,}/{row['active_detector_targets']:,}`",
        f"- detector: `{row['detector_label']}` ({row['detector_mode']}); active fraction `{row['active_detector_fraction']:.3f}`; outer-ring fraction `{row['active_fraction_at_qmax']:.3f}`",
        f"- ACFO first/cached: `{row['acfo_first_s']:.3f}/{row['acfo_cached_median_s']:.3f} s`",
        f"- FINUFFT {row['finufft_threads']}-thread setup/first/cached: `{row['finufft_setup_s']:.3f}/{row['finufft_first_s']:.3f}/{row['finufft_cached_median_s']:.3f} s`",
        f"- warm speedup FINUFFT/ACFO: `{row['warm_speedup_finufft_over_acfo']:.2f}x`",
        f"- complex/intensity/ring L2: `{row['complex_l2_acfo_vs_finufft']:.3e}` / `{row['intensity_l2_acfo_vs_finufft']:.3e}` / `{row['ring_l2_acfo_vs_finufft']:.3e}`",
        f"- q-row intensity relative L2 median/p99: `{row['intensity_row_relative_l2_median']:.3e}` / `{row['intensity_row_relative_l2_p99']:.3e}`",
        f"- break-even repeat: `{row['break_even_repeat']}`",
        "",
        "| T | ACFO total s | FINUFFT total s | FINUFFT/ACFO |",
        "|---:|---:|---:|---:|",
    ]
    for item in totals:
        md.append(
            f"| {item['T']} | {item['acfo_total_s']:.3f} | {item['finufft_total_s']:.3f} | {item['speedup_finufft_over_acfo']:.2f}x |"
        )
    md.extend(
        [
            "",
            "T=10/100 totals are projections from measured setup, first, and cached medians; they are not 10/100 fully executed workflows.",
            f"GPU baseline: {row['gpu_baseline_status']}.",
        ]
    )
    args.output.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(row, indent=2))
    print(f"wrote {args.output} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
