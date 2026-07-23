from __future__ import annotations

import argparse
import ctypes
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
from benchmark_physical_scaling import timed_call  # noqa: E402
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


def timed(function):
    start = time.perf_counter()
    value = function()
    return value, time.perf_counter() - start


def physical_memory_status() -> dict[str, float | None]:
    """Return total/available physical memory without an optional dependency."""

    total = None
    available = None
    if os.name == "nt":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            total = int(status.ullTotalPhys)
            available = int(status.ullAvailPhys)
    elif hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            total = page_size * int(os.sysconf("SC_PHYS_PAGES"))
            available = page_size * int(os.sysconf("SC_AVPHYS_PAGES"))
        except (OSError, ValueError):
            pass
    return {
        "total_mib": None if total is None else total / 1024**2,
        "available_mib": None if available is None else available / 1024**2,
        "available_fraction": (
            None
            if total is None or available is None or total <= 0
            else available / total
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One isolated q-sampling case for a 1M repeated protein crystal."
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--axis", choices=("resolution", "range"), required=True)
    parser.add_argument("--q-min", type=float, required=True)
    parser.add_argument("--q-max", type=float, required=True)
    parser.add_argument("--nq", type=int, required=True)
    parser.add_argument("--nphi", type=int, required=True)
    parser.add_argument("--wavelength-nm", type=float, default=0.08)
    parser.add_argument("--harmonic-margin", type=int, default=48)
    parser.add_argument("--finufft-eps", type=float, default=1e-6)
    parser.add_argument("--finufft-threads", type=int, default=4)
    parser.add_argument(
        "--finufft-mode",
        choices=("reusable", "chunked", "skip"),
        default="reusable",
    )
    parser.add_argument("--finufft-q-block-size", type=int, default=2)
    parser.add_argument("--finufft-wall-threshold-s", type=float)
    parser.add_argument("--memory-sample-interval-s", type=float, default=0.02)
    parser.add_argument(
        "--finufft-max-process-rss-fraction", type=float, default=0.75
    )
    parser.add_argument(
        "--finufft-min-available-memory-fraction", type=float, default=0.20
    )
    parser.add_argument("--compare-legacy", action="store_true")
    parser.add_argument(
        "--lattice-backend", choices=("direct", "separable"), default="direct"
    )
    parser.add_argument("--compare-direct-lattice", action="store_true")
    parser.add_argument(
        "--coefficient-backend",
        choices=("baseline", "fused_phase", "cached_phase"),
        default="baseline",
    )
    parser.add_argument(
        "--unit",
        type=Path,
        default=Path(
            "structures/processed/protein_nanocrystal_lysozyme_1iee_1x1x1_fixed.npz"
        ),
    )
    parser.add_argument(
        "--supercell",
        type=Path,
        default=Path(
            "structures/processed/protein_nanocrystal_lysozyme_1iee_5x5x5_fixed.npz"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.nq < 2 or args.nphi < 4:
        raise ValueError("nq must be >=2 and nphi must be >=4")
    if args.finufft_q_block_size <= 0:
        raise ValueError("finufft-q-block-size must be positive")
    if args.finufft_wall_threshold_s is not None and args.finufft_wall_threshold_s <= 0:
        raise ValueError("finufft-wall-threshold-s must be positive")
    if args.memory_sample_interval_s <= 0:
        raise ValueError("memory-sample-interval-s must be positive")
    if not 0.0 < args.finufft_max_process_rss_fraction <= 1.0:
        raise ValueError("finufft-max-process-rss-fraction must be in (0, 1]")
    if not 0.0 <= args.finufft_min_available_memory_fraction < 1.0:
        raise ValueError(
            "finufft-min-available-memory-fraction must be in [0, 1)"
        )

    total_start = time.perf_counter()
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
    unit_weights = np.ones(unit_coords.shape[0], dtype=np.complex128)
    source_weights = np.ones(coords.shape[0], dtype=np.complex128)

    target_start = time.perf_counter()
    q_report = np.linspace(args.q_min, args.q_max, args.nq, dtype=np.float64)
    q_solver = np.asarray(
        [q_to_inv_nm(value, "inv_angstrom") for value in q_report]
    )
    q_perp, q_z_rows = ewald_ring(q_solver, args.wavelength_nm)
    ff_mapping = build_form_factors(unit_elements, q_solver, "xray_f0")
    form_factors = normalize_form_factors(
        element_order, q_solver, ff_mapping
    ).astype(np.complex128, copy=False)
    phi = (np.arange(args.nphi) + 0.5) * (2.0 * np.pi / args.nphi)
    qx = np.ascontiguousarray(
        (q_perp[:, None] * np.cos(phi)[None, :]).ravel()
    )
    qy = np.ascontiguousarray(
        (q_perp[:, None] * np.sin(phi)[None, :]).ravel()
    )
    qz = np.ascontiguousarray(
        np.broadcast_to(q_z_rows[:, None], (args.nq, args.nphi)).ravel()
    )
    target_q_indices = np.repeat(np.arange(args.nq), args.nphi)
    target_preparation_seconds = time.perf_counter() - target_start

    plan, factorized_plan_setup_seconds = timed(
        lambda: PreparedExactCoordinateHarmonicPlan(
            unit_coords,
            q_perp,
            q_z_rows,
            phi,
            element_indices=unit_e,
            form_factors=form_factors,
            harmonic_margin=args.harmonic_margin,
            prepare_direct_basis=False,
            coefficient_backend=args.coefficient_backend,
        )
    )
    if not plan.fft_supported:
        raise RuntimeError(
            f"nphi={args.nphi} is too small for max cutoff {plan.max_cutoff}"
        )
    if args.lattice_backend == "separable":
        lattice, lattice_setup_seconds = timed(
            lambda: translation_lattice_factor_separable(
                qx,
                qy,
                qz,
                translations,
                metadata["supercell"],
            ).reshape(args.nq, args.nphi)
        )
    else:
        lattice, lattice_setup_seconds = timed(
            lambda: translation_lattice_factor(
                qx, qy, qz, translations
            ).reshape(args.nq, args.nphi)
        )
    lattice_comparison = None
    if args.compare_direct_lattice:
        direct_lattice, direct_lattice_seconds = timed(
            lambda: translation_lattice_factor(
                qx, qy, qz, translations
            ).reshape(args.nq, args.nphi)
        )
        lattice_comparison = {
            "direct_seconds": direct_lattice_seconds,
            "selected_seconds": lattice_setup_seconds,
            "direct_over_selected_speedup": (
                direct_lattice_seconds / lattice_setup_seconds
            ),
            "complex_l2": relative_l2(direct_lattice, lattice),
            "intensity_l2": relative_l2(
                intensity(direct_lattice), intensity(lattice)
            ),
        }
        del direct_lattice
        gc.collect()

    def factorized_execute():
        unit, _ = plan.execute(
            atom_weights=unit_weights, synthesis_backend="fft"
        )
        return unit * lattice, dict(plan.last_profile)

    (factorized_first, factorized_first_profile), factorized_first_seconds = timed(
        factorized_execute
    )
    gc.collect()
    (factorized_hot, factorized_hot_profile), factorized_hot_seconds = timed(
        factorized_execute
    )
    print(
        f"{args.label}: factorized setup {factorized_plan_setup_seconds + lattice_setup_seconds:.3f} s; "
        f"first/hot {factorized_first_seconds:.3f}/{factorized_hot_seconds:.3f} s; "
        f"coefficient {factorized_hot_profile['coefficient_contraction_seconds']:.3f} s; "
        f"synthesis {factorized_hot_profile['azimuth_synthesis_seconds']:.3f} s",
        flush=True,
    )

    legacy_comparison = None
    if args.compare_legacy:
        def legacy_execute() -> np.ndarray:
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
            return unit * lattice

        legacy_value, legacy_seconds = timed(legacy_execute)
        legacy_comparison = {
            "seconds": legacy_seconds,
            "legacy_over_prepared_hot": legacy_seconds / factorized_hot_seconds,
            "complex_l2_vs_prepared": relative_l2(
                legacy_value, factorized_hot
            ),
            "intensity_l2_vs_prepared": relative_l2(
                intensity(legacy_value), intensity(factorized_hot)
            ),
        }
        del legacy_value
        gc.collect()
        print(
            f"{args.label}: legacy {legacy_seconds:.3f} s; "
            f"legacy/prepared {legacy_comparison['legacy_over_prepared_hot']:.3f}x; "
            f"complex L2 {legacy_comparison['complex_l2_vs_prepared']:.3e}",
            flush=True,
        )

    if args.finufft_mode == "skip":
        factorized_specific_setup = (
            factorized_plan_setup_seconds + lattice_setup_seconds
        )
        result = {
            "schema": "protein-lattice-q-sampling-case-v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "label": args.label,
            "axis": args.axis,
            "contract": {
                "unit": args.unit.as_posix(),
                "supercell": args.supercell.as_posix(),
                "q_min_inv_angstrom": args.q_min,
                "q_max_inv_angstrom": args.q_max,
                "nq": args.nq,
                "dq_inv_angstrom": float(q_report[1] - q_report[0]),
                "nphi": args.nphi,
                "wavelength_nm": args.wavelength_nm,
                "harmonic_margin": args.harmonic_margin,
                "finufft_mode": "skip",
                "finufft_q_block_size": args.finufft_q_block_size,
                "finufft_wall_threshold_seconds": args.finufft_wall_threshold_s,
                "finufft_max_process_rss_fraction": args.finufft_max_process_rss_fraction,
                "finufft_min_available_memory_fraction": args.finufft_min_available_memory_fraction,
                "memory_sample_interval_seconds": args.memory_sample_interval_s,
                "lattice_backend": args.lattice_backend,
                "coefficient_backend": args.coefficient_backend,
            },
            "atom_count": int(coords.shape[0]),
            "cell_count": int(translations.shape[0]),
            "unit_atom_count": int(unit_coords.shape[0]),
            "unit_structure": unit_metadata.get("structure_id", args.unit.stem),
            "supercell_shape": metadata.get("supercell"),
            "repetition_residual_nm": repetition_residual,
            "target_count": int(qx.size),
            "maximum_harmonic": plan.max_cutoff,
            "timing_seconds": {
                "shared_target_preparation": target_preparation_seconds,
                "factorized": {
                    "plan_setup": factorized_plan_setup_seconds,
                    "lattice_setup": lattice_setup_seconds,
                    "specific_setup_total": factorized_specific_setup,
                    "first_execute": factorized_first_seconds,
                    "hot_execute": factorized_hot_seconds,
                    "first_total_excluding_shared": (
                        factorized_specific_setup + factorized_first_seconds
                    ),
                    "first_profile": factorized_first_profile,
                    "hot_profile": factorized_hot_profile,
                },
                "finufft": {"mode": "skip"},
            },
            "speedup_finufft_over_factorized": None,
            "legacy_comparison": legacy_comparison,
            "lattice_comparison": lattice_comparison,
            "cross_error": None,
            "gates": {
                "maximum_harmonic_below_nyquist": (
                    plan.max_cutoff < args.nphi // 2
                ),
                "repetition_residual_le_1e_9_nm": repetition_residual <= 1e-9,
                "factorized_repeat_l2_le_1e_13": (
                    relative_l2(factorized_first, factorized_hot) <= 1e-13
                ),
                "legacy_comparison_l2_le_1e_12": (
                    legacy_comparison is None
                    or legacy_comparison["complex_l2_vs_prepared"] <= 1e-12
                ),
                "lattice_comparison_l2_le_1e_9": (
                    lattice_comparison is None
                    or lattice_comparison["complex_l2"] <= 1e-9
                ),
            },
            "elapsed_seconds": time.perf_counter() - total_start,
            "claim_boundary": [
                "Factorized-only profile; FINUFFT timing is intentionally omitted.",
                "The execute uses prepared coordinates/cutoffs and uniform-phi FFT synthesis.",
            ],
        }
        result["passed"] = all(result["gates"].values())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}", flush=True)
        return

    stop_reasons = []
    if args.finufft_mode == "reusable":
        (plans, masks), finufft_setup_seconds = timed(
            lambda: build_finufft_plans(
                coords,
                element_indices,
                qx,
                qy,
                qz,
                n_elements=len(element_order),
                eps=args.finufft_eps,
                threads=args.finufft_threads,
            )
        )

        def finufft_execute() -> np.ndarray:
            active = np.zeros(qx.size, dtype=np.complex128)
            for element_index, (finufft_plan, mask) in enumerate(zip(plans, masks)):
                values = finufft_plan.execute(
                    np.ascontiguousarray(source_weights[mask])
                )
                active += values * form_factors[element_index, target_q_indices]
            return active.reshape(args.nq, args.nphi)

        finufft_first, finufft_first_seconds = timed(finufft_execute)
        gc.collect()
        finufft_hot, finufft_hot_seconds = timed(finufft_execute)
        finufft_wall_seconds = finufft_setup_seconds + finufft_first_seconds
        finufft_timing = {
            "mode": "reusable",
            "plan_setup": finufft_setup_seconds,
            "first_execute": finufft_first_seconds,
            "hot_execute": finufft_hot_seconds,
            "first_total_excluding_shared": finufft_wall_seconds,
        }
        speedup_hot = finufft_hot_seconds / factorized_hot_seconds
        finufft_for_cross = finufft_hot
    else:
        chunked_start = time.perf_counter()
        finufft_setup_seconds = 0.0
        finufft_first_seconds = 0.0
        cleanup_seconds = 0.0
        finufft_first = np.zeros(
            (args.nq, args.nphi), dtype=np.complex128
        )
        block_count = 0
        block_rows = []
        completed_nq = 0
        initial_memory = physical_memory_status()
        for q_start in range(0, args.nq, args.finufft_q_block_size):
            block_start = time.perf_counter()
            q_stop = min(q_start + args.finufft_q_block_size, args.nq)
            target_slice = slice(q_start * args.nphi, q_stop * args.nphi)
            (plans, masks), block_setup, setup_memory = timed_call(
                lambda target_slice=target_slice: build_finufft_plans(
                    coords,
                    element_indices,
                    qx[target_slice],
                    qy[target_slice],
                    qz[target_slice],
                    n_elements=len(element_order),
                    eps=args.finufft_eps,
                    threads=args.finufft_threads,
                ),
                measure_memory=True,
                sample_interval_s=args.memory_sample_interval_s,
            )
            finufft_setup_seconds += block_setup

            def execute_block() -> np.ndarray:
                active = np.zeros(
                    (q_stop - q_start) * args.nphi, dtype=np.complex128
                )
                block_indices = np.repeat(
                    np.arange(q_start, q_stop), args.nphi
                )
                for element_index, (finufft_plan, mask) in enumerate(
                    zip(plans, masks)
                ):
                    values = finufft_plan.execute(
                        np.ascontiguousarray(source_weights[mask])
                    )
                    active += values * form_factors[
                        element_index, block_indices
                    ]
                return active.reshape(q_stop - q_start, args.nphi)

            block_value, block_execute, execute_memory = timed_call(
                execute_block,
                measure_memory=True,
                sample_interval_s=args.memory_sample_interval_s,
            )
            finufft_first_seconds += block_execute
            finufft_first[q_start:q_stop] = block_value
            cleanup_start = time.perf_counter()
            del plans, masks, block_value
            gc.collect()
            block_cleanup = time.perf_counter() - cleanup_start
            cleanup_seconds += block_cleanup
            block_count += 1
            completed_nq = q_stop
            block_memory = physical_memory_status()
            peak_rss_values = [
                memory["peak_rss_mib"]
                for memory in (setup_memory, execute_memory)
                if memory is not None and memory.get("peak_rss_mib") is not None
            ]
            peak_rss_mib = max(peak_rss_values, default=None)
            total_mib = block_memory["total_mib"]
            peak_rss_fraction = (
                None
                if peak_rss_mib is None or total_mib is None or total_mib <= 0
                else peak_rss_mib / total_mib
            )
            block_wall = time.perf_counter() - block_start
            cumulative_wall = time.perf_counter() - chunked_start
            block_rows.append(
                {
                    "block_index": block_count - 1,
                    "q_start_index": q_start,
                    "q_stop_index": q_stop,
                    "q_min_inv_angstrom": float(q_report[q_start]),
                    "q_max_inv_angstrom": float(q_report[q_stop - 1]),
                    "target_count": int((q_stop - q_start) * args.nphi),
                    "setup_seconds": block_setup,
                    "execute_seconds": block_execute,
                    "cleanup_seconds": block_cleanup,
                    "block_wall_seconds": block_wall,
                    "cumulative_wall_seconds": cumulative_wall,
                    "setup_memory": setup_memory,
                    "execute_memory": execute_memory,
                    "physical_memory_after_cleanup": block_memory,
                    "peak_process_rss_fraction": peak_rss_fraction,
                }
            )
            print(
                f"{args.label}: chunk {block_count} q[{q_start}:{q_stop}] "
                f"setup {block_setup:.3f} s execute {block_execute:.3f} s",
                flush=True,
            )
            if (
                args.finufft_wall_threshold_s is not None
                and cumulative_wall >= args.finufft_wall_threshold_s
            ):
                stop_reasons.append("wall_time_threshold")
            if (
                peak_rss_fraction is not None
                and peak_rss_fraction >= args.finufft_max_process_rss_fraction
            ):
                stop_reasons.append("process_rss_fraction_threshold")
            if (
                block_memory["available_fraction"] is not None
                and block_memory["available_fraction"]
                <= args.finufft_min_available_memory_fraction
            ):
                stop_reasons.append("available_memory_fraction_threshold")
            if stop_reasons:
                print(
                    f"{args.label}: stopping after {completed_nq}/{args.nq} q rows: "
                    f"{', '.join(stop_reasons)}",
                    flush=True,
                )
                break
        finufft_wall_seconds = time.perf_counter() - chunked_start
        finufft_hot_seconds = None
        censored = completed_nq < args.nq
        finufft_timing = {
            "mode": "chunked_streaming",
            "q_block_size": args.finufft_q_block_size,
            "block_count": block_count,
            "summed_plan_setup": finufft_setup_seconds,
            "summed_execute": finufft_first_seconds,
            "summed_cleanup": cleanup_seconds,
            "streamed_wall_excluding_shared": finufft_wall_seconds,
            "block_rows": block_rows,
            "completed_nq": completed_nq,
            "completion_fraction": completed_nq / args.nq,
            "censored": censored,
            "stop_reasons": stop_reasons,
            "stop_contract": {
                "wall_threshold_seconds": args.finufft_wall_threshold_s,
                "max_process_rss_fraction": args.finufft_max_process_rss_fraction,
                "min_available_memory_fraction": args.finufft_min_available_memory_fraction,
                "initial_physical_memory": initial_memory,
            },
        }
        speedup_hot = None
        finufft_for_cross = finufft_first[:completed_nq]
        factorized_for_cross = factorized_hot[:completed_nq]
    if args.finufft_mode == "reusable":
        censored = False
        completed_nq = args.nq
        factorized_for_cross = factorized_hot
    cross_error = {
        "complex_l2": relative_l2(finufft_for_cross, factorized_for_cross),
        "intensity_l2": relative_l2(
            intensity(finufft_for_cross), intensity(factorized_for_cross)
        ),
        "factorized_first_hot_complex_l2": relative_l2(
            factorized_first, factorized_hot
        ),
        "completed_nq": completed_nq,
    }
    factorized_specific_setup = (
        factorized_plan_setup_seconds + lattice_setup_seconds
    )
    result = {
        "schema": "protein-lattice-q-sampling-case-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "axis": args.axis,
        "contract": {
            "unit": args.unit.as_posix(),
            "supercell": args.supercell.as_posix(),
            "q_min_inv_angstrom": args.q_min,
            "q_max_inv_angstrom": args.q_max,
            "nq": args.nq,
            "dq_inv_angstrom": float(q_report[1] - q_report[0]),
            "nphi": args.nphi,
            "wavelength_nm": args.wavelength_nm,
            "harmonic_margin": args.harmonic_margin,
            "finufft_eps": args.finufft_eps,
            "finufft_threads": args.finufft_threads,
            "finufft_mode": args.finufft_mode,
            "finufft_q_block_size": args.finufft_q_block_size,
            "finufft_wall_threshold_seconds": args.finufft_wall_threshold_s,
            "finufft_max_process_rss_fraction": args.finufft_max_process_rss_fraction,
            "finufft_min_available_memory_fraction": args.finufft_min_available_memory_fraction,
            "memory_sample_interval_seconds": args.memory_sample_interval_s,
            "lattice_backend": args.lattice_backend,
            "coefficient_backend": args.coefficient_backend,
        },
        "atom_count": int(coords.shape[0]),
        "cell_count": int(translations.shape[0]),
        "unit_atom_count": int(unit_coords.shape[0]),
        "unit_structure": unit_metadata.get("structure_id", args.unit.stem),
        "supercell_shape": metadata.get("supercell"),
        "repetition_residual_nm": repetition_residual,
        "target_count": int(qx.size),
        "maximum_harmonic": plan.max_cutoff,
        "timing_seconds": {
            "shared_target_preparation": target_preparation_seconds,
            "factorized": {
                "plan_setup": factorized_plan_setup_seconds,
                "lattice_setup": lattice_setup_seconds,
                "specific_setup_total": factorized_specific_setup,
                "first_execute": factorized_first_seconds,
                "hot_execute": factorized_hot_seconds,
                "first_total_excluding_shared": (
                    factorized_specific_setup + factorized_first_seconds
                ),
                "first_profile": factorized_first_profile,
                "hot_profile": factorized_hot_profile,
            },
            "finufft": finufft_timing,
        },
        "speedup_finufft_over_factorized": {
            "first_total_excluding_shared": (
                finufft_wall_seconds
                / (factorized_specific_setup + factorized_first_seconds)
            ),
            "hot_execute": speedup_hot,
            "first_total_is_lower_bound": censored,
        },
        "cross_error": cross_error,
        "legacy_comparison": legacy_comparison,
        "lattice_comparison": lattice_comparison,
        "gates": {
            "maximum_harmonic_below_nyquist": plan.max_cutoff < args.nphi // 2,
            "repetition_residual_le_1e_9_nm": repetition_residual <= 1e-9,
            "cross_complex_l2_le_2e_6": cross_error["complex_l2"] <= 2e-6,
            "cross_intensity_l2_le_5e_6": cross_error["intensity_l2"] <= 5e-6,
            "factorized_repeat_l2_le_1e_13": (
                cross_error["factorized_first_hot_complex_l2"] <= 1e-13
            ),
            "lattice_comparison_l2_le_1e_9": (
                lattice_comparison is None
                or lattice_comparison["complex_l2"] <= 1e-9
            ),
            "censored_stop_has_reason": (not censored or bool(stop_reasons)),
        },
        "elapsed_seconds": time.perf_counter() - total_start,
        "claim_boundary": [
            "This is a single-process exploratory scaling case, not a 10/30 timing claim.",
            "Setup and execute are reported separately; streamed chunking also reports cleanup and wall total.",
            "The factorized execute uses prepared coordinates/cutoffs and uniform-phi FFT synthesis.",
            "FINUFFT eps=1e-6 is a practical timing baseline; direct NDFT correctness is separate.",
            (
                "A censored chunked row reports a measured lower bound, not an extrapolated complete runtime."
                if censored
                else "The chunked FINUFFT row completed all requested q rows."
            ),
        ],
    }
    result["passed"] = all(result["gates"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"{args.label}: FINUFFT mode {args.finufft_mode}; setup sum {finufft_setup_seconds:.3f} s; "
        f"execute sum {finufft_first_seconds:.3f} s; wall {finufft_wall_seconds:.3f} s; "
        f"first-total speedup {result['speedup_finufft_over_factorized']['first_total_excluding_shared']:.3f}x; "
        f"cross L2 {cross_error['complex_l2']:.3e}",
        flush=True,
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
