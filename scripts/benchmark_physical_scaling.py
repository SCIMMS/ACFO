from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waxs_cake import (  # noqa: E402
    PreparedCakePlan,
    choose_physical_grid,
    make_cylindrical_histogram,
    nufft_amplitude,
    nufft_amplitude_chunked,
)
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402
from waxs_cake.presets import FAST_PRESET_NAMES, apply_fast_preset  # noqa: E402


def synthetic_water_box(n_atoms: int, side_nm: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(
        -0.5 * side_nm,
        0.5 * side_nm,
        size=(n_atoms, 3),
    )


def parse_hist_dtype(value: str) -> np.dtype | None:
    return None if value == "default" else np.dtype(value)


def parse_complex_dtype(value: str) -> np.dtype | None:
    return None if value == "auto" else np.dtype(value)


def median_time(func, repeats: int) -> tuple[object, float, list[float]]:
    value = None
    times = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        value = func()
        times.append(time.perf_counter() - start)
    return value, float(median(times)), times


def current_rss_bytes() -> int | None:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(ProcessMemoryCounters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(
            handle,
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            return None
        return int(counters.WorkingSetSize)

    statm = Path("/proc/self/statm")
    if statm.exists():
        try:
            resident_pages = int(statm.read_text(encoding="utf-8").split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, IndexError, ValueError):
            return None
    return None


def bytes_to_mib(value: int | None) -> float | None:
    return None if value is None else value / 1024**2


def timed_call(
    func,
    *,
    measure_memory: bool,
    sample_interval_s: float,
) -> tuple[object, float, dict | None]:
    if not measure_memory:
        start = time.perf_counter()
        value = func()
        return value, time.perf_counter() - start, None

    gc.collect()
    before = current_rss_bytes()
    peak = before
    samples: list[int] = []
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak
        while not stop.is_set():
            rss = current_rss_bytes()
            if rss is not None:
                samples.append(rss)
                peak = rss if peak is None else max(peak, rss)
            stop.wait(sample_interval_s)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        value = func()
    finally:
        seconds = time.perf_counter() - start
        stop.set()
        thread.join()
    after = current_rss_bytes()
    for rss in (after,):
        if rss is not None:
            peak = rss if peak is None else max(peak, rss)
    if peak is None:
        memory = {
            "rss_before_mib": None,
            "rss_after_mib": None,
            "rss_after_delta_mib": None,
            "peak_rss_mib": None,
            "peak_rss_delta_mib": None,
            "rss_samples": len(samples),
        }
    else:
        memory = {
            "rss_before_mib": bytes_to_mib(before),
            "rss_after_mib": bytes_to_mib(after),
            "rss_after_delta_mib": None
            if before is None or after is None
            else bytes_to_mib(after - before),
            "peak_rss_mib": bytes_to_mib(peak),
            "peak_rss_delta_mib": None
            if before is None
            else bytes_to_mib(max(0, peak - before)),
            "rss_samples": len(samples),
        }
    return value, seconds, memory


def memory_fields(prefix: str, memory: dict | None) -> dict:
    keys = (
        "rss_before_mib",
        "rss_after_mib",
        "rss_after_delta_mib",
        "peak_rss_mib",
        "peak_rss_delta_mib",
        "rss_samples",
    )
    if memory is None:
        return {f"{prefix}_{key}": None for key in keys}
    return {f"{prefix}_{key}": memory.get(key) for key in keys}


def peak_delta_text(memory: dict | None) -> str:
    if memory is None or memory.get("peak_rss_delta_mib") is None:
        return "n/a"
    return f"{memory['peak_rss_delta_mib']:.1f} MiB"


def grid_summary(grid) -> dict:
    row = asdict(grid)
    row.update(
        {
            "height_nm": grid.height_nm,
            "n_bins_per_element": grid.n_bins_per_element,
            "dr_nm": grid.dr_nm,
            "dz_nm": grid.dz_nm,
            "outer_arc_nm": grid.outer_arc_nm,
            "hist_float32_mib_per_element": grid.n_bins_per_element * 4 / 1024**2,
            "hist_complex64_mib_per_element": grid.n_bins_per_element * 8 / 1024**2,
        }
    )
    return row


def occupancy_summary(hist: np.ndarray) -> dict:
    nonzero = hist != 0
    row_counts = np.count_nonzero(nonzero, axis=-1)
    active_row_counts = row_counts[row_counts > 0]
    active_flat = int(np.count_nonzero(nonzero))
    active_rz = int(active_row_counts.size)
    active_er = int(np.count_nonzero(np.any(nonzero, axis=(2, 3))))
    active_r = int(np.count_nonzero(np.any(nonzero, axis=(0, 2, 3))))
    total_flat = int(hist.size)
    total_rz = int(hist.shape[0] * hist.shape[1] * hist.shape[2])
    total_er = int(hist.shape[0] * hist.shape[1])
    total_r = int(hist.shape[1])
    atoms_in_hist = float(np.sum(hist.real))
    if active_row_counts.size:
        row_percentiles = np.percentile(active_row_counts, [50, 90, 99])
        row_max = int(np.max(active_row_counts))
    else:
        row_percentiles = np.zeros(3)
        row_max = 0
    return {
        "active_flat_bins": active_flat,
        "total_flat_bins": total_flat,
        "active_flat_fraction": active_flat / total_flat if total_flat else 0.0,
        "active_rz_profiles": active_rz,
        "total_rz_profiles": total_rz,
        "active_rz_fraction": active_rz / total_rz if total_rz else 0.0,
        "active_er_profiles": active_er,
        "total_er_profiles": total_er,
        "active_er_fraction": active_er / total_er if total_er else 0.0,
        "active_r_count": active_r,
        "total_r_count": total_r,
        "active_r_fraction": active_r / total_r if total_r else 0.0,
        "mean_active_flat_bins_per_active_rz_profile": active_flat / active_rz
        if active_rz
        else 0.0,
        "median_active_flat_bins_per_active_rz_profile": float(row_percentiles[0]),
        "p90_active_flat_bins_per_active_rz_profile": float(row_percentiles[1]),
        "p99_active_flat_bins_per_active_rz_profile": float(row_percentiles[2]),
        "max_active_flat_bins_per_active_rz_profile": row_max,
        "mean_atoms_per_active_flat_bin": atoms_in_hist / active_flat
        if active_flat
        else 0.0,
        "mean_atoms_per_active_rz_profile": atoms_in_hist / active_rz
        if active_rz
        else 0.0,
        "mean_active_flat_bins_per_active_er_profile": active_flat / active_er
        if active_er
        else 0.0,
        "mean_atoms_per_active_er_profile": atoms_in_hist / active_er
        if active_er
        else 0.0,
    }


def run_case(
    n_atoms: int,
    *,
    args,
    seed: int,
) -> dict:
    grid = choose_physical_grid(
        n_atoms,
        bin_width_nm=args.bin_width_nm,
        qmax=args.qmax,
        q_unit=args.q_unit,
        n_phi_detector=args.nphi_detector,
        harmonic_margin=args.harmonic_margin,
        angular_rule=args.angular_rule,
    )
    row = {"atoms": n_atoms, "grid": grid_summary(grid)}
    print(
        f"{n_atoms}: side={grid.box_side_nm:.2f} nm "
        f"rmax={grid.r_max_nm:.2f} nm bins={grid.n_r}x{grid.n_z}x{grid.n_phi} "
        f"dr={grid.dr_nm:.3f} dz={grid.dz_nm:.3f} arc={grid.outer_arc_nm:.3f} nm "
        f"bins/elem={grid.n_bins_per_element:,}"
    )
    if args.dry_run:
        return row

    coords = synthetic_water_box(n_atoms, grid.box_side_nm, seed)
    q = np.linspace(args.qmin, args.qmax, args.nq)
    if args.q_unit == "inv_angstrom":
        # Coordinates are in nm. Keep q*r dimensionless.
        q_solver = 10.0 * q
    else:
        q_solver = q
    phi = (np.arange(grid.n_phi) + 0.5) * (2.0 * np.pi / grid.n_phi)

    hist_dtype = parse_hist_dtype(args.hist_dtype)
    complex_dtype = parse_complex_dtype(args.complex_dtype)

    binned, hist_s, hist_times = median_time(
        lambda: make_cylindrical_histogram(
            coords,
            n_r=grid.n_r,
            n_z=grid.n_z,
            n_phi=grid.n_phi,
            r_max=grid.r_max_nm,
            z_range=grid.z_range_nm,
            backend=args.hist_backend,
            hist_dtype=hist_dtype,
            angle_lut_size=args.angle_lut_size,
            angle_lut_mode=args.angle_lut_mode,
        ),
        args.repeats,
    )
    occupancy_start = time.perf_counter()
    occupancy = occupancy_summary(binned.hist)
    occupancy_s = time.perf_counter() - occupancy_start
    def make_plan() -> PreparedCakePlan:
        return PreparedCakePlan(
            binned,
            q_solver,
            args.wavelength_nm,
            q_block_size=args.q_block_size,
            circular_backend=args.circular_backend,
            complex_dtype=complex_dtype,
            harmonic_bandlimit_margin=args.harmonic_bandlimit_margin,
        )

    plan, plan_s, plan_times = median_time(make_plan, args.repeats)
    amp, solve_first_s, solve_first_memory = timed_call(
        plan.circular_fft,
        measure_memory=args.measure_memory,
        sample_interval_s=args.memory_sample_interval_s,
    )
    _, solve_cached_s, solve_times = median_time(plan.circular_fft, args.repeats)
    dense_curve = np.mean(intensity(amp), axis=1)

    curve_plan_s = None
    curve_plan_times = []
    curve_first_s = None
    curve_cached_s = None
    curve_cached_times = []
    curve_total_s = None
    curve_cached_total_s = None
    curve_rel_l2_vs_dense_map = None
    curve = None
    r_grouped_curve_plan_s = None
    r_grouped_curve_plan_times = []
    r_grouped_curve_first_s = None
    r_grouped_curve_cached_s = None
    r_grouped_curve_cached_times = []
    r_grouped_curve_total_s = None
    r_grouped_curve_cached_total_s = None
    r_grouped_curve_rel_l2_vs_dense_map = None
    r_grouped_curve = None
    r_dependent_curve_plan_s = None
    r_dependent_curve_plan_times = []
    r_dependent_curve_first_s = None
    r_dependent_curve_cached_s = None
    r_dependent_curve_cached_times = []
    r_dependent_curve_total_s = None
    r_dependent_curve_cached_total_s = None
    r_dependent_curve_rel_l2_vs_dense_map = None
    r_dependent_curve = None
    r_dependent_cake_first_s = None
    r_dependent_cake_cached_s = None
    r_dependent_cake_cached_times = []
    r_dependent_cake_total_s = None
    r_dependent_cake_cached_total_s = None
    r_dependent_cake_rel_l2_vs_dense = None
    r_dependent_cake_intensity_rel_l2_vs_dense = None
    r_dependent_cake_first_memory = None
    r_dependent_cake = None
    if args.benchmark_curve:
        curve_plan, curve_plan_s, curve_plan_times = median_time(make_plan, args.repeats)
        curve_start = time.perf_counter()
        curve = curve_plan.ring_average_intensity()
        curve_first_s = time.perf_counter() - curve_start
        curve_rel_l2_vs_dense_map = relative_l2(curve, dense_curve)
        _, curve_cached_s, curve_cached_times = median_time(
            curve_plan.ring_average_intensity,
            args.repeats,
        )
        curve_total_s = hist_s + curve_plan_s + curve_first_s
        curve_cached_total_s = hist_s + curve_plan_s + curve_cached_s
    if args.benchmark_r_grouped_curve:
        r_grouped_plan, r_grouped_curve_plan_s, r_grouped_curve_plan_times = median_time(
            make_plan,
            args.repeats,
        )
        r_grouped_start = time.perf_counter()
        r_grouped_curve = r_grouped_plan.ring_average_intensity_r_grouped()
        r_grouped_curve_first_s = time.perf_counter() - r_grouped_start
        r_grouped_curve_rel_l2_vs_dense_map = relative_l2(
            r_grouped_curve,
            dense_curve,
        )
        _, r_grouped_curve_cached_s, r_grouped_curve_cached_times = median_time(
            r_grouped_plan.ring_average_intensity_r_grouped,
            args.repeats,
        )
        r_grouped_curve_total_s = (
            hist_s + r_grouped_curve_plan_s + r_grouped_curve_first_s
        )
        r_grouped_curve_cached_total_s = (
            hist_s + r_grouped_curve_plan_s + r_grouped_curve_cached_s
        )
    if args.benchmark_r_dependent_curve:
        r_dep_plan, r_dependent_curve_plan_s, r_dependent_curve_plan_times = median_time(
            make_plan,
            args.repeats,
        )
        r_dep_start = time.perf_counter()
        r_dependent_curve = r_dep_plan.ring_average_intensity_r_dependent_bandlimit(
            margin=args.r_dependent_margin,
            cutoff_bin_size=args.r_dependent_cutoff_bin_size,
        )
        r_dependent_curve_first_s = time.perf_counter() - r_dep_start
        r_dependent_curve_rel_l2_vs_dense_map = relative_l2(
            r_dependent_curve,
            dense_curve,
        )
        _, r_dependent_curve_cached_s, r_dependent_curve_cached_times = median_time(
            lambda: r_dep_plan.ring_average_intensity_r_dependent_bandlimit(
                margin=args.r_dependent_margin,
                cutoff_bin_size=args.r_dependent_cutoff_bin_size,
            ),
            args.repeats,
        )
        r_dependent_curve_total_s = (
            hist_s + r_dependent_curve_plan_s + r_dependent_curve_first_s
        )
        r_dependent_curve_cached_total_s = (
            hist_s + r_dependent_curve_plan_s + r_dependent_curve_cached_s
        )

    if args.benchmark_r_dependent_cake:
        r_dependent_cake, r_dependent_cake_first_s, r_dependent_cake_first_memory = (
            timed_call(
                lambda: plan.circular_fft_r_dependent_bandlimit(
                    margin=args.r_dependent_margin,
                    cutoff_bin_size=args.r_dependent_cutoff_bin_size,
                    analytic_kernel=args.r_dependent_analytic_kernel,
                    analytic_kernel_table_dx=args.r_dependent_analytic_kernel_table_dx,
                    z_projection=args.r_dependent_z_projection,
                    r_block_size=args.r_dependent_r_block_size,
                    fused_analytic_kernel=args.r_dependent_fused_analytic_kernel,
                ),
                measure_memory=args.measure_memory,
                sample_interval_s=args.memory_sample_interval_s,
            )
        )
        r_dependent_cake_rel_l2_vs_dense = relative_l2(r_dependent_cake, amp)
        r_dependent_cake_intensity_rel_l2_vs_dense = relative_l2(
            intensity(r_dependent_cake),
            intensity(amp),
        )
        _, r_dependent_cake_cached_s, r_dependent_cake_cached_times = median_time(
            lambda: plan.circular_fft_r_dependent_bandlimit(
                margin=args.r_dependent_margin,
                cutoff_bin_size=args.r_dependent_cutoff_bin_size,
                analytic_kernel=args.r_dependent_analytic_kernel,
                analytic_kernel_table_dx=args.r_dependent_analytic_kernel_table_dx,
                z_projection=args.r_dependent_z_projection,
                r_block_size=args.r_dependent_r_block_size,
                fused_analytic_kernel=args.r_dependent_fused_analytic_kernel,
            ),
            args.repeats,
        )
        r_dependent_cake_total_s = hist_s + plan_s + r_dependent_cake_first_s
        r_dependent_cake_cached_total_s = hist_s + plan_s + r_dependent_cake_cached_s

    sparse_first_s = None
    sparse_cached_s = None
    sparse_cached_times = []
    sparse_rel_l2 = None
    if args.benchmark_sparse:
        sparse_start = time.perf_counter()
        sparse_amp = plan.circular_fft_sparse_rz(
            active_chunk_size=args.active_chunk_size,
        )
        sparse_first_s = time.perf_counter() - sparse_start
        sparse_rel_l2 = relative_l2(sparse_amp, amp)
        _, sparse_cached_s, sparse_cached_times = median_time(
            lambda: plan.circular_fft_sparse_rz(
                active_chunk_size=args.active_chunk_size,
            ),
            args.repeats,
        )

    sparse_profile_first_s = None
    sparse_profile_cached_s = None
    sparse_profile_cached_times = []
    sparse_profile_rel_l2 = None
    sparse_profile_skipped = None
    sparse_profile_work_estimate = (
        occupancy["active_flat_bins"] * grid.n_phi
        + occupancy["active_rz_profiles"] * grid.n_phi * args.nq
    )
    if args.benchmark_sparse_profiles:
        if sparse_profile_work_estimate > args.max_sparse_flat_work:
            sparse_profile_skipped = (
                f"work estimate {sparse_profile_work_estimate} exceeds "
                f"--max-sparse-flat-work {args.max_sparse_flat_work}"
            )
        else:
            sparse_start = time.perf_counter()
            sparse_profile_amp = plan.circular_fft_sparse_profiles()
            sparse_profile_first_s = time.perf_counter() - sparse_start
            sparse_profile_rel_l2 = relative_l2(sparse_profile_amp, amp)
            _, sparse_profile_cached_s, sparse_profile_cached_times = median_time(
                plan.circular_fft_sparse_profiles,
                args.repeats,
            )

    adaptive_profile_first_s = None
    adaptive_profile_cached_s = None
    adaptive_profile_cached_times = []
    adaptive_profile_rel_l2 = None
    adaptive_profile_skipped = None
    adaptive_profile_stats = None
    adaptive_profile_work_estimate = sparse_profile_work_estimate
    if args.benchmark_adaptive_profiles:
        if adaptive_profile_work_estimate > args.max_sparse_flat_work:
            adaptive_profile_skipped = (
                f"work estimate {adaptive_profile_work_estimate} exceeds "
                f"--max-sparse-flat-work {args.max_sparse_flat_work}"
            )
        else:
            adaptive_start = time.perf_counter()
            adaptive_profile_amp = plan.circular_fft_adaptive_profiles(
                dense_row_factor=args.adaptive_row_dense_factor,
                dense_batch_size=args.adaptive_dense_batch_size,
            )
            adaptive_profile_first_s = time.perf_counter() - adaptive_start
            adaptive_profile_rel_l2 = relative_l2(adaptive_profile_amp, amp)
            adaptive_profile_stats = plan.last_adaptive_profile_stats
            _, adaptive_profile_cached_s, adaptive_profile_cached_times = median_time(
                lambda: plan.circular_fft_adaptive_profiles(
                    dense_row_factor=args.adaptive_row_dense_factor,
                    dense_batch_size=args.adaptive_dense_batch_size,
                ),
                args.repeats,
            )

    sparse_source_projection_first_s = None
    sparse_source_projection_cached_s = None
    sparse_source_projection_cached_times = []
    sparse_source_projection_amp = None
    sparse_source_projection_rel_l2 = None
    sparse_source_projection_skipped = None
    sparse_source_projection_first_memory = None
    sparse_source_projection_work_estimate = (
        occupancy["active_flat_bins"] * args.nq
        + occupancy["active_er_profiles"]
        * args.nq
        * grid.n_phi
        * np.log2(max(grid.n_phi, 2))
    )
    if args.benchmark_sparse_source_projection:
        if sparse_source_projection_work_estimate > args.max_sparse_flat_work:
            sparse_source_projection_skipped = (
                f"work estimate {sparse_source_projection_work_estimate} exceeds "
                f"--max-sparse-flat-work {args.max_sparse_flat_work}"
            )
        else:
            (
                sparse_source_projection_amp,
                sparse_source_projection_first_s,
                sparse_source_projection_first_memory,
            ) = timed_call(
                lambda: plan.circular_fft_sparse_source_projection(
                    profile_chunk_size=args.source_profile_chunk_size,
                ),
                measure_memory=args.measure_memory,
                sample_interval_s=args.memory_sample_interval_s,
            )
            sparse_source_projection_rel_l2 = relative_l2(
                sparse_source_projection_amp,
                amp,
            )
            _, sparse_source_projection_cached_s, sparse_source_projection_cached_times = (
                median_time(
                    lambda: plan.circular_fft_sparse_source_projection(
                        profile_chunk_size=args.source_profile_chunk_size,
                    ),
                    args.repeats,
                )
            )

    sparse_source_r_dependent_first_s = None
    sparse_source_r_dependent_cached_s = None
    sparse_source_r_dependent_cached_times = []
    sparse_source_r_dependent_amp = None
    sparse_source_r_dependent_rel_l2 = None
    sparse_source_r_dependent_intensity_rel_l2 = None
    sparse_source_r_dependent_rel_l2_vs_r_dependent = None
    sparse_source_r_dependent_skipped = None
    sparse_source_r_dependent_first_memory = None
    sparse_source_r_dependent_work_estimate = sparse_source_projection_work_estimate
    if args.benchmark_sparse_source_r_dependent:
        if sparse_source_r_dependent_work_estimate > args.max_sparse_flat_work:
            sparse_source_r_dependent_skipped = (
                f"work estimate {sparse_source_r_dependent_work_estimate} exceeds "
                f"--max-sparse-flat-work {args.max_sparse_flat_work}"
            )
        else:
            (
                sparse_source_r_dependent_amp,
                sparse_source_r_dependent_first_s,
                sparse_source_r_dependent_first_memory,
            ) = timed_call(
                lambda: plan.circular_fft_sparse_source_r_dependent(
                    margin=args.r_dependent_margin,
                    cutoff_bin_size=args.r_dependent_cutoff_bin_size,
                    analytic_kernel=args.r_dependent_analytic_kernel,
                    analytic_kernel_table_dx=args.r_dependent_analytic_kernel_table_dx,
                    profile_chunk_size=args.source_profile_chunk_size,
                ),
                measure_memory=args.measure_memory,
                sample_interval_s=args.memory_sample_interval_s,
            )
            sparse_source_r_dependent_rel_l2 = relative_l2(
                sparse_source_r_dependent_amp,
                amp,
            )
            sparse_source_r_dependent_intensity_rel_l2 = relative_l2(
                intensity(sparse_source_r_dependent_amp),
                intensity(amp),
            )
            if r_dependent_cake is not None:
                sparse_source_r_dependent_rel_l2_vs_r_dependent = relative_l2(
                    sparse_source_r_dependent_amp,
                    r_dependent_cake,
                )
            _, sparse_source_r_dependent_cached_s, sparse_source_r_dependent_cached_times = (
                median_time(
                    lambda: plan.circular_fft_sparse_source_r_dependent(
                        margin=args.r_dependent_margin,
                        cutoff_bin_size=args.r_dependent_cutoff_bin_size,
                        analytic_kernel=args.r_dependent_analytic_kernel,
                        analytic_kernel_table_dx=args.r_dependent_analytic_kernel_table_dx,
                        profile_chunk_size=args.source_profile_chunk_size,
                    ),
                    args.repeats,
                )
            )

    sparse_flat_first_s = None
    sparse_flat_cached_s = None
    sparse_flat_cached_times = []
    sparse_flat_rel_l2 = None
    sparse_flat_skipped = None
    sparse_flat_work_estimate = occupancy["active_flat_bins"] * grid.n_phi * args.nq
    if args.benchmark_sparse_flat:
        if sparse_flat_work_estimate > args.max_sparse_flat_work:
            sparse_flat_skipped = (
                f"work estimate {sparse_flat_work_estimate} exceeds "
                f"--max-sparse-flat-work {args.max_sparse_flat_work}"
            )
        else:
            sparse_start = time.perf_counter()
            sparse_flat_amp = plan.circular_fft_sparse_flat(
                active_chunk_size=args.active_chunk_size,
            )
            sparse_flat_first_s = time.perf_counter() - sparse_start
            sparse_flat_rel_l2 = relative_l2(sparse_flat_amp, amp)
            _, sparse_flat_cached_s, sparse_flat_cached_times = median_time(
                lambda: plan.circular_fft_sparse_flat(
                    active_chunk_size=args.active_chunk_size,
                ),
                args.repeats,
            )

    nufft_s = None
    nufft_first_s = None
    nufft_times = []
    nufft_memory = None
    amp_err = None
    intensity_err = None
    r_dependent_cake_err_nufft = None
    r_dependent_cake_intensity_err_nufft = None
    sparse_source_r_dependent_err_nufft = None
    sparse_source_r_dependent_intensity_err_nufft = None
    curve_err_nufft = None
    if not args.skip_nufft:
        nufft_func = (
            (lambda: nufft_amplitude(coords, q_solver, args.wavelength_nm, phi))
            if args.nufft_q_block_size is None
            else (
                lambda: nufft_amplitude_chunked(
                    coords,
                    q_solver,
                    args.wavelength_nm,
                    phi,
                    q_block_size=args.nufft_q_block_size,
                )
            )
        )
        nufft_repeats = max(1, min(args.repeats, 3))
        if args.measure_memory:
            nufft_amp, nufft_first_s, nufft_memory = timed_call(
                nufft_func,
                measure_memory=True,
                sample_interval_s=args.memory_sample_interval_s,
            )
            nufft_times = [nufft_first_s]
            for _ in range(nufft_repeats - 1):
                _, elapsed, _ = timed_call(
                    nufft_func,
                    measure_memory=False,
                    sample_interval_s=args.memory_sample_interval_s,
                )
                nufft_times.append(elapsed)
            nufft_s = float(median(nufft_times))
        else:
            nufft_amp, nufft_s, nufft_times = median_time(
                nufft_func,
                nufft_repeats,
            )
            nufft_first_s = nufft_times[0] if nufft_times else None
        amp_err = relative_l2(amp, nufft_amp)
        intensity_err = relative_l2(intensity(amp), intensity(nufft_amp))
        if r_dependent_cake is not None:
            r_dependent_cake_err_nufft = relative_l2(r_dependent_cake, nufft_amp)
            r_dependent_cake_intensity_err_nufft = relative_l2(
                intensity(r_dependent_cake),
                intensity(nufft_amp),
            )
        if sparse_source_r_dependent_amp is not None:
            sparse_source_r_dependent_err_nufft = relative_l2(
                sparse_source_r_dependent_amp,
                nufft_amp,
            )
            sparse_source_r_dependent_intensity_err_nufft = relative_l2(
                intensity(sparse_source_r_dependent_amp),
                intensity(nufft_amp),
            )
        if curve is not None:
            curve_err_nufft = relative_l2(curve, np.mean(intensity(nufft_amp), axis=1))

    total_s = hist_s + plan_s + solve_first_s
    cached_total_s = hist_s + plan_s + solve_cached_s
    sparse_profile_total_s = (
        None
        if sparse_profile_first_s is None
        else hist_s + plan_s + sparse_profile_first_s
    )
    sparse_profile_cached_total_s = (
        None
        if sparse_profile_cached_s is None
        else hist_s + plan_s + sparse_profile_cached_s
    )
    adaptive_profile_total_s = (
        None
        if adaptive_profile_first_s is None
        else hist_s + plan_s + adaptive_profile_first_s
    )
    adaptive_profile_cached_total_s = (
        None
        if adaptive_profile_cached_s is None
        else hist_s + plan_s + adaptive_profile_cached_s
    )
    sparse_source_projection_total_s = (
        None
        if sparse_source_projection_first_s is None
        else hist_s + plan_s + sparse_source_projection_first_s
    )
    sparse_source_projection_cached_total_s = (
        None
        if sparse_source_projection_cached_s is None
        else hist_s + plan_s + sparse_source_projection_cached_s
    )
    sparse_source_r_dependent_total_s = (
        None
        if sparse_source_r_dependent_first_s is None
        else hist_s + plan_s + sparse_source_r_dependent_first_s
    )
    sparse_source_r_dependent_cached_total_s = (
        None
        if sparse_source_r_dependent_cached_s is None
        else hist_s + plan_s + sparse_source_r_dependent_cached_s
    )
    row.update(
        {
            "hist_s": hist_s,
            "occupancy_s": occupancy_s,
            **occupancy,
            "plan_s": plan_s,
            "solve_s": solve_first_s,
            "solve_first_s": solve_first_s,
            "solve_cached_s": solve_cached_s,
            **memory_fields("solve_first", solve_first_memory),
            "curve_plan_s": curve_plan_s,
            "curve_first_s": curve_first_s,
            "curve_cached_s": curve_cached_s,
            "curve_total_s": curve_total_s,
            "curve_cached_total_s": curve_cached_total_s,
            "curve_rel_l2_vs_dense_map": curve_rel_l2_vs_dense_map,
            "curve_rel_l2_vs_nufft_curve": curve_err_nufft,
            "r_grouped_curve_plan_s": r_grouped_curve_plan_s,
            "r_grouped_curve_first_s": r_grouped_curve_first_s,
            "r_grouped_curve_cached_s": r_grouped_curve_cached_s,
            "r_grouped_curve_total_s": r_grouped_curve_total_s,
            "r_grouped_curve_cached_total_s": r_grouped_curve_cached_total_s,
            "r_grouped_curve_rel_l2_vs_dense_map": (
                r_grouped_curve_rel_l2_vs_dense_map
            ),
            "r_dependent_curve_plan_s": r_dependent_curve_plan_s,
            "r_dependent_curve_first_s": r_dependent_curve_first_s,
            "r_dependent_curve_cached_s": r_dependent_curve_cached_s,
            "r_dependent_curve_total_s": r_dependent_curve_total_s,
            "r_dependent_curve_cached_total_s": r_dependent_curve_cached_total_s,
            "r_dependent_curve_rel_l2_vs_dense_map": (
                r_dependent_curve_rel_l2_vs_dense_map
            ),
            "r_dependent_cake_first_s": r_dependent_cake_first_s,
            "r_dependent_cake_cached_s": r_dependent_cake_cached_s,
            "r_dependent_cake_total_s": r_dependent_cake_total_s,
            "r_dependent_cake_cached_total_s": r_dependent_cake_cached_total_s,
            "r_dependent_cake_rel_l2_vs_dense": r_dependent_cake_rel_l2_vs_dense,
            "r_dependent_cake_intensity_rel_l2_vs_dense": (
                r_dependent_cake_intensity_rel_l2_vs_dense
            ),
            "r_dependent_cake_rel_l2_vs_nufft": r_dependent_cake_err_nufft,
            "r_dependent_cake_intensity_rel_l2_vs_nufft": (
                r_dependent_cake_intensity_err_nufft
            ),
            **memory_fields(
                "r_dependent_cake_first",
                r_dependent_cake_first_memory,
            ),
            "sparse_first_s": sparse_first_s,
            "sparse_cached_s": sparse_cached_s,
            "sparse_rel_l2_vs_dense": sparse_rel_l2,
            "sparse_profile_work_estimate": sparse_profile_work_estimate,
            "sparse_profile_first_s": sparse_profile_first_s,
            "sparse_profile_cached_s": sparse_profile_cached_s,
            "sparse_profile_rel_l2_vs_dense": sparse_profile_rel_l2,
            "sparse_profile_skipped": sparse_profile_skipped,
            "sparse_profile_total_s": sparse_profile_total_s,
            "sparse_profile_cached_total_s": sparse_profile_cached_total_s,
            "adaptive_profile_work_estimate": adaptive_profile_work_estimate,
            "adaptive_profile_first_s": adaptive_profile_first_s,
            "adaptive_profile_cached_s": adaptive_profile_cached_s,
            "adaptive_profile_rel_l2_vs_dense": adaptive_profile_rel_l2,
            "adaptive_profile_skipped": adaptive_profile_skipped,
            "adaptive_profile_total_s": adaptive_profile_total_s,
            "adaptive_profile_cached_total_s": adaptive_profile_cached_total_s,
            "adaptive_profile_stats": adaptive_profile_stats,
            "sparse_source_projection_work_estimate": (
                sparse_source_projection_work_estimate
            ),
            "sparse_source_projection_first_s": sparse_source_projection_first_s,
            "sparse_source_projection_cached_s": sparse_source_projection_cached_s,
            "sparse_source_projection_rel_l2_vs_dense": (
                sparse_source_projection_rel_l2
            ),
            "sparse_source_projection_skipped": sparse_source_projection_skipped,
            "sparse_source_projection_total_s": sparse_source_projection_total_s,
            "sparse_source_projection_cached_total_s": (
                sparse_source_projection_cached_total_s
            ),
            **memory_fields(
                "sparse_source_projection_first",
                sparse_source_projection_first_memory,
            ),
            "sparse_source_r_dependent_work_estimate": (
                sparse_source_r_dependent_work_estimate
            ),
            "sparse_source_r_dependent_first_s": sparse_source_r_dependent_first_s,
            "sparse_source_r_dependent_cached_s": sparse_source_r_dependent_cached_s,
            "sparse_source_r_dependent_rel_l2_vs_dense": (
                sparse_source_r_dependent_rel_l2
            ),
            "sparse_source_r_dependent_intensity_rel_l2_vs_dense": (
                sparse_source_r_dependent_intensity_rel_l2
            ),
            "sparse_source_r_dependent_rel_l2_vs_r_dependent": (
                sparse_source_r_dependent_rel_l2_vs_r_dependent
            ),
            "sparse_source_r_dependent_rel_l2_vs_nufft": (
                sparse_source_r_dependent_err_nufft
            ),
            "sparse_source_r_dependent_intensity_rel_l2_vs_nufft": (
                sparse_source_r_dependent_intensity_err_nufft
            ),
            "sparse_source_r_dependent_skipped": sparse_source_r_dependent_skipped,
            "sparse_source_r_dependent_total_s": sparse_source_r_dependent_total_s,
            "sparse_source_r_dependent_cached_total_s": (
                sparse_source_r_dependent_cached_total_s
            ),
            **memory_fields(
                "sparse_source_r_dependent_first",
                sparse_source_r_dependent_first_memory,
            ),
            "sparse_flat_work_estimate": sparse_flat_work_estimate,
            "sparse_flat_first_s": sparse_flat_first_s,
            "sparse_flat_cached_s": sparse_flat_cached_s,
            "sparse_flat_rel_l2_vs_dense": sparse_flat_rel_l2,
            "sparse_flat_skipped": sparse_flat_skipped,
            "total_s": total_s,
            "cached_total_s": cached_total_s,
            "nufft_s": nufft_s,
            "nufft_first_s": nufft_first_s,
            **memory_fields("nufft", nufft_memory),
            "speedup_vs_nufft": None if nufft_s is None else nufft_s / total_s,
            "solve_first_solver_speedup_vs_nufft": None
            if nufft_s is None
            else nufft_s / solve_first_s,
            "solve_cached_solver_speedup_vs_nufft": None
            if nufft_s is None
            else nufft_s / solve_cached_s,
            "sparse_profile_speedup_vs_nufft": None
            if nufft_s is None or sparse_profile_total_s is None
            else nufft_s / sparse_profile_total_s,
            "adaptive_profile_speedup_vs_nufft": None
            if nufft_s is None or adaptive_profile_total_s is None
            else nufft_s / adaptive_profile_total_s,
            "sparse_source_projection_speedup_vs_nufft": None
            if nufft_s is None or sparse_source_projection_total_s is None
            else nufft_s / sparse_source_projection_total_s,
            "sparse_source_r_dependent_speedup_vs_nufft": None
            if nufft_s is None or sparse_source_r_dependent_total_s is None
            else nufft_s / sparse_source_r_dependent_total_s,
            "curve_speedup_vs_nufft": None
            if nufft_s is None or curve_total_s is None
            else nufft_s / curve_total_s,
            "r_grouped_curve_speedup_vs_nufft": None
            if nufft_s is None or r_grouped_curve_total_s is None
            else nufft_s / r_grouped_curve_total_s,
            "r_dependent_curve_speedup_vs_nufft": None
            if nufft_s is None or r_dependent_curve_total_s is None
            else nufft_s / r_dependent_curve_total_s,
            "r_dependent_cake_speedup_vs_nufft": None
            if nufft_s is None or r_dependent_cake_total_s is None
            else nufft_s / r_dependent_cake_total_s,
            "r_dependent_cake_first_solver_speedup_vs_nufft": None
            if nufft_s is None or r_dependent_cake_first_s is None
            else nufft_s / r_dependent_cake_first_s,
            "r_dependent_cake_cached_solver_speedup_vs_nufft": None
            if nufft_s is None or r_dependent_cake_cached_s is None
            else nufft_s / r_dependent_cake_cached_s,
            "sparse_source_r_dependent_first_solver_speedup_vs_nufft": None
            if nufft_s is None or sparse_source_r_dependent_first_s is None
            else nufft_s / sparse_source_r_dependent_first_s,
            "sparse_source_r_dependent_cached_solver_speedup_vs_nufft": None
            if nufft_s is None or sparse_source_r_dependent_cached_s is None
            else nufft_s / sparse_source_r_dependent_cached_s,
            "amp_rel_l2_vs_nufft": amp_err,
            "intensity_rel_l2_vs_nufft": intensity_err,
            "hist_times": hist_times,
            "plan_times": plan_times,
            "solve_times": solve_times,
            "curve_plan_times": curve_plan_times,
            "curve_cached_times": curve_cached_times,
            "r_grouped_curve_plan_times": r_grouped_curve_plan_times,
            "r_grouped_curve_cached_times": r_grouped_curve_cached_times,
            "r_dependent_curve_plan_times": r_dependent_curve_plan_times,
            "r_dependent_curve_cached_times": r_dependent_curve_cached_times,
            "r_dependent_cake_cached_times": r_dependent_cake_cached_times,
            "sparse_profile_cached_times": sparse_profile_cached_times,
            "adaptive_profile_cached_times": adaptive_profile_cached_times,
            "sparse_source_projection_cached_times": (
                sparse_source_projection_cached_times
            ),
            "sparse_source_r_dependent_cached_times": (
                sparse_source_r_dependent_cached_times
            ),
            "sparse_cached_times": sparse_cached_times,
            "sparse_flat_cached_times": sparse_flat_cached_times,
            "nufft_times": nufft_times,
        }
    )
    print(
        f"  total={total_s:.4f}s cached_total={cached_total_s:.4f}s "
        f"hist={hist_s:.4f}s plan={plan_s:.4f}s "
        f"solve_first={solve_first_s:.4f}s solve_cached={solve_cached_s:.4f}s "
        f"solve_peak_delta={peak_delta_text(solve_first_memory)} "
        f"nufft={nufft_s} nufft_peak_delta={peak_delta_text(nufft_memory)}"
    )
    print(
        "  occupancy: "
        f"flat={occupancy['active_flat_bins']:,}/{occupancy['total_flat_bins']:,} "
        f"({occupancy['active_flat_fraction']:.2%}), "
        f"rz={occupancy['active_rz_profiles']:,}/{occupancy['total_rz_profiles']:,} "
        f"({occupancy['active_rz_fraction']:.2%}), "
        f"er={occupancy['active_er_profiles']:,}/{occupancy['total_er_profiles']:,} "
        f"({occupancy['active_er_fraction']:.2%}), "
        f"active_R={occupancy['active_r_count']:,}/{occupancy['total_r_count']:,}, "
        f"flat/rz mean={occupancy['mean_active_flat_bins_per_active_rz_profile']:.1f} "
        f"flat/er mean={occupancy['mean_active_flat_bins_per_active_er_profile']:.1f} "
        f"p50={occupancy['median_active_flat_bins_per_active_rz_profile']:.0f} "
        f"p90={occupancy['p90_active_flat_bins_per_active_rz_profile']:.0f} "
        f"p99={occupancy['p99_active_flat_bins_per_active_rz_profile']:.0f}"
    )
    if args.benchmark_sparse:
        print(
            "  sparse_rz: "
            f"first={sparse_first_s:.4f}s cached={sparse_cached_s:.4f}s "
            f"err={sparse_rel_l2:.3g}"
        )
    if args.benchmark_curve:
        print(
            "  curve_1d: "
            f"total={curve_total_s:.4f}s "
            f"first={curve_first_s:.4f}s "
            f"cached={curve_cached_s:.4f}s "
            f"err_map={curve_rel_l2_vs_dense_map:.3g}"
        )
    if args.benchmark_r_grouped_curve:
        print(
            "  curve_1d_r_grouped: "
            f"total={r_grouped_curve_total_s:.4f}s "
            f"first={r_grouped_curve_first_s:.4f}s "
            f"cached={r_grouped_curve_cached_s:.4f}s "
            f"err_map={r_grouped_curve_rel_l2_vs_dense_map:.3g}"
        )
    if args.benchmark_r_dependent_curve:
        print(
            "  curve_1d_r_dependent: "
            f"total={r_dependent_curve_total_s:.4f}s "
            f"first={r_dependent_curve_first_s:.4f}s "
            f"cached={r_dependent_curve_cached_s:.4f}s "
            f"err_map={r_dependent_curve_rel_l2_vs_dense_map:.3g} "
            f"margin={args.r_dependent_margin} "
            f"bin={args.r_dependent_cutoff_bin_size}"
        )
    if args.benchmark_r_dependent_cake:
        print(
            "  cake_2d_r_dependent: "
            f"total={r_dependent_cake_total_s:.4f}s "
            f"first={r_dependent_cake_first_s:.4f}s "
            f"cached={r_dependent_cake_cached_s:.4f}s "
                f"peak_delta={peak_delta_text(r_dependent_cake_first_memory)} "
                f"amp_err={r_dependent_cake_rel_l2_vs_dense:.3g} "
                f"I_err={r_dependent_cake_intensity_rel_l2_vs_dense:.3g} "
                f"I_err_nufft={r_dependent_cake_intensity_err_nufft} "
                f"margin={args.r_dependent_margin} "
            f"bin={args.r_dependent_cutoff_bin_size} "
            f"analytic_kernel={args.r_dependent_analytic_kernel} "
            f"table_dx={args.r_dependent_analytic_kernel_table_dx} "
            f"z_projection={args.r_dependent_z_projection} "
            f"r_block={args.r_dependent_r_block_size} "
            f"fused_analytic={args.r_dependent_fused_analytic_kernel}"
        )
    if args.benchmark_sparse_profiles:
        if sparse_profile_skipped:
            print(f"  sparse_profiles: skipped ({sparse_profile_skipped})")
        else:
            print(
                "  sparse_profiles: "
                f"first={sparse_profile_first_s:.4f}s "
                f"cached={sparse_profile_cached_s:.4f}s "
                f"total={sparse_profile_total_s:.4f}s "
                f"err={sparse_profile_rel_l2:.3g}"
            )
    if args.benchmark_adaptive_profiles:
        if adaptive_profile_skipped:
            print(f"  adaptive_profiles: skipped ({adaptive_profile_skipped})")
        else:
            dense_fraction = (
                adaptive_profile_stats or {}
            ).get("mean_dense_profile_fraction")
            print(
                "  adaptive_profiles: "
                f"first={adaptive_profile_first_s:.4f}s "
                f"cached={adaptive_profile_cached_s:.4f}s "
                f"total={adaptive_profile_total_s:.4f}s "
                f"dense_rows={dense_fraction:.2%} "
                f"err={adaptive_profile_rel_l2:.3g}"
            )
    if args.benchmark_sparse_source_projection:
        if sparse_source_projection_skipped:
            print(
                "  sparse_source_projection: "
                f"skipped ({sparse_source_projection_skipped})"
            )
        else:
            print(
                "  sparse_source_projection: "
                f"first={sparse_source_projection_first_s:.4f}s "
                f"cached={sparse_source_projection_cached_s:.4f}s "
                f"total={sparse_source_projection_total_s:.4f}s "
                f"peak_delta={peak_delta_text(sparse_source_projection_first_memory)} "
                f"err={sparse_source_projection_rel_l2:.3g} "
                f"profile_chunk={args.source_profile_chunk_size}"
            )
    if args.benchmark_sparse_source_r_dependent:
        if sparse_source_r_dependent_skipped:
            print(
                "  sparse_source_r_dependent: "
                f"skipped ({sparse_source_r_dependent_skipped})"
            )
        else:
            print(
                "  sparse_source_r_dependent: "
                f"first={sparse_source_r_dependent_first_s:.4f}s "
                f"cached={sparse_source_r_dependent_cached_s:.4f}s "
                f"total={sparse_source_r_dependent_total_s:.4f}s "
                f"peak_delta={peak_delta_text(sparse_source_r_dependent_first_memory)} "
                f"amp_err={sparse_source_r_dependent_rel_l2:.3g} "
                f"I_err={sparse_source_r_dependent_intensity_rel_l2:.3g} "
                f"rdep_err={sparse_source_r_dependent_rel_l2_vs_r_dependent} "
                f"profile_chunk={args.source_profile_chunk_size} "
                f"analytic_kernel={args.r_dependent_analytic_kernel} "
                f"table_dx={args.r_dependent_analytic_kernel_table_dx}"
            )
    if args.benchmark_sparse_flat:
        if sparse_flat_skipped:
            print(f"  sparse_flat: skipped ({sparse_flat_skipped})")
        else:
            print(
                "  sparse_flat: "
                f"first={sparse_flat_first_s:.4f}s "
                f"cached={sparse_flat_cached_s:.4f}s "
                f"err={sparse_flat_rel_l2:.3g}"
            )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms", nargs="+", type=int, default=[10_000, 100_000, 1_000_000])
    parser.add_argument("--bin-width-nm", type=float, default=0.1)
    parser.add_argument("--qmin", type=float, default=0.05)
    parser.add_argument("--qmax", type=float, default=2.2)
    parser.add_argument("--q-unit", choices=["inv_angstrom", "inv_nm"], default="inv_angstrom")
    parser.add_argument("--nq", type=int, default=40)
    parser.add_argument("--nphi-detector", type=int, default=180)
    parser.add_argument("--harmonic-margin", type=int, default=16)
    parser.add_argument("--angular-rule", choices=["bandlimit", "arc"], default="bandlimit")
    parser.add_argument("--wavelength-nm", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--measure-memory", action="store_true")
    parser.add_argument("--memory-sample-interval-s", type=float, default=0.002)
    parser.add_argument("--benchmark-curve", action="store_true")
    parser.add_argument("--benchmark-r-grouped-curve", action="store_true")
    parser.add_argument("--benchmark-r-dependent-curve", action="store_true")
    parser.add_argument("--benchmark-r-dependent-cake", action="store_true")
    parser.add_argument("--r-dependent-margin", type=int, default=16)
    parser.add_argument("--r-dependent-cutoff-bin-size", type=int, default=16)
    parser.add_argument("--r-dependent-analytic-kernel", action="store_true")
    parser.add_argument("--r-dependent-analytic-kernel-table-dx", type=float, default=None)
    parser.add_argument("--r-dependent-z-projection", action="store_true")
    parser.add_argument("--r-dependent-r-block-size", type=int, default=None)
    parser.add_argument("--r-dependent-fused-analytic-kernel", action="store_true")
    parser.add_argument("--benchmark-sparse", action="store_true")
    parser.add_argument("--benchmark-sparse-profiles", action="store_true")
    parser.add_argument("--benchmark-adaptive-profiles", action="store_true")
    parser.add_argument("--benchmark-sparse-source-projection", action="store_true")
    parser.add_argument("--benchmark-sparse-source-r-dependent", action="store_true")
    parser.add_argument("--benchmark-sparse-flat", action="store_true")
    parser.add_argument("--active-chunk-size", type=int, default=256)
    parser.add_argument("--adaptive-row-dense-factor", type=float, default=1.0)
    parser.add_argument("--adaptive-dense-batch-size", type=int, default=2048)
    parser.add_argument("--source-profile-chunk-size", type=int, default=64)
    parser.add_argument(
        "--max-sparse-flat-work",
        type=float,
        default=5.0e9,
        help=(
            "Skip sparse-flat/profile timing when the sparse work estimate exceeds "
            "this threshold."
        ),
    )
    parser.add_argument(
        "--fast-preset",
        choices=FAST_PRESET_NAMES,
        default="production",
        help="Apply a fast-path option macro before physical grid timing.",
    )
    parser.add_argument(
        "--hist-backend",
        choices=["numpy", "numba", "numba-parallel", "cpp"],
        default="cpp",
    )
    parser.add_argument(
        "--hist-dtype",
        choices=["default", "int64", "uint32", "float32", "float64"],
        default="float32",
    )
    parser.add_argument("--angle-lut-size", type=int, default=32)
    parser.add_argument("--angle-lut-mode", choices=["nearest", "cubic"], default="cubic")
    parser.add_argument("--circular-backend", choices=["auto", "numpy", "cpp"], default="cpp")
    parser.add_argument("--complex-dtype", choices=["auto", "complex64", "complex128"], default="auto")
    parser.add_argument("--q-block-size", type=int, default=128)
    parser.add_argument("--harmonic-bandlimit-margin", type=int, default=None)
    parser.add_argument("--skip-nufft", action="store_true")
    parser.add_argument(
        "--nufft-q-block-size",
        type=int,
        default=None,
        help="Evaluate FINUFFT in q-blocks to reduce memory use.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark_results/physical_scaling.json"),
    )
    args = parser.parse_args()
    if args.memory_sample_interval_s <= 0:
        raise ValueError("--memory-sample-interval-s must be positive")
    apply_fast_preset(args)

    rows = []
    for i, n_atoms in enumerate(args.atoms):
        rows.append(run_case(n_atoms, args=args, seed=args.seed + i))

    result = {
        "case": {
            "atoms": args.atoms,
            "bin_width_nm": args.bin_width_nm,
            "qmin": args.qmin,
            "qmax": args.qmax,
            "q_unit": args.q_unit,
            "nq": args.nq,
            "nphi_detector": args.nphi_detector,
            "note": "nphi_detector is a lower bound; timed output uses the physical grid n_phi.",
            "harmonic_margin": args.harmonic_margin,
            "angular_rule": args.angular_rule,
            "wavelength_nm": args.wavelength_nm,
            "measure_memory": args.measure_memory,
            "memory_sample_interval_s": args.memory_sample_interval_s,
            "hist_backend": args.hist_backend,
            "hist_dtype": args.hist_dtype,
            "angle_lut_size": args.angle_lut_size,
            "angle_lut_mode": args.angle_lut_mode,
            "circular_backend": args.circular_backend,
            "complex_dtype": args.complex_dtype,
            "harmonic_bandlimit_margin": args.harmonic_bandlimit_margin,
            "nufft_q_block_size": args.nufft_q_block_size,
            "dry_run": args.dry_run,
            "benchmark_curve": args.benchmark_curve,
            "benchmark_r_grouped_curve": args.benchmark_r_grouped_curve,
            "benchmark_r_dependent_curve": args.benchmark_r_dependent_curve,
            "benchmark_r_dependent_cake": args.benchmark_r_dependent_cake,
            "r_dependent_margin": args.r_dependent_margin,
            "r_dependent_cutoff_bin_size": args.r_dependent_cutoff_bin_size,
            "r_dependent_analytic_kernel": args.r_dependent_analytic_kernel,
            "r_dependent_analytic_kernel_table_dx": (
                args.r_dependent_analytic_kernel_table_dx
            ),
            "r_dependent_z_projection": args.r_dependent_z_projection,
            "r_dependent_r_block_size": args.r_dependent_r_block_size,
            "r_dependent_fused_analytic_kernel": (
                args.r_dependent_fused_analytic_kernel
            ),
            "benchmark_sparse": args.benchmark_sparse,
            "benchmark_sparse_profiles": args.benchmark_sparse_profiles,
            "benchmark_adaptive_profiles": args.benchmark_adaptive_profiles,
            "benchmark_sparse_source_projection": (
                args.benchmark_sparse_source_projection
            ),
            "benchmark_sparse_source_r_dependent": (
                args.benchmark_sparse_source_r_dependent
            ),
            "benchmark_sparse_flat": args.benchmark_sparse_flat,
            "active_chunk_size": args.active_chunk_size,
            "adaptive_row_dense_factor": args.adaptive_row_dense_factor,
            "adaptive_dense_batch_size": args.adaptive_dense_batch_size,
            "source_profile_chunk_size": args.source_profile_chunk_size,
            "max_sparse_flat_work": args.max_sparse_flat_work,
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
