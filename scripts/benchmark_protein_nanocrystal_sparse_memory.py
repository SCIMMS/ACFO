from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_physical_scaling import (  # noqa: E402
    bytes_to_mib,
    current_rss_bytes,
    memory_fields,
    timed_call,
)
from validate_public_waxs_crystals import choose_grid  # noqa: E402
from validate_public_waxs_structures import build_form_factors, load_structure  # noqa: E402
from waxs_cake import (  # noqa: E402
    PreparedCakePlan,
    cylindrical_flat_indices,
    make_sparse_cylindrical_histogram_from_flat_indices,
)
from waxs_cake.metrics import relative_l2  # noqa: E402
from waxs_cake.physical_scaling import q_to_inv_nm  # noqa: E402


def rss_delta_mib(after: int | None, before: int | None) -> float | None:
    if after is None or before is None:
        return None
    return bytes_to_mib(after - before)


def build_sparse_structure(
    coords: np.ndarray,
    elements: np.ndarray,
    grid: dict,
    *,
    index_backend: str,
):
    element_order, element_indices = np.unique(elements, return_inverse=True)
    flat = cylindrical_flat_indices(
        coords,
        element_indices=element_indices,
        n_elements=element_order.size,
        backend=index_backend,
        n_r=grid["n_r"],
        n_z=grid["n_z"],
        n_phi=grid["n_phi"],
        r_max=grid["r_max_nm"],
        z_range=grid["z_range_nm"],
    )
    return make_sparse_cylindrical_histogram_from_flat_indices(
        flat,
        n_elements=element_order.size,
        n_r=grid["n_r"],
        n_z=grid["n_z"],
        n_phi=grid["n_phi"],
        r_max=grid["r_max_nm"],
        z_range=grid["z_range_nm"],
        element_order=element_order,
        value_dtype=np.float32,
    )


def benchmark(args: argparse.Namespace) -> dict:
    load_start = time.perf_counter()
    coords, elements, metadata = load_structure(args.structure)
    load_s = time.perf_counter() - load_start
    gc.collect()
    rss_loaded = current_rss_bytes()

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

    sparse, build_s, build_memory = timed_call(
        lambda: build_sparse_structure(
            coords,
            elements,
            grid,
            index_backend=args.index_backend,
        ),
        measure_memory=True,
        sample_interval_s=args.memory_sample_interval_s,
    )
    gc.collect()
    rss_sparse_steady = current_rss_bytes()

    plan, plan_s, plan_memory = timed_call(
        lambda: PreparedCakePlan(
            sparse,
            q_solver,
            args.wavelength_nm,
            form_factors=form_factors,
            circular_backend=args.circular_backend,
            complex_dtype=np.complex64,
            q_block_size=args.q_block_size,
        ),
        measure_memory=True,
        sample_interval_s=args.memory_sample_interval_s,
    )
    gc.collect()
    rss_plan_steady = current_rss_bytes()

    solve = lambda: plan.circular_fft_sparse_source_r_dependent(
        margin=args.r_dependent_margin,
        cutoff_bin_size=args.cutoff_bin_size,
        analytic_kernel=True,
        q_block_size=args.q_block_size,
        profile_chunk_size=args.profile_chunk_size,
    )
    first_amp, first_s, first_memory = timed_call(
        solve,
        measure_memory=True,
        sample_interval_s=args.memory_sample_interval_s,
    )
    first_norm = float(np.linalg.norm(first_amp))
    first_intensity_sum = float(np.sum(np.abs(first_amp) ** 2, dtype=np.float64))
    del first_amp
    gc.collect()
    rss_prepared_steady = current_rss_bytes()

    cached_amp, cached_memory_s, cached_memory = timed_call(
        solve,
        measure_memory=True,
        sample_interval_s=args.memory_sample_interval_s,
    )
    cached_norm = float(np.linalg.norm(cached_amp))
    del cached_amp
    gc.collect()

    cached_times: list[float] = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        amp = solve()
        cached_times.append(time.perf_counter() - start)
        del amp
        gc.collect()

    dense_values = len(sparse.elements) * grid["n_r"] * grid["n_z"] * grid["n_phi"]
    return {
        "structure_path": args.structure.as_posix(),
        "structure_id": metadata.get("structure_id", args.structure.stem),
        "atoms": int(coords.shape[0]),
        "elements": list(sparse.elements),
        "qmin": args.qmin,
        "qmax": args.qmax,
        "q_unit": args.q_unit,
        "nq": args.nq,
        "wavelength_nm": args.wavelength_nm,
        "bin_width_nm": args.bin_width_nm,
        "n_r": grid["n_r"],
        "n_z": grid["n_z"],
        "n_phi": grid["n_phi"],
        "active_flat_bins": int(sparse.active_values.size),
        "active_flat_fraction": float(sparse.active_values.size / dense_values),
        "sparse_structure_storage_mib": sparse.sparse_storage_nbytes / 1024**2,
        "dense_hist_float32_gib": dense_values * 4 / 1024**3,
        "load_s": load_s,
        "rss_loaded_mib": bytes_to_mib(rss_loaded),
        "representation_build_s": build_s,
        **memory_fields("representation_build", build_memory),
        "representation_steady_rss_mib": bytes_to_mib(rss_sparse_steady),
        "representation_steady_delta_from_loaded_mib": rss_delta_mib(
            rss_sparse_steady,
            rss_loaded,
        ),
        "plan_build_s": plan_s,
        **memory_fields("plan_build", plan_memory),
        "plan_steady_rss_mib": bytes_to_mib(rss_plan_steady),
        "plan_steady_delta_from_representation_mib": rss_delta_mib(
            rss_plan_steady,
            rss_sparse_steady,
        ),
        "first_solve_s": first_s,
        **memory_fields("first_solve", first_memory),
        "prepared_steady_rss_mib": bytes_to_mib(rss_prepared_steady),
        "prepared_steady_delta_from_plan_mib": rss_delta_mib(
            rss_prepared_steady,
            rss_plan_steady,
        ),
        "cached_memory_s": cached_memory_s,
        **memory_fields("cached_solve", cached_memory),
        "cached_median_s": float(median(cached_times)),
        "cached_times": cached_times,
        "first_amplitude_norm": first_norm,
        "cached_amplitude_norm": cached_norm,
        "first_cached_norm_rel_difference": abs(first_norm - cached_norm)
        / max(first_norm, 1e-300),
        "first_intensity_sum": first_intensity_sum,
        "all_finite": bool(np.isfinite(first_norm) and np.isfinite(first_intensity_sum)),
        "first_total_from_loaded_s": build_s + plan_s + first_s,
        "index_backend": args.index_backend,
        "circular_backend": args.circular_backend,
        "form_factor_model": args.form_factor_model,
        "q_block_size": args.q_block_size,
        "profile_chunk_size": args.profile_chunk_size,
        "r_dependent_margin": args.r_dependent_margin,
        "cutoff_bin_size": args.cutoff_bin_size,
        "memory_sample_interval_s": args.memory_sample_interval_s,
        "python": sys.version,
        "platform": platform.platform(),
    }


def write_markdown(row: dict, path: Path) -> None:
    lines = [
        f"# Sparse protein-nanocrystal memory benchmark: {row['structure_id']}",
        "",
        f"- atoms: `{row['atoms']:,}`",
        f"- grid: `{row['n_r']} x {row['n_z']} x {row['n_phi']}` at qmax `{row['qmax']} {row['q_unit']}`",
        f"- active flat bins: `{row['active_flat_bins']:,}` (`{row['active_flat_fraction']:.5%}`)",
        f"- sparse structure arrays: `{row['sparse_structure_storage_mib']:.2f} MiB`",
        f"- dense float32 histogram equivalent: `{row['dense_hist_float32_gib']:.2f} GiB`",
        "",
        "| phase | time s | peak RSS delta MiB | retained/steady delta MiB |",
        "|---|---:|---:|---:|",
        f"| representation build | {row['representation_build_s']:.4f} | {row['representation_build_peak_rss_delta_mib']:.2f} | {row['representation_steady_delta_from_loaded_mib']:.2f} |",
        f"| plan build | {row['plan_build_s']:.4f} | {row['plan_build_peak_rss_delta_mib']:.2f} | {row['plan_steady_delta_from_representation_mib']:.2f} |",
        f"| first sparse solve | {row['first_solve_s']:.4f} | {row['first_solve_peak_rss_delta_mib']:.2f} | {row['prepared_steady_delta_from_plan_mib']:.2f} |",
        f"| cached sparse solve | {row['cached_median_s']:.4f} | {row['cached_solve_peak_rss_delta_mib']:.2f} | n/a |",
        "",
        "The sparse-array byte count is object storage only. RSS deltas include allocator and solver-cache effects and are the values to use for the memory gate.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure true sparse representation and sparse-source WAXS memory on a protein nanocrystal."
    )
    parser.add_argument("structure", type=Path)
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
    parser.add_argument("--q-block-size", type=int, default=2)
    parser.add_argument("--profile-chunk-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--form-factor-model", default="xray_f0")
    parser.add_argument("--index-backend", choices=["numpy", "cpp"], default="cpp")
    parser.add_argument("--circular-backend", choices=["numpy", "cpp"], default="cpp")
    parser.add_argument("--memory-sample-interval-s", type=float, default=0.002)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/protein_nanocrystal_sparse_memory.json"),
    )
    args = parser.parse_args()
    if args.nq <= 0 or args.q_block_size <= 0 or args.profile_chunk_size <= 0:
        raise ValueError("nq, q-block-size, and profile-chunk-size must be positive")

    row = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    write_markdown(row, args.output.with_suffix(".md"))
    print(json.dumps(row, indent=2))
    print(f"wrote {args.output} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
