from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_physical_scaling import (  # noqa: E402
    grid_summary,
    occupancy_summary,
    parse_complex_dtype,
    parse_hist_dtype,
    synthetic_water_box,
)
from waxs_cake import (  # noqa: E402
    PreparedCakePlan,
    choose_physical_grid,
    make_cylindrical_histogram,
)
from waxs_cake.metrics import relative_l2  # noqa: E402
from waxs_cake.presets import FAST_PRESET_NAMES, apply_fast_preset  # noqa: E402


def timed_once(func):
    gc.collect()
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def median_time(func, repeats: int):
    value = None
    times = []
    for _ in range(repeats):
        value, elapsed = timed_once(func)
        times.append(elapsed)
    return value, float(median(times)), times


def run_case(n_atoms: int, *, args, seed: int) -> dict:
    grid = choose_physical_grid(
        n_atoms,
        bin_width_nm=args.bin_width_nm,
        qmax=args.qmax,
        q_unit=args.q_unit,
        n_phi_detector=args.nphi_detector,
        harmonic_margin=args.harmonic_margin,
        angular_rule=args.angular_rule,
    )
    print(
        f"{n_atoms}: side={grid.box_side_nm:.2f} nm "
        f"bins={grid.n_r}x{grid.n_z}x{grid.n_phi} "
        f"bins/elem={grid.n_bins_per_element:,}"
    )

    coords = synthetic_water_box(n_atoms, grid.box_side_nm, seed)
    q = np.linspace(args.qmin, args.qmax, args.nq)
    q_solver = 10.0 * q if args.q_unit == "inv_angstrom" else q

    hist_dtype = parse_hist_dtype(args.hist_dtype)
    complex_dtype = parse_complex_dtype(args.complex_dtype)
    binned, hist_s = timed_once(
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
        )
    )
    occupancy = occupancy_summary(binned.hist)

    plan, plan_s = timed_once(
        lambda: PreparedCakePlan(
            binned,
            q_solver,
            args.wavelength_nm,
            q_block_size=args.q_block_size,
            circular_backend=args.circular_backend,
            harmonic_bandlimit_margin=args.harmonic_bandlimit_margin,
            complex_dtype=complex_dtype,
            cache_kernel_fft=args.cache_kernels,
        )
    )

    dense_amp, dense_first_s = timed_once(
        lambda: plan.circular_fft(q_block_size=args.q_block_size)
    )
    _, dense_cached_s, dense_cached_times = median_time(
        lambda: plan.circular_fft(q_block_size=args.q_block_size),
        args.repeats,
    )

    _, source_profile_cache_s = timed_once(lambda: plan.active_er_profile_count)
    rows = []
    for chunk in args.chunks:
        sparse_amp, first_s = timed_once(
            lambda chunk=chunk: plan.circular_fft_sparse_source_projection(
                q_block_size=args.q_block_size,
                profile_chunk_size=chunk,
            )
        )
        rel_l2 = relative_l2(sparse_amp, dense_amp)
        _, cached_s, cached_times = median_time(
            lambda chunk=chunk: plan.circular_fft_sparse_source_projection(
                q_block_size=args.q_block_size,
                profile_chunk_size=chunk,
            ),
            args.repeats,
        )
        row = {
            "chunk": int(chunk),
            "first_s": first_s,
            "cached_s": cached_s,
            "cached_times": cached_times,
            "rel_l2_vs_dense": rel_l2,
            "solve_speedup_first": dense_first_s / first_s if first_s else float("inf"),
            "solve_speedup_cached": (
                dense_cached_s / cached_s if cached_s else float("inf")
            ),
            "one_shot_total_s": hist_s + plan_s + source_profile_cache_s + first_s,
            "one_shot_speedup_vs_dense_total": (
                (hist_s + plan_s + dense_first_s)
                / (hist_s + plan_s + source_profile_cache_s + first_s)
            ),
        }
        rows.append(row)
        print(
            f"  chunk={chunk:<4d} first={first_s:.4f}s cached={cached_s:.4f}s "
            f"x_first={row['solve_speedup_first']:.2f} "
            f"x_cached={row['solve_speedup_cached']:.2f} "
            f"err={rel_l2:.3g}"
        )

    best_first = min(rows, key=lambda r: r["first_s"])
    best_cached = min(rows, key=lambda r: r["cached_s"])
    print(
        f"  dense first={dense_first_s:.4f}s cached={dense_cached_s:.4f}s; "
        f"best_first={best_first['chunk']} best_cached={best_cached['chunk']}"
    )

    return {
        "grid": grid_summary(grid),
        "hist_s": hist_s,
        "plan_s": plan_s,
        "dense_first_s": dense_first_s,
        "dense_cached_s": dense_cached_s,
        "dense_cached_times": dense_cached_times,
        "source_profile_cache_s": source_profile_cache_s,
        **occupancy,
        "chunks": rows,
        "best_first_chunk": int(best_first["chunk"]),
        "best_cached_chunk": int(best_cached["chunk"]),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    flat_rows = []
    for case in rows:
        grid = case["grid"]
        for chunk in case["chunks"]:
            flat_rows.append(
                {
                    "atoms": grid["n_atoms"],
                    "n_r": grid["n_r"],
                    "n_z": grid["n_z"],
                    "n_phi": grid["n_phi"],
                    "active_flat_fraction": case["active_flat_fraction"],
                    "active_rz_fraction": case["active_rz_fraction"],
                    "active_er_fraction": case["active_er_fraction"],
                    "chunk": chunk["chunk"],
                    "dense_first_s": case["dense_first_s"],
                    "dense_cached_s": case["dense_cached_s"],
                    "sparse_first_s": chunk["first_s"],
                    "sparse_cached_s": chunk["cached_s"],
                    "solve_speedup_first": chunk["solve_speedup_first"],
                    "solve_speedup_cached": chunk["solve_speedup_cached"],
                    "rel_l2_vs_dense": chunk["rel_l2_vs_dense"],
                    "one_shot_total_s": chunk["one_shot_total_s"],
                    "one_shot_speedup_vs_dense_total": (
                        chunk["one_shot_speedup_vs_dense_total"]
                    ),
                }
            )
    if not flat_rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms", nargs="+", type=int, default=[250_000, 500_000, 1_000_000])
    parser.add_argument("--chunks", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument("--bin-width-nm", type=float, default=0.1)
    parser.add_argument("--qmin", type=float, default=0.05)
    parser.add_argument("--qmax", type=float, default=6.3)
    parser.add_argument("--q-unit", choices=["inv_angstrom", "inv_nm"], default="inv_angstrom")
    parser.add_argument("--nq", type=int, default=40)
    parser.add_argument("--nphi-detector", type=int, default=180)
    parser.add_argument("--harmonic-margin", type=int, default=16)
    parser.add_argument("--angular-rule", choices=["bandlimit", "arc"], default="bandlimit")
    parser.add_argument("--wavelength-nm", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--fast-preset", choices=FAST_PRESET_NAMES, default="production")
    parser.add_argument("--hist-backend", choices=["numpy", "numba", "numba-parallel", "cpp"], default="cpp")
    parser.add_argument("--hist-dtype", choices=["default", "int64", "uint32", "float32", "float64"], default="float32")
    parser.add_argument("--angle-lut-size", type=int, default=32)
    parser.add_argument("--angle-lut-mode", choices=["nearest", "cubic"], default="cubic")
    parser.add_argument("--circular-backend", choices=["auto", "numpy", "cpp"], default="cpp")
    parser.add_argument("--complex-dtype", choices=["auto", "complex64", "complex128"], default="auto")
    parser.add_argument("--q-block-size", type=int, default=128)
    parser.add_argument("--harmonic-bandlimit-margin", type=int, default=None)
    parser.add_argument("--cache-kernels", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark_results/sparse_source_projection_chunk_sweep.json"),
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("benchmark_results/sparse_source_projection_chunk_sweep.csv"),
    )
    args = parser.parse_args()
    apply_fast_preset(args)

    rows = [
        run_case(n_atoms, args=args, seed=args.seed + i)
        for i, n_atoms in enumerate(args.atoms)
    ]
    result = {
        "case": {
            "atoms": args.atoms,
            "chunks": args.chunks,
            "bin_width_nm": args.bin_width_nm,
            "qmin": args.qmin,
            "qmax": args.qmax,
            "q_unit": args.q_unit,
            "nq": args.nq,
            "nphi_detector": args.nphi_detector,
            "harmonic_margin": args.harmonic_margin,
            "angular_rule": args.angular_rule,
            "wavelength_nm": args.wavelength_nm,
            "repeats": args.repeats,
            "fast_preset": args.fast_preset,
            "hist_backend": args.hist_backend,
            "hist_dtype": args.hist_dtype,
            "angle_lut_size": args.angle_lut_size,
            "angle_lut_mode": args.angle_lut_mode,
            "circular_backend": args.circular_backend,
            "complex_dtype": args.complex_dtype,
            "q_block_size": args.q_block_size,
            "harmonic_bandlimit_margin": args.harmonic_bandlimit_margin,
            "cache_kernels": args.cache_kernels,
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(args.csv_out, rows)
    print(f"wrote {args.out}")
    print(f"wrote {args.csv_out}")


if __name__ == "__main__":
    main()
