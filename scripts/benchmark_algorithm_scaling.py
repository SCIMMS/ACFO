"""Direct one-shot scaling comparison for cake-map algorithms.

Unlike ``benchmark_physical_scaling.py``, this script times each algorithm on a
fresh ``PreparedCakePlan``. That keeps dense circular FFT, R-dependent analytic,
fused R-dependent analytic, and NUFFT from borrowing each other's caches.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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
    nufft_amplitude,
    nufft_amplitude_chunked,
)
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402
from waxs_cake.presets import FAST_PRESET_NAMES, apply_fast_preset  # noqa: E402


def median_time(func, repeats: int):
    value = None
    times = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        value = func()
        times.append(time.perf_counter() - start)
    return value, float(median(times)), times


def fit_power_law(rows: list[dict], key: str) -> float | None:
    points = [
        (float(row["atoms"]), float(row[key]))
        for row in rows
        if row.get(key) is not None and row[key] > 0
    ]
    if len(points) < 2:
        return None
    x = np.log([p[0] for p in points])
    y = np.log([p[1] for p in points])
    return float(np.polyfit(x, y, 1)[0])


def fit_power_laws(rows: list[dict], keys: list[str]) -> dict[str, float | None]:
    return {key: fit_power_law(rows, key) for key in keys}


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
        f"bins={grid.n_r}x{grid.n_z}x{grid.n_phi}"
    )

    coords = synthetic_water_box(n_atoms, grid.box_side_nm, seed)
    q = np.linspace(args.qmin, args.qmax, args.nq)
    q_solver = 10.0 * q if args.q_unit == "inv_angstrom" else q
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
    occupancy = occupancy_summary(binned.hist)

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

    dense_plan, dense_plan_s, dense_plan_times = median_time(make_plan, args.repeats)
    dense_amp, dense_s, dense_times = median_time(dense_plan.circular_fft, args.repeats)

    rdep_plan, rdep_plan_s, rdep_plan_times = median_time(make_plan, args.repeats)
    rdep_amp, rdep_s, rdep_times = median_time(
        lambda: rdep_plan.circular_fft_r_dependent_bandlimit(
            margin=args.r_dependent_margin,
            cutoff_bin_size=args.r_dependent_cutoff_bin_size,
            analytic_kernel=True,
        ),
        args.repeats,
    )

    fused_plan, fused_plan_s, fused_plan_times = median_time(make_plan, args.repeats)
    fused_amp, fused_s, fused_times = median_time(
        lambda: fused_plan.circular_fft_r_dependent_bandlimit(
            margin=args.r_dependent_margin,
            cutoff_bin_size=args.r_dependent_cutoff_bin_size,
            analytic_kernel=True,
            fused_analytic_kernel=True,
        ),
        args.repeats,
    )

    nufft_s = None
    nufft_times: list[float] = []
    nufft_amp = None
    if not args.skip_nufft:
        if args.nufft_q_block_size is None:
            nufft_func = lambda: nufft_amplitude(
                coords,
                q_solver,
                args.wavelength_nm,
                phi,
            )
        else:
            nufft_func = lambda: nufft_amplitude_chunked(
                coords,
                q_solver,
                args.wavelength_nm,
                phi,
                q_block_size=args.nufft_q_block_size,
            )
        nufft_amp, nufft_s, nufft_times = median_time(
            nufft_func,
            max(1, min(args.repeats, 3)),
        )

    row = {
        "atoms": n_atoms,
        "grid": grid_summary(grid),
        **occupancy,
        "hist_s": hist_s,
        "dense_plan_s": dense_plan_s,
        "dense_s": dense_s,
        "dense_total_s": hist_s + dense_plan_s + dense_s,
        "rdep_plan_s": rdep_plan_s,
        "rdep_analytic_s": rdep_s,
        "rdep_analytic_total_s": hist_s + rdep_plan_s + rdep_s,
        "rdep_fused_s": fused_s,
        "rdep_fused_plan_s": fused_plan_s,
        "rdep_fused_total_s": hist_s + fused_plan_s + fused_s,
        "nufft_s": nufft_s,
        "rdep_analytic_rel_l2_vs_dense": relative_l2(rdep_amp, dense_amp),
        "rdep_analytic_intensity_rel_l2_vs_dense": relative_l2(
            intensity(rdep_amp),
            intensity(dense_amp),
        ),
        "rdep_fused_rel_l2_vs_dense": relative_l2(fused_amp, dense_amp),
        "rdep_fused_intensity_rel_l2_vs_dense": relative_l2(
            intensity(fused_amp),
            intensity(dense_amp),
        ),
        "rdep_fused_rel_l2_vs_rdep_analytic": relative_l2(fused_amp, rdep_amp),
        "nufft_rel_l2_vs_dense": None
        if nufft_amp is None
        else relative_l2(nufft_amp, dense_amp),
        "nufft_intensity_rel_l2_vs_dense": None
        if nufft_amp is None
        else relative_l2(intensity(nufft_amp), intensity(dense_amp)),
        "rdep_speedup_vs_dense": dense_s / rdep_s if rdep_s else None,
        "rdep_fused_speedup_vs_dense": dense_s / fused_s if fused_s else None,
        "rdep_speedup_vs_nufft": None if nufft_s is None else nufft_s / rdep_s,
        "rdep_fused_speedup_vs_nufft": None
        if nufft_s is None
        else nufft_s / fused_s,
        "dense_times": dense_times,
        "rdep_analytic_times": rdep_times,
        "rdep_fused_times": fused_times,
        "nufft_times": nufft_times,
        "hist_times": hist_times,
        "dense_plan_times": dense_plan_times,
        "rdep_plan_times": rdep_plan_times,
        "rdep_fused_plan_times": fused_plan_times,
    }
    print(
        "  "
        f"dense={dense_s:.4f}s "
        f"rdep={rdep_s:.4f}s "
        f"fused={fused_s:.4f}s "
        f"nufft={nufft_s} "
        f"I_err={row['rdep_analytic_intensity_rel_l2_vs_dense']:.3g}"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms", nargs="+", type=int, default=[100_000, 250_000, 500_000, 1_000_000])
    parser.add_argument("--bin-width-nm", type=float, default=0.1)
    parser.add_argument("--qmin", type=float, default=0.05)
    parser.add_argument("--qmax", type=float, default=6.3)
    parser.add_argument("--q-unit", choices=["inv_angstrom", "inv_nm"], default="inv_angstrom")
    parser.add_argument("--nq", type=int, default=40)
    parser.add_argument("--nphi-detector", type=int, default=180)
    parser.add_argument("--harmonic-margin", type=int, default=16)
    parser.add_argument("--angular-rule", choices=["bandlimit", "arc"], default="bandlimit")
    parser.add_argument("--wavelength-nm", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--r-dependent-margin", type=int, default=16)
    parser.add_argument("--r-dependent-cutoff-bin-size", type=int, default=16)
    parser.add_argument(
        "--fast-preset",
        choices=FAST_PRESET_NAMES,
        default="production",
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
    parser.add_argument("--nufft-q-block-size", type=int, default=2)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark_results/algorithm_scaling_highq_direct.json"),
    )
    args = parser.parse_args()
    apply_fast_preset(args)

    rows = [
        run_case(n_atoms, args=args, seed=args.seed + i)
        for i, n_atoms in enumerate(args.atoms)
    ]
    scaling_keys = [
        "hist_s",
        "dense_s",
        "dense_total_s",
        "rdep_analytic_s",
        "rdep_analytic_total_s",
        "rdep_fused_s",
        "rdep_fused_total_s",
        "nufft_s",
    ]
    result = {
        "case": {
            **vars(args),
            "out": str(args.out),
        },
        "scaling_exponents_vs_atoms": fit_power_laws(rows, scaling_keys),
        "rows": rows,
    }
    result["case"]["atoms"] = args.atoms
    result["case"]["out"] = str(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(json.dumps(result["scaling_exponents_vs_atoms"], indent=2))


if __name__ == "__main__":
    main()
