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
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waxs_cake import (  # noqa: E402
    PreparedCakePlan,
    choose_physical_grid,
    direct_amplitude,
    make_cylindrical_histogram,
)
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402


def synthetic_water_box(n_atoms: int, side_nm: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(
        -0.5 * side_nm,
        0.5 * side_nm,
        size=(n_atoms, 3),
    )


def parse_atom_counts(value: str) -> list[int]:
    counts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not counts:
        raise ValueError("at least one atom count is required")
    if any(count <= 0 for count in counts):
        raise ValueError("atom counts must be positive")
    return counts


def parse_hist_dtype(value: str) -> np.dtype | None:
    return None if value == "default" else np.dtype(value)


def parse_complex_dtype(value: str) -> np.dtype | None:
    return None if value == "auto" else np.dtype(value)


def median_time(func, repeats: int) -> tuple[object, float, list[float]]:
    value = None
    times: list[float] = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        value = func()
        times.append(time.perf_counter() - start)
    return value, float(median(times)), times


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def direct_ring_curve(
    coords: np.ndarray,
    q_solver: np.ndarray,
    wavelength: float,
    phi: np.ndarray,
) -> np.ndarray:
    amp = direct_amplitude(coords, q_solver, wavelength, phi)
    return np.mean(intensity(amp), axis=1)


def rotated_direct_ring_curve(
    coords: np.ndarray,
    q_solver: np.ndarray,
    wavelength: float,
    phi: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    out = np.zeros(q_solver.size, dtype=float)
    for _ in range(samples):
        rotation = random_rotation_matrix(rng)
        rotated = coords @ rotation.T
        out += direct_ring_curve(rotated, q_solver, wavelength, phi)
    return out / float(samples)


def debye_intensity(
    coords: np.ndarray,
    q_solver: np.ndarray,
    *,
    block_size: int,
    q_block_size: int,
) -> np.ndarray:
    """Exact unit-form-factor Debye intensity.

    The returned curve is sum_ij sinc(q * r_ij / pi), where numpy's sinc uses
    sin(pi x) / (pi x). Coordinates and q_solver must use reciprocal units.
    """

    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    q_block_size = int(q_block_size)
    if q_block_size <= 0:
        raise ValueError("q_block_size must be positive")

    coords = np.ascontiguousarray(coords, dtype=np.float64)
    q_solver = np.asarray(q_solver, dtype=np.float64)
    out = np.zeros(q_solver.size, dtype=np.float64)
    n_atoms = coords.shape[0]

    for start in range(0, n_atoms, block_size):
        stop = min(start + block_size, n_atoms)
        diff = coords[start:stop, None, :] - coords[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        for q_start in range(0, q_solver.size, q_block_size):
            q_stop = min(q_start + q_block_size, q_solver.size)
            q_sel = q_solver[q_start:q_stop]
            qr = q_sel[:, None, None] * dist[None, :, :]
            out[q_start:q_stop] += np.sum(np.sinc(qr / np.pi), axis=(1, 2))
    return out


def debye_intensity_pdist(coords: np.ndarray, q_solver: np.ndarray) -> np.ndarray:
    """Exact unit-form-factor Debye intensity using SciPy's C distance kernel."""

    coords = np.ascontiguousarray(coords, dtype=np.float64)
    q_solver = np.asarray(q_solver, dtype=np.float64)
    n_atoms = coords.shape[0]
    distances = pdist(coords)
    out = np.empty(q_solver.size, dtype=np.float64)
    for i, q_value in enumerate(q_solver):
        out[i] = n_atoms + 2.0 * np.sum(np.sinc((q_value * distances) / np.pi))
    return out


def build_our_curve(
    coords: np.ndarray,
    q_solver: np.ndarray,
    *,
    grid,
    args,
) -> tuple[np.ndarray, dict[str, float | list[float] | str]]:
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
    plan, plan_s, plan_times = median_time(
        lambda: PreparedCakePlan(
            binned,
            q_solver,
            args.wavelength_nm,
            q_block_size=args.q_block_size,
            circular_backend=args.circular_backend,
            complex_dtype=complex_dtype,
            harmonic_bandlimit_margin=args.harmonic_bandlimit_margin,
        ),
        args.repeats,
    )

    if args.curve_backend == "r-dependent":
        curve_func = lambda: plan.ring_average_intensity_r_dependent_bandlimit(
            margin=args.r_dependent_margin,
            cutoff_bin_size=args.r_dependent_cutoff_bin_size,
        )
    else:
        curve_func = plan.ring_average_intensity

    curve, curve_s, curve_times = median_time(curve_func, args.repeats)
    timings: dict[str, float | list[float] | str] = {
        "hist_s": hist_s,
        "plan_s": plan_s,
        "curve_s": curve_s,
        "curve_total_s": hist_s + plan_s + curve_s,
        "hist_times": hist_times,
        "plan_times": plan_times,
        "curve_times": curve_times,
        "hist_dtype_actual": str(binned.hist.dtype),
        "complex_dtype_actual": str(plan.complex_dtype),
    }
    return curve, timings


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
    coords = synthetic_water_box(n_atoms, grid.box_side_nm, seed)
    q_report = np.linspace(args.qmin, args.qmax, args.nq)
    q_solver = 10.0 * q_report if args.q_unit == "inv_angstrom" else q_report
    phi = (np.arange(grid.n_phi) + 0.5) * (2.0 * np.pi / grid.n_phi)

    print(
        f"{n_atoms}: side={grid.box_side_nm:.2f} nm "
        f"bins={grid.n_r}x{grid.n_z}x{grid.n_phi} q=[{args.qmin}, {args.qmax}]"
    )

    if args.debye_backend == "pdist":
        debye_func = lambda: debye_intensity_pdist(coords, q_solver)
    else:
        debye_func = lambda: debye_intensity(
            coords,
            q_solver,
            block_size=args.debye_block_size,
            q_block_size=args.debye_q_block_size,
        )
    debye, debye_s, debye_times = median_time(debye_func, args.repeats)
    direct_curve, direct_s, direct_times = median_time(
        lambda: direct_ring_curve(coords, q_solver, args.wavelength_nm, phi),
        args.repeats,
    )
    if args.orientation_samples > 0:
        rotated_curve, rotated_s, rotated_times = median_time(
            lambda: rotated_direct_ring_curve(
                coords,
                q_solver,
                args.wavelength_nm,
                phi,
                samples=args.orientation_samples,
                seed=seed + 10_000,
            ),
            args.repeats,
        )
    else:
        rotated_curve = None
        rotated_s = None
        rotated_times = []

    our_curve, curve_timings = build_our_curve(coords, q_solver, grid=grid, args=args)

    row = {
        "atoms": n_atoms,
        "seed": seed,
        "box_side_nm": grid.box_side_nm,
        "r_max_nm": grid.r_max_nm,
        "n_r": grid.n_r,
        "n_z": grid.n_z,
        "n_phi": grid.n_phi,
        "qmin": args.qmin,
        "qmax": args.qmax,
        "nq": args.nq,
        "q_unit": args.q_unit,
        "wavelength_nm": args.wavelength_nm,
        "bin_width_nm": args.bin_width_nm,
        "curve_backend": args.curve_backend,
        "debye_backend": args.debye_backend,
        "hist_backend": args.hist_backend,
        "angle_lut_size": args.angle_lut_size,
        "angle_lut_mode": args.angle_lut_mode,
        "circular_backend": args.circular_backend,
        "q_block_size": args.q_block_size,
        "harmonic_bandlimit_margin": args.harmonic_bandlimit_margin,
        "debye_s": debye_s,
        "direct_ring_s": direct_s,
        "rotated_ring_s": rotated_s,
        "debye_times": debye_times,
        "direct_ring_times": direct_times,
        "rotated_ring_times": rotated_times,
        **curve_timings,
        "direct_ring_rel_l2_vs_debye": relative_l2(direct_curve, debye),
        "our_curve_rel_l2_vs_direct_ring": relative_l2(our_curve, direct_curve),
        "our_curve_rel_l2_vs_debye": relative_l2(our_curve, debye),
        "speedup_curve_total_vs_debye": debye_s / curve_timings["curve_total_s"],
        "speedup_curve_solve_vs_debye": debye_s / curve_timings["curve_s"],
    }
    if rotated_curve is not None:
        row.update(
            {
                "rotated_ring_rel_l2_vs_debye": relative_l2(rotated_curve, debye),
                "our_curve_rel_l2_vs_rotated_ring": relative_l2(
                    our_curve,
                    rotated_curve,
                ),
            }
        )
    else:
        row.update(
            {
                "rotated_ring_rel_l2_vs_debye": None,
                "our_curve_rel_l2_vs_rotated_ring": None,
            }
        )
    return row


def write_outputs(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    scalar_rows = [
        {key: value for key, value in row.items() if not isinstance(value, list)}
        for row in rows
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)


def print_table(rows: list[dict]) -> None:
    headers = [
        "atoms",
        "nphi",
        "debye_s",
        "direct_s",
        "curve_total_s",
        "x_total",
        "ring_vs_debye",
        "curve_vs_ring",
        "curve_vs_debye",
    ]
    print("\t".join(headers))
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["atoms"]),
                    str(row["n_phi"]),
                    f"{row['debye_s']:.4f}",
                    f"{row['direct_ring_s']:.4f}",
                    f"{row['curve_total_s']:.4f}",
                    f"{row['speedup_curve_total_vs_debye']:.2f}",
                    f"{row['direct_ring_rel_l2_vs_debye']:.3g}",
                    f"{row['our_curve_rel_l2_vs_direct_ring']:.3g}",
                    f"{row['our_curve_rel_l2_vs_debye']:.3g}",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atom-counts", default="250,500,1000,2000")
    parser.add_argument("--qmin", type=float, default=0.05)
    parser.add_argument("--qmax", type=float, default=2.2)
    parser.add_argument("--nq", type=int, default=24)
    parser.add_argument(
        "--q-unit",
        choices=["inv_angstrom", "inv_nm"],
        default="inv_angstrom",
    )
    parser.add_argument("--wavelength-nm", type=float, default=0.1)
    parser.add_argument("--bin-width-nm", type=float, default=0.1)
    parser.add_argument("--nphi-detector", type=int, default=180)
    parser.add_argument("--harmonic-margin", type=int, default=16)
    parser.add_argument(
        "--angular-rule",
        choices=["bandlimit", "arc"],
        default="bandlimit",
    )
    parser.add_argument("--hist-backend", choices=["numpy", "cpp"], default="cpp")
    parser.add_argument(
        "--hist-dtype",
        choices=["default", "uint32", "float32", "float64"],
        default="float32",
    )
    parser.add_argument("--angle-lut-size", type=int, default=32)
    parser.add_argument("--angle-lut-mode", choices=["nearest", "cubic"], default="cubic")
    parser.add_argument("--circular-backend", choices=["auto", "numpy", "cpp"], default="cpp")
    parser.add_argument("--complex-dtype", choices=["auto", "complex64", "complex128"], default="auto")
    parser.add_argument("--q-block-size", type=int, default=128)
    parser.add_argument("--harmonic-bandlimit-margin", type=int, default=None)
    parser.add_argument(
        "--curve-backend",
        choices=["r-grouped", "r-dependent"],
        default="r-grouped",
    )
    parser.add_argument("--r-dependent-margin", type=int, default=16)
    parser.add_argument("--r-dependent-cutoff-bin-size", type=int, default=16)
    parser.add_argument("--debye-block-size", type=int, default=128)
    parser.add_argument("--debye-q-block-size", type=int, default=8)
    parser.add_argument(
        "--debye-backend",
        choices=["blocked", "pdist"],
        default="pdist",
    )
    parser.add_argument("--orientation-samples", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/debye_curve_comparison.json"),
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    atom_counts = parse_atom_counts(args.atom_counts)
    rows = [
        run_case(n_atoms, args=args, seed=args.seed + i)
        for i, n_atoms in enumerate(atom_counts)
    ]
    print_table(rows)
    write_outputs(rows, args.output)
    print(f"wrote {args.output} and {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
