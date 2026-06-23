from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waxs_cake import PreparedCakePlan, make_cylindrical_histogram, nufft_amplitude  # noqa: E402
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402
from waxs_cake.presets import FAST_PRESET_NAMES, apply_fast_preset  # noqa: E402


@dataclass(frozen=True)
class BenchCase:
    name: str
    atoms: int
    nq: int
    nphi: int
    nr: int
    nz: int
    qmax: float
    radius: float = 20.0
    height: float = 20.0
    qmin: float = 0.05
    wavelength: float = 1.0


CASES: dict[str, list[BenchCase]] = {
    "quick": [
        BenchCase("10k_base", 10_000, 40, 180, 48, 48, 2.2),
        BenchCase("100k_base", 100_000, 40, 180, 48, 48, 2.2),
        BenchCase("1m_base", 1_000_000, 40, 180, 48, 48, 2.2),
        BenchCase("1m_high_phi", 1_000_000, 40, 720, 48, 48, 2.2),
    ],
    "default": [
        BenchCase("10k_base", 10_000, 40, 180, 48, 48, 2.2),
        BenchCase("100k_base", 100_000, 40, 180, 48, 48, 2.2),
        BenchCase("1m_base", 1_000_000, 40, 180, 48, 48, 2.2),
        BenchCase("1m_dense_q", 1_000_000, 200, 180, 48, 48, 2.2),
        BenchCase("1m_high_phi", 1_000_000, 40, 720, 48, 48, 2.2),
        BenchCase("100k_high_q", 100_000, 40, 180, 48, 48, 5.0),
        BenchCase("1m_high_q", 1_000_000, 40, 180, 48, 48, 5.0),
        BenchCase("1m_fine_bins", 1_000_000, 40, 180, 96, 96, 2.2),
        BenchCase("1m_low_q", 1_000_000, 40, 180, 48, 48, 0.5),
    ],
}


def synthetic_cylinder(
    n_atoms: int,
    radius: float,
    height: float,
    seed: int,
    *,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.random(n_atoms, dtype=dtype))
    beta = (2.0 * np.pi) * rng.random(n_atoms, dtype=dtype)
    z = height * (rng.random(n_atoms, dtype=dtype) - 0.5)
    coords = np.empty((n_atoms, 3), dtype=dtype)
    coords[:, 0] = r * np.cos(beta)
    coords[:, 1] = r * np.sin(beta)
    coords[:, 2] = z
    return coords


def warm_numba(case: BenchCase) -> None:
    coords = synthetic_cylinder(256, case.radius, case.height, 123)
    make_cylindrical_histogram(
        coords,
        n_r=case.nr,
        n_z=case.nz,
        n_phi=case.nphi,
        r_max=case.radius,
        z_range=(-0.5 * case.height, 0.5 * case.height),
        backend="numba-parallel",
    )


def parse_hist_dtype(value: str) -> np.dtype | None:
    return None if value == "default" else np.dtype(value)


def parse_complex_dtype(value: str) -> np.dtype | None:
    return None if value == "auto" else np.dtype(value)


def time_median(func, repeats: int) -> tuple[object, float, list[float]]:
    times = []
    value = None
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        value = func()
        times.append(time.perf_counter() - start)
    return value, float(median(times)), times


def run_case(
    case: BenchCase,
    *,
    repeats: int,
    seed: int,
    hist_backend: str,
    hist_dtype: np.dtype | None,
    angle_lut_size: int,
    angle_lut_mode: str,
    cache_kernels: bool,
    kernel_interp_dx: float | None,
    circular_backend: str,
    complex_dtype: np.dtype | None,
    q_block_size: int,
    harmonic_bandlimit_margin: int | None,
) -> dict:
    coords = synthetic_cylinder(case.atoms, case.radius, case.height, seed)
    q = np.linspace(case.qmin, case.qmax, case.nq)
    phi = (np.arange(case.nphi) + 0.5) * (2.0 * np.pi / case.nphi)

    def build_hist():
        return make_cylindrical_histogram(
            coords,
            n_r=case.nr,
            n_z=case.nz,
            n_phi=case.nphi,
            r_max=case.radius,
            z_range=(-0.5 * case.height, 0.5 * case.height),
            backend=hist_backend,
            hist_dtype=hist_dtype,
            angle_lut_size=angle_lut_size,
            angle_lut_mode=angle_lut_mode,
        )

    binned, hist_s, hist_times = time_median(build_hist, repeats)

    def build_plan():
        return PreparedCakePlan(
            binned,
            q,
            case.wavelength,
            cache_kernel_fft=cache_kernels,
            kernel_interpolation_dx=kernel_interp_dx,
            circular_backend=circular_backend,
            complex_dtype=complex_dtype,
            q_block_size=q_block_size,
            harmonic_bandlimit_margin=harmonic_bandlimit_margin,
        )

    plan, plan_s, plan_times = time_median(build_plan, repeats)

    def circular():
        return plan.circular_fft()

    circular_amp, circular_s, circular_times = time_median(circular, repeats)

    def nufft():
        return nufft_amplitude(coords, q, case.wavelength, phi)

    nufft_amp, nufft_s, nufft_times = time_median(nufft, repeats)

    amp_rel_l2 = relative_l2(circular_amp, nufft_amp)
    intensity_rel_l2 = relative_l2(intensity(circular_amp), intensity(nufft_amp))
    total_s = hist_s + plan_s + circular_s

    return {
        **asdict(case),
        "hist_backend": hist_backend,
        "hist_dtype": str(binned.hist.dtype),
        "angle_lut_size": angle_lut_size,
        "angle_lut_mode": angle_lut_mode,
        "cache_kernels": cache_kernels,
        "kernel_interp_dx": kernel_interp_dx,
        "circular_backend": circular_backend,
        "complex_dtype": str(plan.complex_dtype),
        "q_block_size": q_block_size,
        "harmonic_bandlimit_margin": harmonic_bandlimit_margin,
        "hist_s": hist_s,
        "plan_s": plan_s,
        "circular_s": circular_s,
        "total_s": total_s,
        "nufft_s": nufft_s,
        "speedup_total_vs_nufft": nufft_s / total_s if total_s else float("inf"),
        "speedup_solve_vs_nufft": nufft_s / circular_s
        if circular_s
        else float("inf"),
        "amp_rel_l2_vs_nufft": amp_rel_l2,
        "intensity_rel_l2_vs_nufft": intensity_rel_l2,
        "hist_times": hist_times,
        "plan_times": plan_times,
        "circular_times": circular_times,
        "nufft_times": nufft_times,
    }


def print_table(rows: list[dict]) -> None:
    headers = [
        "case",
        "atoms",
        "nq",
        "nphi",
        "bins",
        "qmax",
        "hist",
        "circ",
        "total",
        "nufft",
        "x_total",
        "amp_err",
        "int_err",
    ]
    print("\t".join(headers))
    for row in rows:
        print(
            "\t".join(
                [
                    row["name"],
                    f"{row['atoms']}",
                    f"{row['nq']}",
                    f"{row['nphi']}",
                    f"{row['nr']}x{row['nz']}",
                    f"{row['qmax']:.2g}",
                    f"{row['hist_s']:.4f}",
                    f"{row['circular_s']:.4f}",
                    f"{row['total_s']:.4f}",
                    f"{row['nufft_s']:.4f}",
                    f"{row['speedup_total_vs_nufft']:.2f}",
                    f"{row['amp_rel_l2_vs_nufft']:.3g}",
                    f"{row['intensity_rel_l2_vs_nufft']:.3g}",
                ]
            )
        )


def write_outputs(rows: list[dict], out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    scalar_rows = [
        {key: value for key, value in row.items() if not isinstance(value, list)}
        for row in rows
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(CASES), default="default")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--fast-preset",
        choices=FAST_PRESET_NAMES,
        default="none",
        help=(
            "Apply a fast-path option macro. 'production' uses C++/float32/"
            "cubic32/auto-complex; 'production-bandlimited' also uses "
            "harmonic_bandlimit_margin=16."
        ),
    )
    parser.add_argument(
        "--hist-backend",
        choices=["numpy", "numba", "numba-parallel", "cpp"],
        default="numba-parallel",
    )
    parser.add_argument(
        "--hist-dtype",
        choices=["default", "int64", "uint32", "float32", "float64"],
        default="default",
    )
    parser.add_argument("--angle-lut-size", type=int, default=0)
    parser.add_argument(
        "--angle-lut-mode",
        choices=["nearest", "cubic"],
        default="nearest",
    )
    parser.add_argument("--cache-kernels", action="store_true")
    parser.add_argument("--kernel-interp-dx", type=float, default=None)
    parser.add_argument(
        "--circular-backend",
        choices=["auto", "numpy", "cpp"],
        default="auto",
    )
    parser.add_argument(
        "--complex-dtype",
        choices=["auto", "complex64", "complex128"],
        default="auto",
    )
    parser.add_argument("--q-block-size", type=int, default=128)
    parser.add_argument("--harmonic-bandlimit-margin", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("benchmark_results/matrix"))
    args = parser.parse_args()
    apply_fast_preset(args)

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.hist_backend.startswith("numba"):
        warm_numba(CASES[args.preset][0])

    rows = []
    for i, case in enumerate(CASES[args.preset]):
        print(f"\n[{i + 1}/{len(CASES[args.preset])}] {case.name}")
        rows.append(
            run_case(
                case,
                repeats=args.repeats,
                seed=args.seed + i,
                hist_backend=args.hist_backend,
                hist_dtype=parse_hist_dtype(args.hist_dtype),
                angle_lut_size=args.angle_lut_size,
                angle_lut_mode=args.angle_lut_mode,
                cache_kernels=args.cache_kernels,
                kernel_interp_dx=args.kernel_interp_dx,
                circular_backend=args.circular_backend,
                complex_dtype=parse_complex_dtype(args.complex_dtype),
                q_block_size=args.q_block_size,
                harmonic_bandlimit_margin=args.harmonic_bandlimit_margin,
            )
        )
        print_table([rows[-1]])

    print("\nsummary")
    print_table(rows)
    write_outputs(rows, args.out)
    print(f"\nwrote {args.out.with_suffix('.json')} and {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
