from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waxs_cake import PreparedCakePlan, make_cylindrical_histogram  # noqa: E402
from waxs_cake.presets import FAST_PRESET_NAMES, apply_fast_preset  # noqa: E402


def synthetic_cylinder(n_atoms: int, radius: float, height: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.random(n_atoms))
    beta = 2.0 * np.pi * rng.random(n_atoms)
    z = height * (rng.random(n_atoms) - 0.5)
    coords = np.empty((n_atoms, 3), dtype=np.float64)
    coords[:, 0] = r * np.cos(beta)
    coords[:, 1] = r * np.sin(beta)
    coords[:, 2] = z
    return coords


def timed(func):
    start = time.perf_counter()
    value = func()
    return value, time.perf_counter() - start


def median_time(func, repeats: int):
    value = None
    times = []
    for _ in range(repeats):
        gc.collect()
        value, elapsed = timed(func)
        times.append(elapsed)
    return value, float(median(times)), times


def warm_numba(coords: np.ndarray, args) -> None:
    make_cylindrical_histogram(
        coords[: min(coords.shape[0], 512)],
        hist_dtype=parse_hist_dtype(args.hist_dtype),
        angle_lut_size=args.angle_lut_size,
        angle_lut_mode=args.angle_lut_mode,
        n_r=args.nr,
        n_z=args.nz,
        n_phi=args.nphi,
        r_max=args.radius,
        z_range=(-0.5 * args.height, 0.5 * args.height),
        backend=args.hist_backend,
    )


def parse_hist_dtype(value: str) -> np.dtype | None:
    return None if value == "default" else np.dtype(value)


def parse_complex_dtype(value: str) -> np.dtype | None:
    return None if value == "auto" else np.dtype(value)


def breakdown(plan: PreparedCakePlan, q_block_size: int, repeats: int) -> dict:
    indices = np.arange(plan.q.size)

    def run_kernel_only():
        total = 0
        for start in range(0, indices.size, q_block_size):
            sel = indices[start : start + q_block_size]
            total += plan._kernel_hat_block(sel).size
        return total

    def run_z_only():
        total = 0
        for start in range(0, indices.size, q_block_size):
            sel = indices[start : start + q_block_size]
            total += plan._z_reduced_block(sel).size
        return total

    def run_contract_only():
        total = 0
        for start in range(0, indices.size, q_block_size):
            sel = indices[start : start + q_block_size]
            khat = plan._kernel_hat_block(sel)
            z_reduced = plan._z_reduced_block(sel)
            if z_reduced.shape[1] == 1:
                total += (
                    plan.form_factors[0, sel, None]
                    * np.einsum(
                        "brh,brh->bh",
                        z_reduced[:, 0],
                        khat,
                        optimize=True,
                    )
                ).size
            else:
                total += np.einsum(
                    "eb,berh,brh->bh",
                    plan.form_factors[:, sel],
                    z_reduced,
                    khat,
                    optimize=True,
                ).size
        return total

    def run_ahat():
        return plan.circular_ahat(q_block_size=q_block_size)

    def run_ifft():
        ahat = plan.circular_ahat(q_block_size=q_block_size)
        return np.fft.ifft(ahat, axis=-1)

    _, kernel_s, kernel_times = median_time(run_kernel_only, repeats)
    _, z_s, z_times = median_time(run_z_only, repeats)
    _, contract_s, contract_times = median_time(run_contract_only, repeats)
    ahat, ahat_s, ahat_times = median_time(run_ahat, repeats)
    _, ifft_total_s, ifft_total_times = median_time(run_ifft, repeats)
    _, ifft_only_s, ifft_only_times = median_time(
        lambda: np.fft.ifft(ahat, axis=-1),
        repeats,
    )

    return {
        "q_block_size": q_block_size,
        "kernel_s": kernel_s,
        "z_reduction_s": z_s,
        "contract_s_includes_kernel_z": contract_s,
        "ahat_s": ahat_s,
        "ifft_only_s": ifft_only_s,
        "ifft_total_s": ifft_total_s,
        "kernel_times": kernel_times,
        "z_reduction_times": z_times,
        "contract_times": contract_times,
        "ahat_times": ahat_times,
        "ifft_only_times": ifft_only_times,
        "ifft_total_times": ifft_total_times,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms", type=int, default=1_000_000)
    parser.add_argument("--nq", type=int, default=40)
    parser.add_argument("--nphi", type=int, default=180)
    parser.add_argument("--nr", type=int, default=48)
    parser.add_argument("--nz", type=int, default=48)
    parser.add_argument("--qmin", type=float, default=0.05)
    parser.add_argument("--qmax", type=float, default=2.2)
    parser.add_argument("--radius", type=float, default=20.0)
    parser.add_argument("--height", type=float, default=20.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=151)
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
    parser.add_argument(
        "--complex-dtype",
        choices=["auto", "complex64", "complex128"],
        default="auto",
    )
    parser.add_argument(
        "--circular-backend",
        choices=["auto", "numpy", "cpp"],
        default="numpy",
    )
    parser.add_argument(
        "--q-block-sizes",
        nargs="+",
        type=int,
        default=[8, 16, 32, 64, 128],
    )
    parser.add_argument("--harmonic-bandlimit-margin", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark_results/one_shot_breakdown.json"),
    )
    args = parser.parse_args()
    apply_fast_preset(args)

    coords = synthetic_cylinder(args.atoms, args.radius, args.height, args.seed)
    if args.hist_backend.startswith("numba"):
        warm_numba(coords, args)

    binned, hist_s, hist_times = median_time(
        lambda: make_cylindrical_histogram(
            coords,
            hist_dtype=parse_hist_dtype(args.hist_dtype),
            angle_lut_size=args.angle_lut_size,
            angle_lut_mode=args.angle_lut_mode,
            n_r=args.nr,
            n_z=args.nz,
            n_phi=args.nphi,
            r_max=args.radius,
            z_range=(-0.5 * args.height, 0.5 * args.height),
            backend=args.hist_backend,
        ),
        args.repeats,
    )
    q = np.linspace(args.qmin, args.qmax, args.nq)
    plan, plan_s, plan_times = median_time(
        lambda: PreparedCakePlan(
            binned,
            q,
            1.0,
            circular_backend=args.circular_backend,
            complex_dtype=parse_complex_dtype(args.complex_dtype),
            harmonic_bandlimit_margin=args.harmonic_bandlimit_margin,
        ),
        args.repeats,
    )

    rows = []
    for block_size in args.q_block_sizes:
        row = breakdown(plan, block_size, args.repeats)
        rows.append(row)
        print(
            f"block={block_size:>3d} kernel={row['kernel_s']:.5f}s "
            f"z={row['z_reduction_s']:.5f}s ahat={row['ahat_s']:.5f}s "
            f"ifft={row['ifft_only_s']:.5f}s"
        )

    result = {
        "case": {
            "atoms": args.atoms,
            "nq": args.nq,
            "nphi": args.nphi,
            "nr": args.nr,
            "nz": args.nz,
            "qmax": args.qmax,
            "hist_backend": args.hist_backend,
            "hist_dtype": str(binned.hist.dtype),
            "angle_lut_size": args.angle_lut_size,
            "angle_lut_mode": args.angle_lut_mode,
            "circular_backend": args.circular_backend,
            "complex_dtype": str(plan.complex_dtype),
            "harmonic_bandlimit_margin": args.harmonic_bandlimit_margin,
        },
        "hist_s": hist_s,
        "hist_times": hist_times,
        "plan_s": plan_s,
        "plan_times": plan_times,
        "blocks": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"hist={hist_s:.5f}s plan={plan_s:.5f}s")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
