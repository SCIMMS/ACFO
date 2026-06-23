from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waxs_cake import PreparedCakePlan, make_cylindrical_histogram  # noqa: E402
from waxs_cake.metrics import relative_l2  # noqa: E402


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
    values = []
    last = None
    for _ in range(repeats):
        last, elapsed = timed(func)
        values.append(elapsed)
    return last, float(median(values)), values


def make_form_factor_sets(q: np.ndarray, count: int) -> list[dict[str, np.ndarray]]:
    sets = []
    q_scaled = (q - q.min()) / max(np.ptp(q), 1e-12)
    for i in range(count):
        scale = 0.05 * (i + 1)
        phase = 0.02 * i
        sets.append({"X": (1.0 + scale * q_scaled) * np.exp(1j * phase * q_scaled)})
    return sets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms", type=int, default=1_000_000)
    parser.add_argument("--nq", type=int, default=40)
    parser.add_argument("--nphi", type=int, default=180)
    parser.add_argument("--nr", type=int, default=48)
    parser.add_argument("--nz", type=int, default=48)
    parser.add_argument("--qmax", type=float, default=2.2)
    parser.add_argument("--radius", type=float, default=20.0)
    parser.add_argument("--height", type=float, default=20.0)
    parser.add_argument("--sweeps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cache-kernels", action="store_true")
    parser.add_argument("--cache-z", action="store_true")
    parser.add_argument(
        "--hist-backend",
        choices=["numpy", "numba", "numba-parallel", "cpp"],
        default="numba-parallel",
    )
    parser.add_argument(
        "--circular-backend",
        choices=["auto", "numpy", "cpp"],
        default="auto",
    )
    parser.add_argument("--q-block-size", type=int, default=128)
    parser.add_argument("--harmonic-bandlimit-margin", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("benchmark_results/reuse.json"))
    args = parser.parse_args()

    coords = synthetic_cylinder(args.atoms, args.radius, args.height, 41)
    q = np.linspace(0.05, args.qmax, args.nq)

    if args.hist_backend.startswith("numba"):
        make_cylindrical_histogram(
            coords[:256],
            n_r=args.nr,
            n_z=args.nz,
            n_phi=args.nphi,
            r_max=args.radius,
            z_range=(-0.5 * args.height, 0.5 * args.height),
            backend=args.hist_backend,
        )

    binned, hist_s = timed(
        lambda: make_cylindrical_histogram(
            coords,
            n_r=args.nr,
            n_z=args.nz,
            n_phi=args.nphi,
            r_max=args.radius,
            z_range=(-0.5 * args.height, 0.5 * args.height),
            backend=args.hist_backend,
        )
    )
    plan, plan_s = timed(
        lambda: PreparedCakePlan(
            binned,
            q,
            1.0,
            cache_kernel_fft=args.cache_kernels,
            circular_backend=args.circular_backend,
            q_block_size=args.q_block_size,
            harmonic_bandlimit_margin=args.harmonic_bandlimit_margin,
        )
    )
    if args.cache_z:
        _, z_cache_s = timed(plan.precompute_z_reduced)
    else:
        z_cache_s = 0.0

    ff_sets = make_form_factor_sets(q, args.sweeps)

    def run_sweep():
        return [plan.circular_fft_with_form_factors(ff) for ff in ff_sets]

    sweep_values, sweep_s, sweep_times = median_time(run_sweep, args.repeats)

    def run_ring_average():
        return [plan.ring_average_intensity(form_factors=ff) for ff in ff_sets]

    ring_values, ring_s, ring_times = median_time(run_ring_average, args.repeats)

    amp0 = sweep_values[0]
    ring_expected = np.mean(np.abs(amp0) ** 2, axis=1)
    ring_err = relative_l2(ring_values[0], ring_expected)

    result = {
        "atoms": args.atoms,
        "nq": args.nq,
        "nphi": args.nphi,
        "bins": [args.nr, args.nz],
        "qmax": args.qmax,
        "sweeps": args.sweeps,
        "cache_kernels": args.cache_kernels,
        "cache_z": args.cache_z,
        "hist_backend": args.hist_backend,
        "circular_backend": args.circular_backend,
        "q_block_size": args.q_block_size,
        "harmonic_bandlimit_margin": args.harmonic_bandlimit_margin,
        "hist_s": hist_s,
        "plan_s": plan_s,
        "z_cache_s": z_cache_s,
        "sweep_s": sweep_s,
        "sweep_per_template_s": sweep_s / args.sweeps,
        "ring_s": ring_s,
        "ring_per_template_s": ring_s / args.sweeps,
        "ring_average_rel_l2": ring_err,
        "sweep_times": sweep_times,
        "ring_times": ring_times,
    }

    print(json.dumps(result, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
