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

from waxs_cake import PreparedCakePlan, make_cylindrical_histogram, nufft_amplitude  # noqa: E402
from waxs_cake.metrics import intensity, relative_l2  # noqa: E402
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


def parse_hist_dtype(value: str) -> np.dtype | None:
    return None if value == "default" else np.dtype(value)


def parse_complex_dtype(value: str) -> np.dtype | None:
    return None if value == "auto" else np.dtype(value)


def time_samples(func, *, repeats: int, warmups: int) -> tuple[object, dict]:
    value = None
    for _ in range(warmups):
        value = func()

    times = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        value = func()
        times.append(time.perf_counter() - start)

    return value, {
        "median_s": float(median(times)),
        "min_s": float(min(times)),
        "max_s": float(max(times)),
        "times": times,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms", type=int, default=1_000_000)
    parser.add_argument("--nq", type=int, default=40)
    parser.add_argument("--nphi", type=int, default=180)
    parser.add_argument("--nr", type=int, default=48)
    parser.add_argument("--nz", type=int, default=24)
    parser.add_argument("--qmin", type=float, default=0.05)
    parser.add_argument("--qmax", type=float, default=2.2)
    parser.add_argument("--radius", type=float, default=20.0)
    parser.add_argument("--height", type=float, default=20.0)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=421)
    parser.add_argument(
        "--fast-preset",
        choices=FAST_PRESET_NAMES,
        default="production",
        help=(
            "Apply a fast-path option macro. Use 'production-bandlimited' for "
            "the high-phi band-limited path."
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
    parser.add_argument("--skip-nufft", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark_results/bottleneck.json"),
    )
    args = parser.parse_args()
    apply_fast_preset(args)

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")

    coords = synthetic_cylinder(args.atoms, args.radius, args.height, args.seed)
    q = np.linspace(args.qmin, args.qmax, args.nq)
    phi = (np.arange(args.nphi) + 0.5) * (2.0 * np.pi / args.nphi)

    def build_hist():
        return make_cylindrical_histogram(
            coords,
            n_r=args.nr,
            n_z=args.nz,
            n_phi=args.nphi,
            r_max=args.radius,
            z_range=(-0.5 * args.height, 0.5 * args.height),
            backend=args.hist_backend,
            hist_dtype=parse_hist_dtype(args.hist_dtype),
            angle_lut_size=args.angle_lut_size,
            angle_lut_mode=args.angle_lut_mode,
        )

    binned, hist = time_samples(build_hist, repeats=args.repeats, warmups=args.warmups)

    def build_plan():
        return PreparedCakePlan(
            binned,
            q,
            1.0,
            q_block_size=args.q_block_size,
            circular_backend=args.circular_backend,
            complex_dtype=parse_complex_dtype(args.complex_dtype),
            harmonic_bandlimit_margin=args.harmonic_bandlimit_margin,
        )

    plan, plan_stats = time_samples(
        build_plan,
        repeats=args.repeats,
        warmups=args.warmups,
    )
    amp, solve = time_samples(
        plan.circular_fft,
        repeats=args.repeats,
        warmups=args.warmups,
    )

    nufft = None
    nufft_stats = None
    if not args.skip_nufft:
        nufft, nufft_stats = time_samples(
            lambda: nufft_amplitude(coords, q, 1.0, phi),
            repeats=max(1, min(args.repeats, 5)),
            warmups=0,
        )

    total_median = hist["median_s"] + plan_stats["median_s"] + solve["median_s"]
    fractions = {
        "histogram": hist["median_s"] / total_median,
        "plan": plan_stats["median_s"] / total_median,
        "solve": solve["median_s"] / total_median,
    }

    result = {
        "case": {
            "atoms": args.atoms,
            "nq": args.nq,
            "nphi": args.nphi,
            "nr": args.nr,
            "nz": args.nz,
            "qmax": args.qmax,
            "fast_preset": args.fast_preset,
            "hist_backend": args.hist_backend,
            "hist_dtype": str(binned.hist.dtype),
            "angle_lut_size": args.angle_lut_size,
            "angle_lut_mode": args.angle_lut_mode,
            "circular_backend": args.circular_backend,
            "complex_dtype": str(plan.complex_dtype),
            "q_block_size": args.q_block_size,
            "harmonic_bandlimit_margin": args.harmonic_bandlimit_margin,
        },
        "histogram": hist,
        "plan": plan_stats,
        "solve": solve,
        "total_median_s": total_median,
        "median_fraction": fractions,
        "nufft": nufft_stats,
    }
    if nufft is not None:
        result["speedup_vs_nufft"] = nufft_stats["median_s"] / total_median
        result["amp_rel_l2_vs_nufft"] = relative_l2(amp, nufft)
        result["intensity_rel_l2_vs_nufft"] = relative_l2(intensity(amp), intensity(nufft))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        f"total={total_median:.5f}s hist={hist['median_s']:.5f}s "
        f"plan={plan_stats['median_s']:.5f}s solve={solve['median_s']:.5f}s"
    )
    print(
        "fractions: "
        f"hist={fractions['histogram']:.2%} "
        f"plan={fractions['plan']:.2%} "
        f"solve={fractions['solve']:.2%}"
    )
    if nufft_stats is not None:
        print(f"speedup_vs_nufft={result['speedup_vs_nufft']:.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
