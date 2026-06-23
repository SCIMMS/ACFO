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


def median_time(func, repeats: int):
    value = None
    times = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        value = func()
        times.append(time.perf_counter() - start)
    return value, float(median(times)), times


def parse_bins(items: list[str]) -> list[tuple[int, int]]:
    out = []
    for item in items:
        left, right = item.lower().split("x", maxsplit=1)
        out.append((int(left), int(right)))
    return out


def parse_hist_dtype(value: str) -> np.dtype | None:
    return None if value == "default" else np.dtype(value)


def parse_complex_dtype(value: str) -> np.dtype | None:
    return None if value == "auto" else np.dtype(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms", type=int, default=1_000_000)
    parser.add_argument("--nq", type=int, default=40)
    parser.add_argument("--nphi", type=int, default=180)
    parser.add_argument("--qmin", type=float, default=0.05)
    parser.add_argument("--qmax", type=float, default=2.2)
    parser.add_argument("--radius", type=float, default=20.0)
    parser.add_argument("--height", type=float, default=20.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=251)
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
        "--bins",
        nargs="+",
        default=["48x48", "48x32", "48x24", "48x16", "64x24", "64x16", "96x32", "96x24"],
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
        default=Path("benchmark_results/binning_frontier.json"),
    )
    args = parser.parse_args()
    apply_fast_preset(args)

    coords = synthetic_cylinder(args.atoms, args.radius, args.height, args.seed)
    q = np.linspace(args.qmin, args.qmax, args.nq)
    phi = (np.arange(args.nphi) + 0.5) * (2.0 * np.pi / args.nphi)

    if args.hist_backend.startswith("numba"):
        nr0, nz0 = parse_bins(args.bins)[0]
        make_cylindrical_histogram(
            coords[:512],
            n_r=nr0,
            n_z=nz0,
            n_phi=args.nphi,
            r_max=args.radius,
            z_range=(-0.5 * args.height, 0.5 * args.height),
            backend=args.hist_backend,
            hist_dtype=parse_hist_dtype(args.hist_dtype),
            angle_lut_size=args.angle_lut_size,
            angle_lut_mode=args.angle_lut_mode,
        )

    nufft = None
    nufft_s = None
    nufft_times = []
    if not args.skip_nufft:
        nufft, nufft_s, nufft_times = median_time(
            lambda: nufft_amplitude(coords, q, 1.0, phi),
            args.repeats,
        )

    rows = []
    for nr, nz in parse_bins(args.bins):
        binned, hist_s, hist_times = median_time(
            lambda nr=nr, nz=nz: make_cylindrical_histogram(
                coords,
                n_r=nr,
                n_z=nz,
                n_phi=args.nphi,
                r_max=args.radius,
                z_range=(-0.5 * args.height, 0.5 * args.height),
                backend=args.hist_backend,
                hist_dtype=parse_hist_dtype(args.hist_dtype),
                angle_lut_size=args.angle_lut_size,
                angle_lut_mode=args.angle_lut_mode,
            ),
            args.repeats,
        )
        plan, plan_s, plan_times = median_time(
            lambda binned=binned: PreparedCakePlan(
                binned,
                q,
                1.0,
                q_block_size=args.q_block_size,
                circular_backend=args.circular_backend,
                complex_dtype=parse_complex_dtype(args.complex_dtype),
                harmonic_bandlimit_margin=args.harmonic_bandlimit_margin,
            ),
            args.repeats,
        )
        amp, solve_s, solve_times = median_time(plan.circular_fft, args.repeats)
        total_s = hist_s + plan_s + solve_s
        row = {
            "nr": nr,
            "nz": nz,
            "n_bins": nr * nz * args.nphi,
            "hist_s": hist_s,
            "plan_s": plan_s,
            "solve_s": solve_s,
            "total_s": total_s,
            "hist_times": hist_times,
            "plan_times": plan_times,
            "solve_times": solve_times,
        }
        if nufft is not None:
            row["speedup_vs_nufft"] = nufft_s / total_s if total_s else float("inf")
            row["amp_rel_l2_vs_nufft"] = relative_l2(amp, nufft)
            row["intensity_rel_l2_vs_nufft"] = relative_l2(intensity(amp), intensity(nufft))
        rows.append(row)
        int_err = row.get("intensity_rel_l2_vs_nufft")
        print(
            f"{nr}x{nz}: total={total_s:.4f}s hist={hist_s:.4f}s solve={solve_s:.4f}s "
            f"x={row.get('speedup_vs_nufft', float('nan')):.2f} "
            f"int_err={'' if int_err is None else f'{int_err:.3g}'}"
        )

    result = {
        "case": {
            "atoms": args.atoms,
            "nq": args.nq,
            "nphi": args.nphi,
            "qmax": args.qmax,
            "radius": args.radius,
            "height": args.height,
            "hist_backend": args.hist_backend,
            "hist_dtype": str(parse_hist_dtype(args.hist_dtype) or "default"),
            "angle_lut_size": args.angle_lut_size,
            "angle_lut_mode": args.angle_lut_mode,
            "circular_backend": args.circular_backend,
            "complex_dtype": args.complex_dtype,
            "q_block_size": args.q_block_size,
            "harmonic_bandlimit_margin": args.harmonic_bandlimit_margin,
        },
        "nufft_s": nufft_s,
        "nufft_times": nufft_times,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
