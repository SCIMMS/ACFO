from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waxs_cake import (  # noqa: E402
    PreparedCakePlan,
    direct_amplitude,
    make_cylindrical_histogram,
    nufft_amplitude,
)
from waxs_cake.geometry import ewald_ring  # noqa: E402
from waxs_cake.metrics import intensity, max_relative_abs, relative_l2  # noqa: E402
from waxs_cake.presets import FAST_PRESET_NAMES, apply_fast_preset  # noqa: E402


def synthetic_cluster(
    n_atoms: int,
    radius: float,
    height: float,
    seed: int,
    *,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coords = []
    while len(coords) < n_atoms:
        trial = rng.uniform(-1.0, 1.0, size=(max(1024, n_atoms), 3)).astype(dtype)
        keep = np.sum(trial[:, :2] ** 2, axis=1) <= 1.0
        picked = trial[keep]
        picked[:, :2] *= radius
        picked[:, 2] *= 0.5 * height
        coords.extend(picked.tolist())
    return np.asarray(coords[:n_atoms], dtype=dtype)


def timed(label: str, func):
    start = time.perf_counter()
    value = func()
    elapsed = time.perf_counter() - start
    print(f"{label:>14s}: {elapsed:8.4f} s")
    return value, elapsed


def summarize(name: str, candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    return {
        "amp_rel_l2": relative_l2(candidate, reference),
        "amp_max_rel": max_relative_abs(candidate, reference),
        "intensity_rel_l2": relative_l2(intensity(candidate), intensity(reference)),
    }


def parse_hist_dtype(value: str) -> np.dtype | None:
    return None if value == "default" else np.dtype(value)


def parse_complex_dtype(value: str) -> np.dtype | None:
    return None if value == "auto" else np.dtype(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms", type=int, default=1500)
    parser.add_argument("--radius", type=float, default=20.0)
    parser.add_argument("--height", type=float, default=20.0)
    parser.add_argument("--nq", type=int, default=40)
    parser.add_argument("--nphi", type=int, default=180)
    parser.add_argument("--nr", type=int, default=48)
    parser.add_argument("--nz", type=int, default=48)
    parser.add_argument("--qmin", type=float, default=0.05)
    parser.add_argument("--qmax", type=float, default=2.2)
    parser.add_argument("--wavelength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--fast-preset",
        choices=FAST_PRESET_NAMES,
        default="none",
        help=(
            "Apply a reproducible fast-path macro. 'production' uses C++ "
            "histogram/circular solver, float32 histogram, cubic32 angle LUT, "
            "and auto complex dtype. 'production-bandlimited' also sets "
            "harmonic_bandlimit_margin=16."
        ),
    )
    parser.add_argument("--coord-dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument("--binning-dtype", choices=["preserve", "float64", "float32"], default="preserve")
    parser.add_argument(
        "--hist-backend",
        choices=["numpy", "numba", "numba-parallel", "cpp"],
        default="numpy",
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
    parser.add_argument("--cutoff-tol", type=float, default=1e-8)
    parser.add_argument("--switch-fraction", type=float, default=0.5)
    parser.add_argument("--q-block-size", type=int, default=128)
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
    parser.add_argument("--harmonic-bandlimit-margin", type=int, default=None)
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--skip-nufft", action="store_true")
    args = parser.parse_args()
    apply_fast_preset(args)

    coord_dtype = np.float32 if args.coord_dtype == "float32" else np.float64
    binning_dtype = None if args.binning_dtype == "preserve" else args.binning_dtype
    coords = synthetic_cluster(
        args.atoms,
        args.radius,
        args.height,
        args.seed,
        dtype=coord_dtype,
    )
    q = np.linspace(args.qmin, args.qmax, args.nq)
    phi = (np.arange(args.nphi) + 0.5) * (2.0 * np.pi / args.nphi)
    print(
        "case:",
        json.dumps(
            {
                "atoms": args.atoms,
                "q": [args.qmin, args.qmax, args.nq],
                "nphi": args.nphi,
                "bins": [args.nr, args.nz, args.nphi],
                "radius": args.radius,
                "height": args.height,
                "coord_dtype": args.coord_dtype,
                "binning_dtype": args.binning_dtype,
                "hist_backend": args.hist_backend,
                "hist_dtype": args.hist_dtype,
                "angle_lut_size": args.angle_lut_size,
                "angle_lut_mode": args.angle_lut_mode,
                "circular_backend": args.circular_backend,
                "complex_dtype": args.complex_dtype,
                "harmonic_bandlimit_margin": args.harmonic_bandlimit_margin,
            }
        ),
    )

    reference = None
    results = {}
    timings = {}

    if args.hist_backend.startswith("numba"):
        make_cylindrical_histogram(
            coords[: min(coords.shape[0], 128)],
            n_r=args.nr,
            n_z=args.nz,
            n_phi=args.nphi,
            r_max=args.radius,
            z_range=(-0.5 * args.height, 0.5 * args.height),
            binning_dtype=binning_dtype,
            hist_dtype=parse_hist_dtype(args.hist_dtype),
            angle_lut_size=args.angle_lut_size,
            angle_lut_mode=args.angle_lut_mode,
            backend=args.hist_backend,
        )

    binned, timings["histogram"] = timed(
        "histogram",
        lambda: make_cylindrical_histogram(
            coords,
            n_r=args.nr,
            n_z=args.nz,
            n_phi=args.nphi,
            r_max=args.radius,
            z_range=(-0.5 * args.height, 0.5 * args.height),
            binning_dtype=binning_dtype,
            hist_dtype=parse_hist_dtype(args.hist_dtype),
            angle_lut_size=args.angle_lut_size,
            angle_lut_mode=args.angle_lut_mode,
            backend=args.hist_backend,
        ),
    )

    if not args.skip_direct:
        reference, timings["direct"] = timed(
            "direct",
            lambda: direct_amplitude(coords, q, args.wavelength, phi),
        )

    plan, timings["prepare_plan"] = timed(
        "prepare_plan",
        lambda: PreparedCakePlan(
            binned,
            q,
            args.wavelength,
            cutoff_tol=args.cutoff_tol,
            switch_fraction=args.switch_fraction,
            q_block_size=args.q_block_size,
            cache_kernel_fft=args.cache_kernels,
            kernel_interpolation_dx=args.kernel_interp_dx,
            circular_backend=args.circular_backend,
            complex_dtype=parse_complex_dtype(args.complex_dtype),
            harmonic_bandlimit_margin=args.harmonic_bandlimit_margin,
        ),
    )
    circ, timings["circular_fft"] = timed(
        "circular_fft",
        lambda: plan.circular_fft(),
    )
    (jac, cutoffs), timings["jacobi"] = timed(
        "jacobi",
        lambda: plan.jacobi_anger(),
    )
    (hybrid, hybrid_cutoffs, use_jacobi), timings["hybrid"] = timed(
        "hybrid",
        lambda: plan.hybrid(),
    )

    if not args.skip_nufft:
        try:
            nufft, timings["nufft"] = timed(
                "nufft",
                lambda: nufft_amplitude(coords, q, args.wavelength, phi),
            )
            if reference is None:
                reference = nufft
            else:
                results["nufft_vs_direct"] = summarize("nufft", nufft, reference)
        except Exception as exc:
            print(f"{'nufft':>14s}: skipped ({exc})")

    if reference is not None:
        results["circular_fft"] = summarize("circular_fft", circ, reference)
        results["jacobi"] = summarize("jacobi", jac, reference)
        results["hybrid"] = summarize("hybrid", hybrid, reference)
    results["jacobi_vs_circular"] = summarize("jacobi_vs_circular", jac, circ)
    results["hybrid_vs_circular"] = summarize("hybrid_vs_circular", hybrid, circ)

    q_perp, _ = ewald_ring(q, args.wavelength)
    q_radius = q_perp * binned.r_max
    print(
        "planner:",
        json.dumps(
            {
                "cutoff_min": int(np.min(cutoffs)),
                "cutoff_max": int(np.max(cutoffs)),
                "cutoff_mean": float(np.mean(cutoffs)),
                "max_qperp_rmax": float(np.max(q_radius)),
                "angular_nyquist": int(args.nphi // 2),
                "nyquist_limited_q_count": int(np.sum(q_radius > args.nphi // 2)),
                "jacobi_q_count": int(np.sum(use_jacobi)),
                "circular_q_count": int(np.sum(~use_jacobi)),
            }
        ),
    )
    print("timings:", json.dumps(timings, indent=2))
    print("errors:", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
