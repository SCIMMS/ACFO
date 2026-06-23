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

from waxs_cake import (  # noqa: E402
    PreparedCakePlan,
    cylindrical_flat_indices,
    make_cylindrical_histogram,
    make_cylindrical_histogram_from_flat_indices,
)
from waxs_cake.metrics import relative_l2  # noqa: E402


def synthetic_cylinder(
    n_atoms: int,
    radius: float,
    occupied_height: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.random(n_atoms))
    beta = 2.0 * np.pi * rng.random(n_atoms)
    z = occupied_height * (rng.random(n_atoms) - 0.5)
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


def parse_dtype(value: str) -> np.dtype | None:
    return None if value in {"default", "auto"} else np.dtype(value)


def precompute_indices(
    coords: np.ndarray,
    *,
    n_r: int,
    n_z: int,
    n_phi: int,
    r_max: float,
    z_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    radius = np.sqrt(x * x + y * y)
    r_idx = (radius * (n_r / r_max)).astype(np.intp)
    z_idx = ((z - z_range[0]) * (n_z / (z_range[1] - z_range[0]))).astype(np.intp)
    beta = np.arctan2(y, x)
    beta[beta < 0.0] += 2.0 * np.pi
    beta_idx = (beta * (n_phi / (2.0 * np.pi))).astype(np.intp)
    np.clip(r_idx, 0, n_r - 1, out=r_idx)
    np.clip(z_idx, 0, n_z - 1, out=z_idx)
    np.clip(beta_idx, 0, n_phi - 1, out=beta_idx)
    return r_idx, z_idx, beta_idx


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
    parser.add_argument("--occupied-height", type=float, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument(
        "--hist-dtype",
        choices=["default", "uint32", "float32", "float64"],
        default="float32",
    )
    parser.add_argument(
        "--complex-dtype",
        choices=["auto", "complex64", "complex128"],
        default="auto",
    )
    parser.add_argument(
        "--circular-backend",
        choices=["auto", "numpy", "cpp"],
        default="cpp",
    )
    parser.add_argument("--q-block-size", type=int, default=128)
    parser.add_argument("--active-chunk-size", type=int, default=256)
    parser.add_argument("--angle-lut-size", type=int, default=0)
    parser.add_argument(
        "--angle-lut-mode",
        choices=["nearest", "cubic"],
        default="nearest",
    )
    parser.add_argument(
        "--skip-index-validation",
        action="store_true",
        help="Skip range validation when accumulating trusted precomputed flat indices.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark_results/precomputed_sparse.json"),
    )
    args = parser.parse_args()

    occupied_height = args.height if args.occupied_height is None else args.occupied_height
    coords = synthetic_cylinder(args.atoms, args.radius, occupied_height, args.seed)
    q = np.linspace(args.qmin, args.qmax, args.nq)
    z_range = (-0.5 * args.height, 0.5 * args.height)
    hist_dtype = parse_dtype(args.hist_dtype)
    complex_dtype = parse_dtype(args.complex_dtype)

    hist_kwargs = {
        "n_r": args.nr,
        "n_z": args.nz,
        "n_phi": args.nphi,
        "r_max": args.radius,
        "z_range": z_range,
        "hist_dtype": hist_dtype,
        "backend": "cpp",
    }

    binned, hist_s, hist_times = median_time(
        lambda: make_cylindrical_histogram(coords, **hist_kwargs),
        args.repeats,
    )
    flat_indices, index_s, index_times = median_time(
        lambda: cylindrical_flat_indices(
            coords,
            n_r=args.nr,
            n_z=args.nz,
            n_phi=args.nphi,
            r_max=args.radius,
            z_range=z_range,
            backend="cpp",
            angle_lut_size=args.angle_lut_size,
            angle_lut_mode=args.angle_lut_mode,
        ),
        args.repeats,
    )
    binned_from_idx, accumulate_s, accumulate_times = median_time(
        lambda: make_cylindrical_histogram_from_flat_indices(
            flat_indices,
            n_r=args.nr,
            n_z=args.nz,
            n_phi=args.nphi,
            r_max=args.radius,
            z_range=z_range,
            hist_dtype=hist_dtype,
            backend="cpp",
            validate_indices=not args.skip_index_validation,
        ),
        args.repeats,
    )

    plan, plan_s, plan_times = median_time(
        lambda: PreparedCakePlan(
            binned,
            q,
            1.0,
            circular_backend=args.circular_backend,
            complex_dtype=complex_dtype,
            q_block_size=args.q_block_size,
        ),
        args.repeats,
    )
    dense, dense_s, dense_times = median_time(plan.circular_fft, args.repeats)
    sparse, sparse_s, sparse_times = median_time(
        lambda: plan.circular_fft_sparse_rz(active_chunk_size=args.active_chunk_size),
        args.repeats,
    )

    result = {
        "case": {
            "atoms": args.atoms,
            "nq": args.nq,
            "nphi": args.nphi,
            "nr": args.nr,
            "nz": args.nz,
            "height": args.height,
            "occupied_height": occupied_height,
            "hist_dtype": str(binned.hist.dtype),
            "complex_dtype": str(plan.complex_dtype),
            "active_chunk_size": args.active_chunk_size,
            "angle_lut_size": args.angle_lut_size,
            "angle_lut_mode": args.angle_lut_mode,
            "validate_indices": not args.skip_index_validation,
        },
        "active_rz_count": plan.active_rz_count,
        "total_rz_count": int(binned.hist.shape[0] * binned.hist.shape[1] * binned.hist.shape[2]),
        "hist_s": hist_s,
        "index_precompute_s": index_s,
        "index_accumulate_s": accumulate_s,
        "index_total_s": index_s + accumulate_s,
        "plan_s": plan_s,
        "dense_s": dense_s,
        "sparse_s": sparse_s,
        "sparse_rel_l2_vs_dense": relative_l2(sparse, dense),
        "from_indices_exact": bool(np.array_equal(binned_from_idx.hist, binned.hist)),
        "from_indices_rel_l2": relative_l2(binned_from_idx.hist, binned.hist),
        "hist_times": hist_times,
        "index_times": index_times,
        "accumulate_times": accumulate_times,
        "plan_times": plan_times,
        "dense_times": dense_times,
        "sparse_times": sparse_times,
    }

    print(
        "active_rz="
        f"{result['active_rz_count']}/{result['total_rz_count']} "
        f"hist={hist_s:.5f}s idx={index_s:.5f}s acc={accumulate_s:.5f}s "
        f"dense={dense_s:.5f}s sparse={sparse_s:.5f}s "
        f"err={result['sparse_rel_l2_vs_dense']:.3g}"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
