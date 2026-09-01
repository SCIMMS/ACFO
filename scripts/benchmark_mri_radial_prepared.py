from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from scipy import special

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waxs_cake.metrics import relative_l2  # noqa: E402


@dataclass(frozen=True)
class RadialMriCase:
    name: str
    n: int
    n_radial: int
    n_angles: int
    kmax_fraction: float
    mmax_values: tuple[int, ...]


CASES = {
    "tiny": RadialMriCase(
        name="tiny",
        n=24,
        n_radial=18,
        n_angles=48,
        kmax_fraction=0.75,
        mmax_values=(24, 36, 48, 64),
    ),
    "small": RadialMriCase(
        name="small",
        n=32,
        n_radial=24,
        n_angles=64,
        kmax_fraction=0.85,
        mmax_values=(32, 48, 64, 80, 96),
    ),
    "medium": RadialMriCase(
        name="medium",
        n=48,
        n_radial=32,
        n_angles=96,
        kmax_fraction=0.75,
        mmax_values=(48, 64, 80, 96, 128),
    ),
    "angle_heavy": RadialMriCase(
        name="angle_heavy",
        n=32,
        n_radial=24,
        n_angles=512,
        kmax_fraction=0.85,
        mmax_values=(48, 64, 80, 96),
    ),
}


def mode_indices(n: int) -> np.ndarray:
    return np.arange(-(n // 2), n - (n // 2), dtype=np.float64)


def radial_trajectory(case: RadialMriCase) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho = np.linspace(
        np.pi * case.kmax_fraction / case.n_radial,
        np.pi * case.kmax_fraction,
        case.n_radial,
        dtype=np.float64,
    )
    theta = np.arange(case.n_angles, dtype=np.float64) * (2.0 * np.pi / case.n_angles)
    kx = rho[:, None] * np.cos(theta)[None, :]
    ky = rho[:, None] * np.sin(theta)[None, :]
    return rho, theta, kx.reshape(-1), ky.reshape(-1)


def synthetic_phantom(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coords = mode_indices(n) / max(1.0, n / 2.0)
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    image = np.zeros((n, n), dtype=np.float64)
    blobs = [
        (1.00, -0.25, -0.15, 0.28, 0.20),
        (0.75, 0.22, 0.18, 0.18, 0.30),
        (-0.35, 0.05, -0.38, 0.16, 0.12),
        (0.25, -0.42, 0.36, 0.12, 0.16),
    ]
    for amp, x0, y0, sx, sy in blobs:
        image += amp * np.exp(-0.5 * (((xx - x0) / sx) ** 2 + ((yy - y0) / sy) ** 2))
    image += 0.02 * rng.standard_normal((n, n))
    image *= np.hanning(n)[:, None] * np.hanning(n)[None, :]
    return np.ascontiguousarray(image.astype(np.complex128))


def median_time_per_call(func, *, repeats: int, loops: int, warmups: int = 1) -> tuple[Any, float, list[float]]:
    value = None
    for _ in range(warmups):
        for _ in range(loops):
            value = func()
    times: list[float] = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        for _ in range(loops):
            value = func()
        times.append((time.perf_counter() - start) / float(loops))
    return value, float(median(times)), times


def build_direct_matrix(
    n: int,
    kx: np.ndarray,
    ky: np.ndarray,
    *,
    isign: int,
) -> np.ndarray:
    ii, jj = np.meshgrid(mode_indices(n), mode_indices(n), indexing="ij")
    phase = kx[:, None] * ii.reshape(1, -1) + ky[:, None] * jj.reshape(1, -1)
    return np.exp(1j * float(isign) * phase)


def direct_execute(matrix: np.ndarray, image: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    return (matrix @ image.reshape(-1)).reshape(out_shape)


@dataclass
class HarmonicGeometry:
    n: int
    rho: np.ndarray
    theta: np.ndarray
    m_values: np.ndarray
    radius_values: np.ndarray
    group_indices: list[np.ndarray]
    group_phase: list[np.ndarray]
    bessel: np.ndarray
    angular: np.ndarray
    mode_phase: np.ndarray
    cache_mib: float


def build_harmonic_geometry(
    n: int,
    rho: np.ndarray,
    theta: np.ndarray,
    *,
    mmax: int,
    isign: int,
) -> HarmonicGeometry:
    ii, jj = np.meshgrid(mode_indices(n), mode_indices(n), indexing="ij")
    ii_flat = ii.reshape(-1)
    jj_flat = jj.reshape(-1)
    radius_sq = (ii_flat.astype(np.int64) ** 2) + (jj_flat.astype(np.int64) ** 2)
    beta = np.arctan2(jj_flat, ii_flat)
    unique_r2 = np.unique(radius_sq)
    group_indices: list[np.ndarray] = []
    group_phase: list[np.ndarray] = []
    m_values = np.arange(-mmax, mmax + 1, dtype=np.int64)
    for r2 in unique_r2:
        idx = np.flatnonzero(radius_sq == r2)
        group_indices.append(idx)
        group_phase.append(np.exp(-1j * np.outer(m_values, beta[idx])))

    radius_values = np.sqrt(unique_r2.astype(np.float64))
    z = rho[:, None, None] * radius_values[None, :, None]
    bessel = special.jv(m_values[None, None, :], z)
    angular = np.exp(1j * np.outer(m_values, theta))
    mode_phase = np.exp(0.5j * float(isign) * np.pi * m_values)

    cache_bytes = bessel.nbytes + angular.nbytes + mode_phase.nbytes + radius_values.nbytes
    cache_bytes += sum(arr.nbytes for arr in group_phase)
    return HarmonicGeometry(
        n=n,
        rho=rho,
        theta=theta,
        m_values=m_values,
        radius_values=radius_values,
        group_indices=group_indices,
        group_phase=group_phase,
        bessel=bessel,
        angular=angular,
        mode_phase=mode_phase,
        cache_mib=float(cache_bytes / (1024.0 * 1024.0)),
    )


def harmonic_execute(geom: HarmonicGeometry, image: np.ndarray) -> np.ndarray:
    flat = image.reshape(-1)
    grouped_modes = np.empty(
        (len(geom.group_indices), geom.m_values.size),
        dtype=np.complex128,
    )
    for group_index, (idx, phase) in enumerate(zip(geom.group_indices, geom.group_phase)):
        grouped_modes[group_index, :] = phase @ flat[idx]

    radial_modes = np.einsum("rgm,gm->rm", geom.bessel, grouped_modes, optimize=True)
    radial_modes *= geom.mode_phase[None, :]
    return radial_modes @ geom.angular


def build_finufft_plan(n: int, kx: np.ndarray, ky: np.ndarray, *, eps: float, isign: int):
    try:
        import finufft
    except ImportError as exc:  # pragma: no cover - dependency controlled by env
        raise RuntimeError("finufft is required for this benchmark") from exc

    plan = finufft.Plan(2, (n, n), 1, eps=eps, isign=isign)
    plan.setpts(kx, ky)
    return plan


def finufft_execute(plan, image: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    return plan.execute(image).reshape(out_shape)


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(relative_l2(np.asarray(a), np.asarray(b)))


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def run_case(case: RadialMriCase, *, args: argparse.Namespace) -> dict[str, Any]:
    image = synthetic_phantom(case.n, args.seed)
    rho, theta, kx, ky = radial_trajectory(case)
    out_shape = (rho.size, theta.size)

    start = time.perf_counter()
    fin_plan = build_finufft_plan(case.n, kx, ky, eps=args.eps, isign=args.isign)
    fin_build_s = time.perf_counter() - start
    fin_out, fin_s, fin_times = median_time_per_call(
        lambda: finufft_execute(fin_plan, image, out_shape),
        repeats=args.repeats,
        loops=args.loops,
    )

    direct_allowed = (kx.size * case.n * case.n) <= args.direct_max_elements
    direct_out = None
    direct_s = None
    direct_build_s = None
    direct_times: list[float] = []
    direct_rel_l2_vs_finufft = None
    dense_cache_mib = None
    if direct_allowed:
        start = time.perf_counter()
        direct_matrix = build_direct_matrix(case.n, kx, ky, isign=args.isign)
        direct_build_s = time.perf_counter() - start
        dense_cache_mib = float(direct_matrix.nbytes / (1024.0 * 1024.0))
        direct_out, direct_s, direct_times = median_time_per_call(
            lambda: direct_execute(direct_matrix, image, out_shape),
            repeats=args.repeats,
            loops=args.loops,
        )
        direct_rel_l2_vs_finufft = rel_l2(direct_out, fin_out)

    reference = direct_out if direct_out is not None else fin_out
    reference_name = "direct_dense" if direct_out is not None else "finufft_plan"

    rows: list[dict[str, Any]] = []
    for mmax in case.mmax_values if args.mmax is None else tuple(args.mmax):
        start = time.perf_counter()
        geom = build_harmonic_geometry(case.n, rho, theta, mmax=mmax, isign=args.isign)
        build_s = time.perf_counter() - start
        harmonic_out, hot_s, hot_times = median_time_per_call(
            lambda geom=geom: harmonic_execute(geom, image),
            repeats=args.repeats,
            loops=args.loops,
        )
        rows.append(
            {
                "mmax": int(mmax),
                "modes": int(2 * mmax + 1),
                "radius_groups": int(geom.radius_values.size),
                "build_s": float(build_s),
                "hot_s": float(hot_s),
                "hot_times_s": hot_times,
                "cache_mib": geom.cache_mib,
                "rel_l2_vs_reference": rel_l2(harmonic_out, reference),
                "rel_l2_vs_finufft": rel_l2(harmonic_out, fin_out),
                "speedup_vs_finufft_hot": float(fin_s / hot_s) if hot_s else float("inf"),
                "speedup_vs_direct_hot": None if direct_s is None else float(direct_s / hot_s),
            }
        )

    return {
        "case": asdict(case),
        "trajectory_samples": int(kx.size),
        "image_coefficients": int(case.n * case.n),
        "isign": int(args.isign),
        "eps": float(args.eps),
        "reference": reference_name,
        "finufft_plan_build_s": float(fin_build_s),
        "finufft_hot_s": float(fin_s),
        "finufft_hot_times_s": fin_times,
        "direct_allowed": bool(direct_allowed),
        "direct_dense_build_s": direct_build_s,
        "direct_dense_hot_s": direct_s,
        "direct_dense_hot_times_s": direct_times,
        "direct_dense_cache_mib": dense_cache_mib,
        "direct_rel_l2_vs_finufft": direct_rel_l2_vs_finufft,
        "harmonic_rows": rows,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Radial MRI prepared-operator benchmark",
        "",
        "This is a small feasibility benchmark for applying the prepared harmonic",
        "operator idea to a 2D radial non-Cartesian MRI-style Fourier operator.",
        "It is not a clinical MRI reconstruction benchmark.",
        "",
        "The FINUFFT baseline uses a reused type-2 plan with radial sample points",
        "set once. The harmonic path groups Cartesian Fourier/image coefficients",
        "by integer radius and evaluates a truncated Jacobi-Anger expansion.",
        "",
    ]
    for result in payload["results"]:
        case = result["case"]
        lines.extend(
            [
                f"## Case `{case['name']}`",
                "",
                "| field | value |",
                "| --- | ---: |",
                f"| grid | `{case['n']} x {case['n']}` |",
                f"| radial samples | `{case['n_radial']}` |",
                f"| angles | `{case['n_angles']}` |",
                f"| trajectory samples | `{result['trajectory_samples']}` |",
                f"| FINUFFT plan build s | `{result['finufft_plan_build_s']:.6g}` |",
                f"| FINUFFT hot s | `{result['finufft_hot_s']:.6g}` |",
            ]
        )
        if result["direct_allowed"]:
            lines.extend(
                [
                    f"| dense direct build s | `{result['direct_dense_build_s']:.6g}` |",
                    f"| dense direct hot s | `{result['direct_dense_hot_s']:.6g}` |",
                    f"| dense direct cache MiB | `{result['direct_dense_cache_mib']:.3f}` |",
                    f"| direct rel-L2 vs FINUFFT | `{result['direct_rel_l2_vs_finufft']:.3g}` |",
                ]
            )
        lines.extend(
            [
                "",
                "| mmax | modes | radius groups | build s | hot s | cache MiB | rel-L2 vs ref | hot speedup vs FINUFFT |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["harmonic_rows"]:
            lines.append(
                "| {mmax} | {modes} | {radius_groups} | {build_s:.6g} | {hot_s:.6g} | "
                "{cache_mib:.3f} | {rel_l2_vs_reference:.3g} | {speedup_vs_finufft_hot:.3g} |".format(
                    **row
                )
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Small 2D radial MRI feasibility benchmark for a prepared harmonic operator.",
    )
    p.add_argument("--cases", nargs="+", choices=sorted(CASES), default=["small"])
    p.add_argument("--mmax", nargs="*", type=int, default=None)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--loops", type=int, default=5)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--eps", type=float, default=1e-12)
    p.add_argument("--isign", type=int, choices=[-1, 1], default=-1)
    p.add_argument("--direct-max-elements", type=int, default=10_000_000)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmark_results" / "mri_radial_prepared_small.json",
    )
    p.add_argument(
        "--summary-md",
        type=Path,
        default=ROOT / "benchmark_results" / "mri_radial_prepared_small.md",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    results = []
    for name in args.cases:
        print(f"[mri-radial] running case {name}")
        result = run_case(CASES[name], args=args)
        results.append(result)
        best = min(result["harmonic_rows"], key=lambda row: row["rel_l2_vs_reference"])
        print(
            "  finufft_hot={:.6g}s best_mmax={} best_err={:.3g} best_hot={:.6g}s".format(
                result["finufft_hot_s"],
                best["mmax"],
                best["rel_l2_vs_reference"],
                best["hot_s"],
            )
        )

    payload = {
        "config": vars(args),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    if args.summary_md:
        write_markdown(args.summary_md, payload)
    print(json.dumps({"out": str(args.out), "summary_md": str(args.summary_md)}, indent=2))


if __name__ == "__main__":
    main()
