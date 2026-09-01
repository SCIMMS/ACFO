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
from scipy import sparse, special

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waxs_cake.metrics import relative_l2  # noqa: E402


@dataclass(frozen=True)
class StackOfStarsCase:
    name: str
    n_xy: int
    n_z: int
    n_radial: int
    n_angles: int
    n_kz: int
    kmax_fraction: float
    mmax_values: tuple[int, ...]


CASES = {
    "tiny": StackOfStarsCase(
        name="tiny",
        n_xy=20,
        n_z=12,
        n_radial=16,
        n_angles=48,
        n_kz=12,
        kmax_fraction=0.80,
        mmax_values=(20, 30, 40, 50),
    ),
    "small": StackOfStarsCase(
        name="small",
        n_xy=32,
        n_z=16,
        n_radial=24,
        n_angles=64,
        n_kz=16,
        kmax_fraction=0.85,
        mmax_values=(32, 48, 64, 80),
    ),
    "angle_z_heavy": StackOfStarsCase(
        name="angle_z_heavy",
        n_xy=32,
        n_z=24,
        n_radial=24,
        n_angles=256,
        n_kz=24,
        kmax_fraction=0.85,
        mmax_values=(48, 64, 80, 96),
    ),
}


def mode_indices(n: int) -> np.ndarray:
    return np.arange(-(n // 2), n - (n // 2), dtype=np.float64)


def stack_of_stars_trajectory(
    case: StackOfStarsCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho = np.linspace(
        np.pi * case.kmax_fraction / case.n_radial,
        np.pi * case.kmax_fraction,
        case.n_radial,
        dtype=np.float64,
    )
    theta = np.arange(case.n_angles, dtype=np.float64) * (2.0 * np.pi / case.n_angles)
    kz_modes = mode_indices(case.n_kz)
    kz = 2.0 * np.pi * kz_modes / float(case.n_z)
    kx = rho[:, None, None] * np.cos(theta)[None, :, None] * np.ones(
        (1, 1, case.n_kz),
        dtype=np.float64,
    )
    ky = rho[:, None, None] * np.sin(theta)[None, :, None] * np.ones(
        (1, 1, case.n_kz),
        dtype=np.float64,
    )
    kz_grid = np.ones((case.n_radial, case.n_angles, 1), dtype=np.float64) * kz[None, None, :]
    return rho, theta, kz, kx.reshape(-1), ky.reshape(-1), kz_grid.reshape(-1)


def synthetic_volume(n_xy: int, n_z: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xy = mode_indices(n_xy) / max(1.0, n_xy / 2.0)
    zz = mode_indices(n_z) / max(1.0, n_z / 2.0)
    x, y, z = np.meshgrid(xy, xy, zz, indexing="ij")
    volume = np.zeros((n_xy, n_xy, n_z), dtype=np.float64)
    blobs = [
        (1.00, -0.24, -0.14, -0.18, 0.30, 0.20, 0.26),
        (0.65, 0.22, 0.20, 0.24, 0.18, 0.28, 0.20),
        (-0.30, 0.02, -0.36, 0.10, 0.16, 0.12, 0.30),
        (0.24, -0.40, 0.34, -0.28, 0.12, 0.14, 0.18),
    ]
    for amp, x0, y0, z0, sx, sy, sz in blobs:
        volume += amp * np.exp(
            -0.5 * (((x - x0) / sx) ** 2 + ((y - y0) / sy) ** 2 + ((z - z0) / sz) ** 2)
        )
    window = np.hanning(n_xy)[:, None, None] * np.hanning(n_xy)[None, :, None] * np.hanning(n_z)[None, None, :]
    volume = (volume + 0.01 * rng.standard_normal(volume.shape)) * window
    return np.ascontiguousarray(volume.astype(np.complex128))


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


def build_finufft3d_plan(
    n_xy: int,
    n_z: int,
    kx: np.ndarray,
    ky: np.ndarray,
    kz: np.ndarray,
    *,
    eps: float,
    isign: int,
):
    try:
        import finufft
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("finufft is required for this benchmark") from exc

    plan = finufft.Plan(2, (n_xy, n_xy, n_z), 1, eps=eps, isign=isign)
    plan.setpts(kx, ky, kz)
    return plan


def finufft3d_execute(plan, volume: np.ndarray, out_shape: tuple[int, int, int]) -> np.ndarray:
    return plan.execute(volume).reshape(out_shape)


@dataclass
class SeparableFinufft2dPlan:
    plan: Any
    z_phase: np.ndarray
    out_shape: tuple[int, int, int]
    cache_mib: float


def build_separable_finufft2d_plan(
    case: StackOfStarsCase,
    rho: np.ndarray,
    theta: np.ndarray,
    kz: np.ndarray,
    *,
    eps: float,
    isign: int,
) -> SeparableFinufft2dPlan:
    try:
        import finufft
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("finufft is required for this benchmark") from exc

    kx2 = (rho[:, None] * np.cos(theta)[None, :]).reshape(-1)
    ky2 = (rho[:, None] * np.sin(theta)[None, :]).reshape(-1)
    plan = finufft.Plan(2, (case.n_xy, case.n_xy), case.n_kz, eps=eps, isign=isign)
    plan.setpts(kx2, ky2)
    z_index = mode_indices(case.n_z)
    z_phase = np.exp(1j * float(isign) * np.outer(kz, z_index))
    cache_mib = float(z_phase.nbytes / (1024.0 * 1024.0))
    return SeparableFinufft2dPlan(
        plan=plan,
        z_phase=z_phase,
        out_shape=(case.n_radial, case.n_angles, case.n_kz),
        cache_mib=cache_mib,
    )


def separable_finufft2d_execute(plan: SeparableFinufft2dPlan, volume: np.ndarray) -> np.ndarray:
    z_transformed = np.einsum("xyz,kz->xyk", volume, plan.z_phase, optimize=True)
    batch = np.ascontiguousarray(np.moveaxis(z_transformed, 2, 0))
    out = plan.plan.execute(batch)
    return out.reshape(plan.z_phase.shape[0], plan.out_shape[0], plan.out_shape[1]).transpose(1, 2, 0)


@dataclass
class HarmonicStackGeometry:
    case: StackOfStarsCase
    m_values: np.ndarray
    radius_values: np.ndarray
    group_matrix: sparse.csr_matrix
    bessel: np.ndarray
    angular: np.ndarray | None
    z_phase: np.ndarray
    mode_phase: np.ndarray
    angular_backend: str
    z_order: str
    fft_mode_indices: np.ndarray
    fft_has_alias: bool
    cache_mib: float


def build_group_matrix(n_xy: int, m_values: np.ndarray) -> tuple[np.ndarray, sparse.csr_matrix]:
    ix, iy = np.meshgrid(mode_indices(n_xy), mode_indices(n_xy), indexing="ij")
    ix_flat = ix.reshape(-1)
    iy_flat = iy.reshape(-1)
    radius_sq = (ix_flat.astype(np.int64) ** 2) + (iy_flat.astype(np.int64) ** 2)
    beta = np.arctan2(iy_flat, ix_flat)
    unique_r2 = np.unique(radius_sq)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    for group_idx, r2 in enumerate(unique_r2):
        idx = np.flatnonzero(radius_sq == r2)
        phase = np.exp(-1j * np.outer(m_values, beta[idx]))
        base_rows = group_idx * m_values.size + np.arange(m_values.size)
        rows.append(np.repeat(base_rows, idx.size))
        cols.append(np.tile(idx, m_values.size))
        data.append(phase.reshape(-1))

    matrix = sparse.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(unique_r2.size * m_values.size, n_xy * n_xy),
    )
    return np.sqrt(unique_r2.astype(np.float64)), matrix


def build_harmonic_stack_geometry(
    case: StackOfStarsCase,
    rho: np.ndarray,
    theta: np.ndarray,
    kz: np.ndarray,
    *,
    mmax: int,
    isign: int,
    angular_backend: str,
    z_order: str,
) -> HarmonicStackGeometry:
    m_values = np.arange(-mmax, mmax + 1, dtype=np.int64)
    radius_values, group_matrix = build_group_matrix(case.n_xy, m_values)
    bessel = special.jv(m_values[None, None, :], rho[:, None, None] * radius_values[None, :, None])
    angular = None
    if angular_backend == "dense":
        angular = np.exp(1j * np.outer(m_values, theta))
    z_phase = np.exp(1j * float(isign) * np.outer(kz, mode_indices(case.n_z)))
    mode_phase = np.exp(0.5j * float(isign) * np.pi * m_values)
    fft_mode_indices = np.mod(m_values, case.n_angles).astype(np.int64)
    fft_has_alias = bool(np.unique(fft_mode_indices).size != fft_mode_indices.size)
    cache_bytes = (
        group_matrix.data.nbytes
        + group_matrix.indices.nbytes
        + group_matrix.indptr.nbytes
        + bessel.nbytes
        + z_phase.nbytes
        + mode_phase.nbytes
        + radius_values.nbytes
    )
    if angular is not None:
        cache_bytes += angular.nbytes
    return HarmonicStackGeometry(
        case=case,
        m_values=m_values,
        radius_values=radius_values,
        group_matrix=group_matrix,
        bessel=bessel,
        angular=angular,
        z_phase=z_phase,
        mode_phase=mode_phase,
        angular_backend=angular_backend,
        z_order=z_order,
        fft_mode_indices=fft_mode_indices,
        fft_has_alias=fft_has_alias,
        cache_mib=float(cache_bytes / (1024.0 * 1024.0)),
    )


def _angular_fft_from_modes(geom: HarmonicStackGeometry, kz_modes: np.ndarray) -> np.ndarray:
    coeffs = np.zeros(
        (geom.case.n_radial, geom.case.n_kz, geom.case.n_angles),
        dtype=np.complex128,
    )
    values = kz_modes.transpose(0, 2, 1)
    if geom.fft_has_alias:
        np.add.at(coeffs, (slice(None), slice(None), geom.fft_mode_indices), values)
    else:
        coeffs[:, :, geom.fft_mode_indices] = values
    angular = geom.case.n_angles * np.fft.ifft(coeffs, axis=2)
    return angular.transpose(0, 2, 1)


def harmonic_stack_execute_z_late(geom: HarmonicStackGeometry, volume: np.ndarray) -> np.ndarray:
    flat_xy_z = volume.reshape(geom.case.n_xy * geom.case.n_xy, geom.case.n_z)
    grouped = geom.group_matrix @ flat_xy_z
    grouped = grouped.reshape(geom.radius_values.size, geom.m_values.size, geom.case.n_z)
    radial_modes = np.einsum("rgm,gmz->rmz", geom.bessel, grouped, optimize=True)
    radial_modes *= geom.mode_phase[None, :, None]
    kz_modes = np.einsum("rmz,kz->rmk", radial_modes, geom.z_phase, optimize=True)
    if geom.angular_backend == "dense":
        if geom.angular is None:
            raise RuntimeError("dense angular backend requires precomputed angular matrix")
        return np.einsum("rmk,mt->rtk", kz_modes, geom.angular, optimize=True)
    return _angular_fft_from_modes(geom, kz_modes)


def harmonic_stack_execute_z_first(geom: HarmonicStackGeometry, volume: np.ndarray) -> np.ndarray:
    z_transformed = np.einsum("xyz,kz->xyk", volume, geom.z_phase, optimize=True)
    flat_xy_k = z_transformed.reshape(geom.case.n_xy * geom.case.n_xy, geom.case.n_kz)
    grouped = geom.group_matrix @ flat_xy_k
    grouped = grouped.reshape(geom.radius_values.size, geom.m_values.size, geom.case.n_kz)
    kz_modes = np.einsum("rgm,gmk->rmk", geom.bessel, grouped, optimize=True)
    kz_modes *= geom.mode_phase[None, :, None]
    if geom.angular_backend == "dense":
        if geom.angular is None:
            raise RuntimeError("dense angular backend requires precomputed angular matrix")
        return np.einsum("rmk,mt->rtk", kz_modes, geom.angular, optimize=True)
    return _angular_fft_from_modes(geom, kz_modes)


def harmonic_stack_execute(geom: HarmonicStackGeometry, volume: np.ndarray) -> np.ndarray:
    if geom.z_order == "first":
        return harmonic_stack_execute_z_first(geom, volume)
    return harmonic_stack_execute_z_late(geom, volume)


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


def run_case(case: StackOfStarsCase, *, args: argparse.Namespace) -> dict[str, Any]:
    volume = synthetic_volume(case.n_xy, case.n_z, args.seed)
    rho, theta, kz, kx_flat, ky_flat, kz_flat = stack_of_stars_trajectory(case)
    out_shape = (case.n_radial, case.n_angles, case.n_kz)

    start = time.perf_counter()
    plan3d = build_finufft3d_plan(
        case.n_xy,
        case.n_z,
        kx_flat,
        ky_flat,
        kz_flat,
        eps=args.eps,
        isign=args.isign,
    )
    fin3d_build_s = time.perf_counter() - start
    fin3d_out, fin3d_s, fin3d_times = median_time_per_call(
        lambda: finufft3d_execute(plan3d, volume, out_shape),
        repeats=args.repeats,
        loops=args.loops,
    )

    start = time.perf_counter()
    sep_plan = build_separable_finufft2d_plan(
        case,
        rho,
        theta,
        kz,
        eps=args.eps,
        isign=args.isign,
    )
    sep_build_s = time.perf_counter() - start
    sep_out, sep_s, sep_times = median_time_per_call(
        lambda: separable_finufft2d_execute(sep_plan, volume),
        repeats=args.repeats,
        loops=args.loops,
    )

    rows: list[dict[str, Any]] = []
    for mmax in case.mmax_values if args.mmax is None else tuple(args.mmax):
        start = time.perf_counter()
        geom = build_harmonic_stack_geometry(
            case,
            rho,
            theta,
            kz,
            mmax=mmax,
            isign=args.isign,
            angular_backend=args.angular_backend,
            z_order=args.z_order,
        )
        build_s = time.perf_counter() - start
        harmonic_out, hot_s, hot_times = median_time_per_call(
            lambda geom=geom: harmonic_stack_execute(geom, volume),
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
                "angular_backend": geom.angular_backend,
                "z_order": geom.z_order,
                "rel_l2_vs_finufft3d": rel_l2(harmonic_out, fin3d_out),
                "rel_l2_vs_separable": rel_l2(harmonic_out, sep_out),
                "speedup_vs_finufft3d_hot": float(fin3d_s / hot_s) if hot_s else float("inf"),
                "speedup_vs_separable_hot": float(sep_s / hot_s) if hot_s else float("inf"),
            }
        )

    return {
        "case": asdict(case),
        "trajectory_samples": int(kx_flat.size),
        "volume_coefficients": int(case.n_xy * case.n_xy * case.n_z),
        "isign": int(args.isign),
        "eps": float(args.eps),
        "finufft3d_plan_build_s": float(fin3d_build_s),
        "finufft3d_hot_s": float(fin3d_s),
        "finufft3d_hot_times_s": fin3d_times,
        "separable_plan_build_s": float(sep_build_s),
        "separable_hot_s": float(sep_s),
        "separable_hot_times_s": sep_times,
        "separable_cache_mib": sep_plan.cache_mib,
        "separable_rel_l2_vs_finufft3d": rel_l2(sep_out, fin3d_out),
        "harmonic_rows": rows,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Stack-of-stars MRI prepared-operator benchmark",
        "",
        "This benchmark tests the favorable 3D radial-stack phase structure",
        "`rho R cos(theta-beta) + k_z z`.",
        "",
        "Baselines:",
        "",
        "- `FINUFFT 3D plan`: reused type-2 plan for all stack-of-stars points.",
        "- `separable 2D FINUFFT`: explicit z transform plus batched reused 2D radial FINUFFT plan.",
        "- `harmonic prepared`: radius-grouped Jacobi-Anger contraction plus z phase reuse.",
        "",
        "This is a toy operator benchmark, not a full clinical MRI reconstruction.",
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
                f"| grid | `{case['n_xy']} x {case['n_xy']} x {case['n_z']}` |",
                f"| radial samples | `{case['n_radial']}` |",
                f"| angles | `{case['n_angles']}` |",
                f"| kz samples | `{case['n_kz']}` |",
                f"| trajectory samples | `{result['trajectory_samples']}` |",
                f"| FINUFFT 3D build s | `{result['finufft3d_plan_build_s']:.6g}` |",
                f"| FINUFFT 3D hot s | `{result['finufft3d_hot_s']:.6g}` |",
                f"| separable build s | `{result['separable_plan_build_s']:.6g}` |",
                f"| separable hot s | `{result['separable_hot_s']:.6g}` |",
                f"| separable rel-L2 vs FINUFFT 3D | `{result['separable_rel_l2_vs_finufft3d']:.3g}` |",
                "",
        "| mmax | modes | radius groups | angular | z order | build s | hot s | cache MiB | rel-L2 vs FINUFFT 3D | speedup vs FINUFFT 3D | speedup vs separable |",
        "| ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["harmonic_rows"]:
            lines.append(
                "| {mmax} | {modes} | {radius_groups} | {angular_backend} | {z_order} | {build_s:.6g} | {hot_s:.6g} | "
                "{cache_mib:.3f} | {rel_l2_vs_finufft3d:.3g} | "
                "{speedup_vs_finufft3d_hot:.3g} | {speedup_vs_separable_hot:.3g} |".format(**row)
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Toy 3D stack-of-stars MRI benchmark for favorable prepared harmonic geometry.",
    )
    p.add_argument("--cases", nargs="+", choices=sorted(CASES), default=["small"])
    p.add_argument("--mmax", nargs="*", type=int, default=None)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--loops", type=int, default=3)
    p.add_argument("--seed", type=int, default=31415)
    p.add_argument("--eps", type=float, default=1e-12)
    p.add_argument("--isign", type=int, choices=[-1, 1], default=-1)
    p.add_argument("--angular-backend", choices=["fft", "dense"], default="fft")
    p.add_argument("--z-order", choices=["first", "late"], default="first")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmark_results" / "mri_stack_of_stars_prepared.json",
    )
    p.add_argument(
        "--summary-md",
        type=Path,
        default=ROOT / "benchmark_results" / "mri_stack_of_stars_prepared.md",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    results = []
    for name in args.cases:
        print(f"[mri-stack] running case {name}")
        result = run_case(CASES[name], args=args)
        results.append(result)
        best = min(result["harmonic_rows"], key=lambda row: row["rel_l2_vs_finufft3d"])
        print(
            "  fin3d={:.6g}s sep={:.6g}s best_mmax={} best_err={:.3g} best_hot={:.6g}s".format(
                result["finufft3d_hot_s"],
                result["separable_hot_s"],
                best["mmax"],
                best["rel_l2_vs_finufft3d"],
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
