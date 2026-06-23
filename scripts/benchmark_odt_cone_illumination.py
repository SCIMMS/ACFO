from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from benchmark_odt_ewald_cap_operator import (
    ROOT,
    QSamples,
    ShiftedAxisFactorization,
    StructuredOdtPlan,
    _axis_grid_adjoint_fft_compact,
    _axis_grid_forward_fft,
    build_shifted_axis_phases,
    build_structured_kernel,
    complex_dot,
    detector_directions,
    make_cylindrical_object,
    recommended_h_cutoff,
    relative_complex_error,
    relative_l2,
    resolve_structured_backend,
    structured_adjoint,
    structured_forward,
)


@dataclass(frozen=True)
class ConeModeCache:
    base_q: QSamples
    flat_q: QSamples
    factorization: ShiftedAxisFactorization
    source_factors: np.ndarray
    weights: np.ndarray
    mode_orders: np.ndarray
    flat_kernel: Any
    flat_plan: StructuredOdtPlan
    cone_plan: StructuredOdtPlan


def parse_int_list(value: str) -> list[int]:
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not out or any(item <= 0 for item in out):
        raise ValueError("expected positive comma-separated integers")
    return out


def parse_float_list(value: str) -> list[float]:
    out = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not out or any(item < 0.0 for item in out):
        raise ValueError("expected non-negative comma-separated floats")
    return out


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    value_f = float(value)
    if value_f == 0.0:
        return "0"
    if abs(value_f) < 1e-3 or abs(value_f) >= 1e3:
        return f"{value_f:.{digits}e}"
    return f"{value_f:.{digits}f}"


def median_time(func, *, repeats: int) -> tuple[Any, float, list[float]]:
    result = None
    times: list[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        result = func()
        times.append(time.perf_counter() - start)
    if result is None:
        raise RuntimeError("timed function did not run")
    return result, float(median(times)), times


def speedup(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def kernel_mib(kernel: Any) -> float:
    return (kernel.radial.nbytes + kernel.axial.nbytes + kernel.angular.nbytes) / (
        1024.0 * 1024.0
    )


def cone_illumination_directions(*, n_illum: int, illumination_na: float) -> tuple[np.ndarray, np.ndarray]:
    if n_illum <= 0:
        raise ValueError("n_illum must be positive")
    if illumination_na < 0.0 or illumination_na >= 1.0:
        raise ValueError("illumination_na must be in [0, 1)")
    phi = np.linspace(0.0, 2.0 * np.pi, n_illum, endpoint=False, dtype=float)
    sx = illumination_na * np.cos(phi)
    sy = illumination_na * np.sin(phi)
    sz = np.sqrt(np.maximum(1.0 - illumination_na * illumination_na, 0.0))
    return np.column_stack([sx, sy, np.full_like(sx, sz)]), phi


def q_samples_from_vectors(q_vectors: np.ndarray, illumination_index: np.ndarray) -> QSamples:
    q_vectors = np.asarray(q_vectors, dtype=float)
    illumination_index = np.asarray(illumination_index, dtype=np.int64)
    if q_vectors.ndim != 2 or q_vectors.shape[1] != 3:
        raise ValueError("q_vectors must have shape (n, 3)")
    if illumination_index.shape != (q_vectors.shape[0],):
        raise ValueError("illumination_index shape does not match q vectors")
    q_perp = np.hypot(q_vectors[:, 0], q_vectors[:, 1])
    phi = np.mod(np.arctan2(q_vectors[:, 1], q_vectors[:, 0]), 2.0 * np.pi)
    return QSamples(
        qx=np.ascontiguousarray(q_vectors[:, 0]),
        qy=np.ascontiguousarray(q_vectors[:, 1]),
        qz=np.ascontiguousarray(q_vectors[:, 2]),
        q_perp=np.ascontiguousarray(q_perp),
        phi=np.ascontiguousarray(phi),
        illumination_index=np.ascontiguousarray(illumination_index),
    )


def cone_q_samples(
    *,
    k: float,
    detector_na: float,
    cap_radial: int,
    cap_phi: int,
    illumination: np.ndarray,
) -> tuple[QSamples, QSamples]:
    detector = detector_directions(
        detector_na=detector_na,
        cap_radial=cap_radial,
        cap_phi=cap_phi,
    )
    base_q_vectors = k * (detector - np.array([[0.0, 0.0, 1.0]], dtype=float))
    base_q = q_samples_from_vectors(
        base_q_vectors,
        np.zeros(detector.shape[0], dtype=np.int64),
    )
    blocks: list[np.ndarray] = []
    illum_index: list[np.ndarray] = []
    for index, s_in in enumerate(illumination):
        blocks.append(k * (detector - s_in[None, :]))
        illum_index.append(np.full(detector.shape[0], index, dtype=np.int64))
    flat_q = q_samples_from_vectors(np.vstack(blocks), np.concatenate(illum_index))
    return flat_q, base_q


def mode_order_sequence(rank: int) -> np.ndarray:
    if rank <= 0:
        raise ValueError("rank must be positive")
    orders = [0]
    order = 1
    while len(orders) < rank:
        orders.append(order)
        if len(orders) < rank:
            orders.append(-order)
        order += 1
    return np.asarray(orders, dtype=np.int64)


def source_mode_weights(phi: np.ndarray, mode_orders: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.exp(1j * mode_orders[:, None] * phi[None, :]) / np.sqrt(float(phi.size))
    )


def random_mode_residual(*, rank: int, base_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(rank, base_count))
    envelope = 1.0 + 0.20 * rng.standard_normal((rank, base_count))
    mode_scale = 1.0 + 0.04 * np.arange(rank, dtype=float)[:, None]
    return (envelope * mode_scale * np.exp(1j * phase)).astype(np.complex128)


def build_cone_mode_cache(
    args: argparse.Namespace,
    *,
    obj: Any,
    n_illum: int,
    cmd_rank: int,
    illumination_na: float,
    include_flat_kernel: bool = True,
) -> ConeModeCache:
    illumination, illum_phi = cone_illumination_directions(
        n_illum=n_illum,
        illumination_na=illumination_na,
    )
    flat_q, base_q = cone_q_samples(
        k=args.k,
        detector_na=args.detector_na,
        cap_radial=args.cap_radial,
        cap_phi=args.cap_phi,
        illumination=illumination,
    )
    flat_h_cutoff = (
        recommended_h_cutoff(flat_q, args.r_max, args.n_beta, args.h_margin)
        if args.h_cutoff is None
        else int(args.h_cutoff)
    )
    cone_h_cutoff = (
        recommended_h_cutoff(base_q, args.r_max, args.n_beta, args.h_margin)
        if args.h_cutoff is None
        else int(args.h_cutoff)
    )
    flat_plan = StructuredOdtPlan.build(
        r_axis=obj.r_axis,
        z_axis=obj.z_axis,
        beta_axis=obj.beta_axis,
        h_cutoff=flat_h_cutoff,
    )
    cone_plan = StructuredOdtPlan.build(
        r_axis=obj.r_axis,
        z_axis=obj.z_axis,
        beta_axis=obj.beta_axis,
        h_cutoff=cone_h_cutoff,
    )
    mode_orders = mode_order_sequence(cmd_rank)
    if np.max(np.abs(mode_orders)) >= n_illum:
        raise ValueError("CMD rank uses Fourier orders that alias on the illumination ring")
    weights = source_mode_weights(illum_phi, mode_orders)
    phases = build_shifted_axis_phases(cone_plan, k=args.k, illumination=illumination)
    source_factors = np.tensordot(weights, phases, axes=(1, 0))
    source_factors = np.ascontiguousarray(source_factors)
    cone_kernel = build_structured_kernel(cone_plan, base_q)
    flat_kernel = build_structured_kernel(flat_plan, flat_q) if include_flat_kernel else None
    factorization = ShiftedAxisFactorization(
        base_q=base_q,
        illumination=np.zeros((cmd_rank, 3), dtype=float),
        phase=source_factors,
        beta_twiddle=np.empty((0, 0), dtype=np.complex128),
        kernel=cone_kernel,
        cap_radial=args.cap_radial,
        cap_phi=args.cap_phi,
    )
    return ConeModeCache(
        base_q=base_q,
        flat_q=flat_q,
        factorization=factorization,
        source_factors=source_factors,
        weights=weights,
        mode_orders=mode_orders,
        flat_kernel=flat_kernel,
        flat_plan=flat_plan,
        cone_plan=cone_plan,
    )


def flat_mode_pair(
    args: argparse.Namespace,
    *,
    obj: Any,
    cache: ConeModeCache,
    residual_modes: np.ndarray,
    backend: str,
) -> tuple[np.ndarray, np.ndarray]:
    base_count = cache.base_q.count
    n_illum = cache.weights.shape[1]
    flat_forward = structured_forward(
        cache.flat_plan,
        obj.coeff,
        cache.flat_q,
        kernel=cache.flat_kernel,
        backend=backend,
        cpp_threads=args.cpp_threads,
    ).reshape(n_illum, base_count)
    mode_forward = np.einsum("mi,ib->mb", cache.weights, flat_forward, optimize=True)
    flat_residual = np.einsum(
        "mi,mb->ib",
        np.conj(cache.weights),
        residual_modes,
        optimize=True,
    ).reshape(n_illum * base_count)
    mode_adjoint = structured_adjoint(
        cache.flat_plan,
        cache.flat_q,
        flat_residual,
        kernel=cache.flat_kernel,
        backend=backend,
        cpp_threads=args.cpp_threads,
    )
    return mode_forward, mode_adjoint


def cone_mode_pair(
    args: argparse.Namespace,
    *,
    obj: Any,
    cache: ConeModeCache,
    residual_modes: np.ndarray,
    backend: str,
) -> tuple[np.ndarray, np.ndarray]:
    coeff_modes = cache.source_factors * obj.coeff[None, :, :, :]
    coeff_h_full = np.fft.ifft(coeff_modes, axis=3) * float(cache.cone_plan.n_beta)
    coeff_h_all = np.ascontiguousarray(coeff_h_full[:, :, :, cache.cone_plan.h_indices])
    mode_forward = _axis_grid_forward_fft(
        cache.cone_plan,
        coeff_h_all,
        cache.factorization,
        backend=backend,
        cpp_threads=args.cpp_threads,
    ).reshape(cache.source_factors.shape[0], cache.base_q.count)

    compact = _axis_grid_adjoint_fft_compact(
        cache.cone_plan,
        cache.factorization,
        np.ascontiguousarray(residual_modes.reshape(-1)),
        backend=backend,
        cpp_threads=args.cpp_threads,
    )
    coeff_adjoint_full = np.zeros(
        (
            cache.source_factors.shape[0],
            cache.cone_plan.r_axis.size,
            cache.cone_plan.z_axis.size,
            cache.cone_plan.n_beta,
        ),
        dtype=np.complex128,
    )
    coeff_adjoint_full[:, :, :, cache.cone_plan.h_indices] = compact
    axis_adjoint = np.fft.fft(coeff_adjoint_full, axis=3)
    mode_adjoint = np.sum(np.conj(cache.source_factors) * axis_adjoint, axis=0)
    return mode_forward, mode_adjoint


def cache_mib(cache: ConeModeCache) -> float:
    return (
        kernel_mib(cache.factorization.kernel)
        + cache.source_factors.nbytes / (1024.0 * 1024.0)
        + cache.weights.nbytes / (1024.0 * 1024.0)
    )


def benchmark_case(
    args: argparse.Namespace,
    *,
    n_illum: int,
    cmd_rank: int,
    illumination_na: float,
) -> dict[str, Any]:
    obj = make_cylindrical_object(
        n_r=args.n_r,
        n_z=args.n_z,
        n_beta=args.n_beta,
        r_max=args.r_max,
        z_max=args.z_max,
        phantom=args.phantom,
        seed=args.seed,
    )
    backend = resolve_structured_backend(args.structured_backend)
    _, cone_build_s, cone_build_times = median_time(
        lambda: build_cone_mode_cache(
            args,
            obj=obj,
            n_illum=n_illum,
            cmd_rank=cmd_rank,
            illumination_na=illumination_na,
            include_flat_kernel=False,
        ),
        repeats=args.build_repeats,
    )
    cache = build_cone_mode_cache(
        args,
        obj=obj,
        n_illum=n_illum,
        cmd_rank=cmd_rank,
        illumination_na=illumination_na,
        include_flat_kernel=True,
    )
    residual_modes = random_mode_residual(
        rank=cmd_rank,
        base_count=cache.base_q.count,
        seed=args.seed + 7919 + 17 * cmd_rank + n_illum,
    )

    (flat_forward, flat_adjoint), flat_hot_s, flat_hot_times = median_time(
        lambda: flat_mode_pair(
            args,
            obj=obj,
            cache=cache,
            residual_modes=residual_modes,
            backend=backend,
        ),
        repeats=args.hot_repeats,
    )
    (cone_forward, cone_adjoint), cone_hot_s, cone_hot_times = median_time(
        lambda: cone_mode_pair(
            args,
            obj=obj,
            cache=cache,
            residual_modes=residual_modes,
            backend=backend,
        ),
        repeats=args.hot_repeats,
    )
    dot_error = relative_complex_error(
        complex_dot(cone_forward, residual_modes),
        complex_dot(obj.coeff, cone_adjoint),
    )

    row: dict[str, Any] = {
        "status": "ok",
        "n_beta": args.n_beta,
        "n_r": args.n_r,
        "n_z": args.n_z,
        "r_max": args.r_max,
        "z_max": args.z_max,
        "phantom": args.phantom,
        "k": args.k,
        "detector_na": args.detector_na,
        "illumination_na": illumination_na,
        "n_illum": n_illum,
        "cmd_rank": cmd_rank,
        "mode_orders": " ".join(str(int(item)) for item in cache.mode_orders),
        "cap_radial": args.cap_radial,
        "cap_phi": args.cap_phi,
        "flat_q_samples": cache.flat_q.count,
        "base_q_samples": cache.base_q.count,
        "flat_h_cutoff": int(np.max(np.abs(cache.flat_plan.h_values))),
        "cone_h_cutoff": int(np.max(np.abs(cache.cone_plan.h_values))),
        "flat_used_modes": cache.flat_plan.used_modes,
        "cone_used_modes": cache.cone_plan.used_modes,
        "unique_flat_q_perp_rounded12": int(np.unique(np.round(cache.flat_q.q_perp, 12)).size),
        "unique_base_q_perp_rounded12": int(np.unique(np.round(cache.base_q.q_perp, 12)).size),
        "structured_backend": backend,
        "cpp_threads": args.cpp_threads,
        "cone_build_s": cone_build_s,
        "cone_cache_mib": cache_mib(cache),
        "flat_kernel_mib": kernel_mib(cache.flat_kernel),
        "cone_axis_kernel_mib": kernel_mib(cache.factorization.kernel),
        "source_factors_mib": cache.source_factors.nbytes / (1024.0 * 1024.0),
        "flat_hot_pair_s": flat_hot_s,
        "cone_hot_pair_s": cone_hot_s,
        "flat_vs_cone_hot_speedup": speedup(flat_hot_s, cone_hot_s),
        "cone_forward_l2_vs_flat": relative_l2(cone_forward, flat_forward),
        "cone_adjoint_l2_vs_flat": relative_l2(cone_adjoint, flat_adjoint),
        "cone_adjoint_dot_error": dot_error,
        "cone_build_times_s": " ".join(f"{item:.9g}" for item in cone_build_times),
        "flat_hot_pair_times_s": " ".join(f"{item:.9g}" for item in flat_hot_times),
        "cone_hot_pair_times_s": " ".join(f"{item:.9g}" for item in cone_hot_times),
    }
    for iterations in args.iteration_counts:
        flat_amortized = flat_hot_s
        cone_amortized = (cone_build_s + float(iterations) * cone_hot_s) / float(
            iterations
        )
        row[f"flat_amortized_pair_s_n{iterations}"] = flat_amortized
        row[f"cone_amortized_pair_s_n{iterations}"] = cone_amortized
        row[f"flat_vs_cone_amortized_speedup_n{iterations}"] = speedup(
            flat_amortized,
            cone_amortized,
        )
    return row


def case_label(row: dict[str, Any]) -> str:
    return f"illum={row['n_illum']} rank={row['cmd_rank']}"


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    report_iteration = payload["config"]["report_iteration"]
    lines = [
        "# ODT cone-illumination CMD benchmark",
        "",
        "This benchmark tests whether a conical illumination ring can be collapsed into coherent/CMD source modes instead of evaluating every illumination direction as an independent shifted q-list.",
        "",
        "- Flat baseline: evaluate all `n_illum * cap_radial * cap_phi` shifted q samples, then combine illumination fields into CMD modes.",
        "- Cone-mode path: precompute CMD source factors on the cylindrical object grid, then evaluate only axis-cap detector samples for each retained source mode.",
        "- This is a complex field-mode benchmark. Intensity formation is intentionally outside the operator timing.",
        "- Values above `1x` mean the cone-mode path is faster than the flat shifted-q baseline.",
        "",
        "## Results",
        "",
        "| case | q flat | q axis | modes | flat hot s | cone hot s | hot speedup | N speedup | fwd err | adj err | dot err |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {fq} | {bq} | `{modes}` | `{fh}` | `{ch}` | `{hs}` | `{ns}` | `{fe}` | `{ae}` | `{de}` |".format(
                case=case_label(row),
                fq=row["flat_q_samples"],
                bq=row["base_q_samples"],
                modes=row["mode_orders"],
                fh=fmt(row["flat_hot_pair_s"], 5),
                ch=fmt(row["cone_hot_pair_s"], 5),
                hs=fmt(row["flat_vs_cone_hot_speedup"], 4),
                ns=fmt(row[f"flat_vs_cone_amortized_speedup_n{report_iteration}"], 4),
                fe=fmt(row["cone_forward_l2_vs_flat"], 4),
                ae=fmt(row["cone_adjoint_l2_vs_flat"], 4),
                de=fmt(row["cone_adjoint_dot_error"], 4),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Rank 1 corresponds to the coherent uniform cone mode.",
            "- Higher ranks mimic a low-rank CMD expansion in illumination azimuth.",
        "- The cone-mode path should improve when `n_illum` is large and CMD rank is small.",
        "- The path loses advantage as CMD rank approaches the number of illumination samples.",
        f"- `N speedup` uses N={report_iteration} repeated forward-adjoint pairs and includes one cone-mode cache build; the flat baseline is treated as already cached.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path_png: Path, path_svg: Path, payload: dict[str, Any]) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    import matplotlib.pyplot as plt

    rows = payload["rows"]
    report_iteration = payload["config"]["report_iteration"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), dpi=180)
    ax_hot, ax_err = axes
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xlabel("CMD rank")
    ax_hot.axhline(1.0, color="#444444", lw=1.0, ls="--", alpha=0.75)
    ax_hot.set_ylabel("flat / cone speedup")
    ax_err.set_yscale("log")
    ax_err.set_ylabel("relative L2 vs flat")
    for n_illum in sorted({int(row["n_illum"]) for row in rows}):
        subset = [row for row in rows if int(row["n_illum"]) == n_illum]
        subset.sort(key=lambda row: row["cmd_rank"])
        x = [row["cmd_rank"] for row in subset]
        ax_hot.plot(
            x,
            [row["flat_vs_cone_hot_speedup"] for row in subset],
            marker="o",
            lw=1.7,
            label=f"hot illum={n_illum}",
        )
        ax_hot.plot(
            x,
            [row[f"flat_vs_cone_amortized_speedup_n{report_iteration}"] for row in subset],
            marker="s",
            lw=1.2,
            ls="--",
            label=f"N={report_iteration} illum={n_illum}",
        )
        ax_err.plot(
            x,
            [row["cone_forward_l2_vs_flat"] for row in subset],
            marker="o",
            lw=1.7,
            label=f"fwd illum={n_illum}",
        )
        ax_err.plot(
            x,
            [row["cone_adjoint_l2_vs_flat"] for row in subset],
            marker="s",
            lw=1.2,
            ls="--",
            label=f"adj illum={n_illum}",
        )
    ax_hot.set_title("A. Cone-mode speed")
    ax_err.set_title("B. Operator agreement")
    ax_hot.legend(frameon=False, fontsize=7)
    ax_err.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png)
    fig.savefig(path_svg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark conical ODT illumination collapsed into coherent/CMD source modes."
    )
    parser.add_argument("--output-prefix", default="benchmark_results/odt_cone_illumination")
    parser.add_argument("--n-illum-values", default="8,16,32")
    parser.add_argument("--cmd-ranks", default="1,3,5,9")
    parser.add_argument("--illumination-na-values", default="0.2")
    parser.add_argument("--iteration-counts", default="1,2,4,8,16,32")
    parser.add_argument("--report-iteration", type=int, default=32)
    parser.add_argument("--phantom", choices=["beads", "random_beads", "shell"], default="beads")
    parser.add_argument("--n-r", type=int, default=16)
    parser.add_argument("--n-z", type=int, default=15)
    parser.add_argument("--n-beta", type=int, default=192)
    parser.add_argument("--r-max", type=float, default=1.5)
    parser.add_argument("--z-max", type=float, default=1.2)
    parser.add_argument("--k", type=float, default=8.0)
    parser.add_argument("--detector-na", type=float, default=0.45)
    parser.add_argument("--cap-radial", type=int, default=8)
    parser.add_argument("--cap-phi", type=int, default=32)
    parser.add_argument("--h-cutoff", type=int, default=None)
    parser.add_argument("--h-margin", type=int, default=18)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--structured-backend", choices=["auto", "numpy", "cpp"], default="cpp")
    parser.add_argument("--cpp-threads", type=int, default=0)
    parser.add_argument("--build-repeats", type=int, default=2)
    parser.add_argument("--hot-repeats", type=int, default=3)
    args = parser.parse_args()
    args.n_illum_values = parse_int_list(args.n_illum_values)
    args.cmd_ranks = parse_int_list(args.cmd_ranks)
    args.illumination_na_values = parse_float_list(args.illumination_na_values)
    args.iteration_counts = parse_int_list(args.iteration_counts)
    if args.report_iteration not in args.iteration_counts:
        raise ValueError("report-iteration must be included in iteration-counts")
    if args.cpp_threads < 0:
        raise ValueError("cpp-threads must be non-negative")
    if args.cap_radial <= 0 or args.cap_phi <= 0:
        raise ValueError("cap-radial and cap-phi must be positive")
    return args


def main() -> None:
    args = parse_args()
    rows = [
        benchmark_case(
            args,
            n_illum=n_illum,
            cmd_rank=cmd_rank,
            illumination_na=illumination_na,
        )
        for illumination_na in args.illumination_na_values
        for n_illum in args.n_illum_values
        for cmd_rank in args.cmd_ranks
        if cmd_rank <= n_illum
    ]
    payload = {
        "config": {
            **vars(args),
            "n_illum_values": args.n_illum_values,
            "cmd_ranks": args.cmd_ranks,
            "illumination_na_values": args.illumination_na_values,
            "iteration_counts": args.iteration_counts,
        },
        "rows": rows,
    }
    output_prefix = ROOT / args.output_prefix
    write_json(output_prefix.with_suffix(".json"), payload)
    write_csv(output_prefix.with_suffix(".csv"), rows)
    write_summary(output_prefix.with_name(output_prefix.name + "_summary.md"), payload)
    write_plot(
        output_prefix.with_name(output_prefix.name + ".png"),
        output_prefix.with_name(output_prefix.name + ".svg"),
        payload,
    )
    print(
        json.dumps(
            {
                "json": str(output_prefix.with_suffix(".json")),
                "csv": str(output_prefix.with_suffix(".csv")),
                "summary_md": str(output_prefix.with_name(output_prefix.name + "_summary.md")),
                "png": str(output_prefix.with_name(output_prefix.name + ".png")),
                "svg": str(output_prefix.with_name(output_prefix.name + ".svg")),
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
