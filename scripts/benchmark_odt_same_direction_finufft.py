from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from benchmark_odt_cone_axis_decomposition import default_l_cutoff, fmt
from benchmark_odt_cone_illumination import parse_int_list
from benchmark_odt_ewald_cap_operator import (
    StructuredOdtPlan,
    complex_dot,
    finufft_adjoint,
    finufft_forward,
    make_cylindrical_object,
    random_residual,
    recommended_h_cutoff,
    relative_complex_error,
    relative_l2,
    resolve_structured_backend,
)
from benchmark_odt_same_direction_zperp import (
    build_magnitude_svd,
    build_same_direction_decomposition,
    compressed_adjoint,
    compressed_forward,
    same_direction_adjoint,
    same_direction_forward,
    same_direction_illumination,
    same_direction_q_samples,
    speedup,
)


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


def parse_grid_list(text: str) -> list[tuple[int, int]]:
    grids: list[tuple[int, int]] = []
    for item in text.split(","):
        token = item.strip().lower().replace(" ", "")
        if not token:
            continue
        if "x" not in token:
            raise ValueError(f"cap grid entry must look like 8x32, got {item!r}")
        left, right = token.split("x", 1)
        grids.append((int(left), int(right)))
    if not grids:
        raise ValueError("at least one cap grid is required")
    return grids


def benchmark_case(
    args: argparse.Namespace,
    *,
    n_mag: int,
    cap_radial: int,
    cap_phi: int,
) -> list[dict[str, Any]]:
    obj = make_cylindrical_object(
        n_r=args.n_r,
        n_z=args.n_z,
        n_beta=args.n_beta,
        r_max=args.r_max,
        z_max=args.z_max,
        phantom=args.phantom,
        seed=args.seed,
    )
    magnitudes = np.linspace(args.min_illumination_na, args.max_illumination_na, n_mag)
    illumination = same_direction_illumination(
        magnitudes=magnitudes,
        direction_phi=args.direction_phi,
    )
    flat_q, base_q = same_direction_q_samples(
        k=args.k,
        detector_na=args.detector_na,
        cap_radial=cap_radial,
        cap_phi=cap_phi,
        illumination=illumination,
    )
    axis_h_cutoff = (
        args.h_cutoff
        if args.h_cutoff is not None
        else recommended_h_cutoff(base_q, args.r_max, args.n_beta, args.h_margin)
    )
    axis_plan = StructuredOdtPlan.build(
        r_axis=obj.r_axis,
        z_axis=obj.z_axis,
        beta_axis=obj.beta_axis,
        h_cutoff=axis_h_cutoff,
    )
    l_cutoff = default_l_cutoff(
        k=args.k,
        illumination_na=args.max_illumination_na,
        r_max=args.r_max,
        margin=args.l_margin,
        n_beta=args.n_beta,
    )
    decomp, decomp_build_s, decomp_build_times = median_time(
        lambda: build_same_direction_decomposition(
            axis_plan,
            k=args.k,
            detector_na=args.detector_na,
            cap_radial=cap_radial,
            cap_phi=cap_phi,
            magnitudes=magnitudes,
            direction_phi=args.direction_phi,
            l_cutoff=l_cutoff,
        ),
        repeats=args.build_repeats,
    )
    compression, svd_build_s, svd_build_times = median_time(
        lambda: build_magnitude_svd(decomp),
        repeats=args.build_repeats,
    )
    backend = resolve_structured_backend(args.structured_backend)
    residual = random_residual(decomp.flat_q, seed=args.seed + 2003 + n_mag + cap_radial + cap_phi)

    def grouped_forward_func():
        return same_direction_forward(
            obj.coeff,
            decomp,
            backend=backend,
            cpp_threads=args.cpp_threads,
            grouped=True,
        )

    def grouped_adjoint_func():
        return same_direction_adjoint(
            residual,
            decomp,
            backend=backend,
            cpp_threads=args.cpp_threads,
            grouped=True,
        )

    grouped_forward, grouped_forward_s, grouped_forward_times = median_time(
        grouped_forward_func,
        repeats=args.hot_repeats,
    )
    grouped_adjoint, grouped_adjoint_s, grouped_adjoint_times = median_time(
        grouped_adjoint_func,
        repeats=args.hot_repeats,
    )
    grouped_s = grouped_forward_s + grouped_adjoint_s

    finufft_forward_value = None
    finufft_adjoint_value = None
    finufft_forward_s = None
    finufft_adjoint_s = None
    finufft_s = None
    finufft_forward_times: list[float] = []
    finufft_adjoint_times: list[float] = []
    finufft_skip_reason = None
    if not args.skip_finufft:
        try:
            finufft_forward_value, finufft_forward_s, finufft_forward_times = median_time(
                lambda: finufft_forward(obj, decomp.flat_q, eps=args.finufft_eps),
                repeats=args.finufft_repeats,
            )
            finufft_adjoint_value, finufft_adjoint_s, finufft_adjoint_times = median_time(
                lambda: finufft_adjoint(obj, decomp.flat_q, residual, eps=args.finufft_eps),
                repeats=args.finufft_repeats,
            )
            finufft_s = finufft_forward_s + finufft_adjoint_s
        except Exception as exc:  # pragma: no cover - optional dependency/runtime path
            finufft_skip_reason = str(exc)

    rows: list[dict[str, Any]] = []

    def base_row(
        method: str,
        pair_s: float | None,
        forward_s: float | None,
        adjoint_s: float | None,
        forward_times: list[float],
        adjoint_times: list[float],
        forward: np.ndarray | None,
        adjoint: np.ndarray | None,
        *,
        rank: int | None = None,
    ) -> dict[str, Any]:
        unique_flat_q_perp = int(np.unique(np.round(decomp.flat_q.q_perp, 12)).size)
        unique_base_q_perp = int(np.unique(np.round(decomp.base_q.q_perp, 12)).size)
        return {
            "status": "ok" if pair_s is not None else "skipped",
            "method": method,
            "rank": rank,
            "n_mag": n_mag,
            "n_beta": args.n_beta,
            "n_r": args.n_r,
            "n_z": args.n_z,
            "r_max": args.r_max,
            "z_max": args.z_max,
            "phantom": args.phantom,
            "k": args.k,
            "detector_na": args.detector_na,
            "min_illumination_na": float(magnitudes.min()),
            "max_illumination_na": float(magnitudes.max()),
            "direction_phi": args.direction_phi,
            "cap_radial": cap_radial,
            "cap_phi": cap_phi,
            "cap_grid": cap_radial * cap_phi,
            "flat_q_samples": decomp.flat_q.count,
            "axis_q_samples": decomp.base_q.count,
            "unique_flat_q_perp_rounded12": unique_flat_q_perp,
            "unique_base_q_perp_rounded12": unique_base_q_perp,
            "flat_q_per_unique_q_perp": float(decomp.flat_q.count) / float(unique_flat_q_perp),
            "flat_q_perp_max": float(np.max(decomp.flat_q.q_perp)),
            "flat_q_abs_max": float(
                np.max(
                    np.sqrt(
                        decomp.flat_q.qx * decomp.flat_q.qx
                        + decomp.flat_q.qy * decomp.flat_q.qy
                        + decomp.flat_q.qz * decomp.flat_q.qz
                    )
                )
            ),
            "l_cutoff": l_cutoff,
            "l_modes": int(decomp.l_values.size),
            "axis_h_cutoff": axis_h_cutoff,
            "axis_used_modes": axis_plan.used_modes,
            "structured_backend": backend,
            "cpp_threads": args.cpp_threads,
            "finufft_eps": None if args.skip_finufft else args.finufft_eps,
            "decomp_build_s": decomp_build_s,
            "svd_build_s": svd_build_s,
            "decomp_factor_mib": (
                decomp.transverse_by_mag.nbytes + decomp.axial_phase_by_mag.nbytes
            )
            / (1024.0 * 1024.0),
            "source_points": int(obj.weights.size),
            "pair_s": pair_s,
            "forward_s": forward_s,
            "adjoint_s": adjoint_s,
            "grouped_pair_s": grouped_s,
            "grouped_forward_s": grouped_forward_s,
            "grouped_adjoint_s": grouped_adjoint_s,
            "finufft_pair_s": finufft_s,
            "finufft_forward_s": finufft_forward_s,
            "finufft_adjoint_s": finufft_adjoint_s,
            "finufft_over_method_speedup": speedup(finufft_s, pair_s),
            "finufft_over_method_forward_speedup": speedup(finufft_forward_s, forward_s),
            "finufft_over_method_adjoint_speedup": speedup(finufft_adjoint_s, adjoint_s),
            "method_over_finufft_speedup": speedup(pair_s, finufft_s),
            "method_over_finufft_forward_speedup": speedup(forward_s, finufft_forward_s),
            "method_over_finufft_adjoint_speedup": speedup(adjoint_s, finufft_adjoint_s),
            "method_forward_l2_vs_grouped": None
            if forward is None
            else relative_l2(forward, grouped_forward),
            "method_adjoint_l2_vs_grouped": None
            if adjoint is None
            else relative_l2(adjoint, grouped_adjoint),
            "adjoint_dot_error": None
            if forward is None or adjoint is None
            else relative_complex_error(
                complex_dot(forward, residual),
                complex_dot(obj.coeff, adjoint),
            ),
            "hot_pair_times_s": "",
            "hot_forward_times_s": " ".join(f"{item:.9g}" for item in forward_times),
            "hot_adjoint_times_s": " ".join(f"{item:.9g}" for item in adjoint_times),
            "decomp_build_times_s": " ".join(f"{item:.9g}" for item in decomp_build_times),
            "svd_build_times_s": " ".join(f"{item:.9g}" for item in svd_build_times),
            "finufft_skip_reason": finufft_skip_reason,
        }

    rows.append(
        base_row(
            "grouped_same_direction_z0",
            grouped_s,
            grouped_forward_s,
            grouped_adjoint_s,
            grouped_forward_times,
            grouped_adjoint_times,
            grouped_forward,
            grouped_adjoint,
        )
    )

    for rank in args.svd_rank_values:
        if rank <= 0 or rank > compression.singular_values.size:
            continue

        def compressed_forward_func(rank: int = rank):
            return compressed_forward(
                obj.coeff,
                decomp,
                compression,
                rank=rank,
                backend=backend,
                cpp_threads=args.cpp_threads,
            )

        def compressed_adjoint_func(rank: int = rank):
            return compressed_adjoint(
                residual,
                decomp,
                compression,
                rank=rank,
                backend=backend,
                cpp_threads=args.cpp_threads,
            )

        forward, forward_s, forward_times = median_time(
            compressed_forward_func,
            repeats=args.hot_repeats,
        )
        adjoint, adjoint_s, adjoint_times = median_time(
            compressed_adjoint_func,
            repeats=args.hot_repeats,
        )
        pair_s = forward_s + adjoint_s
        row = base_row(
            f"svd_rank_{rank}",
            pair_s,
            forward_s,
            adjoint_s,
            forward_times,
            adjoint_times,
            forward,
            adjoint,
            rank=rank,
        )
        row["svd_energy_fraction"] = float(
            np.sum(compression.singular_values[:rank] ** 2)
            / max(np.sum(compression.singular_values**2), 1e-300)
        )
        rows.append(row)

    if finufft_s is not None:
        rows.append(
            base_row(
                "finufft_type3",
                finufft_s,
                finufft_forward_s,
                finufft_adjoint_s,
                finufft_forward_times,
                finufft_adjoint_times,
                finufft_forward_value,
                finufft_adjoint_value,
            )
        )
    else:
        rows.append(base_row("finufft_type3", None, None, None, [], [], None, None))

    return rows


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    lines = [
        "# ODT same-direction z_perp vs FINUFFT",
        "",
        "This benchmark compares the same-direction z_perp grouped/SVD paths against FINUFFT type-3 on the same flat shifted q-list.",
        "",
        "## Results",
        "",
        "| n_mag | cap grid | flat q | method | rank | fwd s | adj s | FINUFFT/fwd | FINUFFT/adj | fwd err vs grouped | adj err vs grouped |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {n_mag} | {grid} | {fq} | {method} | {rank} | `{fwd_s}` | `{adj_s}` | `{fwd_speed}` | `{adj_speed}` | `{fwd}` | `{adj}` |".format(
                n_mag=row["n_mag"],
                grid=row["cap_grid"],
                fq=row["flat_q_samples"],
                method=row["method"],
                rank="" if row["rank"] is None else row["rank"],
                fwd_s=fmt(row.get("forward_s"), 5),
                adj_s=fmt(row.get("adjoint_s"), 5),
                fwd_speed=fmt(row.get("finufft_over_method_forward_speedup"), 4),
                adj_speed=fmt(row.get("finufft_over_method_adjoint_speedup"), 4),
                fwd=fmt(row.get("method_forward_l2_vs_grouped"), 4),
                adj=fmt(row.get("method_adjoint_l2_vs_grouped"), 4),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `FINUFFT/fwd` and `FINUFFT/adj` greater than 1 mean the listed method is faster than FINUFFT in that direction.",
            "- FINUFFT forward and adjoint are timed separately as type-3 calls on the full shifted q-list.",
            "- `pair_s` remains available in the JSON/CSV as `forward_s + adjoint_s`; the table intentionally reports the split timings.",
            "- `grouped_same_direction_z0` is exact relative to the phase-ramp factorization; `svd_rank_*` is approximate unless the retained rank spans the magnitude stack.",
        ]
    )
    if any(row.get("finufft_skip_reason") for row in rows):
        reasons = sorted({row["finufft_skip_reason"] for row in rows if row.get("finufft_skip_reason")})
        lines.extend(["", "FINUFFT skip reasons:"])
        for reason in reasons:
            lines.append(f"- `{reason}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path_png: Path, path_svg: Path, payload: dict[str, Any]) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib_cache"))
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in payload["rows"]
        if row["method"] != "finufft_type3" and row.get("finufft_over_method_speedup") is not None
    ]
    if not rows:
        return
    metric_specs = [
        (
            "finufft_over_method_forward_speedup",
            "FINUFFT / method forward time",
            "Forward speedup",
        ),
        (
            "finufft_over_method_adjoint_speedup",
            "FINUFFT / method adjoint time",
            "Adjoint speedup",
        ),
    ]
    metric_specs = [
        spec for spec in metric_specs if any(row.get(spec[0]) is not None for row in rows)
    ]
    if not metric_specs:
        metric_specs = [
            (
                "finufft_over_method_speedup",
                "FINUFFT / method split-pair time",
                "Forward + adjoint speedup",
            )
        ]
    methods = []
    for row in rows:
        if row["method"] not in methods:
            methods.append(row["method"])
    grids = sorted(
        {
            (row["k"], row["n_mag"], row["cap_grid"], row["flat_q_samples"])
            for row in rows
        }
    )
    if len({item[0] for item in grids}) > 1:
        x_labels = [f"k={fmt(k, 3)}\nq={flat_q}" for k, _, _, flat_q in grids]
    else:
        x_labels = [f"{n_mag}x{grid}\nq={flat_q}" for _, n_mag, grid, flat_q in grids]
    x = np.arange(len(grids))
    fig, axes = plt.subplots(
        1,
        len(metric_specs),
        figsize=(5.4 * len(metric_specs), 4.8),
        squeeze=False,
    )
    for ax, (metric, ylabel, title) in zip(axes[0], metric_specs):
        for method in methods:
            values = []
            for k, n_mag, cap_grid, flat_q in grids:
                row = next(
                    (
                        item
                        for item in rows
                        if item["method"] == method
                        and item["k"] == k
                        and item["n_mag"] == n_mag
                        and item["cap_grid"] == cap_grid
                        and item["flat_q_samples"] == flat_q
                    ),
                    None,
                )
                values.append(np.nan if row is None else row.get(metric, np.nan))
            ax.plot(x, values, marker="o", label=method)
        ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("case")
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Same-direction z_perp split speedup over FINUFFT", y=1.02)
    fig.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_svg)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark same-direction z_perp paths against FINUFFT.")
    parser.add_argument("--output-prefix", default="benchmark_results/odt_same_direction_zperp_finufft")
    parser.add_argument("--n-mag-values", default="8,16,32")
    parser.add_argument("--cap-grids", default="8x32,12x48,16x64")
    parser.add_argument("--svd-rank-values", default="4,8")
    parser.add_argument("--min-illumination-na", type=float, default=0.02)
    parser.add_argument("--max-illumination-na", type=float, default=0.2)
    parser.add_argument("--direction-phi", type=float, default=0.0)
    parser.add_argument("--phantom", choices=["beads", "random_beads", "shell"], default="beads")
    parser.add_argument("--n-r", type=int, default=16)
    parser.add_argument("--n-z", type=int, default=15)
    parser.add_argument("--n-beta", type=int, default=384)
    parser.add_argument("--r-max", type=float, default=1.5)
    parser.add_argument("--z-max", type=float, default=1.2)
    parser.add_argument("--k", type=float, default=8.0)
    parser.add_argument("--detector-na", type=float, default=0.45)
    parser.add_argument("--h-cutoff", type=int, default=None)
    parser.add_argument("--h-margin", type=int, default=18)
    parser.add_argument("--l-margin", type=int, default=18)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--structured-backend", choices=["auto", "numpy", "cpp"], default="cpp")
    parser.add_argument("--cpp-threads", type=int, default=0)
    parser.add_argument("--build-repeats", type=int, default=2)
    parser.add_argument("--hot-repeats", type=int, default=5)
    parser.add_argument("--finufft-repeats", type=int, default=3)
    parser.add_argument("--finufft-eps", type=float, default=1e-12)
    parser.add_argument("--skip-finufft", action="store_true")
    args = parser.parse_args()
    args.n_mag_values = parse_int_list(args.n_mag_values)
    args.svd_rank_values = parse_int_list(args.svd_rank_values)
    args.cap_grid_values = parse_grid_list(args.cap_grids)
    if args.cpp_threads < 0:
        raise ValueError("cpp-threads must be non-negative")
    if args.finufft_eps <= 0.0:
        raise ValueError("finufft-eps must be positive")
    return args


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for n_mag in args.n_mag_values:
        for cap_radial, cap_phi in args.cap_grid_values:
            rows.extend(
                benchmark_case(
                    args,
                    n_mag=n_mag,
                    cap_radial=cap_radial,
                    cap_phi=cap_phi,
                )
            )
    payload = {
        "args": {
            "n_mag_values": args.n_mag_values,
            "cap_grid_values": args.cap_grid_values,
            "svd_rank_values": args.svd_rank_values,
            "n_beta": args.n_beta,
            "n_r": args.n_r,
            "n_z": args.n_z,
            "finufft_eps": args.finufft_eps,
            "skip_finufft": args.skip_finufft,
        },
        "rows": rows,
    }
    output_prefix = Path(args.output_prefix)
    write_json(output_prefix.with_suffix(".json"), payload)
    write_csv(output_prefix.with_suffix(".csv"), rows)
    write_summary(output_prefix.with_name(output_prefix.name + "_summary.md"), payload)
    write_plot(
        output_prefix.with_suffix(".png"),
        output_prefix.with_suffix(".svg"),
        payload,
    )
    print(json.dumps({"json": str(output_prefix.with_suffix(".json")), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
