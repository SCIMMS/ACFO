from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from benchmark_odt_ewald_cap_operator import (
    ROOT,
    build_structured_kernel,
    ewald_cap_q_samples,
    finufft_adjoint,
    finufft_forward,
    make_cylindrical_object,
    random_residual,
    recommended_h_cutoff,
    relative_l2,
    resolve_structured_backend,
    structured_adjoint,
    structured_forward,
    StructuredOdtPlan,
)


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    if any(value <= 0 for value in values):
        raise ValueError("integer values must be positive")
    return values


def median_time(func, *, repeats: int) -> tuple[Any, float, list[float]]:
    result = None
    times: list[float] = []
    for _ in range(max(1, repeats)):
        start = __import__("time").perf_counter()
        result = func()
        times.append(__import__("time").perf_counter() - start)
    if result is None:
        raise RuntimeError("timed function did not run")
    return result, float(median(times)), times


def speedup(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    value_f = float(value)
    if value_f == 0.0:
        return "0"
    if abs(value_f) < 1e-3 or abs(value_f) >= 1e3:
        return f"{value_f:.{digits}e}"
    return f"{value_f:.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def case_label(row: dict[str, Any]) -> str:
    return f"{row['geometry']} nb={row['n_beta']}"


def benchmark_case(args: argparse.Namespace, *, geometry: str, n_beta: int) -> dict[str, Any]:
    obj = make_cylindrical_object(
        n_r=args.n_r,
        n_z=args.n_z,
        n_beta=n_beta,
        r_max=args.r_max,
        z_max=args.z_max,
        phantom=args.phantom,
        seed=args.seed,
    )
    q = ewald_cap_q_samples(
        k=args.k,
        detector_na=args.detector_na,
        cap_radial=args.cap_radial,
        cap_phi=args.cap_phi,
        geometry=geometry,
        n_illum=args.n_illum if geometry == "shifted" else 1,
        illumination_na=args.illumination_na,
    )
    h_cutoff = (
        recommended_h_cutoff(q, args.r_max, n_beta, args.h_margin)
        if args.h_cutoff is None
        else int(args.h_cutoff)
    )
    plan = StructuredOdtPlan.build(
        r_axis=obj.r_axis,
        z_axis=obj.z_axis,
        beta_axis=obj.beta_axis,
        h_cutoff=h_cutoff,
    )
    backend = resolve_structured_backend(args.structured_backend)
    residual = random_residual(q, seed=args.seed + 7919)

    kernel, kernel_build_s, kernel_build_times = median_time(
        lambda: build_structured_kernel(plan, q),
        repeats=args.build_repeats,
    )
    kernel_mib = (
        kernel.radial.nbytes + kernel.axial.nbytes + kernel.angular.nbytes
    ) / (1024.0 * 1024.0)

    def structured_pair(active_kernel):
        forward = structured_forward(
            plan,
            obj.coeff,
            q,
            kernel=active_kernel,
            backend=backend,
            cpp_threads=args.cpp_threads,
        )
        adjoint = structured_adjoint(
            plan,
            q,
            residual,
            kernel=active_kernel,
            backend=backend,
            cpp_threads=args.cpp_threads,
        )
        return forward, adjoint

    (structured_forward_value, structured_adjoint_value), structured_hot_pair_s, structured_hot_times = (
        median_time(lambda: structured_pair(kernel), repeats=args.hot_repeats)
    )
    _, structured_cold_pair_s, structured_cold_times = median_time(
        lambda: structured_pair(build_structured_kernel(plan, q)),
        repeats=args.cold_repeats,
    )

    finufft_error = None
    finufft_adj_error = None
    finufft_pair_s = None
    finufft_pair_times: list[float] = []
    finufft_skip_reason = None
    try:
        (finufft_forward_value, finufft_adjoint_value), finufft_pair_s, finufft_pair_times = (
            median_time(
                lambda: (
                    finufft_forward(obj, q, eps=args.finufft_eps),
                    finufft_adjoint(obj, q, residual, eps=args.finufft_eps),
                ),
                repeats=args.hot_repeats,
            )
        )
        finufft_error = relative_l2(structured_forward_value, finufft_forward_value)
        finufft_adj_error = relative_l2(structured_adjoint_value, finufft_adjoint_value)
    except Exception as exc:  # pragma: no cover - optional dependency/runtime path
        finufft_skip_reason = str(exc)

    build_break_even_iterations = None
    if finufft_pair_s is not None and finufft_pair_s > structured_hot_pair_s:
        build_break_even_iterations = kernel_build_s / (finufft_pair_s - structured_hot_pair_s)

    row: dict[str, Any] = {
        "status": "ok",
        "geometry": geometry,
        "phantom": args.phantom,
        "n_r": args.n_r,
        "n_z": args.n_z,
        "n_beta": n_beta,
        "r_max": args.r_max,
        "z_max": args.z_max,
        "k": args.k,
        "detector_na": args.detector_na,
        "illumination_na": args.illumination_na,
        "n_illum": int(np.max(q.illumination_index) + 1),
        "cap_radial": args.cap_radial,
        "cap_phi": args.cap_phi,
        "q_samples": q.count,
        "unique_q_perp_rounded12": int(np.unique(np.round(q.q_perp, 12)).size),
        "unique_phi_rounded12": int(np.unique(np.round(q.phi, 12)).size),
        "object_bins": obj.bins,
        "h_cutoff": h_cutoff,
        "used_modes": plan.used_modes,
        "structured_backend": backend,
        "requested_structured_backend": args.structured_backend,
        "cpp_threads": args.cpp_threads,
        "kernel_build_s": kernel_build_s,
        "kernel_mib": kernel_mib,
        "structured_hot_pair_s": structured_hot_pair_s,
        "structured_cold_pair_s": structured_cold_pair_s,
        "finufft_pair_s": finufft_pair_s,
        "finufft_vs_structured_hot_pair_speedup": speedup(
            finufft_pair_s,
            structured_hot_pair_s,
        ),
        "finufft_vs_structured_cold_pair_speedup": speedup(
            finufft_pair_s,
            structured_cold_pair_s,
        ),
        "build_break_even_iterations": build_break_even_iterations,
        "structured_forward_l2_vs_finufft": finufft_error,
        "structured_adjoint_l2_vs_finufft": finufft_adj_error,
        "finufft_skip_reason": finufft_skip_reason,
        "kernel_build_times_s": " ".join(f"{value:.9g}" for value in kernel_build_times),
        "structured_hot_pair_times_s": " ".join(
            f"{value:.9g}" for value in structured_hot_times
        ),
        "structured_cold_pair_times_s": " ".join(
            f"{value:.9g}" for value in structured_cold_times
        ),
        "finufft_pair_times_s": " ".join(f"{value:.9g}" for value in finufft_pair_times),
    }

    for iterations in args.iteration_counts:
        amortized = (kernel_build_s + float(iterations) * structured_hot_pair_s) / float(
            iterations
        )
        row[f"structured_amortized_pair_s_n{iterations}"] = amortized
        row[f"finufft_vs_structured_amortized_speedup_n{iterations}"] = speedup(
            finufft_pair_s,
            amortized,
        )

    return row


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    iteration_counts = payload["config"]["iteration_counts"]
    lines = [
        "# ODT Ewald-cap build amortization benchmark",
        "",
        "This benchmark measures the setup cost that is excluded from hot forward/adjoint timings.",
        "The key quantity is how many repeated forward-adjoint pairs are needed to amortize the structured Bessel/phase kernel build against FINUFFT.",
        "",
        "## Results",
        "",
        "| case | q | bins | modes | kernel build s | kernel MiB | structured hot pair s | structured cold pair s | FINUFFT pair s | hot speedup | cold speedup | break-even pairs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {q} | {bins} | {modes} | `{build}` | `{mib}` | `{hot}` | `{cold}` | `{finufft}` | `{hot_sp}` | `{cold_sp}` | `{be}` |".format(
                case=case_label(row),
                q=row["q_samples"],
                bins=row["object_bins"],
                modes=row["used_modes"],
                build=fmt(row["kernel_build_s"], 5),
                mib=fmt(row["kernel_mib"], 4),
                hot=fmt(row["structured_hot_pair_s"], 5),
                cold=fmt(row["structured_cold_pair_s"], 5),
                finufft=fmt(row["finufft_pair_s"], 5),
                hot_sp=fmt(row["finufft_vs_structured_hot_pair_speedup"], 4),
                cold_sp=fmt(row["finufft_vs_structured_cold_pair_speedup"], 4),
                be=fmt(row["build_break_even_iterations"], 4),
            )
        )

    lines.extend(["", "## Amortized Speedup"])
    header = "| case | " + " | ".join(f"N={count}" for count in iteration_counts) + " |"
    sep = "| --- | " + " | ".join("---:" for _ in iteration_counts) + " |"
    lines.extend(["", header, sep])
    for row in rows:
        values = [
            fmt(row[f"finufft_vs_structured_amortized_speedup_n{count}"], 4)
            for count in iteration_counts
        ]
        lines.append(f"| {case_label(row)} | " + " | ".join(f"`{value}`" for value in values) + " |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `hot pair` is one cached structured forward plus one cached structured adjoint.",
            "- `cold pair` rebuilds the structured kernel and then runs one forward-adjoint pair.",
            "- One-shot ODT calls should use the cold number; reconstruction/inverse-design loops should use the amortized number.",
            "- A break-even value near 30 means a 100-step reconstruction loop can hide setup cost, but a single forward call cannot.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path_png: Path, path_svg: Path, payload: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    rows = payload["rows"]
    iteration_counts = payload["config"]["iteration_counts"]
    labels = [case_label(row) for row in rows]
    x = np.arange(len(iteration_counts), dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    ax.axhline(1.0, color="#444444", lw=1.0, ls="--", alpha=0.75)
    for row in rows:
        y = [
            row[f"finufft_vs_structured_amortized_speedup_n{count}"]
            for count in iteration_counts
        ]
        ax.plot(x, y, marker="o", lw=1.8, label=case_label(row))
    ax.set_xticks(x)
    ax.set_xticklabels([str(count) for count in iteration_counts])
    ax.set_xlabel("forward-adjoint pairs after one kernel build")
    ax.set_ylabel("FINUFFT / structured amortized speedup")
    ax.set_title("ODT build amortization: when cached structured CPU wins")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=8, ncols=2)
    fig.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png)
    fig.savefig(path_svg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ODT structured-kernel build amortization."
    )
    parser.add_argument("--output-prefix", default="benchmark_results/odt_build_amortization")
    parser.add_argument("--geometries", default="axis,shifted")
    parser.add_argument("--n-beta-values", default="96,192,384")
    parser.add_argument("--iteration-counts", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--phantom", choices=["beads", "random_beads", "shell"], default="beads")
    parser.add_argument("--n-r", type=int, default=16)
    parser.add_argument("--n-z", type=int, default=15)
    parser.add_argument("--r-max", type=float, default=1.5)
    parser.add_argument("--z-max", type=float, default=1.2)
    parser.add_argument("--k", type=float, default=8.0)
    parser.add_argument("--detector-na", type=float, default=0.45)
    parser.add_argument("--illumination-na", type=float, default=0.25)
    parser.add_argument("--n-illum", type=int, default=9)
    parser.add_argument("--cap-radial", type=int, default=8)
    parser.add_argument("--cap-phi", type=int, default=32)
    parser.add_argument("--h-cutoff", type=int, default=None)
    parser.add_argument("--h-margin", type=int, default=18)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--finufft-eps", type=float, default=1e-12)
    parser.add_argument("--structured-backend", choices=["auto", "numpy", "cpp"], default="cpp")
    parser.add_argument("--cpp-threads", type=int, default=0)
    parser.add_argument("--build-repeats", type=int, default=3)
    parser.add_argument("--hot-repeats", type=int, default=3)
    parser.add_argument("--cold-repeats", type=int, default=3)
    args = parser.parse_args()

    args.geometries = [part.strip() for part in args.geometries.split(",") if part.strip()]
    if any(value not in {"axis", "shifted"} for value in args.geometries):
        raise ValueError("geometries must contain only axis and/or shifted")
    if not args.geometries:
        raise ValueError("expected at least one geometry")
    args.n_beta_values = parse_int_list(args.n_beta_values)
    args.iteration_counts = parse_int_list(args.iteration_counts)
    if args.cpp_threads < 0:
        raise ValueError("cpp-threads must be non-negative")
    if args.build_repeats <= 0 or args.hot_repeats <= 0 or args.cold_repeats <= 0:
        raise ValueError("repeat counts must be positive")
    return args


def main() -> None:
    args = parse_args()
    rows = [
        benchmark_case(args, geometry=geometry, n_beta=n_beta)
        for geometry in args.geometries
        for n_beta in args.n_beta_values
    ]
    payload = {
        "config": {
            **vars(args),
            "geometries": args.geometries,
            "n_beta_values": args.n_beta_values,
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
