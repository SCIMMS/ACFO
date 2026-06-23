from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from benchmark_odt_build_amortization import (
    ROOT,
    benchmark_case,
    fmt,
    parse_int_list,
    write_csv,
    write_json,
)


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one float")
    if any(value < 0.0 for value in values):
        raise ValueError("illumination NA values must be non-negative")
    return values


def case_label(row: dict[str, Any]) -> str:
    if row["geometry"] == "axis":
        return f"axis nb={row['n_beta']}"
    return f"illum={row['illumination_na']:.3g} nb={row['n_beta']}"


def make_case_args(args: argparse.Namespace, *, illumination_na: float) -> SimpleNamespace:
    return SimpleNamespace(
        n_r=args.n_r,
        n_z=args.n_z,
        r_max=args.r_max,
        z_max=args.z_max,
        phantom=args.phantom,
        seed=args.seed,
        k=args.k,
        detector_na=args.detector_na,
        cap_radial=args.cap_radial,
        cap_phi=args.cap_phi,
        n_illum=args.n_illum,
        illumination_na=illumination_na,
        h_cutoff=args.h_cutoff,
        h_margin=args.h_margin,
        structured_backend=args.structured_backend,
        cpp_threads=args.cpp_threads,
        build_repeats=args.build_repeats,
        hot_repeats=args.hot_repeats,
        cold_repeats=args.cold_repeats,
        finufft_eps=args.finufft_eps,
        iteration_counts=args.iteration_counts,
    )


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    iteration_count = payload["config"]["report_iteration"]
    lines = [
        "# ODT illumination transition sweep",
        "",
        "This sweep measures how the C++ structured ODT operator changes as illumination moves from axis-like to off-axis shifted caps.",
        "Rows above `1x` are faster than FINUFFT for the corresponding forward-adjoint pair definition.",
        "",
        "## Results",
        "",
        "| case | illum NA | n_beta | q | unique q_perp | modes | build s | hot pair s | FINUFFT pair s | hot speedup | cold speedup | break-even | amortized speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {case} | `{illum}` | {nb} | {q} | {uq} | {modes} | `{build}` | `{hot}` | `{finufft}` | `{hot_sp}` | `{cold_sp}` | `{be}` | `{amort}` |".format(
                case=case_label(row),
                illum=fmt(row["illumination_na"], 4),
                nb=row["n_beta"],
                q=row["q_samples"],
                uq=row["unique_q_perp_rounded12"],
                modes=row["used_modes"],
                build=fmt(row["kernel_build_s"], 5),
                hot=fmt(row["structured_hot_pair_s"], 5),
                finufft=fmt(row["finufft_pair_s"], 5),
                hot_sp=fmt(row["finufft_vs_structured_hot_pair_speedup"], 4),
                cold_sp=fmt(row["finufft_vs_structured_cold_pair_speedup"], 4),
                be=fmt(row["build_break_even_iterations"], 4),
                amort=fmt(
                    row[f"finufft_vs_structured_amortized_speedup_n{iteration_count}"],
                    4,
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The single-axis row is the strongest symmetry case and is not workload-identical to multi-illumination shifted rows.",
            "- The shifted `illum NA=0` row keeps the multi-illumination q-count but collapses illumination directions onto the optic axis.",
            "- `unique q_perp` is the direct proxy for Bessel-build reuse; hot contraction can still be limited by total q samples and array movement.",
            f"- `amortized speedup` uses N={iteration_count} forward-adjoint pairs after one kernel build.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path_png: Path, path_svg: Path, payload: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    rows = payload["rows"]
    report_iteration = payload["config"]["report_iteration"]
    shifted_rows = [row for row in rows if row["geometry"] == "shifted"]
    n_beta_values = payload["config"]["n_beta_values"]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), dpi=180)
    ax_speed, ax_reuse = axes
    ax_speed.axhline(1.0, color="#444444", lw=1.0, ls="--", alpha=0.75)
    for n_beta in n_beta_values:
        subset = [row for row in shifted_rows if row["n_beta"] == n_beta]
        subset.sort(key=lambda row: row["illumination_na"])
        x = [row["illumination_na"] for row in subset]
        hot = [row["finufft_vs_structured_hot_pair_speedup"] for row in subset]
        amortized = [
            row[f"finufft_vs_structured_amortized_speedup_n{report_iteration}"]
            for row in subset
        ]
        ax_speed.plot(x, hot, marker="o", lw=1.8, label=f"hot nb={n_beta}")
        ax_speed.plot(x, amortized, marker="s", lw=1.3, ls="--", label=f"N={report_iteration} nb={n_beta}")
        ax_reuse.plot(
            x,
            [row["unique_q_perp_rounded12"] for row in subset],
            marker="o",
            lw=1.8,
            label=f"nb={n_beta}",
        )
    ax_speed.set_xlabel("illumination NA")
    ax_speed.set_ylabel("FINUFFT / structured speedup")
    ax_speed.set_title("A. Shifted-cap speedup")
    ax_speed.grid(alpha=0.25)
    ax_speed.legend(frameon=False, fontsize=7, ncols=2)

    ax_reuse.set_xlabel("illumination NA")
    ax_reuse.set_ylabel("unique q_perp values")
    ax_reuse.set_title("B. Bessel radial reuse")
    ax_reuse.grid(alpha=0.25)
    ax_reuse.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png)
    fig.savefig(path_svg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep ODT illumination NA from axis-like to shifted caps."
    )
    parser.add_argument("--output-prefix", default="benchmark_results/odt_illumination_transition")
    parser.add_argument("--illumination-na-values", default="0,0.02,0.05,0.1,0.15,0.2,0.25,0.3")
    parser.add_argument("--n-beta-values", default="192,384")
    parser.add_argument("--iteration-counts", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--report-iteration", type=int, default=32)
    parser.add_argument("--include-axis-baseline", action="store_true")
    parser.add_argument("--phantom", choices=["beads", "random_beads", "shell"], default="beads")
    parser.add_argument("--n-r", type=int, default=16)
    parser.add_argument("--n-z", type=int, default=15)
    parser.add_argument("--r-max", type=float, default=1.5)
    parser.add_argument("--z-max", type=float, default=1.2)
    parser.add_argument("--k", type=float, default=8.0)
    parser.add_argument("--detector-na", type=float, default=0.45)
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
    args.illumination_na_values = parse_float_list(args.illumination_na_values)
    args.n_beta_values = parse_int_list(args.n_beta_values)
    args.iteration_counts = parse_int_list(args.iteration_counts)
    if args.report_iteration not in args.iteration_counts:
        raise ValueError("report-iteration must be included in iteration-counts")
    if args.cpp_threads < 0:
        raise ValueError("cpp-threads must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for n_beta in args.n_beta_values:
        if args.include_axis_baseline:
            rows.append(
                benchmark_case(
                    make_case_args(args, illumination_na=0.0),
                    geometry="axis",
                    n_beta=n_beta,
                )
            )
        for illumination_na in args.illumination_na_values:
            rows.append(
                benchmark_case(
                    make_case_args(args, illumination_na=illumination_na),
                    geometry="shifted",
                    n_beta=n_beta,
                )
            )

    payload = {
        "config": {
            **vars(args),
            "illumination_na_values": args.illumination_na_values,
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
