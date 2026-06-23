from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from benchmark_odt_ewald_cap_operator import (
    ROOT,
    build_shifted_axis_factorization,
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
    structured_adjoint_shifted_axis_factored,
    structured_adjoint_shifted_axis_fft_factored,
    structured_forward,
    structured_forward_shifted_axis_factored,
    structured_forward_shifted_axis_fft_factored,
    StructuredOdtPlan,
)


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    if any(item <= 0 for item in values):
        raise ValueError("integer values must be positive")
    return values


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one float")
    if any(item < 0.0 for item in values):
        raise ValueError("float values must be non-negative")
    return values


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


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    value_f = float(value)
    if value_f == 0.0:
        return "0"
    if abs(value_f) < 1e-3 or abs(value_f) >= 1e3:
        return f"{value_f:.{digits}e}"
    return f"{value_f:.{digits}f}"


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


def kernel_mib(kernel) -> float:
    return (kernel.radial.nbytes + kernel.axial.nbytes + kernel.angular.nbytes) / (
        1024.0 * 1024.0
    )


def factorization_mib(factorization) -> float:
    return (factorization.phase.nbytes / (1024.0 * 1024.0)) + kernel_mib(
        factorization.kernel
    )


def benchmark_case(
    args: argparse.Namespace,
    *,
    n_beta: int,
    illumination_na: float,
) -> dict[str, Any]:
    obj = make_cylindrical_object(
        n_r=args.n_r,
        n_z=args.n_z,
        n_beta=n_beta,
        r_max=args.r_max,
        z_max=args.z_max,
        phantom=args.phantom,
        seed=args.seed,
    )
    q_flat = ewald_cap_q_samples(
        k=args.k,
        detector_na=args.detector_na,
        cap_radial=args.cap_radial,
        cap_phi=args.cap_phi,
        geometry="shifted",
        n_illum=args.n_illum,
        illumination_na=illumination_na,
    )
    q_base = ewald_cap_q_samples(
        k=args.k,
        detector_na=args.detector_na,
        cap_radial=args.cap_radial,
        cap_phi=args.cap_phi,
        geometry="axis",
        n_illum=1,
        illumination_na=0.0,
    )
    flat_h_cutoff = (
        recommended_h_cutoff(q_flat, args.r_max, n_beta, args.h_margin)
        if args.h_cutoff is None
        else int(args.h_cutoff)
    )
    factored_h_cutoff = (
        recommended_h_cutoff(q_base, args.r_max, n_beta, args.h_margin)
        if args.h_cutoff is None
        else int(args.h_cutoff)
    )
    flat_plan = StructuredOdtPlan.build(
        r_axis=obj.r_axis,
        z_axis=obj.z_axis,
        beta_axis=obj.beta_axis,
        h_cutoff=flat_h_cutoff,
    )
    factored_plan = StructuredOdtPlan.build(
        r_axis=obj.r_axis,
        z_axis=obj.z_axis,
        beta_axis=obj.beta_axis,
        h_cutoff=factored_h_cutoff,
    )
    backend = resolve_structured_backend(args.structured_backend)
    residual = random_residual(q_flat, seed=args.seed + 7919)

    flat_kernel, flat_build_s, flat_build_times = median_time(
        lambda: build_structured_kernel(flat_plan, q_flat),
        repeats=args.build_repeats,
    )
    factorization, factored_build_s, factored_build_times = median_time(
        lambda: build_shifted_axis_factorization(
            factored_plan,
            k=args.k,
            detector_na=args.detector_na,
            cap_radial=args.cap_radial,
            cap_phi=args.cap_phi,
            n_illum=args.n_illum,
            illumination_na=illumination_na,
        ),
        repeats=args.build_repeats,
    )

    def flat_pair(active_kernel):
        forward = structured_forward(
            flat_plan,
            obj.coeff,
            q_flat,
            kernel=active_kernel,
            backend=backend,
            cpp_threads=args.cpp_threads,
        )
        adjoint = structured_adjoint(
            flat_plan,
            q_flat,
            residual,
            kernel=active_kernel,
            backend=backend,
            cpp_threads=args.cpp_threads,
        )
        return forward, adjoint

    def factored_pair(active_factorization):
        forward = structured_forward_shifted_axis_factored(
            factored_plan,
            obj.coeff,
            active_factorization,
            backend=backend,
            cpp_threads=args.cpp_threads,
        )
        adjoint = structured_adjoint_shifted_axis_factored(
            factored_plan,
            active_factorization,
            residual,
            backend=backend,
            cpp_threads=args.cpp_threads,
        )
        return forward, adjoint

    def fft_factored_pair(active_factorization):
        forward = structured_forward_shifted_axis_fft_factored(
            factored_plan,
            obj.coeff,
            active_factorization,
            backend=backend,
            cpp_threads=args.cpp_threads,
            phase_backend=args.phase_backend,
        )
        adjoint = structured_adjoint_shifted_axis_fft_factored(
            factored_plan,
            active_factorization,
            residual,
            backend=backend,
            cpp_threads=args.cpp_threads,
            phase_backend=args.phase_backend,
        )
        return forward, adjoint

    (flat_forward, flat_adjoint), flat_hot_pair_s, flat_hot_times = median_time(
        lambda: flat_pair(flat_kernel),
        repeats=args.hot_repeats,
    )
    (factored_forward, factored_adjoint), factored_hot_pair_s, factored_hot_times = (
        median_time(lambda: factored_pair(factorization), repeats=args.hot_repeats)
    )
    (fft_factored_forward, fft_factored_adjoint), fft_factored_hot_pair_s, fft_factored_hot_times = (
        median_time(lambda: fft_factored_pair(factorization), repeats=args.hot_repeats)
    )
    _, flat_cold_pair_s, flat_cold_times = median_time(
        lambda: flat_pair(build_structured_kernel(flat_plan, q_flat)),
        repeats=args.cold_repeats,
    )
    _, factored_cold_pair_s, factored_cold_times = median_time(
        lambda: factored_pair(
            build_shifted_axis_factorization(
                factored_plan,
                k=args.k,
                detector_na=args.detector_na,
                cap_radial=args.cap_radial,
                cap_phi=args.cap_phi,
                n_illum=args.n_illum,
                illumination_na=illumination_na,
            )
        ),
        repeats=args.cold_repeats,
    )
    _, fft_factored_cold_pair_s, fft_factored_cold_times = median_time(
        lambda: fft_factored_pair(
            build_shifted_axis_factorization(
                factored_plan,
                k=args.k,
                detector_na=args.detector_na,
                cap_radial=args.cap_radial,
                cap_phi=args.cap_phi,
                n_illum=args.n_illum,
                illumination_na=illumination_na,
            )
        ),
        repeats=args.cold_repeats,
    )

    finufft_pair_s = None
    finufft_pair_times: list[float] = []
    finufft_forward_error = None
    finufft_adjoint_error = None
    finufft_skip_reason = None
    try:
        (finufft_forward_value, finufft_adjoint_value), finufft_pair_s, finufft_pair_times = (
            median_time(
                lambda: (
                    finufft_forward(obj, q_flat, eps=args.finufft_eps),
                    finufft_adjoint(obj, q_flat, residual, eps=args.finufft_eps),
                ),
                repeats=args.hot_repeats,
            )
        )
        finufft_forward_error = relative_l2(factored_forward, finufft_forward_value)
        finufft_adjoint_error = relative_l2(factored_adjoint, finufft_adjoint_value)
    except Exception as exc:  # pragma: no cover - optional dependency/runtime path
        finufft_skip_reason = str(exc)

    row: dict[str, Any] = {
        "status": "ok",
        "n_beta": n_beta,
        "n_r": args.n_r,
        "n_z": args.n_z,
        "r_max": args.r_max,
        "z_max": args.z_max,
        "k": args.k,
        "detector_na": args.detector_na,
        "illumination_na": illumination_na,
        "n_illum": args.n_illum,
        "cap_radial": args.cap_radial,
        "cap_phi": args.cap_phi,
        "q_samples": q_flat.count,
        "base_q_samples": q_base.count,
        "unique_flat_q_perp_rounded12": int(np.unique(np.round(q_flat.q_perp, 12)).size),
        "unique_base_q_perp_rounded12": int(np.unique(np.round(q_base.q_perp, 12)).size),
        "flat_h_cutoff": flat_h_cutoff,
        "factored_h_cutoff": factored_h_cutoff,
        "flat_used_modes": flat_plan.used_modes,
        "factored_used_modes": factored_plan.used_modes,
        "structured_backend": backend,
        "phase_backend": args.phase_backend,
        "cpp_threads": args.cpp_threads,
        "flat_build_s": flat_build_s,
        "factored_build_s": factored_build_s,
        "flat_kernel_mib": kernel_mib(flat_kernel),
        "factored_cache_mib": factorization_mib(factorization),
        "flat_hot_pair_s": flat_hot_pair_s,
        "factored_hot_pair_s": factored_hot_pair_s,
        "fft_factored_hot_pair_s": fft_factored_hot_pair_s,
        "flat_cold_pair_s": flat_cold_pair_s,
        "factored_cold_pair_s": factored_cold_pair_s,
        "fft_factored_cold_pair_s": fft_factored_cold_pair_s,
        "finufft_pair_s": finufft_pair_s,
        "flat_vs_factored_build_speedup": speedup(flat_build_s, factored_build_s),
        "flat_vs_factored_hot_pair_speedup": speedup(flat_hot_pair_s, factored_hot_pair_s),
        "flat_vs_fft_factored_hot_pair_speedup": speedup(
            flat_hot_pair_s,
            fft_factored_hot_pair_s,
        ),
        "flat_vs_factored_cold_pair_speedup": speedup(flat_cold_pair_s, factored_cold_pair_s),
        "flat_vs_fft_factored_cold_pair_speedup": speedup(
            flat_cold_pair_s,
            fft_factored_cold_pair_s,
        ),
        "finufft_vs_flat_hot_pair_speedup": speedup(finufft_pair_s, flat_hot_pair_s),
        "finufft_vs_factored_hot_pair_speedup": speedup(
            finufft_pair_s,
            factored_hot_pair_s,
        ),
        "finufft_vs_fft_factored_hot_pair_speedup": speedup(
            finufft_pair_s,
            fft_factored_hot_pair_s,
        ),
        "factored_forward_l2_vs_flat": relative_l2(factored_forward, flat_forward),
        "factored_adjoint_l2_vs_flat": relative_l2(factored_adjoint, flat_adjoint),
        "fft_factored_forward_l2_vs_flat": relative_l2(
            fft_factored_forward,
            flat_forward,
        ),
        "fft_factored_adjoint_l2_vs_flat": relative_l2(
            fft_factored_adjoint,
            flat_adjoint,
        ),
        "factored_forward_l2_vs_finufft": finufft_forward_error,
        "factored_adjoint_l2_vs_finufft": finufft_adjoint_error,
        "finufft_skip_reason": finufft_skip_reason,
        "flat_build_times_s": " ".join(f"{item:.9g}" for item in flat_build_times),
        "factored_build_times_s": " ".join(f"{item:.9g}" for item in factored_build_times),
        "flat_hot_pair_times_s": " ".join(f"{item:.9g}" for item in flat_hot_times),
        "factored_hot_pair_times_s": " ".join(f"{item:.9g}" for item in factored_hot_times),
        "fft_factored_hot_pair_times_s": " ".join(
            f"{item:.9g}" for item in fft_factored_hot_times
        ),
        "flat_cold_pair_times_s": " ".join(f"{item:.9g}" for item in flat_cold_times),
        "factored_cold_pair_times_s": " ".join(f"{item:.9g}" for item in factored_cold_times),
        "fft_factored_cold_pair_times_s": " ".join(
            f"{item:.9g}" for item in fft_factored_cold_times
        ),
        "finufft_pair_times_s": " ".join(f"{item:.9g}" for item in finufft_pair_times),
    }

    for iterations in args.iteration_counts:
        flat_amortized = (flat_build_s + float(iterations) * flat_hot_pair_s) / float(
            iterations
        )
        factored_amortized = (
            factored_build_s + float(iterations) * factored_hot_pair_s
        ) / float(iterations)
        fft_factored_amortized = (
            factored_build_s + float(iterations) * fft_factored_hot_pair_s
        ) / float(iterations)
        row[f"flat_amortized_pair_s_n{iterations}"] = flat_amortized
        row[f"factored_amortized_pair_s_n{iterations}"] = factored_amortized
        row[f"fft_factored_amortized_pair_s_n{iterations}"] = fft_factored_amortized
        row[f"flat_vs_factored_amortized_speedup_n{iterations}"] = speedup(
            flat_amortized,
            factored_amortized,
        )
        row[f"flat_vs_fft_factored_amortized_speedup_n{iterations}"] = speedup(
            flat_amortized,
            fft_factored_amortized,
        )
        row[f"finufft_vs_factored_amortized_speedup_n{iterations}"] = speedup(
            finufft_pair_s,
            factored_amortized,
        )
        row[f"finufft_vs_fft_factored_amortized_speedup_n{iterations}"] = speedup(
            finufft_pair_s,
            fft_factored_amortized,
        )

    return row


def case_label(row: dict[str, Any]) -> str:
    return f"illum={row['illumination_na']:.3g} nb={row['n_beta']}"


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    report_iteration = payload["config"]["report_iteration"]
    lines = [
        "# ODT shifted-axis factorization benchmark",
        "",
        "This benchmark compares the current flat shifted-q operator with a phase-ramp factorization:",
        "",
        "`F(k(s_out - s_in)) = F_axis[object * exp(i k((0,0,1)-s_in) dot x)]`.",
        "",
        "Values above `1x` in the flat/factored columns mean the factored path is faster than the current flat shifted-q path.",
        "",
        "## Results",
        "",
        "| case | q | base q | flat modes | axis modes | flat build s | axis build s | build speedup | flat hot s | contract hot s | axis-FFT hot s | flat/axis-FFT hot | FINUFFT s | axis-FFT vs FINUFFT | N axis-FFT speedup | fwd err | adj err |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {q} | {base_q} | {fm} | {gm} | `{fb}` | `{gb}` | `{bs}` | `{fh}` | `{gh}` | `{ffth}` | `{ffths}` | `{fin}` | `{fins}` | `{amort}` | `{fe}` | `{ae}` |".format(
                case=case_label(row),
                q=row["q_samples"],
                base_q=row["base_q_samples"],
                fm=row["flat_used_modes"],
                gm=row["factored_used_modes"],
                fb=fmt(row["flat_build_s"], 5),
                gb=fmt(row["factored_build_s"], 5),
                bs=fmt(row["flat_vs_factored_build_speedup"], 4),
                fh=fmt(row["flat_hot_pair_s"], 5),
                gh=fmt(row["factored_hot_pair_s"], 5),
                ffth=fmt(row["fft_factored_hot_pair_s"], 5),
                ffths=fmt(row["flat_vs_fft_factored_hot_pair_speedup"], 4),
                fin=fmt(row["finufft_pair_s"], 5),
                fins=fmt(row["finufft_vs_fft_factored_hot_pair_speedup"], 4),
                amort=fmt(
                    row[f"flat_vs_fft_factored_amortized_speedup_n{report_iteration}"],
                    4,
                ),
                fe=fmt(row["fft_factored_forward_l2_vs_flat"], 4),
                ae=fmt(row["fft_factored_adjoint_l2_vs_flat"], 4),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The contract factored path reuses the axis detector cap kernel but still loops over detector phi samples.",
            "- The axis-FFT factored path additionally folds angular harmonics modulo `cap_phi` and evaluates detector phi samples with FFTs.",
            "- Build speedup reflects the smaller axis detector kernel plus the cached phase-ramp cost.",
            f"- `N axis-FFT speedup` uses N={report_iteration} forward-adjoint pairs after one setup.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path_png: Path, path_svg: Path, payload: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    rows = payload["rows"]
    n_beta_values = payload["config"]["n_beta_values"]
    report_iteration = payload["config"]["report_iteration"]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5), dpi=180)
    ax_build, ax_hot = axes
    for ax in axes:
        ax.axhline(1.0, color="#444444", lw=1.0, ls="--", alpha=0.75)
        ax.grid(alpha=0.25)
        ax.set_xlabel("illumination NA")
        ax.set_ylabel("flat / factored speedup")
    for n_beta in n_beta_values:
        subset = [row for row in rows if row["n_beta"] == n_beta]
        subset.sort(key=lambda row: row["illumination_na"])
        x = [row["illumination_na"] for row in subset]
        ax_build.plot(
            x,
            [row["flat_vs_factored_build_speedup"] for row in subset],
            marker="o",
            lw=1.8,
            label=f"build nb={n_beta}",
        )
        ax_hot.plot(
            x,
            [row["flat_vs_fft_factored_hot_pair_speedup"] for row in subset],
            marker="o",
            lw=1.8,
            label=f"hot nb={n_beta}",
        )
        ax_hot.plot(
            x,
            [
                row[f"flat_vs_fft_factored_amortized_speedup_n{report_iteration}"]
                for row in subset
            ],
            marker="s",
            lw=1.3,
            ls="--",
            label=f"N={report_iteration} nb={n_beta}",
        )
    ax_build.set_title("A. Setup cost")
    ax_hot.set_title("B. Hot and amortized pairs")
    ax_build.legend(frameon=False, fontsize=8)
    ax_hot.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png)
    fig.savefig(path_svg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark phase-ramp factorization for shifted ODT caps."
    )
    parser.add_argument("--output-prefix", default="benchmark_results/odt_shift_factorization")
    parser.add_argument("--illumination-na-values", default="0,0.02,0.05,0.1,0.2,0.3")
    parser.add_argument("--n-beta-values", default="192,384")
    parser.add_argument("--iteration-counts", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--report-iteration", type=int, default=32)
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
    parser.add_argument("--phase-backend", choices=["fft", "selected-dft"], default="fft")
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
    rows = [
        benchmark_case(args, n_beta=n_beta, illumination_na=illumination_na)
        for n_beta in args.n_beta_values
        for illumination_na in args.illumination_na_values
    ]
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
