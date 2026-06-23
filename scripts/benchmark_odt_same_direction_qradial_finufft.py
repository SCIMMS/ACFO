from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_odt_cone_axis_decomposition import fmt
from benchmark_odt_cone_illumination import parse_int_list
from benchmark_odt_same_direction_finufft import benchmark_case, write_plot


def parse_q_cases(text: str) -> list[tuple[float, int, int]]:
    cases: list[tuple[float, int, int]] = []
    for item in text.split(","):
        token = item.strip().lower().replace(" ", "")
        if not token:
            continue
        if ":" not in token or "x" not in token:
            raise ValueError(f"q case must look like 8:8x32, got {item!r}")
        k_text, grid_text = token.split(":", 1)
        radial_text, phi_text = grid_text.split("x", 1)
        cases.append((float(k_text), int(radial_text), int(phi_text)))
    if not cases:
        raise ValueError("at least one q-radial case is required")
    return cases


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


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    lines = [
        "# ODT same-direction radial-q growth vs FINUFFT",
        "",
        "This benchmark scales the detector radial extent by increasing both `k` and the detector cap grid. The intent is to probe the regime where the flat q-list grows roughly like a 2D detector area, while the same-direction factorization reuses harmonic/rank structure.",
        "",
        "## Results",
        "",
        "| k | cap grid | flat q | q_perp max | method | rank | fwd s | adj s | FINUFFT/fwd | FINUFFT/adj | fwd err vs grouped | adj err vs grouped |",
        "| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {k} | {grid} | {fq} | `{qp}` | {method} | {rank} | `{fwd_s}` | `{adj_s}` | `{fwd_speed}` | `{adj_speed}` | `{fwd}` | `{adj}` |".format(
                k=fmt(row["k"], 4),
                grid=row["cap_grid"],
                fq=row["flat_q_samples"],
                qp=fmt(row.get("flat_q_perp_max"), 4),
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
    finite_rows = [
        row
        for row in rows
        if row["method"] != "finufft_type3" and row.get("finufft_over_method_speedup") is not None
    ]
    rank8 = [row for row in finite_rows if row["method"] == "svd_rank_8"]
    rank4 = [row for row in finite_rows if row["method"] == "svd_rank_4"]
    grouped = [row for row in finite_rows if row["method"] == "grouped_same_direction_z0"]
    lines.extend(["", "## Interpretation", ""])
    lines.append("- `FINUFFT/fwd` and `FINUFFT/adj` greater than 1 mean the listed method is faster than FINUFFT in that direction.")
    lines.append("- FINUFFT forward and adjoint are timed separately as type-3 calls on the full shifted q-list.")
    lines.append("- `pair_s` remains available in the JSON/CSV as `forward_s + adjoint_s`; this summary table intentionally reports split timings.")
    lines.append("- The q-radial cases intentionally co-scale `k`, `cap_radial`, and `cap_phi`; this increases both q range and detector q count.")
    if rows:
        q_perp_values = [row.get("flat_q_perp_max") for row in rows if row.get("flat_q_perp_max") is not None]
        flat_q_values = [row["flat_q_samples"] for row in rows]
        finufft_rows = [row for row in rows if row["method"] == "finufft_type3" and row.get("forward_s") is not None]
        if q_perp_values and flat_q_values:
            lines.append(
                "- This is the stricter FINUFFT stress case: `q_perp max` grows from `{qlo}` to `{qhi}`, and the flat q-list grows from {flo:,} to {fhi:,} samples.".format(
                    qlo=fmt(min(q_perp_values), 4),
                    qhi=fmt(max(q_perp_values), 4),
                    flo=min(flat_q_values),
                    fhi=max(flat_q_values),
                )
            )
        if finufft_rows:
            lines.append(
                "- FINUFFT forward time grows from `{fwd_lo}` s to `{fwd_hi}` s; FINUFFT adjoint time grows from `{adj_lo}` s to `{adj_hi}` s over this sweep.".format(
                    fwd_lo=fmt(min(row["forward_s"] for row in finufft_rows), 5),
                    fwd_hi=fmt(max(row["forward_s"] for row in finufft_rows), 5),
                    adj_lo=fmt(min(row["adjoint_s"] for row in finufft_rows), 5),
                    adj_hi=fmt(max(row["adjoint_s"] for row in finufft_rows), 5),
                )
            )
    if rank8:
        lines.append(
            "- Rank 8 forward speedup range: {flo}x-{fhi}x; adjoint speedup range: {alo}x-{ahi}x; adjoint errors range from `{elo}` to `{ehi}`.".format(
                flo=fmt(min(row["finufft_over_method_forward_speedup"] for row in rank8), 4),
                fhi=fmt(max(row["finufft_over_method_forward_speedup"] for row in rank8), 4),
                alo=fmt(min(row["finufft_over_method_adjoint_speedup"] for row in rank8), 4),
                ahi=fmt(max(row["finufft_over_method_adjoint_speedup"] for row in rank8), 4),
                elo=fmt(min(row["method_adjoint_l2_vs_grouped"] for row in rank8), 4),
                ehi=fmt(max(row["method_adjoint_l2_vs_grouped"] for row in rank8), 4),
            )
        )
    if rank4:
        lines.append(
            "- Rank 4 screening forward speedup range: {flo}x-{fhi}x; adjoint speedup range: {alo}x-{ahi}x; adjoint errors range from `{elo}` to `{ehi}`.".format(
                flo=fmt(min(row["finufft_over_method_forward_speedup"] for row in rank4), 4),
                fhi=fmt(max(row["finufft_over_method_forward_speedup"] for row in rank4), 4),
                alo=fmt(min(row["finufft_over_method_adjoint_speedup"] for row in rank4), 4),
                ahi=fmt(max(row["finufft_over_method_adjoint_speedup"] for row in rank4), 4),
                elo=fmt(min(row["method_adjoint_l2_vs_grouped"] for row in rank4), 4),
                ehi=fmt(max(row["method_adjoint_l2_vs_grouped"] for row in rank4), 4),
            )
        )
    if grouped:
        lines.append(
            "- Exact grouped forward speedup range: {flo}x-{fhi}x; adjoint speedup range: {alo}x-{ahi}x, with numerical-precision agreement to itself.".format(
                flo=fmt(min(row["finufft_over_method_forward_speedup"] for row in grouped), 4),
                fhi=fmt(max(row["finufft_over_method_forward_speedup"] for row in grouped), 4),
                alo=fmt(min(row["finufft_over_method_adjoint_speedup"] for row in grouped), 4),
                ahi=fmt(max(row["finufft_over_method_adjoint_speedup"] for row in grouped), 4),
            )
        )

    def best_rank_line(threshold: float) -> str | None:
        pieces = []
        for k in sorted({row["k"] for row in rows}):
            candidates = [
                row
                for row in finite_rows
                if row["k"] == k
                and row["rank"] is not None
                and row.get("method_adjoint_l2_vs_grouped") is not None
                and row["method_adjoint_l2_vs_grouped"] <= threshold
            ]
            if not candidates:
                continue
            best = min(candidates, key=lambda row: row["pair_s"])
            pieces.append(
                "k={k}: rank {rank}, fwd {fwd}x, adj {adj}x".format(
                    k=fmt(k, 4),
                    rank=best["rank"],
                    fwd=fmt(best["finufft_over_method_forward_speedup"], 4),
                    adj=fmt(best["finufft_over_method_adjoint_speedup"], 4),
                )
            )
        if not pieces:
            return None
        return "- Fastest split-sum ranks for adjoint error <= `{thr}`: {items}.".format(
            thr=fmt(threshold, 1),
            items="; ".join(pieces),
        )

    for threshold in (1e-6, 1e-12):
        line = best_rank_line(threshold)
        if line is not None:
            lines.append(line)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark same-direction radial-q growth against FINUFFT.")
    parser.add_argument("--output-prefix", default="benchmark_results/odt_same_direction_zperp_finufft_qradial_growth")
    parser.add_argument("--q-cases", default="8:8x32,16:16x64,32:32x128,48:48x192")
    parser.add_argument("--n-mag", type=int, default=32)
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
    args.q_cases = parse_q_cases(args.q_cases)
    args.svd_rank_values = parse_int_list(args.svd_rank_values)
    args.n_mag_values = [args.n_mag]
    args.cap_grid_values = [(radial, phi) for _, radial, phi in args.q_cases]
    if args.cpp_threads < 0:
        raise ValueError("cpp-threads must be non-negative")
    if args.finufft_eps <= 0.0:
        raise ValueError("finufft-eps must be positive")
    return args


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for k, cap_radial, cap_phi in args.q_cases:
        case_args = copy.copy(args)
        case_args.k = k
        rows.extend(
            benchmark_case(
                case_args,
                n_mag=args.n_mag,
                cap_radial=cap_radial,
                cap_phi=cap_phi,
            )
        )
    payload = {
        "args": {
            "q_cases": args.q_cases,
            "n_mag": args.n_mag,
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
    write_plot(output_prefix.with_suffix(".png"), output_prefix.with_suffix(".svg"), payload)
    print(json.dumps({"json": str(output_prefix.with_suffix(".json")), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
