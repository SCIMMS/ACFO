from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark_odt_iterative_reconstruction import (  # noqa: E402
    ReconstructionContext,
    coeff_norm2,
    data_norm2,
    finufft_adjoint_coeff,
    finufft_forward_coeff,
    plot_history,
    structured_adjoint,
    structured_forward,
)


@dataclass(frozen=True)
class CompositeContext:
    ring: ReconstructionContext
    axis: ReconstructionContext | None

    @property
    def q_count(self) -> int:
        total = int(self.ring.flat_q.count)
        if self.axis is not None:
            total += int(self.axis.flat_q.count)
        return total

    @property
    def build_steps(self) -> list[dict[str, Any]]:
        steps = [{**row, "operator": "ring"} for row in self.ring.build_steps]
        if self.axis is not None:
            steps.extend({**row, "operator": "axis"} for row in self.axis.build_steps)
        return steps


def args_for_operator(
    args: argparse.Namespace,
    *,
    n_illum: int,
    illumination_na: float,
) -> argparse.Namespace:
    payload = vars(args).copy()
    payload["n_illum"] = int(n_illum)
    payload["illumination_na"] = float(illumination_na)
    return argparse.Namespace(**payload)


def build_composite_context(args: argparse.Namespace) -> CompositeContext:
    from benchmark_odt_iterative_reconstruction import build_context

    ring_na = math.sin(math.radians(float(args.illumination_angle_deg)))
    ring_args = args_for_operator(args, n_illum=args.ring_illum, illumination_na=ring_na)
    ring_ctx = build_context(ring_args)
    axis_ctx = None
    if not args.skip_axis_illumination:
        axis_args = args_for_operator(args, n_illum=1, illumination_na=0.0)
        axis_ctx = build_context(axis_args)
    return CompositeContext(ring=ring_ctx, axis=axis_ctx)


def split_residual(ctx: CompositeContext, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    ring_count = int(ctx.ring.flat_q.count)
    ring_residual = residual[:ring_count]
    axis_residual = None
    if ctx.axis is not None:
        axis_residual = residual[ring_count:]
        if axis_residual.shape != (ctx.axis.flat_q.count,):
            raise ValueError("axis residual length does not match axis q-count")
    return ring_residual, axis_residual


def composite_forward(
    ctx: CompositeContext,
    coeff: np.ndarray,
    args: argparse.Namespace,
    *,
    use_finufft: bool,
) -> np.ndarray:
    if use_finufft:
        parts = [
            finufft_forward_coeff(
                ctx.ring,
                coeff,
                eps=args.finufft_eps,
                q_batch_size=args.finufft_q_batch_size,
            )
        ]
        if ctx.axis is not None:
            parts.append(
                finufft_forward_coeff(
                    ctx.axis,
                    coeff,
                    eps=args.finufft_eps,
                    q_batch_size=args.finufft_q_batch_size,
                )
            )
        return np.concatenate(parts)
    parts = [structured_forward(ctx.ring, coeff, args)]
    if ctx.axis is not None:
        parts.append(structured_forward(ctx.axis, coeff, args))
    return np.concatenate(parts)


def composite_adjoint(
    ctx: CompositeContext,
    residual: np.ndarray,
    args: argparse.Namespace,
    *,
    use_finufft: bool,
) -> np.ndarray:
    ring_residual, axis_residual = split_residual(ctx, residual)
    if use_finufft:
        grad = finufft_adjoint_coeff(
            ctx.ring,
            ring_residual,
            eps=args.finufft_eps,
            q_batch_size=args.finufft_q_batch_size,
        )
        if ctx.axis is not None and axis_residual is not None:
            grad = grad + finufft_adjoint_coeff(
                ctx.axis,
                axis_residual,
                eps=args.finufft_eps,
                q_batch_size=args.finufft_q_batch_size,
            )
        return grad
    grad = structured_adjoint(ctx.ring, ring_residual)
    if ctx.axis is not None and axis_residual is not None:
        grad = grad + structured_adjoint(ctx.axis, axis_residual)
    return grad


def run_steepest_descent(
    *,
    label: str,
    ctx: CompositeContext,
    data: np.ndarray,
    true_coeff: np.ndarray,
    args: argparse.Namespace,
    iterations: int,
    use_finufft: bool,
) -> list[dict[str, Any]]:
    x = np.zeros_like(true_coeff)
    pred = np.zeros_like(data)
    data_norm = max(float(np.linalg.norm(data.ravel())), 1e-300)
    true_norm = max(float(np.linalg.norm(true_coeff.ravel())), 1e-300)
    rows: list[dict[str, Any]] = [
        {
            "method": label,
            "iteration": 0,
            "loss_rel": float(np.linalg.norm((pred - data).ravel()) / data_norm),
            "object_rel_l2": float(np.linalg.norm((x - true_coeff).ravel()) / true_norm),
            "alpha": 0.0,
            "iter_s": 0.0,
            "adjoint_s": 0.0,
            "line_forward_s": 0.0,
            "cumulative_iter_s": 0.0,
        }
    ]
    cumulative = 0.0
    for iteration in range(1, int(iterations) + 1):
        residual = pred - data
        start = time.perf_counter()
        adj_start = time.perf_counter()
        grad = composite_adjoint(ctx, residual, args, use_finufft=use_finufft)
        adjoint_s = time.perf_counter() - adj_start
        fw_start = time.perf_counter()
        a_grad = composite_forward(ctx, grad, args, use_finufft=use_finufft)
        line_forward_s = time.perf_counter() - fw_start
        alpha = coeff_norm2(grad) / max(data_norm2(a_grad), 1e-300)
        x = x - alpha * grad
        pred = pred - alpha * a_grad
        elapsed = time.perf_counter() - start
        cumulative += elapsed
        rows.append(
            {
                "method": label,
                "iteration": iteration,
                "loss_rel": float(np.linalg.norm((pred - data).ravel()) / data_norm),
                "object_rel_l2": float(np.linalg.norm((x - true_coeff).ravel()) / true_norm),
                "alpha": float(alpha),
                "iter_s": float(elapsed),
                "adjoint_s": float(adjoint_s),
                "line_forward_s": float(line_forward_s),
                "cumulative_iter_s": float(cumulative),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def final_row(rows: list[dict[str, Any]], method: str) -> dict[str, Any] | None:
    method_rows = [row for row in rows if row["method"] == method]
    if not method_rows:
        return None
    return max(method_rows, key=lambda row: int(row["iteration"]))


def write_summary_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    structured_final = final_row(rows, "structured")
    finufft_final = final_row(rows, "finufft")
    lines = [
        "# ODT realistic lab-geometry reconstruction benchmark",
        "",
        "This benchmark approximates a practical illumination-scanning ODT acquisition as a circular illumination ring plus an optional normal-incidence hologram. The ring and axis operators are prepared separately, then combined as one reconstruction operator.",
        "",
        "## Configuration",
        "",
        f"- ring illumination count: `{summary['ring_illum']}`",
        f"- axis illumination included: `{summary['axis_illumination_included']}`",
        f"- total illumination count: `{summary['total_illumination_count']}`",
        f"- illumination ring angle: `{summary['illumination_angle_deg']}` deg",
        f"- illumination ring NA: `{summary['illumination_ring_na']:.6f}`",
        f"- detector NA: `{summary['detector_na']}`",
        f"- cap: `{summary['cap_radial']} x {summary['cap_phi']}`",
        f"- total q samples: `{summary['total_q_samples']}`",
        f"- object bins: `{summary['object_bins']}`",
        f"- structured geometry build: `{summary['structured_geometry_build_s']:.6f} s`",
        f"- FINUFFT q batch size: `{summary['finufft_q_batch_size']}`",
        "",
        "## Reconstruction Readout",
        "",
        "| method | iterations | final loss rel | final object rel-L2 | median iter s | median adjoint s | median forward s | total incl. setup s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if structured_final is not None:
        lines.append(
            "| structured | {it} | {loss} | {obj} | {med} | {adj} | {fw} | {tot} |".format(
                it=int(structured_final["iteration"]),
                loss=f"{float(structured_final['loss_rel']):.6g}",
                obj=f"{float(structured_final['object_rel_l2']):.6g}",
                med=f"{summary['structured_median_iter_s']:.6f}",
                adj=f"{summary['structured_median_adjoint_s']:.6f}",
                fw=f"{summary['structured_median_forward_s']:.6f}",
                tot=f"{summary['structured_total_including_geometry_s']:.6f}",
            )
        )
    if finufft_final is not None:
        lines.append(
            "| FINUFFT | {it} | {loss} | {obj} | {med} | {adj} | {fw} | {tot} |".format(
                it=int(finufft_final["iteration"]),
                loss=f"{float(finufft_final['loss_rel']):.6g}",
                obj=f"{float(finufft_final['object_rel_l2']):.6g}",
                med=f"{summary['finufft_median_iter_s']:.6f}",
                adj=f"{summary['finufft_median_adjoint_s']:.6f}",
                fw=f"{summary['finufft_median_forward_s']:.6f}",
                tot=f"{summary['finufft_total_s']:.6f}",
            )
        )
    if summary.get("structured_total_at_finufft_final_iteration_s") is not None:
        lines.extend(
            [
                "",
                "Matched-iteration setup-inclusive comparison:",
                "",
                "- structured at FINUFFT final iteration: `{:.6f} s`".format(
                    summary["structured_total_at_finufft_final_iteration_s"]
                ),
                "- FINUFFT at same iteration count: `{:.6f} s`".format(
                    summary["finufft_total_s"]
                ),
                "- FINUFFT / structured matched total: `{:.3f}x`".format(
                    summary["total_speedup_finufft_over_structured"]
                ),
            ]
        )
    if summary.get("break_even_iteration") is not None:
        lines.append(
            f"- Break-even including structured geometry setup occurs at iteration `{summary['break_even_iteration']}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The structured path pays separate one-time preparation costs for the circular ring and normal-axis measurement, then reuses both operators inside the iterative loop.",
            "- FINUFFT is evaluated in q batches when requested; this is a memory-safety baseline and still represents the same generic type-3 operator.",
            "- This is still synthetic weak-scattering reconstruction, but the illumination count and ring-plus-axis geometry are closer to practical ODT acquisition than the earlier 32-view cone-only benchmark.",
        ]
    )
    if summary.get("finufft_data_l2_vs_structured") is not None:
        lines.append(
            f"- FINUFFT/structured data agreement: `{summary['finufft_data_l2_vs_structured']:.6g}` rel-L2."
        )
    if summary.get("figure"):
        lines.append(f"- figure: `{summary['figure']}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    ctx = build_composite_context(args)
    true_coeff = np.ascontiguousarray(ctx.ring.obj.coeff * float(args.object_scale))

    data_start = time.perf_counter()
    data = composite_forward(ctx, true_coeff, args, use_finufft=False)
    synthetic_data_forward_s = time.perf_counter() - data_start

    rows: list[dict[str, Any]] = []
    structured_rows = run_steepest_descent(
        label="structured",
        ctx=ctx,
        data=data,
        true_coeff=true_coeff,
        args=args,
        iterations=args.iterations,
        use_finufft=False,
    )
    rows.extend(structured_rows)

    finufft_error = None
    finufft_rows: list[dict[str, Any]] = []
    if args.include_finufft:
        try:
            finufft_rows = run_steepest_descent(
                label="finufft",
                ctx=ctx,
                data=data,
                true_coeff=true_coeff,
                args=args,
                iterations=args.finufft_iterations or args.iterations,
                use_finufft=True,
            )
            rows.extend(finufft_rows)
        except Exception as exc:  # pragma: no cover - optional dependency/runtime path.
            finufft_error = str(exc)

    structured_iter_times = [
        float(row["iter_s"]) for row in structured_rows if int(row["iteration"]) > 0
    ]
    structured_adjoint_times = [
        float(row["adjoint_s"]) for row in structured_rows if int(row["iteration"]) > 0
    ]
    structured_forward_times = [
        float(row["line_forward_s"]) for row in structured_rows if int(row["iteration"]) > 0
    ]
    finufft_iter_times = [float(row["iter_s"]) for row in finufft_rows if int(row["iteration"]) > 0]
    finufft_adjoint_times = [
        float(row["adjoint_s"]) for row in finufft_rows if int(row["iteration"]) > 0
    ]
    finufft_forward_times = [
        float(row["line_forward_s"]) for row in finufft_rows if int(row["iteration"]) > 0
    ]
    geometry_build_s = float(sum(row["s"] for row in ctx.build_steps))
    structured_total = geometry_build_s + float(structured_rows[-1]["cumulative_iter_s"])
    finufft_total = None if not finufft_rows else float(finufft_rows[-1]["cumulative_iter_s"])
    structured_total_at_finufft_final_iteration_s = None
    total_speedup_finufft_over_structured = None
    if finufft_rows:
        finufft_final_iteration = int(finufft_rows[-1]["iteration"])
        structured_by_it = {
            int(row["iteration"]): float(row["cumulative_iter_s"]) for row in structured_rows
        }
        if finufft_final_iteration in structured_by_it:
            structured_total_at_finufft_final_iteration_s = (
                geometry_build_s + structured_by_it[finufft_final_iteration]
            )
            total_speedup_finufft_over_structured = (
                finufft_total / structured_total_at_finufft_final_iteration_s
                if finufft_total is not None
                else None
            )
    break_even_iteration = None
    if finufft_rows:
        fin_by_it = {int(row["iteration"]): float(row["cumulative_iter_s"]) for row in finufft_rows}
        for row in structured_rows:
            iteration = int(row["iteration"])
            if iteration == 0 or iteration not in fin_by_it:
                continue
            structured_total_at_it = geometry_build_s + float(row["cumulative_iter_s"])
            if structured_total_at_it <= fin_by_it[iteration]:
                break_even_iteration = iteration
                break

    ring_na = math.sin(math.radians(float(args.illumination_angle_deg)))
    summary = {
        "ring_illum": int(args.ring_illum),
        "axis_illumination_included": not bool(args.skip_axis_illumination),
        "total_illumination_count": int(args.ring_illum)
        + (0 if args.skip_axis_illumination else 1),
        "illumination_angle_deg": float(args.illumination_angle_deg),
        "illumination_ring_na": float(ring_na),
        "detector_na": float(args.detector_na),
        "cap_radial": int(args.cap_radial),
        "cap_phi": int(args.cap_phi),
        "ring_q_samples": int(ctx.ring.flat_q.count),
        "axis_q_samples": 0 if ctx.axis is None else int(ctx.axis.flat_q.count),
        "total_q_samples": int(ctx.q_count),
        "object_bins": int(true_coeff.size),
        "n_beta": int(args.n_beta),
        "n_r": int(args.n_r),
        "n_z": int(args.n_z),
        "k": float(args.k),
        "forward_execute_mode": args.forward_execute_mode,
        "forward_kernel_mode": args.forward_kernel_mode,
        "finufft_q_batch_size": int(args.finufft_q_batch_size),
        "structured_geometry_build_s": geometry_build_s,
        "synthetic_data_forward_s": float(synthetic_data_forward_s),
        "structured_median_iter_s": float(median(structured_iter_times)),
        "structured_median_adjoint_s": float(median(structured_adjoint_times)),
        "structured_median_forward_s": float(median(structured_forward_times)),
        "structured_total_including_geometry_s": float(structured_total),
        "structured_total_at_finufft_final_iteration_s": structured_total_at_finufft_final_iteration_s,
        "structured_final_loss_rel": float(structured_rows[-1]["loss_rel"]),
        "structured_final_object_rel_l2": float(structured_rows[-1]["object_rel_l2"]),
        "finufft_median_iter_s": None
        if not finufft_iter_times
        else float(median(finufft_iter_times)),
        "finufft_median_adjoint_s": None
        if not finufft_adjoint_times
        else float(median(finufft_adjoint_times)),
        "finufft_median_forward_s": None
        if not finufft_forward_times
        else float(median(finufft_forward_times)),
        "finufft_total_s": finufft_total,
        "finufft_final_loss_rel": None if not finufft_rows else float(finufft_rows[-1]["loss_rel"]),
        "finufft_final_object_rel_l2": None
        if not finufft_rows
        else float(finufft_rows[-1]["object_rel_l2"]),
        "finufft_error": finufft_error,
        "iter_speedup_finufft_over_structured": None
        if not finufft_iter_times
        else float(median(finufft_iter_times) / median(structured_iter_times)),
        "total_speedup_finufft_over_structured": None
        if total_speedup_finufft_over_structured is None
        else float(total_speedup_finufft_over_structured),
        "break_even_iteration": break_even_iteration,
        "build_steps": ctx.build_steps,
        "figure": str(args.figure) if args.figure else None,
        "history_csv": str(args.csv),
    }
    if args.include_operator_agreement:
        finufft_data = composite_forward(ctx, true_coeff, args, use_finufft=True)
        summary["finufft_data_l2_vs_structured"] = float(
            np.linalg.norm((finufft_data - data).ravel())
            / max(float(np.linalg.norm(data.ravel())), 1e-300)
        )

    write_csv(args.csv, rows)
    write_json(args.out, {"config": vars(args), "summary": summary, "history": rows})
    if args.figure:
        plot_history(args.figure, rows, summary)
    if args.summary_md:
        write_summary_markdown(args.summary_md, summary, rows)
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Synthetic ODT reconstruction benchmark with a practical ring-plus-axis lab geometry."
    )
    p.add_argument("--n-beta", type=int, default=384)
    p.add_argument("--n-r", type=int, default=16)
    p.add_argument("--n-z", type=int, default=15)
    p.add_argument("--r-max", type=float, default=1.0)
    p.add_argument("--z-max", type=float, default=0.8)
    p.add_argument("--phantom", choices=["beads", "random_beads", "shell"], default="random_beads")
    p.add_argument("--object-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--k", type=float, default=17.307319527958313)
    p.add_argument("--detector-na", type=float, default=0.9240924092409241)
    p.add_argument("--illumination-angle-deg", type=float, default=49.0)
    p.add_argument("--ring-illum", type=int, default=100)
    p.add_argument("--skip-axis-illumination", action="store_true")
    p.add_argument("--cap-radial", type=int, default=128)
    p.add_argument("--cap-phi", type=int, default=512)
    p.add_argument("--h-cutoff", type=int, default=None)
    p.add_argument("--h-margin", type=int, default=20)
    p.add_argument("--l-margin", type=int, default=18)
    p.add_argument("--cone-l-prune-threshold", type=float, default=1e-12)
    p.add_argument("--cpp-threads", type=int, default=16)
    p.add_argument(
        "--forward-execute-mode",
        choices=["prepared", "wrapper"],
        default="prepared",
    )
    p.add_argument(
        "--forward-kernel-mode",
        choices=["compact", "partitioned"],
        default="partitioned",
    )
    p.add_argument(
        "--native-prepared-plan-mode",
        choices=["auto", "direct", "gathered", "gathered-zmajor"],
        default="auto",
    )
    p.add_argument("--native-prepared-gather-threshold", type=int, default=8192)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--include-finufft", action="store_true")
    p.add_argument("--finufft-iterations", type=int, default=None)
    p.add_argument("--finufft-eps", type=float, default=1e-12)
    p.add_argument(
        "--finufft-q-batch-size",
        type=int,
        default=1_048_576,
        help="Split FINUFFT target/source q samples into chunks. 0 uses one full FINUFFT call.",
    )
    p.add_argument("--include-operator-agreement", action="store_true")
    p.add_argument("--noise-rel", type=float, default=0.0)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_realistic_geometry_reconstruction.json",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_realistic_geometry_reconstruction_history.csv",
    )
    p.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_realistic_geometry_reconstruction.png",
    )
    p.add_argument(
        "--summary-md",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_realistic_geometry_reconstruction_summary.md",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
