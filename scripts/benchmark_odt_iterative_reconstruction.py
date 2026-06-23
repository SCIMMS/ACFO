from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark_odt_cone_axis_decomposition import (  # noqa: E402
    build_cone_axis_decomposition,
    default_l_cutoff,
    decomposed_forward,
)
from benchmark_odt_cone_illumination import (  # noqa: E402
    cone_illumination_directions,
    cone_q_samples,
)
from benchmark_odt_ewald_cap_operator import (  # noqa: E402
    StructuredOdtPlan,
    _cpp_odt_module,
    finufft_adjoint,
    make_cylindrical_object,
    recommended_h_cutoff,
    relative_l2,
)
from profile_odt_cone_axis_bottleneck import axis_factor_pack  # noqa: E402
from profile_odt_cone_axis_execute_only import (  # noqa: E402
    PreparedAdjointExecute,
    native_plan_effective_mode,
    prepared_adjoint_execute,
)


@dataclass(frozen=True)
class PreparedForwardExecute:
    decomp: Any
    radial: np.ndarray
    axial: np.ndarray
    mode_phase: np.ndarray
    slots: np.ndarray
    cap_radial: int
    cap_phi: int
    n_illum: int
    cpp_threads: int
    cpp_odt: Any


@dataclass(frozen=True)
class ReconstructionContext:
    obj: Any
    flat_q: Any
    base_q: Any
    plan: StructuredOdtPlan
    decomp: Any
    prepared: PreparedAdjointExecute
    prepared_forward: PreparedForwardExecute
    build_steps: list[dict[str, Any]]
    l_cutoff: int
    axis_h_cutoff: int


def timed(label: str, func: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    start = time.perf_counter()
    value = func()
    elapsed = time.perf_counter() - start
    return value, {"step": label, "s": float(elapsed)}


def norm_rel(candidate: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.linalg.norm(np.ravel(reference)))
    if denom == 0.0:
        return float(np.linalg.norm(np.ravel(candidate)))
    return float(np.linalg.norm(np.ravel(candidate - reference)) / denom)


def coeff_norm2(coeff: np.ndarray) -> float:
    return float(np.vdot(np.ravel(coeff), np.ravel(coeff)).real)


def data_norm2(data: np.ndarray) -> float:
    return float(np.vdot(data.ravel(), data.ravel()).real)


def build_context(args: argparse.Namespace) -> ReconstructionContext:
    build_steps: list[dict[str, Any]] = []

    obj, row = timed(
        "object_grid_and_phantom_build",
        lambda: make_cylindrical_object(
            n_r=args.n_r,
            n_z=args.n_z,
            n_beta=args.n_beta,
            r_max=args.r_max,
            z_max=args.z_max,
            phantom=args.phantom,
            seed=args.seed,
        ),
    )
    build_steps.append(row)

    q_tuple, row = timed(
        "lab_illumination_and_detector_q_build",
        lambda: (
            lambda illumination: (
                *cone_q_samples(
                    k=args.k,
                    detector_na=args.detector_na,
                    cap_radial=args.cap_radial,
                    cap_phi=args.cap_phi,
                    illumination=illumination,
                ),
            )
        )(
            cone_illumination_directions(
                n_illum=args.n_illum,
                illumination_na=args.illumination_na,
            )[0]
        ),
    )
    flat_q, base_q = q_tuple
    build_steps.append(row)

    axis_h_cutoff = (
        recommended_h_cutoff(base_q, args.r_max, args.n_beta, args.h_margin)
        if args.h_cutoff is None
        else int(args.h_cutoff)
    )
    plan, row = timed(
        "structured_axis_plan_build",
        lambda: StructuredOdtPlan.build(
            r_axis=obj.r_axis,
            z_axis=obj.z_axis,
            beta_axis=obj.beta_axis,
            h_cutoff=axis_h_cutoff,
        ),
    )
    build_steps.append(row)

    l_cutoff = default_l_cutoff(
        k=args.k,
        illumination_na=args.illumination_na,
        r_max=args.r_max,
        margin=args.l_margin,
        n_beta=args.n_beta,
    )
    decomp, row = timed(
        "cone_axis_decomposition_build",
        lambda: build_cone_axis_decomposition(
            plan,
            k=args.k,
            detector_na=args.detector_na,
            cap_radial=args.cap_radial,
            cap_phi=args.cap_phi,
            illumination_na=args.illumination_na,
            n_illum=args.n_illum,
            l_cutoff=l_cutoff,
            adaptive_l_threshold=args.cone_l_prune_threshold,
        ),
    )
    build_steps.append(row)

    factors, row = timed("execute_factor_pack", lambda: axis_factor_pack(decomp))
    radial, axial, mode_phase, slots = factors
    build_steps.append(row)

    cpp_odt = _cpp_odt_module(required=True)
    native_prepared_tables = None
    native_prepared_plan = None
    if (
        decomp.active_l_offsets is not None
        and decomp.active_l_indices is not None
        and hasattr(cpp_odt, "cone_axis_prepare_adjoint_pruned")
    ):
        native_prepared_tables, row = timed(
            "native_adjoint_prepare_pruned",
            lambda: cpp_odt.cone_axis_prepare_adjoint_pruned(
                radial,
                axial,
                mode_phase,
                decomp.transverse_coeff,
                decomp.psi_phase,
                decomp.axial_phase,
                decomp.source_slots,
                decomp.active_l_offsets,
                decomp.active_l_indices,
            ),
        )
        build_steps.append(row)
    if native_prepared_tables is not None and hasattr(cpp_odt, "ConeAxisPreparedAdjointPlan"):
        native_prepared_plan, row = timed(
            "native_prepared_adjoint_plan_build",
            lambda: cpp_odt.ConeAxisPreparedAdjointPlan(
                slots,
                decomp.active_l_offsets,
                *native_prepared_tables,
                int(decomp.plan.n_beta),
            ),
        )
        build_steps.append(row)

    prepared = PreparedAdjointExecute(
        decomp=decomp,
        radial=radial,
        axial=axial,
        mode_phase=mode_phase,
        slots=slots,
        native_prepared_tables=native_prepared_tables,
        native_prepared_plan=native_prepared_plan,
        native_prepared_plan_mode=args.native_prepared_plan_mode,
        native_prepared_gather_threshold=args.native_prepared_gather_threshold,
        cap_radial=int(decomp.factorization.cap_radial),
        cap_phi=int(decomp.factorization.cap_phi),
        n_illum=int(decomp.illumination_phi.size),
        cpp_threads=args.cpp_threads,
        cpp_odt=cpp_odt,
    )
    prepared_forward = PreparedForwardExecute(
        decomp=decomp,
        radial=radial,
        axial=axial,
        mode_phase=mode_phase,
        slots=slots,
        cap_radial=int(decomp.factorization.cap_radial),
        cap_phi=int(decomp.factorization.cap_phi),
        n_illum=int(decomp.illumination_phi.size),
        cpp_threads=args.cpp_threads,
        cpp_odt=cpp_odt,
    )
    return ReconstructionContext(
        obj=obj,
        flat_q=flat_q,
        base_q=base_q,
        plan=plan,
        decomp=decomp,
        prepared=prepared,
        prepared_forward=prepared_forward,
        build_steps=build_steps,
        l_cutoff=l_cutoff,
        axis_h_cutoff=axis_h_cutoff,
    )


def prepared_forward_execute(
    ctx: ReconstructionContext,
    coeff: np.ndarray,
    *,
    kernel_mode: str,
) -> np.ndarray:
    prepared = ctx.prepared_forward
    decomp = prepared.decomp
    coeff_h_full = np.ascontiguousarray(np.fft.ifft(coeff, axis=2) * float(decomp.plan.n_beta))
    if decomp.active_l_offsets is not None and decomp.active_l_indices is not None:
        if kernel_mode == "partitioned":
            if not hasattr(prepared.cpp_odt, "cone_axis_forward_fold_pruned_partitioned"):
                raise RuntimeError(
                    "waxs_cake._cpp_odt lacks cone_axis_forward_fold_pruned_partitioned"
                )
            folded = prepared.cpp_odt.cone_axis_forward_fold_pruned_partitioned(
                coeff_h_full,
                prepared.radial,
                prepared.axial,
                prepared.mode_phase,
                prepared.slots,
                decomp.transverse_coeff,
                decomp.psi_phase,
                decomp.axial_phase,
                decomp.source_slots,
                decomp.active_l_offsets,
                decomp.active_l_indices,
                int(prepared.cap_phi),
                int(prepared.cpp_threads),
            )
        else:
            folded = prepared.cpp_odt.cone_axis_forward_fold_pruned(
                coeff_h_full,
                prepared.radial,
                prepared.axial,
                prepared.mode_phase,
                prepared.slots,
                decomp.transverse_coeff,
                decomp.psi_phase,
                decomp.axial_phase,
                decomp.source_slots,
                decomp.active_l_offsets,
                decomp.active_l_indices,
                int(prepared.cap_phi),
                int(prepared.cpp_threads),
            )
    else:
        if kernel_mode == "partitioned":
            raise RuntimeError("partitioned forward kernel currently requires active l pruning")
        folded = prepared.cpp_odt.cone_axis_forward_fold(
            coeff_h_full,
            prepared.radial,
            prepared.axial,
            prepared.mode_phase,
            prepared.slots,
            decomp.transverse_coeff,
            decomp.psi_phase,
            decomp.axial_phase,
            decomp.source_slots,
            int(prepared.cap_phi),
            int(prepared.cpp_threads),
        )
    return np.fft.fft(folded, axis=2).reshape(
        prepared.n_illum * prepared.cap_radial * prepared.cap_phi
    )


def structured_forward(ctx: ReconstructionContext, coeff: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.forward_execute_mode == "prepared":
        return prepared_forward_execute(ctx, coeff, kernel_mode=args.forward_kernel_mode)
    return decomposed_forward(
        coeff,
        ctx.decomp,
        backend="cpp",
        cpp_threads=args.cpp_threads,
        forward_mode="fused",
    )


def structured_adjoint(ctx: ReconstructionContext, residual: np.ndarray) -> np.ndarray:
    return prepared_adjoint_execute(ctx.prepared, residual)


def finufft_forward_coeff(
    ctx: ReconstructionContext,
    coeff: np.ndarray,
    *,
    eps: float,
    q_batch_size: int = 0,
) -> np.ndarray:
    try:
        import finufft
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("finufft is not installed") from exc
    obj = ctx.obj
    q = ctx.flat_q
    q_batch_size = int(q_batch_size)
    coeff_flat = np.ascontiguousarray(coeff.ravel()).astype(np.complex128, copy=False)
    if q_batch_size > 0 and q_batch_size < q.count:
        out = np.empty(q.count, dtype=np.complex128)
        for start in range(0, q.count, q_batch_size):
            stop = min(start + q_batch_size, q.count)
            out[start:stop] = finufft.nufft3d3(
                obj.x,
                obj.y,
                obj.z,
                coeff_flat,
                q.qx[start:stop],
                q.qy[start:stop],
                q.qz[start:stop],
                eps=eps,
                isign=1,
            )
        return out
    return finufft.nufft3d3(
        obj.x,
        obj.y,
        obj.z,
        coeff_flat,
        q.qx,
        q.qy,
        q.qz,
        eps=eps,
        isign=1,
    )


def finufft_adjoint_coeff(
    ctx: ReconstructionContext,
    residual: np.ndarray,
    *,
    eps: float,
    q_batch_size: int = 0,
) -> np.ndarray:
    q_batch_size = int(q_batch_size)
    if q_batch_size <= 0 or q_batch_size >= ctx.flat_q.count:
        return finufft_adjoint(ctx.obj, ctx.flat_q, residual, eps=eps)
    try:
        import finufft
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("finufft is not installed") from exc
    obj = ctx.obj
    q = ctx.flat_q
    residual = np.asarray(residual, dtype=np.complex128)
    if residual.shape != (q.count,):
        raise ValueError("residual shape must match q sample count")
    out = np.zeros(obj.bins, dtype=np.complex128)
    for start in range(0, q.count, q_batch_size):
        stop = min(start + q_batch_size, q.count)
        out += finufft.nufft3d3(
            q.qx[start:stop],
            q.qy[start:stop],
            q.qz[start:stop],
            residual[start:stop],
            obj.x,
            obj.y,
            obj.z,
            eps=eps,
            isign=-1,
        )
    return out.reshape(obj.coeff.shape)


def run_steepest_descent(
    *,
    label: str,
    ctx: ReconstructionContext,
    data: np.ndarray,
    true_coeff: np.ndarray,
    args: argparse.Namespace,
    iterations: int,
    use_finufft: bool,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    x = np.zeros_like(true_coeff)
    pred = np.zeros_like(data)
    data_norm = max(float(np.linalg.norm(data.ravel())), 1e-300)
    true_norm = max(float(np.linalg.norm(true_coeff.ravel())), 1e-300)
    history: list[dict[str, Any]] = [
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
        if use_finufft:
            adj_start = time.perf_counter()
            grad = finufft_adjoint_coeff(
                ctx,
                residual,
                eps=args.finufft_eps,
                q_batch_size=args.finufft_q_batch_size,
            )
            adjoint_s = time.perf_counter() - adj_start
            fw_start = time.perf_counter()
            a_grad = finufft_forward_coeff(
                ctx,
                grad,
                eps=args.finufft_eps,
                q_batch_size=args.finufft_q_batch_size,
            )
            line_forward_s = time.perf_counter() - fw_start
        else:
            adj_start = time.perf_counter()
            grad = structured_adjoint(ctx, residual)
            adjoint_s = time.perf_counter() - adj_start
            fw_start = time.perf_counter()
            a_grad = structured_forward(ctx, grad, args)
            line_forward_s = time.perf_counter() - fw_start
        denom = max(data_norm2(a_grad), 1e-300)
        alpha = coeff_norm2(grad) / denom
        x = x - alpha * grad
        pred = pred - alpha * a_grad
        elapsed = time.perf_counter() - start
        cumulative += elapsed
        history.append(
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
    return history, x


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


def plot_history(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted({row["method"] for row in rows})
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)

    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        x = [int(row["iteration"]) for row in method_rows]
        axes[0].semilogy(x, [float(row["loss_rel"]) for row in method_rows], marker="o", label=method)
        axes[1].semilogy(x, [float(row["object_rel_l2"]) for row in method_rows], marker="o", label=method)
        axes[2].plot(
            x,
            [
                float(summary["structured_geometry_build_s"]) + float(row["cumulative_iter_s"])
                if method == "structured"
                else float(row["cumulative_iter_s"])
                for row in method_rows
            ],
            marker="o",
            label=method,
        )
    axes[0].set_title("Residual loss")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("||Ax-y|| / ||y||")
    axes[1].set_title("Object error")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("||x-x_true|| / ||x_true||")
    axes[2].set_title("Cumulative reconstruction time")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("s")
    for ax in axes:
        ax.legend(frameon=True, fontsize=8)
    fig.suptitle("Synthetic ODT iterative reconstruction", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def final_row(rows: list[dict[str, Any]], method: str) -> dict[str, Any] | None:
    method_rows = [row for row in rows if row["method"] == method]
    if not method_rows:
        return None
    return max(method_rows, key=lambda row: int(row["iteration"]))


def write_summary_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    structured_final = final_row(rows, "structured")
    finufft_final = final_row(rows, "finufft")
    lines = [
        "# ODT synthetic iterative reconstruction benchmark",
        "",
        "This benchmark treats the lab-frame cone-axis ODT geometry as a reusable prepared operator. A synthetic object generates measurements, then a steepest-descent reconstruction loop repeatedly applies the adjoint and a line-search forward projection.",
        "",
        "## Configuration",
        "",
        f"- object bins: `{summary['object_bins']}`",
        f"- cone samples: `{summary['cone_samples']}`",
        f"- cap: `{summary['cap_radial']} x {summary['cap_phi']}`",
        f"- n_illum: `{summary['n_illum']}`",
        f"- n_beta: `{summary['n_beta']}`",
        f"- forward execute mode: `{summary['forward_execute_mode']}`",
        f"- forward kernel mode: `{summary['forward_kernel_mode']}`",
        f"- FINUFFT q batch size: `{summary['finufft_q_batch_size']}`",
        f"- prepared plan effective mode: `{summary['native_prepared_plan_effective_mode']}`",
        f"- structured geometry build: `{summary['structured_geometry_build_s']:.6f} s`",
        "",
        "## Reconstruction readout",
        "",
        "| method | iterations | final loss rel | final object rel-L2 | median iter s | total incl. setup s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    if structured_final is not None:
        lines.append(
            "| structured | {it} | {loss:.6g} | {obj:.6g} | {med:.6f} | {tot:.6f} |".format(
                it=int(structured_final["iteration"]),
                loss=float(structured_final["loss_rel"]),
                obj=float(structured_final["object_rel_l2"]),
                med=float(summary["structured_median_iter_s"]),
                tot=float(summary["structured_total_including_geometry_s"]),
            )
        )
    if finufft_final is not None:
        lines.append(
            "| FINUFFT | {it} | {loss:.6g} | {obj:.6g} | {med:.6f} | {tot:.6f} |".format(
                it=int(finufft_final["iteration"]),
                loss=float(finufft_final["loss_rel"]),
                obj=float(finufft_final["object_rel_l2"]),
                med=float(summary["finufft_median_iter_s"]),
                tot=float(summary["finufft_total_s"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a self-consistency reconstruction demo, not an experimental ODT reconstruction.",
            "- The structured path pays a one-time lab-geometry preparation cost, then reuses the same prepared adjoint in every iteration.",
            "- FINUFFT is used as a generic type-3 Fourier baseline and pays its internal setup on every forward/adjoint call.",
            "- The relevant high-throughput claim is repeated reconstruction or repeated residual/backpropagation for a fixed calibrated lab geometry.",
        ]
    )
    if summary.get("break_even_iteration") is not None:
        lines.append(f"- Break-even including structured geometry setup occurs at iteration `{summary['break_even_iteration']}` in this run.")
    if summary.get("figure"):
        lines.extend(["", f"- figure: `{summary['figure']}`"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    ctx = build_context(args)
    true_coeff = np.ascontiguousarray(ctx.obj.coeff * float(args.object_scale))

    data, data_row = timed("synthetic_data_forward", lambda: structured_forward(ctx, true_coeff, args))
    if args.noise_rel > 0.0:
        rng = np.random.default_rng(args.seed + 1709)
        sigma = float(args.noise_rel) * max(float(np.linalg.norm(data.ravel())), 1e-300) / np.sqrt(data.size)
        data = data + sigma * (
            rng.standard_normal(data.shape) + 1j * rng.standard_normal(data.shape)
        ) / np.sqrt(2.0)

    rows: list[dict[str, Any]] = []
    structured_rows, structured_x = run_steepest_descent(
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
            finufft_rows, _ = run_steepest_descent(
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
    finufft_iter_times = [float(row["iter_s"]) for row in finufft_rows if int(row["iteration"]) > 0]
    geometry_build_s = float(sum(row["s"] for row in ctx.build_steps))
    structured_total_including_geometry_s = geometry_build_s + float(
        structured_rows[-1]["cumulative_iter_s"]
    )
    finufft_total_s = (
        None if not finufft_rows else float(finufft_rows[-1]["cumulative_iter_s"])
    )
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
            if finufft_total_s is not None:
                total_speedup_finufft_over_structured = (
                    finufft_total_s / structured_total_at_finufft_final_iteration_s
                )
    break_even_iteration = None
    if finufft_rows:
        fin_by_it = {int(row["iteration"]): float(row["cumulative_iter_s"]) for row in finufft_rows}
        for row in structured_rows:
            iteration = int(row["iteration"])
            if iteration == 0 or iteration not in fin_by_it:
                continue
            structured_total = geometry_build_s + float(row["cumulative_iter_s"])
            if structured_total <= fin_by_it[iteration]:
                break_even_iteration = iteration
                break

    summary = {
        "n_beta": int(args.n_beta),
        "n_r": int(args.n_r),
        "n_z": int(args.n_z),
        "object_bins": int(ctx.obj.coeff.size),
        "cap_radial": int(args.cap_radial),
        "cap_phi": int(args.cap_phi),
        "n_illum": int(args.n_illum),
        "cone_samples": int(ctx.flat_q.count),
        "k": float(args.k),
        "detector_na": float(args.detector_na),
        "illumination_na": float(args.illumination_na),
        "axis_h_cutoff": int(ctx.axis_h_cutoff),
        "l_cutoff": int(ctx.l_cutoff),
        "forward_execute_mode": args.forward_execute_mode,
        "forward_kernel_mode": args.forward_kernel_mode,
        "finufft_q_batch_size": int(args.finufft_q_batch_size),
        "native_prepared_plan_effective_mode": native_plan_effective_mode(ctx.prepared),
        "structured_geometry_build_s": geometry_build_s,
        "synthetic_data_forward_s": float(data_row["s"]),
        "structured_median_iter_s": float(median(structured_iter_times)),
        "structured_total_including_geometry_s": structured_total_including_geometry_s,
        "structured_final_loss_rel": float(structured_rows[-1]["loss_rel"]),
        "structured_final_object_rel_l2": float(structured_rows[-1]["object_rel_l2"]),
        "finufft_median_iter_s": None if not finufft_iter_times else float(median(finufft_iter_times)),
        "finufft_total_s": finufft_total_s,
        "structured_total_at_finufft_final_iteration_s": structured_total_at_finufft_final_iteration_s,
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
        finufft_data = finufft_forward_coeff(
            ctx,
            true_coeff,
            eps=args.finufft_eps,
            q_batch_size=args.finufft_q_batch_size,
        )
        summary["finufft_data_l2_vs_structured"] = relative_l2(finufft_data, data)

    write_csv(args.csv, rows)
    write_json(args.out, {"config": vars(args), "summary": summary, "history": rows})
    if args.figure:
        plot_history(args.figure, rows, summary)
    if args.summary_md:
        write_summary_markdown(args.summary_md, summary, rows)
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Synthetic iterative reconstruction demo for prepared cone-axis ODT operators."
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
    p.add_argument("--illumination-na", type=float, default=0.877887788778878)
    p.add_argument("--n-illum", type=int, default=32)
    p.add_argument("--cap-radial", type=int, default=64)
    p.add_argument("--cap-phi", type=int, default=256)
    p.add_argument("--h-cutoff", type=int, default=None)
    p.add_argument("--h-margin", type=int, default=20)
    p.add_argument("--l-margin", type=int, default=18)
    p.add_argument("--cone-l-prune-threshold", type=float, default=1e-12)
    p.add_argument("--cpp-threads", type=int, default=16)
    p.add_argument(
        "--forward-execute-mode",
        choices=["prepared", "wrapper"],
        default="prepared",
        help="Use cached Python-side forward geometry arrays or the original decomposed_forward wrapper.",
    )
    p.add_argument(
        "--forward-kernel-mode",
        choices=["compact", "partitioned"],
        default="compact",
        help="Prepared forward C++ kernel variant. Partitioned currently requires adaptive l pruning.",
    )
    p.add_argument(
        "--native-prepared-plan-mode",
        choices=["auto", "direct", "gathered", "gathered-zmajor"],
        default="auto",
    )
    p.add_argument("--native-prepared-gather-threshold", type=int, default=8192)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--include-finufft", action="store_true")
    p.add_argument("--finufft-iterations", type=int, default=None)
    p.add_argument("--finufft-eps", type=float, default=1e-12)
    p.add_argument(
        "--finufft-q-batch-size",
        type=int,
        default=0,
        help="Split FINUFFT target/source q samples into chunks. 0 uses one full FINUFFT call.",
    )
    p.add_argument("--include-operator-agreement", action="store_true")
    p.add_argument("--noise-rel", type=float, default=0.0)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_iterative_reconstruction.json",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_iterative_reconstruction_history.csv",
    )
    p.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_iterative_reconstruction_figure.png",
    )
    p.add_argument(
        "--summary-md",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_iterative_reconstruction_summary.md",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
