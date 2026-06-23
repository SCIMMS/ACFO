from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any, Callable

import numpy as np
from scipy import special

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark_odt_cone_axis_decomposition import (  # noqa: E402
    ConeAxisDecomposition,
    build_radial_l_pruning,
    default_l_cutoff,
)
from benchmark_odt_cone_illumination import (  # noqa: E402
    cone_illumination_directions,
    q_samples_from_vectors,
)
from benchmark_odt_ewald_cap_operator import (  # noqa: E402
    QSamples,
    ShiftedAxisFactorization,
    StructuredOdtPlan,
    _cpp_odt_module,
    build_structured_kernel,
    make_cylindrical_object,
    recommended_h_cutoff,
    relative_l2,
)
from benchmark_odt_iterative_reconstruction import (  # noqa: E402
    PreparedForwardExecute,
    ReconstructionContext,
    run_steepest_descent,
    structured_forward,
)
from profile_odt_cone_axis_bottleneck import axis_factor_pack  # noqa: E402
from profile_odt_cone_axis_execute_only import (  # noqa: E402
    PreparedAdjointExecute,
    native_plan_effective_mode,
)


def timed(label: str, func: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    start = time.perf_counter()
    value = func()
    return value, {"step": label, "s": float(time.perf_counter() - start)}


def parse_float_list(value: str) -> list[float]:
    out = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise ValueError("expected a comma-separated list of floats")
    return out


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    value_f = float(value)
    if value_f == 0.0:
        return "0"
    if abs(value_f) < 1e-3 or abs(value_f) >= 1e3:
        return f"{value_f:.{digits}e}"
    return f"{value_f:.{digits}f}"


def detector_directions_model(
    *,
    detector_na: float,
    cap_radial: int,
    cap_phi: int,
    axial_model: str,
) -> np.ndarray:
    if detector_na <= 0.0 or detector_na >= 1.0:
        raise ValueError("detector_na must be in (0, 1)")
    if axial_model not in {"full", "paraxial", "flat"}:
        raise ValueError("axial_model must be full, paraxial, or flat")
    radial = (np.arange(cap_radial, dtype=float) + 0.5) * detector_na / float(cap_radial)
    phi = np.linspace(0.0, 2.0 * np.pi, cap_phi, endpoint=False, dtype=float)
    rr, pp = np.meshgrid(radial, phi, indexing="ij")
    sx = rr.ravel() * np.cos(pp.ravel())
    sy = rr.ravel() * np.sin(pp.ravel())
    rho2 = sx * sx + sy * sy
    if axial_model == "full":
        sz = np.sqrt(np.maximum(1.0 - rho2, 0.0))
    elif axial_model == "paraxial":
        sz = 1.0 - 0.5 * rho2
    else:
        sz = np.ones_like(sx)
    return np.column_stack([sx, sy, sz])


def cone_q_samples_axial_model(
    *,
    k: float,
    detector_na: float,
    cap_radial: int,
    cap_phi: int,
    illumination: np.ndarray,
    axial_model: str,
) -> tuple[QSamples, QSamples]:
    detector = detector_directions_model(
        detector_na=detector_na,
        cap_radial=cap_radial,
        cap_phi=cap_phi,
        axial_model=axial_model,
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


def build_cone_axis_decomposition_axial_model(
    plan: StructuredOdtPlan,
    *,
    k: float,
    detector_na: float,
    cap_radial: int,
    cap_phi: int,
    illumination_na: float,
    n_illum: int,
    l_cutoff: int,
    adaptive_l_threshold: float | None,
    axial_model: str,
) -> ConeAxisDecomposition:
    illumination, illumination_phi = cone_illumination_directions(
        n_illum=n_illum,
        illumination_na=illumination_na,
    )
    flat_q, base_q = cone_q_samples_axial_model(
        k=k,
        detector_na=detector_na,
        cap_radial=cap_radial,
        cap_phi=cap_phi,
        illumination=illumination,
        axial_model=axial_model,
    )
    l_values = np.arange(-int(l_cutoff), int(l_cutoff) + 1, dtype=np.int64)
    source_slots = np.mod(
        plan.h_values[:, None] + l_values[None, :],
        plan.n_beta,
    ).astype(np.int64)
    arg = float(k) * float(illumination_na) * plan.r_axis[None, :]
    transverse = ((-1j) ** l_values)[:, None] * special.jv(l_values[:, None], arg)
    active_l_offsets = None
    active_l_indices = None
    if adaptive_l_threshold is not None and adaptive_l_threshold > 0.0:
        active_l_offsets, active_l_indices = build_radial_l_pruning(
            transverse,
            threshold=float(adaptive_l_threshold),
        )
    psi_phase = np.exp(-1j * illumination_phi[:, None] * l_values[None, :])
    cos_alpha = np.sqrt(max(1.0 - float(illumination_na) ** 2, 0.0))
    axial_phase = np.exp(1j * float(k) * (1.0 - cos_alpha) * plan.z_axis)
    factorization = ShiftedAxisFactorization(
        base_q=base_q,
        illumination=illumination,
        phase=np.empty((0, 0, 0, 0), dtype=np.complex128),
        beta_twiddle=np.empty((0, 0), dtype=np.complex128),
        kernel=build_structured_kernel(plan, base_q),
        cap_radial=cap_radial,
        cap_phi=cap_phi,
    )
    return ConeAxisDecomposition(
        base_q=base_q,
        flat_q=flat_q,
        factorization=factorization,
        illumination_phi=np.ascontiguousarray(illumination_phi),
        l_values=np.ascontiguousarray(l_values),
        transverse_coeff=np.ascontiguousarray(transverse),
        psi_phase=np.ascontiguousarray(psi_phase),
        axial_phase=np.ascontiguousarray(axial_phase),
        source_slots=np.ascontiguousarray(source_slots),
        plan=plan,
        active_l_offsets=active_l_offsets,
        active_l_indices=active_l_indices,
        active_l_threshold=(
            None
            if adaptive_l_threshold is None or adaptive_l_threshold <= 0.0
            else float(adaptive_l_threshold)
        ),
    )


def build_model_context(
    *,
    args: argparse.Namespace,
    obj: Any,
    plan: StructuredOdtPlan,
    l_cutoff: int,
    axis_h_cutoff: int,
    detector_na: float,
    axial_model: str,
) -> ReconstructionContext:
    build_steps: list[dict[str, Any]] = []
    decomp, row = timed(
        f"{axial_model}_cone_axis_decomposition_build",
        lambda: build_cone_axis_decomposition_axial_model(
            plan,
            k=args.k,
            detector_na=detector_na,
            cap_radial=args.cap_radial,
            cap_phi=args.cap_phi,
            illumination_na=args.illumination_na,
            n_illum=args.n_illum,
            l_cutoff=l_cutoff,
            adaptive_l_threshold=args.cone_l_prune_threshold,
            axial_model=axial_model,
        ),
    )
    build_steps.append(row)
    factors, row = timed(f"{axial_model}_execute_factor_pack", lambda: axis_factor_pack(decomp))
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
            f"{axial_model}_native_adjoint_prepare_pruned",
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
            f"{axial_model}_native_prepared_adjoint_plan_build",
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
        flat_q=decomp.flat_q,
        base_q=decomp.base_q,
        plan=plan,
        decomp=decomp,
        prepared=prepared,
        prepared_forward=prepared_forward,
        build_steps=build_steps,
        l_cutoff=l_cutoff,
        axis_h_cutoff=axis_h_cutoff,
    )


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


def final_row(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    method_rows = [row for row in rows if row["method"] == method]
    if not method_rows:
        raise ValueError(f"missing method rows for {method}")
    return max(method_rows, key=lambda row: int(row["iteration"]))


def plot_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = ["full", "paraxial", "flat"]
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)
    panels = [
        ("oracle_data_mismatch_rel", "True-object data mismatch", True),
        ("final_loss_rel", "Final residual loss", True),
        ("final_object_rel_l2", "Final object rel-L2", True),
        ("median_iter_s", "Median iteration time", False),
    ]
    for ax, (key, title, log_y) in zip(axes.ravel(), panels):
        for model in models:
            model_rows = [row for row in rows if row["model"] == model]
            x = [float(row["detector_na"]) for row in model_rows]
            y = [float(row[key]) for row in model_rows]
            ax.plot(x, y, marker="o", label=model)
        if log_y:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("detector NA")
        ax.set_ylabel(key)
        ax.legend(frameon=True, fontsize=8)
    fig.suptitle("Curved Ewald model importance in synthetic ODT reconstruction", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Curved Ewald importance benchmark",
        "",
        "Synthetic measurements are generated with the full curved Ewald-cap model. Reconstruction then uses one of three detector axial models while keeping the same object, illumination ring, prepared forward kernel, and prepared adjoint kernel.",
        "",
        "- `full`: exact detector `k_z = k sqrt(1 - NA_r^2)`.",
        "- `paraxial`: second-order detector `k_z / k = 1 - NA_r^2 / 2`.",
        "- `flat`: tangent-plane detector `k_z / k = 1`; this removes detector-side Ewald curvature.",
        "",
        "## Configuration",
        "",
        f"- object bins: `{summary['object_bins']}`",
        f"- n_illum: `{summary['n_illum']}`",
        f"- cap: `{summary['cap_radial']} x {summary['cap_phi']}`",
        f"- iterations per model: `{summary['iterations']}`",
        f"- illumination_na: `{summary['illumination_na']}`",
        f"- k: `{summary['k']}`",
        f"- forward kernel mode: `{summary['forward_kernel_mode']}`",
        "",
        "## Results",
        "",
        "| detector NA | model | max abs delta qz vs full | oracle data mismatch | final loss | object rel-L2 | median iter s | build s |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {na} | {model} | {dqz} | {oracle} | {loss} | {obj} | {iter_s} | {build_s} |".format(
                na=fmt(row["detector_na"], 3),
                model=row["model"],
                dqz=fmt(row["max_abs_delta_qz_vs_full"], 4),
                oracle=fmt(row["oracle_data_mismatch_rel"], 4),
                loss=fmt(row["final_loss_rel"], 4),
                obj=fmt(row["final_object_rel_l2"], 4),
                iter_s=fmt(row["median_iter_s"], 4),
                build_s=fmt(row["model_build_s"], 4),
            )
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- The `oracle data mismatch` is `||A_model x_true - A_full x_true|| / ||A_full x_true||`; it isolates geometry error before reconstruction iterations.",
            "- If curved Ewald geometry matters, the flat/paraxial models should show a mismatch and a nonzero reconstruction residual that grow with detector NA.",
            "- Runtime should remain close across models because only the detector axial phase changes; the algorithmic cost is dominated by the same prepared cylindrical harmonic contractions.",
        ]
    )
    if summary.get("figure"):
        lines.append(f"- figure: `{summary['figure']}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    detector_na_values = parse_float_list(args.detector_na_list)
    obj, object_step = timed(
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
    true_coeff = np.ascontiguousarray(obj.coeff * float(args.object_scale))
    all_rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []

    for detector_na in detector_na_values:
        illumination, _ = cone_illumination_directions(
            n_illum=args.n_illum,
            illumination_na=args.illumination_na,
        )
        _, base_q_full = cone_q_samples_axial_model(
            k=args.k,
            detector_na=detector_na,
            cap_radial=args.cap_radial,
            cap_phi=args.cap_phi,
            illumination=illumination,
            axial_model="full",
        )
        axis_h_cutoff = (
            recommended_h_cutoff(base_q_full, args.r_max, args.n_beta, args.h_margin)
            if args.h_cutoff is None
            else int(args.h_cutoff)
        )
        plan, plan_step = timed(
            f"detector_na_{detector_na:.3f}_structured_axis_plan_build",
            lambda: StructuredOdtPlan.build(
                r_axis=obj.r_axis,
                z_axis=obj.z_axis,
                beta_axis=obj.beta_axis,
                h_cutoff=axis_h_cutoff,
            ),
        )
        l_cutoff = default_l_cutoff(
            k=args.k,
            illumination_na=args.illumination_na,
            r_max=args.r_max,
            margin=args.l_margin,
            n_beta=args.n_beta,
        )
        contexts: dict[str, ReconstructionContext] = {}
        for model in ["full", "paraxial", "flat"]:
            contexts[model] = build_model_context(
                args=args,
                obj=obj,
                plan=plan,
                l_cutoff=l_cutoff,
                axis_h_cutoff=axis_h_cutoff,
                detector_na=detector_na,
                axial_model=model,
            )

        full_ctx = contexts["full"]
        data_full, data_step = timed(
            f"detector_na_{detector_na:.3f}_synthetic_full_data_forward",
            lambda: structured_forward(full_ctx, true_coeff, args),
        )
        data_norm = max(float(np.linalg.norm(data_full.ravel())), 1e-300)
        case_summaries.append(
            {
                "detector_na": float(detector_na),
                "axis_h_cutoff": int(axis_h_cutoff),
                "l_cutoff": int(l_cutoff),
                "full_q_samples": int(full_ctx.flat_q.count),
                "full_qz_curvature_max": float(np.max(np.abs(full_ctx.base_q.qz))),
                "shared_object_build_s": float(object_step["s"]),
                "plan_build_s": float(plan_step["s"]),
                "synthetic_data_forward_s": float(data_step["s"]),
            }
        )

        for model, ctx in contexts.items():
            model_data, model_data_step = timed(
                f"detector_na_{detector_na:.3f}_{model}_true_object_forward",
                lambda ctx=ctx: structured_forward(ctx, true_coeff, args),
            )
            model_history, _ = run_steepest_descent(
                label=model,
                ctx=ctx,
                data=data_full,
                true_coeff=true_coeff,
                args=args,
                iterations=args.iterations,
                use_finufft=False,
            )
            histories.extend(
                {
                    **row,
                    "detector_na": float(detector_na),
                    "model": model,
                }
                for row in model_history
            )
            final = final_row(model_history, model)
            iter_times = [float(row["iter_s"]) for row in model_history if int(row["iteration"]) > 0]
            qz_delta = ctx.base_q.qz - full_ctx.base_q.qz
            all_rows.append(
                {
                    "detector_na": float(detector_na),
                    "model": model,
                    "q_samples": int(ctx.flat_q.count),
                    "axis_h_cutoff": int(axis_h_cutoff),
                    "l_cutoff": int(l_cutoff),
                    "native_prepared_plan_effective_mode": native_plan_effective_mode(ctx.prepared),
                    "max_abs_delta_qz_vs_full": float(np.max(np.abs(qz_delta))),
                    "rms_delta_qz_vs_full": float(np.sqrt(np.mean(qz_delta * qz_delta))),
                    "oracle_data_mismatch_rel": relative_l2(model_data, data_full),
                    "model_true_object_forward_s": float(model_data_step["s"]),
                    "final_loss_rel": float(final["loss_rel"]),
                    "final_object_rel_l2": float(final["object_rel_l2"]),
                    "median_iter_s": float(median(iter_times)),
                    "median_adjoint_s": float(
                        median(
                            float(row["adjoint_s"])
                            for row in model_history
                            if int(row["iteration"]) > 0
                        )
                    ),
                    "median_line_forward_s": float(
                        median(
                            float(row["line_forward_s"])
                            for row in model_history
                            if int(row["iteration"]) > 0
                        )
                    ),
                    "cumulative_iter_s": float(final["cumulative_iter_s"]),
                    "model_build_s": float(sum(step["s"] for step in ctx.build_steps)),
                    "data_norm": float(data_norm),
                }
            )

    summary = {
        "n_beta": int(args.n_beta),
        "n_r": int(args.n_r),
        "n_z": int(args.n_z),
        "object_bins": int(obj.coeff.size),
        "cap_radial": int(args.cap_radial),
        "cap_phi": int(args.cap_phi),
        "n_illum": int(args.n_illum),
        "iterations": int(args.iterations),
        "k": float(args.k),
        "illumination_na": float(args.illumination_na),
        "detector_na_values": detector_na_values,
        "forward_execute_mode": args.forward_execute_mode,
        "forward_kernel_mode": args.forward_kernel_mode,
        "object_build_s": float(object_step["s"]),
        "case_summaries": case_summaries,
        "figure": str(args.figure) if args.figure else None,
        "csv": str(args.csv),
        "history_csv": str(args.history_csv),
    }
    write_csv(args.csv, all_rows)
    write_csv(args.history_csv, histories)
    write_json(args.out, {"config": vars(args), "summary": summary, "rows": all_rows, "history": histories})
    if args.figure:
        plot_summary(args.figure, all_rows)
    if args.summary_md:
        write_summary_markdown(args.summary_md, summary, all_rows)
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Measure when detector-side curved Ewald geometry matters in synthetic ODT reconstruction."
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
    p.add_argument("--detector-na-list", type=str, default="0.3,0.6,0.85,0.95")
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
    p.add_argument("--iterations", type=int, default=16)
    p.add_argument("--finufft-eps", type=float, default=1e-12)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_curved_ewald_importance.json",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_curved_ewald_importance_summary.csv",
    )
    p.add_argument(
        "--history-csv",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_curved_ewald_importance_history.csv",
    )
    p.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_curved_ewald_importance.png",
    )
    p.add_argument(
        "--summary-md",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_curved_ewald_importance_summary.md",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
