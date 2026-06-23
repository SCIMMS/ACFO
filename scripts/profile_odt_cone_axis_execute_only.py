from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from benchmark_odt_cone_axis_decomposition import (
    ROOT,
    build_cone_axis_decomposition,
    default_l_cutoff,
    decomposed_adjoint,
    fmt,
)
from benchmark_odt_cone_illumination import (
    cone_illumination_directions,
    cone_q_samples,
    parse_int_list,
)
from benchmark_odt_ewald_cap_operator import (
    StructuredOdtPlan,
    _cpp_odt_module,
    finufft_adjoint,
    make_cylindrical_object,
    random_residual,
    recommended_h_cutoff,
    relative_l2,
)
from profile_odt_cone_axis_bottleneck import (
    axis_factor_pack,
    fft_friendly_length,
    is_fft_friendly_length,
    speedup,
    time_call,
)


@dataclass(frozen=True)
class PreparedAdjointExecute:
    decomp: Any
    radial: np.ndarray
    axial: np.ndarray
    mode_phase: np.ndarray
    slots: np.ndarray
    native_prepared_tables: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
    native_prepared_plan: Any | None
    native_prepared_plan_mode: str
    native_prepared_gather_threshold: int
    cap_radial: int
    cap_phi: int
    n_illum: int
    cpp_threads: int
    cpp_odt: Any


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


def detector_ifft(ctx: PreparedAdjointExecute, residual: np.ndarray) -> np.ndarray:
    residual_grid = residual.reshape(ctx.n_illum, ctx.cap_radial, ctx.cap_phi)
    return np.fft.ifft(residual_grid, axis=2) * float(ctx.cap_phi)


def detector_ifft_batch(ctx: PreparedAdjointExecute, residual_batch: np.ndarray) -> np.ndarray:
    residual_grid = residual_batch.reshape(
        residual_batch.shape[0],
        ctx.n_illum,
        ctx.cap_radial,
        ctx.cap_phi,
    )
    return np.fft.ifft(residual_grid, axis=3) * float(ctx.cap_phi)


def cpp_adjoint_execute(ctx: PreparedAdjointExecute, residual_modes: np.ndarray) -> np.ndarray:
    decomp = ctx.decomp
    residual_modes = np.ascontiguousarray(residual_modes)
    if ctx.native_prepared_plan is not None:
        effective_mode = native_plan_effective_mode(ctx)
        if effective_mode == "gathered-zmajor":
            return ctx.native_prepared_plan.execute_gathered_zmajor(
                residual_modes,
                int(ctx.cpp_threads),
            )
        if effective_mode == "gathered":
            return ctx.native_prepared_plan.execute_gathered(
                residual_modes,
                int(ctx.cpp_threads),
            )
        return ctx.native_prepared_plan.execute(
            residual_modes,
            int(ctx.cpp_threads),
        )
    if ctx.native_prepared_tables is not None:
        return ctx.cpp_odt.cone_axis_adjoint_unfold_scatter_pruned_prepared(
            residual_modes,
            ctx.slots,
            decomp.active_l_offsets,
            *ctx.native_prepared_tables,
            int(decomp.plan.n_beta),
            int(ctx.cpp_threads),
        )
    if decomp.active_l_offsets is not None and decomp.active_l_indices is not None:
        return ctx.cpp_odt.cone_axis_adjoint_unfold_scatter_pruned(
            residual_modes,
            ctx.radial,
            ctx.axial,
            ctx.mode_phase,
            ctx.slots,
            decomp.transverse_coeff,
            decomp.psi_phase,
            decomp.axial_phase,
            decomp.source_slots,
            decomp.active_l_offsets,
            decomp.active_l_indices,
            int(decomp.plan.n_beta),
            int(ctx.cpp_threads),
        )
    return ctx.cpp_odt.cone_axis_adjoint_unfold_scatter(
        residual_modes,
        ctx.radial,
        ctx.axial,
        ctx.mode_phase,
        ctx.slots,
        decomp.transverse_coeff,
        decomp.psi_phase,
        decomp.axial_phase,
        decomp.source_slots,
        int(decomp.plan.n_beta),
        int(ctx.cpp_threads),
    )


def native_plan_effective_mode(ctx: PreparedAdjointExecute) -> str | None:
    if ctx.native_prepared_plan is None:
        return None
    if ctx.native_prepared_plan_mode == "auto":
        if ctx.cap_radial * ctx.cap_phi >= ctx.native_prepared_gather_threshold:
            return "gathered-zmajor"
        return "direct"
    return ctx.native_prepared_plan_mode


def cpp_adjoint_execute_batch(
    ctx: PreparedAdjointExecute,
    residual_modes_batch: np.ndarray,
) -> np.ndarray:
    decomp = ctx.decomp
    residual_modes_batch = np.ascontiguousarray(residual_modes_batch)
    if (
        ctx.native_prepared_tables is not None
        and hasattr(ctx.cpp_odt, "cone_axis_adjoint_unfold_scatter_pruned_prepared_batch")
    ):
        return ctx.cpp_odt.cone_axis_adjoint_unfold_scatter_pruned_prepared_batch(
            residual_modes_batch,
            ctx.slots,
            decomp.active_l_offsets,
            *ctx.native_prepared_tables,
            int(decomp.plan.n_beta),
            int(ctx.cpp_threads),
        )
    return np.stack(
        [cpp_adjoint_execute(ctx, residual_modes_batch[index]) for index in range(residual_modes_batch.shape[0])],
        axis=0,
    )


def prepared_adjoint_execute(ctx: PreparedAdjointExecute, residual: np.ndarray) -> np.ndarray:
    residual_modes = detector_ifft(ctx, residual)
    out_h = cpp_adjoint_execute(ctx, residual_modes)
    return np.fft.fft(out_h, axis=2)


def prepared_adjoint_execute_batch(
    ctx: PreparedAdjointExecute,
    residual_batch: np.ndarray,
) -> np.ndarray:
    residual_modes = detector_ifft_batch(ctx, residual_batch)
    out_h = cpp_adjoint_execute_batch(ctx, residual_modes)
    return np.fft.fft(out_h, axis=3)


def prepared_adjoint_execute_loop(
    ctx: PreparedAdjointExecute,
    residual_batch: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [prepared_adjoint_execute(ctx, residual_batch[index]) for index in range(residual_batch.shape[0])],
        axis=0,
    )


def random_residual_batch(flat_q: Any, *, seed: int, batch: int) -> np.ndarray:
    return np.ascontiguousarray(
        np.stack(
            [
                random_residual(flat_q, seed=seed + 104729 * index)
                for index in range(int(batch))
            ],
            axis=0,
        )
    )


def add_case_metadata(row: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    for key in (
        "requested_n_beta",
        "n_beta",
        "requested_cap_phi",
        "cap_phi",
        "n_beta_fft_friendly",
        "cap_phi_fft_friendly",
        "fft_friendly_grid",
        "fft_grid_adjusted",
        "n_illum",
        "cpp_threads",
        "cap_radial",
        "cone_samples",
        "l_cutoff",
        "l_modes",
        "axis_used_modes",
        "cone_l_active_fraction",
        "native_prepared_adjoint",
        "native_prepared_plan",
        "native_prepared_plan_mode",
        "native_prepared_plan_effective_mode",
        "native_prepared_gather_threshold",
        "batch_size",
    ):
        if key in case:
            row[key] = case[key]
    return row


def timed_step(
    rows: list[dict[str, Any]],
    label: str,
    group: str,
    func: Callable[[], Any],
    *,
    repeats: int,
    warmup: int,
    case_meta: dict[str, Any] | None = None,
) -> Any:
    value, row = time_call(label, group, func, repeats=repeats, warmup=warmup)
    if case_meta is not None:
        add_case_metadata(row, case_meta)
    rows.append(row)
    return value


def profile_case(
    args: argparse.Namespace,
    *,
    n_beta: int,
    requested_n_beta: int,
) -> dict[str, Any]:
    build_rows: list[dict[str, Any]] = []
    exec_rows: list[dict[str, Any]] = []

    obj = timed_step(
        build_rows,
        "object_build",
        "build",
        lambda: make_cylindrical_object(
            n_r=args.n_r,
            n_z=args.n_z,
            n_beta=n_beta,
            r_max=args.r_max,
            z_max=args.z_max,
            phantom=args.phantom,
            seed=args.seed,
        ),
        repeats=args.build_repeats,
        warmup=args.build_warmup,
    )

    illumination, flat_q, base_q = timed_step(
        build_rows,
        "illumination_and_q_build",
        "build",
        lambda: (
            lambda illum: (
                illum,
                *cone_q_samples(
                    k=args.k,
                    detector_na=args.detector_na,
                    cap_radial=args.cap_radial,
                    cap_phi=args.cap_phi,
                    illumination=illum,
                ),
            )
        )(
            cone_illumination_directions(
                n_illum=args.n_illum,
                illumination_na=args.illumination_na,
            )[0]
        ),
        repeats=args.build_repeats,
        warmup=args.build_warmup,
    )

    axis_h_cutoff = (
        recommended_h_cutoff(base_q, args.r_max, n_beta, args.h_margin)
        if args.h_cutoff is None
        else int(args.h_cutoff)
    )
    plan = timed_step(
        build_rows,
        "structured_plan_build",
        "build",
        lambda: StructuredOdtPlan.build(
            r_axis=obj.r_axis,
            z_axis=obj.z_axis,
            beta_axis=obj.beta_axis,
            h_cutoff=axis_h_cutoff,
        ),
        repeats=args.build_repeats,
        warmup=args.build_warmup,
    )

    l_cutoff = default_l_cutoff(
        k=args.k,
        illumination_na=args.illumination_na,
        r_max=args.r_max,
        margin=args.l_margin,
        n_beta=n_beta,
    )
    decomp = timed_step(
        build_rows,
        "cone_axis_decomposition_build",
        "build",
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
        repeats=args.build_repeats,
        warmup=args.build_warmup,
    )

    factors = timed_step(
        build_rows,
        "execute_factor_pack",
        "prepare",
        lambda: axis_factor_pack(decomp),
        repeats=args.build_repeats,
        warmup=args.build_warmup,
    )
    radial, axial, mode_phase, slots = factors

    cpp_odt = _cpp_odt_module(required=True)
    native_prepared_tables = None
    native_prepared_plan = None
    native_prepare_available = (
        not args.disable_native_prepared
        and decomp.active_l_offsets is not None
        and decomp.active_l_indices is not None
        and hasattr(cpp_odt, "cone_axis_prepare_adjoint_pruned")
        and hasattr(cpp_odt, "cone_axis_adjoint_unfold_scatter_pruned_prepared")
    )
    if native_prepare_available:
        native_prepared_tables = timed_step(
            build_rows,
            "native_adjoint_prepare_pruned",
            "prepare",
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
            repeats=args.build_repeats,
            warmup=args.build_warmup,
        )
    native_plan_available = (
        native_prepared_tables is not None
        and not args.disable_native_prepared_plan
        and hasattr(cpp_odt, "ConeAxisPreparedAdjointPlan")
    )
    if native_plan_available:
        native_prepared_plan = timed_step(
            build_rows,
            "native_prepared_adjoint_plan_build",
            "prepare",
            lambda: cpp_odt.ConeAxisPreparedAdjointPlan(
                slots,
                decomp.active_l_offsets,
                *native_prepared_tables,
                int(decomp.plan.n_beta),
            ),
            repeats=args.build_repeats,
            warmup=args.build_warmup,
        )

    residual = timed_step(
        build_rows,
        "residual_build",
        "build",
        lambda: random_residual(
            flat_q,
            seed=args.seed + 7919 + args.n_illum + args.l_margin,
        ),
        repeats=args.build_repeats,
        warmup=args.build_warmup,
    )

    cap_phi = decomp.factorization.cap_phi
    cap_radial = decomp.factorization.cap_radial
    n_illum = int(decomp.illumination_phi.size)
    expected = n_illum * cap_radial * cap_phi
    if residual.shape != (expected,):
        raise ValueError("residual shape does not match cone stack")

    if decomp.active_l_offsets is not None and decomp.active_l_indices is not None:
        if not hasattr(cpp_odt, "cone_axis_adjoint_unfold_scatter_pruned"):
            raise RuntimeError("waxs_cake._cpp_odt lacks cone_axis_adjoint_unfold_scatter_pruned")
    elif not hasattr(cpp_odt, "cone_axis_adjoint_unfold_scatter"):
        raise RuntimeError("waxs_cake._cpp_odt lacks cone_axis_adjoint_unfold_scatter")

    ctx = PreparedAdjointExecute(
        decomp=decomp,
        radial=radial,
        axial=axial,
        mode_phase=mode_phase,
        slots=slots,
        native_prepared_tables=native_prepared_tables,
        native_prepared_plan=native_prepared_plan,
        native_prepared_plan_mode=args.native_prepared_plan_mode,
        native_prepared_gather_threshold=args.native_prepared_gather_threshold,
        cap_radial=cap_radial,
        cap_phi=cap_phi,
        n_illum=n_illum,
        cpp_threads=args.cpp_threads,
        cpp_odt=cpp_odt,
    )
    effective_plan_mode = native_plan_effective_mode(ctx)

    case_meta = {
        "requested_n_beta": requested_n_beta,
        "n_beta": n_beta,
        "requested_cap_phi": args.requested_cap_phi,
        "cap_phi": args.cap_phi,
        "n_beta_fft_friendly": is_fft_friendly_length(n_beta),
        "cap_phi_fft_friendly": is_fft_friendly_length(args.cap_phi),
        "fft_friendly_grid": args.fft_friendly_grid,
        "fft_grid_adjusted": requested_n_beta != n_beta
        or int(args.requested_cap_phi) != int(args.cap_phi),
        "n_illum": args.n_illum,
        "cpp_threads": args.cpp_threads,
        "cap_radial": cap_radial,
        "cone_samples": expected,
        "l_cutoff": l_cutoff,
        "l_modes": int(2 * l_cutoff + 1),
        "axis_used_modes": plan.used_modes,
        "cone_l_active_fraction": (
            None
            if decomp.active_l_indices is None
            else float(decomp.active_l_indices.size)
            / float(decomp.transverse_coeff.shape[0] * decomp.transverse_coeff.shape[1])
        ),
        "native_prepared_adjoint": native_prepared_tables is not None,
        "native_prepared_plan": native_prepared_plan is not None,
        "native_prepared_plan_mode": (
            args.native_prepared_plan_mode if native_prepared_plan is not None else None
        ),
        "native_prepared_plan_effective_mode": effective_plan_mode,
        "native_prepared_gather_threshold": (
            args.native_prepared_gather_threshold if native_prepared_plan is not None else None
        ),
    }
    for row in build_rows:
        add_case_metadata(row, case_meta)

    residual_modes = timed_step(
        exec_rows,
        "detector_phi_ifft_residual",
        "execute_component",
        lambda: detector_ifft(ctx, residual),
        repeats=args.repeats,
        warmup=args.warmup,
        case_meta=case_meta,
    )
    out_h = timed_step(
        exec_rows,
        (
            f"cpp_cone_adjoint_unfold_scatter_pruned_prepared_plan_{args.native_prepared_plan_mode}"
            if native_prepared_plan is not None
            else "cpp_cone_adjoint_unfold_scatter_pruned_prepared"
            if native_prepared_tables is not None
            else (
                "cpp_cone_adjoint_unfold_scatter_pruned"
                if decomp.active_l_offsets is not None and decomp.active_l_indices is not None
                else "cpp_cone_adjoint_unfold_scatter"
            )
        ),
        "execute_component",
        lambda: cpp_adjoint_execute(ctx, residual_modes),
        repeats=args.repeats,
        warmup=args.warmup,
        case_meta=case_meta,
    )
    component_adjoint = timed_step(
        exec_rows,
        "beta_fft_h_to_coeff",
        "execute_component",
        lambda: np.fft.fft(out_h, axis=2),
        repeats=args.repeats,
        warmup=args.warmup,
        case_meta=case_meta,
    )
    prepared_full = timed_step(
        exec_rows,
        "prepared_adjoint_execute_full",
        "execute_full",
        lambda: prepared_adjoint_execute(ctx, residual),
        repeats=args.repeats,
        warmup=args.warmup,
        case_meta=case_meta,
    )
    prepared_full_s = exec_rows[-1]["median_s"]
    wrapper_full = timed_step(
        exec_rows,
        "decomposed_adjoint_full_wrapper",
        "execute_full",
        lambda: decomposed_adjoint(
            residual,
            decomp,
            backend="cpp",
            cpp_threads=args.cpp_threads,
            adjoint_mode="fused",
        ),
        repeats=args.repeats,
        warmup=args.warmup,
        case_meta=case_meta,
    )
    wrapper_full_s = exec_rows[-1]["median_s"]

    finufft_adjoint_s = None
    finufft_l2 = None
    finufft_skip_reason = None
    if args.include_finufft_adjoint:
        try:
            finufft_value = timed_step(
                exec_rows,
                "finufft_adjoint",
                "finufft",
                lambda: finufft_adjoint(
                    obj,
                    flat_q,
                    residual,
                    eps=args.finufft_eps,
                ),
                repeats=args.finufft_repeats,
                warmup=args.finufft_warmup,
                case_meta=case_meta,
            )
            finufft_row = exec_rows[-1]
            finufft_row["finufft_eps"] = args.finufft_eps
            finufft_adjoint_s = finufft_row["median_s"]
            finufft_l2 = relative_l2(finufft_value, prepared_full)
            finufft_row["finufft_adjoint_l2_vs_prepared"] = finufft_l2
        except Exception as exc:  # pragma: no cover - optional dependency/runtime path.
            finufft_skip_reason = str(exc)

    batch_results: list[dict[str, Any]] = []
    if args.batch_sizes:
        max_batch = max(args.batch_sizes)
        residual_batch_max = timed_step(
            build_rows,
            f"residual_batch_build_B{max_batch}",
            "build",
            lambda: random_residual_batch(
                flat_q,
                seed=args.seed + 7919 + args.n_illum + args.l_margin,
                batch=max_batch,
            ),
            repeats=args.build_repeats,
            warmup=args.build_warmup,
        )
        add_case_metadata(build_rows[-1], case_meta)
        for batch in args.batch_sizes:
            residual_batch = np.ascontiguousarray(residual_batch_max[:batch])
            batch_meta = {**case_meta, "batch_size": int(batch)}
            residual_modes_batch = timed_step(
                exec_rows,
                f"detector_phi_ifft_residual_batch_B{batch}",
                "batch_execute_component",
                lambda residual_batch=residual_batch: detector_ifft_batch(ctx, residual_batch),
                repeats=args.batch_repeats,
                warmup=args.batch_warmup,
                case_meta=batch_meta,
            )
            out_h_batch = timed_step(
                exec_rows,
                f"cpp_cone_adjoint_batch_B{batch}",
                "batch_execute_component",
                lambda residual_modes_batch=residual_modes_batch: cpp_adjoint_execute_batch(
                    ctx,
                    residual_modes_batch,
                ),
                repeats=args.batch_repeats,
                warmup=args.batch_warmup,
                case_meta=batch_meta,
            )
            component_batch = timed_step(
                exec_rows,
                f"beta_fft_batch_B{batch}",
                "batch_execute_component",
                lambda out_h_batch=out_h_batch: np.fft.fft(out_h_batch, axis=3),
                repeats=args.batch_repeats,
                warmup=args.batch_warmup,
                case_meta=batch_meta,
            )
            batch_full = timed_step(
                exec_rows,
                f"prepared_adjoint_batch_full_B{batch}",
                "batch_execute_full",
                lambda residual_batch=residual_batch: prepared_adjoint_execute_batch(
                    ctx,
                    residual_batch,
                ),
                repeats=args.batch_repeats,
                warmup=args.batch_warmup,
                case_meta=batch_meta,
            )
            batch_full_s = exec_rows[-1]["median_s"]
            batch_component_s = sum(
                row["median_s"]
                for row in exec_rows
                if row.get("batch_size") == batch
                and row["group"] == "batch_execute_component"
            )
            loop_s = None
            loop_l2 = None
            if args.include_batch_loop_baseline:
                loop_full = timed_step(
                    exec_rows,
                    f"prepared_adjoint_loop_full_B{batch}",
                    "batch_loop_baseline",
                    lambda residual_batch=residual_batch: prepared_adjoint_execute_loop(
                        ctx,
                        residual_batch,
                    ),
                    repeats=args.batch_repeats,
                    warmup=args.batch_warmup,
                    case_meta=batch_meta,
                )
                loop_s = exec_rows[-1]["median_s"]
                loop_l2 = relative_l2(loop_full, batch_full)

            batch_l2_first = relative_l2(batch_full[0], prepared_full)
            batch_entry = {
                "batch_size": int(batch),
                "batch_full_s": batch_full_s,
                "batch_per_residual_s": batch_full_s / float(batch),
                "batch_component_s": batch_component_s,
                "batch_component_per_residual_s": batch_component_s / float(batch),
                "batch_component_l2": relative_l2(component_batch, batch_full),
                "batch_first_l2_vs_single": batch_l2_first,
                "single_prepared_s": prepared_full_s,
                "throughput_speedup_vs_single": speedup(
                    prepared_full_s,
                    batch_full_s / float(batch),
                ),
                "loop_full_s": loop_s,
                "loop_per_residual_s": None if loop_s is None else loop_s / float(batch),
                "batch_speedup_vs_loop": speedup(loop_s, batch_full_s),
                "loop_l2_vs_batch": loop_l2,
            }
            for row in exec_rows:
                if row.get("batch_size") == batch:
                    row["batch_full_s"] = batch_full_s
                    row["batch_per_residual_s"] = batch_entry["batch_per_residual_s"]
                    row["batch_component_s"] = batch_component_s
                    row["batch_component_per_residual_s"] = batch_entry[
                        "batch_component_per_residual_s"
                    ]
                    row["throughput_speedup_vs_single"] = batch_entry[
                        "throughput_speedup_vs_single"
                    ]
            batch_results.append(batch_entry)

    build_total_s = sum(row["median_s"] for row in build_rows)
    component_execute_s = sum(
        row["median_s"] for row in exec_rows if row["group"] == "execute_component"
    )
    for row in build_rows + exec_rows:
        row["build_total_s"] = build_total_s
        row["component_execute_s"] = component_execute_s
        row["prepared_execute_full_s"] = prepared_full_s
        row["wrapper_execute_full_s"] = wrapper_full_s
        row["pct_of_prepared_execute_full"] = (
            None
            if prepared_full_s <= 0.0
            else 100.0 * float(row["median_s"]) / prepared_full_s
        )

    return {
        **case_meta,
        "object_bins": int(obj.coeff.size),
        "build_total_s": build_total_s,
        "component_execute_s": component_execute_s,
        "prepared_execute_full_s": prepared_full_s,
        "wrapper_execute_full_s": wrapper_full_s,
        "wrapper_over_prepared_speedup": speedup(wrapper_full_s, prepared_full_s),
        "component_vs_prepared_ratio": speedup(component_execute_s, prepared_full_s),
        "build_over_prepared_ratio": speedup(build_total_s, prepared_full_s),
        "component_l2_vs_prepared": relative_l2(component_adjoint, prepared_full),
        "wrapper_l2_vs_prepared": relative_l2(wrapper_full, prepared_full),
        "finufft_adjoint_s": finufft_adjoint_s,
        "finufft_adjoint_l2_vs_prepared": finufft_l2,
        "finufft_over_prepared_speedup": speedup(finufft_adjoint_s, prepared_full_s),
        "finufft_skip_reason": finufft_skip_reason,
        "native_prepared_adjoint": native_prepared_tables is not None,
        "batch_results": batch_results,
        "steps": build_rows + exec_rows,
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# ODT cone-axis execute-only profile",
        "",
        "This profile separates geometry/object/decomposition setup from prepared adjoint execution. The prepared execution path reuses packed axis factors and calls the same C++ adjoint kernel used by the decomposed wrapper.",
        "",
        "## Case Summary",
        "",
        "| n_beta | cap radial | cap phi | cone samples | build total s | prepared execute s | wrapper execute s | wrapper/prepared | component sum s | component L2 | FINUFFT s | FINUFFT/prepared |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in payload["cases"]:
        lines.append(
            "| {nb} | {cr} | {cp} | {samples} | {build} | {prep} | {wrap} | {sw} | {comp} | {cl2} | {fin} | {fsw} |".format(
                nb=case["n_beta"],
                cr=case["cap_radial"],
                cp=case["cap_phi"],
                samples=case["cone_samples"],
                build=fmt(case["build_total_s"], 5),
                prep=fmt(case["prepared_execute_full_s"], 5),
                wrap=fmt(case["wrapper_execute_full_s"], 5),
                sw=fmt(case["wrapper_over_prepared_speedup"], 3),
                comp=fmt(case["component_execute_s"], 5),
                cl2=fmt(case["component_l2_vs_prepared"], 3),
                fin=fmt(case["finufft_adjoint_s"], 5),
                fsw=fmt(case["finufft_over_prepared_speedup"], 3),
            )
        )
        if case.get("native_prepared_adjoint"):
            lines.append("")
            lines.append(
                f"Native prepared adjoint tables were enabled for n_beta={case['n_beta']}."
            )
        if case.get("native_prepared_plan"):
            lines.append("Native prepared adjoint plan execution was enabled.")
            lines.append(
                "Native prepared plan mode: {mode} (effective: {effective}, gather threshold: {threshold}).".format(
                    mode=case.get("native_prepared_plan_mode"),
                    effective=case.get("native_prepared_plan_effective_mode"),
                    threshold=case.get("native_prepared_gather_threshold"),
                )
            )

    batch_cases = [
        (case, item)
        for case in payload["cases"]
        for item in case.get("batch_results", [])
    ]
    if batch_cases:
        lines.extend(
            [
                "",
                "## Batch Throughput",
                "",
                "| n_beta | cap radial | cap phi | batch | batch full s | per residual s | speedup vs single | loop full s | batch vs loop | first L2 vs single |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for case, item in batch_cases:
            lines.append(
                "| {nb} | {cr} | {cp} | {batch} | {full} | {per} | {speed} | {loop} | {loop_speed} | {l2} |".format(
                    nb=case["n_beta"],
                    cr=case["cap_radial"],
                    cp=case["cap_phi"],
                    batch=item["batch_size"],
                    full=fmt(item["batch_full_s"], 5),
                    per=fmt(item["batch_per_residual_s"], 5),
                    speed=fmt(item["throughput_speedup_vs_single"], 3),
                    loop=fmt(item["loop_full_s"], 5),
                    loop_speed=fmt(item["batch_speedup_vs_loop"], 3),
                    l2=fmt(item["batch_first_l2_vs_single"], 3),
                )
            )

    lines.extend(["", "## Step Timings", ""])
    for case in payload["cases"]:
        lines.extend(
            [
                f"### n_beta={case['n_beta']}, cap_radial={case['cap_radial']}, cap_phi={case['cap_phi']}",
                "",
                "| group | step | median s | prepared execute % |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for row in sorted(case["steps"], key=lambda item: item["median_s"], reverse=True):
            lines.append(
                "| {group} | {step} | {time} | {pct} |".format(
                    group=row["group"],
                    step=row["step"],
                    time=fmt(row["median_s"], 5),
                    pct=fmt(row["pct_of_prepared_execute_full"], 3),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Readout Guide",
            "",
            "- `build total s` is not a production total; it is a sum of separately timed setup stages and is intended to expose amortization pressure.",
            "- `prepared execute s` is the current ODTConeAxisPlan proxy: geometry-dependent arrays are packed once outside the timed call, but the C++ kernel still owns its internal temporary allocations.",
            "- `wrapper execute s` is the current Python wrapper path through `decomposed_adjoint(..., adjoint_mode='fused')`.",
            "- If `prepared execute s` is much lower than `wrapper execute s`, a Python-level prepared plan can help immediately. If not, the next meaningful gains require moving C++ temporaries into a native plan or batching residuals.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Separate ODT cone-axis adjoint setup cost from execute-only cost."
    )
    parser.add_argument("--output-prefix", default="benchmark_results/odt_cone_axis_execute_only")
    parser.add_argument("--n-beta-values", default="384")
    parser.add_argument("--n-illum", type=int, default=32)
    parser.add_argument("--illumination-na", type=float, default=0.877887788778878)
    parser.add_argument("--l-margin", type=int, default=18)
    parser.add_argument("--phantom", choices=["beads", "random_beads", "shell"], default="beads")
    parser.add_argument("--n-r", type=int, default=16)
    parser.add_argument("--n-z", type=int, default=15)
    parser.add_argument("--r-max", type=float, default=1.5)
    parser.add_argument("--z-max", type=float, default=1.2)
    parser.add_argument("--k", type=float, default=17.307319527958313)
    parser.add_argument("--detector-na", type=float, default=0.9240924092409241)
    parser.add_argument("--cap-radial", type=int, default=16)
    parser.add_argument("--cap-phi", type=int, default=64)
    parser.add_argument(
        "--fft-friendly-grid",
        action="store_true",
        help="Round n_beta and cap_phi upward to SciPy/pocketFFT-friendly complex FFT lengths.",
    )
    parser.add_argument("--h-cutoff", type=int, default=None)
    parser.add_argument("--h-margin", type=int, default=18)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cpp-threads", type=int, default=0)
    parser.add_argument("--cone-l-prune-threshold", type=float, default=1e-12)
    parser.add_argument("--include-finufft-adjoint", action="store_true")
    parser.add_argument(
        "--disable-native-prepared",
        action="store_true",
        help="Use the legacy C++ adjoint kernel even when native prepared tables are available.",
    )
    parser.add_argument(
        "--disable-native-prepared-plan",
        action="store_true",
        help="Use tuple-based native prepared execution even when the C++ prepared plan object is available.",
    )
    parser.add_argument(
        "--native-prepared-plan-mode",
        choices=["auto", "direct", "gathered", "gathered-zmajor"],
        default="auto",
        help="Execution mode for ConeAxisPreparedAdjointPlan.",
    )
    parser.add_argument(
        "--native-prepared-gather-threshold",
        type=int,
        default=8192,
        help="Use gathered plan execution in auto mode when cap_radial * cap_phi is at least this value.",
    )
    parser.add_argument("--finufft-eps", type=float, default=1e-12)
    parser.add_argument("--finufft-repeats", type=int, default=5)
    parser.add_argument("--finufft-warmup", type=int, default=1)
    parser.add_argument(
        "--batch-sizes",
        default="",
        help="Comma-separated batch sizes for prepared adjoint batch throughput profiling.",
    )
    parser.add_argument(
        "--include-batch-loop-baseline",
        action="store_true",
        help="Also time a Python loop over B single prepared-adjoint calls.",
    )
    parser.add_argument(
        "--batch-repeats",
        type=int,
        default=0,
        help="Repeats for batch timings. Defaults to --repeats when 0.",
    )
    parser.add_argument(
        "--batch-warmup",
        type=int,
        default=-1,
        help="Warmup calls for batch timings. Defaults to --warmup when negative.",
    )
    parser.add_argument("--build-repeats", type=int, default=7)
    parser.add_argument("--build-warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    requested_n_beta_values = parse_int_list(args.n_beta_values)
    args.requested_n_beta_values = requested_n_beta_values
    args.requested_cap_phi = int(args.cap_phi)
    effective_pairs: list[dict[str, int]] = []
    seen_effective: set[int] = set()
    for requested in requested_n_beta_values:
        effective = fft_friendly_length(requested) if args.fft_friendly_grid else requested
        if effective not in seen_effective:
            effective_pairs.append({"requested": requested, "effective": effective})
            seen_effective.add(effective)
    args.n_beta_cases = effective_pairs
    args.n_beta_values = [item["effective"] for item in effective_pairs]
    if args.fft_friendly_grid:
        args.cap_phi = fft_friendly_length(args.cap_phi)
    if args.cap_phi <= 0 or args.cap_radial <= 0:
        raise ValueError("cap-radial and cap-phi must be positive")
    if args.cpp_threads < 0:
        raise ValueError("cpp-threads must be non-negative")
    if args.cone_l_prune_threshold < 0.0:
        raise ValueError("cone-l-prune-threshold must be non-negative")
    if args.repeats <= 0 or args.warmup < 0:
        raise ValueError("repeats must be positive and warmup must be non-negative")
    if args.build_repeats <= 0 or args.build_warmup < 0:
        raise ValueError("build repeats must be positive and warmup must be non-negative")
    if args.finufft_eps <= 0.0:
        raise ValueError("finufft-eps must be positive")
    args.batch_sizes = parse_int_list(args.batch_sizes) if args.batch_sizes else []
    if any(batch <= 0 for batch in args.batch_sizes):
        raise ValueError("batch sizes must be positive")
    if args.batch_repeats <= 0:
        args.batch_repeats = args.repeats
    if args.batch_warmup < 0:
        args.batch_warmup = args.warmup
    if args.batch_repeats <= 0 or args.batch_warmup < 0:
        raise ValueError("batch repeats must be positive and warmup must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    cases = [
        profile_case(args, n_beta=item["effective"], requested_n_beta=item["requested"])
        for item in args.n_beta_cases
    ]
    payload = {
        "config": {**vars(args), "n_beta_values": args.n_beta_values},
        "cases": cases,
    }
    output_prefix = ROOT / args.output_prefix
    write_json(output_prefix.with_suffix(".json"), payload)
    rows = [row for case in cases for row in case["steps"]]
    write_csv(output_prefix.with_suffix(".csv"), rows)
    write_summary(output_prefix.with_name(output_prefix.name + "_summary.md"), payload)
    print(
        json.dumps(
            {
                "json": str(output_prefix.with_suffix(".json")),
                "csv": str(output_prefix.with_suffix(".csv")),
                "summary_md": str(output_prefix.with_name(output_prefix.name + "_summary.md")),
                "cases": [
                    {
                        "n_beta": case["n_beta"],
                        "cap_radial": case["cap_radial"],
                        "cap_phi": case["cap_phi"],
                        "cone_samples": case["cone_samples"],
                        "build_total_s": case["build_total_s"],
                        "prepared_execute_full_s": case["prepared_execute_full_s"],
                        "wrapper_execute_full_s": case["wrapper_execute_full_s"],
                        "wrapper_over_prepared_speedup": case[
                            "wrapper_over_prepared_speedup"
                        ],
                        "finufft_over_prepared_speedup": case[
                            "finufft_over_prepared_speedup"
                        ],
                        "batch_results": case["batch_results"],
                    }
                    for case in cases
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
