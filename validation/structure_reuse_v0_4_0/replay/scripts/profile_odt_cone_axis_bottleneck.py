from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from statistics import median
from typing import Any, Callable

import numpy as np

from benchmark_odt_cone_axis_decomposition import (
    ROOT,
    build_cone_axis_decomposition,
    default_l_cutoff,
    decomposed_adjoint,
    decomposed_forward,
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


def fft_friendly_length(n: int) -> int:
    n = int(n)
    if n <= 0:
        raise ValueError("FFT grid lengths must be positive")
    try:
        from scipy.fft import next_fast_len
    except Exception as exc:  # pragma: no cover - dependency is expected in benchmark envs.
        raise RuntimeError("scipy is required for --fft-friendly-grid") from exc
    return int(next_fast_len(n, real=False))


def is_fft_friendly_length(n: int) -> bool:
    return int(n) == fft_friendly_length(int(n))


def time_call(
    label: str,
    group: str,
    func: Callable[[], Any],
    *,
    repeats: int,
    warmup: int,
) -> tuple[Any, dict[str, Any]]:
    result = None
    for _ in range(max(0, warmup)):
        result = func()

    times: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(max(1, repeats)):
            start = time.perf_counter()
            result = func()
            times.append(time.perf_counter() - start)
    finally:
        if gc_was_enabled:
            gc.enable()

    if result is None:
        raise RuntimeError(f"{label} did not run")
    return result, {
        "group": group,
        "step": label,
        "median_s": float(median(times)),
        "min_s": float(min(times)),
        "max_s": float(max(times)),
        "times_s": " ".join(f"{item:.9g}" for item in times),
    }


def speedup(reference_s: float | None, candidate_s: float | None) -> float | None:
    if reference_s is None or candidate_s is None or candidate_s <= 0.0:
        return None
    return float(reference_s) / float(candidate_s)


def axis_factor_pack(decomp: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    plan = decomp.plan
    cap_phi = decomp.factorization.cap_phi
    kernel = decomp.factorization.kernel
    if kernel.radial.shape[1] == decomp.factorization.cap_radial:
        radial = kernel.radial
        axial = kernel.axial
    else:
        radial = kernel.radial[:, ::cap_phi, :]
        axial = kernel.axial[::cap_phi, :]
    mode_phase = kernel.angular[:, 0]
    slots = np.mod(plan.h_values, cap_phi).astype(np.int64)
    return (
        np.ascontiguousarray(radial),
        np.ascontiguousarray(axial),
        np.ascontiguousarray(mode_phase),
        np.ascontiguousarray(slots),
    )


def profile_case(
    args: argparse.Namespace,
    *,
    n_beta: int,
    requested_n_beta: int | None = None,
) -> dict[str, Any]:
    requested_n_beta = n_beta if requested_n_beta is None else int(requested_n_beta)
    obj = make_cylindrical_object(
        n_r=args.n_r,
        n_z=args.n_z,
        n_beta=n_beta,
        r_max=args.r_max,
        z_max=args.z_max,
        phantom=args.phantom,
        seed=args.seed,
    )
    illumination, _ = cone_illumination_directions(
        n_illum=args.n_illum,
        illumination_na=args.illumination_na,
    )
    flat_q, base_q = cone_q_samples(
        k=args.k,
        detector_na=args.detector_na,
        cap_radial=args.cap_radial,
        cap_phi=args.cap_phi,
        illumination=illumination,
    )
    axis_h_cutoff = (
        recommended_h_cutoff(base_q, args.r_max, n_beta, args.h_margin)
        if args.h_cutoff is None
        else int(args.h_cutoff)
    )
    plan = StructuredOdtPlan.build(
        r_axis=obj.r_axis,
        z_axis=obj.z_axis,
        beta_axis=obj.beta_axis,
        h_cutoff=axis_h_cutoff,
    )
    l_cutoff = default_l_cutoff(
        k=args.k,
        illumination_na=args.illumination_na,
        r_max=args.r_max,
        margin=args.l_margin,
        n_beta=n_beta,
    )
    decomp = build_cone_axis_decomposition(
        plan,
        k=args.k,
        detector_na=args.detector_na,
        cap_radial=args.cap_radial,
        cap_phi=args.cap_phi,
        illumination_na=args.illumination_na,
        n_illum=args.n_illum,
        l_cutoff=l_cutoff,
        adaptive_l_threshold=args.cone_l_prune_threshold,
    )
    residual = random_residual(flat_q, seed=args.seed + 7919 + args.n_illum + args.l_margin)
    n_beta_fft_friendly = is_fft_friendly_length(n_beta)
    cap_phi_fft_friendly = is_fft_friendly_length(args.cap_phi)
    fft_grid_adjusted = (
        requested_n_beta != n_beta or int(args.requested_cap_phi) != int(args.cap_phi)
    )

    cpp_odt = _cpp_odt_module(required=True)
    if not hasattr(cpp_odt, "cone_axis_forward_fold"):
        raise RuntimeError("waxs_cake._cpp_odt lacks cone_axis_forward_fold")
    if not hasattr(cpp_odt, "cone_axis_adjoint_unfold_scatter"):
        raise RuntimeError("waxs_cake._cpp_odt lacks cone_axis_adjoint_unfold_scatter")

    steps: list[dict[str, Any]] = []

    factors, row = time_call(
        "axis_factor_pack",
        "shared",
        lambda: axis_factor_pack(decomp),
        repeats=args.repeats,
        warmup=args.warmup,
    )
    steps.append(row)
    radial, axial, mode_phase, slots = factors

    cap_phi = decomp.factorization.cap_phi
    cap_radial = decomp.factorization.cap_radial
    expected = decomp.illumination_phi.size * cap_radial * cap_phi
    if residual.shape != (expected,):
        raise ValueError("residual shape does not match cone stack")

    forward_from_steps = None
    forward_full = None
    if not args.adjoint_only:
        coeff_h_full, row = time_call(
            "beta_ifft_coeff_to_h",
            "forward",
            lambda: np.fft.ifft(obj.coeff, axis=2) * float(plan.n_beta),
            repeats=args.repeats,
            warmup=args.warmup,
        )
        steps.append(row)

        forward_kernel_name = (
            "cpp_cone_forward_fold_pruned"
            if decomp.active_l_offsets is not None and decomp.active_l_indices is not None
            else "cpp_cone_forward_fold"
        )
        if forward_kernel_name.endswith("_pruned") and not hasattr(cpp_odt, "cone_axis_forward_fold_pruned"):
            raise RuntimeError("waxs_cake._cpp_odt lacks cone_axis_forward_fold_pruned")
        folded, row = time_call(
            forward_kernel_name,
            "forward",
            lambda: (
                cpp_odt.cone_axis_forward_fold_pruned(
                    np.ascontiguousarray(coeff_h_full),
                    radial,
                    axial,
                    mode_phase,
                    slots,
                    decomp.transverse_coeff,
                    decomp.psi_phase,
                    decomp.axial_phase,
                    decomp.source_slots,
                    decomp.active_l_offsets,
                    decomp.active_l_indices,
                    int(cap_phi),
                    int(args.cpp_threads),
                )
                if decomp.active_l_offsets is not None and decomp.active_l_indices is not None
                else cpp_odt.cone_axis_forward_fold(
                    np.ascontiguousarray(coeff_h_full),
                    radial,
                    axial,
                    mode_phase,
                    slots,
                    decomp.transverse_coeff,
                    decomp.psi_phase,
                    decomp.axial_phase,
                    decomp.source_slots,
                    int(cap_phi),
                    int(args.cpp_threads),
                )
            ),
            repeats=args.repeats,
            warmup=args.warmup,
        )
        steps.append(row)

        forward_from_steps, row = time_call(
            "detector_phi_fft",
            "forward",
            lambda: np.fft.fft(folded, axis=2).reshape(expected),
            repeats=args.repeats,
            warmup=args.warmup,
        )
        steps.append(row)

        forward_full, row = time_call(
            "decomposed_forward_full",
            "full",
            lambda: decomposed_forward(
                obj.coeff,
                decomp,
                backend="cpp",
                cpp_threads=args.cpp_threads,
                forward_mode="fused",
            ),
            repeats=args.repeats,
            warmup=args.warmup,
        )
        steps.append(row)

    residual_grid = residual.reshape(args.n_illum, cap_radial, cap_phi)
    residual_modes, row = time_call(
        "detector_phi_ifft_residual",
        "adjoint",
        lambda: np.fft.ifft(residual_grid, axis=2) * float(cap_phi),
        repeats=args.repeats,
        warmup=args.warmup,
    )
    steps.append(row)

    adjoint_kernel_name = (
        "cpp_cone_adjoint_unfold_scatter_pruned"
        if decomp.active_l_offsets is not None and decomp.active_l_indices is not None
        else "cpp_cone_adjoint_unfold_scatter"
    )
    if adjoint_kernel_name.endswith("_pruned") and not hasattr(
        cpp_odt,
        "cone_axis_adjoint_unfold_scatter_pruned",
    ):
        raise RuntimeError("waxs_cake._cpp_odt lacks cone_axis_adjoint_unfold_scatter_pruned")
    out_h, row = time_call(
        adjoint_kernel_name,
        "adjoint",
        lambda: (
            cpp_odt.cone_axis_adjoint_unfold_scatter_pruned(
                np.ascontiguousarray(residual_modes),
                radial,
                axial,
                mode_phase,
                slots,
                decomp.transverse_coeff,
                decomp.psi_phase,
                decomp.axial_phase,
                decomp.source_slots,
                decomp.active_l_offsets,
                decomp.active_l_indices,
                int(plan.n_beta),
                int(args.cpp_threads),
            )
            if decomp.active_l_offsets is not None and decomp.active_l_indices is not None
            else cpp_odt.cone_axis_adjoint_unfold_scatter(
                np.ascontiguousarray(residual_modes),
                radial,
                axial,
                mode_phase,
                slots,
                decomp.transverse_coeff,
                decomp.psi_phase,
                decomp.axial_phase,
                decomp.source_slots,
                int(plan.n_beta),
                int(args.cpp_threads),
            )
        ),
        repeats=args.repeats,
        warmup=args.warmup,
    )
    steps.append(row)

    adjoint_from_steps, row = time_call(
        "beta_fft_h_to_coeff",
        "adjoint",
        lambda: np.fft.fft(out_h, axis=2),
        repeats=args.repeats,
        warmup=args.warmup,
    )
    steps.append(row)

    adjoint_full, row = time_call(
        "decomposed_adjoint_full",
        "full",
        lambda: decomposed_adjoint(
            residual,
            decomp,
            backend="cpp",
            cpp_threads=args.cpp_threads,
            adjoint_mode="fused",
        ),
        repeats=args.repeats,
        warmup=args.warmup,
    )
    steps.append(row)
    decomposed_adjoint_full_s = row["median_s"]

    finufft_adjoint_s = None
    finufft_adjoint_l2 = None
    finufft_adjoint_skip_reason = None
    if args.include_finufft_adjoint:
        try:
            finufft_adjoint_value, row = time_call(
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
            )
            finufft_adjoint_s = row["median_s"]
            finufft_adjoint_l2 = relative_l2(finufft_adjoint_value, adjoint_full)
            row["finufft_eps"] = args.finufft_eps
            row["finufft_adjoint_l2_vs_decomp"] = finufft_adjoint_l2
            row["finufft_over_decomp_adjoint_speedup"] = speedup(
                finufft_adjoint_s, decomposed_adjoint_full_s
            )
            steps.append(row)
        except Exception as exc:  # pragma: no cover - optional dependency/runtime path.
            finufft_adjoint_skip_reason = str(exc)

    if not args.adjoint_only:
        _, row = time_call(
            "forward_minus_data_residual",
            "loop_extra",
            lambda: forward_full - residual,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        steps.append(row)

        _, row = time_call(
            "gradient_axpy_update",
            "loop_extra",
            lambda: obj.coeff - 1.0e-3 * adjoint_full,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        steps.append(row)

    full_pair_s = sum(
        row["median_s"]
        for row in steps
        if row["step"]
        in (
            {"decomposed_adjoint_full"}
            if args.adjoint_only
            else {"decomposed_forward_full", "decomposed_adjoint_full"}
        )
    )
    component_pair_s = sum(
        row["median_s"]
        for row in steps
        if row["step"]
        in (
            {
                "detector_phi_ifft_residual",
                "cpp_cone_adjoint_unfold_scatter",
                "cpp_cone_adjoint_unfold_scatter_pruned",
                "beta_fft_h_to_coeff",
            }
            if args.adjoint_only
            else {
                "beta_ifft_coeff_to_h",
                "cpp_cone_forward_fold",
                "cpp_cone_forward_fold_pruned",
                "detector_phi_fft",
                "detector_phi_ifft_residual",
                "cpp_cone_adjoint_unfold_scatter",
                "cpp_cone_adjoint_unfold_scatter_pruned",
                "beta_fft_h_to_coeff",
            }
        )
    )
    for row in steps:
        row["requested_n_beta"] = requested_n_beta
        row["n_beta"] = n_beta
        row["requested_cap_phi"] = args.requested_cap_phi
        row["cap_phi"] = args.cap_phi
        row["n_beta_fft_friendly"] = n_beta_fft_friendly
        row["cap_phi_fft_friendly"] = cap_phi_fft_friendly
        row["fft_friendly_grid"] = args.fft_friendly_grid
        row["fft_grid_adjusted"] = fft_grid_adjusted
        row["adjoint_only"] = args.adjoint_only
        row["n_illum"] = args.n_illum
        row["cpp_threads"] = args.cpp_threads
        row["cone_l_prune_threshold"] = args.cone_l_prune_threshold
        row["cone_l_active_fraction"] = (
            None
            if decomp.active_l_indices is None
            else float(decomp.active_l_indices.size)
            / float(decomp.transverse_coeff.shape[0] * decomp.transverse_coeff.shape[1])
        )
        row["l_cutoff"] = l_cutoff
        row["l_modes"] = int(2 * l_cutoff + 1)
        row["pct_of_full_pair"] = (
            None if full_pair_s <= 0.0 else 100.0 * float(row["median_s"]) / full_pair_s
        )
        row["pct_of_component_pair"] = (
            None if component_pair_s <= 0.0 else 100.0 * float(row["median_s"]) / component_pair_s
        )

    return {
        "requested_n_beta": requested_n_beta,
        "n_beta": n_beta,
        "requested_cap_phi": args.requested_cap_phi,
        "cap_phi": args.cap_phi,
        "n_beta_fft_friendly": n_beta_fft_friendly,
        "cap_phi_fft_friendly": cap_phi_fft_friendly,
        "fft_friendly_grid": args.fft_friendly_grid,
        "fft_grid_adjusted": fft_grid_adjusted,
        "adjoint_only": args.adjoint_only,
        "n_illum": args.n_illum,
        "cpp_threads": args.cpp_threads,
        "cone_l_prune_threshold": args.cone_l_prune_threshold,
        "cone_l_active_fraction": (
            None
            if decomp.active_l_indices is None
            else float(decomp.active_l_indices.size)
            / float(decomp.transverse_coeff.shape[0] * decomp.transverse_coeff.shape[1])
        ),
        "illumination_na": args.illumination_na,
        "cap_radial": args.cap_radial,
        "axis_h_cutoff": axis_h_cutoff,
        "axis_used_modes": plan.used_modes,
        "l_cutoff": l_cutoff,
        "l_modes": int(2 * l_cutoff + 1),
        "object_bins": int(obj.coeff.size),
        "cone_samples": int(expected),
        "full_pair_s": full_pair_s,
        "decomposed_adjoint_full_s": decomposed_adjoint_full_s,
        "component_pair_s": component_pair_s,
        "forward_component_l2": (
            None
            if args.adjoint_only or forward_from_steps is None or forward_full is None
            else relative_l2(forward_from_steps, forward_full)
        ),
        "adjoint_component_l2": relative_l2(adjoint_from_steps, adjoint_full),
        "finufft_adjoint_s": finufft_adjoint_s,
        "finufft_adjoint_l2_vs_decomp": finufft_adjoint_l2,
        "finufft_over_decomp_adjoint_speedup": speedup(
            finufft_adjoint_s, decomposed_adjoint_full_s
        ),
        "finufft_adjoint_skip_reason": finufft_adjoint_skip_reason,
        "steps": steps,
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    cases = payload["cases"]
    lines = [
        "# ODT cone-axis bottleneck profile",
        "",
        "This profile decomposes the fused full-rank cone z0/zperp path into FFT, C++ fused-kernel, and simple reconstruction-loop array operations.",
        "",
        "When run with `--adjoint-only`, forward-side timings are skipped and the reported full/component sums refer only to the adjoint path.",
        "",
        "FFT-sensitive axes are the object beta grid (`n_beta`) and detector azimuth grid (`cap_phi`). With `--fft-friendly-grid`, requested lengths are rounded upward to SciPy/pocketFFT-friendly complex FFT lengths and both requested/effective lengths are recorded.",
        "",
        "## Case Summary",
        "",
        "| requested n_beta | n_beta | requested cap_phi | cap_phi | FFT friendly | threads | l modes | full pair s | component sum s | fwd check | adj check | dominant component |",
        "| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in cases:
        component_steps = [
            row
            for row in case["steps"]
            if row["group"] in {"forward", "adjoint"}
        ]
        dominant = max(component_steps, key=lambda row: row["median_s"])
        fft_status = (
            "yes"
            if case["n_beta_fft_friendly"] and case["cap_phi_fft_friendly"]
            else "no"
        )
        if case["fft_grid_adjusted"]:
            fft_status += " (adjusted)"
        lines.append(
            "| {req_nb} | {nb} | {req_cp} | {cp} | {fft} | {threads} | {lm} | {full} | {comp} | {fe} | {ae} | {dom} `{dt}` |".format(
                req_nb=case["requested_n_beta"],
                nb=case["n_beta"],
                req_cp=case["requested_cap_phi"],
                cp=case["cap_phi"],
                fft=fft_status,
                threads=case["cpp_threads"],
                lm=case["l_modes"],
                full=fmt(case["full_pair_s"], 5),
                comp=fmt(case["component_pair_s"], 5),
                fe=fmt(case["forward_component_l2"], 3),
                ae=fmt(case["adjoint_component_l2"], 3),
                dom=dominant["step"],
                dt=fmt(dominant["median_s"], 5),
            )
        )

    lines.extend(["", "## Step Timings", ""])
    for case in cases:
        lines.extend(
            [
                f"### n_beta={case['n_beta']}",
                "",
                "| group | step | median s | pair % | component % |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in sorted(case["steps"], key=lambda item: item["median_s"], reverse=True):
            lines.append(
                "| {group} | {step} | {time} | {pair_pct} | {component_pct} |".format(
                    group=row["group"],
                    step=row["step"],
                    time=fmt(row["median_s"], 5),
                    pair_pct=fmt(row["pct_of_full_pair"], 3),
                    component_pct=fmt(row["pct_of_component_pair"], 3),
                )
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile bottlenecks in fused cone-axis ODT.")
    parser.add_argument("--output-prefix", default="benchmark_results/odt_cone_axis_bottleneck_profile")
    parser.add_argument("--n-beta-values", default="192,384")
    parser.add_argument("--n-illum", type=int, default=32)
    parser.add_argument("--illumination-na", type=float, default=0.2)
    parser.add_argument("--l-margin", type=int, default=18)
    parser.add_argument("--phantom", choices=["beads", "random_beads", "shell"], default="beads")
    parser.add_argument("--n-r", type=int, default=16)
    parser.add_argument("--n-z", type=int, default=15)
    parser.add_argument("--r-max", type=float, default=1.5)
    parser.add_argument("--z-max", type=float, default=1.2)
    parser.add_argument("--k", type=float, default=8.0)
    parser.add_argument("--detector-na", type=float, default=0.45)
    parser.add_argument("--cap-radial", type=int, default=8)
    parser.add_argument("--cap-phi", type=int, default=32)
    parser.add_argument(
        "--fft-friendly-grid",
        action="store_true",
        help="Round n_beta and cap_phi upward to SciPy/pocketFFT-friendly complex FFT lengths.",
    )
    parser.add_argument("--h-cutoff", type=int, default=None)
    parser.add_argument("--h-margin", type=int, default=18)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cpp-threads", type=int, default=0)
    parser.add_argument("--cone-l-prune-threshold", type=float, default=0.0)
    parser.add_argument(
        "--adjoint-only",
        action="store_true",
        help="Skip forward timings and report only detector-IFFT, fused adjoint, beta-FFT, and full adjoint timings.",
    )
    parser.add_argument(
        "--include-finufft-adjoint",
        action="store_true",
        help="Also time FINUFFT type-3 adjoint on the same flat q-list and residual.",
    )
    parser.add_argument("--finufft-eps", type=float, default=1e-12)
    parser.add_argument("--finufft-repeats", type=int, default=5)
    parser.add_argument("--finufft-warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
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
    if args.cpp_threads < 0:
        raise ValueError("cpp-threads must be non-negative")
    if args.cap_phi <= 0 or args.cap_radial <= 0:
        raise ValueError("cap-radial and cap-phi must be positive")
    if args.cone_l_prune_threshold < 0.0:
        raise ValueError("cone-l-prune-threshold must be non-negative")
    if args.finufft_eps <= 0.0:
        raise ValueError("finufft-eps must be positive")
    if args.finufft_repeats <= 0 or args.finufft_warmup < 0:
        raise ValueError("finufft repeats must be positive and warmup must be non-negative")
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
                        "requested_n_beta": case["requested_n_beta"],
                        "n_beta": case["n_beta"],
                        "requested_cap_phi": case["requested_cap_phi"],
                        "cap_phi": case["cap_phi"],
                        "fft_grid_adjusted": case["fft_grid_adjusted"],
                        "finufft_adjoint_s": case["finufft_adjoint_s"],
                        "finufft_over_decomp_adjoint_speedup": case[
                            "finufft_over_decomp_adjoint_speedup"
                        ],
                        "full_pair_s": case["full_pair_s"],
                        "component_pair_s": case["component_pair_s"],
                    }
                    for case in cases
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
