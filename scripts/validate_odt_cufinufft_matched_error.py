from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_high_na_torch_gpu import (  # noqa: E402
    device_name,
    import_torch,
    resolve_device,
    synchronize,
)
from benchmark_odt_cufinufft_gpu_baseline import (  # noqa: E402
    cufinufft_adjoint,
    cufinufft_forward,
    flatten_cufinufft_forward,
    make_cufinufft_composite,
    parser as baseline_parser,
    sum_cufinufft_adjoint,
    synchronize_cupy,
    timed_cupy,
    timed_torch,
    timing_distribution,
)
from benchmark_odt_realistic_geometry_reconstruction import (  # noqa: E402
    build_composite_context,
    split_residual,
)
from benchmark_odt_torch_gpu_reconstruction import (  # noqa: E402
    TorchCompositeOdtPlan,
)
from validate_odt_128cubed_independent_subset import (  # noqa: E402
    direct_adjoint_selected_objects,
    direct_forward_subset,
    rel_l2,
)


def parse_eps_values(text: str) -> list[float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("eps-values must contain positive finite values")
    if len(set(values)) != len(values):
        raise ValueError("eps-values must not contain duplicates")
    return sorted(values, reverse=True)


def concat_q(context: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blocks = [context.ring.flat_q]
    if context.axis is not None:
        blocks.append(context.axis.flat_q)
    return tuple(
        np.concatenate([np.asarray(getattr(block, name)) for block in blocks])
        for name in ("qx", "qy", "qz")
    )  # type: ignore[return-value]


def exact_dot_error(
    forward: np.ndarray,
    residual: np.ndarray,
    coeff: np.ndarray,
    adjoint: np.ndarray,
) -> float:
    lhs = np.vdot(forward.astype(np.complex128), residual.astype(np.complex128))
    rhs = np.vdot(coeff.astype(np.complex128), adjoint.astype(np.complex128))
    return float(
        abs(lhs - rhs)
        / max(abs(lhs) + abs(rhs), np.finfo(np.float64).tiny)
    )


def select_matched(rows: list[dict[str, Any]], ours: dict[str, float]) -> dict[str, Any]:
    strict = [
        row
        for row in rows
        if row["forward_rel_l2_vs_direct"] <= ours["forward_rel_l2_vs_direct"]
        and row["adjoint_rel_l2_vs_direct"] <= ours["adjoint_rel_l2_vs_direct"]
    ]
    worst_tier = [
        row
        for row in rows
        if row["worst_rel_l2_vs_direct"] <= ours["worst_rel_l2_vs_direct"]
    ]
    chosen = strict[0] if strict else (worst_tier[0] if worst_tier else None)
    return {
        "criterion": (
            "direction-wise no worse than ACFO"
            if strict
            else (
                "worst forward/adjoint rel-L2 no worse than ACFO"
                if worst_tier
                else "no tested cuFINUFFT epsilon matched ACFO"
            )
        ),
        "eps": None if chosen is None else float(chosen["eps"]),
        "row": chosen,
        "strict_directional_match_exists": bool(strict),
        "worst_tier_match_exists": bool(worst_tier),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    eps_values = parse_eps_values(args.eps_values)
    if args.q_subset_count <= 0 or args.q_chunk <= 0 or args.object_chunk <= 0:
        raise ValueError("direct-reference sizes must be positive")

    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA device required")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    context_start = time.perf_counter()
    context = build_composite_context(args)
    context_s = time.perf_counter() - context_start
    plan_start = time.perf_counter()
    plan = TorchCompositeOdtPlan.from_context(
        context,
        torch=torch,
        device=device,
        dtype=args.dtype,
        low_memory_adjoint=args.low_memory_adjoint,
        radial_block_size=args.radial_block_size,
        illumination_block_size=args.illumination_block_size,
        prune_axis_l0=args.prune_axis_l0,
        axial_lowrank_rank=args.axial_lowrank_rank,
        ring_adaptive_l_packed_threshold=args.ring_adaptive_l_packed_threshold,
    )
    synchronize(torch, device)
    plan_s = time.perf_counter() - plan_start

    coeff_128 = np.ascontiguousarray(
        context.ring.obj.coeff.astype(np.complex128, copy=False)
    )
    coeff_64 = np.ascontiguousarray(coeff_128.astype(np.complex64))
    coeff_t = torch.as_tensor(coeff_64, dtype=torch.complex64, device=device)
    qx, qy, qz = concat_q(context)
    if qx.size != plan.q_count:
        raise RuntimeError("composite q ordering/count mismatch")
    q_subset = np.unique(
        np.linspace(0, qx.size - 1, args.q_subset_count, dtype=np.int64)
    )

    rng = np.random.default_rng(args.direct_seed)
    residual_values_128 = (
        rng.standard_normal(q_subset.size) + 1j * rng.standard_normal(q_subset.size)
    ).astype(np.complex128)
    residual_128 = np.zeros(qx.size, dtype=np.complex128)
    residual_128[q_subset] = residual_values_128
    residual_64 = np.ascontiguousarray(residual_128.astype(np.complex64))
    residual_t = torch.as_tensor(residual_64, dtype=torch.complex64, device=device)

    direct_forward_start = time.perf_counter()
    direct_forward = direct_forward_subset(
        np.asarray(context.ring.obj.x, dtype=np.float64),
        np.asarray(context.ring.obj.y, dtype=np.float64),
        np.asarray(context.ring.obj.z, dtype=np.float64),
        coeff_128.ravel(),
        qx[q_subset],
        qy[q_subset],
        qz[q_subset],
        q_chunk=args.q_chunk,
    )
    direct_forward_s = time.perf_counter() - direct_forward_start
    direct_adjoint_start = time.perf_counter()
    direct_adjoint = direct_adjoint_selected_objects(
        np.asarray(context.ring.obj.x, dtype=np.float64),
        np.asarray(context.ring.obj.y, dtype=np.float64),
        np.asarray(context.ring.obj.z, dtype=np.float64),
        residual_values_128,
        qx[q_subset],
        qy[q_subset],
        qz[q_subset],
        object_chunk=args.object_chunk,
    ).reshape(coeff_128.shape)
    direct_adjoint_s = time.perf_counter() - direct_adjoint_start
    direct_dot = exact_dot_error(
        direct_forward,
        residual_values_128,
        coeff_128.ravel(),
        direct_adjoint.ravel(),
    )

    with torch.inference_mode():
        ours_forward_t, ours_forward_s, ours_forward_times = timed_torch(
            torch,
            device,
            lambda: plan.forward(coeff_t),
            repeats=args.repeats,
            warmups=args.warmups,
        )
        ours_adjoint_t, ours_adjoint_s, ours_adjoint_times = timed_torch(
            torch,
            device,
            lambda: plan.adjoint(residual_t),
            repeats=args.repeats,
            warmups=args.warmups,
        )
    ours_forward = ours_forward_t.detach().cpu().numpy().astype(np.complex128)
    ours_adjoint = ours_adjoint_t.detach().cpu().numpy().astype(np.complex128)
    ours = {
        "forward_rel_l2_vs_direct": rel_l2(
            ours_forward[q_subset], direct_forward
        ),
        "adjoint_rel_l2_vs_direct": rel_l2(ours_adjoint, direct_adjoint),
        "forward_hot_s": float(ours_forward_s),
        "adjoint_hot_s": float(ours_adjoint_s),
        "pair_hot_s": float(ours_forward_s + ours_adjoint_s),
        "forward_timing": timing_distribution(ours_forward_times),
        "adjoint_timing": timing_distribution(ours_adjoint_times),
        "dot_error_on_direct_subset": exact_dot_error(
            ours_forward[q_subset],
            residual_values_128,
            coeff_128.ravel(),
            ours_adjoint.ravel(),
        ),
    }
    ours["worst_rel_l2_vs_direct"] = max(
        ours["forward_rel_l2_vs_direct"], ours["adjoint_rel_l2_vs_direct"]
    )

    cu_np_dtype = (
        np.complex64 if args.cufinufft_dtype == "complex64" else np.complex128
    )
    coeff_cu = np.ascontiguousarray(coeff_128.astype(cu_np_dtype))
    residual_cu = np.ascontiguousarray(residual_128.astype(cu_np_dtype))
    ring_residual, axis_residual = split_residual(context, residual_cu)
    rows: list[dict[str, Any]] = []
    cp = None
    cufinufft_version = None
    cupy_version = None
    for eps in eps_values:
        setup_start = time.perf_counter()
        cu_op = make_cufinufft_composite(
            context,
            dtype=args.cufinufft_dtype,
            plan_mode="plan",
            eps=eps,
        )
        cp = cu_op.cp
        synchronize_cupy(cp)
        setup_s = time.perf_counter() - setup_start
        cufinufft_version = getattr(cu_op.cufinufft, "__version__", None)
        cupy_version = getattr(cp, "__version__", None)
        coeff_gpu = cp.asarray(coeff_cu.ravel())
        residual_parts = [cp.asarray(ring_residual)]
        if axis_residual is not None:
            residual_parts.append(cp.asarray(axis_residual))
        forward_parts, forward_s, forward_times = timed_cupy(
            cp,
            lambda: cufinufft_forward(cu_op, coeff_gpu, eps=eps),
            repeats=args.repeats,
            warmups=args.warmups,
        )
        adjoint_parts, adjoint_s, adjoint_times = timed_cupy(
            cp,
            lambda: cufinufft_adjoint(cu_op, residual_parts, eps=eps),
            repeats=args.repeats,
            warmups=args.warmups,
        )
        forward = flatten_cufinufft_forward(forward_parts).astype(np.complex128)
        adjoint = sum_cufinufft_adjoint(
            adjoint_parts, coeff_128.shape
        ).astype(np.complex128)
        row = {
            "eps": float(eps),
            "setup_s": float(setup_s),
            "forward_rel_l2_vs_direct": rel_l2(
                forward[q_subset], direct_forward
            ),
            "adjoint_rel_l2_vs_direct": rel_l2(adjoint, direct_adjoint),
            "forward_hot_s": float(forward_s),
            "adjoint_hot_s": float(adjoint_s),
            "pair_hot_s": float(forward_s + adjoint_s),
            "forward_timing": timing_distribution(forward_times),
            "adjoint_timing": timing_distribution(adjoint_times),
            "dot_error_on_direct_subset": exact_dot_error(
                forward[q_subset],
                residual_values_128,
                coeff_128.ravel(),
                adjoint.ravel(),
            ),
            "forward_rel_l2_vs_acfo": rel_l2(
                forward[q_subset], ours_forward[q_subset]
            ),
            "adjoint_rel_l2_vs_acfo": rel_l2(adjoint, ours_adjoint),
        }
        row["worst_rel_l2_vs_direct"] = max(
            row["forward_rel_l2_vs_direct"], row["adjoint_rel_l2_vs_direct"]
        )
        rows.append(row)
        print(
            f"eps={eps:.0e}: direct fwd={row['forward_rel_l2_vs_direct']:.3e}, "
            f"adj={row['adjoint_rel_l2_vs_direct']:.3e}, "
            f"pair={1000.0 * row['pair_hot_s']:.3f} ms",
            flush=True,
        )
        del (
            cu_op,
            coeff_gpu,
            residual_parts,
            forward_parts,
            adjoint_parts,
            forward,
            adjoint,
        )
        gc.collect()
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()

    matched = select_matched(rows, ours)
    result = {
        "schema": "odt-cufinufft-matched-error-direct-subset-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": {
            "name": device_name(torch, device),
            "torch": getattr(torch, "__version__", None),
            "cupy": cupy_version,
            "cufinufft": cufinufft_version,
            "acfo_dtype": args.dtype,
            "cufinufft_dtype": args.cufinufft_dtype,
        },
        "problem": {
            "object_shape": list(coeff_128.shape),
            "object_bins": int(coeff_128.size),
            "illumination_count": int(args.ring_illum)
            + (0 if args.skip_axis_illumination else 1),
            "detector_shape": [int(args.cap_radial), int(args.cap_phi)],
            "total_q_samples": int(qx.size),
            "direct_q_subset": int(q_subset.size),
            "q_selection": "uniform indices spanning the full concatenated ring-plus-axis q ordering",
            "direct_forward_support": "all object bins and selected q nodes",
            "direct_adjoint_support": "selected q residual and all object bins",
            "direct_reference_dtype": "complex128 NumPy exponent sum",
            "seed": int(args.direct_seed),
        },
        "candidate_contract": {
            "h_cutoff": int(args.h_cutoff),
            "prune_axis_l0": bool(args.prune_axis_l0),
            "axial_lowrank_rank_requested": int(args.axial_lowrank_rank),
            "ring_axial_lowrank_rank": int(plan.ring.axial_lowrank_rank),
            "ring_adaptive_l_packed_threshold": float(
                args.ring_adaptive_l_packed_threshold
            ),
            "ring_adaptive_l_active_fraction": float(
                plan.ring.adaptive_l_active_fraction
            ),
        },
        "build_timing_s": {
            "context": float(context_s),
            "acfo_plan": float(plan_s),
        },
        "direct_reference": {
            "forward_s": float(direct_forward_s),
            "adjoint_s": float(direct_adjoint_s),
            "dot_error": float(direct_dot),
        },
        "acfo": ours,
        "cufinufft_sweep": rows,
        "matched_error_selection": matched,
        "gates": {
            "direct_dot_error_le_1e-12": bool(direct_dot <= 1e-12),
            "finite_acfo_errors": bool(
                np.isfinite(ours["forward_rel_l2_vs_direct"])
                and np.isfinite(ours["adjoint_rel_l2_vs_direct"])
            ),
            "all_cufinufft_errors_finite": bool(
                all(np.isfinite(row["worst_rel_l2_vs_direct"]) for row in rows)
            ),
            "selection_outcome_resolved": bool(
                matched["eps"] is not None
                or (
                    rows
                    and min(row["worst_rel_l2_vs_direct"] for row in rows)
                    > ours["worst_rel_l2_vs_direct"]
                )
            ),
        },
        "claim_boundary": [
            "The independent reference is the literal complex128 Cartesian exponent sum, not ACFO, FINUFFT or cuFINUFFT.",
            "The forward reference uses every object bin but a frozen 4096-node q subset; the adjoint uses the same q subset and returns every object bin.",
            "cuFINUFFT executes the complete q cloud for every epsilon; only accuracy scoring is restricted to the frozen direct subset.",
            "Small-case hot timings diagnose tolerance behavior but are not the publication-scale speedup. The selected epsilon must be rerun with the frozen 256-cubed AB/BA protocol.",
            "The ACFO candidate uses the production H28/rank16/adaptive-L contract, but the smaller geometry is an accuracy audit rather than proof of full-scale direct error.",
            "ACFO remains complex64 in every run. cufinufft_dtype is explicit so a complex128 cuFINUFFT run can test whether higher precision is required to match the ACFO direct error.",
        ],
    }
    result["passed"] = bool(all(result["gates"].values()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.summary_md:
        lines = [
            "# ODT ACFO-cuFINUFFT matched-error direct audit",
            "",
            f"- independent direct dot error: `{direct_dot:.3e}`",
            f"- ACFO forward/adjoint direct error: `{ours['forward_rel_l2_vs_direct']:.3e}` / `{ours['adjoint_rel_l2_vs_direct']:.3e}`",
            f"- ACFO worst direct error: `{ours['worst_rel_l2_vs_direct']:.3e}`",
            f"- selected cuFINUFFT epsilon: `{matched['eps']}`",
            f"- selection criterion: `{matched['criterion']}`",
            "",
            "| eps | forward direct L2 | adjoint direct L2 | worst | pair ms |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                f"| {row['eps']:.0e} | {row['forward_rel_l2_vs_direct']:.3e} | "
                f"{row['adjoint_rel_l2_vs_direct']:.3e} | "
                f"{row['worst_rel_l2_vs_direct']:.3e} | "
                f"{1000.0 * row['pair_hot_s']:.3f} |"
            )
        lines.extend(["", *[f"- {item}" for item in result["claim_boundary"]]])
        args.summary_md.parent.mkdir(parents=True, exist_ok=True)
        args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def parser() -> argparse.ArgumentParser:
    p = baseline_parser()
    p.description = (
        "Select a cuFINUFFT epsilon against the same independent direct-sum "
        "forward and adjoint reference used to score the ACFO candidate."
    )
    p.add_argument(
        "--eps-values",
        default="1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5,3e-6,1e-6",
    )
    p.add_argument("--q-subset-count", type=int, default=4096)
    p.add_argument("--q-chunk", type=int, default=128)
    p.add_argument("--object-chunk", type=int, default=128)
    p.add_argument("--direct-seed", type=int, default=20260714)
    p.set_defaults(
        dtype="complex64",
        cufinufft_dtype="complex64",
        low_memory_adjoint=True,
        radial_block_size=32,
        illumination_block_size=4,
        prune_axis_l0=True,
        axial_lowrank_rank=16,
        ring_adaptive_l_packed_threshold=1e-6,
        skip_native_prepared_adjoint=True,
        compact_axisymmetric_kernel=True,
        n_beta=64,
        n_r=24,
        n_z=24,
        ring_illum=8,
        skip_axis_illumination=False,
        cap_radial=24,
        cap_phi=64,
        h_cutoff=28,
        cpp_threads=4,
        repeats=5,
        warmups=2,
        pair_repeats=0,
        pair_warmups=0,
        out=ROOT
        / "benchmark_results"
        / "odt_cufinufft_matched_error_direct_subset.json",
        summary_md=ROOT
        / "benchmark_results"
        / "odt_cufinufft_matched_error_direct_subset_ko.md",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    result = run(args)
    compact = {
        "passed": result["passed"],
        "gates": result["gates"],
        "direct_dot_error": result["direct_reference"]["dot_error"],
        "acfo": {
            "forward": result["acfo"]["forward_rel_l2_vs_direct"],
            "adjoint": result["acfo"]["adjoint_rel_l2_vs_direct"],
            "worst": result["acfo"]["worst_rel_l2_vs_direct"],
        },
        "matched": result["matched_error_selection"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
