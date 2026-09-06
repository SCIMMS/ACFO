from __future__ import annotations

import gc
import json
import sys
import time
from dataclasses import replace
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
    make_cufinufft_composite,
)
from benchmark_odt_ewald_cap_operator import CylindricalObject  # noqa: E402
from benchmark_odt_realistic_geometry_reconstruction import (  # noqa: E402
    CompositeContext,
    build_composite_context,
)
from benchmark_odt_torch_gpu_reconstruction import (  # noqa: E402
    TorchCompositeOdtPlan,
    parser as base_parser,
    torch_dtypes,
)


def timing_summary(times: list[float]) -> dict[str, float | int]:
    values = np.asarray(times, dtype=np.float64)
    return {
        "count": int(values.size),
        "total_s": float(values.sum()),
        "mean_s": float(values.mean()),
        "median_s": float(np.median(values)),
        "p05_s": float(np.percentile(values, 5)),
        "p95_s": float(np.percentile(values, 95)),
        "min_s": float(values.min()),
        "max_s": float(values.max()),
    }


def centered_z_indices(n_full: int, n_selected: int) -> np.ndarray:
    if n_full <= 0 or n_selected <= 0 or n_selected > n_full:
        raise ValueError("require 0 < n_selected <= n_full")
    start = (n_full - n_selected) // 2
    return np.arange(start, start + n_selected, dtype=np.int64)


def selected_object(obj: CylindricalObject, z_indices: np.ndarray) -> CylindricalObject:
    n_r, n_z, n_beta = obj.coeff.shape
    index = np.asarray(z_indices, dtype=np.int64)

    def selected_flat(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values).reshape(n_r, n_z, n_beta)
        return np.ascontiguousarray(array[:, index, :].ravel())

    coeff = np.ascontiguousarray(obj.coeff[:, index, :])
    return CylindricalObject(
        r_axis=obj.r_axis,
        z_axis=np.ascontiguousarray(obj.z_axis[index]),
        beta_axis=obj.beta_axis,
        coeff=coeff,
        x=selected_flat(obj.x),
        y=selected_flat(obj.y),
        z=selected_flat(obj.z),
        weights=np.ascontiguousarray(coeff.ravel()),
        volume_weights=selected_flat(obj.volume_weights),
    )


def selected_context(ctx: CompositeContext, z_indices: np.ndarray) -> CompositeContext:
    ring = replace(ctx.ring, obj=selected_object(ctx.ring.obj, z_indices))
    axis = None
    if ctx.axis is not None:
        axis = replace(ctx.axis, obj=selected_object(ctx.axis.obj, z_indices))
    return CompositeContext(ring=ring, axis=axis)


def relative_l2_torch(torch: Any, candidate: Any, reference: Any) -> float:
    denom = torch.clamp(torch.linalg.vector_norm(reference), min=1e-30)
    return float((torch.linalg.vector_norm(candidate - reference) / denom).detach().cpu().item())


def relative_l2_numpy(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_128 = np.asarray(candidate, dtype=np.complex128)
    reference_128 = np.asarray(reference, dtype=np.complex128)
    denominator = max(float(np.linalg.norm(reference_128.ravel())), 1e-30)
    return float(np.linalg.norm((candidate_128 - reference_128).ravel()) / denominator)


def common_result(args: Any, ctx: CompositeContext, z_indices: np.ndarray) -> dict[str, Any]:
    return {
        "schema": "odt-selected-z-gpu-hot-pair-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": vars(args),
        "full_data_used": True,
        "full_n_z": int(args.n_z),
        "selected_n_z": int(z_indices.size),
        "selected_z_indices": z_indices.tolist(),
        "selected_z_positions": ctx.ring.obj.z_axis[z_indices].tolist(),
        "full_object_bins": int(np.prod(ctx.ring.obj.coeff.shape)),
        "selected_object_bins": int(args.n_r * z_indices.size * args.n_beta),
        "q_samples": int(ctx.q_count),
        "total_illumination_count": int(args.ring_illum)
        + (0 if args.skip_axis_illumination else 1),
        "claim_boundary": [
            "All illumination and detector q samples are used.",
            "The selected pair is a restricted-support reconstruction operator on full-grid z planes.",
            "The selected adjoint alone exactly equals slicing the full-data adjoint.",
        ],
    }


def run_acfo(args: Any, ctx: CompositeContext, z_indices: np.ndarray) -> dict[str, Any]:
    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA device required")
    torch.cuda.empty_cache()

    setup_start = time.perf_counter()
    plan = TorchCompositeOdtPlan.from_context(
        ctx,
        torch=torch,
        device=device,
        dtype=args.dtype,
        low_memory_adjoint=True,
        radial_block_size=args.radial_block_size,
        illumination_block_size=args.illumination_block_size,
        forward_mode="illumination-reduced",
        adjoint_mode="illumination-reduced",
    )
    _, _, np_complex, _ = torch_dtypes(torch, args.dtype)
    coeff_np = np.ascontiguousarray(
        ctx.ring.obj.coeff[:, z_indices, :].astype(np_complex, copy=False)
    )
    if args.real_object:
        coeff_np = np.ascontiguousarray(np.real(coeff_np).astype(np_complex))
    coeff = torch.as_tensor(coeff_np, dtype=plan.complex_dtype, device=device)
    index = torch.as_tensor(z_indices, dtype=torch.long, device=device)
    synchronize(torch, device)
    setup_s = time.perf_counter() - setup_start

    scale = complex(args.residual_scale_real, args.residual_scale_imag)
    with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
        first_start = time.perf_counter()
        first_forward = plan.forward_selected_z(coeff, index)
        residual = first_forward * scale
        first_adjoint = plan.adjoint_selected_z(residual, index)
        synchronize(torch, device)
        first_pair_s = time.perf_counter() - first_start

        dot_lhs = torch.vdot(first_forward.reshape(-1), residual.reshape(-1))
        dot_rhs = torch.vdot(coeff.reshape(-1), first_adjoint.reshape(-1))
        dot_error = float(
            (
                torch.abs(dot_lhs - dot_rhs)
                / torch.clamp(torch.abs(dot_lhs) + torch.abs(dot_rhs), min=1e-30)
            )
            .detach()
            .cpu()
            .item()
        )

        forward_restriction_rel_l2 = None
        adjoint_slice_rel_l2 = None
        if args.validate_full_restriction:
            coeff_full = torch.zeros(
                (args.n_r, args.n_z, args.n_beta),
                dtype=plan.complex_dtype,
                device=device,
            )
            coeff_full.index_copy_(1, index, coeff)
            full_forward = plan.forward(coeff_full)
            full_adjoint = plan.adjoint(residual)
            synchronize(torch, device)
            forward_restriction_rel_l2 = relative_l2_torch(
                torch, first_forward, full_forward
            )
            adjoint_slice_rel_l2 = relative_l2_torch(
                torch, first_adjoint, full_adjoint.index_select(1, index)
            )
            del coeff_full, full_forward, full_adjoint
            torch.cuda.empty_cache()

        def pair_call() -> tuple[Any, Any]:
            forward = plan.forward_selected_z(coeff, index)
            adjoint = plan.adjoint_selected_z(residual, index)
            return forward, adjoint

        for _ in range(args.warmups):
            pair_call()
            synchronize(torch, device)
        torch.cuda.reset_peak_memory_stats(device)
        times: list[float] = []
        for _ in range(args.repeats):
            synchronize(torch, device)
            start = time.perf_counter()
            pair_call()
            synchronize(torch, device)
            times.append(time.perf_counter() - start)

    result = common_result(args, ctx, z_indices)
    result.update(
        {
            "backend": "ACFO selected-z illumination-reduced PyTorch/CUDA",
            "device": str(device),
            "device_name": device_name(torch, device),
            "torch_version": getattr(torch, "__version__", None),
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "setup_s": float(setup_s),
            "first_pair_s": float(first_pair_s),
            "pair_timing": timing_summary(times),
            "gpu_hot_peak_allocated_mib": float(
                torch.cuda.max_memory_allocated(device) / 1024**2
            ),
            "dot_error": dot_error,
            "forward_restriction_rel_l2": forward_restriction_rel_l2,
            "adjoint_slice_rel_l2": adjoint_slice_rel_l2,
            "passed": bool(
                np.all(np.isfinite(times))
                and dot_error <= 1e-6
                and (
                    forward_restriction_rel_l2 is None
                    or forward_restriction_rel_l2 <= 2e-6
                )
                and (
                    adjoint_slice_rel_l2 is None or adjoint_slice_rel_l2 <= 2e-6
                )
            ),
        }
    )
    return result


def run_cufinufft(args: Any, ctx: CompositeContext, z_indices: np.ndarray) -> dict[str, Any]:
    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA device required")
    selected_ctx = selected_context(ctx, z_indices)

    setup_start = time.perf_counter()
    op = make_cufinufft_composite(
        selected_ctx,
        dtype=args.dtype,
        plan_mode="plan",
        eps=args.cufinufft_eps,
    )
    cp = op.cp
    _, _, np_complex, _ = torch_dtypes(torch, args.dtype)
    coeff_np = np.ascontiguousarray(
        selected_ctx.ring.obj.coeff.astype(np_complex, copy=False).ravel()
    )
    if args.real_object:
        coeff_np = np.ascontiguousarray(np.real(coeff_np).astype(np_complex))
    coeff = cp.asarray(coeff_np)
    cp.cuda.get_current_stream().synchronize()
    setup_s = time.perf_counter() - setup_start

    scale = complex(args.residual_scale_real, args.residual_scale_imag)
    first_start = time.perf_counter()
    first_forward = cufinufft_forward(op, coeff, eps=args.cufinufft_eps)
    residual = [part * scale for part in first_forward]
    first_adjoint_parts = cufinufft_adjoint(op, residual, eps=args.cufinufft_eps)
    first_adjoint = first_adjoint_parts[0]
    if len(first_adjoint_parts) > 1:
        first_adjoint = first_adjoint + first_adjoint_parts[1]
    cp.cuda.get_current_stream().synchronize()
    first_pair_s = time.perf_counter() - first_start

    dot_lhs = sum(cp.vdot(forward, res) for forward, res in zip(first_forward, residual))
    dot_rhs = cp.vdot(coeff, first_adjoint)
    dot_error = float(
        (cp.abs(dot_lhs - dot_rhs) / cp.maximum(cp.abs(dot_lhs) + cp.abs(dot_rhs), 1e-30)).get()
    )

    def pair_call() -> tuple[list[Any], Any]:
        forward = cufinufft_forward(op, coeff, eps=args.cufinufft_eps)
        adjoint_parts = cufinufft_adjoint(op, residual, eps=args.cufinufft_eps)
        adjoint = adjoint_parts[0]
        if len(adjoint_parts) > 1:
            adjoint = adjoint + adjoint_parts[1]
        return forward, adjoint

    for _ in range(args.warmups):
        pair_call()
        cp.cuda.get_current_stream().synchronize()
    times: list[float] = []
    for _ in range(args.repeats):
        cp.cuda.get_current_stream().synchronize()
        start = time.perf_counter()
        pair_call()
        cp.cuda.get_current_stream().synchronize()
        times.append(time.perf_counter() - start)

    pool = cp.get_default_memory_pool()
    result = common_result(args, ctx, z_indices)
    result.update(
        {
            "backend": "cuFINUFFT selected-z reusable type-3 plans",
            "device": str(device),
            "device_name": device_name(torch, device),
            "cupy_version": getattr(cp, "__version__", None),
            "cufinufft_version": getattr(op.cufinufft, "__version__", None),
            "setup_s": float(setup_s),
            "first_pair_s": float(first_pair_s),
            "pair_timing": timing_summary(times),
            "cupy_pool_used_mib": float(pool.used_bytes() / 1024**2),
            "cupy_pool_total_mib": float(pool.total_bytes() / 1024**2),
            "dot_error": dot_error,
            "passed": bool(np.all(np.isfinite(times)) and dot_error <= 5e-5),
        }
    )
    return result


def run_cross_validation(
    args: Any, ctx: CompositeContext, z_indices: np.ndarray
) -> dict[str, Any]:
    """Compare selected-z ACFO and cuFINUFFT outputs on identical full data."""
    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA device required")
    torch.cuda.empty_cache()

    plan = TorchCompositeOdtPlan.from_context(
        ctx,
        torch=torch,
        device=device,
        dtype=args.dtype,
        low_memory_adjoint=True,
        radial_block_size=args.radial_block_size,
        illumination_block_size=args.illumination_block_size,
        forward_mode="illumination-reduced",
        adjoint_mode="illumination-reduced",
    )
    _, _, np_complex, _ = torch_dtypes(torch, args.dtype)
    coeff_np = np.ascontiguousarray(
        ctx.ring.obj.coeff[:, z_indices, :].astype(np_complex, copy=False)
    )
    if args.real_object:
        coeff_np = np.ascontiguousarray(np.real(coeff_np).astype(np_complex))
    coeff_t = torch.as_tensor(coeff_np, dtype=plan.complex_dtype, device=device)
    index_t = torch.as_tensor(z_indices, dtype=torch.long, device=device)
    scale = complex(args.residual_scale_real, args.residual_scale_imag)
    with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
        acfo_forward_t = plan.forward_selected_z(coeff_t, index_t)
        residual_t = acfo_forward_t * scale
        acfo_adjoint_t = plan.adjoint_selected_z(residual_t, index_t)
        synchronize(torch, device)
        acfo_forward = np.asarray(acfo_forward_t.detach().cpu().numpy())
        residual_np = np.asarray(residual_t.detach().cpu().numpy())
        acfo_adjoint = np.asarray(acfo_adjoint_t.detach().cpu().numpy())

    del plan, coeff_t, index_t, acfo_forward_t, residual_t, acfo_adjoint_t
    gc.collect()
    torch.cuda.empty_cache()

    selected_ctx = selected_context(ctx, z_indices)
    op = make_cufinufft_composite(
        selected_ctx,
        dtype=args.dtype,
        plan_mode="plan",
        eps=args.cufinufft_eps,
    )
    cp = op.cp
    coeff_cp = cp.asarray(np.ascontiguousarray(coeff_np.ravel()))
    ring_count = int(ctx.ring.flat_q.count)
    residual_parts = [cp.asarray(np.ascontiguousarray(residual_np[:ring_count]))]
    if ctx.axis is not None:
        residual_parts.append(cp.asarray(np.ascontiguousarray(residual_np[ring_count:])))
    cu_forward_parts = cufinufft_forward(op, coeff_cp, eps=args.cufinufft_eps)
    cu_adjoint_parts = cufinufft_adjoint(op, residual_parts, eps=args.cufinufft_eps)
    cp.cuda.get_current_stream().synchronize()
    cu_forward = np.concatenate([np.asarray(part.get()) for part in cu_forward_parts])
    cu_adjoint = np.asarray(cu_adjoint_parts[0].get())
    if len(cu_adjoint_parts) > 1:
        cu_adjoint = cu_adjoint + np.asarray(cu_adjoint_parts[1].get())
    cu_adjoint = cu_adjoint.reshape(acfo_adjoint.shape)

    forward_rel_l2 = relative_l2_numpy(cu_forward, acfo_forward)
    adjoint_rel_l2 = relative_l2_numpy(cu_adjoint, acfo_adjoint)
    result = common_result(args, ctx, z_indices)
    result.update(
        {
            "schema": "odt-selected-z-gpu-cross-validation-v1",
            "backend": "ACFO vs cuFINUFFT selected-z output comparison",
            "device": str(device),
            "device_name": device_name(torch, device),
            "cufinufft_eps": float(args.cufinufft_eps),
            "cufinufft_forward_rel_l2_vs_acfo": forward_rel_l2,
            "cufinufft_adjoint_rel_l2_vs_acfo": adjoint_rel_l2,
            "passed": bool(forward_rel_l2 <= 2e-3 and adjoint_rel_l2 <= 2e-3),
        }
    )
    return result


def parser() -> Any:
    p = base_parser()
    p.description = "GPU hot execute-only selected-z ODT forward-adjoint benchmark."
    p.add_argument(
        "--backend", choices=["acfo", "cufinufft", "compare"], required=True
    )
    p.add_argument("--n-selected", type=int, required=True)
    p.add_argument("--cufinufft-eps", type=float, default=1e-6)
    p.add_argument("--residual-scale-real", type=float, default=0.1)
    p.add_argument("--residual-scale-imag", type=float, default=0.2)
    p.add_argument("--validate-full-restriction", action="store_true")
    p.set_defaults(
        device="cuda",
        dtype="complex64",
        low_memory_adjoint=True,
        radial_block_size=16,
        illumination_block_size=4,
        forward_mode="illumination-reduced",
        adjoint_mode="illumination-reduced",
        compact_axisymmetric_kernel=True,
        real_object=True,
        n_beta=256,
        n_r=256,
        n_z=256,
        ring_illum=120,
        cap_radial=256,
        cap_phi=256,
        repeats=20,
        warmups=5,
        out=ROOT / "benchmark_results" / "odt_selected_z_gpu.json",
        summary_md=None,
    )
    return p


def main() -> None:
    args = parser().parse_args()
    if args.repeats <= 0 or args.warmups < 0:
        raise ValueError("repeats must be positive and warmups nonnegative")
    z_indices = centered_z_indices(int(args.n_z), int(args.n_selected))
    ctx = build_composite_context(args)
    if args.backend == "acfo":
        result = run_acfo(args, ctx, z_indices)
    elif args.backend == "cufinufft":
        result = run_cufinufft(args, ctx, z_indices)
    else:
        result = run_cross_validation(args, ctx, z_indices)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
