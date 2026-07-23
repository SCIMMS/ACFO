from __future__ import annotations

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
from benchmark_odt_realistic_geometry_reconstruction import (  # noqa: E402
    build_composite_context,
)
from benchmark_odt_torch_gpu_reconstruction import (  # noqa: E402
    TorchCompositeOdtPlan,
    parser as base_parser,
    timed_cuda,
    torch_dtypes,
)


def rel_l2(torch: Any, candidate: Any, reference: Any) -> float:
    denominator = torch.clamp(torch.linalg.vector_norm(reference), min=1e-30)
    return float((torch.linalg.vector_norm(candidate - reference) / denominator).item())


def dot_error(torch: Any, coeff: Any, forward: Any, residual: Any, adjoint: Any) -> float:
    left = torch.vdot(forward.reshape(-1), residual.reshape(-1))
    right = torch.vdot(coeff.reshape(-1), adjoint.reshape(-1))
    denominator = torch.clamp(torch.abs(left) + torch.abs(right), min=1e-30)
    return float((torch.abs(left - right) / denominator).item())


def set_modes(plan: TorchCompositeOdtPlan, mode: str) -> None:
    plan.ring.forward_mode = mode
    plan.ring.adjoint_mode = mode
    if plan.axis is not None:
        plan.axis.forward_mode = mode
        plan.axis.adjoint_mode = mode


def main() -> None:
    parser = base_parser()
    parser.description = "Validate illumination-reduced ODT GPU contractions against legacy streaming."
    parser.set_defaults(
        forward_mode="auto",
        adjoint_mode="auto",
        out=ROOT / "benchmark_results" / "odt_torch_reduced_contraction_validation.json",
        summary_md=None,
    )
    args = parser.parse_args()
    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA device required")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    setup_start = time.perf_counter()
    context = build_composite_context(args)
    plan = TorchCompositeOdtPlan.from_context(
        context,
        torch=torch,
        device=device,
        dtype=args.dtype,
        low_memory_adjoint=args.low_memory_adjoint,
        radial_block_size=args.radial_block_size,
        illumination_block_size=args.illumination_block_size,
        forward_mode="legacy",
        adjoint_mode="legacy",
    )
    _, _, np_complex, _ = torch_dtypes(torch, args.dtype)
    coeff_np = np.ascontiguousarray(context.ring.obj.coeff.astype(np_complex, copy=False))
    if args.real_object:
        coeff_np = np.ascontiguousarray(np.real(coeff_np).astype(np_complex))
    coeff = torch.as_tensor(coeff_np, dtype=plan.complex_dtype, device=device)
    synchronize(torch, device)
    setup_s = time.perf_counter() - setup_start

    with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
        set_modes(plan, "legacy")
        legacy_forward, legacy_forward_s, legacy_forward_times = timed_cuda(
            torch,
            device,
            lambda: plan.forward(coeff),
            repeats=args.repeats,
            warmups=args.warmups,
        )
        residual = legacy_forward * (0.1 + 0.2j)
        legacy_adjoint, legacy_adjoint_s, legacy_adjoint_times = timed_cuda(
            torch,
            device,
            lambda: plan.adjoint(residual),
            repeats=args.repeats,
            warmups=args.warmups,
        )

        set_modes(plan, "auto")
        optimized_forward, optimized_forward_s, optimized_forward_times = timed_cuda(
            torch,
            device,
            lambda: plan.forward(coeff),
            repeats=args.repeats,
            warmups=args.warmups,
        )
        optimized_adjoint, optimized_adjoint_s, optimized_adjoint_times = timed_cuda(
            torch,
            device,
            lambda: plan.adjoint(residual),
            repeats=args.repeats,
            warmups=args.warmups,
        )

    forward_rel = rel_l2(torch, optimized_forward, legacy_forward)
    adjoint_rel = rel_l2(torch, optimized_adjoint, legacy_adjoint)
    legacy_dot = dot_error(torch, coeff, legacy_forward, residual, legacy_adjoint)
    optimized_dot = dot_error(torch, coeff, optimized_forward, residual, optimized_adjoint)
    tolerance = 2e-6 if args.dtype == "complex64" else 1e-11
    result = {
        "schema": "odt-torch-reduced-contraction-validation-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "device_name": device_name(torch, device),
        "config": vars(args),
        "object_bins": int(coeff.numel()),
        "q_samples": int(plan.q_count),
        "setup_s": float(setup_s),
        "resolved_modes": {
            "ring_forward": plan.ring.resolved_forward_mode,
            "ring_adjoint": plan.ring.resolved_adjoint_mode,
            "axis_forward": None if plan.axis is None else plan.axis.resolved_forward_mode,
            "axis_adjoint": None if plan.axis is None else plan.axis.resolved_adjoint_mode,
        },
        "accuracy": {
            "optimized_forward_rel_l2_vs_legacy": forward_rel,
            "optimized_adjoint_rel_l2_vs_legacy": adjoint_rel,
            "legacy_dot_error": legacy_dot,
            "optimized_dot_error": optimized_dot,
            "relative_l2_tolerance": tolerance,
        },
        "timing": {
            "legacy_forward_median_s": legacy_forward_s,
            "legacy_adjoint_median_s": legacy_adjoint_s,
            "optimized_forward_median_s": optimized_forward_s,
            "optimized_adjoint_median_s": optimized_adjoint_s,
            "forward_speedup": legacy_forward_s / optimized_forward_s,
            "adjoint_speedup": legacy_adjoint_s / optimized_adjoint_s,
            "legacy_forward_times_s": legacy_forward_times,
            "legacy_adjoint_times_s": legacy_adjoint_times,
            "optimized_forward_times_s": optimized_forward_times,
            "optimized_adjoint_times_s": optimized_adjoint_times,
        },
        "gpu_peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        "gpu_peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / 1024**2),
    }
    result["passed"] = bool(
        forward_rel <= tolerance
        and adjoint_rel <= tolerance
        and legacy_dot <= 1e-6
        and optimized_dot <= 1e-6
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
