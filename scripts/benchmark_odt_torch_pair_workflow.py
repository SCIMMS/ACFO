from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_high_na_torch_gpu import device_name, import_torch, resolve_device, synchronize  # noqa: E402
from benchmark_odt_realistic_geometry_reconstruction import build_composite_context  # noqa: E402
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


def main() -> None:
    parser = base_parser()
    parser.description = "Standalone ACFO ODT forward-adjoint pair workflow."
    parser.set_defaults(
        out=ROOT / "benchmark_results" / "odt_torch_256cubed_100pair.json",
        csv=ROOT / "benchmark_results" / "odt_torch_256cubed_100pair.csv",
        summary_md=None,
    )
    args = parser.parse_args()
    if args.repeats <= 0 or args.warmups < 0:
        raise ValueError("repeats must be positive and warmups nonnegative")

    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA device required")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    setup_start = time.perf_counter()
    ctx = build_composite_context(args)
    plan = TorchCompositeOdtPlan.from_context(
        ctx,
        torch=torch,
        device=device,
        dtype=args.dtype,
        low_memory_adjoint=args.low_memory_adjoint,
        radial_block_size=args.radial_block_size,
        illumination_block_size=args.illumination_block_size,
        forward_mode=args.forward_mode,
        adjoint_mode=args.adjoint_mode,
    )
    _, _, np_complex, _ = torch_dtypes(torch, args.dtype)
    coeff_np = np.ascontiguousarray(ctx.ring.obj.coeff.astype(np_complex, copy=False))
    if args.real_object:
        coeff_np = np.ascontiguousarray(np.real(coeff_np).astype(np_complex))
    coeff = torch.as_tensor(coeff_np, dtype=plan.complex_dtype, device=device)
    synchronize(torch, device)
    setup_s = time.perf_counter() - setup_start

    with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
        first_start = time.perf_counter()
        first_forward = plan.forward(coeff)
        residual = first_forward * (0.1 + 0.2j)
        first_adjoint = plan.adjoint(residual)
        synchronize(torch, device)
        first_pair_s = time.perf_counter() - first_start

        dot_lhs = torch.vdot(first_forward.reshape(-1), residual.reshape(-1))
        dot_rhs = torch.vdot(coeff.reshape(-1), first_adjoint.reshape(-1))
        dot_error = float(
            (torch.abs(dot_lhs - dot_rhs) / torch.clamp(torch.abs(dot_lhs) + torch.abs(dot_rhs), min=1e-30))
            .detach()
            .cpu()
            .item()
        )

        def pair_call():
            forward = plan.forward(coeff)
            adjoint = plan.adjoint(residual)
            return forward, adjoint

        for index in range(args.warmups):
            pair_call()
            synchronize(torch, device)
            if (index + 1) % 5 == 0 or index + 1 == args.warmups:
                print(f"warmup: {index + 1}/{args.warmups}", flush=True)

        times: list[float] = []
        workflow_start = time.perf_counter()
        last = None
        for index in range(args.repeats):
            synchronize(torch, device)
            start = time.perf_counter()
            last = pair_call()
            synchronize(torch, device)
            times.append(time.perf_counter() - start)
            if (index + 1) % 10 == 0 or index + 1 == args.repeats:
                print(f"measured pair: {index + 1}/{args.repeats}", flush=True)
        measured_wall_s = time.perf_counter() - workflow_start
        assert last is not None

    result = {
        "schema": "odt-torch-pair-workflow-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "ACFO compact axisymmetric streaming PyTorch/C++",
        "device": str(device),
        "device_name": device_name(torch, device),
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "config": vars(args),
        "object_bins": int(coeff.numel()),
        "q_samples": int(plan.q_count),
        "total_illumination_count": int(args.ring_illum) + (0 if args.skip_axis_illumination else 1),
        "ring_forward_mode_resolved": plan.ring.resolved_forward_mode,
        "axis_forward_mode_resolved": (
            None if plan.axis is None else plan.axis.resolved_forward_mode
        ),
        "ring_adjoint_mode_resolved": plan.ring.resolved_adjoint_mode,
        "axis_adjoint_mode_resolved": (
            None if plan.axis is None else plan.axis.resolved_adjoint_mode
        ),
        "setup_s": setup_s,
        "first_forward_adjoint_pair_s": first_pair_s,
        "forward_adjoint_dot_error_complex64": dot_error,
        "pair_timing": timing_summary(times),
        "measured_workflow_wall_s": measured_wall_s,
        "gpu_peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        "gpu_peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / 1024**2),
        "passed": bool(np.all(np.isfinite(times)) and dot_error <= 1e-6),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
