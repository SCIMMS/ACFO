from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_high_na_torch_gpu import device_name, import_torch, resolve_device  # noqa: E402
from benchmark_odt_cufinufft_gpu_baseline import (  # noqa: E402
    cufinufft_adjoint,
    cufinufft_forward,
    make_cufinufft_composite,
    parser as base_parser,
)
from benchmark_odt_realistic_geometry_reconstruction import build_composite_context  # noqa: E402
from benchmark_odt_torch_gpu_reconstruction import torch_dtypes  # noqa: E402


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
    parser.description = "Standalone reusable-plan cuFINUFFT ODT forward-adjoint pair workflow."
    parser.set_defaults(
        cufinufft_plan_mode="plan",
        out=ROOT / "benchmark_results" / "odt_cufinufft_256cubed_100pair.json",
        csv=ROOT / "benchmark_results" / "odt_cufinufft_256cubed_100pair.csv",
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

    setup_start = time.perf_counter()
    ctx = build_composite_context(args)
    op = make_cufinufft_composite(
        ctx,
        dtype=args.dtype,
        plan_mode=args.cufinufft_plan_mode,
        eps=args.cufinufft_eps,
    )
    cp = op.cp
    _, _, np_complex, _ = torch_dtypes(torch, args.dtype)
    coeff_np = np.ascontiguousarray(ctx.ring.obj.coeff.astype(np_complex, copy=False))
    coeff = cp.asarray(coeff_np.ravel())
    cp.cuda.get_current_stream().synchronize()
    setup_s = time.perf_counter() - setup_start

    first_start = time.perf_counter()
    first_forward = cufinufft_forward(op, coeff, eps=args.cufinufft_eps)
    residual = [part * complex(args.residual_scale_real, args.residual_scale_imag) for part in first_forward]
    first_adjoint = cufinufft_adjoint(op, residual, eps=args.cufinufft_eps)
    cp.cuda.get_current_stream().synchronize()
    first_pair_s = time.perf_counter() - first_start

    def pair_call():
        forward = cufinufft_forward(op, coeff, eps=args.cufinufft_eps)
        adjoint = cufinufft_adjoint(op, residual, eps=args.cufinufft_eps)
        return forward, adjoint

    for index in range(args.warmups):
        pair_call()
        cp.cuda.get_current_stream().synchronize()
        if (index + 1) % 5 == 0 or index + 1 == args.warmups:
            print(f"warmup: {index + 1}/{args.warmups}", flush=True)

    times: list[float] = []
    workflow_start = time.perf_counter()
    last = None
    for index in range(args.repeats):
        cp.cuda.get_current_stream().synchronize()
        start = time.perf_counter()
        last = pair_call()
        cp.cuda.get_current_stream().synchronize()
        times.append(time.perf_counter() - start)
        if (index + 1) % 10 == 0 or index + 1 == args.repeats:
            print(f"measured pair: {index + 1}/{args.repeats}", flush=True)
    measured_wall_s = time.perf_counter() - workflow_start
    assert last is not None and first_adjoint

    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    result = {
        "schema": "odt-cufinufft-pair-workflow-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "cuFINUFFT reusable type-3 plans",
        "device": str(device),
        "device_name": device_name(torch, device),
        "cupy_version": getattr(cp, "__version__", None),
        "cufinufft_version": getattr(op.cufinufft, "__version__", None),
        "config": vars(args),
        "object_bins": int(coeff.size),
        "q_samples": int(op.ring.q_count + (0 if op.axis is None else op.axis.q_count)),
        "total_illumination_count": int(args.ring_illum) + (0 if args.skip_axis_illumination else 1),
        "setup_s": setup_s,
        "first_forward_adjoint_pair_s": first_pair_s,
        "pair_timing": timing_summary(times),
        "measured_workflow_wall_s": measured_wall_s,
        "cupy_pool_used_mib": float(pool.used_bytes() / 1024**2),
        "cupy_pool_total_mib": float(pool.total_bytes() / 1024**2),
        "gpu_free_mib_after": float(free_bytes / 1024**2),
        "gpu_total_mib": float(total_bytes / 1024**2),
        "passed": bool(np.all(np.isfinite(times))),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
