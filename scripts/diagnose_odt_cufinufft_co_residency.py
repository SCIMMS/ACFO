from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

from benchmark_odt_cufinufft_gpu_baseline import (
    cufinufft_adjoint,
    cufinufft_forward,
    make_cufinufft_composite,
)
from benchmark_odt_realistic_geometry_reconstruction import (
    build_composite_context,
    split_residual,
)
from benchmark_odt_torch_gpu_reconstruction import (
    TorchCompositeOdtPlan,
    import_torch,
    parser,
    resolve_device,
    timed_cuda,
    to_numpy,
    torch_dtypes,
)


def main() -> None:
    args = parser().parse_args([])
    args.device = "cuda"
    args.dtype = "complex64"
    args.n_beta = 256
    args.n_r = 256
    args.n_z = 256
    args.ring_illum = 120
    args.cap_radial = 256
    args.cap_phi = 256
    args.h_cutoff = 28
    args.l_margin = 18
    args.cone_l_prune_threshold = 1e-12
    args.cpp_threads = 4
    args.skip_native_prepared_adjoint = True
    args.compact_axisymmetric_kernel = True

    torch = import_torch()
    device = resolve_device(torch, args.device)
    context = build_composite_context(args)
    plan = TorchCompositeOdtPlan.from_context(
        context,
        torch=torch,
        device=device,
        dtype=args.dtype,
        low_memory_adjoint=True,
        radial_block_size=32,
        illumination_block_size=4,
        forward_mode="auto",
        adjoint_mode="auto",
        prune_axis_l0=True,
        axial_lowrank_rank=16,
        ring_adaptive_l_packed_threshold=1e-6,
    )
    _, _, np_complex, _ = torch_dtypes(torch, args.dtype)
    coeff_np = np.ascontiguousarray(
        context.ring.obj.coeff.astype(np_complex, copy=False)
    )
    coeff = torch.as_tensor(coeff_np, dtype=plan.complex_dtype, device=device)
    with torch.inference_mode():
        data = plan.forward(coeff)
        residual = data * (0.1 + 0.2j)

        def ours_pair():
            return plan.forward(coeff), plan.adjoint(residual)

        _, before_s, before_times = timed_cuda(
            torch, device, ours_pair, repeats=5, warmups=3
        )

        setup_start = time.perf_counter()
        cu_op = make_cufinufft_composite(
            context, dtype=args.dtype, plan_mode="plan", eps=1e-6
        )
        cu_op.cp.cuda.runtime.deviceSynchronize()
        setup_s = time.perf_counter() - setup_start
        _, after_plan_s, after_plan_times = timed_cuda(
            torch, device, ours_pair, repeats=5, warmups=3
        )

        residual_np = to_numpy(torch, device, residual).astype(np_complex, copy=False)
        ring_residual, axis_residual = split_residual(context, residual_np)
        coeff_cu = cu_op.cp.asarray(coeff_np.reshape(-1))
        residual_cu = [cu_op.cp.asarray(ring_residual)]
        if axis_residual is not None:
            residual_cu.append(cu_op.cp.asarray(axis_residual))
        cufinufft_forward(cu_op, coeff_cu, eps=1e-6)
        cufinufft_adjoint(cu_op, residual_cu, eps=1e-6)
        cu_op.cp.cuda.runtime.deviceSynchronize()
        _, after_execute_s, after_execute_times = timed_cuda(
            torch, device, ours_pair, repeats=5, warmups=3
        )

        cp = cu_op.cp
        del coeff_cu, residual_cu, cu_op
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.runtime.deviceSynchronize()
        _, after_release_s, after_release_times = timed_cuda(
            torch, device, ours_pair, repeats=5, warmups=3
        )

    payload = {
        "before_cufinufft_plan": {"median_s": before_s, "times_s": before_times},
        "cufinufft_setup_and_sync_s": setup_s,
        "after_cufinufft_plan": {
            "median_s": after_plan_s,
            "times_s": after_plan_times,
        },
        "after_one_cufinufft_pair": {
            "median_s": after_execute_s,
            "times_s": after_execute_times,
        },
        "after_cufinufft_release": {
            "median_s": after_release_s,
            "times_s": after_release_times,
        },
    }
    out = ROOT / "benchmark_results" / "odt_cufinufft_co_residency_diagnostic.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
