from __future__ import annotations

import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

from benchmark_odt_adaptive_l_packed_sweep import rel_l2, timed_pair
from benchmark_odt_realistic_geometry_reconstruction import build_composite_context
from benchmark_odt_torch_gpu_reconstruction import (
    TorchCompositeOdtPlan,
    device_name,
    import_torch,
    parser,
    resolve_device,
    torch_dtypes,
)


def make_plan(
    context: Any,
    *,
    torch: Any,
    device: Any,
    args: Any,
    rank: int,
    adaptive_threshold: float,
) -> TorchCompositeOdtPlan:
    return TorchCompositeOdtPlan.from_context(
        context,
        torch=torch,
        device=device,
        dtype=args.dtype,
        low_memory_adjoint=True,
        radial_block_size=args.radial_block_size,
        illumination_block_size=args.illumination_block_size,
        forward_mode="auto",
        adjoint_mode="auto",
        prune_axis_l0=True,
        axial_lowrank_rank=rank,
        ring_adaptive_l_packed_threshold=adaptive_threshold,
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
    args.l_margin = 18
    args.cone_l_prune_threshold = 1e-12
    args.cpp_threads = 4
    args.skip_native_prepared_adjoint = True
    args.compact_axisymmetric_kernel = True
    args.radial_block_size = 32
    args.illumination_block_size = 4
    reference_h = 36
    candidate_h = 28
    candidate_rank = 16
    adaptive_threshold = 1e-6
    tolerance = 2e-6

    torch = import_torch()
    device = resolve_device(torch, args.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    reference_args = copy.deepcopy(args)
    reference_args.h_cutoff = reference_h
    candidate_args = copy.deepcopy(args)
    candidate_args.h_cutoff = candidate_h
    start = time.perf_counter()
    reference_context = build_composite_context(reference_args)
    reference_context_build_s = time.perf_counter() - start
    start = time.perf_counter()
    candidate_context = build_composite_context(candidate_args)
    candidate_context_build_s = time.perf_counter() - start
    start = time.perf_counter()
    reference_plan = make_plan(
        reference_context,
        torch=torch,
        device=device,
        args=args,
        rank=0,
        adaptive_threshold=0.0,
    )
    reference_plan_build_s = time.perf_counter() - start
    start = time.perf_counter()
    candidate_plan = make_plan(
        candidate_context,
        torch=torch,
        device=device,
        args=args,
        rank=candidate_rank,
        adaptive_threshold=adaptive_threshold,
    )
    candidate_plan_build_s = time.perf_counter() - start
    if reference_plan.q_count != candidate_plan.q_count:
        raise RuntimeError("reference and candidate q counts differ")

    complex_dtype, real_dtype, np_complex, _ = torch_dtypes(torch, args.dtype)
    coeff_np = np.ascontiguousarray(
        reference_context.ring.obj.coeff.astype(np_complex, copy=False)
    )
    coeff = torch.as_tensor(coeff_np, dtype=complex_dtype, device=device)
    generator = torch.Generator(device=device).manual_seed(20260713)
    stress_coeff = torch.complex(
        torch.randn(coeff.shape, generator=generator, device=device, dtype=real_dtype),
        torch.randn(coeff.shape, generator=generator, device=device, dtype=real_dtype),
    )
    stress_residual = torch.complex(
        torch.randn(
            reference_plan.q_count,
            generator=generator,
            device=device,
            dtype=real_dtype,
        ),
        torch.randn(
            reference_plan.q_count,
            generator=generator,
            device=device,
            dtype=real_dtype,
        ),
    )

    with torch.inference_mode():
        reference_forward = reference_plan.forward(coeff)
        physical_residual = reference_forward * (0.1 + 0.2j)
        reference_adjoint = reference_plan.adjoint(physical_residual)
        reference_stress_forward = reference_plan.forward(stress_coeff)
        reference_stress_adjoint = reference_plan.adjoint(stress_residual)
        candidate_forward = candidate_plan.forward(coeff)
        candidate_adjoint = candidate_plan.adjoint(physical_residual)
        candidate_stress_forward = candidate_plan.forward(stress_coeff)
        candidate_stress_adjoint = candidate_plan.adjoint(stress_residual)

        errors = {
            "physical_forward_rel_l2": rel_l2(
                torch, candidate_forward, reference_forward
            ),
            "physical_adjoint_rel_l2": rel_l2(
                torch, candidate_adjoint, reference_adjoint
            ),
            "stress_forward_rel_l2": rel_l2(
                torch, candidate_stress_forward, reference_stress_forward
            ),
            "stress_adjoint_rel_l2": rel_l2(
                torch, candidate_stress_adjoint, reference_stress_adjoint
            ),
        }
        worst_error = max(errors.values())
        lhs = torch.vdot(
            candidate_stress_forward.reshape(-1), stress_residual.reshape(-1)
        )
        rhs = torch.vdot(
            stress_coeff.reshape(-1), candidate_stress_adjoint.reshape(-1)
        )
        dot_error = float(
            (
                torch.abs(lhs - rhs)
                / torch.clamp(torch.abs(lhs) + torch.abs(rhs), min=1e-30)
            )
            .detach()
            .cpu()
            .item()
        )
        reference_pair_s, reference_times = timed_pair(
            torch,
            device,
            reference_plan,
            coeff,
            physical_residual,
            repeats=10,
            warmups=5,
        )
        candidate_pair_s, candidate_times = timed_pair(
            torch,
            device,
            candidate_plan,
            coeff,
            physical_residual,
            repeats=10,
            warmups=5,
        )

    payload = {
        "schema": "odt-final-packed-candidate-validation-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": device_name(torch, device),
        "problem": {
            "object_shape": list(coeff.shape),
            "q_samples": int(reference_plan.q_count),
            "illuminations": 121,
        },
        "reference": {
            "h_cutoff": reference_h,
            "axial_rank": 0,
            "adaptive_l_threshold": 0.0,
            "context_build_s": reference_context_build_s,
            "plan_build_s": reference_plan_build_s,
            "pair_median_s": reference_pair_s,
            "pair_times_s": reference_times,
        },
        "candidate": {
            "h_cutoff": candidate_h,
            "axial_rank": candidate_rank,
            "adaptive_l_threshold": adaptive_threshold,
            "adaptive_l_active_fraction": candidate_plan.ring.adaptive_l_active_fraction,
            "axis_l_modes": candidate_plan.axis.n_l,
            "context_build_s": candidate_context_build_s,
            "plan_build_s": candidate_plan_build_s,
            "pair_median_s": candidate_pair_s,
            "pair_times_s": candidate_times,
            "speedup_vs_reference": reference_pair_s / candidate_pair_s,
        },
        "errors": errors,
        "worst_rel_l2": worst_error,
        "dot_error": dot_error,
        "tolerance": tolerance,
        "passed": bool(worst_error <= tolerance and dot_error <= tolerance),
        "torch_peak_allocated_mib": float(
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        ),
    }
    out = ROOT / "benchmark_results" / "odt_final_packed_candidate_validation.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = ROOT / "benchmark_results" / "odt_final_packed_candidate_validation_ko.md"
    md.write_text(
        "\n".join(
            [
                "# ODT final packed candidate validation",
                "",
                "H=36 dense structured operator를 reference로 두고 H=28, axial rank=16, adaptive-L threshold=1e-6 후보를 직접 비교했다.",
                "",
                f"- worst physical/stress rel-L2: `{worst_error:.6g}`",
                f"- forward/adjoint dot error: `{dot_error:.6g}`",
                f"- tolerance: `{tolerance:.6g}`",
                f"- pass: `{payload['passed']}`",
                f"- reference pair median: `{1000.0 * reference_pair_s:.3f}` ms",
                f"- candidate pair median: `{1000.0 * candidate_pair_s:.3f}` ms",
                f"- speedup: `{reference_pair_s / candidate_pair_s:.3f}x`",
                f"- active ring (r,L) fraction: `{candidate_plan.ring.adaptive_l_active_fraction:.6f}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
