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
from benchmark_odt_realistic_geometry_reconstruction import (  # noqa: E402
    build_composite_context,
)
from benchmark_odt_torch_gpu_reconstruction import (  # noqa: E402
    TorchCompositeOdtPlan,
    parser as base_parser,
    torch_dtypes,
)


def rel_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference.ravel())), np.finfo(np.float64).tiny)
    return float(np.linalg.norm((candidate - reference).ravel()) / denominator)


def dot_error(
    coeff: np.ndarray,
    forward: np.ndarray,
    residual: np.ndarray,
    adjoint: np.ndarray,
) -> float:
    lhs = np.vdot(forward.astype(np.complex128, copy=False).ravel(), residual.astype(np.complex128, copy=False).ravel())
    rhs = np.vdot(coeff.astype(np.complex128, copy=False).ravel(), adjoint.astype(np.complex128, copy=False).ravel())
    denominator = max(float(abs(lhs) + abs(rhs)), np.finfo(np.float64).tiny)
    return float(abs(lhs - rhs) / denominator)


def build_plan(
    context: Any,
    *,
    torch: Any,
    device: Any,
    dtype: str,
    low_memory: bool,
    radial_block_size: int,
    illumination_block_size: int,
) -> TorchCompositeOdtPlan:
    return TorchCompositeOdtPlan.from_context(
        context,
        torch=torch,
        device=device,
        dtype=dtype,
        low_memory_adjoint=low_memory,
        radial_block_size=radial_block_size,
        illumination_block_size=illumination_block_size,
        forward_mode="auto",
        adjoint_mode="auto",
    )


def main() -> None:
    parser = base_parser()
    parser.description = "Compare resident and stream-reduced ODT paths on the same data."
    parser.add_argument("--stream-radial-block-size", type=int, default=16)
    parser.add_argument("--stream-illumination-block-size", type=int, default=4)
    parser.set_defaults(
        compact_axisymmetric_kernel=True,
        skip_native_prepared_adjoint=True,
        real_object=True,
        forward_mode="auto",
        adjoint_mode="auto",
        out=ROOT / "benchmark_results" / "odt_resident_stream_equivalence.json",
        summary_md=None,
    )
    args = parser.parse_args()

    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA device required")

    context = build_composite_context(args)
    _, _, np_complex, _ = torch_dtypes(torch, args.dtype)
    coeff_np = np.ascontiguousarray(context.ring.obj.coeff.astype(np_complex, copy=False))
    if args.real_object:
        coeff_np = np.ascontiguousarray(np.real(coeff_np).astype(np_complex))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    resident_setup_start = time.perf_counter()
    resident_plan = build_plan(
        context,
        torch=torch,
        device=device,
        dtype=args.dtype,
        low_memory=False,
        radial_block_size=0,
        illumination_block_size=0,
    )
    resident_coeff = torch.as_tensor(coeff_np, dtype=resident_plan.complex_dtype, device=device)
    synchronize(torch, device)
    resident_setup_s = time.perf_counter() - resident_setup_start
    with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
        resident_forward = resident_plan.forward(resident_coeff)
        resident_residual = resident_forward * (0.1 + 0.2j)
        resident_adjoint = resident_plan.adjoint(resident_residual)
        synchronize(torch, device)
        resident_forward_np = np.ascontiguousarray(resident_forward.detach().cpu().numpy())
        residual_np = np.ascontiguousarray(resident_residual.detach().cpu().numpy())
        resident_adjoint_np = np.ascontiguousarray(resident_adjoint.detach().cpu().numpy())
    resident_peak_allocated_mib = float(torch.cuda.max_memory_allocated(device) / 1024**2)
    resident_peak_reserved_mib = float(torch.cuda.max_memory_reserved(device) / 1024**2)
    resident_modes = {
        "ring_forward": resident_plan.ring.resolved_forward_mode,
        "ring_adjoint": resident_plan.ring.resolved_adjoint_mode,
        "axis_forward": None if resident_plan.axis is None else resident_plan.axis.resolved_forward_mode,
        "axis_adjoint": None if resident_plan.axis is None else resident_plan.axis.resolved_adjoint_mode,
    }

    del resident_adjoint, resident_residual, resident_forward, resident_coeff, resident_plan
    gc.collect()
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(device)
    stream_setup_start = time.perf_counter()
    stream_plan = build_plan(
        context,
        torch=torch,
        device=device,
        dtype=args.dtype,
        low_memory=True,
        radial_block_size=args.stream_radial_block_size,
        illumination_block_size=args.stream_illumination_block_size,
    )
    stream_coeff = torch.as_tensor(coeff_np, dtype=stream_plan.complex_dtype, device=device)
    stream_residual = torch.as_tensor(residual_np, dtype=stream_plan.complex_dtype, device=device)
    synchronize(torch, device)
    stream_setup_s = time.perf_counter() - stream_setup_start
    with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
        stream_forward = stream_plan.forward(stream_coeff)
        stream_adjoint = stream_plan.adjoint(stream_residual)
        synchronize(torch, device)
        stream_forward_np = np.ascontiguousarray(stream_forward.detach().cpu().numpy())
        stream_adjoint_np = np.ascontiguousarray(stream_adjoint.detach().cpu().numpy())
    stream_peak_allocated_mib = float(torch.cuda.max_memory_allocated(device) / 1024**2)
    stream_peak_reserved_mib = float(torch.cuda.max_memory_reserved(device) / 1024**2)
    stream_modes = {
        "ring_forward": stream_plan.ring.resolved_forward_mode,
        "ring_adjoint": stream_plan.ring.resolved_adjoint_mode,
        "axis_forward": None if stream_plan.axis is None else stream_plan.axis.resolved_forward_mode,
        "axis_adjoint": None if stream_plan.axis is None else stream_plan.axis.resolved_adjoint_mode,
    }

    tolerance = 2e-6 if args.dtype == "complex64" else 1e-11
    forward_rel = rel_l2(stream_forward_np, resident_forward_np)
    adjoint_rel = rel_l2(stream_adjoint_np, resident_adjoint_np)
    resident_dot = dot_error(coeff_np, resident_forward_np, residual_np, resident_adjoint_np)
    stream_dot = dot_error(coeff_np, stream_forward_np, residual_np, stream_adjoint_np)
    result = {
        "schema": "odt-resident-stream-equivalence-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "device_name": device_name(torch, device),
        "dtype": args.dtype,
        "problem": {
            "n_r": args.n_r,
            "n_z": args.n_z,
            "n_beta": args.n_beta,
            "object_bins": int(coeff_np.size),
            "ring_illum": args.ring_illum,
            "axis_included": not args.skip_axis_illumination,
            "cap_radial": args.cap_radial,
            "cap_phi": args.cap_phi,
            "q_samples": int(stream_plan.q_count),
        },
        "resident": {
            "setup_s": resident_setup_s,
            "modes": resident_modes,
            "peak_allocated_mib": resident_peak_allocated_mib,
            "peak_reserved_mib": resident_peak_reserved_mib,
        },
        "stream": {
            "setup_s": stream_setup_s,
            "radial_block_size": args.stream_radial_block_size,
            "illumination_block_size": args.stream_illumination_block_size,
            "modes": stream_modes,
            "peak_allocated_mib": stream_peak_allocated_mib,
            "peak_reserved_mib": stream_peak_reserved_mib,
        },
        "accuracy": {
            "stream_forward_rel_l2_vs_resident": forward_rel,
            "stream_adjoint_rel_l2_vs_resident": adjoint_rel,
            "resident_dot_error_complex128_accum": resident_dot,
            "stream_dot_error_complex128_accum": stream_dot,
            "relative_l2_tolerance": tolerance,
            "dot_tolerance": 1e-6,
        },
        "passed": bool(
            forward_rel <= tolerance
            and adjoint_rel <= tolerance
            and resident_dot <= 1e-6
            and stream_dot <= 1e-6
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
