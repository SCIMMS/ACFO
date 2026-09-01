from __future__ import annotations

import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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
from benchmark_odt_realistic_geometry_reconstruction import build_composite_context  # noqa: E402
from benchmark_odt_torch_gpu_reconstruction import (  # noqa: E402
    TorchCompositeOdtPlan,
    parser as base_parser,
    torch_dtypes,
)


def main() -> None:
    parser = base_parser()
    parser.description = "Probe whether an unblocked resident ODT plan fits on the current GPU."
    parser.set_defaults(
        compact_axisymmetric_kernel=True,
        skip_native_prepared_adjoint=True,
        real_object=True,
        forward_mode="auto",
        adjoint_mode="auto",
        out=ROOT / "benchmark_results" / "odt_resident_memory_gate.json",
        summary_md=None,
    )
    args = parser.parse_args()

    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA device required")
    properties = torch.cuda.get_device_properties(device)
    total_vram_mib = float(properties.total_memory / 1024**2)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    status = "not-run"
    error_type = None
    error_message = None
    pair_s = None
    object_bins = int(args.n_r * args.n_z * args.n_beta)
    q_samples = int(
        (args.ring_illum + (0 if args.skip_axis_illumination else 1))
        * args.cap_radial
        * args.cap_phi
    )
    try:
        context = build_composite_context(args)
        plan = TorchCompositeOdtPlan.from_context(
            context,
            torch=torch,
            device=device,
            dtype=args.dtype,
            low_memory_adjoint=False,
            radial_block_size=0,
            illumination_block_size=0,
            forward_mode="auto",
            adjoint_mode="auto",
        )
        _, _, np_complex, _ = torch_dtypes(torch, args.dtype)
        coeff_np = np.ascontiguousarray(context.ring.obj.coeff.astype(np_complex, copy=False))
        if args.real_object:
            coeff_np = np.ascontiguousarray(np.real(coeff_np).astype(np_complex))
        coeff = torch.as_tensor(coeff_np, dtype=plan.complex_dtype, device=device)
        synchronize(torch, device)
        pair_start = time.perf_counter()
        with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
            forward = plan.forward(coeff)
            residual = forward * (0.1 + 0.2j)
            _ = plan.adjoint(residual)
            synchronize(torch, device)
        pair_s = float(time.perf_counter() - pair_start)
        status = "fit"
    except torch.OutOfMemoryError as exc:
        status = "oom"
        error_type = type(exc).__name__
        error_message = str(exc)
    finally:
        elapsed_s = float(time.perf_counter() - start)
        peak_allocated_mib = float(torch.cuda.max_memory_allocated(device) / 1024**2)
        peak_reserved_mib = float(torch.cuda.max_memory_reserved(device) / 1024**2)
        gc.collect()
        torch.cuda.empty_cache()

    result = {
        "schema": "odt-resident-memory-gate-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "device_name": device_name(torch, device),
        "total_vram_mib": total_vram_mib,
        "dtype": args.dtype,
        "problem": {
            "n_r": args.n_r,
            "n_z": args.n_z,
            "n_beta": args.n_beta,
            "object_bins": object_bins,
            "ring_illum": args.ring_illum,
            "axis_included": not args.skip_axis_illumination,
            "cap_radial": args.cap_radial,
            "cap_phi": args.cap_phi,
            "q_samples": q_samples,
        },
        "resident_configuration": {
            "low_memory_adjoint": False,
            "radial_block_size": 0,
            "illumination_block_size": 0,
            "forward_mode": "auto",
            "adjoint_mode": "auto",
        },
        "status": status,
        "resident_feasible": status == "fit",
        "elapsed_s": elapsed_s,
        "pair_s": pair_s,
        "gpu_peak_allocated_mib": peak_allocated_mib,
        "gpu_peak_reserved_mib": peak_reserved_mib,
        "error_type": error_type,
        "error_message": error_message,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
