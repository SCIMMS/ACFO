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

from benchmark_odt_cufinufft_gpu_baseline import (  # noqa: E402
    import_cufinufft_modules,
    parser as baseline_parser,
)
from benchmark_odt_realistic_geometry_reconstruction import (  # noqa: E402
    build_composite_context,
)


def memory_snapshot(cp: Any, label: str) -> dict[str, Any]:
    free, total = cp.cuda.runtime.memGetInfo()
    return {
        "label": label,
        "free_mib": float(free / 1024**2),
        "total_mib": float(total / 1024**2),
        "used_mib": float((total - free) / 1024**2),
        "cupy_pool_used_mib": float(
            cp.get_default_memory_pool().used_bytes() / 1024**2
        ),
        "cupy_pool_total_mib": float(
            cp.get_default_memory_pool().total_bytes() / 1024**2
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    context_start = time.perf_counter()
    context = build_composite_context(args)
    context_s = time.perf_counter() - context_start
    cp, cufinufft = import_cufinufft_modules()
    snapshots = [memory_snapshot(cp, "initial_after_cpu_context")]
    stages: list[dict[str, Any]] = []
    forward_plan = None
    arrays: list[Any] = []
    try:
        obj = context.ring.obj
        q = context.ring.flat_q
        allocation_start = time.perf_counter()
        arrays = [
            cp.asarray(np.asarray(obj.x, dtype=np.float64)),
            cp.asarray(np.asarray(obj.y, dtype=np.float64)),
            cp.asarray(np.asarray(obj.z, dtype=np.float64)),
            cp.asarray(np.asarray(q.qx, dtype=np.float64)),
            cp.asarray(np.asarray(q.qy, dtype=np.float64)),
            cp.asarray(np.asarray(q.qz, dtype=np.float64)),
        ]
        cp.cuda.runtime.deviceSynchronize()
        stages.append(
            {
                "stage": "copy_forward_points",
                "status": "ok",
                "elapsed_s": float(time.perf_counter() - allocation_start),
            }
        )
        snapshots.append(memory_snapshot(cp, "after_forward_point_arrays"))

        plan_start = time.perf_counter()
        forward_plan = cufinufft.Plan(
            3, 3, eps=args.cufinufft_eps, isign=1, dtype="complex128"
        )
        cp.cuda.runtime.deviceSynchronize()
        stages.append(
            {
                "stage": "make_forward_plan",
                "status": "ok",
                "elapsed_s": float(time.perf_counter() - plan_start),
            }
        )
        snapshots.append(memory_snapshot(cp, "after_forward_plan_constructor"))

        setpts_start = time.perf_counter()
        forward_plan.setpts(*arrays)
        cp.cuda.runtime.deviceSynchronize()
        stages.append(
            {
                "stage": "forward_setpts",
                "status": "ok",
                "elapsed_s": float(time.perf_counter() - setpts_start),
            }
        )
        snapshots.append(memory_snapshot(cp, "after_forward_setpts"))
    except Exception as exc:
        stages.append(
            {
                "stage": (
                    "forward_setpts"
                    if forward_plan is not None
                    else "copy_or_make_forward_plan"
                ),
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        try:
            snapshots.append(memory_snapshot(cp, "at_failure"))
        except Exception:
            pass

    success = bool(stages and stages[-1]["status"] == "ok")
    result = {
        "schema": "odt-cufinufft-c128-full-plan-diagnostic-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "problem": {
            "object_shape": list(context.ring.obj.coeff.shape),
            "ring_q_samples": int(context.ring.flat_q.count),
            "axis_q_samples": (
                0 if context.axis is None else int(context.axis.flat_q.count)
            ),
            "dtype": "complex128",
            "eps": float(args.cufinufft_eps),
            "acfo_plan_present": False,
        },
        "context_build_s": float(context_s),
        "stages": stages,
        "memory_snapshots": snapshots,
        "forward_setpts_succeeded_without_acfo_plan": success,
        "interpretation": (
            "If this succeeds while the mixed-resident benchmark fails at the same "
            "forward setpts call, the 8 GB failure is co-residency pressure rather "
            "than an invalid complex128 cuFINUFFT geometry."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    del forward_plan, arrays
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return result


def parser() -> argparse.ArgumentParser:
    p = baseline_parser()
    p.description = (
        "Diagnose whether the production complex128 cuFINUFFT forward plan can "
        "set points without a co-resident ACFO GPU plan."
    )
    p.set_defaults(
        dtype="complex64",
        cufinufft_dtype="complex128",
        cufinufft_eps=1e-7,
        n_beta=256,
        n_r=256,
        n_z=256,
        ring_illum=120,
        cap_radial=256,
        cap_phi=256,
        h_cutoff=28,
        cpp_threads=4,
        skip_native_prepared_adjoint=True,
        compact_axisymmetric_kernel=True,
        out=ROOT
        / "benchmark_results"
        / "odt_cufinufft_c128_full_plan_diagnostic.json",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
