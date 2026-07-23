from __future__ import annotations

import argparse
import gc
import hashlib
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
    cufinufft_adjoint,
    cufinufft_forward,
    make_cufinufft_composite,
    parser as baseline_parser,
    synchronize_cupy,
    timed_cupy,
    timing_distribution,
)
from benchmark_odt_realistic_geometry_reconstruction import (  # noqa: E402
    build_composite_context,
)


DEFAULT_ACFO_REFERENCE = (
    ROOT
    / "benchmark_results"
    / "odt_h28_rank16_adaptive_l1e-6_vs_cufinufft_abba30_final.json"
)
DEFAULT_ACCURACY_AUDIT = (
    ROOT
    / "benchmark_results"
    / "odt_cufinufft_matched_error_direct_subset_c128.json"
)


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def memory_snapshot(cp: Any, label: str) -> dict[str, float | str]:
    free, total = cp.cuda.runtime.memGetInfo()
    return {
        "label": label,
        "free_mib": float(free / 1024**2),
        "used_mib": float((total - free) / 1024**2),
        "total_mib": float(total / 1024**2),
        "cupy_pool_used_mib": float(
            cp.get_default_memory_pool().used_bytes() / 1024**2
        ),
        "cupy_pool_total_mib": float(
            cp.get_default_memory_pool().total_bytes() / 1024**2
        ),
    }


def load_acfo_reference(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    pair_s = summary.get("ours_forward_adjoint_pair_s")
    if pair_s is None:
        raise ValueError("ACFO reference lacks ours_forward_adjoint_pair_s")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "pair_median_s": float(pair_s),
        "device_name": summary.get("device_name"),
        "dtype": summary.get("dtype"),
        "pair_protocol": summary.get("pair_timing_protocol"),
        "candidate": {
            "h_cutoff": payload.get("config", {}).get("h_cutoff"),
            "axial_lowrank_rank": payload.get("config", {}).get(
                "axial_lowrank_rank"
            ),
            "ring_adaptive_l_packed_threshold": payload.get("config", {}).get(
                "ring_adaptive_l_packed_threshold"
            ),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.cufinufft_dtype == "same":
        cufinufft_dtype = args.dtype
    else:
        cufinufft_dtype = args.cufinufft_dtype
    if args.pair_repeats <= 0 or args.pair_warmups < 0:
        raise ValueError("pair repeats/warmups must be positive/non-negative")

    context_start = time.perf_counter()
    context = build_composite_context(args)
    context_s = time.perf_counter() - context_start
    setup_start = time.perf_counter()
    op = make_cufinufft_composite(
        context,
        dtype=cufinufft_dtype,
        plan_mode="plan",
        eps=args.cufinufft_eps,
    )
    cp = op.cp
    synchronize_cupy(cp)
    setup_s = time.perf_counter() - setup_start
    snapshots = [memory_snapshot(cp, "after_all_forward_adjoint_plans")]

    np_dtype = (
        np.complex64 if cufinufft_dtype == "complex64" else np.complex128
    )
    coeff_np = np.ascontiguousarray(
        context.ring.obj.coeff.astype(np_dtype, copy=False).ravel()
    )
    coeff_gpu = cp.asarray(coeff_np)
    first_forward = cufinufft_forward(
        op, coeff_gpu, eps=args.cufinufft_eps
    )
    residual_parts = [
        part * complex(args.residual_scale_real, args.residual_scale_imag)
        for part in first_forward
    ]
    synchronize_cupy(cp)
    snapshots.append(memory_snapshot(cp, "after_coeff_forward_and_residual"))

    forward_value, forward_s, forward_times = timed_cupy(
        cp,
        lambda: cufinufft_forward(op, coeff_gpu, eps=args.cufinufft_eps),
        repeats=args.repeats,
        warmups=args.warmups,
    )
    adjoint_value, adjoint_s, adjoint_times = timed_cupy(
        cp,
        lambda: cufinufft_adjoint(
            op, residual_parts, eps=args.cufinufft_eps
        ),
        repeats=args.repeats,
        warmups=args.warmups,
    )
    pair_value, pair_s, pair_times = timed_cupy(
        cp,
        lambda: (
            cufinufft_forward(op, coeff_gpu, eps=args.cufinufft_eps),
            cufinufft_adjoint(op, residual_parts, eps=args.cufinufft_eps),
        ),
        repeats=args.pair_repeats,
        warmups=args.pair_warmups,
    )
    snapshots.append(memory_snapshot(cp, "after_timed_execution"))

    acfo_reference = load_acfo_reference(args.acfo_reference)
    accuracy_audit = {
        "path": str(args.accuracy_audit),
        "sha256": file_sha256(args.accuracy_audit),
    }
    speedup = (
        None
        if acfo_reference is None
        else float(pair_s / acfo_reference["pair_median_s"])
    )
    result = {
        "schema": "odt-cufinufft-only-production-timing-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": {
            "id": int(cp.cuda.runtime.getDevice()),
            "name": cp.cuda.runtime.getDeviceProperties(
                cp.cuda.runtime.getDevice()
            )["name"].decode("utf-8"),
            "cupy": getattr(cp, "__version__", None),
            "cufinufft": getattr(op.cufinufft, "__version__", None),
        },
        "problem": {
            "object_shape": list(context.ring.obj.coeff.shape),
            "object_bins": int(context.ring.obj.coeff.size),
            "ring_q_samples": int(op.ring.q_count),
            "axis_q_samples": 0 if op.axis is None else int(op.axis.q_count),
            "total_q_samples": int(
                op.ring.q_count + (0 if op.axis is None else op.axis.q_count)
            ),
            "ring_illum": int(args.ring_illum),
            "detector_shape": [int(args.cap_radial), int(args.cap_phi)],
        },
        "cufinufft": {
            "dtype": cufinufft_dtype,
            "eps": float(args.cufinufft_eps),
            "plan_mode": "plan",
            "context_build_s": float(context_s),
            "operator_setup_s": float(setup_s),
            "forward_timing": timing_distribution(forward_times),
            "adjoint_timing": timing_distribution(adjoint_times),
            "pair_timing": timing_distribution(pair_times),
            "forward_median_s": float(forward_s),
            "adjoint_median_s": float(adjoint_s),
            "pair_median_s": float(pair_s),
        },
        "memory_snapshots": snapshots,
        "acfo_reference": acfo_reference,
        "accuracy_audit": accuracy_audit,
        "speedup_acfo_vs_matched_cufinufft": speedup,
        "claim_boundary": [
            "This process contains only the cuFINUFFT plans and data; the ACFO plan is deliberately absent because the matched complex128 cuFINUFFT and ACFO plans cannot coexist on the 8 GB RTX 2070 SUPER.",
            "The linked ACFO timing is the frozen 30-repeat alternating AB/BA result from the same 256-cubed geometry and GPU, but the mixed-precision speedup is therefore a separate-process comparison rather than a co-resident AB/BA measurement.",
            "The linked small direct-sum audit, not cross-backend agreement, selects complex128 eps=1e-7 as the first direction-wise cuFINUFFT match to the complex64 ACFO error.",
            "Camera acquisition, transfer, remap and reconstruction iteration count are outside this raw full-q forward/adjoint pair timing.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"cuFINUFFT-only pair: {pair_s:.6f} s; "
        f"ACFO speedup: {speedup if speedup is not None else 'n/a'}",
        flush=True,
    )
    del (
        forward_value,
        adjoint_value,
        pair_value,
        first_forward,
        residual_parts,
        coeff_gpu,
        op,
    )
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return result


def parser() -> argparse.ArgumentParser:
    p = baseline_parser()
    p.description = (
        "Time a production cuFINUFFT forward/adjoint pair without a co-resident "
        "ACFO plan, for matched-error complex128 runs on an 8 GB GPU."
    )
    p.add_argument("--acfo-reference", type=Path, default=DEFAULT_ACFO_REFERENCE)
    p.add_argument("--accuracy-audit", type=Path, default=DEFAULT_ACCURACY_AUDIT)
    p.set_defaults(
        dtype="complex64",
        cufinufft_dtype="complex128",
        cufinufft_eps=1e-7,
        n_beta=256,
        n_r=256,
        n_z=256,
        ring_illum=120,
        skip_axis_illumination=False,
        cap_radial=256,
        cap_phi=256,
        h_cutoff=28,
        cpp_threads=4,
        skip_native_prepared_adjoint=True,
        compact_axisymmetric_kernel=True,
        repeats=1,
        warmups=1,
        pair_repeats=5,
        pair_warmups=2,
        out=ROOT
        / "benchmark_results"
        / "odt_cufinufft_matched_c128_full_pair5.json",
        csv=ROOT
        / "benchmark_results"
        / "odt_cufinufft_matched_c128_full_pair5_unused.csv",
        summary_md=None,
    )
    return p


def main() -> None:
    args = parser().parse_args()
    result = run(args)
    compact = {
        "problem": result["problem"],
        "cufinufft": result["cufinufft"],
        "speedup": result["speedup_acfo_vs_matched_cufinufft"],
        "memory": result["memory_snapshots"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
