from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark_high_na_torch_gpu import (  # noqa: E402
    device_name,
    import_torch,
    package_version,
    resolve_device,
    synchronize,
)
from benchmark_odt_realistic_geometry_reconstruction import (  # noqa: E402
    CompositeContext,
    build_composite_context,
    composite_adjoint,
    composite_forward,
    split_residual,
)
from benchmark_odt_torch_gpu_reconstruction import (  # noqa: E402
    TorchCompositeOdtPlan,
    rel_l2,
    torch_dtypes,
    to_numpy,
)


@dataclass(frozen=True)
class CuFinufftBlock:
    x_src: Any
    y_src: Any
    z_src: Any
    qx_tgt: Any
    qy_tgt: Any
    qz_tgt: Any
    qx_src: Any
    qy_src: Any
    qz_src: Any
    x_tgt: Any
    y_tgt: Any
    z_tgt: Any
    coeff_shape: tuple[int, ...]
    q_count: int
    q_scale: float
    forward_plan: Any
    adjoint_plan: Any
    forward_out: Any
    adjoint_out: Any


@dataclass(frozen=True)
class CuFinufftComposite:
    cp: Any
    cufinufft: Any
    dtype: str
    plan_mode: str
    complex_dtype: Any
    ring: CuFinufftBlock
    axis: CuFinufftBlock | None


def add_gpu_dll_directories() -> None:
    os.environ.setdefault("CUPY_CACHE_DIR", str(ROOT / ".cupy_cache"))
    for path in (
        ROOT / ".venv" / "Lib" / "site-packages" / "torch" / "lib",
        ROOT / ".venv" / "Lib" / "site-packages" / "cufinufft",
        ROOT / ".venv" / "Lib" / "site-packages" / "cufinufft.libs",
    ):
        if path.exists() and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(path))


def import_cufinufft_modules() -> tuple[Any, Any]:
    add_gpu_dll_directories()
    try:
        import cupy as cp
        import cufinufft
    except ImportError as exc:
        raise RuntimeError("cupy/cufinufft is not installed or cannot load its CUDA DLLs") from exc
    return cp, cufinufft


def cupy_dtypes(cp: Any, dtype: str) -> tuple[Any, Any, Any, Any]:
    if dtype == "complex64":
        return cp.complex64, cp.float32, np.complex64, np.float32
    if dtype == "complex128":
        return cp.complex128, cp.float64, np.complex128, np.float64
    raise ValueError("dtype must be complex64 or complex128")


def q_source_scale(q: Any) -> float:
    max_abs = max(
        float(np.max(np.abs(q.qx))),
        float(np.max(np.abs(q.qy))),
        float(np.max(np.abs(q.qz))),
        1.0,
    )
    return max(1.0, 1.001 * max_abs / math.pi)


def make_block(
    cp: Any,
    cufinufft: Any,
    ctx: Any,
    *,
    dtype: str,
    complex_dtype: Any,
    np_real: Any,
    plan_mode: str,
    eps: float,
) -> CuFinufftBlock:
    obj = ctx.obj
    q = ctx.flat_q
    scale = q_source_scale(q)
    x_src = cp.asarray(np.asarray(obj.x, dtype=np_real))
    y_src = cp.asarray(np.asarray(obj.y, dtype=np_real))
    z_src = cp.asarray(np.asarray(obj.z, dtype=np_real))
    qx_tgt = cp.asarray(np.asarray(q.qx, dtype=np_real))
    qy_tgt = cp.asarray(np.asarray(q.qy, dtype=np_real))
    qz_tgt = cp.asarray(np.asarray(q.qz, dtype=np_real))
    qx_src = cp.asarray(np.asarray(q.qx / scale, dtype=np_real))
    qy_src = cp.asarray(np.asarray(q.qy / scale, dtype=np_real))
    qz_src = cp.asarray(np.asarray(q.qz / scale, dtype=np_real))
    x_tgt = cp.asarray(np.asarray(obj.x * scale, dtype=np_real))
    y_tgt = cp.asarray(np.asarray(obj.y * scale, dtype=np_real))
    z_tgt = cp.asarray(np.asarray(obj.z * scale, dtype=np_real))
    forward_plan = None
    adjoint_plan = None
    forward_out = None
    adjoint_out = None
    if plan_mode == "plan":
        forward_plan = cufinufft.Plan(3, 3, eps=eps, isign=1, dtype=dtype)
        forward_plan.setpts(x_src, y_src, z_src, qx_tgt, qy_tgt, qz_tgt)
        adjoint_plan = cufinufft.Plan(3, 3, eps=eps, isign=-1, dtype=dtype)
        adjoint_plan.setpts(qx_src, qy_src, qz_src, x_tgt, y_tgt, z_tgt)
        forward_out = cp.empty(int(q.count), dtype=complex_dtype)
        adjoint_out = cp.empty(int(obj.x.size), dtype=complex_dtype)
    return CuFinufftBlock(
        x_src=x_src,
        y_src=y_src,
        z_src=z_src,
        qx_tgt=qx_tgt,
        qy_tgt=qy_tgt,
        qz_tgt=qz_tgt,
        qx_src=qx_src,
        qy_src=qy_src,
        qz_src=qz_src,
        x_tgt=x_tgt,
        y_tgt=y_tgt,
        z_tgt=z_tgt,
        coeff_shape=tuple(obj.coeff.shape),
        q_count=int(q.count),
        q_scale=float(scale),
        forward_plan=forward_plan,
        adjoint_plan=adjoint_plan,
        forward_out=forward_out,
        adjoint_out=adjoint_out,
    )


def make_cufinufft_composite(ctx: CompositeContext, *, dtype: str, plan_mode: str, eps: float) -> CuFinufftComposite:
    cp, cufinufft = import_cufinufft_modules()
    complex_dtype, _, np_complex, np_real = cupy_dtypes(cp, dtype)
    return CuFinufftComposite(
        cp=cp,
        cufinufft=cufinufft,
        dtype=dtype,
        plan_mode=plan_mode,
        complex_dtype=complex_dtype,
        ring=make_block(
            cp,
            cufinufft,
            ctx.ring,
            dtype=dtype,
            complex_dtype=complex_dtype,
            np_real=np_real,
            plan_mode=plan_mode,
            eps=eps,
        ),
        axis=None
        if ctx.axis is None
        else make_block(
            cp,
            cufinufft,
            ctx.axis,
            dtype=dtype,
            complex_dtype=complex_dtype,
            np_real=np_real,
            plan_mode=plan_mode,
            eps=eps,
        ),
    )


def cufinufft_forward_block(op: CuFinufftComposite, block: CuFinufftBlock, coeff_gpu: Any, *, eps: float) -> Any:
    if block.forward_plan is not None:
        return block.forward_plan.execute(coeff_gpu, out=block.forward_out)
    return op.cufinufft.nufft3d3(
        block.x_src,
        block.y_src,
        block.z_src,
        coeff_gpu,
        block.qx_tgt,
        block.qy_tgt,
        block.qz_tgt,
        eps=eps,
        isign=1,
    )


def cufinufft_adjoint_block(op: CuFinufftComposite, block: CuFinufftBlock, residual_gpu: Any, *, eps: float) -> Any:
    if block.adjoint_plan is not None:
        return block.adjoint_plan.execute(residual_gpu, out=block.adjoint_out)
    return op.cufinufft.nufft3d3(
        block.qx_src,
        block.qy_src,
        block.qz_src,
        residual_gpu,
        block.x_tgt,
        block.y_tgt,
        block.z_tgt,
        eps=eps,
        isign=-1,
    )


def cufinufft_forward(op: CuFinufftComposite, coeff_gpu: Any, *, eps: float) -> list[Any]:
    parts = [cufinufft_forward_block(op, op.ring, coeff_gpu, eps=eps)]
    if op.axis is not None:
        parts.append(cufinufft_forward_block(op, op.axis, coeff_gpu, eps=eps))
    return parts


def cufinufft_adjoint(op: CuFinufftComposite, residual_parts: list[Any], *, eps: float) -> list[Any]:
    parts = [cufinufft_adjoint_block(op, op.ring, residual_parts[0], eps=eps)]
    if op.axis is not None:
        parts.append(cufinufft_adjoint_block(op, op.axis, residual_parts[1], eps=eps))
    return parts


def synchronize_cupy(cp: Any) -> None:
    # cuFINUFFT may enqueue setup/execute work on a plan-owned stream.  A
    # current-stream sync can therefore return while device work is still in
    # flight and contaminate the next backend's timing.
    cp.cuda.runtime.deviceSynchronize()


def timed_cupy(cp: Any, func, *, repeats: int, warmups: int) -> tuple[Any, float, list[float]]:
    value = None
    for _ in range(max(0, warmups)):
        value = func()
        synchronize_cupy(cp)
    times: list[float] = []
    for _ in range(max(1, repeats)):
        synchronize_cupy(cp)
        start = time.perf_counter()
        value = func()
        synchronize_cupy(cp)
        times.append(time.perf_counter() - start)
    if value is None:
        raise RuntimeError("timed CuPy function did not run")
    return value, float(median(times)), times


def timed_torch(torch: Any, device: Any, func, *, repeats: int, warmups: int) -> tuple[Any, float, list[float]]:
    value = None
    for _ in range(max(0, warmups)):
        value = func()
        synchronize(torch, device)
    times: list[float] = []
    for _ in range(max(1, repeats)):
        synchronize(torch, device)
        start = time.perf_counter()
        value = func()
        synchronize(torch, device)
        times.append(time.perf_counter() - start)
    if value is None:
        raise RuntimeError("timed torch function did not run")
    return value, float(median(times)), times


def timed_alternating_backend_pairs(
    torch: Any,
    device: Any,
    cp: Any,
    ours_func,
    cufinufft_func,
    *,
    repeats: int,
    warmups: int,
) -> tuple[Any, Any, list[float], list[float]]:
    ours_value = None
    cufinufft_value = None
    ours_times: list[float] = []
    cufinufft_times: list[float] = []
    total = max(0, warmups) + max(1, repeats)
    for pair_index in range(total):
        order = (("ours", ours_func), ("cufinufft", cufinufft_func))
        if pair_index % 2:
            order = tuple(reversed(order))
        for label, func in order:
            if label == "ours":
                synchronize(torch, device)
                start = time.perf_counter()
                ours_value = func()
                synchronize(torch, device)
                elapsed = time.perf_counter() - start
                if pair_index >= warmups:
                    ours_times.append(elapsed)
            else:
                synchronize_cupy(cp)
                start = time.perf_counter()
                cufinufft_value = func()
                synchronize_cupy(cp)
                elapsed = time.perf_counter() - start
                if pair_index >= warmups:
                    cufinufft_times.append(elapsed)
        if (pair_index + 1) % 5 == 0 or pair_index + 1 == total:
            print(f"pair protocol: completed {pair_index + 1}/{total}", flush=True)
    return ours_value, cufinufft_value, ours_times, cufinufft_times


def timing_distribution(times: list[float]) -> dict[str, float | int | None]:
    if not times:
        return {
            "count": 0,
            "median_s": None,
            "q1_s": None,
            "q3_s": None,
            "min_s": None,
            "max_s": None,
            "mean_s": None,
            "std_s": None,
        }
    values = np.asarray(times, dtype=np.float64)
    return {
        "count": int(values.size),
        "median_s": float(np.median(values)),
        "q1_s": float(np.quantile(values, 0.25)),
        "q3_s": float(np.quantile(values, 0.75)),
        "min_s": float(np.min(values)),
        "max_s": float(np.max(values)),
        "mean_s": float(np.mean(values)),
        "std_s": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
    }


def median_time(func, *, repeats: int, warmups: int) -> tuple[Any, float, list[float]]:
    value = None
    for _ in range(max(0, warmups)):
        value = func()
    times: list[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        value = func()
        times.append(time.perf_counter() - start)
    if value is None:
        raise RuntimeError("timed function did not run")
    return value, float(median(times)), times


def flatten_cufinufft_forward(parts: list[Any]) -> np.ndarray:
    arrays = [np.asarray(part.get()) for part in parts]
    return np.concatenate(arrays)


def sum_cufinufft_adjoint(parts: list[Any], shape: tuple[int, ...]) -> np.ndarray:
    out = np.zeros(int(np.prod(shape)), dtype=np.complex128)
    for part in parts:
        out += np.asarray(part.get(), dtype=np.complex128)
    return out.reshape(shape)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ODT cuFINUFFT GPU baseline",
        "",
        "This benchmark compares the same ring-plus-axis ODT operator using the structured PyTorch GPU path and a generic cuFINUFFT 3D type-3 GPU baseline.",
        "",
        "## Configuration",
        "",
        f"- device: `{summary['device_name']}`",
        f"- torch: `{summary['torch_version']}`",
        f"- cupy: `{summary['cupy_version']}`",
        f"- cufinufft: `{summary['cufinufft_version']}`",
        f"- dtype: `{summary['dtype']}`",
        f"- cuFINUFFT dtype: `{summary.get('cufinufft_dtype', summary['dtype'])}`",
        f"- eps: `{summary['eps']}`",
        f"- cuFINUFFT mode: `{summary['cufinufft_plan_mode']}`",
        f"- total q samples: `{summary['total_q_samples']}`",
        f"- object bins: `{summary['object_bins']}`",
        f"- cap: `{summary['cap_radial']} x {summary['cap_phi']}`",
        f"- total illuminations: `{summary['total_illumination_count']}`",
        f"- axis included: `{summary['axis_illumination_included']}`",
        f"- axis L=0 pruning: `{summary['prune_axis_l0']}`",
        f"- axial low-rank requested: `{summary['axial_lowrank_rank_requested']}`",
        f"- ring/axis L modes: `{summary['ring_l_modes']}` / `{summary['axis_l_modes']}`",
        f"- ring adaptive-L threshold/active fraction: "
        f"`{summary['ring_adaptive_l_packed_threshold']}` / "
        f"`{summary['ring_adaptive_l_active_fraction']:.6f}`",
        "",
        "## Hot Timings",
        "",
        f"- ours operator setup: `{1000.0 * summary['ours_operator_setup_s']:.3f}` ms",
        f"- cuFINUFFT operator setup: `{1000.0 * summary['cufinufft_operator_setup_s']:.3f}` ms",
        "",
        "| path | median ms | speedup vs cuFINUFFT |",
        "| --- | ---: | ---: |",
        f"| ours forward | {1000.0 * summary['ours_forward_s']:.3f} | {summary['cufinufft_forward_s'] / summary['ours_forward_s']:.3f}x |",
        f"| cuFINUFFT forward | {1000.0 * summary['cufinufft_forward_s']:.3f} | 1.000x |",
        f"| ours adjoint | {1000.0 * summary['ours_adjoint_s']:.3f} | {summary['cufinufft_adjoint_s'] / summary['ours_adjoint_s']:.3f}x |",
        f"| cuFINUFFT adjoint | {1000.0 * summary['cufinufft_adjoint_s']:.3f} | 1.000x |",
        "",
        "## Interleaved forward-adjoint pair",
        "",
        f"- protocol: `{summary['pair_timing_protocol']['method_order']}`",
        f"- measured repeats per backend: `{summary['pair_timing_protocol']['measured_repeats_per_backend']}`",
        f"- ACFO median: `{1000.0 * summary['ours_forward_adjoint_pair_s']:.3f}` ms",
        f"- cuFINUFFT median: `{1000.0 * summary['cufinufft_forward_adjoint_pair_s']:.3f}` ms",
        f"- ACFO speedup: `{summary['ours_speedup_vs_cufinufft_pair']:.3f}x`",
        "",
        "## Accuracy",
        "",
        f"- cuFINUFFT forward rel-L2 vs ours: `{summary['cufinufft_forward_rel_l2_vs_ours']:.6g}`",
        f"- cuFINUFFT adjoint rel-L2 vs ours: `{summary['cufinufft_adjoint_rel_l2_vs_ours']:.6g}`",
    ]
    if summary.get("cpu_finufft_forward_s") is not None:
        lines.extend(
            [
                "",
                "## CPU FINUFFT",
                "",
                f"- CPU FINUFFT forward: `{1000.0 * summary['cpu_finufft_forward_s']:.3f}` ms",
                f"- CPU FINUFFT adjoint: `{1000.0 * summary['cpu_finufft_adjoint_s']:.3f}` ms",
                f"- cuFINUFFT forward rel-L2 vs CPU FINUFFT: `{summary['cufinufft_forward_rel_l2_vs_cpu_finufft']:.6g}`",
                f"- cuFINUFFT adjoint rel-L2 vs CPU FINUFFT: `{summary['cufinufft_adjoint_rel_l2_vs_cpu_finufft']:.6g}`",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("this benchmark requires a CUDA device")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    ctx = build_composite_context(args)
    start = time.perf_counter()
    torch_plan = TorchCompositeOdtPlan.from_context(
        ctx,
        torch=torch,
        device=device,
        dtype=args.dtype,
        low_memory_adjoint=args.low_memory_adjoint,
        radial_block_size=args.radial_block_size,
        illumination_block_size=args.illumination_block_size,
        prune_axis_l0=args.prune_axis_l0,
        axial_lowrank_rank=args.axial_lowrank_rank,
        ring_adaptive_l_packed_threshold=args.ring_adaptive_l_packed_threshold,
    )
    torch_setup_s = time.perf_counter() - start

    _, _, np_complex, _ = torch_dtypes(torch, args.dtype)
    true_coeff_np = np.ascontiguousarray(ctx.ring.obj.coeff.astype(np_complex, copy=False))
    true_coeff_t = torch.as_tensor(true_coeff_np, dtype=torch_plan.complex_dtype, device=device)
    # Initialize PyTorch FFT/cuBLAS/index kernels before cuFINUFFT creates its
    # plans.  On Windows, letting cuFINUFFT be the first executable CUDA path
    # can leave the later packed PyTorch path in a severely degraded context.
    with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
        prime_forward_t = torch_plan.forward(true_coeff_t)
        torch_plan.adjoint(
            prime_forward_t
            * complex(args.residual_scale_real, args.residual_scale_imag)
        )
        synchronize(torch, device)

    cufinufft_dtype = (
        args.dtype
        if getattr(args, "cufinufft_dtype", "same") == "same"
        else args.cufinufft_dtype
    )
    start = time.perf_counter()
    cu_op = make_cufinufft_composite(
        ctx,
        dtype=cufinufft_dtype,
        plan_mode=args.cufinufft_plan_mode,
        eps=args.cufinufft_eps,
    )
    cp = cu_op.cp
    synchronize_cupy(cp)
    cu_setup_s = time.perf_counter() - start

    cu_np_complex = (
        np.complex64 if cufinufft_dtype == "complex64" else np.complex128
    )
    cu_coeff_np = np.ascontiguousarray(
        true_coeff_np.astype(cu_np_complex, copy=False)
    )
    coeff_gpu = cp.asarray(cu_coeff_np.ravel())

    with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
        ours_forward_t, ours_forward_s, ours_forward_times = timed_torch(
            torch,
            device,
            lambda: torch_plan.forward(true_coeff_t),
            repeats=args.repeats,
            warmups=args.warmups,
        )
        residual_t = ours_forward_t * complex(args.residual_scale_real, args.residual_scale_imag)
        ours_adjoint_t, ours_adjoint_s, ours_adjoint_times = timed_torch(
            torch,
            device,
            lambda: torch_plan.adjoint(residual_t),
            repeats=args.repeats,
            warmups=args.warmups,
        )

    residual_np = to_numpy(torch, device, residual_t).astype(np_complex, copy=False)
    cu_residual_np = np.ascontiguousarray(
        residual_np.astype(cu_np_complex, copy=False)
    )
    ring_residual_np, axis_residual_np = split_residual(ctx, cu_residual_np)
    residual_parts_gpu = [cp.asarray(ring_residual_np)]
    if axis_residual_np is not None:
        residual_parts_gpu.append(cp.asarray(axis_residual_np))

    cu_forward_parts, cu_forward_s, cu_forward_times = timed_cupy(
        cp,
        lambda: cufinufft_forward(cu_op, coeff_gpu, eps=args.cufinufft_eps),
        repeats=args.repeats,
        warmups=args.warmups,
    )
    cu_adjoint_parts, cu_adjoint_s, cu_adjoint_times = timed_cupy(
        cp,
        lambda: cufinufft_adjoint(cu_op, residual_parts_gpu, eps=args.cufinufft_eps),
        repeats=args.repeats,
        warmups=args.warmups,
    )

    pair_ours_times: list[float] = []
    pair_cufinufft_times: list[float] = []
    if args.pair_repeats > 0:
        _, _, pair_ours_times, pair_cufinufft_times = timed_alternating_backend_pairs(
            torch,
            device,
            cp,
            lambda: (torch_plan.forward(true_coeff_t), torch_plan.adjoint(residual_t)),
            lambda: (
                cufinufft_forward(cu_op, coeff_gpu, eps=args.cufinufft_eps),
                cufinufft_adjoint(cu_op, residual_parts_gpu, eps=args.cufinufft_eps),
            ),
            repeats=args.pair_repeats,
            warmups=args.pair_warmups,
        )
        ours_pair_s = float(median(pair_ours_times))
        cufinufft_pair_s = float(median(pair_cufinufft_times))
    else:
        ours_pair_s = float(ours_forward_s + ours_adjoint_s)
        cufinufft_pair_s = float(cu_forward_s + cu_adjoint_s)

    ours_forward_np = to_numpy(torch, device, ours_forward_t).astype(np.complex128, copy=False)
    ours_adjoint_np = to_numpy(torch, device, ours_adjoint_t).astype(np.complex128, copy=False)
    cu_forward_np = flatten_cufinufft_forward(cu_forward_parts).astype(np.complex128, copy=False)
    cu_adjoint_np = sum_cufinufft_adjoint(cu_adjoint_parts, true_coeff_np.shape)

    cpu_forward_s = None
    cpu_adjoint_s = None
    cpu_forward_l2 = None
    cpu_adjoint_l2 = None
    if args.include_cpu_finufft:
        cpu_forward, cpu_forward_s, cpu_forward_times = median_time(
            lambda: composite_forward(ctx, true_coeff_np.astype(np.complex128), args, use_finufft=True),
            repeats=args.cpu_repeats,
            warmups=args.cpu_warmups,
        )
        cpu_adjoint, cpu_adjoint_s, cpu_adjoint_times = median_time(
            lambda: composite_adjoint(ctx, residual_np.astype(np.complex128), args, use_finufft=True),
            repeats=args.cpu_repeats,
            warmups=args.cpu_warmups,
        )
        cpu_forward_l2 = rel_l2(cu_forward_np, cpu_forward)
        cpu_adjoint_l2 = rel_l2(cu_adjoint_np, cpu_adjoint)
    else:
        cpu_forward_times = []
        cpu_adjoint_times = []

    peak_mib = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    cupy_peak_mib = None
    try:
        cupy_peak_mib = float(cp.get_default_memory_pool().total_bytes() / (1024.0 * 1024.0))
    except Exception:
        pass

    summary = {
        "device": str(device),
        "device_name": device_name(torch, device),
        "torch_version": package_version("torch"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cupy_version": getattr(cp, "__version__", None),
        "cufinufft_version": getattr(cu_op.cufinufft, "__version__", None),
        "dtype": args.dtype,
        "cufinufft_dtype": cufinufft_dtype,
        "eps": float(args.cufinufft_eps),
        "cufinufft_plan_mode": args.cufinufft_plan_mode,
        "low_memory_adjoint": bool(args.low_memory_adjoint),
        "radial_block_size": int(args.radial_block_size),
        "illumination_block_size": int(args.illumination_block_size),
        "prune_axis_l0": bool(args.prune_axis_l0),
        "axial_lowrank_rank_requested": int(args.axial_lowrank_rank),
        "ring_axial_lowrank_rank": int(torch_plan.ring.axial_lowrank_rank),
        "axis_axial_lowrank_rank": (
            None
            if torch_plan.axis is None
            else int(torch_plan.axis.axial_lowrank_rank)
        ),
        "ring_l_modes": int(torch_plan.ring.n_l),
        "axis_l_modes": (
            None if torch_plan.axis is None else int(torch_plan.axis.n_l)
        ),
        "ring_adaptive_l_packed_threshold": float(
            args.ring_adaptive_l_packed_threshold
        ),
        "ring_adaptive_l_active_fraction": float(
            torch_plan.ring.adaptive_l_active_fraction
        ),
        "skip_native_prepared_adjoint": bool(args.skip_native_prepared_adjoint),
        "compact_axisymmetric_kernel": bool(args.compact_axisymmetric_kernel),
        "cap_radial": int(args.cap_radial),
        "cap_phi": int(args.cap_phi),
        "ring_illum": int(args.ring_illum),
        "axis_illumination_included": not bool(args.skip_axis_illumination),
        "total_illumination_count": int(args.ring_illum) + (0 if args.skip_axis_illumination else 1),
        "total_q_samples": int(torch_plan.q_count),
        "object_bins": int(true_coeff_np.size),
        "ring_q_scale_for_adjoint": float(cu_op.ring.q_scale),
        "axis_q_scale_for_adjoint": None if cu_op.axis is None else float(cu_op.axis.q_scale),
        "ours_operator_setup_s": float(torch_setup_s),
        "cufinufft_operator_setup_s": float(cu_setup_s),
        "ours_forward_s": float(ours_forward_s),
        "ours_adjoint_s": float(ours_adjoint_s),
        "cufinufft_forward_s": float(cu_forward_s),
        "cufinufft_adjoint_s": float(cu_adjoint_s),
        "ours_forward_adjoint_pair_s": ours_pair_s,
        "cufinufft_forward_adjoint_pair_s": cufinufft_pair_s,
        "ours_speedup_vs_cufinufft_pair": float(cufinufft_pair_s / ours_pair_s),
        "pair_timing_protocol": {
            "method_order": "alternating_ab_ba" if args.pair_repeats > 0 else "sum_of_separate_medians",
            "warmups_per_backend": int(args.pair_warmups),
            "measured_repeats_per_backend": int(args.pair_repeats),
            "ours_times_s": pair_ours_times,
            "cufinufft_times_s": pair_cufinufft_times,
            "ours_distribution": timing_distribution(pair_ours_times),
            "cufinufft_distribution": timing_distribution(pair_cufinufft_times),
        },
        "ours_speedup_vs_cufinufft_forward": float(cu_forward_s / ours_forward_s),
        "ours_speedup_vs_cufinufft_adjoint": float(cu_adjoint_s / ours_adjoint_s),
        "cufinufft_forward_rel_l2_vs_ours": rel_l2(cu_forward_np, ours_forward_np),
        "cufinufft_adjoint_rel_l2_vs_ours": rel_l2(cu_adjoint_np, ours_adjoint_np),
        "cpu_finufft_forward_s": None if cpu_forward_s is None else float(cpu_forward_s),
        "cpu_finufft_adjoint_s": None if cpu_adjoint_s is None else float(cpu_adjoint_s),
        "cufinufft_forward_rel_l2_vs_cpu_finufft": cpu_forward_l2,
        "cufinufft_adjoint_rel_l2_vs_cpu_finufft": cpu_adjoint_l2,
        "torch_peak_allocated_mib": peak_mib,
        "cupy_pool_total_mib": cupy_peak_mib,
    }
    rows = [
        {
            "path": "ours_forward",
            "median_s": float(ours_forward_s),
            "times_s": " ".join(f"{value:.9g}" for value in ours_forward_times),
        },
        {
            "path": "ours_adjoint",
            "median_s": float(ours_adjoint_s),
            "times_s": " ".join(f"{value:.9g}" for value in ours_adjoint_times),
        },
        {
            "path": "cufinufft_forward",
            "median_s": float(cu_forward_s),
            "times_s": " ".join(f"{value:.9g}" for value in cu_forward_times),
        },
        {
            "path": "cufinufft_adjoint",
            "median_s": float(cu_adjoint_s),
            "times_s": " ".join(f"{value:.9g}" for value in cu_adjoint_times),
        },
    ]
    if args.include_cpu_finufft:
        rows.extend(
            [
                {
                    "path": "cpu_finufft_forward",
                    "median_s": float(cpu_forward_s),
                    "times_s": " ".join(f"{value:.9g}" for value in cpu_forward_times),
                },
                {
                    "path": "cpu_finufft_adjoint",
                    "median_s": float(cpu_adjoint_s),
                    "times_s": " ".join(f"{value:.9g}" for value in cpu_adjoint_times),
                },
            ]
        )

    write_csv(args.csv, rows)
    write_json(args.out, {"config": vars(args), "summary": summary, "rows": rows})
    if args.summary_md:
        write_summary(args.summary_md, summary)
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare ODT structured GPU operator against cuFINUFFT type-3 GPU baseline.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    p.add_argument(
        "--cufinufft-dtype",
        choices=["same", "complex64", "complex128"],
        default="same",
        help="cuFINUFFT precision; 'same' preserves the legacy same-dtype protocol.",
    )
    p.add_argument("--low-memory-adjoint", action="store_true")
    p.add_argument("--radial-block-size", type=int, default=0)
    p.add_argument("--illumination-block-size", type=int, default=0)
    p.add_argument("--prune-axis-l0", action="store_true")
    p.add_argument("--axial-lowrank-rank", type=int, default=0)
    p.add_argument("--ring-adaptive-l-packed-threshold", type=float, default=0.0)
    p.add_argument("--skip-native-prepared-adjoint", action="store_true")
    p.add_argument("--compact-axisymmetric-kernel", action="store_true")
    p.add_argument("--n-beta", type=int, default=384)
    p.add_argument("--n-r", type=int, default=16)
    p.add_argument("--n-z", type=int, default=15)
    p.add_argument("--r-max", type=float, default=1.0)
    p.add_argument("--z-max", type=float, default=0.8)
    p.add_argument("--phantom", choices=["beads", "random_beads", "shell"], default="random_beads")
    p.add_argument("--object-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--k", type=float, default=17.307319527958313)
    p.add_argument("--detector-na", type=float, default=0.9240924092409241)
    p.add_argument("--illumination-angle-deg", type=float, default=49.0)
    p.add_argument("--ring-illum", type=int, default=100)
    p.add_argument("--skip-axis-illumination", action="store_true")
    p.add_argument("--cap-radial", type=int, default=64)
    p.add_argument("--cap-phi", type=int, default=256)
    p.add_argument("--h-cutoff", type=int, default=None)
    p.add_argument("--h-margin", type=int, default=20)
    p.add_argument("--l-margin", type=int, default=18)
    p.add_argument("--cone-l-prune-threshold", type=float, default=1e-12)
    p.add_argument("--cpp-threads", type=int, default=16)
    p.add_argument("--forward-execute-mode", choices=["prepared", "wrapper"], default="prepared")
    p.add_argument("--forward-kernel-mode", choices=["compact", "partitioned"], default="partitioned")
    p.add_argument(
        "--native-prepared-plan-mode",
        choices=["auto", "direct", "gathered", "gathered-zmajor"],
        default="auto",
    )
    p.add_argument("--native-prepared-gather-threshold", type=int, default=8192)
    p.add_argument("--finufft-eps", type=float, default=1e-12)
    p.add_argument("--finufft-q-batch-size", type=int, default=1_048_576)
    p.add_argument("--cufinufft-eps", type=float, default=1e-6)
    p.add_argument("--cufinufft-plan-mode", choices=["plan", "simple"], default="plan")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--pair-repeats", type=int, default=0)
    p.add_argument("--pair-warmups", type=int, default=0)
    p.add_argument("--include-cpu-finufft", action="store_true")
    p.add_argument("--cpu-repeats", type=int, default=1)
    p.add_argument("--cpu-warmups", type=int, default=0)
    p.add_argument("--residual-scale-real", type=float, default=0.1)
    p.add_argument("--residual-scale-imag", type=float, default=0.2)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_cufinufft_gpu_baseline.json",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_cufinufft_gpu_baseline.csv",
    )
    p.add_argument(
        "--summary-md",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_cufinufft_gpu_baseline_summary.md",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
