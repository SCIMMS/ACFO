from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark_aidt_public_measured_adapter import (  # noqa: E402
    build_operator_args,
    json_default,
    measured_polar_residual_from_contract,
)
from benchmark_high_na_torch_gpu import (  # noqa: E402
    device_name,
    import_torch,
    package_version,
    resolve_device,
    synchronize,
)
from benchmark_odt_cufinufft_gpu_baseline import (  # noqa: E402
    cufinufft_adjoint_block,
    cufinufft_forward_block,
    cupy_dtypes,
    import_cufinufft_modules,
    make_block,
    synchronize_cupy,
)
from benchmark_odt_gpu_reconstruction_compare import (  # noqa: E402
    cufinufft_adjoint_torch,
    cufinufft_forward_torch,
)
from benchmark_odt_iterative_reconstruction import build_context  # noqa: E402
from benchmark_odt_torch_gpu_reconstruction import (  # noqa: E402
    TorchConeAxisOdtPlan,
    rel_l2,
    torch_dtypes,
)


def coeff_norm2_torch(torch: Any, value: Any) -> Any:
    return torch.sum(torch.real(torch.conj(value) * value))


def coeff_norm2_cupy(cp: Any, value: Any) -> Any:
    return cp.sum(cp.real(cp.conj(value) * value))


@dataclass(frozen=True)
class UpdateLoopResult:
    rows: list[dict[str, Any]]
    x_cpu: np.ndarray


def run_torch_update_loop(
    *,
    label: str,
    torch: Any,
    device: Any,
    coeff_shape: tuple[int, ...],
    coeff_dtype: Any,
    data: Any,
    forward,
    adjoint,
    sync,
    iterations: int,
    warmup_updates: int,
) -> UpdateLoopResult:
    x = torch.zeros(coeff_shape, dtype=coeff_dtype, device=device)
    pred = torch.zeros_like(data)
    data_norm = torch.clamp(torch.linalg.vector_norm(data), min=1e-30)
    rows: list[dict[str, Any]] = [
        {
            "method": label,
            "iteration": 0,
            "loss_rel": 1.0,
            "alpha": 0.0,
            "iter_s": 0.0,
            "adjoint_s": 0.0,
            "line_forward_s": 0.0,
            "update_s": 0.0,
            "cumulative_iter_s": 0.0,
        }
    ]
    cumulative = 0.0
    with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
        for _ in range(max(0, int(warmup_updates))):
            residual = pred - data
            grad = adjoint(residual)
            a_grad = forward(grad)
            alpha = coeff_norm2_torch(torch, grad) / torch.clamp(
                coeff_norm2_torch(torch, a_grad),
                min=1e-30,
            )
            _ = x - alpha * grad
            _ = pred - alpha * a_grad
            sync()
        for iteration in range(1, int(iterations) + 1):
            residual = pred - data
            sync()
            iter_start = time.perf_counter()

            adjoint_start = time.perf_counter()
            grad = adjoint(residual)
            sync()
            adjoint_s = time.perf_counter() - adjoint_start

            forward_start = time.perf_counter()
            a_grad = forward(grad)
            sync()
            forward_s = time.perf_counter() - forward_start

            alpha = coeff_norm2_torch(torch, grad) / torch.clamp(
                coeff_norm2_torch(torch, a_grad),
                min=1e-30,
            )
            x = x - alpha * grad
            pred = pred - alpha * a_grad
            loss = torch.linalg.vector_norm(pred - data) / data_norm
            sync()
            update_s = time.perf_counter() - iter_start
            cumulative += update_s
            rows.append(
                {
                    "method": label,
                    "iteration": int(iteration),
                    "loss_rel": float(loss.detach().cpu().item()),
                    "alpha": float(alpha.detach().cpu().item()),
                    "iter_s": float(update_s),
                    "adjoint_s": float(adjoint_s),
                    "line_forward_s": float(forward_s),
                    "update_s": float(update_s),
                    "cumulative_iter_s": float(cumulative),
                }
            )
    sync()
    return UpdateLoopResult(rows=rows, x_cpu=x.detach().cpu().numpy())


def run_prepared_update_loop(
    *,
    torch: Any,
    device: Any,
    plan: Any,
    data: Any,
    iterations: int,
    warmup_updates: int = 0,
) -> UpdateLoopResult:
    return run_torch_update_loop(
        label="prepared_gpu",
        torch=torch,
        device=device,
        coeff_shape=(plan.n_r, plan.n_z, plan.n_beta),
        coeff_dtype=plan.complex_dtype,
        data=data,
        forward=plan.forward,
        adjoint=plan.adjoint,
        sync=lambda: synchronize(torch, device),
        iterations=iterations,
        warmup_updates=warmup_updates,
    )


def run_cufinufft_update_loop(
    *,
    cp: Any,
    cu_op: Any,
    block: Any,
    data: Any,
    coeff_shape: tuple[int, ...],
    eps: float,
    iterations: int,
) -> UpdateLoopResult:
    x = cp.zeros(int(np.prod(coeff_shape)), dtype=data.dtype)
    pred = cp.zeros_like(data)
    data_norm = cp.maximum(cp.linalg.norm(data), cp.asarray(1e-30, dtype=cp.float32))
    rows: list[dict[str, Any]] = [
        {
            "method": "cufinufft_gpu",
            "iteration": 0,
            "loss_rel": 1.0,
            "alpha": 0.0,
            "iter_s": 0.0,
            "adjoint_s": 0.0,
            "line_forward_s": 0.0,
            "update_s": 0.0,
            "cumulative_iter_s": 0.0,
        }
    ]
    cumulative = 0.0
    for iteration in range(1, int(iterations) + 1):
        residual = pred - data
        synchronize_cupy(cp)
        iter_start = time.perf_counter()

        adjoint_start = time.perf_counter()
        grad = cufinufft_adjoint_block(cu_op, block, residual, eps=eps)
        synchronize_cupy(cp)
        adjoint_s = time.perf_counter() - adjoint_start

        forward_start = time.perf_counter()
        a_grad = cufinufft_forward_block(cu_op, block, grad, eps=eps)
        synchronize_cupy(cp)
        forward_s = time.perf_counter() - forward_start

        alpha = coeff_norm2_cupy(cp, grad) / cp.maximum(
            coeff_norm2_cupy(cp, a_grad),
            cp.asarray(1e-30, dtype=cp.float32),
        )
        x = x - alpha * grad
        pred = pred - alpha * a_grad
        loss = cp.linalg.norm(pred - data) / data_norm
        synchronize_cupy(cp)
        update_s = time.perf_counter() - iter_start
        cumulative += update_s
        rows.append(
            {
                "method": "cufinufft_gpu",
                "iteration": int(iteration),
                "loss_rel": float(loss.get()),
                "alpha": float(alpha.get()),
                "iter_s": float(update_s),
                "adjoint_s": float(adjoint_s),
                "line_forward_s": float(forward_s),
                "update_s": float(update_s),
                "cumulative_iter_s": float(cumulative),
            }
        )
    synchronize_cupy(cp)
    return UpdateLoopResult(rows=rows, x_cpu=np.asarray(x.get()).reshape(coeff_shape))


def method_stats(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    selected = [row for row in rows if row["method"] == method]
    active = [row for row in selected if int(row["iteration"]) > 0]
    final = max(selected, key=lambda row: int(row["iteration"]))
    return {
        "iterations": int(final["iteration"]),
        "final_loss_rel": float(final["loss_rel"]),
        "median_iter_s": float(median(float(row["iter_s"]) for row in active)),
        "median_adjoint_s": float(median(float(row["adjoint_s"]) for row in active)),
        "median_forward_s": float(median(float(row["line_forward_s"]) for row in active)),
        "cumulative_iter_s": float(final["cumulative_iter_s"]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def write_summary(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    prepared = summary["prepared_gpu"]
    cufinufft = summary["cufinufft_gpu"]
    lines = [
        "# Public aIDT measured-data update loop",
        "",
        "This benchmark uses the public aIDT Diatom I measured intensity stack as the target. The adapter Fourier-transforms each measured frame, samples a polar detector cap, and then solves a linear measured-target update problem `min ||A x - y_measured||^2` with the same geometry for the prepared GPU operator and cuFINUFFT GPU Plan.",
        "",
        "This is a measured-data update loop, not yet a full physical aIDT intensity reconstruction.",
        "",
        "## Configuration",
        "",
        f"- device: `{summary['device_name']}`",
        f"- dtype: `{summary['dtype']}`",
        f"- cap: `{summary['cap_radial']} x {summary['cap_phi']}`",
        f"- illumination frames: `{summary['n_illum']}`",
        f"- measured samples: `{summary['q_samples']}`",
        f"- object bins: `{summary['object_bins']}`",
        f"- iterations: `{summary['iterations']}`",
        f"- warmup updates: `{summary['warmup_updates']}`",
        f"- detector NA: `{summary['detector_na']}`",
        f"- illumination NA mean/std: `{summary['illumination_na_mean']:.6g}` / `{summary['illumination_na_std']:.6g}`",
        f"- prepared setup: `{summary['prepared_setup_s']:.6f}` s",
        f"- cuFINUFFT setup: `{summary['cufinufft_setup_s']:.6f}` s",
        "",
        "## Update Loop",
        "",
        "| method | final loss rel | median iter ms | median adjoint ms | median forward ms | cumulative iter s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        "| prepared GPU | {loss:.6g} | {iter_ms:.3f} | {adj_ms:.3f} | {fw_ms:.3f} | {cum:.6f} |".format(
            loss=prepared["final_loss_rel"],
            iter_ms=1000.0 * prepared["median_iter_s"],
            adj_ms=1000.0 * prepared["median_adjoint_s"],
            fw_ms=1000.0 * prepared["median_forward_s"],
            cum=prepared["cumulative_iter_s"],
        ),
        "| cuFINUFFT GPU | {loss:.6g} | {iter_ms:.3f} | {adj_ms:.3f} | {fw_ms:.3f} | {cum:.6f} |".format(
            loss=cufinufft["final_loss_rel"],
            iter_ms=1000.0 * cufinufft["median_iter_s"],
            adj_ms=1000.0 * cufinufft["median_adjoint_s"],
            fw_ms=1000.0 * cufinufft["median_forward_s"],
            cum=cufinufft["cumulative_iter_s"],
        ),
        "",
        "## Readout",
        "",
        f"- cuFINUFFT / prepared median iteration time: `{summary['cufinufft_iter_speedup_vs_prepared']:.3f}x`.",
        f"- cuFINUFFT / prepared cumulative loop time: `{summary['cufinufft_cumulative_speedup_vs_prepared']:.3f}x`.",
        f"- final loss absolute delta: `{summary['final_loss_abs_delta']:.6g}`.",
        f"- final object-update rel-L2 delta: `{summary['final_object_update_rel_l2']:.6g}`.",
        "",
        "## Boundary",
        "",
        "- The measured data affect the target vector and all residuals in the loop.",
        "- The update model is a linear deconvolution/update rule on the Fourier-domain measured residual, not a calibrated nonlinear aIDT solver with background, pupil, and regularization.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    adapter_start = time.perf_counter()
    measured_residual, adapter_meta = measured_polar_residual_from_contract(
        args.contract,
        cap_radial=args.cap_radial,
        cap_phi=args.cap_phi,
        detector_na=args.detector_na,
        fft_norm=args.fft_norm,
        interpolation_order=args.interpolation_order,
        normalize=not args.no_residual_normalize,
        max_frames=args.max_frames,
    )
    adapter_s = time.perf_counter() - adapter_start
    op_args = build_operator_args(args, measured_residual)

    context_start = time.perf_counter()
    ctx = build_context(op_args)
    context_s = time.perf_counter() - context_start

    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("this benchmark requires a CUDA device")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    cp, cufinufft = import_cufinufft_modules()

    prepared_start = time.perf_counter()
    torch_plan = TorchConeAxisOdtPlan.from_context(ctx, torch=torch, device=device, dtype=args.dtype)
    prepared_setup_s = time.perf_counter() - prepared_start

    complex_dtype, _, np_complex, np_real = torch_dtypes(torch, args.dtype)
    cu_complex_dtype, _, _, _ = cupy_dtypes(cp, args.dtype)
    simple_ctx = SimpleNamespace(obj=ctx.obj, flat_q=ctx.flat_q)
    cu_start = time.perf_counter()
    cu_block = make_block(
        cp,
        cufinufft,
        simple_ctx,
        dtype=args.dtype,
        complex_dtype=cu_complex_dtype,
        np_real=np_real,
        plan_mode=args.cufinufft_plan_mode,
        eps=args.cufinufft_eps,
    )
    cufinufft_setup_s = time.perf_counter() - cu_start
    cu_op = SimpleNamespace(cp=cp, cufinufft=cufinufft, ring=cu_block, axis=None)

    residual_np = measured_residual.residual.astype(np_complex, copy=False)
    data_t = torch.as_tensor(residual_np, dtype=complex_dtype, device=device)
    data_cp = cp.asarray(residual_np)

    prepared_result = run_prepared_update_loop(
        torch=torch,
        device=device,
        plan=torch_plan,
        data=data_t,
        iterations=args.iterations,
        warmup_updates=args.warmup_updates,
    )
    def sync_cufinufft() -> None:
        synchronize_cupy(cp)
        synchronize(torch, device)

    cufinufft_result = run_torch_update_loop(
        label="cufinufft_gpu",
        torch=torch,
        device=device,
        coeff_shape=tuple(ctx.obj.coeff.shape),
        coeff_dtype=complex_dtype,
        data=data_t,
        forward=lambda coeff: cufinufft_forward_torch(
            torch,
            cu_op,
            coeff,
            eps=args.cufinufft_eps,
        ),
        adjoint=lambda residual: cufinufft_adjoint_torch(
            torch,
            cu_op,
            residual,
            eps=args.cufinufft_eps,
            coeff_shape=tuple(ctx.obj.coeff.shape),
        ),
        sync=sync_cufinufft,
        iterations=args.cufinufft_iterations or args.iterations,
        warmup_updates=args.warmup_updates,
    )

    rows = prepared_result.rows + cufinufft_result.rows
    prepared_stats = method_stats(rows, "prepared_gpu")
    cufinufft_stats = method_stats(rows, "cufinufft_gpu")
    final_loss_abs_delta = abs(prepared_stats["final_loss_rel"] - cufinufft_stats["final_loss_rel"])
    object_update_rel = rel_l2(
        cufinufft_result.x_cpu.astype(np.complex128, copy=False),
        prepared_result.x_cpu.astype(np.complex128, copy=False),
    )
    summary = {
        **adapter_meta,
        "device": str(device),
        "device_name": device_name(torch, device),
        "torch_version": package_version("torch"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cupy_version": getattr(cp, "__version__", None),
        "cufinufft_version": getattr(cufinufft, "__version__", None),
        "dtype": args.dtype,
        "cufinufft_eps": float(args.cufinufft_eps),
        "cufinufft_plan_mode": args.cufinufft_plan_mode,
        "adapter_s": float(adapter_s),
        "context_build_s": float(context_s),
        "prepared_setup_s": float(prepared_setup_s),
        "cufinufft_setup_s": float(cufinufft_setup_s),
        "k": float(args.k),
        "cap_radial": int(args.cap_radial),
        "cap_phi": int(args.cap_phi),
        "n_illum": int(measured_residual.frame_order.size),
        "q_samples": int(measured_residual.q_count),
        "object_bins": int(np.prod(ctx.obj.coeff.shape)),
        "n_r": int(args.n_r),
        "n_z": int(args.n_z),
        "n_beta": int(args.n_beta),
        "iterations": int(args.iterations),
        "warmup_updates": int(args.warmup_updates),
        "prepared_gpu": prepared_stats,
        "cufinufft_gpu": cufinufft_stats,
        "cufinufft_iter_speedup_vs_prepared": float(
            cufinufft_stats["median_iter_s"] / prepared_stats["median_iter_s"]
        ),
        "cufinufft_cumulative_speedup_vs_prepared": float(
            cufinufft_stats["cumulative_iter_s"] / prepared_stats["cumulative_iter_s"]
        ),
        "final_loss_abs_delta": float(final_loss_abs_delta),
        "final_object_update_rel_l2": float(object_update_rel),
        "torch_peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
        "cupy_pool_total_mib": float(cp.get_default_memory_pool().total_bytes() / (1024.0 * 1024.0)),
        "history_csv": str(args.csv),
        "summary_md": str(args.summary_md),
    }
    write_csv(args.csv, rows)
    write_json(args.out, {"config": vars(args), "summary": summary, "history": rows})
    if args.summary_md:
        write_summary(args.summary_md, summary, rows)
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a measured-target update loop on public aIDT data with prepared GPU and cuFINUFFT GPU operators.")
    p.add_argument("--contract", type=Path, default=ROOT / "benchmark_results" / "aidt_diatom_public_contract.npz")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    p.add_argument("--cap-radial", type=int, default=32)
    p.add_argument("--cap-phi", type=int, default=128)
    p.add_argument("--detector-na", type=float, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--fft-norm", choices=["none", "ortho"], default="ortho")
    p.add_argument("--interpolation-order", type=int, choices=[0, 1, 3], default=1)
    p.add_argument("--no-residual-normalize", action="store_true")
    p.add_argument("--n-beta", type=int, default=256)
    p.add_argument("--n-r", type=int, default=12)
    p.add_argument("--n-z", type=int, default=11)
    p.add_argument("--r-max", type=float, default=1.0)
    p.add_argument("--z-max", type=float, default=0.8)
    p.add_argument("--phantom", choices=["beads", "random_beads", "shell"], default="random_beads")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--k", type=float, default=17.93636473255199)
    p.add_argument("--h-cutoff", type=int, default=None)
    p.add_argument("--h-margin", type=int, default=20)
    p.add_argument("--l-margin", type=int, default=18)
    p.add_argument("--cone-l-prune-threshold", type=float, default=1e-12)
    p.add_argument("--cpp-threads", type=int, default=16)
    p.add_argument("--forward-kernel-mode", choices=["compact", "partitioned"], default="partitioned")
    p.add_argument("--native-prepared-plan-mode", choices=["auto", "direct", "gathered", "gathered-zmajor"], default="auto")
    p.add_argument("--native-prepared-gather-threshold", type=int, default=8192)
    p.add_argument("--cufinufft-eps", type=float, default=1e-6)
    p.add_argument("--cufinufft-plan-mode", choices=["plan", "simple"], default="plan")
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--warmup-updates", type=int, default=1)
    p.add_argument("--cufinufft-iterations", type=int, default=None)
    p.add_argument("--out", type=Path, default=ROOT / "benchmark_results" / "aidt_public_measured_update_loop.json")
    p.add_argument("--csv", type=Path, default=ROOT / "benchmark_results" / "aidt_public_measured_update_loop_history.csv")
    p.add_argument("--summary-md", type=Path, default=ROOT / "benchmark_results" / "aidt_public_measured_update_loop.md")
    return p


def main() -> None:
    args = parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
