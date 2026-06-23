from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from statistics import median
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

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
    import_cufinufft_modules,
    make_cufinufft_composite,
    synchronize_cupy,
)
from benchmark_odt_realistic_geometry_reconstruction import (  # noqa: E402
    build_composite_context,
    composite_adjoint,
    composite_forward,
)
from benchmark_odt_torch_gpu_reconstruction import (  # noqa: E402
    TorchCompositeOdtPlan,
    rel_l2,
    torch_dtypes,
)


def coeff_norm2_torch(torch: Any, value: Any) -> Any:
    return torch.sum(torch.real(torch.conj(value) * value))


def torch_to_cupy(cp: Any, value: Any) -> Any:
    return cp.from_dlpack(value.detach().contiguous())


def cupy_to_torch(torch: Any, value: Any) -> Any:
    return torch.from_dlpack(value)


def cufinufft_forward_torch(
    torch: Any,
    cu_op: Any,
    coeff: Any,
    *,
    eps: float,
) -> Any:
    cp = cu_op.cp
    coeff_gpu = torch_to_cupy(cp, coeff.reshape(-1))
    parts_cp = [cufinufft_forward_block(cu_op, cu_op.ring, coeff_gpu, eps=eps)]
    if cu_op.axis is not None:
        parts_cp.append(cufinufft_forward_block(cu_op, cu_op.axis, coeff_gpu, eps=eps))
    synchronize_cupy(cp)
    parts = [cupy_to_torch(torch, part) for part in parts_cp]
    if len(parts) == 1:
        return parts[0]
    return torch.cat(parts, dim=0)


def cufinufft_adjoint_torch(
    torch: Any,
    cu_op: Any,
    residual: Any,
    *,
    eps: float,
    coeff_shape: tuple[int, ...],
) -> Any:
    cp = cu_op.cp
    ring_count = int(cu_op.ring.q_count)
    ring_residual = torch_to_cupy(cp, residual[:ring_count])
    parts_cp = [cufinufft_adjoint_block(cu_op, cu_op.ring, ring_residual, eps=eps)]
    if cu_op.axis is not None:
        axis_residual = torch_to_cupy(cp, residual[ring_count:])
        parts_cp.append(cufinufft_adjoint_block(cu_op, cu_op.axis, axis_residual, eps=eps))
    synchronize_cupy(cp)
    grad = cupy_to_torch(torch, parts_cp[0]).reshape(coeff_shape)
    if len(parts_cp) > 1:
        grad = grad + cupy_to_torch(torch, parts_cp[1]).reshape(coeff_shape)
    return grad


def run_torch_resident_reconstruction(
    *,
    label: str,
    torch: Any,
    device: Any,
    true_coeff: Any,
    data: Any,
    forward: Callable[[Any], Any],
    adjoint: Callable[[Any], Any],
    sync: Callable[[], None],
    iterations: int,
) -> tuple[list[dict[str, Any]], Any]:
    x = torch.zeros_like(true_coeff)
    pred = torch.zeros_like(data)
    data_norm = torch.clamp(torch.linalg.vector_norm(data), min=1e-30)
    true_norm = torch.clamp(torch.linalg.vector_norm(true_coeff), min=1e-30)
    rows: list[dict[str, Any]] = [
        {
            "method": label,
            "iteration": 0,
            "loss_rel": 1.0,
            "object_rel_l2": 1.0,
            "alpha": 0.0,
            "iter_s": 0.0,
            "update_s": 0.0,
            "adjoint_s": 0.0,
            "line_forward_s": 0.0,
            "cumulative_iter_s": 0.0,
        }
    ]
    cumulative = 0.0
    with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
        for iteration in range(1, int(iterations) + 1):
            residual = pred - data
            sync()
            iter_start = time.perf_counter()

            adj_start = time.perf_counter()
            grad = adjoint(residual)
            sync()
            adjoint_s = time.perf_counter() - adj_start

            fw_start = time.perf_counter()
            a_grad = forward(grad)
            sync()
            forward_s = time.perf_counter() - fw_start

            alpha = coeff_norm2_torch(torch, grad) / torch.clamp(
                coeff_norm2_torch(torch, a_grad),
                min=1e-30,
            )
            x = x - alpha * grad
            pred = pred - alpha * a_grad
            sync()
            update_s = time.perf_counter() - iter_start
            loss = torch.linalg.vector_norm(pred - data) / data_norm
            obj_err = torch.linalg.vector_norm(x - true_coeff) / true_norm
            sync()
            elapsed = time.perf_counter() - iter_start
            cumulative += elapsed
            rows.append(
                {
                    "method": label,
                    "iteration": int(iteration),
                    "loss_rel": float(loss.detach().cpu().item()),
                    "object_rel_l2": float(obj_err.detach().cpu().item()),
                    "alpha": float(alpha.detach().cpu().item()),
                    "iter_s": float(elapsed),
                    "update_s": float(update_s),
                    "adjoint_s": float(adjoint_s),
                    "line_forward_s": float(forward_s),
                    "cumulative_iter_s": float(cumulative),
                }
            )
    return rows, x


def coeff_norm2_np(value: np.ndarray) -> float:
    return float(np.vdot(value.ravel(), value.ravel()).real)


def run_cpu_structured_reconstruction(
    *,
    ctx: Any,
    args: argparse.Namespace,
    true_coeff: np.ndarray,
    data: np.ndarray,
    iterations: int,
) -> list[dict[str, Any]]:
    x = np.zeros_like(true_coeff)
    pred = np.zeros_like(data)
    data_norm = max(float(np.linalg.norm(data.ravel())), 1e-300)
    true_norm = max(float(np.linalg.norm(true_coeff.ravel())), 1e-300)
    rows: list[dict[str, Any]] = [
        {
            "method": "cpu_structured",
            "iteration": 0,
            "loss_rel": 1.0,
            "object_rel_l2": 1.0,
            "alpha": 0.0,
            "iter_s": 0.0,
            "update_s": 0.0,
            "adjoint_s": 0.0,
            "line_forward_s": 0.0,
            "cumulative_iter_s": 0.0,
        }
    ]
    cumulative = 0.0
    for iteration in range(1, int(iterations) + 1):
        residual = pred - data
        iter_start = time.perf_counter()
        adj_start = time.perf_counter()
        grad = composite_adjoint(ctx, residual, args, use_finufft=False)
        adjoint_s = time.perf_counter() - adj_start
        fw_start = time.perf_counter()
        a_grad = composite_forward(ctx, grad, args, use_finufft=False)
        forward_s = time.perf_counter() - fw_start
        alpha = coeff_norm2_np(grad) / max(coeff_norm2_np(a_grad), 1e-300)
        x = x - alpha * grad
        pred = pred - alpha * a_grad
        update_s = time.perf_counter() - iter_start
        elapsed = time.perf_counter() - iter_start
        cumulative += elapsed
        rows.append(
            {
                "method": "cpu_structured",
                "iteration": int(iteration),
                "loss_rel": float(np.linalg.norm((pred - data).ravel()) / data_norm),
                "object_rel_l2": float(np.linalg.norm((x - true_coeff).ravel()) / true_norm),
                "alpha": float(alpha),
                "iter_s": float(elapsed),
                "update_s": float(update_s),
                "adjoint_s": float(adjoint_s),
                "line_forward_s": float(forward_s),
                "cumulative_iter_s": float(cumulative),
            }
        )
    return rows


def method_stats(rows: list[dict[str, Any]], method: str) -> dict[str, Any] | None:
    selected = [row for row in rows if row["method"] == method]
    if not selected:
        return None
    active = [row for row in selected if int(row["iteration"]) > 0]
    final = max(selected, key=lambda row: int(row["iteration"]))
    return {
        "iterations": int(final["iteration"]),
        "final_loss_rel": float(final["loss_rel"]),
        "final_object_rel_l2": float(final["object_rel_l2"]),
        "median_iter_s": None if not active else float(median(float(row["iter_s"]) for row in active)),
        "median_update_s": None if not active else float(median(float(row["update_s"]) for row in active)),
        "median_adjoint_s": None if not active else float(median(float(row["adjoint_s"]) for row in active)),
        "median_forward_s": None
        if not active
        else float(median(float(row["line_forward_s"]) for row in active)),
        "cumulative_iter_s": float(final["cumulative_iter_s"]),
    }


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


def write_summary(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    methods = [method for method in ("ours_gpu", "cufinufft_gpu", "cpu_structured") if method_stats(rows, method)]
    lines = [
        "# ODT GPU reconstruction comparison",
        "",
        "This benchmark runs the same steepest-descent reconstruction loop with a structured PyTorch GPU operator and a cuFINUFFT type-3 GPU Plan baseline. CPU structured reconstruction is optional and included when requested.",
        "",
        "## Configuration",
        "",
        f"- device: `{summary['device_name']}`",
        f"- dtype: `{summary['dtype']}`",
        f"- q samples: `{summary['total_q_samples']}`",
        f"- object bins: `{summary['object_bins']}`",
        f"- cap: `{summary['cap_radial']} x {summary['cap_phi']}`",
        f"- total illuminations: `{summary['total_illumination_count']}`",
        f"- ours setup: `{summary['ours_setup_s']:.6f}` s",
        f"- cuFINUFFT setup: `{summary['cufinufft_setup_s']:.6f}` s",
        f"- cuFINUFFT forward data rel-L2 vs ours: `{summary['cufinufft_data_rel_l2_vs_ours']:.6g}`",
        "",
        "## Reconstruction",
        "",
        "| method | iterations | final loss rel | final object rel-L2 | median update ms | median iter ms | median adjoint ms | median forward ms | cumulative iter s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        stats = method_stats(rows, method)
        if stats is None:
            continue
        lines.append(
            "| {method} | {it} | {loss:.6g} | {obj:.6g} | {update_ms:.3f} | {iter_ms:.3f} | {adj_ms:.3f} | {fw_ms:.3f} | {cum:.6f} |".format(
                method=method,
                it=stats["iterations"],
                loss=stats["final_loss_rel"],
                obj=stats["final_object_rel_l2"],
                update_ms=1000.0 * stats["median_update_s"],
                iter_ms=1000.0 * stats["median_iter_s"],
                adj_ms=1000.0 * stats["median_adjoint_s"],
                fw_ms=1000.0 * stats["median_forward_s"],
                cum=stats["cumulative_iter_s"],
            )
        )
    if summary.get("cufinufft_iter_speedup_vs_ours") is not None:
        lines.extend(
            [
                "",
                "## Readout",
                "",
                f"- cuFINUFFT GPU / ours GPU median update time: `{summary['cufinufft_update_speedup_vs_ours']:.3f}x`.",
                f"- cuFINUFFT GPU / ours GPU median diagnostic iteration time: `{summary['cufinufft_iter_speedup_vs_ours']:.3f}x`.",
                f"- cuFINUFFT GPU / ours GPU cumulative reconstruction time: `{summary['cufinufft_cumulative_speedup_vs_ours']:.3f}x`.",
            ]
        )
    if summary.get("cpu_iter_speedup_vs_ours") is not None:
        lines.append(
            f"- CPU structured / ours GPU median iteration time: `{summary['cpu_iter_speedup_vs_ours']:.3f}x`."
        )
    lines.extend(
        [
            "- The same target data is used for both GPU reconstructions; small final-loss differences include the measured cuFINUFFT/operator numerical mismatch.",
            "- This is still a synthetic self-consistency reconstruction, but it directly tests the repeated forward-adjoint-update loop relevant to live ODT reconstruction.",
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
    cp, _ = import_cufinufft_modules()

    ctx = build_composite_context(args)
    start = time.perf_counter()
    torch_plan = TorchCompositeOdtPlan.from_context(ctx, torch=torch, device=device, dtype=args.dtype)
    ours_setup_s = time.perf_counter() - start
    start = time.perf_counter()
    cu_op = make_cufinufft_composite(
        ctx,
        dtype=args.dtype,
        plan_mode=args.cufinufft_plan_mode,
        eps=args.cufinufft_eps,
    )
    cufinufft_setup_s = time.perf_counter() - start

    _, _, np_complex, _ = torch_dtypes(torch, args.dtype)
    true_coeff_np = np.ascontiguousarray((ctx.ring.obj.coeff * float(args.object_scale)).astype(np_complex))
    true_coeff_t = torch.as_tensor(true_coeff_np, dtype=torch_plan.complex_dtype, device=device)

    def sync_ours() -> None:
        synchronize(torch, device)

    def sync_cu() -> None:
        synchronize_cupy(cp)
        synchronize(torch, device)

    with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
        data_t = torch_plan.forward(true_coeff_t).detach().clone()
        sync_ours()
        cu_data_t = cufinufft_forward_torch(torch, cu_op, true_coeff_t, eps=args.cufinufft_eps).detach().clone()
        sync_cu()
        data_norm = torch.clamp(torch.linalg.vector_norm(data_t), min=1e-30)
        cu_data_rel = float((torch.linalg.vector_norm(cu_data_t - data_t) / data_norm).detach().cpu().item())

        a_grad_buffer = torch.empty_like(data_t)

        def ours_forward(coeff: Any) -> Any:
            return torch_plan.forward_into(coeff, a_grad_buffer)

        def ours_adjoint(residual: Any) -> Any:
            return torch_plan.adjoint(residual)

        ours_rows, ours_x = run_torch_resident_reconstruction(
            label="ours_gpu",
            torch=torch,
            device=device,
            true_coeff=true_coeff_t,
            data=data_t,
            forward=ours_forward,
            adjoint=ours_adjoint,
            sync=sync_ours,
            iterations=args.iterations,
        )

        def cu_forward(coeff: Any) -> Any:
            return cufinufft_forward_torch(torch, cu_op, coeff, eps=args.cufinufft_eps)

        def cu_adjoint(residual: Any) -> Any:
            return cufinufft_adjoint_torch(
                torch,
                cu_op,
                residual,
                eps=args.cufinufft_eps,
                coeff_shape=true_coeff_np.shape,
            )

        cu_rows, cu_x = run_torch_resident_reconstruction(
            label="cufinufft_gpu",
            torch=torch,
            device=device,
            true_coeff=true_coeff_t,
            data=data_t,
            forward=cu_forward,
            adjoint=cu_adjoint,
            sync=sync_cu,
            iterations=args.cufinufft_iterations or args.iterations,
        )

    rows = ours_rows + cu_rows
    cpu_rows: list[dict[str, Any]] = []
    if args.include_cpu_structured:
        data_np = data_t.detach().cpu().numpy().astype(np.complex128, copy=False)
        cpu_rows = run_cpu_structured_reconstruction(
            ctx=ctx,
            args=args,
            true_coeff=true_coeff_np.astype(np.complex128, copy=False),
            data=data_np,
            iterations=args.cpu_iterations or args.iterations,
        )
        rows.extend(cpu_rows)

    ours_stats = method_stats(rows, "ours_gpu")
    cu_stats = method_stats(rows, "cufinufft_gpu")
    cpu_stats = method_stats(rows, "cpu_structured")
    summary = {
        "device": str(device),
        "device_name": device_name(torch, device),
        "torch_version": package_version("torch"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cupy_version": getattr(cp, "__version__", None),
        "cufinufft_version": getattr(cu_op.cufinufft, "__version__", None),
        "dtype": args.dtype,
        "cufinufft_eps": float(args.cufinufft_eps),
        "cufinufft_plan_mode": args.cufinufft_plan_mode,
        "cap_radial": int(args.cap_radial),
        "cap_phi": int(args.cap_phi),
        "ring_illum": int(args.ring_illum),
        "axis_illumination_included": not bool(args.skip_axis_illumination),
        "total_illumination_count": int(args.ring_illum) + (0 if args.skip_axis_illumination else 1),
        "total_q_samples": int(torch_plan.q_count),
        "object_bins": int(true_coeff_np.size),
        "ours_setup_s": float(ours_setup_s),
        "cufinufft_setup_s": float(cufinufft_setup_s),
        "cufinufft_data_rel_l2_vs_ours": float(cu_data_rel),
        "ours": ours_stats,
        "cufinufft_gpu": cu_stats,
        "cpu_structured": cpu_stats,
        "cufinufft_iter_speedup_vs_ours": None
        if ours_stats is None or cu_stats is None
        else float(cu_stats["median_iter_s"] / ours_stats["median_iter_s"]),
        "cufinufft_update_speedup_vs_ours": None
        if ours_stats is None or cu_stats is None
        else float(cu_stats["median_update_s"] / ours_stats["median_update_s"]),
        "cufinufft_cumulative_speedup_vs_ours": None
        if ours_stats is None or cu_stats is None
        else float(cu_stats["cumulative_iter_s"] / ours_stats["cumulative_iter_s"]),
        "cpu_iter_speedup_vs_ours": None
        if ours_stats is None or cpu_stats is None
        else float(cpu_stats["median_iter_s"] / ours_stats["median_iter_s"]),
        "ours_gpu_peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
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
    p = argparse.ArgumentParser(
        description="Compare ODT reconstruction loops using structured GPU and cuFINUFFT GPU operators."
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
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
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--cufinufft-iterations", type=int, default=None)
    p.add_argument("--include-cpu-structured", action="store_true")
    p.add_argument("--cpu-iterations", type=int, default=None)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_gpu_reconstruction_compare.json",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_gpu_reconstruction_compare_history.csv",
    )
    p.add_argument(
        "--summary-md",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_gpu_reconstruction_compare_summary.md",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
