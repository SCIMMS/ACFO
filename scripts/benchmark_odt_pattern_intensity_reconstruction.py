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
    import_cufinufft_modules,
    make_cufinufft_composite,
    synchronize_cupy,
)
from benchmark_odt_pattern_layer_gpu import (  # noqa: E402
    TorchPatternedRingPlan,
    cufinufft_adjoint_torch,
    cufinufft_forward_torch,
    fmt,
    make_pattern_matrix_np,
    parse_case,
    speedup,
)
from benchmark_odt_realistic_geometry_reconstruction import build_composite_context  # noqa: E402
from benchmark_odt_torch_gpu_reconstruction import (  # noqa: E402
    TorchCompositeOdtPlan,
    torch_dtypes,
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def coeff_norm(torch: Any, value: Any) -> Any:
    return torch.clamp(torch.linalg.vector_norm(value.reshape(-1)), min=1e-30)


def aligned_object_error(torch: Any, candidate: Any, truth: Any) -> float:
    cand = candidate.reshape(-1)
    ref = truth.reshape(-1)
    phase = torch.sum(torch.conj(cand) * ref)
    phase = phase / torch.clamp(torch.abs(phase), min=1e-30)
    aligned = candidate * phase
    return float((coeff_norm(torch, aligned - truth) / coeff_norm(torch, truth)).detach().cpu().item())


def random_warm_start(torch: Any, device: Any, true_coeff: Any, *, rel_noise: float, seed: int) -> Any:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    real_noise = torch.randn(true_coeff.shape, generator=generator, device=device, dtype=true_coeff.real.dtype)
    imag_noise = torch.randn(true_coeff.shape, generator=generator, device=device, dtype=true_coeff.real.dtype)
    noise = torch.complex(real_noise, imag_noise).to(dtype=true_coeff.dtype)
    noise = noise * (float(rel_noise) * coeff_norm(torch, true_coeff) / coeff_norm(torch, noise))
    return true_coeff + noise


class PatternIntensityOperator:
    def __init__(
        self,
        *,
        label: str,
        torch: Any,
        device: Any,
        plan: TorchPatternedRingPlan,
        cufinufft: Any | None,
        cufinufft_eps: float,
        coeff_shape: tuple[int, ...],
        use_cufinufft: bool,
    ) -> None:
        self.label = label
        self.torch = torch
        self.device = device
        self.plan = plan
        self.cufinufft = cufinufft
        self.cufinufft_eps = float(cufinufft_eps)
        self.coeff_shape = coeff_shape
        self.use_cufinufft = bool(use_cufinufft)

    def stack_forward(self, coeff: Any) -> Any:
        if not self.use_cufinufft:
            return self.plan.stack_forward(coeff)
        if self.cufinufft is None:
            raise RuntimeError("missing cuFINUFFT operator")
        return cufinufft_forward_torch(
            self.torch,
            self.cufinufft,
            coeff,
            eps=self.cufinufft_eps,
        ).reshape(self.plan.n_illum, self.plan.base_pixels)

    def stack_adjoint(self, ring_residual: Any) -> Any:
        if not self.use_cufinufft:
            return self.plan.ring.adjoint(ring_residual.reshape(-1))
        if self.cufinufft is None:
            raise RuntimeError("missing cuFINUFFT operator")
        return cufinufft_adjoint_torch(
            self.torch,
            self.cufinufft,
            ring_residual.reshape(-1),
            eps=self.cufinufft_eps,
            coeff_shape=self.coeff_shape,
        )

    def coherent_eval_grad(self, coeff: Any, data: Any) -> tuple[Any, Any, Any]:
        stack = self.stack_forward(coeff)
        field = self.plan.mix_stack(stack)
        intensity = self.torch.real(field.conj() * field)
        residual = intensity - data
        weighted_field = residual.to(dtype=field.dtype) * field
        ring_residual = self.plan.backmix(weighted_field.reshape(-1))
        grad = self.stack_adjoint(ring_residual)
        return residual, grad, intensity

    def incoherent_eval_grad(self, coeff: Any, data: Any) -> tuple[Any, Any, Any]:
        stack = self.stack_forward(coeff)
        stack_intensity = self.torch.real(stack.conj() * stack)
        intensity = self.plan.incoherent_weights @ stack_intensity
        residual = intensity - data
        weighted_stack = (self.plan.incoherent_weights.transpose(0, 1) @ residual).to(
            dtype=stack.dtype
        ) * stack
        grad = self.stack_adjoint(weighted_stack)
        return residual, grad, intensity


def run_intensity_loop(
    *,
    operator: PatternIntensityOperator,
    model: str,
    x0: Any,
    true_coeff: Any,
    data: Any,
    iterations: int,
    step_rel: float,
    sync: Callable[[], None],
) -> list[dict[str, Any]]:
    torch = operator.torch
    x = x0.clone()
    data_norm = coeff_norm(torch, data)
    true_norm = coeff_norm(torch, true_coeff)
    eval_grad = operator.coherent_eval_grad if model == "coherent" else operator.incoherent_eval_grad
    rows: list[dict[str, Any]] = []
    cumulative = 0.0
    for iteration in range(1, int(iterations) + 1):
        sync()
        start = time.perf_counter()
        fw_start = time.perf_counter()
        residual, grad, _ = eval_grad(x, data)
        sync()
        forward_grad_s = time.perf_counter() - fw_start
        grad_norm = coeff_norm(torch, grad)
        step = float(step_rel) * true_norm / grad_norm
        x = x - step * grad
        sync()
        elapsed = time.perf_counter() - start
        cumulative += elapsed
        loss = coeff_norm(torch, residual) / data_norm
        rows.append(
            {
                "method": operator.label,
                "model": model,
                "iteration": int(iteration),
                "loss_rel": float(loss.detach().cpu().item()),
                "object_rel_l2_aligned": aligned_object_error(torch, x, true_coeff),
                "step": float(step.detach().cpu().item()) if hasattr(step, "detach") else float(step),
                "grad_norm_over_true_norm": float((grad_norm / true_norm).detach().cpu().item()),
                "iter_s": float(elapsed),
                "forward_grad_s": float(forward_grad_s),
                "cumulative_iter_s": float(cumulative),
            }
        )
    return rows


def method_stats(rows: list[dict[str, Any]], *, method: str, model: str) -> dict[str, Any]:
    selected = [row for row in rows if row["method"] == method and row["model"] == model]
    if not selected:
        raise ValueError(f"no rows for {method}/{model}")
    final = selected[-1]
    return {
        "iterations": int(final["iteration"]),
        "final_loss_rel": float(final["loss_rel"]),
        "final_object_rel_l2_aligned": float(final["object_rel_l2_aligned"]),
        "median_iter_s": float(median(float(row["iter_s"]) for row in selected)),
        "cumulative_iter_s": float(final["cumulative_iter_s"]),
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["history"]
    summaries = payload["summary_rows"]
    lines = [
        "# ODT pattern-intensity inverse-loop proxy",
        "",
        "This benchmark runs warm-start intensity-only gradient updates on top of the validated FPDT/FS-ODT fixed-pattern layer.",
        "It is still synthetic self-consistency data, but it is a full nonlinear update loop rather than only a forward-cost diagnostic.",
        "",
        "## Results",
        "",
        "| case | model | updates | ours median update ms | cuFINUFFT median update ms | update speedup | ours final loss | cu final loss | final loss delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| {case} | {model} | {it} | `{oms}` | `{cms}` | `{sp}` | `{oloss}` | `{closs}` | `{delta}` |".format(
                case=row["case"],
                model=row["model"],
                it=row["iterations"],
                oms=fmt(1000.0 * row["ours_median_iter_s"], 4),
                cms=fmt(1000.0 * row["cufinufft_median_iter_s"], 4),
                sp=fmt(row["update_speedup"], 4),
                oloss=fmt(row["ours_final_loss_rel"], 4),
                closs=fmt(row["cufinufft_final_loss_rel"], 4),
                delta=fmt(row["final_loss_abs_delta"], 4),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `coherent` uses `|P A_stack x|^2` and the manual Wirtinger-gradient proxy `A_stack* P* ((I - I_target) field)`.",
            "- `incoherent` uses `P_abs |A_stack x|^2` and backprojects the intensity residual through the fixed nonnegative pattern weights.",
            "- Both loops use the same warm start, same target data, same step rule, and same iteration count for the prepared GPU operator and the cuFINUFFT GPU Plan baseline.",
            "- This is not yet a real-data FPDT/FS-ODT solver; it tests whether the nonlinear intensity update loop preserves the prepared-operator advantage.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path: Path, summaries: list[dict[str, Any]]) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    import matplotlib.pyplot as plt

    labels = [f"{row['case']}\n{row['model']}" for row in summaries]
    values = [row["update_speedup"] for row in summaries]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(np.arange(len(labels)), values)
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("cuFINUFFT update time / ours update time")
    ax.set_title("ODT pattern-intensity inverse-loop speedup")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def benchmark_case(
    args: argparse.Namespace,
    *,
    label: str,
    n_illum: int,
    n_patterns: int,
    active: int,
    cap_radial: int,
    cap_phi: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("pattern-intensity reconstruction benchmark requires CUDA")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    case_args = argparse.Namespace(**vars(args))
    case_args.ring_illum = int(n_illum)
    case_args.cap_radial = int(cap_radial)
    case_args.cap_phi = int(cap_phi)
    case_args.skip_axis_illumination = True
    ctx = build_composite_context(case_args)
    composite = TorchCompositeOdtPlan.from_context(ctx, torch=torch, device=device, dtype=args.dtype)
    pattern_np = make_pattern_matrix_np(
        n_patterns=n_patterns,
        n_illum=n_illum,
        active=active,
        seed=args.seed + n_illum * 13 + n_patterns * 17 + active,
        kind=args.pattern_kind,
    )
    plan = TorchPatternedRingPlan.from_ring(
        torch=torch,
        device=device,
        ring=composite.ring,
        pattern_np=pattern_np,
        dtype=args.dtype,
    )

    _, _, np_complex, _ = torch_dtypes(torch, args.dtype)
    true_coeff_np = np.ascontiguousarray((ctx.ring.obj.coeff * args.object_scale).astype(np_complex, copy=False))
    true_coeff = torch.as_tensor(true_coeff_np, dtype=composite.complex_dtype, device=device)
    x0 = random_warm_start(
        torch,
        device,
        true_coeff,
        rel_noise=args.warm_start_noise_rel,
        seed=args.seed + 8009,
    )
    cp, _ = import_cufinufft_modules()
    cu_op = make_cufinufft_composite(
        ctx,
        dtype=args.dtype,
        plan_mode=args.cufinufft_plan_mode,
        eps=args.cufinufft_eps,
    )
    ours_op = PatternIntensityOperator(
        label="ours_gpu",
        torch=torch,
        device=device,
        plan=plan,
        cufinufft=None,
        cufinufft_eps=args.cufinufft_eps,
        coeff_shape=true_coeff_np.shape,
        use_cufinufft=False,
    )
    cu_operator = PatternIntensityOperator(
        label="cufinufft_gpu",
        torch=torch,
        device=device,
        plan=plan,
        cufinufft=cu_op,
        cufinufft_eps=args.cufinufft_eps,
        coeff_shape=true_coeff_np.shape,
        use_cufinufft=True,
    )

    def sync_ours() -> None:
        synchronize(torch, device)

    def sync_cu() -> None:
        synchronize_cupy(cp)
        synchronize(torch, device)

    history: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
        coherent_data = ours_op.coherent_eval_grad(true_coeff, torch.zeros((plan.n_patterns, plan.base_pixels), dtype=true_coeff.real.dtype, device=device))[2].detach()
        incoherent_data = ours_op.incoherent_eval_grad(true_coeff, torch.zeros((plan.n_patterns, plan.base_pixels), dtype=true_coeff.real.dtype, device=device))[2].detach()
        for model, data in (("coherent", coherent_data), ("incoherent", incoherent_data)):
            ours_rows = run_intensity_loop(
                operator=ours_op,
                model=model,
                x0=x0,
                true_coeff=true_coeff,
                data=data,
                iterations=args.iterations,
                step_rel=args.step_rel,
                sync=sync_ours,
            )
            cu_rows = run_intensity_loop(
                operator=cu_operator,
                model=model,
                x0=x0,
                true_coeff=true_coeff,
                data=data,
                iterations=args.iterations,
                step_rel=args.step_rel,
                sync=sync_cu,
            )
            for row in ours_rows + cu_rows:
                row.update(
                    {
                        "case": label,
                        "n_illum": int(n_illum),
                        "n_patterns": int(n_patterns),
                        "active_per_pattern": int(active),
                        "cap_radial": int(cap_radial),
                        "cap_phi": int(cap_phi),
                        "stack_q_samples": int(n_illum * plan.base_pixels),
                        "pattern_q_samples": int(plan.q_count),
                        "object_bins": int(true_coeff.numel()),
                    }
                )
            history.extend(ours_rows + cu_rows)
            ours_stats = method_stats(ours_rows, method="ours_gpu", model=model)
            cu_stats = method_stats(cu_rows, method="cufinufft_gpu", model=model)
            summary_rows.append(
                {
                    "case": label,
                    "model": model,
                    "iterations": int(args.iterations),
                    "n_illum": int(n_illum),
                    "n_patterns": int(n_patterns),
                    "active_per_pattern": int(active),
                    "cap_radial": int(cap_radial),
                    "cap_phi": int(cap_phi),
                    "stack_q_samples": int(n_illum * plan.base_pixels),
                    "pattern_q_samples": int(plan.q_count),
                    "object_bins": int(true_coeff.numel()),
                    "ours_median_iter_s": ours_stats["median_iter_s"],
                    "cufinufft_median_iter_s": cu_stats["median_iter_s"],
                    "update_speedup": speedup(cu_stats["median_iter_s"], ours_stats["median_iter_s"]),
                    "ours_cumulative_iter_s": ours_stats["cumulative_iter_s"],
                    "cufinufft_cumulative_iter_s": cu_stats["cumulative_iter_s"],
                    "cumulative_speedup": speedup(cu_stats["cumulative_iter_s"], ours_stats["cumulative_iter_s"]),
                    "ours_final_loss_rel": ours_stats["final_loss_rel"],
                    "cufinufft_final_loss_rel": cu_stats["final_loss_rel"],
                    "final_loss_abs_delta": abs(ours_stats["final_loss_rel"] - cu_stats["final_loss_rel"]),
                    "ours_final_object_rel_l2_aligned": ours_stats["final_object_rel_l2_aligned"],
                    "cufinufft_final_object_rel_l2_aligned": cu_stats["final_object_rel_l2_aligned"],
                    "device_name": device_name(torch, device),
                    "torch_version": package_version("torch"),
                    "torch_cuda_version": getattr(torch.version, "cuda", None),
                    "cupy_version": getattr(cp, "__version__", None),
                    "cufinufft_version": getattr(cu_op.cufinufft, "__version__", None),
                    "dtype": args.dtype,
                    "pattern_kind": args.pattern_kind,
                    "warm_start_noise_rel": float(args.warm_start_noise_rel),
                    "step_rel": float(args.step_rel),
                    "torch_peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
                    "cupy_pool_total_mib": float(cp.get_default_memory_pool().total_bytes() / (1024.0 * 1024.0)),
                }
            )
    return history, summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run warm-start intensity-only inverse-loop proxies on the fixed ODT pattern layer."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["medium=64:16:8:32x128"],
        help="Cases as label=n_illum:n_patterns:active:cap_radialxcap_phi.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    parser.add_argument("--pattern-kind", choices=["phase", "binary", "rolling"], default="phase")
    parser.add_argument("--n-beta", type=int, default=256)
    parser.add_argument("--n-r", type=int, default=12)
    parser.add_argument("--n-z", type=int, default=11)
    parser.add_argument("--r-max", type=float, default=1.0)
    parser.add_argument("--z-max", type=float, default=0.8)
    parser.add_argument("--phantom", choices=["beads", "random_beads", "shell"], default="random_beads")
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--k", type=float, default=17.307319527958313)
    parser.add_argument("--detector-na", type=float, default=0.9240924092409241)
    parser.add_argument("--illumination-angle-deg", type=float, default=49.0)
    parser.add_argument("--h-cutoff", type=int, default=None)
    parser.add_argument("--h-margin", type=int, default=20)
    parser.add_argument("--l-margin", type=int, default=18)
    parser.add_argument("--cone-l-prune-threshold", type=float, default=1e-12)
    parser.add_argument("--cpp-threads", type=int, default=16)
    parser.add_argument("--forward-execute-mode", choices=["prepared", "wrapper"], default="prepared")
    parser.add_argument("--forward-kernel-mode", choices=["compact", "partitioned"], default="partitioned")
    parser.add_argument(
        "--native-prepared-plan-mode",
        choices=["auto", "direct", "gathered", "gathered-zmajor"],
        default="auto",
    )
    parser.add_argument("--native-prepared-gather-threshold", type=int, default=8192)
    parser.add_argument("--finufft-eps", type=float, default=1e-12)
    parser.add_argument("--finufft-q-batch-size", type=int, default=1_048_576)
    parser.add_argument("--cufinufft-eps", type=float, default=1e-6)
    parser.add_argument("--cufinufft-plan-mode", choices=["plan", "simple"], default="plan")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--step-rel", type=float, default=0.04)
    parser.add_argument("--warm-start-noise-rel", type=float, default=0.15)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_pattern_intensity_reconstruction",
    )
    args = parser.parse_args()
    args.cases = [parse_case(value) for value in args.cases]
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.step_rel <= 0.0:
        raise ValueError("step-rel must be positive")
    if args.warm_start_noise_rel < 0.0:
        raise ValueError("warm-start-noise-rel must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    history: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for label, n_illum, n_patterns, active, cap_radial, cap_phi in args.cases:
        case_history, case_summary = benchmark_case(
            args,
            label=label,
            n_illum=n_illum,
            n_patterns=n_patterns,
            active=active,
            cap_radial=cap_radial,
            cap_phi=cap_phi,
        )
        history.extend(case_history)
        summary_rows.extend(case_summary)
    prefix = args.output_prefix if args.output_prefix.is_absolute() else ROOT / args.output_prefix
    payload = {
        "config": {
            "cases": [
                {
                    "label": label,
                    "n_illum": n_illum,
                    "n_patterns": n_patterns,
                    "active": active,
                    "cap_radial": cap_radial,
                    "cap_phi": cap_phi,
                }
                for label, n_illum, n_patterns, active, cap_radial, cap_phi in args.cases
            ],
            "dtype": args.dtype,
            "pattern_kind": args.pattern_kind,
            "iterations": args.iterations,
            "step_rel": args.step_rel,
            "warm_start_noise_rel": args.warm_start_noise_rel,
            "cufinufft_eps": args.cufinufft_eps,
            "cufinufft_plan_mode": args.cufinufft_plan_mode,
        },
        "summary_rows": summary_rows,
        "history": history,
    }
    write_json(prefix.with_suffix(".json"), payload)
    write_csv(prefix.with_suffix(".csv"), summary_rows)
    write_csv(prefix.with_name(prefix.name + "_history.csv"), history)
    write_summary(prefix.with_suffix(".md"), payload)
    write_plot(prefix.with_suffix(".png"), summary_rows)
    print(json.dumps({"summary_rows": len(summary_rows), "history_rows": len(history)}, indent=2))


if __name__ == "__main__":
    main()
