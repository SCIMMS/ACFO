from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

from benchmark_odt_realistic_geometry_reconstruction import build_composite_context
from benchmark_odt_torch_gpu_reconstruction import (
    TorchCompositeOdtPlan,
    device_name,
    import_torch,
    parser as torch_parser,
    resolve_device,
    synchronize,
    torch_dtypes,
)


def comma_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("thresholds must be positive comma-separated floats")
    return values


def rel_l2(torch: Any, candidate: Any, reference: Any) -> float:
    numerator = torch.linalg.vector_norm(candidate - reference)
    denominator = torch.clamp(torch.linalg.vector_norm(reference), min=1e-30)
    return float((numerator / denominator).detach().cpu().item())


def timed_pair(
    torch: Any,
    device: Any,
    plan: TorchCompositeOdtPlan,
    coeff: Any,
    residual: Any,
    *,
    repeats: int,
    warmups: int,
) -> tuple[float, list[float]]:
    for _ in range(max(0, warmups)):
        plan.forward(coeff)
        plan.adjoint(residual)
        synchronize(torch, device)
    times: list[float] = []
    for _ in range(max(1, repeats)):
        synchronize(torch, device)
        start = time.perf_counter()
        plan.forward(coeff)
        plan.adjoint(residual)
        synchronize(torch, device)
        times.append(time.perf_counter() - start)
    return float(median(times)), times


def make_plan(
    context: Any,
    *,
    torch: Any,
    device: Any,
    args: argparse.Namespace,
    threshold: float,
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
        axial_lowrank_rank=args.axial_lowrank_rank,
        ring_adaptive_l_packed_threshold=threshold,
    )


def distribution(times: list[float]) -> dict[str, float | int]:
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


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    reference = payload["reference"]
    rows = payload["rows"]
    lines = [
        "# ODT ring adaptive-L packed GPU sweep",
        "",
        "H=28, axial rank=16 production candidate에서 ring의 radial-dependent L support를 packed layout으로 실행한 결과이다. Axis illumination은 algebraically exact L=0 pruning을 유지한다.",
        "",
        "## 조건",
        "",
        f"- GPU: `{payload['device']['name']}`",
        f"- object: `{payload['problem']['object_shape']}`",
        f"- q samples: `{payload['problem']['q_samples']}`",
        f"- illuminations: `{payload['problem']['illuminations']}`",
        f"- dense reference pair median: `{1000.0 * reference['pair_median_s']:.3f}` ms",
        f"- error tolerance: `{payload['config']['operator_tolerance']}`",
        "",
        "## 결과",
        "",
        "| threshold | active L fraction | pair ms | speedup | physical max rel-L2 | stress max rel-L2 | dot error | pass |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['threshold']:.0e} | {row['active_l_fraction']:.6f} | "
            f"{1000.0 * row['pair_median_s']:.3f} | {row['speedup_vs_dense']:.3f}x | "
            f"{row['physical_worst_rel_l2']:.3e} | {row['stress_worst_rel_l2']:.3e} | "
            f"{row['dot_error']:.3e} | {'yes' if row['passed'] else 'no'} |"
        )
    passing = [row for row in rows if row["passed"]]
    fastest = min(passing, key=lambda row: row["pair_median_s"]) if passing else None
    lines.extend(["", "## 판정", ""])
    if fastest is None:
        lines.append("- 설정한 tolerance를 통과한 packed threshold가 없다.")
    else:
        lines.append(
            f"- 통과한 조건 중 최속은 threshold `{fastest['threshold']:.0e}`: "
            f"`{1000.0 * fastest['pair_median_s']:.3f}` ms, dense 대비 "
            f"`{fastest['speedup_vs_dense']:.3f}x`."
        )
    lines.append(
        "- packed 결과는 동일한 근사 연산자의 forward/adjoint를 사용하며 dot-product error로 수반 일관성을 별도 확인했다."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("adaptive-L production sweep requires CUDA")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    context_start = time.perf_counter()
    context = build_composite_context(args)
    context_build_s = time.perf_counter() - context_start
    complex_dtype, real_dtype, np_complex, _ = torch_dtypes(torch, args.dtype)
    coeff_np = np.ascontiguousarray(
        context.ring.obj.coeff.astype(np_complex, copy=False)
    )
    coeff = torch.as_tensor(coeff_np, dtype=complex_dtype, device=device)
    generator = torch.Generator(device=device).manual_seed(args.stress_seed)
    stress_coeff = torch.complex(
        torch.randn(
            coeff.shape, generator=generator, device=device, dtype=real_dtype
        ),
        torch.randn(
            coeff.shape, generator=generator, device=device, dtype=real_dtype
        ),
    )

    reference_plan = make_plan(
        context, torch=torch, device=device, args=args, threshold=0.0
    )
    with torch.inference_mode():
        reference_forward = reference_plan.forward(coeff)
        residual = reference_forward * (0.1 + 0.2j)
        reference_adjoint = reference_plan.adjoint(residual)
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
        reference_stress_forward = reference_plan.forward(stress_coeff)
        reference_stress_adjoint = reference_plan.adjoint(stress_residual)
        reference_pair_s, reference_times = timed_pair(
            torch,
            device,
            reference_plan,
            coeff,
            residual,
            repeats=args.repeats,
            warmups=args.warmups,
        )

        rows: list[dict[str, Any]] = []
        for threshold in args.thresholds:
            build_start = time.perf_counter()
            plan = make_plan(
                context,
                torch=torch,
                device=device,
                args=args,
                threshold=float(threshold),
            )
            plan_build_s = time.perf_counter() - build_start
            forward = plan.forward(coeff)
            adjoint = plan.adjoint(residual)
            stress_forward = plan.forward(stress_coeff)
            stress_adjoint = plan.adjoint(stress_residual)
            physical_forward_error = rel_l2(torch, forward, reference_forward)
            physical_adjoint_error = rel_l2(torch, adjoint, reference_adjoint)
            stress_forward_error = rel_l2(
                torch, stress_forward, reference_stress_forward
            )
            stress_adjoint_error = rel_l2(
                torch, stress_adjoint, reference_stress_adjoint
            )
            lhs = torch.vdot(stress_forward.reshape(-1), stress_residual.reshape(-1))
            rhs = torch.vdot(stress_coeff.reshape(-1), stress_adjoint.reshape(-1))
            dot_error = float(
                (
                    torch.abs(lhs - rhs)
                    / torch.clamp(torch.abs(lhs) + torch.abs(rhs), min=1e-30)
                )
                .detach()
                .cpu()
                .item()
            )
            pair_s, times = timed_pair(
                torch,
                device,
                plan,
                coeff,
                residual,
                repeats=args.repeats,
                warmups=args.warmups,
            )
            physical_worst = max(physical_forward_error, physical_adjoint_error)
            stress_worst = max(stress_forward_error, stress_adjoint_error)
            worst = max(physical_worst, stress_worst)
            row = {
                "threshold": float(threshold),
                "active_l_fraction": float(plan.ring.adaptive_l_active_fraction),
                "active_l_pairs": int(
                    plan.ring.adaptive_l_offsets[-1]
                    if plan.ring.adaptive_l_offsets is not None
                    else plan.ring.n_r * plan.ring.n_l
                ),
                "dense_l_pairs": int(plan.ring.n_r * plan.ring.n_l),
                "plan_build_s": float(plan_build_s),
                "physical_forward_rel_l2": physical_forward_error,
                "physical_adjoint_rel_l2": physical_adjoint_error,
                "physical_worst_rel_l2": physical_worst,
                "stress_forward_rel_l2": stress_forward_error,
                "stress_adjoint_rel_l2": stress_adjoint_error,
                "stress_worst_rel_l2": stress_worst,
                "worst_rel_l2": worst,
                "dot_error": dot_error,
                "pair_median_s": pair_s,
                "pair_times_s": times,
                "pair_distribution": distribution(times),
                "speedup_vs_dense": float(reference_pair_s / pair_s),
                "passed": bool(
                    worst <= args.operator_tolerance
                    and dot_error <= args.dot_tolerance
                ),
            }
            rows.append(row)
            print(
                f"adaptive-L {threshold:.0e}: active={row['active_l_fraction']:.4f}, "
                f"pair={1000.0 * pair_s:.3f} ms, worst={worst:.3e}, "
                f"pass={row['passed']}",
                flush=True,
            )

    peak_mib = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    payload = {
        "schema": "odt-adaptive-l-packed-sweep-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": {
            "name": device_name(torch, device),
            "torch": getattr(torch, "__version__", None),
            "cuda": getattr(torch.version, "cuda", None),
            "peak_allocated_mib": peak_mib,
        },
        "config": {
            "h_cutoff": int(args.h_cutoff),
            "axial_lowrank_rank": int(args.axial_lowrank_rank),
            "radial_block_size": int(args.radial_block_size),
            "illumination_block_size": int(args.illumination_block_size),
            "thresholds": [float(value) for value in args.thresholds],
            "operator_tolerance": float(args.operator_tolerance),
            "dot_tolerance": float(args.dot_tolerance),
            "warmups": int(args.warmups),
            "repeats": int(args.repeats),
            "stress_seed": int(args.stress_seed),
        },
        "problem": {
            "object_shape": list(coeff.shape),
            "q_samples": int(reference_plan.q_count),
            "illuminations": int(args.ring_illum)
            + (0 if args.skip_axis_illumination else 1),
            "ring_h_modes": int(reference_plan.ring.n_h),
            "ring_l_modes": int(reference_plan.ring.n_l),
            "axis_l_modes": (
                None if reference_plan.axis is None else int(reference_plan.axis.n_l)
            ),
        },
        "context_build_s": float(context_build_s),
        "reference": {
            "pair_median_s": reference_pair_s,
            "pair_times_s": reference_times,
            "pair_distribution": distribution(reference_times),
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.summary_md is not None:
        write_markdown(args.summary_md, payload)
    return payload


def parser() -> argparse.ArgumentParser:
    p = torch_parser()
    p.description = "Sweep the production ring adaptive-L packed PyTorch GPU path."
    p.set_defaults(
        device="cuda",
        dtype="complex64",
        n_beta=256,
        n_r=256,
        n_z=256,
        ring_illum=120,
        cap_radial=256,
        cap_phi=256,
        h_cutoff=28,
        l_margin=18,
        cone_l_prune_threshold=1e-12,
        cpp_threads=4,
        skip_native_prepared_adjoint=True,
        compact_axisymmetric_kernel=True,
        radial_block_size=32,
        illumination_block_size=4,
        axial_lowrank_rank=16,
    )
    p.add_argument(
        "--thresholds",
        type=comma_floats,
        default=[1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4],
    )
    p.add_argument("--operator-tolerance", type=float, default=2e-6)
    p.add_argument("--dot-tolerance", type=float, default=2e-6)
    p.add_argument("--stress-seed", type=int, default=20260713)
    p.set_defaults(warmups=5, repeats=10)
    p.set_defaults(
        out=ROOT / "benchmark_results" / "odt_adaptive_l_packed_sweep.json",
        summary_md=ROOT
        / "benchmark_results"
        / "odt_adaptive_l_packed_sweep_ko.md",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    payload = run(args)
    print(json.dumps({"reference": payload["reference"], "rows": payload["rows"]}, indent=2))


if __name__ == "__main__":
    main()
