from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_high_na_torch_gpu import import_torch, resolve_device, synchronize  # noqa: E402
from benchmark_odt_realistic_geometry_reconstruction import build_composite_context  # noqa: E402
from benchmark_odt_torch_gpu_reconstruction import (  # noqa: E402
    TorchCompositeOdtPlan,
    parser as gpu_parser,
)


def rel_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(reference.ravel())), np.finfo(np.float64).tiny)
    return float(np.linalg.norm((candidate - reference).ravel()) / denom)


def direct_forward_subset(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    weights: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    *,
    q_chunk: int,
) -> np.ndarray:
    out = np.empty(qx.size, dtype=np.complex128)
    for start in range(0, qx.size, q_chunk):
        stop = min(start + q_chunk, qx.size)
        phase = (
            qx[start:stop, None] * x[None, :]
            + qy[start:stop, None] * y[None, :]
            + qz[start:stop, None] * z[None, :]
        )
        out[start:stop] = np.exp(1j * phase) @ weights
    return out


def direct_adjoint_selected_objects(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    residual: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    *,
    object_chunk: int,
) -> np.ndarray:
    out = np.empty(x.size, dtype=np.complex128)
    for start in range(0, x.size, object_chunk):
        stop = min(start + object_chunk, x.size)
        phase = (
            qx[:, None] * x[None, start:stop]
            + qy[:, None] * y[None, start:stop]
            + qz[:, None] * z[None, start:stop]
        )
        out[start:stop] = np.exp(-1j * phase).T @ residual
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independent direct-exponent subset validation for the 128-cubed ODT operator."
    )
    parser.add_argument("--support-count", type=int, default=2048)
    parser.add_argument("--q-subset-count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--q-chunk", type=int, default=128)
    parser.add_argument("--object-chunk", type=int, default=128)
    parser.add_argument("--forward-gate", type=float, default=1e-6)
    parser.add_argument("--adjoint-gate", type=float, default=1e-9)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/odt_128cubed_independent_subset.json"),
    )
    args = parser.parse_args()

    config = gpu_parser().parse_args([])
    config.device = "cuda"
    config.dtype = "complex128"
    config.low_memory_adjoint = True
    config.n_beta = 128
    config.n_r = 128
    config.n_z = 128
    config.ring_illum = 60
    config.skip_axis_illumination = False
    config.cap_radial = 128
    config.cap_phi = 128
    config.cpp_threads = 4

    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, config.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    build_start = time.perf_counter()
    ctx = build_composite_context(config)
    plan = TorchCompositeOdtPlan.from_context(
        ctx,
        torch=torch,
        device=device,
        dtype=config.dtype,
        low_memory_adjoint=True,
    )
    build_s = time.perf_counter() - build_start

    n_object = int(ctx.ring.obj.coeff.size)
    rng = np.random.default_rng(args.seed)
    support = np.sort(rng.choice(n_object, size=args.support_count, replace=False))
    weights = (
        rng.standard_normal(args.support_count)
        + 1j * rng.standard_normal(args.support_count)
    ).astype(np.complex128)
    coeff = np.zeros(n_object, dtype=np.complex128)
    coeff[support] = weights
    coeff = coeff.reshape(ctx.ring.obj.coeff.shape)
    coeff_t = torch.as_tensor(coeff, dtype=torch.complex128, device=device)

    synchronize(torch, device)
    start = time.perf_counter()
    with torch.inference_mode():
        data_t = plan.forward(coeff_t)
    synchronize(torch, device)
    forward_s = time.perf_counter() - start

    qx = np.concatenate(
        [ctx.ring.flat_q.qx]
        + ([] if ctx.axis is None else [ctx.axis.flat_q.qx])
    )
    qy = np.concatenate(
        [ctx.ring.flat_q.qy]
        + ([] if ctx.axis is None else [ctx.axis.flat_q.qy])
    )
    qz = np.concatenate(
        [ctx.ring.flat_q.qz]
        + ([] if ctx.axis is None else [ctx.axis.flat_q.qz])
    )
    if qx.size != plan.q_count:
        raise RuntimeError("composite q ordering/count mismatch")
    q_subset = np.unique(
        np.linspace(0, qx.size - 1, args.q_subset_count, dtype=np.int64)
    )
    obj_x = ctx.ring.obj.x[support]
    obj_y = ctx.ring.obj.y[support]
    obj_z = ctx.ring.obj.z[support]

    direct_start = time.perf_counter()
    direct_forward = direct_forward_subset(
        obj_x,
        obj_y,
        obj_z,
        weights,
        qx[q_subset],
        qy[q_subset],
        qz[q_subset],
        q_chunk=args.q_chunk,
    )
    direct_forward_s = time.perf_counter() - direct_start
    gpu_forward_subset = data_t[q_subset].detach().cpu().numpy()
    forward_error = rel_l2(gpu_forward_subset, direct_forward)

    residual_values = (
        rng.standard_normal(q_subset.size) + 1j * rng.standard_normal(q_subset.size)
    ).astype(np.complex128)
    residual = np.zeros(qx.size, dtype=np.complex128)
    residual[q_subset] = residual_values
    residual_t = torch.as_tensor(residual, dtype=torch.complex128, device=device)
    synchronize(torch, device)
    start = time.perf_counter()
    with torch.inference_mode():
        grad_t = plan.adjoint(residual_t)
    synchronize(torch, device)
    adjoint_s = time.perf_counter() - start

    direct_start = time.perf_counter()
    direct_adjoint = direct_adjoint_selected_objects(
        obj_x,
        obj_y,
        obj_z,
        residual_values,
        qx[q_subset],
        qy[q_subset],
        qz[q_subset],
        object_chunk=args.object_chunk,
    )
    direct_adjoint_s = time.perf_counter() - direct_start
    gpu_grad_selected = grad_t.reshape(-1)[support].detach().cpu().numpy()
    adjoint_error = rel_l2(gpu_grad_selected, direct_adjoint)

    lhs = np.vdot(gpu_forward_subset.astype(np.complex128), residual_values)
    rhs = np.vdot(weights, gpu_grad_selected.astype(np.complex128))
    subset_dot_error = float(
        abs(lhs - rhs) / max(abs(lhs) + abs(rhs), np.finfo(np.float64).tiny)
    )

    result = {
        "schema": "odt-128cubed-independent-subset-v1",
        "problem": {
            "object_shape": [128, 128, 128],
            "object_bins": n_object,
            "illumination_count": 61,
            "detector_shape": [128, 128],
            "total_q_samples": int(qx.size),
            "dtype": "complex128",
            "low_memory_adjoint": True,
        },
        "subset": {
            "active_object_support": int(support.size),
            "q_samples": int(q_subset.size),
            "selection": "seeded random object support and uniform full-range q indices",
            "seed": args.seed,
        },
        "metrics": {
            "forward_complex_l2_vs_direct": forward_error,
            "adjoint_selected_object_l2_vs_direct": adjoint_error,
            "subset_dot_error": subset_dot_error,
            "forward_gate": args.forward_gate,
            "adjoint_gate": args.adjoint_gate,
            "forward_gate_pass": bool(forward_error <= args.forward_gate),
            "adjoint_gate_pass": bool(adjoint_error <= args.adjoint_gate),
        },
        "timing_s": {
            "context_and_plan_build": build_s,
            "gpu_full_forward": forward_s,
            "direct_subset_forward": direct_forward_s,
            "gpu_full_adjoint": adjoint_s,
            "direct_selected_object_adjoint": direct_adjoint_s,
        },
        "memory": {
            "gpu_basis_mib": plan.basis_mib,
            "gpu_peak_allocated_mib": float(
                torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            ),
        },
        "limitations": [
            "The full 128-cubed structured forward and adjoint are executed, but the independent Cartesian exponent reference is evaluated on selected q nodes and selected object support.",
            "The sparse-support reference validates operator phases and adjoint signs without materializing a 2,097,152 by 999,424 direct matrix.",
            "This is an operator validation, not a reconstruction-accuracy result.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# ODT 128-cubed independent subset validation",
        "",
        f"- full problem: `128 x 128 x 128`, 61 illuminations, detector `128 x 128`, `{qx.size:,}` q samples",
        f"- independent subset: `{support.size:,}` active object bins x `{q_subset.size:,}` q samples",
        f"- forward complex L2: `{forward_error:.3e}` ({'PASS' if forward_error <= args.forward_gate else 'FAIL'})",
        f"- selected-object adjoint L2: `{adjoint_error:.3e}` ({'PASS' if adjoint_error <= args.adjoint_gate else 'FAIL'})",
        f"- subset dot error: `{subset_dot_error:.3e}`",
        f"- GPU peak allocated: `{result['memory']['gpu_peak_allocated_mib']:.2f} MiB`",
        "",
        "The full structured operator is executed. The independent direct exponent sum is restricted to selected source and detector nodes to avoid materializing the full direct matrix.",
    ]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {args.output} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
