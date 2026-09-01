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
from benchmark_odt_128cubed_cg_reconstruction import (  # noqa: E402
    real_complex,
    regularization_normal,
)
from benchmark_odt_realistic_geometry_reconstruction import build_composite_context  # noqa: E402
from benchmark_odt_torch_gpu_reconstruction import (  # noqa: E402
    TorchCompositeOdtPlan,
    parser as gpu_parser,
)


def norm(torch, value):
    return torch.clamp(torch.linalg.vector_norm(value), min=1e-30)


def data_lipschitz(torch, plan, *, iterations: int, seed: int) -> float:
    torch.manual_seed(seed)
    real = torch.randn(
        (plan.ring.n_r, plan.ring.n_z, plan.ring.n_beta),
        dtype=plan.real_dtype,
        device=plan.device,
    )
    value = torch.complex(real, torch.zeros_like(real))
    value = value / norm(torch, value)
    estimate = 0.0
    with torch.inference_mode():
        for _ in range(iterations):
            normal = real_complex(torch, plan.adjoint(plan.forward(value)))
            estimate = float(torch.real(torch.vdot(value.reshape(-1), normal.reshape(-1))).item())
            value = normal / norm(torch, normal)
    return estimate


def solve(
    torch,
    plan,
    data,
    truth,
    *,
    regularization: float,
    data_lipschitz_value: float,
    iterations: int,
    record_every: int,
) -> dict:
    # The forward-difference normal operator has spectral norm <= 12 in 3-D.
    lipschitz = 1.10 * data_lipschitz_value + 12.0 * regularization
    step = 1.0 / lipschitz
    x = torch.zeros_like(truth)
    y = x.clone()
    momentum = 1.0
    truth_norm = norm(torch, truth)
    history = []
    start = time.perf_counter()
    with torch.inference_mode():
        for iteration in range(1, iterations + 1):
            residual = plan.forward(y) - data
            gradient = real_complex(torch, plan.adjoint(residual))
            if regularization:
                gradient = gradient + regularization * regularization_normal(
                    torch, y, "gradient"
                )
            x_new_real = torch.clamp(torch.real(y - step * gradient), min=0.0)
            x_new = torch.complex(x_new_real, torch.zeros_like(x_new_real))
            momentum_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
            y_new = x_new + ((momentum - 1.0) / momentum_new) * (x_new - x)
            # Adaptive restart suppresses oscillation after the positivity projection.
            restart = torch.real(
                torch.vdot((x_new - x).reshape(-1), (y_new - x_new).reshape(-1))
            ) > 0
            if bool(restart.item()):
                momentum_new = 1.0
                y_new = x_new
            x, y, momentum = x_new, y_new, momentum_new
            if iteration % record_every == 0 or iteration == iterations:
                history.append(
                    {
                        "iteration": iteration,
                        "object_nrmse": float((norm(torch, x - truth) / truth_norm).item()),
                    }
                )
        prediction = plan.forward(x)
        data_residual = float((norm(torch, prediction - data) / norm(torch, data)).item())
        clean_residual = None
    synchronize(torch, plan.device)
    return {
        "regularization": regularization,
        "lipschitz": lipschitz,
        "step": step,
        "iterations": iterations,
        "solve_s": time.perf_counter() - start,
        "object_nrmse": history[-1]["object_nrmse"],
        "data_residual": data_residual,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Nonnegative FISTA for the 128-cubed ODT beads gate.")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--power-iterations", type=int, default=12)
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--noise-db", type=float, default=30.0)
    parser.add_argument("--lambdas", default="0,1e7,1e8,1e9")
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/odt_128cubed_beads_30db_nonnegative_fista.json"),
    )
    args = parser.parse_args()
    config = gpu_parser().parse_args([])
    config.device = "cuda"
    config.dtype = "complex64"
    config.low_memory_adjoint = True
    config.real_object = True
    config.n_beta = config.n_r = config.n_z = 128
    config.ring_illum = 60
    config.skip_axis_illumination = False
    config.cap_radial = config.cap_phi = 128
    config.cpp_threads = 4
    config.skip_native_prepared_adjoint = False
    config.compact_axisymmetric_kernel = False
    config.radial_block_size = 0
    config.illumination_block_size = 0

    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, config.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    build_start = time.perf_counter()
    context = build_composite_context(config)
    plan = TorchCompositeOdtPlan.from_context(
        context,
        torch=torch,
        device=device,
        dtype=config.dtype,
        low_memory_adjoint=True,
    )
    truth_np = np.ascontiguousarray(np.real(context.ring.obj.coeff).astype(np.complex64))
    truth = torch.as_tensor(truth_np, dtype=torch.complex64, device=device)
    with torch.inference_mode():
        clean_data = plan.forward(truth)
        torch.manual_seed(args.seed + 1709)
        noise = torch.complex(
            torch.randn_like(torch.real(clean_data)),
            torch.randn_like(torch.real(clean_data)),
        )
        noise *= norm(torch, clean_data) * 10.0 ** (-args.noise_db / 20.0) / norm(torch, noise)
        data = clean_data + noise
    lipschitz_start = time.perf_counter()
    data_lip = data_lipschitz(
        torch, plan, iterations=args.power_iterations, seed=args.seed + 33
    )
    synchronize(torch, device)
    lipschitz_s = time.perf_counter() - lipschitz_start
    rows = [
        solve(
            torch,
            plan,
            data,
            truth,
            regularization=float(value),
            data_lipschitz_value=data_lip,
            iterations=args.iterations,
            record_every=args.record_every,
        )
        for value in args.lambdas.split(",")
        if value.strip()
    ]
    best = min(rows, key=lambda row: row["object_nrmse"])
    result = {
        "schema": "odt-128cubed-nonnegative-fista-v1",
        "problem": {
            "object_shape": [128, 128, 128],
            "illumination_count": 61,
            "detector_shape": [128, 128],
            "noise_db": args.noise_db,
            "constraint": "real nonnegative",
            "regularizer": "3-D forward-difference quadratic gradient",
        },
        "data_lipschitz_estimate": data_lip,
        "timing_s": {
            "context_plan_and_data": lipschitz_start - build_start,
            "power_iteration": lipschitz_s,
        },
        "rows": rows,
        "best": best,
        "gate_nrmse_le_5pct": best["object_nrmse"] <= 0.05,
        "passed": best["object_nrmse"] <= 0.05,
        "gpu_peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**result, "rows": [{k: v for k, v in row.items() if k != "history"} for row in rows]}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
