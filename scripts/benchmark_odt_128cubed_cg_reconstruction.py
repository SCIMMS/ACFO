from __future__ import annotations

import argparse
import csv
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


def real_complex(torch, value):
    real = torch.real(value)
    return torch.complex(real, torch.zeros_like(real))


def real_dot(torch, a, b):
    return torch.sum(torch.real(torch.conj(a) * b))


def regularization_normal(torch, value, kind: str):
    if kind == "identity":
        return value
    if kind != "gradient":
        raise ValueError("regularizer must be identity or gradient")
    out = torch.zeros_like(value)
    diff_r = value[1:, :, :] - value[:-1, :, :]
    out[:-1, :, :] -= diff_r
    out[1:, :, :] += diff_r
    diff_z = value[:, 1:, :] - value[:, :-1, :]
    out[:, :-1, :] -= diff_z
    out[:, 1:, :] += diff_z
    out += 2.0 * value - torch.roll(value, 1, dims=2) - torch.roll(value, -1, dims=2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="128-cubed real-object ODT CG normal-equation reconstruction.")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--truth-mode", choices=["beads", "adjoint_range"], default="beads")
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--tikhonov", type=float, default=0.0)
    parser.add_argument("--regularizer", choices=["identity", "gradient"], default="identity")
    parser.add_argument("--noise-db", type=float)
    parser.add_argument("--illumination-angle-deg", type=float, default=49.0)
    parser.add_argument("--ring-illum", type=int, default=60)
    parser.add_argument("--target-nrmse", type=float, default=0.01)
    parser.add_argument("--record-every", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/odt_128cubed_cg_reconstruction.json"),
    )
    args = parser.parse_args()

    config = gpu_parser().parse_args([])
    config.device = "cuda"
    config.dtype = "complex64"
    config.low_memory_adjoint = True
    config.real_object = True
    config.n_beta = 128
    config.n_r = 128
    config.n_z = 128
    config.ring_illum = args.ring_illum
    config.illumination_angle_deg = args.illumination_angle_deg
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

    if args.truth_mode == "beads":
        true_np = np.ascontiguousarray(np.real(ctx.ring.obj.coeff).astype(np.complex64))
        true_x = torch.as_tensor(true_np, dtype=torch.complex64, device=device)
    else:
        torch.manual_seed(args.seed)
        probe_real = torch.randn(plan.q_count, dtype=torch.float32, device=device)
        probe_imag = torch.randn(plan.q_count, dtype=torch.float32, device=device)
        probe = torch.complex(probe_real, probe_imag)
        with torch.inference_mode():
            true_x = real_complex(torch, plan.adjoint(probe))
            true_x = true_x / torch.clamp(torch.linalg.vector_norm(true_x), min=1e-30)
        true_np = np.empty((0,), dtype=np.complex64)
    true_norm = torch.clamp(torch.linalg.vector_norm(true_x), min=1e-30)
    with torch.inference_mode():
        data = plan.forward(true_x)
        if args.noise_db is not None:
            torch.manual_seed(args.seed + 1709)
            noise = torch.complex(
                torch.randn_like(torch.real(data)),
                torch.randn_like(torch.real(data)),
            )
            target_noise_norm = torch.linalg.vector_norm(data) * 10.0 ** (
                -float(args.noise_db) / 20.0
            )
            noise = noise * (
                target_noise_norm
                / torch.clamp(torch.linalg.vector_norm(noise), min=1e-30)
            )
            data = data + noise
        data_norm = torch.clamp(torch.linalg.vector_norm(data), min=1e-30)
        b = real_complex(torch, plan.adjoint(data))
        x = torch.zeros_like(true_x)
        pred = torch.zeros_like(data)
        r = b.clone()
        p = r.clone()
        rr = real_dot(torch, r, r)

        history = []
        start_all = time.perf_counter()
        converged_iteration = None
        for iteration in range(1, args.iterations + 1):
            synchronize(torch, device)
            iter_start = time.perf_counter()
            ap = plan.forward(p)
            normal_p = real_complex(torch, plan.adjoint(ap))
            if args.tikhonov:
                normal_p = normal_p + float(args.tikhonov) * regularization_normal(
                    torch, p, args.regularizer
                )
            denom = torch.clamp(real_dot(torch, p, normal_p), min=1e-30)
            alpha = rr / denom
            x = x + alpha * p
            pred = pred + alpha * ap
            r_new = r - alpha * normal_p
            rr_new = real_dot(torch, r_new, r_new)
            beta = rr_new / torch.clamp(rr, min=1e-30)
            p = r_new + beta * p
            r = r_new
            rr = rr_new
            synchronize(torch, device)
            iter_s = time.perf_counter() - iter_start

            if iteration % args.record_every == 0 or iteration == args.iterations:
                object_nrmse = float((torch.linalg.vector_norm(x - true_x) / true_norm).cpu().item())
                data_residual = float((torch.linalg.vector_norm(pred - data) / data_norm).cpu().item())
                normal_residual = float((torch.sqrt(torch.clamp(rr, min=0.0)) / torch.clamp(torch.linalg.vector_norm(b), min=1e-30)).cpu().item())
                history.append(
                    {
                        "iteration": iteration,
                        "object_nrmse": object_nrmse,
                        "data_residual": data_residual,
                        "normal_residual": normal_residual,
                        "alpha": float(alpha.cpu().item()),
                        "beta": float(beta.cpu().item()),
                        "iter_s": iter_s,
                    }
                )
                if converged_iteration is None and object_nrmse <= args.target_nrmse:
                    converged_iteration = iteration
                    break

        synchronize(torch, device)
        solve_s = time.perf_counter() - start_all

    final = history[-1]
    result = {
        "schema": "odt-128cubed-real-cg-v1",
        "problem": {
            "object_shape": [128, 128, 128],
            "object_bins": int(true_x.numel()),
            "illumination_count": args.ring_illum + 1,
            "illumination_angle_deg": args.illumination_angle_deg,
            "detector_shape": [128, 128],
            "total_q_samples": int(plan.q_count),
            "object_constraint": "real-valued",
            "truth_mode": args.truth_mode,
            "truth_seed": args.seed,
            "dtype": "complex64",
            "tikhonov": args.tikhonov,
            "regularizer": args.regularizer,
            "noise_db": args.noise_db,
        },
        "solver": "conjugate gradient on real-subspace normal equations",
        "target_nrmse": args.target_nrmse,
        "converged_iteration": converged_iteration,
        "final": final,
        "gate_pass": bool(final["object_nrmse"] <= args.target_nrmse),
        "timing_s": {
            "context_and_plan_build": build_s,
            "solve": solve_s,
            "median_recorded_iteration": float(np.median([row["iter_s"] for row in history])),
        },
        "memory": {
            "gpu_basis_mib": plan.basis_mib,
            "gpu_peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
        },
        "history": history,
        "limitations": [
            "The inverse uses a real-object constraint matched to the phantom.",
            "No positivity or support mask is used.",
            f"The selected regularizer is {args.regularizer}; its strength is {args.tikhonov}.",
            "This is synthetic self-consistency; independent operator accuracy is reported separately.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    lines = [
        "# ODT 128-cubed real-object CG reconstruction",
        "",
        f"- final iteration: `{final['iteration']}`",
        f"- object NRMSE: `{final['object_nrmse']:.6g}`",
        f"- data residual: `{final['data_residual']:.6g}`",
        f"- normal residual: `{final['normal_residual']:.6g}`",
        f"- target NRMSE: `{args.target_nrmse}`",
        f"- gate: `{'PASS' if result['gate_pass'] else 'FAIL'}`",
        f"- GPU peak allocated: `{result['memory']['gpu_peak_allocated_mib']:.2f} MiB`",
        "",
        "The solve uses the real-object constraint only; no positivity, support, or spatial regularization is applied.",
    ]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2))
    print(f"wrote {args.output}, {csv_path}, and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
