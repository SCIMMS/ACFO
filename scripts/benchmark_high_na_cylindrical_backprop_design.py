from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_high_na_backpropagation import focal_grid  # noqa: E402
from benchmark_high_na_debye_wolf import gauss_theta_grid, relative_l2  # noqa: E402
from benchmark_high_na_gpu_dense_baseline import (  # noqa: E402
    TorchDenseDirectVectorialDebyeWolf,
    target_points_from_cylindrical,
)
from benchmark_high_na_torch_gpu import (  # noqa: E402
    TorchSeparableHarmonicDebyeWolfPlan,
    device_name,
    import_torch,
    make_pupil_batch,
    package_version,
    resolve_device,
    timed_torch,
    to_numpy,
)
from benchmark_high_na_vectorial_backpropagation import (  # noqa: E402
    richards_wolf_jones_matrix,
    vectorial_h_cutoff_for_workload,
    vectorial_pupil_jones,
    workloads,
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": config, "rows": rows}, indent=2), encoding="utf-8")


def roi_weight(
    *,
    rho_axis: np.ndarray,
    psi_axis: np.ndarray,
    z_axis: np.ndarray,
    rho_max: float,
    z_max: float,
    case: str,
    rho_fraction: float,
    rho_sigma_fraction: float,
    z_sigma_fraction: float,
    psi_modulation: float,
    psi_order: int,
) -> np.ndarray:
    rho = rho_axis[:, None, None]
    psi = psi_axis[None, :, None]
    z = z_axis[None, None, :]
    if case == "all":
        weight = np.ones((rho_axis.size, psi_axis.size, z_axis.size), dtype=float)
    elif case == "central":
        rho_sigma = max(rho_sigma_fraction * rho_max, np.finfo(float).eps)
        z_sigma = max(z_sigma_fraction * max(z_max, 1.0), np.finfo(float).eps)
        weight = np.exp(-0.5 * (rho / rho_sigma) ** 2 - 0.5 * (z / z_sigma) ** 2)
        weight = np.broadcast_to(weight, (rho_axis.size, psi_axis.size, z_axis.size)).copy()
    elif case == "annular":
        rho_center = rho_fraction * rho_max
        rho_sigma = max(rho_sigma_fraction * rho_max, np.finfo(float).eps)
        z_sigma = max(z_sigma_fraction * max(z_max, 1.0), np.finfo(float).eps)
        weight = np.exp(
            -0.5 * ((rho - rho_center) / rho_sigma) ** 2
            - 0.5 * (z / z_sigma) ** 2
        )
        weight = np.broadcast_to(weight, (rho_axis.size, psi_axis.size, z_axis.size)).copy()
    else:
        raise ValueError("roi_case must be all, central, or annular")
    if psi_modulation != 0.0:
        modulation = 1.0 + psi_modulation * np.cos(int(psi_order) * psi)
        weight = weight * np.clip(modulation, 0.0, None)
    max_weight = float(np.max(weight))
    if max_weight <= 0.0:
        raise ValueError("ROI weight is zero everywhere")
    return (weight / max_weight).reshape(-1)


def field_match_loss_and_residual(torch: Any, field: Any, target: Any, weight: Any) -> tuple[Any, Any]:
    diff = field - target
    norm = torch.clamp(torch.sum(weight).real * field.shape[0] * field.shape[1], min=1.0)
    weighted = weight[None, None, :] * diff / norm
    loss = 0.5 * torch.sum(weight[None, None, :] * torch.abs(diff) ** 2).real / norm
    return loss, weighted


def phase_gradient(torch: Any, pupil: Any, pupil_gradient: Any) -> Any:
    return torch.sum(torch.imag(torch.conj(pupil) * pupil_gradient), dim=1)


def phase_step_pupil(torch: Any, pupil: Any, phase_grad: Any, step_radians: float) -> Any:
    max_abs = torch.amax(torch.abs(phase_grad).reshape(phase_grad.shape[0], -1), dim=1)
    scale = torch.clamp(max_abs, min=1e-30).reshape(-1, 1, 1)
    delta = -float(step_radians) * phase_grad / scale
    return pupil * torch.exp(1j * delta[:, None, :, :])


def relative_scalar_error(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def run_skipped(args: argparse.Namespace, reason: str) -> list[dict[str, Any]]:
    return [
        {
            "status": "skipped",
            "skip_reason": reason,
            "workload_set": args.workload_set,
            "workload_index": args.workload_index,
            "device": args.device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
        }
    ]


def run_case(args: argparse.Namespace) -> list[dict[str, Any]]:
    torch = import_torch()
    if torch is None:
        return run_skipped(args, "torch is not installed")
    try:
        device = resolve_device(torch, args.device)
    except RuntimeError as exc:
        return run_skipped(args, str(exc))

    workload = workloads(args.workload_set)[args.workload_index]
    theta, theta_weights = gauss_theta_grid(workload.ntheta, workload.theta_max)
    phi = np.linspace(0.0, 2.0 * np.pi, workload.nphi, endpoint=False, dtype=float)
    rho_axis, psi_axis, z_axis, rho, psi, z_cyl = focal_grid(
        nrho=workload.nrho,
        npsi=workload.npsi,
        nz=workload.nz,
        rho_max=workload.rho_max,
        z_max=workload.z_max,
    )
    x, y, z = target_points_from_cylindrical(rho, psi, z_cyl)
    mixing = richards_wolf_jones_matrix(theta, phi, apodization=args.apodization)
    base_pupil = vectorial_pupil_jones(
        workload.pupil_case,
        theta,
        phi,
        theta_max=workload.theta_max,
        strength=workload.aberration_strength,
        vortex_charge=workload.vortex_charge,
    )
    pupil_batch = make_pupil_batch(
        base_pupil,
        theta,
        phi,
        theta_max=workload.theta_max,
        batch_size=args.batch_size,
        phase_strength=args.initial_phase_strength,
    )
    teacher_batch = make_pupil_batch(
        base_pupil,
        theta,
        phi,
        theta_max=workload.theta_max,
        batch_size=args.batch_size,
        phase_strength=args.teacher_phase_strength,
    )
    weight_np = roi_weight(
        rho_axis=rho_axis,
        psi_axis=psi_axis,
        z_axis=z_axis,
        rho_max=workload.rho_max,
        z_max=workload.z_max,
        case=args.roi_case,
        rho_fraction=args.roi_rho_fraction,
        rho_sigma_fraction=args.roi_rho_sigma_fraction,
        z_sigma_fraction=args.roi_z_sigma_fraction,
        psi_modulation=args.roi_psi_modulation,
        psi_order=args.roi_psi_order,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    dense = TorchDenseDirectVectorialDebyeWolf(
        torch=torch,
        theta=theta,
        theta_weights=theta_weights,
        phi=phi,
        x=x,
        y=y,
        z=z,
        k=workload.k,
        mixing=mixing,
        device=device,
        dtype=args.dtype,
        chunk_targets=args.chunk_targets,
    )
    h_cutoff = vectorial_h_cutoff_for_workload(workload, args.h_margin)
    separable = TorchSeparableHarmonicDebyeWolfPlan.build(
        torch=torch,
        nphi=workload.nphi,
        theta=theta,
        theta_weights=theta_weights,
        rho_axis=rho_axis,
        psi_axis=psi_axis,
        z_axis=z_axis,
        k=workload.k,
        h_cutoff=h_cutoff,
        device=device,
        dtype=args.dtype,
        basis_mode=args.basis_mode,
        contract_mode=args.contract_mode,
    )
    pupil_t = torch.as_tensor(pupil_batch, dtype=dense.complex_dtype, device=device)
    teacher_t = torch.as_tensor(teacher_batch, dtype=dense.complex_dtype, device=device)
    weight_t = torch.as_tensor(weight_np, dtype=dense.real_dtype, device=device)
    mixing_t = separable.as_tensor(mixing)

    with torch.no_grad():
        target = dense.forward(teacher_t).detach()

    def dense_iteration() -> tuple[Any, Any, Any]:
        field = dense.forward(pupil_t)
        loss, residual = field_match_loss_and_residual(torch, field, target, weight_t)
        grad = dense.adjoint(residual)
        phase_grad = phase_gradient(torch, pupil_t, grad)
        return loss, grad, phase_grad

    def separable_iteration() -> tuple[Any, Any, Any]:
        field = separable.evaluate_vectorial_batch(pupil_t, mixing_t)
        loss, residual = field_match_loss_and_residual(torch, field, target, weight_t)
        grad = separable.adjoint_vectorial_batch(residual, mixing_t)
        phase_grad = phase_gradient(torch, pupil_t, grad)
        return loss, grad, phase_grad

    dense_value, dense_s, dense_times = timed_torch(
        torch,
        device,
        dense_iteration,
        repeats=args.repeats,
        warmups=args.warmups,
    )
    separable_value, separable_s, separable_times = timed_torch(
        torch,
        device,
        separable_iteration,
        repeats=args.repeats,
        warmups=args.warmups,
    )
    dense_loss_t, dense_grad_t, dense_phase_t = dense_value
    separable_loss_t, separable_grad_t, separable_phase_t = separable_value

    with torch.no_grad():
        sep_step_pupil = phase_step_pupil(torch, pupil_t, separable_phase_t, args.phase_step_radians)
        dense_step_pupil = phase_step_pupil(torch, pupil_t, dense_phase_t, args.phase_step_radians)

        sep_field_after_sep_step = separable.evaluate_vectorial_batch(sep_step_pupil, mixing_t)
        sep_loss_after_sep_step, _ = field_match_loss_and_residual(
            torch,
            sep_field_after_sep_step,
            target,
            weight_t,
        )
        dense_field_after_sep_step = dense.forward(sep_step_pupil)
        dense_loss_after_sep_step, _ = field_match_loss_and_residual(
            torch,
            dense_field_after_sep_step,
            target,
            weight_t,
        )
        dense_field_after_dense_step = dense.forward(dense_step_pupil)
        dense_loss_after_dense_step, _ = field_match_loss_and_residual(
            torch,
            dense_field_after_dense_step,
            target,
            weight_t,
        )

    dense_grad_np = to_numpy(torch, device, dense_grad_t)
    separable_grad_np = to_numpy(torch, device, separable_grad_t)
    dense_phase_np = to_numpy(torch, device, dense_phase_t)
    separable_phase_np = to_numpy(torch, device, separable_phase_t)

    dense_loss = float(dense_loss_t.detach().cpu().item())
    separable_loss = float(separable_loss_t.detach().cpu().item())
    sep_loss_after_sep_step = float(sep_loss_after_sep_step.detach().cpu().item())
    dense_loss_after_sep_step = float(dense_loss_after_sep_step.detach().cpu().item())
    dense_loss_after_dense_step = float(dense_loss_after_dense_step.detach().cpu().item())

    row: dict[str, Any] = {
        "status": "ok",
        "objective": "weighted_cylindrical_roi_field_match",
        "workload": workload.name,
        "workload_set": args.workload_set,
        "workload_index": args.workload_index,
        "device": str(device),
        "device_name": device_name(torch, device),
        "torch_version": package_version("torch"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "ntheta": workload.ntheta,
        "nphi": workload.nphi,
        "nrho": workload.nrho,
        "npsi": workload.npsi,
        "nz": workload.nz,
        "targets_per_mask": int(rho.size),
        "field_components": 3,
        "pupil_components": 2,
        "roi_case": args.roi_case,
        "roi_weight_sum": float(np.sum(weight_np)),
        "roi_weight_nonzero_fraction": float(np.count_nonzero(weight_np > 1e-6) / weight_np.size),
        "h_cutoff": int(h_cutoff),
        "separable_used_modes": int(separable.used_modes),
        "separable_basis_mode": separable.basis_mode,
        "separable_contract_mode": separable.contract_mode,
        "separable_basis_mib": float(separable.basis_mib),
        "dense_source_mib": float(dense.source_mib),
        "dense_iteration_hot_s": float(dense_s),
        "separable_iteration_hot_s": float(separable_s),
        "speedup_dense_vs_separable_iteration": float(dense_s / separable_s),
        "dense_loss": dense_loss,
        "separable_loss": separable_loss,
        "loss_relative_error_dense_vs_separable": relative_scalar_error(dense_loss, separable_loss),
        "pupil_gradient_l2_dense_vs_separable": relative_l2(separable_grad_np, dense_grad_np),
        "phase_gradient_l2_dense_vs_separable": relative_l2(separable_phase_np, dense_phase_np),
        "phase_step_radians": float(args.phase_step_radians),
        "separable_loss_after_separable_step": sep_loss_after_sep_step,
        "dense_loss_after_separable_step": dense_loss_after_sep_step,
        "dense_loss_after_dense_step": dense_loss_after_dense_step,
        "separable_step_loss_decrease_fraction": float(
            (separable_loss - sep_loss_after_sep_step) / max(abs(separable_loss), 1e-300)
        ),
        "dense_loss_decrease_fraction_from_separable_step": float(
            (dense_loss - dense_loss_after_sep_step) / max(abs(dense_loss), 1e-300)
        ),
        "dense_loss_decrease_fraction_from_dense_step": float(
            (dense_loss - dense_loss_after_dense_step) / max(abs(dense_loss), 1e-300)
        ),
        "dense_iteration_times_s": " ".join(f"{value:.9g}" for value in dense_times),
        "separable_iteration_times_s": " ".join(f"{value:.9g}" for value in separable_times),
        "gpu_peak_allocated_mib": None
        if device.type != "cuda"
        else float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
        "note": "teacher target is generated with dense direct CUDA; objective is field matching on a cylindrical ROI",
    }
    return [row]


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    value_f = float(value)
    if value_f == 0.0:
        return "0"
    if abs(value_f) < 1e-3 or abs(value_f) >= 1e3:
        return f"{value_f:.{digits}e}"
    return f"{value_f:.{digits}f}"


def write_summary(path: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# High-NA Cylindrical Backpropagation Design Benchmark",
        "",
        "This benchmark wraps the vectorial forward and adjoint kernels in a simple cylindrical ROI field-matching objective.",
        "The target field is generated by dense direct CUDA on the same cylindrical grid; the separable path is compared against the same loss and phase-gradient step.",
        "",
        "## Config",
        "",
        f"- workload_set: `{config['workload_set']}`",
        f"- workload_index: `{config['workload_index']}`",
        f"- roi_case: `{config['roi_case']}`",
        f"- device: `{config['device']}`",
        f"- dtype: `{config['dtype']}`",
        f"- basis_mode: `{config.get('basis_mode', 'separate')}`",
        f"- contract_mode: `{config.get('contract_mode', 'einsum')}`",
        f"- batch_size: `{config['batch_size']}`",
        "",
        "## Results",
        "",
        "| workload | batch | targets | ROI sum | dense iter s | separable iter s | speedup | loss rel err | pupil grad L2 | phase grad L2 | sep-step dense loss decrease |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row.get("status") != "ok":
            lines.append(f"| {row.get('status')} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            lines.append("")
            lines.append(f"Skip reason: `{row.get('skip_reason')}`")
            continue
        lines.append(
            "| {workload} | {batch} | {targets} | {roi_sum} | {dense_s} | {sep_s} | {speedup} | {loss_err} | {pupil_l2} | {phase_l2} | {loss_dec} |".format(
                workload=row["workload"],
                batch=row["batch_size"],
                targets=row["targets_per_mask"],
                roi_sum=fmt(row["roi_weight_sum"]),
                dense_s=fmt(row["dense_iteration_hot_s"]),
                sep_s=fmt(row["separable_iteration_hot_s"]),
                speedup=fmt(row["speedup_dense_vs_separable_iteration"]),
                loss_err=fmt(row["loss_relative_error_dense_vs_separable"]),
                pupil_l2=fmt(row["pupil_gradient_l2_dense_vs_separable"]),
                phase_l2=fmt(row["phase_gradient_l2_dense_vs_separable"]),
                loss_dec=fmt(row["dense_loss_decrease_fraction_from_separable_step"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is the first benchmark where the cylindrical target grid is part of an inverse-design style objective, not just a forward output layout.",
            "- `dense iter s` and `separable iter s` include forward propagation, weighted ROI residual construction, adjoint propagation, and shared phase-gradient construction.",
            "- The dense reference is still direct CUDA quadrature; a domain-package matched backpropagation baseline remains future work.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a cylindrical ROI vectorial High-NA backpropagation design step."
    )
    parser.add_argument("--output-prefix", default="benchmark_results/high_na_cylindrical_backprop_design")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    parser.add_argument("--workload-set", choices=["quick", "representative", "large"], default="representative")
    parser.add_argument("--workload-index", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--chunk-targets", type=int, default=1024)
    parser.add_argument("--h-margin", type=int, default=5)
    parser.add_argument("--basis-mode", choices=["separate", "fused"], default="separate")
    parser.add_argument("--contract-mode", choices=["einsum", "matmul"], default="einsum")
    parser.add_argument("--apodization", choices=["sqrt-cos", "none"], default="sqrt-cos")
    parser.add_argument("--initial-phase-strength", type=float, default=0.15)
    parser.add_argument("--teacher-phase-strength", type=float, default=0.55)
    parser.add_argument("--roi-case", choices=["all", "central", "annular"], default="annular")
    parser.add_argument("--roi-rho-fraction", type=float, default=0.55)
    parser.add_argument("--roi-rho-sigma-fraction", type=float, default=0.18)
    parser.add_argument("--roi-z-sigma-fraction", type=float, default=0.30)
    parser.add_argument("--roi-psi-modulation", type=float, default=0.0)
    parser.add_argument("--roi-psi-order", type=int, default=3)
    parser.add_argument("--phase-step-radians", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = vars(args).copy()
    rows = run_case(args)
    output_prefix = ROOT / args.output_prefix
    write_csv(output_prefix.with_suffix(".csv"), rows)
    write_json(output_prefix.with_suffix(".json"), config, rows)
    write_summary(output_prefix.with_name(output_prefix.name + "_summary.md"), config, rows)
    print(json.dumps({"config": config, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
