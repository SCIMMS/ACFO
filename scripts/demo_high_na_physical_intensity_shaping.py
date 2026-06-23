from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_high_na_backpropagation import focal_grid  # noqa: E402
from benchmark_high_na_debye_wolf import gauss_theta_grid  # noqa: E402
from benchmark_high_na_torch_gpu import (  # noqa: E402
    TorchSeparableHarmonicDebyeWolfPlan,
    device_name,
    import_torch,
    package_version,
    resolve_device,
    synchronize,
    to_numpy,
)
from benchmark_high_na_vectorial_backpropagation import (  # noqa: E402
    richards_wolf_jones_matrix,
    vectorial_h_cutoff_for_workload,
    vectorial_pupil_jones,
    workloads,
)


RESULTS = ROOT / "benchmark_results"

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
COLOR = {
    "blue": "#5477C4",
    "blue_light": "#CEDFFE",
    "orange": "#CC6F47",
    "olive": "#71B436",
    "gold": "#B8A037",
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "font.monospace": ["Consolas", "DejaVu Sans Mono", "monospace"],
        },
    )


def target_weights(
    *,
    rho_axis: np.ndarray,
    psi_axis: np.ndarray,
    z_axis: np.ndarray,
    rho_max: float,
    z_max: float,
    rho_fraction: float,
    rho_sigma_fraction: float,
    z_sigma_fraction: float,
    angular_order: int,
    angular_floor: float,
    angular_phase: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho = rho_axis[:, None, None]
    psi = psi_axis[None, :, None]
    z = z_axis[None, None, :]
    rho_center = rho_fraction * rho_max
    rho_sigma = max(rho_sigma_fraction * rho_max, np.finfo(float).eps)
    z_sigma = max(z_sigma_fraction * max(z_max, 1.0), np.finfo(float).eps)
    ring = np.exp(-0.5 * ((rho - rho_center) / rho_sigma) ** 2 - 0.5 * (z / z_sigma) ** 2)
    lobes = ((1.0 + np.cos(int(angular_order) * psi + angular_phase)) * 0.5) ** 2
    angular = angular_floor + (1.0 - angular_floor) * lobes
    target = ring * angular
    target = np.broadcast_to(target, (rho_axis.size, psi_axis.size, z_axis.size)).copy()
    target /= max(float(target.max()), np.finfo(float).eps)
    off_target = ring * (1.0 - lobes)
    off_target = np.broadcast_to(off_target, target.shape).copy()
    off_target /= max(float(off_target.max()), np.finfo(float).eps)
    center_sigma = max(0.18 * rho_max, np.finfo(float).eps)
    center = np.exp(-0.5 * (rho / center_sigma) ** 2 - 0.5 * (z / z_sigma) ** 2)
    center = np.broadcast_to(center, target.shape).copy()
    center /= max(float(center.max()), np.finfo(float).eps)
    ring_only = np.broadcast_to(ring, target.shape).copy()
    ring_only /= max(float(ring_only.max()), np.finfo(float).eps)
    return target.reshape(-1), center.reshape(-1), ring_only.reshape(-1), off_target.reshape(-1)


def intensity(field: Any, torch: Any) -> Any:
    return torch.sum(torch.abs(field) ** 2, dim=1)


def fraction_and_derivative(torch: Any, intensity_t: Any, weight: Any) -> tuple[Any, Any]:
    total = torch.clamp(torch.sum(intensity_t, dim=1, keepdim=True), min=1e-30)
    numerator = torch.sum(weight[None, :] * intensity_t, dim=1, keepdim=True)
    fraction = numerator / total
    derivative = (weight[None, :] * total - numerator) / (total * total)
    return fraction.squeeze(1), derivative


def loss_residual_and_metrics(
    *,
    torch: Any,
    field: Any,
    target_weight: Any,
    center_weight: Any,
    off_target_weight: Any,
    center_penalty: float,
    off_target_penalty: float,
) -> tuple[Any, Any, dict[str, Any]]:
    inten = intensity(field, torch)
    target_frac, d_target = fraction_and_derivative(torch, inten, target_weight)
    center_frac, d_center = fraction_and_derivative(torch, inten, center_weight)
    off_target_frac, d_off_target = fraction_and_derivative(torch, inten, off_target_weight)
    loss = (
        -torch.mean(target_frac)
        + float(center_penalty) * torch.mean(center_frac)
        + float(off_target_penalty) * torch.mean(off_target_frac)
    )
    d_loss_d_intensity = (
        -d_target + float(center_penalty) * d_center + float(off_target_penalty) * d_off_target
    ) / float(field.shape[0])
    residual = 2.0 * d_loss_d_intensity[:, None, :] * field
    metrics = {
        "loss": float(loss.detach().cpu().item()),
        "target_fraction": float(torch.mean(target_frac).detach().cpu().item()),
        "center_fraction": float(torch.mean(center_frac).detach().cpu().item()),
        "off_target_fraction": float(torch.mean(off_target_frac).detach().cpu().item()),
    }
    return loss, residual, metrics


def phase_gradient(torch: Any, pupil: Any, pupil_gradient: Any) -> Any:
    return torch.sum(torch.imag(torch.conj(pupil) * pupil_gradient), dim=1)


def rms_normalize(torch: Any, value: Any) -> Any:
    rms = torch.sqrt(torch.mean(value * value))
    return value / torch.clamp(rms, min=1e-30)


def wrap_phase(torch: Any, phase_angle: Any) -> Any:
    two_pi = 2.0 * float(np.pi)
    return torch.remainder(phase_angle + float(np.pi), two_pi) - float(np.pi)


def apply_phase(base_pupil: Any, phase_angle: Any, *, torch: Any, complex_dtype: Any) -> Any:
    phase_factor = torch.exp(1j * phase_angle.to(complex_dtype))
    return base_pupil * phase_factor[:, None, :, :]


def evaluate_metrics_np(
    *,
    intensity_np: np.ndarray,
    target_weight: np.ndarray,
    center_weight: np.ndarray,
    ring_weight: np.ndarray,
    off_target_weight: np.ndarray,
    rho_axis: np.ndarray,
    psi_axis: np.ndarray,
    z_axis: np.ndarray,
    angular_order: int,
) -> dict[str, float]:
    flat = intensity_np.reshape(-1)
    total = max(float(np.sum(flat)), np.finfo(float).eps)
    z_index = int(np.argmin(np.abs(z_axis)))
    grid = intensity_np.reshape(rho_axis.size, psi_axis.size, z_axis.size)
    ring_grid = ring_weight.reshape(rho_axis.size, psi_axis.size, z_axis.size)
    angular_profile = np.sum(grid[:, :, z_index] * ring_grid[:, :, z_index], axis=0)
    angular_total = max(float(np.sum(angular_profile)), np.finfo(float).eps)
    coeff = np.sum(angular_profile * np.exp(-1j * int(angular_order) * psi_axis)) / angular_total
    return {
        "target_fraction": float(np.sum(target_weight * flat) / total),
        "center_fraction": float(np.sum(center_weight * flat) / total),
        "ring_fraction": float(np.sum(ring_weight * flat) / total),
        "off_target_fraction": float(np.sum(off_target_weight * flat) / total),
        "angular_m_contrast": float(2.0 * np.abs(coeff)),
        "total_intensity": total,
    }


def tensor_from_np(torch: Any, array: np.ndarray, *, dtype: Any, device: Any) -> Any:
    return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype, device=device)


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    workload = workloads(args.workload_set)[args.workload_index]
    theta, theta_weights = gauss_theta_grid(workload.ntheta, workload.theta_max)
    phi = np.linspace(0.0, 2.0 * np.pi, workload.nphi, endpoint=False, dtype=float)
    rho_axis, psi_axis, z_axis, _, _, _ = focal_grid(
        nrho=workload.nrho,
        npsi=workload.npsi,
        nz=workload.nz,
        rho_max=workload.rho_max,
        z_max=workload.z_max,
    )
    mixing = richards_wolf_jones_matrix(theta, phi, apodization=args.apodization)
    base_np = vectorial_pupil_jones(
        workload.pupil_case,
        theta,
        phi,
        theta_max=workload.theta_max,
        strength=workload.aberration_strength,
        vortex_charge=workload.vortex_charge,
    )
    target_np, center_np, ring_np, off_target_np = target_weights(
        rho_axis=rho_axis,
        psi_axis=psi_axis,
        z_axis=z_axis,
        rho_max=workload.rho_max,
        z_max=workload.z_max,
        rho_fraction=args.target_rho_fraction,
        rho_sigma_fraction=args.target_rho_sigma_fraction,
        z_sigma_fraction=args.target_z_sigma_fraction,
        angular_order=args.angular_order,
        angular_floor=args.angular_floor,
        angular_phase=args.angular_phase,
    )
    h_cutoff = vectorial_h_cutoff_for_workload(workload, args.h_margin)
    plan = TorchSeparableHarmonicDebyeWolfPlan.build(
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
    )
    complex_dtype = plan.complex_dtype
    real_dtype = plan.real_dtype
    base_t = tensor_from_np(torch, base_np[None, ...], dtype=complex_dtype, device=device)
    mixing_t = tensor_from_np(torch, mixing, dtype=complex_dtype, device=device)
    target_t = tensor_from_np(torch, target_np, dtype=real_dtype, device=device)
    center_t = tensor_from_np(torch, center_np, dtype=real_dtype, device=device)
    off_target_t = tensor_from_np(torch, off_target_np, dtype=real_dtype, device=device)

    phase_angle = torch.zeros((1, workload.ntheta, workload.nphi), dtype=real_dtype, device=device)
    momentum_buffer = torch.zeros_like(phase_angle)
    adam_m = torch.zeros_like(phase_angle)
    adam_v = torch.zeros_like(phase_angle)

    history: list[dict[str, Any]] = []
    start = time.perf_counter()

    def current_pupil() -> Any:
        return apply_phase(base_t, phase_angle, torch=torch, complex_dtype=complex_dtype)

    def forward_loss_grad() -> tuple[Any, Any, dict[str, Any], Any]:
        pupil = current_pupil()
        field = plan.evaluate_vectorial_batch(pupil, mixing_t)
        _, residual, metrics = loss_residual_and_metrics(
            torch=torch,
            field=field,
            target_weight=target_t,
            center_weight=center_t,
            off_target_weight=off_target_t,
            center_penalty=args.center_penalty,
            off_target_penalty=args.off_target_penalty,
        )
        pupil_grad = plan.adjoint_vectorial_batch(residual, mixing_t)
        grad = phase_gradient(torch, pupil, pupil_grad)
        return field, grad, metrics, pupil

    with torch.no_grad():
        field0, grad0, metrics0, _ = forward_loss_grad()
        initial_intensity = to_numpy(torch, device, intensity(field0, torch)[0])
        history.append(
            {
                "step": 0,
                "optimizer": args.optimizer,
                "accepted": True,
                "step_radians": 0.0,
                "gradient_rms": 0.0,
                "direction_rms": 0.0,
                **metrics0,
            }
        )

        for step in range(1, args.steps + 1):
            _, grad, metrics, _ = forward_loss_grad()
            grad_real = torch.real(grad)
            gradient_rms = torch.sqrt(torch.mean(grad_real * grad_real))
            normalized = rms_normalize(torch, grad_real)

            if args.optimizer == "line-search":
                direction = normalized
            elif args.optimizer == "momentum":
                momentum_buffer.mul_(float(args.momentum_beta)).add_(normalized)
                direction = rms_normalize(torch, momentum_buffer)
            elif args.optimizer == "adam":
                adam_m.mul_(float(args.adam_beta1)).add_(normalized, alpha=1.0 - float(args.adam_beta1))
                adam_v.mul_(float(args.adam_beta2)).addcmul_(normalized, normalized, value=1.0 - float(args.adam_beta2))
                bias_m = 1.0 - float(args.adam_beta1) ** step
                bias_v = 1.0 - float(args.adam_beta2) ** step
                adam_direction = (adam_m / bias_m) / (torch.sqrt(adam_v / bias_v) + float(args.adam_eps))
                direction = rms_normalize(torch, adam_direction)
            else:
                raise ValueError(f"unknown optimizer: {args.optimizer}")

            direction_rms = torch.sqrt(torch.mean(direction * direction))
            best_phase_angle = phase_angle
            best_metrics = metrics
            best_step = 0.0
            candidate_steps = [
                args.phase_step_radians,
                0.5 * args.phase_step_radians,
                0.25 * args.phase_step_radians,
                1.5 * args.phase_step_radians,
                -0.5 * args.phase_step_radians,
            ]
            for step_size in candidate_steps:
                trial_phase_angle = wrap_phase(torch, phase_angle - float(step_size) * direction)
                trial_pupil = apply_phase(base_t, trial_phase_angle, torch=torch, complex_dtype=complex_dtype)
                trial_field = plan.evaluate_vectorial_batch(trial_pupil, mixing_t)
                _, _, trial_metrics = loss_residual_and_metrics(
                    torch=torch,
                    field=trial_field,
                    target_weight=target_t,
                    center_weight=center_t,
                    off_target_weight=off_target_t,
                    center_penalty=args.center_penalty,
                    off_target_penalty=args.off_target_penalty,
                )
                if trial_metrics["loss"] < best_metrics["loss"]:
                    best_metrics = trial_metrics
                    best_phase_angle = trial_phase_angle
                    best_step = float(step_size)
            accepted = best_step != 0.0
            phase_angle = best_phase_angle
            history.append(
                {
                    "step": step,
                    "optimizer": args.optimizer,
                    "accepted": accepted,
                    "step_radians": best_step,
                    "gradient_rms": float(gradient_rms.detach().cpu().item()),
                    "direction_rms": float(direction_rms.detach().cpu().item()),
                    **best_metrics,
                }
            )
            if args.stop_patience > 0 and len(history) > args.stop_patience:
                recent = history[-args.stop_patience :]
                if max(item["target_fraction"] for item in recent) - min(item["target_fraction"] for item in recent) < args.stop_delta:
                    break

        field_final, _, metrics_final, pupil_final = forward_loss_grad()
        synchronize(torch, device)
        elapsed_s = time.perf_counter() - start
        final_intensity = to_numpy(torch, device, intensity(field_final, torch)[0])
        phase_np = to_numpy(torch, device, phase_angle[0])

    initial_metrics_np = evaluate_metrics_np(
        intensity_np=initial_intensity,
        target_weight=target_np,
        center_weight=center_np,
        ring_weight=ring_np,
        off_target_weight=off_target_np,
        rho_axis=rho_axis,
        psi_axis=psi_axis,
        z_axis=z_axis,
        angular_order=args.angular_order,
    )
    final_metrics_np = evaluate_metrics_np(
        intensity_np=final_intensity,
        target_weight=target_np,
        center_weight=center_np,
        ring_weight=ring_np,
        off_target_weight=off_target_np,
        rho_axis=rho_axis,
        psi_axis=psi_axis,
        z_axis=z_axis,
        angular_order=args.angular_order,
    )
    summary = {
        "status": "ok",
        "demo": "phase_only_cylindrical_intensity_shaping",
        "workload": workload.name,
        "pupil_case": workload.pupil_case,
        "device": str(device),
        "device_name": device_name(torch, device),
        "torch_version": package_version("torch"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "dtype": args.dtype,
        "optimizer": args.optimizer,
        "steps_requested": args.steps,
        "steps_completed": int(history[-1]["step"]),
        "elapsed_s": float(elapsed_s),
        "ms_per_step": float(1000.0 * elapsed_s / max(1, history[-1]["step"])),
        "h_cutoff": int(h_cutoff),
        "used_modes": int(plan.used_modes),
        "targets": int(workload.nrho * workload.npsi * workload.nz),
        "target_order": int(args.angular_order),
        "center_penalty": float(args.center_penalty),
        "off_target_penalty": float(args.off_target_penalty),
        "initial_target_fraction": initial_metrics_np["target_fraction"],
        "final_target_fraction": final_metrics_np["target_fraction"],
        "target_fraction_gain": final_metrics_np["target_fraction"] / max(initial_metrics_np["target_fraction"], 1e-30),
        "initial_center_fraction": initial_metrics_np["center_fraction"],
        "final_center_fraction": final_metrics_np["center_fraction"],
        "initial_off_target_fraction": initial_metrics_np["off_target_fraction"],
        "final_off_target_fraction": final_metrics_np["off_target_fraction"],
        "off_target_fraction_gain": final_metrics_np["off_target_fraction"] / max(initial_metrics_np["off_target_fraction"], 1e-30),
        "target_to_off_target_gain": (final_metrics_np["target_fraction"] / max(final_metrics_np["off_target_fraction"], 1e-30))
        / max(initial_metrics_np["target_fraction"] / max(initial_metrics_np["off_target_fraction"], 1e-30), 1e-30),
        "initial_ring_fraction": initial_metrics_np["ring_fraction"],
        "final_ring_fraction": final_metrics_np["ring_fraction"],
        "initial_m_contrast": initial_metrics_np["angular_m_contrast"],
        "final_m_contrast": final_metrics_np["angular_m_contrast"],
        "final_loss": float(metrics_final["loss"]),
        "gpu_peak_allocated_mib": None
        if device.type != "cuda"
        else float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
    }
    arrays_path = ROOT / args.output_prefix
    npz_path = arrays_path.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        rho_axis=rho_axis,
        psi_axis=psi_axis,
        z_axis=z_axis,
        target_weight=target_np.reshape(workload.nrho, workload.npsi, workload.nz),
        center_weight=center_np.reshape(workload.nrho, workload.npsi, workload.nz),
        ring_weight=ring_np.reshape(workload.nrho, workload.npsi, workload.nz),
        off_target_weight=off_target_np.reshape(workload.nrho, workload.npsi, workload.nz),
        initial_intensity=initial_intensity.reshape(workload.nrho, workload.npsi, workload.nz),
        final_intensity=final_intensity.reshape(workload.nrho, workload.npsi, workload.nz),
        phase=phase_np,
    )
    return {
        "summary": summary,
        "history": history,
        "arrays_npz": str(npz_path),
        "rho_axis": rho_axis,
        "psi_axis": psi_axis,
        "z_axis": z_axis,
        "target_weight": target_np.reshape(workload.nrho, workload.npsi, workload.nz),
        "initial_intensity": initial_intensity.reshape(workload.nrho, workload.npsi, workload.nz),
        "final_intensity": final_intensity.reshape(workload.nrho, workload.npsi, workload.nz),
        "phase": phase_np,
    }


def plot_demo(result: dict[str, Any], output_prefix: Path) -> tuple[Path, Path]:
    use_chart_theme()
    rho_axis = result["rho_axis"]
    psi_axis = result["psi_axis"]
    z_axis = result["z_axis"]
    z_index = int(np.argmin(np.abs(z_axis)))
    target = result["target_weight"][:, :, z_index]
    initial = result["initial_intensity"][:, :, z_index]
    final = result["final_intensity"][:, :, z_index]
    phase = result["phase"]
    history = result["history"]
    summary = result["summary"]

    initial_plot = initial / max(float(np.max(initial)), np.finfo(float).eps)
    final_plot = final / max(float(np.max(final)), np.finfo(float).eps)

    fig = plt.figure(figsize=(15.2, 9.7), dpi=180)
    grid = fig.add_gridspec(2, 3, left=0.060, right=0.965, top=0.840, bottom=0.085, wspace=0.34, hspace=0.42)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    ax_curve = fig.add_subplot(grid[1, :2])
    ax_phase = fig.add_subplot(grid[1, 2])

    fig.text(
        0.060,
        0.972,
        "Phase-only pupil optimization forms a three-lobed annular High-NA focus",
        ha="left",
        va="top",
        fontsize=16,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.060,
        0.922,
        "Vectorial Debye-Wolf separable GPU solver on cylindrical (rho, psi, z) grid. Maps show z=0 intensity; target is an annular m=3 ROI.",
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
    )

    extent = [0.0, 2.0 * np.pi, float(rho_axis[0]), float(rho_axis[-1])]
    panels = [
        ("A. Target ROI weight", target, "Target weight"),
        ("B. Initial intensity", initial_plot, "Intensity / panel max"),
        ("C. Optimized intensity", final_plot, "Intensity / panel max"),
    ]
    for ax, (title, data, cbar_label) in zip(axes, panels):
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=sns.blend_palette([TOKENS["panel"], "#CEDFFE", "#5477C4", "#2E4780"], as_cmap=True),
            vmin=0.0,
        )
        ax.set_title(title, loc="left", fontsize=11, fontweight="semibold")
        ax.set_xlabel("Azimuth psi (rad)")
        ax.set_ylabel("Radius rho")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(np.pi))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: "0" if abs(value) < 1e-6 else ("pi" if abs(value - np.pi) < 1e-6 else ("2pi" if abs(value - 2 * np.pi) < 1e-6 else ""))))
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.018)
        cbar.set_label(cbar_label, fontsize=8.5, color=TOKENS["muted"])
        cbar.ax.tick_params(labelsize=7.5, colors=TOKENS["muted"])

    steps = [row["step"] for row in history]
    target_fraction = [row["target_fraction"] for row in history]
    center_fraction = [row["center_fraction"] for row in history]
    off_target_fraction = [row["off_target_fraction"] for row in history]
    loss = [row["loss"] for row in history]
    ax_curve.plot(steps, target_fraction, color=COLOR["blue"], label="Target ROI fraction", linewidth=1.5)
    ax_curve.plot(steps, off_target_fraction, color=COLOR["orange"], label="Off-target ring fraction", linewidth=1.5)
    ax_curve.plot(steps, center_fraction, color=COLOR["gold"], label="Center leakage fraction", linewidth=1.5)
    ax_curve.set_xlabel("Optimization step")
    ax_curve.set_ylabel("Energy-weighted fraction")
    ax_curve.set_title("D. Objective improves over phase-only updates", loc="left", fontsize=11, fontweight="semibold")
    ax_curve.legend(frameon=False, loc="upper left", ncol=2)
    ax_curve.grid(axis="both", color=TOKENS["grid"])
    ax2 = ax_curve.twinx()
    ax2.plot(steps, loss, color=COLOR["olive"], linestyle=":", label="Loss", linewidth=1.3)
    ax2.set_ylabel("Loss")
    ax2.grid(False)
    ax2.tick_params(colors=TOKENS["muted"])
    ax_curve.text(
        0.03,
        0.08,
        f"Target x{summary['target_fraction_gain']:.2f}; target/off x{summary['target_to_off_target_gain']:.2f}; m=3 {summary['final_m_contrast']:.2f}",
        transform=ax_curve.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color=TOKENS["ink"],
        bbox={"facecolor": TOKENS["panel"], "edgecolor": TOKENS["axis"], "boxstyle": "round,pad=0.25"},
    )

    im_phase = ax_phase.imshow(
        phase,
        aspect="auto",
        origin="lower",
        extent=[0.0, 2.0 * np.pi, 0.0, float(phase.shape[0] - 1)],
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    ax_phase.set_title("E. Learned shared pupil phase", loc="left", fontsize=11, fontweight="semibold")
    ax_phase.set_xlabel("Azimuth phi (rad)")
    ax_phase.set_ylabel("Theta sample")
    cbar = fig.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.018)
    cbar.set_label("Phase (rad)", fontsize=8.5, color=TOKENS["muted"])
    cbar.ax.tick_params(labelsize=7.5, colors=TOKENS["muted"])

    fig.text(
        0.060,
        0.025,
        f"Local snapshot: {summary['device_name']}, {summary['optimizer']}, torch {summary['torch_version']}, {summary['steps_completed']} steps, {summary['ms_per_step']:.2f} ms/step including Python loop and search overhead; not a hot-kernel timing.",
        ha="left",
        va="bottom",
        fontsize=8,
        color=TOKENS["muted"],
    )
    for ax in [*axes, ax_curve, ax_phase]:
        ax.tick_params(axis="both", labelsize=8.5, colors=TOKENS["muted"])
        sns.despine(ax=ax)
    png_path = output_prefix.with_name(output_prefix.name + "_figure.png")
    svg_path = output_prefix.with_name(output_prefix.name + "_figure.svg")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a physical High-NA cylindrical intensity-shaping demo.")
    parser.add_argument("--output-prefix", default="benchmark_results/high_na_physical_demo_intensity_shaping")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    parser.add_argument("--workload-set", choices=["quick", "representative", "large"], default="representative")
    parser.add_argument("--workload-index", type=int, default=1)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--optimizer", choices=["line-search", "momentum", "adam"], default="adam")
    parser.add_argument("--phase-step-radians", type=float, default=0.08)
    parser.add_argument("--momentum-beta", type=float, default=0.85)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.99)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--center-penalty", type=float, default=0.5)
    parser.add_argument("--off-target-penalty", type=float, default=0.5)
    parser.add_argument("--target-rho-fraction", type=float, default=0.58)
    parser.add_argument("--target-rho-sigma-fraction", type=float, default=0.14)
    parser.add_argument("--target-z-sigma-fraction", type=float, default=0.26)
    parser.add_argument("--angular-order", type=int, default=3)
    parser.add_argument("--angular-floor", type=float, default=0.08)
    parser.add_argument("--angular-phase", type=float, default=0.0)
    parser.add_argument("--h-margin", type=int, default=6)
    parser.add_argument("--apodization", choices=["sqrt-cos", "none"], default="sqrt-cos")
    parser.add_argument("--stop-patience", type=int, default=0)
    parser.add_argument("--stop-delta", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_prefix = ROOT / args.output_prefix
    result = run_demo(args)
    png_path, svg_path = plot_demo(result, output_prefix)
    history_path = output_prefix.with_name(output_prefix.name + "_history.csv")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")
    summary_md_path = output_prefix.with_name(output_prefix.name + "_summary.md")
    write_csv(history_path, result["history"])
    write_json(
        summary_path,
        {
            "config": vars(args),
            "summary": result["summary"],
            "history_csv": str(history_path),
            "arrays_npz": result["arrays_npz"],
            "figure_png": str(png_path),
            "figure_svg": str(svg_path),
        },
    )
    lines = [
        "# High-NA physical intensity-shaping demo",
        "",
        "This demo uses the separable vectorial GPU Debye-Wolf backend to optimize a shared phase-only pupil mask for a three-lobed annular focal intensity target on a cylindrical grid.",
        "",
        "## Results",
        "",
        f"- steps completed: `{result['summary']['steps_completed']}`",
        f"- optimizer: `{result['summary']['optimizer']}`",
        f"- target ROI fraction: `{result['summary']['initial_target_fraction']:.6g}` -> `{result['summary']['final_target_fraction']:.6g}` (`{result['summary']['target_fraction_gain']:.2f}x`)",
        f"- center leakage fraction: `{result['summary']['initial_center_fraction']:.6g}` -> `{result['summary']['final_center_fraction']:.6g}`",
        f"- off-target ring fraction: `{result['summary']['initial_off_target_fraction']:.6g}` -> `{result['summary']['final_off_target_fraction']:.6g}`",
        f"- target/off-target gain: `{result['summary']['target_to_off_target_gain']:.3g}x`",
        f"- annular m={result['summary']['target_order']} contrast: `{result['summary']['initial_m_contrast']:.6g}` -> `{result['summary']['final_m_contrast']:.6g}`",
        f"- elapsed: `{result['summary']['elapsed_s']:.4f} s`, `{result['summary']['ms_per_step']:.3f} ms/step`",
        f"- figure: `{png_path}`",
        "",
        "## Caveats",
        "",
        "- This is a local physical demo, not a matched comparison against an external microscope PSF package.",
        "- The objective maximizes a weighted cylindrical intensity fraction with center-leakage and optional off-target-ring penalties; it is not yet a full experimental optical-design problem.",
    ]
    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": result["summary"],
                "history_csv": str(history_path),
                "arrays_npz": result["arrays_npz"],
                "figure_png": str(png_path),
                "figure_svg": str(svg_path),
                "summary_md": str(summary_md_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
