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
from benchmark_high_na_vectorial_backpropagation import richards_wolf_jones_matrix  # noqa: E402
from demo_high_na_physical_intensity_shaping import (  # noqa: E402
    COLOR,
    TOKENS,
    apply_phase,
    intensity,
    rms_normalize,
    use_chart_theme,
    wrap_phase,
)


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


def x_polarized_jones(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    out = np.zeros((2, theta.size, phi.size), dtype=np.complex128)
    out[0] = 1.0 + 0.0j
    return out


def wrapped_angle(value: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * value))


def make_multitrap_target(
    *,
    rho_axis: np.ndarray,
    psi_axis: np.ndarray,
    z_axis: np.ndarray,
    trap_count: int,
    rho_fraction: float,
    rho_sigma_fraction: float,
    psi_sigma: float,
    z_sigma_fraction: float,
    angular_phase: float,
    background_floor: float,
) -> dict[str, np.ndarray]:
    rho_max = float(rho_axis[-1])
    z_max = max(float(np.max(np.abs(z_axis))), 1.0)
    rho = rho_axis[:, None, None]
    psi = psi_axis[None, :, None]
    z = z_axis[None, None, :]
    rho_center = float(rho_fraction) * rho_max
    rho_sigma = max(float(rho_sigma_fraction) * rho_max, np.finfo(float).eps)
    z_sigma = max(float(z_sigma_fraction) * z_max, np.finfo(float).eps)
    trap_weights: list[np.ndarray] = []
    broad_weights: list[np.ndarray] = []
    for index in range(int(trap_count)):
        center = float(angular_phase) + 2.0 * np.pi * index / float(trap_count)
        dpsi = wrapped_angle(psi - center)
        trap = np.exp(
            -0.5 * ((rho - rho_center) / rho_sigma) ** 2
            -0.5 * (dpsi / max(float(psi_sigma), np.finfo(float).eps)) ** 2
            -0.5 * (z / z_sigma) ** 2
        )
        trap = np.broadcast_to(trap, (rho_axis.size, psi_axis.size, z_axis.size)).copy()
        trap /= max(float(np.max(trap)), np.finfo(float).eps)
        trap_weights.append(trap)
        broad = np.exp(
            -0.5 * ((rho - rho_center) / (2.0 * rho_sigma)) ** 2
            -0.5 * (dpsi / max(2.0 * float(psi_sigma), np.finfo(float).eps)) ** 2
            -0.5 * (z / (1.5 * z_sigma)) ** 2
        )
        broad = np.broadcast_to(broad, trap.shape).copy()
        broad /= max(float(np.max(broad)), np.finfo(float).eps)
        broad_weights.append(broad)

    traps = np.stack(trap_weights, axis=0)
    broad = np.stack(broad_weights, axis=0)
    target = np.sum(traps, axis=0)
    target /= max(float(np.max(target)), np.finfo(float).eps)
    if background_floor > 0.0:
        target = np.clip(target + float(background_floor), 0.0, None)
    trap_union = np.clip(np.sum(broad, axis=0), 0.0, 1.0)
    background = 1.0 - trap_union
    return {
        "target": target.astype(np.float32),
        "trap_weights": traps.astype(np.float32),
        "trap_union": trap_union.astype(np.float32),
        "background": background.astype(np.float32),
    }


def normalized_intensity_loss(
    *,
    torch: Any,
    field: Any,
    target_norm: Any,
    sample_weight: Any,
    fit_weight: Any,
) -> tuple[Any, Any, dict[str, float]]:
    pred = intensity(field, torch)
    selected = fit_weight[None, :] * pred
    total = torch.clamp(torch.sum(selected, dim=1, keepdim=True), min=1e-30)
    pred_norm = selected / total
    error = pred_norm - target_norm
    weighted_error = sample_weight[None, :] * error
    loss = 0.5 * torch.mean(torch.sum(weighted_error * error, dim=1))
    projection = torch.sum(weighted_error * pred_norm, dim=1, keepdim=True)
    d_loss_d_intensity = fit_weight[None, :] * (weighted_error - projection) / (total * float(field.shape[0]))
    residual = 2.0 * d_loss_d_intensity[:, None, :] * field
    target_norm_l2 = torch.linalg.vector_norm(target_norm, dim=1)
    rel_l2 = torch.linalg.vector_norm(error, dim=1) / torch.clamp(target_norm_l2, min=1e-30)
    cosine = torch.sum(pred_norm * target_norm, dim=1) / torch.clamp(
        torch.linalg.vector_norm(pred_norm, dim=1) * target_norm_l2,
        min=1e-30,
    )
    return (
        loss,
        residual,
        {
            "loss": float(loss.detach().cpu().item()),
            "target_rel_l2": float(torch.mean(rel_l2).detach().cpu().item()),
            "target_cosine": float(torch.mean(cosine).detach().cpu().item()),
        },
    )


def trap_metrics(
    intensity_np: np.ndarray,
    target_np: np.ndarray,
    trap_weights_np: np.ndarray,
    trap_union_np: np.ndarray,
    background_np: np.ndarray,
) -> dict[str, float]:
    flat = intensity_np.reshape(-1).astype(float)
    total = max(float(np.sum(flat)), np.finfo(float).eps)
    target_flat = target_np.reshape(-1).astype(float)
    target_norm = target_flat / max(float(np.sum(target_flat)), np.finfo(float).eps)
    pred_norm = flat / total
    trap_energies = np.array(
        [
            float(np.sum(weight.reshape(-1) * flat) / total)
            for weight in trap_weights_np
        ],
        dtype=float,
    )
    mean_energy = max(float(np.mean(trap_energies)), np.finfo(float).eps)
    background_fraction = float(np.sum(background_np.reshape(-1) * flat) / total)
    trap_union_fraction = float(np.sum(trap_union_np.reshape(-1) * flat) / total)
    trap_area = max(float(np.sum(trap_union_np)), np.finfo(float).eps)
    background_area = max(float(np.sum(background_np)), np.finfo(float).eps)
    trap_density = float(np.sum(trap_union_np.reshape(-1) * flat) / trap_area)
    background_density = float(np.sum(background_np.reshape(-1) * flat) / background_area)
    cosine = float(
        np.dot(pred_norm, target_norm)
        / max(float(np.linalg.norm(pred_norm) * np.linalg.norm(target_norm)), np.finfo(float).eps)
    )
    rel_l2 = float(
        np.linalg.norm(pred_norm - target_norm)
        / max(float(np.linalg.norm(target_norm)), np.finfo(float).eps)
    )
    out = {
        "target_cosine": cosine,
        "target_rel_l2": rel_l2,
        "trap_union_fraction": trap_union_fraction,
        "background_fraction": background_fraction,
        "trap_to_background_density": trap_density / max(background_density, np.finfo(float).eps),
        "trap_uniformity_cv": float(np.std(trap_energies) / mean_energy),
        "trap_min_over_max": float(np.min(trap_energies) / max(float(np.max(trap_energies)), np.finfo(float).eps)),
    }
    for index, value in enumerate(trap_energies):
        out[f"trap_{index}_fraction"] = float(value)
    return out


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    theta, theta_weights = gauss_theta_grid(args.ntheta, args.theta_max)
    phi = np.linspace(0.0, 2.0 * np.pi, args.nphi, endpoint=False, dtype=float)
    rho_axis, psi_axis, z_axis, _, _, _ = focal_grid(
        nrho=args.nrho,
        npsi=args.npsi,
        nz=args.nz,
        rho_max=args.rho_max,
        z_max=args.z_max,
    )
    h_cutoff = min(
        args.nphi // 2,
        int(np.ceil(args.k * args.rho_max * np.sin(args.theta_max))) + int(args.h_margin),
    )
    target = make_multitrap_target(
        rho_axis=rho_axis,
        psi_axis=psi_axis,
        z_axis=z_axis,
        trap_count=args.trap_count,
        rho_fraction=args.trap_rho_fraction,
        rho_sigma_fraction=args.trap_rho_sigma_fraction,
        psi_sigma=args.trap_psi_sigma,
        z_sigma_fraction=args.trap_z_sigma_fraction,
        angular_phase=args.trap_angular_phase,
        background_floor=args.target_background_floor,
    )
    target_np = target["target"]
    trap_weights_np = target["trap_weights"]
    trap_union_np = target["trap_union"]
    background_np = target["background"]

    mixing = richards_wolf_jones_matrix(theta, phi, apodization=args.apodization)
    base_np = x_polarized_jones(theta, phi)
    initial_phase_np = np.zeros((args.ntheta, args.nphi), dtype=np.float32)
    if args.initial_seed_strength != 0.0:
        radial = np.sin(theta)[:, None] / max(np.sin(args.theta_max), np.finfo(float).eps)
        initial_phase_np = (
            float(args.initial_seed_strength)
            * radial**2
            * np.cos(int(args.trap_count) * phi[None, :] + 0.37)
        ).astype(np.float32)

    plan = TorchSeparableHarmonicDebyeWolfPlan.build(
        torch=torch,
        nphi=args.nphi,
        theta=theta,
        theta_weights=theta_weights,
        rho_axis=rho_axis,
        psi_axis=psi_axis,
        z_axis=z_axis,
        k=args.k,
        h_cutoff=h_cutoff,
        device=device,
        dtype=args.dtype,
    )
    complex_dtype = plan.complex_dtype
    real_dtype = plan.real_dtype
    base_t = torch.as_tensor(np.ascontiguousarray(base_np[None, ...]), dtype=complex_dtype, device=device)
    mixing_t = torch.as_tensor(np.ascontiguousarray(mixing), dtype=complex_dtype, device=device)
    fit_weight_np = np.ones(args.nrho * args.npsi * args.nz, dtype=np.float32)
    target_flat = target_np.reshape(-1)
    target_norm_np = fit_weight_np * target_flat
    target_norm_np = target_norm_np / max(float(np.sum(target_norm_np)), np.finfo(float).eps)
    target_t = torch.as_tensor(np.ascontiguousarray(target_norm_np[None, :]), dtype=real_dtype, device=device)
    fit_weight_t = torch.as_tensor(np.ascontiguousarray(fit_weight_np), dtype=real_dtype, device=device)
    target_boost = target_np.reshape(-1) / max(float(np.max(target_np)), np.finfo(float).eps)
    sample_weight_np = 1.0 + float(args.target_weight_boost) * target_boost
    sample_weight_t = torch.as_tensor(np.ascontiguousarray(sample_weight_np), dtype=real_dtype, device=device)

    phase_angle = torch.as_tensor(
        np.ascontiguousarray(initial_phase_np[None, ...]),
        dtype=real_dtype,
        device=device,
    )
    adam_m = torch.zeros_like(phase_angle)
    adam_v = torch.zeros_like(phase_angle)
    history: list[dict[str, Any]] = []
    start = time.perf_counter()

    def forward_loss_grad() -> tuple[Any, Any, dict[str, float]]:
        pupil = apply_phase(base_t, phase_angle, torch=torch, complex_dtype=complex_dtype)
        field = plan.evaluate_vectorial_batch(pupil, mixing_t)
        _, residual, metrics = normalized_intensity_loss(
            torch=torch,
            field=field,
            target_norm=target_t,
            sample_weight=sample_weight_t,
            fit_weight=fit_weight_t,
        )
        pupil_grad = plan.adjoint_vectorial_batch(residual, mixing_t)
        grad = torch.sum(torch.imag(torch.conj(pupil) * pupil_grad), dim=1)
        return field, grad, metrics

    with torch.no_grad():
        field0, _, metrics0 = forward_loss_grad()
        initial_intensity = to_numpy(torch, device, intensity(field0, torch)[0]).reshape(args.nrho, args.npsi, args.nz)
        history.append(
            {
                "step": 0,
                "accepted": True,
                "step_radians": 0.0,
                "gradient_rms": 0.0,
                **metrics0,
                **{f"initial_{key}": value for key, value in trap_metrics(initial_intensity, target_np, trap_weights_np, trap_union_np, background_np).items()},
            }
        )

        for step in range(1, args.steps + 1):
            _, grad, metrics = forward_loss_grad()
            grad_real = torch.real(grad)
            gradient_rms = torch.sqrt(torch.mean(grad_real * grad_real))
            normalized = rms_normalize(torch, grad_real)
            adam_m.mul_(float(args.adam_beta1)).add_(normalized, alpha=1.0 - float(args.adam_beta1))
            adam_v.mul_(float(args.adam_beta2)).addcmul_(normalized, normalized, value=1.0 - float(args.adam_beta2))
            bias_m = 1.0 - float(args.adam_beta1) ** step
            bias_v = 1.0 - float(args.adam_beta2) ** step
            direction = (adam_m / bias_m) / (torch.sqrt(adam_v / bias_v) + float(args.adam_eps))
            direction = rms_normalize(torch, direction)

            best_phase = phase_angle
            best_metrics = metrics
            best_step = 0.0
            for step_size in [
                args.phase_step_radians,
                0.5 * args.phase_step_radians,
                0.25 * args.phase_step_radians,
                1.5 * args.phase_step_radians,
                -0.5 * args.phase_step_radians,
            ]:
                trial_phase = wrap_phase(torch, phase_angle - float(step_size) * direction)
                trial_pupil = apply_phase(base_t, trial_phase, torch=torch, complex_dtype=complex_dtype)
                trial_field = plan.evaluate_vectorial_batch(trial_pupil, mixing_t)
                _, _, trial_metrics = normalized_intensity_loss(
                    torch=torch,
                    field=trial_field,
                    target_norm=target_t,
                    sample_weight=sample_weight_t,
                    fit_weight=fit_weight_t,
                )
                if trial_metrics["loss"] < best_metrics["loss"]:
                    best_metrics = trial_metrics
                    best_phase = trial_phase
                    best_step = float(step_size)
            phase_angle = best_phase
            history.append(
                {
                    "step": step,
                    "accepted": best_step != 0.0,
                    "step_radians": best_step,
                    "gradient_rms": float(gradient_rms.detach().cpu().item()),
                    **best_metrics,
                }
            )
            if args.stop_patience > 0 and len(history) > args.stop_patience:
                recent = history[-args.stop_patience :]
                if max(item["target_cosine"] for item in recent) - min(item["target_cosine"] for item in recent) < args.stop_delta:
                    break

        final_field, _, final_metrics = forward_loss_grad()
        synchronize(torch, device)
        elapsed_s = time.perf_counter() - start
        final_intensity = to_numpy(torch, device, intensity(final_field, torch)[0]).reshape(args.nrho, args.npsi, args.nz)
        phase_np = to_numpy(torch, device, phase_angle[0])

    initial_metrics = trap_metrics(initial_intensity, target_np, trap_weights_np, trap_union_np, background_np)
    final_metrics_np = trap_metrics(final_intensity, target_np, trap_weights_np, trap_union_np, background_np)
    summary = {
        "status": "ok",
        "demo": "phase_only_multitrap_high_na",
        "device": str(device),
        "device_name": device_name(torch, device),
        "torch_version": package_version("torch"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "dtype": args.dtype,
        "trap_count": int(args.trap_count),
        "steps_requested": int(args.steps),
        "steps_completed": int(history[-1]["step"]),
        "elapsed_s": float(elapsed_s),
        "ms_per_step": float(1000.0 * elapsed_s / max(1, int(history[-1]["step"]))),
        "h_cutoff": int(h_cutoff),
        "used_modes": int(plan.used_modes),
        "targets": int(args.nrho * args.npsi * args.nz),
        "target_weight_boost": float(args.target_weight_boost),
        "initial_loss": float(metrics0["loss"]),
        "final_loss": float(final_metrics["loss"]),
        "loss_reduction": float(metrics0["loss"] / max(float(final_metrics["loss"]), 1e-30)),
        **{f"initial_{key}": value for key, value in initial_metrics.items()},
        **{f"final_{key}": value for key, value in final_metrics_np.items()},
        "gpu_peak_allocated_mib": None
        if device.type != "cuda"
        else float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
    }

    output_prefix = ROOT / args.output_prefix
    npz_path = output_prefix.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        theta=theta,
        phi=phi,
        rho_axis=rho_axis,
        psi_axis=psi_axis,
        z_axis=z_axis,
        target_intensity=target_np,
        trap_weights=trap_weights_np,
        trap_union=trap_union_np,
        background_weight=background_np,
        initial_intensity=initial_intensity,
        final_intensity=final_intensity,
        phase=phase_np,
    )
    return {
        "summary": summary,
        "history": history,
        "arrays_npz": str(npz_path),
        "rho_axis": rho_axis,
        "psi_axis": psi_axis,
        "z_axis": z_axis,
        "target_intensity": target_np,
        "trap_weights": trap_weights_np,
        "initial_intensity": initial_intensity,
        "final_intensity": final_intensity,
        "phase": phase_np,
    }


def normalized_panel(data: np.ndarray) -> np.ndarray:
    return data / max(float(np.max(data)), np.finfo(float).eps)


def plot_demo(result: dict[str, Any], output_prefix: Path) -> tuple[Path, Path]:
    use_chart_theme()
    rho_axis = result["rho_axis"]
    psi_axis = result["psi_axis"]
    z_axis = result["z_axis"]
    z_index = int(np.argmin(np.abs(z_axis)))
    target = normalized_panel(result["target_intensity"][:, :, z_index])
    initial = normalized_panel(result["initial_intensity"][:, :, z_index])
    final = normalized_panel(result["final_intensity"][:, :, z_index])
    phase = result["phase"]
    history = result["history"]
    summary = result["summary"]

    fig = plt.figure(figsize=(15.6, 9.8), dpi=180)
    grid = fig.add_gridspec(2, 3, left=0.060, right=0.965, top=0.845, bottom=0.085, wspace=0.34, hspace=0.42)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    ax_bar = fig.add_subplot(grid[1, 0])
    ax_history = fig.add_subplot(grid[1, 1])
    ax_phase = fig.add_subplot(grid[1, 2])

    fig.text(
        0.060,
        0.975,
        f"Phase-only vectorial High-NA pupil forms {summary['trap_count']} discrete focal traps",
        ha="left",
        va="top",
        fontsize=16,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.060,
        0.928,
        "The objective matches a normalized multi-trap intensity target on a cylindrical focal grid; metrics report trap energy, uniformity, and background leakage.",
        ha="left",
        va="top",
        fontsize=9.4,
        color=TOKENS["muted"],
    )

    extent = [0.0, 2.0 * np.pi, float(rho_axis[0]), float(rho_axis[-1])]
    cmap_int = sns.blend_palette([TOKENS["panel"], "#CEDFFE", "#5477C4", "#2E4780"], as_cmap=True)
    for ax, data, title in [
        (axes[0], target, "A. Target traps"),
        (axes[1], initial, "B. Initial focus"),
        (axes[2], final, "C. Optimized traps"),
    ]:
        im = ax.imshow(data, aspect="auto", origin="lower", extent=extent, cmap=cmap_int, vmin=0.0)
        ax.set_title(title, loc="left", fontsize=11, fontweight="semibold")
        ax.set_xlabel("Focal azimuth psi (rad)")
        ax.set_ylabel("Radius rho")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(np.pi))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda value, _: "0"
                if abs(value) < 1e-6
                else ("pi" if abs(value - np.pi) < 1e-6 else ("2pi" if abs(value - 2 * np.pi) < 1e-6 else ""))
            )
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.018)
        cbar.set_label("Intensity / panel max", fontsize=8.3, color=TOKENS["muted"])
        cbar.ax.tick_params(labelsize=7.4, colors=TOKENS["muted"])

    trap_count = int(summary["trap_count"])
    x = np.arange(trap_count)
    initial_traps = [float(summary[f"initial_trap_{index}_fraction"]) for index in range(trap_count)]
    final_traps = [float(summary[f"final_trap_{index}_fraction"]) for index in range(trap_count)]
    width = 0.38
    ax_bar.bar(x - width / 2, initial_traps, width, label="initial", color=COLOR["blue_light"])
    ax_bar.bar(x + width / 2, final_traps, width, label="optimized", color=COLOR["blue"])
    ax_bar.set_title("D. Trap energy fractions", loc="left", fontsize=11, fontweight="semibold")
    ax_bar.set_xlabel("Trap index")
    ax_bar.set_ylabel("Fraction of total intensity")
    ax_bar.legend(frameon=False)
    ax_bar.grid(axis="y", color=TOKENS["grid"])

    steps = [row["step"] for row in history]
    target_cos = [row["target_cosine"] for row in history]
    target_l2 = [row["target_rel_l2"] for row in history]
    loss = [row["loss"] for row in history]
    ax_history.plot(steps, target_cos, color=COLOR["orange"], linewidth=1.4, label="Target cosine")
    ax_history.plot(steps, target_l2, color=COLOR["blue"], linewidth=1.4, label="Target rel-L2")
    ax_history.set_title("E. Intensity target fit", loc="left", fontsize=11, fontweight="semibold")
    ax_history.set_xlabel("Optimization step")
    ax_history.set_ylabel("Metric")
    ax_history.legend(frameon=False, loc="upper right")
    ax_history.grid(axis="both", color=TOKENS["grid"])
    ax_loss = ax_history.twinx()
    ax_loss.plot(steps, loss, color=COLOR["olive"], linestyle=":", linewidth=1.1)
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(False)
    ax_history.text(
        0.04,
        0.08,
        f"cos {summary['initial_target_cosine']:.3f} -> {summary['final_target_cosine']:.3f}\n"
        f"bg {summary['initial_background_fraction']:.3f} -> {summary['final_background_fraction']:.3f}\n"
        f"uniform CV {summary['final_trap_uniformity_cv']:.3f}",
        transform=ax_history.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
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
    ax_phase.set_title("F. Learned phase-only pupil mask", loc="left", fontsize=11, fontweight="semibold")
    ax_phase.set_xlabel("Pupil azimuth phi (rad)")
    ax_phase.set_ylabel("Theta sample")
    cbar = fig.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.018)
    cbar.set_label("Phase (rad)", fontsize=8.3, color=TOKENS["muted"])
    cbar.ax.tick_params(labelsize=7.4, colors=TOKENS["muted"])

    fig.text(
        0.060,
        0.026,
        f"Local snapshot: {summary['device_name']}, torch {summary['torch_version']}, {summary['steps_completed']} Adam steps, {summary['ms_per_step']:.2f} ms/step including Python loop and line search.",
        ha="left",
        va="bottom",
        fontsize=8,
        color=TOKENS["muted"],
    )
    for ax in [*axes, ax_bar, ax_history, ax_phase]:
        ax.tick_params(axis="both", labelsize=8.4, colors=TOKENS["muted"])
        sns.despine(ax=ax)

    png_path = output_prefix.with_name(output_prefix.name + "_figure.png")
    svg_path = output_prefix.with_name(output_prefix.name + "_figure.svg")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize a phase-only vectorial High-NA pupil for a discrete multi-trap focal pattern.")
    parser.add_argument("--output-prefix", default="benchmark_results/high_na_multitrap_phase_mask")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    parser.add_argument("--ntheta", type=int, default=28)
    parser.add_argument("--nphi", type=int, default=128)
    parser.add_argument("--nrho", type=int, default=23)
    parser.add_argument("--npsi", type=int, default=96)
    parser.add_argument("--nz", type=int, default=5)
    parser.add_argument("--rho-max", type=float, default=2.2)
    parser.add_argument("--z-max", type=float, default=1.0)
    parser.add_argument("--theta-max", type=float, default=1.0)
    parser.add_argument("--k", type=float, default=10.0)
    parser.add_argument("--h-margin", type=int, default=16)
    parser.add_argument("--apodization", choices=["sqrt-cos", "none"], default="sqrt-cos")
    parser.add_argument("--trap-count", type=int, default=5)
    parser.add_argument("--trap-rho-fraction", type=float, default=0.58)
    parser.add_argument("--trap-rho-sigma-fraction", type=float, default=0.075)
    parser.add_argument("--trap-psi-sigma", type=float, default=0.115)
    parser.add_argument("--trap-z-sigma-fraction", type=float, default=0.22)
    parser.add_argument("--trap-angular-phase", type=float, default=0.18)
    parser.add_argument("--target-background-floor", type=float, default=0.0)
    parser.add_argument("--target-weight-boost", type=float, default=35.0)
    parser.add_argument("--initial-seed-strength", type=float, default=0.03)
    parser.add_argument("--steps", type=int, default=130)
    parser.add_argument("--phase-step-radians", type=float, default=0.075)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.99)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--stop-patience", type=int, default=0)
    parser.add_argument("--stop-delta", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trap_count < 2:
        raise ValueError("trap_count must be at least 2")
    output_prefix = ROOT / args.output_prefix
    result = run_demo(args)
    png_path, svg_path = plot_demo(result, output_prefix)
    history_path = output_prefix.with_name(output_prefix.name + "_history.csv")
    summary_json_path = output_prefix.with_name(output_prefix.name + "_summary.json")
    summary_md_path = output_prefix.with_name(output_prefix.name + "_summary.md")
    write_csv(history_path, result["history"])
    write_json(
        summary_json_path,
        {
            "config": vars(args),
            "summary": result["summary"],
            "history_csv": str(history_path),
            "arrays_npz": result["arrays_npz"],
            "figure_png": str(png_path),
            "figure_svg": str(svg_path),
        },
    )
    summary = result["summary"]
    lines = [
        "# High-NA phase-only multi-trap demo",
        "",
        "A shared phase-only pupil mask is optimized to form discrete focal traps on a vectorial high-NA cylindrical grid.",
        "",
        "## Results",
        "",
        f"- traps: `{summary['trap_count']}`",
        f"- steps completed: `{summary['steps_completed']}`",
        f"- target cosine: `{summary['initial_target_cosine']:.6g}` -> `{summary['final_target_cosine']:.6g}`",
        f"- target rel-L2: `{summary['initial_target_rel_l2']:.6g}` -> `{summary['final_target_rel_l2']:.6g}`",
        f"- trap-union energy fraction: `{summary['initial_trap_union_fraction']:.6g}` -> `{summary['final_trap_union_fraction']:.6g}`",
        f"- background energy fraction: `{summary['initial_background_fraction']:.6g}` -> `{summary['final_background_fraction']:.6g}`",
        f"- trap/background density ratio: `{summary['initial_trap_to_background_density']:.6g}` -> `{summary['final_trap_to_background_density']:.6g}`",
        f"- trap uniformity CV: `{summary['initial_trap_uniformity_cv']:.6g}` -> `{summary['final_trap_uniformity_cv']:.6g}`",
        f"- trap min/max energy: `{summary['initial_trap_min_over_max']:.6g}` -> `{summary['final_trap_min_over_max']:.6g}`",
        f"- elapsed: `{summary['elapsed_s']:.4f} s`, `{summary['ms_per_step']:.3f} ms/step`",
        f"- figure: `{png_path}`",
        "",
        "## Interpretation",
        "",
        "- This is an intensity-only design demo. Unlike aberration correction, there is no unique expected pupil phase.",
        "- The useful readout is target-pattern agreement, trap uniformity, and background leakage.",
        "- This target is intentionally discrete and non-axisymmetric, so it exercises the repeated phase-mask design regime rather than a simple analytic donut.",
    ]
    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": summary,
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
