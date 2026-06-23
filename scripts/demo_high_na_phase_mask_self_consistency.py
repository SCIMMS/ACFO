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


def make_hidden_phase(
    *,
    theta: np.ndarray,
    phi: np.ndarray,
    theta_max: float,
    strength: float,
    hidden_order: int,
) -> np.ndarray:
    radial = (theta[:, None] / max(float(theta_max), np.finfo(float).eps)).clip(0.0, 1.0)
    az = phi[None, :]
    phase = (
        0.65 * radial**2 * np.cos(int(hidden_order) * az + 0.35)
        + 0.35 * radial * np.sin((int(hidden_order) + 2) * az - 0.7)
        + 0.20 * radial**3 * np.cos(2.0 * az + 1.1)
    )
    phase = float(strength) * phase
    return np.angle(np.exp(1j * phase)).astype(np.float32, copy=False)


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
    target_norm_safe = torch.linalg.vector_norm(target_norm, dim=1)
    rel_l2 = torch.linalg.vector_norm(error, dim=1) / torch.clamp(target_norm_safe, min=1e-30)
    cosine = torch.sum(pred_norm * target_norm, dim=1) / torch.clamp(
        torch.linalg.vector_norm(pred_norm, dim=1) * target_norm_safe,
        min=1e-30,
    )
    return (
        loss,
        residual,
        {
            "loss": float(loss.detach().cpu().item()),
            "intensity_rel_l2": float(torch.mean(rel_l2).detach().cpu().item()),
            "intensity_cosine": float(torch.mean(cosine).detach().cpu().item()),
        },
    )


def make_fit_weight(*, nrho: int, npsi: int, nz: int, z_axis: np.ndarray, scope: str) -> np.ndarray:
    weight = np.zeros((nrho, npsi, nz), dtype=np.float32)
    if scope == "volume":
        weight.fill(1.0)
    elif scope == "z0":
        weight[:, :, int(np.argmin(np.abs(z_axis)))] = 1.0
    else:
        raise ValueError(f"unknown fit scope: {scope}")
    return weight.reshape(-1)


def normalized_metrics_np(intensity_np: np.ndarray, target_np: np.ndarray, fit_weight_np: np.ndarray) -> dict[str, float]:
    selected = fit_weight_np * intensity_np
    target_selected = fit_weight_np * target_np
    selected_norm = selected / max(float(np.sum(selected)), np.finfo(float).eps)
    target_norm = target_selected / max(float(np.sum(target_selected)), np.finfo(float).eps)
    target_norm_l2 = max(float(np.linalg.norm(target_norm)), np.finfo(float).eps)
    rel_l2 = float(np.linalg.norm(selected_norm - target_norm) / target_norm_l2)
    cosine = float(
        np.dot(selected_norm, target_norm)
        / max(float(np.linalg.norm(selected_norm) * np.linalg.norm(target_norm)), np.finfo(float).eps)
    )
    return {"rel_l2": rel_l2, "cosine": cosine}


def phase_metrics(
    *,
    recovered: np.ndarray,
    hidden: np.ndarray,
    weight: np.ndarray,
) -> dict[str, float]:
    w = np.asarray(weight, dtype=float)
    w = w / max(float(np.sum(w)), np.finfo(float).eps)
    diff = np.angle(np.exp(1j * (recovered - hidden)))
    offset = np.angle(np.sum(w * np.exp(1j * diff)))
    aligned = np.angle(np.exp(1j * (diff - offset)))
    corr = abs(np.sum(w * np.exp(1j * aligned)))
    rmse = np.sqrt(float(np.sum(w * aligned * aligned)))
    return {"phase_correlation": float(corr), "phase_rmse_rad": float(rmse), "phase_global_offset_rad": float(offset)}


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
    hidden_phase_np = make_hidden_phase(
        theta=theta,
        phi=phi,
        theta_max=workload.theta_max,
        strength=args.hidden_phase_strength,
        hidden_order=args.hidden_order,
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
    base_t = torch.as_tensor(np.ascontiguousarray(base_np[None, ...]), dtype=complex_dtype, device=device)
    mixing_t = torch.as_tensor(np.ascontiguousarray(mixing), dtype=complex_dtype, device=device)
    hidden_phase_t = torch.as_tensor(hidden_phase_np[None, ...], dtype=real_dtype, device=device)

    with torch.no_grad():
        hidden_pupil = apply_phase(base_t, hidden_phase_t, torch=torch, complex_dtype=complex_dtype)
        target_field = plan.evaluate_vectorial_batch(hidden_pupil, mixing_t)
        target_intensity_t = intensity(target_field, torch)
        fit_weight_np = make_fit_weight(
            nrho=workload.nrho,
            npsi=workload.npsi,
            nz=workload.nz,
            z_axis=z_axis,
            scope=args.fit_scope,
        )
        fit_weight_t = torch.as_tensor(np.ascontiguousarray(fit_weight_np), dtype=real_dtype, device=device)
        target_selected_t = fit_weight_t[None, :] * target_intensity_t
        target_total = torch.clamp(torch.sum(target_selected_t, dim=1, keepdim=True), min=1e-30)
        target_norm_t = target_selected_t / target_total
        nfit = max(float(np.sum(fit_weight_np > 0.0)), 1.0)
        sample_weight = fit_weight_t * (target_norm_t[0] + float(args.background_weight) / nfit)
        sample_weight = sample_weight / torch.clamp(torch.sum(sample_weight) / float(nfit), min=1e-30)
        target_intensity = to_numpy(torch, device, target_intensity_t[0])

    phase_angle = torch.zeros((1, workload.ntheta, workload.nphi), dtype=real_dtype, device=device)
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
            target_norm=target_norm_t,
            sample_weight=sample_weight,
            fit_weight=fit_weight_t,
        )
        pupil_grad = plan.adjoint_vectorial_batch(residual, mixing_t)
        grad = torch.sum(torch.imag(torch.conj(pupil) * pupil_grad), dim=1)
        return field, grad, metrics

    with torch.no_grad():
        field0, _, metrics0 = forward_loss_grad()
        initial_intensity = to_numpy(torch, device, intensity(field0, torch)[0])
        history.append(
            {
                "step": 0,
                "accepted": True,
                "step_radians": 0.0,
                "gradient_rms": 0.0,
                **metrics0,
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
                    target_norm=target_norm_t,
                    sample_weight=sample_weight,
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
                if max(item["intensity_rel_l2"] for item in recent) - min(item["intensity_rel_l2"] for item in recent) < args.stop_delta:
                    break

        recovered_field, _, metrics_final = forward_loss_grad()
        synchronize(torch, device)
        elapsed_s = time.perf_counter() - start
        recovered_intensity = to_numpy(torch, device, intensity(recovered_field, torch)[0])
        recovered_phase_np = to_numpy(torch, device, phase_angle[0])

    if base_np.ndim == 3:
        pupil_weight = np.sum(np.abs(base_np) ** 2, axis=0)
    elif base_np.ndim == 4:
        pupil_weight = np.sum(np.abs(base_np) ** 2, axis=(0, 1))
    else:
        raise ValueError(f"unexpected pupil shape: {base_np.shape}")
    phase_stats = phase_metrics(recovered=recovered_phase_np, hidden=hidden_phase_np, weight=pupil_weight)
    volume_weight_np = np.ones_like(fit_weight_np, dtype=np.float32)
    z0_weight_np = make_fit_weight(
        nrho=workload.nrho,
        npsi=workload.npsi,
        nz=workload.nz,
        z_axis=z_axis,
        scope="z0",
    )
    initial_fit_metrics = normalized_metrics_np(initial_intensity, target_intensity, fit_weight_np)
    final_fit_metrics = normalized_metrics_np(recovered_intensity, target_intensity, fit_weight_np)
    initial_volume_metrics = normalized_metrics_np(initial_intensity, target_intensity, volume_weight_np)
    final_volume_metrics = normalized_metrics_np(recovered_intensity, target_intensity, volume_weight_np)
    initial_z0_metrics = normalized_metrics_np(initial_intensity, target_intensity, z0_weight_np)
    final_z0_metrics = normalized_metrics_np(recovered_intensity, target_intensity, z0_weight_np)

    summary = {
        "status": "ok",
        "demo": "phase_mask_self_consistency",
        "workload": workload.name,
        "pupil_case": workload.pupil_case,
        "device": str(device),
        "device_name": device_name(torch, device),
        "torch_version": package_version("torch"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "dtype": args.dtype,
        "optimizer": "adam",
        "fit_scope": args.fit_scope,
        "fit_samples": int(np.sum(fit_weight_np > 0.0)),
        "steps_requested": int(args.steps),
        "steps_completed": int(history[-1]["step"]),
        "elapsed_s": float(elapsed_s),
        "ms_per_step": float(1000.0 * elapsed_s / max(1, history[-1]["step"])),
        "h_cutoff": int(h_cutoff),
        "used_modes": int(plan.used_modes),
        "targets": int(workload.nrho * workload.npsi * workload.nz),
        "hidden_phase_strength": float(args.hidden_phase_strength),
        "hidden_order": int(args.hidden_order),
        "initial_fit_intensity_rel_l2": initial_fit_metrics["rel_l2"],
        "final_fit_intensity_rel_l2": final_fit_metrics["rel_l2"],
        "fit_intensity_rel_l2_reduction": initial_fit_metrics["rel_l2"] / max(final_fit_metrics["rel_l2"], 1e-30),
        "initial_fit_intensity_cosine": initial_fit_metrics["cosine"],
        "final_fit_intensity_cosine": final_fit_metrics["cosine"],
        "initial_intensity_rel_l2": initial_volume_metrics["rel_l2"],
        "final_intensity_rel_l2": final_volume_metrics["rel_l2"],
        "intensity_rel_l2_reduction": initial_volume_metrics["rel_l2"] / max(final_volume_metrics["rel_l2"], 1e-30),
        "initial_intensity_cosine": initial_volume_metrics["cosine"],
        "final_intensity_cosine": final_volume_metrics["cosine"],
        "initial_z0_intensity_rel_l2": initial_z0_metrics["rel_l2"],
        "final_z0_intensity_rel_l2": final_z0_metrics["rel_l2"],
        "initial_z0_intensity_cosine": initial_z0_metrics["cosine"],
        "final_z0_intensity_cosine": final_z0_metrics["cosine"],
        **phase_stats,
        "final_loss": float(metrics_final["loss"]),
        "gpu_peak_allocated_mib": None
        if device.type != "cuda"
        else float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
    }
    output_prefix = ROOT / args.output_prefix
    npz_path = output_prefix.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        rho_axis=rho_axis,
        psi_axis=psi_axis,
        z_axis=z_axis,
        theta=theta,
        phi=phi,
        hidden_phase=hidden_phase_np,
        recovered_phase=recovered_phase_np,
        initial_intensity=initial_intensity.reshape(workload.nrho, workload.npsi, workload.nz),
        target_intensity=target_intensity.reshape(workload.nrho, workload.npsi, workload.nz),
        recovered_intensity=recovered_intensity.reshape(workload.nrho, workload.npsi, workload.nz),
        fit_weight=fit_weight_np.reshape(workload.nrho, workload.npsi, workload.nz),
    )
    return {
        "summary": summary,
        "history": history,
        "arrays_npz": str(npz_path),
        "rho_axis": rho_axis,
        "psi_axis": psi_axis,
        "z_axis": z_axis,
        "hidden_phase": hidden_phase_np,
        "recovered_phase": recovered_phase_np,
        "initial_intensity": initial_intensity.reshape(workload.nrho, workload.npsi, workload.nz),
        "target_intensity": target_intensity.reshape(workload.nrho, workload.npsi, workload.nz),
        "recovered_intensity": recovered_intensity.reshape(workload.nrho, workload.npsi, workload.nz),
        "fit_weight": fit_weight_np.reshape(workload.nrho, workload.npsi, workload.nz),
    }


def plot_demo(result: dict[str, Any], output_prefix: Path) -> tuple[Path, Path]:
    use_chart_theme()
    rho_axis = result["rho_axis"]
    psi_axis = result["psi_axis"]
    z_axis = result["z_axis"]
    z_index = int(np.argmin(np.abs(z_axis)))
    target = result["target_intensity"][:, :, z_index]
    initial = result["initial_intensity"][:, :, z_index]
    recovered = result["recovered_intensity"][:, :, z_index]
    hidden_phase = result["hidden_phase"]
    recovered_phase = result["recovered_phase"]
    history = result["history"]
    summary = result["summary"]

    target_plot = target / max(float(np.max(target)), np.finfo(float).eps)
    initial_plot = initial / max(float(np.max(initial)), np.finfo(float).eps)
    recovered_plot = recovered / max(float(np.max(recovered)), np.finfo(float).eps)
    residual_plot = recovered_plot - target_plot

    fig = plt.figure(figsize=(15.2, 9.7), dpi=180)
    grid = fig.add_gridspec(2, 3, left=0.060, right=0.965, top=0.840, bottom=0.085, wspace=0.34, hspace=0.42)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    ax_curve = fig.add_subplot(grid[1, :2])
    ax_phase = fig.add_subplot(grid[1, 2])
    fig.text(
        0.060,
        0.972,
        "Self-consistency: recover a phase-only mask from its synthetic High-NA intensity",
        ha="left",
        va="top",
        fontsize=15.2,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.060,
        0.922,
        f"A hidden pupil phase generates the target through the vectorial Debye-Wolf solver; Adam fits a phase-only mask using {summary['fit_scope']} intensity samples.",
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
    )

    extent = [0.0, 2.0 * np.pi, float(rho_axis[0]), float(rho_axis[-1])]
    cmap_int = sns.blend_palette([TOKENS["panel"], "#CEDFFE", "#5477C4", "#2E4780"], as_cmap=True)
    panels = [
        ("A. Target intensity from hidden phase", target_plot, "Intensity / panel max", cmap_int, 0.0, None),
        ("B. Zero-phase initial intensity", initial_plot, "Intensity / panel max", cmap_int, 0.0, None),
        ("C. Recovered-mask intensity", recovered_plot, "Intensity / panel max", cmap_int, 0.0, None),
    ]
    for ax, (title, data, label, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(data, aspect="auto", origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, loc="left", fontsize=11, fontweight="semibold")
        ax.set_xlabel("Azimuth psi (rad)")
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
        cbar.set_label(label, fontsize=8.5, color=TOKENS["muted"])
        cbar.ax.tick_params(labelsize=7.5, colors=TOKENS["muted"])

    steps = [row["step"] for row in history]
    rel_l2 = [row["intensity_rel_l2"] for row in history]
    cosine = [row["intensity_cosine"] for row in history]
    loss = [row["loss"] for row in history]
    ax_curve.plot(steps, rel_l2, color=COLOR["blue"], linewidth=1.6, label="Normalized intensity rel-L2")
    ax_curve.plot(steps, cosine, color=COLOR["orange"], linewidth=1.4, label="Intensity cosine")
    ax_curve.set_title("D. Intensity self-consistency improves", loc="left", fontsize=11, fontweight="semibold")
    ax_curve.set_xlabel("Optimization step")
    ax_curve.set_ylabel("Intensity metric")
    ax_curve.legend(frameon=False, loc="upper right")
    ax_curve.grid(axis="both", color=TOKENS["grid"])
    ax_loss = ax_curve.twinx()
    ax_loss.plot(steps, loss, color=COLOR["olive"], linestyle=":", linewidth=1.2, label="Loss")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(False)
    ax_loss.tick_params(colors=TOKENS["muted"])
    ax_curve.text(
        0.03,
        0.08,
        f"fit L2 {summary['initial_fit_intensity_rel_l2']:.3f} -> {summary['final_fit_intensity_rel_l2']:.3f}; volume L2 {summary['final_intensity_rel_l2']:.3f}; phase corr {summary['phase_correlation']:.3f}",
        transform=ax_curve.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color=TOKENS["ink"],
        bbox={"facecolor": TOKENS["panel"], "edgecolor": TOKENS["axis"], "boxstyle": "round,pad=0.25"},
    )

    phase_stack = np.concatenate([hidden_phase, recovered_phase], axis=0)
    im_phase = ax_phase.imshow(
        phase_stack,
        aspect="auto",
        origin="lower",
        extent=[0.0, 2.0 * np.pi, 0.0, float(phase_stack.shape[0] - 1)],
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    ax_phase.axhline(hidden_phase.shape[0] - 0.5, color=TOKENS["axis"], linewidth=1.0)
    ax_phase.set_title("E. Hidden phase (bottom) and recovered phase (top)", loc="left", fontsize=10.5, fontweight="semibold")
    ax_phase.set_xlabel("Azimuth phi (rad)")
    ax_phase.set_ylabel("Theta sample")
    cbar = fig.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.018)
    cbar.set_label("Phase (rad)", fontsize=8.5, color=TOKENS["muted"])
    cbar.ax.tick_params(labelsize=7.5, colors=TOKENS["muted"])

    fig.text(
        0.060,
        0.025,
        f"Local snapshot: {summary['device_name']}, torch {summary['torch_version']}, {summary['steps_completed']} Adam steps, {summary['ms_per_step']:.2f} ms/step including Python loop and search overhead.",
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

    residual_png = output_prefix.with_name(output_prefix.name + "_residual.png")
    fig2, ax = plt.subplots(figsize=(5.8, 4.2), dpi=180)
    im = ax.imshow(residual_plot, aspect="auto", origin="lower", extent=extent, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title("Recovered - target normalized intensity at z=0", loc="left", fontsize=10.5, fontweight="semibold")
    ax.set_xlabel("Azimuth psi (rad)")
    ax.set_ylabel("Radius rho")
    fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    sns.despine(ax=ax)
    fig2.savefig(residual_png, dpi=220, bbox_inches="tight")
    plt.close(fig2)
    return png_path, svg_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run High-NA phase-mask self-consistency recovery.")
    parser.add_argument("--output-prefix", default="benchmark_results/high_na_phase_mask_self_consistency")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    parser.add_argument("--workload-set", choices=["quick", "representative", "large"], default="representative")
    parser.add_argument("--workload-index", type=int, default=1)
    parser.add_argument("--fit-scope", choices=["volume", "z0"], default="volume")
    parser.add_argument("--steps", type=int, default=140)
    parser.add_argument("--phase-step-radians", type=float, default=0.06)
    parser.add_argument("--hidden-phase-strength", type=float, default=1.1)
    parser.add_argument("--hidden-order", type=int, default=3)
    parser.add_argument("--background-weight", type=float, default=0.05)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.99)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
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
    summary = result["summary"]
    lines = [
        "# High-NA phase-mask self-consistency demo",
        "",
        "A hidden phase-only pupil mask generates a synthetic focal intensity volume. Starting from a zero phase mask, the optimizer then fits another phase-only mask to reproduce that normalized intensity.",
        "",
        "## Results",
        "",
        f"- steps completed: `{summary['steps_completed']}`",
        f"- fit scope: `{summary['fit_scope']}` (`{summary['fit_samples']}` samples)",
        f"- fit-scope normalized intensity rel-L2: `{summary['initial_fit_intensity_rel_l2']:.6g}` -> `{summary['final_fit_intensity_rel_l2']:.6g}` (`{summary['fit_intensity_rel_l2_reduction']:.2f}x` reduction)",
        f"- fit-scope intensity cosine: `{summary['initial_fit_intensity_cosine']:.6g}` -> `{summary['final_fit_intensity_cosine']:.6g}`",
        f"- full-volume normalized intensity rel-L2: `{summary['initial_intensity_rel_l2']:.6g}` -> `{summary['final_intensity_rel_l2']:.6g}` (`{summary['intensity_rel_l2_reduction']:.2f}x` reduction)",
        f"- full-volume intensity cosine: `{summary['initial_intensity_cosine']:.6g}` -> `{summary['final_intensity_cosine']:.6g}`",
        f"- z=0 normalized intensity rel-L2: `{summary['initial_z0_intensity_rel_l2']:.6g}` -> `{summary['final_z0_intensity_rel_l2']:.6g}`",
        f"- phase correlation against hidden phase: `{summary['phase_correlation']:.6g}`",
        f"- phase RMSE after global offset alignment: `{summary['phase_rmse_rad']:.6g} rad`",
        f"- elapsed: `{summary['elapsed_s']:.4f} s`, `{summary['ms_per_step']:.3f} ms/step`",
        f"- figure: `{png_path}`",
        "",
        "## Interpretation",
        "",
        "- This is a self-consistency test for the forward/adjoint optimizer, not a uniqueness proof for phase retrieval.",
        "- The primary pass/fail signal is whether the recovered phase-only mask reproduces the hidden-mask intensity in the requested fit scope and whether it generalizes to the full through-focus volume.",
        "- Phase agreement is reported separately because intensity-only phase recovery can be non-unique.",
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
