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


def x_polarized_jones(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    out = np.zeros((2, theta.size, phi.size), dtype=np.complex128)
    out[0] = 1.0 + 0.0j
    return out


def aberration_phase(
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    theta_max: float,
    strength: float,
    case: str,
) -> np.ndarray:
    radial = (np.sin(theta)[:, None] / max(np.sin(theta_max), np.finfo(float).eps)).clip(0.0, 1.0)
    az = phi[None, :]
    if case == "coma_astig_spherical":
        phase = (
            0.75 * radial**3 * np.cos(az - 0.35)
            + 0.55 * radial**2 * np.cos(2.0 * az + 0.65)
            + 0.35 * (2.0 * radial**4 - radial**2)
        )
    elif case == "astigmatism":
        phase = radial**2 * np.cos(2.0 * az + 0.45)
    elif case == "coma":
        phase = (3.0 * radial**3 - 2.0 * radial) * np.cos(az - 0.25)
    else:
        raise ValueError("aberration case must be coma_astig_spherical, astigmatism, or coma")
    phase = float(strength) * phase
    return np.angle(np.exp(1j * phase)).astype(np.float32, copy=False)


def fit_weight(
    *,
    nrho: int,
    npsi: int,
    nz: int,
    rho_axis: np.ndarray,
    z_axis: np.ndarray,
    scope: str,
    rho_fraction: float,
    z_fraction: float,
) -> np.ndarray:
    if scope == "volume":
        return np.ones(nrho * npsi * nz, dtype=np.float32)
    if scope == "z0":
        weight = np.zeros((nrho, npsi, nz), dtype=np.float32)
        weight[:, :, int(np.argmin(np.abs(z_axis)))] = 1.0
        return weight.reshape(-1)
    if scope == "central_volume":
        rho = rho_axis[:, None, None]
        z = z_axis[None, None, :]
        rho_sigma = max(float(rho_fraction) * float(rho_axis[-1]), np.finfo(float).eps)
        z_sigma = max(float(z_fraction) * max(float(np.max(np.abs(z_axis))), 1.0), np.finfo(float).eps)
        weight = np.exp(-0.5 * (rho / rho_sigma) ** 2 - 0.5 * (z / z_sigma) ** 2)
        weight = np.broadcast_to(weight, (nrho, npsi, nz)).copy()
        return (weight / max(float(np.max(weight)), np.finfo(float).eps)).astype(np.float32).reshape(-1)
    raise ValueError("fit scope must be volume, z0, or central_volume")


def field_loss_residual_metrics(
    *,
    torch: Any,
    field: Any,
    target: Any,
    weight: Any,
) -> tuple[Any, Any, dict[str, float]]:
    diff = field - target
    denom = torch.clamp(torch.sum(weight[None, None, :] * torch.abs(target) ** 2).real, min=1e-30)
    weighted_power = weight[None, None, :] * torch.abs(diff) ** 2
    loss = 0.5 * torch.sum(weighted_power).real / denom
    residual = weight[None, None, :] * diff / denom
    rel_l2 = torch.sqrt(torch.sum(weighted_power).real / denom)
    dot = torch.sum(torch.conj(target) * field * weight[None, None, :])
    cosine = torch.abs(dot) / torch.clamp(
                torch.sqrt(torch.sum(torch.abs(field) ** 2 * weight[None, None, :]).real)
        * torch.sqrt(torch.sum(torch.abs(target) ** 2 * weight[None, None, :]).real),
        min=1e-30,
    )
    return (
        loss,
        residual,
        {
            "loss": float(loss.detach().cpu().item()),
            "field_rel_l2": float(rel_l2.detach().cpu().item()),
            "field_cosine": float(cosine.detach().cpu().item()),
        },
    )


def field_metrics_np(field: np.ndarray, target: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    weight3 = weight.reshape(1, -1)
    diff = (field - target).reshape(3, -1)
    target_f = target.reshape(3, -1)
    field_f = field.reshape(3, -1)
    denom = max(float(np.sum(weight3 * np.abs(target_f) ** 2)), np.finfo(float).eps)
    rel_l2 = float(np.sqrt(np.sum(weight3 * np.abs(diff) ** 2) / denom))
    dot = np.sum(np.conj(target_f) * field_f * weight3)
    cosine = float(
        abs(dot)
        / max(
            float(
                np.sqrt(np.sum(weight3 * np.abs(field_f) ** 2))
                * np.sqrt(np.sum(weight3 * np.abs(target_f) ** 2))
            ),
            np.finfo(float).eps,
        )
    )
    return {"field_rel_l2": rel_l2, "field_cosine": cosine}


def intensity_metrics_np(
    intensity_np: np.ndarray,
    ideal_np: np.ndarray,
    *,
    rho_axis: np.ndarray,
    z_axis: np.ndarray,
) -> dict[str, float]:
    peak_ideal = max(float(np.max(ideal_np)), np.finfo(float).eps)
    z_index = int(np.argmin(np.abs(z_axis)))
    central_radius = 0.25 * float(rho_axis[-1])
    center_mask = (rho_axis[:, None, None] <= central_radius).astype(float)
    center_mask = np.broadcast_to(center_mask, intensity_np.shape)
    total = max(float(np.sum(intensity_np)), np.finfo(float).eps)
    ideal_total = max(float(np.sum(ideal_np)), np.finfo(float).eps)
    selected = intensity_np[:, :, z_index]
    ideal_selected = ideal_np[:, :, z_index]
    selected_norm = selected / max(float(np.sum(selected)), np.finfo(float).eps)
    ideal_norm = ideal_selected / max(float(np.sum(ideal_selected)), np.finfo(float).eps)
    return {
        "peak_over_ideal": float(np.max(intensity_np) / peak_ideal),
        "z0_intensity_rel_l2": float(
            np.linalg.norm(selected_norm - ideal_norm)
            / max(float(np.linalg.norm(ideal_norm)), np.finfo(float).eps)
        ),
        "center_energy_fraction": float(np.sum(center_mask * intensity_np) / total),
        "center_energy_fraction_over_ideal": float(
            (np.sum(center_mask * intensity_np) / total)
            / max(float(np.sum(center_mask * ideal_np) / ideal_total), np.finfo(float).eps)
        ),
    }


def phase_alignment_metrics(recovered: np.ndarray, expected: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    w = np.asarray(weight, dtype=float)
    w = w / max(float(np.sum(w)), np.finfo(float).eps)
    diff = np.angle(np.exp(1j * (recovered - expected)))
    offset = np.angle(np.sum(w * np.exp(1j * diff)))
    aligned = np.angle(np.exp(1j * (diff - offset)))
    return {
        "correction_phase_correlation": float(abs(np.sum(w * np.exp(1j * aligned)))),
        "correction_phase_rmse_rad": float(np.sqrt(np.sum(w * aligned * aligned))),
        "correction_phase_global_offset_rad": float(offset),
    }


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
    mixing = richards_wolf_jones_matrix(theta, phi, apodization=args.apodization)
    ideal_np = x_polarized_jones(theta, phi)
    aberration_np = aberration_phase(
        theta,
        phi,
        theta_max=args.theta_max,
        strength=args.aberration_strength,
        case=args.aberration_case,
    )
    weight_np = fit_weight(
        nrho=args.nrho,
        npsi=args.npsi,
        nz=args.nz,
        rho_axis=rho_axis,
        z_axis=z_axis,
        scope=args.fit_scope,
        rho_fraction=args.fit_rho_fraction,
        z_fraction=args.fit_z_fraction,
    )

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
    ideal_t = torch.as_tensor(np.ascontiguousarray(ideal_np[None, ...]), dtype=complex_dtype, device=device)
    mixing_t = torch.as_tensor(np.ascontiguousarray(mixing), dtype=complex_dtype, device=device)
    aberration_t = torch.as_tensor(np.ascontiguousarray(aberration_np[None, ...]), dtype=real_dtype, device=device)
    weight_t = torch.as_tensor(np.ascontiguousarray(weight_np), dtype=real_dtype, device=device)

    with torch.no_grad():
        aberrated_t = apply_phase(ideal_t, aberration_t, torch=torch, complex_dtype=complex_dtype)
        target_field_t = plan.evaluate_vectorial_batch(ideal_t, mixing_t).detach()
        initial_field_t = plan.evaluate_vectorial_batch(aberrated_t, mixing_t).detach()
        _, _, initial_metrics_t = field_loss_residual_metrics(
            torch=torch,
            field=initial_field_t,
            target=target_field_t,
            weight=weight_t,
        )
        target_field_np = to_numpy(torch, device, target_field_t[0]).reshape(3, args.nrho, args.npsi, args.nz)
        initial_field_np = to_numpy(torch, device, initial_field_t[0]).reshape(3, args.nrho, args.npsi, args.nz)

    phase_angle = torch.zeros((1, args.ntheta, args.nphi), dtype=real_dtype, device=device)
    adam_m = torch.zeros_like(phase_angle)
    adam_v = torch.zeros_like(phase_angle)
    history: list[dict[str, Any]] = [
        {
            "step": 0,
            "accepted": True,
            "step_radians": 0.0,
            "gradient_rms": 0.0,
            **initial_metrics_t,
        }
    ]
    start = time.perf_counter()

    def forward_loss_grad() -> tuple[Any, Any, dict[str, float], Any]:
        corrected_pupil = apply_phase(aberrated_t, phase_angle, torch=torch, complex_dtype=complex_dtype)
        field = plan.evaluate_vectorial_batch(corrected_pupil, mixing_t)
        _, residual, metrics = field_loss_residual_metrics(
            torch=torch,
            field=field,
            target=target_field_t,
            weight=weight_t,
        )
        pupil_grad = plan.adjoint_vectorial_batch(residual, mixing_t)
        grad = torch.sum(torch.imag(torch.conj(corrected_pupil) * pupil_grad), dim=1)
        return field, grad, metrics, corrected_pupil

    with torch.no_grad():
        for step in range(1, args.steps + 1):
            _, grad, metrics, _ = forward_loss_grad()
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
                trial_pupil = apply_phase(aberrated_t, trial_phase, torch=torch, complex_dtype=complex_dtype)
                trial_field = plan.evaluate_vectorial_batch(trial_pupil, mixing_t)
                _, _, trial_metrics = field_loss_residual_metrics(
                    torch=torch,
                    field=trial_field,
                    target=target_field_t,
                    weight=weight_t,
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
                if max(item["field_rel_l2"] for item in recent) - min(item["field_rel_l2"] for item in recent) < args.stop_delta:
                    break

        final_field_t, _, final_metrics_t, _ = forward_loss_grad()
        synchronize(torch, device)
        elapsed_s = time.perf_counter() - start
        final_field_np = to_numpy(torch, device, final_field_t[0]).reshape(3, args.nrho, args.npsi, args.nz)
        recovered_phase_np = to_numpy(torch, device, phase_angle[0])

    ideal_intensity = np.sum(np.abs(target_field_np) ** 2, axis=0)
    initial_intensity = np.sum(np.abs(initial_field_np) ** 2, axis=0)
    final_intensity = np.sum(np.abs(final_field_np) ** 2, axis=0)
    full_weight_np = np.ones_like(weight_np, dtype=np.float32)
    z0_weight_np = fit_weight(
        nrho=args.nrho,
        npsi=args.npsi,
        nz=args.nz,
        rho_axis=rho_axis,
        z_axis=z_axis,
        scope="z0",
        rho_fraction=args.fit_rho_fraction,
        z_fraction=args.fit_z_fraction,
    )
    pupil_weight = np.sum(np.abs(ideal_np) ** 2, axis=0)
    expected_correction = np.angle(np.exp(-1j * aberration_np)).astype(np.float32)
    phase_stats = phase_alignment_metrics(recovered_phase_np, expected_correction, pupil_weight)

    initial_full = field_metrics_np(initial_field_np, target_field_np, full_weight_np)
    final_full = field_metrics_np(final_field_np, target_field_np, full_weight_np)
    initial_z0 = field_metrics_np(initial_field_np, target_field_np, z0_weight_np)
    final_z0 = field_metrics_np(final_field_np, target_field_np, z0_weight_np)
    initial_intensity_metrics = intensity_metrics_np(initial_intensity, ideal_intensity, rho_axis=rho_axis, z_axis=z_axis)
    final_intensity_metrics = intensity_metrics_np(final_intensity, ideal_intensity, rho_axis=rho_axis, z_axis=z_axis)

    summary = {
        "status": "ok",
        "demo": "aberrated_high_na_focus_correction",
        "device": str(device),
        "device_name": device_name(torch, device),
        "torch_version": package_version("torch"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "dtype": args.dtype,
        "aberration_case": args.aberration_case,
        "aberration_strength": float(args.aberration_strength),
        "fit_scope": args.fit_scope,
        "fit_samples": int(np.count_nonzero(weight_np > 1e-7)),
        "steps_requested": int(args.steps),
        "steps_completed": int(history[-1]["step"]),
        "elapsed_s": float(elapsed_s),
        "ms_per_step": float(1000.0 * elapsed_s / max(1, int(history[-1]["step"]))),
        "h_cutoff": int(h_cutoff),
        "used_modes": int(plan.used_modes),
        "targets": int(args.nrho * args.npsi * args.nz),
        "initial_fit_field_rel_l2": float(history[0]["field_rel_l2"]),
        "final_fit_field_rel_l2": float(final_metrics_t["field_rel_l2"]),
        "fit_field_rel_l2_reduction": float(history[0]["field_rel_l2"] / max(float(final_metrics_t["field_rel_l2"]), 1e-30)),
        "initial_fit_field_cosine": float(history[0]["field_cosine"]),
        "final_fit_field_cosine": float(final_metrics_t["field_cosine"]),
        "initial_volume_field_rel_l2": initial_full["field_rel_l2"],
        "final_volume_field_rel_l2": final_full["field_rel_l2"],
        "initial_volume_field_cosine": initial_full["field_cosine"],
        "final_volume_field_cosine": final_full["field_cosine"],
        "initial_z0_field_rel_l2": initial_z0["field_rel_l2"],
        "final_z0_field_rel_l2": final_z0["field_rel_l2"],
        "initial_z0_field_cosine": initial_z0["field_cosine"],
        "final_z0_field_cosine": final_z0["field_cosine"],
        "initial_peak_over_ideal": initial_intensity_metrics["peak_over_ideal"],
        "final_peak_over_ideal": final_intensity_metrics["peak_over_ideal"],
        "initial_z0_intensity_rel_l2": initial_intensity_metrics["z0_intensity_rel_l2"],
        "final_z0_intensity_rel_l2": final_intensity_metrics["z0_intensity_rel_l2"],
        "initial_center_energy_fraction": initial_intensity_metrics["center_energy_fraction"],
        "final_center_energy_fraction": final_intensity_metrics["center_energy_fraction"],
        **phase_stats,
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
        fit_weight=weight_np.reshape(args.nrho, args.npsi, args.nz),
        aberration_phase=aberration_np,
        expected_correction_phase=expected_correction,
        recovered_correction_phase=recovered_phase_np,
        ideal_intensity=ideal_intensity,
        aberrated_intensity=initial_intensity,
        corrected_intensity=final_intensity,
        ideal_field=target_field_np,
        aberrated_field=initial_field_np,
        corrected_field=final_field_np,
    )
    return {
        "summary": summary,
        "history": history,
        "arrays_npz": str(npz_path),
        "theta": theta,
        "phi": phi,
        "rho_axis": rho_axis,
        "psi_axis": psi_axis,
        "z_axis": z_axis,
        "aberration_phase": aberration_np,
        "expected_correction_phase": expected_correction,
        "recovered_correction_phase": recovered_phase_np,
        "ideal_intensity": ideal_intensity,
        "aberrated_intensity": initial_intensity,
        "corrected_intensity": final_intensity,
    }


def normalized_panel(data: np.ndarray) -> np.ndarray:
    return data / max(float(np.max(data)), np.finfo(float).eps)


def plot_demo(result: dict[str, Any], output_prefix: Path) -> tuple[Path, Path]:
    use_chart_theme()
    rho_axis = result["rho_axis"]
    psi_axis = result["psi_axis"]
    z_axis = result["z_axis"]
    z_index = int(np.argmin(np.abs(z_axis)))
    ideal = normalized_panel(result["ideal_intensity"][:, :, z_index])
    aberrated = normalized_panel(result["aberrated_intensity"][:, :, z_index])
    corrected = normalized_panel(result["corrected_intensity"][:, :, z_index])
    aberration = result["aberration_phase"]
    expected = result["expected_correction_phase"]
    recovered = result["recovered_correction_phase"]
    phase_error = np.angle(np.exp(1j * (recovered - expected)))
    history = result["history"]
    summary = result["summary"]

    fig = plt.figure(figsize=(15.8, 10.2), dpi=180)
    grid = fig.add_gridspec(2, 4, left=0.055, right=0.975, top=0.845, bottom=0.085, wspace=0.33, hspace=0.42)
    ax_aberr = fig.add_subplot(grid[0, 0])
    ax_ideal = fig.add_subplot(grid[0, 1])
    ax_initial = fig.add_subplot(grid[0, 2])
    ax_final = fig.add_subplot(grid[0, 3])
    ax_expected = fig.add_subplot(grid[1, 0])
    ax_recovered = fig.add_subplot(grid[1, 1])
    ax_history = fig.add_subplot(grid[1, 2])
    ax_error = fig.add_subplot(grid[1, 3])

    fig.text(
        0.055,
        0.975,
        "Phase-only correction restores an aberrated vectorial High-NA focus",
        ha="left",
        va="top",
        fontsize=16,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.055,
        0.928,
        f"Known pupil aberration is applied to an x-polarized aperture; the optimizer learns a shared correction phase by matching the ideal vector field over {summary['fit_scope']} samples.",
        ha="left",
        va="top",
        fontsize=9.4,
        color=TOKENS["muted"],
    )

    phase_extent = [0.0, 2.0 * np.pi, 0.0, float(aberration.shape[0] - 1)]
    for ax, data, title in [
        (ax_aberr, aberration, "A. Imposed aberration"),
        (ax_expected, expected, "E. Ideal correction (-aberration)"),
        (ax_recovered, recovered, "F. Recovered correction"),
        (ax_error, phase_error, "H. Wrapped correction error"),
    ]:
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=phase_extent,
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
        )
        ax.set_title(title, loc="left", fontsize=10.4, fontweight="semibold")
        ax.set_xlabel("Pupil azimuth phi (rad)")
        ax.set_ylabel("Theta sample")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.018)
        cbar.set_label("Phase (rad)", fontsize=8.0, color=TOKENS["muted"])
        cbar.ax.tick_params(labelsize=7.4, colors=TOKENS["muted"])

    intensity_extent = [0.0, 2.0 * np.pi, float(rho_axis[0]), float(rho_axis[-1])]
    cmap_int = sns.blend_palette([TOKENS["panel"], "#CEDFFE", "#5477C4", "#2E4780"], as_cmap=True)
    for ax, data, title in [
        (ax_ideal, ideal, "B. Ideal focus"),
        (ax_initial, aberrated, "C. Aberrated focus"),
        (ax_final, corrected, "D. Corrected focus"),
    ]:
        im = ax.imshow(data, aspect="auto", origin="lower", extent=intensity_extent, cmap=cmap_int, vmin=0.0)
        ax.set_title(title, loc="left", fontsize=10.4, fontweight="semibold")
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
        cbar.set_label("Intensity / panel max", fontsize=8.0, color=TOKENS["muted"])
        cbar.ax.tick_params(labelsize=7.4, colors=TOKENS["muted"])

    steps = [row["step"] for row in history]
    field_l2 = [row["field_rel_l2"] for row in history]
    cosine = [row["field_cosine"] for row in history]
    loss = [row["loss"] for row in history]
    ax_history.plot(steps, field_l2, color=COLOR["blue"], linewidth=1.5, label="Field rel-L2")
    ax_history.plot(steps, cosine, color=COLOR["orange"], linewidth=1.4, label="Field cosine")
    ax_history.set_title("G. Correction objective", loc="left", fontsize=10.4, fontweight="semibold")
    ax_history.set_xlabel("Optimization step")
    ax_history.set_ylabel("Metric")
    ax_history.grid(axis="both", color=TOKENS["grid"])
    ax_history.legend(frameon=False, loc="upper right")
    ax_loss = ax_history.twinx()
    ax_loss.plot(steps, loss, color=COLOR["olive"], linestyle=":", linewidth=1.2, label="Loss")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(False)
    ax_history.text(
        0.04,
        0.08,
        f"field L2 {summary['initial_fit_field_rel_l2']:.3f} -> {summary['final_fit_field_rel_l2']:.3f}\nphase RMSE {summary['correction_phase_rmse_rad']:.3f} rad",
        transform=ax_history.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        color=TOKENS["ink"],
        bbox={"facecolor": TOKENS["panel"], "edgecolor": TOKENS["axis"], "boxstyle": "round,pad=0.25"},
    )

    fig.text(
        0.055,
        0.026,
        f"Local snapshot: {summary['device_name']}, torch {summary['torch_version']}, {summary['steps_completed']} Adam steps, {summary['ms_per_step']:.2f} ms/step. The phase comparison is piston-aligned.",
        ha="left",
        va="bottom",
        fontsize=8,
        color=TOKENS["muted"],
    )
    for ax in [ax_aberr, ax_ideal, ax_initial, ax_final, ax_expected, ax_recovered, ax_history, ax_error]:
        ax.tick_params(axis="both", labelsize=8.2, colors=TOKENS["muted"])
        sns.despine(ax=ax)

    png_path = output_prefix.with_name(output_prefix.name + "_figure.png")
    svg_path = output_prefix.with_name(output_prefix.name + "_figure.svg")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover a phase-only correction for an aberrated vectorial High-NA focus.")
    parser.add_argument("--output-prefix", default="benchmark_results/high_na_aberration_correction")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    parser.add_argument("--ntheta", type=int, default=28)
    parser.add_argument("--nphi", type=int, default=128)
    parser.add_argument("--nrho", type=int, default=17)
    parser.add_argument("--npsi", type=int, default=64)
    parser.add_argument("--nz", type=int, default=5)
    parser.add_argument("--rho-max", type=float, default=2.0)
    parser.add_argument("--z-max", type=float, default=1.0)
    parser.add_argument("--theta-max", type=float, default=1.0)
    parser.add_argument("--k", type=float, default=10.0)
    parser.add_argument("--h-margin", type=int, default=10)
    parser.add_argument("--apodization", choices=["sqrt-cos", "none"], default="sqrt-cos")
    parser.add_argument("--aberration-case", choices=["coma_astig_spherical", "astigmatism", "coma"], default="coma_astig_spherical")
    parser.add_argument("--aberration-strength", type=float, default=1.15)
    parser.add_argument("--fit-scope", choices=["volume", "z0", "central_volume"], default="volume")
    parser.add_argument("--fit-rho-fraction", type=float, default=0.55)
    parser.add_argument("--fit-z-fraction", type=float, default=0.75)
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--phase-step-radians", type=float, default=0.08)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.99)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--stop-patience", type=int, default=0)
    parser.add_argument("--stop-delta", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        "# High-NA aberration-correction demo",
        "",
        "A known pupil aberration is applied to an x-polarized vectorial high-NA aperture. A shared phase-only correction mask is optimized to recover the ideal vector field on the same cylindrical focal grid.",
        "",
        "## Results",
        "",
        f"- aberration: `{summary['aberration_case']}`, strength `{summary['aberration_strength']}` rad scale",
        f"- fit scope: `{summary['fit_scope']}` (`{summary['fit_samples']}` samples)",
        f"- steps completed: `{summary['steps_completed']}`",
        f"- fit-scope field rel-L2: `{summary['initial_fit_field_rel_l2']:.6g}` -> `{summary['final_fit_field_rel_l2']:.6g}` (`{summary['fit_field_rel_l2_reduction']:.2f}x` reduction)",
        f"- fit-scope field cosine: `{summary['initial_fit_field_cosine']:.6g}` -> `{summary['final_fit_field_cosine']:.6g}`",
        f"- full-volume field rel-L2: `{summary['initial_volume_field_rel_l2']:.6g}` -> `{summary['final_volume_field_rel_l2']:.6g}`",
        f"- z=0 intensity rel-L2: `{summary['initial_z0_intensity_rel_l2']:.6g}` -> `{summary['final_z0_intensity_rel_l2']:.6g}`",
        f"- peak intensity / ideal peak: `{summary['initial_peak_over_ideal']:.6g}` -> `{summary['final_peak_over_ideal']:.6g}`",
        f"- correction phase correlation vs `-aberration`: `{summary['correction_phase_correlation']:.6g}`",
        f"- correction phase RMSE vs `-aberration`: `{summary['correction_phase_rmse_rad']:.6g} rad`",
        f"- elapsed: `{summary['elapsed_s']:.4f} s`, `{summary['ms_per_step']:.3f} ms/step`",
        f"- figure: `{png_path}`",
        "",
        "## Interpretation",
        "",
        "- This is a controlled wavefront-correction test: the expected phase-only solution is the negative of the imposed aberration, up to piston.",
        "- The objective matches the complex vector field, so it validates the forward/adjoint correction machinery before moving to intensity-only or experimental-style objectives.",
        "- A future intensity-only version will be a harder phase-retrieval-style demo and should be interpreted separately.",
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
