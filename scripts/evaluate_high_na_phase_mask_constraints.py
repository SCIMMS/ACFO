from __future__ import annotations

import argparse
import csv
import json
import os
import sys
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

from benchmark_high_na_debye_wolf import gauss_theta_grid  # noqa: E402
from benchmark_high_na_torch_gpu import (  # noqa: E402
    TorchSeparableHarmonicDebyeWolfPlan,
    device_name,
    import_torch,
    package_version,
    resolve_device,
    to_numpy,
)
from benchmark_high_na_vectorial_backpropagation import (  # noqa: E402
    richards_wolf_jones_matrix,
    vectorial_h_cutoff_for_workload,
    vectorial_pupil_jones,
    workloads,
)
from demo_high_na_phase_mask_self_consistency import (  # noqa: E402
    make_fit_weight,
    normalized_metrics_np,
    phase_metrics,
)
from demo_high_na_physical_intensity_shaping import (  # noqa: E402
    COLOR,
    TOKENS,
    apply_phase,
    intensity,
    use_chart_theme,
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


def parse_levels(value: str) -> list[int]:
    levels: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        levels.append(int(item))
    return levels


def parse_float_list(value: str) -> list[float]:
    values: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def wrap_phase_np(phase: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * phase)).astype(np.float32, copy=False)


def quantize_phase(phase: np.ndarray, levels: int) -> np.ndarray:
    if levels <= 0:
        return wrap_phase_np(phase)
    step = 2.0 * np.pi / float(levels)
    wrapped = np.mod(phase + np.pi, 2.0 * np.pi)
    quantized = np.mod(np.round(wrapped / step), levels) * step - np.pi
    return wrap_phase_np(quantized)


def add_phase_noise(phase: np.ndarray, rms: float, rng: np.random.Generator) -> np.ndarray:
    if rms <= 0.0:
        return wrap_phase_np(phase)
    noise = rng.normal(loc=0.0, scale=float(rms), size=phase.shape)
    return wrap_phase_np(phase + noise)


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.array([1.0], dtype=float)
    radius = max(1, int(np.ceil(3.0 * float(sigma))))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / float(sigma)) ** 2)
    kernel /= max(float(np.sum(kernel)), np.finfo(float).eps)
    return kernel


def convolve_theta_reflect(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    radius = kernel.size // 2
    if radius == 0:
        return values
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="reflect")
    out = np.zeros_like(values, dtype=np.complex128)
    for i, weight in enumerate(kernel):
        out += weight * padded[i : i + values.shape[0], :]
    return out


def convolve_phi_periodic(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    radius = kernel.size // 2
    if radius == 0:
        return values
    out = np.zeros_like(values, dtype=np.complex128)
    for i, weight in enumerate(kernel):
        out += weight * np.roll(values, shift=i - radius, axis=1)
    return out


def blur_phase(phase: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return wrap_phase_np(phase)
    phasor = np.exp(1j * phase)
    kernel = gaussian_kernel1d(float(sigma))
    blurred = convolve_theta_reflect(phasor, kernel)
    blurred = convolve_phi_periodic(blurred, kernel)
    return np.angle(blurred).astype(np.float32, copy=False)


def evaluate_phase_mask(
    *,
    torch: Any,
    plan: Any,
    base_t: Any,
    mixing_t: Any,
    phase_np: np.ndarray,
    target_np: np.ndarray,
    fit_weight_np: np.ndarray,
    z0_weight_np: np.ndarray,
    hidden_phase_np: np.ndarray,
    recovered_phase_np: np.ndarray,
    pupil_weight_np: np.ndarray,
    device: Any,
    complex_dtype: Any,
) -> tuple[np.ndarray, dict[str, float]]:
    phase_t = torch.as_tensor(np.ascontiguousarray(phase_np[None, ...]), dtype=plan.real_dtype, device=device)
    with torch.no_grad():
        pupil = apply_phase(base_t, phase_t, torch=torch, complex_dtype=complex_dtype)
        field = plan.evaluate_vectorial_batch(pupil, mixing_t)
        intensity_np = to_numpy(torch, device, intensity(field, torch)[0])
    flat_i = intensity_np.reshape(-1)
    flat_t = target_np.reshape(-1)
    full_weight = np.ones_like(flat_t, dtype=np.float32)
    full_metrics = normalized_metrics_np(flat_i, flat_t, full_weight)
    fit_metrics = normalized_metrics_np(flat_i, flat_t, fit_weight_np.reshape(-1))
    z0_metrics = normalized_metrics_np(flat_i, flat_t, z0_weight_np.reshape(-1))
    phase_vs_hidden = phase_metrics(recovered=phase_np, hidden=hidden_phase_np, weight=pupil_weight_np)
    phase_vs_recovered = phase_metrics(recovered=phase_np, hidden=recovered_phase_np, weight=pupil_weight_np)
    metrics = {
        "full_rel_l2": full_metrics["rel_l2"],
        "full_cosine": full_metrics["cosine"],
        "fit_rel_l2": fit_metrics["rel_l2"],
        "fit_cosine": fit_metrics["cosine"],
        "z0_rel_l2": z0_metrics["rel_l2"],
        "z0_cosine": z0_metrics["cosine"],
        "phase_corr_vs_hidden": phase_vs_hidden["phase_correlation"],
        "phase_rmse_vs_hidden_rad": phase_vs_hidden["phase_rmse_rad"],
        "phase_corr_vs_recovered": phase_vs_recovered["phase_correlation"],
        "phase_rmse_vs_recovered_rad": phase_vs_recovered["phase_rmse_rad"],
        "total_intensity_ratio": float(np.sum(flat_i) / max(float(np.sum(flat_t)), np.finfo(float).eps)),
    }
    return intensity_np.reshape(target_np.shape), metrics


def load_config(summary_json: Path) -> dict[str, Any]:
    if not summary_json.exists():
        return {}
    return json.loads(summary_json.read_text(encoding="utf-8")).get("config", {})


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    input_npz = ROOT / args.input_npz
    summary_json = ROOT / args.summary_json
    data = np.load(input_npz)
    config = load_config(summary_json)

    workload_set = args.workload_set or config.get("workload_set", "representative")
    workload_index = int(args.workload_index if args.workload_index is not None else config.get("workload_index", 1))
    dtype = args.dtype or config.get("dtype", "complex64")
    h_margin = int(args.h_margin if args.h_margin is not None else config.get("h_margin", 6))
    apodization = args.apodization or config.get("apodization", "sqrt-cos")
    fit_scope = args.fit_scope or config.get("fit_scope", "volume")

    workload = workloads(workload_set)[workload_index]
    theta = np.asarray(data["theta"], dtype=float)
    phi = np.asarray(data["phi"], dtype=float)
    rho_axis = np.asarray(data["rho_axis"], dtype=float)
    psi_axis = np.asarray(data["psi_axis"], dtype=float)
    z_axis = np.asarray(data["z_axis"], dtype=float)
    recovered_phase_np = np.asarray(data["recovered_phase"], dtype=np.float32)
    hidden_phase_np = np.asarray(data["hidden_phase"], dtype=np.float32)
    target_np = np.asarray(data["target_intensity"], dtype=np.float32)
    fit_weight_np = np.asarray(data["fit_weight"], dtype=np.float32) if "fit_weight" in data.files else make_fit_weight(
        nrho=target_np.shape[0],
        npsi=target_np.shape[1],
        nz=target_np.shape[2],
        z_axis=z_axis,
        scope=fit_scope,
    ).reshape(target_np.shape)
    z0_weight_np = make_fit_weight(
        nrho=target_np.shape[0],
        npsi=target_np.shape[1],
        nz=target_np.shape[2],
        z_axis=z_axis,
        scope="z0",
    ).reshape(target_np.shape)

    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    theta_weights = gauss_theta_grid(workload.ntheta, workload.theta_max)[1]
    mixing = richards_wolf_jones_matrix(theta, phi, apodization=apodization)
    base_np = vectorial_pupil_jones(
        workload.pupil_case,
        theta,
        phi,
        theta_max=workload.theta_max,
        strength=workload.aberration_strength,
        vortex_charge=workload.vortex_charge,
    )
    h_cutoff = vectorial_h_cutoff_for_workload(workload, h_margin)
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
        dtype=dtype,
    )
    base_t = torch.as_tensor(np.ascontiguousarray(base_np[None, ...]), dtype=plan.complex_dtype, device=device)
    mixing_t = torch.as_tensor(np.ascontiguousarray(mixing), dtype=plan.complex_dtype, device=device)
    if base_np.ndim == 3:
        pupil_weight_np = np.sum(np.abs(base_np) ** 2, axis=0)
    elif base_np.ndim == 4:
        pupil_weight_np = np.sum(np.abs(base_np) ** 2, axis=(0, 1))
    else:
        raise ValueError(f"unexpected pupil shape: {base_np.shape}")

    rows: list[dict[str, Any]] = []
    intensities: dict[str, np.ndarray] = {}
    phases: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(int(args.noise_seed))
    cases: list[tuple[str, str, float, int, np.ndarray]] = [
        ("continuous", "continuous", 0.0, 0, wrap_phase_np(recovered_phase_np))
    ]
    for level in parse_levels(args.quantization_levels):
        cases.append((f"{level}-level", "quantized", float(level), int(level), quantize_phase(recovered_phase_np, int(level))))
    for noise_rms in parse_float_list(args.noise_rms_values):
        cases.append((f"noise-{noise_rms:g}rad", "noise", float(noise_rms), 0, add_phase_noise(recovered_phase_np, noise_rms, rng)))
    for blur_sigma in parse_float_list(args.blur_sigma_values):
        cases.append((f"blur-{blur_sigma:g}px", "blur", float(blur_sigma), 0, blur_phase(recovered_phase_np, blur_sigma)))

    for label, family, perturbation, levels, phase_np in cases:
        intensity_np, metrics = evaluate_phase_mask(
            torch=torch,
            plan=plan,
            base_t=base_t,
            mixing_t=mixing_t,
            phase_np=phase_np,
            target_np=target_np,
            fit_weight_np=fit_weight_np,
            z0_weight_np=z0_weight_np,
            hidden_phase_np=hidden_phase_np,
            recovered_phase_np=recovered_phase_np,
            pupil_weight_np=pupil_weight_np,
            device=device,
            complex_dtype=plan.complex_dtype,
        )
        row = {
            "mask": label,
            "family": family,
            "perturbation": perturbation,
            "levels": levels,
            **metrics,
        }
        rows.append(row)
        intensities[label] = intensity_np
        phases[label] = phase_np

    summary = {
        "status": "ok",
        "demo": "phase_mask_constraint_evaluation",
        "input_npz": str(input_npz),
        "workload": workload.name,
        "pupil_case": workload.pupil_case,
        "fit_scope": fit_scope,
        "device": str(device),
        "device_name": device_name(torch, device),
        "torch_version": package_version("torch"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "dtype": dtype,
        "h_cutoff": int(h_cutoff),
        "used_modes": int(plan.used_modes),
        "targets": int(np.prod(target_np.shape)),
        "continuous_full_rel_l2": rows[0]["full_rel_l2"],
        "continuous_z0_rel_l2": rows[0]["z0_rel_l2"],
    }
    return {
        "summary": summary,
        "rows": rows,
        "rho_axis": rho_axis,
        "psi_axis": psi_axis,
        "z_axis": z_axis,
        "target_intensity": target_np,
        "intensities": intensities,
        "phases": phases,
        "hidden_phase": hidden_phase_np,
        "recovered_phase": recovered_phase_np,
    }


def plot_results(result: dict[str, Any], output_prefix: Path) -> tuple[Path, Path]:
    use_chart_theme()
    rows = result["rows"]
    rho_axis = result["rho_axis"]
    psi_axis = result["psi_axis"]
    z_axis = result["z_axis"]
    target = result["target_intensity"]
    z_index = int(np.argmin(np.abs(z_axis)))
    target_z = target[:, :, z_index]
    cont_z = result["intensities"]["continuous"][:, :, z_index]
    showcase_label = "noise-0.1rad" if "noise-0.1rad" in result["intensities"] else (
        "blur-1px" if "blur-1px" in result["intensities"] else (
            "8-level" if "8-level" in result["intensities"] else rows[min(len(rows) - 1, 1)]["mask"]
        )
    )
    showcase_z = result["intensities"][showcase_label][:, :, z_index]
    phase_label = "blur-1px" if "blur-1px" in result["phases"] else (
        "4-level" if "4-level" in result["phases"] else rows[-1]["mask"]
    )
    phase_show = result["phases"][phase_label]

    target_plot = target_z / max(float(np.max(target_z)), np.finfo(float).eps)
    cont_plot = cont_z / max(float(np.max(cont_z)), np.finfo(float).eps)
    showcase_plot = showcase_z / max(float(np.max(showcase_z)), np.finfo(float).eps)

    fig = plt.figure(figsize=(15.2, 9.7), dpi=180)
    grid = fig.add_gridspec(2, 3, left=0.060, right=0.965, top=0.840, bottom=0.085, wspace=0.34, hspace=0.42)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    ax_curve = fig.add_subplot(grid[1, :2])
    ax_phase = fig.add_subplot(grid[1, 2])
    fig.text(
        0.060,
        0.972,
        "Phase-mask constraint sweep for High-NA self-consistency",
        ha="left",
        va="top",
        fontsize=15.2,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.060,
        0.922,
        "Recovered continuous phase masks are quantized, perturbed with phase noise, blurred, and re-evaluated with the same vectorial Debye-Wolf forward model.",
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
    )
    extent = [0.0, 2.0 * np.pi, float(rho_axis[0]), float(rho_axis[-1])]
    cmap_int = sns.blend_palette([TOKENS["panel"], "#CEDFFE", "#5477C4", "#2E4780"], as_cmap=True)
    panels = [
        ("A. Target intensity", target_plot, "Intensity / panel max"),
        ("B. Continuous mask", cont_plot, "Intensity / panel max"),
        (f"C. {showcase_label} mask", showcase_plot, "Intensity / panel max"),
    ]
    for ax, (title, data, label) in zip(axes, panels):
        im = ax.imshow(data, aspect="auto", origin="lower", extent=extent, cmap=cmap_int, vmin=0.0)
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

    x = np.arange(len(rows))
    labels = [str(row["mask"]) for row in rows]
    families = {
        "continuous": {"marker": "o", "color": COLOR["blue"], "label": "continuous"},
        "quantized": {"marker": "s", "color": COLOR["orange"], "label": "quantized"},
        "noise": {"marker": "^", "color": COLOR["olive"], "label": "phase noise"},
        "blur": {"marker": "D", "color": COLOR["gold"], "label": "phase blur"},
    }
    for family, style in families.items():
        xs = [i for i, row in enumerate(rows) if row["family"] == family]
        if not xs:
            continue
        ys = [float(rows[i]["full_rel_l2"]) for i in xs]
        ax_curve.plot(xs, ys, marker=style["marker"], linewidth=1.45, color=style["color"], label=style["label"])
    ax_curve.set_title("D. Mask constraints degrade intensity according to perturbation strength", loc="left", fontsize=11, fontweight="semibold")
    ax_curve.set_xlabel("Mask or perturbation")
    ax_curve.set_ylabel("Normalized intensity rel-L2")
    ax_curve.set_xticks(x)
    ax_curve.set_xticklabels(labels, rotation=35, ha="right")
    ax_curve.legend(frameon=False, loc="upper left")
    ax_curve.grid(axis="both", color=TOKENS["grid"])
    ax2 = ax_curve.twinx()
    ax2.scatter(x, [float(row["phase_rmse_vs_recovered_rad"]) for row in rows], marker="x", s=20, color=TOKENS["muted"], label="Phase RMSE")
    ax2.set_ylabel("Phase RMSE vs continuous (rad)")
    ax2.grid(False)
    ax2.tick_params(colors=TOKENS["muted"])

    im_phase = ax_phase.imshow(
        phase_show,
        aspect="auto",
        origin="lower",
        extent=[0.0, 2.0 * np.pi, 0.0, float(phase_show.shape[0] - 1)],
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    ax_phase.set_title(f"E. {phase_label} phase mask", loc="left", fontsize=11, fontweight="semibold")
    ax_phase.set_xlabel("Azimuth phi (rad)")
    ax_phase.set_ylabel("Theta sample")
    cbar = fig.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.018)
    cbar.set_label("Phase (rad)", fontsize=8.5, color=TOKENS["muted"])
    cbar.ax.tick_params(labelsize=7.5, colors=TOKENS["muted"])
    fig.text(
        0.060,
        0.025,
        f"Local snapshot: {result['summary']['device_name']}, torch {result['summary']['torch_version']}; fit scope {result['summary']['fit_scope']}.",
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
    parser = argparse.ArgumentParser(description="Evaluate phase-mask quantization constraints for a High-NA self-consistency result.")
    parser.add_argument("--input-npz", default="benchmark_results/high_na_phase_mask_self_consistency.npz")
    parser.add_argument("--summary-json", default="benchmark_results/high_na_phase_mask_self_consistency_summary.json")
    parser.add_argument("--output-prefix", default="benchmark_results/high_na_phase_mask_constraints")
    parser.add_argument("--quantization-levels", default="64,32,16,8,4,2")
    parser.add_argument("--noise-rms-values", default="0.02,0.05,0.1,0.2")
    parser.add_argument("--blur-sigma-values", default="0.25,0.5,1.0,1.5")
    parser.add_argument("--noise-seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default=None)
    parser.add_argument("--workload-set", choices=["quick", "representative", "large"], default=None)
    parser.add_argument("--workload-index", type=int, default=None)
    parser.add_argument("--fit-scope", choices=["volume", "z0"], default=None)
    parser.add_argument("--h-margin", type=int, default=None)
    parser.add_argument("--apodization", choices=["sqrt-cos", "none"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_prefix = ROOT / args.output_prefix
    result = run_evaluation(args)
    png_path, svg_path = plot_results(result, output_prefix)
    csv_path = output_prefix.with_name(output_prefix.name + "_metrics.csv")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")
    summary_md_path = output_prefix.with_name(output_prefix.name + "_summary.md")
    write_csv(csv_path, result["rows"])
    write_json(
        summary_path,
        {
            "config": vars(args),
            "summary": result["summary"],
            "metrics_csv": str(csv_path),
            "figure_png": str(png_path),
            "figure_svg": str(svg_path),
        },
    )
    rows = result["rows"]
    best_practical = next((row for row in rows if row["mask"] == "8-level"), rows[min(len(rows) - 1, 1)])
    noise_01 = next((row for row in rows if row["mask"] == "noise-0.1rad"), None)
    blur_1 = next((row for row in rows if row["mask"] == "blur-1px"), None)
    lines = [
        "# High-NA phase-mask constraint evaluation",
        "",
        "The recovered continuous phase-only mask from the self-consistency demo was quantized, perturbed with Gaussian phase noise, blurred as a complex phasor, and re-evaluated with the same vectorial Debye-Wolf forward model.",
        "",
        "## Results",
        "",
        "| Mask | Family | Perturbation | Full rel-L2 | z=0 rel-L2 | Full cosine | Phase RMSE vs continuous |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['mask']}` | `{row['family']}` | {row['perturbation']:.6g} | {row['full_rel_l2']:.6g} | {row['z0_rel_l2']:.6g} | {row['full_cosine']:.6g} | {row['phase_rmse_vs_recovered_rad']:.6g} rad |"
        )
    lines.extend(["", "## Takeaway", ""])
    lines.append(f"- Continuous full-volume rel-L2 is `{rows[0]['full_rel_l2']:.6g}`.")
    lines.append(
        f"- `{best_practical['mask']}` full-volume rel-L2 is `{best_practical['full_rel_l2']:.6g}` with phase RMSE `{best_practical['phase_rmse_vs_recovered_rad']:.6g} rad` versus the continuous mask."
    )
    if noise_01 is not None:
        lines.append(
            f"- `noise-0.1rad` full-volume rel-L2 is `{noise_01['full_rel_l2']:.6g}` with phase RMSE `{noise_01['phase_rmse_vs_recovered_rad']:.6g} rad`."
        )
    if blur_1 is not None:
        lines.append(
            f"- `blur-1px` full-volume rel-L2 is `{blur_1['full_rel_l2']:.6g}` with phase RMSE `{blur_1['phase_rmse_vs_recovered_rad']:.6g} rad`."
        )
    lines.extend(
        [
            "- Coarse quantization, high phase noise, and strong blur are mask/manufacturability stress tests; they do not change the underlying vectorial propagation model.",
            f"- Figure: `{png_path}`",
        ]
    )
    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": result["summary"],
                "metrics_csv": str(csv_path),
                "figure_png": str(png_path),
                "figure_svg": str(svg_path),
                "summary_md": str(summary_md_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
