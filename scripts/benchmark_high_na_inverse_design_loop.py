from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import seaborn as sns

SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_high_na_backpropagation import focal_grid  # noqa: E402
from benchmark_high_na_cylindrical_backprop_design import (  # noqa: E402
    field_match_loss_and_residual,
    phase_gradient,
    phase_step_pupil,
    roi_weight,
)
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
    synchronize,
    to_numpy,
)
from benchmark_high_na_vectorial_backpropagation import (  # noqa: E402
    richards_wolf_jones_matrix,
    vectorial_h_cutoff_for_workload,
    vectorial_pupil_jones,
    workloads,
)


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        parsed = int(part)
        if parsed <= 0:
            raise ValueError("integer list values must be positive")
        out.append(parsed)
    if not out:
        raise ValueError("integer list is empty")
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def timed_loop(
    *,
    torch: Any,
    device: Any,
    func,
    repeats: int,
    warmups: int,
) -> tuple[Any, float, list[float]]:
    context = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    value = None
    with context():
        for _ in range(max(0, warmups)):
            value = func()
            synchronize(torch, device)
        times: list[float] = []
        for _ in range(max(1, repeats)):
            gc.collect()
            synchronize(torch, device)
            start = time.perf_counter()
            value = func()
            synchronize(torch, device)
            times.append(time.perf_counter() - start)
    if value is None:
        raise RuntimeError("timed loop did not run")
    return value, float(median(times)), times


def dense_loss_for_pupil(
    *,
    torch: Any,
    dense: TorchDenseDirectVectorialDebyeWolf,
    pupil: Any,
    target: Any,
    weight: Any,
) -> float:
    context = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    with context():
        field = dense.forward(pupil)
        loss, _ = field_match_loss_and_residual(torch, field, target, weight)
    return float(loss.detach().cpu().item())


def run_design_loop(
    *,
    torch: Any,
    initial_pupil: Any,
    target: Any,
    weight: Any,
    mixing: Any,
    n_iterations: int,
    phase_step_radians: float,
    forward,
    adjoint,
) -> tuple[Any, Any]:
    pupil = initial_pupil.clone()
    loss = None
    for _ in range(n_iterations):
        field = forward(pupil)
        loss, residual = field_match_loss_and_residual(torch, field, target, weight)
        grad = adjoint(residual)
        phase_grad = phase_gradient(torch, pupil, grad)
        pupil = phase_step_pupil(torch, pupil, phase_grad, phase_step_radians)
    if loss is None:
        raise RuntimeError("design loop did not execute")
    return pupil, loss


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    torch = import_torch()
    if torch is None:
        return {
            "config": vars(args).copy(),
            "rows": [{"status": "skipped", "skip_reason": "torch is not installed"}],
        }
    try:
        device = resolve_device(torch, args.device)
    except RuntimeError as exc:
        return {
            "config": vars(args).copy(),
            "rows": [{"status": "skipped", "skip_reason": str(exc)}],
        }

    workload = workloads(args.workload_set)[args.workload_index]
    iteration_counts = parse_int_list(args.iterations)
    batch_sizes = parse_int_list(args.batch_sizes)

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
        _ = torch.empty((1,), dtype=torch.float32, device=device)
        synchronize(torch, device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    synchronize(torch, device)
    dense_setup_start = time.perf_counter()
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
    synchronize(torch, device)
    dense_setup_s = time.perf_counter() - dense_setup_start

    h_cutoff = vectorial_h_cutoff_for_workload(workload, args.h_margin)
    synchronize(torch, device)
    separable_setup_start = time.perf_counter()
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
    mixing_t = separable.as_tensor(mixing)
    synchronize(torch, device)
    separable_setup_s = time.perf_counter() - separable_setup_start

    weight_t = torch.as_tensor(weight_np, dtype=dense.real_dtype, device=device)
    rows: list[dict[str, Any]] = []
    target_generation_s_by_batch: dict[int, float] = {}

    for batch_size in batch_sizes:
        pupil_batch = make_pupil_batch(
            base_pupil,
            theta,
            phi,
            theta_max=workload.theta_max,
            batch_size=batch_size,
            phase_strength=args.initial_phase_strength,
        )
        teacher_batch = make_pupil_batch(
            base_pupil,
            theta,
            phi,
            theta_max=workload.theta_max,
            batch_size=batch_size,
            phase_strength=args.teacher_phase_strength,
        )
        initial_pupil = torch.as_tensor(pupil_batch, dtype=dense.complex_dtype, device=device)
        teacher_pupil = torch.as_tensor(teacher_batch, dtype=dense.complex_dtype, device=device)

        context = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        synchronize(torch, device)
        target_start = time.perf_counter()
        with context():
            target = dense.forward(teacher_pupil).detach()
            dense_initial_field = dense.forward(initial_pupil)
            dense_initial_loss_t, _ = field_match_loss_and_residual(
                torch,
                dense_initial_field,
                target,
                weight_t,
            )
            separable_initial_field = separable.evaluate_vectorial_batch(initial_pupil, mixing_t)
            separable_initial_loss_t, _ = field_match_loss_and_residual(
                torch,
                separable_initial_field,
                target,
                weight_t,
            )
        synchronize(torch, device)
        target_generation_s_by_batch[batch_size] = time.perf_counter() - target_start
        dense_initial_loss = float(dense_initial_loss_t.detach().cpu().item())
        separable_initial_loss = float(separable_initial_loss_t.detach().cpu().item())

        def dense_forward(pupil: Any) -> Any:
            return dense.forward(pupil)

        def dense_adjoint(residual: Any) -> Any:
            return dense.adjoint(residual)

        def separable_forward(pupil: Any) -> Any:
            return separable.evaluate_vectorial_batch(pupil, mixing_t)

        def separable_adjoint(residual: Any) -> Any:
            return separable.adjoint_vectorial_batch(residual, mixing_t)

        for n_iterations in iteration_counts:
            dense_value, dense_hot_s, dense_times = timed_loop(
                torch=torch,
                device=device,
                func=lambda n=n_iterations: run_design_loop(
                    torch=torch,
                    initial_pupil=initial_pupil,
                    target=target,
                    weight=weight_t,
                    mixing=mixing_t,
                    n_iterations=n,
                    phase_step_radians=args.phase_step_radians,
                    forward=dense_forward,
                    adjoint=dense_adjoint,
                ),
                repeats=args.repeats,
                warmups=args.warmups,
            )
            sep_value, sep_hot_s, sep_times = timed_loop(
                torch=torch,
                device=device,
                func=lambda n=n_iterations: run_design_loop(
                    torch=torch,
                    initial_pupil=initial_pupil,
                    target=target,
                    weight=weight_t,
                    mixing=mixing_t,
                    n_iterations=n,
                    phase_step_radians=args.phase_step_radians,
                    forward=separable_forward,
                    adjoint=separable_adjoint,
                ),
                repeats=args.repeats,
                warmups=args.warmups,
            )
            dense_final_pupil, dense_final_loss_t = dense_value
            sep_final_pupil, sep_final_loss_t = sep_value
            dense_final_loss = float(dense_final_loss_t.detach().cpu().item())
            sep_final_loss = float(sep_final_loss_t.detach().cpu().item())
            dense_loss_of_sep_final = dense_loss_for_pupil(
                torch=torch,
                dense=dense,
                pupil=sep_final_pupil,
                target=target,
                weight=weight_t,
            )
            dense_loss_of_dense_final = dense_loss_for_pupil(
                torch=torch,
                dense=dense,
                pupil=dense_final_pupil,
                target=target,
                weight=weight_t,
            )
            dense_pupil_np = to_numpy(torch, device, dense_final_pupil)
            sep_pupil_np = to_numpy(torch, device, sep_final_pupil)
            final_pupil_l2 = relative_l2(sep_pupil_np, dense_pupil_np)

            dense_total_with_setup = dense_setup_s + dense_hot_s
            sep_total_with_setup = separable_setup_s + sep_hot_s
            dense_ms_per_iter = 1e3 * dense_hot_s / float(n_iterations)
            sep_ms_per_iter = 1e3 * sep_hot_s / float(n_iterations)
            hot_speedup = dense_hot_s / sep_hot_s
            setup_inclusive_speedup = dense_total_with_setup / sep_total_with_setup
            slope_delta_s = dense_hot_s / float(n_iterations) - sep_hot_s / float(n_iterations)
            if slope_delta_s > 0.0:
                break_even_iterations = max(
                    0.0,
                    (separable_setup_s - dense_setup_s) / slope_delta_s,
                )
            else:
                break_even_iterations = None

            rows.append(
                {
                    "status": "ok",
                    "objective": "weighted_cylindrical_roi_field_match_phase_only_loop",
                    "workload": workload.name,
                    "workload_set": args.workload_set,
                    "workload_index": args.workload_index,
                    "device": str(device),
                    "device_name": device_name(torch, device),
                    "torch_version": package_version("torch"),
                    "torch_cuda_version": getattr(torch.version, "cuda", None),
                    "dtype": args.dtype,
                    "batch_size": int(batch_size),
                    "n_iterations": int(n_iterations),
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
                    "roi_weight_nonzero_fraction": float(
                        np.count_nonzero(weight_np > 1e-6) / weight_np.size
                    ),
                    "h_cutoff": int(h_cutoff),
                    "separable_used_modes": int(separable.used_modes),
                    "separable_basis_mode": separable.basis_mode,
                    "separable_contract_mode": separable.contract_mode,
                    "separable_basis_mib": float(separable.basis_mib),
                    "dense_source_mib": float(dense.source_mib),
                    "dense_setup_s": float(dense_setup_s),
                    "separable_setup_s": float(separable_setup_s),
                    "target_generation_s": float(target_generation_s_by_batch[batch_size]),
                    "dense_hot_s": float(dense_hot_s),
                    "separable_hot_s": float(sep_hot_s),
                    "dense_ms_per_iteration": float(dense_ms_per_iter),
                    "separable_ms_per_iteration": float(sep_ms_per_iter),
                    "hot_loop_speedup_dense_vs_separable": float(hot_speedup),
                    "dense_total_with_setup_s": float(dense_total_with_setup),
                    "separable_total_with_setup_s": float(sep_total_with_setup),
                    "setup_inclusive_speedup_dense_vs_separable": float(setup_inclusive_speedup),
                    "break_even_iterations_estimate": break_even_iterations,
                    "dense_initial_loss": dense_initial_loss,
                    "separable_initial_loss": separable_initial_loss,
                    "dense_final_loss": dense_final_loss,
                    "separable_final_loss": sep_final_loss,
                    "dense_loss_of_dense_final_pupil": dense_loss_of_dense_final,
                    "dense_loss_of_separable_final_pupil": dense_loss_of_sep_final,
                    "dense_loss_decrease_fraction": float(
                        (dense_initial_loss - dense_loss_of_dense_final)
                        / max(abs(dense_initial_loss), 1e-300)
                    ),
                    "separable_loss_decrease_fraction_evaluated_by_dense": float(
                        (dense_initial_loss - dense_loss_of_sep_final)
                        / max(abs(dense_initial_loss), 1e-300)
                    ),
                    "final_pupil_l2_separable_vs_dense": float(final_pupil_l2),
                    "dense_times_s": " ".join(f"{value:.9g}" for value in dense_times),
                    "separable_times_s": " ".join(f"{value:.9g}" for value in sep_times),
                    "gpu_peak_allocated_mib": None
                    if device.type != "cuda"
                    else float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
                    "note": (
                        "target generation is excluded from hot/setup loop timing; "
                        "both paths use the same dense-generated teacher target"
                    ),
                }
            )

    config = vars(args).copy()
    config.update(
        {
            "iteration_counts": iteration_counts,
            "batch_sizes": batch_sizes,
            "resolved_device": str(device),
            "device_name": device_name(torch, device),
            "workload": workload.__dict__,
        }
    )
    return {"config": config, "rows": rows}


def plot_results(path_png: Path, path_svg: Path, rows: list[dict[str, Any]]) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return
    sns.set_theme(style="whitegrid", context="talk")
    colors = {
        "dense": "#4b5563",
        "separable": "#2563eb",
        "speedup": "#c2410c",
    }
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.0))

    batch_sizes = sorted({int(row["batch_size"]) for row in ok_rows})
    for batch_size in batch_sizes:
        sub = sorted(
            [row for row in ok_rows if int(row["batch_size"]) == batch_size],
            key=lambda row: int(row["n_iterations"]),
        )
        x = np.array([int(row["n_iterations"]) for row in sub], dtype=float)
        dense_hot = np.array([float(row["dense_hot_s"]) for row in sub], dtype=float)
        sep_hot = np.array([float(row["separable_hot_s"]) for row in sub], dtype=float)
        speedup = np.array(
            [float(row["hot_loop_speedup_dense_vs_separable"]) for row in sub],
            dtype=float,
        )
        axes[0].plot(
            x,
            dense_hot,
            marker="o",
            color=colors["dense"],
            alpha=0.45 + 0.12 * batch_sizes.index(batch_size),
            label=f"dense b{batch_size}",
        )
        axes[0].plot(
            x,
            sep_hot,
            marker="s",
            color=colors["separable"],
            alpha=0.45 + 0.12 * batch_sizes.index(batch_size),
            label=f"separable b{batch_size}",
        )
        axes[1].plot(
            x,
            speedup,
            marker="o",
            label=f"batch {batch_size}",
        )
        axes[2].plot(
            x,
            np.array([float(row["dense_ms_per_iteration"]) for row in sub], dtype=float),
            marker="o",
            color=colors["dense"],
            alpha=0.45 + 0.12 * batch_sizes.index(batch_size),
            label=f"dense b{batch_size}",
        )
        axes[2].plot(
            x,
            np.array([float(row["separable_ms_per_iteration"]) for row in sub], dtype=float),
            marker="s",
            color=colors["separable"],
            alpha=0.45 + 0.12 * batch_sizes.index(batch_size),
            label=f"separable b{batch_size}",
        )

    axes[0].set_title("A. Hot-loop total time", loc="left", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Phase-update iterations")
    axes[0].set_ylabel("Time (s)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].set_title("B. Structured-loop speedup", loc="left", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Phase-update iterations")
    axes[1].set_ylabel("Dense / separable hot time")
    axes[1].set_xscale("log")
    axes[1].axhline(1.0, color="#111827", linewidth=1.0, alpha=0.45)
    axes[1].legend(fontsize=9)

    axes[2].set_title("C. Per-iteration cost", loc="left", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Phase-update iterations")
    axes[2].set_ylabel("ms / iteration")
    axes[2].set_xscale("log")
    axes[2].legend(fontsize=8, ncol=2)

    fig.suptitle(
        "High-NA repeated inverse-design loop benchmark",
        x=0.055,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.02,
        "Loop: vectorial forward, weighted cylindrical ROI residual, adjoint, shared phase-gradient, phase-only update.",
        fontsize=9,
        color="#6b7280",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_svg)
    plt.close(fig)


def write_summary(
    path: Path,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    figure_png: Path,
) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# High-NA repeated inverse-design loop benchmark",
        "",
        "This benchmark measures the repeated phase-mask design primitive:",
        "",
        "`vectorial forward -> weighted cylindrical ROI residual -> adjoint -> shared phase-gradient -> phase-only update`.",
        "",
        "The target field is generated once with the dense direct CUDA path and excluded from loop timing.",
        "",
        "## Config",
        "",
        f"- workload: `{config.get('workload', {}).get('name', 'n/a')}`",
        f"- device: `{config.get('resolved_device', config.get('device'))}`",
        f"- device name: `{config.get('device_name', 'n/a')}`",
        f"- dtype: `{config.get('dtype')}`",
        f"- basis mode: `{config.get('basis_mode', 'separate')}`",
        f"- contract mode: `{config.get('contract_mode', 'einsum')}`",
        f"- iterations: `{config.get('iteration_counts')}`",
        f"- batch sizes: `{config.get('batch_sizes')}`",
        f"- roi case: `{config.get('roi_case')}`",
        "",
        "## Results",
        "",
        "| batch | iterations | dense hot s | separable hot s | speedup | dense ms/iter | separable ms/iter | setup-inclusive speedup | final dense-loss decrease from separable step | final pupil L2 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ok_rows:
        lines.append(
            "| {batch} | {iters} | {dense_hot} | {sep_hot} | {speedup} | {dense_ms} | {sep_ms} | {setup_speedup} | {loss_dec} | {pupil_l2} |".format(
                batch=row["batch_size"],
                iters=row["n_iterations"],
                dense_hot=fmt(row["dense_hot_s"]),
                sep_hot=fmt(row["separable_hot_s"]),
                speedup=fmt(row["hot_loop_speedup_dense_vs_separable"]),
                dense_ms=fmt(row["dense_ms_per_iteration"]),
                sep_ms=fmt(row["separable_ms_per_iteration"]),
                setup_speedup=fmt(row["setup_inclusive_speedup_dense_vs_separable"]),
                loss_dec=fmt(row["separable_loss_decrease_fraction_evaluated_by_dense"]),
                pupil_l2=fmt(row["final_pupil_l2_separable_vs_dense"]),
            )
        )

    if ok_rows:
        max_n = max(int(row["n_iterations"]) for row in ok_rows)
        representative = [
            row for row in ok_rows if int(row["n_iterations"]) == max_n
        ]
        lines.extend(
            [
                "",
                f"## Long-loop readout at {max_n} iterations",
                "",
                "| batch | hot speedup | setup-inclusive speedup | dense ms/iter | separable ms/iter | break-even iterations |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in representative:
            lines.append(
                "| {batch} | {hot} | {setup} | {dense_ms} | {sep_ms} | {break_even} |".format(
                    batch=row["batch_size"],
                    hot=fmt(row["hot_loop_speedup_dense_vs_separable"]),
                    setup=fmt(row["setup_inclusive_speedup_dense_vs_separable"]),
                    dense_ms=fmt(row["dense_ms_per_iteration"]),
                    sep_ms=fmt(row["separable_ms_per_iteration"]),
                    break_even=fmt(row["break_even_iterations_estimate"]),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a matched same-device inverse-design primitive, not a full external package optimizer comparison.",
            "- The useful slope is `ms/iteration` after fixed geometry and basis setup have been paid once.",
            "- Setup-inclusive speedup is reported to show whether the structured plan pays off for short loops.",
            "- The dense direct CUDA path remains the correctness/timing reference for this matched cylindrical ROI benchmark.",
            f"- figure: `{figure_png}`",
        ]
    )
    if rows and rows[0].get("status") == "skipped":
        lines.extend(["", f"Skipped: `{rows[0].get('skip_reason')}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark repeated vectorial High-NA phase-mask inverse-design loops."
    )
    parser.add_argument("--output-prefix", default="benchmark_results/high_na_inverse_design_loop")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    parser.add_argument("--workload-set", choices=["quick", "representative", "large"], default="representative")
    parser.add_argument("--workload-index", type=int, default=1)
    parser.add_argument("--iterations", default="1,10,50,100,300")
    parser.add_argument("--batch-sizes", default="1,8,32")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
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
    output_prefix = ROOT / args.output_prefix
    result = run_case(args)
    rows = result["rows"]
    figure_png = output_prefix.with_name(output_prefix.name + "_figure.png")
    figure_svg = output_prefix.with_name(output_prefix.name + "_figure.svg")
    plot_results(figure_png, figure_svg, rows)
    write_csv(output_prefix.with_suffix(".csv"), rows)
    write_json(output_prefix.with_suffix(".json"), result)
    write_summary(
        output_prefix.with_name(output_prefix.name + "_summary.md"),
        result["config"],
        rows,
        figure_png,
    )
    print(
        json.dumps(
            {
                "config": result["config"],
                "rows": rows,
                "csv": str(output_prefix.with_suffix(".csv")),
                "json": str(output_prefix.with_suffix(".json")),
                "summary_md": str(output_prefix.with_name(output_prefix.name + "_summary.md")),
                "figure_png": str(figure_png),
                "figure_svg": str(figure_svg),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
