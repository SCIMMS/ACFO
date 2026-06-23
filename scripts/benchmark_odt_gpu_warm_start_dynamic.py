from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark_high_na_torch_gpu import (  # noqa: E402
    device_name,
    import_torch,
    package_version,
    resolve_device,
    synchronize,
)
from benchmark_odt_realistic_geometry_reconstruction import (  # noqa: E402
    build_composite_context,
)
from benchmark_odt_torch_gpu_reconstruction import (  # noqa: E402
    TorchCompositeOdtPlan,
    coeff_norm2,
    torch_dtypes,
)


@dataclass(frozen=True)
class DynamicGrid:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    r_axis: np.ndarray
    z_axis: np.ndarray
    beta_axis: np.ndarray
    volume: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.volume.shape


@dataclass
class UpdateResult:
    x: Any
    pred: Any
    row: dict[str, Any]


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed < 0:
            raise ValueError("update counts must be non-negative")
        out.append(parsed)
    if not out:
        raise ValueError("at least one update count is required")
    return out


def median_or_none(values: list[float]) -> float | None:
    return None if not values else float(median(values))


def torch_scalar(value: Any) -> float:
    return float(value.detach().cpu().item())


def build_dynamic_grid(ctx: Any) -> DynamicGrid:
    obj = ctx.ring.obj
    r_axis = np.asarray(obj.r_axis, dtype=np.float64)
    z_axis = np.asarray(obj.z_axis, dtype=np.float64)
    beta_axis = np.asarray(obj.beta_axis, dtype=np.float64)
    rr, zz, bb = np.meshgrid(r_axis, z_axis, beta_axis, indexing="ij")
    volume = np.asarray(obj.volume_weights, dtype=np.float64).reshape(rr.shape)
    return DynamicGrid(
        x=np.ascontiguousarray(rr * np.cos(bb)),
        y=np.ascontiguousarray(rr * np.sin(bb)),
        z=np.ascontiguousarray(zz),
        r_axis=r_axis,
        z_axis=z_axis,
        beta_axis=beta_axis,
        volume=np.ascontiguousarray(volume),
    )


def dynamic_coefficients(
    grid: DynamicGrid,
    *,
    frame: int,
    frames: int,
    object_scale: float,
    motion_fraction: float,
    phase_drift_rad: float,
    np_complex: Any,
) -> np.ndarray:
    r_max = float(np.max(grid.r_axis))
    z_extent = max(float(np.max(np.abs(grid.z_axis))), 1e-12)
    t = float(frame) / float(max(frames - 1, 1))
    motion = float(motion_fraction) * r_max
    phase = 2.0 * math.pi * t

    radial_shell = np.sqrt(grid.x**2 + grid.y**2 + (0.70 * grid.z) ** 2)
    values = 0.10 * np.exp(-0.5 * ((radial_shell - 0.58 * r_max) / (0.22 * r_max)) ** 2)
    values = values.astype(np.complex128) * (1.0 + 0.08j * np.cos(2.0 * np.arctan2(grid.y, grid.x)))

    beads = [
        (-0.46 * r_max, 0.20 * r_max, -0.34 * z_extent, 0.16 * r_max, 1.00 + 0.10j, 0.00),
        (0.34 * r_max, -0.36 * r_max, 0.18 * z_extent, 0.13 * r_max, 0.72 - 0.18j, 0.31),
        (0.10 * r_max, 0.42 * r_max, 0.46 * z_extent, 0.12 * r_max, 0.56 + 0.30j, 0.62),
        (-0.06 * r_max, -0.12 * r_max, -0.04 * z_extent, 0.23 * r_max, 0.35 + 0.04j, 0.79),
    ]
    for cx0, cy0, cz0, sigma, amp0, offset in beads:
        theta = phase + 2.0 * math.pi * offset
        cx = cx0 + motion * math.sin(theta)
        cy = cy0 + 0.75 * motion * math.cos(theta)
        cz = cz0 + 0.35 * motion * (z_extent / max(r_max, 1e-12)) * math.sin(theta + 0.4)
        amp = amp0 * (1.0 + 0.04 * math.sin(1.7 * theta))
        dist2 = (grid.x - cx) ** 2 + (grid.y - cy) ** 2 + (grid.z - cz) ** 2
        values += amp * np.exp(-0.5 * dist2 / max(float(sigma) ** 2, 1e-300))

    drift = np.exp(1j * float(phase_drift_rad) * math.sin(phase))
    coeff = float(object_scale) * drift * values * grid.volume
    return np.ascontiguousarray(coeff.astype(np_complex, copy=False))


def timed_data(
    *,
    torch: Any,
    device: Any,
    plan: TorchCompositeOdtPlan,
    true_coeff: Any,
) -> tuple[Any, float]:
    synchronize(torch, device)
    start = time.perf_counter()
    data = plan.forward(true_coeff)
    synchronize(torch, device)
    return data, float(time.perf_counter() - start)


def add_measurement_noise(
    torch: Any,
    plan: TorchCompositeOdtPlan,
    data: Any,
    *,
    noise_rel: float,
    noise_model: str,
    noise_temporal_model: str,
    noise_seed: int,
    frame: int,
    frames: int,
    real_dtype: Any,
    synthetic_noise_rel: float,
    synthetic_noise_seed: int,
) -> Any:
    if float(noise_rel) <= 0.0 and float(synthetic_noise_rel) <= 0.0:
        return data

    model_offsets = {
        "independent": 101,
        "global_gain": 211,
        "illumination_gain": 307,
        "radial_background": 401,
        "phase_ramp": 503,
    }
    temporal_offsets = {
        "frame_independent": 10_001,
        "static": 20_003,
        "smooth": 30_011,
    }
    if noise_temporal_model not in temporal_offsets:
        raise ValueError(f"unsupported noise temporal model: {noise_temporal_model}")
    try:
        generator = torch.Generator(device=data.device)
    except TypeError:
        generator = torch.Generator(device=str(data.device))
    seed = (
        int(noise_seed)
        + model_offsets.get(str(noise_model), 0)
        + temporal_offsets[str(noise_temporal_model)]
    )
    if noise_temporal_model == "frame_independent":
        seed += 1_000_003 * (int(frame) + 17)
    generator.manual_seed(seed)

    def randn_complex(shape: tuple[int, ...]) -> Any:
        real = torch.randn(shape, dtype=real_dtype, device=data.device, generator=generator)
        imag = torch.randn(shape, dtype=real_dtype, device=data.device, generator=generator)
        return torch.complex(real, imag).to(dtype=data.dtype)

    def blocks(value: Any) -> list[tuple[Any, Any, int, int]]:
        out = []
        offset = 0
        for subplan in (plan.ring, plan.axis):
            if subplan is None:
                continue
            count = int(subplan.q_count)
            view = value[offset : offset + count].reshape(
                int(subplan.n_illum),
                int(subplan.cap_radial),
                int(subplan.cap_phi),
            )
            out.append((subplan, view, offset, count))
            offset += count
        return out

    def temporal_multiplier() -> float:
        if noise_temporal_model in {"frame_independent", "static"}:
            return 1.0
        t = float(frame) / float(max(int(frames) - 1, 1))
        return 0.35 + 0.65 * (0.5 + 0.5 * math.sin(2.0 * math.pi * (t + 0.13)))

    data_norm = torch.linalg.vector_norm(data)
    out = data

    if float(noise_rel) > 0.0:
        if noise_model == "independent":
            noise = randn_complex(tuple(data.shape))
        elif noise_model == "global_gain":
            angle = 2.0 * math.pi * float(torch.rand((), device=data.device, generator=generator).item())
            gain = complex(math.cos(angle), math.sin(angle))
            noise = data * gain
        elif noise_model == "illumination_gain":
            noise = torch.empty_like(data)
            for subplan, block, offset, count in blocks(data):
                gain = randn_complex((int(subplan.n_illum), 1, 1))
                noise[offset : offset + count].copy_((block * gain).reshape(-1))
        elif noise_model == "radial_background":
            noise = torch.empty_like(data)
            for subplan, _block, offset, count in blocks(data):
                n_illum = int(subplan.n_illum)
                cap_radial = int(subplan.cap_radial)
                cap_phi = int(subplan.cap_phi)
                r = torch.linspace(0.0, 1.0, cap_radial, dtype=real_dtype, device=data.device)
                phi = torch.linspace(
                    0.0,
                    2.0 * math.pi,
                    cap_phi + 1,
                    dtype=real_dtype,
                    device=data.device,
                )[:-1]
                center = 0.25 + 0.55 * torch.rand((n_illum, 1, 1), dtype=real_dtype, device=data.device, generator=generator)
                width = 0.10 + 0.20 * torch.rand((n_illum, 1, 1), dtype=real_dtype, device=data.device, generator=generator)
                phase = 2.0 * math.pi * torch.rand((n_illum, 1, 1), dtype=real_dtype, device=data.device, generator=generator)
                radial = torch.exp(-0.5 * ((r.reshape(1, cap_radial, 1) - center) / width) ** 2)
                angular = 1.0 + 0.25 * torch.cos(phi.reshape(1, 1, cap_phi) + phase)
                amp = randn_complex((n_illum, 1, 1))
                noise[offset : offset + count].copy_((amp * radial * angular).reshape(-1))
        elif noise_model == "phase_ramp":
            noise = torch.empty_like(data)
            for subplan, block, offset, count in blocks(data):
                n_illum = int(subplan.n_illum)
                cap_radial = int(subplan.cap_radial)
                cap_phi = int(subplan.cap_phi)
                r = torch.linspace(0.0, 1.0, cap_radial, dtype=real_dtype, device=data.device)
                phi = torch.linspace(
                    0.0,
                    2.0 * math.pi,
                    cap_phi + 1,
                    dtype=real_dtype,
                    device=data.device,
                )[:-1]
                direction = 2.0 * math.pi * torch.rand((n_illum, 1, 1), dtype=real_dtype, device=data.device, generator=generator)
                ramp = r.reshape(1, cap_radial, 1) * torch.cos(phi.reshape(1, 1, cap_phi) - direction)
                phase = torch.exp(1j * ramp).to(dtype=data.dtype)
                noise[offset : offset + count].copy_((block * (phase - 1.0)).reshape(-1))
        else:
            raise ValueError(f"unsupported noise model: {noise_model}")

        noise_norm = torch.clamp(torch.linalg.vector_norm(noise), min=1e-30)
        drift_rel = float(noise_rel) * temporal_multiplier()
        out = out + noise * (drift_rel * data_norm / noise_norm)

    if float(synthetic_noise_rel) > 0.0:
        try:
            synthetic_generator = torch.Generator(device=data.device)
        except TypeError:
            synthetic_generator = torch.Generator(device=str(data.device))
        synthetic_generator.manual_seed(int(synthetic_noise_seed) + 9_176_291 * (int(frame) + 23))
        real = torch.randn(tuple(data.shape), dtype=real_dtype, device=data.device, generator=synthetic_generator)
        imag = torch.randn(tuple(data.shape), dtype=real_dtype, device=data.device, generator=synthetic_generator)
        synthetic_noise = torch.complex(real, imag).to(dtype=data.dtype)
        synthetic_norm = torch.clamp(torch.linalg.vector_norm(synthetic_noise), min=1e-30)
        out = out + synthetic_noise * (float(synthetic_noise_rel) * data_norm / synthetic_norm)
    return out


def run_updates(
    plan: TorchCompositeOdtPlan,
    *,
    data: Any,
    true_coeff: Any,
    updates: int,
    x_init: Any | None,
    pred_init: Any | None,
    mode: str,
    frame: int,
    data_generation_s: float,
) -> UpdateResult:
    torch = plan.torch
    device = plan.device
    x = torch.zeros_like(true_coeff) if x_init is None else x_init.clone()
    pred = torch.zeros_like(data) if pred_init is None else pred_init.clone()
    data_norm = torch.clamp(torch.linalg.vector_norm(data), min=1e-30)
    true_norm = torch.clamp(torch.linalg.vector_norm(true_coeff), min=1e-30)
    alpha = torch.zeros((), dtype=plan.real_dtype, device=device)
    adjoint_times: list[float] = []
    forward_times: list[float] = []
    a_grad_buffer = torch.empty_like(data)

    synchronize(torch, device)
    start = time.perf_counter()
    for _ in range(int(updates)):
        residual = pred - data

        synchronize(torch, device)
        adj_start = time.perf_counter()
        grad = plan.adjoint(residual)
        synchronize(torch, device)
        adjoint_times.append(float(time.perf_counter() - adj_start))

        fw_start = time.perf_counter()
        a_grad = plan.forward_into(grad, a_grad_buffer)
        synchronize(torch, device)
        forward_times.append(float(time.perf_counter() - fw_start))

        alpha = coeff_norm2(torch, grad) / torch.clamp(coeff_norm2(torch, a_grad), min=1e-30)
        x = x - alpha * grad
        pred = pred - alpha * a_grad

    synchronize(torch, device)
    update_s = float(time.perf_counter() - start)
    loss_rel = torch.linalg.vector_norm(pred - data) / data_norm
    object_rel_l2 = torch.linalg.vector_norm(x - true_coeff) / true_norm
    synchronize(torch, device)
    row = {
        "mode": mode,
        "frame": int(frame),
        "updates": int(updates),
        "data_generation_s": float(data_generation_s),
        "update_s": update_s,
        "frame_total_with_synthetic_data_s": float(update_s + data_generation_s),
        "fps_excluding_synthetic_data": float(1.0 / update_s) if update_s > 0.0 else None,
        "fps_including_synthetic_data": float(1.0 / (update_s + data_generation_s))
        if update_s + data_generation_s > 0.0
        else None,
        "loss_rel": torch_scalar(loss_rel),
        "object_rel_l2": torch_scalar(object_rel_l2),
        "alpha_last": torch_scalar(alpha),
        "adjoint_s_total": float(sum(adjoint_times)),
        "line_forward_s_total": float(sum(forward_times)),
        "adjoint_s_median": median_or_none(adjoint_times),
        "line_forward_s_median": median_or_none(forward_times),
    }
    return UpdateResult(x=x, pred=pred, row=row)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        mode = str(row["mode"])
        if mode == "initial":
            continue
        groups.setdefault((mode, int(row["updates"])), []).append(row)

    out: list[dict[str, Any]] = []
    for (mode, updates), group_rows in sorted(groups.items()):
        update_s = [float(row["update_s"]) for row in group_rows]
        data_s = [float(row["data_generation_s"]) for row in group_rows]
        object_err = [float(row["object_rel_l2"]) for row in group_rows]
        loss = [float(row["loss_rel"]) for row in group_rows]
        fps = [
            float(row["fps_excluding_synthetic_data"])
            for row in group_rows
            if row["fps_excluding_synthetic_data"] is not None
        ]
        out.append(
            {
                "mode": mode,
                "updates": int(updates),
                "frames": int(len(group_rows)),
                "median_update_s": float(median(update_s)),
                "median_fps_excluding_synthetic_data": float(median(fps)) if fps else None,
                "median_data_generation_s": float(median(data_s)),
                "mean_object_rel_l2": float(np.mean(object_err)),
                "final_object_rel_l2": float(object_err[-1]),
                "mean_loss_rel": float(np.mean(loss)),
                "final_loss_rel": float(loss[-1]),
            }
        )
    return out


def add_warm_vs_cold(summary_rows: list[dict[str, Any]]) -> None:
    by_key = {(row["mode"], int(row["updates"])): row for row in summary_rows}
    for row in summary_rows:
        if row["mode"] != "warm_start":
            continue
        cold = by_key.get(("cold_start", int(row["updates"])))
        if cold is None:
            continue
        row["mean_object_rel_l2_vs_cold_ratio"] = (
            float(row["mean_object_rel_l2"]) / max(float(cold["mean_object_rel_l2"]), 1e-300)
        )
        row["final_object_rel_l2_vs_cold_ratio"] = (
            float(row["final_object_rel_l2"]) / max(float(cold["final_object_rel_l2"]), 1e-300)
        )


def plot_history(
    path: Path,
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    *,
    target_fps: float,
) -> None:
    cache_dir = ROOT / ".matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    ax_err, ax_time = axes

    for mode in ("warm_start", "cold_start", "reference"):
        mode_rows = [row for row in rows if row["mode"] == mode]
        updates_values = sorted({int(row["updates"]) for row in mode_rows})
        for updates in updates_values:
            group = [row for row in mode_rows if int(row["updates"]) == updates]
            group.sort(key=lambda row: int(row["frame"]))
            label = f"{mode}, {updates} update" + ("s" if updates != 1 else "")
            ax_err.plot(
                [int(row["frame"]) for row in group],
                [float(row["object_rel_l2"]) for row in group],
                marker="o",
                linewidth=1.5,
                markersize=3.5,
                label=label,
            )
    ax_err.set_xlabel("frame")
    ax_err.set_ylabel("object rel-L2")
    ax_err.set_yscale("log")
    ax_err.grid(True, alpha=0.25)
    ax_err.legend(fontsize=7)

    labels = []
    values = []
    for row in summary_rows:
        if row["mode"] not in {"warm_start", "cold_start"}:
            continue
        labels.append(f"{row['mode'].replace('_', ' ')}\n{row['updates']} upd")
        values.append(1000.0 * float(row["median_update_s"]))
    ax_time.bar(np.arange(len(labels)), values, color="#4c78a8")
    if target_fps > 0.0:
        ax_time.axhline(
            1000.0 / target_fps,
            color="#d62728",
            linestyle="--",
            linewidth=1.2,
            label=f"{target_fps:g} FPS",
        )
    ax_time.set_xticks(np.arange(len(labels)))
    ax_time.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax_time.set_ylabel("median update latency (ms)")
    ax_time.grid(True, axis="y", alpha=0.25)
    ax_time.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary_markdown(
    path: Path,
    *,
    summary: dict[str, Any],
    summary_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# ODT GPU dynamic warm-start benchmark",
        "",
        "This benchmark keeps the realistic ring-plus-axis ODT geometry fixed and changes only the synthetic object over a time-lapse sequence. Warm-start rows reuse both the previous reconstruction `x` and the previous prediction `A x`, so each new frame only pays the requested number of update steps.",
        "",
        "## Configuration",
        "",
        f"- device: `{summary['device_name']}`",
        f"- torch: `{summary['torch_version']}`",
        f"- dtype: `{summary['dtype']}`",
        f"- total illuminations: `{summary['total_illumination_count']}`",
        f"- total q samples: `{summary['total_q_samples']}`",
        f"- object bins: `{summary['object_bins']}`",
        f"- cap: `{summary['cap_radial']} x {summary['cap_phi']}`",
        f"- frames: `{summary['frames']}`",
        f"- target FPS: `{summary['target_fps']}`",
        f"- target frame budget: `{summary['target_frame_budget_ms']:.2f}` ms",
        f"- initial mode: `{summary['initial_mode']}`",
        f"- initial iterations: `{summary['initial_iterations']}`",
        f"- warmup updates: `{summary['warmup_updates']}`",
        f"- update counts: `{summary['updates_per_frame']}`",
        f"- synthetic motion fraction: `{summary['motion_fraction']}`",
        f"- synthetic phase drift rad: `{summary['phase_drift_rad']}`",
        f"- measurement noise model: `{summary['noise_model']}`",
        f"- measurement noise temporal model: `{summary['noise_temporal_model']}`",
        f"- measurement noise rel-L2: `{summary['noise_rel']}`",
        f"- synthetic independent noise rel-L2: `{summary['synthetic_noise_rel']}`",
        f"- GPU basis memory: `{summary['gpu_basis_mib']:.3f} MiB`",
        f"- GPU peak allocated: `{summary['gpu_peak_allocated_mib']}` MiB",
        "",
        "## Aggregate Results",
        "",
        "| mode | updates/frame | median latency ms | FPS excl. synthetic data | mean object rel-L2 | final object rel-L2 | mean loss rel |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            f"{row['mode']} | "
            f"{int(row['updates'])} | "
            f"{1000.0 * float(row['median_update_s']):.2f} | "
            f"{float(row['median_fps_excluding_synthetic_data']):.1f} | "
            f"{float(row['mean_object_rel_l2']):.4g} | "
            f"{float(row['final_object_rel_l2']):.4g} | "
            f"{float(row['mean_loss_rel']):.4g} |"
        )

    lines.extend(
        [
            "",
            "## Warm-Start Readout",
            "",
        ]
    )
    warm_rows = [row for row in summary_rows if row["mode"] == "warm_start"]
    for row in warm_rows:
        ratio = row.get("mean_object_rel_l2_vs_cold_ratio")
        ratio_text = "n/a" if ratio is None else f"{float(ratio):.3f}x"
        lines.append(
            f"- `{int(row['updates'])}` update/frame: median latency `{1000.0 * float(row['median_update_s']):.2f}` ms, "
            f"throughput `{float(row['median_fps_excluding_synthetic_data']):.1f}` FPS, "
            f"mean object error `{float(row['mean_object_rel_l2']):.4g}`; warm/cold error ratio `{ratio_text}`."
        )

    if summary.get("figure"):
        lines.extend(["", f"- figure: `{summary['figure']}`"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The latency columns exclude synthetic data generation because real acquisition would provide the measured field; the synthetic forward time is recorded separately in the CSV/JSON.",
            "- This is still a PyTorch tensor prototype. It demonstrates the algorithmic warm-start path, not the final low-level CUDA ceiling.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    ctx = build_composite_context(args)
    plan = TorchCompositeOdtPlan.from_context(ctx, torch=torch, device=device, dtype=args.dtype)
    grid = build_dynamic_grid(ctx)
    _, real_dtype, np_complex, _ = torch_dtypes(torch, args.dtype)
    updates_per_frame = parse_int_list(args.updates_per_frame)

    rows: list[dict[str, Any]] = []
    with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
        true0_np = dynamic_coefficients(
            grid,
            frame=0,
            frames=args.frames,
            object_scale=args.object_scale,
            motion_fraction=args.motion_fraction,
            phase_drift_rad=args.phase_drift_rad,
            np_complex=np_complex,
        )
        true0 = torch.as_tensor(true0_np, dtype=plan.complex_dtype, device=device)
        data0, data0_s = timed_data(torch=torch, device=device, plan=plan, true_coeff=true0)
        data0 = add_measurement_noise(
            torch,
            plan,
            data0,
            noise_rel=args.noise_rel,
            noise_model=args.noise_model,
            noise_temporal_model=args.noise_temporal_model,
            noise_seed=args.noise_seed,
            frame=0,
            frames=args.frames,
            real_dtype=real_dtype,
            synthetic_noise_rel=args.synthetic_noise_rel,
            synthetic_noise_seed=args.synthetic_noise_seed,
        )

        if int(args.warmup_updates) > 0:
            run_updates(
                plan,
                data=data0,
                true_coeff=true0,
                updates=args.warmup_updates,
                x_init=None,
                pred_init=None,
                mode="warmup",
                frame=-1,
                data_generation_s=0.0,
            )

        if args.initial_mode == "oracle":
            initial_x = true0.clone()
            initial_pred = data0.clone()
            initial_loss = 0.0
            initial_error = 0.0
            initial_update_s = 0.0
        else:
            initial = run_updates(
                plan,
                data=data0,
                true_coeff=true0,
                updates=args.initial_iterations,
                x_init=None,
                pred_init=None,
                mode="initial",
                frame=0,
                data_generation_s=data0_s,
            )
            initial_x = initial.x
            initial_pred = initial.pred
            rows.append(initial.row)
            initial_loss = float(initial.row["loss_rel"])
            initial_error = float(initial.row["object_rel_l2"])
            initial_update_s = float(initial.row["update_s"])

        if args.initial_mode == "oracle":
            rows.append(
                {
                    "mode": "initial",
                    "frame": 0,
                    "updates": 0,
                    "data_generation_s": data0_s,
                    "update_s": initial_update_s,
                    "frame_total_with_synthetic_data_s": data0_s,
                    "fps_excluding_synthetic_data": None,
                    "fps_including_synthetic_data": None,
                    "loss_rel": initial_loss,
                    "object_rel_l2": initial_error,
                    "alpha_last": 0.0,
                    "adjoint_s_total": 0.0,
                    "line_forward_s_total": 0.0,
                    "adjoint_s_median": None,
                    "line_forward_s_median": None,
                }
            )

        for updates in updates_per_frame:
            x = initial_x.clone()
            pred = initial_pred.clone()
            for frame in range(1, int(args.frames)):
                coeff_np = dynamic_coefficients(
                    grid,
                    frame=frame,
                    frames=args.frames,
                    object_scale=args.object_scale,
                    motion_fraction=args.motion_fraction,
                    phase_drift_rad=args.phase_drift_rad,
                    np_complex=np_complex,
                )
                true_coeff = torch.as_tensor(coeff_np, dtype=plan.complex_dtype, device=device)
                data, data_s = timed_data(torch=torch, device=device, plan=plan, true_coeff=true_coeff)
                data = add_measurement_noise(
                    torch,
                    plan,
                    data,
                    noise_rel=args.noise_rel,
                    noise_model=args.noise_model,
                    noise_temporal_model=args.noise_temporal_model,
                    noise_seed=args.noise_seed,
                    frame=frame,
                    frames=args.frames,
                    real_dtype=real_dtype,
                    synthetic_noise_rel=args.synthetic_noise_rel,
                    synthetic_noise_seed=args.synthetic_noise_seed,
                )
                result = run_updates(
                    plan,
                    data=data,
                    true_coeff=true_coeff,
                    updates=updates,
                    x_init=x,
                    pred_init=pred,
                    mode="warm_start",
                    frame=frame,
                    data_generation_s=data_s,
                )
                rows.append(result.row)
                x = result.x
                pred = result.pred

        if args.include_cold_start:
            for updates in updates_per_frame:
                for frame in range(1, int(args.frames)):
                    coeff_np = dynamic_coefficients(
                        grid,
                        frame=frame,
                        frames=args.frames,
                        object_scale=args.object_scale,
                        motion_fraction=args.motion_fraction,
                        phase_drift_rad=args.phase_drift_rad,
                        np_complex=np_complex,
                    )
                    true_coeff = torch.as_tensor(coeff_np, dtype=plan.complex_dtype, device=device)
                    data, data_s = timed_data(
                        torch=torch,
                        device=device,
                        plan=plan,
                        true_coeff=true_coeff,
                    )
                    data = add_measurement_noise(
                        torch,
                        plan,
                        data,
                        noise_rel=args.noise_rel,
                        noise_model=args.noise_model,
                        noise_temporal_model=args.noise_temporal_model,
                        noise_seed=args.noise_seed,
                        frame=frame,
                        frames=args.frames,
                        real_dtype=real_dtype,
                        synthetic_noise_rel=args.synthetic_noise_rel,
                        synthetic_noise_seed=args.synthetic_noise_seed,
                    )
                    result = run_updates(
                        plan,
                        data=data,
                        true_coeff=true_coeff,
                        updates=updates,
                        x_init=None,
                        pred_init=None,
                        mode="cold_start",
                        frame=frame,
                        data_generation_s=data_s,
                    )
                    rows.append(result.row)

        if args.reference_iterations > 0:
            for frame in range(int(args.frames)):
                coeff_np = dynamic_coefficients(
                    grid,
                    frame=frame,
                    frames=args.frames,
                    object_scale=args.object_scale,
                    motion_fraction=args.motion_fraction,
                    phase_drift_rad=args.phase_drift_rad,
                    np_complex=np_complex,
                )
                true_coeff = torch.as_tensor(coeff_np, dtype=plan.complex_dtype, device=device)
                data, data_s = timed_data(torch=torch, device=device, plan=plan, true_coeff=true_coeff)
                data = add_measurement_noise(
                    torch,
                    plan,
                    data,
                    noise_rel=args.noise_rel,
                    noise_model=args.noise_model,
                    noise_temporal_model=args.noise_temporal_model,
                    noise_seed=args.noise_seed,
                    frame=frame,
                    frames=args.frames,
                    real_dtype=real_dtype,
                    synthetic_noise_rel=args.synthetic_noise_rel,
                    synthetic_noise_seed=args.synthetic_noise_seed,
                )
                result = run_updates(
                    plan,
                    data=data,
                    true_coeff=true_coeff,
                    updates=args.reference_iterations,
                    x_init=None,
                    pred_init=None,
                    mode="reference",
                    frame=frame,
                    data_generation_s=data_s,
                )
                rows.append(result.row)

    summary_rows = aggregate_rows(rows)
    add_warm_vs_cold(summary_rows)

    peak_mib = None
    if device.type == "cuda":
        peak_mib = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    ring_na = math.sin(math.radians(float(args.illumination_angle_deg)))
    summary = {
        "device": str(device),
        "device_name": device_name(torch, device),
        "torch_version": package_version("torch"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "dtype": args.dtype,
        "ring_illum": int(args.ring_illum),
        "axis_illumination_included": not bool(args.skip_axis_illumination),
        "total_illumination_count": int(args.ring_illum)
        + (0 if args.skip_axis_illumination else 1),
        "illumination_angle_deg": float(args.illumination_angle_deg),
        "illumination_ring_na": float(ring_na),
        "detector_na": float(args.detector_na),
        "cap_radial": int(args.cap_radial),
        "cap_phi": int(args.cap_phi),
        "total_q_samples": int(plan.q_count),
        "object_bins": int(np.prod(grid.shape)),
        "gpu_basis_mib": float(plan.basis_mib),
        "gpu_peak_allocated_mib": peak_mib,
        "frames": int(args.frames),
        "target_fps": float(args.target_fps),
        "target_frame_budget_ms": float(1000.0 / args.target_fps)
        if float(args.target_fps) > 0.0
        else float("inf"),
        "initial_mode": args.initial_mode,
        "initial_iterations": int(args.initial_iterations),
        "warmup_updates": int(args.warmup_updates),
        "initial_loss_rel": float(initial_loss),
        "initial_object_rel_l2": float(initial_error),
        "initial_update_s": float(initial_update_s),
        "updates_per_frame": updates_per_frame,
        "reference_iterations": int(args.reference_iterations),
        "motion_fraction": float(args.motion_fraction),
        "phase_drift_rad": float(args.phase_drift_rad),
        "noise_model": str(args.noise_model),
        "noise_temporal_model": str(args.noise_temporal_model),
        "noise_rel": float(args.noise_rel),
        "noise_seed": int(args.noise_seed),
        "synthetic_noise_rel": float(args.synthetic_noise_rel),
        "synthetic_noise_seed": int(args.synthetic_noise_seed),
        "summary_rows": summary_rows,
        "history_csv": str(args.csv),
        "figure": str(args.figure) if args.figure else None,
    }

    write_csv(args.csv, rows)
    write_json(args.out, {"config": vars(args), "summary": summary, "summary_rows": summary_rows, "history": rows})
    if args.figure:
        plot_history(args.figure, rows, summary_rows, target_fps=float(args.target_fps))
    if args.summary_md:
        write_summary_markdown(args.summary_md, summary=summary, summary_rows=summary_rows)
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dynamic ODT GPU warm-start benchmark for realistic ring-plus-axis geometry."
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    p.add_argument("--n-beta", type=int, default=384)
    p.add_argument("--n-r", type=int, default=16)
    p.add_argument("--n-z", type=int, default=15)
    p.add_argument("--r-max", type=float, default=1.0)
    p.add_argument("--z-max", type=float, default=0.8)
    p.add_argument("--phantom", choices=["beads", "random_beads", "shell"], default="random_beads")
    p.add_argument("--object-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--k", type=float, default=17.307319527958313)
    p.add_argument("--detector-na", type=float, default=0.9240924092409241)
    p.add_argument("--illumination-angle-deg", type=float, default=49.0)
    p.add_argument("--ring-illum", type=int, default=100)
    p.add_argument("--skip-axis-illumination", action="store_true")
    p.add_argument("--cap-radial", type=int, default=128)
    p.add_argument("--cap-phi", type=int, default=512)
    p.add_argument("--h-cutoff", type=int, default=None)
    p.add_argument("--h-margin", type=int, default=20)
    p.add_argument("--l-margin", type=int, default=18)
    p.add_argument("--cone-l-prune-threshold", type=float, default=1e-12)
    p.add_argument("--cpp-threads", type=int, default=16)
    p.add_argument(
        "--forward-execute-mode",
        choices=["prepared", "wrapper"],
        default="prepared",
    )
    p.add_argument(
        "--forward-kernel-mode",
        choices=["compact", "partitioned"],
        default="partitioned",
    )
    p.add_argument(
        "--native-prepared-plan-mode",
        choices=["auto", "direct", "gathered", "gathered-zmajor"],
        default="auto",
    )
    p.add_argument("--native-prepared-gather-threshold", type=int, default=8192)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--target-fps", type=float, default=30.0)
    p.add_argument("--updates-per-frame", default="1,2,3")
    p.add_argument("--initial-mode", choices=["cold_start", "oracle"], default="cold_start")
    p.add_argument("--initial-iterations", type=int, default=10)
    p.add_argument("--warmup-updates", type=int, default=1)
    p.add_argument("--reference-iterations", type=int, default=8)
    p.add_argument("--include-cold-start", action="store_true")
    p.add_argument("--motion-fraction", type=float, default=0.08)
    p.add_argument("--phase-drift-rad", type=float, default=0.12)
    p.add_argument(
        "--noise-rel",
        type=float,
        default=0.0,
        help="Relative measurement perturbation: ||delta data||_2 / ||clean data||_2.",
    )
    p.add_argument(
        "--noise-model",
        choices=[
            "independent",
            "global_gain",
            "illumination_gain",
            "radial_background",
            "phase_ramp",
        ],
        default="independent",
        help="Measurement perturbation family, normalized to --noise-rel.",
    )
    p.add_argument(
        "--noise-temporal-model",
        choices=["frame_independent", "static", "smooth"],
        default="frame_independent",
        help="Temporal behavior of the structured measurement perturbation.",
    )
    p.add_argument("--noise-seed", type=int, default=12345)
    p.add_argument(
        "--synthetic-noise-rel",
        type=float,
        default=0.0,
        help="Additional frame-independent complex Gaussian detector/read noise, normalized to clean data.",
    )
    p.add_argument("--synthetic-noise-seed", type=int, default=54321)
    p.add_argument("--finufft-eps", type=float, default=1e-12)
    p.add_argument("--finufft-q-batch-size", type=int, default=1_048_576)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_torch_gpu_dynamic_warm_start.json",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_torch_gpu_dynamic_warm_start_history.csv",
    )
    p.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_torch_gpu_dynamic_warm_start.png",
    )
    p.add_argument(
        "--summary-md",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_torch_gpu_dynamic_warm_start_summary.md",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    if args.frames < 2:
        raise ValueError("--frames must be at least 2")
    summary = run(args)
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
