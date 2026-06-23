from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from benchmark_high_na_torch_gpu import (  # noqa: E402
    device_name,
    import_torch,
    package_version,
    resolve_device,
    synchronize,
)
from benchmark_odt_cufinufft_gpu_baseline import (  # noqa: E402
    import_cufinufft_modules,
    make_cufinufft_composite,
    synchronize_cupy,
)
from benchmark_odt_pattern_intensity_reconstruction import (  # noqa: E402
    PatternIntensityOperator,
    coeff_norm,
    method_stats,
    random_warm_start,
    run_intensity_loop,
)
from benchmark_odt_pattern_layer_gpu import (  # noqa: E402
    TorchPatternedRingPlan,
    fmt,
    make_pattern_matrix_np,
    parse_case,
    speedup,
)
from benchmark_odt_realistic_geometry_reconstruction import build_composite_context  # noqa: E402
from benchmark_odt_torch_gpu_reconstruction import (  # noqa: E402
    TorchCompositeOdtPlan,
    torch_dtypes,
)


def parse_float_list(value: str) -> list[float]:
    out = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not out:
        raise ValueError("at least one float is required")
    return out


def parse_str_list(value: str) -> list[str]:
    out = [item.strip() for item in value.split(",") if item.strip()]
    if not out:
        raise ValueError("at least one value is required")
    return out


def slug_float(value: float) -> str:
    text = f"{float(value):.6g}".replace("-", "m").replace(".", "p")
    return text.replace("+", "")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def perturb_pattern_matrix(
    pattern: np.ndarray,
    *,
    kind: str,
    rel: float,
    seed: int,
) -> np.ndarray:
    if kind == "none" or float(rel) == 0.0:
        return np.ascontiguousarray(pattern.copy())
    rng = np.random.default_rng(int(seed))
    out = pattern.astype(np.complex128, copy=True)
    support = np.abs(out) > 0.0
    if kind == "phase":
        phase = np.zeros_like(out, dtype=np.float64)
        phase[support] = rng.normal(loc=0.0, scale=float(rel), size=int(np.count_nonzero(support)))
        out *= np.exp(1j * phase)
    elif kind == "amplitude":
        gain = np.ones_like(out, dtype=np.float64)
        gain[support] += float(rel) * rng.normal(size=int(np.count_nonzero(support)))
        gain = np.maximum(gain, 0.05)
        out *= gain
    elif kind == "phase_amplitude":
        phase = np.zeros_like(out, dtype=np.float64)
        phase[support] = rng.normal(loc=0.0, scale=float(rel), size=int(np.count_nonzero(support)))
        gain = np.ones_like(out, dtype=np.float64)
        gain[support] += float(rel) * rng.normal(size=int(np.count_nonzero(support)))
        gain = np.maximum(gain, 0.05)
        out *= gain * np.exp(1j * phase)
    else:
        raise ValueError(f"unsupported pattern mismatch kind: {kind}")
    row_norms = np.linalg.norm(out, axis=1)
    out = out / np.maximum(row_norms[:, None], 1e-300)
    return np.ascontiguousarray(out)


def add_intensity_perturbation(
    torch: Any,
    data: Any,
    *,
    model: str,
    rel: float,
    seed: int,
    n_patterns: int,
    cap_radial: int,
    cap_phi: int,
) -> tuple[Any, float]:
    if model == "none" or float(rel) == 0.0:
        return data, 0.0
    try:
        generator = torch.Generator(device=data.device)
    except TypeError:
        generator = torch.Generator(device=str(data.device))
    generator.manual_seed(int(seed))
    real_dtype = data.dtype
    data_norm = torch.clamp(torch.linalg.vector_norm(data), min=1e-30)

    if model == "independent":
        noise = torch.randn(tuple(data.shape), dtype=real_dtype, device=data.device, generator=generator)
    elif model == "pattern_gain":
        gain = torch.randn((int(n_patterns), 1), dtype=real_dtype, device=data.device, generator=generator)
        noise = data * gain
    elif model == "radial_background":
        radial = torch.linspace(0.0, 1.0, int(cap_radial), dtype=real_dtype, device=data.device)
        phi = torch.linspace(0.0, 2.0 * math.pi, int(cap_phi) + 1, dtype=real_dtype, device=data.device)[:-1]
        center = 0.25 + 0.55 * torch.rand(
            (int(n_patterns), 1, 1),
            dtype=real_dtype,
            device=data.device,
            generator=generator,
        )
        width = 0.10 + 0.20 * torch.rand(
            (int(n_patterns), 1, 1),
            dtype=real_dtype,
            device=data.device,
            generator=generator,
        )
        phase = 2.0 * math.pi * torch.rand(
            (int(n_patterns), 1, 1),
            dtype=real_dtype,
            device=data.device,
            generator=generator,
        )
        radial_profile = torch.exp(-0.5 * ((radial.reshape(1, int(cap_radial), 1) - center) / width) ** 2)
        angular = 1.0 + 0.25 * torch.cos(phi.reshape(1, 1, int(cap_phi)) + phase)
        amp = torch.randn((int(n_patterns), 1, 1), dtype=real_dtype, device=data.device, generator=generator)
        noise = (amp * radial_profile * angular).reshape(int(n_patterns), int(cap_radial) * int(cap_phi))
    else:
        raise ValueError(f"unsupported detector perturbation model: {model}")

    noise_norm = torch.clamp(torch.linalg.vector_norm(noise), min=1e-30)
    perturb = noise * (float(rel) * data_norm / noise_norm)
    out = data + perturb
    actual_rel = torch.linalg.vector_norm(out - data) / data_norm
    return out, float(actual_rel.detach().cpu().item())


def build_case(
    args: argparse.Namespace,
    *,
    label: str,
    n_illum: int,
    n_patterns: int,
    active: int,
    cap_radial: int,
    cap_phi: int,
) -> dict[str, Any]:
    torch = import_torch()
    if torch is None:
        raise RuntimeError("torch is not installed")
    device = resolve_device(torch, args.device)
    if device.type != "cuda":
        raise RuntimeError("pattern-intensity robustness sweep requires CUDA")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    case_args = argparse.Namespace(**vars(args))
    case_args.ring_illum = int(n_illum)
    case_args.cap_radial = int(cap_radial)
    case_args.cap_phi = int(cap_phi)
    case_args.skip_axis_illumination = True
    ctx = build_composite_context(case_args)
    composite = TorchCompositeOdtPlan.from_context(ctx, torch=torch, device=device, dtype=args.dtype)
    pattern_np = make_pattern_matrix_np(
        n_patterns=n_patterns,
        n_illum=n_illum,
        active=active,
        seed=args.seed + n_illum * 13 + n_patterns * 17 + active,
        kind=args.pattern_kind,
    )
    plan = TorchPatternedRingPlan.from_ring(
        torch=torch,
        device=device,
        ring=composite.ring,
        pattern_np=pattern_np,
        dtype=args.dtype,
    )
    _, _, np_complex, _ = torch_dtypes(torch, args.dtype)
    true_coeff_np = np.ascontiguousarray((ctx.ring.obj.coeff * args.object_scale).astype(np_complex, copy=False))
    true_coeff = torch.as_tensor(true_coeff_np, dtype=composite.complex_dtype, device=device)
    x0 = random_warm_start(
        torch,
        device,
        true_coeff,
        rel_noise=args.warm_start_noise_rel,
        seed=args.seed + 8009,
    )
    cp, _ = import_cufinufft_modules()
    cu_op = make_cufinufft_composite(
        ctx,
        dtype=args.dtype,
        plan_mode=args.cufinufft_plan_mode,
        eps=args.cufinufft_eps,
    )
    ours_op = PatternIntensityOperator(
        label="ours_gpu",
        torch=torch,
        device=device,
        plan=plan,
        cufinufft=None,
        cufinufft_eps=args.cufinufft_eps,
        coeff_shape=true_coeff_np.shape,
        use_cufinufft=False,
    )
    cu_operator = PatternIntensityOperator(
        label="cufinufft_gpu",
        torch=torch,
        device=device,
        plan=plan,
        cufinufft=cu_op,
        cufinufft_eps=args.cufinufft_eps,
        coeff_shape=true_coeff_np.shape,
        use_cufinufft=True,
    )
    return {
        "torch": torch,
        "device": device,
        "cp": cp,
        "ctx": ctx,
        "composite": composite,
        "plan": plan,
        "pattern_np": pattern_np,
        "true_coeff": true_coeff,
        "true_coeff_np_shape": true_coeff_np.shape,
        "x0": x0,
        "ours_op": ours_op,
        "cu_operator": cu_operator,
    }


def make_target_data(
    args: argparse.Namespace,
    case: dict[str, Any],
    *,
    n_illum: int,
    n_patterns: int,
    active: int,
    cap_radial: int,
    cap_phi: int,
    model: str,
    detector_model: str,
    detector_rel: float,
    pattern_mismatch: str,
    pattern_mismatch_rel: float,
    condition_seed: int,
) -> tuple[Any, dict[str, Any]]:
    torch = case["torch"]
    device = case["device"]
    pattern_target_np = perturb_pattern_matrix(
        case["pattern_np"],
        kind=pattern_mismatch,
        rel=pattern_mismatch_rel,
        seed=condition_seed + 1009,
    )
    target_plan = TorchPatternedRingPlan.from_ring(
        torch=torch,
        device=device,
        ring=case["composite"].ring,
        pattern_np=pattern_target_np,
        dtype=args.dtype,
    )
    target_op = PatternIntensityOperator(
        label="target_gpu",
        torch=torch,
        device=device,
        plan=target_plan,
        cufinufft=None,
        cufinufft_eps=args.cufinufft_eps,
        coeff_shape=case["true_coeff_np_shape"],
        use_cufinufft=False,
    )
    zero = torch.zeros((int(n_patterns), int(cap_radial) * int(cap_phi)), dtype=case["true_coeff"].real.dtype, device=device)
    if model == "coherent":
        clean = target_op.coherent_eval_grad(case["true_coeff"], zero)[2].detach()
    else:
        clean = target_op.incoherent_eval_grad(case["true_coeff"], zero)[2].detach()
    noisy, actual_detector_rel = add_intensity_perturbation(
        torch,
        clean,
        model=detector_model,
        rel=detector_rel,
        seed=condition_seed + 2027,
        n_patterns=n_patterns,
        cap_radial=cap_radial,
        cap_phi=cap_phi,
    )
    nominal_zero = torch.zeros_like(clean)
    if model == "coherent":
        nominal_clean = case["ours_op"].coherent_eval_grad(case["true_coeff"], nominal_zero)[2].detach()
    else:
        nominal_clean = case["ours_op"].incoherent_eval_grad(case["true_coeff"], nominal_zero)[2].detach()
    mismatch_rel = coeff_norm(torch, noisy - nominal_clean) / coeff_norm(torch, nominal_clean)
    target_meta = {
        "detector_actual_rel": float(actual_detector_rel),
        "target_vs_nominal_data_rel": float(mismatch_rel.detach().cpu().item()),
    }
    return noisy.detach(), target_meta


def run_condition(
    args: argparse.Namespace,
    case: dict[str, Any],
    *,
    label: str,
    n_illum: int,
    n_patterns: int,
    active: int,
    cap_radial: int,
    cap_phi: int,
    model: str,
    detector_model: str,
    detector_rel: float,
    pattern_mismatch: str,
    pattern_mismatch_rel: float,
    condition_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch = case["torch"]
    device = case["device"]
    cp = case["cp"]

    def sync_ours() -> None:
        synchronize(torch, device)

    def sync_cu() -> None:
        synchronize_cupy(cp)
        synchronize(torch, device)

    data, target_meta = make_target_data(
        args,
        case,
        n_illum=n_illum,
        n_patterns=n_patterns,
        active=active,
        cap_radial=cap_radial,
        cap_phi=cap_phi,
        model=model,
        detector_model=detector_model,
        detector_rel=detector_rel,
        pattern_mismatch=pattern_mismatch,
        pattern_mismatch_rel=pattern_mismatch_rel,
        condition_seed=condition_seed,
    )
    with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
        ours_rows = run_intensity_loop(
            operator=case["ours_op"],
            model=model,
            x0=case["x0"],
            true_coeff=case["true_coeff"],
            data=data,
            iterations=args.iterations,
            step_rel=args.step_rel,
            sync=sync_ours,
        )
        cu_rows = run_intensity_loop(
            operator=case["cu_operator"],
            model=model,
            x0=case["x0"],
            true_coeff=case["true_coeff"],
            data=data,
            iterations=args.iterations,
            step_rel=args.step_rel,
            sync=sync_cu,
        )
    common = {
        "case": label,
        "model": model,
        "detector_model": detector_model,
        "detector_rel": float(detector_rel),
        "pattern_mismatch": pattern_mismatch,
        "pattern_mismatch_rel": float(pattern_mismatch_rel),
        "detector_actual_rel": float(target_meta["detector_actual_rel"]),
        "target_vs_nominal_data_rel": float(target_meta["target_vs_nominal_data_rel"]),
        "n_illum": int(n_illum),
        "n_patterns": int(n_patterns),
        "active_per_pattern": int(active),
        "cap_radial": int(cap_radial),
        "cap_phi": int(cap_phi),
        "stack_q_samples": int(n_illum * cap_radial * cap_phi),
        "pattern_q_samples": int(n_patterns * cap_radial * cap_phi),
        "object_bins": int(case["true_coeff"].numel()),
    }
    for row in ours_rows + cu_rows:
        row.update(common)
    ours_stats = method_stats(ours_rows, method="ours_gpu", model=model)
    cu_stats = method_stats(cu_rows, method="cufinufft_gpu", model=model)
    summary = {
        **common,
        "iterations": int(args.iterations),
        "ours_median_iter_s": ours_stats["median_iter_s"],
        "cufinufft_median_iter_s": cu_stats["median_iter_s"],
        "update_speedup": speedup(cu_stats["median_iter_s"], ours_stats["median_iter_s"]),
        "ours_cumulative_iter_s": ours_stats["cumulative_iter_s"],
        "cufinufft_cumulative_iter_s": cu_stats["cumulative_iter_s"],
        "cumulative_speedup": speedup(cu_stats["cumulative_iter_s"], ours_stats["cumulative_iter_s"]),
        "ours_final_loss_rel": ours_stats["final_loss_rel"],
        "cufinufft_final_loss_rel": cu_stats["final_loss_rel"],
        "final_loss_abs_delta": abs(ours_stats["final_loss_rel"] - cu_stats["final_loss_rel"]),
        "ours_final_object_rel_l2_aligned": ours_stats["final_object_rel_l2_aligned"],
        "cufinufft_final_object_rel_l2_aligned": cu_stats["final_object_rel_l2_aligned"],
        "final_object_rel_l2_abs_delta": abs(
            ours_stats["final_object_rel_l2_aligned"] - cu_stats["final_object_rel_l2_aligned"]
        ),
        "device_name": device_name(torch, device),
        "torch_version": package_version("torch"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cupy_version": getattr(cp, "__version__", None),
        "dtype": args.dtype,
        "pattern_kind": args.pattern_kind,
        "warm_start_noise_rel": float(args.warm_start_noise_rel),
        "step_rel": float(args.step_rel),
        "condition_seed": int(condition_seed),
        "torch_peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
        "cupy_pool_total_mib": float(cp.get_default_memory_pool().total_bytes() / (1024.0 * 1024.0)),
    }
    return ours_rows + cu_rows, summary


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["summary_rows"]
    lines = [
        "# ODT FPDT/FS-ODT pattern-intensity robustness sweep",
        "",
        "This benchmark perturbs the synthetic intensity target while keeping the reconstruction operator fixed.",
        "It tests whether detector perturbations and pattern-calibration mismatch change the numerical parity between the prepared GPU operator and cuFINUFFT GPU Plan.",
        "",
        "The data are still synthetic; this is not a real experimental reconstruction.",
        "",
        "## Results",
        "",
        "| case | model | detector perturbation | pattern mismatch | target mismatch | ours ms | cuFINUFFT ms | speedup | final loss delta | object error delta |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        detector = f"{row['detector_model']}:{fmt(row['detector_rel'], 3)}"
        pattern = f"{row['pattern_mismatch']}:{fmt(row['pattern_mismatch_rel'], 3)}"
        lines.append(
            "| {case} | {model} | {detector} | {pattern} | {mismatch} | {ours} | {cu} | {sp} | {loss_delta} | {obj_delta} |".format(
                case=row["case"],
                model=row["model"],
                detector=detector,
                pattern=pattern,
                mismatch=fmt(row["target_vs_nominal_data_rel"], 4),
                ours=fmt(1000.0 * row["ours_median_iter_s"], 4),
                cu=fmt(1000.0 * row["cufinufft_median_iter_s"], 4),
                sp=fmt(row["update_speedup"], 4),
                loss_delta=fmt(row["final_loss_abs_delta"], 4),
                obj_delta=fmt(row["final_object_rel_l2_abs_delta"], 4),
            )
        )
    speedups = [float(row["update_speedup"]) for row in rows if row.get("update_speedup") is not None]
    loss_deltas = [float(row["final_loss_abs_delta"]) for row in rows]
    object_deltas = [float(row["final_object_rel_l2_abs_delta"]) for row in rows]
    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- speedup range: `{fmt(min(speedups), 4)}x` to `{fmt(max(speedups), 4)}x`",
            f"- maximum final loss delta: `{fmt(max(loss_deltas), 4)}`",
            f"- maximum aligned-object error delta: `{fmt(max(object_deltas), 4)}`",
            "",
            "## Claim Boundary",
            "",
            "Supported by this sweep:",
            "",
            "- detector perturbations and target-pattern calibration mismatch do not break prepared-vs-cuFINUFFT numerical parity in this synthetic proxy;",
            "- the prepared GPU path keeps the main speed advantage inside perturbed nonlinear intensity update loops;",
            "- target mismatch is explicitly measured against the nominal clean operator.",
            "",
            "Still not supported:",
            "",
            "- real measured FPDT/FS-ODT reconstruction;",
            "- full ptychographic probe/position update;",
            "- convergence guarantees from arbitrary initialization.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    cache_dir = ROOT / ".matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = []
    values = []
    colors = []
    palette = {"coherent": "#1f77b4", "incoherent": "#2ca02c"}
    for row in rows:
        labels.append(
            f"{row['case']}\n{row['model']}\n{row['detector_model']}/{row['pattern_mismatch']}"
        )
        values.append(float(row["update_speedup"]))
        colors.append(palette.get(str(row["model"]), "#7f7f7f"))
    fig, ax = plt.subplots(figsize=(max(9.0, 0.72 * len(labels)), 5.0))
    ax.bar(np.arange(len(labels)), values, color=colors)
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1)
    ax.set_ylabel("cuFINUFFT update time / ours update time")
    ax.set_title("ODT pattern-intensity robustness speedup")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    detector_models = parse_str_list(args.detector_models)
    detector_rels = parse_float_list(args.detector_rels)
    pattern_mismatches = parse_str_list(args.pattern_mismatches)
    pattern_mismatch_rels = parse_float_list(args.pattern_mismatch_rels)
    models = parse_str_list(args.models)
    valid_models = {"coherent", "incoherent"}
    if set(models) - valid_models:
        raise ValueError(f"unsupported models: {sorted(set(models) - valid_models)}")
    if len(detector_models) != len(detector_rels):
        raise ValueError("detector-models and detector-rels must have the same length")
    if len(pattern_mismatches) != len(pattern_mismatch_rels):
        raise ValueError("pattern-mismatches and pattern-mismatch-rels must have the same length")
    if len(detector_models) != len(pattern_mismatches):
        raise ValueError("detector and pattern condition lists must have the same length")

    history: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for label, n_illum, n_patterns, active, cap_radial, cap_phi in args.cases:
        case = build_case(
            args,
            label=label,
            n_illum=n_illum,
            n_patterns=n_patterns,
            active=active,
            cap_radial=cap_radial,
            cap_phi=cap_phi,
        )
        for condition_index, (detector_model, detector_rel, pattern_mismatch, pattern_mismatch_rel) in enumerate(
            zip(detector_models, detector_rels, pattern_mismatches, pattern_mismatch_rels)
        ):
            for model in models:
                condition_seed = int(args.seed + 100_003 * condition_index + 9_173 * (0 if model == "coherent" else 1))
                rows, summary = run_condition(
                    args,
                    case,
                    label=label,
                    n_illum=n_illum,
                    n_patterns=n_patterns,
                    active=active,
                    cap_radial=cap_radial,
                    cap_phi=cap_phi,
                    model=model,
                    detector_model=detector_model,
                    detector_rel=detector_rel,
                    pattern_mismatch=pattern_mismatch,
                    pattern_mismatch_rel=pattern_mismatch_rel,
                    condition_seed=condition_seed,
                )
                history.extend(rows)
                summary_rows.append(summary)
    payload = {
        "config": {
            **vars(args),
            "cases": [
                {
                    "label": label,
                    "n_illum": n_illum,
                    "n_patterns": n_patterns,
                    "active": active,
                    "cap_radial": cap_radial,
                    "cap_phi": cap_phi,
                }
                for label, n_illum, n_patterns, active, cap_radial, cap_phi in args.cases
            ],
        },
        "summary_rows": summary_rows,
        "history": history,
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep synthetic detector and pattern-calibration perturbations for the ODT pattern-intensity inverse loop."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["large=96:24:12:64x256"],
        help="Cases as label=n_illum:n_patterns:active:cap_radialxcap_phi.",
    )
    parser.add_argument("--models", default="coherent,incoherent")
    parser.add_argument(
        "--detector-models",
        default="none,independent,pattern_gain,radial_background",
        help="Comma-separated perturbation families, paired with --detector-rels.",
    )
    parser.add_argument("--detector-rels", default="0,0.05,0.1,0.1")
    parser.add_argument(
        "--pattern-mismatches",
        default="none,none,amplitude,phase_amplitude",
        help="Comma-separated target-pattern mismatch families, paired with --pattern-mismatch-rels.",
    )
    parser.add_argument("--pattern-mismatch-rels", default="0,0,0.05,0.05")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    parser.add_argument("--pattern-kind", choices=["phase", "binary", "rolling"], default="phase")
    parser.add_argument("--n-beta", type=int, default=256)
    parser.add_argument("--n-r", type=int, default=12)
    parser.add_argument("--n-z", type=int, default=11)
    parser.add_argument("--r-max", type=float, default=1.0)
    parser.add_argument("--z-max", type=float, default=0.8)
    parser.add_argument("--phantom", choices=["beads", "random_beads", "shell"], default="random_beads")
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--k", type=float, default=17.307319527958313)
    parser.add_argument("--detector-na", type=float, default=0.9240924092409241)
    parser.add_argument("--illumination-angle-deg", type=float, default=49.0)
    parser.add_argument("--h-cutoff", type=int, default=None)
    parser.add_argument("--h-margin", type=int, default=20)
    parser.add_argument("--l-margin", type=int, default=18)
    parser.add_argument("--cone-l-prune-threshold", type=float, default=1e-12)
    parser.add_argument("--cpp-threads", type=int, default=16)
    parser.add_argument("--forward-execute-mode", choices=["prepared", "wrapper"], default="prepared")
    parser.add_argument("--forward-kernel-mode", choices=["compact", "partitioned"], default="partitioned")
    parser.add_argument(
        "--native-prepared-plan-mode",
        choices=["auto", "direct", "gathered", "gathered-zmajor"],
        default="auto",
    )
    parser.add_argument("--native-prepared-gather-threshold", type=int, default=8192)
    parser.add_argument("--finufft-eps", type=float, default=1e-12)
    parser.add_argument("--finufft-q-batch-size", type=int, default=1_048_576)
    parser.add_argument("--cufinufft-eps", type=float, default=1e-6)
    parser.add_argument("--cufinufft-plan-mode", choices=["plan", "simple"], default="plan")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--step-rel", type=float, default=0.0002)
    parser.add_argument("--warm-start-noise-rel", type=float, default=0.1)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_pattern_intensity_robustness",
    )
    args = parser.parse_args()
    args.cases = [parse_case(value) for value in args.cases]
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.step_rel <= 0.0:
        raise ValueError("step-rel must be positive")
    if args.warm_start_noise_rel < 0.0:
        raise ValueError("warm-start-noise-rel must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    payload = run(args)
    payload["elapsed_s"] = float(time.perf_counter() - started)
    prefix = args.output_prefix if args.output_prefix.is_absolute() else ROOT / args.output_prefix
    write_json(prefix.with_suffix(".json"), payload)
    write_csv(prefix.with_suffix(".csv"), payload["summary_rows"])
    write_csv(prefix.with_name(prefix.name + "_history.csv"), payload["history"])
    write_summary(prefix.with_suffix(".md"), payload)
    write_plot(prefix.with_suffix(".png"), payload["summary_rows"])
    print(json.dumps({"summary_rows": len(payload["summary_rows"]), "history_rows": len(payload["history"])}, indent=2))


if __name__ == "__main__":
    main()
