from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_odt_gpu_warm_start_dynamic import (  # noqa: E402
    json_default,
    parser as dynamic_parser,
    run as run_dynamic,
    write_json,
)


def parse_float_list(value: str) -> list[float]:
    out: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    if not out:
        raise ValueError("at least one float value is required")
    return out


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            parsed = int(item)
            if parsed < 0:
                raise ValueError("update counts must be non-negative")
            out.append(parsed)
    if not out:
        raise ValueError("at least one update value is required")
    return out


def parse_str_list(value: str) -> list[str]:
    out = [item.strip() for item in value.split(",") if item.strip()]
    if not out:
        raise ValueError("at least one string value is required")
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


def plot_sweep(path: Path, rows: list[dict[str, Any]], *, target_fps: float) -> None:
    cache_dir = ROOT / ".matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    warm_rows = [row for row in rows if row["mode"] == "warm_start"]
    models = sorted(
        {
            f"{row.get('noise_model', 'independent')}/{row.get('noise_temporal_model', 'frame_independent')}"
            for row in warm_rows
        }
    )
    motions = sorted({float(row["motion_fraction"]) for row in warm_rows})
    noises = sorted({float(row["noise_rel"]) for row in warm_rows})
    width = max(5.0, 4.3 * len(motions))
    height = max(4.2, 3.3 * len(models))
    fig, axes = plt.subplots(
        len(models),
        len(motions),
        figsize=(width, height),
        sharey=True,
        constrained_layout=True,
    )
    if len(models) == 1 and len(motions) == 1:
        axes_grid = [[axes]]
    elif len(models) == 1:
        axes_grid = [list(axes)]
    elif len(motions) == 1:
        axes_grid = [[ax] for ax in axes]
    else:
        axes_grid = axes

    cmap = plt.get_cmap("viridis")
    colors = {noise: cmap(i / max(len(noises) - 1, 1)) for i, noise in enumerate(noises)}
    for model_index, model in enumerate(models):
        for motion_index, motion in enumerate(motions):
            ax = axes_grid[model_index][motion_index]
            for noise in noises:
                group = [
                    row
                    for row in warm_rows
                    if f"{row.get('noise_model', 'independent')}/{row.get('noise_temporal_model', 'frame_independent')}" == model
                    and math.isclose(float(row["motion_fraction"]), motion)
                    and math.isclose(float(row["noise_rel"]), noise)
                ]
                group.sort(key=lambda row: int(row["updates"]))
                if not group:
                    continue
                ax.plot(
                    [int(row["updates"]) for row in group],
                    [float(row["mean_object_rel_l2"]) for row in group],
                    marker="o",
                    linewidth=1.6,
                    markersize=4,
                    color=colors[noise],
                    label=f"noise {noise:g}",
                )
            title = f"motion {motion:g}" if model_index == 0 else ""
            if motion_index == 0:
                title = f"{model}\n{title}".strip()
            ax.set_title(title)
            ax.set_xlabel("updates/frame")
            ax.grid(True, alpha=0.25)
            if target_fps > 0.0:
                budget_ms = 1000.0 / float(target_fps)
                feasible = sorted(
                    {
                        int(row["updates"])
                        for row in warm_rows
                        if f"{row.get('noise_model', 'independent')}/{row.get('noise_temporal_model', 'frame_independent')}" == model
                        and math.isclose(float(row["motion_fraction"]), motion)
                        and 1000.0 * float(row["median_update_s"]) <= budget_ms
                    }
                )
                if feasible:
                    ax.axvspan(0.5, max(feasible) + 0.5, color="#4c78a8", alpha=0.08)
            if motion_index == 0:
                ax.set_ylabel("mean object rel-L2")
    axes_grid[0][-1].legend(fontsize=8, loc="best")
    fig.suptitle("ODT GPU warm-start correlated-noise update error sweep")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary_markdown(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    warm_rows = [row for row in rows if row["mode"] == "warm_start"]
    lines = [
        "# ODT GPU noise/motion update-error sweep",
        "",
        "This sweep uses the same ring-plus-axis ODT geometry and changes only the synthetic object motion amplitude, the measurement perturbation family, and the relative perturbation norm.",
        "",
        "## Configuration",
        "",
        f"- frames: `{config['frames']}`",
        f"- updates/frame: `{config['updates_per_frame']}`",
        f"- noise models: `{config['noise_models']}`",
        f"- noise temporal models: `{config['noise_temporal_models']}`",
        f"- motion fractions: `{config['motion_fractions']}`",
        f"- noise rel-L2 values: `{config['noise_rel_values']}`",
        f"- synthetic independent noise rel-L2: `{config['synthetic_noise_rel']}`",
        f"- target FPS: `{config['target_fps']}`",
        f"- initial iterations: `{config['initial_iterations']}`",
        f"- reference iterations per case: `{config['reference_iterations']}`",
        "",
        "## Warm-Start Error Surface",
        "",
        "| noise model | temporal model | motion | drift rel | synthetic noise rel | updates/frame | median latency ms | FPS | mean object rel-L2 | final object rel-L2 | mean loss rel |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in warm_rows:
        lines.append(
            "| "
            f"{row.get('noise_model', 'independent')} | "
            f"{row.get('noise_temporal_model', 'frame_independent')} | "
            f"{float(row['motion_fraction']):.4g} | "
            f"{float(row['noise_rel']):.4g} | "
            f"{float(row.get('synthetic_noise_rel', 0.0)):.4g} | "
            f"{int(row['updates'])} | "
            f"{1000.0 * float(row['median_update_s']):.2f} | "
            f"{float(row['median_fps_excluding_synthetic_data']):.1f} | "
            f"{float(row['mean_object_rel_l2']):.4g} | "
            f"{float(row['final_object_rel_l2']):.4g} | "
            f"{float(row['mean_loss_rel']):.4g} |"
        )

    target_fps = float(config["target_fps"])
    if target_fps > 0.0:
        budget_s = 1.0 / target_fps
        lines.extend(["", f"## Best Within {target_fps:g} FPS Budget", ""])
        lines.append("| noise model | temporal model | motion | drift rel | synthetic noise rel | best updates | median latency ms | mean object rel-L2 |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        keys = sorted(
            {
                (
                    str(row.get("noise_model", "independent")),
                    str(row.get("noise_temporal_model", "frame_independent")),
                    float(row["motion_fraction"]),
                    float(row["noise_rel"]),
                    float(row.get("synthetic_noise_rel", 0.0)),
                )
                for row in warm_rows
            }
        )
        for model, temporal, motion, noise, synthetic_noise in keys:
            feasible = [
                row
                for row in warm_rows
                if str(row.get("noise_model", "independent")) == model
                and str(row.get("noise_temporal_model", "frame_independent")) == temporal
                and math.isclose(float(row["motion_fraction"]), motion)
                and math.isclose(float(row["noise_rel"]), noise)
                and math.isclose(float(row.get("synthetic_noise_rel", 0.0)), synthetic_noise)
                and float(row["median_update_s"]) <= budget_s
            ]
            if not feasible:
                continue
            best = min(feasible, key=lambda row: float(row["mean_object_rel_l2"]))
            lines.append(
                "| "
                f"{model} | "
                f"{temporal} | "
                f"{motion:.4g} | "
                f"{noise:.4g} | "
                f"{synthetic_noise:.4g} | "
                f"{int(best['updates'])} | "
                f"{1000.0 * float(best['median_update_s']):.2f} | "
                f"{float(best['mean_object_rel_l2']):.4g} |"
            )

    if config.get("figure"):
        lines.extend(["", f"- figure: `{config['figure']}`"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_dynamic_args(
    args: argparse.Namespace,
    *,
    motion: float,
    noise: float,
    noise_model: str,
    noise_temporal_model: str,
) -> argparse.Namespace:
    dyn = dynamic_parser().parse_args([])
    for name in (
        "device",
        "dtype",
        "n_beta",
        "n_r",
        "n_z",
        "r_max",
        "z_max",
        "phantom",
        "object_scale",
        "seed",
        "k",
        "detector_na",
        "illumination_angle_deg",
        "ring_illum",
        "skip_axis_illumination",
        "cap_radial",
        "cap_phi",
        "h_cutoff",
        "h_margin",
        "l_margin",
        "cone_l_prune_threshold",
        "cpp_threads",
        "forward_execute_mode",
        "forward_kernel_mode",
        "native_prepared_plan_mode",
        "native_prepared_gather_threshold",
        "frames",
        "target_fps",
        "updates_per_frame",
        "initial_mode",
        "initial_iterations",
        "warmup_updates",
        "reference_iterations",
        "include_cold_start",
        "phase_drift_rad",
        "finufft_eps",
        "finufft_q_batch_size",
        "noise_seed",
        "synthetic_noise_rel",
        "synthetic_noise_seed",
    ):
        setattr(dyn, name, getattr(args, name))
    dyn.motion_fraction = float(motion)
    dyn.noise_rel = float(noise)
    dyn.noise_model = str(noise_model)
    dyn.noise_temporal_model = str(noise_temporal_model)
    case = f"{noise_model}_{noise_temporal_model}_m{slug_float(motion)}_n{slug_float(noise)}"
    dyn.out = args.output_dir / f"{args.prefix}_{case}.json"
    dyn.csv = args.output_dir / f"{args.prefix}_{case}_history.csv"
    dyn.figure = None
    dyn.summary_md = None
    return dyn


def run(args: argparse.Namespace) -> dict[str, Any]:
    motions = parse_float_list(args.motion_fractions)
    noises = parse_float_list(args.noise_rel_values)
    noise_models = parse_str_list(args.noise_models)
    noise_temporal_models = parse_str_list(args.noise_temporal_models)
    valid_noise_models = {
        "independent",
        "global_gain",
        "illumination_gain",
        "radial_background",
        "phase_ramp",
    }
    invalid = sorted(set(noise_models) - valid_noise_models)
    if invalid:
        raise ValueError(f"unsupported noise models: {invalid}")
    valid_temporal_models = {"frame_independent", "static", "smooth"}
    invalid_temporal = sorted(set(noise_temporal_models) - valid_temporal_models)
    if invalid_temporal:
        raise ValueError(f"unsupported noise temporal models: {invalid_temporal}")
    parse_int_list(args.updates_per_frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    for noise_model in noise_models:
        for noise_temporal_model in noise_temporal_models:
            for motion in motions:
                for noise in noises:
                    dyn_args = make_dynamic_args(
                        args,
                        motion=motion,
                        noise=noise,
                        noise_model=noise_model,
                        noise_temporal_model=noise_temporal_model,
                    )
                    summary = run_dynamic(dyn_args)
                    case_summaries.append(summary)
                    for summary_row in summary["summary_rows"]:
                        row = dict(summary_row)
                        row["noise_model"] = str(noise_model)
                        row["noise_temporal_model"] = str(noise_temporal_model)
                        row["motion_fraction"] = float(motion)
                        row["noise_rel"] = float(noise)
                        row["synthetic_noise_rel"] = float(args.synthetic_noise_rel)
                        row["frames"] = int(summary_row["frames"])
                        row["case_json"] = str(dyn_args.out)
                        row["case_history_csv"] = str(dyn_args.csv)
                        rows.append(row)

    figure = args.figure
    if figure:
        plot_sweep(figure, rows, target_fps=float(args.target_fps))
    config = vars(args).copy()
    config["noise_models"] = noise_models
    config["noise_temporal_models"] = noise_temporal_models
    config["motion_fractions"] = motions
    config["noise_rel_values"] = noises
    config["figure"] = str(figure) if figure else None
    payload = {"config": config, "rows": rows, "case_summaries": case_summaries}
    write_csv(args.csv, rows)
    write_json(args.out, payload)
    if args.summary_md:
        write_summary_markdown(args.summary_md, rows=rows, config=config)
    return {"config": config, "rows": rows}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sweep ODT GPU warm-start error over motion and measurement noise.")
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
    p.add_argument("--updates-per-frame", default="1,2,3,5,8")
    p.add_argument("--initial-mode", choices=["cold_start", "oracle"], default="cold_start")
    p.add_argument("--initial-iterations", type=int, default=10)
    p.add_argument("--warmup-updates", type=int, default=1)
    p.add_argument("--reference-iterations", type=int, default=0)
    p.add_argument("--include-cold-start", action="store_true")
    p.add_argument("--motion-fractions", default="0.02,0.08,0.16")
    p.add_argument("--phase-drift-rad", type=float, default=0.12)
    p.add_argument(
        "--noise-models",
        default="independent,illumination_gain,radial_background,phase_ramp",
        help="Comma-separated noise models: independent, global_gain, illumination_gain, radial_background, phase_ramp.",
    )
    p.add_argument(
        "--noise-temporal-models",
        default="frame_independent",
        help="Comma-separated temporal models: frame_independent, static, smooth.",
    )
    p.add_argument("--noise-rel-values", default="0,0.01,0.03,0.1")
    p.add_argument("--noise-seed", type=int, default=12345)
    p.add_argument(
        "--synthetic-noise-rel",
        type=float,
        default=0.0,
        help="Additional frame-independent complex Gaussian detector/read noise normalized to clean data.",
    )
    p.add_argument("--synthetic-noise-seed", type=int, default=54321)
    p.add_argument("--finufft-eps", type=float, default=1e-12)
    p.add_argument("--finufft-q-batch-size", type=int, default=1_048_576)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_noise_motion_update_sweep_cases",
    )
    p.add_argument(
        "--prefix",
        default="odt_noise_motion_update",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_noise_motion_update_sweep.json",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_noise_motion_update_sweep.csv",
    )
    p.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_noise_motion_update_sweep.png",
    )
    p.add_argument(
        "--summary-md",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_noise_motion_update_sweep_summary.md",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    summary = run(args)
    compact = {
        "config": summary["config"],
        "row_count": len(summary["rows"]),
    }
    print(json.dumps(compact, indent=2, default=json_default))


if __name__ == "__main__":
    main()
