from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_checkpoints(value: str) -> list[int]:
    checkpoints = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not checkpoints:
        raise ValueError("expected at least one checkpoint")
    if any(checkpoint <= 0 for checkpoint in checkpoints):
        raise ValueError("checkpoints must be positive")
    return sorted(set(checkpoints))


def parse_input(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path_text = value.split("=", 1)
        label = label.strip()
        path = Path(path_text.strip())
    else:
        path = Path(value)
        label = path.stem
    if not label:
        raise ValueError("input label must not be empty")
    if not path.is_absolute():
        path = ROOT / path
    return label, path


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rows_by_method(history: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for row in history:
        method = str(row["method"])
        iteration = int(row["iteration"])
        grouped.setdefault(method, {})[iteration] = row
    return grouped


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    value_f = float(value)
    if value_f == 0.0:
        return "0"
    if abs(value_f) < 1e-3 or abs(value_f) >= 1e3:
        return f"{value_f:.{digits}e}"
    return f"{value_f:.{digits}f}"


def extract_case_rows(label: str, path: Path, checkpoints: list[int]) -> list[dict[str, Any]]:
    payload = load_payload(path)
    summary = payload["summary"]
    history = payload["history"]
    grouped = rows_by_method(history)
    ours = grouped.get("ours_gpu", {})
    cufinufft = grouped.get("cufinufft_gpu", {})
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        ours_row = ours.get(checkpoint)
        cu_row = cufinufft.get(checkpoint)
        if ours_row is None or cu_row is None:
            continue
        ours_hot = float(ours_row["cumulative_iter_s"])
        cu_hot = float(cu_row["cumulative_iter_s"])
        ours_setup = float(summary["ours_setup_s"])
        cu_setup = float(summary["cufinufft_setup_s"])
        ours_total = ours_setup + ours_hot
        cu_total = cu_setup + cu_hot
        rows.append(
            {
                "case": label,
                "source_json": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                "checkpoint_updates": int(checkpoint),
                "q_samples": int(summary["total_q_samples"]),
                "object_bins": int(summary["object_bins"]),
                "cap_radial": int(summary["cap_radial"]),
                "cap_phi": int(summary["cap_phi"]),
                "ring_illum": int(summary["ring_illum"]),
                "total_illumination_count": int(summary["total_illumination_count"]),
                "ours_setup_s": ours_setup,
                "cufinufft_setup_s": cu_setup,
                "ours_hot_cumulative_s": ours_hot,
                "cufinufft_hot_cumulative_s": cu_hot,
                "ours_total_including_setup_s": ours_total,
                "cufinufft_total_including_setup_s": cu_total,
                "hot_loop_speedup_cufinufft_over_ours": ratio(cu_hot, ours_hot),
                "setup_included_speedup_cufinufft_over_ours": ratio(cu_total, ours_total),
                "ours_hot_updates_per_s": ratio(float(checkpoint), ours_hot),
                "cufinufft_hot_updates_per_s": ratio(float(checkpoint), cu_hot),
                "ours_setup_included_updates_per_s": ratio(float(checkpoint), ours_total),
                "cufinufft_setup_included_updates_per_s": ratio(float(checkpoint), cu_total),
                "ours_loss_rel": float(ours_row["loss_rel"]),
                "cufinufft_loss_rel": float(cu_row["loss_rel"]),
                "loss_rel_abs_delta": abs(float(ours_row["loss_rel"]) - float(cu_row["loss_rel"])),
                "ours_object_rel_l2": float(ours_row["object_rel_l2"]),
                "cufinufft_object_rel_l2": float(cu_row["object_rel_l2"]),
                "object_rel_l2_abs_delta": abs(
                    float(ours_row["object_rel_l2"]) - float(cu_row["object_rel_l2"])
                ),
                "cufinufft_data_rel_l2_vs_ours": float(summary["cufinufft_data_rel_l2_vs_ours"]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": config, "rows": rows}, indent=2), encoding="utf-8")


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# ODT GPU reconstruction checkpoint sweep",
        "",
        "This table extracts 1/4/8/16/32-update checkpoints from full reconstruction histories.",
        "Both operators use prepared GPU-resident paths; the `setup-included` columns add the measured operator setup time once.",
        "",
        "## Summary",
        "",
        "| case | q samples | updates | ours hot ms | cuFINUFFT hot ms | hot speedup | ours total ms | cuFINUFFT total ms | total speedup | ours loss | cu loss | loss delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {q} | {updates} | {ohot} | {chot} | {hsp} | {otot} | {ctot} | {tsp} | {oloss} | {closs} | {dloss} |".format(
                case=row["case"],
                q=row["q_samples"],
                updates=row["checkpoint_updates"],
                ohot=fmt(1000.0 * row["ours_hot_cumulative_s"], 3),
                chot=fmt(1000.0 * row["cufinufft_hot_cumulative_s"], 3),
                hsp=fmt(row["hot_loop_speedup_cufinufft_over_ours"], 3),
                otot=fmt(1000.0 * row["ours_total_including_setup_s"], 3),
                ctot=fmt(1000.0 * row["cufinufft_total_including_setup_s"], 3),
                tsp=fmt(row["setup_included_speedup_cufinufft_over_ours"], 3),
                oloss=fmt(row["ours_loss_rel"], 4),
                closs=fmt(row["cufinufft_loss_rel"], 4),
                dloss=fmt(row["loss_rel_abs_delta"], 3),
            )
        )
    if rows:
        by_case = sorted({str(row["case"]) for row in rows})
        lines.extend(["", "## Readout", ""])
        for case in by_case:
            case_rows = [row for row in rows if row["case"] == case]
            last = max(case_rows, key=lambda row: int(row["checkpoint_updates"]))
            lines.append(
                "- `{case}` reaches `{updates}` updates with hot-loop speedup `{hot}`x and setup-included speedup `{total}`x; final loss delta is `{delta}`.".format(
                    case=case,
                    updates=last["checkpoint_updates"],
                    hot=fmt(last["hot_loop_speedup_cufinufft_over_ours"], 3),
                    total=fmt(last["setup_included_speedup_cufinufft_over_ours"], 3),
                    delta=fmt(last["loss_rel_abs_delta"], 3),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The hot-loop numbers are the right metric for calibrated repeated reconstruction.",
            "- Setup-included numbers show whether the advantage survives short reconstruction bursts.",
            "- Loss deltas are measured at the same checkpoint update count and expose numerical drift from the cuFINUFFT/operator mismatch.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharex=True)
    for case in sorted({str(row["case"]) for row in rows}):
        case_rows = sorted(
            [row for row in rows if row["case"] == case],
            key=lambda row: int(row["checkpoint_updates"]),
        )
        x = [row["checkpoint_updates"] for row in case_rows]
        hot = [row["hot_loop_speedup_cufinufft_over_ours"] for row in case_rows]
        total = [row["setup_included_speedup_cufinufft_over_ours"] for row in case_rows]
        axes[0].plot(x, hot, marker="o", label=case)
        axes[1].plot(x, total, marker="o", label=case)
    for ax, title in zip(axes, ("Hot reconstruction loop", "Including one setup/build")):
        ax.axhline(1.0, color="0.5", linestyle="--", linewidth=1.0)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("reconstruction updates")
        ax.set_ylabel("cuFINUFFT time / ours time")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("ODT GPU reconstruction speedup at fixed update checkpoints")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize ODT GPU reconstruction histories at fixed update checkpoints."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input JSON files, optionally as label=path.",
    )
    parser.add_argument("--checkpoints", default="1,4,8,16,32")
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=ROOT / "benchmark_results" / "odt_reconstruction_checkpoint_sweep",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = parse_checkpoints(args.checkpoints)
    rows: list[dict[str, Any]] = []
    parsed_inputs = [parse_input(value) for value in args.inputs]
    for label, path in parsed_inputs:
        rows.extend(extract_case_rows(label, path, checkpoints))
    prefix = args.out_prefix if args.out_prefix.is_absolute() else ROOT / args.out_prefix
    config = {
        "inputs": [{"label": label, "path": str(path)} for label, path in parsed_inputs],
        "checkpoints": checkpoints,
    }
    write_csv(prefix.with_suffix(".csv"), rows)
    write_json(prefix.with_suffix(".json"), rows, config)
    write_markdown(prefix.with_suffix(".md"), rows)
    write_plot(prefix.with_suffix(".png"), rows)
    print(json.dumps({"rows": len(rows), "csv": str(prefix.with_suffix(".csv"))}, indent=2))


if __name__ == "__main__":
    main()
