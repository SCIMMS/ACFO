from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
(ROOT / ".matplotlib_cache").mkdir(exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, str):
        return value
    value_f = float(value)
    if value_f == 0.0:
        return "0"
    if abs(value_f) < 1e-3 or abs(value_f) >= 1e3:
        return f"{value_f:.{digits}e}"
    return f"{value_f:.{digits}f}"


def pyfocus_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    config = payload["config"]
    rows: list[dict[str, Any]] = []
    for row in payload["rows"]:
        rows.append(
            {
                "lane": "pyfocus_cpu_xy",
                "baseline": "PyCustomFocus/PyFocus",
                "case": row["case"],
                "device": "cpu",
                "package_version": config["pycustomfocus_version"],
                "matched_accuracy": "shape_only",
                "batch": 1,
                "baseline_targets": row["cartesian_targets"],
                "ours_targets": row["separable_polar_targets"],
                "shape_l2": row["intensity_shape_l2_pyfocus_vs_ours_direct"],
                "correlation": row["intensity_pearson_pyfocus_vs_ours_direct"],
                "peak_radius_offset_nm": row["peak_radius_offset_nm"],
                "baseline_s": row["pyfocus_s"],
                "ours_direct_s": row["ours_cartesian_direct_s"],
                "ours_hot_s": row["ours_separable_hot_s"],
                "ours_one_shot_s": row["ours_separable_one_shot_s"],
                "speedup_direct_vs_baseline": row["speedup_direct_vs_pyfocus"],
                "speedup_hot_vs_baseline": row["speedup_separable_hot_vs_pyfocus"],
                "speedup_one_shot_vs_baseline": row["speedup_separable_one_shot_vs_pyfocus"],
                "note": "PyFocus native Cartesian XY intensity shape; local separable timing uses equal-size polar grid",
            }
        )
    return rows


def psf_generator_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        for row in payload["rows"]:
            if row.get("status") != "ok":
                rows.append(
                    {
                        "lane": "psf_generator_gpu_cartesian",
                        "baseline": "psf-generator",
                        "case": "skipped",
                        "device": "n/a",
                        "package_version": "n/a",
                        "matched_accuracy": "no",
                        "batch": "",
                        "baseline_targets": "",
                        "ours_targets": "",
                        "shape_l2": "",
                        "correlation": "",
                        "peak_radius_offset_nm": "",
                        "baseline_s": "",
                        "ours_direct_s": "",
                        "ours_hot_s": "",
                        "ours_one_shot_s": "",
                        "speedup_direct_vs_baseline": "",
                        "speedup_hot_vs_baseline": "",
                        "speedup_one_shot_vs_baseline": "",
                        "note": row.get("skip_reason", "skipped"),
                    }
                )
                continue
            rows.append(
                {
                    "lane": "psf_generator_gpu_cartesian",
                    "baseline": "psf-generator VectorialCartesianPropagator",
                    "case": f"batch_{row['batch_size_ours']}",
                    "device": row["device_name"],
                    "package_version": row["psf_generator_version"],
                    "matched_accuracy": "timing_only",
                    "batch": row["batch_size_ours"],
                    "baseline_targets": row["psf_targets"],
                    "ours_targets": row["ours_targets_per_mask"],
                    "shape_l2": "",
                    "correlation": "",
                    "peak_radius_offset_nm": "",
                    "baseline_s": row["psf_hot_s"],
                    "ours_direct_s": "",
                    "ours_hot_s": row["ours_hot_s"],
                    "ours_one_shot_s": "",
                    "speedup_direct_vs_baseline": "",
                    "speedup_hot_vs_baseline": row[
                        "wall_time_ratio_psf_generator_cartesian_to_ours_structured"
                    ],
                    "speedup_one_shot_vs_baseline": "",
                    "note": "Timing only; psf-generator uses dense Cartesian stack and the local path uses structured cylindrical output",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "lane",
        "baseline",
        "case",
        "device",
        "package_version",
        "matched_accuracy",
        "batch",
        "baseline_targets",
        "ours_targets",
        "shape_l2",
        "correlation",
        "peak_radius_offset_nm",
        "baseline_s",
        "ours_direct_s",
        "ours_hot_s",
        "ours_one_shot_s",
        "speedup_direct_vs_baseline",
        "speedup_hot_vs_baseline",
        "speedup_one_shot_vs_baseline",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    pyfocus = [row for row in rows if row["lane"] == "pyfocus_cpu_xy"]
    psf = [row for row in rows if row["lane"] == "psf_generator_gpu_cartesian"]
    max_l2 = max(float(row["shape_l2"]) for row in pyfocus)
    min_corr = min(float(row["correlation"]) for row in pyfocus)
    min_hot = min(float(row["speedup_hot_vs_baseline"]) for row in pyfocus)
    max_hot = max(float(row["speedup_hot_vs_baseline"]) for row in pyfocus)
    min_one = min(float(row["speedup_one_shot_vs_baseline"]) for row in pyfocus)
    max_one = max(float(row["speedup_one_shot_vs_baseline"]) for row in pyfocus)
    psf_ok = [row for row in psf if row["case"] != "skipped"]

    lines = [
        "# High-NA external package comparison",
        "",
        "This rollup separates two external-package checks:",
        "",
        "- PyFocus/PyCustomFocus validates vectorial focal-pattern shape on its native Cartesian XY plane.",
        "- psf-generator gives a same-GPU vectorial package timing anchor, but not a matched accuracy row.",
        "",
        "## PyFocus shape comparison",
        "",
        "| case | shape L2 | corr | peak-radius offset nm | PyFocus s | ours direct s | ours hot speedup | ours one-shot speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pyfocus:
        lines.append(
            "| {case} | {l2} | {corr} | {radius} | {base} | {direct} | {hot}x | {one}x |".format(
                case=row["case"],
                l2=fmt(row["shape_l2"]),
                corr=fmt(row["correlation"], 6),
                radius=fmt(row["peak_radius_offset_nm"]),
                base=fmt(row["baseline_s"]),
                direct=fmt(row["ours_direct_s"]),
                hot=fmt(row["speedup_hot_vs_baseline"], 2),
                one=fmt(row["speedup_one_shot_vs_baseline"], 2),
            )
        )
    lines.extend(
        [
            "",
            "Readout:",
            "",
            f"- PyFocus intensity-shape agreement is strong in this snapshot: max scale-fit L2 `{max_l2:.3e}`, min correlation `{min_corr:.9f}`.",
            f"- The local direct Cartesian reference is slower than PyFocus on this tiny grid, so this is not a direct-loop speed claim.",
            f"- The local separable structured-grid path is `{min_hot:.1f}x` to `{max_hot:.1f}x` faster in the hot loop and `{min_one:.1f}x` to `{max_one:.1f}x` faster including plan build.",
            "- The PyFocus comparison supports vectorial formulation and pattern agreement; it does not prove a full Cartesian package replacement.",
            "",
            "## psf-generator GPU timing anchor",
            "",
            "| batch | device | psf targets | ours targets/mask | psf hot s | ours hot s | wall-time ratio |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if not psf_ok:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    for row in psf_ok:
        lines.append(
            "| {batch} | {device} | {base_targets} | {ours_targets} | {base} | {ours} | {ratio}x |".format(
                batch=row["batch"],
                device=row["device"],
                base_targets=row["baseline_targets"],
                ours_targets=row["ours_targets"],
                base=fmt(row["baseline_s"]),
                ours=fmt(row["ours_hot_s"]),
                ratio=fmt(row["speedup_hot_vs_baseline"], 2),
            )
        )
    lines.extend(
        [
            "",
            "Readout:",
            "",
            "- psf-generator is a serious PyTorch/CUDA vectorial optics baseline, but this row is timing-only because coordinates, pupil semantics, and batching semantics are not yet matched.",
            "- The current local advantage is clearest for repeated masks or coherent modes on a structured cylindrical ROI.",
            "- Unsupported claim remains: faster than optimized full dense-Cartesian FFT-Debye or PSF package evaluation for the same Cartesian focal volume.",
            "",
            "## Next matched baseline",
            "",
            "The next validation step is a dense-Cartesian adapter where both paths use the same pupil, normalization, target coordinates, defocus stack, and batching convention.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def save_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    pyfocus = [row for row in rows if row["lane"] == "pyfocus_cpu_xy"]
    psf = [
        row
        for row in rows
        if row["lane"] == "psf_generator_gpu_cartesian" and row["case"] != "skipped"
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)

    cases = [row["case"] for row in pyfocus]
    x = np.arange(len(cases))
    axes[0].bar(x, [float(row["shape_l2"]) for row in pyfocus], color="#4c78a8")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, cases, rotation=20, ha="right")
    axes[0].set_title("PyFocus shape error")
    axes[0].set_ylabel("scale-fit L2")

    width = 0.36
    hot = [float(row["speedup_hot_vs_baseline"]) for row in pyfocus]
    one = [float(row["speedup_one_shot_vs_baseline"]) for row in pyfocus]
    axes[1].bar(x - width / 2, hot, width, label="hot", color="#54a24b")
    axes[1].bar(x + width / 2, one, width, label="one-shot", color="#e45756")
    axes[1].set_xticks(x, cases, rotation=20, ha="right")
    axes[1].set_title("Structured vs PyFocus")
    axes[1].set_ylabel("speedup")
    axes[1].legend(frameon=False)

    if psf:
        batches = [str(row["batch"]) for row in psf]
        axes[2].bar(
            np.arange(len(psf)),
            [float(row["speedup_hot_vs_baseline"]) for row in psf],
            color="#f58518",
        )
        axes[2].set_xticks(np.arange(len(psf)), batches)
    axes[2].set_title("psf-generator timing anchor")
    axes[2].set_xlabel("local batch")
    axes[2].set_ylabel("wall-time ratio")

    fig.suptitle("High-NA external package comparison")
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a PyFocus and psf-generator external package comparison rollup."
    )
    parser.add_argument(
        "--pyfocus-json",
        default="benchmark_results/high_na_pyfocus_vectorial_package.json",
    )
    parser.add_argument(
        "--psf-json",
        nargs="*",
        default=[
            "benchmark_results/high_na_psf_generator_baseline_representative.json",
            "benchmark_results/high_na_psf_generator_baseline_representative_b8.json",
            "benchmark_results/high_na_psf_generator_baseline_representative_b32.json",
        ],
    )
    parser.add_argument(
        "--output-prefix",
        default="benchmark_results/high_na_external_package_comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pyfocus_payload = load_json(ROOT / args.pyfocus_json)
    psf_payloads = [load_json(ROOT / path) for path in args.psf_json]
    rows = pyfocus_rows(pyfocus_payload) + psf_generator_rows(psf_payloads)
    output_prefix = ROOT / args.output_prefix
    write_csv(output_prefix.with_suffix(".csv"), rows)
    write_summary(output_prefix.with_suffix(".md"), rows)
    save_plot(output_prefix.with_suffix(".png"), rows)
    print(
        json.dumps(
            {
                "output_prefix": str(output_prefix),
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
