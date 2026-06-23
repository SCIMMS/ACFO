from __future__ import annotations

import csv
import os
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns


RESULTS = ROOT / "benchmark_results"


FONT_FAMILY = ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
MONO_FONT_FAMILY = ["Consolas", "DejaVu Sans Mono", "monospace"]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLOR_FAMILIES = {
    "blue": {"xlight": "#EAF1FE", "light": "#CEDFFE", "base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"},
    "orange": {"xlight": "#FFEDDE", "light": "#FFBDA1", "base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"},
    "olive": {"xlight": "#D8ECBD", "light": "#BEEB96", "base": "#A3D576", "mid": "#71B436", "dark": "#386411"},
    "gold": {"xlight": "#FFF4C2", "light": "#FFEA8F", "base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"},
    "pink": {"xlight": "#FCDAD6", "light": "#F5BACC", "base": "#F390CA", "mid": "#BD569B", "dark": "#8A3A6F"},
}

NEUTRAL = {
    "xlight": "#F4F5F7",
    "light": "#E2E5EA",
    "base": "#C5CAD3",
    "mid": "#7A828F",
    "dark": "#464C55",
}


def read_one_csv(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path}, got {len(rows)}")
    return rows[0]


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def ms(value_s: float) -> float:
    return 1000.0 * value_s


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "figure.edgecolor": "none",
            "savefig.facecolor": TOKENS["surface"],
            "savefig.edgecolor": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.monospace": MONO_FONT_FAMILY,
            "patch.linewidth": 1.0,
        },
    )


def add_header(fig: Any, title: str, subtitle: str) -> None:
    fig.text(
        0.055,
        0.976,
        textwrap.fill(title, 92, break_long_words=False),
        ha="left",
        va="top",
        fontsize=16,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.055,
        0.925,
        textwrap.fill(subtitle, 142, break_long_words=False),
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
    )


def label_bars(ax: Any, bars: Any, *, fmt: str = "{:.2f}") -> None:
    xmax = ax.get_xlim()[1]
    for bar in bars:
        value = float(bar.get_width())
        ax.text(
            value + xmax * 0.018,
            bar.get_y() + bar.get_height() / 2.0,
            fmt.format(value),
            ha="left",
            va="center",
            fontsize=8.5,
            color=TOKENS["ink"],
            fontfamily=MONO_FONT_FAMILY,
        )


def main() -> None:
    use_chart_theme()

    dense = read_one_csv(RESULTS / "high_na_gpu_dense_baseline_opt_representative_cyl_b8.csv")
    roi_b8 = read_one_csv(RESULTS / "high_na_cylindrical_backprop_design_opt_representative_b8.csv")
    roi_b32 = read_one_csv(RESULTS / "high_na_cylindrical_backprop_design_opt_representative_b32.csv")
    roi_m3 = read_one_csv(RESULTS / "high_na_cylindrical_backprop_design_opt_representative_b32_psi_mod.csv")

    before_pair = read_one_csv(RESULTS / "high_na_torch_gpu_vectorial_representative_b32.csv")
    after_pair = read_one_csv(RESULTS / "high_na_torch_gpu_vectorial_opt_representative_b32.csv")
    before_roi_b32 = read_one_csv(RESULTS / "high_na_cylindrical_backprop_design_representative_b32.csv")
    before_roi_m3 = read_one_csv(RESULTS / "high_na_cylindrical_backprop_design_representative_b32_psi_mod.csv")

    figure_data = [
        {"section": "same_target_pair", "case": "dense_direct", "time_ms": ms(f(dense, "dense_forward_plus_adjoint_hot_s"))},
        {"section": "same_target_pair", "case": "separable", "time_ms": ms(f(dense, "separable_forward_plus_adjoint_hot_s"))},
        {"section": "roi_backprop", "case": "annular_b8_dense", "time_ms": ms(f(roi_b8, "dense_iteration_hot_s"))},
        {"section": "roi_backprop", "case": "annular_b8_separable", "time_ms": ms(f(roi_b8, "separable_iteration_hot_s"))},
        {"section": "roi_backprop", "case": "annular_b32_dense", "time_ms": ms(f(roi_b32, "dense_iteration_hot_s"))},
        {"section": "roi_backprop", "case": "annular_b32_separable", "time_ms": ms(f(roi_b32, "separable_iteration_hot_s"))},
        {"section": "roi_backprop", "case": "m3_b32_dense", "time_ms": ms(f(roi_m3, "dense_iteration_hot_s"))},
        {"section": "roi_backprop", "case": "m3_b32_separable", "time_ms": ms(f(roi_m3, "separable_iteration_hot_s"))},
    ]
    with (RESULTS / "high_na_gpu_comparison_figure_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "case", "time_ms"])
        writer.writeheader()
        writer.writerows(figure_data)

    fig = plt.figure(figsize=(14.5, 9.8), dpi=180)
    grid = fig.add_gridspec(2, 2, left=0.070, right=0.985, top=0.790, bottom=0.095, wspace=0.39, hspace=0.58)
    ax_pair = fig.add_subplot(grid[0, 0])
    ax_roi = fig.add_subplot(grid[0, 1])
    ax_acc = fig.add_subplot(grid[1, 0])
    ax_opt = fig.add_subplot(grid[1, 1])

    add_header(
        fig,
        "Structured cylindrical High-NA GPU path wins where backprop reuses the geometry",
        "Local RTX 2070 SUPER snapshot, Torch 2.12.1+cu126, complex64. Dense direct CUDA is the same-target correctness reference; psf-generator-style dense Cartesian output is not claimed here.",
    )

    dense_color = COLOR_FAMILIES["orange"]["base"]
    dense_edge = COLOR_FAMILIES["orange"]["dark"]
    sep_color = COLOR_FAMILIES["blue"]["base"]
    sep_edge = COLOR_FAMILIES["blue"]["dark"]

    pair_labels = ["Dense direct\nCUDA", "Separable\nCUDA"]
    pair_values = [
        ms(f(dense, "dense_forward_plus_adjoint_hot_s")),
        ms(f(dense, "separable_forward_plus_adjoint_hot_s")),
    ]
    bars = ax_pair.barh(
        [1, 0],
        pair_values,
        color=[dense_color, sep_color],
        edgecolor=[dense_edge, sep_edge],
        linewidth=1.0,
    )
    ax_pair.set_yticks([1, 0], pair_labels)
    ax_pair.set_xlabel("Forward + adjoint hot time (ms)")
    ax_pair.set_title("A. Same cylindrical target, forward + adjoint", loc="left", fontsize=11, fontweight="semibold")
    ax_pair.set_xlim(0, max(pair_values) * 1.28)
    ax_pair.grid(axis="x", color=TOKENS["grid"])
    ax_pair.grid(axis="y", visible=False)
    label_bars(ax_pair, bars)
    ax_pair.text(
        pair_values[1] * 1.08,
        -0.25,
        f"{f(dense, 'speedup_dense_vs_separable_pair'):.1f}x faster",
        color=COLOR_FAMILIES["blue"]["dark"],
        fontsize=9,
        fontweight="semibold",
    )

    roi_cases = [
        ("Annular\nbatch 8", roi_b8),
        ("Annular\nbatch 32", roi_b32),
        ("Annular + m=3\nbatch 32", roi_m3),
    ]
    x = np.arange(len(roi_cases))
    width = 0.34
    dense_values = [ms(f(row, "dense_iteration_hot_s")) for _, row in roi_cases]
    sep_values = [ms(f(row, "separable_iteration_hot_s")) for _, row in roi_cases]
    ax_roi.bar(
        x - width / 2,
        dense_values,
        width=width,
        label="Dense direct",
        color=dense_color,
        edgecolor=dense_edge,
        linewidth=1.0,
    )
    ax_roi.bar(
        x + width / 2,
        sep_values,
        width=width,
        label="Separable",
        color=sep_color,
        edgecolor=sep_edge,
        linewidth=1.0,
    )
    for i, (_, row) in enumerate(roi_cases):
        ax_roi.text(
            x[i] + width / 2,
            sep_values[i] + 0.35,
            f"{f(row, 'speedup_dense_vs_separable_iteration'):.1f}x",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=COLOR_FAMILIES["blue"]["dark"],
            fontweight="semibold",
        )
    ax_roi.set_xticks(x, [label for label, _ in roi_cases])
    ax_roi.set_ylabel("Iteration time (ms)")
    ax_roi.set_title("B. Cylindrical ROI backprop objective", loc="left", fontsize=11, fontweight="semibold")
    ax_roi.legend(frameon=False, loc="upper right", ncol=2)
    ax_roi.grid(axis="y", color=TOKENS["grid"])
    ax_roi.grid(axis="x", visible=False)
    ax_roi.set_ylim(0, max(dense_values) * 1.22)

    acc_items = [
        ("Forward field L2\nsame target", f(dense, "forward_l2_dense_vs_separable")),
        ("Adjoint pupil L2\nsame target", f(dense, "adjoint_l2_dense_vs_separable")),
        ("Phase grad L2\nannular b32", f(roi_b32, "phase_gradient_l2_dense_vs_separable")),
        ("Phase grad L2\nm=3 ROI b32", f(roi_m3, "phase_gradient_l2_dense_vs_separable")),
    ]
    y = np.arange(len(acc_items))[::-1]
    values = [value for _, value in acc_items]
    ax_acc.hlines(y, 1e-7, values, color=NEUTRAL["base"], linewidth=1.0)
    ax_acc.scatter(
        values,
        y,
        s=58,
        color=COLOR_FAMILIES["olive"]["base"],
        edgecolor=COLOR_FAMILIES["olive"]["dark"],
        linewidth=1.0,
        zorder=3,
    )
    for label_y, value in zip(y, values):
        ax_acc.text(
            value * 1.18,
            label_y,
            f"{value:.1e}",
            ha="left",
            va="center",
            fontsize=8.5,
            color=TOKENS["ink"],
            fontfamily=MONO_FONT_FAMILY,
        )
    ax_acc.set_yticks(y, [label for label, _ in acc_items])
    ax_acc.set_xscale("log")
    ax_acc.set_xlim(1e-7, 8e-6)
    ax_acc.xaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    ax_acc.set_xlabel("Relative error")
    ax_acc.set_title("C. Accuracy stays in the screening range", loc="left", fontsize=11, fontweight="semibold")
    ax_acc.grid(axis="x", color=TOKENS["grid"], which="both")
    ax_acc.grid(axis="y", visible=False)

    opt_cases = [
        ("Fwd+adj\nbatch 32", ms(f(before_pair, "torch_forward_plus_adjoint_hot_s")), ms(f(after_pair, "torch_forward_plus_adjoint_hot_s"))),
        ("ROI\nbatch 32", ms(f(before_roi_b32, "separable_iteration_hot_s")), ms(f(roi_b32, "separable_iteration_hot_s"))),
        ("m=3 ROI\nbatch 32", ms(f(before_roi_m3, "separable_iteration_hot_s")), ms(f(roi_m3, "separable_iteration_hot_s"))),
    ]
    opt_y = np.arange(len(opt_cases))[::-1]
    before = [item[1] for item in opt_cases]
    after = [item[2] for item in opt_cases]
    ax_opt.hlines(opt_y, after, before, color=NEUTRAL["base"], linewidth=1.5)
    ax_opt.scatter(before, opt_y, s=56, color=NEUTRAL["light"], edgecolor=NEUTRAL["dark"], linewidth=1.0, label="Before simple pass")
    ax_opt.scatter(after, opt_y, s=62, color=COLOR_FAMILIES["blue"]["base"], edgecolor=COLOR_FAMILIES["blue"]["dark"], linewidth=1.0, label="After")
    for label_y, b, a in zip(opt_y, before, after):
        ax_opt.text(
            max(b, a) + 0.12,
            label_y,
            f"{b / a:.2f}x",
            ha="left",
            va="center",
            fontsize=8.5,
            color=COLOR_FAMILIES["blue"]["dark"],
            fontweight="semibold",
        )
    ax_opt.set_yticks(opt_y, [item[0] for item in opt_cases])
    ax_opt.set_xlabel("Separable hot time (ms)")
    ax_opt.set_title("D. Simple optimization pass mostly helps backprop", loc="left", fontsize=11, fontweight="semibold")
    ax_opt.legend(frameon=False, loc="upper right")
    ax_opt.grid(axis="x", color=TOKENS["grid"])
    ax_opt.grid(axis="y", visible=False)
    ax_opt.set_xlim(0, max(before) * 1.25)

    for ax in (ax_pair, ax_roi, ax_acc, ax_opt):
        ax.tick_params(axis="both", labelsize=8.5, colors=TOKENS["muted"])
        ax.xaxis.label.set_color(TOKENS["ink"])
        ax.yaxis.label.set_color(TOKENS["ink"])
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(TOKENS["axis"])
        sns.despine(ax=ax)

    note = (
        "Source CSVs: high_na_gpu_dense_baseline_opt_representative_cyl_b8, "
        "high_na_cylindrical_backprop_design_opt_representative_* and "
        "high_na_torch_gpu_vectorial_opt_representative_b32."
    )
    fig.text(0.055, 0.022, note, ha="left", va="bottom", fontsize=8, color=TOKENS["muted"])

    png_path = RESULTS / "high_na_gpu_comparison_figure.png"
    svg_path = RESULTS / "high_na_gpu_comparison_figure.svg"
    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path)
    print(json_dumps({"png": str(png_path), "svg": str(svg_path)}))


def json_dumps(value: dict[str, str]) -> str:
    import json

    return json.dumps(value, indent=2)


if __name__ == "__main__":
    main()
