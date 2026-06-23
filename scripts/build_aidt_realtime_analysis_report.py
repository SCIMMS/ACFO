from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile_limits(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> tuple[float, float]:
    return float(np.nanpercentile(arr, lo)), float(np.nanpercentile(arr, hi))


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.08,
        1.07,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=16,
        fontweight="bold",
    )


def build_figure(*, out_png: Path, out_svg: Path) -> None:
    contract = np.load(ROOT / "benchmark_results" / "aidt_diatom_public_contract.npz")
    source_xy = np.asarray(contract["source_na_xy"], dtype=float)
    objective_na = float(contract["objective_na"])

    recon = np.load(ROOT / "benchmark_results" / "aidt_public_transfer_reconstruction_256.npz")
    n_re = np.asarray(recon["n_re"])
    n_im = np.asarray(recon["n_im"])
    z = np.asarray(recon["depth_values_um"])
    z0 = int(np.argmin(np.abs(z)))
    n_re_slice = n_re[:, :, z0]
    n_im_projection = np.max(np.abs(n_im), axis=2)

    full = load_json(ROOT / "benchmark_results" / "aidt_10hz_full700_opt_repeat.json")["summary"]
    timing = {
        "H2D copy": 0.009735499988892116,
        "FFT": float(full["gpu_fft_median_s"]),
        "RHS": float(full["gpu_rhs_median_s"]),
        "Solve": float(full["gpu_solve_median_s"]),
    }
    core_s = float(full["gpu_run_median_s"])
    copy_core_s = 0.1065680000174325

    speedups = np.array([1.0, 1.05, 1.075, 1.10, 1.25, 1.50, 2.0, 3.0])
    projected_s = timing["H2D copy"] + core_s / speedups
    projected_hz = 1.0 / projected_s

    fig = plt.figure(figsize=(12.0, 8.6), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.2], height_ratios=[1.0, 1.0])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_aspect("equal")
    theta = np.linspace(0.0, 2.0 * np.pi, 360)
    ax_a.plot(objective_na * np.cos(theta), objective_na * np.sin(theta), color="#4d4d4d", lw=1.3)
    ax_a.scatter(source_xy[:, 0], source_xy[:, 1], s=44, c="#0072B2", edgecolor="white", lw=0.7, zorder=3)
    ax_a.scatter([0], [0], s=40, c="#D55E00", marker="x", lw=1.8, zorder=4)
    ax_a.set_xlim(-0.82, 0.82)
    ax_a.set_ylim(-0.82, 0.82)
    ax_a.set_xlabel("source NA x")
    ax_a.set_ylabel("source NA y")
    ax_a.set_title("Measured annular aIDT geometry")
    ax_a.grid(alpha=0.25)
    ax_a.text(
        0.03,
        0.04,
        "24 illuminations\n700 x 700 detector\n35 z slices",
        transform=ax_a.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bdbdbd", alpha=0.92),
    )
    add_panel_label(ax_a, "A")

    gs_b = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.08)
    ax_b1 = fig.add_subplot(gs_b[0, 0])
    ax_b2 = fig.add_subplot(gs_b[0, 1])
    vmin, vmax = percentile_limits(n_re_slice, 0.5, 99.5)
    im1 = ax_b1.imshow(n_re_slice, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    ax_b1.set_title(f"n_re, z={z[z0]:.1f} um")
    ax_b1.set_xticks([])
    ax_b1.set_yticks([])
    cbar1 = fig.colorbar(im1, ax=ax_b1, fraction=0.046, pad=0.02)
    cbar1.ax.tick_params(labelsize=8)
    vmin2, vmax2 = percentile_limits(n_im_projection, 0.5, 99.7)
    im2 = ax_b2.imshow(n_im_projection, cmap="magma", vmin=vmin2, vmax=vmax2, origin="lower")
    ax_b2.set_title("max |n_im| projection")
    ax_b2.set_xticks([])
    ax_b2.set_yticks([])
    cbar2 = fig.colorbar(im2, ax=ax_b2, fraction=0.046, pad=0.02)
    cbar2.ax.tick_params(labelsize=8)
    add_panel_label(ax_b1, "B")

    ax_c = fig.add_subplot(gs[1, 0])
    colors_stage = {
        "H2D copy": "#CC79A7",
        "FFT": "#56B4E9",
        "RHS": "#E69F00",
        "Solve": "#009E73",
    }
    y_positions = [1, 0]
    labels = ["copy + core", "GPU-resident core"]
    left = 0.0
    for name in ["H2D copy", "FFT", "RHS", "Solve"]:
        val = timing[name] * 1000.0
        ax_c.barh(y_positions[0], val, left=left, color=colors_stage[name], label=name)
        left += val
    left = 0.0
    for name in ["FFT", "RHS", "Solve"]:
        val = timing[name] * 1000.0
        ax_c.barh(y_positions[1], val, left=left, color=colors_stage[name])
        left += val
    ax_c.axvline(100.0, color="#333333", lw=1.2, ls="--")
    ax_c.text(100.8, 1.28, "10 Hz budget", fontsize=9, va="center")
    ax_c.set_yticks(y_positions, labels)
    ax_c.set_xlim(0, 116)
    ax_c.set_xlabel("milliseconds per update")
    ax_c.set_title("Full 700 x 700 x 35 timing budget")
    ax_c.legend(loc="lower right", frameon=False, fontsize=8)
    ax_c.text(101.0, 1.0, f"{copy_core_s * 1000:.1f} ms", fontsize=9, va="center")
    ax_c.text(core_s * 1000 + 1.0, 0.0, f"{core_s * 1000:.1f} ms", fontsize=9, va="center")
    add_panel_label(ax_c, "C")

    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.plot(speedups, projected_hz, marker="o", color="#0072B2", lw=2.0)
    ax_d.axhline(10.0, color="#333333", lw=1.2, ls="--")
    ax_d.axvline(1.075, color="#D55E00", lw=1.2, ls=":")
    ax_d.scatter([1.075], [10.0], s=70, color="#D55E00", zorder=5)
    ax_d.set_xlabel("GPU core speedup vs RTX 2070 SUPER")
    ax_d.set_ylabel("sequential copy + core rate (Hz)")
    ax_d.set_title("Projection from measured full condition")
    ax_d.set_xlim(0.95, 3.05)
    ax_d.set_ylim(8.5, 25.0)
    ax_d.grid(alpha=0.25)
    ax_d.text(
        1.13,
        10.6,
        "1.075x compute speedup\nreaches 10 Hz with H2D copy",
        fontsize=9,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bdbdbd", alpha=0.92),
    )
    ax_d.text(
        1.78,
        20.5,
        "Conservative RHS-only\nmulti-GPU projection:\n2 GPUs -> 11.98 Hz",
        fontsize=9,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bdbdbd", alpha=0.92),
    )
    add_panel_label(ax_d, "D")

    fig.suptitle("ODT/aIDT real-time processing feasibility", fontsize=17, fontweight="bold")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_svg)
    plt.close(fig)


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def build_pdf(*, figure_png: Path, out_pdf: Path) -> None:
    styles = getSampleStyleSheet()
    title = styles["Title"]
    heading = styles["Heading2"]
    normal = styles["BodyText"]
    normal.fontSize = 9.5
    normal.leading = 12.0
    small = ParagraphStyle("Small", parent=normal, fontSize=8.2, leading=10.0, alignment=TA_LEFT)
    table_cell = ParagraphStyle("TableCell", parent=normal, fontSize=7.9, leading=9.2, spaceAfter=0)
    table_header = ParagraphStyle(
        "TableHeader", parent=table_cell, fontName="Helvetica-Bold", textColor=colors.HexColor("#111111")
    )

    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=letter,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )
    story = [
        Paragraph("ODT/aIDT Benchmark Freeze: Real-Time Processing Feasibility", title),
        Spacer(1, 0.12 * inch),
        paragraph(
            "Scope: this is not an end-to-end live microscope demonstration. "
            "It freezes the ODT/aIDT benchmark evidence showing that the processing-side "
            "prepared GPU operator has entered the real-time regime on realistic public measured geometry.",
            normal,
        ),
        Spacer(1, 0.15 * inch),
        Image(str(figure_png), width=7.3 * inch, height=5.23 * inch),
        Spacer(1, 0.08 * inch),
        paragraph(
            "Figure. A, public annular aIDT measured geometry. B, reconstruction from the public Diatom I data. "
            "C, full-condition timing budget on RTX 2070 SUPER. D, measured-speedup projection for real-time feasibility.",
            small,
        ),
        PageBreak(),
        Paragraph("Frozen Main-Text Numbers", heading),
        Spacer(1, 0.08 * inch),
    ]

    raw_table_data = [
        ["Condition", "Seconds/update", "Hz", "Use in manuscript"],
        ["700 x 700 x 35, GPU-resident core", "0.0970", "10.31", "Processing core crosses 10 Hz"],
        ["700 x 700 x 35, H2D copy + core", "0.1066", "9.38", "Near real time on old GPU"],
        ["700 x 700 x 18, H2D copy + core", "0.0625", "15.99", "Stable copy-included 10 Hz regime"],
        ["512 x 512 x 35, H2D copy + core", "0.0556", "18.00", "Stable detector-crop 10 Hz regime"],
        ["Full 700 x 700 x 35 speedup needed for 10 Hz", "1.075x", "-", "Projection boundary"],
        ["RHS-only 2-GPU projection", "0.0835", "11.98", "Conservative multi-GPU feasibility"],
    ]
    table_data = [
        [paragraph(str(value), table_header if row_index == 0 else table_cell) for value in row]
        for row_index, row in enumerate(raw_table_data)
    ]
    table = Table(table_data, colWidths=[2.5 * inch, 1.02 * inch, 0.55 * inch, 2.85 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111111")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.1),
                ("LEADING", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8C8C8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFAFA")]),
            ]
        )
    )
    story.extend(
        [
            table,
            Spacer(1, 0.18 * inch),
            Paragraph("Recommended Claim Boundary", heading),
            paragraph(
                "For realistic measured aIDT geometry, the prepared GPU operator makes real-time analysis practical "
                "on the processing side. On an older RTX 2070 SUPER, the full-condition GPU-resident update already "
                "reaches 10 Hz, and the copy-included path is within a 1.075x compute speedup of 10 Hz. "
                "The claim should not be phrased as a completed end-to-end live microscope demonstration until "
                "acquisition, preprocessing, and transfer scheduling are measured in the target system.",
                normal,
            ),
            Spacer(1, 0.16 * inch),
            Paragraph("Why ODT Benchmarking Can Stop Here", heading),
            paragraph(
                "The benchmark now satisfies the evidence needed for the ODT application: real public data and geometry, "
                "a realistic full detector/depth condition, measured timing that reaches or nearly reaches 10 Hz, "
                "stable relaxed conditions that exceed 10 Hz even with host-to-GPU transfer, and a conservative projection "
                "showing that current GPUs, overlapped transfer, or modest multi-GPU partitioning should close the remaining gap.",
                normal,
            ),
            Spacer(1, 0.16 * inch),
            Paragraph("References Used for Acquisition-Hardware Boundary", heading),
            paragraph(
                "Allahgholi et al., The Adaptive Gain Integrating Pixel Detector at the European XFEL, arXiv:1808.00256. "
                "Munnich et al., Integrated Detector Control and Calibration Processing at the European XFEL, arXiv:1601.01794.",
                small,
            ),
            Spacer(1, 0.16 * inch),
            Paragraph("Local Source Artifacts", heading),
            paragraph(
                "Figure: benchmark_results/aidt_realtime_analysis_figure.png<br/>"
                "Projection: benchmark_results/aidt_realtime_projection_summary.md<br/>"
                "10 Hz condition summary: benchmark_results/aidt_10hz_condition_summary.md",
                small,
            ),
        ]
    )
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figure-png",
        type=Path,
        default=ROOT / "benchmark_results" / "aidt_realtime_analysis_figure.png",
    )
    parser.add_argument(
        "--figure-svg",
        type=Path,
        default=ROOT / "benchmark_results" / "aidt_realtime_analysis_figure.svg",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=ROOT / "benchmark_results" / "aidt_realtime_analysis_report.pdf",
    )
    args = parser.parse_args()
    build_figure(out_png=args.figure_png, out_svg=args.figure_svg)
    build_pdf(figure_png=args.figure_png, out_pdf=args.pdf)
    print(args.figure_png)
    print(args.figure_svg)
    print(args.pdf)


if __name__ == "__main__":
    main()
