from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import fitz
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib_cache"),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results"
DOCS = ROOT / "docs"
SUPPORT = DOCS / "acfo_ncs_execution_update_support"
PDF = DOCS / "ACFO_NCS_validation_execution_update_ko.pdf"
NOTES = DOCS / "acfo_ncs_validation_execution_update_source_notes.md"
RECEIPT = SUPPORT / "build_receipt.json"

BLUE = "#2458A6"
GOLD = "#D99B2B"
ORANGE = "#D66A35"
INK = "#1F2937"
GREY = "#667085"
LIGHT = "#F3F6FA"
GRID = "#D8DEE8"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def load_optional(name: str) -> dict | None:
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def register_fonts() -> tuple[str, str]:
    regular = Path("C:/Windows/Fonts/malgun.ttf")
    bold = Path("C:/Windows/Fonts/malgunbd.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("Malgun", regular))
        pdfmetrics.registerFont(TTFont("MalgunBold", bold))
        return "Malgun", "MalgunBold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = register_fonts()


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleKo", parent=base["Title"], fontName=FONT_BOLD, fontSize=22,
            leading=29, textColor=colors.HexColor(INK), alignment=TA_LEFT, spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleKo", parent=base["Normal"], fontName=FONT, fontSize=10,
            leading=15, textColor=colors.HexColor(GREY), spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "H1Ko", parent=base["Heading1"], fontName=FONT_BOLD, fontSize=16,
            leading=22, textColor=colors.HexColor(BLUE), spaceBefore=8, spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "H2Ko", parent=base["Heading2"], fontName=FONT_BOLD, fontSize=12,
            leading=17, textColor=colors.HexColor(INK), spaceBefore=7, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "BodyKo", parent=base["BodyText"], fontName=FONT, fontSize=9.2,
            leading=14.2, textColor=colors.HexColor(INK), spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "SmallKo", parent=base["BodyText"], fontName=FONT, fontSize=7.6,
            leading=11, textColor=colors.HexColor(GREY), spaceAfter=4,
        ),
        "callout": ParagraphStyle(
            "CalloutKo", parent=base["BodyText"], fontName=FONT_BOLD, fontSize=10.2,
            leading=15.5, textColor=colors.HexColor(INK), backColor=colors.HexColor(LIGHT),
            borderColor=colors.HexColor(GRID), borderWidth=0.6, borderPadding=8, spaceAfter=9,
        ),
        "center": ParagraphStyle(
            "CenterKo", parent=base["BodyText"], fontName=FONT, fontSize=8,
            leading=11, textColor=colors.HexColor(GREY), alignment=TA_CENTER,
        ),
    }


ST = styles()


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, ST[style])


def table(data, widths, *, header=True, font_size=7.7):
    header_style = ParagraphStyle(
        f"TableHeader{font_size}", fontName=FONT_BOLD, fontSize=font_size,
        leading=font_size + 2.4, textColor=colors.white,
    )
    cell_style = ParagraphStyle(
        f"TableCell{font_size}", fontName=FONT, fontSize=font_size,
        leading=font_size + 2.7, textColor=colors.HexColor(INK),
    )
    wrapped = []
    for row_index, row in enumerate(data):
        style = header_style if header and row_index == 0 else cell_style
        wrapped.append([Paragraph(escape(str(item)), style) for item in row])
    value = Table(wrapped, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 3.2),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(INK)),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(GRID)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BLUE)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ])
    for row in range(1 if header else 0, len(data)):
        if row % 2 == 0:
            commands.append(("BACKGROUND", (0, row), (-1, row), colors.HexColor("#F8FAFC")))
    value.setStyle(TableStyle(commands))
    return value


def plot_high_na(payload: dict, path: Path) -> None:
    rows = payload["vector_charge_sweep"]
    charges = [row["vortex_charge"] for row in rows]
    series = {
        "Geometric only": [row["variants"]["geometric_only"]["complex_l2"] for row in rows],
        "Raw Jones adaptive": [row["variants"]["adaptive_raw_jones"]["complex_l2"] for row in rows],
        "Effective-vector adaptive": [row["variants"]["adaptive_effective_vector"]["complex_l2"] for row in rows],
    }
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    colors_map = {"Geometric only": ORANGE, "Raw Jones adaptive": GOLD, "Effective-vector adaptive": BLUE}
    markers = {"Geometric only": "o", "Raw Jones adaptive": "s", "Effective-vector adaptive": "D"}
    for label, values in series.items():
        ax.semilogy(charges, values, marker=markers[label], color=colors_map[label], linewidth=2, label=label)
    ax.axhline(1e-6, color=GREY, linestyle="--", linewidth=1.1, label="L2 gate 1e-6")
    ax.set_xlabel("Vortex charge")
    ax.set_ylabel("Complex-field relative L2")
    ax.set_xticks(charges)
    ax.set_ylim(1e-16, 2)
    ax.grid(True, which="major", color=GRID, linewidth=0.7)
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="lower right")
    ax.set_title("High-NA vector harmonic support sweep", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_waxs(payload: dict, path: Path) -> None:
    rows = payload["rows"]
    labels = [str(row["nq"]) for row in rows]
    speed = [row["warm_speedup"] if row["cached_repeat_count"] else row["first_speedup"] for row in rows]
    memory = [row["memory_reduction_ratio"] for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.2, 3.25))
    width = 0.34
    bars1 = ax.bar(x - width / 2, speed, width, color=BLUE, label="FINUFFT / ACFO time")
    bars2 = ax.bar(x + width / 2, memory, width, facecolor="white", edgecolor=GOLD, linewidth=1.8, label="FINUFFT / ACFO peak RSS")
    ax.axhline(1.5, color=GREY, linestyle="--", linewidth=1, label="1.5x robust speed gate")
    ax.axhline(4.0, color="#9AA4B2", linestyle=":", linewidth=1, label="4x memory target")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Nq")
    ax.set_ylabel("Ratio (higher favors ACFO)")
    ax.set_ylim(0, 6.2)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.legend(
        frameon=False,
        fontsize=7.5,
        ncol=2,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
    )
    ax.bar_label(bars1, fmt="%.2fx", fontsize=8, padding=2)
    ax.bar_label(bars2, fmt="%.2fx", fontsize=8, padding=2)
    ax.set_title("Detector-aware WAXS scaling", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_q_sampling(payload: dict, path: Path) -> None:
    range_rows = payload["fixed_dq_range_sweep"]["rows"]
    resolution_rows = payload["fixed_range_resolution_sweep"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharey=True)
    panels = [
        (
            axes[0],
            [row["q_max_inv_angstrom"] for row in range_rows],
            [row["first_total_speedup"] for row in range_rows],
            BLUE,
            "Fixed dq: chunked first-total",
            "qmax (Å⁻¹)",
        ),
        (
            axes[1],
            [row["nq"] for row in resolution_rows],
            [row["hot_speedup"] for row in resolution_rows],
            ORANGE,
            "Fixed q range: reusable hot",
            "Nq",
        ),
    ]
    for ax, x_values, y_values, color, title, xlabel in panels:
        ax.vlines(x_values, 1.0, y_values, color=color, linewidth=1.8, alpha=0.75)
        ax.scatter(
            x_values,
            y_values,
            s=48,
            facecolor="white" if color == ORANGE else color,
            edgecolor=color,
            linewidth=1.8,
            zorder=3,
        )
        for x_value, y_value in zip(x_values, y_values):
            ax.annotate(
                f"{y_value:.1f}×",
                (x_value, y_value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=INK,
            )
        ax.axhline(1.0, color=GREY, linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_ylim(1, 600)
        ax.set_xticks(x_values)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
        ax.grid(axis="y", which="major", color=GRID, linewidth=0.7)
    axes[0].set_ylabel("FINUFFT / factorized time")
    fig.suptitle("Protein-lattice q-sampling speed ratios", x=0.07, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_missing_cone(rows: list[dict], path: Path) -> None:
    angles = [row["angle"] for row in rows]
    values = [100.0 * row["nrmse"] for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.15))
    bars = ax.bar([str(v) for v in angles], values, color=BLUE, edgecolor="#163F7A")
    ax.axhline(5.0, color=ORANGE, linestyle="--", linewidth=1.4, label="Imaging target 5%")
    ax.set_xlabel("Illumination ring angle (deg)")
    ax.set_ylabel("Object NRMSE (%)")
    ax.set_ylim(0, max(values) * 1.22)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.bar_label(bars, fmt="%.2f%%", fontsize=8, padding=3)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Acquisition coverage changes beads recovery", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor(GRID))
    canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor(GREY))
    canvas.drawString(18 * mm, 9 * mm, "ACFO NCS validation execution update")
    canvas.drawRightString(192 * mm, 9 * mm, f"{doc.page}")
    canvas.restoreState()


def build() -> dict:
    SUPPORT.mkdir(parents=True, exist_ok=True)
    high_na = load("high_na_harmonic_support_risk.json")
    waxs = load("waxs_detector_aware_decision.json")
    odt256 = load("odt_256cubed_streaming_decision.json")
    odt128 = load("odt_128cubed_gate_decision.json")
    vector = load("uniaxial_vector_born_direct_64cubed.json")
    dispersion = load("uniaxial_meep_dispersion_highres_decision.json")
    phase = load("uniaxial_meep_3d_phase_gate_decision.json")
    amplitude = load("uniaxial_meep_3d_amplitude_gate_decision.json")
    green = load("uniaxial_green_tensor_residue_64cubed.json")
    asymptotic = load("uniaxial_meep_3d_amplitude_asymptotic_decision.json")
    curvature = load("waxs_curvature_isolated_decision.json")
    release = load("acfo_ncs_reduced_release_suite.json")
    fresh_dependency = load_optional("acfo_ncs_fresh_dependency_rerun.json")
    waxs_direct = load("waxs_direct_finufft_triad.json")
    waxs_direct_sweep = load("waxs_direct_reference_sweep.json")
    waxs_source_convergence = load("waxs_source_discretization_convergence.json")
    waxs_exact_beta = load("waxs_exact_beta_harmonic_bridge.json")
    waxs_lattice = load("protein_nanocrystal_lattice_factorization.json")
    lattice_crossover = load("protein_lattice_finufft_512.json")
    lattice_abba = load("protein_lattice_finufft_512_abba.json")
    prepared_lattice_abba = load("protein_lattice_prepared_finufft_512_abba.json")
    prepared_abba_decision = load("protein_lattice_prepared_abba_decision.json")
    q_sampling = load("protein_lattice_q_sampling_decision.json")
    contraction = load("exact_beta_contraction_optimization_decision.json")
    highq_threshold = load("protein_lattice_highq_threshold_strategy.json")
    waxs_binning = load("protein_nanocrystal_sparse_accuracy_1iee_1x1x1_highq.json")
    tip3p_dense = load("tip3p_dense_highq_exact_beta_20frames.json")
    tip3p_backends = load("tip3p_exact_beta_backend_comparison.json")
    tip3p_scaling = load("tip3p_exact_beta_factorized_q_scaling.json")
    tip3p_finufft = load("tip3p_exact_beta_finufft_512.json")
    component_aware = load("uniaxial_meep_component_aware_amplitude_decision.json")
    finufft_1e6 = next(row for row in waxs_direct["finufft_sweep"] if row["eps"] == 1e-6)
    finufft_1e10 = next(row for row in waxs_direct["finufft_sweep"] if row["eps"] == 1e-10)
    direct_case = {row["name"]: row for row in waxs_direct_sweep["cases"]}
    source_nphi = {
        row["nphi_source"]: row for row in waxs_source_convergence["azimuth_sweep"]
    }
    nonlinear_r16 = component_aware["cases"]["nonlinear_r16"]
    nonlinear_detectable = nonlinear_r16["metrics"]["coupling_ge_10pct"]
    clean_source_status = "final archive verification is issued as an external sidecar after packaging"
    fresh_dependency_status = (
        f"PASS; {fresh_dependency['environment']['packages']['numpy']}, CUDA torch {fresh_dependency['environment']['packages']['torch']}"
        if fresh_dependency is not None and fresh_dependency.get("passed")
        else "미완료"
    )
    angle_rows = []
    for angle in (49, 60, 70, 80):
        if angle == 49:
            item = load("odt_128cubed_beads_30db_grad_1e8.json")
        else:
            item = load(f"odt_128cubed_beads_30db_angle{angle}_grad1e8.json")
        angle_rows.append({"angle": angle, "nrmse": item["final"]["object_nrmse"], "data_residual": item["final"]["data_residual"]})

    high_path = SUPPORT / "high_na_charge_sweep.png"
    waxs_path = SUPPORT / "waxs_detector_scaling.png"
    cone_path = SUPPORT / "odt_missing_cone_angles.png"
    q_sampling_path = SUPPORT / "protein_lattice_q_sampling.png"
    highq_threshold_path = ROOT / "benchmark_results/protein_lattice_highq_threshold_strategy.png"
    plot_high_na(high_na, high_path)
    plot_waxs(waxs, waxs_path)
    plot_missing_cone(angle_rows, cone_path)
    plot_q_sampling(q_sampling, q_sampling_path)

    doc = SimpleDocTemplate(
        str(PDF), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=19 * mm, title="ACFO NCS validation execution update",
        author="ACFO validation program",
    )
    story = []
    story.extend([
        P("ACFO NCS validation 실행 결과 업데이트", "title"),
        P("High-NA cutoff 수정, detector-aware WAXS, ODT 256³ streaming과 missing-cone 판정 분리", "subtitle"),
        P("기술 요약", "h1"),
        P(
            "<b>WAXS operator와 exact-coordinate dense-source 정확도는 확보했지만 dense-disorder 속도 우위는 성립하지 않았다.</b> "
            "High-NA 고차 벡터 cutoff는 charge 0–24 direct-reference sweep을 통과했고, detector-aware WAXS는 Nq=512 local 10/30 alternating protocol에서 1.98×, "
            "ODT 256³는 8 GiB GPU에서 streaming 가능하지만 실제 100-pair 비교에서 ACFO 1,107.95 s, cuFINUFFT 859.55 s로 comparative performance gate가 실패했다. "
            "WAXS full-harmonic은 direct NDFT와 6.7e-14, TIP3P 20-frame exact-beta는 최대 2.46e-12로 맞는다. Perfect 1M lattice와 0.1% sparse-defect control도 2.43e-11이다. "
            "그러나 50,430-atom, Nq512 exact-coordinate 비교에서 fused path는 38.37 s, FINUFFT hot median은 0.387 s로 약 99배 느렸다. 반대로 exact repeated crystal은 216k에서 FINUFFT가 빠르지만, 1.001M atoms의 prepared fused 10 warm-up/30 AB/BA에서는 paired median 33.480×와 p05 24.243×를 얻었다. 후속 분리 sweep은 fixed-dq q-range 증가가 factorized에 유리하고 dense Nq 증가가 FINUFFT에 상대적으로 유리함을 확인했다. 새 동일 폭 high-q 위치 sweep에서는 measured speedup이 79.2×→487.8×였고, q=6.7–8.0 Å⁻¹ Nq64는 180 s stop rule에서 >343.0× 하한을 확보했다. Prepared median은 legacy median보다 4.427× 빨랐고 output L2는 4.81e-14다. Dense speed claim은 FAIL이지만 million-atom periodic 동일 장비 prepared gate는 PASS이며, 독립 장비 rerun, ODT adjoint와 full nonlinear amplitude가 남는다.", "callout"),
        table([
            ["영역", "현재 판정", "핵심 수치", "남은 항목"],
            ["High-NA", "PASS", "charge 24 L2 4.38e-9; max work +39.4%", "큰 pupil family 추가 반복"],
            ["WAXS", "Dense speed FAIL / 1M periodic local PASS", "prepared 10/30 median 33.480×; p05 24.243×", "독립 장비 반복"],
            ["ODT operator", "Streaming PASS / speed FAIL", "100 pair: ACFO 1108 s; cuFINUFFT 860 s", "adjoint 최적화 또는 proof-of-concept"],
            ["ODT imaging", "별도 robustness", "angle 49→70 deg: NRMSE 17.11→5.65%", "true second-axis/prior study"],
            ["Maxwell", "Green PASS; Yee-aware scoped PASS", "detectable support L2 1.286%; full gate FAIL", "source/grid convergence"],
        ], [27*mm, 35*mm, 68*mm, 32*mm], font_size=7.2),
        Spacer(1, 7 * mm),
        P("판정 원칙", "h2"),
        P("정확도, 성능, 물리 reference와 inverse identifiability를 서로 대체하지 않는다. 특히 missing cone은 측정 결측 문제이며 beads NRMSE를 ACFO forward/adjoint correctness gate로 사용하지 않는다."),
        PageBreak(),
    ])

    story.extend([
        P("High-NA: post-mixing 유효 harmonic 전체를 보존해야 한다", "h1"),
        P("Richards–Wolf vector mixing 이후의 effective pupil support를 각 rho-local cutoff에 필수 모드로 합쳤다. raw Jones pupil만 검사하면 charge 18에서 중요한 |h|=16–20 중 일부를 놓친다."),
        Image(str(high_path), width=170*mm, height=78*mm),
        P("그림 1. 32×96 pupil quadrature, 12×24×5 focal grid, sin(theta_max)=0.8. 점선은 complex-field L2 1e-6 gate다.", "center"),
        table([
            ["charge", "effective |h|", "geometric L2", "raw adaptive", "effective adaptive", "work ratio"],
            *[[str(row["vortex_charge"]), f"{min(row['effective_vector_significant_h_abs'])}–{max(row['effective_vector_significant_h_abs'])}",
               f"{row['variants']['geometric_only']['complex_l2']:.2e}",
               f"{row['variants']['adaptive_raw_jones']['complex_l2']:.2e}",
               f"{row['variants']['adaptive_effective_vector']['complex_l2']:.2e}",
               f"{row['variants']['adaptive_effective_vector']['mode_rho_work_ratio_vs_geometric']:.3f}x"]
              for row in high_na["vector_charge_sweep"]],
        ], [18*mm, 28*mm, 30*mm, 30*mm, 34*mm, 24*mm], font_size=7.1),
        P("<b>해석.</b> charge 22–24에서 geometric-only는 사실상 전체장을 잃지만 corrected adaptive는 1e-8보다 작다. 이 수정은 보수적 correctness fix이며 넓은 harmonic support pupil에서 최소 작업량을 보장한다는 주장은 아니다."),
        PageBreak(),
    ])

    story.extend([
        P("Detector-aware WAXS: dense radial sampling에서 속도 이점이 커진다", "h1"),
        P("15.5 keV, 100 mm EIGER2 X 4M rectangle에 들어오는 partial-arc node만 FINUFFT가 계산하도록 비교했다. ACFO 시간은 full ring 계산을 모두 포함하므로 detector mask에 대해 보수적인 비교다."),
        Image(str(waxs_path), width=170*mm, height=77*mm),
        P("그림 2. 파란 막대는 timing ratio, 열린 금색 막대는 peak-RSS ratio다. Nq1024 timing은 반복 없는 first-run probe다.", "center"),
        table([
            ["Nq", "active/full", "timing", "speedup", "complex L2", "row p99", "memory ratio", "fringe samples"],
            *[[str(row["nq"]), f"{row['active_targets']:,}/{row['full_targets']:,}",
               "cached" if row["cached_repeat_count"] else "first",
               f"{(row['warm_speedup'] if row['cached_repeat_count'] else row['first_speedup']):.2f}x",
               f"{row['complex_l2']:.2e}", f"{row['intensity_row_l2_p99']:.2e}",
               f"{row['memory_reduction_ratio']:.2f}x", f"{row['samples_per_3x3x3_fringe']:.2f}"]
              for row in waxs["rows"]],
        ], [13*mm, 32*mm, 17*mm, 18*mm, 25*mm, 24*mm, 21*mm, 22*mm], font_size=6.8),
        P(f"<b>곡률 해석.</b> 별도 curvature-isolated 실험의 speedup–curvature Pearson correlation은 {curvature['curvature_only_test']['pearson_abs_qz_fraction_vs_speedup']:.3f}였다. 따라서 high-q 이점은 curvature 크기 단독 효과가 아니라 target count, angular bandwidth와 geometry reuse가 결합된 workload regime으로 기술해야 한다."),
        P("<b>제한.</b> rectangle envelope만 포함하며 module gap, bad pixel, beamstop은 아직 없다. Nq512 local timing은 10 warm-up/30 measured AB/BA alternating protocol로 확정했지만, 외부 머신 반복은 남아 있다. Nq1024는 first-run probe다."),
        P("작은 문제의 direct oracle", "h2"),
        P("curved-Ewald target은 Cartesian 격자가 아니므로 일반 FFT가 아니라 complex128 explicit nonuniform phase sum을 독립 oracle로 사용했다. ACFO와 FINUFFT는 동일한 binned source와 4,948개 active target에서 각각 oracle과 비교했다."),
        table([
            ["비교", "complex L2", "intensity L2", "의미"],
            ["ACFO vs direct NDFT", f"{waxs_direct['acfo']['complex_l2_vs_direct']:.2e}", f"{waxs_direct['acfo']['intensity_l2_vs_direct']:.2e}", "operator error"],
            ["FINUFFT eps=1e-6 vs direct", f"{finufft_1e6['complex_l2_vs_direct']:.2e}", f"{finufft_1e6['intensity_l2_vs_direct']:.2e}", "large-case baseline setting"],
            ["FINUFFT eps=1e-10 vs direct", f"{finufft_1e10['complex_l2_vs_direct']:.2e}", f"{finufft_1e10['intensity_l2_vs_direct']:.2e}", "converged numerical reference"],
            ["ACFO bins vs exact atoms", f"{waxs_binning['sparse_vs_exact_atom_complex_l2']:.2e}", f"{waxs_binning['sparse_vs_exact_atom_intensity_l2']:.2e}", "binning + operator"],
        ], [52*mm, 28*mm, 28*mm, 64*mm], font_size=7.0),
        P("FINUFFT도 근사법이지만 eps를 1e-10으로 낮추면 direct NDFT와 3.76e-12까지 수렴한다. 따라서 작은 문제의 correctness oracle은 direct NDFT, 큰 문제의 practical baseline은 FINUFFT로 역할을 분리한다."),
        PageBreak(),
    ])

    direct_rows = []
    for name, label in (
        ("low_q_physical", "low q, physical"),
        ("mid_q_physical", "mid q, physical"),
        ("high_q_physical", "high q, physical"),
        ("high_q_planar", "high q, planar"),
        ("high_q_half_curvature", "high q, half curvature"),
    ):
        row = direct_case[name]
        f6 = next(item for item in row["finufft_vs_direct_binned"] if item["eps"] == 1e-6)
        ft = row["finufft_vs_direct_binned"][-1]
        direct_rows.append([
            label,
            f"{row['acfo_vs_direct_binned']['complex_l2']:.2e}",
            f"{row['acfo_full_harmonic_vs_direct_binned']['complex_l2']:.2e}",
            f"{f6['complex_l2']:.2e}",
            f"{ft['complex_l2']:.2e}",
        ])
    story.extend([
        P("WAXS 정확도: operator PASS와 source-representation FAIL을 분리한다", "h1"),
        P("8,008-atom unit cell에서 q 대역, 곡률, angular sampling과 FINUFFT tolerance를 바꾸며 모두 동일한 binned source를 complex128 direct NDFT와 비교했다. Full-harmonic complex128 ACFO는 문서의 small-case 1e-10 gate를 여유 있게 통과하고 production R-dependent 경로는 1e-6 수준이다."),
        table([
            ["case", "ACFO production", "ACFO full-harmonic", "FINUFFT 1e-6", "FINUFFT tight"],
            *direct_rows,
        ], [42*mm, 31*mm, 35*mm, 31*mm, 31*mm], font_size=7.2),
        Spacer(1, 5 * mm),
        P("고-q exact-atom representation 수렴", "h2"),
        P("같은-bin operator 비교만으로는 atom-to-cylinder discretization을 볼 수 없다. 실제 detector target 1,012개를 고정하고 source Nphi만 바꾼 결과, fine r/z bin에서도 Nphi 750의 pixel-intensity L2는 20.20%였다. 현재 production 0.1 nm/Nphi 750의 high-q 전용 intensity L2는 77.65%, ring L2는 22.55%다."),
        table([
            ["source contract", "pixel intensity L2", "ring L2", "intensity NCC", "해석"],
            ["0.1 nm / Nphi 750", "77.65%", "22.55%", "—", "현재 coarse source contract의 unit-cell probe"],
            ["fine bin / Nphi 750", f"{100*source_nphi[750]['intensity_l2']:.2f}%", f"{100*source_nphi[750]['ring_intensity_l2']:.3f}%", f"{source_nphi[750]['intensity_ncc']:.4f}", "azimuth bin이 지배"],
            ["fine bin / Nphi 3,000", f"{100*source_nphi[3000]['intensity_l2']:.2f}%", f"{100*source_nphi[3000]['ring_intensity_l2']:.3f}%", f"{source_nphi[3000]['intensity_ncc']:.4f}", "source Nphi convergence"],
            ["fine bin / Nphi 24,000", f"{100*source_nphi[24000]['intensity_l2']:.2f}%", f"{100*source_nphi[24000]['ring_intensity_l2']:.3f}%", f"{source_nphi[24000]['intensity_ncc']:.4f}", "pixel 1% 경계"],
            ["fine bin / Nphi 96,000", f"{100*source_nphi[96000]['intensity_l2']:.3f}%", f"{100*source_nphi[96000]['ring_intensity_l2']:.3f}%", f"{source_nphi[96000]['intensity_ncc']:.5f}", "exploratory 1% PASS"],
        ], [47*mm, 29*mm, 24*mm, 28*mm, 44*mm], font_size=7.0),
        P("Sparse object storage는 모두 0.275 MiB지만 FFT work는 source Nphi에 비례한다. Nphi 96,000은 source-grid convergence를 확인한 brute-force diagnostic이며 필요한 detector/output 해상도가 아니다. 다음 exact-beta bridge가 source precision과 output sampling을 분리한다.", "callout"),
        P("판정: direct NDFT는 NUFFT와 ACFO operator를 검증했고 두 operator gate는 PASS다. Coarse whole-object representation은 FAIL이며 perfect/sparse-defect crystal과 dense-disorder general path를 분리해 판정한다."),
        PageBreak(),
    ])

    exact = waxs_exact_beta["exact_coordinate_harmonic"]
    fine_rz = waxs_exact_beta["rz_quantization_sweep"][-1]
    defect = waxs_lattice["sparse_defect_control"]
    story.extend([
        P("Exact-beta와 crystal lattice control: source precision을 output Nphi와 분리했다", "h1"),
        P("Per-atom beta와 좌표를 Jacobi–Anger source phase에 직접 넣고 C++ Miller Bessel kernel을 atom chunk로 streaming했다. Detector Nphi는 720으로 고정했고 retained harmonic 350은 Nyquist 360보다 작다."),
        table([
            ["control", "complex/pixel L2", "ring L2", "time", "범위"],
            ["exact coordinate vs direct NDFT", f"{exact['complex_l2']:.2e}", f"{exact['ring_intensity_l2']:.2e}", f"{exact['seconds']:.3f} s", "unit-cell exact-beta bridge"],
            ["fine R/z + exact beta", f"pixel {100*fine_rz['intensity_l2']:.3f}%", f"{100*fine_rz['ring_intensity_l2']:.3f}%", f"{fine_rz['seconds']:.3f} s", "exploratory representation"],
            ["direct NDFT", "oracle", "oracle", f"{waxs_exact_beta['direct_ndft_seconds']:.3f} s", "1,012 active targets"],
        ], [46*mm, 34*mm, 28*mm, 25*mm, 39*mm], font_size=7.2),
        P(f"Temporary Miller kernel upper bound는 {waxs_exact_beta['streamed_kernel_upper_bound_mib']:.3f} MiB다. 따라서 exact beta를 보존하기 위해 96,000-point output FFT를 만들 필요는 없다.", "callout"),
        P("Perfect nanocrystal과 sparse defect", "h2"),
        table([
            ["case", "atoms/cells", "full 1,012-target", "direct subset", "complex L2"],
            *[[row["label"], f"{row['atom_count']:,}/{row['cell_count']}", f"{row['factorized_full_target_seconds']:.3f} s", f"64 target {row['direct_subset_seconds']:.3f} s", f"{row['subset_complex_l2']:.2e}"] for row in waxs_lattice["supercells"]],
            ["5x5x5 + 0.1% displaced", f"{defect['defect_count']:,} delta atoms", f"{defect['factorized_plus_correction_seconds']:.3f} s", f"64 target {defect['direct_modified_subset_seconds']:.3f} s", f"{defect['subset_complex_l2']:.2e}"],
        ], [38*mm, 34*mm, 32*mm, 37*mm, 31*mm], font_size=7.1),
        P("Unit-cell structure factor × finite lattice sum은 perfect repeated crystal의 표준 결정학적 specialization이며 ACFO novelty가 아니다. Sparse vacancy/substitution/position defect는 exact delta correction으로 검증할 수 있다. Dense thermal disorder, solvent/MD snapshot과 cell마다 다른 변형은 이 특수경로를 사용할 수 없으므로 scalable radial-bin + exact-beta/sub-bin contraction이 다음 P0다."),
        PageBreak(),
    ])

    dense_agg = tip3p_dense["aggregate"]
    old_backend = tip3p_backends["backends"]["cpp_miller"]
    fused_backend = tip3p_backends["backends"]["cpp_fused"]
    story.extend([
        P("Dense TIP3P: exact-coordinate correctness는 20개 MD frame에서 통과했다", "h1"),
        P("8 nm TIP3P water box의 50,430 atoms를 DCD에서 직접 읽어 q=5.0, 6.3 Å⁻¹ detector row에 투영했다. 각 frame은 exact atom direct NDFT와 비교했으며 FINUFFT를 correctness oracle로 사용하지 않았다."),
        table([
            ["검증 항목", "결과", "gate/비교 기준", "판정"],
            ["20-frame exact-beta complex L2", f"mean {dense_agg['exact_beta_complex_l2']['mean']:.2e}; max {dense_agg['exact_beta_complex_l2']['max']:.2e}", "모든 frame ≤1e-9", "PASS"],
            ["20-frame intensity L2", f"mean {dense_agg['exact_beta_intensity_l2']['mean']:.2e}; max {dense_agg['exact_beta_intensity_l2']['max']:.2e}", "모든 frame ≤1e-9", "PASS"],
            ["direct / fused median time", f"{dense_agg['direct_seconds']['median']:.3f} / {dense_agg['exact_beta_seconds']['median']:.3f} s", f"direct/fused {dense_agg['direct_over_exact_beta_speedup']['median']:.2f}×", "local q=2"],
            ["0.1 nm coarse intensity L2", f"mean {100*dense_agg['coarse_intensity_l2']['mean']:.1f}%; max {100*dense_agg['coarse_intensity_l2']['max']:.1f}%", "exact atoms 기준", "FAIL"],
            ["frame-to-frame intensity 변화", f"median {100*dense_agg['frame_variation_intensity_l2_vs_frame0']['median']:.1f}%", "frame 0 기준", "multi-frame 필요"],
        ], [48*mm, 55*mm, 43*mm, 26*mm], font_size=7.2),
        Spacer(1, 5 * mm),
        P("동일 q=8 workload에서 fused contraction이 Python 임시행렬을 제거했다", "h2"),
        table([
            ["backend", "time", "direct/backend", "complex L2 vs direct", "의미"],
            ["direct NDFT", f"{tip3p_backends['direct_ndft_seconds']:.3f} s", "oracle", "oracle", "2,720 active targets"],
            ["기존 cpp_miller", f"{old_backend['seconds']:.3f} s", f"{old_backend['direct_over_backend_speedup']:.3f}×", f"{old_backend['complex_l2_vs_direct']:.2e}", "Bessel만 C++; beta 합은 NumPy"],
            ["cpp_fused", f"{fused_backend['seconds']:.3f} s", f"{fused_backend['direct_over_backend_speedup']:.2f}×", f"{fused_backend['complex_l2_vs_direct']:.2e}", "Miller+beta+atom 합산 C++"],
        ], [35*mm, 25*mm, 30*mm, 39*mm, 43*mm], font_size=7.2),
        P(f"Fused는 기존 cpp_miller보다 {tip3p_backends['fused_over_cpp_miller_speedup']:.2f}× 빨라졌지만 두 경로의 complex 결과는 1e-12 이내로 같다. 즉 이 개선은 정확도 완화가 아니라 구현상 temporary와 Python 호출 제거에서 왔다.", "callout"),
        P("범위: 이 결과는 선택한 고-q 두 row와 20개 TIP3P frame의 dense-source correctness를 닫는다. anomalous dispersion, 실험 background와 solvent model 편향은 포함하지 않는다.", "small"),
        PageBreak(),
    ])

    scaling_rows = tip3p_scaling["rows"]
    finufft_row = tip3p_finufft["finufft"]
    fused_512 = tip3p_finufft["cpp_fused"]
    story.extend([
        P("Dense Nq=512: factorized memory는 통과했지만 FINUFFT 속도비교는 실패했다", "h1"),
        P("Element index와 소수의 form-factor row를 유지해 atom×q complex coefficient matrix를 만들지 않았다. 아래 시간은 coordinate preprocessing, fused contraction과 harmonic evaluation을 포함하며 file I/O와 form-factor 생성은 제외한다."),
        table([
            ["Nq", "output targets", "median time", "피한 atom×q matrix", "계산상 관리 배열"],
            *[[str(row['nq']), f"{row['output_targets']:,}", f"{row['seconds']['median']:.3f} s", f"{row['memory_accounting_mib']['avoided_atom_q_complex_matrix']:.1f} MiB", f"{row['memory_accounting_mib']['accounted_total']:.1f} MiB"] for row in scaling_rows],
        ], [22*mm, 36*mm, 34*mm, 43*mm, 37*mm], font_size=7.4),
        P("계산상 관리 배열은 peak RSS가 아니다. 좌표, element indices, weights, form factors, harmonic coefficients, output, per-q basis와 thread scratch만 합산했다.", "small"),
        Spacer(1, 4 * mm),
        P("동일 50,430 atoms × 512 q × 768 azimuth의 reusable FINUFFT", "h2"),
        table([
            ["method", "setup", "first/hot", "sampled peak RSS delta", "cross complex L2"],
            ["cpp_fused exact-beta", "0", f"{fused_512['seconds']:.3f} s", f"{fused_512['memory']['peak_rss_delta_mib']:.1f} MiB", "reference side of cross-check"],
            [f"FINUFFT eps={finufft_row['eps']:.0e}", f"{finufft_row['setup_seconds']:.3f} s", f"{finufft_row['first_seconds']:.3f}/{finufft_row['hot_seconds']['median']:.3f} s", f"setup {finufft_row['setup_memory']['peak_rss_delta_mib']:.1f}; execute {finufft_row['first_memory']['peak_rss_delta_mib']:.1f} MiB", f"{tip3p_finufft['cross_error']['complex_l2']:.2e}"],
        ], [41*mm, 25*mm, 36*mm, 46*mm, 24*mm], font_size=7.0),
        P(f"<b>Comparative performance FAIL.</b> FINUFFT first-total은 fused의 {tip3p_finufft['finufft_first_total_over_fused_ratio']:.4f}배, hot time은 {tip3p_finufft['finufft_hot_over_fused_ratio']:.4f}배다. 역수로 보면 FINUFFT가 각각 약 {1/tip3p_finufft['finufft_first_total_over_fused_ratio']:.0f}×, {1/tip3p_finufft['finufft_hot_over_fused_ratio']:.0f}× 빠르다. eps=1e-6 cross complex L2 {tip3p_finufft['cross_error']['complex_l2']:.2e}는 practical baseline 수준이며 direct NDFT oracle을 대체하지 않는다.", "callout"),
        P("해석: exact-beta factorization은 dense coordinate 정확도와 bounded working arrays를 제공하지만, q가 조밀하고 source가 비구조적인 이 workload는 FINUFFT에 훨씬 유리하다. 따라서 dense-disorder general speed advantage를 주장하지 않고, 반복 lattice/geometry reuse 또는 구조화된 source에서의 별도 regime claim만 유지해야 한다."),
        PageBreak(),
    ])

    crystal_small, crystal_large = lattice_crossover["rows"]
    abba_summary = lattice_abba["measured_summary"]
    abba_factorized = abba_summary["factorized_seconds"]
    abba_finufft = abba_summary["finufft_seconds"]
    abba_speedup = abba_summary["paired_speedup"]
    prepared_summary = prepared_lattice_abba["measured_summary"]
    prepared_factorized = prepared_summary["factorized_seconds"]
    prepared_finufft = prepared_summary["finufft_seconds"]
    prepared_speedup = prepared_summary["paired_speedup"]
    prepared_order_gap = prepared_abba_decision["order_groups"]["relative_gap"]["paired_speedup_median_relative_gap"]
    story.extend([
        P("Perfect repeated crystal: prepared fused 10/30 로컬 gate를 닫았다", "h1"),
        P("동일 1.001M atoms, 512×768 full-polar target, FINUFFT eps=1e-6·4 threads와 15 AB/15 BA 순서를 유지했다. Legacy는 매회 unit-cell exact-beta state를 다시 구성하고, prepared fused는 coordinate/cutoff를 재사용하며 fused-phase coefficient와 FFT azimuth synthesis를 실행한다. 두 경로 모두 finite lattice factor를 재사용한다."),
        table([
            ["case", "atoms/cells", "protocol", "factorized", "FINUFFT", "speedup", "cross complex L2"],
            [crystal_small["label"], f"{crystal_small['atom_count']:,}/{crystal_small['cell_count']}", "single local hot", f"{crystal_small['factorized']['hot_seconds']['median']:.3f} s", f"{crystal_small['finufft']['hot_seconds']['median']:.3f} s", f"{crystal_small['speedup']['hot_finufft_over_factorized']:.3f}×", f"{crystal_small['cross_error']['complex_l2']:.2e}"],
            ["1M legacy", f"{crystal_large['atom_count']:,}/{crystal_large['cell_count']}", "10 warm-up / 30 AB/BA", f"{abba_factorized['median']:.3f} s", f"{abba_finufft['median']:.3f} s", f"{abba_speedup['median']:.3f}×", f"{lattice_abba['cross_error']['complex_l2']:.2e}"],
            ["1M prepared fused", f"{prepared_lattice_abba['atom_count']:,}/{prepared_lattice_abba['cell_count']}", "10 warm-up / 30 AB/BA", f"{prepared_factorized['median']:.3f} s", f"{prepared_finufft['median']:.3f} s", f"{prepared_speedup['median']:.3f}×", f"{prepared_lattice_abba['cross_error']['complex_l2']:.2e}"],
        ], [20*mm, 27*mm, 34*mm, 25*mm, 25*mm, 20*mm, 23*mm], font_size=6.5),
        P("Factorized speedup은 FINUFFT time / factorized time이다. 216k 반례에서는 FINUFFT가 빠르지만, 1M prepared median은 33.480×다. Legacy와 prepared의 FINUFFT median 차이는 로컬 run 변동이며 paired ratio로 판정한다."),
        Spacer(1, 3 * mm),
        P("1.001M 30-pair distribution", "h2"),
        table([
            ["protocol", "factorized median", "FINUFFT median", "paired median", "paired p05", "CV / order gap"],
            ["legacy", f"{abba_factorized['median']:.3f} s", f"{abba_finufft['median']:.3f} s", f"{abba_speedup['median']:.3f}×", f"{abba_speedup['p05']:.3f}×", f"CV {100*abba_speedup['cv']:.1f}%"],
            ["prepared fused", f"{prepared_factorized['median']:.3f} s", f"{prepared_finufft['median']:.3f} s", f"{prepared_speedup['median']:.3f}×", f"{prepared_speedup['p05']:.3f}×", f"CV {100*prepared_speedup['cv']:.1f}%; AB/BA {100*prepared_order_gap:.2f}%"],
        ], [28*mm, 31*mm, 31*mm, 30*mm, 27*mm, 40*mm], font_size=6.8),
        Spacer(1, 4 * mm),
        P("Setup과 first-total 해석", "h2"),
        table([
            ["prepared plan", "separable lattice", "FINUFFT plan", "derived prepared first-total", "derived FINUFFT first-total", "derived ratio"],
            [f"{prepared_lattice_abba['prepared_plan_setup_seconds']:.3f} s", f"{prepared_lattice_abba['lattice_setup_seconds']:.3f} s", f"{prepared_lattice_abba['finufft_setup_seconds']:.3f} s", f"{prepared_abba_decision['setup']['prepared_first_total_using_measured_median_seconds']:.3f} s", f"{prepared_abba_decision['setup']['finufft_first_total_using_measured_median_seconds']:.3f} s", f"{prepared_abba_decision['setup']['first_total_speedup_using_measured_medians']:.3f}×*"],
        ], [28*mm, 28*mm, 28*mm, 35*mm, 35*mm, 28*mm], font_size=6.6),
        P("* setup + measured median으로 만든 descriptive first-total이며 paired statistic이 아니다.", "small"),
        P(f"<b>준비된 로컬 timing gate PASS.</b> prepared factorized median은 legacy보다 {prepared_abba_decision['legacy_comparison']['legacy_over_prepared_factorized_median']:.3f}× 빨랐고 output L2는 {prepared_lattice_abba['legacy_comparison']['legacy_vs_prepared_complex_l2']:.2e}다. Paired median/p05는 {prepared_speedup['median']:.3f}×/{prepared_speedup['p05']:.3f}×이고 AB/BA median gap은 {100*prepared_order_gap:.2f}%다. 216k 반례와 동일 장비 한계 때문에 독립 머신 반복 전에는 publication timing을 최종 확정하지 않는다.", "callout"),
        P("이 lattice factor는 표준 결정학적 specialization이며 ACFO novelty가 아니다. Direct NDFT correctness는 별도 q=3/64-target subset에서 1M complex L2 2.43e-11로 확인했다. Dense thermal disorder에는 이 factorization을 적용하지 않는다.", "small"),
        PageBreak(),
    ])

    range_rows = q_sampling["fixed_dq_range_sweep"]["rows"]
    resolution_rows = q_sampling["fixed_range_resolution_sweep"]["rows"]
    optimization = q_sampling["exact_optimization_profile_nq512"]
    contraction_backend = contraction["backend_comparison"]
    story.extend([
        P("q 범위와 q resolution은 반대 방향으로 작용했다", "h1"),
        P("1.001M-atom exact repeated crystal에서 두 축을 분리했다. 왼쪽은 dq≈0.160 Å⁻¹를 유지하며 qmax와 물리 최소 FFT-friendly Nphi를 늘리고 FINUFFT를 q-block=2로 streaming한 first-total 비교다. 오른쪽은 q=5–6.3 Å⁻¹와 Nphi=768을 고정하고 Nq만 늘린 reusable hot 비교다."),
        Image(str(q_sampling_path), width=170*mm, height=71*mm),
        P("그림 4. y축은 FINUFFT time / factorized time이며 높을수록 factorized 경로가 빠르다. 두 패널은 protocol이 다르므로 패널 사이의 절대값보다 각 패널 내부 추세를 해석한다. Nq512 hot 점은 prepared fused 10/30 AB/BA paired median이다.", "center"),
        table([
            ["fixed-dq qmax", "Nq/Nphi", "factorized first-total", "chunked FINUFFT wall", "speedup", "cross L2"],
            *[[f"{row['q_max_inv_angstrom']:.2f} Å⁻¹", f"{row['nq']}/{row['nphi']}", f"{row['factorized_first_total_seconds']:.3f} s", f"{row['chunked_finufft_streamed_wall_seconds']:.3f} s", f"{row['first_total_speedup']:.1f}×", f"{row['cross_complex_l2']:.2e}"] for row in range_rows],
        ], [27*mm, 26*mm, 37*mm, 39*mm, 23*mm, 25*mm], font_size=6.8),
        Spacer(1, 3 * mm),
        table([
            ["fixed range Nq", "dq", "factorized hot", "FINUFFT hot", "hot speedup", "근거"],
            *[[str(row['nq']), f"{row['dq_inv_angstrom']:.5f} Å⁻¹", f"{row['factorized_hot_seconds']:.3f} s", f"{row['finufft_hot_seconds']:.3f} s", f"{row['hot_speedup']:.1f}×", "single local" if row['nq'] < 512 else "10/30 paired median"] for row in resolution_rows],
        ], [25*mm, 30*mm, 31*mm, 31*mm, 27*mm, 35*mm], font_size=6.8),
        P("Prepared path와 fused C++가 synthesis·lattice·contraction 병목을 줄였다", "h2"),
        table([
            ["항목", "이전", "prepared/separable", "개선", "정확도"],
            ["Nq512 hot exact path", f"{optimization['legacy_seconds']:.3f} s", f"{optimization['prepared_hot_seconds']:.3f} s", f"{optimization['legacy_over_prepared_speedup']:.3f}×", f"complex L2 {optimization['legacy_vs_prepared_complex_l2']:.2e}"],
            ["Nq512 lattice setup", f"{optimization['direct_lattice_seconds']:.3f} s", f"{optimization['separable_lattice_seconds']:.3f} s", f"{optimization['direct_over_separable_lattice_speedup']:.1f}×", f"complex L2 {optimization['lattice_complex_l2']:.2e}"],
            ["C++ coefficient A/B", f"baseline {contraction_backend['baseline_coefficient_median_seconds']:.3f} s", f"fused {contraction_backend['fused_phase_coefficient_median_seconds']:.3f} s", f"paired median {contraction_backend['baseline_over_fused_paired_median']:.3f}×", f"complex L2 {contraction_backend['fused_vs_baseline_complex_l2']:.1e}"],
            ["optional phase cache", f"fused {contraction_backend['fused_phase_coefficient_median_seconds']:.3f} s", f"cached {contraction_backend['cached_phase_coefficient_median_seconds']:.3f} s", f"{contraction_backend['fused_over_cached_paired_median']:.3f}×; +{contraction_backend['cached_phase_mib']:.2f} MiB", f"complex L2 {contraction_backend['cached_vs_fused_complex_l2']:.2e}"],
        ], [35*mm, 34*mm, 42*mm, 36*mm, 35*mm], font_size=6.4),
        P(f"<b>판정.</b> 사용자의 가설대로 dense q resolution은 FINUFFT에 상대적으로 유리했고, fixed-dq로 q 범위를 넓히면 chunked FINUFFT의 불리함이 커졌다. qmax 2.13→8.06에서 first-total speedup은 47.0×→147.8×로 증가했고, 고정 q 범위 Nq 32→512에서는 hot speedup이 314.3×→{resolution_rows[-1]['hot_speedup']:.1f}×로 감소했다. Low-memory fused-phase는 kernel p05 1.161×를 확보했고 10/30 factorized median은 legacy 대비 {prepared_abba_decision['legacy_comparison']['legacy_over_prepared_factorized_median']:.3f}× 빨랐다. 42.89 MiB cache는 추가 median 1.169×만 제공해 반복 hot용 option으로 남긴다. Nq512 local prepared AB/BA gate는 PASS이며 독립 머신 반복이 남았다. 현재 exact 병목은 Miller Bessel recurrence와 axial/harmonic accumulation이다.", "callout"),
        PageBreak(),
    ])

    highq_calibration = highq_threshold["q_block_calibration"]
    highq_position = highq_threshold["position_sweep"]["rows"]
    highq_resolution = highq_threshold["resolution_sweep"]["rows"]
    highq_holdout = highq_threshold["extrapolation_validation"]
    resolution_table_rows = []
    for row in highq_resolution:
        if row["finufft_wall_seconds"] is None:
            measured = "미실행"
            measured_speedup = "-"
        elif row["finufft_censored"]:
            measured = (
                f">{row['finufft_wall_seconds']:.1f} s "
                f"({row['finufft_completed_nq']}/{row['nq']} rows)"
            )
            measured_speedup = f">{row['measured_speedup_or_lower_bound']:.1f}×"
        else:
            measured = f"{row['finufft_wall_seconds']:.1f} s"
            measured_speedup = f"{row['measured_speedup_or_lower_bound']:.1f}×"
        projected = row["extrapolation"].get("predicted_full_finufft_seconds")
        resolution_table_rows.append(
            [
                str(row["nq"]),
                f"{row['factorized_first_total_seconds']:.3f} s",
                measured,
                measured_speedup,
                "-" if projected is None else f"{projected:.1f} s (projection)",
            ]
        )
    story.extend([
        P("High-q 위치 sweep과 threshold-censored FINUFFT 비교", "h1"),
        P("1.001M-atom repeated protein crystal에서 폭 1.30 Å⁻¹와 Nq=16을 고정하고 q-window를 0.05–1.35에서 6.70–8.00 Å⁻¹까지 이동했다. Nphi는 각 qmax의 exact-coordinate harmonic cutoff를 만족하는 최소 FFT-friendly 값으로 정했다. 이 측정은 물리적으로 필요한 Nphi 증가를 포함한 high-q workload 효과이며, 순수 곡률만의 효과는 별도 curvature-isolation control이 담당한다."),
        Image(str(highq_threshold_path), width=170*mm, height=68*mm),
        P("그림 5. 왼쪽은 완료된 동일 폭 q-window 실측값이다. 오른쪽에서 원은 완료 측정, 삼각형과 화살표는 180 s censored lower bound, 점선과 음영은 holdout gate를 통과한 보조 projection이다.", "center"),
        P("q-block calibration", "h2"),
        table([
            ["q-block", "FINUFFT wall", "peak process RSS / total", "판정"],
            *[[
                str(row["q_block_size"]),
                f"{row['finufft_wall_seconds']:.1f} s",
                f"{100*row['peak_process_rss_fraction']:.1f}%",
                "선택" if row["q_block_size"] == highq_calibration["selected_q_block"] else "비선택",
            ] for row in highq_calibration["rows"]],
        ], [31*mm, 43*mm, 58*mm, 39*mm], font_size=7.4),
        P("High-q resolution과 stop rule", "h2"),
        table([
            ["Nq", "ACFO first-total", "FINUFFT measured", "measured speedup/lower bound", "holdout-gated full estimate"],
            *resolution_table_rows,
        ], [17*mm, 35*mm, 48*mm, 43*mm, 43*mm], font_size=6.8),
        P(f"<b>판정.</b> 동일 폭 위치 sweep의 measured speedup은 {highq_position[0]['measured_speedup_or_lower_bound']:.1f}×에서 {highq_position[-1]['measured_speedup_or_lower_bound']:.1f}×로 증가했다. q=6.70–8.00 Å⁻¹에서 Nq=32는 FINUFFT {highq_resolution[0]['finufft_wall_seconds']:.2f} s 대 ACFO {highq_resolution[0]['factorized_first_total_seconds']:.3f} s, 즉 {highq_resolution[0]['measured_speedup_or_lower_bound']:.1f}×의 완료 측정이다. Nq=64는 {highq_resolution[1]['finufft_completed_nq']}/64 rows와 {highq_resolution[1]['finufft_wall_seconds']:.2f} s에서 중단되어 >{highq_resolution[1]['measured_speedup_or_lower_bound']:.1f}×만 1차 근거로 사용한다. q-center 선형 block model은 가장 높은 q {100*highq_holdout['holdout_relative_error']:.3f}% holdout error로 15% gate를 통과했지만, projection은 planning/SI 보조값이며 측정 timing으로 인용하지 않는다.", "callout"),
        PageBreak(),
    ])

    selected = odt256["selected"]
    r32 = odt256["block_comparison"]["r32_i4_first"]
    odt100 = odt256["actual_100pair"]
    odt_match = odt256["matched_accuracy"]
    story.extend([
        P("ODT 256³: prepared-table-free streaming을 8 GiB GPU에서 실행했다", "h1"),
        P("기존 실패는 GPU 연산이 아니라 detector azimuth마다 동일 radial kernel을 복제한 9.12 GiB CPU table과 18.25 GiB prepared-adjoint table이었다. compact axisymmetric context와 radial/illumination 이중 streaming으로 두 materialization을 제거했다."),
        table([
            ["항목", "radial 32 / illum 4", "radial 16 / illum 4 (선택)"],
            ["GPU peak allocated", f"{r32['gpu_peak_allocated_mib']:.1f} MiB", f"{selected['gpu_peak_allocated_mib']:.1f} MiB"],
            ["forward", f"{r32['gpu_forward_hot_s']:.3f} s", f"{selected['gpu_forward_hot_s']:.3f} s"],
            ["adjoint", f"{r32['gpu_adjoint_hot_s']:.3f} s", f"{selected['gpu_adjoint_hot_s']:.3f} s"],
            ["forward-adjoint pair", f"{r32['gpu_forward_adjoint_pair_hot_s']:.3f} s", f"{selected['gpu_forward_adjoint_pair_hot_s']:.3f} s"],
            ["dot error, complex128 accumulation", f"{r32['forward_adjoint_dot_error_complex128_accum']:.3e}", f"{selected['forward_adjoint_dot_error_complex128_accum']:.3e}"],
        ], [60*mm, 55*mm, 58*mm], font_size=8),
        Spacer(1, 5 * mm),
        P("검증 범위", "h2"),
        P(f"실제 문제는 16,777,216 object coefficients, {selected['total_illumination_count']} illuminations, 256×256 detector, {selected['total_q_samples']:,} q samples다. basis는 {selected['gpu_basis_mib']:.2f} MiB이며 peak에는 forward, adjoint, pair와 reconstruction update가 포함된다."),
        P("128³ independent Cartesian exponent subset은 forward 4.155e-12, selected-object adjoint 3.895e-12, full complex128 dot 9.688e-16이다. 256³ 전체 independent direct matrix는 현실적으로 불가능하므로 scale에서는 dot consistency와 128³ independent subset을 결합한다."),
        P("실제 100-pair standalone 비교", "h2"),
        table([
            ["backend", "setup", "100-pair wall", "pair median", "accuracy cross-check"],
            ["ACFO", f"{odt100['acfo_setup_s']:.2f} s", f"{odt100['acfo_measured_wall_s']:.2f} s", f"{odt100['acfo_pair_median_s']:.3f} s", f"dot {selected['forward_adjoint_dot_error_complex128_accum']:.2e}"],
            ["cuFINUFFT", f"{odt100['cufinufft_setup_s']:.2f} s", f"{odt100['cufinufft_measured_wall_s']:.2f} s", f"{odt100['cufinufft_pair_median_s']:.3f} s", f"F/A L2 {odt_match['forward_rel_l2_vs_acfo']:.1e}/{odt_match['adjoint_rel_l2_vs_acfo']:.1e}"],
        ], [34*mm, 25*mm, 36*mm, 32*mm, 46*mm], font_size=7.1),
        P(f"<b>Comparative performance FAIL.</b> steady 100-pair baseline/ACFO ratio는 {odt100['baseline_over_acfo_steady_speedup']:.3f}×이며 ACFO가 cuFINUFFT보다 {odt100['acfo_slowdown_vs_cufinufft']:.3f}× 느렸다. 기존 995.3 s projection은 실제 1,107.95 s 측정으로 대체한다. 현재 ODT는 memory-feasible proof-of-concept로 제한하고 structured adjoint 최적화 전에는 speed advantage를 주장하지 않는다.", "callout"),
        PageBreak(),
    ])

    story.extend([
        P("Missing cone은 ACFO 오류가 아니라 acquisition 결측 문제다", "h1"),
        P("ACFO operator gate, 관측 가능한 subspace의 numerical inverse control, physical imaging robustness를 분리한다. 128³ adjoint-range truth는 NRMSE 0.997%로 통과했지만 broad-support beads는 동일 정보를 포함하지 않는다."),
        Image(str(cone_path), width=170*mm, height=74*mm),
        P("그림 3. 동일 128³ beads, 30 dB complex noise, 60 ring + 1 axis, gradient penalty lambda=1e8. illumination angle만 변경했다.", "center"),
        table([
            ["angle", "object NRMSE", "data residual", "해석"],
            *[[f"{row['angle']} deg", f"{100*row['nrmse']:.2f}%", f"{100*row['data_residual']:.3f}%",
               "coverage probe; true second-axis 아님"] for row in angle_rows],
        ], [24*mm, 32*mm, 32*mm, 84*mm], font_size=7.8),
        P("operator와 prior를 고정하고 acquisition coverage만 바꿨을 때 NRMSE가 17.11%에서 5.65%까지 변한다. 낮은 data residual과 높은 object error의 공존은 forward 구현 실패가 아니라 null-space 비식별성을 가리킨다."),
        P("quadratic gradient lambda sweep의 최적값은 17.11%, nonnegative FISTA 200 iteration 최적값은 21.15%였다. 이 negative result는 prior만으로 결측 정보를 복원했다고 주장하지 못하게 하는 중요한 경계다."),
        PageBreak(),
    ])

    dispersion_fine = dispersion["rows"][-1]
    phase_fine = phase["rows"][-1]
    story.extend([
        P("Maxwell/비선형 검증: Green-tensor와 Yee-aware amplitude를 분리했다", "h1"),
        table([
            ["검증", "판정", "주요 수치", "주장 범위"],
            ["64³ vector Born", "PASS", f"extraordinary L2 {vector['cases'][0]['vector_complex_l2']:.2e}", "tensor/eigenpolarization first-Born algebra"],
            ["Maxwell Green residue", "PASS", f"corrected field L2 {green['cases'][0]['green_field_acfo_vs_direct_complex_l2']:.2e}", "homogeneous dyadic first-Born amplitude"],
            ["PyMeep 2-D dispersion", "PASS", f"ellipse L2 {100*dispersion_fine['correct_relative_l2']:.4f}%", "uniaxial dispersion ridge"],
            ["PyMeep 3-D phase", "PASS", f"phase-slope L2 {100*phase_fine['calibrated_extraordinary_ellipse_relative_l2']:.4f}%", "grid/time/boundary checked phase curvature"],
            ["PyMeep single-Ex amplitude", "Control PASS", "L2 4.625%; NCC 0.9996; ordinary 1.881%", "exact Yee-source representation control"],
            ["Yee-aware nonlinear amplitude", "Scoped PASS", f"detectable L2 {100*nonlinear_detectable['complex_l2']:.3f}%; NCC {nonlinear_detectable['intensity_ncc']:.4f}", "10% detectable support only"],
            ["Publication full amplitude", "FAIL", "FDTD grid 41.75%; source oracle 37.32%", "grid/source and forced control unresolved"],
        ], [34*mm, 29*mm, 65*mm, 44*mm], font_size=7.3),
        Spacer(1, 5 * mm),
        P("Green-tensor PASS와 PyMeep amplitude FAIL을 분리한다", "h2"),
        P("새 reference는 Cartesian Maxwell wave operator의 nullspace와 resolvent limit에서 dyadic pole residue를 계산한다. Corrected ACFO–direct Green field L2는 extraordinary 1.10e-14, ordinary 9.82e-15다. 이 과정에서 기존 extraordinary scalar residue가 Maxwell field norm을 5.216x 크게 만든다는 문제도 검출했다."),
        P("PyMeep의 Ex/Ey/Ez Yee source array를 staggered 좌표와 integration weight까지 그대로 export해 각 component의 Fourier source를 Maxwell residue에 적용했다. 이 exact source contract를 쓰면 resolution-16 nonlinear case의 10% detectable support에서 complex L2 1.286%, NCC 0.9965, ordinary calibration 1.357%가 된다."),
        P(f"Raw peak error는 {nonlinear_detectable['peak_error_deg']:.0f}°이지만 reference top-two peak margin이 {100*nonlinear_detectable['reference_peak_margin_fraction']:.4f}%뿐이어서 이 support에서 peak 위치는 식별 가능하지 않다. 이 규칙은 raw peak 값을 삭제하지 않고 peak gate만 non-identifiable로 판정한다."),
        P(f"그러나 joint detectable support의 resolution 12→16 변화는 FDTD {100*component_aware['grid_convergence']['calibrated_fdt_l2_r12_to_r16']:.2f}%, exact source oracle {100*component_aware['grid_convergence']['source_oracle_l2_r12_to_r16']:.2f}%다. 즉 source discretization 자체가 미수렴이며 forced-sphere control도 해결되지 않았다. Scoped detectable-support amplitude는 PASS지만 publication full amplitude는 FAIL이다.", "callout"),
        PageBreak(),
    ])

    story.extend([
        P("제출 전 남은 작업과 권장 claim", "h1"),
        P("현재 권장 claim boundary", "h2"),
        P("ACFO operator는 동일 cylindrical representation에서 direct NDFT에 production 1e-6, full-harmonic 1e-10보다 좋은 정확도로 일치한다. Exact-beta bridge, TIP3P 20-frame dense-source correctness와 perfect/sparse-defect lattice control도 exact atoms에 1e-9 이하로 맞는다. Dense TIP3P Nq512에서는 reusable FINUFFT가 first-total 약 40×, hot 약 99× 빠르므로 dense general speed advantage는 주장하지 않는다. Exact repeated crystal도 216k에서는 FINUFFT가 빠르지만 1.001M prepared fused 10/30 AB/BA에서 paired median 33.480×, p05 24.243×로 로컬 timing gate를 통과했다. Factorized median은 legacy보다 4.427× 빨랐고 AB/BA median gap은 2.04%다. Fixed-dq q-range와 fixed-range Nq sweep은 각각 factorized-favorable와 FINUFFT-relative-favorable 추세를 분리했다. 동일 폭 high-q position sweep은 물리적 Nphi 증가를 포함해 measured speedup 79.2×→487.8×를 보였고, q=6.7–8.0 Å⁻¹에서는 Nq32 538.2× 완료 측정과 Nq64 >343.0× censored lower bound를 얻었다. 외삽 timing은 holdout-gated 보조값이며 claim에 사용하지 않는다. Prepared output은 legacy와 L2 4.81e-14로 같고 local gate는 닫혔지만, claim은 tested million-atom periodic regime과 동일 장비에 한정하며 독립 반복이 남는다."),
        P("즉시 실행할 잔여 항목", "h2"),
        table([
            ["우선순위", "작업", "완료 기준", "현재 상태"],
            ["P0", "Dense-disorder source path", "20-frame direct NDFT + Nq512 corrected path", "correctness PASS; speed FAIL"],
            ["P0", "WAXS periodic timing 재현", "prepared 10/30 + independent-machine rerun", "prepared 10/30 PASS; independent pending"],
            ["P0", "ODT claim decision", "실제 100 pair + cuFINUFFT", "speed FAIL; adjoint 최적화/범위 축소"],
            ["P0", "Release regression", "archive, source rebuild, one-command suite", clean_source_status],
            ["P1", "Dependency/clean rerun", "격리 venv + final archive source rebuild", f"historical fresh 172; current local 181; {clean_source_status}"],
            ["P1", "Independent rerun", "다른 machine/operator가 핵심 표 재생성", "미실행"],
            ["P1", "Amplitude validation", "Yee source/grid convergence + forced control", "detectable scoped PASS; full FAIL"],
            ["P2", "True second-axis imaging", "acquisition gain과 prior gain 분리", "multi-angle probe만 존재"],
        ], [17*mm, 48*mm, 66*mm, 41*mm], font_size=7.4),
        Spacer(1, 6 * mm),
        P("사용하면 안 되는 표현", "h2"),
        P("curvature 자체가 speedup을 만든다, perfect lattice specialization이 dense disorder도 해결한다, 기존 1.98× same-bin timing이 corrected dense atomistic 속도 우위다, dense TIP3P에서 ACFO가 FINUFFT보다 빠르다, 모든 repeated crystal에서 factorized path가 빠르다, 표준 lattice sum을 ACFO novelty로 주장한다, 현재 ODT가 cuFINUFFT보다 빠르다, PyMeep bridge가 full amplitude를 검증했다는 표현은 사용하지 않는다."),
        P("보고서 판정", "h2"),
        P("High-NA, WAXS correctness/controls, ODT streaming과 Maxwell scoped amplitude는 근거로 사용할 수 있다. Dense-disorder speed는 FAIL이지만 1M exact repeated-crystal prepared 10/30은 median 33.480×, p05 24.243×로 로컬 gate를 통과했다. q-range/resolution과 동일 폭 high-q 위치 실험도 사용자의 해석을 지지하고 Nq64 censored run은 >343.0× 측정 하한을 제공한다. Prepared factorized median은 legacy보다 4.427× 빨랐고 output을 유지했다. High-q projection은 보조값이며 독립 반복, ODT speed와 full amplitude가 남아 submission readiness는 미완료다.", "callout"),
    ])

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    pdf = fitz.open(PDF)
    texts = [page.get_text() for page in pdf]
    combined = "\n".join(texts)
    anchors = ["High-NA", "Detector-aware WAXS", "WAXS 정확도", "Exact-beta", "Dense TIP3P", "Dense Nq=512", "Perfect repeated crystal", "q 범위와 q resolution", "High-q 위치 sweep", "ODT 256", "Missing cone", "Maxwell", "제출 전 남은 작업"]
    missing = [anchor for anchor in anchors if anchor not in combined]
    if missing:
        raise RuntimeError(f"missing PDF anchors: {missing}")
    if any(not text.strip() for text in texts):
        raise RuntimeError("blank page detected")
    preview_paths = []
    # Inspect the title, WAXS direct/source split, ODT, Maxwell, and final pages.
    for index in sorted({0, 2, 3, len(pdf)//2, len(pdf)//2 + 1, 8, 9, len(pdf)-2, len(pdf)-1}):
        path = SUPPORT / f"preview_page_{index+1:02d}.png"
        pdf[index].get_pixmap(matrix=fitz.Matrix(1.35, 1.35), alpha=False).save(path)
        preview_paths.append(path.relative_to(ROOT).as_posix())

    sources = [
        "benchmark_results/high_na_harmonic_support_risk.json",
        "benchmark_results/waxs_detector_aware_decision.json",
        "benchmark_results/waxs_curvature_isolated_decision.json",
        "benchmark_results/waxs_direct_finufft_triad.json",
        "benchmark_results/waxs_direct_reference_sweep.json",
        "benchmark_results/waxs_source_discretization_convergence.json",
        "benchmark_results/waxs_exact_beta_harmonic_bridge.json",
        "benchmark_results/protein_nanocrystal_lattice_factorization.json",
        "benchmark_results/protein_lattice_finufft_512.json",
        "benchmark_results/protein_lattice_finufft_512_abba.json",
        "benchmark_results/protein_lattice_prepared_finufft_512_abba.json",
        "benchmark_results/protein_lattice_prepared_finufft_512_abba.md",
        "benchmark_results/protein_lattice_prepared_abba_decision.json",
        "benchmark_results/protein_lattice_prepared_abba_decision.md",
        "benchmark_results/protein_lattice_q_sampling_decision.json",
        "benchmark_results/protein_lattice_q_sampling_resolution_nq512_optimized_profile.json",
        "benchmark_results/protein_lattice_q_sampling_resolution_nq512_fused_phase_profile.json",
        "benchmark_results/exact_beta_contraction_backends_nq512.json",
        "benchmark_results/exact_beta_contraction_optimization_decision.json",
        "benchmark_results/protein_lattice_highq_threshold_strategy.json",
        "benchmark_results/protein_lattice_highq_threshold_strategy.md",
        "benchmark_results/protein_lattice_highq_threshold_strategy.png",
        "benchmark_results/protein_nanocrystal_sparse_accuracy_1iee_1x1x1_highq.json",
        "benchmark_results/tip3p_dense_highq_exact_beta_20frames.json",
        "benchmark_results/tip3p_exact_beta_backend_comparison.json",
        "benchmark_results/tip3p_exact_beta_factorized_q_scaling.json",
        "benchmark_results/tip3p_exact_beta_finufft_512.json",
        "benchmark_results/odt_128cubed_gate_decision.json",
        "benchmark_results/odt_256cubed_streaming_decision.json",
        "benchmark_results/odt_torch_256cubed_100pair.json",
        "benchmark_results/odt_cufinufft_256cubed_100pair.json",
        "benchmark_results/odt_cufinufft_gpu_256cubed_plan_pair2.json",
        "benchmark_results/odt_128cubed_beads_30db_nonnegative_fista.json",
        "benchmark_results/uniaxial_vector_born_direct_64cubed.json",
        "benchmark_results/uniaxial_green_tensor_residue_64cubed.json",
        "benchmark_results/uniaxial_meep_3d_amplitude_asymptotic_decision.json",
        "benchmark_results/uniaxial_meep_component_aware_amplitude_decision.json",
        "benchmark_results/uniaxial_meep_dispersion_highres_decision.json",
        "benchmark_results/uniaxial_meep_3d_phase_gate_decision.json",
        "benchmark_results/uniaxial_meep_3d_amplitude_gate_decision.json",
        "benchmark_results/acfo_ncs_reduced_release_suite.json",
    ]
    if fresh_dependency is not None:
        sources.append("benchmark_results/acfo_ncs_fresh_dependency_rerun.json")
    NOTES.write_text(
        "# ACFO NCS validation execution update — source notes\n\n"
        "## Report job\n\nTechnical audience; answer-first status of the current validation program and the remaining submission gates.\n\n"
        "## Chart map\n\n"
        "- High-NA charge sweep: ordered line chart, complex-field L2 by charge and cutoff policy; log scale; blue/gold/orange with markers.\n"
        "- Detector-aware WAXS: grouped bars for timing and peak-memory ratios by Nq; exact values retained in the adjacent table.\n"
        "- Missing-cone coverage: categorical bars for object NRMSE by illumination angle; this is an acquisition probe, not a causal proof or true second-axis result.\n\n"
        "- Protein-lattice q sampling: two-panel faceted dot/lollipop comparison with a shared log ratio axis; fixed-dq chunked first-total and fixed-range reusable hot protocols are kept in separate panels and not joined as one series.\n\n"
        "- High-q threshold strategy: equal-width measured speedup is separated from the log-time resolution panel; completed FINUFFT, censored lower bound, and holdout-gated projection use distinct markers/line styles.\n\n"
        "## Required technical-report structure mapping\n\n"
        "- Title and technical summary: page 1.\n"
        "- Key findings with visual/table evidence: High-NA, detector-aware WAXS, direct/source split, dense TIP3P, ODT, missing-cone and Maxwell sections.\n"
        "- Scope, data and metric definitions: direct-NDFT oracle, same-source versus exact-atom contracts, TIP3P frame/target definitions, and each section's adjacent evidence note.\n"
        "- Methodology: harmonic cutoff, exact-beta, lattice factorization, streaming, and Yee-source descriptions adjacent to their findings.\n"
        "- Limitations and robustness: negative controls and claim boundaries in every section.\n"
        "- Recommended next steps and further questions: final submission-gate table and prohibited-claim list.\n\n"
        "## Omitted chart\n\nODT radial-block memory/timing has only two observations, so an exact table is more honest than a chart. Dense TIP3P q-scaling has five exact lookup rows and its decisive comparison is a two-method timing table. The two-size repeated-crystal crossover remains an exact table; the new q-range and resolution sweeps provide the ordered multi-point visual. For the 30 paired 1M timing observations, the exact median, p05, p95 and CV table keeps the paired gate visible; a two-method box plot would obscure the paired speedup and add little decision value. Maxwell evidence mixes incompatible metrics and is also shown as a table.\n\n"
        "## Sources\n\n" + "\n".join(f"- `{item}`" for item in sources) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "schema": "acfo-ncs-execution-update-pdf-receipt-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pdf": PDF.relative_to(ROOT).as_posix(),
        "pages": len(pdf),
        "size_bytes": PDF.stat().st_size,
        "nonblank_pages": len(texts),
        "text_characters": len(combined),
        "anchors": anchors,
        "previews": preview_paths,
        "sources": sources,
        "delivery_mode": "direct PDF fallback because report-to-pdf sub-skill is unavailable in this runtime",
    }
    RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
