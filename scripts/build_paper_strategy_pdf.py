from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "benchmark_results"
CHART_DIR = OUT_DIR / "paper_strategy_assets"
PDF_PATH = OUT_DIR / "curved_manifold_fourier_paper_strategy_ko.pdf"
SOURCE_NOTES_PATH = OUT_DIR / "curved_manifold_fourier_paper_strategy_source_notes.md"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLOR_FAMILIES = {
    "blue": {"base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780", "light": "#CEDFFE"},
    "gold": {"base": "#FFE15B", "mid": "#B8A037", "dark": "#736422", "light": "#FFEA8F"},
    "orange": {"base": "#F0986E", "mid": "#CC6F47", "dark": "#804126", "light": "#FFBDA1"},
    "olive": {"base": "#A3D576", "mid": "#71B436", "dark": "#386411", "light": "#BEEB96"},
    "pink": {"base": "#F390CA", "mid": "#BD569B", "dark": "#8A3A6F", "light": "#F5BACC"},
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def load_waxs_qmax_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_DIR.glob("qmax_scaling_1m_dq0p160_q*.json")):
        payload = load_json(path)
        row = payload["rows"][0]
        rows.append(
            {
                "qmax": float(payload["case"]["qmax"]),
                "nq": int(payload["case"]["nq"]),
                "n_phi": int(row["grid"]["n_phi"]),
                "targets": int(payload["case"]["nq"]) * int(row["grid"]["n_phi"]),
                "rdep_s": float(row["rdep_analytic_s"]),
                "fused_s": float(row["rdep_fused_s"]),
                "nufft_s": float(row["nufft_s"]),
                "speedup_rdep": float(row["rdep_speedup_vs_nufft"]),
                "speedup_fused": float(row["rdep_fused_speedup_vs_nufft"]),
                "intensity_error": float(row["rdep_analytic_intensity_rel_l2_vs_dense"]),
            }
        )
    return sorted(rows, key=lambda row: row["qmax"])


def load_high_na_rows() -> list[dict[str, Any]]:
    path = OUT_DIR / "high_na_pupil_spectrum_option_matrix_eps1e-12.csv"
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] == "adaptive_sparse" and row["workload"] in {
                "benign_mixed_representative_modes16",
                "vortex_extra_h_representative_modes16",
                "vortex_extra_h_large_volume_modes16",
                "benign_mixed_large_modes64",
            }:
                rows.append(
                    {
                        "workload": row["workload"],
                        "speedup": fnum(row["speedup_total_vs_finufft"]),
                        "l2_finufft": fnum(row["field_l2_vs_finufft"]),
                        "required_h": row["required_h_values"],
                        "mode_rho_work": int(fnum(row["mode_rho_work"], 0)),
                    }
                )
    order = [
        "benign_mixed_representative_modes16",
        "vortex_extra_h_representative_modes16",
        "vortex_extra_h_large_volume_modes16",
        "benign_mixed_large_modes64",
    ]
    return sorted(rows, key=lambda row: order.index(row["workload"]))


def load_odt_rows() -> list[dict[str, Any]]:
    cases = [
        ("mid", OUT_DIR / "odt_gpu_reconstruction_compare_mid.json"),
        ("large", OUT_DIR / "odt_gpu_reconstruction_compare_large.json"),
        ("xlarge", OUT_DIR / "odt_gpu_reconstruction_compare_xlarge.json"),
    ]
    rows: list[dict[str, Any]] = []
    for label, path in cases:
        summary = load_json(path)["summary"]
        rows.append(
            {
                "case": label,
                "q_samples": int(summary["total_q_samples"]),
                "object_bins": int(summary["object_bins"]),
                "ours_update_ms": 1000.0 * float(summary["ours"]["median_update_s"]),
                "cufinufft_update_ms": 1000.0 * float(summary["cufinufft_gpu"]["median_update_s"]),
                "cpu_update_ms": None
                if summary.get("cpu_structured") is None
                else 1000.0 * float(summary["cpu_structured"]["median_update_s"]),
                "ours_loss": float(summary["ours"]["final_loss_rel"]),
                "cu_loss": float(summary["cufinufft_gpu"]["final_loss_rel"]),
                "ours_obj": float(summary["ours"]["final_object_rel_l2"]),
                "cu_obj": float(summary["cufinufft_gpu"]["final_object_rel_l2"]),
                "speedup_cu": float(summary["cufinufft_update_speedup_vs_ours"]),
                "speedup_cpu": None
                if summary.get("cpu_iter_speedup_vs_ours") is None
                else float(summary["cpu_iter_speedup_vs_ours"]),
            }
        )
    return rows


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "grid.color": TOKENS["grid"],
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Malgun Gothic", "DejaVu Sans", "Arial"],
        },
    )


def add_chart_header(fig, ax, title: str, subtitle: str) -> None:
    ax.set_title("")
    fig.subplots_adjust(top=0.80, left=0.16, right=0.96, bottom=0.16)
    left = ax.get_position().x0
    fig.text(left, 0.965, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=TOKENS["ink"])
    fig.text(left, 0.915, subtitle, ha="left", va="top", fontsize=9, color=TOKENS["muted"])
    sns.despine(ax=ax)


def save_charts(waxs: list[dict[str, Any]], high_na: list[dict[str, Any]], odt: list[dict[str, Any]]) -> dict[str, Path]:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    use_chart_theme()
    outputs: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    xs = [row["qmax"] for row in waxs]
    ax.plot(xs, [row["speedup_rdep"] for row in waxs], marker="o", color=COLOR_FAMILIES["blue"]["mid"], label="R-dependent analytic")
    ax.plot(xs, [row["speedup_fused"] for row in waxs], marker="o", color=COLOR_FAMILIES["orange"]["mid"], label="Fused Miller")
    ax.set_xlabel("qmax (A^-1)")
    ax.set_ylabel("Speedup vs chunked NUFFT")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0fx"))
    ax.legend(frameon=False, loc="upper left")
    add_chart_header(
        fig,
        ax,
        "WAXS high-q fixed-dq separation grows with detector range",
        "1M atoms; speedup compares structured cake-map solvers against chunked NUFFT.",
    )
    outputs["waxs"] = CHART_DIR / "waxs_qmax_speedup.png"
    fig.savefig(outputs["waxs"], bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    labels = [
        "mixed 16 modes",
        "vortex rep.",
        "vortex large",
        "mixed 64 modes",
    ]
    speeds = [row["speedup"] for row in high_na]
    ax.barh(labels, speeds, color=COLOR_FAMILIES["olive"]["base"], edgecolor=COLOR_FAMILIES["olive"]["dark"])
    ax.set_xlabel("Total speedup vs FINUFFT")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1fx"))
    for i, value in enumerate(speeds):
        ax.text(value + 0.04, i, f"{value:.2f}x", va="center", fontsize=8, color=TOKENS["ink"])
    add_chart_header(
        fig,
        ax,
        "High-NA adaptive harmonic path is competitive after correctness repair",
        "Auto-thread local rows; adaptive sparse fixes high-vortex misses and keeps benign rows accurate.",
    )
    outputs["high_na"] = CHART_DIR / "high_na_adaptive_speedup.png"
    fig.savefig(outputs["high_na"], bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    xvals = [row["q_samples"] for row in odt]
    ax.plot(xvals, [row["ours_update_ms"] for row in odt], marker="o", color=COLOR_FAMILIES["blue"]["mid"], label="ours GPU")
    ax.plot(xvals, [row["cufinufft_update_ms"] for row in odt], marker="o", color=COLOR_FAMILIES["orange"]["mid"], label="cuFINUFFT GPU")
    cpu_x = [row["q_samples"] for row in odt if row["cpu_update_ms"] is not None]
    cpu_y = [row["cpu_update_ms"] for row in odt if row["cpu_update_ms"] is not None]
    if cpu_x:
        ax.plot(cpu_x, cpu_y, marker="o", color=COLOR_FAMILIES["pink"]["mid"], label="CPU structured")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("q samples")
    ax.set_ylabel("Median update time (ms)")
    ax.legend(frameon=False, loc="upper left")
    add_chart_header(
        fig,
        ax,
        "ODT reconstruction updates stay in the real-time regime",
        "Eight-step steepest descent; update excludes diagnostic loss/object-error readout.",
    )
    outputs["odt"] = CHART_DIR / "odt_reconstruction_update_time.png"
    fig.savefig(outputs["odt"], bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    labels = [row["case"] for row in odt]
    speedups = [row["speedup_cu"] for row in odt]
    ax.barh(labels, speedups, color=COLOR_FAMILIES["gold"]["base"], edgecolor=COLOR_FAMILIES["gold"]["dark"])
    ax.set_xlabel("cuFINUFFT GPU / ours GPU update time")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0fx"))
    for i, value in enumerate(speedups):
        ax.text(value + 2.0, i, f"{value:.1f}x", va="center", fontsize=8, color=TOKENS["ink"])
    add_chart_header(
        fig,
        ax,
        "ODT advantage grows as detector sampling increases",
        "Plan-reuse cuFINUFFT type-3 GPU baseline on the local RTX 2070 SUPER.",
    )
    outputs["odt_speedup"] = CHART_DIR / "odt_gpu_speedup.png"
    fig.savefig(outputs["odt_speedup"], bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)
    return outputs


def register_fonts() -> tuple[str, str]:
    regular_candidates = [
        Path("C:/Windows/Fonts/NotoSansKR-VF.ttf"),
        Path("C:/Windows/Fonts/malgun.ttf"),
    ]
    bold_candidates = [
        Path("C:/Windows/Fonts/malgunbd.ttf"),
        Path("C:/Windows/Fonts/NotoSansKR-VF.ttf"),
    ]
    regular = next((path for path in regular_candidates if path.exists()), None)
    bold = next((path for path in bold_candidates if path.exists()), regular)
    if regular is None:
        return "Helvetica", "Helvetica-Bold"
    pdfmetrics.registerFont(TTFont("KoreanRegular", str(regular)))
    pdfmetrics.registerFont(TTFont("KoreanBold", str(bold)))
    return "KoreanRegular", "KoreanBold"


def make_styles(font: str, bold_font: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleK",
            parent=base["Title"],
            fontName=bold_font,
            fontSize=20,
            leading=25,
            alignment=TA_CENTER,
            textColor=colors.HexColor(TOKENS["ink"]),
            spaceAfter=14,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleK",
            parent=base["Normal"],
            fontName=font,
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor(TOKENS["muted"]),
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "H1K",
            parent=base["Heading1"],
            fontName=bold_font,
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#2E4780"),
            spaceBefore=10,
            spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "H2K",
            parent=base["Heading2"],
            fontName=bold_font,
            fontSize=11.5,
            leading=15,
            textColor=colors.HexColor(TOKENS["ink"]),
            spaceBefore=8,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "BodyK",
            parent=base["BodyText"],
            fontName=font,
            fontSize=9.3,
            leading=14.2,
            textColor=colors.HexColor(TOKENS["ink"]),
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "SmallK",
            parent=base["BodyText"],
            fontName=font,
            fontSize=8,
            leading=11,
            textColor=colors.HexColor(TOKENS["muted"]),
            spaceAfter=4,
        ),
        "bullet": ParagraphStyle(
            "BulletK",
            parent=base["BodyText"],
            fontName=font,
            fontSize=9.0,
            leading=13.2,
            leftIndent=8,
            firstLineIndent=0,
            textColor=colors.HexColor(TOKENS["ink"]),
        ),
    }


def p(text: str, styles: dict[str, ParagraphStyle], style: str = "body") -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), styles[style])


def bullet(items: list[str], styles: dict[str, ParagraphStyle]) -> ListFlowable:
    return ListFlowable(
        [ListItem(p(item, styles, "bullet"), leftIndent=10) for item in items],
        bulletType="bullet",
        leftIndent=12,
        bulletFontName=styles["body"].fontName,
        bulletFontSize=7,
        bulletColor=colors.HexColor("#5477C4"),
    )


def table(data: list[list[Any]], col_widths: list[float], styles: dict[str, ParagraphStyle]) -> Table:
    rendered = []
    for row in data:
        rendered.append([cell if hasattr(cell, "wrap") else p(str(cell), styles, "small") for cell in row])
    t = Table(rendered, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF1FE")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(TOKENS["ink"])),
                ("FONTNAME", (0, 0), (-1, 0), styles["body"].fontName),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D7DBE7")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def image(path: Path, width_cm: float = 15.8) -> Image:
    probe = Image(str(path))
    width = width_cm * cm
    height = probe.imageHeight * width / probe.imageWidth
    return Image(str(path), width=width, height=height)


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f} ms"


def build_story(
    waxs: list[dict[str, Any]],
    high_na: list[dict[str, Any]],
    odt: list[dict[str, Any]],
    charts: dict[str, Path],
) -> None:
    font, bold_font = register_fonts()
    styles = make_styles(font, bold_font)
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.55 * cm,
        leftMargin=1.55 * cm,
        topMargin=1.45 * cm,
        bottomMargin=1.45 * cm,
        title="Curved-Manifold Fourier Operator Paper Strategy",
        author="Codex",
    )

    odt_xlarge = odt[-1]
    waxs_last = waxs[-1]
    high_na_min = min(row["speedup"] for row in high_na)
    high_na_max = max(row["speedup"] for row in high_na)

    story: list[Any] = []
    story.append(p("Curved-Manifold Fourier Operator 논문 작성 전략", styles, "title"))
    story.append(
        p(
            "WAXS validation, High-NA optics application, ODT real-time reconstruction을 하나의 논문 구조로 묶기 위한 기술 보고서<br/>Generated: 2026-06-21",
            styles,
            "subtitle",
        )
    )

    story.append(p("기술 요약: 이제 논문 중심축은 충분히 선명하다", styles, "h1"))
    story.append(
        p(
            "현재 결과는 단순히 세 응용을 나열하는 수준이 아니라, 공통된 계산 구조를 단계적으로 증명하는 스토리로 구성할 수 있다. WAXS는 정확도와 scaling을 검증하는 home-field validation, High-NA는 다른 물리 영역으로의 확장 가능성, ODT는 반복 inverse problem에서의 실제 계산 impact를 담당한다.",
            styles,
        )
    )
    story.append(
        bullet(
            [
                f"<b>WAXS:</b> 1M atoms fixed-dq qmax sweep에서 R-dependent analytic solver가 chunked NUFFT 대비 {waxs[0]['speedup_rdep']:.1f}x에서 {waxs_last['speedup_rdep']:.1f}x까지 벌어지고, intensity error는 대략 1e-6 이하로 유지된다.",
                f"<b>High-NA:</b> adaptive pupil-spectrum 보정은 high-vortex failure를 direct-reference 기준으로 복구하며, auto-thread local rows에서 FINUFFT 대비 약 {high_na_min:.2f}x-{high_na_max:.2f}x total speedup을 보인다.",
                f"<b>ODT:</b> 같은 steepest-descent reconstruction loop에서 cuFINUFFT GPU Plan 대비 update가 {odt[0]['speedup_cu']:.1f}x에서 {odt_xlarge['speedup_cu']:.1f}x 빠르다. xlarge case는 {odt_xlarge['ours_update_ms']:.2f} ms/update, 즉 약 {1000.0 / odt_xlarge['ours_update_ms']:.0f} updates/s이다.",
                "<b>논문 메시지:</b> geometry-aware Fourier factorization은 curved measurement manifold에서 generic NUFFT보다 작은 실제 workload를 만들고, GPU에서는 그 구조가 반복 reconstruction에 특히 잘 맞는다.",
            ],
            styles,
        )
    )

    story.append(p("논문 spine: validation에서 real-time inverse problem으로 간다", styles, "h1"))
    story.append(
        table(
            [
                ["논문 역할", "Application", "증명해야 하는 것", "현재 근거", "남은 gate"],
                [
                    "Correctness anchor",
                    "WAXS",
                    "구조화된 cylindrical/cake-map factorization이 reference와 맞고 q-range가 커질수록 이득이 커진다.",
                    "fixed-dq qmax sweep; R-dependent analytic vs chunked NUFFT speedup 25x-59x; intensity error ~1e-6 이하.",
                    "realistic structure NPZ suite, Debye/direct/NUFFT 기준의 source-data package.",
                ],
                [
                    "Cross-field application",
                    "High-NA optics",
                    "같은 harmonic/curved-manifold 구조가 focal-field propagation과 inverse phase-mask demo로 확장된다.",
                    "adaptive pupil spectrum, Debye-Wolf/FINUFFT comparison, phase-mask self-consistency and physical demos.",
                    "vectorial Richards-Wolf validation, PyFocus/external package comparison의 더 엄밀한 조건 정리.",
                ],
                [
                    "Impact demonstration",
                    "ODT real-time reconstruction",
                    "반복 reconstruction loop에서 generic type-3 NUFFT GPU보다 구조적으로 싸다.",
                    "cuFINUFFT GPU Plan 대비 14.5x-104.9x update speedup; loss/object error trajectory matches.",
                    "warm-start time-series demo, noise/model mismatch sweep, real lab geometry parameter set.",
                ],
            ],
            [2.8 * cm, 2.8 * cm, 4.6 * cm, 4.7 * cm, 3.8 * cm],
            styles,
        )
    )

    story.append(PageBreak())
    story.append(p("현재 수치 근거: 세 영역이 서로 다른 약점을 보완한다", styles, "h1"))
    story.append(p("WAXS는 이 방법이 reference와 맞는지를 가장 직접적으로 보여준다. High-NA는 arbitrary pupil spectrum에서 생기는 validity boundary와 adaptive correction을 보여준다. ODT는 forward-only benchmark가 아니라 실제 reconstruction update loop에서의 비용 차이를 보여준다.", styles))
    story.append(image(charts["waxs"]))
    story.append(Spacer(1, 6))
    story.append(image(charts["high_na"]))
    story.append(PageBreak())
    story.append(image(charts["odt"]))
    story.append(Spacer(1, 6))
    story.append(image(charts["odt_speedup"]))

    story.append(p("ODT 결과는 단독 논문급이다", styles, "h1"))
    story.append(
        table(
            [
                ["Case", "q samples", "ours GPU update", "cuFINUFFT GPU update", "CPU structured update", "cuFINUFFT/ours", "final loss match"],
                *[
                    [
                        row["case"],
                        f"{row['q_samples']:,}",
                        fmt_ms(row["ours_update_ms"]),
                        fmt_ms(row["cufinufft_update_ms"]),
                        fmt_ms(row["cpu_update_ms"]),
                        f"{row['speedup_cu']:.1f}x",
                        f"{row['ours_loss']:.6f} vs {row['cu_loss']:.6f}",
                    ]
                    for row in odt
                ],
            ],
            [2.1 * cm, 2.6 * cm, 2.7 * cm, 3.1 * cm, 3.1 * cm, 2.4 * cm, 3.0 * cm],
            styles,
        )
    )
    story.append(
        p(
            "이 표의 핵심은 속도뿐 아니라 trajectory agreement다. final loss와 object relative L2가 거의 같기 때문에, 현재 결과는 단순한 approximate shortcut이 아니라 같은 reconstruction loop를 더 싼 operator로 실행했다는 증거가 된다.",
            styles,
        )
    )

    story.append(p("추천 논문 구조", styles, "h1"))
    story.append(
        bullet(
            [
                "<b>Title 후보:</b> Geometry-aware Fourier operators on curved measurement manifolds for scattering, focusing, and real-time tomographic reconstruction.",
                "<b>Abstract 구조:</b> curved manifold Fourier evaluation의 병목 제시 → cylindrical/harmonic factorization 설명 → WAXS validation → High-NA adaptive focal-field extension → ODT real-time reconstruction speedup → code/source-data availability.",
                "<b>Main Fig. 1:</b> 공통 수학 구조. source representation, curved detector/focal/ODT manifold, harmonic/cylindrical factorization, reusable operator build와 hot loop 분리.",
                "<b>Main Fig. 2:</b> WAXS validation and fixed-dq scaling. Debye/direct/NUFFT agreement, qmax scaling, memory reduction.",
                "<b>Main Fig. 3:</b> High-NA focal-field application. adaptive pupil-spectrum correction, vectorial extension, phase-only mask demos.",
                "<b>Main Fig. 4:</b> ODT reconstruction benchmark. ours GPU vs cuFINUFFT GPU Plan vs CPU structured, update time and trajectory agreement.",
                "<b>Main Fig. 5:</b> real-time/warm-start ODT demo. moving object or changing phase, previous frame initialization, updates-per-frame vs error.",
            ],
            styles,
        )
    )

    story.append(p("주장 수위와 저널 전략", styles, "h1"))
    story.append(
        table(
            [
                ["Target", "가능한 주장", "필수 보강", "위험"],
                [
                    "Nature Computational Science",
                    "Curved-manifold Fourier operator framework that transfers across scattering, focusing, and inverse tomographic reconstruction.",
                    "세 영역 모두 source-backed, external baselines, reproducible package, real-time ODT demo.",
                    "ODT가 synthetic self-consistency에 머물면 framework claim이 과해 보일 수 있음.",
                ],
                [
                    "Science Advances / Nature Communications",
                    "General computational physics/imaging method with strong cross-domain validation and real-time inverse-problem impact.",
                    "ODT warm-start demo, realistic geometry/noise, High-NA vectorial validation.",
                    "WAXS와 High-NA가 부록 수준으로 보이면 novelty가 ODT로만 축소됨.",
                ],
                [
                    "Communications Physics",
                    "Physics-facing structured Fourier method with WAXS/High-NA/ODT demonstrations.",
                    "accuracy and baseline fairness를 매우 명확히 정리.",
                    "computational-science급 broad impact보다는 physics methods로 읽힐 가능성.",
                ],
                [
                    "ODT standalone methods paper",
                    "Fixed lab geometry enables real-time ODT reconstruction updates beyond generic GPU NUFFT.",
                    "실제/realistic ODT geometry와 warm-start movie-style benchmark.",
                    "cross-field framework novelty는 약해지고 ODT community specificity가 커짐.",
                ],
            ],
            [3.3 * cm, 5.0 * cm, 5.0 * cm, 4.0 * cm],
            styles,
        )
    )

    story.append(PageBreak())
    story.append(p("앞으로 해야 할 일: acceptance criteria 중심", styles, "h1"))
    story.append(p("다음 작업은 더 빠르게 만드는 것보다, 논문 주장에 필요한 evidence gap을 닫는 쪽이 우선이다. 특히 ODT는 이미 강한 성능 신호가 있으므로 real-time reconstruction claim을 검증 가능한 demo로 바꾸는 것이 가장 중요하다.", styles))
    story.append(
        table(
            [
                ["Priority", "작업", "완료 기준", "논문상 역할"],
                [
                    "P0",
                    "ODT warm-start time-series reconstruction",
                    "이전 frame 초기값으로 4/8/16 update에서 error vs motion amplitude를 정리; 15-30 FPS budget에서 사용 가능한 영역 표시.",
                    "Fig. 5 real-time claim의 핵심.",
                ],
                [
                    "P0",
                    "ODT cuFINUFFT fairness audit",
                    "Plan reuse, dtype, eps, batch/memory 조건, setup/hot/update/diagnostic time 분리; source code와 result JSON 고정.",
                    "generic GPU NUFFT baseline에 대한 reviewer 방어.",
                ],
                [
                    "P0",
                    "WAXS validation dataset package",
                    "normalized NPZ schema, isotropic MD snapshot, anisotropic crystal controls, Debye/direct/NUFFT agreement table.",
                    "방법의 correctness home field.",
                ],
                [
                    "P1",
                    "High-NA vectorial validation",
                    "vectorial Richards-Wolf/Debye-Wolf reference와 ours agreement; PyFocus 또는 focal-field package baseline의 가정 차이 명시.",
                    "cross-field generality의 약점 보강.",
                ],
                [
                    "P1",
                    "Noise/model-mismatch ODT sweep",
                    "noise가 ours와 cuFINUFFT trajectory 차이를 유의미하게 바꾸지 않는지 확인; synthetic noise는 secondary evidence로 제한.",
                    "실험 robustness claim.",
                ],
                [
                    "P1",
                    "Reproducibility bundle",
                    "single command benchmark rerun, fixed seeds, machine/hardware table, raw JSON/CSV/source notes, figure generation scripts.",
                    "NCS/SA급 제출의 auditability.",
                ],
                [
                    "P2",
                    "Low-level GPU optimization estimate",
                    "현재 PyTorch 대비 fused CUDA/Triton이 줄일 수 있는 overhead를 profiling으로 분리; RTX 5090 projection은 보조로만 사용.",
                    "future ceiling과 real-time margin 설명.",
                ],
            ],
            [1.5 * cm, 4.2 * cm, 7.2 * cm, 4.5 * cm],
            styles,
        )
    )

    story.append(p("주의해야 할 claim boundary", styles, "h1"))
    story.append(
        bullet(
            [
                "NUFFT를 보편적으로 대체한다고 쓰면 안 된다. 현재 가장 방어 가능한 표현은 structured curved-manifold workloads에서 geometry-aware factorization이 더 작은 operator를 만든다는 것이다.",
                "High-NA는 아직 vectorial/full package replacement claim이 아니다. scalar/structured-grid 기반 application evidence로 쓰고, vectorial validation을 다음 gate로 둔다.",
                "ODT는 아직 synthetic self-consistency reconstruction이다. real-time/high-throughput claim은 warm-start time-series와 realistic lab geometry를 붙여야 제출 수준이 된다.",
                "GPU 결과는 local RTX 2070 SUPER snapshot이다. hardware-independent claim 대신 scaling trend와 workload structure를 강조해야 한다.",
                "PDF에 들어간 정확한 시간은 현재 repo/build/result snapshot에 묶인 값이다. 코드나 dependency가 바뀌면 rerun 후 figure/source notes를 갱신해야 한다.",
            ],
            styles,
        )
    )

    story.append(p("권장 실행 순서", styles, "h1"))
    story.append(
        bullet(
            [
                "1주차: ODT warm-start demo와 cuFINUFFT fairness audit을 먼저 완성한다. 이 결과가 논문 impact의 중심이다.",
                "2주차: WAXS validation package를 정리해서 method correctness를 reviewer가 따라갈 수 있게 만든다.",
                "3주차: High-NA vectorial/external package comparison을 최소 viable 수준으로 고정한다.",
                "4주차: Figure 1-5와 Methods skeleton을 작성하고, source-data/reproducibility package를 함께 묶는다.",
                "이후: NCS형 broad framework 원고와 ODT standalone 원고를 병렬 outline으로 유지하되, 먼저 broad framework가 reviewer에게 과하지 않은지 내부 검토한다.",
            ],
            styles,
        )
    )

    story.append(p("Source and reproducibility note", styles, "h1"))
    story.append(
        p(
            "이 보고서는 workspace의 기존 JSON/CSV/MD benchmark 산출물을 읽어 생성했다. 핵심 source는 WAXS fixed-dq qmax JSON, High-NA option/workload summaries, ODT GPU reconstruction compare JSON/CSV이다. 별도 source notes 파일에 경로와 claim boundary를 기록했다.",
            styles,
        )
    )

    doc.build(story)


def write_source_notes(waxs: list[dict[str, Any]], high_na: list[dict[str, Any]], odt: list[dict[str, Any]], charts: dict[str, Path]) -> None:
    lines = [
        "# Curved-manifold Fourier paper strategy source notes",
        "",
        f"Generated artifact: `{PDF_PATH.relative_to(ROOT)}`",
        "",
        "## Report contract",
        "",
        "- Audience: technical.",
        "- Delivery mode: PDF.",
        "- Scope: manuscript strategy and next-work plan based on current local WAXS, High-NA, and ODT benchmark artifacts.",
        "- Generated on: 2026-06-21.",
        "",
        "## Source files",
        "",
        "- `docs/r_dependent_analytic_final_summary.md`",
        "- `benchmark_results/qmax_scaling_1m_dq0p160_q2p13.json`",
        "- `benchmark_results/qmax_scaling_1m_dq0p160_q4p06.json`",
        "- `benchmark_results/qmax_scaling_1m_dq0p160_q6p30.json`",
        "- `benchmark_results/qmax_scaling_1m_dq0p160_q8p06.json`",
        "- `benchmark_results/high_na_pupil_spectrum_option_matrix_summary.md`",
        "- `benchmark_results/high_na_workload_matrix_summary.md`",
        "- `benchmark_results/high_na_pupil_spectrum_option_matrix_eps1e-12.csv`",
        "- `benchmark_results/odt_gpu_reconstruction_compare_mid.json`",
        "- `benchmark_results/odt_gpu_reconstruction_compare_large.json`",
        "- `benchmark_results/odt_gpu_reconstruction_compare_xlarge.json`",
        "- `benchmark_results/odt_gpu_reconstruction_compare_overview.md`",
        "",
        "## Generated chart assets",
        "",
        *[f"- `{path.relative_to(ROOT)}`" for path in charts.values()],
        "",
        "## Key extracted metrics",
        "",
        f"- WAXS fixed-dq qmax speedup range vs chunked NUFFT: `{waxs[0]['speedup_rdep']:.2f}x` to `{waxs[-1]['speedup_rdep']:.2f}x` for R-dependent analytic.",
        f"- High-NA adaptive sparse total speedup range vs FINUFFT: `{min(row['speedup'] for row in high_na):.2f}x` to `{max(row['speedup'] for row in high_na):.2f}x` on selected auto-thread rows.",
        f"- ODT cuFINUFFT GPU / ours GPU update speedup range: `{odt[0]['speedup_cu']:.2f}x` to `{odt[-1]['speedup_cu']:.2f}x`.",
        "",
        "## Claim boundaries",
        "",
        "- WAXS: claim geometry-aware WAXS/cake-map factorization, not universal NUFFT replacement.",
        "- High-NA: claim structured focal-field application and adaptive harmonic repair, not full vectorial/package replacement until external validation is complete.",
        "- ODT: claim synthetic fixed-geometry reconstruction-loop speedup now; real-time experimental claim requires warm-start time-series and realistic geometry/noise validation.",
    ]
    SOURCE_NOTES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    waxs = load_waxs_qmax_rows()
    high_na = load_high_na_rows()
    odt = load_odt_rows()
    charts = save_charts(waxs, high_na, odt)
    build_story(waxs, high_na, odt, charts)
    write_source_notes(waxs, high_na, odt, charts)
    print(
        json.dumps(
            {
                "pdf": str(PDF_PATH),
                "source_notes": str(SOURCE_NOTES_PATH),
                "charts": {key: str(value) for key, value in charts.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
