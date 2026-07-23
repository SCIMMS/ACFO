from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "acfo_ncs_validation_closure_20260714"
PDF_PATH = OUT_DIR / "ACFO_NCS_validation_closure_report_ko.pdf"
MD_PATH = OUT_DIR / "ACFO_NCS_validation_closure_report_ko.md"
SOURCE_PATH = OUT_DIR / "report_source_inventory.json"
CHART_DIR = OUT_DIR / "charts"
MPL_DIR = ROOT / ".matplotlib_cache"


SOURCES = {
    "claim_manifest": ROOT / "benchmark_results" / "acfo_claim_artifact_manifest.json",
    "waxs_detector": ROOT / "benchmark_results" / "waxs_detector_aware_decision.json",
    "waxs_protein": ROOT
    / "benchmark_results"
    / "waxs_protein_exact_beta_followup_decision.json",
    "aidt": ROOT / "benchmark_results" / "aidt_10hz_full700_opt_repeat.json",
    "odt_probe": ROOT
    / "benchmark_results"
    / "odt_banded_cartesian_final_packed_probe.json",
    "odt_scale": ROOT
    / "benchmark_results"
    / "odt_banded_cartesian_final_packed_full_timing.json",
    "odt_same_dtype": ROOT
    / "benchmark_results"
    / "odt_h28_rank16_adaptive_l1e-6_vs_cufinufft_abba30_final.json",
    "odt_direct_c64": ROOT
    / "benchmark_results"
    / "odt_cufinufft_matched_error_direct_subset.json",
    "odt_direct_c128": ROOT
    / "benchmark_results"
    / "odt_cufinufft_matched_error_direct_subset_c128.json",
    "odt_matched_full": ROOT
    / "benchmark_results"
    / "odt_cufinufft_matched_c128_full_pair5.json",
    "odt_c128_diagnostic": ROOT
    / "benchmark_results"
    / "odt_cufinufft_c128_full_plan_diagnostic.json",
    "odt_temporal": ROOT
    / "benchmark_results"
    / "odt_banded_cartesian_temporal_warm_start.json",
    "general_curvature": ROOT
    / "benchmark_results"
    / "linbo3_one_way_shg_cascade_holdout_angle25_r24.json",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value: float, digits: int = 3) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def percent(value: float, digits: int = 3) -> str:
    return f"{100.0 * value:.{digits}f}%"


def collect() -> dict[str, Any]:
    missing = [str(path) for path in SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing report evidence: " + ", ".join(missing))
    data = {name: load(path) for name, path in SOURCES.items()}
    waxs_row = next(row for row in data["waxs_detector"]["rows"] if row["nq"] == 512)
    aidt = data["aidt"]["summary"]
    probe_cases = {row["selected_n_z"]: row for row in data["odt_probe"]["cases"]}
    scale_cases = {row["selected_n_z"]: row for row in data["odt_scale"]["cases"]}
    temporal_rows = {
        (row["mode"], row["updates"]): row
        for row in data["odt_temporal"]["summary_rows"]
    }
    same = data["odt_same_dtype"]["summary"]
    c64 = data["odt_direct_c64"]
    c128 = data["odt_direct_c128"]
    matched = data["odt_matched_full"]
    curvature_row = data["general_curvature"]["rows"][0]
    shell = curvature_row["shells"][0]
    c64_min = min(
        row["worst_rel_l2_vs_direct"] for row in c64["cufinufft_sweep"]
    )
    metrics = {
        "waxs": {
            "detector_speedup": float(waxs_row["warm_speedup"]),
            "detector_p05_speedup": float(waxs_row["paired_timing"]["p05_speedup"]),
            "detector_complex_l2": float(waxs_row["complex_l2"]),
            "detector_memory_reduction": float(waxs_row["memory_reduction_ratio"]),
            **data["waxs_protein"]["metrics"],
        },
        "aidt": {
            "run_ms": 1000.0 * float(aidt["gpu_run_median_s"]),
            "hz": 1.0 / float(aidt["gpu_run_median_s"]),
            "setup_s": float(aidt["gpu_setup_s"]),
            "peak_mib": float(aidt["torch_peak_allocated_mib"]),
            "input_shape": aidt["processed_shape"],
            "depth_count": int(aidt["depth_count"]),
        },
        "odt": {
            "probe": {
                "worst_operator_l2": max(
                    row["operator_errors_vs_h36"]["worst_rel_l2"]
                    for row in probe_cases.values()
                ),
                "worst_reconstruction_difference": max(
                    row["remap_vs_ideal_reconstruction_rel_l2"]
                    for row in probe_cases.values()
                ),
                "z8_ms": 1000.0
                * probe_cases[8]["integrated_remap_mode_update_timing"]["median_s"],
                "z8_hz": float(probe_cases[8]["integrated_updates_per_second"]),
            },
            "scale": {
                str(z): {
                    "steady_ms": 1000.0
                    * row["integrated_steady_remap_mode_normal_pair_timing"]["median_s"],
                    "steady_hz": float(row["integrated_steady_updates_per_second"]),
                    "first_ms": 1000.0
                    * row["integrated_new_frame_first_update_timing"]["median_s"],
                    "first_hz": float(
                        row["integrated_new_frame_first_updates_per_second"]
                    ),
                    "peak_mib": float(row["memory"]["peak_allocated_mib"]),
                }
                for z, row in scale_cases.items()
            },
            "same_dtype": {
                "acfo_pair_ms": 1000.0 * float(same["ours_forward_adjoint_pair_s"]),
                "cufinufft_pair_s": float(same["cufinufft_forward_adjoint_pair_s"]),
                "speedup": float(same["ours_speedup_vs_cufinufft_pair"]),
                "forward_cross_l2": float(same["cufinufft_forward_rel_l2_vs_ours"]),
                "adjoint_cross_l2": float(same["cufinufft_adjoint_rel_l2_vs_ours"]),
            },
            "matched": {
                "direct_dot_error": float(c128["direct_reference"]["dot_error"]),
                "acfo_forward_l2": float(c128["acfo"]["forward_rel_l2_vs_direct"]),
                "acfo_adjoint_l2": float(c128["acfo"]["adjoint_rel_l2_vs_direct"]),
                "acfo_worst_l2": float(c128["acfo"]["worst_rel_l2_vs_direct"]),
                "c64_best_worst_l2": float(c64_min),
                "c128_eps": float(c128["matched_error_selection"]["eps"]),
                "c128_forward_l2": float(
                    c128["matched_error_selection"]["row"]["forward_rel_l2_vs_direct"]
                ),
                "c128_adjoint_l2": float(
                    c128["matched_error_selection"]["row"]["adjoint_rel_l2_vs_direct"]
                ),
                "c128_pair_s": float(matched["cufinufft"]["pair_median_s"]),
                "c128_pair_min_s": float(matched["cufinufft"]["pair_timing"]["min_s"]),
                "c128_pair_max_s": float(matched["cufinufft"]["pair_timing"]["max_s"]),
                "c128_pair_std_s": float(matched["cufinufft"]["pair_timing"]["std_s"]),
                "speedup": float(matched["speedup_acfo_vs_matched_cufinufft"]),
                "plan_used_mib": float(matched["memory_snapshots"][0]["used_mib"]),
            },
            "temporal": {
                "mode_error_min": min(
                    row["remapped_modes_vs_candidate_ideal_rel_l2"]
                    for row in data["odt_temporal"]["frame_data"]
                ),
                "mode_error_max": max(
                    row["remapped_modes_vs_candidate_ideal_rel_l2"]
                    for row in data["odt_temporal"]["frame_data"]
                ),
                "rows": temporal_rows,
                "peak_mib": float(data["odt_temporal"]["gpu_peak_allocated_mib"]),
            },
        },
        "general_curvature": {
            "cell_center_acfo_vs_direct": float(
                curvature_row["cell_center_source"]["acfo_vs_direct_relative_l2"]
            ),
            "exact_yee_acfo_vs_direct": float(
                curvature_row["exact_yee_source"]["acfo_vs_direct_relative_l2"]
            ),
            "source_transfer": float(
                curvature_row["exact_yee_source"][
                    "direct_vs_cell_center_direct_relative_l2"
                ]
            ),
            "fdtd_vs_exact_direct": float(
                shell["fdtd_vs_exact_yee_direct_relative_l2"]
            ),
            "fdtd_vs_exact_acfo": float(shell["fdtd_vs_exact_yee_acfo_relative_l2"]),
            "ordinary": float(
                shell["branch_fdtd_vs_exact_yee_direct_relative_l2"]["ordinary"]
            ),
            "extraordinary": float(
                shell["branch_fdtd_vs_exact_yee_direct_relative_l2"]["extraordinary"]
            ),
            "gates": len(data["general_curvature"]["gates"]),
        },
    }
    return {"raw": data, "metrics": metrics}


def configure_matplotlib() -> None:
    MPL_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPL_DIR)


def build_charts(metrics: dict[str, Any]) -> dict[str, Path]:
    configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    font_path = Path("C:/Windows/Fonts/malgun.ttf")
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": "#24313f",
            "axes.labelcolor": "#24313f",
            "xtick.color": "#52606d",
            "ytick.color": "#52606d",
        }
    )
    blue = "#2f6f9f"
    orange = "#d78232"
    neutral = "#7b8794"
    grid = "#d9e0e6"

    scale = metrics["odt"]["scale"]
    z_values = [64, 128, 256]
    steady = [scale[str(z)]["steady_ms"] for z in z_values]
    first = [scale[str(z)]["first_ms"] for z in z_values]
    x = np.arange(len(z_values))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    bars1 = ax.bar(x - width / 2, steady, width, color=blue, label="steady hot update")
    bars2 = ax.bar(x + width / 2, first, width, color=orange, label="new-frame first update")
    ax.axhline(100.0, color=neutral, linestyle="--", linewidth=1.3, label="10 Hz budget (100 ms)")
    ax.set_title("ODT integrated hot latency by reconstructed z depth", loc="left", fontsize=13, weight="bold")
    ax.set_ylabel("median latency (ms)")
    ax.set_xlabel("selected z slices in 256 × z × 256 object")
    ax.set_xticks(x, [str(z) for z in z_values])
    ax.set_ylim(0, max(first) * 1.18)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=3, loc="upper left", fontsize=8)
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 4,
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.text(
        0.01,
        0.01,
        "RTX 2070 SUPER; GPU-resident Cartesian camera payload. Steady includes remap+mode+normal pair; first update additionally includes data adjoint.",
        fontsize=7.4,
        color="#52606d",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    scale_path = CHART_DIR / "odt_integrated_latency_scaling.png"
    fig.savefig(scale_path, dpi=200)
    plt.close(fig)

    temporal = metrics["odt"]["temporal"]["rows"]
    updates = [1, 2, 3, 5]
    errors = [100.0 * temporal[("warm_start", u)]["mean_object_rel_l2"] for u in updates]
    hz = [temporal[("warm_start", u)]["median_hot_hz"] for u in updates]
    reference_error = 100.0 * temporal[("reference", 20)]["mean_object_rel_l2"]
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    bars = ax.bar(x=np.arange(len(updates)), height=errors, width=0.58, color=blue)
    ax.axhline(reference_error, color=orange, linewidth=1.7, linestyle="--", label="20-update cold reference")
    ax.set_title("Warm-start tracking error versus updates per frame", loc="left", fontsize=13, weight="bold")
    ax.set_ylabel("mean object relative L2 (%)")
    ax.set_xlabel("final packed updates per frame")
    ax.set_xticks(np.arange(len(updates)), [str(value) for value in updates])
    ax.set_ylim(0, max(errors) * 1.38)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    for bar, error, rate in zip(bars, errors, hz):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            error + 0.12,
            f"{error:.2f}%\n{rate:.2f} Hz",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.text(
        0.01,
        0.01,
        "Eight-frame 256×8×256 noiseless sequence; 1% motion, 0.02 rad drift. Each frame originates from an independent Cartesian detector field.",
        fontsize=7.4,
        color="#52606d",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    temporal_path = CHART_DIR / "odt_temporal_warm_start_tradeoff.png"
    fig.savefig(temporal_path, dpi=200)
    plt.close(fig)
    return {"scale": scale_path, "temporal": temporal_path}


def build_source_inventory(metrics: dict[str, Any], charts: dict[str, Path]) -> dict[str, Any]:
    def json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value

    inventory = {
        "schema": "acfo-ncs-validation-closure-report-sources-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audience": "technical",
        "delivery_surface": "PDF",
        "as_of_date": "2026-07-14 Asia/Seoul",
        "report_spine": {
            "question": "Which ACFO validation and feasibility gates are now locally closed, what exact claims do the measured numbers support, and what remains before publication?",
            "answer": "The local computational novelty case is supported, led by integrated and temporally tracked ODT. WAXS, aIDT and one general-curvature holdout provide bounded validation. External replication and experimental end-to-end evidence remain mandatory.",
        },
        "chart_map": [
            {
                "section": "ODT integrated scale",
                "question": "How does the integrated GPU-resident hot path scale with reconstructed z depth?",
                "family": "comparison",
                "type": "grouped bar with 10 Hz reference",
                "fields": ["selected_n_z", "steady_ms", "first_update_ms"],
                "takeaway": "64 z exceeds 10 Hz steady, 128 z is near the boundary, and 256 z is 6.66 Hz on the RTX 2070 SUPER.",
                "palette": "hard two-root cap: blue/orange plus neutral reference",
                "artifact": str(charts["scale"].relative_to(ROOT)),
            },
            {
                "section": "ODT temporal feasibility",
                "question": "What tracking error and frame rate result from 1/2/3/5 warm updates?",
                "family": "comparison and benchmark",
                "type": "bar with cold-reference line and direct Hz labels",
                "fields": ["updates", "mean_object_rel_l2", "median_hot_hz"],
                "takeaway": "One update reaches 15.91 Hz at 3.76% mean error; three updates approach the 20-update reference within 1.054x.",
                "palette": "hard two-root cap: blue bars and orange reference",
                "artifact": str(charts["temporal"].relative_to(ROOT)),
            },
        ],
        "visual_omissions": [
            "WAXS uses exact lookup tables because detector-aware, unit-cell, supercell and dense-MD rows have different denominators and baselines.",
            "aIDT has one frozen public-condition timing row, so a chart would add no information beyond the exact table.",
            "General curvature is a single no-refit holdout with several non-comparable error definitions, so a definition-first table is safer than a ranked chart.",
        ],
        "metrics": json_safe(metrics),
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in SOURCES.items()
        },
    }
    SOURCE_PATH.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return inventory


def make_pdf(metrics: dict[str, Any], charts: dict[str, Path]) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        Image,
        KeepTogether,
        PageBreak,
        PageTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    pdfmetrics.registerFont(TTFont("Malgun", "C:/Windows/Fonts/malgun.ttf"))
    pdfmetrics.registerFont(TTFont("MalgunBold", "C:/Windows/Fonts/malgunbd.ttf"))
    pdfmetrics.registerFontFamily(
        "Malgun", normal="Malgun", bold="MalgunBold", italic="Malgun", boldItalic="MalgunBold"
    )
    PAGE_W, PAGE_H = A4
    blue = colors.HexColor("#2F6F9F")
    dark = colors.HexColor("#24313F")
    mid = colors.HexColor("#52606D")
    pale = colors.HexColor("#EAF2F8")
    orange_pale = colors.HexColor("#FFF1E2")
    line = colors.HexColor("#D9E0E6")
    white = colors.white

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "title",
        parent=styles["Title"],
        fontName="MalgunBold",
        fontSize=24,
        leading=32,
        textColor=dark,
        alignment=TA_LEFT,
        wordWrap="CJK",
        spaceAfter=8 * mm,
    )
    subtitle = ParagraphStyle(
        "subtitle",
        fontName="Malgun",
        fontSize=11,
        leading=17,
        textColor=mid,
        wordWrap="CJK",
    )
    h1 = ParagraphStyle(
        "h1",
        fontName="MalgunBold",
        fontSize=17,
        leading=23,
        textColor=dark,
        wordWrap="CJK",
        spaceBefore=4 * mm,
        spaceAfter=4 * mm,
    )
    h2 = ParagraphStyle(
        "h2",
        fontName="MalgunBold",
        fontSize=12.5,
        leading=18,
        textColor=blue,
        wordWrap="CJK",
        spaceBefore=3 * mm,
        spaceAfter=2 * mm,
    )
    body = ParagraphStyle(
        "body",
        fontName="Malgun",
        fontSize=9.2,
        leading=15.2,
        textColor=dark,
        wordWrap="CJK",
        spaceAfter=2.4 * mm,
    )
    small = ParagraphStyle(
        "small",
        fontName="Malgun",
        fontSize=7.7,
        leading=11.5,
        textColor=mid,
        wordWrap="CJK",
    )
    bullet = ParagraphStyle(
        "bullet",
        parent=body,
        leftIndent=5 * mm,
        firstLineIndent=-3.3 * mm,
        bulletIndent=1.5 * mm,
        spaceAfter=1.5 * mm,
    )
    callout = ParagraphStyle(
        "callout",
        parent=body,
        fontSize=10,
        leading=16,
        borderColor=blue,
        borderWidth=1,
        borderPadding=8,
        backColor=pale,
        spaceBefore=2 * mm,
        spaceAfter=4 * mm,
    )
    warning = ParagraphStyle(
        "warning",
        parent=body,
        borderColor=colors.HexColor("#D78232"),
        borderWidth=1,
        borderPadding=8,
        backColor=orange_pale,
        spaceBefore=2 * mm,
        spaceAfter=4 * mm,
    )
    table_header = ParagraphStyle(
        "table_header",
        fontName="MalgunBold",
        fontSize=7.6,
        leading=10.5,
        textColor=white,
        alignment=TA_CENTER,
        wordWrap="CJK",
    )
    table_cell = ParagraphStyle(
        "table_cell",
        fontName="Malgun",
        fontSize=7.2,
        leading=10.4,
        textColor=dark,
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    table_num = ParagraphStyle(
        "table_num",
        parent=table_cell,
        alignment=TA_CENTER,
    )

    def P(text: str, style: ParagraphStyle = body) -> Paragraph:
        return Paragraph(text, style)

    def B(text: str) -> Paragraph:
        return Paragraph("• " + text, bullet)

    def cell(value: Any, numeric: bool = False) -> Paragraph:
        return Paragraph(str(value), table_num if numeric else table_cell)

    def header(values: Iterable[str]) -> list[Paragraph]:
        return [Paragraph(value, table_header) for value in values]

    def styled_table(
        rows: list[list[Any]], widths: list[float], repeat_rows: int = 1
    ) -> Table:
        table = Table(rows, colWidths=widths, repeatRows=repeat_rows, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), blue),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), 0.4, line),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, colors.HexColor("#F7F9FB")]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return table

    def page(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        if doc.page > 1:
            canvas.setStrokeColor(line)
            canvas.line(18 * mm, 15 * mm, PAGE_W - 18 * mm, 15 * mm)
            canvas.setFont("Malgun", 7.5)
            canvas.setFillColor(mid)
            canvas.drawString(18 * mm, 10 * mm, "ACFO NCS validation closure report")
            canvas.drawRightString(PAGE_W - 18 * mm, 10 * mm, str(doc.page))
        canvas.restoreState()

    frame = Frame(18 * mm, 18 * mm, PAGE_W - 36 * mm, PAGE_H - 32 * mm, id="main")
    template = PageTemplate(id="normal", frames=[frame], onPage=page)
    doc = BaseDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title="ACFO NCS validation closure report",
        author="ACFO validation workspace",
    )
    doc.addPageTemplates([template])
    story: list[Any] = []

    # Title page
    story.extend(
        [
            Spacer(1, 25 * mm),
            P("ACFO NCS 검증 종결 및 주장 경계", title),
            P(
                "WAXS · aIDT · ODT · general curvature의 로컬 검증 결과, 핵심 수치, 재현 경로와 남은 외부 gate",
                subtitle,
            ),
            Spacer(1, 14 * mm),
            P(
                "<b>결론.</b> 현재 로컬 계산 증거는 ACFO의 가능성과 논문 novelty를 지지한다. 특히 ODT는 Cartesian detector remap, final packed operator, 독립 direct matched-error, 8-frame warm-start tracking까지 하나의 주장 체계로 닫혔다. 다만 독립 장비 재현과 실제 acquisition-to-GPU 실험은 아직 남아 있으므로, ‘실험적 end-to-end real-time’이 아니라 ‘GPU-resident prepared reconstruction 가능성’으로 주장해야 한다.",
                callout,
            ),
            Spacer(1, 12 * mm),
            styled_table(
                [
                    header(["보고서 기준일", "주요 GPU", "범위", "상태"]),
                    [
                        cell("2026-07-14 (Asia/Seoul)"),
                        cell("NVIDIA RTX 2070 SUPER, 8 GB"),
                        cell("저장된 로컬 결과와 이번 통합 실행"),
                        cell("내부 계산 gate 종결 / 외부 검증 대기"),
                    ],
                ],
                [38 * mm, 44 * mm, 54 * mm, 38 * mm],
            ),
            Spacer(1, 28 * mm),
            P("작성 원칙", h2),
            B("setup, hot execution, 첫 frame update, acquisition/transfer를 분리했다."),
            B("ACFO와 NUFFT를 서로의 reference로 두지 않고 가능한 곳에서 독립 direct 기준을 사용했다."),
            B("속도 이득이 없는 TIP3P Nq=512 결과도 포함해 유리한 조건만 선택하는 해석을 피했다."),
        ]
    )
    story.append(PageBreak())

    # Technical summary
    story.append(P("기술 요약: 핵심 novelty는 ODT에서 가장 강하게 남는다", h1))
    story.append(
        P(
            "외부 AI의 초기 계획은 재현 package를 우선 만들고 WAXS exact-beta와 ODT warm-start를 새로 닫자는 방향이었다. 실제 repository를 감사한 결과 package와 WAXS exact-beta의 상당 부분은 이미 존재했다. 따라서 이번 작업은 가장 큰 공백이었던 <b>ODT detector/operator 통합, matched-error 공정성, temporal tracking</b>에 집중했고 세 gate를 모두 통과시켰다."
        )
    )
    summary_rows = [
        header(["영역", "현재 근거", "논문에서 가능한 주장", "남은 핵심 gate"]),
        [
            cell("WAXS"),
            cell("protein crystal exact-beta/direct, 1M ordered lattice, detector-aware local timing"),
            cell("곡률·detector-aware validation/control"),
            cell("독립 장비 timing, 실제 crystal photon/orientation"),
        ],
        [
            cell("aIDT"),
            cell("24×700×700 → 700×700×35 GPU core, 10.31 Hz"),
            cell("고정 geometry processing core의 10 Hz급 가능성"),
            cell("preprocess/H2D/acquisition 포함 end-to-end"),
        ],
        [
            cell("ODT"),
            cell("Cartesian remap + final packed + direct fairness + temporal"),
            cell("GPU-resident prepared reconstruction 및 warm tracking"),
            cell("독립 GPU, noise/model mismatch, 실제 complex camera"),
        ],
        [
            cell("general curvature"),
            cell("38° calibration → 25° no-refit SHG cascade holdout"),
            cell("임의 곡률/비선형 one-way cascade 가능성"),
            cell("복수 holdout/material, full nonlinear feedback"),
        ],
    ]
    story.append(styled_table(summary_rows, [27 * mm, 56 * mm, 53 * mm, 38 * mm]))
    story.append(Spacer(1, 4 * mm))
    story.append(P("이번 작업에서 새로 닫힌 항목", h2))
    story.append(B("ODT Cartesian detector → cached remap → angular modes → final H28/rank16/adaptive-L forward/adjoint를 한 실행으로 연결했다."))
    story.append(B("작은 exact complex128 exponent sum으로 ACFO와 cuFINUFFT의 forward/adjoint 오차를 각각 측정해 matched-error 기준을 만들었다."))
    story.append(B("물리 Cartesian camera 8-frame sequence에서 warm/cold/20-update reference를 비교해 1 update/frame 15.91 Hz를 보였다."))
    story.append(B("WAXS oriented protein object를 dilute single protein이 아니라 ordered protein crystal/single-crystal nanocrystal로 고정했다."))
    story.append(PageBreak())

    # Integrated ODT
    o = metrics["odt"]
    story.append(P("ODT 통합 경로는 정확도와 규모별 실행을 동시에 닫았다", h1))
    story.append(
        P(
            f"작은 물리 probe에서 H36 structured reference 대비 최악 operator 오차는 <b>{fmt(o['probe']['worst_operator_l2'])}</b>, remap과 ideal input의 reconstruction 차이는 <b>{percent(o['probe']['worst_reconstruction_difference'])}</b>였다. z=8에서는 remap과 mode reduction을 포함한 hot update가 <b>{o['probe']['z8_ms']:.2f} ms ({o['probe']['z8_hz']:.2f} Hz)</b>였다. 따라서 detector interpolation과 packed approximation을 분리해서도, 결합해서도 gate가 통과했다."
        )
    )
    story.append(Image(str(charts["scale"]), width=174 * mm, height=95 * mm))
    story.append(
        P(
            "그림은 동일한 final packed path를 64/128/256 z로 확장한 timing-only 결과다. steady row는 remap+mode+normal pair, first-update row는 새 frame의 RHS adjoint까지 포함한다. 64 z는 steady 12.27 Hz, 128 z는 9.64 Hz로 10 Hz 경계에 있고, full 256 z는 6.66 Hz다. 이는 수렴된 전체 reconstruction 시간이 아니라 반복 solver의 1회 update 비용이다.",
            small,
        )
    )
    scale_table = [header(["object", "steady", "first update", "peak allocation"])]
    for z in (64, 128, 256):
        row = o["scale"][str(z)]
        scale_table.append(
            [
                cell(f"256×{z}×256"),
                cell(f"{row['steady_ms']:.2f} ms / {row['steady_hz']:.2f} Hz", True),
                cell(f"{row['first_ms']:.2f} ms / {row['first_hz']:.2f} Hz", True),
                cell(f"{row['peak_mib']:.1f} MiB", True),
            ]
        )
    story.append(styled_table(scale_table, [39 * mm, 49 * mm, 49 * mm, 37 * mm]))
    story.append(PageBreak())

    # Matched error
    story.append(P("matched-error 비교는 기존 81.24×를 약화시키지 않고 오히려 경계를 명확히 했다", h1))
    same = o["same_dtype"]
    match = o["matched"]
    story.append(
        P(
            f"동일 complex64·동시상주 AB/BA 30회에서 ACFO pair는 <b>{same['acfo_pair_ms']:.3f} ms</b>, cuFINUFFT는 <b>{same['cufinufft_pair_s']:.3f} s</b>로 <b>{same['speedup']:.2f}×</b>였다. 그러나 cross-backend 차이({fmt(same['forward_cross_l2'])}/{fmt(same['adjoint_cross_l2'])})는 독립 exact error가 아니었다. 이번에는 동일 q/object에 대한 literal complex128 exponent sum을 기준으로 두 방법을 각각 평가했다."
        )
    )
    matched_table = [
        header(["방법", "정밀도/설정", "forward direct L2", "adjoint direct L2", "production pair"]),
        [
            cell("ACFO final packed"),
            cell("complex64, H28/rank16/adaptive-L"),
            cell(fmt(match["acfo_forward_l2"]), True),
            cell(fmt(match["acfo_adjoint_l2"]), True),
            cell(f"{same['acfo_pair_ms']:.3f} ms", True),
        ],
        [
            cell("cuFINUFFT same dtype"),
            cell("complex64, eps sweep"),
            cell("direction별 match 없음"),
            cell(f"best worst={fmt(match['c64_best_worst_l2'])}"),
            cell(f"{same['cufinufft_pair_s']:.3f} s", True),
        ],
        [
            cell("cuFINUFFT matched"),
            cell(f"complex128, eps={match['c128_eps']:.0e}"),
            cell(fmt(match["c128_forward_l2"]), True),
            cell(fmt(match["c128_adjoint_l2"]), True),
            cell(f"{match['c128_pair_s']:.3f} s", True),
        ],
    ]
    story.append(styled_table(matched_table, [33 * mm, 49 * mm, 32 * mm, 32 * mm, 28 * mm]))
    story.append(Spacer(1, 3 * mm))
    story.append(
        P(
            f"독립 direct dot error는 {fmt(match['direct_dot_error'])}였다. ACFO complex64의 최악 direct error {fmt(match['acfo_worst_l2'])}를 방향별로 처음 만족한 cuFINUFFT 조건은 complex128/eps=10⁻⁷이었다. full 256³에서 cuFINUFFT pair 5회 median은 <b>{match['c128_pair_s']:.3f} s</b> (min {match['c128_pair_min_s']:.3f}, max {match['c128_pair_max_s']:.3f}, std {match['c128_pair_std_s']:.3f})이고, frozen ACFO median 대비 <b>{match['speedup']:.2f}×</b>다.",
                callout,
            )
    )
    story.append(
        P(
            f"중요 caveat: matched cuFINUFFT plan만으로도 plan 직후 약 {match['plan_used_mib']:.0f} MiB를 사용해 8 GB GPU에서 ACFO plan과 동시상주하지 못했다. 따라서 310.94×는 동일 GPU의 <b>별도 프로세스</b> 비교이고, 81.24×만 동시상주 AB/BA 결과다. 두 수치를 하나로 합치면 안 된다.",
            warning,
        )
    )
    story.append(PageBreak())

    # Temporal
    story.append(P("한 번의 warm update는 15.91 Hz에서 구조 변화를 추적했다", h1))
    temporal = o["temporal"]
    warm1 = temporal["rows"][("warm_start", 1)]
    warm3 = temporal["rows"][("warm_start", 3)]
    warm5 = temporal["rows"][("warm_start", 5)]
    story.append(
        P(
            f"각 frame은 121-view 320×320 Cartesian complex detector field에서 시작했고, remap된 modes와 candidate ideal modes의 차이는 {percent(temporal['mode_error_min'])}–{percent(temporal['mode_error_max'])}였다. 초기 frame은 cold 20 updates로 복원한 뒤, 이후 7 frames에서 이전 x와 A x를 재사용했다. 1 update/frame은 <b>{warm1['median_total_hot_s']*1000:.2f} ms ({warm1['median_hot_hz']:.2f} Hz)</b>, 평균 object L2 <b>{percent(warm1['mean_object_rel_l2'], 2)}</b>였다."
        )
    )
    story.append(Image(str(charts["temporal"]), width=174 * mm, height=95 * mm))
    story.append(
        P(
            "1 update는 속도 우선 tracking, 3 updates는 reference 근접, 5 updates는 이 sequence에서 20-update cold reference보다 낮은 평균 오차를 보였다. reference는 동일 approximate/remapped operator의 20-update cold solve이며 독립 inverse ground truth는 아니다.",
            small,
        )
    )
    temp_table = [header(["warm updates", "median hot", "mean object L2", "cold 대비", "20-update reference 대비"])]
    for update in (1, 2, 3, 5):
        row = temporal["rows"][("warm_start", update)]
        temp_table.append(
            [
                cell(str(update), True),
                cell(f"{row['median_total_hot_s']*1000:.2f} ms / {row['median_hot_hz']:.2f} Hz", True),
                cell(percent(row["mean_object_rel_l2"], 2), True),
                cell(f"{row['mean_object_error_vs_cold_ratio']:.3f}×", True),
                cell(f"{row['mean_object_error_vs_reference_ratio']:.3f}×", True),
            ]
        )
    story.append(styled_table(temp_table, [24 * mm, 49 * mm, 34 * mm, 31 * mm, 36 * mm]))
    story.append(PageBreak())

    # WAXS
    story.append(P("WAXS는 새로운 범용 가속 주장보다 정확도와 detector-aware 검증 역할이 적절하다", h1))
    w = metrics["waxs"]
    unit = w["protein_unit_cell"]
    crystal = w["ordered_protein_supercells"]
    prepared = w["prepared_1m_abba"]
    dense = w["dense_md_control"]
    story.append(
        P(
            "실험 object는 <b>ordered lysozyme protein crystal 또는 single-crystal protein nanocrystal</b>로 고정한다. dilute isolated single protein은 WAXS photon count가 부족할 수 있어 현재 실험 claim에서 제외한다. 이 정의에 따르면 추가 dense protein-MD local rerun은 필요하지 않다."
        )
    )
    waxs_table = [
        header(["검증 층", "규모/조건", "주요 수치", "해석"]),
        [
            cell("protein exact-beta"),
            cell(f"{unit['atoms']:,} atoms, q={unit['q_inv_angstrom']} Å⁻¹"),
            cell(f"direct complex L2 {fmt(unit['exact_beta_complex_l2_vs_direct'])}"),
            cell("protein unit-cell exact-coordinate correctness"),
        ],
        [
            cell("ordered supercell"),
            cell(f"{crystal['5x5x5_atoms']:,} atoms"),
            cell(f"direct-subset L2 {fmt(crystal['5x5x5_direct_subset_complex_l2'])}"),
            cell("perfect crystal/lattice control; known specialization"),
        ],
        [
            cell("prepared AB/BA"),
            cell("1M crystal, Nq=512, 10/30"),
            cell(f"median {prepared['paired_speedup_median']:.2f}×; p05 {prepared['paired_speedup_p05']:.2f}×"),
            cell("local timing gate PASS; external replication pending"),
        ],
        [
            cell("detector-aware"),
            cell("EIGER2 X 4M envelope, Nq=512"),
            cell(f"{w['detector_speedup']:.3f}×; complex L2 {fmt(w['detector_complex_l2'])}"),
            cell("same-bin curved detector comparison"),
        ],
        [
            cell("dense MD negative control"),
            cell(f"TIP3P {dense['frames']} frames, {dense['atoms']:,} atoms"),
            cell(f"Nq=512 exact-beta {dense['nq512_exact_beta_s']:.3f} s vs FINUFFT {dense['nq512_finufft_hot_median_s']:.3f} s"),
            cell("generic dense exact-coordinate workload에서는 FINUFFT 우세"),
        ],
    ]
    story.append(styled_table(waxs_table, [35 * mm, 48 * mm, 48 * mm, 43 * mm]))
    story.append(
        P(
            "따라서 WAXS에서 주장해야 할 것은 ‘ACFO가 항상 NUFFT보다 빠르다’가 아니다. ordered lattice와 repeated geometry가 있는 protein crystal, detector curvature가 큰 high-q 구간, 그리고 prepared reuse가 결합될 때 이득이 생긴다는 조건부 결과다. TIP3P Nq=512 FAIL은 이 경계를 정직하게 고정한다.",
            warning,
        )
    )
    story.append(PageBreak())

    # aIDT and general curvature
    story.append(P("aIDT는 10 Hz급 core를, general curvature는 single holdout 가능성을 제공한다", h1))
    a = metrics["aidt"]
    story.append(P("aIDT: GPU-resident processing core", h2))
    story.append(
        P(
            f"public 24-illumination 조건에서 24×700×700 입력을 700×700×35 volume으로 복원하는 GPU core median은 <b>{a['run_ms']:.3f} ms ({a['hz']:.3f} Hz)</b>였다. setup {a['setup_s']:.3f} s, peak allocation {a['peak_mib']:.1f} MiB다. 이 수치는 실제 장비의 camera transfer, preprocessing, scheduling과 output statistics를 제외한다."
        )
    )
    story.append(P("General curvature: 38° calibration → 25° no-refit holdout", h2))
    g = metrics["general_curvature"]
    curvature_table = [
        header(["비교", "relative L2", "의미"]),
        [cell("cell-center ACFO vs direct"), cell(percent(g["cell_center_acfo_vs_direct"]), True), cell("factorization/contraction error")],
        [cell("exact Yee-source ACFO vs direct"), cell(percent(g["exact_yee_acfo_vs_direct"]), True), cell("Yee field를 사용한 ACFO contraction")],
        [cell("exact Yee direct vs cell-center direct"), cell(percent(g["source_transfer"]), True), cell("연속장→Yee source transfer/discretization")],
        [cell("SH FDTD vs exact-Yee direct"), cell(percent(g["fdtd_vs_exact_direct"]), True), cell("one-way impressed-current propagation 포함")],
        [cell("SH FDTD vs exact-Yee ACFO"), cell(percent(g["fdtd_vs_exact_acfo"]), True), cell("전체 cascade 차이; 주요 4.449% 수치")],
    ]
    story.append(styled_table(curvature_table, [63 * mm, 32 * mm, 79 * mm]))
    story.append(
        P(
            f"10/10 gates가 통과했지만, 이는 homogeneous LiNbO₃의 사전 고정된 r24 한 사례다. 4.449%를 ACFO tensor contraction 자체의 오차로 읽으면 안 된다. exact Yee-source ACFO/direct 차이는 0.798%이고, 더 큰 부분은 source transfer와 SH FDTD propagation을 포함한다.",
            callout,
        )
    )
    story.append(PageBreak())

    # Definitions/methodology
    story.append(P("측정 정의와 검증 설계", h1))
    story.append(P("시간 경계", h2))
    story.append(B("setup: geometry/context/plan/setpts/cache 생성. 반복 frame의 hot time과 분리한다."))
    story.append(B("steady hot update: GPU에 complex camera가 이미 있고 geometry가 고정된 상태의 remap+mode+forward/adjoint normal pair."))
    story.append(B("new-frame first update: steady 경로에 data adjoint/RHS 형성을 한 번 추가한다."))
    story.append(B("acquisition-to-GPU: detector readout, network/PCIe transfer, hologram demodulation. 현재 10 Hz core 주장에 포함되지 않는다."))
    story.append(P("정확도 기준", h2))
    story.append(B("ODT packed approximation: H36 full-rank dense-L structured reference."))
    story.append(B("ODT matched-error: 동일 source/q의 literal complex128 exponent sum; ACFO와 cuFINUFFT를 독립적으로 채점."))
    story.append(B("WAXS protein: literal direct NDFT unit cell, ordered-supercell direct subset, FINUFFT cross-check."))
    story.append(B("general curvature: direct χ(2) tensor contraction, exact injected Yee source, SH impressed-current FDTD를 단계별로 분리."))
    story.append(P("통계와 hardware", h2))
    story.append(B("동일 dtype ODT headline은 5 warmups + 30 alternating AB/BA pairs."))
    story.append(B("matched complex128 cuFINUFFT full run은 VRAM 제약 때문에 별도 프로세스 2 warmups + 5 measured pairs."))
    story.append(B("WAXS prepared crystal headline은 10 warmups + 30 measured AB/BA pairs; 독립 machine은 아직 없음."))
    story.append(B("모든 주요 GPU 수치는 장기간 사용한 RTX 2070 SUPER 8 GB에서 얻었다. 더 빠른 GPU projection은 measured claim이 아니다."))
    story.append(PageBreak())

    # Limitations
    story.append(P("제한과 robustness: 로컬 novelty는 지지되지만 실험적 종결은 아니다", h1))
    limitations = [
        "독립 장비 재현이 없다. 동일 시스템의 열/driver/환경 특성이 모든 timing에 공통으로 남는다.",
        "ODT temporal sequence는 8 frames, 1% motion, 0.02 rad drift, noise 없음이다. detector noise, demodulation error, aberration, multiple scattering을 포함하지 않는다.",
        "matched-error 310.94×는 complex128 cuFINUFFT와 complex64 ACFO의 별도 프로세스 비교다. 동시상주 공정성은 complex64 81.24× row가 담당한다.",
        "ODT 1 update/frame은 tracking update이지 수렴된 3D reconstruction 전체 시간이 아니다. 허용 오차에 따라 1/2/3/5 updates를 선택해야 한다.",
        "aIDT 10.31 Hz는 GPU core이며 camera-to-volume end-to-end가 아니다.",
        "WAXS perfect protein lattice factorization은 알려진 exact specialization이다. 이것만으로 ACFO novelty를 주장하지 않는다.",
        "general curvature는 단일 no-refit holdout이다. arbitrary curvature의 범용성은 복수 geometry/material holdout이 필요하다.",
    ]
    for item in limitations:
        story.append(B(item))
    story.append(
        P(
            "공유 가능성 판정: <b>Share with caveats.</b> 내부 계산 결과와 주장 경계는 논문 설계에 사용할 수 있지만, ‘실제 장비에서 end-to-end real-time’과 ‘외부 재현된 speedup’ 문구는 아직 사용할 수 없다.",
            warning,
        )
    )
    story.append(PageBreak())

    # Next steps
    story.append(P("다음 단계는 더 많은 로컬 최적화보다 외부·실험 검증이 우선이다", h1))
    next_table = [
        header(["우선순위", "작업", "통과 기준", "이유"]),
        [cell("P0"), cell("독립 GPU 재현"), cell("frozen WAXS 10/30, ODT same-dtype와 matched protocol 재실행"), cell("논문 timing의 마지막 공통 gate")],
        [cell("P0"), cell("실제 ODT complex camera adapter"), cell("camera→GPU→demod/remap→1/3 update latency와 error"), cell("end-to-end real-time 문구를 결정")],
        [cell("P0"), cell("ODT noise/model mismatch sweep"), cell("noise, aberration, geometry error에서 tracking drift와 failure boundary"), cell("8-frame noiseless 결과의 범위 확장")],
        [cell("P1"), cell("aIDT transfer 포함"), cell("pinned memory/stream overlap과 preprocess를 포함한 frame budget"), cell("10.31 Hz core를 장비 workflow로 연결")],
        [cell("P1"), cell("general-curvature holdout 확대"), cell("복수 각도/곡률/material에서 no-refit 재현"), cell("single-case에서 broader claim으로 확장")],
        [cell("P1"), cell("release evidence freeze"), cell("새 artifact 28개를 새 release manifest/ZIP에 포함"), cell("현재 prior release에 최신 결과가 없음")],
    ]
    story.append(styled_table(next_table, [18 * mm, 39 * mm, 72 * mm, 45 * mm]))
    story.append(P("현재 하지 않을 작업", h2))
    story.append(B("dense disordered protein-MD exact-beta: ordered protein-crystal claim에 필수적이지 않다."))
    story.append(B("RTX 5090 projection을 measured result처럼 제시: 실제 external rerun 전에는 전망으로만 둔다."))
    story.append(B("모든 ODT size·noise·iteration 조합을 소진: 가능성 논문에는 frozen representative gates가 충분하다."))
    story.append(P("추가로 결정해야 할 질문", h2))
    story.append(B("실험 ODT의 camera pixel/view/NA와 허용 tracking error는 얼마인가?"))
    story.append(B("외부 재현에 사용할 GPU는 동일 세대, RTX 5090, 또는 32 GB급 중 무엇인가?"))
    story.append(B("general curvature를 main novelty로 둘지, cross-domain validation으로 둘지?"))
    story.append(B("protein crystal의 실제 orientation 안정성과 photon budget을 어떤 장비에서 검증할 수 있는가?"))
    story.append(PageBreak())

    # Reproduction
    story.append(P("재현 명령과 증거 파일", h1))
    story.append(
        P(
            "아래 명령은 repository root의 <font name='MalgunBold'>.venv</font> Python을 기준으로 한다. 큰 production timing은 수 분 이상 걸릴 수 있으며, 외부 장비 재현 시 raw JSON을 덮어쓰지 말고 machine ID가 포함된 새 파일로 저장해야 한다."
        )
    )
    commands = [
        ("ODT 통합 probe", ".\\.venv\\Scripts\\python.exe scripts\\benchmark_odt_banded_cartesian_final_packed.py"),
        ("ODT 64/128/256 timing", ".\\.venv\\Scripts\\python.exe scripts\\benchmark_odt_banded_cartesian_final_packed_full_timing.py"),
        ("ODT direct matched sweep", ".\\.venv\\Scripts\\python.exe scripts\\validate_odt_cufinufft_matched_error.py"),
        ("ODT matched complex128", ".\\.venv\\Scripts\\python.exe scripts\\validate_odt_cufinufft_matched_error.py --cufinufft-dtype complex128 --eps-values 1e-4,3e-5,1e-5,3e-6,1e-6,3e-7,1e-7,3e-8,1e-8"),
        ("ODT full matched pair", ".\\.venv\\Scripts\\python.exe scripts\\benchmark_odt_cufinufft_only.py --repeats 1 --warmups 0"),
        ("ODT temporal", ".\\.venv\\Scripts\\python.exe scripts\\benchmark_odt_banded_cartesian_temporal_warm_start.py"),
        ("WAXS follow-up decision", ".\\.venv\\Scripts\\python.exe scripts\\build_waxs_protein_exact_beta_followup_decision.py"),
        ("claim manifest", ".\\.venv\\Scripts\\python.exe scripts\\build_acfo_claim_artifact_manifest.py"),
        ("이 PDF", ".\\.venv\\Scripts\\python.exe scripts\\build_acfo_ncs_validation_closure_report.py"),
    ]
    cmd_rows = [header(["항목", "명령"])]
    for label, command in commands:
        cmd_rows.append([cell(label), cell(command)])
    story.append(styled_table(cmd_rows, [43 * mm, 131 * mm]))
    story.append(Spacer(1, 3 * mm))
    story.append(P("핵심 evidence", h2))
    evidence = [
        "benchmark_results/odt_banded_cartesian_final_packed_probe.json",
        "benchmark_results/odt_banded_cartesian_final_packed_full_timing.json",
        "benchmark_results/odt_cufinufft_matched_error_direct_subset_c128.json",
        "benchmark_results/odt_cufinufft_matched_c128_full_pair5.json",
        "benchmark_results/odt_banded_cartesian_temporal_warm_start.json",
        "benchmark_results/waxs_protein_exact_beta_followup_decision.json",
        "benchmark_results/aidt_10hz_full700_opt_repeat.json",
        "benchmark_results/linbo3_one_way_shg_cascade_holdout_angle25_r24.json",
        "benchmark_results/acfo_claim_artifact_manifest.json",
    ]
    for path in evidence:
        story.append(B(path))
    story.append(
        P(
            "전체 source hash와 chart contract는 report_source_inventory.json에 저장했다. 이 PDF의 수치는 해당 JSON과 위 evidence 파일에서 다시 계산된다.",
            small,
        )
    )
    doc.build(story)


def write_markdown(metrics: dict[str, Any]) -> None:
    warm1 = metrics["odt"]["temporal"]["rows"][("warm_start", 1)]
    lines = [
        "# ACFO NCS 검증 종결 및 주장 경계",
        "",
        "## 기술 요약",
        "",
        "로컬 계산 증거는 ACFO의 가능성과 논문 novelty를 지지한다. ODT의 Cartesian detector 통합, matched-error 공정성, temporal warm-start gate가 모두 닫혔다. 외부 장비 재현과 실제 acquisition-to-GPU 검증은 남아 있다.",
        "",
        "## 핵심 수치",
        "",
        f"- ODT integrated probe: worst H36 operator L2 `{metrics['odt']['probe']['worst_operator_l2']:.3e}`, remap reconstruction difference `{100*metrics['odt']['probe']['worst_reconstruction_difference']:.4f}%`.",
        f"- ODT full 256-z steady: `{metrics['odt']['scale']['256']['steady_ms']:.3f} ms`, `{metrics['odt']['scale']['256']['steady_hz']:.3f} Hz`.",
        f"- ODT matched-error: ACFO complex64 vs cuFINUFFT complex128 eps=1e-7, production speedup `{metrics['odt']['matched']['speedup']:.3f}x` (separate process).",
        f"- ODT temporal 1 update: `{warm1['median_total_hot_s']*1000:.3f} ms`, `{warm1['median_hot_hz']:.3f} Hz`, mean object L2 `{100*warm1['mean_object_rel_l2']:.3f}%`.",
        f"- aIDT GPU core: `{metrics['aidt']['run_ms']:.3f} ms`, `{metrics['aidt']['hz']:.3f} Hz`.",
        f"- WAXS protein unit exact-beta/direct: `{metrics['waxs']['protein_unit_cell']['exact_beta_complex_l2_vs_direct']:.3e}`; 1M crystal direct subset `{metrics['waxs']['ordered_protein_supercells']['5x5x5_direct_subset_complex_l2']:.3e}`.",
        f"- General curvature: exact Yee ACFO/direct `{100*metrics['general_curvature']['exact_yee_acfo_vs_direct']:.3f}%`; SH FDTD/exact-Yee ACFO `{100*metrics['general_curvature']['fdtd_vs_exact_acfo']:.3f}%`.",
        "",
        "## 남은 필수 작업",
        "",
        "1. 독립 GPU에서 frozen timing protocol 재실행.",
        "2. 실제 ODT complex camera의 acquisition, transfer, demodulation, remap, update를 한 budget으로 측정.",
        "3. ODT noise/model mismatch temporal sweep.",
        "4. aIDT transfer 포함 end-to-end 측정.",
        "5. general-curvature no-refit holdout 확대.",
    ]
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = collect()
    charts = build_charts(result["metrics"])
    build_source_inventory(result["metrics"], charts)
    write_markdown(result["metrics"])
    make_pdf(result["metrics"], charts)
    print(
        json.dumps(
            {
                "pdf": str(PDF_PATH),
                "markdown": str(MD_PATH),
                "source_inventory": str(SOURCE_PATH),
                "charts": {name: str(path) for name, path in charts.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
