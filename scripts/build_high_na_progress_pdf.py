from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
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
OUT = ROOT / "benchmark_results" / "high_na_progress_report.pdf"


def load_json(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def register_fonts() -> tuple[str, str]:
    malgun = Path("C:/Windows/Fonts/malgun.ttf")
    malgun_bold = Path("C:/Windows/Fonts/malgunbd.ttf")
    if malgun.exists() and malgun_bold.exists():
        pdfmetrics.registerFont(TTFont("MalgunGothic", str(malgun)))
        pdfmetrics.registerFont(TTFont("MalgunGothic-Bold", str(malgun_bold)))
        return "MalgunGothic", "MalgunGothic-Bold"
    body = "HYSMyeongJo-Medium"
    bold = "HYGothic-Medium"
    pdfmetrics.registerFont(UnicodeCIDFont(body))
    pdfmetrics.registerFont(UnicodeCIDFont(bold))
    return body, bold


def make_styles() -> dict[str, ParagraphStyle]:
    body_font, bold_font = register_fonts()
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "KoreanTitle",
            parent=sample["Title"],
            fontName=bold_font,
            fontSize=22,
            leading=28,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#111827"),
            spaceAfter=14,
        ),
        "subtitle": ParagraphStyle(
            "KoreanSubtitle",
            parent=sample["Normal"],
            fontName=body_font,
            fontSize=10.5,
            leading=15,
            textColor=colors.HexColor("#4b5563"),
            spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "KoreanH1",
            parent=sample["Heading1"],
            fontName=bold_font,
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#111827"),
            spaceBefore=12,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "KoreanH2",
            parent=sample["Heading2"],
            fontName=bold_font,
            fontSize=12.5,
            leading=16,
            textColor=colors.HexColor("#1f2937"),
            spaceBefore=10,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "KoreanBody",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#111827"),
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "KoreanSmall",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#4b5563"),
            spaceAfter=4,
        ),
        "table": ParagraphStyle(
            "KoreanTable",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#111827"),
        ),
        "table_head": ParagraphStyle(
            "KoreanTableHead",
            parent=sample["BodyText"],
            fontName=bold_font,
            fontSize=8,
            leading=10,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "caption": ParagraphStyle(
            "KoreanCaption",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#6b7280"),
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
    }


def p(text: str, styles: dict[str, ParagraphStyle], name: str = "body") -> Paragraph:
    return Paragraph(text, styles[name])


def bullet(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(f"• {text}", styles["body"])


def table(
    rows: list[list[Any]],
    styles: dict[str, ParagraphStyle],
    *,
    col_widths: list[float] | None = None,
) -> Table:
    wrapped: list[list[Any]] = []
    for row_index, row in enumerate(rows):
        style_name = "table_head" if row_index == 0 else "table"
        wrapped.append(
            [
                cell
                if hasattr(cell, "wrap")
                else Paragraph(str(cell), styles[style_name])
                for cell in row
            ]
        )
    result = Table(wrapped, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
            ]
        )
    )
    return result


def scaled_image(path: Path, max_width: float = 7.0 * inch, max_height: float = 4.55 * inch) -> Image:
    with PILImage.open(path) as img:
        width_px, height_px = img.size
    scale = min(max_width / width_px, max_height / height_px)
    return Image(str(path), width=width_px * scale, height=height_px * scale)


def add_figure(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    title: str,
    path: str,
    caption: str,
    *,
    page_break: bool = True,
) -> None:
    if page_break:
        story.append(PageBreak())
    image_path = ROOT / path
    story.append(p(title, styles, "h1"))
    story.append(scaled_image(image_path))
    story.append(p(caption, styles, "caption"))


def inverse_loop_rows() -> list[list[str]]:
    data = load_json("benchmark_results/high_na_inverse_design_loop.json")
    rows = [
        [
            "batch",
            "iterations",
            "dense ms/iter",
            "ours ms/iter",
            "hot speedup",
            "setup-inclusive",
            "loss decrease",
        ]
    ]
    for row in data["rows"]:
        if row.get("status") != "ok" or int(row["n_iterations"]) != 300:
            continue
        rows.append(
            [
                row["batch_size"],
                row["n_iterations"],
                f"{row['dense_ms_per_iteration']:.2f}",
                f"{row['separable_ms_per_iteration']:.2f}",
                f"{row['hot_loop_speedup_dense_vs_separable']:.2f}x",
                f"{row['setup_inclusive_speedup_dense_vs_separable']:.2f}x",
                f"{100.0 * row['separable_loss_decrease_fraction_evaluated_by_dense']:.1f}%",
            ]
        )
    return rows


def build_story(styles: dict[str, ParagraphStyle]) -> list[Any]:
    story: list[Any] = []
    story.append(p("High-NA Optics Extension Progress Report", styles, "title"))
    story.append(
        p(
            "Local benchmark snapshot generated in the Atomic WAXS Cake-Map Simulator workspace on 2026-06-20. "
            "The report summarizes the current evidence for a geometry-aware High-NA Debye-Wolf/Richards-Wolf propagator, "
            "with emphasis on repeated phase-mask inverse-design loops on fixed cylindrical focal grids.",
            styles,
            "subtitle",
        )
    )
    story.append(p("Executive Summary", styles, "h1"))
    story.extend(
        [
            bullet(
                "핵심 claim은 '다른 optics package가 못 하는 물리'가 아니라, 고정된 focal geometry/ROI에서 반복 phase-mask update를 더 싸게 계산한다는 점이다.",
                styles,
            ),
            bullet(
                "현재 GPU 구현은 아직 Python/Torch orchestration 중심이고 custom CUDA/Triton/C++ fused kernel 수준의 low-level 최적화가 들어가지 않았다.",
                styles,
            ),
            bullet(
                "그 상태에서도 representative vectorial inverse-design primitive에서 300 update 기준 dense direct CUDA 대비 4.35x-7.20x hot-loop speedup을 보였다.",
                styles,
            ),
            bullet(
                "물리 demo는 self-consistency, known-aberration correction, intensity-only multi-trap으로 나누어 forward/adjoint와 phase-only design 가능성을 검증했다.",
                styles,
            ),
            bullet(
                "아직 broad package replacement claim은 이르다. PyFocus/psf-generator와 matched optimizer-loop baseline이 다음 관문이다.",
                styles,
            ),
        ]
    )

    story.append(p("Current Evidence Map", styles, "h1"))
    story.append(
        table(
            [
                ["Lane", "What was checked", "Current readout", "Claim boundary"],
                [
                    "Debye-Wolf / FINUFFT baseline",
                    "Scalar structured-grid rows, adaptive pupil-spectrum cutoff, direct/FINUFFT references.",
                    "Adaptive sparse recovers high-azimuthal failures; representative rows show speedups vs FINUFFT in favorable regimes.",
                    "Not a universal NUFFT or dense Cartesian FFT-Debye replacement.",
                ],
                [
                    "Vectorial propagation",
                    "Richards-Wolf Jones pupil mapping, vectorial forward/adjoint, dense CUDA comparison.",
                    "Same-device vectorial cylindrical ROI gives multi-x speedups with small gradient error.",
                    "External domain-package vectorial optimizer baseline remains open.",
                ],
                [
                    "Inverse-design loop",
                    "Forward, weighted ROI residual, adjoint, phase-gradient, phase-only update repeated for N iterations.",
                    "300-step loop is 4.35x-7.20x faster than dense direct CUDA depending on batch size.",
                    "Target generation excluded; comparison is same-device dense CUDA, not yet PyFocus/psf-generator optimizer loop.",
                ],
                [
                    "Physical demos",
                    "Hidden-mask self-consistency, aberration correction, multi-trap intensity shaping.",
                    "Phase recovery and focal-pattern formation are stable enough for application-level demos.",
                    "Multi-trap itself is not novel; it is a stress test for repeated non-axisymmetric design.",
                ],
            ],
            styles,
            col_widths=[1.25 * inch, 2.0 * inch, 2.0 * inch, 1.9 * inch],
        )
    )

    story.append(p("Repeated Inverse-Design Benchmark", styles, "h1"))
    story.append(
        p(
            "This benchmark measures the primitive that matters for phase-mask design: vectorial forward propagation, ROI residual construction, adjoint propagation, shared phase-gradient construction, and a phase-only update. "
            "The teacher target is generated once with dense direct CUDA and excluded from loop timing.",
            styles,
        )
    )
    story.append(table(inverse_loop_rows(), styles, col_widths=[0.65 * inch, 0.8 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.1 * inch, 1.15 * inch]))
    story.append(
        p(
            "Interpretation: the slope of the repeated loop is already favorable before low-level kernel fusion. "
            "The current result supports a focused claim about cheaper repeated structured-ROI inverse design, not a blanket claim about all High-NA propagation.",
            styles,
        )
    )
    add_figure(
        story,
        styles,
        "Figure 1. Repeated inverse-design loop cost",
        "benchmark_results/high_na_inverse_design_loop_figure.png",
        "Dense direct CUDA and structured separable vectorial paths were compared on the same representative cylindrical ROI. The right panel shows the per-update slope that drives long optimizer loops.",
        page_break=False,
    )

    story.append(PageBreak())
    story.append(p("Physical Demo Summary", styles, "h1"))
    story.append(
        table(
            [
                ["Demo", "Primary result", "Why it matters"],
                [
                    "Self-consistency",
                    "Intensity rel-L2 0.397 -> 0.00321; intensity cosine 0.929 -> 0.999995; phase corr 0.996.",
                    "Checks whether the phase-only optimizer can reproduce a hidden-mask focal intensity volume.",
                ],
                [
                    "Aberration correction",
                    "Field rel-L2 0.278 -> 0.00718; field cosine 0.961 -> 0.999978; correction phase corr 0.999.",
                    "Known-answer validation: expected correction is the negative imposed aberration up to piston.",
                ],
                [
                    "Multi-trap intensity shaping",
                    "Target cosine 0.00169 -> 0.717; trap energy fraction 0.00374 -> 0.2146; trap/background density 0.0647 -> 4.71.",
                    "Application-style stress test with no unique target phase and non-axisymmetric discrete traps.",
                ],
            ],
            styles,
            col_widths=[1.45 * inch, 2.4 * inch, 3.0 * inch],
        )
    )
    add_figure(
        story,
        styles,
        "Figure 2. Self-consistency phase-mask recovery",
        "benchmark_results/high_na_phase_mask_self_consistency_volume_figure.png",
        "A hidden phase-only pupil mask generated the target intensity volume; the optimizer recovered an equivalent mask that reproduces the through-focus intensity.",
    )
    add_figure(
        story,
        styles,
        "Figure 3. Known-aberration correction",
        "benchmark_results/high_na_aberration_correction_figure.png",
        "The optimizer learned a shared phase-only correction for an imposed coma/astigmatism/spherical-aberration mixture and restored the vectorial High-NA focus.",
    )
    add_figure(
        story,
        styles,
        "Figure 4. Phase-only multi-trap design",
        "benchmark_results/high_na_multitrap_phase_mask_figure.png",
        "A single phase-only vectorial pupil mask forms five discrete focal traps on the cylindrical grid. The demo is a design-workload stress test rather than a novelty claim for multi-trap physics.",
    )

    story.append(PageBreak())
    story.append(p("External Package And GPU Context", styles, "h1"))
    story.extend(
        [
            bullet(
                "PyFocus/PyCustomFocus agreement is strong on native Cartesian XY shape checks: max scale-fit L2 about 1.1e-4 and correlation essentially 1.0.",
                styles,
            ),
            bullet(
                "The local structured hot loop is 62x-74x faster than PyFocus in the current shape-comparison snapshot, but this is not yet a matched Cartesian package-replacement benchmark.",
                styles,
            ),
            bullet(
                "psf-generator provides a serious PyTorch/CUDA timing anchor; current rows show 3.7x-5.9x wall-time ratios, but coordinate, pupil, and batching semantics are not fully matched.",
                styles,
            ),
        ]
    )
    add_figure(
        story,
        styles,
        "Figure 5. External package comparison snapshot",
        "benchmark_results/high_na_external_package_comparison.png",
        "External package checks are useful anchors but should be framed as shape/timing evidence until a matched optimizer-loop adapter is added.",
        page_break=False,
    )
    add_figure(
        story,
        styles,
        "Figure 6. GPU structured-vs-dense comparison",
        "benchmark_results/high_na_gpu_comparison_figure.png",
        "Same-device GPU comparisons show where structured cylindrical propagation and backpropagation start to separate from dense direct quadrature.",
    )

    story.append(PageBreak())
    story.append(p("Recommended Claim For Manuscript Framing", styles, "h1"))
    story.append(
        p(
            "<b>Safe central claim:</b> A geometry-aware harmonic factorization makes repeated High-NA phase-mask design loops cheaper when the focal geometry is fixed and the target is naturally cylindrical or sparse.",
            styles,
        )
    )
    story.append(
        p(
            "<b>Avoid:</b> claiming a universal replacement for FINUFFT, PyFocus, psf-generator, or dense Cartesian FFT-Debye methods. The current evidence is strongest for structured cylindrical ROI workloads and repeated mask/coherent-mode evaluations.",
            styles,
        )
    )
    story.append(p("Next Benchmark Steps", styles, "h2"))
    story.extend(
        [
            bullet("Add a matched psf-generator optimizer-loop baseline using the same pupil, target normalization, defocus stack, and batch semantics.", styles),
            bullet("Add a PyFocus repeated-mask timing adapter; if gradient support is impractical, keep it as forward-loop evidence only.", styles),
            bullet("Fuse phase application, vectorial mixing, forward/adjoint contraction, residual weighting, and phase-gradient construction in lower-level kernels.", styles),
            bullet("Measure peak RSS and allocator behavior; current basis/cache MiB does not capture transient GPU allocations.", styles),
            bullet("Rerun after each C++/CUDA/Triton or build-setting change and keep exact timings local to machine/date/hardware.", styles),
        ]
    )

    story.append(p("Source Artifacts", styles, "h1"))
    story.append(
        table(
            [
                ["Artifact", "Path"],
                ["Inverse loop benchmark", "scripts/benchmark_high_na_inverse_design_loop.py"],
                ["Inverse loop results", "benchmark_results/high_na_inverse_design_loop_summary.md"],
                ["Self-consistency demo", "benchmark_results/high_na_phase_mask_self_consistency_volume_summary.md"],
                ["Aberration correction demo", "benchmark_results/high_na_aberration_correction_summary.md"],
                ["Multi-trap demo", "benchmark_results/high_na_multitrap_phase_mask_summary.md"],
                ["External package rollup", "benchmark_results/high_na_external_package_comparison.md"],
                ["Publication validation", "validation/high_na_si/"],
            ],
            styles,
            col_widths=[2.15 * inch, 4.75 * inch],
        )
    )
    return story


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawRightString(A4[0] - 0.5 * inch, 0.35 * inch, f"{doc.page}")
    canvas.restoreState()


def main() -> None:
    styles = make_styles()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.55 * inch,
        title="High-NA Optics Extension Progress Report",
        author="Atomic WAXS Cake-Map Simulator",
    )
    story = build_story(styles)
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(OUT)


if __name__ == "__main__":
    main()
