from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

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
OUT = ROOT / "benchmark_results" / "waxs_highna_odt_integrated_progress_report.pdf"
SOURCE_NOTES = ROOT / "benchmark_results" / "waxs_highna_odt_integrated_progress_source_notes.md"


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
            "KRTitle",
            parent=sample["Title"],
            fontName=bold_font,
            fontSize=21,
            leading=27,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#111827"),
            spaceAfter=9,
        ),
        "subtitle": ParagraphStyle(
            "KRSubtitle",
            parent=sample["Normal"],
            fontName=body_font,
            fontSize=9.3,
            leading=13.3,
            textColor=colors.HexColor("#4b5563"),
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "KRH1",
            parent=sample["Heading1"],
            fontName=bold_font,
            fontSize=14.2,
            leading=18.2,
            textColor=colors.HexColor("#111827"),
            spaceBefore=10,
            spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "KRH2",
            parent=sample["Heading2"],
            fontName=bold_font,
            fontSize=11.1,
            leading=14.4,
            textColor=colors.HexColor("#1f2937"),
            spaceBefore=7,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "KRBody",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=8.8,
            leading=12.5,
            textColor=colors.HexColor("#111827"),
            spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "KRSmall",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=7.3,
            leading=9.8,
            textColor=colors.HexColor("#4b5563"),
            spaceAfter=3,
        ),
        "table": ParagraphStyle(
            "KRTable",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=7.0,
            leading=8.7,
            textColor=colors.HexColor("#111827"),
        ),
        "table_head": ParagraphStyle(
            "KRTableHead",
            parent=sample["BodyText"],
            fontName=bold_font,
            fontSize=6.8,
            leading=8.5,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "caption": ParagraphStyle(
            "KRCaption",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=7.0,
            leading=9.0,
            textColor=colors.HexColor("#6b7280"),
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=7,
        ),
    }


def p(text: str, styles: dict[str, ParagraphStyle], name: str = "body") -> Paragraph:
    return Paragraph(escape(text), styles[name])


def bullet(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(f"- {escape(text)}", styles["body"])


def table(
    rows: list[list[object]],
    styles: dict[str, ParagraphStyle],
    *,
    col_widths: list[float] | None = None,
    font_size: float | None = None,
) -> Table:
    table_style = styles["table"]
    head_style = styles["table_head"]
    if font_size is not None:
        table_style = ParagraphStyle(
            f"{table_style.name}_{font_size}",
            parent=table_style,
            fontSize=font_size,
            leading=font_size + 1.7,
        )
        head_style = ParagraphStyle(
            f"{head_style.name}_{font_size}",
            parent=head_style,
            fontSize=max(6.0, font_size - 0.2),
            leading=font_size + 1.5,
        )
    wrapped: list[list[object]] = []
    for row_idx, row in enumerate(rows):
        style = head_style if row_idx == 0 else table_style
        wrapped.append(
            [
                cell
                if hasattr(cell, "wrap")
                else Paragraph(escape(str(cell)), style)
                for cell in row
            ]
        )
    result = Table(wrapped, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("GRID", (0, 0), (-1, -1), 0.28, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3.2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3.2),
                ("TOPPADDING", (0, 0), (-1, -1), 2.6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
            ]
        )
    )
    return result


def image_block(
    rel_path: str,
    caption: str,
    styles: dict[str, ParagraphStyle],
    *,
    max_width: float = 7.0 * inch,
    max_height: float = 4.2 * inch,
) -> KeepTogether:
    path = ROOT / rel_path
    with PILImage.open(path) as img:
        width_px, height_px = img.size
    scale = min(max_width / width_px, max_height / height_px)
    img = Image(str(path), width=width_px * scale, height=height_px * scale)
    img.hAlign = "CENTER"
    return KeepTogether([img, p(caption, styles, "caption")])


def load_json(rel_path: str) -> dict:
    return json.loads((ROOT / rel_path).read_text(encoding="utf-8"))


def load_csv(rel_path: str) -> list[dict[str, str]]:
    with (ROOT / rel_path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fmt_s(value: float) -> str:
    if value < 0.01:
        return f"{value * 1000:.2f} ms"
    return f"{value:.3f} s"


def fmt_x(value: float) -> str:
    return f"{value:.2f}x"


def waxs_highq_rows() -> list[list[str]]:
    data = load_json("benchmark_results/algorithm_scaling_highq_direct_fftfriendly_nufft_qb2.json")
    rows = [["atoms", "n_phi", "R-dep", "fused", "NUFFT", "NUFFT/R-dep", "I error"]]
    for row in data["rows"]:
        atoms = int(row["atoms"])
        rows.append(
            [
                f"{atoms // 1000}k" if atoms < 1_000_000 else "1M",
                str(row["grid"]["n_phi"]),
                fmt_s(row["rdep_analytic_s"]),
                fmt_s(row["rdep_fused_s"]),
                fmt_s(row["nufft_s"]),
                fmt_x(row["rdep_speedup_vs_nufft"]),
                f"{row['rdep_analytic_intensity_rel_l2_vs_dense']:.1e}",
            ]
        )
    return rows


def waxs_qmax_rows() -> list[list[str]]:
    rows = [["qmax", "Nq", "n_phi", "targets", "R-dep", "fused", "NUFFT", "NUFFT/R-dep"]]
    for path in sorted((ROOT / "benchmark_results").glob("qmax_scaling_1m_dq0p160_q*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        case = data["case"]
        row = data["rows"][0]
        n_phi = int(row["grid"]["n_phi"])
        nq = int(case["nq"])
        rows.append(
            [
                f"{case['qmax']:.2f}",
                str(nq),
                str(n_phi),
                f"{nq * n_phi:,}",
                fmt_s(row["rdep_analytic_s"]),
                fmt_s(row["rdep_fused_s"]),
                fmt_s(row["nufft_s"]),
                fmt_x(row["rdep_speedup_vs_nufft"]),
            ]
        )
    return rows


def high_na_adaptive_rows() -> list[list[str]]:
    return [
        ["case", "failure without adaptive", "adaptive correction", "speed / work readout"],
        [
            "vortex h=18 small",
            "geometric-only L2 ~= 1.0",
            "adds h=18, direct L2 4.3e-10",
            "mode-rho work 77 -> 82",
        ],
        [
            "vortex h=30 representative",
            "geometric-only L2 ~= 1.0",
            "adds h=30, direct L2 4.0e-08",
            "sparse work 354 vs dense-prefix 449",
        ],
        [
            "benign mixed representative",
            "no extra harmonics needed",
            "FINUFFT L2 1.5e-12",
            "adaptive sparse total speedup 2.03x vs FINUFFT",
        ],
        [
            "large benign / vortex",
            "broad high-order content remains the risk",
            "FINUFFT L2 1.4e-12 to 3.9e-06",
            "speedup 1.46x to 2.58x vs FINUFFT",
        ],
    ]


def high_na_gpu_rows() -> list[list[str]]:
    return [
        ["benchmark", "matched baseline", "local structured path", "accuracy / implication"],
        [
            "same-target CUDA vectorial pair",
            "dense direct forward+adjoint",
            "4.92x faster",
            "forward L2 3.78e-06, adjoint L2 8.38e-07",
        ],
        [
            "PyFocus/PyCustomFocus XY shape",
            "external vectorial package",
            "62x-74x hot, 17x-21x one-shot",
            "shape L2 <= 1.11e-04, corr ~= 1.0",
        ],
        [
            "psf-generator CUDA timing anchor",
            "external vectorial PSF package",
            "3.73x-5.85x wall-time ratio",
            "timing-only; target grids are not yet matched",
        ],
        [
            "cylindrical inverse-design loop",
            "dense direct CUDA objective",
            "4.3x-7.2x at 300 iterations",
            "phase gradient L2 around 1e-6 in representative rows",
        ],
    ]


def odt_finufft_rows() -> list[list[str]]:
    paths = sorted(
        (ROOT / "benchmark_results").glob("odt_cone_axis_final_finufft_r*_phi*.json"),
        key=lambda pth: int(pth.stem.split("_r")[1].split("_phi")[0]),
    )
    rows = [["cap", "mode", "samples", "prepared", "FINUFFT", "speedup", "rel-L2", "setup"]]
    for path in paths:
        case = json.loads(path.read_text(encoding="utf-8"))["cases"][0]
        rows.append(
            [
                f"{case['cap_radial']} x {case['cap_phi']}",
                case["native_prepared_plan_effective_mode"],
                f"{case['cone_samples']:,}",
                fmt_s(case["prepared_execute_full_s"]),
                fmt_s(case["finufft_adjoint_s"]),
                fmt_x(case["finufft_over_prepared_speedup"]),
                f"{case['finufft_adjoint_l2_vs_prepared']:.2e}",
                fmt_s(case["build_total_s"]),
            ]
        )
    return rows


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawRightString(A4[0] - 0.50 * inch, 0.35 * inch, f"{doc.page}")
    canvas.restoreState()


def section_list(items: Iterable[str], styles: dict[str, ParagraphStyle]) -> list[Paragraph]:
    return [bullet(item, styles) for item in items]


def build_story(styles: dict[str, ParagraphStyle]) -> list[object]:
    story: list[object] = []
    today = date(2026, 6, 21)
    story.append(p("WAXS, High-NA optics, ODT 알고리즘 확장 진행 요약", styles, "title"))
    story.append(
        p(
            f"작성일: {today.isoformat()}. 범위: Atomic WAXS Cake-Map Simulator 워크스페이스에서 수행한 "
            "WAXS cake-map solver, High-NA Debye-Wolf/Richards-Wolf prototype, ODT cone-axis adjoint 최적화와 벤치마크.",
            styles,
            "subtitle",
        )
    )

    story.append(p("기술 요약", styles, "h1"))
    story.extend(
        section_list(
            [
                "세 분야 모두에서 공통 패턴은 같다. 일반 NUFFT나 dense quadrature가 target grid를 직접 다루는 대신, 물리적 좌표계가 만드는 반복 구조를 찾아 harmonic/FFT/structured-grid factorization으로 바꾸었다.",
                "WAXS는 현재 가장 강한 검증 축이다. 1M atom, qmax=6.3 A^-1 high-q run에서 R-dependent analytic solver가 FINUFFT 대비 약 54x, fixed-dq q-range sweep에서는 qmax가 커질수록 약 25x에서 59x까지 이득이 증가했다.",
                "High-NA optics는 WAXS 구조가 다른 물리계로 이전될 수 있음을 보여주는 중간 단계다. adaptive pupil spectrum 보정으로 high-azimuthal failure를 해결했고, vectorial/GPU/backpropagation demo와 외부 package 비교까지 확보했다.",
                "ODT는 WAXS보다 inverse/backpropagation 구조와 더 직접적으로 맞는다. 최종 cone-axis adjoint-only prepared path는 FINUFFT type-3 adjoint 대비 cap 64 x 256에서 8.27x, cap 128 x 512에서 9.29x 빠르며 L2 차이는 약 9e-10이다.",
                "현재 가장 방어적인 논문 스토리는 'WAXS에서 검증된 geometry-aware Fourier factorization이 High-NA와 ODT의 structured inverse/backpropagation workloads에서도 반복 평가 비용을 줄인다'이다. 아직 universal NUFFT replacement나 full Cartesian optics replacement로 쓰면 안 된다.",
            ],
            styles,
        )
    )

    story.append(p("공통 방법론: curved manifold를 직접 계산하지 않고 분해한다", styles, "h1"))
    story.append(
        table(
            [
                ["domain", "structured variable", "factorization", "validated baseline", "strong regime"],
                ["WAXS", "(R, z, beta) histogram + Ewald ring", "beta FFT, circular harmonics, R-dependent h cutoff", "dense circular, chunked FINUFFT", "high-q fixed-dq cake maps"],
                ["High-NA optics", "(rho, psi, z) focal grid + pupil azimuth", "pupil/focal azimuth harmonics, positive-rho no-copy, adaptive pupil spectrum", "direct Debye-Wolf, FINUFFT, PyFocus, psf-generator anchors", "many coherent modes, repeated phase masks, cylindrical ROI"],
                ["ODT", "cone-axis detector cap + object cylindrical bins", "axis decomposition, h/l slots, prepared adjoint, residual gather, z-major l accumulation", "FINUFFT nufft3d3 adjoint", "adjoint-only screening and repeated residuals"],
            ],
            styles,
            col_widths=[0.8 * inch, 1.45 * inch, 2.15 * inch, 1.55 * inch, 1.35 * inch],
            font_size=6.7,
        )
    )

    story.append(PageBreak())
    story.append(p("WAXS: 가장 강한 검증 축", styles, "h1"))
    story.append(
        p(
            "WAXS에서는 cylindrical histogram, beta FFT, R-dependent harmonic cutoff, analytic Miller/Bessel kernel, "
            "no-copy positive half-spectrum, fused Miller kernel, FFT-friendly n_phi rounding까지 적용했다. "
            "핵심은 fixed dq에서 q range가 커질수록 detector target 수와 angular sampling이 함께 증가하므로, "
            "WAXS-specific factorization의 이득이 더 분명해진다는 점이다.",
            styles,
        )
    )
    story.append(p("High-q atom-count sweep", styles, "h2"))
    story.append(
        table(
            waxs_highq_rows(),
            styles,
            col_widths=[0.70 * inch, 0.55 * inch, 0.80 * inch, 0.80 * inch, 0.80 * inch, 0.90 * inch, 0.85 * inch],
        )
    )
    story.append(p("Fixed-dq q-range sweep", styles, "h2"))
    story.append(
        table(
            waxs_qmax_rows(),
            styles,
            col_widths=[0.62 * inch, 0.42 * inch, 0.56 * inch, 0.78 * inch, 0.75 * inch, 0.75 * inch, 0.78 * inch, 0.86 * inch],
        )
    )
    story.append(
        p(
            "해석: WAXS 결과는 이미 논문 1편의 중심축으로 충분히 강하다. 다만 주장 경계는 WAXS cake-map geometry에 묶어야 한다. "
            "NUFFT 일반 대체가 아니라, curved Ewald/circular structure를 이용한 domain-specific factorization이다.",
            styles,
        )
    )
    story.append(
        image_block(
            "docs/waxs_optimization_experiment_summary_2026_06_16_ko_page1.png",
            "WAXS 최적화 요약의 첫 페이지. 최종 PDF에서는 원천 수치를 JSON에서 다시 읽어 표로 재구성했다.",
            styles,
            max_height=6.4 * inch,
        )
    )

    story.append(PageBreak())
    story.append(p("High-NA optics: scalar에서 vectorial/backpropagation까지 확장", styles, "h1"))
    story.append(
        p(
            "High-NA 쪽은 Debye-Wolf solver를 reference로 두고 시작해 FINUFFT/direct baseline, pupil-spectrum adaptive cutoff, "
            "GPU vectorial backend, PyFocus/psf-generator 비교, phase-only physical demo까지 확장했다. "
            "현재 핵심은 full Cartesian PSF replacement가 아니라 structured cylindrical ROI에서 반복 mask/coherent-mode 평가가 싸진다는 점이다.",
            styles,
        )
    )
    story.append(p("Adaptive pupil spectrum이 정확도 경계를 해결했다", styles, "h2"))
    story.append(
        table(
            high_na_adaptive_rows(),
            styles,
            col_widths=[1.35 * inch, 1.65 * inch, 1.65 * inch, 1.55 * inch],
        )
    )
    story.append(p("GPU/vectorial/external baseline readout", styles, "h2"))
    story.append(
        table(
            high_na_gpu_rows(),
            styles,
            col_widths=[1.35 * inch, 1.55 * inch, 1.35 * inch, 2.0 * inch],
        )
    )
    story.append(
        p(
            "해석: High-NA 결과는 cross-field validation으로 의미가 있다. 특히 adaptive cutoff는 단순 속도 최적화가 아니라 "
            "high-azimuthal pupil content에서 틀리는 물리적 failure mode를 고친 결과다. 하지만 외부 package와 완전히 같은 Cartesian target, "
            "normalization, batching semantics를 맞춘 비교는 아직 남아 있다.",
            styles,
        )
    )
    story.append(
        image_block(
            "benchmark_results/high_na_external_package_comparison.png",
            "PyFocus shape agreement와 psf-generator GPU timing anchor. 외부 package 비교는 formulation/속도 anchor이며 full matched replacement claim은 아니다.",
            styles,
            max_height=2.25 * inch,
        )
    )
    story.append(
        image_block(
            "benchmark_results/high_na_inverse_design_loop_figure.png",
            "반복 phase-mask inverse-design loop. Dense direct CUDA 대비 structured separable loop의 per-iteration 비용이 낮다.",
            styles,
            max_height=2.25 * inch,
        )
    )

    story.append(PageBreak())
    story.append(p("High-NA physical demos: phase-only mask가 계산 loop의 의미를 만든다", styles, "h1"))
    story.extend(
        section_list(
            [
                "Self-consistency: hidden phase-only pupil mask로 만든 intensity volume을 다시 맞추는 테스트에서 normalized intensity rel-L2가 0.397 -> 0.00321로 감소했고, hidden phase와의 correlation은 0.996이었다.",
                "Aberration correction: known aberration의 negative phase를 회복하는 controlled vector-field objective에서 field rel-L2가 0.278 -> 0.00718, phase RMSE가 0.034 rad까지 내려갔다.",
                "Multi-trap: phase-only pupil mask로 5개의 non-axisymmetric trap을 만들었고, trap/background density ratio가 0.0647 -> 4.71로 증가했다.",
                "Repeated loop benchmark: 300 iteration에서 separable loop는 dense CUDA 대비 batch 1/8/32에서 각각 약 4.3x/5.4x/7.2x hot speedup을 보였다.",
            ],
            styles,
        )
    )
    story.append(
        image_block(
            "benchmark_results/high_na_aberration_correction_figure.png",
            "Known aberration을 phase-only correction으로 되돌리는 demo. 이 경우는 기대 해가 명확하므로 forward/adjoint correction 검증에 적합하다.",
            styles,
            max_height=4.3 * inch,
        )
    )
    story.append(
        image_block(
            "benchmark_results/high_na_multitrap_phase_mask_figure.png",
            "Discrete multi-trap phase-only design. 비축대칭 target이므로 반복 mask optimization workload를 직접 자극한다.",
            styles,
            max_height=4.0 * inch,
        )
    )

    story.append(PageBreak())
    story.append(p("ODT: adjoint-only에서 구조적 이득이 가장 선명하다", styles, "h1"))
    story.append(
        p(
            "ODT cone-axis 작업은 forward/adjoint pair를 먼저 구현하고 검증한 뒤, 논문적으로 더 중요한 adjoint-only screening/backpropagation "
            "경로를 따로 밀었다. Native C++ prepared plan, residual h-slot gather, z-major l-accumulation, selective ODT-only build, "
            "AVX2 build mode가 적용되었다.",
            styles,
        )
    )
    story.append(p("Optimization sequence", styles, "h2"))
    story.append(
        table(
            [
                ["step", "implemented result", "readout"],
                ["forward/adjoint pair", "safe C++ pair path and pytests", "validated pair before isolating adjoint"],
                ["native prepared plan", "ConeAxisPreparedAdjointPlan", "large q-area tuple call 대비 about 1.48x in A/B run"],
                ["residual gather", "contiguous (h, illum, u) residual buffer", "large q-area direct 대비 about 1.32x"],
                ["z-major l-accum", "l_accum[z, local_l] scatter layout", "large q-area gathered 대비 about 1.69x in A/B run"],
                ["build optimization", "WAXS_CPP_EXTENSIONS=odt, WAXS_CPP_OPT=avx2", "ODT-only force rebuild around 18-22 s"],
            ],
            styles,
            col_widths=[1.20 * inch, 2.65 * inch, 2.80 * inch],
        )
    )
    story.append(p("Final FINUFFT comparison", styles, "h2"))
    story.append(
        table(
            odt_finufft_rows(),
            styles,
            col_widths=[0.70 * inch, 1.15 * inch, 0.90 * inch, 0.75 * inch, 0.75 * inch, 0.62 * inch, 0.72 * inch, 0.72 * inch],
            font_size=6.6,
        )
    )
    story.append(
        p(
            "해석: cap/q-area가 작을 때도 이득은 있지만, 강한 주장은 cap 64 x 256 이상에서 나온다. "
            "FINUFFT와의 relative L2가 약 9e-10으로 유지되므로, 단순한 근사 shortcut이 아니라 같은 operator를 더 싸게 평가하는 결과로 볼 수 있다. "
            "다만 build/setup은 커지므로 repeated residual 또는 iterative adjoint에서 amortization을 전제로 말해야 한다.",
            styles,
        )
    )
    story.append(
        image_block(
            "benchmark_results/odt_cone_axis_final_finufft_comparison.png",
            "ODT cone-axis final adjoint-only comparison. High q-area에서 FINUFFT 대비 8-9x 구간으로 올라간다.",
            styles,
            max_height=3.55 * inch,
        )
    )

    story.append(PageBreak())
    story.append(p("논문 전략: WAXS 단독, cross-field 확장, 또는 통합 framework", styles, "h1"))
    story.append(
        table(
            [
                ["strategy", "strength", "risk", "best target"],
                ["WAXS-first", "가장 직접적이고 수치가 강함. 1M high-q, memory, fixed-dq scaling story가 선명함.", "cross-field novelty는 제한적.", "JSR / Communications Physics급 methods paper"],
                ["WAXS + High-NA", "검증된 home field와 optics transfer를 동시에 보여줌.", "High-NA external matched baseline이 아직 일부 남음.", "Communications Physics, Nature Communications 도전 가능성"],
                ["WAXS + High-NA + ODT", "forward와 inverse/backpropagation 양쪽에서 같은 factorization 철학을 보여줌.", "세 분야를 한 논문에 묶으면 claim 관리가 어려움.", "Nature Computational Science급 확장 전략의 후보"],
            ],
            styles,
            col_widths=[1.05 * inch, 2.45 * inch, 1.85 * inch, 1.35 * inch],
        )
    )
    story.append(
        p(
            "추천: 지금 당장 가장 안전한 1차 논문은 WAXS 중심이다. 다만 Nature Communications 이상을 노릴 경우에는 "
            "High-NA와 ODT를 별도 application validation으로 붙이는 전략이 더 강하다. 특히 ODT adjoint는 WAXS의 real-to-reciprocal 구조와 방향성이 맞아 "
            "세 번째 application으로 성공하면 computational science 계열에 필요한 일반성 주장이 더 설득력을 얻는다.",
            styles,
        )
    )

    story.append(p("제한과 robustness check", styles, "h1"))
    story.extend(
        section_list(
            [
                "WAXS: fixed-dq high-q sweep은 강하지만, 실제 재료/분자 benchmark input을 더 늘려야 한다. NUFFT 비교는 q-block chunking을 공정 baseline으로 유지해야 한다.",
                "High-NA: scalar/structured-grid benchmark와 vectorial/GPU demo는 확보됐지만, full Cartesian external package와 완전히 matched된 comparison은 아직 다음 단계다.",
                "ODT: adjoint-only execute time은 강하지만 build/setup amortization 조건을 명확히 해야 한다. Forward/adjoint pair 전체 claim과 adjoint-only claim을 분리해야 한다.",
                "세 분야 통합 claim: 'curved manifold Fourier factorization'이라는 상위 언어는 가능하지만, 각 domain의 좌표계와 validity boundary를 표로 분리해서 과잉 일반화를 피해야 한다.",
            ],
            styles,
        )
    )

    story.append(p("다음 단계", styles, "h1"))
    story.extend(
        section_list(
            [
                "WAXS: realistic NPZ benchmark set을 확정하고, fixed-dq high-q figure와 memory figure를 publication-quality로 재생성한다.",
                "High-NA: matched external vectorial package workload를 하나 더 만들고, vectorial Richards-Wolf assumptions와 normalization을 문서화한다.",
                "ODT: adjoint-only repeated-residual benchmark를 추가해 build/setup amortization curve를 만든다. q-area와 radial extent를 더 키워 FINUFFT memory scaling이 불리해지는 영역을 명시한다.",
                "통합 논문: 세 application을 하나의 표준 narrative로 묶는다면, WAXS는 validation, High-NA는 optics transfer, ODT는 inverse/backpropagation transfer로 역할을 분리한다.",
            ],
            styles,
        )
    )

    story.append(p("Source inventory", styles, "h1"))
    source_items = [
        "docs/r_dependent_analytic_final_summary.md",
        "benchmark_results/algorithm_scaling_highq_direct_fftfriendly_nufft_qb2.json",
        "benchmark_results/qmax_scaling_1m_dq0p160_q*.json",
        "benchmark_results/high_na_pupil_spectrum_option_matrix_summary.md",
        "benchmark_results/high_na_workload_matrix_summary.md",
        "benchmark_results/high_na_gpu_external_baseline_rollup.md",
        "benchmark_results/high_na_external_package_comparison.md",
        "benchmark_results/high_na_aberration_correction_summary.md",
        "benchmark_results/high_na_multitrap_phase_mask_summary.md",
        "benchmark_results/high_na_inverse_design_loop_summary.md",
        "benchmark_results/odt_cone_axis_native_plan_summary.md",
        "benchmark_results/odt_cone_axis_residual_gather_summary.md",
        "benchmark_results/odt_cone_axis_laccum_zmajor_summary.md",
        "benchmark_results/odt_cone_axis_final_finufft_comparison_summary.md",
    ]
    for item in source_items:
        story.append(p(item, styles, "small"))
    return story


def write_source_notes() -> None:
    notes = """# WAXS / High-NA / ODT integrated progress report source notes

Generated artifact:

- `benchmark_results/waxs_highna_odt_integrated_progress_report.pdf`

Report contract:

- Audience: technical.
- Scope: local benchmark and demo artifacts in the Atomic WAXS Cake-Map Simulator workspace.
- Delivery mode: PDF only.
- Generated on: 2026-06-21.

Primary source files:

- `docs/r_dependent_analytic_final_summary.md`
- `benchmark_results/algorithm_scaling_highq_direct_fftfriendly_nufft_qb2.json`
- `benchmark_results/qmax_scaling_1m_dq0p160_q*.json`
- `benchmark_results/high_na_pupil_spectrum_option_matrix_summary.md`
- `benchmark_results/high_na_workload_matrix_summary.md`
- `benchmark_results/high_na_gpu_external_baseline_rollup.md`
- `benchmark_results/high_na_external_package_comparison.md`
- `benchmark_results/high_na_phase_mask_self_consistency_summary.md`
- `benchmark_results/high_na_aberration_correction_summary.md`
- `benchmark_results/high_na_multitrap_phase_mask_summary.md`
- `benchmark_results/high_na_inverse_design_loop_summary.md`
- `benchmark_results/odt_cone_axis_native_plan_summary.md`
- `benchmark_results/odt_cone_axis_residual_gather_summary.md`
- `benchmark_results/odt_cone_axis_laccum_zmajor_summary.md`
- `benchmark_results/odt_cone_axis_final_finufft_comparison_summary.md`

Visual map:

- WAXS overview: `docs/waxs_optimization_experiment_summary_2026_06_16_ko_page1.png`
- High-NA external package comparison: `benchmark_results/high_na_external_package_comparison.png`
- High-NA inverse-design loop: `benchmark_results/high_na_inverse_design_loop_figure.png`
- High-NA aberration correction: `benchmark_results/high_na_aberration_correction_figure.png`
- High-NA multi-trap phase mask: `benchmark_results/high_na_multitrap_phase_mask_figure.png`
- ODT final FINUFFT comparison: `benchmark_results/odt_cone_axis_final_finufft_comparison.png`

Claim boundaries:

- WAXS: domain-specific cake-map factorization, not a universal NUFFT replacement.
- High-NA: structured cylindrical ROI and repeated coherent-mode/mask workloads, not a full Cartesian optics package replacement.
- ODT: adjoint-only repeated residual/backpropagation workloads, with build/setup amortization reported separately.
"""
    SOURCE_NOTES.write_text(notes, encoding="utf-8")


def main() -> None:
    styles = make_styles()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.48 * inch,
        bottomMargin=0.55 * inch,
        title="WAXS, High-NA optics, ODT integrated progress report",
        author="Codex",
    )
    doc.build(build_story(styles), onFirstPage=add_page_number, onLaterPages=add_page_number)
    write_source_notes()
    print(OUT)
    print(SOURCE_NOTES)


if __name__ == "__main__":
    main()
