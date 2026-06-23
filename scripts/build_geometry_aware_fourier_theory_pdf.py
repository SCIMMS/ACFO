from __future__ import annotations

import csv
import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Iterable
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
OUT = ROOT / "benchmark_results" / "geometry_aware_fourier_factorization_theory_draft_ko.pdf"
SOURCE_NOTES = ROOT / "benchmark_results" / "geometry_aware_fourier_factorization_theory_draft_sources.md"

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
(ROOT / ".matplotlib_cache").mkdir(exist_ok=True)


INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#4b5563")
GRID = colors.HexColor("#d1d5db")
SOFT = colors.HexColor("#f9fafb")
HEADER = colors.HexColor("#1f2937")
BLUE = colors.HexColor("#dbeafe")
GREEN = colors.HexColor("#dcfce7")
AMBER = colors.HexColor("#fef3c7")
ROSE = colors.HexColor("#ffe4e6")
VIOLET = colors.HexColor("#ede9fe")


def register_fonts() -> tuple[str, str]:
    regular = Path("C:/Windows/Fonts/malgun.ttf")
    bold = Path("C:/Windows/Fonts/malgunbd.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("MalgunGothic", str(regular)))
        pdfmetrics.registerFont(TTFont("MalgunGothic-Bold", str(bold)))
        return "MalgunGothic", "MalgunGothic-Bold"
    body = "HYSMyeongJo-Medium"
    head = "HYGothic-Medium"
    pdfmetrics.registerFont(UnicodeCIDFont(body))
    pdfmetrics.registerFont(UnicodeCIDFont(head))
    return body, head


def make_styles() -> dict[str, ParagraphStyle]:
    body_font, bold_font = register_fonts()
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleKR",
            parent=sample["Title"],
            fontName=bold_font,
            fontSize=19.5,
            leading=25,
            alignment=TA_LEFT,
            textColor=INK,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleKR",
            parent=sample["Normal"],
            fontName=body_font,
            fontSize=9.2,
            leading=13,
            textColor=MUTED,
            spaceAfter=7,
        ),
        "h1": ParagraphStyle(
            "Heading1KR",
            parent=sample["Heading1"],
            fontName=bold_font,
            fontSize=14.0,
            leading=18,
            textColor=INK,
            spaceBefore=8,
            spaceAfter=5,
        ),
        "h2": ParagraphStyle(
            "Heading2KR",
            parent=sample["Heading2"],
            fontName=bold_font,
            fontSize=11.0,
            leading=14,
            textColor=colors.HexColor("#1f2937"),
            spaceBefore=6,
            spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "BodyKR",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=8.75,
            leading=12.2,
            textColor=INK,
            spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "SmallKR",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=7.1,
            leading=9.2,
            textColor=MUTED,
            spaceAfter=2.5,
        ),
        "caption": ParagraphStyle(
            "CaptionKR",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=7.0,
            leading=9.0,
            textColor=colors.HexColor("#6b7280"),
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=7,
        ),
        "equation": ParagraphStyle(
            "EquationKR",
            parent=sample["Code"],
            fontName=body_font,
            fontSize=8.6,
            leading=12,
            textColor=colors.HexColor("#172554"),
            backColor=colors.HexColor("#eff6ff"),
            borderColor=colors.HexColor("#bfdbfe"),
            borderWidth=0.35,
            borderPadding=5,
            spaceBefore=4,
            spaceAfter=5,
        ),
        "table": ParagraphStyle(
            "TableKR",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=6.75,
            leading=8.4,
            textColor=INK,
        ),
        "table_head": ParagraphStyle(
            "TableHeadKR",
            parent=sample["BodyText"],
            fontName=bold_font,
            fontSize=6.7,
            leading=8.4,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "box": ParagraphStyle(
            "BoxKR",
            parent=sample["BodyText"],
            fontName=bold_font,
            fontSize=7.2,
            leading=9.2,
            textColor=INK,
            alignment=TA_CENTER,
        ),
    }


def clean_markup(text: str) -> str:
    return escape(text).replace("\n", "<br/>")


def p(text: str, styles: dict[str, ParagraphStyle], style: str = "body") -> Paragraph:
    return Paragraph(clean_markup(text), styles[style])


def bullet(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return p("- " + text, styles, "body")


def bullets(items: Iterable[str], styles: dict[str, ParagraphStyle]) -> list[Paragraph]:
    return [bullet(item, styles) for item in items]


def table(
    rows: list[list[Any]],
    styles: dict[str, ParagraphStyle],
    *,
    col_widths: list[float] | None = None,
    font_size: float | None = None,
) -> Table:
    table_style = styles["table"]
    head_style = styles["table_head"]
    if font_size is not None:
        table_style = ParagraphStyle(
            table_style.name + str(font_size),
            parent=table_style,
            fontSize=font_size,
            leading=font_size + 1.6,
        )
        head_style = ParagraphStyle(
            head_style.name + str(font_size),
            parent=head_style,
            fontSize=max(5.8, font_size - 0.2),
            leading=font_size + 1.5,
        )
    wrapped: list[list[Any]] = []
    for i, row in enumerate(rows):
        style = head_style if i == 0 else table_style
        wrapped.append(
            [
                cell
                if hasattr(cell, "wrap")
                else Paragraph(clean_markup(str(cell)), style)
                for cell in row
            ]
        )
    result = Table(wrapped, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HEADER),
                ("GRID", (0, 0), (-1, -1), 0.25, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3.0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3.0),
                ("TOPPADDING", (0, 0), (-1, -1), 2.4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SOFT]),
            ]
        )
    )
    return result


def equation(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return p(text, styles, "equation")


def image_block(
    rel_path: str,
    caption: str,
    styles: dict[str, ParagraphStyle],
    *,
    max_width: float = 7.0 * inch,
    max_height: float = 4.2 * inch,
) -> KeepTogether | Paragraph:
    path = ROOT / rel_path
    if not path.exists():
        return p(f"[missing image] {rel_path}", styles, "small")
    with PILImage.open(path) as img:
        width_px, height_px = img.size
    scale = min(max_width / width_px, max_height / height_px)
    image = Image(str(path), width=width_px * scale, height=height_px * scale)
    image.hAlign = "CENTER"
    return KeepTogether([image, p(caption, styles, "caption")])


def pipeline_diagram(styles: dict[str, ParagraphStyle]) -> Table:
    labels = [
        "물리 좌표계\nWAXS, pupil, detector cap",
        "structured bins\nradial / axial / azimuthal",
        "harmonic spectrum\nFFT over repeated angle",
        "basis kernels\nBessel, recurrence, phase",
        "prepared contraction\nforward or adjoint",
        "outputs\ncake, focal ROI, backprojection",
    ]
    row: list[Any] = []
    widths: list[float] = []
    colors_for_boxes = [BLUE, GREEN, AMBER, VIOLET, ROSE, BLUE]
    for idx, label in enumerate(labels):
        row.append(Paragraph(clean_markup(label), styles["box"]))
        widths.append(1.05 * inch)
        if idx != len(labels) - 1:
            row.append(Paragraph("→", styles["box"]))
            widths.append(0.22 * inch)
    result = Table([row], colWidths=widths, hAlign="CENTER")
    commands: list[tuple[Any, ...]] = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    box_col = 0
    for color in colors_for_boxes:
        commands.append(("BACKGROUND", (box_col, 0), (box_col, 0), color))
        commands.append(("BOX", (box_col, 0), (box_col, 0), 0.5, colors.HexColor("#9ca3af")))
        box_col += 2
    result.setStyle(TableStyle(commands))
    return result


def fmt_ms(seconds: float) -> str:
    return f"{1000.0 * seconds:.2f} ms"


def fmt_s(seconds: float) -> str:
    if seconds < 0.01:
        return fmt_ms(seconds)
    return f"{seconds:.3f} s"


def load_json(rel_path: str) -> dict[str, Any]:
    return json.loads((ROOT / rel_path).read_text(encoding="utf-8"))


def load_csv(rel_path: str) -> list[dict[str, str]]:
    with (ROOT / rel_path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def waxs_qmax_rows() -> list[list[str]]:
    rows = [["qmax", "targets", "R-dep", "fused", "FINUFFT", "speedup"]]
    for path in sorted((ROOT / "benchmark_results").glob("qmax_scaling_1m_dq0p160_q*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = payload["rows"][0]
        nq = int(payload["case"]["nq"])
        n_phi = int(row["grid"]["n_phi"])
        rows.append(
            [
                f"{float(payload['case']['qmax']):.2f}",
                f"{nq * n_phi:,}",
                fmt_s(float(row["rdep_analytic_s"])),
                fmt_s(float(row["rdep_fused_s"])),
                fmt_s(float(row["nufft_s"])),
                f"{float(row['rdep_speedup_vs_nufft']):.2f}x",
            ]
        )
    return rows


def waxs_highq_rows() -> list[list[str]]:
    payload = load_json("benchmark_results/algorithm_scaling_highq_direct_fftfriendly_nufft_qb2.json")
    rows = [["atoms", "n_phi", "R-dep", "FINUFFT", "speedup", "I error"]]
    for row in payload["rows"]:
        atoms = int(row["atoms"])
        rows.append(
            [
                f"{atoms // 1000}k" if atoms < 1_000_000 else "1M",
                str(row["grid"]["n_phi"]),
                fmt_s(float(row["rdep_analytic_s"])),
                fmt_s(float(row["nufft_s"])),
                f"{float(row['rdep_speedup_vs_nufft']):.2f}x",
                f"{float(row['rdep_analytic_intensity_rel_l2_vs_dense']):.1e}",
            ]
        )
    return rows


def high_na_package_rows() -> list[list[str]]:
    rows = [["case", "comparison", "runtime readout", "accuracy readout"]]
    matched = load_csv("benchmark_results/high_na_psf_generator_matched_cartesian_representative.csv")
    os2 = next(row for row in matched if row["oversample"] == "2")
    rows.append(
        [
            "psf-generator Cartesian stress test",
            "64 x 64 x 5, clear pupil, linear x",
            f"{float(os2['speedup_psf_vs_adapter']):.2f}x faster than package",
            f"adapter-vs-psf intensity L2 {float(os2['adapter_vs_psf_intensity_l2']):.3e}",
        ]
    )
    rows.append(
        [
            "PyFocus XY shape",
            "linear/circular/vortex focal plane",
            "62x-74x hot-loop structured readout",
            "shape L2 <= 1.11e-04, corr ~= 1",
        ]
    )
    rows.append(
        [
            "adaptive pupil spectrum",
            "vortex h=18 / h=30 failure recovery",
            "adds only needed high harmonics",
            "geometric-only L2 ~1 -> 1e-8 to 1e-10",
        ]
    )
    return rows


def odt_reconstruction_rows() -> list[list[str]]:
    rows = [["case", "iterations", "structured total", "FINUFFT total", "total speedup", "operator agreement"]]
    data = load_csv("benchmark_results/odt_reconstruction_benchmark_summary.csv")
    for row in data:
        rows.append(
            [
                row["case"],
                row["iterations"],
                fmt_s(float(row["structured_total_s"])),
                fmt_s(float(row["finufft_total_s"])),
                f"{float(row['total_speedup']):.2f}x",
                f"{float(row['operator_agreement_l2']):.1e}",
            ]
        )
    return rows


def domain_mapping_table(styles: dict[str, ParagraphStyle]) -> Table:
    return table(
        [
            ["domain", "target manifold", "source representation", "factorization handle", "validated role"],
            [
                "WAXS",
                "Ewald ring/cake: (q, phi)",
                "atoms -> H_e(R, z, beta)",
                "FFT_beta, circular harmonics, R-dependent h_max(q,R)",
                "primary forward validation",
            ],
            [
                "High-NA",
                "focal volume: (rho, psi, z)",
                "pupil Jones field P_j(theta, phi)",
                "pupil azimuth harmonics, Bessel radial kernels, adaptive h set",
                "optics transfer and design-loop validation",
            ],
            [
                "ODT",
                "cone-axis detector cap / q manifold",
                "object bins and residuals",
                "axis split, h/l slots, prepared adjoint contraction",
                "inverse/backpropagation transfer",
            ],
        ],
        styles,
        col_widths=[0.75 * inch, 1.35 * inch, 1.55 * inch, 2.0 * inch, 1.35 * inch],
        font_size=6.6,
    )


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawRightString(A4[0] - 0.45 * inch, 0.33 * inch, f"{doc.page}")
    canvas.restoreState()


def build_story(styles: dict[str, ParagraphStyle]) -> list[Any]:
    story: list[Any] = []
    story.append(p("Geometry-aware Fourier factorization", styles, "title"))
    story.append(
        p(
            f"WAXS, High-NA optics, ODT를 하나의 계산 원리로 묶기 위한 theory draft. 작성일: {date(2026, 6, 22).isoformat()}",
            styles,
            "subtitle",
        )
    )
    story.append(
        p(
            "목표는 세 분야의 결과를 단순한 benchmark 모음으로 보이지 않게 만드는 것이다. 공통 메시지는 다음과 같다. 반복되는 wave-field 평가가 curved/cylindrical target manifold 위에서 일어날 때, target을 임의의 점 집합으로 보지 않고 물리 좌표계의 반복 구조를 드러내면 harmonic reuse, FFT, recurrence, prepared contraction으로 계산량을 줄일 수 있다.",
            styles,
        )
    )
    story.append(Spacer(1, 0.08 * inch))
    story.append(pipeline_diagram(styles))
    story.append(Spacer(1, 0.12 * inch))
    story.append(p("한 문장 claim", styles, "h2"))
    story.append(
        equation(
            "We introduce a geometry-aware Fourier factorization for repeated wave-field evaluation on structured curved manifolds. The method exposes radial, axial, and azimuthal separability and replaces target-wise evaluation with harmonic reuse and prepared contractions.",
            styles,
        )
    )
    story.extend(
        bullets(
            [
                "WAXS는 가장 강한 primary validation이다. high-q cake-map에서 geometry-aware factorization의 scaling advantage가 가장 명확하다.",
                "High-NA optics는 같은 수학 구조가 Debye-Wolf/Richards-Wolf focal-volume evaluation으로 이전됨을 보여준다.",
                "ODT는 forward보다 adjoint/backpropagation workload에서 구조적 이득이 더 직접적으로 드러남을 보여준다.",
                "주장 경계는 명확해야 한다. 이 방법은 universal NUFFT replacement가 아니라 structured curved/cylindrical manifold workload를 위한 domain-aware accelerator다.",
            ],
            styles,
        )
    )

    story.append(PageBreak())
    story.append(p("1. 공통 수식: target-wise sum을 manifold factorization으로 바꾼다", styles, "h1"))
    story.append(
        p(
            "세 응용은 겉으로는 다르지만, 계산 핵심은 모두 source coordinate에서 target manifold로 가는 oscillatory kernel의 반복 평가다. 일반 direct method나 NUFFT baseline은 target을 독립적인 비균일 점으로 처리한다. 여기서는 target이 반복 좌표를 가진다는 점을 먼저 사용한다.",
            styles,
        )
    )
    story.append(
        equation(
            "Generic operator:\n"
            "    y(xi) = sum_j a_j w_j K(xi, x_j)\n"
            "    xi = target coordinate on a detector, focal volume, or cone/cap manifold\n"
            "    x_j = atom, pupil sample, object bin, or residual bin",
            styles,
        )
    )
    story.append(
        p(
            "핵심은 kernel K를 모든 xi에 대해 직접 평가하지 않고, target manifold의 각도/반경/축방향 반복 구조에 맞춰 분리하는 것이다. 대표 형태는 아래처럼 쓸 수 있다.",
            styles,
        )
    )
    story.append(
        equation(
            "Structured expansion:\n"
            "    K((u, psi, z), x) ~= sum_h B_h(u, z; x) exp(i h psi)\n\n"
            "Prepared execution:\n"
            "    Y_h(u, z) = sum_b C_{h,b} B_{h,b}(u, z)\n"
            "    y(u, psi, z) = sum_h Y_h(u, z) exp(i h psi)",
            styles,
        )
    )
    story.append(
        p(
            "여기서 b는 원자 bin, pupil theta sample, object bin 등 domain별 source index다. forward에서는 harmonic coefficients를 재조합하고, adjoint에서는 residual을 같은 harmonic basis로 되돌려 source gradient 또는 object update를 얻는다.",
            styles,
        )
    )
    story.append(
        equation(
            "Adjoint reuse:\n"
            "    r_h(u, z) = FFT_psi[ r(u, psi, z) ]\n"
            "    grad_b += sum_{h,u,z} conj(B_{h,b}(u,z)) r_h(u,z)",
            styles,
        )
    )
    story.append(p("이 구조가 통하는 조건", styles, "h2"))
    story.append(
        table(
            [
                ["condition", "meaning", "why it matters"],
                ["structured target manifold", "target이 (radial, axial, azimuthal) 또는 유사 좌표로 반복됨", "angle FFT와 radial/axial basis reuse가 가능"],
                ["bounded or adaptive harmonic bandwidth", "필요한 h만 계산 가능", "dense global target-wise work를 sparse harmonic work로 바꿈"],
                ["fixed geometry reused many times", "same detector, same focal ROI, same lab geometry", "plan/build cost를 반복 mask, mode, residual에 amortize"],
                ["explicit validity boundary", "arbitrary point cloud나 tiny one-shot Cartesian만 목표가 아님", "overclaim을 막고 reviewer attack surface를 줄임"],
            ],
            styles,
            col_widths=[1.55 * inch, 2.45 * inch, 2.75 * inch],
        )
    )

    story.append(PageBreak())
    story.append(p("2. 세 분야는 같은 operator의 서로 다른 instance다", styles, "h1"))
    story.append(domain_mapping_table(styles))
    story.append(p("WAXS instance", styles, "h2"))
    story.append(
        equation(
            "A(q, phi) = sum_j f_j(q) exp(i q(q,phi) dot r_j)\n\n"
            "After cylindrical binning:\n"
            "    H_e(R, z, beta) = sum_{j in bin} f_e(q) delta(R_j,z_j,beta_j)\n"
            "    A(q,phi) ~= sum_{R,z,h} H_hat_h(R,z) M_h(q,R,z) exp(i h phi)",
            styles,
        )
    )
    story.append(p("High-NA instance", styles, "h2"))
    story.append(
        equation(
            "E_c(rho, psi, z) = int dtheta dphi sum_j M_cj(theta,phi) P_j(theta,phi)\n"
            "                  exp(i k z cos(theta)) exp(i k rho sin(theta) cos(phi-psi))\n\n"
            "Azimuthal expansion:\n"
            "    exp(i a cos(phi-psi)) = sum_h i^h J_h(a) exp(i h (phi-psi))",
            styles,
        )
    )
    story.append(p("ODT instance", styles, "h2"))
    story.append(
        equation(
            "Forward:    y(q) = sum_x f(x) exp(i q dot x)\n"
            "Adjoint:    g(x) = sum_q r(q) exp(-i q dot x)\n\n"
            "Cone-axis geometry exposes repeated q_perp directions and axial factors,\n"
            "so the adjoint can reuse prepared h/l slots instead of calling a generic type-3 NUFFT for every residual.",
            styles,
        )
    )
    story.append(
        p(
            "중요한 점은 세 instance가 같은 모양의 수식을 공유하지만, 물리적 source와 target, 오차 기준, 유리한 workload는 다르다는 것이다. 따라서 paper에서는 공통 vocabulary를 쓰되 domain별 claim boundary를 표로 분리해야 한다.",
            styles,
        )
    )

    story.append(PageBreak())
    story.append(p("3. WAXS validation: 가장 직접적인 flagship evidence", styles, "h1"))
    story.append(
        p(
            "WAXS는 이 framework의 home field다. atomistic source가 cylindrical histogram으로 안정적으로 투영되고, detector target은 Ewald ring/cake geometry를 갖는다. high-q, fixed-dq 조건에서는 q range가 커질수록 target 수와 angular bandwidth가 함께 증가하므로 구조적 이득이 커진다.",
            styles,
        )
    )
    story.append(
        table(
            waxs_highq_rows(),
            styles,
            col_widths=[0.75 * inch, 0.65 * inch, 0.85 * inch, 0.90 * inch, 0.78 * inch, 0.80 * inch],
        )
    )
    story.append(Spacer(1, 0.05 * inch))
    story.append(
        table(
            waxs_qmax_rows(),
            styles,
            col_widths=[0.60 * inch, 0.95 * inch, 0.85 * inch, 0.85 * inch, 0.85 * inch, 0.82 * inch],
        )
    )
    story.append(
        image_block(
            "benchmark_results/paper_strategy_assets/waxs_qmax_speedup.png",
            "Fixed-dq q-range sweep. qmax가 커질수록 target count와 angular bandwidth가 함께 커져 WAXS-specific factorization의 이득이 증가한다.",
            styles,
            max_height=3.2 * inch,
        )
    )
    story.append(
        p(
            "Manuscript role: WAXS는 correctness와 scaling을 동시에 보여주는 primary validation이다. 여기서 claim은 universal NUFFT replacement가 아니라 WAXS cake-map geometry의 반복 구조를 이용한 controlled acceleration이다.",
            styles,
        )
    )

    story.append(PageBreak())
    story.append(p("4. High-NA optics transfer: Debye-Wolf focal volume도 같은 구조를 갖는다", styles, "h1"))
    story.append(
        p(
            "High-NA optics에서는 source가 atom이 아니라 pupil field이고, target은 detector가 아니라 focal volume이다. 하지만 azimuthal phase structure는 동일하게 harmonic expansion으로 분리된다. 특히 phase mask, coherent mode, inverse-design loop처럼 같은 focal ROI를 반복 평가하는 경우 plan reuse가 의미를 갖는다.",
            styles,
        )
    )
    story.append(
        table(
            high_na_package_rows(),
            styles,
            col_widths=[1.55 * inch, 1.70 * inch, 1.55 * inch, 1.85 * inch],
        )
    )
    story.append(
        image_block(
            "benchmark_results/paper_strategy_assets/high_na_adaptive_speedup.png",
            "High-NA adaptive cutoff summary. geometric-only cutoff는 high-azimuthal pupil content에서 실패하고, adaptive pupil spectrum이 필요한 harmonic만 추가해 복구한다.",
            styles,
            max_height=2.55 * inch,
        )
    )
    story.append(
        image_block(
            "benchmark_results/high_na_external_package_comparison.png",
            "External package anchor. PyFocus shape agreement와 psf-generator timing은 optics-domain sanity check로 사용한다. Cartesian은 non-native stress test로만 해석한다.",
            styles,
            max_height=2.35 * inch,
        )
    )
    story.append(
        p(
            "Manuscript role: High-NA는 cross-field transfer를 보여준다. 가장 안전한 표현은 full Cartesian PSF rasterizer replacement가 아니라 structured cylindrical ROI와 repeated phase-mask/coherent-mode workload의 acceleration이다.",
            styles,
        )
    )

    story.append(PageBreak())
    story.append(p("5. ODT transfer: inverse/backpropagation에서 구조적 이득이 가장 잘 맞는다", styles, "h1"))
    story.append(
        p(
            "ODT는 forward보다 adjoint/backpropagation에서 WAXS와 방향성이 더 잘 맞는다. residual on detector manifold를 object space로 되돌리는 연산은 real-space source update를 만드는 adjoint이며, fixed lab geometry가 반복되므로 prepared plan의 amortization이 자연스럽다.",
            styles,
        )
    )
    story.append(
        table(
            odt_reconstruction_rows(),
            styles,
            col_widths=[1.1 * inch, 0.7 * inch, 1.0 * inch, 1.0 * inch, 0.9 * inch, 1.0 * inch],
        )
    )
    story.append(
        image_block(
            "benchmark_results/paper_strategy_assets/odt_gpu_speedup.png",
            "ODT GPU update speedup. detector sampling이 커질수록 cuFINUFFT 대비 update speedup이 커진다.",
            styles,
            max_height=2.6 * inch,
        )
    )
    story.append(
        image_block(
            "benchmark_results/odt_reconstruction_benchmark_summary.png",
            "ODT reconstruction amortization. 반복 residual/update workload에서 setup cost가 빠르게 amortize된다.",
            styles,
            max_height=2.7 * inch,
        )
    )
    story.append(
        p(
            "Manuscript role: ODT는 inverse/backpropagation transfer다. full reconstruction solver 전체를 대체한다고 쓰지 말고, prepared adjoint and repeated reconstruction update acceleration으로 경계를 잡는 것이 안전하다.",
            styles,
        )
    )

    story.append(PageBreak())
    story.append(p("6. Regime map: 언제 빠르고 언제 주장하지 않는가", styles, "h1"))
    story.append(
        table(
            [
                ["regime", "direct", "NUFFT / cuFINUFFT", "structured factorization"],
                ["tiny one-shot Cartesian", "often competitive", "often unnecessary", "build/interpolation overhead can dominate"],
                ["large arbitrary nonuniform targets", "too slow", "strong generic baseline", "not claimed unless geometry is structured"],
                ["structured curved/cylindrical manifold", "scales poorly", "works but ignores repeated coordinates", "native target regime"],
                ["many masks / modes / residuals", "repeated full work", "plan reuse helps but generic", "basis and map reuse are decisive"],
                ["high angular bandwidth with local cutoff", "very expensive", "large target/working grid", "adaptive or R-dependent h saves work"],
            ],
            styles,
            col_widths=[1.55 * inch, 1.35 * inch, 1.55 * inch, 2.35 * inch],
        )
    )
    story.append(p("Claim boundary table", styles, "h2"))
    story.append(
        table(
            [
                ["avoid", "use instead", "reason"],
                ["universal NUFFT replacement", "geometry-aware Fourier factorization", "benefit depends on structured target manifold"],
                ["general Fourier transform accelerator", "structured wave-field evaluation on curved/cylindrical manifolds", "arbitrary non-separable targets are outside scope"],
                ["full high-NA optical simulator replacement", "acceleration for structured cylindrical ROI and repeated mask/coherent-mode workloads", "Cartesian PSF is a stress test, not the native target"],
                ["full ODT reconstruction solver", "prepared adjoint acceleration for repeated tomographic backpropagation", "current strongest evidence is adjoint/update loop"],
            ],
            styles,
            col_widths=[1.55 * inch, 2.25 * inch, 2.75 * inch],
        )
    )
    story.append(
        p(
            "NCS급 논문에서는 강한 claim만큼 제한 조건을 선명하게 써야 한다. Fig. 5 또는 SI에는 direct, NUFFT, structured CPU, structured GPU의 유리한 영역을 같은 축에서 비교하는 regime map이 필요하다.",
            styles,
        )
    )

    story.append(PageBreak())
    story.append(p("7. Manuscript-level integration plan", styles, "h1"))
    story.append(
        table(
            [
                ["figure", "content", "purpose"],
                ["Fig. 1 framework", "operator template, manifold coordinates, harmonic reuse pipeline", "benchmark collection이 아니라 하나의 method임을 보여줌"],
                ["Fig. 2 WAXS flagship", "Ewald/cake geometry, fixed-dq qmax scaling, memory/RSS, ablation", "primary validation"],
                ["Fig. 3 High-NA transfer", "Debye-Wolf ROI, adaptive cutoff, matched package stress test, phase-mask demo", "optics domain transfer"],
                ["Fig. 4 ODT inverse transfer", "cone-axis geometry, prepared adjoint, amortization, adjointness", "inverse/backpropagation transfer"],
                ["Fig. 5 regime map", "target size/bandwidth vs repetition count", "when to use direct, NUFFT, or structured method"],
            ],
            styles,
            col_widths=[1.05 * inch, 3.2 * inch, 2.25 * inch],
        )
    )
    story.append(p("Immediate next tasks", styles, "h2"))
    story.extend(
        bullets(
            [
                "WAXS realistic NPZ benchmark와 memory/RSS figure를 publication-quality로 재생성한다.",
                "High-NA matched package comparison은 psf-generator stress test를 SI 또는 Fig. 3 inset으로 정리하고, 가능하면 package/discretization error floor를 더 줄인다.",
                "ODT는 reconstruction/update amortization, adjointness, memory, realistic lab geometry를 하나의 benchmark table로 고정한다.",
                "모든 figure는 raw JSON/CSV와 generation script에서 재생성되도록 정리한다.",
                "Supplementary theory에는 세 domain의 kernel factorization과 validity boundary를 더 엄밀하게 적는다.",
            ],
            styles,
        )
    )
    story.append(
        p(
            "Draft conclusion: 현재 결과는 NCS 통합 논문으로 밀 가능성이 있다. 다만 결정적인 일은 추가적인 10-20% 최적화가 아니라, 공통 수식, figure architecture, reproducibility package를 통해 세 응용이 하나의 계산 원리에서 나온다는 점을 분명히 만드는 것이다.",
            styles,
        )
    )

    story.append(p("Source files used", styles, "h1"))
    for source in [
        "benchmark_results/algorithm_scaling_highq_direct_fftfriendly_nufft_qb2.json",
        "benchmark_results/qmax_scaling_1m_dq0p160_q*.json",
        "benchmark_results/high_na_psf_generator_matched_cartesian_representative.csv",
        "benchmark_results/high_na_external_package_comparison.md",
        "benchmark_results/high_na_pupil_spectrum_option_matrix_summary.md",
        "benchmark_results/odt_reconstruction_benchmark_summary.csv",
        "benchmark_results/odt_gpu_reconstruction_compare_overview.md",
        "benchmark_results/paper_strategy_assets/*.png",
    ]:
        story.append(p(source, styles, "small"))
    return story


def write_source_notes() -> None:
    notes = """# Geometry-aware Fourier factorization theory draft source notes

Generated artifact:

- `benchmark_results/geometry_aware_fourier_factorization_theory_draft_ko.pdf`

Purpose:

- Theory draft for a unified WAXS, High-NA optics, and ODT framework paper.
- Korean explanatory text with equations and existing benchmark/demo images.

Primary numeric sources:

- `benchmark_results/algorithm_scaling_highq_direct_fftfriendly_nufft_qb2.json`
- `benchmark_results/qmax_scaling_1m_dq0p160_q*.json`
- `benchmark_results/high_na_psf_generator_matched_cartesian_representative.csv`
- `benchmark_results/high_na_external_package_comparison.md`
- `benchmark_results/high_na_pupil_spectrum_option_matrix_summary.md`
- `benchmark_results/odt_reconstruction_benchmark_summary.csv`
- `benchmark_results/odt_gpu_reconstruction_compare_overview.md`

Primary visual sources:

- `benchmark_results/paper_strategy_assets/waxs_qmax_speedup.png`
- `benchmark_results/paper_strategy_assets/high_na_adaptive_speedup.png`
- `benchmark_results/high_na_external_package_comparison.png`
- `benchmark_results/paper_strategy_assets/odt_gpu_speedup.png`
- `benchmark_results/odt_reconstruction_benchmark_summary.png`

Claim boundaries:

- Not a universal NUFFT replacement.
- Not a full dense Cartesian high-NA PSF rasterizer replacement.
- Not a full ODT reconstruction solver replacement.
- Framed as geometry-aware acceleration for repeated wave-field evaluation on structured curved/cylindrical manifolds.
"""
    SOURCE_NOTES.write_text(notes, encoding="utf-8")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = make_styles()
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=0.45 * inch,
        rightMargin=0.45 * inch,
        topMargin=0.48 * inch,
        bottomMargin=0.55 * inch,
        title="Geometry-aware Fourier factorization theory draft",
        author="Codex",
    )
    doc.build(build_story(styles), onFirstPage=add_page_number, onLaterPages=add_page_number)
    write_source_notes()
    print(OUT)
    print(SOURCE_NOTES)


if __name__ == "__main__":
    main()
