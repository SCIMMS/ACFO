from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
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


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "benchmark_results"
PDF_PATH = OUT_DIR / "curved_ewald_operator_manuscript_direction_ko.pdf"
SOURCE_NOTES_PATH = OUT_DIR / "curved_ewald_operator_manuscript_direction_source_notes.md"


PALETTE = {
    "ink": "#20242E",
    "muted": "#626C7F",
    "line": "#D9DEE9",
    "soft": "#F3F6FA",
    "panel": "#FFFFFF",
    "blue": "#446BA8",
    "green": "#3E7A5C",
    "gold": "#9B7B2F",
    "red": "#A24A4A",
    "violet": "#6B5EA8",
}


def register_fonts() -> tuple[str, str]:
    font_dir = Path("C:/Windows/Fonts")
    regular = font_dir / "malgun.ttf"
    bold = font_dir / "malgunbd.ttf"
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("KoreanRegular", str(regular)))
        pdfmetrics.registerFont(TTFont("KoreanBold", str(bold)))
        return "KoreanRegular", "KoreanBold"
    return "Helvetica", "Helvetica-Bold"


BODY_FONT, BOLD_FONT = register_fonts()


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName=BOLD_FONT,
            fontSize=20,
            leading=27,
            alignment=TA_CENTER,
            textColor=colors.HexColor(PALETTE["ink"]),
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["BodyText"],
            fontName=BODY_FONT,
            fontSize=9,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor(PALETTE["muted"]),
            spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName=BOLD_FONT,
            fontSize=14,
            leading=18,
            textColor=colors.HexColor(PALETTE["ink"]),
            spaceBefore=12,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=BOLD_FONT,
            fontSize=11,
            leading=15,
            textColor=colors.HexColor(PALETTE["ink"]),
            spaceBefore=8,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName=BODY_FONT,
            fontSize=9,
            leading=13,
            textColor=colors.HexColor(PALETTE["ink"]),
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName=BODY_FONT,
            fontSize=7.7,
            leading=10.5,
            textColor=colors.HexColor(PALETTE["muted"]),
            spaceAfter=4,
        ),
        "table": ParagraphStyle(
            "table",
            parent=base["BodyText"],
            fontName=BODY_FONT,
            fontSize=7.4,
            leading=9.6,
            textColor=colors.HexColor(PALETTE["ink"]),
        ),
        "table_bold": ParagraphStyle(
            "table_bold",
            parent=base["BodyText"],
            fontName=BOLD_FONT,
            fontSize=7.4,
            leading=9.6,
            textColor=colors.HexColor(PALETTE["ink"]),
        ),
        "quote": ParagraphStyle(
            "quote",
            parent=base["BodyText"],
            fontName=BODY_FONT,
            fontSize=8.4,
            leading=12,
            leftIndent=8,
            rightIndent=8,
            textColor=colors.HexColor(PALETTE["ink"]),
            borderColor=colors.HexColor(PALETTE["line"]),
            borderWidth=0.5,
            borderPadding=7,
            backColor=colors.HexColor("#FAFBFD"),
            spaceBefore=4,
            spaceAfter=10,
        ),
    }


S = styles()


def p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def cell(text: str, bold: bool = False) -> Paragraph:
    return Paragraph(text, S["table_bold" if bold else "table"])


def bullet(items: list[str]) -> ListFlowable:
    return ListFlowable(
        [ListItem(p(item, "body"), leftIndent=11) for item in items],
        bulletType="bullet",
        start="disc",
        leftIndent=14,
        bulletFontName=BODY_FONT,
        bulletFontSize=7,
    )


def table(data, widths, header_rows: int = 1, font_size: float | None = None) -> Table:
    prepared = []
    for r, row in enumerate(data):
        prepared.append([cell(str(x), bold=r < header_rows) for x in row])
    t = Table(prepared, colWidths=widths, hAlign="LEFT", repeatRows=header_rows)
    style = [
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#E9EEF7")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(PALETTE["ink"])),
        ("FONTNAME", (0, 0), (-1, header_rows - 1), BOLD_FONT),
        ("FONTNAME", (0, header_rows), (-1, -1), BODY_FONT),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(PALETTE["line"])),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, header_rows), (-1, -1), [colors.white, colors.HexColor("#FBFCFE")]),
    ]
    if font_size is not None:
        style.append(("FONTSIZE", (0, 0), (-1, -1), font_size))
    t.setStyle(TableStyle(style))
    return t


def flow_boxes(rows: list[tuple[str, str, str]]) -> Table:
    data = []
    for title, role, color in rows:
        data.append(
            [
                Paragraph(
                    f"<font name='{BOLD_FONT}' color='{color}'>{escape(title)}</font><br/>"
                    f"<font name='{BODY_FONT}' color='{PALETTE['ink']}'>{escape(role)}</font>",
                    S["table"],
                )
            ]
        )
    t = Table(data, colWidths=[16.8 * cm], hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFFFF")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(PALETTE["line"])),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor(PALETTE["line"])),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return t


def build_story() -> list:
    story: list = []
    story.append(p("Curved-Ewald Operator Manuscript Direction", "title"))
    story.append(
        p(
            "High-NA를 main validation이 아니라 선행연구 anchor 및 Supplement bridge로 낮추고, WAXS/ODT를 main evidence로 세우는 작성 방향성<br/>Generated: 2026-06-23",
            "subtitle",
        )
    )

    story.append(p("Executive Decision", "h1"))
    story.append(
        p(
            "현재 논문은 <b>factorization 자체를 새로 발견했다</b>고 주장하면 위험하다. 더 안전하고 강한 주장은, "
            "<b>이미 알려진 circular-harmonic / Fourier-Bessel 계열 factorization을 curved Ewald/cap Fourier geometry에 prepared forward/adjoint operator로 적용했다</b>는 것이다."
        )
    )
    story.append(
        p(
            "따라서 High-NA는 main Results의 중심이 아니라, Introduction과 Supplement에서 known reduction을 회수하는 anchor로 배치한다. "
            "Main Results는 WAXS와 ODT에 집중한다."
        )
    )
    story.append(
        p(
            "Safe central claim: We do not introduce the circular-harmonic identity itself. "
            "We show that this known class of factorization can be prepared as reusable curved-Ewald/cap Fourier operators and applied to WAXS and ODT geometries.",
            "quote",
        )
    )

    story.append(p("Paper Logic", "h1"))
    story.append(
        flow_boxes(
            [
                (
                    "Known Anchor: High-NA optics",
                    "Debye-Wolf/Richards-Wolf의 Fourier-Bessel 구조는 이미 알려져 있다. 논문에서는 이 구조가 낯선 수학이 아니라는 배경과 sanity check로 사용한다.",
                    PALETTE["violet"],
                ),
                (
                    "New-Domain Validation: WAXS",
                    "동일한 harmonic/cylindrical 구조가 atomistic WAXS curved-Ewald cake map에서 reference와 맞고 scaling 이득을 준다는 것을 보여준다.",
                    PALETTE["blue"],
                ),
                (
                    "Impact Case: ODT/aIDT",
                    "fixed lab geometry와 repeated forward/adjoint update에서 prepared operator가 real-time processing feasibility로 이어질 수 있음을 보여준다.",
                    PALETTE["green"],
                ),
            ]
        )
    )

    story.append(p("Recommended Main-Text Structure", "h1"))
    story.append(
        table(
            [
                ["Section", "Role", "Main content", "Claim boundary"],
                [
                    "Abstract",
                    "One-paragraph claim map",
                    "curved Ewald/cap Fourier 병목, prepared operator, WAXS validation, ODT impact, High-NA known bridge를 압축",
                    "universal NUFFT/PSF/reconstruction replacement라고 쓰지 않음",
                ],
                [
                    "Introduction",
                    "Problem + prior-art positioning",
                    "High-NA에서 known Fourier-Bessel reduction이 이미 중요하다는 점을 먼저 밝히고, 그 구조를 WAXS/ODT curved Ewald operator로 확장하는 질문 제시",
                    "High-NA를 novelty처럼 보이게 하지 않음",
                ],
                [
                    "Theory",
                    "General operator formulation",
                    "curved manifold Fourier samples, angular harmonic expansion, setup/hot-loop/adjoint, build amortization, favorable/unfavorable regime",
                    "identity 자체가 아니라 prepared operatorization이 기여",
                ],
                [
                    "Result 1: WAXS",
                    "Main correctness/scaling validation",
                    "direct Debye/cake, form factors, public structure/CIF, TIP3P trajectory frame-average, fixed-dq high-q scaling",
                    "WAXS geometry-aware method이지 generic Fourier replacement가 아님",
                ],
                [
                    "Result 2: ODT/aIDT",
                    "Main application impact",
                    "operator validation, adjoint identity, GPU warm-start update, public aIDT full-condition 10 Hz projection",
                    "end-to-end live microscope가 아니라 processing-side feasibility",
                ],
                [
                    "Regime Map",
                    "Generalization without overclaiming",
                    "geometry reuse, detector/cap size, repeated evaluations, setup cost, Cartesian penalty를 축으로 favorable regime 정리",
                    "불리한 영역도 명시",
                ],
                [
                    "Discussion",
                    "Interpretation and limitations",
                    "왜 WAXS/ODT에서 이득이 생기는지, High-NA prior art와의 관계, NUFFT/FINUFFT/cuFINUFFT와의 관계, 한계와 다음 단계",
                    "실험/패키지/저수준 CUDA 미완성 범위 분리",
                ],
            ],
            [2.6 * cm, 3.1 * cm, 7.1 * cm, 4.0 * cm],
        )
    )

    story.append(PageBreak())
    story.append(p("Where High-NA Should Go", "h1"))
    story.append(
        p(
            "High-NA를 Result section으로 크게 세우면 reviewer가 novelty를 그쪽에서 찾을 가능성이 있다. "
            "그 경우 '이미 알려진 Fourier-Bessel/Richards-Wolf reduction 아닌가?'라는 비판이 쉬워진다. "
            "따라서 main text에서는 짧게 연결하고, 상세 검증은 Supplement에 둔다."
        )
    )
    story.append(
        table(
            [
                ["Placement", "Recommended content", "Purpose"],
                [
                    "Introduction",
                    "High-NA Debye-Wolf/Richards-Wolf에서 같은 harmonic reduction이 알려져 있다는 사실을 명시",
                    "prior art를 인정하고, 새 claim의 위치를 낮춤",
                ],
                [
                    "Theory",
                    "known motivating example로 1 paragraph 또는 boxed note",
                    "factorization family가 물리적으로 익숙한 구조임을 보여줌",
                ],
                [
                    "Supplementary Note",
                    "scalar/vectorial validation, adaptive pupil spectrum, PyFocus/PyCustomFocus/psf-generator, cuFINUFFT 비교",
                    "reviewer 방어자료와 practical demo",
                ],
                [
                    "Main Results",
                    "원칙적으로 제외. 필요하면 regime map에서 작은 marker로만 언급",
                    "main novelty가 High-NA로 읽히는 것을 방지",
                ],
            ],
            [3.1 * cm, 8.1 * cm, 5.6 * cm],
        )
    )

    story.append(p("Evidence Already Ready", "h1"))
    story.append(
        table(
            [
                ["Evidence lane", "Current readout", "Use in manuscript"],
                [
                    "WAXS TIP3P trajectory",
                    "5-frame 8 nm TIP3P validation. Full all-atom cake: mean speedup 250.93x vs direct, mean 2D L2 3.404e-04, max 4.922e-04. Sphere Debye-checkable crops stay below 6.0e-4 hist-vs-direct.",
                    "Result 1에서 single-snapshot artifact가 아니라는 solvent/amorphous validation으로 사용",
                ],
                [
                    "High-NA readiness",
                    "Scalar adaptive sparse recovers vortex rows to 4.3e-10 / 4.0e-08; vectorial separable path matches direct near 3.443e-15 forward and 8.843e-16 adjoint; GPU vectorial/cylindrical target rows show strong speedups.",
                    "Supplement에서 known-case recovery 및 package cross-check로 사용",
                ],
                [
                    "ODT warm-start",
                    "RTX 2070 SUPER PyTorch prototype: 1 update/frame 7.48 ms, 3 updates/frame 23.44 ms, 8 updates/frame 59.32 ms; 15-frame dynamic sequence on fixed geometry.",
                    "Result 2에서 repeated inverse/backpropagation use case의 practical value로 사용",
                ],
                [
                    "Public aIDT projection",
                    "Full 24 x 700 x 700, 700 x 700 x 35 condition: GPU-resident core 0.0970 s = 10.31 Hz; copy+core 0.1066 s = 9.38 Hz; overlap or 1.075x compute speedup reaches 10 Hz.",
                    "processing-side real-time feasibility claim의 핵심 수치로 사용",
                ],
            ],
            [3.3 * cm, 8.4 * cm, 5.1 * cm],
        )
    )

    story.append(p("Main Figures", "h1"))
    story.append(
        table(
            [
                ["Figure", "Content", "Message"],
                [
                    "Fig. 1",
                    "Prepared curved-Ewald/cap Fourier operator schematic: source representation, curved manifold, harmonic factors, setup vs hot-loop forward/adjoint",
                    "한 개의 operator idea가 WAXS/ODT에 공통으로 쓰임",
                ],
                [
                    "Fig. 2",
                    "WAXS validation and scaling: direct/reference agreement, TIP3P frame-average, fixed-dq high-q scaling, NUFFT/direct comparison",
                    "main correctness/scaling validation",
                ],
                [
                    "Fig. 3",
                    "ODT operator and reconstruction impact: adjoint identity, warm-start update count vs error/FPS, public aIDT 10 Hz projection",
                    "repeated inverse problem에서의 impact",
                ],
                [
                    "Fig. 4",
                    "Regime map: geometry reuse, detector/cap size, target coordinates, setup amortization, Cartesian penalty",
                    "어디서 이기고 어디서 불리한지 명확화",
                ],
            ],
            [2.2 * cm, 8.2 * cm, 6.4 * cm],
        )
    )

    story.append(p("Supplement Structure", "h1"))
    story.append(
        bullet(
            [
                "Supplementary Note 1: relation to established High-NA Fourier-Bessel / circular-harmonic formulations.",
                "Supplementary Note 2: scalar and vectorial High-NA validation, adaptive pupil spectrum, package comparison, Cartesian stress test.",
                "Supplementary Note 3: WAXS benchmark details, form-factor model, CIF/PDB/TIP3P preprocessing, Debye/direct/NUFFT fairness.",
                "Supplementary Note 4: ODT forward/adjoint identities, GPU implementation, warm-start protocol, public aIDT projection.",
                "Supplementary Note 5: reproducibility guide, commands, environment, committed-vs-generated data table.",
            ]
        )
    )

    story.append(PageBreak())
    story.append(p("Recommended Wording", "h1"))
    story.append(p("Possible title", "h2"))
    story.append(
        p(
            "Prepared curved-Ewald Fourier operators for accelerated scattering and diffraction tomography",
            "quote",
        )
    )
    story.append(p("Core contribution sentence", "h2"))
    story.append(
        p(
            "We formulate a known circular-harmonic/Fourier-Bessel factorization as reusable prepared forward and adjoint operators for rotationally structured curved Ewald and cap-Fourier manifolds, and demonstrate its value in WAXS simulation and ODT/aIDT reconstruction workflows.",
            "quote",
        )
    )
    story.append(p("High-NA boundary sentence", "h2"))
    story.append(
        p(
            "High-NA optics is used here as an established reference point for the same harmonic structure, not as the primary novelty claim; detailed High-NA validation is reported in the Supplement.",
            "quote",
        )
    )
    story.append(p("ODT boundary sentence", "h2"))
    story.append(
        p(
            "The ODT/aIDT results support processing-side real-time feasibility on fixed geometries; they do not yet constitute an end-to-end live microscope demonstration.",
            "quote",
        )
    )

    story.append(p("Do Not Claim", "h1"))
    story.append(
        bullet(
            [
                "Do not claim that the circular-harmonic identity or Fourier-Bessel reduction is new.",
                "Do not claim a universal NUFFT replacement; the method is geometry-aware and regime-dependent.",
                "Do not claim dense Cartesian High-NA PSF replacement from the current evidence.",
                "Do not claim completed end-to-end real-time ODT acquisition; claim processing-side feasibility.",
                "Do not make High-NA a main validation section unless the paper is reframed as an optics paper.",
            ]
        )
    )

    story.append(p("Immediate Manuscript Tasks", "h1"))
    story.append(
        table(
            [
                ["Priority", "Task", "Output"],
                [
                    "1",
                    "Write the main claim map and section skeleton using the structure above.",
                    "1-2 page manuscript outline",
                ],
                [
                    "2",
                    "Make Fig. 1 and Fig. 4 first, because they control whether the cross-domain story reads as one method or a loose collection.",
                    "theory schematic + regime map",
                ],
                [
                    "3",
                    "Freeze WAXS and ODT benchmark tables for main text; move High-NA package comparisons to SI.",
                    "source-backed main figure data tables",
                ],
                [
                    "4",
                    "Create reproducibility/source-data table: committed, generated, external, omitted.",
                    "Methods/SI reproducibility package",
                ],
            ],
            [2.1 * cm, 10.4 * cm, 4.3 * cm],
        )
    )

    story.append(Spacer(1, 12))
    story.append(
        p(
            "Source files used for the evidence snapshot are listed in the companion source-notes markdown file generated with this PDF.",
            "small",
        )
    )
    return story


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(BODY_FONT, 7)
    canvas.setFillColor(colors.HexColor(PALETTE["muted"]))
    canvas.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"{doc.page}")
    canvas.drawString(1.6 * cm, 1.0 * cm, "Curved-Ewald operator manuscript direction")
    canvas.restoreState()


def write_source_notes() -> None:
    lines = [
        "# Curved-Ewald operator manuscript direction source notes",
        "",
        "Generated: 2026-06-23",
        "",
        "## Scope",
        "",
        "- This note records the manuscript direction after deciding that High-NA should be treated as a prior-art-connected anchor and Supplement bridge, not as the main validation section.",
        "- Main-text evidence should center on WAXS correctness/scaling and ODT/aIDT repeated inverse-processing impact.",
        "",
        "## Source artifacts",
        "",
        "- `benchmark_results/solvent_water_tip3p_8nm_traj5_frame_average_summary.md`: TIP3P trajectory frame-average WAXS validation.",
        "- `benchmark_results/high_na_publication_readiness_summary.md`: High-NA claim-boundary and package-comparison readiness summary.",
        "- `benchmark_results/odt_torch_gpu_dynamic_warm_start_update_saturation_summary.md`: ODT dynamic warm-start update/FPS/error summary.",
        "- `benchmark_results/aidt_realtime_projection_summary.md`: public aIDT full-condition 10 Hz processing-side projection.",
        "",
        "## Main claim boundary",
        "",
        "- The circular-harmonic/Fourier-Bessel identity is not the novelty.",
        "- The manuscript should claim prepared curved-Ewald/cap Fourier operatorization and geometry-specific reuse.",
        "- WAXS is the main correctness/scaling validation.",
        "- ODT/aIDT is the main impact validation.",
        "- High-NA belongs mainly in Introduction/Methods/Supplement as a known reduction recovered by the same formulation.",
    ]
    SOURCE_NOTES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_source_notes()
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.55 * cm,
        leftMargin=1.55 * cm,
        topMargin=1.55 * cm,
        bottomMargin=1.45 * cm,
        title="Curved-Ewald operator manuscript direction",
        author="Atomic WAXS Cake-Map Simulator",
    )
    doc.build(build_story(), onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(PDF_PATH)
    print(SOURCE_NOTES_PATH)


if __name__ == "__main__":
    main()
