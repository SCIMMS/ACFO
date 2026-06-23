from __future__ import annotations

from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "curved_ewald_factorization_strategy_ko.pdf"
SOURCE_NOTES = ROOT / "benchmark_results" / "curved_ewald_factorization_strategy_sources.md"


INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#4b5563")
GRID = colors.HexColor("#d1d5db")
HEADER = colors.HexColor("#172554")
SOFT_BLUE = colors.HexColor("#eff6ff")
SOFT_GREEN = colors.HexColor("#ecfdf5")
SOFT_AMBER = colors.HexColor("#fffbeb")
SOFT_ROSE = colors.HexColor("#fff1f2")
SOFT_GRAY = colors.HexColor("#f9fafb")


SOURCES = [
    {
        "key": "Boichenko 2022/2023",
        "url": "https://arxiv.org/abs/2212.10978",
        "role": "High-NA Richards-Wolf circularly polarized vortex-mode reduction; known precedent, not our novelty.",
    },
    {
        "key": "Kirisits et al. 2024/2025",
        "url": "https://arxiv.org/abs/2407.01793",
        "role": "Generalized Fourier diffraction theorem and filtered backpropagation for diffraction tomography.",
    },
    {
        "key": "Chen et al. 2021",
        "url": "https://arxiv.org/abs/2101.11709",
        "role": "Curved Ewald sphere problem in electron-microscopy reconstruction.",
    },
    {
        "key": "Horstmeyer and Yang 2015",
        "url": "https://arxiv.org/abs/1510.08756",
        "role": "Fourier ptychographic diffraction tomography geometry and iterative reconstruction baseline.",
    },
    {
        "key": "Zuo et al. 2019",
        "url": "https://arxiv.org/abs/1904.09386",
        "role": "High-throughput Fourier ptychographic diffraction tomography application context.",
    },
    {
        "key": "Pratley et al. 2018/2019",
        "url": "https://arxiv.org/abs/1807.09239",
        "role": "Adjacent radio-interferometry example of radial/Hankel structure for wide-field correction.",
    },
    {
        "key": "Zhao et al. 2014/2015",
        "url": "https://arxiv.org/abs/1412.0781",
        "role": "Adjacent Fourier-Bessel basis and NUFFT use in cryo-EM image analysis.",
    },
]


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
            fontSize=19,
            leading=24,
            alignment=TA_LEFT,
            textColor=INK,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleKR",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=9.0,
            leading=13,
            textColor=MUTED,
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "Heading1KR",
            parent=sample["Heading1"],
            fontName=bold_font,
            fontSize=13.6,
            leading=17.2,
            textColor=colors.HexColor("#1e3a8a"),
            spaceBefore=8,
            spaceAfter=5,
        ),
        "h2": ParagraphStyle(
            "Heading2KR",
            parent=sample["Heading2"],
            fontName=bold_font,
            fontSize=10.7,
            leading=14,
            textColor=INK,
            spaceBefore=6,
            spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "BodyKR",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=8.65,
            leading=12.25,
            textColor=INK,
            spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "SmallKR",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=7.1,
            leading=9.25,
            textColor=MUTED,
            spaceAfter=2.5,
        ),
        "table": ParagraphStyle(
            "TableKR",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=6.85,
            leading=8.55,
            textColor=INK,
        ),
        "table_head": ParagraphStyle(
            "TableHeadKR",
            parent=sample["BodyText"],
            fontName=bold_font,
            fontSize=6.85,
            leading=8.55,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "equation": ParagraphStyle(
            "EquationKR",
            parent=sample["Code"],
            fontName=body_font,
            fontSize=8.2,
            leading=11.6,
            textColor=colors.HexColor("#172554"),
            backColor=SOFT_BLUE,
            borderColor=colors.HexColor("#bfdbfe"),
            borderWidth=0.35,
            borderPadding=5,
            spaceBefore=4,
            spaceAfter=5,
        ),
        "callout": ParagraphStyle(
            "CalloutKR",
            parent=sample["BodyText"],
            fontName=bold_font,
            fontSize=8.1,
            leading=11.4,
            textColor=colors.HexColor("#713f12"),
            backColor=SOFT_AMBER,
            borderColor=colors.HexColor("#fde68a"),
            borderWidth=0.35,
            borderPadding=6,
            spaceBefore=4,
            spaceAfter=5,
        ),
        "box": ParagraphStyle(
            "BoxKR",
            parent=sample["BodyText"],
            fontName=bold_font,
            fontSize=7.1,
            leading=9.0,
            textColor=INK,
            alignment=TA_CENTER,
        ),
    }


def text(value: str) -> str:
    return escape(value).replace("\n", "<br/>")


def p(value: str, styles: dict[str, ParagraphStyle], style: str = "body") -> Paragraph:
    return Paragraph(text(value), styles[style])


def bullets(items: list[str], styles: dict[str, ParagraphStyle]) -> list[Paragraph]:
    return [p("- " + item, styles) for item in items]


def make_table(
    rows: list[list[str]],
    styles: dict[str, ParagraphStyle],
    col_widths: list[float],
    *,
    row_colors: list | None = None,
) -> Table:
    rendered: list[list[Paragraph]] = []
    for row_idx, row in enumerate(rows):
        style = styles["table_head"] if row_idx == 0 else styles["table"]
        rendered.append([Paragraph(text(str(cell)), style) for cell in row])

    tbl = Table(rendered, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    commands: list[tuple] = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER),
        ("GRID", (0, 0), (-1, -1), 0.28, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), row_colors or [colors.white, SOFT_GRAY]),
    ]
    tbl.setStyle(TableStyle(commands))
    return tbl


def pipeline(styles: dict[str, ParagraphStyle]) -> Table:
    labels = [
        ("Known optics\nreduction", SOFT_BLUE),
        ("Curved Ewald\noperator view", SOFT_GREEN),
        ("WAXS\nvalidation", SOFT_AMBER),
        ("ODT h/l\nextension", SOFT_ROSE),
        ("Prepared\nforward/adjoint", SOFT_BLUE),
    ]
    row: list[Paragraph] = []
    widths: list[float] = []
    for idx, (label, _) in enumerate(labels):
        row.append(Paragraph(text(label), styles["box"]))
        widths.append(1.23 * inch)
        if idx < len(labels) - 1:
            row.append(Paragraph("->", styles["box"]))
            widths.append(0.25 * inch)
    tbl = Table([row], colWidths=widths, hAlign="CENTER")
    commands: list[tuple] = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    box_col = 0
    for _, fill in labels:
        commands.append(("BACKGROUND", (box_col, 0), (box_col, 0), fill))
        commands.append(("BOX", (box_col, 0), (box_col, 0), 0.45, colors.HexColor("#9ca3af")))
        box_col += 2
    tbl.setStyle(TableStyle(commands))
    return tbl


def page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawRightString(A4[0] - 0.45 * inch, 0.34 * inch, str(doc.page))
    canvas.restoreState()


def build_story(styles: dict[str, ParagraphStyle]) -> list:
    story: list = []

    story.append(p("Curved Ewald Factorization 전략 메모", styles, "title"))
    story.append(
        p(
            f"rotationally structured curved Ewald/cap Fourier evaluation을 prepared forward/adjoint operator로 정식화하기 위한 논문 전략. 작성일: {date(2026, 6, 22).isoformat()}",
            styles,
            "subtitle",
        )
    )
    story.append(pipeline(styles))
    story.append(Spacer(1, 0.10 * inch))
    story.append(
        p(
            "핵심 전략은 High-NA optics를 새 수학 발견의 증거로 쓰지 않고, 이미 알려진 wave-optics reduction의 선행 사례로 인정한 뒤, 같은 원리를 WAXS와 ODT의 curved Ewald operator로 확장했다는 점을 전면에 세우는 것이다.",
            styles,
            "callout",
        )
    )
    story.extend(
        bullets(
            [
                "Main novelty: rotationally structured curved Ewald sphere/cap 위의 Fourier evaluation을 reusable prepared forward/adjoint operator로 만든 것.",
                "Main evidence: WAXS는 correctness와 scaling validation, ODT는 repeated inverse/backpropagation impact.",
                "High-NA role: Boichenko류 circular-harmonic/Fourier-Bessel reduction이 이미 알려진 구조임을 보여주는 prior-art-connected bridge.",
                "Claim boundary: circular harmonic identity 자체가 novelty가 아니라, prepared curved-manifold operatorization과 WAXS/ODT geometry exploitation이 novelty.",
            ],
            styles,
        )
    )

    story.append(p("논문에서 사용할 중심 문장", styles, "h2"))
    story.append(
        p(
            "Circular-harmonic reductions of rotational wave integrals are classical and have appeared in high-NA focusing through vortex-mode and Fourier-Bessel formulations. Here we show that the same structural idea can be turned into prepared forward and adjoint operators for Fourier evaluation on curved Ewald spheres and caps. WAXS provides a direct correctness and scaling validation, while ODT exposes the main inverse-problem advantage through repeated backpropagation on detector-illumination Ewald-cap geometries.",
            styles,
            "equation",
        )
    )

    story.append(PageBreak())
    story.append(p("1. 공통 수학 구조", styles, "h1"))
    story.append(
        p(
            "모든 예시는 원래 서로 다른 물리 문제처럼 보이지만, 계산 관점에서는 source distribution의 Fourier transform을 detector, focal volume, 또는 Ewald cap 위에서 반복 평가하는 문제다. rotational geometry가 있으면 target-wise nonuniform point cloud로 보지 않고, 각도 좌표를 분리해 harmonic coefficient와 radial/axial kernel을 재사용할 수 있다.",
            styles,
        )
    )
    story.append(
        p(
            "F(q) = integral rho(r) exp(i q dot r) dr\n\n"
            "q dot r = q_perp R cos(phi_q - beta) + q_z z\n\n"
            "exp(i q_perp R cos(phi_q - beta))\n"
            "  = sum_h i^h J_h(q_perp R) exp(i h phi_q) exp(-i h beta)",
            styles,
            "equation",
        )
    )
    story.extend(
        bullets(
            [
                "source azimuth beta 방향은 Fourier coefficient로 압축된다.",
                "target azimuth phi_q 방향은 exp(i h phi_q) basis로 재구성된다.",
                "radial/axial dependence는 Bessel kernel J_h(q_perp R)와 axial phase exp(i q_z z)에 들어간다.",
                "동일 geometry에서 여러 mask, mode, residual, iteration을 반복하면 prepared operator가 amortize된다.",
            ],
            styles,
        )
    )

    story.append(p("도메인별 mapping", styles, "h2"))
    story.append(
        make_table(
            [
                ["Domain", "Known / new role", "Manifold", "Factorization handle", "Paper role"],
                [
                    "High-NA",
                    "Known precedent",
                    "focal volume (rho, psi, z)",
                    "pupil azimuth harmonic, Bessel radial kernel",
                    "intro bridge + secondary practical benchmark",
                ],
                [
                    "WAXS",
                    "Direct application",
                    "curved Ewald ring / cake map",
                    "cylindrical histogram, beta FFT, R-dependent harmonic cutoff",
                    "correctness and scaling anchor",
                ],
                [
                    "ODT",
                    "Extended application",
                    "detector cap with illumination shift",
                    "detector harmonic h plus illumination harmonic l; prepared adjoint",
                    "inverse/backpropagation impact anchor",
                ],
            ],
            styles,
            [0.8 * inch, 1.25 * inch, 1.45 * inch, 2.15 * inch, 1.35 * inch],
        )
    )

    story.append(PageBreak())
    story.append(p("2. Prior art 경계", styles, "h1"))
    story.append(
        p(
            "중요한 방어선은 '같은 수학 identity는 오래되었지만, curved Ewald/cap Fourier evaluation을 prepared forward/adjoint operator로 구성한 것이 contribution'이라는 점이다. 특히 Boichenko는 High-NA에서 강한 선행 사례로 인용해야 한다.",
            styles,
        )
    )
    story.append(
        make_table(
            [
                ["Prior art", "What it covers", "Overlap", "How we position our work"],
                [
                    "Boichenko",
                    "arbitrary entrance beam을 circularly polarized vortex vector beams로 분해하고 Richards-Wolf focal field를 rho,z와 phi로 factorize.",
                    "High-NA circular-harmonic reduction과 매우 가깝다.",
                    "반드시 인용. High-NA에서는 novelty가 아니라 known reduction recovery와 modern package/GPU benchmark로 제한.",
                ],
                [
                    "Fourier diffraction theorem / ODT",
                    "ODT/FPDT에서 Ewald sphere/cap Fourier coverage와 filtered backpropagation을 정식화.",
                    "Ewald/cap physics와 inverse problem baseline이 겹친다.",
                    "우리 novelty는 theorem 자체가 아니라 h/l geometry-aware prepared operator와 repeated adjoint acceleration.",
                ],
                [
                    "Cryo-EM Ewald curvature",
                    "curved Ewald sphere가 projection approximation을 깨는 regime과 reconstruction algorithm.",
                    "curved Ewald importance가 겹친다.",
                    "우리 방법은 cryo-EM reconstruction solver가 아니라 rotational Ewald/cap Fourier operatorization.",
                ],
                [
                    "Radio interferometry w-projection",
                    "wide-field correction에서 radial Hankel kernel과 gridding correction을 사용.",
                    "Hankel/radial prepared kernel 철학이 adjacent.",
                    "인접 사례로 인용 가능. Ewald scattering operator와 h/l ODT structure와는 다름.",
                ],
            ],
            styles,
            [1.05 * inch, 2.0 * inch, 1.65 * inch, 2.30 * inch],
        )
    )

    story.append(p("안전한 claim boundary", styles, "h2"))
    story.append(
        make_table(
            [
                ["Avoid", "Use instead", "Reason"],
                [
                    "new circular harmonic identity",
                    "classical harmonic identity turned into a prepared curved-Ewald operator",
                    "identity 자체는 Boichenko와 고전 Fourier-Bessel/Hankel theory에 걸린다.",
                ],
                [
                    "universal NUFFT replacement",
                    "domain-aware accelerator for structured curved/cylindrical manifolds",
                    "arbitrary nonuniform point cloud는 우리 native regime이 아니다.",
                ],
                [
                    "High-NA method is new",
                    "High-NA is a prior-art-connected bridge and benchmark stress test",
                    "High-NA factorization 자체는 이미 알려져 있다.",
                ],
                [
                    "ODT full reconstruction solver replacement",
                    "prepared adjoint/backpropagation acceleration for repeated ODT updates",
                    "현재 가장 강한 구조적 이득은 adjoint/update loop에서 나온다.",
                ],
            ],
            styles,
            [1.65 * inch, 2.35 * inch, 2.85 * inch],
            row_colors=[colors.white, SOFT_GRAY],
        )
    )

    story.append(PageBreak())
    story.append(p("3. Application별 논문 역할", styles, "h1"))
    story.append(p("WAXS: High-NA에서 알려진 원리를 curved Ewald sphere에 적용하는 직접 사례", styles, "h2"))
    story.extend(
        bullets(
            [
                "WAXS는 q = k_out - k_in이 Ewald sphere/ring을 이루기 때문에 curved Ewald Fourier evaluation이라는 framing이 가장 자연스럽다.",
                "source 쪽은 atoms를 cylindrical histogram H(R,z,beta)로 모으고, beta FFT를 통해 occupied angular harmonics를 얻는다.",
                "q range가 커지고 fixed-dq grid가 커질수록 target count와 angular bandwidth가 함께 증가하므로, geometry-aware contraction의 장점이 명확해진다.",
                "논문 내 역할은 '이 방법이 정확하고 scaling이 맞는가?'에 대한 anchor다.",
            ],
            styles,
        )
    )

    story.append(p("ODT: 한 발 더 나아간 h/l factorization", styles, "h2"))
    story.extend(
        bullets(
            [
                "ODT는 q = k_s - k_i 구조 때문에 detector cap과 illumination angle이 동시에 들어간다.",
                "따라서 WAXS처럼 detector angular harmonic만 쓰는 것을 넘어, detector harmonic h와 illumination harmonic l의 조합을 준비할 수 있다.",
                "특히 inverse/reconstruction loop에서는 residual을 object space로 backpropagate하는 adjoint가 반복되므로 prepared operator가 가장 잘 맞는다.",
                "논문 내 역할은 '이 구조가 repeated inverse problem에서 실제 계산 regime을 바꾸는가?'에 대한 impact anchor다.",
            ],
            styles,
        )
    )

    story.append(p("High-NA: known reduction을 recover하는 bridge", styles, "h2"))
    story.extend(
        bullets(
            [
                "High-NA Debye-Wolf/Richards-Wolf의 azimuthal harmonic reduction은 이미 optics prior art가 있다.",
                "따라서 High-NA를 main novelty로 쓰지 않고, introduction에서 '이런 harmonic reduction은 wave optics에서 이미 중요하다'는 연결고리로 사용한다.",
                "우리가 추가로 보일 수 있는 것은 sampled pupil에서 effective vectorial pupil spectrum을 분석하고, adaptive/GPU/package benchmark를 제공하는 practical implementation value다.",
                "논문 내 위치는 main text의 짧은 bridge panel 또는 SI의 자세한 benchmark가 적절하다.",
            ],
            styles,
        )
    )

    story.append(PageBreak())
    story.append(p("4. Manuscript 구조", styles, "h1"))
    story.append(
        make_table(
            [
                ["Figure / Section", "Content", "Purpose"],
                [
                    "Fig. 1 framework",
                    "rotationally structured curved manifold, harmonic separation, build/hot-loop split, forward/adjoint pair",
                    "method가 benchmark 모음이 아니라 하나의 operator view라는 점을 먼저 보여줌",
                ],
                [
                    "Fig. 2 WAXS",
                    "Ewald ring/cake-map geometry, reference agreement, q-range scaling, NUFFT/direct comparison",
                    "correctness and scaling validation",
                ],
                [
                    "Fig. 3 High-NA",
                    "Boichenko prior-art connection, Debye-Wolf/Richards-Wolf recovery, adaptive/vectorial/GPU practical demo",
                    "known wave-optics reduction과의 연결 및 cross-domain credibility",
                ],
                [
                    "Fig. 4 ODT",
                    "detector cap plus illumination geometry, h/l factorization, prepared adjoint, cuFINUFFT comparison",
                    "inverse/backpropagation impact",
                ],
                [
                    "Fig. 5 regime map",
                    "direct vs NUFFT vs structured operator as function of target geometry, repetition count, angular bandwidth",
                    "언제 유리하고 언제 불리한지 명확히 표시",
                ],
            ],
            styles,
            [1.25 * inch, 3.25 * inch, 2.35 * inch],
        )
    )

    story.append(p("Introduction paragraph draft", styles, "h2"))
    story.append(
        p(
            "Many wave-physics measurements are not arbitrary Fourier samples but structured evaluations on detector, focal, or Ewald manifolds. In rotational geometries, the angular part of the phase can be separated by circular harmonics, a classical reduction that has appeared in high-NA focusing and vortex-mode formulations. Here we use this observation not as a new identity, but as an operator-design principle: curved Ewald and Ewald-cap Fourier measurements can be prepared as reusable forward and adjoint operators by exposing their radial, axial, and angular structure.",
            styles,
            "equation",
        )
    )

    story.append(PageBreak())
    story.append(p("5. 앞으로 해야 할 일", styles, "h1"))
    story.append(
        make_table(
            [
                ["Priority", "Task", "Acceptance criterion", "Why it matters"],
                [
                    "P0",
                    "Prior-art matrix 확정",
                    "Boichenko, FDT/ODT, FPDT, cryo-EM Ewald, radio interferometry, Fourier-Bessel image analysis를 same/adjacent/not-same으로 표기",
                    "NCS급 논문에서 novelty boundary를 방어",
                ],
                [
                    "P0",
                    "WAXS validation package",
                    "realistic NPZ, Debye/direct/NUFFT agreement, fixed-dq q-range scaling, memory/build breakdown",
                    "main correctness anchor",
                ],
                [
                    "P0",
                    "ODT prepared adjoint benchmark 고정",
                    "cuFINUFFT fairness, build/hot/update separation, warm-start update error, realistic lab geometry",
                    "main impact anchor",
                ],
                [
                    "P1",
                    "High-NA secondary benchmark 정리",
                    "Boichenko comparison text, PyFocus/vectorial package/FINUFFT/cuFINUFFT comparison, adaptive pupil spectrum ablation",
                    "bridge를 설득력 있게 유지하되 novelty overclaim 방지",
                ],
                [
                    "P1",
                    "Regime map",
                    "Cartesian/non-Cartesian, one-shot/repeated, low/high angular bandwidth 조건별 direct/NUFFT/ours 비교",
                    "제한점과 강점을 한 그림에 표시",
                ],
                [
                    "P1",
                    "Reproducibility bundle",
                    "single-command figure regeneration, source-data table, seeds, hardware, dependency versions",
                    "computational journal submission readiness",
                ],
            ],
            styles,
            [0.7 * inch, 1.75 * inch, 2.75 * inch, 1.85 * inch],
        )
    )

    story.append(p("결론", styles, "h1"))
    story.append(
        p(
            "바뀐 전략의 핵심은 High-NA를 빼는 것이 아니라, High-NA의 역할을 정확히 낮추는 것이다. High-NA는 known circular-harmonic reduction의 선행 사례이자 practical bridge다. 메인 주장은 WAXS와 ODT에서 curved Ewald/cap Fourier evaluation을 prepared forward/adjoint operator로 만든 것이며, 특히 ODT에서는 detector/illumination h/l factorization을 통해 repeated inverse workload에서 이득을 보인다는 점이다.",
            styles,
            "callout",
        )
    )

    story.append(p("Sources", styles, "h1"))
    for source in SOURCES:
        story.append(p(f"{source['key']}: {source['role']}\n{source['url']}", styles, "small"))

    return story


def write_source_notes() -> None:
    lines = [
        "# Curved Ewald factorization strategy source notes",
        "",
        f"Generated PDF: `{OUT.relative_to(ROOT)}`",
        f"Generated on: {date(2026, 6, 22).isoformat()}",
        "",
        "## Main strategy",
        "",
        "- Main novelty: prepared forward/adjoint operatorization for rotationally structured curved Ewald/cap Fourier evaluation.",
        "- Main evidence: WAXS for correctness/scaling validation; ODT for repeated inverse/backpropagation impact.",
        "- High-NA role: prior-art-connected bridge, recovering known circular-harmonic/Fourier-Bessel reduction; adaptive/vectorial/GPU implementation as secondary practical demo.",
        "- Claim boundary: circular harmonic identity itself is not novel; WAXS/ODT geometry-aware operatorization is the novelty.",
        "",
        "## Web sources checked in this turn",
        "",
    ]
    for source in SOURCES:
        lines.append(f"- {source['key']}: {source['url']} -- {source['role']}")
    lines.extend(
        [
            "",
            "## Local context used",
            "",
            "- Existing benchmark directories and previous PDF/report scripts were inspected for output convention.",
            "- The new PDF is a strategy memo, not a fresh benchmark rerun.",
        ]
    )
    SOURCE_NOTES.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = make_styles()
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=0.48 * inch,
        rightMargin=0.48 * inch,
        topMargin=0.48 * inch,
        bottomMargin=0.56 * inch,
        title="Curved Ewald Factorization Strategy Memo",
        author="Codex",
    )
    doc.build(build_story(styles), onFirstPage=page_number, onLaterPages=page_number)
    write_source_notes()
    print(OUT)
    print(SOURCE_NOTES)


if __name__ == "__main__":
    main()
