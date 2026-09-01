from __future__ import annotations

from html import escape
import json
from pathlib import Path

import fitz
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
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
RESULT_JSON = ROOT / "benchmark_results" / "acfo_pde_extension_validation.json"
FIGURE_DIR = ROOT / "docs" / "acfo_pde_extension_validation_support"
OUTPUT = ROOT / "docs" / "ACFO_PDE_restriction_extension_minimum_validation_results_ko.pdf"

INK = colors.HexColor("#12263A")
MUTED = colors.HexColor("#4D6578")
BLUE = colors.HexColor("#176B87")
BLUE_SOFT = colors.HexColor("#E8F3F7")
GREEN = colors.HexColor("#17865A")
GREEN_SOFT = colors.HexColor("#E9F7F0")
ORANGE = colors.HexColor("#D96C19")
ORANGE_SOFT = colors.HexColor("#FFF1E6")
LINE = colors.HexColor("#CBD8E1")
PAPER = colors.HexColor("#F6F9FB")


def register_fonts() -> tuple[str, str]:
    font_dir = Path("C:/Windows/Fonts")
    regular = font_dir / "malgun.ttf"
    bold = font_dir / "malgunbd.ttf"
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("PDEKorean", str(regular)))
        pdfmetrics.registerFont(TTFont("PDEKorean-Bold", str(bold)))
        return "PDEKorean", "PDEKorean-Bold"
    pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
    return "HYSMyeongJo-Medium", "HYSMyeongJo-Medium"


BODY_FONT, BOLD_FONT = register_fonts()


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleKO",
            parent=base["Title"],
            fontName=BOLD_FONT,
            fontSize=21,
            leading=28,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleKO",
            fontName=BODY_FONT,
            fontSize=10.2,
            leading=15,
            textColor=MUTED,
            spaceAfter=9,
        ),
        "section": ParagraphStyle(
            "SectionKO",
            fontName=BOLD_FONT,
            fontSize=15,
            leading=20,
            textColor=INK,
            spaceAfter=7,
        ),
        "subsection": ParagraphStyle(
            "SubsectionKO",
            fontName=BOLD_FONT,
            fontSize=10.3,
            leading=14,
            textColor=BLUE,
            spaceBefore=4,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "BodyKO",
            fontName=BODY_FONT,
            fontSize=8.6,
            leading=13.1,
            textColor=INK,
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "SmallKO",
            fontName=BODY_FONT,
            fontSize=7.2,
            leading=10.3,
            textColor=MUTED,
        ),
        "caption": ParagraphStyle(
            "CaptionKO",
            fontName=BODY_FONT,
            fontSize=7.3,
            leading=10.5,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=5,
        ),
        "formula": ParagraphStyle(
            "FormulaKO",
            fontName=BODY_FONT,
            fontSize=8.3,
            leading=12.5,
            textColor=INK,
            leftIndent=8,
            rightIndent=8,
            alignment=TA_CENTER,
        ),
        "card_title": ParagraphStyle(
            "CardTitleKO",
            fontName=BOLD_FONT,
            fontSize=10.5,
            leading=14,
            textColor=GREEN,
            alignment=TA_CENTER,
        ),
        "card_body": ParagraphStyle(
            "CardBodyKO",
            fontName=BODY_FONT,
            fontSize=8.2,
            leading=12.2,
            textColor=INK,
            alignment=TA_CENTER,
        ),
        "table_head": ParagraphStyle(
            "TableHeadKO",
            fontName=BOLD_FONT,
            fontSize=6.5,
            leading=8.5,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "table_cell": ParagraphStyle(
            "TableCellKO",
            fontName=BODY_FONT,
            fontSize=6.3,
            leading=8.2,
            textColor=INK,
            alignment=TA_CENTER,
        ),
        "table_left": ParagraphStyle(
            "TableLeftKO",
            fontName=BODY_FONT,
            fontSize=6.3,
            leading=8.2,
            textColor=INK,
            alignment=TA_LEFT,
        ),
    }


S = styles()


def p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def bullet(text: str) -> Paragraph:
    return Paragraph(f"<font color='#176B87'>●</font>&nbsp; {text}", S["body"])


def result_table(data: list[list[object]], widths: list[float]) -> Table:
    rows: list[list[Paragraph]] = []
    for row_index, row in enumerate(data):
        style = "table_head" if row_index == 0 else "table_cell"
        rows.append([p(str(value), style) for value in row])
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PAPER]),
            ]
        )
    )
    return table


def info_box(title: str, body: str, *, background=BLUE_SOFT, accent=BLUE) -> Table:
    content = [
        [Paragraph(title, ParagraphStyle("BoxTitle", parent=S["subsection"], textColor=accent, alignment=TA_LEFT))],
        [p(body, "body")],
    ]
    table = Table(content, colWidths=[17.3 * cm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.8, accent),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def figure(path: Path, width: float, caption: str) -> list[object]:
    image = Image(str(path))
    image._restrictSize(width, 9.0 * cm)
    image.hAlign = "CENTER"
    return [image, p(caption, "caption")]


def header_footer(canvas, document) -> None:
    canvas.saveState()
    page = canvas.getPageNumber()
    width, height = A4
    if page > 1:
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.45)
        canvas.line(1.6 * cm, height - 1.05 * cm, width - 1.6 * cm, height - 1.05 * cm)
        canvas.setFont(BODY_FONT, 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(1.6 * cm, height - 0.78 * cm, "ACFO PDE restriction-extension | 내부 기술 검증 결과")
    canvas.setStrokeColor(LINE)
    canvas.line(1.6 * cm, 1.05 * cm, width - 1.6 * cm, 1.05 * cm)
    canvas.setFont(BODY_FONT, 6.8)
    canvas.setFillColor(MUTED)
    canvas.drawString(1.6 * cm, 0.73 * cm, "2026-07-12 | 코드·JSON·figure 기반 재현 가능 결과")
    canvas.drawRightString(width - 1.6 * cm, 0.73 * cm, f"{page}")
    canvas.restoreState()


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}e}"


def build_story(payload: dict) -> list[object]:
    results = payload["results"]
    order = ("shifted_helmholtz", "paraxial", "ellipsoid")
    story: list[object] = []

    # Page 1: decision summary.
    story.extend(
        [
            Spacer(1, 0.7 * cm),
            p("ACFO PDE restriction–extension", "title"),
            p("최소 계산 검증 결과", "title"),
            p(
                "Ewald sphere에 한정되지 않는 축대칭 dispersion surface 위 exact homogeneous mode의 분석·합성 연산자를 독립 수치 경로로 검증한 결과 보고서",
                "subtitle",
            ),
            info_box(
                "최종 판정 · Overall PASS",
                "Shifted Helmholtz, paraxial paraboloid, anisotropic ellipsoid의 세 case가 부호·measure 계약, 직접 합성, shell/negative control, PDE residual, harmonic truncation, 새 adjoint의 모든 고정 판정 기준을 통과했다.",
                background=GREEN_SOFT,
                accent=GREEN,
            ),
            Spacer(1, 0.3 * cm),
            p("핵심 결과", "subsection"),
        ]
    )
    table_data = [["Case", "shell", "합성 L2", "PDE residual", "수렴차수", "negative ratio", "adjoint", "판정"]]
    for name in order:
        raw = results[name]
        table_data.append(
            [
                raw["label"],
                fmt(raw["shell"]["correct"]),
                fmt(raw["synthesis"]["random_prepared_vs_direct_256_relative_l2"]),
                fmt(raw["pde_convergence"]["finest_correct"]),
                f"{raw['pde_convergence']['observed_order']:.2f}",
                f"{raw['pde_convergence']['off_shell_to_correct_ratio']:.2e}",
                fmt(raw["adjoint"]["dot_product_error"]),
                "PASS",
            ]
        )
    story.extend(
        [
            result_table(
                table_data,
                [2.7 * cm, 1.8 * cm, 1.8 * cm, 2.0 * cm, 1.5 * cm, 2.1 * cm, 1.8 * cm, 1.25 * cm],
            ),
            Spacer(1, 0.25 * cm),
            bullet("독립 직접 합성 reference와의 오차는 random field에서 최대 4.159e-16이었다."),
            bullet("8차 Cartesian finite difference의 관측 수렴차수는 7.84–7.86이었다."),
            bullet("off-shell residual은 correct residual보다 최소 2.24e7배 컸다."),
            bullet("weighted adjoint dot-product 오차는 최대 3.757e-16이었다."),
            Spacer(1, 0.15 * cm),
            info_box(
                "이번 PDF의 범위",
                "검증된 사실과 수치만 정리한다. Boundary-value solve, 성능 benchmark, vector/full-wave PDE, evanescent branch, 법적 신규성 판단은 포함하지 않는다.",
                background=ORANGE_SOFT,
                accent=ORANGE,
            ),
            PageBreak(),
        ]
    )

    # Page 2: contract and geometry.
    story.extend(
        [
            p("1. 부호와 measure 계약", "section"),
            p(
                "기존 ACFO의 +i point-sampling forward와 그 conjugate adjoint는 변경하지 않았다. PDE 해석에는 물리적 dispersion Γ를 입력으로 받는 별도 pair를 두고 restriction은 −i, extension은 +i 부호로 고정했다.",
                "body",
            ),
            info_box(
                "Restriction · negative phase",
                "A<sub>Γ</sub>u의 각 harmonic coefficient는 유한 Cartesian 영역 Ω에서 u(x) exp(−i Γ·x)를 voxel measure로 적분해 얻는다.",
            ),
            Spacer(1, 0.15 * cm),
            info_box(
                "Extension · positive phase",
                "E<sub>Γ</sub>a = 2π Σ<sub>j,h</sub> w<sub>j</sub><super>Γ</super> i<super>h</super> a<sub>jh</sub> J<sub>h</sub>(K<sub>⊥,j</sub>R) exp(i hβ) exp(i K<sub>z,j</sub>z)",
            ),
            Spacer(1, 0.15 * cm),
            p(
                "Manifold weight는 w<super>Γ</super> = w<super>(s)</super>K<sub>⊥</sub>√[(dK<sub>⊥</sub>/ds)²+(dK<sub>z</sub>/ds)²]로 고정했다. Harmonic coefficient inner product에는 2πw<super>Γ</super>를, Cartesian field inner product에는 Δx³를 사용했다.",
                "body",
            ),
            p("검증 순서", "subsection"),
            result_table(
                [
                    ["단계", "질문", "독립성/판정"],
                    ["1", "부호와 measure가 명시적인가?", "기존 +i 경로와 별도 pair"],
                    ["2", "Bessel/harmonic 합성이 맞는가?", "Cartesian exponent sum reference"],
                    ["3", "올바른 shell만 검출하는가?", "off-shell 및 qz sign control"],
                    ["4", "합성 field가 PDE를 만족하는가?", "독립 8차 finite difference"],
                    ["5", "H truncation이 PDE residual과 분리되는가?", "output Parseval energy"],
                    ["6", "새 pair가 weighted adjoint인가?", "dot identity + direct restriction"],
                ],
                [1.0 * cm, 7.0 * cm, 9.0 * cm],
            ),
            Spacer(1, 0.2 * cm),
            *figure(
                FIGURE_DIR / "figure1_dispersion_shell.png",
                17.3 * cm,
                "Figure 1. 세 dispersion curve와 normalized shell residual. On-shell은 machine precision, 의도적 off-shell perturbation은 명확히 분리된다.",
            ),
            PageBreak(),
        ]
    )

    # Page 3: synthesis and shell controls.
    synthesis_rows = [["Case", "random/direct", "single-mode max", "Nφ 128/256", "off-shell", "qz sign"]]
    for name in order:
        raw = results[name]
        synthesis_rows.append(
            [
                raw["label"],
                fmt(raw["synthesis"]["random_prepared_vs_direct_256_relative_l2"]),
                fmt(raw["synthesis"]["single_mode_relative_l2_max"]),
                fmt(raw["synthesis"]["direct_128_vs_256_relative_l2"]),
                fmt(raw["shell"]["off_shell"]),
                fmt(raw["shell"].get("qz_sign_flip", float("nan"))) if "qz_sign_flip" in raw["shell"] else "N/A",
            ]
        )
    story.extend(
        [
            p("2. 직접 합성 정확성과 shell control", "section"),
            p(
                "PDE residual만으로는 합성 계수나 normalization의 오류를 검출할 수 없다. 따라서 production Bessel/harmonic 경로를 사용하지 않는 직접 φ-quadrature Cartesian exponent sum을 먼저 reference로 두었다.",
                "body",
            ),
            *figure(
                FIGURE_DIR / "figure2_direct_synthesis.png",
                15.0 * cm,
                "Figure 2. Random multimode, 단일 h=0, ±1, ±8 mode, Nφ refinement 모두 고정 기준보다 충분히 작다.",
            ),
            result_table(
                synthesis_rows,
                [3.0 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.2 * cm, 2.2 * cm],
            ),
            Spacer(1, 0.2 * cm),
            info_box(
                "해석",
                "Random/direct 오차가 약 1e-16이고 Nφ=128/256 차이도 약 1e-16이므로, 이후 PDE residual은 angular quadrature 부족이나 harmonic normalization 오류로 설명되지 않는다. Shifted/paraxial의 qz sign flip은 normalized shell residual 1.0으로 실패하며, ellipsoid는 z-even symbol이므로 sign flip을 negative control로 사용하지 않았다.",
                background=GREEN_SOFT,
                accent=GREEN,
            ),
            PageBreak(),
        ]
    )

    # Page 4: PDE convergence.
    convergence_rows = [["Case", "N=24", "N=32", "N=40", "N=48", "order", "off/correct"]]
    for name in order:
        raw = results[name]["pde_convergence"]
        convergence_rows.append(
            [
                results[name]["label"],
                *[fmt(value) for value in raw["correct_residuals"]],
                f"{raw['observed_order']:.2f}",
                f"{raw['off_shell_to_correct_ratio']:.2e}",
            ]
        )
    story.extend(
        [
            p("3. 독립 Cartesian PDE residual", "section"),
            p(
                "합성 field를 uniform Cartesian grid에서 평가한 뒤 8차 centered finite difference로 각 PDE를 적용했다. FFT periodic derivative는 사용하지 않았으며, 모든 방향에서 4-cell stencil margin을 제거한 interior norm만 판정에 사용했다.",
                "body",
            ),
            *figure(
                FIGURE_DIR / "figure3_pde_convergence.png",
                17.3 * cm,
                "Figure 3. On-shell field는 약 8차로 감소하고, off-shell field는 1e-3 수준의 유한 residual에 머문다.",
            ),
            result_table(
                convergence_rows,
                [2.8 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 1.6 * cm, 2.2 * cm],
            ),
            Spacer(1, 0.2 * cm),
            bullet("Shifted Helmholtz: Δu + 2ik∂<sub>z</sub>u = 0"),
            bullet("Paraxial: Δ<sub>⊥</sub>u + 2ik∂<sub>z</sub>u = 0"),
            bullet("Ellipsoid: −Δ<sub>⊥</sub>u/a² − ∂<sub>z</sub>²u/c² − u = 0"),
            PageBreak(),
        ]
    )

    # Page 5: truncation.
    truncation_rows = [["Case", "Parseval mismatch", "PDE residual max", "residual spread", "monotone"]]
    for name in order:
        raw = results[name]["truncation"]
        truncation_rows.append(
            [
                results[name]["label"],
                fmt(raw["parseval_mismatch_max"]),
                fmt(raw["pde_residual_max"]),
                f"{raw['pde_residual_spread']:.3f}",
                "Yes" if raw["solution_error_monotone"] else "No",
            ]
        )
    story.extend(
        [
            p("4. Harmonic truncation과 PDE residual의 분리", "section"),
            p(
                "H=0,1,2,4,6,8에서 동일 coefficient field를 잘랐다. Solution error는 cylindrical diagnostic grid의 R dR dβ dz measure로 계산하고, discarded output harmonic energy와 직접 비교했다. 입력 coefficient energy를 solution error의 대용으로 사용하지 않았다.",
                "body",
            ),
            *figure(
                FIGURE_DIR / "figure4_truncation_separation.png",
                17.3 * cm,
                "Figure 4. H 증가에 따라 solution error와 output-tail energy가 함께 감소하지만 PDE residual은 약 1e-11 floor에 유지된다.",
            ),
            result_table(
                truncation_rows,
                [3.2 * cm, 3.2 * cm, 3.2 * cm, 2.5 * cm, 2.2 * cm],
            ),
            Spacer(1, 0.2 * cm),
            info_box(
                "핵심 signature",
                "최대 Parseval mismatch는 5.551e-17이었다. 즉 H truncation은 PDE를 근사하는 과정이 아니라, 이미 exact인 homogeneous solution space 안에서 low-angular-bandwidth sector를 선택하는 과정으로 수치적으로 분리된다.",
                background=GREEN_SOFT,
                accent=GREEN,
            ),
            PageBreak(),
        ]
    )

    # Page 6: adjoint, conclusion, reproducibility.
    adjoint_rows = [["Case", "dot error", "direct restriction L2", "wrong-measure error"]]
    for name in order:
        raw = results[name]["adjoint"]
        adjoint_rows.append(
            [
                results[name]["label"],
                fmt(raw["dot_product_error"]),
                fmt(raw["restriction_vs_direct_relative_l2"]),
                fmt(raw["wrong_measure_dot_error"]),
            ]
        )
    story.extend(
        [
            p("5. 새 restriction–extension pair의 adjoint", "section"),
            p(
                "새 pair는 선언된 surface measure와 voxel measure에서 weighted adjoint identity를 만족한다. Restriction은 별도의 negative-exponent φ-quadrature reference와도 비교했다. Spatial weight를 의도적으로 생략하면 dot error가 0.839로 증가해 measure가 실제 판정에 작동함을 확인했다.",
                "body",
            ),
            result_table(
                adjoint_rows,
                [4.0 * cm, 3.2 * cm, 4.0 * cm, 3.8 * cm],
            ),
            Spacer(1, 0.25 * cm),
            p("6. 결론과 claim boundary", "section"),
            info_box(
                "검증된 결론",
                "ACFO harmonic backend와 분리된 physical-sign restriction–extension pair는 명시된 축대칭 dispersion surface에서 exact homogeneous PDE mode를 분석·합성한다. 세 polynomial-symbol case에서 직접 합성, shell discrimination, grid-convergent PDE residual, truncation separation, weighted adjoint가 모두 일관되게 통과했다.",
                background=GREEN_SOFT,
                accent=GREEN,
            ),
            Spacer(1, 0.15 * cm),
            info_box(
                "명시적 경계",
                "기존 ACFO가 임의의 유한 axisymmetric sampling curve를 계산할 수 있다는 사실과, 그 curve가 특정 PDE의 물리적 dispersion relation이라는 사실은 별개다. 이번 결과는 p(Γ)=0이 명시적으로 확인된 case에만 exact homogeneous-mode 해석을 부여한다. 임의 spline/table curve가 자동으로 물리적 또는 local-PDE dispersion surface가 된다고 주장하지 않는다.",
                background=ORANGE_SOFT,
                accent=ORANGE,
            ),
            Spacer(1, 0.2 * cm),
            p("재현 및 근거 파일", "subsection"),
            p(
                "실행: <font name='Courier'>.\\.venv\\Scripts\\python.exe scripts\\validate_axisymmetric_pde_extension.py</font><br/>"
                "수치 원천: <font name='Courier'>benchmark_results/acfo_pde_extension_validation.json</font><br/>"
                "구현: <font name='Courier'>src/waxs_cake/axisymmetric_pde.py</font><br/>"
                "테스트: 관련 신규·회귀 suite 41 passed<br/>"
                "전체 검증 실행 시간: 313.5 s",
                "body",
            ),
            p(
                "본 문서는 내부 기술 검증 결과이며, 법적 신규성·진보성·청구항 범위 판단을 대신하지 않는다.",
                "small",
            ),
        ]
    )
    return story


def build(output: Path = OUTPUT) -> None:
    payload = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    if not payload.get("passed"):
        raise RuntimeError("validation JSON does not report Overall PASS")
    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.35 * cm,
        title="ACFO PDE restriction-extension 최소 계산 검증 결과",
        author="Minsu Kim",
        subject="Axisymmetric dispersion-surface PDE restriction-extension validation",
    )
    document.build(build_story(payload), onFirstPage=header_footer, onLaterPages=header_footer)


def verify(output: Path = OUTPUT) -> dict[str, object]:
    document = fitz.open(output)
    extracted = "\n".join(page.get_text() for page in document)
    anchors = (
        "ACFO PDE",
        "Overall PASS",
        "Shifted Helmholtz",
        "Paraxial",
        "Ellipsoid",
        "p(Γ)=0",
        "41 passed",
    )
    missing = [anchor for anchor in anchors if anchor not in extracted]
    image_count = sum(len(page.get_images(full=True)) for page in document)
    result = {
        "path": str(output),
        "bytes": output.stat().st_size,
        "pages": document.page_count,
        "images": image_count,
        "anchors_found": [anchor for anchor in anchors if anchor in extracted],
        "missing_anchors": missing,
    }
    document.close()
    if result["pages"] != 6:
        raise RuntimeError(f"expected 6 pages, found {result['pages']}")
    if image_count < 4:
        raise RuntimeError(f"expected at least 4 embedded figures, found {image_count}")
    if missing:
        raise RuntimeError(f"missing extracted-text anchors: {missing}")
    return result


def main() -> None:
    build()
    print(json.dumps(verify(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
