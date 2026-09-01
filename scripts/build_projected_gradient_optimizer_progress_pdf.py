from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "projected_gradient_optimizer_progress_2026_06_27.pdf"

INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#4b5563")
GRID = colors.HexColor("#d1d5db")
HEADER = colors.HexColor("#1f2937")
SOFT = colors.HexColor("#f9fafb")
BLUE = colors.HexColor("#dbeafe")
GREEN = colors.HexColor("#dcfce7")
AMBER = colors.HexColor("#fef3c7")


RESULT_FILES = {
    "GPU-CZT cached, clean": ROOT
    / "benchmark_results"
    / "projected_gradient_optimizer_synthetic_shape4_roll_background_cuda_splatjac_gpu_czt_cached_q129.json",
    "CPU-CZT rerun, clean": ROOT
    / "benchmark_results"
    / "projected_gradient_optimizer_synthetic_shape4_roll_background_cuda_splatjac_cpu_czt_q129_rerun.json",
    "GPU-CZT cached, 0.5% noise": ROOT
    / "benchmark_results"
    / "projected_gradient_optimizer_synthetic_shape4_roll_background_noisy_cuda_splatjac_gpu_czt_cached_q129.json",
    "CPU-CZT rerun, 0.5% noise": ROOT
    / "benchmark_results"
    / "projected_gradient_optimizer_synthetic_shape4_roll_background_noisy_cuda_splatjac_cpu_czt_q129_rerun.json",
}


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
            fontSize=9.4,
            leading=13.2,
            textColor=MUTED,
            spaceAfter=8,
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
            textColor=HEADER,
            spaceBefore=6,
            spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "BodyKR",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=8.85,
            leading=12.4,
            textColor=INK,
            spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "SmallKR",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=7.2,
            leading=9.4,
            textColor=MUTED,
            spaceAfter=2.5,
        ),
        "equation": ParagraphStyle(
            "EquationKR",
            parent=sample["Code"],
            fontName=body_font,
            fontSize=8.3,
            leading=11.5,
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
            fontSize=6.9,
            leading=8.6,
            textColor=INK,
        ),
        "table_head": ParagraphStyle(
            "TableHeadKR",
            parent=sample["BodyText"],
            fontName=bold_font,
            fontSize=6.7,
            leading=8.5,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
    }


def clean(text: str) -> str:
    return escape(text).replace("\n", "<br/>")


def p(text: str, styles: dict[str, ParagraphStyle], style: str = "body") -> Paragraph:
    return Paragraph(clean(text), styles[style])


def bullets(items: list[str], styles: dict[str, ParagraphStyle]) -> list[Paragraph]:
    return [p("- " + item, styles) for item in items]


def table(rows: list[list[Any]], styles: dict[str, ParagraphStyle], col_widths: list[float]) -> Table:
    wrapped: list[list[Any]] = []
    for i, row in enumerate(rows):
        style = styles["table_head"] if i == 0 else styles["table"]
        wrapped.append([cell if hasattr(cell, "wrap") else Paragraph(clean(str(cell)), style) for cell in row])
    result = Table(wrapped, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HEADER),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SOFT]),
            ]
        )
    )
    return result


def callout(text: str, styles: dict[str, ParagraphStyle], color: colors.Color = BLUE) -> Table:
    cell = p(text, styles, "body")
    result = Table([[cell]], colWidths=[7.05 * inch], hAlign="LEFT")
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), color),
                ("BOX", (0, 0), (-1, -1), 0.45, colors.HexColor("#93c5fd")),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return result


def load_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def benchmark_rows() -> list[list[str]]:
    rows = [["Run", "Time (s)", "nfev", "RMS log residual", "Recovered parameters"]]
    for label, path in RESULT_FILES.items():
        data = load_result(path)
        if data is None:
            rows.append([label, "missing", "-", "-", str(path.relative_to(ROOT))])
            continue
        fit = data["fit"]
        rec = fit["recovered"]
        rows.append(
            [
                label,
                f"{float(fit['optimizer_s']):.3f}",
                str(int(fit["nfev"])),
                f"{float(fit['final_rms_log_residual']):.3e}",
                (
                    f"axis=({rec['axis_u_a']:.3f}, {rec['axis_v_a']:.3f}, {rec['axis_w_a_fixed']:.1f}) A; "
                    f"shape4={rec['shape4']:.6f}; roll={rec['roll_deg']:.4f} deg; "
                    f"bg={rec['background_frac']:.6f}"
                ),
            ]
        )
    return rows


def footer(canvas, doc) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawRightString(A4[0] - 0.58 * inch, 0.42 * inch, f"page {doc.page}")
    canvas.restoreState()


def build_story(styles: dict[str, ParagraphStyle]) -> list[Any]:
    story: list[Any] = []
    story.append(p("Projected-Gradient Mesh SAXS Optimizer Progress Memo", styles, "title"))
    story.append(
        p(
            f"작성일: {date.today().isoformat()} | 범위: surface-normal / projected-gradient mesh SAXS, "
            "RawKernel gridding, CZT transfer, synthetic shape optimization",
            styles,
            "subtitle",
        )
    )
    story.append(
        callout(
            "현재 결론: projected-gradient 방법은 signed raster와 같은 uniform-grid FT 구조를 타면서도 "
            "surface normal / area gradient 정보를 보존한다. 따라서 단순 forward speed만으로는 signed raster를 "
            "압도하기 어렵지만, differentiable shape optimization 쪽에서 더 강한 장점이 보인다.",
            styles,
            GREEN,
        )
    )

    story.append(p("1. 문제 설정", styles, "h1"))
    story.extend(
        bullets(
            [
                "초기 목표는 surface-mesh-projection method를 sharp-interface SAXS로 확장할 수 있는지 확인하는 것이었다.",
                "Exact triangle integral을 빠른 주경로로 주장하지 않고, sub-face / median-line quadrature로 만든 vector-valued surface source를 기본 근사로 삼았다.",
                "GPU 레벨에서 먼저 signed raster baseline과 비교하고, 이후 CPU 가능성을 판단하는 순서로 진행했다.",
                "실험 q grid는 기존 PONI 파일 post_beamtime_tmp_saxs.poni에서 만든 detector crop을 사용했다.",
            ],
            styles,
        )
    )

    story.append(p("2. 핵심 수식과 구현 방향", styles, "h1"))
    story.append(
        p(
            "Projected-gradient formulation은 표면 normal source를 투영면의 두 성분 G_u, G_v로 누적한 뒤, "
            "2D Fourier transform과 q dot contraction으로 amplitude를 복원한다.",
            styles,
        )
    )
    story.append(
        p(
            "A(q_u,q_v) = (q_u F_u(q_u,q_v) + q_v F_v(q_u,q_v)) / (-i (q_u^2 + q_v^2))",
            styles,
            "equation",
        )
    )
    story.extend(
        bullets(
            [
                "q=0 또는 low-q에서는 1/|q|^2 식을 직접 쓰지 않고 mesh volume/moment expansion으로 처리한다.",
                "Forward path는 triangle sub-face sample을 RawKernel로 CIC splat하여 projected gradient grid를 만든 뒤 CZT로 detector q grid에 보낸다.",
                "Optimization path는 axis_u, axis_v, shape4에 대해 projected-gradient grid의 derivative splat을 계산하고, intensity Jacobian으로 연결한다.",
                "Roll derivative는 q-grid gradient dA/droll = -q_v dA/dq_u + q_u dA/dq_v 근사를 사용해 추가 RawKernel forward 호출을 줄였다.",
            ],
            styles,
        )
    )

    story.append(p("3. 현재 구현된 파일과 기능", styles, "h1"))
    story.append(
        table(
            [
                ["File", "Implemented role"],
                [
                    "scripts/benchmark_projected_gradient_rawkernel.py",
                    "Projected-gradient RawKernel gridding, packed/separate CZT amplitude, signed raster comparison harness.",
                ],
                [
                    "scripts/optimize_projected_gradient_shape.py",
                    "Synthetic/NPZ target optimizer, PONI q-grid, ellipsoid + shape4 + roll + background fit, hybrid Jacobian.",
                ],
                [
                    "scripts/optimize_projected_gradient_shape.py",
                    "CPU derivative splat, CUDA derivative splat, optional GPU derivative-CZT, CuPy CZT plan cache.",
                ],
                [
                    "src/waxs_cake/surface_normal.py",
                    "Surface-normal vector-source circular path, arbitrary-target reference path, low-q moment expansion.",
                ],
                [
                    "tests/test_surface_normal.py",
                    "Sphere/direct/circular/Ewald-ring/flat-plane consistency checks for surface-normal paths.",
                ],
            ],
            styles,
            [2.15 * inch, 4.9 * inch],
        )
    )

    story.append(p("4. Optimizer 실험 조건", styles, "h1"))
    story.append(
        table(
            [
                ["Item", "Value"],
                ["True shape", "axis=(260, 190, 160) A, shape4=0.06, roll=17 deg, background=0.02 of clean max"],
                ["Initial shape", "axis=(240, 210, 160) A, shape4=0, roll=5 deg, scale=0.85, background=0.01"],
                ["Detector/q grid", "PONI-derived 129 x 129 crop, q_min fit = 2.5e-4 A^-1, fit stride = 2"],
                ["Projection grid", "256 x 256 projected-gradient image, square uv bounds, quad_order=2"],
                ["Mesh", "n_lat=20, n_lon=40 ellipsoid mesh; axis_w fixed for single qz=0 projection"],
                ["Optimizer", "scipy least_squares, hybrid Jacobian, forward FD fallback, grid-gradient roll derivative"],
            ],
            styles,
            [1.65 * inch, 5.4 * inch],
        )
    )

    story.append(p("5. 검증과 속도", styles, "h1"))
    story.append(
        p(
            "최종 GPU derivative-CZT 경로는 CPU derivative-CZT와 amplitude를 직접 비교했다. "
            "q65 small check에서 fit mask 기준 relative L2는 axis_u=6.73e-08, axis_v=1.77e-07, "
            "shape4=1.45e-05 수준이었다.",
            styles,
        )
    )
    story.append(table(benchmark_rows(), styles, [1.65 * inch, 0.65 * inch, 0.42 * inch, 0.85 * inch, 3.48 * inch]))
    story.append(
        p(
            "해석: GPU-CZT는 plan cache 없이 첫 시도에서 clean case 4.97 s로 느렸지만, "
            "CuPy CZT 객체를 캐시한 뒤 clean 2.41 s, noisy 2.68 s까지 줄었다. "
            "현재 q129에서는 CPU-CZT 대비 이득이 작지만 반복 optimizer에는 GPU-resident derivative path가 실질적으로 유효하다.",
            styles,
        )
    )

    story.append(p("6. Signed raster 대비 평가", styles, "h1"))
    story.extend(
        bullets(
            [
                "Signed raster가 매우 빠른 이유는 결국 uniform grid projection 뒤 FFT/CZT로 문제를 formulation하기 때문이다.",
                "Projected-gradient도 같은 uniform-grid transfer를 쓰므로, 순수 forward speed만으로는 signed raster보다 크게 빨라지기 어렵다.",
                "대신 surface normal과 면적 gradient가 살아 있으므로 sub-face quadrature 정밀도, q=0 moment handling, shape derivative/optimization에서 장점이 있다.",
                "따라서 논문 claim은 'projection-free sharp-interface mesh SAXS with GPU gridding/CZT and differentiable optimization' 정도가 안전하다.",
            ],
            styles,
        )
    )

    story.append(p("7. 현재 한계", styles, "h1"))
    story.extend(
        bullets(
            [
                "현재 optimizer는 synthetic target에서 검증되었고, real NPZ target fitting은 아직 본격적으로 돌리지 않았다.",
                "단일 qz=0 projection에서는 axis_w가 contrast/background와 강하게 얽혀 있어 현재는 fixed parameter로 두었다.",
                "PONI beam-center crop이 완전 대칭 q grid가 아니어서 packed gradient CZT 최적화는 그대로 적용할 수 없다.",
                "Exact triangle solver는 reference/fallback 위치이고, fast path는 sub-face quadrature 기반 근사다.",
                "Signed raster와의 end-to-end baseline은 raw kernel 복구/동일 q grid 조건에서 다시 맞춰야 한다.",
            ],
            styles,
        )
    )

    story.append(p("8. 다음 단계", styles, "h1"))
    story.append(
        callout(
            "우선순위: (1) signed raster RawKernel baseline 복구 또는 재작성, "
            "(2) projected-gradient optimizer와 동일 PONI/q crop에서 matched benchmark, "
            "(3) real NPZ frame에 대한 constrained fitting, "
            "(4) mesh parameterization/regularization을 포함한 gradient-based shape optimization 확장.",
            styles,
            AMBER,
        )
    )
    story.extend(
        bullets(
            [
                "q crop 크기, projection image size, mesh resolution, quad_order를 sweep해서 forward/optimizer break-even을 정리한다.",
                "GPU-CZT는 현재 optional path로 유지하고, 큰 q grid 또는 많은 derivative parameter에서 더 큰 이득이 나는지 확인한다.",
                "Shape4 하나에서 더 풍부한 spherical basis 또는 vertex-control basis로 확장할 때 regularization과 positivity constraint를 같이 넣는다.",
                "결과가 유지되면 후속 논문 framing은 speed-only가 아니라 differentiable surface-gradient mesh SAXS inverse problem으로 잡는 편이 더 강하다.",
            ],
            styles,
        )
    )

    story.append(Spacer(1, 0.1 * inch))
    story.append(
        p(
            "Generated artifacts: benchmark JSON files under benchmark_results/ and this PDF under docs/. "
            "Validation commands used: py_compile, pytest tests/test_surface_normal.py -q, CPU/GPU derivative-amplitude comparison.",
            styles,
            "small",
        )
    )
    return story


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = make_styles()
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=0.58 * inch,
        leftMargin=0.58 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.6 * inch,
        title="Projected-Gradient Mesh SAXS Optimizer Progress Memo",
        author="Codex",
    )
    doc.build(build_story(styles), onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    main()
