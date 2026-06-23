from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Drawing, Line, Rect, String
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
OUT = ROOT / "benchmark_results" / "odt_cone_axis_optimization_summary.pdf"


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
            "KTitle",
            parent=sample["Title"],
            fontName=bold_font,
            fontSize=21,
            leading=27,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#111827"),
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "KSubtitle",
            parent=sample["Normal"],
            fontName=body_font,
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#4b5563"),
            spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "KH1",
            parent=sample["Heading1"],
            fontName=bold_font,
            fontSize=15,
            leading=20,
            textColor=colors.HexColor("#111827"),
            spaceBefore=10,
            spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "KH2",
            parent=sample["Heading2"],
            fontName=bold_font,
            fontSize=11.5,
            leading=15,
            textColor=colors.HexColor("#1f2937"),
            spaceBefore=8,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "KBody",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=9.2,
            leading=13.2,
            textColor=colors.HexColor("#111827"),
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "KSmall",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#4b5563"),
            spaceAfter=3,
        ),
        "table": ParagraphStyle(
            "KTable",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=7.5,
            leading=9.2,
            textColor=colors.HexColor("#111827"),
        ),
        "table_head": ParagraphStyle(
            "KTableHead",
            parent=sample["BodyText"],
            fontName=bold_font,
            fontSize=7.3,
            leading=9,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "caption": ParagraphStyle(
            "KCaption",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=7.5,
            leading=10,
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
) -> Table:
    wrapped: list[list[object]] = []
    for row_index, row in enumerate(rows):
        style_name = "table_head" if row_index == 0 else "table"
        wrapped.append(
            [
                cell
                if hasattr(cell, "wrap")
                else Paragraph(escape(str(cell)), styles[style_name])
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
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
            ]
        )
    )
    return result


def speedup_bar_chart() -> Drawing:
    rows = [
        ("r16/p64", 2.62),
        ("r32/p128", 4.10),
        ("r64/p256", 4.97),
        ("r128/p512", 5.82),
    ]
    width = 6.7 * inch
    height = 2.15 * inch
    left = 0.65 * inch
    bottom = 0.35 * inch
    chart_w = width - 1.1 * inch
    chart_h = height - 0.75 * inch
    max_v = 6.2
    drawing = Drawing(width, height)
    drawing.add(Line(left, bottom, left + chart_w, bottom, strokeColor=colors.HexColor("#374151"), strokeWidth=0.7))
    drawing.add(Line(left, bottom, left, bottom + chart_h, strokeColor=colors.HexColor("#374151"), strokeWidth=0.7))
    for tick in [0, 2, 4, 6]:
        y = bottom + chart_h * tick / max_v
        drawing.add(Line(left - 3, y, left + chart_w, y, strokeColor=colors.HexColor("#e5e7eb"), strokeWidth=0.4))
        drawing.add(String(left - 25, y - 3, f"{tick}x", fontName="Helvetica", fontSize=7, fillColor=colors.HexColor("#4b5563")))
    bar_gap = 18
    bar_w = (chart_w - bar_gap * (len(rows) + 1)) / len(rows)
    for idx, (label, value) in enumerate(rows):
        x = left + bar_gap + idx * (bar_w + bar_gap)
        bar_h = chart_h * value / max_v
        drawing.add(Rect(x, bottom, bar_w, bar_h, fillColor=colors.HexColor("#2563eb"), strokeColor=colors.HexColor("#1d4ed8")))
        drawing.add(String(x + 2, bottom + bar_h + 6, f"{value:.2f}x", fontName="Helvetica-Bold", fontSize=8, fillColor=colors.HexColor("#111827")))
        drawing.add(String(x - 1, bottom - 14, label, fontName="Helvetica", fontSize=7, fillColor=colors.HexColor("#374151")))
    drawing.add(String(left, height - 10, "FINUFFT adjoint / structured adjoint speedup", fontName="Helvetica-Bold", fontSize=9, fillColor=colors.HexColor("#111827")))
    return drawing


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawRightString(A4[0] - 0.55 * inch, 0.38 * inch, f"{doc.page}")
    canvas.restoreState()


def build_story(styles: dict[str, ParagraphStyle]) -> list[object]:
    story: list[object] = []
    story.append(p("ODT cone-axis forward/adjoint pair 및 adjoint-only 최적화 요약", styles, "title"))
    story.append(
        p(
            "Atomic WAXS Cake-Map Simulator workspace, local Windows/MSVC benchmark snapshot. "
            "범위는 scalar ODT/high-NA cone-axis prototype이며, universal NUFFT replacement 또는 vectorial Richards-Wolf claim은 포함하지 않는다.",
            styles,
            "subtitle",
        )
    )

    story.append(p("1. 결론", styles, "h1"))
    for item in [
        "Forward/adjoint pair에서는 safe C++ path가 full pair 0.12490 s -> 0.07938 s로 개선되었다. 현재 probe 기준 약 1.57x이다.",
        "Forward kernel은 0.05951 s -> 0.04466 s, adjoint kernel은 0.04570 s -> 0.03686 s로 각각 약 1.33x, 1.24x 개선되었다.",
        "Adjoint-only 최적화에서는 isolated C++ adjoint kernel이 0.04106 s -> 0.03644-0.03772 s로 약 1.09x-1.13x 개선되었다.",
        "FINUFFT adjoint 대비 q-area sweep에서는 2.62x -> 5.82x speedup이 관측되었고, relative L2는 약 9e-10 수준으로 맞았다.",
        "WAXS에서 유효했던 FFT-friendly grid, pruning, factor reuse는 이미 상당 부분 적용됐다. 다음 큰 CPU 이득은 prepared plan, SIMD/data layout, batching, thread scaling 쪽에서 나와야 한다.",
    ]:
        story.append(bullet(item, styles))

    story.append(p("2. 공통 workload 및 해석 기준", styles, "h1"))
    story.append(
        table(
            [
                ["항목", "값"],
                ["object grid", "n_beta=384, n_r=16, n_z=15; object bins=92,160"],
                ["illumination / detector", "n_illum=32, cap_radial=16, cap_phi=64 for base probe"],
                ["optics parameters", "k=17.307319527958313, detector_na=0.9240924092409241, illumination_na=0.877887788778878"],
                ["harmonic settings", "axis_used_modes=85, l_cutoff=41, l_modes=83, active-l fraction about 0.738"],
                ["timing policy", "local Windows wall-time medians; full-call timings can be noisy, so component/kernel timings are reported alongside full timings"],
                ["baseline policy", "FINUFFT is used as a generic optimized NUFFT baseline with eps=1e-12 where direct reference is not feasible"],
            ],
            styles,
            col_widths=[1.55 * inch, 5.25 * inch],
        )
    )

    story.append(PageBreak())
    story.append(p("3. Forward/adjoint pair: 구현 및 최적화", styles, "h1"))
    story.append(p("목표는 forward와 adjoint를 같이 쓰는 pair loop에서 validated safe state를 확보하는 것이었다.", styles))
    story.append(
        table(
            [
                ["구분", "내용", "판정"],
                ["forward precompute", "cone_axis_forward_fold_pruned에서 source_weight 및 source_phase(mode_phase[h] * axial_phase[z])를 재사용", "채택"],
                ["adjoint precompute", "source_phase_conj를 미리 구성하여 반복 곱셈 감소", "채택"],
                ["adjoint branch split", "illumination accumulation에서 첫 assignment와 이후 addition을 분리해 inner branch 제거", "채택"],
                ["build option", "WAXS_CPP_OPT=avx2|native|fast opt-in build mode 추가", "실험 가능, 기본값 아님"],
                ["illum-owned forward", "thread-local folded buffer를 없애는 forward loop variant", "느려서 보류"],
                ["larger (illum,h,r) task", "z를 inner loop로 둔 큰 task granularity", "느려서 보류"],
            ],
            styles,
            col_widths=[1.45 * inch, 4.25 * inch, 1.1 * inch],
        )
    )
    story.append(p("Pair timing", styles, "h2"))
    story.append(
        table(
            [
                ["build / kernel", "full pair s", "component sum s", "forward kernel s", "adjoint kernel s", "full forward s", "full adjoint s"],
                ["baseline safe", "0.12490", "0.10850", "0.05951", "0.04570", "0.06624", "0.05866"],
                ["final safe long", "0.07938", "0.08491", "0.04466", "0.03686", "0.04222", "0.03715"],
                ["AVX2 opt-in", "0.08611", "0.08184", "0.04666", "0.03202", "0.05097", "0.03513"],
                ["AVX2 + /fp:fast", "0.10287", "0.11173", "0.05960", "0.04878", "0.06314", "0.03973"],
            ],
            styles,
            col_widths=[1.35 * inch, 0.82 * inch, 0.93 * inch, 0.93 * inch, 0.93 * inch, 0.92 * inch, 0.92 * inch],
        )
    )
    for item in [
        "safe default가 full pair 기준으로 가장 좋았다. AVX2는 isolated adjoint kernel에는 도움이 되지만 forward 손실 때문에 pair total에서는 채택하지 않았다.",
        "/fp:fast는 현재 테스트를 통과했지만 이 path에서는 느렸다.",
        "Incremental _cpp_odt rebuild는 약 19-27 s, full --force rebuild는 약 52-70 s로 관측됐다.",
    ]:
        story.append(bullet(item, styles))

    story.append(PageBreak())
    story.append(p("4. Adjoint-only: 구현 및 최적화", styles, "h1"))
    story.append(
        p(
            "Forward validation이 끝난 뒤에는 forward를 더 최적화하지 않고, inverse/screening에 직접 중요한 adjoint만 분리해서 밀었다. "
            "profiling script에는 --adjoint-only 및 FINUFFT adjoint 비교 옵션을 추가했다.",
            styles,
        )
    )
    story.append(
        table(
            [
                ["구분", "내용", "판정"],
                ["radial-axis precompute", "radial_axis_conj[h,r,u,z] = radial[h,u,r] * conj(axis_axial[u,z])를 adjoint call 내부에서 미리 구성", "채택"],
                ["active-l lookup pack", "active_psi_conj, active_transverse_conj, active_source_slots를 hot loop 밖에서 packing", "채택"],
                ["l-accumulation structure", "compact-cache contraction 대신 l-accumulation adjoint 구조 유지", "채택"],
                ["compact-cache contraction", "compact_cache[illum,z] 기반 contraction", "정확하지만 느리고 불안정해 보류"],
                ["AVX2 default", "full-call 일부에서는 빠르지만 isolated kernel이 안정적으로 개선되지 않음", "기본값 아님"],
            ],
            styles,
            col_widths=[1.55 * inch, 4.15 * inch, 1.1 * inch],
        )
    )
    story.append(p("Adjoint-only timing", styles, "h2"))
    story.append(
        table(
            [
                ["variant", "full adjoint s", "component sum s", "C++ kernel s", "beta FFT s", "detector IFFT s"],
                ["baseline safe", "0.05210", "0.04239", "0.04106", "0.00109", "0.000241"],
                ["radial-axis precompute", "0.04330", "0.03804", "0.03673", "0.00106", "0.000246"],
                ["radial-axis + active lookup", "0.03630", "0.04001", "0.03858", "0.00118", "0.000242"],
                ["final safe rerun", "0.04094", "0.03911", "0.03772", "0.00113", "0.000266"],
                ["final safe rerun 2", "0.06078", "0.03785", "0.03644", "0.00116", "0.000249"],
                ["AVX2 opt-in rerun", "0.03385", "0.04263", "0.04132", "0.00107", "0.000242"],
            ],
            styles,
            col_widths=[1.65 * inch, 1.0 * inch, 1.05 * inch, 1.05 * inch, 0.92 * inch, 1.0 * inch],
        )
    )
    for item in [
        "가장 안정적인 지표는 isolated C++ adjoint kernel이다. full-call median은 Windows scheduling 영향이 크다.",
        "safe-build kernel 기준 개선은 1.09x-1.13x이고, component sum 기준 개선은 1.08x-1.12x이다.",
        "이 단계 이후 easy precompute 계열은 대부분 수확된 상태로 보인다.",
    ]:
        story.append(bullet(item, styles))

    story.append(PageBreak())
    story.append(p("5. FINUFFT adjoint q-area 비교", styles, "h1"))
    story.append(
        p(
            "NUFFT가 q grid 증가에서 불리해지는지를 보기 위해 cap_radial과 cap_phi를 함께 키웠다. "
            "따라서 cone samples는 row마다 4배씩 증가한다.",
            styles,
        )
    )
    story.append(
        table(
            [
                ["cap radial", "cap phi", "cone samples", "our full adjoint s", "component sum s", "C++ kernel s", "FINUFFT adjoint s", "speedup", "rel-L2"],
                ["16", "64", "32,768", "0.057011", "0.045697", "0.044317", "0.149160", "2.62x", "9.24e-10"],
                ["32", "128", "131,072", "0.046367", "0.052146", "0.048757", "0.190191", "4.10x", "9.55e-10"],
                ["64", "256", "524,288", "0.120021", "0.131863", "0.121207", "0.596828", "4.97x", "8.90e-10"],
                ["128", "512", "2,097,152", "0.246886", "0.328248", "0.241004", "1.435957", "5.82x", "9.06e-10"],
            ],
            styles,
            col_widths=[0.62 * inch, 0.55 * inch, 0.92 * inch, 0.9 * inch, 0.9 * inch, 0.82 * inch, 0.9 * inch, 0.62 * inch, 0.57 * inch],
        )
    )
    story.append(Spacer(1, 0.08 * inch))
    story.append(speedup_bar_chart())
    story.append(p("FINUFFT adjoint 대비 structured adjoint full-call speedup. 이 차트는 local benchmark snapshot 기준이다.", styles, "caption"))
    for item in [
        "relative L2가 약 9e-10로 유지되어, 이 비교에서는 timing 차이가 정확도 희생에서 나온 것이 아니다.",
        "q-area가 커질수록 speedup이 증가했다. 이 방향은 WAXS high-q sweep에서 보였던 geometry-aware factorization 이득과 잘 맞는다.",
        "다만 이 결과는 scalar structured cone-axis grid 기준이다. vectorial high-NA 및 외부 optics package baseline은 별도 검증이 필요하다.",
    ]:
        story.append(bullet(item, styles))

    story.append(PageBreak())
    story.append(p("6. 해석과 다음 최적화 방향", styles, "h1"))
    story.append(p("WAXS에서 옮겨온 최적화가 어디까지 작동했는지와, 앞으로 필요한 최적화 수준을 분리해서 봐야 한다.", styles))
    story.append(
        table(
            [
                ["축", "현재 판단"],
                ["이미 먹힌 WAXS식 아이디어", "FFT-friendly grid, harmonic pruning, factor reuse/precompute, C++ fused contraction, NUFFT q-scaling 비교"],
                ["더 어려워진 이유", "현재 component time 대부분이 C++ adjoint kernel 내부 dense complex contraction에 집중됨"],
                ["CPU에서 다음 후보", "prepared adjoint plan으로 per-call precompute를 plan/cache로 이동, SoA/SIMD layout, batched residuals, thread scaling"],
                ["benchmark에서 다음 후보", "peak RSS, q-area 더 큰 sweep, fixed-accuracy eps sweep, repeated residual/batch loop, external vectorial/domain package baseline"],
                ["논문 claim boundary", "structured scalar cone-axis inverse/backpropagation acceleration. universal NUFFT replacement나 vectorial optics replacement로 쓰기에는 아직 검증 부족"],
            ],
            styles,
            col_widths=[1.7 * inch, 5.1 * inch],
        )
    )

    story.append(p("7. Source and validation files", styles, "h1"))
    for item in [
        "benchmark_results/odt_cone_axis_kernel_push_summary.md",
        "benchmark_results/odt_cone_axis_adjoint_push_summary.md",
        "benchmark_results/odt_cone_axis_adjoint_finufft_qarea_summary.md",
        "scripts/profile_odt_cone_axis_bottleneck.py",
        "src/waxs_cake/_cpp_odt.cpp",
        "setup.py",
    ]:
        story.append(bullet(item, styles))
    story.append(
        p(
            "검증: final safe build 후 tests/test_odt_ewald_operator.py 7 tests passed; "
            "profile script py_compile passed; FINUFFT q-area benchmark commands completed and wrote JSON/CSV/markdown outputs.",
            styles,
        )
    )
    return story


def main() -> None:
    styles = make_styles()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=0.48 * inch,
        leftMargin=0.48 * inch,
        topMargin=0.48 * inch,
        bottomMargin=0.55 * inch,
        title="ODT cone-axis optimization summary",
        author="Codex",
    )
    doc.build(build_story(styles), onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(OUT)


if __name__ == "__main__":
    main()
