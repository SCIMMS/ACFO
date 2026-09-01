from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
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
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
STAMP = "2026_06_27"
PDF_OUT = ROOT / "docs" / f"projected_gradient_mesh_saxs_theory_implementation_{STAMP}.pdf"
ZIP_OUT = ROOT / "docs" / f"projected_gradient_mesh_saxs_theory_implementation_package_{STAMP}.zip"
PACKAGE_DIR = ROOT / "benchmark_results" / f"projected_gradient_mesh_saxs_theory_implementation_package_{STAMP}"

CODE_FILES = [
    ROOT / "scripts" / "optimize_projected_gradient_shape.py",
    ROOT / "scripts" / "benchmark_projected_gradient_rawkernel.py",
    ROOT / "src" / "waxs_cake" / "surface_normal.py",
    ROOT / "tests" / "test_surface_normal.py",
    ROOT / "scripts" / "build_projected_gradient_optimizer_progress_pdf.py",
    ROOT / "scripts" / "build_projected_gradient_theory_implementation_package.py",
]

RESULT_FILES = {
    "clean_gpu_czt": ROOT
    / "benchmark_results"
    / "projected_gradient_optimizer_synthetic_shape4_roll_background_cuda_splatjac_gpu_czt_cached_q129.json",
    "center_disk_mask": ROOT
    / "benchmark_results"
    / "projected_gradient_optimizer_synthetic_shape4_roll_background_centerdisk_mask_q129.json",
    "experimental_mask_synthetic": ROOT
    / "benchmark_results"
    / "projected_gradient_optimizer_synthetic_shape4_roll_background_experimental_mask_q129.json",
    "experimental_mask_npz_smoke": ROOT
    / "benchmark_results"
    / "projected_gradient_optimizer_npz_frame262_experimental_mask_smoke_q129.json",
}


INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#4b5563")
GRID = colors.HexColor("#d1d5db")
HEADER = colors.HexColor("#1f2937")
SOFT = colors.HexColor("#f9fafb")
BLUE = colors.HexColor("#dbeafe")
GREEN = colors.HexColor("#dcfce7")
AMBER = colors.HexColor("#fef3c7")
ROSE = colors.HexColor("#ffe4e6")


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
            fontSize=19.0,
            leading=24.0,
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
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "Heading1KR",
            parent=sample["Heading1"],
            fontName=bold_font,
            fontSize=13.7,
            leading=17.5,
            textColor=INK,
            spaceBefore=8,
            spaceAfter=5,
        ),
        "h2": ParagraphStyle(
            "Heading2KR",
            parent=sample["Heading2"],
            fontName=bold_font,
            fontSize=10.8,
            leading=14,
            textColor=HEADER,
            spaceBefore=5,
            spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "BodyKR",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=8.6,
            leading=12.1,
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
        "equation": ParagraphStyle(
            "EquationKR",
            parent=sample["Code"],
            fontName=body_font,
            fontSize=8.15,
            leading=11.4,
            textColor=colors.HexColor("#172554"),
            backColor=colors.HexColor("#eff6ff"),
            borderColor=colors.HexColor("#bfdbfe"),
            borderWidth=0.35,
            borderPadding=5,
            spaceBefore=3,
            spaceAfter=5,
        ),
        "table": ParagraphStyle(
            "TableKR",
            parent=sample["BodyText"],
            fontName=body_font,
            fontSize=6.8,
            leading=8.3,
            textColor=INK,
        ),
        "table_head": ParagraphStyle(
            "TableHeadKR",
            parent=sample["BodyText"],
            fontName=bold_font,
            fontSize=6.55,
            leading=8.3,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
    }


def clean(text: str) -> str:
    return escape(str(text)).replace("\n", "<br/>")


def p(text: str, styles: dict[str, ParagraphStyle], style: str = "body") -> Paragraph:
    return Paragraph(clean(text), styles[style])


def bullets(items: list[str], styles: dict[str, ParagraphStyle]) -> list[Paragraph]:
    return [p("- " + item, styles) for item in items]


def table(rows: list[list[Any]], styles: dict[str, ParagraphStyle], col_widths: list[float]) -> Table:
    wrapped: list[list[Any]] = []
    for i, row in enumerate(rows):
        style = styles["table_head"] if i == 0 else styles["table"]
        wrapped.append([cell if hasattr(cell, "wrap") else Paragraph(clean(cell), style) for cell in row])
    result = Table(wrapped, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HEADER),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3.3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SOFT]),
            ]
        )
    )
    return result


def callout(text: str, styles: dict[str, ParagraphStyle], color: colors.Color = BLUE) -> Table:
    border = colors.HexColor("#93c5fd")
    if color == GREEN:
        border = colors.HexColor("#86efac")
    elif color == AMBER:
        border = colors.HexColor("#fcd34d")
    elif color == ROSE:
        border = colors.HexColor("#fda4af")
    result = Table([[p(text, styles)]], colWidths=[7.05 * inch], hAlign="LEFT")
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), color),
                ("BOX", (0, 0), (-1, -1), 0.45, border),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return result


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def result_row(label: str, path: Path) -> list[str]:
    data = load_json(path)
    if data is None:
        return [label, "missing", "-", "-", "-", str(path.relative_to(ROOT))]
    fit = data["fit"]
    target = fit["target"]
    rec = fit.get("recovered", {})
    mask_note = target.get("mask_source_shape_mode") or target.get("synthetic_mask_mode") or "none"
    return [
        label,
        f"{float(fit['optimizer_s']):.3f}",
        str(int(fit["nfev"])),
        str(int(fit["fit_pixels"])),
        f"{float(fit['final_rms_log_residual']):.3e}",
        (
            f"{mask_note}; axis=({rec.get('axis_u_a', 0):.3f}, {rec.get('axis_v_a', 0):.3f}); "
            f"shape4={rec.get('shape4', 0):.6f}; roll={rec.get('roll_deg', 0):.4f}"
        ),
    ]


def footer(canvas, doc) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawRightString(A4[0] - 0.58 * inch, 0.42 * inch, f"page {doc.page}")
    canvas.restoreState()


def build_story(styles: dict[str, ParagraphStyle]) -> list[Any]:
    story: list[Any] = []
    story.append(p("Projected-Gradient Mesh SAXS: Theory and Implementation Note", styles, "title"))
    story.append(
        p(
            f"작성일: {date.today().isoformat()} | 산출물: 이론/구현 PDF + 코드 스냅샷 ZIP | "
            "범위: surface-normal mesh SAXS, projected-gradient gridding, GPU RawKernel, CZT, inverse optimization",
            styles,
            "subtitle",
        )
    )
    story.append(
        callout(
            "핵심 요지: sharp-interface mesh의 표면 normal 정보를 projected-gradient source로 바꾸면, "
            "signed raster처럼 uniform-grid Fourier transfer를 재사용하면서도 면적/normal/shape derivative를 보존할 수 있다. "
            "따라서 순수 forward speed claim보다 differentiable mesh-SAXS inverse problem claim이 더 강하다.",
            styles,
            GREEN,
        )
    )

    story.append(p("1. Physical starting point", styles, "h1"))
    story.append(
        p(
            "밀도 contrast가 sharp interface에서 piecewise constant라고 두면, volume integral amplitude는 "
            "divergence theorem으로 surface-normal source에 연결된다. 이때 closed, watertight, consistently oriented mesh가 기본 가정이다.",
            styles,
        )
    )
    story.append(
        p(
            "F(q) = integral_V exp(-i q dot r) dr = (1 / (-i |q|^2)) integral_S (q dot n) exp(-i q dot r) dS",
            styles,
            "equation",
        )
    )
    story.extend(
        bullets(
            [
                "이 식은 q=0에서 직접 쓰면 singular하므로, low-q에서는 mesh volume/moment expansion으로 대체한다.",
                "Exact triangle integral은 reference와 fallback으로 남기고, fast path는 sub-face quadrature / median-line style source를 사용한다.",
                "Ewald-ring WAXS/SAXS와 flat qz=0 consistency plane 모두 같은 q geometry abstraction으로 다룰 수 있다.",
            ],
            styles,
        )
    )

    story.append(p("2. Projected-gradient reformulation", styles, "h1"))
    story.append(
        p(
            "Detector가 작은-angle q_u, q_v grid로 주어질 때, 표면 source를 projection plane의 두 gradient 성분으로 누적한다. "
            "각 triangle sample은 위치 (u, v)에 normal-area 성분 G_u, G_v를 CIC 방식으로 splat한다.",
            styles,
        )
    )
    story.append(
        p(
            "A(q_u,q_v) = (q_u FFT_CZT[G_u](q_u,q_v) + q_v FFT_CZT[G_v](q_u,q_v)) / (-i (q_u^2 + q_v^2))",
            styles,
            "equation",
        )
    )
    story.extend(
        bullets(
            [
                "Signed raster는 projected thickness rho(u,v)를 grid에 만든 뒤 CZT한다.",
                "Projected-gradient는 rho 대신 surface normal projected source G_u, G_v를 grid에 만든 뒤 같은 CZT machinery를 사용한다.",
                "그 결과 forward speed의 최종 ceiling은 uniform-grid FT/CZT transfer 비용에 가까워진다.",
                "차별점은 source가 normal/area derivative를 포함하므로 gradient-based shape optimization으로 자연스럽게 확장된다는 점이다.",
            ],
            styles,
        )
    )

    story.append(p("3. GPU implementation", styles, "h1"))
    story.append(
        table(
            [
                ["Component", "Implementation detail"],
                [
                    "Projected-gradient RawKernel",
                    "scripts/benchmark_projected_gradient_rawkernel.py의 PreparedProjectedGradientRawKernel이 triangle samples를 GPU에서 CIC splat한다.",
                ],
                [
                    "CZT contraction",
                    "saxs_module.czt2_from_projected_density와 동일한 convention을 사용하며, q dot F 결합 뒤 low-q moment expansion을 적용한다.",
                ],
                [
                    "Derivative splat",
                    "scripts/optimize_projected_gradient_shape.py의 CUDA kernel이 axis_u, axis_v, shape4에 대한 dG/da grid를 batched complex64로 만든다.",
                ],
                [
                    "GPU derivative-CZT",
                    "CuPy CZT callable plan을 cache하여 반복 least-squares에서 chirp plan 재생성 비용을 줄인다.",
                ],
                [
                    "Mask path",
                    "load_detector_mask가 (500,500,1) mask를 np.squeeze로 2D화하고, detector q crop slice로 잘라 objective mask에 넣는다.",
                ],
            ],
            styles,
            [1.75 * inch, 5.3 * inch],
        )
    )

    story.append(p("4. Optimizer parameterization", styles, "h1"))
    story.extend(
        bullets(
            [
                "Synthetic target은 ellipsoid axes, in-plane roll, optional shape4 radial basis, intensity scale, constant background로 만든다.",
                "Fit parameter는 log(axis_u), log(axis_v), log(intensity_scale), optional shape4, optional roll_rad, optional log(background_frac)이다.",
                "단일 qz=0 projection에서 axis_w는 contrast/background와 거의 degenerate하므로 현재 실험에서는 fixed로 둔다.",
                "Jacobian은 scale/background analytic, roll은 grid-gradient, axis/shape는 splat derivative, 나머지는 finite difference fallback이다.",
            ],
            styles,
        )
    )

    story.append(p("5. Verification summary", styles, "h1"))
    story.append(
        table(
            [
                ["Run", "Time", "nfev", "fit pixels", "RMS", "Recovered / mask note"],
                result_row("Clean synthetic, GPU-CZT cached", RESULT_FILES["clean_gpu_czt"]),
                result_row("Synthetic center-disk mask", RESULT_FILES["center_disk_mask"]),
                result_row("Synthetic experimental mask", RESULT_FILES["experimental_mask_synthetic"]),
                result_row("NPZ experimental mask smoke", RESULT_FILES["experimental_mask_npz_smoke"]),
            ],
            styles,
            [1.62 * inch, 0.55 * inch, 0.38 * inch, 0.55 * inch, 0.72 * inch, 3.23 * inch],
        )
    )
    story.append(
        p(
            "Mask-specific check: E:/XFEL_image/SAXS_mask.npy는 raw shape (500,500,1), squeezed shape (500,500), "
            "full valid fraction 0.841324, q129 crop valid fraction 0.359834, center valid fraction 0.0으로 direct beam block이 확인되었다.",
            styles,
        )
    )
    story.append(
        p(
            "GPU derivative-CZT correctness check에서는 CPU derivative-CZT 대비 fit mask relative L2가 "
            "axis_u 6.73e-08, axis_v 1.77e-07, shape4 1.45e-05 수준이었다. "
            "surface-normal regression test는 pytest tests/test_surface_normal.py -q에서 4 passed로 유지되었다.",
            styles,
        )
    )

    story.append(PageBreak())
    story.append(p("6. Why signed raster is still hard to beat", styles, "h1"))
    story.append(
        callout(
            "Signed raster가 빠른 근본 원인은 projection 이후 문제가 uniform-grid Fourier transform으로 바뀌기 때문이다. "
            "Projected-gradient도 같은 transfer를 쓰는 한, forward-only speed에서 큰 폭의 우위는 기대하기 어렵다.",
            styles,
            AMBER,
        )
    )
    story.extend(
        bullets(
            [
                "GPU RawKernel splat 자체는 매우 빠르지만, detector q grid로 보내는 CZT/FFT 계층이 반복 비용의 큰 부분이 된다.",
                "Projected-gradient의 실질적 이득은 더 정밀한 surface quadrature와 differentiable shape parameter에 있다.",
                "큰 q grid, 많은 derivative parameter, 반복 optimization에서는 GPU-resident derivative path가 더 유리해질 가능성이 있다.",
            ],
            styles,
        )
    )

    story.append(p("7. Current limitations and claim boundary", styles, "h1"))
    story.extend(
        bullets(
            [
                "Fast path는 exact triangle integral이 아니라 sub-face quadrature approximation이다.",
                "Closed oriented mesh, sharp interface, constant density contrast라는 가정을 벗어나면 별도 모델링이 필요하다.",
                "Real NPZ fitting은 mask/squeeze/crop smoke를 통과했지만, 물리적으로 의미 있는 parameterization과 regularization은 아직 다음 단계다.",
                "따라서 claim은 'projection-free sharp-interface mesh SAXS with GPU projected-gradient gridding and differentiable optimization'으로 제한하는 것이 안전하다.",
            ],
            styles,
        )
    )

    story.append(p("8. Code snapshot included in ZIP", styles, "h1"))
    story.append(
        table(
            [["Path in ZIP", "Purpose"]]
            + [
                [f"code/{path.relative_to(ROOT).as_posix()}", "source snapshot"]
                for path in CODE_FILES
            ]
            + [
                ["results/*.json", "benchmark and mask-validation outputs"],
                ["README.md / MANIFEST.json", "package contents and checksums"],
            ],
            styles,
            [2.75 * inch, 4.3 * inch],
        )
    )
    story.append(
        p(
            "Recommended rerun path: use .venv/Scripts/python.exe for PDF/package builders and PyMuPDF verification; "
            "use the same venv for optimizer tests because ReportLab, CuPy, SciPy, and PyMuPDF availability are environment-dependent.",
            styles,
            "small",
        )
    )
    return story


def build_pdf() -> None:
    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = make_styles()
    doc = SimpleDocTemplate(
        str(PDF_OUT),
        pagesize=A4,
        rightMargin=0.58 * inch,
        leftMargin=0.58 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.6 * inch,
        title="Projected-Gradient Mesh SAXS Theory and Implementation Note",
        author="Codex",
    )
    doc.build(build_story(styles), onFirstPage=footer, onLaterPages=footer)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_reset_package_dir() -> None:
    root_resolved = ROOT.resolve()
    package_resolved = PACKAGE_DIR.resolve()
    if root_resolved not in package_resolved.parents:
        raise RuntimeError(f"Refusing to reset package directory outside workspace: {PACKAGE_DIR}")
    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)


def write_package() -> None:
    safe_reset_package_dir()
    entries: list[dict[str, Any]] = []

    pdf_dest = PACKAGE_DIR / "docs" / PDF_OUT.name
    pdf_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PDF_OUT, pdf_dest)
    entries.append(
        {
            "kind": "pdf",
            "source": str(PDF_OUT.relative_to(ROOT)),
            "path": str(pdf_dest.relative_to(PACKAGE_DIR)).replace("\\", "/"),
            "sha256": sha256(pdf_dest),
        }
    )

    for src in CODE_FILES:
        if not src.exists():
            continue
        dest = PACKAGE_DIR / "code" / src.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        entries.append(
            {
                "kind": "code",
                "source": str(src.relative_to(ROOT)),
                "path": str(dest.relative_to(PACKAGE_DIR)).replace("\\", "/"),
                "sha256": sha256(dest),
            }
        )

    for label, src in RESULT_FILES.items():
        if not src.exists():
            continue
        dest = PACKAGE_DIR / "results" / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        entries.append(
            {
                "kind": "result_json",
                "label": label,
                "source": str(src.relative_to(ROOT)),
                "path": str(dest.relative_to(PACKAGE_DIR)).replace("\\", "/"),
                "sha256": sha256(dest),
            }
        )

    readme = PACKAGE_DIR / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Projected-Gradient Mesh SAXS Theory and Implementation Package",
                "",
                f"Generated: {date.today().isoformat()}",
                "",
                "Contents:",
                "- `docs/`: theory and implementation PDF.",
                "- `code/`: source-code snapshot used for the current projected-gradient optimizer experiments.",
                "- `results/`: JSON outputs for clean, masked synthetic, and NPZ mask-smoke runs.",
                "- `MANIFEST.json`: checksums and source paths.",
                "",
                "Primary rerun commands:",
                "- `.venv\\Scripts\\python.exe -m pytest tests\\test_surface_normal.py -q`",
                "- `.venv\\Scripts\\python.exe scripts\\build_projected_gradient_theory_implementation_package.py`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    entries.append(
        {
            "kind": "readme",
            "source": None,
            "path": "README.md",
            "sha256": sha256(readme),
        }
    )

    manifest = {
        "generated": date.today().isoformat(),
        "workspace": str(ROOT),
        "pdf": str(PDF_OUT.relative_to(ROOT)),
        "zip": str(ZIP_OUT.relative_to(ROOT)),
        "entries": entries,
    }
    manifest_path = PACKAGE_DIR / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    entries.append(
        {
            "kind": "manifest",
            "source": None,
            "path": "MANIFEST.json",
            "sha256": sha256(manifest_path),
        }
    )
    manifest["entries"] = entries
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if ZIP_OUT.exists():
        ZIP_OUT.unlink()
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(PACKAGE_DIR.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(PACKAGE_DIR)).replace("\\", "/"))


def main() -> None:
    build_pdf()
    write_package()
    print(PDF_OUT)
    print(ZIP_OUT)


if __name__ == "__main__":
    main()
