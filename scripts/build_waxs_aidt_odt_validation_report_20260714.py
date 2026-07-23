from __future__ import annotations

import html
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results"
DOCS = ROOT / "docs"
SUPPORT = DOCS / "waxs_aidt_odt_validation_report_20260714_support"
BASELINE_DATA = DOCS / "waxs_aidt_odt_progress_report_20260713_support" / "report_data.json"
REPORT_DATA = SUPPORT / "report_data.json"
SOURCE_NOTES = SUPPORT / "source_notes.md"
HTML_OUTPUT = SUPPORT / "report_print.html"
PDF_OUTPUT = DOCS / "ACFO_WAXS_aIDT_ODT_validation_progress_report_2026-07-14_ko.pdf"
CHROME = Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def table(headers: list[str], rows: Iterable[Iterable[Any]], widths: list[str] | None = None) -> str:
    width_html = ""
    if widths:
        width_html = "<colgroup>" + "".join(f'<col style="width:{esc(width)}">' for width in widths) + "</colgroup>"
    head = "".join(f"<th>{esc(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell if isinstance(cell, SafeHtml) else esc(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f'<table>{width_html}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


class SafeHtml(str):
    pass


def badge(text: str, kind: str = "neutral") -> SafeHtml:
    return SafeHtml(f'<span class="badge {esc(kind)}">{esc(text)}</span>')


def linear_bar_chart(rows: list[tuple[str, float, str]], maximum: float, target: float | None = None) -> str:
    rendered: list[str] = []
    for label, value, note in rows:
        width = min(100.0, 100.0 * value / maximum)
        target_marker = ""
        if target is not None:
            target_marker = f'<span class="target" style="left:{100.0 * target / maximum:.3f}%"></span>'
        rendered.append(
            '<div class="bar-row">'
            f'<div class="bar-label">{esc(label)}</div>'
            '<div class="bar-track">'
            f'{target_marker}<div class="bar-fill" style="width:{width:.3f}%"></div>'
            '</div>'
            f'<div class="bar-value">{esc(note)}</div>'
            '</div>'
        )
    return '<div class="bar-chart">' + "".join(rendered) + "</div>"


def log_bar_chart(rows: list[tuple[str, float, str]], maximum: float) -> str:
    denominator = math.log10(maximum)
    rendered: list[str] = []
    for label, value, note in rows:
        width = 100.0 * math.log10(max(value, 1.0)) / denominator
        rendered.append(
            '<div class="bar-row">'
            f'<div class="bar-label">{esc(label)}</div>'
            f'<div class="bar-track log"><div class="bar-fill teal" style="width:{width:.3f}%"></div></div>'
            f'<div class="bar-value">{esc(note)}</div>'
            '</div>'
        )
    return '<div class="bar-chart">' + "".join(rendered) + '<div class="axis-note">막대 길이는 log₁₀ 축, 숫자는 실제 speedup</div></div>'


def metric(label: str, value: str, note: str, color: str = "blue") -> str:
    return (
        f'<div class="metric {esc(color)}">'
        f'<div class="metric-label">{esc(label)}</div>'
        f'<div class="metric-value">{esc(value)}</div>'
        f'<div class="metric-note">{esc(note)}</div>'
        '</div>'
    )


def page(number: int, kicker: str, title: str, body: str, last: bool = False) -> str:
    if number == 1:
        klass = "page cover"
    else:
        klass = "page last" if last else "page"
    return (
        f'<section class="{klass}" id="page-{number:02d}">'
        '<header class="page-header">'
        f'<div><span class="kicker">{esc(kicker)}</span><h1>{esc(title)}</h1></div>'
        f'<div class="page-no">{number:02d}</div>'
        '</header>'
        f'<main>{body}</main>'
        '<footer><span>ACFO validation progress report</span><span>2026-07-14 · local RTX 2070 SUPER measurements unless stated</span></footer>'
        '</section>'
    )


def main() -> None:
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    SUPPORT.mkdir(parents=True, exist_ok=True)

    baseline = load_json(BASELINE_DATA)
    waxs_rows = baseline["datasets"]["waxs_fixed_dq"]
    waxs_evidence = baseline["datasets"]["waxs_evidence"]
    aidt_rows = baseline["datasets"]["aidt_conditions"]
    derived0 = baseline["derived_metrics"]

    final_abba = load_json(RESULTS / "odt_h28_rank16_adaptive_l1e-6_vs_cufinufft_abba30_final.json")
    final_validation = load_json(RESULTS / "odt_final_packed_candidate_validation.json")
    adaptive = load_json(RESULTS / "odt_adaptive_l_packed_sweep.json")
    full_slab = load_json(RESULTS / "odt_full_slab_reconstruction_claim.json")
    banded = load_json(RESULTS / "odt_banded_cartesian_reconstruction.json")
    banded_recheck = load_json(RESULTS / "odt_banded_cartesian_remap_hot_recheck.json")
    old_legacy = load_json(RESULTS / "odt_torch_256cubed_100pair.json")
    old_reduced = load_json(RESULTS / "odt_torch_256cubed_reduced_100pair.json")

    abba = final_abba["summary"]
    assert final_validation["passed"] is True
    assert abs(abba["ours_forward_adjoint_pair_s"] - 0.1113661499985028) < 1e-9
    assert abs(abba["ours_speedup_vs_cufinufft_pair"] - 81.24105933556623) < 1e-9

    adaptive_rows = adaptive["rows"]
    chosen_adaptive = next(row for row in adaptive_rows if row["threshold"] == 1e-6)
    assert chosen_adaptive["passed"] is True
    assert next(row for row in adaptive_rows if row["threshold"] == 1e-5)["passed"] is False

    recon_rows: list[dict[str, Any]] = []
    for case in full_slab["cases"]:
        timing = case["reconstruction"]["iteration_core_timing"]
        final = case["reconstruction"]["final"]
        recon_rows.append({
            "z": int(case["selected_n_z"]),
            "known_support": bool(case["known_z_support"]),
            "median_ms": 1000.0 * timing["median_s"],
            "p95_ms": 1000.0 * timing["p95_s"],
            "hz": 1.0 / timing["median_s"],
            "hundred_update_s": case["reconstruction"]["reconstruction_core_s_including_rhs"],
            "object_nrmse_pct": 100.0 * final["object_nrmse"],
            "data_residual_pct": 100.0 * final["data_residual"],
        })
    full = next(row for row in recon_rows if row["z"] == 256)
    slab128 = next(row for row in recon_rows if row["z"] == 128)
    slab1 = next(row for row in recon_rows if row["z"] == 1)

    banded_rows: list[dict[str, Any]] = []
    remap1 = banded_recheck["cases"][0]["remap_timing"]["median_s"]
    for case in banded["cases"]:
        z = int(case["selected_n_z"])
        remap_s = remap1 if z == 1 else case["remap_timing"]["median_s"]
        core_s = case["cartesian_remap"]["core_iteration_timing"]["median_s"]
        banded_rows.append({
            "z": z,
            "samples_per_view": sum(band["samples_per_view"] for band in banded["bands"]),
            "sample_reduction_pct": 100.0 * (1.0 - sum(band["samples_per_view"] for band in banded["bands"]) / 65536.0),
            "input_rel_pct": 100.0 * case["cartesian_remap_rel_l2_vs_ideal"],
            "object_nrmse_pct": 100.0 * case["cartesian_remap"]["object_rel_l2"],
            "reconstruction_diff_pct": 100.0 * case["remap_vs_ideal_reconstruction_rel_l2"],
            "core_ms": 1000.0 * core_s,
            "remap_ms": 1000.0 * remap_s,
            "total_ms": 1000.0 * (core_s + remap_s),
            "hz": 1.0 / (core_s + remap_s),
        })
    banded1 = next(row for row in banded_rows if row["z"] == 1)
    banded8 = next(row for row in banded_rows if row["z"] == 8)

    old_legacy_pair = float(old_legacy["pair_timing"]["median_s"])
    old_reduced_pair = float(old_reduced["pair_timing"]["median_s"])
    final_pair = float(abba["ours_forward_adjoint_pair_s"])
    final_speedups = [
        {"comparison": "H36 dense structured direct → final", "speedup": final_validation["candidate"]["speedup_vs_reference"]},
        {"comparison": "old illumination-reduced → final", "speedup": old_reduced_pair / final_pair},
        {"comparison": "cuFINUFFT reusable-plan → final", "speedup": abba["ours_speedup_vs_cufinufft_pair"]},
        {"comparison": "legacy ACFO stream → final", "speedup": old_legacy_pair / final_pair},
    ]

    rtx2070_fp32 = 9.1
    rtx5090_fp32 = 104.8
    rtx2070_bw = 448.0
    rtx5090_bw = 1792.0
    compute_ratio = rtx5090_fp32 / rtx2070_fp32
    bandwidth_ratio = rtx5090_bw / rtx2070_bw
    floor_p95_ms = 1000.0 * next(case for case in full_slab["cases"] if case["selected_n_z"] == 1)["reconstruction"]["iteration_core_timing"]["p95_s"]
    full_p95_ms = full["p95_ms"]
    projected_p95_ms_2x_delta = floor_p95_ms + (full_p95_ms - floor_p95_ms) / 2.0
    projected_p95_hz_2x_delta = 1000.0 / projected_p95_ms_2x_delta
    required_full_speedup = full_p95_ms / 100.0

    report_data = {
        "schema": "acfo-waxs-aidt-odt-validation-progress-report-v2",
        "generated_at_utc": generated_at,
        "measurement_date_through": "2026-07-13",
        "report_date": "2026-07-14",
        "hardware": "NVIDIA GeForce RTX 2070 SUPER 8 GiB unless stated",
        "waxs_fixed_dq": waxs_rows,
        "waxs_evidence": waxs_evidence,
        "aidt_conditions": aidt_rows,
        "odt_final_operator": {
            "pair_median_ms": 1000.0 * final_pair,
            "pair_iqr_ms": [1000.0 * abba["pair_timing_protocol"]["ours_distribution"]["q1_s"], 1000.0 * abba["pair_timing_protocol"]["ours_distribution"]["q3_s"]],
            "cufinufft_pair_median_s": abba["cufinufft_forward_adjoint_pair_s"],
            "speedup_vs_cufinufft": abba["ours_speedup_vs_cufinufft_pair"],
            "setup_s": abba["ours_operator_setup_s"],
            "cufinufft_setup_s": abba["cufinufft_operator_setup_s"],
            "active_fraction": abba["ring_adaptive_l_active_fraction"],
            "validation": final_validation,
            "speedups": final_speedups,
        },
        "odt_adaptive_l": adaptive_rows,
        "odt_banded_detector": banded_rows,
        "odt_full_slab": recon_rows,
        "rtx5090_projection": {
            "rtx2070_fp32_tflops": rtx2070_fp32,
            "rtx5090_fp32_tflops": rtx5090_fp32,
            "compute_ratio": compute_ratio,
            "rtx2070_bandwidth_gbs": rtx2070_bw,
            "rtx5090_bandwidth_gbs": rtx5090_bw,
            "bandwidth_ratio": bandwidth_ratio,
            "required_full_p95_speedup_for_10hz": required_full_speedup,
            "conservative_2x_delta_p95_ms": projected_p95_ms_2x_delta,
            "conservative_2x_delta_hz": projected_p95_hz_2x_delta,
        },
    }
    REPORT_DATA.write_text(json.dumps(report_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    waxs_chart = linear_bar_chart(
        [(row["q_label"], float(row["speedup"]), f'{row["speedup"]:.1f}×') for row in waxs_rows],
        maximum=160.0,
    )
    aidt_chart = linear_bar_chart(
        [(row["condition"], float(row["hz"]), f'{row["hz"]:.2f} Hz') for row in aidt_rows],
        maximum=20.0,
        target=10.0,
    )
    odt_speedup_chart = log_bar_chart(
        [(row["comparison"], float(row["speedup"]), f'{row["speedup"]:.2f}×') for row in final_speedups],
        maximum=110.0,
    )
    odt_rate_chart = linear_bar_chart(
        [(f'z={row["z"]}', float(row["hz"]), f'{row["hz"]:.2f} Hz') for row in recon_rows],
        maximum=15.0,
        target=10.0,
    ).replace('class="bar-chart"', 'class="bar-chart compact"', 1)

    css = r"""
    @page { size: A4; margin: 0; }
    * { box-sizing: border-box; }
    body { margin: 0; color: #172033; background: #e9edf3; font-family: "Malgun Gothic", "Noto Sans KR", Arial, sans-serif; }
    .page { width: 210mm; min-height: 297mm; padding: 15mm 15mm 13mm; margin: 0 auto 8mm; background: #fff; page-break-after: always; position: relative; }
    .page.last { page-break-after: auto; }
    body:has(.page:target) .page:not(:target) { display: none; }
    body:has(.page:target) .page:target { margin: 0; }
    .page-header { display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 2px solid #1f5b93; padding-bottom: 5mm; margin-bottom: 6mm; }
    h1 { font-size: 20pt; line-height: 1.22; margin: 1.5mm 0 0; letter-spacing: -0.04em; color: #102944; }
    h2 { font-size: 13pt; color: #174f82; margin: 5mm 0 2mm; letter-spacing: -0.03em; }
    h3 { font-size: 10.2pt; color: #23415d; margin: 4mm 0 1.5mm; }
    p, li { font-size: 8.9pt; line-height: 1.58; margin: 0 0 2.4mm; }
    ul, ol { padding-left: 5.5mm; margin: 1.5mm 0 3mm; }
    .kicker { font-size: 7.7pt; font-weight: 700; letter-spacing: .12em; color: #2b7a78; text-transform: uppercase; }
    .page-no { font-size: 17pt; font-weight: 800; color: #b4c3d2; }
    footer { position: absolute; left: 15mm; right: 15mm; bottom: 7mm; display: flex; justify-content: space-between; border-top: 1px solid #dbe3eb; padding-top: 2.5mm; font-size: 6.8pt; color: #6a7888; }
    .cover { background: linear-gradient(135deg, #0c2741 0%, #154f77 55%, #197a83 100%); color: white; }
    .cover .page-header { border: none; }
    .cover h1 { color: white; font-size: 28pt; margin-top: 8mm; max-width: 150mm; }
    .cover .kicker { color: #9de2d8; }
    .cover .page-no { color: rgba(255,255,255,.35); }
    .cover footer { color: rgba(255,255,255,.7); border-color: rgba(255,255,255,.25); }
    .cover-lead { margin: 12mm 0 8mm; font-size: 13pt; line-height: 1.55; max-width: 168mm; }
    .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 3mm; margin: 5mm 0; }
    .metric { border-radius: 3mm; padding: 4mm; min-height: 30mm; background: #eef5fb; border-top: 3px solid #2e74aa; }
    .cover .metric { background: rgba(255,255,255,.11); border-top-color: #9de2d8; }
    .metric.teal { border-top-color: #2b8c82; background: #edf8f6; }
    .metric.amber { border-top-color: #cc8b25; background: #fff7e8; }
    .metric-label { font-size: 7.1pt; font-weight: 700; color: #5b7187; }
    .cover .metric-label { color: #c7e4ec; }
    .metric-value { font-size: 18pt; font-weight: 800; margin: 2mm 0 1mm; letter-spacing: -0.04em; }
    .metric-note { font-size: 6.8pt; line-height: 1.4; color: #637587; }
    .cover .metric-note { color: #d8e8ee; }
    .callout { border-left: 4px solid #2b7a78; background: #f1f8f7; padding: 3.2mm 4mm; margin: 3mm 0; }
    .callout.warn { border-left-color: #d18a1d; background: #fff7e8; }
    .callout strong { color: #174f82; }
    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 5mm; }
    .three-col { display: grid; grid-template-columns: repeat(3,1fr); gap: 4mm; }
    table { width: 100%; border-collapse: collapse; margin: 2.5mm 0 4mm; table-layout: fixed; }
    th { background: #163b5b; color: white; font-size: 7.1pt; line-height: 1.3; padding: 2.2mm 1.8mm; text-align: left; }
    td { border-bottom: 1px solid #dbe3eb; padding: 1.8mm; vertical-align: top; font-size: 7.15pt; line-height: 1.38; word-break: keep-all; }
    tr:nth-child(even) td { background: #f7f9fb; }
    .badge { display: inline-block; padding: .7mm 1.5mm; border-radius: 2mm; font-size: 6.5pt; font-weight: 700; background: #e8edf2; color: #4a5968; }
    .badge.pass { background: #dff3e8; color: #17653c; }
    .badge.fail { background: #fae5e5; color: #9b2929; }
    .badge.conditional { background: #fff0cf; color: #8a5a00; }
    .bar-chart { margin: 3mm 0 4mm; padding: 3mm 3mm 2mm; border: 1px solid #dbe3eb; border-radius: 2mm; background: #fbfcfd; }
    .bar-row { display: grid; grid-template-columns: 50mm 1fr 24mm; align-items: center; gap: 2mm; min-height: 9mm; }
    .bar-chart.compact .bar-row { min-height: 6.8mm; }
    .bar-label { font-size: 7.2pt; line-height: 1.25; }
    .bar-track { height: 4.5mm; background: #e3eaf0; border-radius: 2mm; position: relative; overflow: visible; }
    .bar-track.log { background: linear-gradient(90deg,#e5edf2 0,#e5edf2 48%,#d6e3e7 49%,#e5edf2 50%,#e5edf2 82%,#d6e3e7 83%,#e5edf2 84%); }
    .bar-fill { height: 100%; background: linear-gradient(90deg,#2d6fa3,#3f91be); border-radius: 2mm; min-width: 1.5mm; }
    .bar-fill.teal { background: linear-gradient(90deg,#267a78,#4ea99e); }
    .bar-value { text-align: right; font-size: 7.2pt; font-weight: 700; }
    .target { position: absolute; top: -1.5mm; bottom: -1.5mm; width: 1px; background: #d18122; z-index: 2; }
    .axis-note { text-align: right; font-size: 6.4pt; color: #6d7c8d; margin-top: 1mm; }
    .small { font-size: 7.2pt; color: #5f6f7f; line-height: 1.45; }
    .formula { font-family: Consolas, monospace; font-size: 7.4pt; background: #f2f5f8; padding: 2mm 3mm; border-radius: 1.5mm; }
    .divider { height: 1px; background: #dbe3eb; margin: 4mm 0; }
    .source-list li { font-size: 6.9pt; line-height: 1.35; margin-bottom: 1.2mm; word-break: break-all; }
    """

    pages: list[str] = []

    cover_body = (
        '<p class="cover-lead">WAXS의 정확도·detector-aware 성능, aIDT의 10 Hz급 GPU-resident 처리, 그리고 ODT의 packed operator·detector geometry·full/slab reconstruction 진전을 하나의 검증 경계 안에서 정리했다.</p>'
        '<div class="metrics">'
        + metric("WAXS detector-aware", "1.976×", "Nq=512, ACFO 21.60 s vs FINUFFT 42.69 s")
        + metric("aIDT hot core", "10.31 Hz", "700×700×35, GPU-resident")
        + metric("ODT final pair", "111.366 ms", "forward+adjoint, reusable geometry")
        + metric("ODT vs cuFINUFFT", "81.24×", "AB/BA 30-repeat median ratio")
        + '</div>'
        '<div class="callout" style="background:rgba(255,255,255,.10);border-color:#9de2d8;color:white;margin-top:9mm">'
        '<strong style="color:#bff0e7">핵심 결론.</strong> ODT는 더 이상 단순 가능성 수준에 머물지 않는다. 최종 packed operator는 direct structured reference 허용오차를 통과했고, full 256³ update는 8.46 Hz, known-support z≤128 slab은 10 Hz 이상이다. 다만 이는 “초기 상태에서 매 프레임 완전 수렴”이 아니라 fixed geometry의 반복 update 처리율이다.</div>'
        '<p class="small" style="color:#d9e8ef;margin-top:8mm">보고서 날짜 2026-07-14 · 저장된 실험 결과는 2026-07-13까지 · 별도 표기가 없으면 RTX 2070 SUPER 8 GiB local measurement</p>'
    )
    pages.append(page(1, "Validation synthesis", "ACFO WAXS · aIDT · ODT\n검증 및 성능 진행 보고서", cover_body))

    scope_table = table(
        ["영역", "현재 닫힌 주장", "아직 닫히지 않은 주장"],
        [
            ["WAXS", "동일 source operator 정확도; detector-aware local speed/memory", "coarse-bin high-q atomistic correctness; 독립 머신 timing"],
            ["aIDT", "700×700×35 GPU-resident core 10.31 Hz", "raw acquisition부터 end-to-end 10 Hz"],
            ["ODT operator", "H28/rank16/adaptive-L 1e-6: direct structured reference PASS; 81.24× vs cuFINUFFT", "모든 geometry/모든 GPU에서 동일 우위"],
            ["ODT reconstruction", "full 8.46 Hz update; z≤128 known-support slab ≥10 Hz", "10 complete cold-start reconstructions/s; noisy experimental recovery"],
        ],
        ["18%", "41%", "41%"],
    )
    body2 = (
        '<div class="callout"><strong>측정 경계:</strong> `hot`은 geometry/setup 이후 반복 실행 중앙값이다. Speedup은 표에 적힌 baseline 중앙값을 ACFO 중앙값으로 나눈 값이다. 다른 실험의 speedup은 곱하지 않는다.</div>'
        + scope_table
        + '<div class="two-col"><div><h2>정확도 지표</h2><p><b>Complex L2</b> = ||candidate-reference||₂ / ||reference||₂. WAXS operator error와 exact-atom source representation error는 별도 gate다. ODT의 cuFINUFFT 상호 차이는 exact-reference error가 아니며, exact 기준은 H36 dense structured direct 검증이다.</p></div>'
        '<div><h2>ODT 성능 지표</h2><p><b>pair</b>는 forward+adjoint 1회다. <b>update rate</b>는 CG normal-equation iteration core의 반복률이다. 100 iterations의 총 시간과 동일한 주장이 아니며, setup·acquisition transfer·hologram demodulation은 별도다.</p></div></div>'
        '<h2>판독 원칙</h2><ol><li>Measured와 projected를 분리한다.</li><li>full volume, known-support slab, detector-remap 실험을 별도 branch로 유지한다.</li><li>10 Hz는 프레임마다 필요한 update 횟수와 함께 해석한다.</li><li>missing cone/conditioning 문제를 operator 오류로 재분류하지 않는다.</li></ol>'
    )
    pages.append(page(2, "Scope & definitions", "무엇을 측정했고 무엇을 주장하는가", body2))

    waxs_table_rows: list[list[Any]] = []
    for row in waxs_evidence:
        status = row["status"]
        status_label = "PASS" if "PASS" in status else ("FAIL" if "FAIL" in status else "조건부")
        status_kind = "pass" if "PASS" in status else ("fail" if "FAIL" in status else "conditional")
        waxs_table_rows.append([row["item"], row["condition"], row["result"], badge(status_label, status_kind)])
    waxs_table = table(
        ["검증 항목", "조건", "결과", "판정"],
        waxs_table_rows,
        ["17%", "31%", "36%", "16%"],
    )
    body3 = (
        '<div class="metrics">'
        + metric("Operator production max", "0.91 ppm", "direct complex128 NDFT 기준")
        + metric("Full harmonic max", "6.73e−14", "same-source direct agreement", "teal")
        + metric("High-q pixel error", "77.65%", "coarse source vs exact atoms", "amber")
        + metric("Exact-beta fine R,z", "0.779%", "unit-cell pixel intensity", "teal")
        + '</div>'
        + '<p>동일 source와 curved target을 사용하는 operator 검증은 통과했다. Full-harmonic complex128 경로는 direct NDFT와 2.53e−15–6.73e−14로 일치했고, production R-dependent 경로도 1.77e−7–9.11e−7였다. 그러나 q=5.0–6.3 Å⁻¹에서 coarse 0.1 nm/Nφ=750 source를 exact atoms와 비교하면 pixel intensity 77.65%, ring 22.55% 차이로 실패했다.</p>'
        + '<p>Per-atom β를 harmonic phase에 직접 넣은 exact-beta bridge는 direct complex L2 1.01e−12, fine R,z pixel 0.779%, ring 0.105%를 달성했다. 이는 source 병목을 줄일 경로를 보였지만 unit-cell reference에 한정된다.</p>'
        + waxs_table
        + '<div class="callout warn"><strong>실험 object 정의가 남아 있다.</strong> dilute single protein은 SAXS 영역에서도 intensity가 충분하지 않을 수 있으므로 oriented protein을 protein crystal, single crystal 또는 oriented nanocrystal 중 하나로 명확히 해야 한다.</div>'
    )
    pages.append(page(3, "WAXS · accuracy", "Operator 정확도와 high-q source gate", body3))

    waxs_perf_table = table(
        ["qmax", "Nq/Nφ", "ACFO first-total", "chunked FINUFFT", "speedup", "complex L2"],
        [[row["q_label"].replace("qmax ", ""), f'{row["nq"]}/{row["nphi"]}', f'{row["acfo_first_total_s"]:.3f} s', f'{row["finufft_chunked_s"]:.2f} s', f'{row["speedup"]:.1f}×', f'{row["complex_l2"]:.2e}'] for row in waxs_rows],
        ["15%", "13%", "19%", "20%", "15%", "18%"],
    )
    body4 = (
        '<p>1M repeated-crystal, full-ring, fixed-dq sweep에서 qmax 2.13→8.06 Å⁻¹로 늘면 first-total speedup이 47.0×→147.8×로 증가했다. 반대로 q range를 고정하고 radial resolution만 Nq 32/128/512로 높였을 때 hot speedup은 314.3×/70.4×/33.48×로 감소했다. 따라서 “high q”와 “dense q resolution”은 같은 효과가 아니다.</p>'
        + waxs_chart
        + waxs_perf_table
        + '<div class="two-col"><div><h2>Detector-aware headline</h2><p>EIGER2 X 4M envelope, Nq=512에서 ACFO 21.60 s, FINUFFT 42.69 s, 1.976×였고 memory ratio는 5.14×였다. 이상적인 full-ring 큰 speedup 대신 이 약 2× 결과를 현실적 headline으로 두는 편이 방어 가능하다.</p></div>'
        + '<div><h2>Curvature 해석</h2><p>물리/planar curvature speedup은 2.431×/2.552×, |qz| fraction과 speedup 상관은 −0.767이었다. high-q 이득을 curvature alone으로 설명할 수 없으며 target 수, bandwidth, reuse가 함께 작동한다.</p></div></div>'
        + '<div class="callout"><strong>현재 WAXS 판정:</strong> operator와 detector-aware 성능은 논문 근거가 있으나, realistic protein nanocrystal/MD source에서 exact-beta 또는 sub-bin contraction 검증이 필요하다.</div>'
    )
    pages.append(page(4, "WAXS · performance", "High-q 이득과 현실 detector 조건", body4))

    aidt_table = table(
        ["조건", "시간", "처리율", "H2D 포함", "경계"],
        [[row["condition"], f'{row["seconds"]*1000:.2f} ms', f'{row["hz"]:.2f} Hz', row["copy_included"], row["claim"]] for row in aidt_rows],
        ["39%", "14%", "14%", "12%", "21%"],
    )
    body5 = (
        '<div class="metrics">'
        + metric("Full public condition", "10.31 Hz", "700×700×35 GPU-resident")
        + metric("+ output statistics", "9.88 Hz", "101.23 ms")
        + metric("+ sequential H2D", "9.37 Hz", "106.70 ms", "amber")
        + metric("Cached support gain", "4.58×", "dense GPU 0.446 s → 0.0975 s", "teal")
        + '</div>'
        + '<p>Public 24×700×700 입력에서 700×700×35 출력 조건의 GPU-resident core는 97.00 ms, 10.31 Hz였다. Active support 26.4%와 cached support transfer가 핵심이며 cache 약 2076 MiB, peak allocated 약 3849 MiB였다.</p>'
        + aidt_chart
        + aidt_table
        + '<p>Measured-data proxy에서는 prepared pair 1.242 ms 대 cuFINUFFT 58.036 ms(46.7×), 8-step update 1.962 ms/iter 대 58.778 ms(30.0×)였지만 calibrated nonlinear full aIDT는 아니다. 위 10 Hz 결과와 이 proxy speedup은 합치지 않는다.</p>'
        + '<div class="callout warn"><strong>주장 문구:</strong> “GPU-resident prepared reconstruction core의 10 Hz”와 “processing-side real-time feasibility”는 지지된다. CPU preprocessing 144.5 ms, acquisition scheduling, raw-data contract가 포함된 end-to-end live microscope 10 Hz는 아직 아니다.</div>'
    )
    pages.append(page(5, "aIDT", "10 Hz급 GPU-resident 처리와 pipeline 경계", body5))

    adaptive_table = table(
        ["adaptive-L threshold", "active fraction", "pair median", "worst rel-L2", "판정"],
        [[f'{row["threshold"]:.0e}', f'{row["active_l_fraction"]*100:.2f}%', f'{row["pair_median_s"]*1000:.3f} ms', f'{row["worst_rel_l2"]:.3e}', badge("PASS" if row["passed"] else "FAIL", "pass" if row["passed"] else "fail")] for row in adaptive_rows],
        ["22%", "19%", "20%", "22%", "17%"],
    )
    body6 = (
        '<div class="metrics">'
        + metric("Final hot pair", "111.366 ms", "IQR 111.147–111.630 ms")
        + metric("cuFINUFFT hot pair", "9.0475 s", "reusable plan, ε=1e−6", "amber")
        + metric("Robust speedup", "81.24×", "30 AB/BA repeats", "teal")
        + metric("Direct worst error", "1.072e−6", "tolerance 2e−6: PASS", "teal")
        + '</div>'
        + '<p>최종 opt-in production candidate는 H=28(57 harmonics), exact axis L=0 pruning(65→1), axial SVD rank 16, adaptive-L threshold 1e−6, complex64다. Ring L pair는 8,398/16,640(50.47%)만 활성이다. 기본 exact 경로는 변경하지 않았다.</p>'
        + odt_speedup_chart
        + '<p>H36 dense structured direct reference 대비 physical forward/adjoint는 2.717e−7/2.217e−7, stress는 1.072e−6/9.537e−7, dot error는 6.983e−7로 모두 2e−6 gate를 통과했다. H36 pair 369.530 ms에서 111.239 ms로 3.322× 단축됐다.</p>'
        + adaptive_table
        + '<div class="callout"><strong>최종 선택:</strong> 1e−6은 50.47% active, 104.294 ms sweep median, worst 9.616e−7로 마지막 PASS 지점이다. 1e−5부터 정확도 gate가 실패하므로 더 빠른 수치를 production claim으로 사용하지 않는다.</div>'
    )
    pages.append(page(6, "ODT · packed operator", "정확도 gate를 유지한 111 ms 연산자", body6))

    banded_table = table(
        ["복원", "samples/view", "remap 입력오차", "object NRMSE", "remap-vs-ideal", "core", "remap", "합계/처리율"],
        [[f'z={row["z"]}', f'{row["samples_per_view"]:,}', f'{row["input_rel_pct"]:.3f}%', f'{row["object_nrmse_pct"]:.3f}%', f'{row["reconstruction_diff_pct"]:.3f}%', f'{row["core_ms"]:.3f} ms', f'{row["remap_ms"]:.3f} ms', f'{row["total_ms"]:.3f} ms / {row["hz"]:.2f} Hz'] for row in banded_rows],
        ["8%", "12%", "13%", "13%", "14%", "12%", "11%", "17%"],
    )
    body7 = (
        '<div class="metrics">'
        + metric("Banded samples/view", "28,672", "uniform 65,536 대비 56.25% 감소")
        + metric("z=1 + remap", f'{banded1["hz"]:.2f} Hz', f'{banded1["total_ms"]:.3f} ms')
        + metric("z=8 + remap", f'{banded8["hz"]:.2f} Hz', f'{banded8["total_ms"]:.3f} ms')
        + metric("Recon difference", "≈0.098%", "remap input vs ideal banded", "teal")
        + '</div>'
        + '<p>실제 입력은 121-view, 320×320 Cartesian complex camera field로 두고, cached bilinear remap을 통해 inner 96×128와 outer 64×256 band로 변환했다. detector pixel → angular sampling → ACFO forward/adjoint 순서의 processing chain이다.</p>'
        + banded_table
        + '<div class="two-col"><div><h2>왜 유의미한가</h2><p>Uniform 256×256 polar sampling보다 samples/view를 56.25% 줄이면서 120-iteration ideal-data reconstruction과 remapped reconstruction의 차이를 약 0.098%로 유지했다. z=1과 z=8 모두 remap을 포함해 10 Hz를 넘었다.</p></div>'
        + '<div><h2>무엇과 합치지 않는가</h2><p>이 geometry branch는 앞 페이지의 final H28/rank16/adaptive-L packed operator와 통합 측정하지 않았다. 따라서 12.52 Hz와 81.24×를 곱하거나 동일 pipeline의 동시 달성으로 쓰지 않는다.</p></div></div>'
        + '<div class="callout warn"><strong>제외 항목:</strong> acquisition transfer, hologram demodulation, 측정 noise는 포함되지 않았다. z=1 초기 remap 11.445 ms는 outlier가 포함된 값이므로 100-repeat hot recheck의 1.852 ms를 사용했다.</div>'
    )
    pages.append(page(7, "ODT · detector geometry", "Cartesian camera에서 banded angular sampling까지", body7))

    recon_table = table(
        ["z", "known support", "median/update", "p95", "rate", "100-update total", "final object NRMSE"],
        [[row["z"], "yes" if row["known_support"] else "no", f'{row["median_ms"]:.3f} ms', f'{row["p95_ms"]:.3f} ms', f'{row["hz"]:.2f} Hz', f'{row["hundred_update_s"]:.3f} s', f'{row["object_nrmse_pct"]:.3f}%'] for row in recon_rows if row["z"] in {256, 128, 64, 32, 8, 1}],
        ["8%", "14%", "18%", "16%", "13%", "17%", "14%"],
    )
    body8 = (
        '<div class="metrics">'
        + metric("Full 256³ update", f'{full["hz"]:.2f} Hz', f'median {full["median_ms"]:.3f} ms')
        + metric("Full p95", f'{full["p95_ms"]:.3f} ms', "10 Hz에 1.190× 추가 개선 필요", "amber")
        + metric("z=128 slab", f'{slab128["hz"]:.2f} Hz', f'median {slab128["median_ms"]:.3f} ms', "teal")
        + metric("Pixel→mode", "1.202 ms", "geometry당 cached preprocessing", "teal")
        + '</div>'
        + '<p>121 illuminations와 active detector modes 1,765,632개를 모두 사용하고, final H28/rank16/adaptive-L 1e−6 연산자로 real-subspace normal equations를 CG 100회 수행했다. Pixel q 7,929,856개에서 mode로의 preprocessing은 median 1.202 ms이며 normal-operator equivalence는 1.255e−7였다.</p>'
        + odt_rate_chart
        + recon_table
        + '<p class="small">표는 대표 z만 표시한다. z=16/4/2의 전체 수치는 위 차트와 report_data.json에 보존했다.</p>'
        + '<p>Full 256³는 118.190 ms/update(8.46 Hz), RHS+100 updates 11.904 s, final object NRMSE 18.57%, data residual 0.177%였다. Data residual 1%는 iteration 31/3.743 s, object NRMSE 20%는 iteration 56/6.697 s에 도달했지만 100회 내 10%에는 도달하지 못했다.</p>'
        + '<p class="small"><b>10 Hz 경계:</b> z≤128은 known-support 조건이다. 10 completed reconstructions/s가 아니라 10 update/s이며, synthetic noiseless self-consistent data다.</p>'
    )
    pages.append(page(8, "ODT · reconstruction", "Full-volume 8.46 Hz, known-support slab 10 Hz 이상", body8))

    projection_table = table(
        ["항목", "RTX 2070 SUPER", "RTX 5090", "비율/해석"],
        [
            ["FP32 peak", f"{rtx2070_fp32:.1f} TFLOPS", f"{rtx5090_fp32:.1f} TFLOPS", f"{compute_ratio:.2f}×"],
            ["Memory bandwidth", f"{rtx2070_bw:.0f} GB/s", f"{rtx5090_bw:.0f} GB/s", f"{bandwidth_ratio:.1f}×"],
            ["VRAM", "8 GiB", "32 GiB", "4×; larger tensor / less streaming"],
            ["Full p95 10 Hz 필요조건", f"{full_p95_ms:.3f} ms", "≤100 ms", f"단 {required_full_speedup:.3f}×"],
            ["보수적 2× delta projection", "z=1 p95 floor 고정", f"{projected_p95_ms_2x_delta:.3f} ms", f"{projected_p95_hz_2x_delta:.2f} Hz"],
        ],
        ["27%", "23%", "23%", "27%"],
    )
    update_table = table(
        ["프레임당 update 수", "100 ms/frame에 필요한 full-volume speedup", "판정"],
        [[1, f'{full_p95_ms/100:.3f}×', "RTX 5090에서 유력한 projection"], [2, f'{2*full_p95_ms/100:.3f}×', "직접 측정 필요"], [3, f'{3*full_p95_ms/100:.3f}×', "더 강한 GPU/최적화 필요"], [5, f'{5*full_p95_ms/100:.3f}×', "현 근거로 보장 불가"], [100, f'{100*full_p95_ms/100:.1f}×', "cold 100-iteration 10 Hz 불가"]],
        ["24%", "40%", "36%"],
    )
    body9 = (
        '<p>현재 full-volume p95는 119.013 ms이므로 10 Hz에 필요한 개선은 1.190×뿐이다. RTX 5090의 공식 peak spec은 RTX 2070 SUPER 대비 FP32 약 11.52×, memory bandwidth 4×, VRAM 4×다. 실제 complex64 kernel은 peak spec 비례로 확정할 수 없으므로 보수적으로 z=1 p95 73.002 ms를 고정 floor로 두고 full-volume 추가분만 2× 가속했다.</p>'
        + projection_table
        + '<div class="callout"><strong>Projection:</strong> 이 보수적 Amdahl-style 계산은 full-volume p95 96.007 ms, 10.42 Hz다. 따라서 RTX 5090에서 <b>한 프레임당 1회의 warm full-volume update</b>는 가능성이 높다. 이는 측정이 아니라 projection이다.</div>'
        + update_table
        + '<h2>실시간 ODT에서 더 중요한 실험</h2><p>연속 프레임에서는 구조가 크게 변하지 않을 수 있으므로 매 프레임 100 iterations가 필요하다는 가정은 지나치게 보수적일 수 있다. 다음 검증은 warm start sequence에서 1/2/3/5 updates per frame, 0.1–1 voxel motion, intensity 1–10%, 100–1000 frames를 sweep하고 tracking error·lag·drift를 함께 측정하는 것이다.</p>'
        + '<p class="small">Tensor Core 가속은 현재 complex64 경로에서 사용하지 않았고 projection에도 포함하지 않았다. 32 GiB VRAM은 현재 2.404 GiB peak case 자체를 자동으로 가속하지 않지만 더 큰 tensor와 streaming 감소에 유리하다.</p>'
    )
    pages.append(page(9, "ODT · real-time projection", "RTX 5090에서 warm full-volume 10 Hz 가능성", body9))

    boundary_table = table(
        ["영역", "논문에서 사용할 수 있는 문장", "반드시 붙일 조건"],
        [
            ["WAXS", "Curved-grid prepared operator는 direct NDFT 정확도와 detector-aware 약 2× local 이득을 보였다.", "Coarse high-q atomistic source는 실패; exact-beta realistic structure 검증 필요"],
            ["aIDT", "700×700×35 GPU-resident prepared core는 10.31 Hz를 달성했다.", "CPU preprocessing·acquisition·full raw-data path 제외"],
            ["ODT operator", "Final packed operator는 direct structured reference gate를 통과하고 cuFINUFFT reusable-plan pair보다 81.24× 빨랐다.", "동일 2070S local AB/BA protocol; cross-backend 차이는 exact error가 아님"],
            ["ODT reconstruction", "Full 256³는 8.46 Hz/update, known-support z≤128 slab은 10 Hz 이상이었다.", "100 completed reconstructions/s 아님; noiseless synthetic; missing cone/conditioning 존재"],
            ["ODT projection", "RTX 5090에서 1 warm update/frame의 full-volume 10 Hz가 가능할 것으로 투영된다.", "공식 peak specs 기반 projection; 직접 측정 아님"],
        ],
        ["17%", "43%", "40%"],
    )
    body10 = (
        boundary_table
        + '<h2>우선순위가 높은 다음 실험</h2><ol><li><b>ODT 통합:</b> banded Cartesian remap과 final packed operator를 한 pipeline에서 동시 측정한다.</li><li><b>ODT temporal:</b> warm-start 연속 프레임에서 updates/frame와 tracking error의 trade-off를 측정한다.</li><li><b>ODT robustness:</b> noise, demodulation, model mismatch와 covariance-aware weighting을 추가한다.</li><li><b>WAXS:</b> realistic protein nanocrystal/MD snapshot에서 exact-beta/sub-bin source gate를 닫고 independent timing을 수행한다.</li><li><b>aIDT:</b> GPU preprocessing과 pinned-memory/stream overlap을 포함한 acquisition contract를 검증한다.</li><li><b>외부 검증:</b> 독립 GPU와 가능하면 RTX 5090에서 동일 protocol을 재실행한다.</li></ol>'
        + '<h2>주요 근거 파일</h2><ul class="source-list">'
        + '<li>benchmark_results/odt_h28_rank16_adaptive_l1e-6_vs_cufinufft_abba30_final.json</li>'
        + '<li>benchmark_results/odt_final_packed_candidate_validation.json · odt_adaptive_l_packed_sweep.json</li>'
        + '<li>benchmark_results/odt_full_slab_reconstruction_claim.json</li>'
        + '<li>benchmark_results/odt_banded_cartesian_reconstruction.json · odt_banded_cartesian_remap_hot_recheck.json</li>'
        + '<li>docs/waxs_aidt_odt_progress_report_20260713_support/report_data.json 및 원 WAXS/aIDT benchmark artifacts</li>'
        + '<li>NVIDIA GeForce RTX 5090 product page; NVIDIA Blackwell/GA102 architecture documents (official links are preserved in source_notes.md)</li>'
        + '</ul>'
        + '<div class="callout"><strong>최종 판단:</strong> 현재 자료만으로도 prepared curved-grid factorization이 WAXS, aIDT, ODT에서 서로 다른 현실적 regime의 계산 병목을 줄일 수 있다는 가능성은 충분히 제시된다. 가장 큰 신규 진전은 ODT에서 정확도 gate를 유지한 111 ms pair와 full/slab update 실측까지 연결되었다는 점이다.</div>'
    )
    pages.append(page(10, "Claim boundary & next steps", "논문 문장, 남은 gate, 재현 경로", body10, last=True))

    document = (
        '<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>ACFO WAXS aIDT ODT validation progress report 2026-07-14</title>'
        f'<style>{css}</style></head><body>'
        + "".join(pages)
        + '</body></html>'
    )
    HTML_OUTPUT.write_text(document, encoding="utf-8")

    source_notes = f"""# ACFO WAXS/aIDT/ODT validation progress report source notes

## Reporting job

- Report date: 2026-07-14.
- Saved experiments included through: 2026-07-13.
- Audience: technical manuscript planning and validation review.
- Hardware boundary: NVIDIA GeForce RTX 2070 SUPER 8 GiB unless a row explicitly says projection.
- Output chain: verified benchmark JSON -> reviewed report_data.json -> static report_print.html -> PDF.
- Measured branches are kept separate: final packed operator, banded detector/remap, and full/slab reconstruction.

## Headline definitions

- Hot pair: forward plus adjoint after reusable geometry/operator setup.
- Update rate: one conjugate-gradient normal-equation iteration per second; it is not a complete reconstruction rate.
- Speedup: named baseline median divided by ACFO median under the same saved protocol.
- Complex L2: norm(candidate-reference) / norm(reference).
- Projection: model-derived estimate, not a local RTX 5090 measurement.

## WAXS and aIDT baseline

- Reviewed 2026-07-13 snapshot: `docs/waxs_aidt_odt_progress_report_20260713_support/report_data.json`.
- Its source inventory points to direct NDFT, exact-beta, detector-aware, q-sampling, curvature, public aIDT 10 Hz, and measured-adapter benchmark artifacts.
- The present report does not reinterpret or inflate those validated baseline values.

## Latest ODT sources

- Final cuFINUFFT AB/BA: `benchmark_results/odt_h28_rank16_adaptive_l1e-6_vs_cufinufft_abba30_final.json`.
- Direct structured-reference validation: `benchmark_results/odt_final_packed_candidate_validation.json`.
- Adaptive-L accuracy/speed sweep: `benchmark_results/odt_adaptive_l_packed_sweep.json`.
- Full/slab reconstruction: `benchmark_results/odt_full_slab_reconstruction_claim.json`.
- Banded detector/remap: `benchmark_results/odt_banded_cartesian_reconstruction.json` and `benchmark_results/odt_banded_cartesian_remap_hot_recheck.json`.
- Historical comparison only: `benchmark_results/odt_torch_256cubed_100pair.json` and `benchmark_results/odt_torch_256cubed_reduced_100pair.json`.

## RTX 5090 projection sources and assumptions

- NVIDIA RTX 5090 product page: https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
- NVIDIA RTX Blackwell GPU architecture PDF: https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
- NVIDIA GA102 GPU architecture PDF: https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf
- Input specs used: RTX 2070 SUPER 9.1 FP32 TFLOPS, 448 GB/s, 8 GiB; RTX 5090 104.8 FP32 TFLOPS, 1792 GB/s, 32 GiB.
- Conservative model: keep the measured z=1 p95 as an unchanged floor and scale only the extra full-volume p95 component by 2x. This gives {projected_p95_ms_2x_delta:.3f} ms or {projected_p95_hz_2x_delta:.2f} Hz.
- Tensor Core acceleration is not assumed.

## Important exclusions

- No multiplication of speedups across different experiment branches.
- Banded geometry and final packed operator have not yet been benchmarked together.
- ODT reconstruction data are synthetic, noiseless, and self-consistent; acquisition transfer and hologram demodulation are excluded.
- Known-support slab results assume no unknown object outside the selected centered z support.
- The 81.24x cuFINUFFT comparison is a same-GPU reusable-plan hot comparison; independent-machine replication remains.
- Earlier asynchronous 2-3 s ACFO comparison files are excluded from the final cuFINUFFT claim.

## Generated artifacts

- Processed values: `{REPORT_DATA.relative_to(ROOT).as_posix()}`.
- Static report source: `{HTML_OUTPUT.relative_to(ROOT).as_posix()}`.
- Generated at UTC: {generated_at}.
"""
    SOURCE_NOTES.write_text(source_notes, encoding="utf-8")

    chrome_result = subprocess.run(
        [
            str(CHROME),
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--no-pdf-header-footer",
            "--allow-file-access-from-files",
            "--virtual-time-budget=8000",
            f"--print-to-pdf={PDF_OUTPUT}",
            HTML_OUTPUT.resolve().as_uri(),
        ],
        check=True,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if not PDF_OUTPUT.exists():
        raise RuntimeError(f"Chrome returned successfully but did not create {PDF_OUTPUT}")

    print(json.dumps({
        "html": str(HTML_OUTPUT),
        "pdf": str(PDF_OUTPUT),
        "report_data": str(REPORT_DATA),
        "source_notes": str(SOURCE_NOTES),
        "pages_planned": len(pages),
        "chrome_stderr": chrome_result.stderr.strip(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
