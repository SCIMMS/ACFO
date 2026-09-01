from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile

import fitz

from build_acfo_ncs_validation_execution_plan_pdf import (
    NODE,
    PLUGIN_ROOT,
    RESULTS,
    ROOT,
    find_browser,
    json_source,
    run,
)


DOCS = ROOT / "docs"
SUPPORT = DOCS / "acfo_ncs_validation_progress_summary_support"
ARTIFACT = SUPPORT / "artifact.json"
HTML = SUPPORT / "report.html"
PDF = DOCS / "ACFO_NCS_validation_progress_summary_ko.pdf"
EDITABLE = DOCS / "acfo_ncs_validation_progress_summary_ko.md"
EXTRACTED = SUPPORT / "report_extracted_text.txt"
DESIGN_NOTES = SUPPORT / "report_design_notes.json"
MARGIN_DATA = SUPPORT / "numerical_gate_margin.json"
RECEIPT = SUPPORT / "build_receipt.json"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def source(source_id: str, label: str, filename: str, description: str) -> dict:
    return json_source(
        source_id,
        label,
        f"benchmark_results/{filename}",
        description,
    )


def build_artifact() -> dict:
    generated_at = datetime.now(timezone.utc).isoformat()
    sparse_1m = load("protein_nanocrystal_sparse_memory_1m_q6p3_nq40.json")
    fused_1m = load("physical_scaling_1m_q6p3_rdep_analytic_fused_memory.json")["rows"][0]
    w3 = load("protein_nanocrystal_w3a_gate_decision.json")
    odt128 = load("odt_128cubed_gate_decision.json")
    odt256 = load("odt_256cubed_memory_gate_decision.json")
    vector = load("uniaxial_vector_born_direct_64cubed.json")
    dispersion = load("uniaxial_meep_dispersion_highres_decision.json")
    dispersion_fine = dispersion["rows"][-1]
    phase = load("uniaxial_meep_3d_phase_gate_decision.json")
    phase_fine = phase["rows"][-1]
    amplitude = load("uniaxial_meep_3d_amplitude_gate_decision.json")
    amp_h04_r16 = next(
        row
        for row in amplitude["cases"]
        if row["source_half_width"] == 0.4 and row["resolution"] == 16.0
    )
    high_na = load("high_na_si_correspondence.json")
    high_na_risk = load("high_na_harmonic_support_risk.json")
    high_na_stress = high_na_risk["vector_charge18_stress"]
    regression = load("acfo_release_regression_2026-07-12.json")
    margin_rows = [
        {
            "check": "WAXS complex L2",
            "margin_orders": math.log10(1e-6 / w3["metrics"]["complex_l2"]),
            "current": f"L2 {w3['metrics']['complex_l2']:.2e}",
            "gate": "≤1e-6",
            "scope": "W3a 216k, Nq=256",
        },
        {
            "check": "ODT forward",
            "margin_orders": math.log10(1e-6 / odt128["metrics"]["forward_complex_l2_vs_direct_subset"]),
            "current": f"L2 {odt128['metrics']['forward_complex_l2_vs_direct_subset']:.2e}",
            "gate": "≤1e-6",
            "scope": "128-cubed direct subset",
        },
        {
            "check": "ODT adjoint",
            "margin_orders": math.log10(1e-9 / odt128["metrics"]["adjoint_selected_object_l2_vs_direct_subset"]),
            "current": f"L2 {odt128['metrics']['adjoint_selected_object_l2_vs_direct_subset']:.2e}",
            "gate": "≤1e-9",
            "scope": "128-cubed direct subset",
        },
        {
            "check": "Vector Born",
            "margin_orders": math.log10(1e-8 / vector["cases"][0]["vector_complex_l2"]),
            "current": f"L2 {vector['cases'][0]['vector_complex_l2']:.2e}",
            "gate": "≤1e-8",
            "scope": "64-cubed extraordinary branch",
        },
        {
            "check": "2-D dispersion",
            "margin_orders": math.log10(0.02 / dispersion_fine["correct_relative_l2"]),
            "current": f"{100*dispersion_fine['correct_relative_l2']:.4f}%",
            "gate": "≤2%",
            "scope": "PyMeep resolution 40",
        },
        {
            "check": "3-D phase",
            "margin_orders": math.log10(0.02 / phase_fine["calibrated_extraordinary_ellipse_relative_l2"]),
            "current": f"{100*phase_fine['calibrated_extraordinary_ellipse_relative_l2']:.4f}%",
            "gate": "≤2%",
            "scope": "PyMeep resolution 28",
        },
        {
            "check": "High-NA vector",
            "margin_orders": math.log10(1e-6 / max(row["vector_complex_l2"] for row in high_na["rows"])),
            "current": f"L2 {max(row['vector_complex_l2'] for row in high_na['rows']):.2e}",
            "gate": "≤1e-6",
            "scope": "full-support, three apertures",
        },
    ]

    sources = [
        source("sparse_1m", "1M protein true-sparse memory", "protein_nanocrystal_sparse_memory_1m_q6p3_nq40.json", "1,001,000-atom sparse representation and solve memory"),
        source("fused_1m", "1M C++ fused solve memory", "physical_scaling_1m_q6p3_rdep_analytic_fused_memory.json", "Historical 0.9 MiB incremental solve-memory record"),
        source("w3", "WAXS W3a gate decision", "protein_nanocrystal_w3a_gate_decision.json", "216k-atom Nq=256 accuracy, speed, memory and break-even gate"),
        source("odt128", "ODT 128-cubed gate decision", "odt_128cubed_gate_decision.json", "Forward, adjoint, dot-product and inverse controls"),
        source("odt256", "ODT 256-cubed memory decision", "odt_256cubed_memory_gate_decision.json", "24 GiB analytical memory feasibility and current allocation limitation"),
        source("vector", "Uniaxial vector-Born direct gate", "uniaxial_vector_born_direct_64cubed.json", "64-cubed LiNbO3 vector first-Born direct comparison"),
        source("dispersion", "PyMeep 2-D uniaxial dispersion gate", "uniaxial_meep_dispersion_highres_decision.json", "Actual LiNbO3 ellipse-versus-sphere spectral ridge"),
        source("phase", "PyMeep 3-D uniaxial phase gate", "uniaxial_meep_3d_phase_gate_decision.json", "3-D phase curvature with grid, time and boundary checks"),
        source("amplitude", "PyMeep 3-D amplitude bridge decision", "uniaxial_meep_3d_amplitude_gate_decision.json", "Finite-radius amplitude bridge and reference limitation diagnosis"),
        source("high_na", "High-NA SI correspondence", "high_na_si_correspondence.json", "Full-support scalar and vector direct-reference correspondence"),
        source("high_na_risk", "High-NA harmonic-support risk audit", "high_na_harmonic_support_risk.json", "Charge-18 post-mixing harmonic stress test"),
        source("regression", "Local regression receipt", "acfo_release_regression_2026-07-12.json", "Current local repository test-suite result"),
        json_source(
            "derived_margin",
            "Derived numerical gate margins",
            "docs/acfo_ncs_validation_progress_summary_support/numerical_gate_margin.json",
            "Favorable numerical margin normalized by each predeclared internal error threshold",
        ),
    ]

    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "body": "# ACFO NCS Validation 진행 결과 요약\n\n2026년 7월 12일 기준 · 실행 항목, 주요 수치, 판정 범위와 남은 작업",
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## 요약: 핵심 수치 기반은 확보됐고, production·inverse·독립 reference가 다음 병목이다\n\n"
                "- **WAXS:** 1M 구조의 true-sparse 경로와 W3a 정확도는 확보했다. 최소 production gate는 속도 3×가 아니라 **메모리 대체 조건으로 PASS**다.\n"
                "- **ODT:** 128³ forward·adjoint·in-range inverse는 scoped PASS다. 256³는 24 GiB 분석적 feasibility만 PASS이며 실제 streaming production은 아직이다.\n"
                "- **비선형 결정:** 64³ vector-Born, 2-D 이방성 dispersion, 3-D phase curvature는 PASS다. full complex amplitude는 ACFO 실패가 아니라 **적절한 anisotropic far-field reference 부재로 현재 판정 불가**다.\n"
                "- **High-NA:** full harmonic support에서는 scalar·vector 모두 machine-precision 수준이다. 다만 고차 vector adaptive cutoff는 post-mixing mode 전체를 보존해야 한다.\n"
                "- **재현성:** 현재 로컬 회귀시험 162/162는 PASS지만 독립 머신, DOI archive, publication-grade 반복 통계가 남아 있다."
            ),
        },
        {
            "id": "status_matrix",
            "type": "markdown",
            "body": (
                "## 지금까지 수행한 항목과 현재 판정\n\n"
                "- **WAXS W3a — PASS via memory alternative:** sparse memory, accuracy와 FINUFFT 비교를 수행했다. Production-sized sparse resource advantage까지 주장할 수 있다.\n"
                "- **ODT 128³ — Scoped PASS:** direct subset, full adjoint와 inverse control을 수행했다. Numerical operator와 in-range inverse 범위다.\n"
                "- **ODT 256³ — Feasibility PASS:** live-array memory를 감사했다. 24 GiB 내 이론적 decomposition만 확인됐다.\n"
                "- **Vector Born — PASS:** 64³ LiNbO3 tensor·편광 direct sum을 수행했다. First-Born vector algebra 범위다.\n"
                "- **PyMeep dispersion — PASS:** 2-D ellipse와 forced sphere를 비교해 실제 이방성 dispersion을 구별했다.\n"
                "- **PyMeep 3-D phase — PASS:** phase slope와 grid/time/boundary 안정성을 확인했다. 3-D phase curvature 범위다.\n"
                "- **PyMeep 3-D amplitude — Reference-limited:** finite-radius field bridge로는 방법 실패를 판정할 수 없으며 validation도 미완료다.\n"
                "- **High-NA SI — PASS:** scalar/vector direct field의 full-support correspondence를 확인했다.\n"
                "- **High-order harmonic — 보강 필요:** current charge-6 SI는 안전하지만 고차 vector fast path의 cutoff rule은 수정이 필요하다."
            ),
        },
        {
            "id": "margin_intro",
            "type": "markdown",
            "sourceId": "derived_margin",
            "body": (
                "## 통과한 numerical gate는 대부분 threshold 안쪽에 충분한 여유가 있다\n\n"
                "아래 값은 각 error ceiling에 대해 `log10(gate/current)`로 정규화한 favorable margin이다. 0보다 크면 해당 내부 threshold를 통과한다. 단위가 다른 수치를 raw scale로 비교하지 않으며, production timing·memory와 reference-limited amplitude는 이 그림에서 제외했다."
            ),
        },
        {
            "id": "margin_chart",
            "type": "chart",
            "chartId": "numerical_margin",
            "layout": "full",
        },
        {
            "id": "margin_note",
            "type": "markdown",
            "body": (
                "### 이 여유는 수치 구현의 기반을 의미하며 publication readiness 전체를 뜻하지 않는다\n\n"
                "WAXS speed, ODT 256³ production, physical inverse와 independent Maxwell amplitude reference는 서로 다른 gate다. 따라서 큰 numerical margin을 그 미완료 항목의 대체 증거로 사용하지 않는다."
            ),
        },
        {
            "id": "waxs_sparse",
            "type": "markdown",
            "sourceId": "sparse_1m",
            "body": (
                "## WAXS: 1M 구조를 dense grid 없이 처리하는 sparse 경로를 확인했다\n\n"
                f"- **실제 구조:** `{sparse_1m['atoms']:,}` atoms, grid `{sparse_1m['n_r']}×{sparse_1m['n_z']}×{sparse_1m['n_phi']}`, qmax `{sparse_1m['qmax']}` Å⁻¹.\n"
                f"- **표현 메모리:** sparse arrays `{sparse_1m['sparse_structure_storage_mib']:.2f} MiB`; dense float32 equivalent `{sparse_1m['dense_hist_float32_gib']:.2f} GiB`.\n"
                f"- **점유율:** active flat-bin fraction `{100*sparse_1m['active_flat_fraction']:.4f}%`; representation build peak delta `{sparse_1m['representation_build_peak_rss_delta_mib']:.2f} MiB`.\n"
                f"- **실행:** first solve `{sparse_1m['first_solve_s']:.2f} s`; cached median `{sparse_1m['cached_median_s']:.2f} s`; cached incremental peak `{sparse_1m['cached_solve_peak_rss_delta_mib']:.2f} MiB`.\n\n"
                "빈 공간이 큰 분자구조에서 sparse representation이 효과적이라는 기존 판단을 실제 1M protein nanocrystal 입력으로 재확인했다."
            ),
        },
        {
            "id": "waxs_fused",
            "type": "markdown",
            "sourceId": "fused_1m",
            "body": (
                "### 과거의 0.9 MiB 기록은 representation 전체가 아니라 fused solve의 추가 working set이다\n\n"
                f"C++ fused 경로의 solve-call peak RSS delta는 `{fused_1m['r_dependent_cake_first_peak_rss_delta_mib']:.4f} MiB`였지만, 같은 시점의 전체 process peak는 `{fused_1m['r_dependent_cake_first_peak_rss_mib']:.1f} MiB`였다. "
                "따라서 0.9 MiB와 현재 sparse object arrays 34.37 MiB는 서로 다른 측정량이다."
            ),
        },
        {
            "id": "waxs_gate",
            "type": "markdown",
            "sourceId": "w3",
            "body": (
                "### W3a 최소 production gate는 정확도와 메모리 조건으로 통과했다\n\n"
                f"- **Regime:** `{w3['regime']['atoms']:,}` atoms, Nq `{w3['regime']['nq']}`, Nφ `{w3['regime']['n_phi']}`, qmax `{w3['regime']['qmax']}` Å⁻¹.\n"
                f"- **정확도:** complex L2 `{w3['metrics']['complex_l2']:.2e}`; ring-relative median/p99 `{w3['metrics']['ring_pointwise_relative_median']:.2e}` / `{w3['metrics']['ring_pointwise_relative_p99']:.2e}`.\n"
                f"- **속도:** T=1 total `{w3['metrics']['t1_total_speedup']:.2f}×`; modeled T=100 `{w3['metrics']['t100_total_speedup_modeled']:.2f}×` — 사전 기준 3×에는 미달.\n"
                f"- **메모리:** whole-process `{w3['metrics']['whole_process_peak_memory_ratio_finufft_over_acfo']:.2f}×`, incremental `{w3['metrics']['incremental_peak_memory_ratio_finufft_over_acfo']:.1f}×` 감소; break-even T=`{w3['metrics']['break_even_repeat']}`.\n\n"
                "따라서 W3a는 memory alternative로 PASS다. 논문용 10 warm-up·30 repeat, 실제 T=10/100, Nq=512와 GPU/독립 머신 비교는 별도다."
            ),
        },
        {
            "id": "odt128",
            "type": "markdown",
            "sourceId": "odt128",
            "body": (
                "## ODT 128³: operator 정확성은 통과했지만 물리 phantom inverse는 아직 아니다\n\n"
                f"- **규모:** `128³`, `{odt128['problem']['illumination_count']}` illuminations, detector `128²`, `{odt128['problem']['total_q_samples']:,}` q samples.\n"
                f"- **독립 direct subset:** forward L2 `{odt128['metrics']['forward_complex_l2_vs_direct_subset']:.2e}`; selected-object adjoint L2 `{odt128['metrics']['adjoint_selected_object_l2_vs_direct_subset']:.2e}`.\n"
                f"- **Full adjoint:** dot-product error `{odt128['metrics']['full_forward_adjoint_dot_error']:.2e}`.\n"
                f"- **In-range inverse:** NRMSE `{100*odt128['metrics']['in_range_inverse_nrmse']:.3f}%`, data residual `{100*odt128['metrics']['in_range_inverse_data_residual']:.3f}%`, iteration `{odt128['metrics']['in_range_inverse_converged_iteration']}`.\n"
                f"- **식별성 경고:** unconstrained beads phantom은 100 CG 후 object NRMSE `{100*odt128['metrics']['beads_inverse_nrmse_after_100_cg']:.1f}%`로 실패했다.\n\n"
                "따라서 128³는 numerical operator와 in-range inverse control 범위에서만 scoped PASS다. missing-cone regularization과 30 dB robustness가 필요하다."
            ),
        },
        {
            "id": "odt256",
            "type": "markdown",
            "sourceId": "odt256",
            "body": (
                "## ODT 256³: 24 GiB feasibility는 보였지만 production backend는 아직 없다\n\n"
                f"- **규모:** `256³`, `{odt256['problem']['total_illumination_count']}` illuminations, detector `256²`, `{odt256['problem']['total_q_samples']:,}` q samples.\n"
                f"- **Unchunked 하한:** forward `{odt256['metrics']['unchunked_forward_live_array_lower_bound_mib']:.1f} MiB`; adjoint `{odt256['metrics']['unchunked_adjoint_live_array_lower_bound_mib']:.1f} MiB`.\n"
                f"- **Illumination block=1 후보:** forward `{odt256['metrics']['illumination_block_1_forward_live_array_lower_bound_mib']:.1f} MiB`; adjoint `{odt256['metrics']['illumination_block_1_adjoint_live_array_lower_bound_mib']:.1f} MiB`.\n"
                f"- **현재 병목:** native prepared table allocation `{odt256['metrics']['native_prepared_adjoint_failed_allocation_mib']/1024:.2f} GiB`.\n\n"
                "이는 분석적 memory feasibility PASS일 뿐이다. prepared-table-free construction, illumination streaming, 실제 peak와 100 forward-adjoint timing이 남아 있다."
            ),
        },
        {
            "id": "vector_born",
            "type": "markdown",
            "sourceId": "vector",
            "body": (
                "## 비선형 결정: 64³ vector first-Born algebra는 direct sum과 일치한다\n\n"
                f"- **Extraordinary branch:** vector complex L2 `{vector['cases'][0]['vector_complex_l2']:.2e}`; `{vector['cases'][0]['q_samples']}` q samples.\n"
                f"- **Ordinary control:** vector complex L2 `{vector['cases'][1]['vector_complex_l2']:.2e}`.\n"
                f"- **입력 규모:** `{vector['cases'][0]['active_cartesian_voxels']:,}` active Cartesian voxels, `{vector['cases'][0]['nonzero_cylindrical_bins']:,}` nonzero cylindrical bins.\n\n"
                "χ(2) tensor contraction과 ordinary/extraordinary eigenpolarization projection을 포함한 first-Born 수치 구현은 PASS다. 이는 pump depletion, interface 또는 full Maxwell propagation 검증은 아니다."
            ),
        },
        {
            "id": "dispersion",
            "type": "markdown",
            "sourceId": "dispersion",
            "body": (
                "## PyMeep 2-D dispersion: actual LiNbO3 ellipse를 forced sphere와 구별했다\n\n"
                f"- **Finest resolution:** `{dispersion_fine['resolution']:.0f}`; correct ellipse radial L2 `{100*dispersion_fine['correct_relative_l2']:.4f}%`.\n"
                f"- **Grid 변화:** resolution 32→40 L2 `{100*dispersion['finest_next_finest_peak_l2']:.4f}%`.\n"
                f"- **Negative control:** forced/correct error ratio `{dispersion_fine['forced_to_correct_error_ratio']:.3f}`; ridge-energy ratio `{dispersion_fine['correct_to_sphere_ridge_energy_ratio']:.3f}`; 최대 separation `{dispersion_fine['maximum_correct_sphere_separation_bins']:.3f}` bins.\n\n"
                "실제 단축 이방성 Maxwell spectrum이 spherical approximation과 구별된다는 reduced physics evidence다."
            ),
        },
        {
            "id": "phase",
            "type": "markdown",
            "sourceId": "phase",
            "body": (
                "## PyMeep 3-D phase curvature: grid·time·boundary 검사를 포함해 통과했다\n\n"
                f"- **Extraordinary ellipse:** calibrated phase-slope L2 `{100*phase_fine['calibrated_extraordinary_ellipse_relative_l2']:.4f}%`.\n"
                f"- **Forced sphere:** sphere/ellipse error ratio `{phase_fine['calibrated_sphere_to_ellipse_error_ratio']:.3f}`.\n"
                f"- **Ordinary control:** L2 `{100*phase_fine['calibrated_ordinary_sphere_relative_l2']:.4f}%`.\n"
                f"- **민감도:** resolution 24→28 `{100*phase['finest_next_finest_calibrated_grid_l2']:.4f}%`; after-source time `{phase['time_sensitivity']['calibrated_phase_l2']:.2e}`; cell/PML boundary `{phase['boundary_sensitivity']['calibrated_phase_l2']:.2e}`.\n\n"
                "actual-uniaxial 3-D Maxwell field의 phase curvature가 ACFO ellipse와 일치하고 forced sphere와 구별된다는 범위에서 PASS다."
            ),
        },
        {
            "id": "amplitude",
            "type": "markdown",
            "sourceId": "amplitude",
            "body": (
                "## Full complex amplitude는 방법 실패가 아니라 reference-limited 상태다\n\n"
                f"- 현재 finite-radius bridge의 half-width `0.4`, resolution `16` 결과는 extraordinary L2 `{100*amp_h04_r16['extraordinary_complex_l2']:.2f}%`, NCC `{amp_h04_r16['intensity_ncc']:.3f}`, peak error `{amp_h04_r16['peak_error_deg']:.0f}°`, forced-sphere ratio `{amp_h04_r16['forced_sphere_wrong_to_correct_ratio']:.3f}`였다.\n"
                f"- 그러나 같은 case의 ACFO factorization–direct-binned L2는 `{amp_h04_r16['algorithm_complex_l2']:.2e}`이므로 이 mismatch는 ACFO 알고리즘 오차로 볼 수 없다.\n"
                f"- 소스 half-width `0.8→0.25`에서 ordinary calibration L2가 `{100*amplitude['diagnostics']['source_shrink_h08_to_h025_at_r12']['ordinary_calibration_l2_before']:.2f}%→{100*amplitude['diagnostics']['source_shrink_h08_to_h025_at_r12']['ordinary_calibration_l2_after']:.2f}%`, radial scatter가 `{100*amplitude['diagnostics']['source_shrink_h08_to_h025_at_r12']['ordinary_radial_scatter_before']:.2f}%→{100*amplitude['diagnostics']['source_shrink_h08_to_h025_at_r12']['ordinary_radial_scatter_after']:.2f}%`로 감소했다. 이는 finite-distance near-field contamination을 지지한다.\n"
                "- Meep near-to-far는 homogeneous isotropic epsilon/mu surface를 요구하므로 actual uniaxial background에서 정당한 far-field reference를 만들 수 없다.\n\n"
                "**판정:** full amplitude가 틀렸다고 결론낼 수 없고, 정확하다고 검증할 수도 없다. 현재 claim은 3-D phase curvature까지로 제한하고, amplitude/NCC/peak는 independent anisotropic Green-tensor 또는 asymptotic-field reference가 준비될 때 재개한다."
            ),
        },
        {
            "id": "high_na",
            "type": "markdown",
            "sourceId": "high_na",
            "body": (
                "## High-NA full-support SI correspondence는 machine precision 수준이다\n\n"
                + "\n".join(
                    f"- **sinθmax={row['sin_theta_max']:.2f}:** scalar complex L2 `{row['scalar_complex_l2']:.2e}`; vector complex L2 `{row['vector_complex_l2']:.2e}`."
                    for row in high_na["rows"]
                )
                + "\n\nFull `|h|≤48` support를 사용한 Debye-Wolf/Richards-Wolf direct-reference 비교는 모든 aperture에서 PASS다."
            ),
        },
        {
            "id": "high_na_risk",
            "type": "markdown",
            "sourceId": "high_na_risk",
            "body": (
                "### 고차 harmonic이 포함되면 현재 rho-adaptive cutoff는 보강이 필요하다\n\n"
                f"Charge-18 stress에서 geometric-only L2 `{high_na_stress['variants']['geometric_only']['complex_l2']:.3f}`, raw-Jones adaptive `{high_na_stress['variants']['adaptive_raw_jones']['complex_l2']:.3f}`, current effective adaptive `{high_na_stress['variants']['adaptive_effective_vector']['complex_l2']:.3f}`였다. "
                f"Post-Richards-Wolf effective modes `16–20`을 모두 보존하면 L2 `{high_na_stress['variants']['manual_all_effective_modes']['complex_l2']:.2e}`로 회복되며 work는 `{high_na_stress['variants']['adaptive_effective_vector']['mode_rho_work']}→{high_na_stress['variants']['manual_all_effective_modes']['mode_rho_work']}`로 약 10.5% 증가한다.\n\n"
                "따라서 현재 charge-6/full-support SI 결론은 유지되지만, high-order vector fast path는 전역 cutoff 밖 mode뿐 아니라 local cutoff 안의 significant post-mixing modes 전체를 보존해야 한다."
            ),
        },
        {
            "id": "methods",
            "type": "markdown",
            "body": (
                "## 무엇을 어떻게 검증했는가\n\n"
                "- WAXS는 동일 binned object와 accuracy setting에서 ACFO와 FINUFFT를 비교하고, 속도와 whole-process/incremental memory를 분리했다.\n"
                "- ODT는 full structured operator와 독립 Cartesian exponent subset, forward-adjoint dot test, CG inverse control을 분리했다.\n"
                "- 비선형 결정은 LiNbO3 tensor·편광 projection direct sum, 2-D Maxwell dispersion, 3-D phase slope를 단계적으로 연결했다.\n"
                "- PyMeep amplitude는 patterned/point-source ratio와 ordinary global calibration을 사용했으며 per-angle fitting을 금지했다.\n"
                "- High-NA는 scalar와 vector field를 동일-node direct Debye-Wolf/Richards-Wolf reference에 비교하고 고차 harmonic stress를 별도로 수행했다.\n\n"
                "모든 수치는 현재 로컬 prototype의 내부 go/no-go 기준이며 Nature Computational Science의 공식 합격선을 의미하지 않는다."
            ),
        },
        {
            "id": "regression",
            "type": "markdown",
            "sourceId": "regression",
            "body": (
                "## 현재 구현은 로컬 회귀시험을 모두 통과했다\n\n"
                f"`{regression['passed_tests']}/{regression['collected_tests']}` tests PASS, failure `{regression['failed_tests']}`, duration `{regression['duration_s']:.2f} s`, Python `{regression['python']}`. "
                "이는 현재 worktree의 local regression일 뿐 fresh install, 독립 머신, archive snapshot을 대체하지 않는다."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## 다음 우선순위\n\n"
                "1. **WAXS publication benchmark:** Nq=512, 실제 T=10/100, 10 warm-up·30 repeat, GPU와 독립 머신 비교를 완료한다.\n"
                "2. **ODT production:** prepared-table-free 256³ construction과 illumination streaming을 구현하고 실제 GPU peak 및 100 forward-adjoint를 측정한다.\n"
                "3. **ODT inverse physics:** missing-cone regularization, beads phantom, 30 dB noise robustness를 고정 geometry에서 검증한다.\n"
                "4. **비선형 amplitude reference:** anisotropic Green-tensor 또는 충분히 큰 domain의 asymptotic field를 제공하는 독립 reference를 확보한다.\n"
                "5. **High-order vector cutoff:** post-mixing significant modes 전체를 rho-local cutoff에 반영한 뒤 charge sweep을 회귀시험에 추가한다.\n"
                "6. **Release:** fresh install, one-command figures, DOI snapshot과 독립 머신 rerun을 완료한다.\n\n"
                "### 남은 의사결정 질문\n\n"
                "- NCS main claim을 WAXS+ODT로 고정하고 nonlinear phase evidence를 SI에 둘 것인가?\n"
                "- 256³ ODT streaming 구현을 제출 전 필수 gate로 둘 것인가, 후속 production milestone로 분리할 것인가?\n"
                "- full-amplitude reference를 외부 Maxwell solver로 확보할지, 자체 anisotropic Green-tensor reference를 구축할지 결정해야 한다."
            ),
        },
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "ACFO NCS Validation 진행 결과 요약",
            "description": "현재까지 실행한 WAXS, ODT, nonlinear crystal, PyMeep, High-NA validation의 주요 수치와 claim boundary",
            "generatedAt": generated_at,
            "cards": [],
            "charts": [
                {
                    "id": "numerical_margin",
                    "title": "통과한 numerical gate의 favorable margin",
                    "subtitle": "log10(gate/current); 0보다 크면 내부 threshold 안쪽. Production 및 reference-limited 항목 제외",
                    "type": "bar",
                    "dataset": "numerical_margin",
                    "sourceId": "derived_margin",
                    "encodings": {
                        "x": {"field": "check", "type": "nominal", "label": "Validation check"},
                        "y": {"field": "margin_orders", "type": "quantitative", "label": "Margin (orders of magnitude)"},
                        "tooltip": [
                            {"field": "current", "type": "nominal", "label": "Current"},
                            {"field": "gate", "type": "nominal", "label": "Internal gate"},
                            {"field": "scope", "type": "nominal", "label": "Scope"},
                        ],
                    },
                    "yAxisTitle": "log10 favorable margin",
                }
            ],
            "tables": [],
            "sources": [
                {"id": item["id"], "label": item["label"], "path": item["path"]}
                for item in sources
            ],
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {"numerical_margin": margin_rows},
        },
        "sources": sources,
    }


def editable_markdown(artifact: dict) -> str:
    return "\n\n".join(
        block["body"]
        for block in artifact["manifest"]["blocks"]
        if block["type"] == "markdown"
    ) + "\n"


def verify_pdf() -> dict:
    document = fitz.open(PDF)
    texts = [page.get_text() for page in document]
    combined = "\n".join(texts)
    anchors = [
        "핵심 수치 기반은 확보됐고",
        "1M 구조의 true-sparse 경로",
        "ODT 128³",
        "reference-limited 상태",
        "High-NA full-support",
        "다음 우선순위",
    ]
    missing = [anchor for anchor in anchors if anchor not in combined]
    if missing:
        raise RuntimeError(f"missing PDF anchors: {missing}")
    nonblank = [index + 1 for index, text in enumerate(texts) if text.strip()]
    if len(nonblank) != len(document):
        raise RuntimeError("blank PDF page detected")
    EXTRACTED.write_text(combined, encoding="utf-8")
    preview_dir = SUPPORT / f"previews_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    preview_dir.mkdir(parents=True, exist_ok=True)
    previews: list[str] = []
    for index, page in enumerate(document):
        path = preview_dir / f"page_{index + 1:02d}.png"
        page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False).save(path)
        previews.append(str(path.relative_to(ROOT)))
    return {
        "pages": len(document),
        "size_bytes": PDF.stat().st_size,
        "text_characters": len(combined),
        "searchable_text": len(combined.strip()) > 1000,
        "nonblank_pages": nonblank,
        "anchors_verified": anchors,
        "page_text_characters": [len(text) for text in texts],
        "previews": previews,
        "metadata": document.metadata,
    }


def main() -> None:
    SUPPORT.mkdir(parents=True, exist_ok=True)
    artifact = build_artifact()
    ARTIFACT.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    MARGIN_DATA.write_text(
        json.dumps(
            artifact["snapshot"]["datasets"]["numerical_margin"],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    EDITABLE.write_text(editable_markdown(artifact), encoding="utf-8")
    DESIGN_NOTES.write_text(
        json.dumps(
            {
                "audience": "technical",
                "delivery_mode": "html-to-pdf",
                "required_structure_mapping": {
                    "technical_summary": "technical_summary",
                    "key_findings": ["status_matrix", "waxs_sparse", "odt128", "vector_born", "high_na"],
                    "scope_and_definitions": "methods",
                    "methodology": "methods",
                    "limitations_and_robustness": ["odt256", "amplitude", "high_na_risk", "regression"],
                    "recommended_next_steps": "next_steps",
                    "further_questions": "next_steps",
                },
                "chart_map": {
                    "numerical_margin": {
                        "question": "How far inside each predeclared numerical error threshold are the passed checks?",
                        "family": "comparison",
                        "type": "bar",
                        "fields": ["check", "margin_orders", "current", "gate", "scope"],
                        "takeaway": "Passed numerical checks are inside their own thresholds, but production and reference-limited gates remain separate.",
                        "palette_policy": "single-root preferred",
                    }
                },
                "chart_omission_reason": "No raw cross-domain metric chart is used because validation axes have heterogeneous units and denominators. The only chart normalizes passed error metrics by their own declared thresholds; exact values and claim boundaries remain in tables and narrative.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    builder = PLUGIN_ROOT / "skills" / "build-report" / "scripts" / "deliver_portable_artifact.mjs"
    delivery = run(
        [str(NODE), str(builder), "--input", str(ARTIFACT), "--output", str(HTML)],
        cwd=PLUGIN_ROOT,
    )
    browser = find_browser()
    if os.environ.get("ACFO_SKIP_PDF_PRINT") == "1":
        if not PDF.exists():
            raise RuntimeError("ACFO_SKIP_PDF_PRINT=1 but PDF does not exist")
        browser_stdout = "skipped; existing PDF verified"
    else:
        with tempfile.TemporaryDirectory(prefix="acfo_summary_chrome_") as profile:
            result = run(
                [
                    str(browser),
                    "--headless=new",
                    "--disable-gpu",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--no-pdf-header-footer",
                    "--allow-file-access-from-files",
                    f"--user-data-dir={profile}",
                    "--virtual-time-budget=10000",
                    f"--print-to-pdf={PDF}",
                    HTML.resolve().as_uri(),
                ]
            )
        browser_stdout = result.stdout.strip()
    verification = verify_pdf()
    receipt = {
        "schema": "acfo-ncs-validation-progress-summary-pdf-build-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "node": str(NODE),
            "browser": str(browser),
        },
        "artifact": str(ARTIFACT.relative_to(ROOT)),
        "editable_source": str(EDITABLE.relative_to(ROOT)),
        "html": str(HTML.relative_to(ROOT)),
        "pdf": str(PDF.relative_to(ROOT)),
        "pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
        "portable_delivery_stdout": delivery.stdout.strip(),
        "browser_stdout": browser_stdout,
        "verification": verification,
    }
    RECEIPT.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
