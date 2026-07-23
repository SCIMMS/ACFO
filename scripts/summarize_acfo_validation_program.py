from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


STAGE_FILES = {
    2: "acfo_stage2_discrete_validation.json",
    3: "acfo_stage3_gaussian_validation.json",
    4: "acfo_stage4_adjoint_validation.json",
    5: "acfo_stage5_harmonic_cutoff.json",
    6: "acfo_stage6_crossover.json",
    7: "acfo_stage7_calibration.json",
    8: "acfo_stage8_anisotropic_born.json",
    9: "acfo_stage9_inverse_reconstruction.json",
    10: "acfo_stage10_odt_double_manifold.json",
}


def load_payloads() -> dict[int, dict[str, object]]:
    payloads = {}
    for stage, filename in STAGE_FILES.items():
        path = ROOT / "benchmark_results" / filename
        if not path.exists():
            raise FileNotFoundError(path)
        payloads[stage] = json.loads(path.read_text(encoding="utf-8"))
    return payloads


def build_summary(payloads: dict[int, dict[str, object]]) -> dict[str, object]:
    stage2 = payloads[2]
    stage3 = payloads[3]
    stage4 = payloads[4]
    stage5 = payloads[5]
    stage6 = payloads[6]
    stage7 = payloads[7]
    stage8 = payloads[8]
    stage9 = payloads[9]
    stage10 = payloads[10]

    stage2_max = max(
        result["complex_error"]["relative_l2"]
        for result in stage2["results"].values()
    )
    stage3_operator_max = max(
        family["operator"]["relative_l2"]
        for level in stage3["levels"]
        for family in level["families"].values()
    )
    stage3_finest_total = max(
        family["total"]["relative_l2"]
        for family in stage3["levels"][-1]["families"].values()
    )
    stage4_dot_max = max(
        max(
            result["euclidean_dot_product_error"],
            result["weighted_dot_product_error"],
        )
        for result in stage4["results"].values()
    )
    stage5_fit = stage5["summary"]["transition_fits"]["1e-08"]
    dense_speedups = [
        row["speedups"]["finufft_execute_over_acfo_apply"]
        for row in stage6["n_phi_sweep"]
    ]
    sparse_crossover = next(
        row
        for row in stage6["sparse_source_target_control"]
        if row["warm_fastest_method"] == "acfo_full_grid"
    )
    ellipsoid_recovery = stage7["ellipsoid"]["recoveries"]
    spline_recovery = stage7["anchored_bspline"]["recoveries"]
    pde_extra = stage8["branches"]["extraordinary"]
    wrong_pde = stage8["wrong_curvature_control"]
    inverse_correct = stage9["results"]["acfo_correct"]
    inverse_wrong = stage9["results"]["acfo_wrong_sphere"]
    double_combinations = stage10["combinations"]
    double_inverse = stage10["inverse_reconstruction"]

    stage1_passed = (ROOT / "docs" / "axisymmetric_manifold_contract.md").exists()
    stage_status = {
        "1": {
            "name": "axisymmetric manifold contract",
            "passed": stage1_passed,
            "artifact": "docs/axisymmetric_manifold_contract.md",
        }
    }
    for stage, payload in payloads.items():
        stage_status[str(stage)] = {
            "name": payload["schema"],
            "passed": bool(payload["passed"]),
            "artifact": f"benchmark_results/{STAGE_FILES[stage]}",
        }

    key_results = {
        "discrete_arbitrary_curvature_max_relative_l2": stage2_max,
        "analytic_gaussian_operator_max_relative_l2": stage3_operator_max,
        "analytic_gaussian_finest_total_relative_l2_max": stage3_finest_total,
        "forward_adjoint_dot_error_max": stage4_dot_max,
        "cutoff_fit_at_1e-8": stage5_fit,
        "dense_single_thread_finufft_over_acfo_warm_speedup_range": [
            min(dense_speedups),
            max(dense_speedups),
        ],
        "sparse_10_source_acfo_crossover_selected_targets": sparse_crossover[
            "selected_targets"
        ],
        "ellipsoid_1pct_noise_parameter_relative_l2": ellipsoid_recovery[
            "noise_1e-02"
        ]["parameter_relative_l2"],
        "spline_1pct_noise_curve_relative_l2": spline_recovery["noise_1e-02"][
            "curve_relative_l2"
        ],
        "extraordinary_acfo_vs_scalar_pde_relative_l2": pde_extra["acfo_vs_pde"][
            "relative_l2"
        ],
        "extraordinary_wrong_to_correct_pde_residual_ratio": wrong_pde[
            "wrong_to_correct_residual_ratio"
        ],
        "single_manifold_known_support_object_relative_l2": inverse_correct[
            "reconstruction_relative_l2"
        ],
        "single_manifold_wrong_sphere_object_relative_l2": inverse_wrong[
            "reconstruction_relative_l2"
        ],
        "single_manifold_unrestricted_condition_number": stage9[
            "identifiability_control"
        ]["condition_number"],
        "double_manifold_forward_relative_l2_max": max(
            result["direct_forward_relative_l2"]
            for result in double_combinations.values()
        ),
        "double_manifold_adjoint_relative_l2_max": max(
            result["direct_adjoint_relative_l2"]
            for result in double_combinations.values()
        ),
        "double_manifold_unrestricted_object_relative_l2": double_inverse["structured"][
            "reconstruction_relative_l2"
        ],
        "double_manifold_structured_direct_reconstruction_relative_l2": double_inverse[
            "structured_vs_direct_reconstruction_relative_l2"
        ],
    }
    overall_passed = all(stage["passed"] for stage in stage_status.values())
    return {
        "schema": "acfo-validation-program-summary-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_passed": overall_passed,
        "source_plan": "ACFO_general_Ewald_curvature_validation_plan_ko.pdf",
        "stage_status": stage_status,
        "key_results": key_results,
        "defensible_claims": [
            "The circular-harmonic factorization extends from an Ewald sphere to arbitrary finite axisymmetric tabulated or analytic meridional curves.",
            "The prepared single-manifold forward and adjoint match independent Cartesian exponent sums and type-3 NUFFT references in the tested discrete regime.",
            "Angular cutoff is primarily controlled by Q_perp,max times active object radius, with a tolerance-dependent Bessel transition margin.",
            "Prepared ACFO is advantageous for dense azimuthal grids and repeated geometry/object use on the measured local single-thread CPU regime, with an explicit sparse crossover boundary.",
            "Known-phantom curvature calibration is differentiable and recovers ellipsoid and anchored spline geometry under the tested noise levels.",
            "A scalar anisotropic first-Born reference supports the physical mapping from spherical and ellipsoidal dispersion branches to ACFO sampling curves.",
            "A double-manifold ODT operator with q=Gamma_out-Gamma_in has a correct truncated double-harmonic forward-adjoint pair and supports a small unrestricted discrete reconstruction.",
        ],
        "explicit_nonclaims_and_gaps": [
            "No experimental instrument data are validated in this program.",
            "The independent physical validation is scalar first Born, not vector Maxwell, multiple scattering, FDTD, or FEM.",
            "Single-manifold unconstrained radial-axial tomography is ill-conditioned; the successful stage-9 reconstruction uses known sparse support.",
            "The stage-10 cached double-mode-pair kernel is a small-problem correctness prototype, not a memory-scalable production implementation or speed claim.",
            "The local CPU crossover results do not establish universal superiority over NUFFT, direct sums, or Cartesian FFT interpolation.",
            "Large-Qz mixed-precision stress tests and a dedicated sphere-specialized overhead comparison remain outside the completed package.",
        ],
    }


def render_markdown(summary: dict[str, object]) -> str:
    metrics = summary["key_results"]
    stages = summary["stage_status"]
    lines = [
        "# ACFO general Ewald curvature validation: 전체 정리",
        "",
        f"전체 판정: **{'PASS' if summary['overall_passed'] else 'FAIL'}**",
        "",
        "이 패키지는 기존 curved-Ewald-sphere 계산을 임의의 축대칭 meridional curvature로 일반화하고, 수학적 정확성, 독립 reference, cutoff 법칙, 성능 crossover, curvature calibration, scalar anisotropic-wave mapping, inverse readiness, ODT double-manifold 확장을 순차적으로 검증한다.",
        "",
        "## 단계별 상태",
        "",
        "| 단계 | 내용 | 판정 | 근거 artifact |",
        "|---:|---|---|---|",
    ]
    for stage, status in stages.items():
        lines.append(
            f"| {stage} | {status['name']} | {'PASS' if status['passed'] else 'FAIL'} | `{status['artifact']}` |"
        )
    lines.extend(
        [
            "",
            "## 핵심 수치",
            "",
            f"- arbitrary-curvature discrete forward 최대 relative L2: `{metrics['discrete_arbitrary_curvature_max_relative_l2']:.3e}`",
            f"- Gaussian analytic benchmark의 순수 operator 최대 relative L2: `{metrics['analytic_gaussian_operator_max_relative_l2']:.3e}`",
            f"- forward-adjoint dot-product 최대 오차: `{metrics['forward_adjoint_dot_error_max']:.3e}`",
            f"- 1e-8 cutoff fit: `H = x + {metrics['cutoff_fit_at_1e-8']['a']:.3f} x^(1/3) + {metrics['cutoff_fit_at_1e-8']['b']:.3f}`, R2=`{metrics['cutoff_fit_at_1e-8']['r_squared']:.4f}`",
            f"- dense local single-thread warm speedup, FINUFFT/ACFO: `{metrics['dense_single_thread_finufft_over_acfo_warm_speedup_range'][0]:.2f}–{metrics['dense_single_thread_finufft_over_acfo_warm_speedup_range'][1]:.2f}x`",
            f"- 10-source sparse control에서 ACFO crossover: `{metrics['sparse_10_source_acfo_crossover_selected_targets']}` selected targets",
            f"- 1% noise curvature error: ellipsoid `{metrics['ellipsoid_1pct_noise_parameter_relative_l2']:.3e}`, spline curve `{metrics['spline_1pct_noise_curve_relative_l2']:.3e}`",
            f"- extraordinary scalar-PDE 대비 ACFO relative L2: `{metrics['extraordinary_acfo_vs_scalar_pde_relative_l2']:.3e}`; wrong/correct residual ratio `{metrics['extraordinary_wrong_to_correct_pde_residual_ratio']:.2f}`",
            f"- single-manifold known-support inverse object error: `{metrics['single_manifold_known_support_object_relative_l2']:.3e}`; forced-sphere error `{metrics['single_manifold_wrong_sphere_object_relative_l2']:.3e}`",
            f"- double-manifold forward/adjoint 최대 direct-reference 오차: `{metrics['double_manifold_forward_relative_l2_max']:.3e}` / `{metrics['double_manifold_adjoint_relative_l2_max']:.3e}`",
            f"- double-manifold unrestricted 96-coefficient inverse object error: `{metrics['double_manifold_unrestricted_object_relative_l2']:.3e}`",
            "",
            "## 현재 방어 가능한 결론",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in summary["defensible_claims"])
    lines.extend(["", "## 명시적으로 제외해야 할 주장과 남은 검증", ""])
    lines.extend(f"- {gap}" for gap in summary["explicit_nonclaims_and_gaps"])
    lines.extend(
        [
            "",
            "## 최종 판단",
            "",
            "계산적 관점에서 ACFO의 핵심 일반화 주장은 성립한다. 즉, object representation과 meridional geometry를 분리한 동일 prepared harmonic backend가 sphere, ellipsoid, open surface, arbitrary spline, anisotropic dispersion branch와 ODT double-manifold 조합을 복소 forward-adjoint pair로 처리한다.",
            "",
            "다만 물리적 claim은 현재 `scalar anisotropic first-Born`까지이며, single-manifold의 일반 3-D inverse claim은 보류해야 한다. 논문 또는 기술문서에서는 WAXS arbitrary-curvature 정확도와 성능 지도를 수치 중심축으로 두고, anisotropic Born과 ODT double-manifold를 물리적·응용 확장으로 제시하는 구성이 가장 방어 가능하다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "benchmark_results" / "acfo_validation_program_summary.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "docs" / "acfo_validation_program_summary_ko.md",
    )
    args = parser.parse_args()
    summary = build_summary(load_payloads())
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
