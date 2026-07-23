from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results"
DOCS = ROOT / "docs"


def load(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def build_summary() -> dict[str, Any]:
    r2 = load("general_curvature_r2_holdout.json")
    r3 = load("general_curvature_r3_green_controls.json")
    dispersion = load("uniaxial_meep_dispersion_highres_decision.json")
    phase = load("uniaxial_meep_3d_phase_gate_decision.json")
    amplitude = load("uniaxial_meep_component_aware_amplitude_decision.json")
    dispersion_finest = dispersion["rows"][-1]
    phase_finest = phase["rows"][-1]
    amplitude_scoped = bool(amplitude["detectable_support_amplitude_pass"])
    amplitude_full = bool(amplitude["publication_full_amplitude_pass"])

    gates = {
        "r2_discrete_arbitrary_curve_forward_adjoint": bool(r2["passed"]),
        "r3_physical_vector_green_branch": bool(r3["passed"]),
        "external_meep_dispersion_geometry": bool(dispersion["passed"]),
        "external_meep_3d_phase_curvature": bool(phase["passed"]),
    }
    scoped_pass = all(gates.values())
    return {
        "schema": "general-curvature-validation-decision-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scoped_general_curvature_pass": scoped_pass,
        "publication_full_amplitude_pass": amplitude_full,
        "gates": gates,
        "evidence": {
            "r2_holdout": {
                "curve_families": r2["problem"]["curve_families"],
                "source_realizations": r2["problem"]["source_realizations"],
                "worst_case": r2["summary"]["worst_case"],
            },
            "r3_green_controls": r3["metrics"],
            "pymeep_dispersion": {
                "correct_relative_l2": dispersion_finest["correct_relative_l2"],
                "forced_to_correct_error_ratio": dispersion_finest[
                    "forced_to_correct_error_ratio"
                ],
                "maximum_correct_sphere_separation_bins": dispersion_finest[
                    "maximum_correct_sphere_separation_bins"
                ],
                "finest_next_finest_peak_l2": dispersion[
                    "finest_next_finest_peak_l2"
                ],
            },
            "pymeep_phase": {
                "calibrated_extraordinary_ellipse_relative_l2": phase_finest[
                    "calibrated_extraordinary_ellipse_relative_l2"
                ],
                "calibrated_sphere_to_ellipse_error_ratio": phase_finest[
                    "calibrated_sphere_to_ellipse_error_ratio"
                ],
                "calibrated_ordinary_sphere_relative_l2": phase_finest[
                    "calibrated_ordinary_sphere_relative_l2"
                ],
                "finest_next_finest_calibrated_grid_l2": phase[
                    "finest_next_finest_calibrated_grid_l2"
                ],
            },
            "pymeep_amplitude_boundary": {
                "detectable_support_amplitude_pass": amplitude_scoped,
                "grid_converged_detectable_support_pass": bool(
                    amplitude["grid_converged_detectable_support_pass"]
                ),
                "publication_full_amplitude_pass": amplitude_full,
                "decision": amplitude["decision"],
            },
        },
        "claim_matrix": [
            {
                "claim": "prepared forward/adjoint is valid on finite axisymmetric meridional curves, not only a sphere",
                "status": "PASS" if r2["passed"] else "FAIL",
                "evidence": "R2 direct Cartesian forward and weighted adjoint holdouts",
            },
            {
                "claim": "a physically nonspherical uniaxial Maxwell branch is supported with vector first-Born normalization",
                "status": "PASS" if r3["passed"] else "FAIL",
                "evidence": "R3 Cartesian Maxwell Green residue, rotation, vector-source, and selection controls",
            },
            {
                "claim": "the nonspherical curvature is observable and a forced sphere is detectably wrong",
                "status": "PASS" if dispersion["passed"] and phase["passed"] else "FAIL",
                "evidence": "independent PyMeep dispersion and 3-D phase gates plus R3 forced-sphere control",
            },
            {
                "claim": "publication-grade full complex FDTD amplitude is reproduced",
                "status": "UNRESOLVED" if not amplitude_full else "PASS",
                "evidence": "detectable support passes, but grid convergence and full forced-sphere amplitude gate remain open",
            },
        ],
        "decision": (
            "The scoped general-curvature claim is supported: the prepared operator is accurate on untuned arbitrary curves, "
            "the same architecture evaluates a physically nonspherical vector Maxwell branch, and independent PyMeep "
            "dispersion/phase results distinguish the correct curvature from a sphere. Full complex FDTD amplitude is not "
            "part of this pass and must not be claimed."
        ),
        "manuscript_boundary": {
            "supported": (
                "prepared reusable forward/adjoint architecture for rotationally structured non-spherical Fourier manifolds; "
                "homogeneous lossless uniaxial vector first-Born example; observable curvature advantage over a forced sphere"
            ),
            "not_supported": (
                "universal NUFFT replacement; arbitrary non-axisymmetric manifolds; lossy non-Hermitian residues; interfaces, "
                "multiple scattering, pump depletion, or publication-grade full nonlinear FDTD amplitude"
            ),
        },
        "inputs": [
            "benchmark_results/general_curvature_r2_holdout.json",
            "benchmark_results/general_curvature_r3_green_controls.json",
            "benchmark_results/uniaxial_meep_dispersion_highres_decision.json",
            "benchmark_results/uniaxial_meep_3d_phase_gate_decision.json",
            "benchmark_results/uniaxial_meep_component_aware_amplitude_decision.json",
        ],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    evidence = summary["evidence"]
    r2 = evidence["r2_holdout"]
    r3 = evidence["r3_green_controls"]
    dispersion = evidence["pymeep_dispersion"]
    phase = evidence["pymeep_phase"]
    amplitude = evidence["pymeep_amplitude_boundary"]
    worst = r2["worst_case"]
    lines = [
        "# General-curvature validation decision",
        "",
        f"Scoped general-curvature pass: **{summary['scoped_general_curvature_pass']}**",
        "",
        "## Claim matrix",
        "",
        "| claim | status | evidence |",
        "|---|:---:|---|",
    ]
    for row in summary["claim_matrix"]:
        lines.append(f"| {row['claim']} | **{row['status']}** | {row['evidence']} |")
    lines.extend(
        [
            "",
            "## New publication controls",
            "",
            f"- R2: `{r2['curve_families']}` curve families x `{r2['source_realizations']}` untuned complex source realizations. complex128 worst forward/adjoint/dot = `{worst['complex128_forward_relative_l2']:.3e}` / `{worst['complex128_weighted_adjoint_relative_l2']:.3e}` / `{worst['complex128_weighted_dot_error']:.3e}`; complex64 = `{worst['complex64_forward_relative_l2']:.3e}` / `{worst['complex64_weighted_adjoint_relative_l2']:.3e}` / `{worst['complex64_weighted_dot_error']:.3e}`.",
            f"- R3 correct extraordinary Green ACFO/direct L2: `{r3['correct_extraordinary']['green_field_acfo_vs_direct_relative_l2']:.3e}`.",
            f"- R3 forced sphere single-global-gain L2: `{r3['forced_sphere_geometry']['single_global_gain_relative_l2']:.3%}`; angle/branch fitting 없음.",
            f"- R3 axis-rotation residue/field L2: `{r3['axis_rotation_covariance']['residue_relative_l2']:.3e}` / `{r3['axis_rotation_covariance']['field_relative_l2']:.3e}`.",
            f"- R3 spatial-vector-source Green L2: `{r3['spatially_varying_vector_source']['green_field_acfo_vs_direct_relative_l2']:.3e}`; source second/first singular ratio `{r3['spatially_varying_vector_source']['second_to_first_singular_value_ratio']:.3f}`.",
            "",
            "## Independent physical curvature evidence already present",
            "",
            f"- PyMeep dispersion: correct ellipse L2 `{dispersion['correct_relative_l2']:.3%}`, forced/correct ratio `{dispersion['forced_to_correct_error_ratio']:.3f}`, max separation `{dispersion['maximum_correct_sphere_separation_bins']:.3f}` bins, finest-grid change `{dispersion['finest_next_finest_peak_l2']:.3%}`.",
            f"- PyMeep 3-D phase: calibrated ellipse L2 `{phase['calibrated_extraordinary_ellipse_relative_l2']:.3%}`, forced/correct ratio `{phase['calibrated_sphere_to_ellipse_error_ratio']:.3f}`, ordinary control `{phase['calibrated_ordinary_sphere_relative_l2']:.3%}`, finest-grid change `{phase['finest_next_finest_calibrated_grid_l2']:.3%}`.",
            "",
            "## Boundary that remains open",
            "",
            f"- Detectable-support amplitude pass: `{amplitude['detectable_support_amplitude_pass']}`.",
            f"- Grid-converged detectable-support pass: `{amplitude['grid_converged_detectable_support_pass']}`.",
            f"- Publication full-amplitude pass: `{amplitude['publication_full_amplitude_pass']}`.",
            "",
            "따라서 full complex FDTD amplitude를 주장하지 않는 한, 이 미해결 항목은 scoped general-curvature novelty를 반증하지 않는다. 논문에서는 discrete operator generality, homogeneous vector Maxwell branch, independent dispersion/phase curvature evidence까지만 연결해야 한다.",
            "",
            "## Manuscript claim boundary",
            "",
            f"Supported: {summary['manuscript_boundary']['supported']}.",
            "",
            f"Not supported: {summary['manuscript_boundary']['not_supported']}.",
            "",
            "Reproduce decision file with:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\summarize_general_curvature_validation.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    summary = build_summary()
    json_path = RESULTS / "general_curvature_validation_decision.json"
    markdown_path = DOCS / "general_curvature_validation_decision_ko.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
