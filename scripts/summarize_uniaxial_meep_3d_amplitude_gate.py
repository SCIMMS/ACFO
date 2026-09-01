from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def metrics(payload: dict) -> dict:
    row = payload["rows"][0]
    return {
        "resolution": row["resolution"],
        "source_half_width": payload["configuration"]["source_half_width"],
        "ordinary_calibration_l2": row["ordinary_calibration_l2"],
        "extraordinary_complex_l2": row["extraordinary_complex_l2"],
        "intensity_ncc": row["intensity_ncc"],
        "peak_error_deg": row["peak_error_deg"],
        "forced_sphere_wrong_to_correct_ratio": row[
            "forced_sphere_wrong_to_correct_ratio"
        ],
        "extraordinary_radial_ratio_scatter": row[
            "extraordinary_radial_ratio_scatter"
        ],
        "ordinary_radial_ratio_scatter": row["ordinary_radial_ratio_scatter"],
        "algorithm_complex_l2": payload["reference"]["algorithm_complex_l2"],
        "representation_complex_l2": payload["reference"][
            "representation_complex_l2"
        ],
    }


def main() -> None:
    cases = [
        metrics(load("uniaxial_meep_3d_amplitude_debug_r12.json")),
        metrics(load("uniaxial_meep_3d_amplitude_debug_h04_r12.json")),
        metrics(load("uniaxial_meep_3d_amplitude_debug_h04_r16.json")),
        metrics(load("uniaxial_meep_3d_amplitude_debug_h025_r12.json")),
    ]
    h04_r12 = cases[1]
    h04_r16 = cases[2]
    h025_r12 = cases[3]
    result = {
        "schema": "uniaxial-meep-3d-amplitude-gate-decision-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "actual-uniaxial 3-D patterned-to-point-source complex-amplitude "
            "bridge; finite-radius DFT-field observation"
        ),
        "cases": cases,
        "diagnostics": {
            "source_shrink_h08_to_h025_at_r12": {
                "ordinary_calibration_l2_before": cases[0]["ordinary_calibration_l2"],
                "ordinary_calibration_l2_after": h025_r12[
                    "ordinary_calibration_l2"
                ],
                "ordinary_radial_scatter_before": cases[0][
                    "ordinary_radial_ratio_scatter"
                ],
                "ordinary_radial_scatter_after": h025_r12[
                    "ordinary_radial_ratio_scatter"
                ],
                "interpretation": (
                    "large improvement confirms finite-distance/near-field contamination"
                ),
            },
            "grid_change_h04_r12_to_r16": {
                "ordinary_calibration_l2_before": h04_r12[
                    "ordinary_calibration_l2"
                ],
                "ordinary_calibration_l2_after": h04_r16[
                    "ordinary_calibration_l2"
                ],
                "extraordinary_complex_l2_before": h04_r12[
                    "extraordinary_complex_l2"
                ],
                "extraordinary_complex_l2_after": h04_r16[
                    "extraordinary_complex_l2"
                ],
                "extraordinary_radial_scatter_before": h04_r12[
                    "extraordinary_radial_ratio_scatter"
                ],
                "extraordinary_radial_scatter_after": h04_r16[
                    "extraordinary_radial_ratio_scatter"
                ],
                "interpretation": (
                    "Yee-grid resampling contributes but does not explain the dominant mismatch"
                ),
            },
            "meep_near_to_far_constraint": {
                "supported_here": False,
                "reason": (
                    "Meep requires near-to-far surfaces to lie in a homogeneous "
                    "material with isotropic epsilon and mu; this probe intentionally "
                    "uses an actual uniaxial background"
                ),
                "official_documentation": (
                    "https://meep.readthedocs.io/en/master/Python_User_Interface/"
                    "#near-to-far-field-spectra"
                ),
            },
        },
        "gates": {
            "ordinary_control_can_reach_5pct": h025_r12[
                "ordinary_calibration_l2"
            ] <= 0.05,
            "extraordinary_complex_l2_le_5pct": h04_r16[
                "extraordinary_complex_l2"
            ] <= 0.05,
            "intensity_ncc_ge_0_98": h04_r16["intensity_ncc"] >= 0.98,
            "peak_error_le_2deg": h04_r16["peak_error_deg"] <= 2.0,
            "forced_sphere_ratio_ge_5": h04_r16[
                "forced_sphere_wrong_to_correct_ratio"
            ] >= 5.0,
            "acfo_algorithm_error_negligible": h04_r16["algorithm_complex_l2"]
            <= 1e-12,
        },
        "passed": False,
        "decision": (
            "FAIL for publication-grade full complex amplitude/NCC/peak. "
            "Do not extend the separately passed 3-D phase-curvature gate to a "
            "full-amplitude Maxwell validation claim."
        ),
        "claim_boundary": {
            "supported": (
                "actual-uniaxial 3-D phase curvature and ellipse-versus-sphere "
                "dispersion discrimination"
            ),
            "not_supported": (
                "actual-uniaxial full angle-resolved complex amplitude, NCC, and peak"
            ),
            "next_valid_route": (
                "an independent anisotropic far-field/Green-tensor reference or a "
                "much larger-domain asymptotic-field extraction; Meep isotropic "
                "near-to-far cannot be used for this background"
            ),
        },
    }
    output = RESULTS / "uniaxial_meep_3d_amplitude_gate_decision.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
