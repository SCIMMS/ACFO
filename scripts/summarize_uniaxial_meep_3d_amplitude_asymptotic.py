from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results"
OUTPUT = RESULTS / "uniaxial_meep_3d_amplitude_asymptotic_decision.json"
MARKDOWN = RESULTS / "uniaxial_meep_3d_amplitude_asymptotic_decision.md"


def load(name: str) -> dict[str, object]:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def complex_array(values: list[list[float]]) -> np.ndarray:
    return np.array([complex(*value) for value in values], dtype=np.complex128)


def relative_l2(model: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(model - reference) / np.linalg.norm(reference))


def row(payload: dict[str, object]) -> dict[str, object]:
    return payload["rows"][0]


def compact(payload: dict[str, object]) -> dict[str, object]:
    item = row(payload)
    return {
        "configuration": payload["configuration"],
        "ordinary_calibration_l2": item["ordinary_calibration_l2"],
        "extraordinary_complex_l2": item["extraordinary_complex_l2"],
        "intensity_ncc": item["intensity_ncc"],
        "peak_error_deg": item["peak_error_deg"],
        "forced_sphere_wrong_to_correct_ratio": item[
            "forced_sphere_wrong_to_correct_ratio"
        ],
        "extraordinary_radial_ratio_scatter": item[
            "extraordinary_radial_ratio_scatter"
        ],
        "ordinary_radial_ratio_scatter": item["ordinary_radial_ratio_scatter"],
        "pattern_runtime_s": item["pattern_runtime_s"],
        "point_runtime_s": item["point_runtime_s"],
    }


def main() -> None:
    nonlinear = load("uniaxial_meep_3d_amplitude_asymptotic_c10_r12_h025.json")
    small = {
        resolution: load(
            f"uniaxial_meep_3d_amplitude_singlex_phi45_c10_r{resolution}_h025.json"
        )
        for resolution in (8, 12, 16)
    }
    large = {
        "r8_c16": load("uniaxial_meep_3d_amplitude_singlex_phi45_c16_r8_h08.json"),
        "r12_c14": load("uniaxial_meep_3d_amplitude_singlex_phi45_c14_r12_h08.json"),
    }
    calibrated = {
        resolution: complex_array(
            row(payload)["calibrated_extraordinary_ratio"]
        )
        for resolution, payload in small.items()
    }
    grid_8_to_12 = relative_l2(calibrated[8], calibrated[12])
    grid_12_to_16 = relative_l2(calibrated[12], calibrated[16])
    errors = np.array(
        [row(small[resolution])["extraordinary_complex_l2"] for resolution in (8, 12, 16)]
    )
    empirical_order = float(
        -np.polyfit(np.log(np.array([8.0, 12.0, 16.0])), np.log(errors), 1)[0]
    )
    small_npz = np.load(
        RESULTS / "uniaxial_meep_3d_amplitude_singlex_phi45_c10_r16_h025.npz"
    )
    large_npz = np.load(
        RESULTS / "uniaxial_meep_3d_amplitude_singlex_phi45_c14_r12_h08.npz"
    )
    small_reference_separation = relative_l2(
        small_npz["sphere_acfo"], small_npz["ellipse_acfo"]
    )
    large_reference_separation = relative_l2(
        large_npz["sphere_acfo"], large_npz["ellipse_acfo"]
    )
    nonlinear_row = row(nonlinear)
    single_r12 = row(small[12])
    single_r16 = row(small[16])
    representation_diagnostic = {
        "matched_geometry": "cell 10, PML 1, source half-width 0.25, radius 2.5-4.0, resolution 12",
        "nonlinear_multicomponent_extraordinary_l2": nonlinear_row[
            "extraordinary_complex_l2"
        ],
        "single_ex_extraordinary_l2": single_r12["extraordinary_complex_l2"],
        "l2_improvement_factor": nonlinear_row["extraordinary_complex_l2"]
        / single_r12["extraordinary_complex_l2"],
        "nonlinear_multicomponent_intensity_ncc": nonlinear_row["intensity_ncc"],
        "single_ex_intensity_ncc": single_r12["intensity_ncc"],
        "nonlinear_multicomponent_peak_error_deg": nonlinear_row["peak_error_deg"],
        "single_ex_peak_error_deg": single_r12["peak_error_deg"],
        "interpretation": (
            "Using one Ex Yee source at 45-deg azimuth removes component-wise source staggering while exciting both ordinary and extraordinary branches. "
            "The matched-geometry improvement identifies the multi-component Yee source/reference contract as a dominant failure mode."
        ),
    }
    gates = {
        "single_component_finest_extraordinary_l2_le_5pct": single_r16[
            "extraordinary_complex_l2"
        ]
        <= 0.05,
        "single_component_finest_intensity_ncc_ge_0_98": single_r16[
            "intensity_ncc"
        ]
        >= 0.98,
        "single_component_finest_peak_error_le_2deg": single_r16[
            "peak_error_deg"
        ]
        <= 2.0,
        "single_component_finest_radial_scatter_le_5pct": single_r16[
            "extraordinary_radial_ratio_scatter"
        ]
        <= 0.05,
        "ordinary_calibration_l2_le_5pct": single_r16["ordinary_calibration_l2"]
        <= 0.05,
        "grid_12_to_16_l2_le_2pct": grid_12_to_16 <= 0.02,
        "forced_sphere_ratio_ge_5": single_r16[
            "forced_sphere_wrong_to_correct_ratio"
        ]
        >= 5.0,
        "large_source_negative_control_bridge_pass": row(large["r12_c14"])[
            "extraordinary_complex_l2"
        ]
        <= 0.05
        and row(large["r12_c14"])["forced_sphere_wrong_to_correct_ratio"] >= 5.0,
    }
    result = {
        "schema": "uniaxial-meep-3d-amplitude-asymptotic-decision-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "actual-uniaxial 3-D PyMeep patterned/point-source amplitude bridge with far-radius raw-field diagnostics",
        "small_source_single_component_resolution_sweep": {
            str(resolution): compact(payload) for resolution, payload in small.items()
        },
        "grid_convergence": {
            "calibrated_field_l2_8_to_12": grid_8_to_12,
            "calibrated_field_l2_12_to_16": grid_12_to_16,
            "extraordinary_error_empirical_order": empirical_order,
        },
        "reference_separability": {
            "half_width_0p25_sphere_vs_ellipse_l2": small_reference_separation,
            "half_width_0p8_sphere_vs_ellipse_l2": large_reference_separation,
            "interpretation": (
                "The small source supports an accuracy check but is intrinsically too broadband for a 5x forced-sphere residual gate. "
                "The larger source is more discriminative analytically but its current Yee/FDTD bridge is not numerically stable."
            ),
        },
        "large_source_controls": {name: compact(payload) for name, payload in large.items()},
        "multicomponent_representation_diagnostic": representation_diagnostic,
        "gates": gates,
        "scoped_single_component_amplitude_pass": all(
            gates[name]
            for name in (
                "single_component_finest_extraordinary_l2_le_5pct",
                "single_component_finest_intensity_ncc_ge_0_98",
                "single_component_finest_peak_error_le_2deg",
                "single_component_finest_radial_scatter_le_5pct",
            )
        ),
        "publication_full_amplitude_pass": all(gates.values()),
        "decision": (
            "The single-Ex/45-deg bridge reaches 4.992% extraordinary complex L2 at resolution 16 with NCC 0.992 and 2-deg peak error, "
            "but the 12-to-16 field change is 6.67%, ordinary calibration is 5.42%, and the forced-sphere ratio is 1.06. "
            "Therefore this is a scoped representation diagnostic, not a publication-grade full nonlinear-source amplitude PASS."
        ),
        "claim_boundary": {
            "supported": "single-component homogeneous anisotropic Maxwell amplitude-shape correspondence and identification of multi-component Yee source staggering as a dominant mismatch",
            "not_supported": "grid-converged full nonlinear-source complex amplitude or a discriminative 3-D forced-sphere negative-control gate",
            "next_valid_route": (
                "Use a component-aware Yee-grid source oracle or collocated current injection, then repeat the same fixed 8/12/16 grid protocol; "
                "do not spend more compute on brute-force domain enlargement with the current source contract."
            ),
        },
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Uniaxial PyMeep 3-D asymptotic amplitude decision",
        "",
        f"- scoped single-component amplitude: **{'PASS' if result['scoped_single_component_amplitude_pass'] else 'FAIL'}**",
        f"- publication full amplitude: **{'PASS' if result['publication_full_amplitude_pass'] else 'FAIL'}**",
        "",
        "| resolution | extraordinary L2 | NCC | peak error | ordinary calibration | radial scatter | forced ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for resolution in (8, 12, 16):
        item = row(small[resolution])
        lines.append(
            f"| {resolution} | {100*item['extraordinary_complex_l2']:.3f}% | {item['intensity_ncc']:.6f} | {item['peak_error_deg']:.1f} deg | "
            f"{100*item['ordinary_calibration_l2']:.3f}% | {100*item['extraordinary_radial_ratio_scatter']:.3f}% | {item['forced_sphere_wrong_to_correct_ratio']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"- calibrated field L2, 8->12: `{grid_8_to_12:.6f}`",
            f"- calibrated field L2, 12->16: `{grid_12_to_16:.6f}`",
            f"- empirical error order: `{empirical_order:.3f}`",
            f"- matched-geometry nonlinear->single-Ex L2 improvement: `{representation_diagnostic['l2_improvement_factor']:.2f}x`",
            "",
            result["decision"],
            "",
            "The next run must change the source/reference contract, not only the domain size.",
        ]
    )
    MARKDOWN.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {OUTPUT} and {MARKDOWN}")


if __name__ == "__main__":
    main()
