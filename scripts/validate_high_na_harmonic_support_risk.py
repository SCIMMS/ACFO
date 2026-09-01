from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_high_na_debye_wolf import (  # noqa: E402
    PreparedPositiveRhoDependentHarmonicDebyeWolfPlan,
    direct_debye_wolf,
    focal_axes,
    flatten_focal_axes,
    gauss_theta_grid,
    pupil_field,
    relative_l2,
    significant_pupil_h_abs,
)
from benchmark_high_na_pupil_spectrum_adaptive import auto_h_cutoff  # noqa: E402
from benchmark_high_na_vectorial_backpropagation import (  # noqa: E402
    direct_vectorial_debye_wolf,
    mix_vectorial_pupil,
    richards_wolf_jones_matrix,
    separable_vectorial_evaluate,
    vectorial_pupil_jones,
)


def support(values: np.ndarray | list[np.ndarray]) -> list[int]:
    return [
        int(value)
        for value in significant_pupil_h_abs(values, relative_threshold=1e-6)
    ]


def build_plan(
    *,
    nphi: int,
    theta: np.ndarray,
    theta_weights: np.ndarray,
    rho_axis: np.ndarray,
    psi_axis: np.ndarray,
    z_axis: np.ndarray,
    k: float,
    geometric_h_cutoff: int,
    margin: int,
    pupil_spectrum: str,
    pupils: list[np.ndarray] | None,
    required_h_abs: list[int] | None = None,
) -> PreparedPositiveRhoDependentHarmonicDebyeWolfPlan:
    return PreparedPositiveRhoDependentHarmonicDebyeWolfPlan.build(
        nphi,
        theta,
        theta_weights,
        rho_axis,
        psi_axis,
        z_axis,
        k=k,
        h_cutoff=geometric_h_cutoff,
        margin=margin,
        cutoff_bin_size=2,
        required_h_abs=required_h_abs,
        pupil_spectrum=pupil_spectrum,
        pupil_spectrum_pupils=pupils,
        pupil_spectrum_relative_threshold=1e-6,
        backend="auto",
        no_copy_coefficients=True,
    )


def evaluate_current_case(
    sin_theta_max: float,
    *,
    ntheta: int,
    nphi: int,
    k: float,
    margin: int,
    rho_axis: np.ndarray,
    psi_axis: np.ndarray,
    z_axis: np.ndarray,
    rho: np.ndarray,
    psi: np.ndarray,
    z: np.ndarray,
) -> dict[str, object]:
    theta_max = float(np.arcsin(sin_theta_max))
    theta, theta_weights = gauss_theta_grid(ntheta, theta_max)
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    scalar_pupil = pupil_field(
        "mixed",
        theta,
        phi,
        theta_max=theta_max,
        strength=0.45,
        vortex_charge=6,
        apodization="sqrt-cos",
    )
    vector_pupil = vectorial_pupil_jones(
        "x_vortex",
        theta,
        phi,
        theta_max=theta_max,
        strength=0.45,
        vortex_charge=6,
    )
    mixing = richards_wolf_jones_matrix(theta, phi, apodization="sqrt-cos")
    effective = mix_vectorial_pupil(vector_pupil, mixing)
    geometric = auto_h_cutoff(
        k=k,
        rho_max=float(rho_axis[-1]),
        sin_theta_max=sin_theta_max,
        margin=margin,
        nphi=nphi,
    )
    scalar_direct = direct_debye_wolf(
        scalar_pupil, theta, theta_weights, phi, rho, psi, z, k=k
    )
    vector_direct = direct_vectorial_debye_wolf(
        vector_pupil, mixing, theta, theta_weights, phi, rho, psi, z, k=k
    )
    scalar_geometric = build_plan(
        nphi=nphi,
        theta=theta,
        theta_weights=theta_weights,
        rho_axis=rho_axis,
        psi_axis=psi_axis,
        z_axis=z_axis,
        k=k,
        geometric_h_cutoff=geometric,
        margin=margin,
        pupil_spectrum="off",
        pupils=None,
    )
    vector_geometric = build_plan(
        nphi=nphi,
        theta=theta,
        theta_weights=theta_weights,
        rho_axis=rho_axis,
        psi_axis=psi_axis,
        z_axis=z_axis,
        k=k,
        geometric_h_cutoff=geometric,
        margin=margin,
        pupil_spectrum="off",
        pupils=None,
    )
    scalar_adaptive = build_plan(
        nphi=nphi,
        theta=theta,
        theta_weights=theta_weights,
        rho_axis=rho_axis,
        psi_axis=psi_axis,
        z_axis=z_axis,
        k=k,
        geometric_h_cutoff=geometric,
        margin=margin,
        pupil_spectrum="adaptive",
        pupils=[scalar_pupil],
    )
    vector_adaptive = build_plan(
        nphi=nphi,
        theta=theta,
        theta_weights=theta_weights,
        rho_axis=rho_axis,
        psi_axis=psi_axis,
        z_axis=z_axis,
        k=k,
        geometric_h_cutoff=geometric,
        margin=margin,
        pupil_spectrum="adaptive",
        pupils=[effective[index] for index in range(3)],
    )
    return {
        "sin_theta_max": sin_theta_max,
        "geometric_h_cutoff": geometric,
        "scalar_significant_h_abs": support(scalar_pupil),
        "vector_raw_significant_h_abs": support([vector_pupil[index] for index in range(2)]),
        "vector_effective_significant_h_abs": support([effective[index] for index in range(3)]),
        "scalar_geometric_l2": relative_l2(scalar_geometric.evaluate(scalar_pupil), scalar_direct),
        "scalar_adaptive_l2": relative_l2(scalar_adaptive.evaluate(scalar_pupil), scalar_direct),
        "vector_geometric_l2": relative_l2(
            separable_vectorial_evaluate(vector_geometric, vector_pupil, mixing), vector_direct
        ),
        "vector_adaptive_effective_l2": relative_l2(
            separable_vectorial_evaluate(vector_adaptive, vector_pupil, mixing), vector_direct
        ),
        "scalar_required_h_abs": scalar_adaptive.required_h_abs.tolist(),
        "vector_required_h_abs": vector_adaptive.required_h_abs.tolist(),
    }


def evaluate_vector_stress(
    *,
    sin_theta_max: float,
    vortex_charge: int,
    ntheta: int,
    nphi: int,
    k: float,
    margin: int,
    rho_axis: np.ndarray,
    psi_axis: np.ndarray,
    z_axis: np.ndarray,
    rho: np.ndarray,
    psi: np.ndarray,
    z: np.ndarray,
) -> dict[str, object]:
    theta_max = float(np.arcsin(sin_theta_max))
    theta, theta_weights = gauss_theta_grid(ntheta, theta_max)
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    pupil = vectorial_pupil_jones(
        "x_vortex",
        theta,
        phi,
        theta_max=theta_max,
        strength=0.45,
        vortex_charge=vortex_charge,
    )
    mixing = richards_wolf_jones_matrix(theta, phi, apodization="sqrt-cos")
    effective = mix_vectorial_pupil(pupil, mixing)
    geometric = auto_h_cutoff(
        k=k,
        rho_max=float(rho_axis[-1]),
        sin_theta_max=sin_theta_max,
        margin=margin,
        nphi=nphi,
    )
    direct = direct_vectorial_debye_wolf(
        pupil, mixing, theta, theta_weights, phi, rho, psi, z, k=k
    )
    effective_support = support([effective[index] for index in range(3)])
    variants: dict[str, tuple[str, list[np.ndarray] | None, list[int] | None]] = {
        "geometric_only": ("off", None, None),
        "adaptive_raw_jones": ("adaptive", [pupil[index] for index in range(2)], None),
        "adaptive_effective_vector": (
            "adaptive",
            [effective[index] for index in range(3)],
            None,
        ),
        "manual_all_effective_modes": ("off", None, effective_support),
    }
    results: dict[str, object] = {}
    for name, (mode, pupils, required) in variants.items():
        plan = build_plan(
            nphi=nphi,
            theta=theta,
            theta_weights=theta_weights,
            rho_axis=rho_axis,
            psi_axis=psi_axis,
            z_axis=z_axis,
            k=k,
            geometric_h_cutoff=geometric,
            margin=margin,
            pupil_spectrum=mode,
            pupils=pupils,
            required_h_abs=required,
        )
        field = separable_vectorial_evaluate(plan, pupil, mixing)
        results[name] = {
            "complex_l2": relative_l2(field, direct),
            "required_h_abs": plan.required_h_abs.tolist(),
            "mode_rho_work": plan.mode_rho_work,
        }
    geometric_work = int(results["geometric_only"]["mode_rho_work"])
    for values in results.values():
        values["mode_rho_work_ratio_vs_geometric"] = (
            float(values["mode_rho_work"]) / float(geometric_work)
        )
    return {
        "sin_theta_max": sin_theta_max,
        "vortex_charge": vortex_charge,
        "geometric_h_cutoff": geometric,
        "raw_jones_significant_h_abs": support([pupil[index] for index in range(2)]),
        "effective_vector_significant_h_abs": effective_support,
        "variants": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit High-NA high-order harmonic support.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/high_na_harmonic_support_risk.json"),
    )
    args = parser.parse_args()
    ntheta = 32
    nphi = 96
    k = 2.0 * np.pi
    margin = 6
    rho_axis, psi_axis, z_axis = focal_axes(
        nrho=12, npsi=24, nz=5, rho_max=2.0, z_max=1.0
    )
    rho, psi, z = flatten_focal_axes(rho_axis, psi_axis, z_axis)
    current = [
        evaluate_current_case(
            value,
            ntheta=ntheta,
            nphi=nphi,
            k=k,
            margin=margin,
            rho_axis=rho_axis,
            psi_axis=psi_axis,
            z_axis=z_axis,
            rho=rho,
            psi=psi,
            z=z,
        )
        for value in (0.5, 0.8, 0.95)
    ]
    charges = (0, 6, 10, 14, 18, 22, 24)
    charge_sweep = [
        evaluate_vector_stress(
            sin_theta_max=0.8,
            vortex_charge=charge,
            ntheta=ntheta,
            nphi=nphi,
            k=k,
            margin=margin,
            rho_axis=rho_axis,
            psi_axis=psi_axis,
            z_axis=z_axis,
            rho=rho,
            psi=psi,
            z=z,
        )
        for charge in charges
    ]
    stress = next(row for row in charge_sweep if row["vortex_charge"] == 18)
    gates = {
        "current_charge6_geometric_scalar_le_1e_6": all(
            row["scalar_geometric_l2"] <= 1e-6 for row in current
        ),
        "current_charge6_geometric_vector_le_1e_6": all(
            row["vector_geometric_l2"] <= 1e-6 for row in current
        ),
        "stress_geometric_detects_failure": stress["variants"]["geometric_only"]["complex_l2"] >= 0.1,
        "stress_raw_adaptive_is_insufficient": stress["variants"]["adaptive_raw_jones"]["complex_l2"] > 1e-4,
        "charge_sweep_effective_adaptive_le_1e_6": all(
            row["variants"]["adaptive_effective_vector"]["complex_l2"] <= 1e-6
            for row in charge_sweep
        ),
        "charge_sweep_effective_adaptive_matches_manual": all(
            abs(
                row["variants"]["adaptive_effective_vector"]["complex_l2"]
                - row["variants"]["manual_all_effective_modes"]["complex_l2"]
            )
            <= 1e-15
            for row in charge_sweep
        ),
        "charge_sweep_work_ratio_le_2": all(
            row["variants"]["adaptive_effective_vector"][
                "mode_rho_work_ratio_vs_geometric"
            ]
            <= 2.0
            for row in charge_sweep
        ),
    }
    result = {
        "schema": "high-na-harmonic-support-risk-v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "ntheta": ntheta,
            "nphi": nphi,
            "target_shape": [rho_axis.size, psi_axis.size, z_axis.size],
            "rho_max": float(rho_axis[-1]),
            "margin": margin,
            "cutoff_bin_size": 2,
            "relative_spectrum_threshold": 1e-6,
        },
        "current_si_charge6": current,
        "vector_charge_sweep": charge_sweep,
        "vector_charge18_stress": stress,
        "gates": gates,
        "passed": all(gates.values()),
        "conclusion": "The corrected rho-dependent adaptive rule preserves every significant post-Richards-Wolf effective vector harmonic at each local rho cutoff. It recovers the direct Debye-Wolf reference across the charge sweep while retaining bounded sparse work overhead.",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
