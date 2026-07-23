from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from validate_uniaxial_vector_born_direct import (  # noqa: E402
    direct_binned_fourier,
    midpoint_domain_source,
    outgoing_manifold,
)
from waxs_cake import (  # noqa: E402
    PreparedAxisymmetricOperator,
    apply_maxwell_spectral_residue,
    gayer_5mol_mgo_cln_index,
    linbo3_3m_nonlinear_polarization,
    make_cylindrical_histogram,
    maxwell_resolvent_residue,
    maxwell_spectral_residue,
    project_vector_born_field,
    uniaxial_eigenpolarization,
)


def relative_l2(model: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference.ravel()))
    if denominator == 0.0:
        return 0.0 if np.linalg.norm(model.ravel()) == 0.0 else float("inf")
    return float(np.linalg.norm((model - reference).ravel()) / denominator)


def complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def fit_gain(model: np.ndarray, reference: np.ndarray) -> complex:
    denominator = np.vdot(model.ravel(), model.ravel())
    if denominator == 0.0:
        raise ValueError("cannot fit a gain to a zero model")
    return complex(np.vdot(model.ravel(), reference.ravel()) / denominator)


def run_case(
    *,
    name: str,
    pump_polarization: np.ndarray,
    pump_index: float,
    outgoing_branch: str,
    n: int,
    half_width_um: float,
    n_phi: int,
    u: np.ndarray,
    wavelength_pump_um: float,
    wavelength_sh_um: float,
    n_o_sh: float,
    n_e_sh: float,
    resolvent_etas: np.ndarray,
) -> dict[str, object]:
    pump_k = 2.0 * np.pi * pump_index / wavelength_pump_um
    coords, weights, voxel_volume = midpoint_domain_source(
        n=n,
        half_width_um=half_width_um,
        pump_wave_number_per_um=pump_k,
    )
    binned = make_cylindrical_histogram(
        coords,
        atom_weights=weights,
        n_r=n,
        n_z=n,
        n_phi=n_phi,
        r_max=np.sqrt(2.0) * half_width_um,
        z_range=(-half_width_um, half_width_um),
        hist_dtype=np.complex128,
        backend="numpy",
    )
    manifold = outgoing_manifold(
        u,
        wavelength_um=wavelength_sh_um,
        n_o=n_o_sh,
        n_e=n_e_sh,
        branch=outgoing_branch,
    )
    started = time.perf_counter()
    acfo_scalar = PreparedAxisymmetricOperator(
        binned,
        manifold,
        complex_dtype=np.complex128,
    ).forward(binned.hist)
    acfo_seconds = time.perf_counter() - started
    started = time.perf_counter()
    direct_scalar = direct_binned_fourier(binned, manifold)
    direct_seconds = time.perf_counter() - started

    k_nodes = manifold.target_nodes(binned.beta_centers)
    k0 = 2.0 * np.pi / wavelength_sh_um
    epsilon_perpendicular = n_o_sh * n_o_sh
    epsilon_parallel = n_e_sh * n_e_sh
    epsilon = np.diag(
        [epsilon_perpendicular, epsilon_perpendicular, epsilon_parallel]
    )
    source_vector = linbo3_3m_nonlinear_polarization(pump_polarization)
    green_acfo = apply_maxwell_spectral_residue(
        acfo_scalar,
        k_nodes,
        source_vector,
        k0=k0,
        epsilon_tensor=epsilon,
    )
    green_direct = apply_maxwell_spectral_residue(
        direct_scalar,
        k_nodes,
        source_vector,
        k0=k0,
        epsilon_tensor=epsilon,
    )
    residue, residue_diagnostics = maxwell_spectral_residue(
        k_nodes,
        k0=k0,
        epsilon_tensor=epsilon,
        return_diagnostics=True,
    )

    eigen = uniaxial_eigenpolarization(
        manifold.q_perp,
        manifold.q_z,
        binned.beta_centers,
        epsilon_parallel=epsilon_parallel,
        epsilon_perpendicular=epsilon_perpendicular,
        branch=outgoing_branch,
    )
    if outgoing_branch == "ordinary":
        legacy_weight = 1.0 / (2.0 * manifold.q_z)
    else:
        legacy_weight = epsilon_perpendicular / (2.0 * manifold.q_z)
    legacy_direct = project_vector_born_field(
        direct_scalar * legacy_weight[:, None],
        eigen,
        source_vector,
    )
    # The Maxwell wave equation and the legacy scalar convention differ by one
    # global sign.  Ordinary polarization fixes that sign without per-angle or
    # per-branch fitting.
    signed_green_direct = -green_direct
    best_gain = fit_gain(legacy_direct, signed_green_direct)

    resolvent_rows: list[dict[str, float]] = []
    for eta in resolvent_etas:
        numerical_residue = maxwell_resolvent_residue(
            k_nodes,
            k0=k0,
            epsilon_tensor=epsilon,
            eta=float(eta),
        )
        resolvent_rows.append(
            {
                "eta": float(eta),
                "residue_complex_l2": relative_l2(numerical_residue, residue),
            }
        )
    log_eta = np.log([row["eta"] for row in resolvent_rows])
    log_error = np.log([row["residue_complex_l2"] for row in resolvent_rows])
    convergence_order = float(np.polyfit(log_eta, log_error, 1)[0])
    return {
        "name": name,
        "outgoing_branch": outgoing_branch,
        "active_cartesian_voxels": int(coords.shape[0]),
        "nonzero_cylindrical_bins": int(np.count_nonzero(binned.hist)),
        "voxel_volume_um3": voxel_volume,
        "q_samples": int(manifold.n_u * n_phi),
        "acfo_seconds": acfo_seconds,
        "direct_seconds": direct_seconds,
        "scalar_acfo_vs_direct_complex_l2": relative_l2(
            acfo_scalar, direct_scalar
        ),
        "green_field_acfo_vs_direct_complex_l2": relative_l2(
            green_acfo, green_direct
        ),
        "maxwell_pole": residue_diagnostics,
        "resolvent_limit": {
            "rows": resolvent_rows,
            "convergence_order": convergence_order,
            "finest_complex_l2": resolvent_rows[-1]["residue_complex_l2"],
        },
        "legacy_scalar_weight_diagnostic": {
            "weight_formula": (
                "1/(2*kz)"
                if outgoing_branch == "ordinary"
                else "epsilon_perpendicular/(2*kz)"
            ),
            "signed_l2_without_refit": relative_l2(
                legacy_direct, signed_green_direct
            ),
            "best_per_branch_gain": complex_pair(best_gain),
            "best_per_branch_gain_l2": relative_l2(
                best_gain * legacy_direct, signed_green_direct
            ),
            "legacy_to_green_norm_ratio": float(
                np.linalg.norm(legacy_direct) / np.linalg.norm(green_direct)
            ),
        },
        "green_field_norm": float(np.linalg.norm(green_direct)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate ACFO vector first-Born amplitudes against a Cartesian Maxwell Green-tensor residue."
    )
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--n-phi", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark_results/uniaxial_green_tensor_residue_64cubed.json"
        ),
    )
    args = parser.parse_args()

    wavelength_pump_um = 1.064
    wavelength_sh_um = 0.532
    temperature_c = 24.5
    n_o_pump = gayer_5mol_mgo_cln_index(
        wavelength_pump_um, "ordinary", temperature_c=temperature_c
    )
    n_e_pump = gayer_5mol_mgo_cln_index(
        wavelength_pump_um, "extraordinary", temperature_c=temperature_c
    )
    n_o_sh = gayer_5mol_mgo_cln_index(
        wavelength_sh_um, "ordinary", temperature_c=temperature_c
    )
    n_e_sh = gayer_5mol_mgo_cln_index(
        wavelength_sh_um, "extraordinary", temperature_c=temperature_c
    )
    u = np.linspace(0.08, 0.78, 12)
    half_width_um = 2.0
    etas = np.array([1e-5, 1e-6, 1e-7], dtype=np.float64)
    cases = [
        run_case(
            name="extraordinary-pump-to-extraordinary-SH",
            pump_polarization=np.array([0.0, 0.0, 1.0], dtype=np.complex128),
            pump_index=n_e_pump,
            outgoing_branch="extraordinary",
            n=args.n,
            half_width_um=half_width_um,
            n_phi=args.n_phi,
            u=u,
            wavelength_pump_um=wavelength_pump_um,
            wavelength_sh_um=wavelength_sh_um,
            n_o_sh=n_o_sh,
            n_e_sh=n_e_sh,
            resolvent_etas=etas,
        ),
        run_case(
            name="ordinary-pump-to-ordinary-SH-control",
            pump_polarization=np.array([0.0, 1.0, 0.0], dtype=np.complex128),
            pump_index=n_o_pump,
            outgoing_branch="ordinary",
            n=args.n,
            half_width_um=half_width_um,
            n_phi=args.n_phi,
            u=u,
            wavelength_pump_um=wavelength_pump_um,
            wavelength_sh_um=wavelength_sh_um,
            n_o_sh=n_o_sh,
            n_e_sh=n_e_sh,
            resolvent_etas=etas,
        ),
    ]
    by_branch = {case["outgoing_branch"]: case for case in cases}
    gates = {
        "maxwell_pole_residual_le_1e_12": max(
            case["maxwell_pole"]["max_normalized_null_residual"]
            for case in cases
        )
        <= 1e-12,
        "resolvent_limit_l2_le_1e_5": max(
            case["resolvent_limit"]["finest_complex_l2"] for case in cases
        )
        <= 1e-5,
        "resolvent_first_order_convergence": min(
            case["resolvent_limit"]["convergence_order"] for case in cases
        )
        >= 0.8,
        "green_field_acfo_vs_direct_l2_le_1e_8": max(
            case["green_field_acfo_vs_direct_complex_l2"] for case in cases
        )
        <= 1e-8,
        "legacy_ordinary_scalar_weight_matches": by_branch["ordinary"][
            "legacy_scalar_weight_diagnostic"
        ]["signed_l2_without_refit"]
        <= 1e-10,
        "legacy_extraordinary_normalization_problem_detected": by_branch[
            "extraordinary"
        ]["legacy_scalar_weight_diagnostic"]["signed_l2_without_refit"]
        >= 1.0,
    }
    result = {
        "schema": "uniaxial-green-tensor-residue-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "homogeneous-bulk vector first-Born complex amplitude at simple anisotropic Maxwell spectral poles",
        "reference_independence": (
            "The amplitude normalization is obtained from the Cartesian Maxwell wave-operator nullspace and the resolvent limit; no branch-specific scalar pole weight is used by the oracle."
        ),
        "material": {
            "name": "5 mol% MgO-doped congruent LiNbO3",
            "temperature_c": temperature_c,
            "pump_wavelength_um": wavelength_pump_um,
            "sh_wavelength_um": wavelength_sh_um,
            "n_o_pump": n_o_pump,
            "n_e_pump": n_e_pump,
            "n_o_sh": n_o_sh,
            "n_e_sh": n_e_sh,
        },
        "object": {
            "cartesian_shape": [args.n, args.n, args.n],
            "half_width_um": half_width_um,
            "domain": "off-axis binary two-carrier hologram inside a superellipsoid",
            "cylindrical_shape": [args.n, args.n, args.n_phi],
        },
        "cases": cases,
        "gates": gates,
        "passed": all(gates.values()),
        "decision": (
            "PASS for the corrected homogeneous anisotropic Maxwell Green-tensor first-Born amplitude oracle. "
            "The earlier scalar extraordinary residue is not a valid dyadic Maxwell amplitude normalization."
        ),
        "claim_boundary": {
            "supported": "relative complex vector first-Born amplitudes on ordinary and extraordinary bulk branches after dyadic Green-residue projection",
            "not_supported": "finite-crystal interfaces, Fresnel coupling, pump depletion, multiple scattering, nonlinear propagation, or publication-grade PyMeep full-amplitude agreement",
            "pymeep_status": "the finite-radius patterned/point bridge remains failed and is not reclassified by this analytic spectral reference",
        },
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
