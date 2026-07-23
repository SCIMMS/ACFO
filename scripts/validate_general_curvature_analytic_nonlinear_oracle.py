from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from waxs_cake import (  # noqa: E402
    AnisotropicGaussianMixture,
    AxisymmetricManifold,
    PreparedAxisymmetricOperator,
    apply_maxwell_spectral_residue,
    binned_structure_sources,
    complex_error_metrics,
    direct_axisymmetric_amplitude,
    gayer_5mol_mgo_cln_index,
    linbo3_3m_nonlinear_polarization,
    make_cylindrical_histogram,
    maxwell_resolvent_residue,
    maxwell_spectral_residue,
)


PUMP_WAVELENGTH_UM = 1.064
SH_WAVELENGTH_UM = 0.532
TEMPERATURE_C = 24.5
PUMP_SURFACE_PARAMETER_DEG = 38.0
PUMP_AZIMUTH_DEG = 1.875
DOMAIN_HALF_WIDTH_UM = 1.4
U_MIN_RAD = 0.08
U_MAX_RAD = 1.35
DEFAULT_N_U = 18
SELECTED_OUTPUT_AZIMUTHS_DEG = (-5.625, 1.875, 9.375)


def relative_l2(model: np.ndarray, reference: np.ndarray) -> float:
    model_array = np.asarray(model)
    reference_array = np.asarray(reference)
    denominator = float(np.linalg.norm(reference_array.ravel()))
    if denominator == 0.0:
        return 0.0 if np.linalg.norm(model_array.ravel()) == 0.0 else float("inf")
    return float(np.linalg.norm((model_array - reference_array).ravel()) / denominator)


def fit_global_gain(model: np.ndarray, reference: np.ndarray) -> complex:
    model_array = np.asarray(model).ravel()
    reference_array = np.asarray(reference).ravel()
    denominator = np.vdot(model_array, model_array)
    if denominator == 0.0:
        raise ValueError("cannot fit a gain to a zero model")
    return complex(np.vdot(model_array, reference_array) / denominator)


def complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def rotation_z(angle_rad: float) -> np.ndarray:
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def make_physical_problem() -> dict[str, Any]:
    """Build one manufactured Gaussian-pump LiNbO3 SHG problem.

    The pump polarization and carrier satisfy the extraordinary bulk
    dispersion relation.  The Gaussian envelope is prescribed rather than
    obtained from a Maxwell propagation solve, which keeps the nonlinear
    source Fourier transform analytic.
    """

    n_o_pump = gayer_5mol_mgo_cln_index(
        PUMP_WAVELENGTH_UM,
        "ordinary",
        temperature_c=TEMPERATURE_C,
    )
    n_e_pump = gayer_5mol_mgo_cln_index(
        PUMP_WAVELENGTH_UM,
        "extraordinary",
        temperature_c=TEMPERATURE_C,
    )
    n_o_sh = gayer_5mol_mgo_cln_index(
        SH_WAVELENGTH_UM,
        "ordinary",
        temperature_c=TEMPERATURE_C,
    )
    n_e_sh = gayer_5mol_mgo_cln_index(
        SH_WAVELENGTH_UM,
        "extraordinary",
        temperature_c=TEMPERATURE_C,
    )

    parameter = np.deg2rad(PUMP_SURFACE_PARAMETER_DEG)
    azimuth = np.deg2rad(PUMP_AZIMUTH_DEG)
    radial = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
    pump_k0 = 2.0 * np.pi / PUMP_WAVELENGTH_UM
    pump_k_perpendicular = pump_k0 * n_e_pump * np.sin(parameter)
    pump_k_z = pump_k0 * n_o_pump * np.cos(parameter)
    pump_wavevector = pump_k_perpendicular * radial + np.array(
        [0.0, 0.0, pump_k_z]
    )
    pump_electric = (
        (pump_k_z / n_o_pump**2) * radial
        - np.array([0.0, 0.0, pump_k_perpendicular / n_e_pump**2])
    )
    pump_electric /= np.linalg.norm(pump_electric)
    nonlinear_vector = linbo3_3m_nonlinear_polarization(pump_electric)

    rotation = rotation_z(np.deg2rad(29.0))
    pump_standard_deviations_um = np.array([0.24, 0.19, 0.27])
    pump_covariance = (
        rotation
        @ np.diag(pump_standard_deviations_um**2)
        @ rotation.T
    )
    pump_mean_um = np.array([0.08, -0.05, 0.04])
    pump_envelope = AnisotropicGaussianMixture(
        coefficients=[1.0],
        means=[pump_mean_um],
        covariances=[pump_covariance],
    )
    # If g(r) has covariance Sigma, g(r)^2 has covariance Sigma/2.
    nonlinear_envelope = AnisotropicGaussianMixture(
        coefficients=[1.0],
        means=[pump_mean_um],
        covariances=[0.5 * pump_covariance],
    )

    return {
        "n_o_pump": n_o_pump,
        "n_e_pump": n_e_pump,
        "n_o_sh": n_o_sh,
        "n_e_sh": n_e_sh,
        "pump_wavevector": pump_wavevector,
        "pump_electric": pump_electric,
        "nonlinear_vector": nonlinear_vector,
        "pump_envelope": pump_envelope,
        "nonlinear_envelope": nonlinear_envelope,
        "pump_covariance": pump_covariance,
        "pump_mean_um": pump_mean_um,
        "pump_standard_deviations_um": pump_standard_deviations_um,
        "sh_k0": 2.0 * np.pi / SH_WAVELENGTH_UM,
        "epsilon_sh": np.diag([n_o_sh**2, n_o_sh**2, n_e_sh**2]),
    }


def chi2_factorization_relative_l2(problem: dict[str, Any]) -> float:
    """Verify P2(r) = chi2:(e g exp(ikr))(e g exp(ikr)) pointwise."""

    rng = np.random.default_rng(20260722)
    coordinates = rng.uniform(-0.45, 0.45, size=(17, 3))
    pump_scalar = problem["pump_envelope"].density(coordinates) * np.exp(
        1j * (coordinates @ problem["pump_wavevector"])
    )
    direct = np.stack(
        [
            linbo3_3m_nonlinear_polarization(
                problem["pump_electric"] * scalar
            )
            for scalar in pump_scalar
        ]
    )
    factorized = pump_scalar[:, None] ** 2 * problem["nonlinear_vector"][None, :]
    return relative_l2(direct, factorized)


def midpoint_source(
    problem: dict[str, Any],
    *,
    n_xyz: int,
    half_width_um: float = DOMAIN_HALF_WIDTH_UM,
) -> tuple[np.ndarray, np.ndarray, float]:
    spacing = 2.0 * half_width_um / n_xyz
    axis = -half_width_um + (np.arange(n_xyz) + 0.5) * spacing
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    coordinates = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    carrier = 2.0 * problem["pump_wavevector"]
    weights = problem["nonlinear_envelope"].density(coordinates)
    weights *= np.exp(1j * (coordinates @ carrier))
    weights *= spacing**3
    return coordinates, weights, spacing**3


def make_binned_source(
    coordinates: np.ndarray,
    weights: np.ndarray,
    *,
    n_r: int,
    n_z: int,
    n_phi: int,
    half_width_um: float = DOMAIN_HALF_WIDTH_UM,
):
    return make_cylindrical_histogram(
        coordinates,
        atom_weights=weights,
        n_r=n_r,
        n_z=n_z,
        n_phi=n_phi,
        r_max=np.sqrt(2.0) * half_width_um,
        z_range=(-half_width_um, half_width_um),
        hist_dtype=np.complex128,
        backend="numpy",
    )


def negative_outgoing_manifold(
    problem: dict[str, Any],
    u: np.ndarray,
    branch: str,
) -> AxisymmetricManifold:
    if branch == "ordinary":
        q_perp = problem["sh_k0"] * problem["n_o_sh"] * np.sin(u)
    elif branch == "extraordinary":
        q_perp = problem["sh_k0"] * problem["n_e_sh"] * np.sin(u)
    else:
        raise ValueError("branch must be ordinary or extraordinary")
    # ACFO evaluates exp(+i q.r).  Setting q=-k_out converts this to the
    # physical source transform exp(-i k_out.r).  The transverse minus sign is
    # represented by the opposite azimuth on the complete ring.
    q_z = -problem["sh_k0"] * problem["n_o_sh"] * np.cos(u)
    return AxisymmetricManifold(
        u,
        q_perp,
        q_z,
        name=f"negative-outgoing-LiNbO3-{branch}",
        interpretation="dispersion-derived",
        frequency_units="inverse_micrometre",
    )


def analytic_source_spectrum(
    problem: dict[str, Any],
    q_nodes: np.ndarray,
) -> np.ndarray:
    # P2(r)=p0*g(r)^2*exp(+i 2kp.r), so the positive-sign transform at q is
    # G2_hat(q+2kp).  With q=-k_out this is G2_hat(2kp-k_out).
    return problem["nonlinear_envelope"].fourier_nodes(
        q_nodes + 2.0 * problem["pump_wavevector"]
    )


def direct_fourier_nodes(
    coordinates: np.ndarray,
    weights: np.ndarray,
    q_nodes: np.ndarray,
    *,
    chunk: int = 4,
) -> np.ndarray:
    nodes = np.asarray(q_nodes, dtype=np.float64).reshape(-1, 3)
    output = np.empty(nodes.shape[0], dtype=np.complex128)
    for start in range(0, nodes.shape[0], chunk):
        stop = min(nodes.shape[0], start + chunk)
        output[start:stop] = weights @ np.exp(
            1j * (coordinates @ nodes[start:stop].T)
        )
    return output.reshape(np.asarray(q_nodes).shape[:-1])


def apply_green(
    problem: dict[str, Any],
    scalar_spectrum: np.ndarray,
    q_nodes: np.ndarray,
    *,
    nonlinear_vector: np.ndarray | None = None,
) -> np.ndarray:
    source_vector = (
        problem["nonlinear_vector"]
        if nonlinear_vector is None
        else np.asarray(nonlinear_vector, dtype=np.complex128)
    )
    return apply_maxwell_spectral_residue(
        scalar_spectrum,
        -np.asarray(q_nodes),
        source_vector,
        k0=problem["sh_k0"],
        epsilon_tensor=problem["epsilon_sh"],
    )


def evaluate_level(
    problem: dict[str, Any],
    *,
    n_xyz: int,
    n_r: int,
    n_phi: int,
    n_u: int = DEFAULT_N_U,
    u_min: float = U_MIN_RAD,
    u_max: float = U_MAX_RAD,
    direct_operator: bool = False,
) -> tuple[dict[str, Any], dict[str, list[np.ndarray]]]:
    started = time.perf_counter()
    coordinates, voxel_weights, voxel_volume = midpoint_source(
        problem,
        n_xyz=n_xyz,
    )
    binned = make_binned_source(
        coordinates,
        voxel_weights,
        n_r=n_r,
        n_z=n_xyz,
        n_phi=n_phi,
    )
    u = np.linspace(u_min, u_max, n_u)
    phi = np.asarray(binned.beta_centers)
    binned_coordinates: np.ndarray | None = None
    binned_weights: np.ndarray | None = None
    if direct_operator:
        binned_coordinates, _, binned_weights = binned_structure_sources(binned)

    branch_rows: dict[str, Any] = {}
    arrays: dict[str, list[np.ndarray]] = {
        "analytic_scalar": [],
        "acfo_scalar": [],
        "analytic_green": [],
        "acfo_green": [],
        "q_nodes": [],
    }
    for branch in ("ordinary", "extraordinary"):
        manifold = negative_outgoing_manifold(problem, u, branch)
        q_nodes = manifold.target_nodes(phi)
        analytic_scalar = analytic_source_spectrum(problem, q_nodes)
        acfo_scalar = PreparedAxisymmetricOperator(
            binned,
            manifold,
            complex_dtype=np.complex128,
        ).forward(binned.hist)
        analytic_green = apply_green(problem, analytic_scalar, q_nodes)
        acfo_green = apply_green(problem, acfo_scalar, q_nodes)
        residue, pole_diagnostics = maxwell_spectral_residue(
            -q_nodes,
            k0=problem["sh_k0"],
            epsilon_tensor=problem["epsilon_sh"],
            return_diagnostics=True,
        )
        del residue

        row: dict[str, Any] = {
            "q_perp_min": float(np.min(manifold.q_perp)),
            "q_perp_max": float(np.max(manifold.q_perp)),
            "outgoing_kz_min": float(np.min(-manifold.q_z)),
            "outgoing_kz_max": float(np.max(-manifold.q_z)),
            "scalar_acfo_vs_analytic": complex_error_metrics(
                acfo_scalar,
                analytic_scalar,
            ).to_dict(),
            "green_acfo_vs_analytic": complex_error_metrics(
                acfo_green,
                analytic_green,
            ).to_dict(),
            "maxwell_pole": pole_diagnostics,
        }
        if direct_operator:
            assert binned_coordinates is not None and binned_weights is not None
            direct_scalar = direct_axisymmetric_amplitude(
                binned_coordinates,
                manifold,
                phi,
                source_weights=binned_weights,
            )
            direct_green = apply_green(problem, direct_scalar, q_nodes)
            row["scalar_acfo_vs_binned_direct"] = complex_error_metrics(
                acfo_scalar,
                direct_scalar,
            ).to_dict()
            row["green_acfo_vs_binned_direct"] = complex_error_metrics(
                acfo_green,
                direct_green,
            ).to_dict()

        arrays["analytic_scalar"].append(analytic_scalar)
        arrays["acfo_scalar"].append(acfo_scalar)
        arrays["analytic_green"].append(analytic_green)
        arrays["acfo_green"].append(acfo_green)
        arrays["q_nodes"].append(q_nodes)
        branch_rows[branch] = row

    combined_analytic_scalar = np.concatenate(
        [value.ravel() for value in arrays["analytic_scalar"]]
    )
    combined_acfo_scalar = np.concatenate(
        [value.ravel() for value in arrays["acfo_scalar"]]
    )
    combined_analytic_green = np.concatenate(
        [value.ravel() for value in arrays["analytic_green"]]
    )
    combined_acfo_green = np.concatenate(
        [value.ravel() for value in arrays["acfo_green"]]
    )
    level = {
        "n_xyz": n_xyz,
        "n_r": n_r,
        "n_z": n_xyz,
        "n_phi": n_phi,
        "n_u": n_u,
        "q_samples_per_branch": int(n_u * n_phi),
        "cartesian_voxels": int(coordinates.shape[0]),
        "nonzero_cylindrical_bins": int(np.count_nonzero(binned.hist)),
        "voxel_volume_um3": voxel_volume,
        "scalar_acfo_vs_analytic_relative_l2": relative_l2(
            combined_acfo_scalar,
            combined_analytic_scalar,
        ),
        "green_acfo_vs_analytic_relative_l2": relative_l2(
            combined_acfo_green,
            combined_analytic_green,
        ),
        "branches": branch_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return level, arrays


def selected_mode_voxel_reference(
    problem: dict[str, Any],
    *,
    n_xyz: int,
    n_u: int,
    u_min: float,
    u_max: float,
) -> dict[str, Any]:
    coordinates, voxel_weights, voxel_volume = midpoint_source(
        problem,
        n_xyz=n_xyz,
    )
    u = np.linspace(u_min, u_max, n_u)
    azimuths = np.deg2rad(SELECTED_OUTPUT_AZIMUTHS_DEG)
    branch_rows: dict[str, Any] = {}
    all_analytic_scalar: list[np.ndarray] = []
    all_voxel_scalar: list[np.ndarray] = []
    all_analytic_green: list[np.ndarray] = []
    all_voxel_green: list[np.ndarray] = []
    for branch in ("ordinary", "extraordinary"):
        if branch == "ordinary":
            k_perp = problem["sh_k0"] * problem["n_o_sh"] * np.sin(u)
        else:
            k_perp = problem["sh_k0"] * problem["n_e_sh"] * np.sin(u)
        k_z = problem["sh_k0"] * problem["n_o_sh"] * np.cos(u)
        outgoing = np.stack(
            [
                np.column_stack(
                    (
                        k_perp * np.cos(azimuth),
                        k_perp * np.sin(azimuth),
                        k_z,
                    )
                )
                for azimuth in azimuths
            ]
        )
        q_nodes = -outgoing
        analytic_scalar = analytic_source_spectrum(problem, q_nodes)
        voxel_scalar = direct_fourier_nodes(
            coordinates,
            voxel_weights,
            q_nodes,
        )
        analytic_green = apply_green(problem, analytic_scalar, q_nodes)
        voxel_green = apply_green(problem, voxel_scalar, q_nodes)
        branch_rows[branch] = {
            "scalar_voxel_vs_analytic": complex_error_metrics(
                voxel_scalar,
                analytic_scalar,
            ).to_dict(),
            "green_voxel_vs_analytic": complex_error_metrics(
                voxel_green,
                analytic_green,
            ).to_dict(),
        }
        all_analytic_scalar.append(analytic_scalar)
        all_voxel_scalar.append(voxel_scalar)
        all_analytic_green.append(analytic_green)
        all_voxel_green.append(voxel_green)
    return {
        "n_xyz": n_xyz,
        "voxel_volume_um3": voxel_volume,
        "output_azimuths_deg": list(SELECTED_OUTPUT_AZIMUTHS_DEG),
        "n_u": n_u,
        "target_count": 2 * len(SELECTED_OUTPUT_AZIMUTHS_DEG) * n_u,
        "scalar_voxel_vs_analytic_relative_l2": relative_l2(
            np.concatenate([value.ravel() for value in all_voxel_scalar]),
            np.concatenate([value.ravel() for value in all_analytic_scalar]),
        ),
        "green_voxel_vs_analytic_relative_l2": relative_l2(
            np.concatenate([value.ravel() for value in all_voxel_green]),
            np.concatenate([value.ravel() for value in all_analytic_green]),
        ),
        "branches": branch_rows,
    }


def resolvent_check(problem: dict[str, Any]) -> dict[str, Any]:
    u = np.array([0.54])
    phi = np.array([0.37])
    rows: dict[str, Any] = {}
    for branch in ("ordinary", "extraordinary"):
        q_node = negative_outgoing_manifold(problem, u, branch).target_nodes(phi)
        k_node = -q_node
        analytic = maxwell_spectral_residue(
            k_node,
            k0=problem["sh_k0"],
            epsilon_tensor=problem["epsilon_sh"],
        )
        eta_rows = []
        for eta in (1e-5, 1e-6, 1e-7):
            numerical = maxwell_resolvent_residue(
                k_node,
                k0=problem["sh_k0"],
                epsilon_tensor=problem["epsilon_sh"],
                eta=eta,
            )
            eta_rows.append(
                {"eta": eta, "relative_l2": relative_l2(numerical, analytic)}
            )
        order = float(
            np.polyfit(
                np.log([row["eta"] for row in eta_rows]),
                np.log([row["relative_l2"] for row in eta_rows]),
                1,
            )[0]
        )
        rows[branch] = {
            "rows": eta_rows,
            "convergence_order": order,
            "finest_relative_l2": eta_rows[-1]["relative_l2"],
        }
    return rows


def wrong_tensor_control(
    problem: dict[str, Any],
    arrays: dict[str, list[np.ndarray]],
) -> dict[str, Any]:
    wrong_vector = linbo3_3m_nonlinear_polarization(
        problem["pump_electric"],
        d33_pm_per_v=0.0,
    )
    correct_fields: list[np.ndarray] = []
    wrong_fields: list[np.ndarray] = []
    for analytic_scalar, q_nodes, correct_field in zip(
        arrays["analytic_scalar"],
        arrays["q_nodes"],
        arrays["analytic_green"],
        strict=True,
    ):
        wrong_field = apply_green(
            problem,
            analytic_scalar,
            q_nodes,
            nonlinear_vector=wrong_vector,
        )
        correct_fields.append(correct_field)
        wrong_fields.append(wrong_field)
    correct = np.concatenate([value.ravel() for value in correct_fields])
    wrong = np.concatenate([value.ravel() for value in wrong_fields])
    gain = fit_global_gain(wrong, correct)
    return {
        "definition": "d33 set to zero; one global complex gain allowed over both branches",
        "true_nonlinear_vector_pm_per_v_scale": [
            complex_pair(value) for value in problem["nonlinear_vector"]
        ],
        "wrong_nonlinear_vector_pm_per_v_scale": [
            complex_pair(value) for value in wrong_vector
        ],
        "raw_relative_l2": relative_l2(wrong, correct),
        "global_gain": complex_pair(gain),
        "gain_aligned_relative_l2": relative_l2(gain * wrong, correct),
        "per_angle_or_per_branch_fitting": False,
    }


def parse_levels(value: str) -> tuple[int, ...]:
    levels = tuple(int(item) for item in value.split(",") if item.strip())
    if len(levels) < 2 or any(level <= 0 for level in levels):
        raise argparse.ArgumentTypeError("levels must contain at least two positive integers")
    if tuple(sorted(set(levels))) != levels:
        raise argparse.ArgumentTypeError("levels must be strictly increasing")
    return levels


def run_validation(
    *,
    levels: Iterable[int] = (48, 64, 80),
    n_u: int = DEFAULT_N_U,
    operator_n_xyz: int = 24,
) -> dict[str, Any]:
    level_values = tuple(int(value) for value in levels)
    problem = make_physical_problem()
    chi2_error = chi2_factorization_relative_l2(problem)

    operator_level, _ = evaluate_level(
        problem,
        n_xyz=operator_n_xyz,
        n_r=int(round(1.5 * operator_n_xyz)),
        n_phi=3 * operator_n_xyz,
        n_u=n_u,
        direct_operator=True,
    )
    convergence_levels: list[dict[str, Any]] = []
    finest_arrays: dict[str, list[np.ndarray]] | None = None
    for n_xyz in level_values:
        level, arrays = evaluate_level(
            problem,
            n_xyz=n_xyz,
            n_r=int(round(1.5 * n_xyz)),
            n_phi=3 * n_xyz,
            n_u=n_u,
        )
        convergence_levels.append(level)
        finest_arrays = arrays
    assert finest_arrays is not None

    selected_voxel = selected_mode_voxel_reference(
        problem,
        n_xyz=level_values[-1],
        n_u=n_u,
        u_min=U_MIN_RAD,
        u_max=U_MAX_RAD,
    )
    wrong_tensor = wrong_tensor_control(problem, finest_arrays)
    resolvent = resolvent_check(problem)

    total_errors = np.array(
        [level["green_acfo_vs_analytic_relative_l2"] for level in convergence_levels]
    )
    empirical_order = float(
        np.log(total_errors[0] / total_errors[-1])
        / np.log(level_values[-1] / level_values[0])
    )
    operator_green_max = max(
        operator_level["branches"][branch]["green_acfo_vs_binned_direct"][
            "relative_l2"
        ]
        for branch in ("ordinary", "extraordinary")
    )
    max_pole_residual = max(
        level["branches"][branch]["maxwell_pole"][
            "max_normalized_null_residual"
        ]
        for level in convergence_levels
        for branch in ("ordinary", "extraordinary")
    )
    finest_resolvent = max(
        resolvent[branch]["finest_relative_l2"]
        for branch in ("ordinary", "extraordinary")
    )
    minimum_resolvent_order = min(
        resolvent[branch]["convergence_order"]
        for branch in ("ordinary", "extraordinary")
    )
    gates = {
        "chi2_factorization_l2_le_1e_13": chi2_error <= 1e-13,
        "maxwell_pole_residual_le_1e_12": max_pole_residual <= 1e-12,
        "resolvent_limit_l2_le_1e_5": finest_resolvent <= 1e-5,
        "resolvent_first_order_ge_0_8": minimum_resolvent_order >= 0.8,
        "operator_green_l2_le_1e_11": operator_green_max <= 1e-11,
        "finest_voxel_green_l2_le_1e_8": selected_voxel[
            "green_voxel_vs_analytic_relative_l2"
        ]
        <= 1e-8,
        "finest_acfo_green_l2_le_2pct": convergence_levels[-1][
            "green_acfo_vs_analytic_relative_l2"
        ]
        <= 0.02,
        "acfo_green_error_strictly_decreases": bool(np.all(np.diff(total_errors) < 0.0)),
        "empirical_resolution_order_ge_1": empirical_order >= 1.0,
        "wrong_d33_gain_aligned_l2_ge_20pct": wrong_tensor[
            "gain_aligned_relative_l2"
        ]
        >= 0.20,
    }
    return {
        "schema": "general-curvature-analytic-nonlinear-oracle-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "prescribed fixed-polarization anisotropic-Gaussian undepleted pump; "
            "LiNbO3 3m nonlinear contraction; analytic continuous source Fourier "
            "transform; homogeneous lossless dyadic Maxwell simple-pole fields on "
            "ordinary and extraordinary curved bulk branches"
        ),
        "fourier_contract": {
            "acfo": "integral P2(r) exp(+i q dot r) dr",
            "physical_source": "integral P2(r) exp(-i k_out dot r) dr",
            "manifold_mapping": "q = -k_out",
            "source_carrier": "P2 contains exp(+i 2 k_pump dot r)",
            "analytic_argument": "q + 2 k_pump = 2 k_pump - k_out",
        },
        "normalization_contract": {
            "coordinates": "micrometres",
            "wavevectors": "inverse micrometres",
            "nonlinear_coefficients": "pm/V scale retained for relative component ratios",
            "omitted_global_factors": [
                "2 epsilon_0 convention factor",
                "-i 2 omega polarization-to-current factor",
                "overall Green-function/radiation normalization",
            ],
            "supported_normalization": "relative complex vector amplitudes with no gain fit in positive comparisons",
            "not_supported_normalization": "absolute SH power or conversion efficiency",
        },
        "material": {
            "name": "5 mol% MgO-doped congruent LiNbO3",
            "temperature_c": TEMPERATURE_C,
            "pump_wavelength_um": PUMP_WAVELENGTH_UM,
            "sh_wavelength_um": SH_WAVELENGTH_UM,
            "n_o_pump": problem["n_o_pump"],
            "n_e_pump": problem["n_e_pump"],
            "n_o_sh": problem["n_o_sh"],
            "n_e_sh": problem["n_e_sh"],
            "epsilon_sh": problem["epsilon_sh"].tolist(),
            "d22_pm_per_v": 4.08,
            "d31_pm_per_v": -4.4,
            "d33_pm_per_v": -25.0,
        },
        "manufactured_pump": {
            "branch": "extraordinary",
            "surface_parameter_deg": PUMP_SURFACE_PARAMETER_DEG,
            "azimuth_deg": PUMP_AZIMUTH_DEG,
            "wavevector_per_um": problem["pump_wavevector"].tolist(),
            "electric_unit_vector": problem["pump_electric"].tolist(),
            "nonlinear_vector_pm_per_v_scale": [
                complex_pair(value) for value in problem["nonlinear_vector"]
            ],
            "envelope_mean_um": problem["pump_mean_um"].tolist(),
            "envelope_standard_deviations_um": problem[
                "pump_standard_deviations_um"
            ].tolist(),
            "envelope_covariance_um2": problem["pump_covariance"].tolist(),
            "nonlinear_envelope_covariance": "pump covariance divided by two",
            "domain_half_width_um": DOMAIN_HALF_WIDTH_UM,
        },
        "angular_sampling": {
            "u_min_rad": U_MIN_RAD,
            "u_max_rad": U_MAX_RAD,
            "n_u": n_u,
            "full_ring_on_each_convergence_level": True,
            "selected_voxel_reference_azimuths_deg": list(
                SELECTED_OUTPUT_AZIMUTHS_DEG
            ),
        },
        "error_decomposition": {
            "operator": "ACFO minus direct Cartesian sum over the same nonzero cylindrical bin centers",
            "voxel_quadrature": "Cartesian midpoint source sum minus continuous analytic Gaussian transform on selected physical modes",
            "total": "ACFO from the finite cylindrical histogram minus the continuous analytic Gaussian transform",
        },
        "chi2_factorization_relative_l2": chi2_error,
        "operator_level": operator_level,
        "convergence_levels": convergence_levels,
        "selected_mode_voxel_reference": selected_voxel,
        "convergence": {
            "green_total_relative_l2": total_errors.tolist(),
            "strictly_decreasing": bool(np.all(np.diff(total_errors) < 0.0)),
            "empirical_resolution_order": empirical_order,
        },
        "maxwell_resolvent_check": resolvent,
        "wrong_tensor_control": wrong_tensor,
        "gates": gates,
        "passed": all(gates.values()),
        "claim_boundary": {
            "supported": (
                "exact-source and discretization-converged relative complex vector "
                "fields for a prescribed homogeneous-bulk undepleted Gaussian-pump "
                "LiNbO3 SHG problem on nonspherical ordinary/extraordinary branches"
            ),
            "not_supported": [
                "self-consistent pump propagation or spatially varying pump polarization",
                "pump depletion, back-conversion, or coupled nonlinear propagation",
                "finite-crystal interfaces, Fresnel coupling, periodic poling, or sharp domain walls",
                "lossy non-Hermitian media or degenerate/grazing poles",
                "absolute SH power or conversion efficiency",
                "FDTD accuracy or agreement",
            ],
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    levels = payload["convergence_levels"]
    operator = payload["operator_level"]
    selected = payload["selected_mode_voxel_reference"]
    wrong = payload["wrong_tensor_control"]
    lines = [
        "# General-curvature analytic nonlinear-polarization oracle",
        "",
        "이 검증은 prescribed anisotropic-Gaussian pump, LiNbO3 3m contraction, exact continuous Fourier transform, dyadic anisotropic Maxwell residue를 하나의 pure-NumPy reference chain으로 연결한다. FDTD는 사용하지 않는다.",
        "",
        "## Fourier and normalization contract",
        "",
        "- ACFO convention: `integral P2(r) exp(+i q.r) dr`",
        "- physical source convention: `integral P2(r) exp(-i k_out.r) dr`",
        "- mapping: `q = -k_out`; analytic Gaussian argument: `2 k_pump - k_out`",
        "- reported field: relative complex vector amplitude only; absolute SH efficiency is outside scope",
        "",
        "## Exact algebra and operator gate",
        "",
        f"- chi2 pointwise factorization L2: `{payload['chi2_factorization_relative_l2']:.3e}`",
        f"- reduced grid: `{operator['n_xyz']}^3`, cylindrical `{operator['n_r']} x {operator['n_z']} x {operator['n_phi']}`",
        "",
        "| branch | scalar ACFO/direct | Green ACFO/direct |",
        "|---|---:|---:|",
    ]
    for branch in ("ordinary", "extraordinary"):
        row = operator["branches"][branch]
        lines.append(
            f"| {branch} | {row['scalar_acfo_vs_binned_direct']['relative_l2']:.3e} | "
            f"{row['green_acfo_vs_binned_direct']['relative_l2']:.3e} |"
        )
    lines.extend(
        [
            "",
            "## Continuous analytic-reference convergence",
            "",
            "| Cartesian | cylindrical | full-ring targets | scalar total L2 | Green total L2 | elapsed s |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for level in levels:
        lines.append(
            f"| {level['n_xyz']}^3 | {level['n_r']} x {level['n_z']} x {level['n_phi']} | "
            f"{2 * level['q_samples_per_branch']} | "
            f"{level['scalar_acfo_vs_analytic_relative_l2']:.3e} | "
            f"{level['green_acfo_vs_analytic_relative_l2']:.3e} | "
            f"{level['elapsed_seconds']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"- selected-mode finest voxel-direct/analytic Green L2: `{selected['green_voxel_vs_analytic_relative_l2']:.3e}`",
            f"- empirical resolution order: `{payload['convergence']['empirical_resolution_order']:.3f}`",
            f"- wrong-d33 one-global-gain residual: `{wrong['gain_aligned_relative_l2']:.3%}`",
            "",
            "## Gates",
            "",
            "| gate | pass |",
            "|---|:---:|",
        ]
    )
    for name, passed in payload["gates"].items():
        lines.append(f"| `{name}` | {passed} |")
    lines.extend(
        [
            "",
            f"Overall pass: **{payload['passed']}**",
            "",
            "Supported claim: exact-source and discretization-converged relative complex vector bulk fields for this prescribed undepleted-pump LiNbO3 problem. FDTD, interfaces, depletion, loss, periodic poling, and absolute conversion efficiency are not claimed.",
            "",
            "Reproduce with:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\validate_general_curvature_analytic_nonlinear_oracle.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a pure-NumPy analytic nonlinear-polarization Green oracle."
    )
    parser.add_argument("--levels", type=parse_levels, default=(48, 64, 80))
    parser.add_argument("--n-u", type=int, default=DEFAULT_N_U)
    parser.add_argument("--operator-n", type=int, default=24)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=(
            ROOT
            / "benchmark_results"
            / "general_curvature_analytic_nonlinear_oracle_20260722.json"
        ),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "general_curvature_analytic_nonlinear_oracle_20260722_ko.md"
        ),
    )
    args = parser.parse_args()
    if args.n_u < 3:
        raise ValueError("n-u must be at least three")
    if args.operator_n < 8:
        raise ValueError("operator-n must be at least eight")

    payload = run_validation(
        levels=args.levels,
        n_u=args.n_u,
        operator_n_xyz=args.operator_n,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": payload["passed"],
                "gates": payload["gates"],
                "chi2_factorization_relative_l2": payload[
                    "chi2_factorization_relative_l2"
                ],
                "convergence": payload["convergence"],
                "selected_mode_voxel_reference": payload[
                    "selected_mode_voxel_reference"
                ]["green_voxel_vs_analytic_relative_l2"],
                "wrong_tensor_gain_aligned_relative_l2": payload[
                    "wrong_tensor_control"
                ]["gain_aligned_relative_l2"],
            },
            indent=2,
        )
    )
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
