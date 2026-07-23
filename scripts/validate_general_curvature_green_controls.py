from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from validate_uniaxial_vector_born_direct import (  # noqa: E402
    direct_binned_fourier,
    midpoint_domain_source,
    outgoing_manifold,
)
from waxs_cake import (  # noqa: E402
    PreparedAxisymmetricOperator,
    gayer_5mol_mgo_cln_index,
    make_cylindrical_histogram,
    maxwell_spectral_residue,
)


def relative_l2(model: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(np.asarray(reference).ravel()))
    if denominator == 0.0:
        return 0.0 if np.linalg.norm(np.asarray(model).ravel()) == 0.0 else float("inf")
    return float(
        np.linalg.norm((np.asarray(model) - np.asarray(reference)).ravel()) / denominator
    )


def fit_global_gain(model: np.ndarray, reference: np.ndarray) -> complex:
    denominator = np.vdot(np.asarray(model).ravel(), np.asarray(model).ravel())
    if denominator == 0.0:
        raise ValueError("cannot fit a gain to a zero model")
    return complex(np.vdot(np.asarray(model).ravel(), np.asarray(reference).ravel()) / denominator)


def complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def apply_residue(residue: np.ndarray, vector_amplitude: np.ndarray) -> np.ndarray:
    amplitude = np.asarray(vector_amplitude, dtype=np.complex128)
    if residue.shape[:-2] != amplitude.shape[:-1] or residue.shape[-2:] != (3, 3):
        raise ValueError("residue and vector amplitude shapes do not match")
    if amplitude.shape[-1] != 3:
        raise ValueError("vector amplitude must end in a Cartesian component axis")
    return np.einsum("...ij,...j->...i", residue, amplitude, optimize=True)


def make_binned(
    coords: np.ndarray,
    weights: np.ndarray,
    *,
    n: int,
    n_phi: int,
    half_width_um: float,
):
    return make_cylindrical_histogram(
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


def scalar_pair(binned, manifold) -> tuple[np.ndarray, np.ndarray]:
    acfo = PreparedAxisymmetricOperator(
        binned,
        manifold,
        complex_dtype=np.complex128,
    ).forward(binned.hist)
    direct = direct_binned_fourier(binned, manifold)
    return acfo, direct


def _rotation_y_90() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


def run_controls(*, n: int, n_phi: int, n_u: int) -> dict[str, Any]:
    wavelength_pump_um = 1.064
    wavelength_sh_um = 0.532
    temperature_c = 24.5
    n_e_pump = gayer_5mol_mgo_cln_index(
        wavelength_pump_um,
        "extraordinary",
        temperature_c=temperature_c,
    )
    n_o_sh = gayer_5mol_mgo_cln_index(
        wavelength_sh_um,
        "ordinary",
        temperature_c=temperature_c,
    )
    n_e_sh = gayer_5mol_mgo_cln_index(
        wavelength_sh_um,
        "extraordinary",
        temperature_c=temperature_c,
    )
    k0 = 2.0 * np.pi / wavelength_sh_um
    epsilon = np.diag([n_o_sh**2, n_o_sh**2, n_e_sh**2])
    half_width_um = 2.0
    # Avoid the degenerate optic axis and exact grazing point in the positive
    # test while still spanning a wide angular range.
    u = np.linspace(0.06, 1.42, n_u)
    extraordinary = outgoing_manifold(
        u,
        wavelength_um=wavelength_sh_um,
        n_o=n_o_sh,
        n_e=n_e_sh,
        branch="extraordinary",
    )
    sphere = outgoing_manifold(
        u,
        wavelength_um=wavelength_sh_um,
        n_o=n_o_sh,
        n_e=n_e_sh,
        branch="ordinary",
    )

    pump_k = 2.0 * np.pi * n_e_pump / wavelength_pump_um
    coords, base_weights, voxel_volume = midpoint_domain_source(
        n=n,
        half_width_um=half_width_um,
        pump_wave_number_per_um=pump_k,
    )
    base_binned = make_binned(
        coords,
        base_weights,
        n=n,
        n_phi=n_phi,
        half_width_um=half_width_um,
    )
    extra_acfo_scalar, extra_direct_scalar = scalar_pair(base_binned, extraordinary)
    _, sphere_direct_scalar = scalar_pair(base_binned, sphere)
    phi = np.asarray(base_binned.beta_centers)
    extra_nodes = extraordinary.target_nodes(phi)
    sphere_nodes = sphere.target_nodes(phi)
    extra_residue, extra_diagnostics = maxwell_spectral_residue(
        extra_nodes,
        k0=k0,
        epsilon_tensor=epsilon,
        return_diagnostics=True,
    )
    sphere_residue, sphere_diagnostics = maxwell_spectral_residue(
        sphere_nodes,
        k0=k0,
        epsilon_tensor=epsilon,
        return_diagnostics=True,
    )

    source_vector = np.array([0.31 + 0.08j, -0.27 + 0.14j, 0.83 - 0.05j])
    correct_vector_amplitude = extra_direct_scalar[..., None] * source_vector
    acfo_vector_amplitude = extra_acfo_scalar[..., None] * source_vector
    forced_vector_amplitude = sphere_direct_scalar[..., None] * source_vector
    correct_field = apply_residue(extra_residue, correct_vector_amplitude)
    acfo_field = apply_residue(extra_residue, acfo_vector_amplitude)
    # Deliberately apply the correct extraordinary Green residue after sampling
    # the object on spherical nodes. This isolates the geometry substitution;
    # it is a negative control, not a claim that the sphere is an extraordinary pole.
    forced_sphere_field = apply_residue(extra_residue, forced_vector_amplitude)
    forced_gain = fit_global_gain(forced_sphere_field, correct_field)
    correct_error = relative_l2(acfo_field, correct_field)
    forced_raw_error = relative_l2(forced_sphere_field, correct_field)
    forced_gain_error = relative_l2(forced_gain * forced_sphere_field, correct_field)

    # Rotate the optic axis z -> x and rotate k, epsilon, and source together.
    # The residue coordinate changes from d/dk_z to d/dk_x.
    rotation = _rotation_y_90()
    rotated_nodes = np.einsum("ij,...j->...i", rotation, extra_nodes)
    rotated_epsilon = rotation @ epsilon @ rotation.T
    rotated_residue, rotated_diagnostics = maxwell_spectral_residue(
        rotated_nodes,
        k0=k0,
        epsilon_tensor=rotated_epsilon,
        propagation_axis=0,
        return_diagnostics=True,
    )
    expected_rotated_residue = np.einsum(
        "ia,...ab,jb->...ij",
        rotation,
        extra_residue,
        rotation,
        optimize=True,
    )
    rotated_source = rotation @ source_vector
    rotated_field = apply_residue(
        rotated_residue,
        extra_direct_scalar[..., None] * rotated_source,
    )
    expected_rotated_field = np.einsum("ij,...j->...i", rotation, correct_field)

    # A genuinely spatially varying vector source: the three Cartesian source
    # components are not a common scalar pattern times one constant vector.
    scaled = coords / half_width_um
    component_weights = np.column_stack(
        (
            base_weights * (0.62 + 0.25 * scaled[:, 0] + 0.14j * scaled[:, 1]),
            base_weights * (-0.38 + 0.22 * scaled[:, 2] + 0.11j * scaled[:, 0]),
            base_weights * (0.51 + 0.18 * scaled[:, 1] - 0.09j * scaled[:, 2]),
        )
    )
    singular_values = np.linalg.svd(component_weights, compute_uv=False)
    component_binned = [
        make_binned(
            coords,
            component_weights[:, component],
            n=n,
            n_phi=n_phi,
            half_width_um=half_width_um,
        )
        for component in range(3)
    ]
    component_pairs = [scalar_pair(binned, extraordinary) for binned in component_binned]
    vector_acfo = np.stack([pair[0] for pair in component_pairs], axis=-1)
    vector_direct = np.stack([pair[1] for pair in component_pairs], axis=-1)
    spatial_field_acfo = apply_residue(extra_residue, vector_acfo)
    spatial_field_direct = apply_residue(extra_residue, vector_direct)

    # Ordinary polarization is transverse and has no z component in this
    # axis-aligned uniaxial medium. A z-only source is therefore forbidden.
    allowed_source = np.array([0.0, 1.0, 0.0], dtype=np.complex128)
    forbidden_source = np.array([0.0, 0.0, 1.0], dtype=np.complex128)
    allowed_field = apply_residue(
        sphere_residue,
        sphere_direct_scalar[..., None] * allowed_source,
    )
    forbidden_field = apply_residue(
        sphere_residue,
        sphere_direct_scalar[..., None] * forbidden_source,
    )
    selection_suppression = float(
        np.linalg.norm(forbidden_field.ravel()) / np.linalg.norm(allowed_field.ravel())
    )

    guard_results: dict[str, Any] = {}
    optic_axis_node = np.array([0.0, 0.0, k0 * n_o_sh])
    try:
        maxwell_spectral_residue(
            optic_axis_node,
            k0=k0,
            epsilon_tensor=epsilon,
        )
    except ValueError as exc:
        guard_results["degenerate_optic_axis_rejected"] = True
        guard_results["degenerate_optic_axis_message"] = str(exc)
    else:
        guard_results["degenerate_optic_axis_rejected"] = False
        guard_results["degenerate_optic_axis_message"] = ""

    grazing_node = np.array([k0 * n_e_sh, 0.0, 0.0])
    try:
        maxwell_spectral_residue(
            grazing_node,
            k0=k0,
            epsilon_tensor=epsilon,
        )
    except ValueError as exc:
        guard_results["grazing_derivative_pole_rejected"] = True
        guard_results["grazing_derivative_message"] = str(exc)
    else:
        guard_results["grazing_derivative_pole_rejected"] = False
        guard_results["grazing_derivative_message"] = ""

    metrics = {
        "correct_extraordinary": {
            "scalar_acfo_vs_direct_relative_l2": relative_l2(
                extra_acfo_scalar,
                extra_direct_scalar,
            ),
            "green_field_acfo_vs_direct_relative_l2": correct_error,
            "maxwell_pole": extra_diagnostics,
        },
        "forced_sphere_geometry": {
            "definition": "spherical Fourier sampling substituted before the unchanged extraordinary Green residue",
            "raw_relative_l2": forced_raw_error,
            "single_global_gain": complex_pair(forced_gain),
            "single_global_gain_relative_l2": forced_gain_error,
            "wrong_to_correct_error_ratio": float(forced_gain_error / correct_error),
            "per_angle_or_per_branch_fitting": False,
        },
        "axis_rotation_covariance": {
            "rotation": rotation.tolist(),
            "optic_axis": "z to x",
            "propagation_derivative": "d/dk_z to d/dk_x",
            "residue_relative_l2": relative_l2(
                rotated_residue,
                expected_rotated_residue,
            ),
            "field_relative_l2": relative_l2(
                rotated_field,
                expected_rotated_field,
            ),
            "rotated_maxwell_pole": rotated_diagnostics,
        },
        "spatially_varying_vector_source": {
            "source_matrix_singular_values": [float(value) for value in singular_values],
            "second_to_first_singular_value_ratio": float(
                singular_values[1] / singular_values[0]
            ),
            "vector_fourier_acfo_vs_direct_relative_l2": relative_l2(
                vector_acfo,
                vector_direct,
            ),
            "green_field_acfo_vs_direct_relative_l2": relative_l2(
                spatial_field_acfo,
                spatial_field_direct,
            ),
        },
        "ordinary_selection_control": {
            "allowed_source": "y",
            "forbidden_source": "z",
            "forbidden_to_allowed_field_norm_ratio": selection_suppression,
            "ordinary_maxwell_pole": sphere_diagnostics,
        },
        "simple_pole_guards": guard_results,
    }
    return {
        "material": {
            "name": "5 mol% MgO-doped congruent LiNbO3",
            "temperature_c": temperature_c,
            "pump_wavelength_um": wavelength_pump_um,
            "sh_wavelength_um": wavelength_sh_um,
            "n_e_pump": n_e_pump,
            "n_o_sh": n_o_sh,
            "n_e_sh": n_e_sh,
            "epsilon_tensor": epsilon.tolist(),
        },
        "problem": {
            "cartesian_shape": [n, n, n],
            "cylindrical_shape": [n, n, n_phi],
            "active_cartesian_voxels": int(coords.shape[0]),
            "base_nonzero_cylindrical_bins": int(np.count_nonzero(base_binned.hist)),
            "voxel_volume_um3": voxel_volume,
            "u_min_rad": float(u[0]),
            "u_max_rad": float(u[-1]),
            "n_u": n_u,
            "n_phi": n_phi,
            "q_samples": int(n_u * n_phi),
        },
        "metrics": metrics,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    correct = metrics["correct_extraordinary"]
    forced = metrics["forced_sphere_geometry"]
    rotation = metrics["axis_rotation_covariance"]
    spatial = metrics["spatially_varying_vector_source"]
    selection = metrics["ordinary_selection_control"]
    guards = metrics["simple_pole_guards"]
    problem = payload["problem"]
    lines = [
        "# General-curvature R3 vector Green-tensor publication controls",
        "",
        "Cartesian Maxwell wave operator의 dyadic spectral residue와 direct Cartesian exponent sum을 물리 reference로 사용했다. ACFO는 Fourier sampling에만 사용되며 reference의 pole normalization에는 관여하지 않는다.",
        "",
        f"- object: `{problem['cartesian_shape']}` Cartesian to `{problem['cylindrical_shape']}` cylindrical",
        f"- angular range: `{problem['u_min_rad']:.2f}` to `{problem['u_max_rad']:.2f}` rad; q samples `{problem['q_samples']}`",
        "",
        "| control | metric | value | pass |",
        "|---|---|---:|:---:|",
        f"| correct extraordinary | Green ACFO/direct L2 | {correct['green_field_acfo_vs_direct_relative_l2']:.3e} | {payload['gates']['correct_green_l2']} |",
        f"| forced sphere | single-global-gain L2 | {forced['single_global_gain_relative_l2']:.3e} | {payload['gates']['forced_sphere_detected']} |",
        f"| axis rotation z to x | residue covariance L2 | {rotation['residue_relative_l2']:.3e} | {payload['gates']['rotation_covariance']} |",
        f"| spatial vector source | Green ACFO/direct L2 | {spatial['green_field_acfo_vs_direct_relative_l2']:.3e} | {payload['gates']['spatial_vector_source']} |",
        f"| ordinary selection | forbidden/allowed norm | {selection['forbidden_to_allowed_field_norm_ratio']:.3e} | {payload['gates']['ordinary_selection']} |",
        f"| simple-pole guards | optic-axis and grazing rejected | {guards['degenerate_optic_axis_rejected']} / {guards['grazing_derivative_pole_rejected']} | {payload['gates']['simple_pole_guards']} |",
        "",
        f"Overall pass: **{payload['passed']}**",
        "",
        "Forced-sphere control은 잘못된 spherical Fourier nodes만 대입하고 extraordinary Green residue는 그대로 둔 geometry-isolation negative control이다. 한 개의 global complex gain만 허용했으며 angle별 또는 branch별 fitting은 사용하지 않았다.",
        "",
        "지원되는 주장: homogeneous, lossless Hermitian uniaxial medium의 nondegenerate simple bulk pole에서 spatially varying vector first-Born source에 대한 prepared forward evaluation. Interface, finite-crystal Fresnel coupling, multiple scattering, pump depletion 및 full nonlinear propagation은 포함하지 않는다.",
        "",
        "Reproduce with:",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe scripts\\validate_general_curvature_green_controls.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--n-phi", type=int, default=64)
    parser.add_argument("--n-u", type=int, default=16)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "benchmark_results" / "general_curvature_r3_green_controls.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "docs" / "general_curvature_r3_green_controls_ko.md",
    )
    args = parser.parse_args()

    result = run_controls(n=args.n, n_phi=args.n_phi, n_u=args.n_u)
    metrics = result["metrics"]
    gates = {
        "correct_green_l2": metrics["correct_extraordinary"][
            "green_field_acfo_vs_direct_relative_l2"
        ]
        <= 1e-10,
        "forced_sphere_detected": metrics["forced_sphere_geometry"][
            "single_global_gain_relative_l2"
        ]
        >= 0.20,
        "rotation_covariance": max(
            metrics["axis_rotation_covariance"]["residue_relative_l2"],
            metrics["axis_rotation_covariance"]["field_relative_l2"],
        )
        <= 1e-10,
        "spatial_vector_source": metrics["spatially_varying_vector_source"][
            "green_field_acfo_vs_direct_relative_l2"
        ]
        <= 1e-10
        and metrics["spatially_varying_vector_source"][
            "second_to_first_singular_value_ratio"
        ]
        >= 0.05,
        "ordinary_selection": metrics["ordinary_selection_control"][
            "forbidden_to_allowed_field_norm_ratio"
        ]
        <= 1e-12,
        "simple_pole_guards": metrics["simple_pole_guards"][
            "degenerate_optic_axis_rejected"
        ]
        and metrics["simple_pole_guards"]["grazing_derivative_pole_rejected"],
    }
    payload: dict[str, Any] = {
        "schema": "general-curvature-r3-green-publication-controls-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "homogeneous lossless-Hermitian uniaxial vector first-Born fields at nondegenerate simple Maxwell bulk poles",
        "reference": {
            "fourier_amplitude": "direct Cartesian complex exponent sum over cylindrical bin centers",
            "field_normalization": "Cartesian Maxwell wave-operator spectral residue",
            "shared_acfo_harmonic_factorization": False,
            "per_angle_or_per_branch_fitting": False,
        },
        **result,
        "gates": gates,
        "passed": all(gates.values()),
        "claim_boundary": {
            "supported": "prepared forward evaluation on a physically nonspherical uniaxial branch, including spatially varying vector sources and rotated material axes",
            "not_supported": "lossy non-Hermitian media, degenerate/grazing poles under the selected residue coordinate, interfaces, multiple scattering, pump depletion, or full nonlinear propagation",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"passed": payload["passed"], "gates": gates, "metrics": metrics}, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
