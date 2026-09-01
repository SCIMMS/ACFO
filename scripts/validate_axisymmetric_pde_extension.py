from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from waxs_cake import (  # noqa: E402
    AxisymmetricManifold,
    AxisymmetricPDEPair,
    parameter_trapezoid_weights,
    surface_radial_weights,
)


@dataclass(frozen=True)
class PDECase:
    name: str
    label: str
    manifold: AxisymmetricManifold
    manifold_weights: np.ndarray
    kind: str
    parameters: dict[str, float]

    def shell_terms(self, manifold: AxisymmetricManifold | None = None) -> tuple[np.ndarray, np.ndarray]:
        geometry = self.manifold if manifold is None else manifold
        qp = geometry.q_perp
        qz = geometry.q_z
        if self.kind == "shifted_helmholtz":
            k = self.parameters["k"]
            numerator = qp * qp + qz * qz + 2.0 * k * qz
            scale = qp * qp + qz * qz + 2.0 * k * np.abs(qz)
        elif self.kind == "paraxial":
            k = self.parameters["k"]
            numerator = qp * qp + 2.0 * k * qz
            scale = qp * qp + 2.0 * k * np.abs(qz)
        elif self.kind == "ellipsoid":
            a = self.parameters["a"]
            c = self.parameters["c"]
            radial = qp * qp / (a * a)
            axial = qz * qz / (c * c)
            numerator = radial + axial - 1.0
            scale = radial + axial + 1.0
        else:  # pragma: no cover - construction below freezes known cases.
            raise ValueError(f"unknown PDE case kind: {self.kind}")
        return numerator, np.maximum(scale, np.finfo(np.float64).tiny)

    def normalized_shell_residual(self, manifold: AxisymmetricManifold | None = None) -> float:
        numerator, scale = self.shell_terms(manifold)
        return float(np.max(np.abs(numerator) / scale))

    def off_shell_manifold(self) -> AxisymmetricManifold:
        frequency_scale = max(
            1.0,
            float(np.max(self.manifold.q_perp)),
            float(np.max(np.abs(self.manifold.q_z))),
        )
        return AxisymmetricManifold(
            self.manifold.u,
            self.manifold.q_perp,
            self.manifold.q_z + 1e-3 * frequency_scale,
            name=f"{self.name}-off-shell",
            interpretation="dispersion-derived",
        )

    def sign_flip_manifold(self) -> AxisymmetricManifold:
        return AxisymmetricManifold(
            self.manifold.u,
            self.manifold.q_perp,
            -self.manifold.q_z,
            name=f"{self.name}-qz-sign-flip",
            interpretation="dispersion-derived",
        )


def make_cases(n_u: int = 20) -> dict[str, PDECase]:
    k = 2.0
    theta = np.linspace(0.15, 1.25, n_u)
    qp = k * np.sin(theta)
    qz = k * (np.cos(theta) - 1.0)
    shifted = AxisymmetricManifold(
        theta,
        qp,
        qz,
        name="shifted-helmholtz",
        interpretation="dispersion-derived",
    )
    shifted_weights = surface_radial_weights(
        qp,
        k * np.cos(theta),
        -k * np.sin(theta),
        parameter_trapezoid_weights(theta),
    )

    s = np.linspace(0.10, 2.0, n_u)
    qp = s.copy()
    qz = -(s * s) / (2.0 * k)
    paraxial = AxisymmetricManifold(
        s,
        qp,
        qz,
        name="paraxial-paraboloid",
        interpretation="dispersion-derived",
    )
    paraxial_weights = surface_radial_weights(
        qp,
        np.ones_like(s),
        -s / k,
        parameter_trapezoid_weights(s),
    )

    a, c = 2.0, 1.25
    theta_e = np.linspace(0.15, np.pi - 0.15, n_u)
    qp = a * np.sin(theta_e)
    qz = c * np.cos(theta_e)
    ellipsoid = AxisymmetricManifold(
        theta_e,
        qp,
        qz,
        name="ellipsoidal-dispersion",
        interpretation="dispersion-derived",
    )
    ellipsoid_weights = surface_radial_weights(
        qp,
        a * np.cos(theta_e),
        -c * np.sin(theta_e),
        parameter_trapezoid_weights(theta_e),
    )

    return {
        "shifted_helmholtz": PDECase(
            "shifted_helmholtz",
            "Shifted Helmholtz",
            shifted,
            shifted_weights,
            "shifted_helmholtz",
            {"k": k},
        ),
        "paraxial": PDECase(
            "paraxial",
            "Paraxial",
            paraxial,
            paraxial_weights,
            "paraxial",
            {"k": k},
        ),
        "ellipsoid": PDECase(
            "ellipsoid",
            "Ellipsoid",
            ellipsoid,
            ellipsoid_weights,
            "ellipsoid",
            {"a": a, "c": c},
        ),
    }


def make_coefficients(case: PDECase, max_h: int, rng: np.random.Generator) -> np.ndarray:
    modes = np.arange(-max_h, max_h + 1)
    coordinate = np.linspace(-1.0, 1.0, case.manifold.n_u)
    radial_envelope = np.exp(-0.5 * ((coordinate - 0.10) / 0.58) ** 2)
    harmonic_envelope = np.exp(-0.5 * (modes / 3.2) ** 2)
    random = rng.normal(size=(case.manifold.n_u, modes.size)) + 1j * rng.normal(
        size=(case.manifold.n_u, modes.size)
    )
    coefficients = random * radial_envelope[:, None] * harmonic_envelope[None, :]
    coefficients[:, max_h] += 0.75 * radial_envelope
    return coefficients.astype(np.complex128)


def cartesian_grid(n: int, half_width: float = 2.0) -> tuple[np.ndarray, np.ndarray, float]:
    axis = np.linspace(-half_width, half_width, n)
    spacing = float(axis[1] - axis[0])
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    coords = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    weights = np.full(coords.shape[0], spacing**3)
    return coords, weights, spacing


def cylindrical_grid(
    n_r: int = 32,
    n_beta: int = 128,
    n_z: int = 32,
    *,
    radius_max: float = 2.0,
    z_half_width: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    r_edges = np.linspace(0.0, radius_max, n_r + 1)
    z_edges = np.linspace(-z_half_width, z_half_width, n_z + 1)
    radius = 0.5 * (r_edges[:-1] + r_edges[1:])
    z = 0.5 * (z_edges[:-1] + z_edges[1:])
    beta = (np.arange(n_beta) + 0.5) * (2.0 * np.pi / n_beta)
    rr, bb, zz = np.meshgrid(radius, beta, z, indexing="ij")
    coords = np.column_stack(
        ((rr * np.cos(bb)).ravel(), (rr * np.sin(bb)).ravel(), zz.ravel())
    )
    dr = float(r_edges[1] - r_edges[0])
    dz = float(z_edges[1] - z_edges[0])
    dbeta = 2.0 * np.pi / n_beta
    weights = (rr * dr * dbeta * dz).ravel()
    return coords, weights, (n_r, n_beta, n_z)


FIRST_DERIVATIVE_8 = {
    -4: 1.0 / 280.0,
    -3: -4.0 / 105.0,
    -2: 1.0 / 5.0,
    -1: -4.0 / 5.0,
    1: 4.0 / 5.0,
    2: -1.0 / 5.0,
    3: 4.0 / 105.0,
    4: -1.0 / 280.0,
}
SECOND_DERIVATIVE_8 = {
    -4: -1.0 / 560.0,
    -3: 8.0 / 315.0,
    -2: -1.0 / 5.0,
    -1: 8.0 / 5.0,
    0: -205.0 / 72.0,
    1: 8.0 / 5.0,
    2: -1.0 / 5.0,
    3: 8.0 / 315.0,
    4: -1.0 / 560.0,
}


def _centered_stencil(field: np.ndarray, axis: int, spacing: float, coefficients: dict[int, float], power: int) -> np.ndarray:
    margin = 4
    shape = field.shape
    if field.ndim != 3 or min(shape) <= 2 * margin:
        raise ValueError("field must be a three-dimensional grid wider than the stencil")
    result = np.zeros(tuple(size - 2 * margin for size in shape), dtype=np.complex128)
    for offset, coefficient in coefficients.items():
        slices = [slice(margin, size - margin) for size in shape]
        slices[axis] = slice(margin + offset, shape[axis] - margin + offset)
        result += coefficient * field[tuple(slices)]
    return result / spacing**power


def first_derivative_8(field: np.ndarray, axis: int, spacing: float) -> np.ndarray:
    return _centered_stencil(field, axis, spacing, FIRST_DERIVATIVE_8, 1)


def second_derivative_8(field: np.ndarray, axis: int, spacing: float) -> np.ndarray:
    return _centered_stencil(field, axis, spacing, SECOND_DERIVATIVE_8, 2)


def pde_residual(case: PDECase, field: np.ndarray, spacing: float) -> float:
    dx2 = second_derivative_8(field, 0, spacing)
    dy2 = second_derivative_8(field, 1, spacing)
    dz2 = second_derivative_8(field, 2, spacing)
    core = field[4:-4, 4:-4, 4:-4]
    if case.kind == "shifted_helmholtz":
        dz = first_derivative_8(field, 2, spacing)
        laplacian = dx2 + dy2 + dz2
        drift = 2j * case.parameters["k"] * dz
        residual = laplacian + drift
        denominator = np.linalg.norm(laplacian.ravel()) + np.linalg.norm(drift.ravel())
    elif case.kind == "paraxial":
        dz = first_derivative_8(field, 2, spacing)
        transverse = dx2 + dy2
        drift = 2j * case.parameters["k"] * dz
        residual = transverse + drift
        denominator = np.linalg.norm(transverse.ravel()) + np.linalg.norm(drift.ravel())
    elif case.kind == "ellipsoid":
        radial = -(dx2 + dy2) / case.parameters["a"] ** 2
        axial = -dz2 / case.parameters["c"] ** 2
        mass = -core
        residual = radial + axial + mass
        denominator = (
            np.linalg.norm(radial.ravel())
            + np.linalg.norm(axial.ravel())
            + np.linalg.norm(mass.ravel())
        )
    else:  # pragma: no cover
        raise ValueError(f"unknown PDE case kind: {case.kind}")
    return float(np.linalg.norm(residual.ravel()) / denominator)


def relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm((actual - reference).ravel()) / np.linalg.norm(reference.ravel()))


def direct_extension_phi(
    pair: AxisymmetricPDEPair,
    coefficients: np.ndarray,
    *,
    n_phi: int,
    point_block_size: int = 1024,
) -> np.ndarray:
    phi = (np.arange(n_phi) + 0.5) * (2.0 * np.pi / n_phi)
    density = coefficients @ np.exp(1j * pair.modes[:, None] * phi[None, :])
    output = np.zeros(pair.field_shape, dtype=np.complex128)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    for start in range(0, pair.field_shape[0], point_block_size):
        stop = min(start + point_block_size, pair.field_shape[0])
        coords = pair.cartesian_coords[start:stop]
        transverse = coords[:, 0, None] * cos_phi + coords[:, 1, None] * sin_phi
        block = np.zeros(stop - start, dtype=np.complex128)
        for index in range(pair.manifold.n_u):
            phase = (
                pair.manifold.q_perp[index] * transverse
                + pair.manifold.q_z[index] * coords[:, 2, None]
            )
            block += pair.manifold_weights[index] * np.mean(
                density[index][None, :] * np.exp(1j * phase), axis=1
            )
        output[start:stop] = 2.0 * np.pi * block
    return output


def direct_restriction_phi(
    pair: AxisymmetricPDEPair,
    field: np.ndarray,
    *,
    n_phi: int,
    point_block_size: int = 1024,
) -> np.ndarray:
    phi = (np.arange(n_phi) + 0.5) * (2.0 * np.pi / n_phi)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    samples = np.zeros((pair.manifold.n_u, n_phi), dtype=np.complex128)
    weighted_field = pair.spatial_weights * field
    for start in range(0, pair.field_shape[0], point_block_size):
        stop = min(start + point_block_size, pair.field_shape[0])
        coords = pair.cartesian_coords[start:stop]
        transverse = coords[:, 0, None] * cos_phi + coords[:, 1, None] * sin_phi
        for index in range(pair.manifold.n_u):
            phase = (
                pair.manifold.q_perp[index] * transverse
                + pair.manifold.q_z[index] * coords[:, 2, None]
            )
            samples[index] += np.sum(
                np.exp(-1j * phase) * weighted_field[start:stop, None], axis=0
            )
    return samples @ np.exp(-1j * phi[:, None] * pair.modes[None, :]) / n_phi


def direct_synthesis_validation(case: PDECase, coefficients: np.ndarray) -> dict[str, float]:
    coords, _, _ = cartesian_grid(16)
    pair = AxisymmetricPDEPair(
        case.manifold,
        coords,
        manifold_weights=case.manifold_weights,
        spatial_weights=np.ones(coords.shape[0]),
        max_h=8,
        point_block_size=4096,
    )
    prepared = pair.extension(coefficients)
    direct_128 = direct_extension_phi(pair, coefficients, n_phi=128)
    direct_256 = direct_extension_phi(pair, coefficients, n_phi=256)
    single_errors = []
    probe_coords = coords[np.linspace(0, coords.shape[0] - 1, 256, dtype=np.intp)]
    probe_pair = AxisymmetricPDEPair(
        case.manifold,
        probe_coords,
        manifold_weights=case.manifold_weights,
        spatial_weights=np.ones(probe_coords.shape[0]),
        max_h=8,
    )
    for mode in (0, -1, 1, -8, 8):
        single = np.zeros_like(coefficients)
        single[:, mode + 8] = coefficients[:, mode + 8]
        single_errors.append(
            relative_l2(
                probe_pair.extension(single),
                direct_extension_phi(probe_pair, single, n_phi=256),
            )
        )
    return {
        "random_prepared_vs_direct_256_relative_l2": relative_l2(prepared, direct_256),
        "direct_128_vs_256_relative_l2": relative_l2(direct_128, direct_256),
        "single_mode_relative_l2_max": float(max(single_errors)),
    }


def convergence_validation(
    case: PDECase,
    coefficients: np.ndarray,
    resolutions: tuple[int, ...],
) -> tuple[dict[str, object], dict[int, np.ndarray], float]:
    correct_errors = []
    off_shell_errors = []
    spacings = []
    finest_components: dict[int, np.ndarray] = {}
    finest_spacing = 0.0
    off_shell = case.off_shell_manifold()
    for n in resolutions:
        coords, spatial_weights, spacing = cartesian_grid(n)
        pair = AxisymmetricPDEPair(
            case.manifold,
            coords,
            manifold_weights=case.manifold_weights,
            spatial_weights=spatial_weights,
            max_h=8,
            point_block_size=8192,
        )
        off_pair = AxisymmetricPDEPair(
            off_shell,
            coords,
            manifold_weights=case.manifold_weights,
            spatial_weights=spatial_weights,
            max_h=8,
            point_block_size=8192,
        )
        field = pair.extension(coefficients).reshape((n, n, n))
        off_field = off_pair.extension(coefficients).reshape((n, n, n))
        correct_errors.append(pde_residual(case, field, spacing))
        off_shell_errors.append(pde_residual(case, off_field, spacing))
        spacings.append(spacing)
        if n == resolutions[-1]:
            finest_spacing = spacing
            for mode_index, mode in enumerate(pair.modes):
                component_coefficients = np.zeros_like(coefficients)
                component_coefficients[:, mode_index] = coefficients[:, mode_index]
                finest_components[int(mode)] = pair.extension(component_coefficients).reshape(
                    (n, n, n)
                )
    slope = float(np.polyfit(np.log(spacings), np.log(correct_errors), 1)[0])
    return (
        {
            "resolutions": list(resolutions),
            "spacings": spacings,
            "correct_residuals": correct_errors,
            "off_shell_residuals": off_shell_errors,
            "observed_order": slope,
            "monotone": bool(np.all(np.diff(correct_errors) < 0.0)),
            "finest_correct": correct_errors[-1],
            "finest_off_shell": off_shell_errors[-1],
            "off_shell_to_correct_ratio": off_shell_errors[-1] / correct_errors[-1],
        },
        finest_components,
        finest_spacing,
    )


def truncation_validation(
    case: PDECase,
    coefficients: np.ndarray,
    cartesian_components: dict[int, np.ndarray],
    spacing: float,
) -> dict[str, object]:
    h_values = (0, 1, 2, 4, 6, 8)
    cyl_coords, cyl_weights, _ = cylindrical_grid()
    cyl_pair = AxisymmetricPDEPair(
        case.manifold,
        cyl_coords,
        manifold_weights=case.manifold_weights,
        spatial_weights=cyl_weights,
        max_h=8,
        point_block_size=8192,
    )
    cylindrical_components: dict[int, np.ndarray] = {}
    for mode_index, mode in enumerate(cyl_pair.modes):
        component_coefficients = np.zeros_like(coefficients)
        component_coefficients[:, mode_index] = coefficients[:, mode_index]
        cylindrical_components[int(mode)] = cyl_pair.extension(component_coefficients)

    reference = sum(cylindrical_components.values())
    reference_norm_squared = float(
        np.sum(cyl_weights * np.abs(reference) ** 2)
    )
    component_energy = {
        mode: float(np.sum(cyl_weights * np.abs(field) ** 2))
        for mode, field in cylindrical_components.items()
    }
    records = []
    for h in h_values:
        kept_cyl = sum(field for mode, field in cylindrical_components.items() if abs(mode) <= h)
        difference = kept_cyl - reference
        solution_error_squared = float(
            np.sum(cyl_weights * np.abs(difference) ** 2) / reference_norm_squared
        )
        tail_energy_fraction = float(
            sum(energy for mode, energy in component_energy.items() if abs(mode) > h)
            / reference_norm_squared
        )
        cartesian_field = sum(
            field for mode, field in cartesian_components.items() if abs(mode) <= h
        )
        records.append(
            {
                "H": h,
                "solution_error": float(np.sqrt(solution_error_squared)),
                "solution_error_squared": solution_error_squared,
                "output_tail_energy_fraction": tail_energy_fraction,
                "parseval_absolute_mismatch": abs(
                    solution_error_squared - tail_energy_fraction
                ),
                "pde_residual": pde_residual(case, cartesian_field, spacing),
            }
        )
    errors = np.array([record["solution_error"] for record in records])
    residuals = np.array([record["pde_residual"] for record in records])
    return {
        "records": records,
        "solution_error_monotone": bool(np.all(np.diff(errors) <= 1e-14)),
        "parseval_mismatch_max": float(
            max(record["parseval_absolute_mismatch"] for record in records)
        ),
        "pde_residual_max": float(np.max(residuals)),
        "pde_residual_spread": float(np.max(residuals) / np.min(residuals)),
    }


def adjoint_validation(case: PDECase, coefficients: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    coords, spatial_weights, _ = cartesian_grid(10)
    pair = AxisymmetricPDEPair(
        case.manifold,
        coords,
        manifold_weights=case.manifold_weights,
        spatial_weights=spatial_weights,
        max_h=8,
        point_block_size=1024,
    )
    field = rng.normal(size=pair.field_shape) + 1j * rng.normal(size=pair.field_shape)
    restriction = pair.restriction(field)
    direct = direct_restriction_phi(pair, field, n_phi=256)
    wrong_pair = AxisymmetricPDEPair(
        case.manifold,
        coords,
        manifold_weights=case.manifold_weights,
        spatial_weights=np.ones(pair.field_shape),
        max_h=8,
        point_block_size=1024,
    )
    left = pair.spatial_inner_product(pair.extension(coefficients), field)
    wrong_right = pair.manifold_inner_product(coefficients, wrong_pair.restriction(field))
    wrong_measure_error = abs(left - wrong_right) / (abs(left) + abs(wrong_right))
    return {
        "dot_product_error": pair.adjoint_test(coefficients, field),
        "restriction_vs_direct_relative_l2": relative_l2(restriction, direct),
        "wrong_measure_dot_error": float(wrong_measure_error),
    }


ACCEPTANCE = {
    "shell_residual_max": 1e-13,
    "off_shell_residual_min": 1e-4,
    "synthesis_relative_l2_max": 1e-11,
    "quadrature_relative_l2_max": 1e-12,
    "pde_observed_order_min": 5.5,
    "pde_finest_residual_max": 5e-6,
    "pde_negative_ratio_min": 100.0,
    "parseval_mismatch_max": 1e-10,
    "truncation_pde_residual_max": 1e-5,
    "truncation_residual_spread_max": 5.0,
    "adjoint_error_max": 1e-12,
    "restriction_relative_l2_max": 1e-11,
    "wrong_measure_error_min": 1e-3,
}


def case_passed(result: dict[str, object]) -> bool:
    synthesis = result["synthesis"]
    shell = result["shell"]
    convergence = result["pde_convergence"]
    truncation = result["truncation"]
    adjoint = result["adjoint"]
    assert all(isinstance(section, dict) for section in (synthesis, shell, convergence, truncation, adjoint))
    return bool(
        shell["correct"] <= ACCEPTANCE["shell_residual_max"]
        and shell["off_shell"] >= ACCEPTANCE["off_shell_residual_min"]
        and synthesis["random_prepared_vs_direct_256_relative_l2"]
        <= ACCEPTANCE["synthesis_relative_l2_max"]
        and synthesis["single_mode_relative_l2_max"]
        <= ACCEPTANCE["synthesis_relative_l2_max"]
        and synthesis["direct_128_vs_256_relative_l2"]
        <= ACCEPTANCE["quadrature_relative_l2_max"]
        and convergence["monotone"]
        and convergence["observed_order"] >= ACCEPTANCE["pde_observed_order_min"]
        and convergence["finest_correct"] <= ACCEPTANCE["pde_finest_residual_max"]
        and convergence["off_shell_to_correct_ratio"]
        >= ACCEPTANCE["pde_negative_ratio_min"]
        and truncation["solution_error_monotone"]
        and truncation["parseval_mismatch_max"] <= ACCEPTANCE["parseval_mismatch_max"]
        and truncation["pde_residual_max"]
        <= ACCEPTANCE["truncation_pde_residual_max"]
        and truncation["pde_residual_spread"]
        <= ACCEPTANCE["truncation_residual_spread_max"]
        and adjoint["dot_product_error"] <= ACCEPTANCE["adjoint_error_max"]
        and adjoint["restriction_vs_direct_relative_l2"]
        <= ACCEPTANCE["restriction_relative_l2_max"]
        and adjoint["wrong_measure_dot_error"]
        >= ACCEPTANCE["wrong_measure_error_min"]
    )


def generate_figures(payload: dict[str, object], figure_dir: Path) -> list[str]:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    results = payload["results"]
    assert isinstance(results, dict)
    colors = {"shifted_helmholtz": "#1f77b4", "paraxial": "#d95f02", "ellipsoid": "#2ca02c"}
    paths: list[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for name, raw in results.items():
        geometry = raw["geometry"]
        axes[0].plot(geometry["q_perp"], geometry["q_z"], marker="o", ms=2.5, label=raw["label"], color=colors[name])
    axes[0].set_xlabel(r"$K_\perp$")
    axes[0].set_ylabel(r"$K_z$")
    axes[0].set_title("Dispersion curves")
    axes[0].legend(frameon=False)
    names = list(results)
    x = np.arange(len(names))
    axes[1].bar(x - 0.18, [results[name]["shell"]["correct"] for name in names], width=0.36, label="on shell")
    axes[1].bar(x + 0.18, [results[name]["shell"]["off_shell"] for name in names], width=0.36, label="off shell")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, [results[name]["label"] for name in names], rotation=15)
    axes[1].set_ylabel("normalized shell residual")
    axes[1].set_title("Shell and negative control")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    path = figure_dir / "figure1_dispersion_shell.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.relative_to(ROOT)))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    metrics = (
        "random_prepared_vs_direct_256_relative_l2",
        "single_mode_relative_l2_max",
        "direct_128_vs_256_relative_l2",
    )
    labels = ("random/direct", "single-mode max", "Nphi 128/256")
    width = 0.24
    for offset, name in enumerate(names):
        ax.bar(
            np.arange(len(metrics)) + (offset - 1) * width,
            [results[name]["synthesis"][metric] for metric in metrics],
            width=width,
            label=results[name]["label"],
            color=colors[name],
        )
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(metrics)), labels)
    ax.set_ylabel("relative L2")
    ax.set_title("Direct synthesis accuracy")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = figure_dir / "figure2_direct_synthesis.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.relative_to(ROOT)))

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9), sharey=True)
    for axis, name in zip(axes, names, strict=True):
        convergence = results[name]["pde_convergence"]
        axis.loglog(convergence["spacings"], convergence["correct_residuals"], "o-", label="on shell")
        axis.loglog(convergence["spacings"], convergence["off_shell_residuals"], "s--", label="off shell")
        axis.set_title(f"{results[name]['label']}\norder={convergence['observed_order']:.2f}")
        axis.set_xlabel("grid spacing")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("normalized PDE residual")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    path = figure_dir / "figure3_pde_convergence.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.relative_to(ROOT)))

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9), sharey=True)
    for axis, name in zip(axes, names, strict=True):
        records = results[name]["truncation"]["records"]
        h = [record["H"] for record in records]
        display_floor = 1e-14
        solution_display = [
            max(record["solution_error"], display_floor) for record in records
        ]
        tail_display = [
            max(np.sqrt(record["output_tail_energy_fraction"]), display_floor)
            for record in records
        ]
        axis.semilogy(h, solution_display, "o-", label="solution error")
        axis.semilogy(h, tail_display, "x--", label="sqrt output tail")
        axis.semilogy(h, [record["pde_residual"] for record in records], "s-.", label="PDE residual")
        axis.set_title(results[name]["label"])
        axis.set_xlabel("H")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("relative metric")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.text(0.5, 0.01, "Zero truncation error at H=8 is displayed at 1e-14.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    path = figure_dir / "figure4_truncation_separation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.relative_to(ROOT)))
    return paths


def render_markdown(payload: dict[str, object]) -> str:
    results = payload["results"]
    assert isinstance(results, dict)
    lines = [
        "# ACFO PDE restriction-extension minimum validation",
        "",
        "The established positive-sign ACFO sampling path is unchanged. This validation uses a separate physical-sign pair: restriction `exp(-i Gamma.x)` and extension `exp(+i Gamma.x)`.",
        "",
        "The manifold inner product uses explicit surface-of-revolution quadrature and the Cartesian field inner product uses the uniform voxel measure. The extension and restriction operate in the signed harmonic coefficient space `h=-H,...,H`.",
        "",
        "## Result table",
        "",
        "| case | shell | synthesis L2 | PDE finest | order | negative ratio | adjoint | pass |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for result in results.values():
        lines.append(
            f"| {result['label']} | {result['shell']['correct']:.3e} | "
            f"{result['synthesis']['random_prepared_vs_direct_256_relative_l2']:.3e} | "
            f"{result['pde_convergence']['finest_correct']:.3e} | "
            f"{result['pde_convergence']['observed_order']:.2f} | "
            f"{result['pde_convergence']['off_shell_to_correct_ratio']:.2f} | "
            f"{result['adjoint']['dot_product_error']:.3e} | "
            f"{'PASS' if result['passed'] else 'FAIL'} |"
        )
    lines.extend(["", f"Overall pass: **{payload['passed']}**", "", "## Figures", ""])
    for index, path in enumerate(payload["figures"], start=1):
        lines.append(f"{index}. `{path}`")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "This result separates two statements. The existing ACFO backend evaluates arbitrary finite axisymmetric sampling geometries. The new result establishes exact homogeneous-mode generation only when the supplied geometry satisfies the explicitly tested symbol relation `p(Gamma)=0`. An arbitrary spline or table is not thereby certified as a physical or local-PDE dispersion surface.",
            "",
            "No boundary-value solve, performance benchmark, vector PDE, evanescent branch, PDF, or patent memo is included.",
            "",
            "## Reproduce",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\validate_axisymmetric_pde_extension.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run(seed: int = 20260712) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    results: dict[str, object] = {}
    for name, case in make_cases().items():
        coefficients = make_coefficients(case, 8, rng)
        synthesis = direct_synthesis_validation(case, coefficients)
        off_shell = case.off_shell_manifold()
        shell: dict[str, float] = {
            "correct": case.normalized_shell_residual(),
            "off_shell": case.normalized_shell_residual(off_shell),
        }
        if case.kind in {"shifted_helmholtz", "paraxial"}:
            shell["qz_sign_flip"] = case.normalized_shell_residual(
                case.sign_flip_manifold()
            )
        convergence, cartesian_components, finest_spacing = convergence_validation(
            case,
            coefficients,
            (24, 32, 40, 48),
        )
        truncation = truncation_validation(
            case,
            coefficients,
            cartesian_components,
            finest_spacing,
        )
        adjoint = adjoint_validation(case, coefficients, rng)
        result: dict[str, object] = {
            "label": case.label,
            "geometry": {
                "q_perp": case.manifold.q_perp.tolist(),
                "q_z": case.manifold.q_z.tolist(),
                "manifold_weights": case.manifold_weights.tolist(),
                "parameters": case.parameters,
            },
            "synthesis": synthesis,
            "shell": shell,
            "pde_convergence": convergence,
            "truncation": truncation,
            "adjoint": adjoint,
        }
        result["passed"] = case_passed(result)
        results[name] = result
    return {
        "schema": "acfo-pde-restriction-extension-validation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "configuration": {
            "n_u": 20,
            "max_h": 8,
            "cartesian_half_width": 2.0,
            "cartesian_resolutions": [24, 32, 40, 48],
            "direct_phi_resolutions": [128, 256],
            "truncation_H": [0, 1, 2, 4, 6, 8],
            "finite_difference_order": 8,
            "interior_margin": 4,
        },
        "acceptance": ACCEPTANCE,
        "results": results,
        "passed": bool(all(result["passed"] for result in results.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "benchmark_results" / "acfo_pde_extension_validation.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "docs" / "acfo_pde_extension_validation.md",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT / "docs" / "acfo_pde_extension_validation_support",
    )
    args = parser.parse_args()

    payload = run(args.seed)
    payload["figures"] = generate_figures(payload, args.figure_dir)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.markdown_output.write_text(render_markdown(payload), encoding="utf-8")
    print(render_markdown(payload))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
