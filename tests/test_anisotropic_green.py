from __future__ import annotations

import numpy as np
import pytest

from waxs_cake import (
    apply_maxwell_spectral_residue,
    maxwell_resolvent_residue,
    maxwell_spectral_residue,
    maxwell_wave_operator,
    uniaxial_eigenpolarization,
)


def _uniaxial_case(branch: str) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    epsilon_perpendicular = 5.2
    epsilon_parallel = 4.7
    epsilon = np.diag(
        [epsilon_perpendicular, epsilon_perpendicular, epsilon_parallel]
    )
    k0 = 1.3
    u = np.array([0.17, 0.41, 0.68])
    phi = np.array([0.13, 0.91])
    if branch == "ordinary":
        q_perp = k0 * np.sqrt(epsilon_perpendicular) * np.sin(u)
    else:
        q_perp = k0 * np.sqrt(epsilon_parallel) * np.sin(u)
    q_z = k0 * np.sqrt(epsilon_perpendicular) * np.cos(u)
    nodes = np.empty((u.size, phi.size, 3), dtype=np.float64)
    nodes[..., 0] = q_perp[:, None] * np.cos(phi)[None, :]
    nodes[..., 1] = q_perp[:, None] * np.sin(phi)[None, :]
    nodes[..., 2] = q_z[:, None]
    return nodes, k0, epsilon, np.array((epsilon_perpendicular, epsilon_parallel))


def test_uniaxial_nodes_are_maxwell_poles() -> None:
    for branch in ("ordinary", "extraordinary"):
        nodes, k0, epsilon, _ = _uniaxial_case(branch)
        operator = maxwell_wave_operator(nodes, k0=k0, epsilon_tensor=epsilon)
        singular_values = np.linalg.svd(operator, compute_uv=False)
        assert np.max(singular_values[..., -1] / singular_values[..., 0]) < 2e-15


def test_ordinary_residue_matches_scalar_weight_but_extraordinary_does_not() -> None:
    ordinary_nodes, k0, epsilon, values = _uniaxial_case("ordinary")
    epsilon_perpendicular, epsilon_parallel = values
    phi = np.arctan2(ordinary_nodes[0, :, 1], ordinary_nodes[0, :, 0])
    q_perp = np.linalg.norm(ordinary_nodes[..., :2], axis=-1)[:, 0]
    q_z = ordinary_nodes[:, 0, 2]
    ordinary_eigen = uniaxial_eigenpolarization(
        q_perp,
        q_z,
        phi,
        epsilon_parallel=epsilon_parallel,
        epsilon_perpendicular=epsilon_perpendicular,
        branch="ordinary",
    )
    ordinary_scalar = (
        1.0 / (2.0 * q_z[:, None, None, None])
        * ordinary_eigen[..., :, None]
        * ordinary_eigen[..., None, :]
    )
    ordinary_green = maxwell_spectral_residue(
        ordinary_nodes, k0=k0, epsilon_tensor=epsilon
    )
    np.testing.assert_allclose(ordinary_scalar, -ordinary_green, atol=2e-14)

    extra_nodes, _, _, _ = _uniaxial_case("extraordinary")
    extra_q_perp = np.linalg.norm(extra_nodes[..., :2], axis=-1)[:, 0]
    extra_q_z = extra_nodes[:, 0, 2]
    extra_eigen = uniaxial_eigenpolarization(
        extra_q_perp,
        extra_q_z,
        phi,
        epsilon_parallel=epsilon_parallel,
        epsilon_perpendicular=epsilon_perpendicular,
        branch="extraordinary",
    )
    legacy_scalar = (
        epsilon_perpendicular
        / (2.0 * extra_q_z[:, None, None, None])
        * extra_eigen[..., :, None]
        * extra_eigen[..., None, :]
    )
    extra_green = maxwell_spectral_residue(
        extra_nodes, k0=k0, epsilon_tensor=epsilon
    )
    mismatch = np.linalg.norm(legacy_scalar + extra_green) / np.linalg.norm(extra_green)
    assert mismatch > 3.0


def test_resolvent_limit_converges_to_spectral_residue() -> None:
    nodes, k0, epsilon, _ = _uniaxial_case("extraordinary")
    node = nodes[1, 1]
    exact = maxwell_spectral_residue(node, k0=k0, epsilon_tensor=epsilon)
    coarse = maxwell_resolvent_residue(
        node, k0=k0, epsilon_tensor=epsilon, eta=1e-4
    )
    fine = maxwell_resolvent_residue(
        node, k0=k0, epsilon_tensor=epsilon, eta=1e-7
    )
    coarse_error = np.linalg.norm(coarse - exact) / np.linalg.norm(exact)
    fine_error = np.linalg.norm(fine - exact) / np.linalg.norm(exact)
    assert fine_error < 0.02 * coarse_error
    assert fine_error < 1e-5


def test_green_residue_application_shape_and_linearity() -> None:
    nodes, k0, epsilon, _ = _uniaxial_case("extraordinary")
    amplitude = np.arange(1, 7, dtype=np.float64).reshape(3, 2).astype(np.complex128)
    source = np.array([0.3, -0.2j, 0.7], dtype=np.complex128)
    field = apply_maxwell_spectral_residue(
        amplitude, nodes, source, k0=k0, epsilon_tensor=epsilon
    )
    doubled = apply_maxwell_spectral_residue(
        2.0 * amplitude, nodes, source, k0=k0, epsilon_tensor=epsilon
    )
    assert field.shape == (3, 2, 3)
    np.testing.assert_allclose(doubled, 2.0 * field, atol=1e-14)


def test_degenerate_optic_axis_pole_is_rejected() -> None:
    epsilon_perpendicular = 5.2
    epsilon_parallel = 4.7
    epsilon = np.diag(
        [epsilon_perpendicular, epsilon_perpendicular, epsilon_parallel]
    )
    k0 = 1.3
    degenerate_node = np.array(
        [0.0, 0.0, k0 * np.sqrt(epsilon_perpendicular)]
    )
    with pytest.raises(ValueError, match="normalized spectral gap"):
        maxwell_spectral_residue(
            degenerate_node,
            k0=k0,
            epsilon_tensor=epsilon,
        )
