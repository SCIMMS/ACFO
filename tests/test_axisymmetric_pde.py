from __future__ import annotations

import numpy as np
import pytest

from waxs_cake import AxisymmetricManifold, AxisymmetricPDEPair
from scripts.validate_axisymmetric_pde_extension import (
    first_derivative_8,
    second_derivative_8,
)


def make_pair(*, max_h: int = 3) -> AxisymmetricPDEPair:
    manifold = AxisymmetricManifold(
        u=np.array([0.2, 0.6, 1.0]),
        q_perp=np.array([0.5, 0.9, 1.2]),
        q_z=np.array([-0.1, -0.3, -0.6]),
    )
    coords = np.array(
        [
            [-0.7, -0.2, -0.5],
            [0.0, 0.0, -0.1],
            [0.4, -0.6, 0.2],
            [0.8, 0.3, 0.7],
        ]
    )
    return AxisymmetricPDEPair(
        manifold,
        coords,
        manifold_weights=np.array([0.12, 0.31, 0.27]),
        spatial_weights=np.array([0.08, 0.11, 0.09, 0.13]),
        max_h=max_h,
        point_block_size=2,
    )


def direct_extension(
    pair: AxisymmetricPDEPair,
    coefficients: np.ndarray,
    *,
    n_phi: int = 512,
) -> np.ndarray:
    phi = (np.arange(n_phi) + 0.5) * (2.0 * np.pi / n_phi)
    angular = np.exp(1j * pair.modes[:, None] * phi[None, :])
    density = coefficients @ angular
    result = np.zeros(pair.field_shape, dtype=np.complex128)
    for index in range(pair.manifold.n_u):
        q = np.column_stack(
            (
                pair.manifold.q_perp[index] * np.cos(phi),
                pair.manifold.q_perp[index] * np.sin(phi),
                np.full(n_phi, pair.manifold.q_z[index]),
            )
        )
        phase = q @ pair.cartesian_coords.T
        result += pair.manifold_weights[index] * np.mean(
            density[index, :, None] * np.exp(1j * phase),
            axis=0,
        )
    return 2.0 * np.pi * result


def direct_restriction(
    pair: AxisymmetricPDEPair,
    field: np.ndarray,
    *,
    n_phi: int = 512,
) -> np.ndarray:
    phi = (np.arange(n_phi) + 0.5) * (2.0 * np.pi / n_phi)
    samples = np.empty((pair.manifold.n_u, n_phi), dtype=np.complex128)
    weighted_field = pair.spatial_weights * field
    for index in range(pair.manifold.n_u):
        q = np.column_stack(
            (
                pair.manifold.q_perp[index] * np.cos(phi),
                pair.manifold.q_perp[index] * np.sin(phi),
                np.full(n_phi, pair.manifold.q_z[index]),
            )
        )
        phase = q @ pair.cartesian_coords.T
        samples[index] = np.exp(-1j * phase) @ weighted_field
    return samples @ np.exp(-1j * phi[:, None] * pair.modes[None, :]) / n_phi


@pytest.mark.parametrize("mode", [0, -1, 1, -3, 3])
def test_single_harmonic_extension_matches_direct_exponent_sum(mode: int) -> None:
    pair = make_pair()
    coefficients = np.zeros(pair.coefficient_shape, dtype=np.complex128)
    coefficients[:, mode + pair.max_h] = np.array(
        [0.8 + 0.2j, -0.4 + 0.6j, 0.3 - 0.7j]
    )
    expected = direct_extension(pair, coefficients)

    assert np.allclose(pair.extension(coefficients), expected, rtol=0.0, atol=2e-14)


def test_random_multimode_extension_and_restriction_match_direct() -> None:
    pair = make_pair()
    rng = np.random.default_rng(20260712)
    coefficients = rng.normal(size=pair.coefficient_shape) + 1j * rng.normal(
        size=pair.coefficient_shape
    )
    field = rng.normal(size=pair.field_shape) + 1j * rng.normal(size=pair.field_shape)

    assert np.allclose(
        pair.extension(coefficients),
        direct_extension(pair, coefficients),
        rtol=0.0,
        atol=5e-14,
    )
    assert np.allclose(
        pair.restriction(field),
        direct_restriction(pair, field),
        rtol=0.0,
        atol=5e-14,
    )


def test_weighted_adjoint_identity_and_measure_control() -> None:
    pair = make_pair()
    rng = np.random.default_rng(11)
    coefficients = rng.normal(size=pair.coefficient_shape) + 1j * rng.normal(
        size=pair.coefficient_shape
    )
    field = rng.normal(size=pair.field_shape) + 1j * rng.normal(size=pair.field_shape)

    assert pair.adjoint_test(coefficients, field) < 1e-13

    wrong = AxisymmetricPDEPair(
        pair.manifold,
        pair.cartesian_coords,
        manifold_weights=pair.manifold_weights,
        spatial_weights=np.ones(pair.field_shape),
        max_h=pair.max_h,
    )
    left = pair.spatial_inner_product(pair.extension(coefficients), field)
    wrong_right = pair.manifold_inner_product(coefficients, wrong.restriction(field))
    mismatch = abs(left - wrong_right) / (abs(left) + abs(wrong_right))
    assert mismatch > 1e-3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"manifold_weights": [1.0, 1.0]}, "manifold_weights"),
        ({"spatial_weights": [1.0, 1.0]}, "spatial_weights"),
        ({"manifold_weights": [1.0, -1.0, 1.0]}, "non-negative"),
        ({"max_h": -1}, "non-negative"),
        ({"point_block_size": 0}, "positive"),
    ],
)
def test_invalid_constructor_inputs(kwargs: dict[str, object], message: str) -> None:
    manifold = AxisymmetricManifold([0.0, 1.0, 2.0], [0.2, 0.4, 0.6], [0.0, -0.1, -0.2])
    defaults: dict[str, object] = {
        "manifold_weights": [1.0, 1.0, 1.0],
        "spatial_weights": [1.0, 1.0, 1.0],
        "max_h": 2,
        "point_block_size": 2,
    }
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=message):
        AxisymmetricPDEPair(manifold, np.eye(3), **defaults)


def test_invalid_value_shapes_and_nonfinite_values() -> None:
    pair = make_pair()
    with pytest.raises(ValueError, match="coefficients"):
        pair.extension(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="field"):
        pair.restriction(np.zeros(3))
    coefficients = np.zeros(pair.coefficient_shape, dtype=np.complex128)
    coefficients[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        pair.extension(coefficients)


def test_eighth_order_derivatives_converge_on_plane_wave() -> None:
    wavevector = np.array([2.1, -1.5, 1.9])
    first_errors = []
    second_errors = []
    spacings = []
    for n in (12, 16, 20, 24):
        axis = np.linspace(-1.0, 1.0, n)
        spacing = axis[1] - axis[0]
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        field = np.exp(1j * (wavevector[0] * x + wavevector[1] * y + wavevector[2] * z))
        core = field[4:-4, 4:-4, 4:-4]
        first = first_derivative_8(field, 2, spacing)
        second = second_derivative_8(field, 0, spacing)
        first_errors.append(
            np.linalg.norm(first - 1j * wavevector[2] * core) / np.linalg.norm(core)
        )
        second_errors.append(
            np.linalg.norm(second + wavevector[0] ** 2 * core) / np.linalg.norm(core)
        )
        spacings.append(spacing)

    first_order = np.polyfit(np.log(spacings), np.log(first_errors), 1)[0]
    second_order = np.polyfit(np.log(spacings), np.log(second_errors), 1)[0]
    assert first_order > 7.0
    assert second_order > 7.0
