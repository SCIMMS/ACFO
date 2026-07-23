from __future__ import annotations

import numpy as np

from waxs_cake import (
    AnisotropicGaussianMixture,
    AxisymmetricManifold,
    sample_gaussian_mixture_midpoint_grid,
)
from scripts.validate_axisymmetric_gaussian_mixture import make_gaussian_mixture


def test_gaussian_transform_at_zero_matches_analytic_integral() -> None:
    mixture = make_gaussian_mixture()
    value = mixture.fourier_nodes(np.zeros((1, 3)))[0]
    expected = sum(
        coefficient * (2.0 * np.pi) ** 1.5 * np.sqrt(np.linalg.det(covariance))
        for coefficient, covariance in zip(
            mixture.coefficients,
            mixture.covariances,
            strict=True,
        )
    )
    assert np.allclose(value, expected, rtol=1e-15, atol=1e-15)


def test_shift_phase_uses_positive_fourier_sign() -> None:
    mean = np.array([[0.4, -0.7, 0.2]])
    covariance = np.array([np.diag([0.5, 0.8, 0.3]) ** 2])
    shifted = AnisotropicGaussianMixture([1.0], mean, covariance)
    centered = AnisotropicGaussianMixture([1.0], np.zeros((1, 3)), covariance)
    node = np.array([[0.6, -0.2, 0.9]])
    ratio = shifted.fourier_nodes(node)[0] / centered.fourier_nodes(node)[0]
    assert np.allclose(ratio, np.exp(1j * (node[0] @ mean[0])), rtol=1e-15, atol=1e-15)


def test_midpoint_grid_converges_to_zero_frequency_integral() -> None:
    mixture = AnisotropicGaussianMixture(
        [1.0],
        [[0.2, -0.1, 0.3]],
        [np.diag([0.55, 0.7, 0.45]) ** 2],
    )
    analytic = mixture.fourier_nodes(np.zeros((1, 3)))[0]
    errors = []
    for n_per_axis in (8, 10, 12):
        _, weights, _ = sample_gaussian_mixture_midpoint_grid(
            mixture,
            half_width=3.5,
            n_per_axis=n_per_axis,
        )
        errors.append(abs(np.sum(weights) - analytic))
    assert errors[2] < errors[1] < errors[0]


def test_gaussian_manifold_output_shape() -> None:
    mixture = make_gaussian_mixture()
    manifold = AxisymmetricManifold(
        u=[0.0, 1.0],
        q_perp=[0.5, 1.0],
        q_z=[-0.2, 0.3],
    )
    result = mixture.fourier_manifold(manifold, [0.0, 1.0, 2.0])
    assert result.shape == (2, 3)
