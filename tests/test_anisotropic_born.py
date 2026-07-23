from __future__ import annotations

import numpy as np

from waxs_cake import (
    AxisymmetricManifold,
    CartesianSpectralBornReference,
    UniaxialScalarDispersion,
)
from scripts.validate_axisymmetric_gaussian_mixture import make_gaussian_mixture


def test_uniaxial_branches_satisfy_their_dispersion_relations() -> None:
    dispersion = UniaxialScalarDispersion(2.56, 1.44, 1.6)
    u = np.linspace(0.02, 1.0, 13)
    ordinary = dispersion.manifold(u, "ordinary")
    extraordinary = dispersion.manifold(u, "extraordinary")
    ordinary_kz = ordinary.q_z + dispersion.incident_kz
    extraordinary_kz = extraordinary.q_z + dispersion.incident_kz

    ordinary_residual = (
        ordinary.q_perp**2 + ordinary_kz**2
        - dispersion.epsilon_perpendicular * dispersion.k0**2
    )
    extraordinary_residual = (
        extraordinary.q_perp**2 / dispersion.epsilon_parallel
        + extraordinary_kz**2 / dispersion.epsilon_perpendicular
        - dispersion.k0**2
    )
    assert np.max(np.abs(ordinary_residual)) < 2e-15
    assert np.max(np.abs(extraordinary_residual)) < 2e-15


def test_cartesian_spectral_reference_matches_direct_sum_at_fft_nodes() -> None:
    rng = np.random.default_rng(19)
    n = 5
    half_width = 2.0
    density = rng.normal(size=(n, n, n)) + 1j * rng.normal(size=(n, n, n))
    reference = CartesianSpectralBornReference(
        density,
        half_width=half_width,
        padding_factor=2,
    )
    indices = np.array([[3, 4, 5], [5, 6, 7], [2, 5, 6]])
    nodes = reference.frequencies[indices]
    actual = reference.fourier_nodes(nodes)

    spacing = 2.0 * half_width / n
    axis = -half_width + (np.arange(n) + 0.5) * spacing
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    coords = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    expected = np.array(
        [
            spacing**3
            * np.sum(density.ravel() * np.exp(1j * (coords @ node)))
            for node in nodes
        ]
    )
    assert np.allclose(actual, expected, rtol=0.0, atol=2e-13)


def test_zero_padding_reduces_cartesian_interpolation_error() -> None:
    mixture = make_gaussian_mixture()
    n = 16
    half_width = 4.5
    spacing = 2.0 * half_width / n
    axis = -half_width + (np.arange(n) + 0.5) * spacing
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    coords = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    density = mixture.density(coords).reshape((n, n, n))
    dispersion = UniaxialScalarDispersion(2.56, 1.44, 1.6)
    manifold = dispersion.manifold(np.linspace(0.04, 0.85, 9), "extraordinary")
    phi = AxisymmetricManifold.uniform_phi(24)
    expected = mixture.fourier_manifold(manifold, phi)

    coarse = CartesianSpectralBornReference(
        density,
        half_width=half_width,
        padding_factor=2,
    ).born_field(manifold, phi)
    fine = CartesianSpectralBornReference(
        density,
        half_width=half_width,
        padding_factor=4,
    ).born_field(manifold, phi)
    coarse_error = np.linalg.norm(coarse - expected) / np.linalg.norm(expected)
    fine_error = np.linalg.norm(fine - expected) / np.linalg.norm(expected)
    assert fine_error < 0.35 * coarse_error
