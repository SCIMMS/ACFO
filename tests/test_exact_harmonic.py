from __future__ import annotations

import numpy as np

from waxs_cake import (
    PreparedExactCoordinateHarmonicPlan,
    exact_coordinate_harmonic_amplitude,
    exact_coordinate_harmonic_amplitude_factorized,
)
from waxs_cake.metrics import relative_l2


def direct_axisymmetric(coords, q_perp, q_z, phi, coefficients):
    qx = q_perp[:, None] * np.cos(phi)[None, :]
    qy = q_perp[:, None] * np.sin(phi)[None, :]
    out = np.empty((q_perp.size, phi.size), dtype=np.complex128)
    for iq in range(q_perp.size):
        phase = (
            coords[:, 0, None] * qx[iq]
            + coords[:, 1, None] * qy[iq]
            + coords[:, 2, None] * q_z[iq]
        )
        out[iq] = np.sum(coefficients[:, iq, None] * np.exp(1j * phase), axis=0)
    return out


def test_exact_coordinate_harmonic_matches_direct_nonuniform_sources() -> None:
    rng = np.random.default_rng(20260713)
    coords = rng.normal(scale=0.35, size=(31, 3))
    q_perp = np.asarray([0.2, 0.8, 1.7])
    q_z = np.asarray([-0.01, -0.09, -0.21])
    phi = (np.arange(47) + 0.37) * (2.0 * np.pi / 47)
    coefficients = rng.normal(size=(coords.shape[0], q_perp.size)) + 1j * rng.normal(
        size=(coords.shape[0], q_perp.size)
    )

    expected = direct_axisymmetric(coords, q_perp, q_z, phi, coefficients)
    got, cutoffs = exact_coordinate_harmonic_amplitude(
        coords,
        q_perp,
        q_z,
        phi,
        atom_coefficients=coefficients,
        harmonic_margin=24,
        atom_chunk_size=7,
        bessel_backend="scipy",
    )

    assert np.all(cutoffs >= 24)
    assert relative_l2(got, expected) < 1e-12


def test_cpp_miller_exact_coordinate_harmonic_matches_scipy() -> None:
    import pytest

    pytest.importorskip("waxs_cake._cpp_solvers")
    rng = np.random.default_rng(20260714)
    coords = rng.normal(scale=0.4, size=(23, 3))
    q_perp = np.asarray([0.3, 1.1])
    q_z = np.asarray([-0.02, -0.13])
    phi = (np.arange(41) + 0.5) * (2.0 * np.pi / 41)
    coefficients = rng.normal(size=(coords.shape[0], q_perp.size))

    expected, _ = exact_coordinate_harmonic_amplitude(
        coords,
        q_perp,
        q_z,
        phi,
        atom_coefficients=coefficients,
        harmonic_margin=24,
        bessel_backend="scipy",
    )
    got, _ = exact_coordinate_harmonic_amplitude(
        coords,
        q_perp,
        q_z,
        phi,
        atom_coefficients=coefficients,
        harmonic_margin=24,
        bessel_backend="cpp_miller",
    )

    assert relative_l2(got, expected) < 1e-12


def test_cpp_fused_exact_coordinate_harmonic_matches_direct() -> None:
    import pytest

    cpp = pytest.importorskip("waxs_cake._cpp_solvers")
    if not hasattr(cpp, "exact_beta_harmonic_coefficients_miller"):
        pytest.skip("C++ extension has not been rebuilt with the fused exact-beta kernel")
    rng = np.random.default_rng(20260715)
    coords = rng.normal(scale=0.45, size=(37, 3))
    q_perp = np.asarray([0.25, 0.9, 1.8])
    q_z = np.asarray([-0.01, -0.08, -0.24])
    phi = (np.arange(53) + 0.29) * (2.0 * np.pi / 53)
    coefficients = rng.normal(size=(coords.shape[0], q_perp.size)) + 1j * rng.normal(
        size=(coords.shape[0], q_perp.size)
    )

    expected = direct_axisymmetric(coords, q_perp, q_z, phi, coefficients)
    got, _ = exact_coordinate_harmonic_amplitude(
        coords,
        q_perp,
        q_z,
        phi,
        atom_coefficients=coefficients,
        harmonic_margin=24,
        bessel_backend="cpp_fused",
    )

    assert relative_l2(got, expected) < 2e-12


def test_cpp_factorized_exact_beta_matches_direct_without_atom_q_matrix() -> None:
    import pytest

    cpp = pytest.importorskip("waxs_cake._cpp_solvers")
    if not hasattr(cpp, "exact_beta_harmonic_coefficients_factorized_miller"):
        pytest.skip("C++ extension has not been rebuilt with the factorized kernel")
    rng = np.random.default_rng(20260716)
    coords = rng.normal(scale=0.38, size=(43, 3))
    q_perp = np.asarray([0.35, 1.05, 1.65])
    q_z = np.asarray([-0.015, -0.11, -0.23])
    phi = (np.arange(59) + 0.41) * (2.0 * np.pi / 59)
    element_indices = rng.integers(0, 3, size=coords.shape[0], dtype=np.int64)
    form_factors = rng.normal(size=(3, q_perp.size)) + 1j * rng.normal(
        size=(3, q_perp.size)
    )
    weights = rng.normal(size=coords.shape[0]) + 1j * rng.normal(
        size=coords.shape[0]
    )
    atom_coefficients = weights[:, None] * form_factors[element_indices]

    expected = direct_axisymmetric(
        coords, q_perp, q_z, phi, atom_coefficients
    )
    got, _ = exact_coordinate_harmonic_amplitude_factorized(
        coords,
        q_perp,
        q_z,
        phi,
        element_indices=element_indices,
        form_factors=form_factors,
        atom_weights=weights,
        harmonic_margin=24,
    )

    assert relative_l2(got, expected) < 2e-12


def test_prepared_factorized_direct_and_fft_synthesis_match_reference() -> None:
    rng = np.random.default_rng(20260717)
    coords = rng.normal(scale=0.17, size=(19, 3))
    q_perp = np.array([0.4, 1.1, 2.0])
    q_z = np.array([-0.03, -0.08, -0.15])
    n_phi = 64
    phi = (np.arange(n_phi) + 0.5) * (2.0 * np.pi / n_phi)
    element_indices = rng.integers(0, 2, size=coords.shape[0], dtype=np.int64)
    form_factors = rng.normal(size=(2, q_perp.size)).astype(np.complex128)
    weights = rng.normal(size=coords.shape[0]) + 1j * rng.normal(
        size=coords.shape[0]
    )
    expected, expected_cutoffs = exact_coordinate_harmonic_amplitude_factorized(
        coords,
        q_perp,
        q_z,
        phi,
        element_indices=element_indices,
        form_factors=form_factors,
        atom_weights=weights,
        harmonic_margin=8,
    )
    plan = PreparedExactCoordinateHarmonicPlan(
        coords,
        q_perp,
        q_z,
        phi,
        element_indices=element_indices,
        form_factors=form_factors,
        harmonic_margin=8,
    )
    direct, direct_cutoffs = plan.execute(
        atom_weights=weights, synthesis_backend="direct"
    )
    direct_profile = dict(plan.last_profile)
    via_fft, fft_cutoffs = plan.execute(
        atom_weights=weights, synthesis_backend="fft"
    )
    fused_plan = PreparedExactCoordinateHarmonicPlan(
        coords,
        q_perp,
        q_z,
        phi,
        element_indices=element_indices,
        form_factors=form_factors,
        harmonic_margin=8,
        coefficient_backend="fused_phase",
    )
    fused, fused_cutoffs = fused_plan.execute(
        atom_weights=weights, synthesis_backend="fft"
    )
    cached_plan = PreparedExactCoordinateHarmonicPlan(
        coords,
        q_perp,
        q_z,
        phi,
        element_indices=element_indices,
        form_factors=form_factors,
        harmonic_margin=8,
        coefficient_backend="cached_phase",
    )
    cached, cached_cutoffs = cached_plan.execute(
        atom_weights=weights, synthesis_backend="fft"
    )
    assert plan.fft_supported
    assert np.array_equal(direct_cutoffs, expected_cutoffs)
    assert np.array_equal(fft_cutoffs, expected_cutoffs)
    assert np.array_equal(fused_cutoffs, expected_cutoffs)
    assert np.array_equal(cached_cutoffs, expected_cutoffs)
    assert relative_l2(direct, expected) < 1e-12
    assert relative_l2(via_fft, expected) < 1e-12
    assert relative_l2(fused, expected) < 1e-12
    assert relative_l2(fused, via_fft) < 1e-12
    assert relative_l2(cached, via_fft) < 1e-12
    assert direct_profile["backend"] == "direct"
    assert plan.last_profile["backend"] == "fft"
    assert plan.setup_seconds >= 0.0


def test_exact_coordinate_harmonic_validates_shapes() -> None:
    coords = np.zeros((2, 3))
    with np.testing.assert_raises_regex(ValueError, "atom_coefficients"):
        exact_coordinate_harmonic_amplitude(
            coords,
            np.ones(2),
            np.zeros(2),
            np.zeros(3),
            atom_coefficients=np.ones((2, 3)),
        )
