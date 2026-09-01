from __future__ import annotations

import numpy as np
import pytest

from waxs_cake import (
    AxisymmetricManifold,
    binned_structure_sources,
    complex_error_metrics,
    direct_axisymmetric_amplitude,
    prepare_axisymmetric_plan,
)
from scripts.validate_axisymmetric_manifold_discrete import (
    make_curvature_family,
    make_validation_object,
)


def test_direct_reference_matches_explicit_node_matrix() -> None:
    coords = np.array([[0.2, -0.4, 0.5], [-0.6, 0.3, -0.2]])
    weights = np.array([1.0 + 0.2j, -0.5 + 0.7j])
    manifold = AxisymmetricManifold(
        u=[0.0, 1.0],
        q_perp=[0.8, 1.3],
        q_z=[-0.2, 0.4],
    )
    phi = np.array([0.1, 1.2, 2.4])

    actual = direct_axisymmetric_amplitude(
        coords,
        manifold,
        phi,
        source_weights=weights,
    )
    nodes = manifold.target_nodes(phi)
    expected = np.empty_like(actual)
    for u_index in range(manifold.n_u):
        expected[u_index] = np.sum(
            weights[:, None]
            * np.exp(1j * (coords @ nodes[u_index].T)),
            axis=0,
        )

    assert np.allclose(actual, expected, rtol=0.0, atol=1e-15)


def test_phase_metric_masks_near_zero_reference() -> None:
    reference = np.array([1.0 + 0.0j, 1e-12 + 0.0j])
    actual = np.array([np.exp(0.2j), -1e-12 + 0.0j])
    metrics = complex_error_metrics(actual, reference, phase_threshold_fraction=1e-3)

    assert metrics.phase_mask_count == 1
    assert metrics.phase_rms_rad == pytest.approx(0.2)


@pytest.mark.parametrize("family", ["sphere", "ellipsoid", "paraboloid", "spline"])
def test_curvature_family_matches_independent_cartesian_sum(family: str) -> None:
    binned = make_validation_object()
    manifold = make_curvature_family()[family]
    coords, elements, weights = binned_structure_sources(binned)
    reference = direct_axisymmetric_amplitude(
        coords,
        manifold,
        binned.beta_centers,
        elements=elements,
        source_weights=weights,
    )
    actual = prepare_axisymmetric_plan(
        binned,
        manifold,
        circular_backend="numpy",
        complex_dtype=np.complex128,
    ).circular_fft()
    metrics = complex_error_metrics(actual, reference)

    assert metrics.relative_l2 < 1e-12
    assert metrics.relative_linf < 1e-12
    assert metrics.phase_rms_rad < 1e-12


def test_validation_object_exposes_qz_errors() -> None:
    binned = make_validation_object()
    coords, elements, weights = binned_structure_sources(binned)
    for manifold in make_curvature_family().values():
        reference = direct_axisymmetric_amplitude(
            coords,
            manifold,
            binned.beta_centers,
            elements=elements,
            source_weights=weights,
        )
        flat = AxisymmetricManifold(
            manifold.u,
            manifold.q_perp,
            np.zeros(manifold.n_u),
        )
        flat_reference = direct_axisymmetric_amplitude(
            coords,
            flat,
            binned.beta_centers,
            elements=elements,
            source_weights=weights,
        )
        relative_l2 = np.linalg.norm(flat_reference - reference) / np.linalg.norm(reference)
        assert relative_l2 > 1e-2
