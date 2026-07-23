from __future__ import annotations

import numpy as np

from waxs_cake import (
    PreparedAxisymmetricOperator,
    PreparedFinufftAxisymmetricReference,
    binned_structure_grid,
    direct_axisymmetric_amplitude,
)
from scripts.benchmark_axisymmetric_crossover import make_dense_benchmark_object
from scripts.validate_axisymmetric_harmonic_cutoff import (
    active_radius_max,
    matched_curvature_family,
)


def test_prepared_object_split_matches_forward() -> None:
    histogram = make_dense_benchmark_object(32, n_r=3, n_z=3)
    manifold = matched_curvature_family(6.0, active_radius_max(histogram), n_u=5)["spline"]
    operator = PreparedAxisymmetricOperator(histogram, manifold)
    prepared = operator.prepare_object(histogram.hist)
    split = operator.apply_prepared_object(prepared)
    full = operator.forward(histogram.hist)
    assert np.allclose(split, full, rtol=0.0, atol=0.0)


def test_finufft_plan_matches_direct_and_acfo() -> None:
    histogram = make_dense_benchmark_object(24, n_r=2, n_z=2)
    manifold = matched_curvature_family(4.0, active_radius_max(histogram), n_u=4)["ellipsoid"]
    coords, _ = binned_structure_grid(histogram)
    weights = np.asarray(histogram.hist).ravel()
    plan = PreparedFinufftAxisymmetricReference(
        coords,
        manifold,
        histogram.beta_centers,
        eps=1e-10,
        nthreads=1,
    )
    nufft = plan.execute(weights)
    direct = direct_axisymmetric_amplitude(
        coords,
        manifold,
        histogram.beta_centers,
        source_weights=weights,
    )
    acfo = PreparedAxisymmetricOperator(histogram, manifold).forward(histogram.hist)

    assert np.linalg.norm(nufft - direct) / np.linalg.norm(direct) < 1e-9
    assert np.linalg.norm(acfo - direct) / np.linalg.norm(direct) < 1e-12


def test_finufft_adjoint_satisfies_euclidean_dot_identity() -> None:
    histogram = make_dense_benchmark_object(16, n_r=2, n_z=2)
    manifold = matched_curvature_family(3.0, active_radius_max(histogram), n_u=4)[
        "ellipsoid"
    ]
    coords, _ = binned_structure_grid(histogram)
    plan = PreparedFinufftAxisymmetricReference(
        coords,
        manifold,
        histogram.beta_centers,
        eps=1e-11,
        nthreads=1,
    )
    rng = np.random.default_rng(51)
    source = rng.normal(size=coords.shape[0]) + 1j * rng.normal(size=coords.shape[0])
    data = rng.normal(size=plan.data_shape) + 1j * rng.normal(size=plan.data_shape)
    left = np.vdot(plan.execute(source), data)
    right = np.vdot(source, plan.adjoint(data))
    relative_error = abs(left - right) / (abs(left) + abs(right))
    assert relative_error < 1e-9
