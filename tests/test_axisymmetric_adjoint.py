from __future__ import annotations

import numpy as np
import pytest

from waxs_cake import (
    PreparedAxisymmetricOperator,
    binned_structure_grid,
    direct_axisymmetric_adjoint,
    prepare_axisymmetric_plan,
)
from scripts.validate_axisymmetric_adjoint import (
    form_factors,
    make_operator_problem,
    weighted_manifold,
)
from scripts.validate_axisymmetric_manifold_discrete import make_curvature_family


def make_problem(family: str = "spline"):
    template, rng = make_operator_problem(seed=73)
    manifold = weighted_manifold(make_curvature_family(n_u=8)[family])
    operator = PreparedAxisymmetricOperator(
        template,
        manifold,
        form_factors=form_factors(),
    )
    data = rng.normal(size=operator.data_shape) + 1j * rng.normal(size=operator.data_shape)
    return template, manifold, operator, data


def test_prepared_forward_matches_existing_circular_path() -> None:
    template, manifold, operator, _ = make_problem("ellipsoid")
    actual = operator.forward(template.hist)
    expected = prepare_axisymmetric_plan(
        template,
        manifold,
        form_factors=form_factors(),
        circular_backend="numpy",
    ).circular_fft()
    assert np.allclose(actual, expected, rtol=0.0, atol=2e-13)


def test_euclidean_adjoint_matches_direct_cartesian_reference() -> None:
    template, manifold, operator, data = make_problem()
    coords, elements = binned_structure_grid(template)
    direct = direct_axisymmetric_adjoint(
        coords,
        manifold,
        operator.phi,
        data,
        elements=elements,
        form_factors=form_factors(),
    ).reshape(operator.object_shape)
    actual = operator.adjoint_euclidean(data)
    relative_l2 = np.linalg.norm(actual - direct) / np.linalg.norm(direct)
    assert relative_l2 < 1e-12


def test_weighted_adjoint_matches_direct_and_dot_identity() -> None:
    template, manifold, operator, data = make_problem("paraboloid")
    coords, elements = binned_structure_grid(template)
    direct = direct_axisymmetric_adjoint(
        coords,
        manifold,
        operator.phi,
        data,
        elements=elements,
        form_factors=form_factors(),
        data_weights=manifold.resolved_data_weights,
    ).reshape(operator.object_shape)
    actual = operator.adjoint_weighted(data)
    relative_l2 = np.linalg.norm(actual - direct) / np.linalg.norm(direct)

    assert relative_l2 < 1e-12
    assert operator.adjoint_test(template.hist, data, weighted=True) < 1e-12


def test_weighted_adjoint_equals_euclidean_adjoint_of_weighted_data() -> None:
    _, manifold, operator, data = make_problem("sphere")
    weighted = operator.adjoint_weighted(data)
    explicit = operator.adjoint_euclidean(data * manifold.resolved_data_weights[:, None])
    assert np.allclose(weighted, explicit, rtol=0.0, atol=0.0)


def test_euclidean_dot_identity() -> None:
    template, _, operator, data = make_problem("sphere")
    assert operator.adjoint_test(template.hist, data) < 1e-12


def test_operator_rejects_wrong_shapes() -> None:
    _, _, operator, data = make_problem()
    with pytest.raises(ValueError, match="object values"):
        operator.forward(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="data values"):
        operator.adjoint_euclidean(data[:, :-1])
