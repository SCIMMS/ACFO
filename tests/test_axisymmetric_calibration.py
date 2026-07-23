from __future__ import annotations

import numpy as np
import pytest

from waxs_cake import (
    AnchoredBSplineCurvatureModel,
    AxisymmetricManifold,
    EllipsoidCurvatureModel,
    PreparedAxisymmetricOperator,
    curvature_loss_and_gradient,
)
from scripts.validate_axisymmetric_manifold_discrete import make_validation_object


def test_geometry_jacobian_action_matches_central_difference() -> None:
    template = make_validation_object()
    u = np.linspace(0.04, 1.1, 7)
    manifold = AxisymmetricManifold(
        u,
        2.6 * np.sin(u),
        1.3 * (np.cos(u) - 1.0),
    )
    operator = PreparedAxisymmetricOperator(template, manifold)
    delta_perp = np.linspace(0.2, -0.1, u.size)
    delta_z = np.linspace(-0.15, 0.25, u.size)
    analytic = operator.geometry_jacobian_action(template.hist, delta_perp, delta_z)

    epsilon = 1e-6
    plus = AxisymmetricManifold(
        u,
        manifold.q_perp + epsilon * delta_perp,
        manifold.q_z + epsilon * delta_z,
    )
    minus = AxisymmetricManifold(
        u,
        manifold.q_perp - epsilon * delta_perp,
        manifold.q_z - epsilon * delta_z,
    )
    finite_difference = (
        PreparedAxisymmetricOperator(template, plus).forward(template.hist)
        - PreparedAxisymmetricOperator(template, minus).forward(template.hist)
    ) / (2.0 * epsilon)
    relative_l2 = np.linalg.norm(analytic - finite_difference) / np.linalg.norm(
        finite_difference
    )
    assert relative_l2 < 2e-9


def test_ellipsoid_loss_gradient_matches_finite_difference() -> None:
    template = make_validation_object()
    model = EllipsoidCurvatureModel(np.linspace(0.02, 1.2, 9))
    truth = np.array([3.0, 1.4])
    data = PreparedAxisymmetricOperator(template, model.manifold(truth)).forward(
        template.hist
    )
    parameters = np.array([2.4, 1.9])
    _, analytic = curvature_loss_and_gradient(template, model, parameters, data)
    finite_difference = np.empty_like(parameters)
    epsilon = 1e-6
    for index in range(parameters.size):
        step = np.zeros_like(parameters)
        step[index] = epsilon
        plus = curvature_loss_and_gradient(template, model, parameters + step, data)[0]
        minus = curvature_loss_and_gradient(template, model, parameters - step, data)[0]
        finite_difference[index] = (plus - minus) / (2.0 * epsilon)
    relative_l2 = np.linalg.norm(analytic - finite_difference) / np.linalg.norm(
        finite_difference
    )
    assert relative_l2 < 2e-8


def test_anchored_bspline_contract_and_basis() -> None:
    model = AnchoredBSplineCurvatureModel(np.linspace(0.0, 1.0, 17))
    q_perp = np.array([0.0, 0.4, 1.8, 1.3, 2.7])
    q_z = np.array([0.0, -0.2, -0.7, -0.3, -1.1])
    parameters = model.parameters_from_controls(q_perp, q_z)
    manifold = model.manifold(parameters)

    assert np.allclose(np.sum(model.basis, axis=1), 1.0)
    assert np.min(model.basis) >= -1e-15
    assert manifold.q_perp[0] == pytest.approx(0.0)
    assert manifold.q_z[0] == pytest.approx(0.0)
    assert manifold.q_perp[-1] == pytest.approx(q_perp[-1])
    assert manifold.q_z[-1] == pytest.approx(q_z[-1])
    with pytest.raises(ValueError, match="non-negative"):
        model.manifold(np.concatenate(([-0.1, 1.0, 1.0, 1.0], q_z[1:])))


def test_geometry_derivative_rejects_unspecified_form_factor_derivative() -> None:
    template = make_validation_object()
    model = EllipsoidCurvatureModel(np.linspace(0.02, 0.9, 5))
    operator = PreparedAxisymmetricOperator(
        template,
        model.manifold([2.0, 1.0]),
        form_factors={"X": np.ones(5)},
    )
    with pytest.raises(ValueError, match="form-factor derivatives"):
        operator.geometry_derivatives(template.hist)
