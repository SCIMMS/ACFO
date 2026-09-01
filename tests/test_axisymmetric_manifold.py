from __future__ import annotations

import numpy as np
import pytest

from waxs_cake import (
    AxisymmetricManifold,
    PreparedCakePlan,
    make_cylindrical_histogram,
    parameter_trapezoid_weights,
    prepare_axisymmetric_plan,
    surface_radial_weights,
)


def test_manifold_contract_and_nodes() -> None:
    manifold = AxisymmetricManifold(
        u=[0.0, 0.5, 1.0],
        q_perp=[0.0, 1.0, 0.5],
        q_z=[0.0, -0.2, -0.4],
    )
    phi = np.array([0.0, 0.5 * np.pi])
    nodes = manifold.target_nodes(phi)

    assert manifold.n_u == 3
    assert nodes.shape == (3, 2, 3)
    assert np.allclose(nodes[1, 0], [1.0, 0.0, -0.2])
    assert np.allclose(nodes[1, 1], [0.0, 1.0, -0.2], atol=1e-15)
    assert np.allclose(manifold.q_norm, np.hypot(manifold.q_perp, manifold.q_z))
    assert np.array_equal(manifold.resolved_data_weights, np.ones(3))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"u": [0.0, 0.0], "q_perp": [0.0, 1.0], "q_z": [0.0, 0.0]}, "strictly increasing"),
        ({"u": [0.0], "q_perp": [-1.0], "q_z": [0.0]}, "non-negative"),
        ({"u": [0.0], "q_perp": [1.0, 2.0], "q_z": [0.0]}, "matching shapes"),
        (
            {"u": [0.0, 1.0], "q_perp": [0.0, 1.0], "q_z": [0.0, 0.0], "data_weights": [0.0, 0.0]},
            "at least one positive",
        ),
    ],
)
def test_manifold_rejects_invalid_contract(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        AxisymmetricManifold(**kwargs)


def test_nonmonotone_geometry_and_duplicate_nodes_are_valid() -> None:
    manifold = AxisymmetricManifold(
        u=[0.0, 1.0, 2.0],
        q_perp=[1.0, 0.5, 1.0],
        q_z=[0.0, 0.2, 0.0],
    )
    assert np.array_equal(manifold.q_perp[[0, 2]], [1.0, 1.0])
    assert np.array_equal(manifold.q_z[[0, 2]], [0.0, 0.0])


def test_callback_and_ewald_factories() -> None:
    u = np.linspace(0.0, 0.8, 5)
    ellipsoid = AxisymmetricManifold.from_callback(
        u,
        lambda value: (2.0 * np.sin(value), 0.5 * (np.cos(value) - 1.0)),
        name="ellipsoid",
    )
    sphere = AxisymmetricManifold.ewald_sphere(np.linspace(0.1, 1.0, 5), 1.0)

    assert np.allclose(ellipsoid.q_perp, 2.0 * np.sin(u))
    assert ellipsoid.interpretation == "sampling"
    assert sphere.interpretation == "dispersion-derived"
    assert sphere.name == "elastic-ewald-sphere"


def test_parameter_and_surface_weights_have_explicit_semantics() -> None:
    u = np.array([0.0, 0.5, 1.0])
    u_weights = parameter_trapezoid_weights(u)
    weights = surface_radial_weights(
        q_perp=[0.0, 1.0, 2.0],
        dq_perp_du=[2.0, 2.0, 2.0],
        dq_z_du=[0.0, 0.0, 0.0],
        u_weights=u_weights,
    )

    assert np.allclose(u_weights, [0.25, 0.5, 0.25])
    assert np.allclose(weights, [0.0, 1.0, 1.0])


def test_adapter_matches_explicit_geometry_plan_and_preserves_q_norm() -> None:
    rng = np.random.default_rng(20260711)
    coords = rng.normal(size=(120, 3))
    binned = make_cylindrical_histogram(coords, n_r=10, n_z=9, n_phi=64)
    u = np.linspace(0.05, 0.9, 6)
    manifold = AxisymmetricManifold.from_callback(
        u,
        lambda value: (1.5 * np.sin(value), 0.7 * (np.cos(value) - 1.0)),
        name="ellipsoidal-test-grid",
    )

    adapted = prepare_axisymmetric_plan(binned, manifold, circular_backend="numpy")
    explicit = PreparedCakePlan(
        binned,
        manifold.q_norm,
        1.0,
        q_perp=manifold.q_perp,
        q_z=manifold.q_z,
        circular_backend="numpy",
    )

    assert np.array_equal(adapted.q, manifold.q_norm)
    assert np.allclose(adapted.circular_fft(), explicit.circular_fft(), rtol=0.0, atol=0.0)


def test_ewald_adapter_is_a_legacy_path_regression() -> None:
    rng = np.random.default_rng(7)
    coords = rng.normal(size=(100, 3))
    binned = make_cylindrical_histogram(coords, n_r=9, n_z=8, n_phi=64)
    q = np.linspace(0.1, 1.0, 5)
    manifold = AxisymmetricManifold.ewald_sphere(q, 1.0)

    legacy = PreparedCakePlan(binned, q, 1.0, circular_backend="numpy").circular_fft()
    adapted = prepare_axisymmetric_plan(
        binned,
        manifold,
        circular_backend="numpy",
    ).circular_fft()

    assert np.allclose(adapted, legacy, rtol=0.0, atol=0.0)
