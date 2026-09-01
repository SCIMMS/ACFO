from __future__ import annotations

import numpy as np

from waxs_cake import (
    AxisymmetricManifold,
    PreparedDoubleManifoldOperator,
    direct_double_manifold_adjoint,
    direct_double_manifold_forward,
)


def make_problem(seed: int = 41):
    r = np.array([0.35, 1.05])
    z = np.array([-0.55, 0.60])
    beta = AxisymmetricManifold.uniform_phi(16)
    outgoing_phi = AxisymmetricManifold.uniform_phi(12)
    incident_phi = AxisymmetricManifold.uniform_phi(8)
    u_out = np.linspace(0.08, 0.90, 3)
    u_in = np.linspace(0.06, 0.70, 2)
    outgoing = AxisymmetricManifold(
        u_out,
        3.0 * np.sin(u_out),
        3.0 * np.cos(u_out),
        name="test-sphere-outgoing",
    )
    incident = AxisymmetricManifold(
        u_in,
        2.7 * np.sin(u_in),
        3.2 * np.cos(u_in),
        name="test-ellipsoid-incident",
    )
    rng = np.random.default_rng(seed)
    object_values = rng.normal(size=(2, 2, 16)) + 1j * rng.normal(size=(2, 2, 16))
    return r, z, beta, outgoing, incident, outgoing_phi, incident_phi, object_values, rng


def test_double_harmonic_forward_matches_direct_type3_nudft() -> None:
    r, z, beta, outgoing, incident, phi_out, phi_in, values, _ = make_problem()
    reference = direct_double_manifold_forward(
        values,
        r,
        z,
        beta,
        outgoing,
        incident,
        phi_out,
        phi_in,
    )
    operator = PreparedDoubleManifoldOperator(
        r,
        z,
        beta,
        outgoing,
        incident,
        phi_out,
        phi_in,
        harmonic_cutoff=16,
    )
    relative_l2 = np.linalg.norm(operator.forward(values) - reference) / np.linalg.norm(
        reference
    )
    assert relative_l2 < 3e-13


def test_double_harmonic_adjoint_matches_direct_and_dot_identity() -> None:
    r, z, beta, outgoing, incident, phi_out, phi_in, values, rng = make_problem(43)
    operator = PreparedDoubleManifoldOperator(
        r,
        z,
        beta,
        outgoing,
        incident,
        phi_out,
        phi_in,
        harmonic_cutoff=16,
    )
    data = rng.normal(size=operator.data_shape) + 1j * rng.normal(
        size=operator.data_shape
    )
    direct = direct_double_manifold_adjoint(
        data,
        r,
        z,
        beta,
        outgoing,
        incident,
        phi_out,
        phi_in,
    )
    relative_l2 = np.linalg.norm(operator.adjoint_euclidean(data) - direct) / np.linalg.norm(
        direct
    )
    assert relative_l2 < 3e-13
    assert operator.adjoint_test(values, data) < 1e-12


def test_double_harmonic_cutoff_converges() -> None:
    r, z, beta, outgoing, incident, phi_out, phi_in, values, _ = make_problem(47)
    reference = direct_double_manifold_forward(
        values,
        r,
        z,
        beta,
        outgoing,
        incident,
        phi_out,
        phi_in,
    )
    errors = []
    for cutoff in (6, 10, 14):
        operator = PreparedDoubleManifoldOperator(
            r,
            z,
            beta,
            outgoing,
            incident,
            phi_out,
            phi_in,
            harmonic_cutoff=cutoff,
        )
        errors.append(
            np.linalg.norm(operator.forward(values) - reference) / np.linalg.norm(reference)
        )
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 1e-10
