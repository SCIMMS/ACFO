from __future__ import annotations

import numpy as np

from waxs_cake import (
    gayer_5mol_mgo_cln_index,
    linbo3_3m_nonlinear_polarization,
    project_vector_born_field,
    uniaxial_eigenpolarization,
)


def test_gayer_indices_are_negative_uniaxial_at_1064_and_532_nm() -> None:
    for wavelength in (1.064, 0.532):
        ordinary = gayer_5mol_mgo_cln_index(wavelength, "ordinary")
        extraordinary = gayer_5mol_mgo_cln_index(wavelength, "extraordinary")
        assert ordinary > extraordinary > 2.0


def test_3m_tensor_contraction_for_z_polarized_pump() -> None:
    polarization = linbo3_3m_nonlinear_polarization(np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(polarization, np.array([0.0, 0.0, -25.0]))


def test_uniaxial_eigenpolarizations_are_normalized_and_transverse_in_d() -> None:
    q_perp = np.array([0.4, 0.8])
    q_z = np.array([2.0, 1.8])
    phi = np.array([0.0, 0.7, 1.3])
    eps_parallel = 4.6
    eps_perpendicular = 5.0
    ordinary = uniaxial_eigenpolarization(
        q_perp, q_z, phi,
        epsilon_parallel=eps_parallel,
        epsilon_perpendicular=eps_perpendicular,
        branch="ordinary",
    )
    extraordinary = uniaxial_eigenpolarization(
        q_perp, q_z, phi,
        epsilon_parallel=eps_parallel,
        epsilon_perpendicular=eps_perpendicular,
        branch="extraordinary",
    )
    np.testing.assert_allclose(np.linalg.norm(ordinary, axis=-1), 1.0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(extraordinary, axis=-1), 1.0, atol=1e-14)
    k = np.stack(
        (
            q_perp[:, None] * np.cos(phi)[None, :],
            q_perp[:, None] * np.sin(phi)[None, :],
            np.broadcast_to(q_z[:, None], (q_z.size, phi.size)),
        ),
        axis=-1,
    )
    d_extraordinary = extraordinary * np.array([eps_perpendicular, eps_perpendicular, eps_parallel])
    np.testing.assert_allclose(np.sum(k * d_extraordinary, axis=-1), 0.0, atol=1e-14)


def test_vector_projection_shape_and_zero_orthogonal_coupling() -> None:
    amplitude = np.ones((2, 3), dtype=np.complex128)
    eigen = np.zeros((2, 3, 3), dtype=np.float64)
    eigen[..., 0] = 1.0
    field = project_vector_born_field(amplitude, eigen, np.array([0.0, 1.0, 0.0]))
    np.testing.assert_array_equal(field, np.zeros_like(field))
