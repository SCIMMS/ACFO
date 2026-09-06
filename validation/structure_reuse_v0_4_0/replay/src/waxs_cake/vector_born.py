"""Minimal vector first-Born ingredients for uniaxial SHG validation.

This module fixes material and tensor conventions needed by the NCS validation
program.  It does not implement pump depletion, interfaces, multiple
scattering, or a full Maxwell boundary-value solver.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


Polarization = Literal["ordinary", "extraordinary"]


_GAYER_5MOL_MGO_CLN = {
    "extraordinary": (5.756, 0.0983, 0.2020, 189.32, 12.52, 1.32e-2, 2.860e-6, 4.700e-8, 6.113e-8, 1.516e-4),
    "ordinary": (5.653, 0.1185, 0.2091, 89.61, 10.85, 1.97e-2, 7.941e-7, 3.134e-8, -4.641e-9, -2.188e-6),
}


def gayer_5mol_mgo_cln_index(
    wavelength_um: float,
    polarization: Polarization,
    *,
    temperature_c: float = 24.5,
) -> float:
    """Return the Gayer et al. index for 5 mol% MgO-doped congruent LN.

    Wavelength is in micrometres.  The reference temperature in the published
    equation is 24.5 degrees Celsius.
    """

    wavelength_um = float(wavelength_um)
    temperature_c = float(temperature_c)
    if polarization not in _GAYER_5MOL_MGO_CLN:
        raise ValueError("polarization must be ordinary or extraordinary")
    if not np.isfinite(wavelength_um) or wavelength_um <= 0.0:
        raise ValueError("wavelength_um must be finite and positive")
    if not np.isfinite(temperature_c):
        raise ValueError("temperature_c must be finite")
    a1, a2, a3, a4, a5, a6, b1, b2, b3, b4 = _GAYER_5MOL_MGO_CLN[polarization]
    f = (temperature_c - 24.5) * (temperature_c + 24.5 + 2.0 * 273.16)
    lam2 = wavelength_um * wavelength_um
    n2 = (
        a1
        + b1 * f
        + (a2 + b2 * f) / (lam2 - (a3 + b3 * f) ** 2)
        + (a4 + b4 * f) / (lam2 - a5 * a5)
        - a6 * lam2
    )
    if not np.isfinite(n2) or n2 <= 0.0:
        raise ValueError("Sellmeier equation is invalid at this wavelength/temperature")
    return float(np.sqrt(n2))


def linbo3_3m_nonlinear_polarization(
    pump_e: np.ndarray,
    *,
    d22_pm_per_v: float = 4.08,
    d31_pm_per_v: float = -4.4,
    d33_pm_per_v: float = -25.0,
) -> np.ndarray:
    """Contract the standard LiNbO3 3m ``d`` matrix with one pump field.

    The contracted input order is ``xx, yy, zz, 2yz, 2xz, 2xy``.  Returned
    values retain the supplied pm/V scale; a global electromagnetic prefactor
    is intentionally omitted because the validation compares normalized
    complex vector fields.
    """

    field = np.asarray(pump_e, dtype=np.complex128)
    if field.shape != (3,) or not np.all(np.isfinite(field)):
        raise ValueError("pump_e must be a finite complex vector with shape (3,)")
    d22 = float(d22_pm_per_v)
    d31 = float(d31_pm_per_v)
    d33 = float(d33_pm_per_v)
    if not np.all(np.isfinite((d22, d31, d33))):
        raise ValueError("nonlinear coefficients must be finite")
    ex, ey, ez = field
    contracted = np.array(
        [ex * ex, ey * ey, ez * ez, 2.0 * ey * ez, 2.0 * ex * ez, 2.0 * ex * ey],
        dtype=np.complex128,
    )
    d_matrix = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, d31, -d22],
            [-d22, d22, 0.0, d31, 0.0, 0.0],
            [d31, d31, d33, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return d_matrix @ contracted


def uniaxial_eigenpolarization(
    q_perp: np.ndarray,
    q_z: np.ndarray,
    phi: np.ndarray,
    *,
    epsilon_parallel: float,
    epsilon_perpendicular: float,
    branch: Polarization,
) -> np.ndarray:
    """Return normalized outgoing electric eigenpolarizations on a ring grid.

    The optic axis is ``z`` and the dielectric tensor is
    ``diag(epsilon_perpendicular, epsilon_perpendicular, epsilon_parallel)``.
    Extraordinary polarization is formed by taking a transverse displacement
    field and applying ``epsilon**-1``.
    """

    q_perp = np.asarray(q_perp, dtype=np.float64)
    q_z = np.asarray(q_z, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    if q_perp.ndim != 1 or q_z.shape != q_perp.shape or phi.ndim != 1:
        raise ValueError("q_perp/q_z must be matching vectors and phi must be a vector")
    if not np.all(np.isfinite(q_perp)) or not np.all(np.isfinite(q_z)) or not np.all(np.isfinite(phi)):
        raise ValueError("wave-vector inputs must be finite")
    eps_par = float(epsilon_parallel)
    eps_perp = float(epsilon_perpendicular)
    if min(eps_par, eps_perp) <= 0.0 or not np.all(np.isfinite((eps_par, eps_perp))):
        raise ValueError("dielectric constants must be finite and positive")
    cos_phi = np.cos(phi)[None, :]
    sin_phi = np.sin(phi)[None, :]
    if branch == "ordinary":
        result = np.empty((q_perp.size, phi.size, 3), dtype=np.float64)
        result[..., 0] = -sin_phi
        result[..., 1] = cos_phi
        result[..., 2] = 0.0
        return result
    if branch != "extraordinary":
        raise ValueError("branch must be ordinary or extraordinary")
    k_norm = np.hypot(q_perp, q_z)
    if np.any(k_norm == 0.0):
        raise ValueError("extraordinary polarization is undefined at zero wave vector")
    d_r = q_z / k_norm
    d_z = -q_perp / k_norm
    e_r = d_r / eps_perp
    e_z = d_z / eps_par
    norm = np.sqrt(e_r * e_r + e_z * e_z)
    e_r = e_r / norm
    e_z = e_z / norm
    result = np.empty((q_perp.size, phi.size, 3), dtype=np.float64)
    result[..., 0] = e_r[:, None] * cos_phi
    result[..., 1] = e_r[:, None] * sin_phi
    result[..., 2] = e_z[:, None]
    return result


def project_vector_born_field(
    scalar_amplitude: np.ndarray,
    eigenpolarization: np.ndarray,
    nonlinear_polarization: np.ndarray,
) -> np.ndarray:
    """Project one nonlinear source vector onto outgoing eigenpolarizations."""

    amplitude = np.asarray(scalar_amplitude, dtype=np.complex128)
    eigen = np.asarray(eigenpolarization, dtype=np.float64)
    source = np.asarray(nonlinear_polarization, dtype=np.complex128)
    if eigen.shape != amplitude.shape + (3,) or source.shape != (3,):
        raise ValueError("amplitude/eigenpolarization/source shapes are inconsistent")
    coupling = np.einsum("...c,c->...", eigen, source)
    return amplitude[..., None] * coupling[..., None] * eigen
