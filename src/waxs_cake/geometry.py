"""Reciprocal-space geometry for WAXS cake maps."""

from __future__ import annotations

import numpy as np


def ewald_ring(q: np.ndarray, wavelength: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``q_perp`` and ``q_z`` for Ewald-sphere rings.

    The input ``q`` is the scattering-vector magnitude in inverse length units.
    The incident wavevector magnitude is ``k = 2*pi / wavelength``. The returned
    ring parameterization is ``(q_perp cos(phi), q_perp sin(phi), q_z)``.
    """

    q = np.asarray(q, dtype=float)
    if wavelength <= 0:
        raise ValueError("wavelength must be positive")
    if np.any(q < 0):
        raise ValueError("q values must be non-negative")

    k = 2.0 * np.pi / wavelength
    if np.any(q > 2.0 * k * (1.0 + 1e-12)):
        raise ValueError("q cannot exceed 2k for elastic Ewald-sphere geometry")

    q_z = -(q**2) / (2.0 * k)
    inside = np.maximum(0.0, 1.0 - (q / (2.0 * k)) ** 2)
    q_perp = q * np.sqrt(inside)
    return q_perp, q_z
