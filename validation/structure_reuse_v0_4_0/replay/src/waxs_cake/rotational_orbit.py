"""Geometry and Fourier-phase utilities for rotational tomography orbits.

The axisymmetric ACFO core samples circles around its local ``z`` axis.  A
single detector-frequency node in rotational ODT traces exactly such a circle
around the sample-rotation axis, but generally starts at a node-dependent
azimuth.  This module separates that physical orbit into the meridional
coordinates consumed by ACFO and the residual Fourier phase shift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def _unit_vector(value: "ArrayLike", *, field: str) -> "NDArray[np.float64]":
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{field} must be a finite three-vector")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"{field} must be nonzero")
    return vector / norm


def axis_frame(axis: "ArrayLike") -> tuple["NDArray[np.float64]", ...]:
    """Return a deterministic right-handed ``(e1, e2, axis)`` frame."""

    axis_unit = _unit_vector(axis, field="axis")
    candidates = np.eye(3, dtype=np.float64)
    seed = candidates[int(np.argmin(np.abs(candidates @ axis_unit)))]
    e1 = seed - np.dot(seed, axis_unit) * axis_unit
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis_unit, e1)
    return e1, e2, axis_unit


@dataclass(frozen=True)
class RotationalOrbitCoordinates:
    """Meridional ACFO coordinates plus one starting phase per orbit."""

    q_perp: "NDArray[np.float64]"
    q_parallel: "NDArray[np.float64]"
    phi0: "NDArray[np.float64]"
    e1: "NDArray[np.float64]"
    e2: "NDArray[np.float64]"
    axis: "NDArray[np.float64]"


def decompose_rotational_orbits(
    base_q: "ArrayLike", axis: "ArrayLike"
) -> RotationalOrbitCoordinates:
    """Decompose base reciprocal vectors into circles around ``axis``."""

    q = np.asarray(base_q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 3 or not np.all(np.isfinite(q)):
        raise ValueError("base_q must be a finite array with shape (n, 3)")
    e1, e2, axis_unit = axis_frame(axis)
    q_parallel = q @ axis_unit
    component1 = q @ e1
    component2 = q @ e2
    q_perp = np.hypot(component1, component2)
    phi0 = np.arctan2(component2, component1)
    phi0 = np.where(q_perp == 0.0, 0.0, phi0)
    return RotationalOrbitCoordinates(
        q_perp=q_perp,
        q_parallel=q_parallel,
        phi0=phi0,
        e1=e1,
        e2=e2,
        axis=axis_unit,
    )


def reconstruct_rotational_orbits(
    coordinates: RotationalOrbitCoordinates,
    angles: "ArrayLike",
) -> "NDArray[np.float64]":
    """Return orbit nodes with shape ``(n_orbit, n_angle, 3)``."""

    angle = np.asarray(angles, dtype=np.float64)
    if angle.ndim != 1 or not np.all(np.isfinite(angle)):
        raise ValueError("angles must be a finite one-dimensional array")
    phase = coordinates.phi0[:, None] + angle[None, :]
    radial = (
        np.cos(phase)[..., None] * coordinates.e1
        + np.sin(phase)[..., None] * coordinates.e2
    )
    return (
        coordinates.q_parallel[:, None, None] * coordinates.axis
        + coordinates.q_perp[:, None, None] * radial
    )


def rotate_vectors_about_axis(
    vectors: "ArrayLike", axis: "ArrayLike", angles: "ArrayLike"
) -> "NDArray[np.float64]":
    """Rotate each vector through every angle using Rodrigues' formula."""

    q = np.asarray(vectors, dtype=np.float64)
    angle = np.asarray(angles, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 3 or not np.all(np.isfinite(q)):
        raise ValueError("vectors must be a finite array with shape (n, 3)")
    if angle.ndim != 1 or not np.all(np.isfinite(angle)):
        raise ValueError("angles must be a finite one-dimensional array")
    axis_unit = _unit_vector(axis, field="axis")
    cos_angle = np.cos(angle)[None, :, None]
    sin_angle = np.sin(angle)[None, :, None]
    parallel = (q @ axis_unit)[:, None, None] * axis_unit
    perpendicular = q[:, None, :] - parallel
    cross = np.cross(axis_unit, q)[:, None, :]
    return parallel + perpendicular * cos_angle + cross * sin_angle


def shift_orbit_fourier_coefficients(
    coefficients: "ArrayLike", phi0: "ArrayLike"
) -> "NDArray[np.complexfloating]":
    """Shift IFFT-series coefficients from ``theta`` to ``theta + phi0``.

    ``coefficients`` must have shape ``(n_orbit, n_angle)`` and use NumPy FFT
    storage order.  The result can be passed directly to ``numpy.fft.ifft``.
    """

    values = np.asarray(coefficients)
    offset = np.asarray(phi0, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("coefficients must have shape (n_orbit, n_angle)")
    if offset.shape != (values.shape[0],) or not np.all(np.isfinite(offset)):
        raise ValueError("phi0 must contain one finite value per orbit")
    modes = np.rint(np.fft.fftfreq(values.shape[1]) * values.shape[1])
    phase = np.exp(1j * offset[:, None] * modes[None, :])
    return values * phase
