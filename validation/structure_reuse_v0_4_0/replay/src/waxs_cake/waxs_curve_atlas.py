"""One-dimensional harmonic atlas helpers for an existing WAXS Ewald curve."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def uniform_atlas_indices(n_rows: int, n_knots: int) -> "NDArray[np.int64]":
    """Choose approximately uniform Ewald-curve rows, including both endpoints."""

    n_rows = int(n_rows)
    n_knots = int(n_knots)
    if n_rows < 2:
        raise ValueError("n_rows must be at least two")
    if n_knots < 2 or n_knots > n_rows:
        raise ValueError("n_knots must be in [2, n_rows]")
    indices = np.unique(
        np.rint(np.linspace(0, n_rows - 1, n_knots)).astype(np.int64)
    )
    if indices.size != n_knots:
        raise RuntimeError("uniform knot construction produced duplicate rows")
    indices.setflags(write=False)
    return indices


def interpolate_ewald_fourier_coefficients(
    q_coordinate: "ArrayLike",
    knot_indices: "ArrayLike",
    knot_coefficients: "ArrayLike",
    *,
    complex_dtype: np.dtype | str | None = None,
) -> "NDArray[np.complexfloating]":
    """Cubic-spline harmonic coefficients from selected rows to a full q grid.

    This interpolates complex *amplitudes* before the azimuthal inverse FFT.
    It does not commute with intensity formation.
    """

    q = np.asarray(q_coordinate, dtype=np.float64)
    indices = np.asarray(knot_indices, dtype=np.int64)
    coefficients = np.asarray(knot_coefficients)
    if q.ndim != 1 or q.size < 2 or not np.all(np.isfinite(q)):
        raise ValueError("q_coordinate must be a finite vector with at least two entries")
    if np.any(np.diff(q) <= 0.0):
        raise ValueError("q_coordinate must be strictly increasing")
    if indices.ndim != 1 or indices.size < 2:
        raise ValueError("knot_indices must contain at least two entries")
    if np.any(np.diff(indices) <= 0) or indices[0] < 0 or indices[-1] >= q.size:
        raise ValueError("knot_indices must be strictly increasing valid row indices")
    if coefficients.ndim != 2 or coefficients.shape[0] != indices.size:
        raise ValueError(
            "knot_coefficients must have shape (n_knots, n_harmonics)"
        )
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("knot_coefficients must contain only finite values")
    dtype = np.dtype(coefficients.dtype if complex_dtype is None else complex_dtype)
    if dtype not in {np.dtype(np.complex64), np.dtype(np.complex128)}:
        raise ValueError("complex_dtype must be complex64 or complex128")
    spline = CubicSpline(q[indices], coefficients, axis=0)
    return np.asarray(spline(q), dtype=dtype)
