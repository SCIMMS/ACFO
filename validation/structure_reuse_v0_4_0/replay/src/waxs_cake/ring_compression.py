"""Diagnostics and reusable primitives for skeleton-ring compression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import qr

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from .axisymmetric_operator import PreparedAxisymmetricOperator


def effective_mode_kernel(
    operator: "PreparedAxisymmetricOperator",
    mode_index: int,
) -> "NDArray[np.complexfloating]":
    """Return the ring-by-source kernel for one stored azimuthal FFT mode.

    Columns use flattened ``(element, r, z)`` order and act on the matching
    slice of the object's azimuthal FFT.
    """

    index = int(mode_index)
    if index < 0 or index >= operator.phi.size:
        raise ValueError(f"mode_index must be in [0, {operator.phi.size})")
    kernel = np.einsum(
        "ej,jz,jr->jerz",
        operator.form_factors,
        operator.z_phase,
        operator.kernel_fft[:, :, index],
        optimize=True,
    )
    return kernel.reshape(operator.manifold.n_u, -1).astype(
        operator.complex_dtype,
        copy=False,
    )


def relative_frobenius_rank(matrix: "ArrayLike", tolerance: float) -> int:
    """Return the smallest SVD rank with relative Frobenius error <= tolerance."""

    values = np.asarray(matrix)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("matrix must be a finite two-dimensional array")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0 or tolerance >= 1.0:
        raise ValueError("tolerance must be finite and in [0, 1)")
    singular_values = np.linalg.svd(values, compute_uv=False)
    energy = np.square(singular_values.astype(np.float64))
    total = float(np.sum(energy))
    if total == 0.0:
        return 0
    target = (1.0 - tolerance * tolerance) * total
    rank = int(np.searchsorted(np.cumsum(energy), target, side="left") + 1)
    return min(rank, singular_values.size)


@dataclass(frozen=True)
class RowSkeleton:
    """Interpolative row skeleton ``matrix ~= interpolation @ matrix[indices]``."""

    indices: "NDArray[np.int64]"
    interpolation: "NDArray[np.complexfloating]"
    relative_frobenius_error: float


def row_skeleton(matrix: "ArrayLike", rank: int) -> RowSkeleton:
    """Build a pivoted-QR row skeleton with a least-squares interpolant."""

    values = np.asarray(matrix)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("matrix must be a finite two-dimensional array")
    rank = int(rank)
    maximum = min(values.shape)
    if rank <= 0 or rank > maximum:
        raise ValueError(f"rank must be in [1, {maximum}]")

    _, _, pivots = qr(values.T, mode="economic", pivoting=True)
    indices = np.asarray(pivots[:rank], dtype=np.int64)
    selected = values[indices]
    interpolation = np.linalg.lstsq(
        selected.T,
        values.T,
        rcond=None,
    )[0].T
    approximation = interpolation @ selected
    denominator = float(np.linalg.norm(values))
    error = float(np.linalg.norm(values - approximation))
    if denominator > 0.0:
        error /= denominator
    indices.setflags(write=False)
    interpolation.setflags(write=False)
    return RowSkeleton(
        indices=indices,
        interpolation=interpolation,
        relative_frobenius_error=error,
    )
