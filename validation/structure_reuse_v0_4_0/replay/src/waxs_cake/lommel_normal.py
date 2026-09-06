"""Lommel-reduced radial normal operators for disk and annulus geometry."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import special


def _disk_product_integral(mode: int, left: np.ndarray, right: np.ndarray, radius: float) -> np.ndarray:
    """Return ``integral_0^R r J_m(a r) J_m(b r) dr`` pairwise."""

    mode = int(mode)
    if mode < 0:
        raise ValueError("mode must be non-negative")
    a = np.asarray(left, dtype=np.float64)[:, None]
    b = np.asarray(right, dtype=np.float64)[None, :]
    r = float(radius)
    if not np.isfinite(r) or r < 0:
        raise ValueError("radius must be finite and non-negative")
    if r == 0.0:
        return np.zeros((a.shape[0], b.shape[1]), dtype=np.float64)
    ar = a * r
    br = b * r
    denominator = a * a - b * b
    numerator = r * (
        a * special.jv(mode + 1, ar) * special.jv(mode, br)
        - b * special.jv(mode, ar) * special.jv(mode + 1, br)
    )
    scale = np.maximum(np.maximum(a * a, b * b), 1.0)
    # The off-diagonal quotient loses all useful digits when a and b nearly
    # coincide.  The midpoint diagonal limit is second-order accurate in
    # |a-b| and is preferable once the squared-wavenumber separation reaches
    # the square-root-epsilon regime.
    diagonal = np.abs(denominator) <= 8.0 * np.sqrt(np.finfo(np.float64).eps) * scale
    output = np.empty_like(denominator)
    np.divide(numerator, denominator, out=output, where=~diagonal)
    if np.any(diagonal):
        k = 0.5 * (a + b)
        kr = k * r
        value = 0.5 * r * r * (
            special.jv(mode, kr) ** 2
            - special.jv(mode - 1, kr) * special.jv(mode + 1, kr)
        )
        output[diagonal] = value[diagonal]
    return output


def lommel_bessel_product_integral(
    mode: int,
    left_wavenumbers: Any,
    right_wavenumbers: Any | None = None,
    *,
    outer_radius: float,
    inner_radius: float = 0.0,
) -> np.ndarray:
    """Pairwise weighted Bessel products on ``inner_radius <= r <= outer_radius``."""

    left = np.asarray(left_wavenumbers, dtype=np.float64)
    right = left if right_wavenumbers is None else np.asarray(right_wavenumbers, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1 or left.size == 0 or right.size == 0:
        raise ValueError("wavenumbers must be non-empty vectors")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)) or np.any(left < 0) or np.any(right < 0):
        raise ValueError("wavenumbers must be finite and non-negative")
    outer = float(outer_radius)
    inner = float(inner_radius)
    if not (np.isfinite(inner) and np.isfinite(outer) and 0.0 <= inner < outer):
        raise ValueError("radii must satisfy 0 <= inner_radius < outer_radius")
    return _disk_product_integral(mode, left, right, outer) - _disk_product_integral(
        mode, left, right, inner
    )


class LommelRadialNormalOperator:
    """Exact normal matrix for a finite Bessel synthesis on a disk/annulus.

    The synthesis is ``f(r)=sum_j spectral_weights[j] * c[j] * J_m(k_j r)``
    with radial inner product ``integral r conj(f) g dr``.
    """

    def __init__(
        self,
        mode: int,
        wavenumbers: Any,
        *,
        outer_radius: float,
        inner_radius: float = 0.0,
        spectral_weights: Any | None = None,
        damping: float = 0.0,
    ) -> None:
        k = np.asarray(wavenumbers, dtype=np.float64)
        if k.ndim != 1 or k.size == 0:
            raise ValueError("wavenumbers must be a non-empty vector")
        if spectral_weights is None:
            weights = np.ones(k.shape, dtype=np.complex128)
        else:
            weights = np.asarray(spectral_weights, dtype=np.complex128)
            if weights.shape != k.shape or not np.all(np.isfinite(weights)):
                raise ValueError("spectral_weights must be finite and match wavenumbers")
        damping = float(damping)
        if not damping >= 0:
            raise ValueError("damping must be non-negative")
        base = lommel_bessel_product_integral(
            mode,
            k,
            outer_radius=outer_radius,
            inner_radius=inner_radius,
        )
        gram = np.conj(weights)[:, None] * base * weights[None, :]
        gram = 0.5 * (gram + gram.conj().T)
        self.mode = int(mode)
        self.wavenumbers = np.array(k, copy=True)
        self.outer_radius = float(outer_radius)
        self.inner_radius = float(inner_radius)
        self.spectral_weights = np.array(weights, copy=True)
        self.damping = damping
        self.gram = gram

    def matvec(self, values: Any, *, include_damping: bool = True) -> np.ndarray:
        vector = np.asarray(values, dtype=np.complex128)
        if vector.shape != self.wavenumbers.shape or not np.all(np.isfinite(vector)):
            raise ValueError("values must be finite and match wavenumbers")
        result = self.gram @ vector
        if include_damping and self.damping:
            result = result + self.damping * vector
        return result

    def diagonal_preconditioner(self) -> np.ndarray:
        diagonal = np.real(np.diag(self.gram)) + self.damping
        floor = np.finfo(np.float64).eps * max(float(np.max(diagonal)), 1.0)
        return 1.0 / np.maximum(diagonal, floor)

    def quadrature_reference(self, *, order: int = 512) -> np.ndarray:
        """Independent Gauss--Legendre Gram reference for validation."""

        order = int(order)
        if order < 16:
            raise ValueError("quadrature order must be at least 16")
        nodes, weights = np.polynomial.legendre.leggauss(order)
        span = self.outer_radius - self.inner_radius
        radius = self.inner_radius + 0.5 * span * (nodes + 1.0)
        radial_weights = 0.5 * span * weights * radius
        basis = special.jv(self.mode, self.wavenumbers[:, None] * radius[None, :])
        basis = self.spectral_weights[:, None] * basis
        return (np.conj(basis) * radial_weights[None, :]) @ basis.T
