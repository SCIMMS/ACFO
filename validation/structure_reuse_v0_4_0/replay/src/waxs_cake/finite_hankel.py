"""Regular finite-Hankel / CPSWF coefficient-space feasibility primitives.

On L2((0,1), dx), use K(x,y)=c sqrt(xy) J_nu(cxy). This is sqrt(c)
times the convention with kernel sqrt(cxy) J_nu(cxy); K*K has spectrum
in [0,1]. Orders nu >= 0 and c > 0 are supported. The regular endpoint
branch is x**(nu+1/2) (no logarithmic solution for nu=0); at x=1 the
regular/natural Sturm--Liouville endpoint is selected. No Dirichlet
condition at x=1 is imposed.

The Jacobi-basis tridiagonal construction is classical (Bouwkamp/Slepian).
This module tests reusable finite materialization, not a new CPSWF solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg, special


def _parameters(order: float, bandwidth: float, dimension: int) -> tuple[float, float, int]:
    nu, c = float(order), float(bandwidth)
    if not np.isfinite(nu) or nu < 0:
        raise ValueError("order must be finite and non-negative")
    if not np.isfinite(c) or c <= 0:
        raise ValueError("bandwidth must be finite and positive")
    if isinstance(dimension, bool) or int(dimension) != dimension or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    return nu, c, int(dimension)


def radial_jacobi_basis(order: float, dimension: int, x: np.ndarray) -> np.ndarray:
    """Orthonormal phi_n=sqrt(2(2n+nu+1))*x^(nu+1/2)*P_n^(0,nu)(2x^2-1)."""
    nu, _, size = _parameters(order, 1.0, dimension)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not np.all(np.isfinite(x)) or np.any((x < 0) | (x > 1)):
        raise ValueError("x must be a finite vector in [0,1]")
    n = np.arange(size)
    return (np.sqrt(2 * (2 * n + nu + 1))[None, :]
            * x[:, None] ** (nu + 0.5)
            * special.eval_jacobi(n[None, :], 0.0, nu, 2 * x[:, None] ** 2 - 1))


def x_squared_tridiagonal(order: float, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """Galerkin multiplication by x^2 in the regular radial Jacobi basis."""
    nu, _, size = _parameters(order, 1.0, dimension)
    n = np.arange(size, dtype=np.float64)
    b = np.zeros(size)
    b[0] = nu / (nu + 2)
    b[1:] = nu**2 / ((2*n[1:] + nu) * (2*n[1:] + nu + 2))
    j = n[:-1]
    a = ((j + 1) * (j + nu + 1)
         / ((2*j + nu + 2) * np.sqrt((2*j + nu + 1) * (2*j + nu + 3))))
    return (1 + b) / 2, a


def commuting_tridiagonal(order: float, bandwidth: float, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """D=-d_x((1-x^2)d_x)+(nu^2-1/4)/x^2+c^2*x^2, regular domain."""
    nu, c, size = _parameters(order, bandwidth, dimension)
    n = np.arange(size)
    d, e = x_squared_tridiagonal(nu, size)
    return (2*n + nu + 0.5) * (2*n + nu + 1.5) + c*c*d, c*c*e


def tridiagonal_apply(diagonal: np.ndarray, off_diagonal: np.ndarray, values: np.ndarray) -> np.ndarray:
    """O(N*nrhs) action with slices, including complex right-hand sides."""
    d, e, a = np.asarray(diagonal), np.asarray(off_diagonal), np.asarray(values)
    if d.ndim != 1 or e.shape != (d.size-1,) or a.ndim not in (1, 2) or a.shape[0] != d.size:
        raise ValueError("incompatible tridiagonal/action shapes")
    scale = (slice(None),) + (None,) * (a.ndim-1)
    out = d[scale] * a
    out[1:] += e[scale] * a[:-1]
    out[:-1] += e[scale] * a[1:]
    return out


def gauss_unit_interval(count: int) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(count, bool) or int(count) != count or count < 1:
        raise ValueError("quadrature count must be positive integer")
    x, w = special.roots_legendre(int(count))
    return (x+1)/2, w/2


def hankel_basis_images(order: float, bandwidth: float, dimension: int, x: np.ndarray) -> np.ndarray:
    """Analytic K phi_n(x)=(-1)^n sqrt(2(2n+nu+1))*J_(2n+nu+1)(cx)/sqrt(x).

    This identity eliminates one numerical integration. Independent direct
    kernel quadrature must still validate the resulting finite materialization.
    """
    nu, c, size = _parameters(order, bandwidth, dimension)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not np.all(np.isfinite(x)) or np.any((x < 0) | (x > 1)):
        raise ValueError("x must be a finite vector in [0,1]")
    n = np.arange(size)
    out = np.zeros((x.size, size))
    mask = x > 0
    out[mask] = ((-1.0)**n * np.sqrt(2*(2*n+nu+1)))[None, :] * special.jv(
        (2*n+nu+1)[None, :], c*x[mask, None]) / np.sqrt(x[mask, None])
    return out


def hankel_galerkin(order: float, bandwidth: float, dimension: int, quadrature: int, *, direct: bool = False) -> np.ndarray:
    """Projected K; direct=True uses independent two-dimensional kernel quadrature.

    Return the unsymmetrized quadrature result so discretization asymmetry is
    visible to callers. Spectral preparation symmetrizes explicitly.
    """
    nu, c, size = _parameters(order, bandwidth, dimension)
    x, w = gauss_unit_interval(quadrature)
    phi = radial_jacobi_basis(nu, size, x)
    if direct:
        xy = x[:, None] * x[None, :]
        kernel = c * np.sqrt(xy) * special.jv(nu, c*xy)
        return (phi.T * w) @ kernel @ (w[:, None] * phi)
    return (phi.T * w) @ hankel_basis_images(nu, c, size, x)


@dataclass(frozen=True)
class PreparedCPSWF:
    """Coefficient-native low-rank K and K*K, with basis conversion charged.

    Only selected complex128 basis vectors and real eigenvalues are retained.
    Finite-section/quadrature/eigenpair errors need an external audit. Dropped
    eigenvalues alone are not a certificate of total continuous-operator error.
    """
    vectors: np.ndarray
    eigenvalues: np.ndarray

    @classmethod
    def from_pair(cls, diagonal: np.ndarray, off_diagonal: np.ndarray, hankel_matrix: np.ndarray,
                  *, relative_tolerance: float = 1e-10) -> tuple[PreparedCPSWF, dict[str, np.ndarray]]:
        if not np.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
            raise ValueError("relative_tolerance must lie in (0,1)")
        h = np.asarray(hankel_matrix, dtype=np.float64)
        if h.shape != (len(diagonal), len(diagonal)) or not np.all(np.isfinite(h)):
            raise ValueError("hankel_matrix must be finite and match diagonal")
        chi, u = linalg.eigh_tridiagonal(diagonal, off_diagonal)
        h = (h+h.T)/2
        projected = u.T @ h @ u
        lam = np.diag(projected).copy()
        keep = np.abs(lam) > relative_tolerance * np.max(np.abs(lam))
        plan = cls(np.ascontiguousarray(u[:, keep], dtype=np.complex128), lam[keep])
        audit = {"chi": chi, "vectors": u, "hankel_eigenvalues": lam,
                 "hankel_in_cpswf_basis": projected, "retained": keep}
        return plan, audit

    @property
    def retained_bytes(self) -> int:
        return self.vectors.nbytes + self.eigenvalues.nbytes

    def apply(self, values: np.ndarray, *, normal: bool = False) -> np.ndarray:
        a = np.asarray(values, dtype=np.complex128)
        if a.ndim not in (1, 2) or a.shape[0] != self.vectors.shape[0]:
            raise ValueError("values must have matching leading dimension")
        lam = self.eigenvalues**2 if normal else self.eigenvalues
        v = self.vectors
        # Stored vectors are exactly real, though complex128 for fair GEMM.
        # Transpose is therefore the Hermitian transpose without conjugate copy.
        coefficients = v.T @ a
        return v @ (lam.reshape((-1,) + (1,)*(a.ndim-1)) * coefficients)


def chebyshev_tridiagonal_apply(diagonal: np.ndarray, off_diagonal: np.ndarray,
                               coefficients: np.ndarray, bounds: tuple[float, float],
                               values: np.ndarray) -> np.ndarray:
    """Clenshaw p(D) action without forming eigenvectors or dense powers."""
    low, high = bounds
    if not np.isfinite(low+high) or high <= low:
        raise ValueError("bounds must be finite and increasing")
    a = np.asarray(values)
    coeff = np.asarray(coefficients, dtype=np.float64)
    if coeff.ndim != 1 or coeff.size == 0:
        raise ValueError("coefficients must be nonempty vector")
    d = (2*np.asarray(diagonal) - (high+low)) / (high-low)
    e = 2*np.asarray(off_diagonal) / (high-low)
    b1, b2 = np.zeros_like(a, dtype=np.result_type(a, float)), np.zeros_like(a, dtype=np.result_type(a, float))
    for term in coeff[:0:-1]:
        b0 = 2*tridiagonal_apply(d, e, b1) - b2 + term*a
        b2, b1 = b1, b0
    return tridiagonal_apply(d, e, b1) - b2 + coeff[0]*a
