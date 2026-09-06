"""Sampled CPSWF subspaces for existing discrete ACFO radial source blocks.

Input coefficients are source strengths with Euclidean inner product. Thus
the regular L2(dx) CPSWFs must be sampled as phi(x)/sqrt(x), not as phi(x).
The resulting basis is orthonormalized in the unchanged discrete metric.
No dense Hankel matrix is needed to generate this geometry-only basis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg, special

from .finite_hankel import commuting_tridiagonal, x_squared_tridiagonal


def sampled_regular_jacobi(order: int, dimension: int, x: np.ndarray) -> np.ndarray:
    """Evaluate phi_n(x)/sqrt(x) by normalized x^2 three-term recurrence."""
    if int(order) != order or order < 0:
        raise ValueError("order must be a nonnegative integer")
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)) or np.any((x <= 0) | (x > 1)):
        raise ValueError("x must be a finite vector in (0,1]")
    diag, off = x_squared_tridiagonal(order, dimension)
    output = np.empty((x.size, dimension))
    output[:, 0] = np.sqrt(2*(order+1))*x**order
    for n in range(dimension-1):
        output[:, n+1] = (x*x-diag[n])*output[:, n]
        if n:
            output[:, n+1] -= off[n-1]*output[:, n-1]
        output[:, n+1] /= off[n]
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("Jacobi recurrence overflow; basis unresolved")
    return output


def sampled_cpswf_basis(radius: np.ndarray, outer_radius: float, q_max: float,
                        order: int, dimension: int, *, dependence_tolerance: float = 1e-12):
    """Return an ordered Euclidean-orthonormal discrete source subspace.

    QR is unpivoted and stops at the first numerically dependent sampled
    column. Arbitrary QR completion vectors are never passed as CPSWF modes.
    """
    if not np.isfinite(outer_radius) or outer_radius <= 0:
        raise ValueError("outer_radius must be finite and positive")
    if not 0 < dependence_tolerance < 1:
        raise ValueError("dependence_tolerance must be in (0,1)")
    radius = np.asarray(radius, dtype=float)
    d, e = commuting_tridiagonal(order, q_max*outer_radius, dimension)
    count = min(radius.size, dimension)
    chi, u = linalg.eigh_tridiagonal(d, e, select="i", select_range=(0, count-1))
    samples = sampled_regular_jacobi(order, dimension, radius/outer_radius)@u
    norms = linalg.norm(samples, axis=0)
    if np.any(norms == 0) or not np.all(np.isfinite(norms)):
        raise FloatingPointError("zero/nonfinite sampled eigenfunction")
    orthogonal, r = linalg.qr(samples/norms, mode="economic")
    bad = np.flatnonzero(np.abs(np.diag(r)) <= dependence_tolerance)
    resolved = int(bad[0]) if bad.size else count
    return orthogonal[:, :resolved].copy(), {"chi": chi, "qr_diagonal": np.diag(r),
                                            "resolved_prefix": resolved, "dimension": dimension}


def radial_block(q_perp: np.ndarray, radius: np.ndarray, order: int) -> np.ndarray:
    q, r = np.asarray(q_perp, dtype=float), np.asarray(radius, dtype=float)
    if int(order) != order or order < 0 or q.ndim != 1 or r.ndim != 1:
        raise ValueError("order/vectors invalid")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(r)) or np.any(q < 0) or np.any(r <= 0):
        raise ValueError("q must be nonnegative and radius positive, all finite")
    return special.jv(order, q[:, None]*r[None, :])


def weighted_families(matrix: np.ndarray, weights: list[np.ndarray]):
    a = np.asarray(matrix, dtype=float)
    blocks, scales = [], []
    for weight in weights:
        w = np.asarray(weight, dtype=float)
        if w.shape != (a.shape[0],) or not np.all(np.isfinite(w)) or np.any(w < 0) or not np.any(w > 0):
            raise ValueError("weights must match rows, finite, nonnegative and nonzero")
        block = np.sqrt(w)[:, None]*a
        scale = float(linalg.svdvals(block)[0])
        if not scale > 0:
            raise ValueError("zero operator cannot define relative rank")
        blocks.append(block/scale)
        scales.append(scale)
    return blocks, np.asarray(scales)


def prefix_error(blocks: list[np.ndarray], basis: np.ndarray, rank: int) -> list[float]:
    p = basis[:, :rank]
    return [float(linalg.svdvals(a-(a@p)@p.T)[0]) for a in blocks]


def smallest_shared_prefix(blocks: list[np.ndarray], basis: np.ndarray, tolerance: float):
    """Binary search nested subspaces with spectral-norm residual checks."""
    if not 0 < tolerance < 1:
        raise ValueError("tolerance must be in (0,1)")
    checked = {}
    def err(k):
        if k not in checked:
            checked[k] = prefix_error(blocks, basis, k)
        return max(checked[k])
    low, high = 0, basis.shape[1]
    if err(high) > tolerance:
        return None, checked
    while low < high:
        middle = (low+high)//2
        if err(middle) <= tolerance:
            high = middle
        else:
            low = middle+1
    err(low)
    if low:
        err(low-1)
    return low, checked


@dataclass(frozen=True)
class PreparedRadialProjection:
    """Real radial kernel / real shared basis, complex128 RHS application.

    Geometry matrices are stored complex128 consistently with dense timing.
    They are exactly real, so transposes are Hermitian transposes without
    temporary conjugation copies. Normal cores are built once per weight.
    """
    basis: np.ndarray
    reduced: np.ndarray
    normal_cores: tuple[np.ndarray, ...]

    @classmethod
    def build(cls, matrix: np.ndarray, basis: np.ndarray, weights: list[np.ndarray]):
        if np.iscomplexobj(matrix) or np.iscomplexobj(basis):
            raise ValueError("this radial primitive requires real geometry matrices")
        a, p = np.asarray(matrix, dtype=float), np.asarray(basis, dtype=float)
        if a.ndim != 2 or p.ndim != 2 or a.shape[1] != p.shape[0] or not np.all(np.isfinite(a)) or not np.all(np.isfinite(p)):
            raise ValueError("incompatible/nonfinite geometry matrices")
        c = a@p
        cores = []
        for w in weights:
            w = np.asarray(w, dtype=float)
            if w.shape != (a.shape[0],) or not np.all(np.isfinite(w)) or np.any(w < 0):
                raise ValueError("invalid normal weights")
            cores.append(np.asarray(c.T@(w[:, None]*c), dtype=complex))
        arrays = [np.asarray(p, dtype=complex), np.asarray(c, dtype=complex), *cores]
        for arr in arrays:
            arr.setflags(write=False)
        return cls(arrays[0], arrays[1], tuple(arrays[2:]))

    def forward(self, values):
        return self.reduced@(self.basis.T@np.asarray(values, dtype=complex))

    def adjoint(self, values):
        return self.basis@(self.reduced.T@np.asarray(values, dtype=complex))

    def normal(self, values, profile=0):
        return self.basis@(self.normal_cores[profile]@(self.basis.T@np.asarray(values, dtype=complex)))

    @property
    def forward_bytes(self):
        return self.basis.nbytes+self.reduced.nbytes

    @property
    def all_operator_bytes(self):
        return self.forward_bytes+sum(core.nbytes for core in self.normal_cores)
