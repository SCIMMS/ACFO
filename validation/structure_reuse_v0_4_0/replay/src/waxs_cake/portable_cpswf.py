"""Portable continuous radial bases with explicit discrete metric bindings.

K(x,y)=c*sqrt(x*y)*J_m(c*x*y) acts on L2(0,1; dx). Commuting-D
eigenvalues chi and K eigenvalues mu are different. Classical Jacobi and
Nyström constructions are used here; no new spectral solver is claimed.
The A-stage builder uses dense residual selection. B-stage uses an audited
finite spectral expansion, with no target-grid dense kernel. Spectral tail
estimates alone are not a rigorous total continuous-error certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import linalg, special

from .finite_hankel import commuting_tridiagonal, gauss_unit_interval, hankel_basis_images
from .radial_cpswf import sampled_regular_jacobi, smallest_shared_prefix


SVD_FALLBACK_EVENTS = []


def robust_svd(matrix):
    """Same full reduced SVD; GESVD retry only after GESDD nonconvergence.

    The failed attempt remains part of elapsed preparation time. Both methods
    are LAPACK algorithms; rank/tolerance/inputs are unchanged on retry.
    """
    try:
        return linalg.svd(matrix, full_matrices=False, lapack_driver="gesdd")
    except linalg.LinAlgError:
        SVD_FALLBACK_EVENTS.append({"shape": list(matrix.shape), "from": "gesdd", "to": "gesvd"})
        return linalg.svd(matrix, full_matrices=False, lapack_driver="gesvd")


def readonly(a, dtype=float):
    b = np.array(a, dtype=dtype, copy=True, order="C")
    if not np.all(np.isfinite(b)):
        raise ValueError("non-finite array")
    b.setflags(write=False)
    return b


def positive_vector(a, name):
    a = np.asarray(a, dtype=float)
    if a.ndim != 1 or not a.size or not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError(f"{name} must be finite, positive, nonempty vector")
    return a


def kernel(order, c, y, x):
    xy = np.asarray(y)[:, None]*np.asarray(x)[None, :]
    return c*np.sqrt(xy)*special.jv(order, c*xy)


@dataclass(frozen=True)
class RadialBinding:
    """Discrete B=diag(left) K(y,x) diag(right), Euclidean coordinates.

    hankel_l2 uses sqrt(quadrature) on BOTH sides. legacy_density keeps the
    existing ODT cell-area input and Euclidean output; its norm is grid-specific.
    """
    x: np.ndarray
    y: np.ndarray
    right: np.ndarray
    left: np.ndarray
    metric: str

    @classmethod
    def make(cls, x, y, wx, wy, *, c, metric="hankel_l2", radius=1.0):
        x, y = positive_vector(x, "x"), positive_vector(y, "y")
        wx, wy = positive_vector(wx, "wx"), positive_vector(wy, "wy")
        if wx.shape != x.shape or wy.shape != y.shape or np.max(x) > 1 or np.max(y) > 1:
            raise ValueError("nodes/weights mismatch or nodes outside (0,1]")
        if np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0) or c <= 0 or radius <= 0:
            raise ValueError("ordered nodes and positive scales required")
        if metric == "hankel_l2":
            right, left = np.sqrt(wx), np.sqrt(wy)
        elif metric == "legacy_density":
            right = 2*np.pi*radius**2*wx*np.sqrt(x)
            left = 1/(c*np.sqrt(y))
        elif metric == "source_strength":
            right, left = 1/np.sqrt(x), 1/(c*np.sqrt(y))
        else:
            raise ValueError("unknown metric")
        return cls(*(readonly(a) for a in (x, y, right, left)), metric)

    def matrix(self, order, c):
        return self.left[:, None]*kernel(order, c, self.y, self.x)*self.right[None, :]


@dataclass(frozen=True)
class ContinuousRadialBasis:
    order: int
    c: float
    coefficients: np.ndarray
    chi: np.ndarray

    @classmethod
    def prepare(cls, order, c, dimension, columns):
        if isinstance(order, bool) or int(order) != order or order < 0:
            raise ValueError("nonnegative integer order required")
        if isinstance(columns, bool) or int(columns) != columns or not 1 <= columns <= dimension:
            raise ValueError("invalid columns")
        d, e = commuting_tridiagonal(order, c, dimension)
        chi, u = linalg.eigh_tridiagonal(d, e, select="i", select_range=(0, columns-1))
        # Canonical signs make save/load and finite-section checks reproducible.
        sign = np.sign(u[np.argmax(abs(u), axis=0), np.arange(columns)])
        return cls(int(order), float(c), readonly(u*sign), readonly(chi))

    @property
    def dimension(self):
        return self.coefficients.shape[0]

    @property
    def columns(self):
        return self.coefficients.shape[1]

    @property
    def retained_bytes(self):
        return self.coefficients.nbytes+self.chi.nbytes

    def sample(self, nodes, *, columns=None, chunk=32):
        x = positive_vector(nodes, "nodes")
        if np.max(x) > 1 or not isinstance(chunk, int) or chunk < 1:
            raise ValueError("invalid nodes/chunk")
        k = self.columns if columns is None else int(columns)
        if not 1 <= k <= self.columns:
            raise ValueError("requested columns unavailable")
        out = np.empty((x.size, k))
        for start in range(0, x.size, chunk):
            z = x[start:start+chunk]
            phi = np.sqrt(z[:, None])*sampled_regular_jacobi(self.order, self.dimension, z)
            out[start:start+chunk] = phi@self.coefficients[:, :k]
        return out

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        meta = json.dumps({"schema": "continuous-radial-basis-v1", "order": self.order, "c": self.c})
        np.savez(path, metadata=meta, coefficients=self.coefficients, chi=self.chi)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta["schema"] != "continuous-radial-basis-v1":
                raise ValueError("unknown cache schema")
            u, chi = data["coefficients"], data["chi"]
            if u.ndim != 2 or chi.shape != (u.shape[1],):
                raise ValueError("cache shape mismatch")
            if linalg.norm(u.T@u-np.eye(u.shape[1]), 2) > 1e-10:
                raise ValueError("cache coefficient orthogonality failed")
            return cls(meta["order"], meta["c"], readonly(u), readonly(chi))

    def fingerprint(self):
        h = hashlib.sha256(json.dumps([self.order, self.c]).encode())
        h.update(self.coefficients.tobytes())
        h.update(self.chi.tobytes())
        return h.hexdigest()


def ordered_qr(samples, *, minimum=1, tolerance=1e-12):
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.shape[0] < minimum:
        raise ValueError("grid under-resolved for required rank")
    norms = linalg.norm(samples, axis=0)
    if np.any(norms == 0) or not np.all(np.isfinite(norms)):
        raise ArithmeticError("zero/nonfinite sampled function")
    q, r = linalg.qr(samples/norms, mode="economic")
    bad = np.flatnonzero(np.abs(np.diag(r)) <= tolerance)
    resolved = int(bad[0]) if bad.size else min(samples.shape)
    if resolved < minimum:
        raise ArithmeticError("grid under-resolved for required rank")
    return q[:, :resolved], resolved


@dataclass(frozen=True)
class PreparedRadialAction:
    """A=C P*, allowing complex factors and a shared input coefficient space."""
    basis: np.ndarray
    reduced: np.ndarray

    @classmethod
    def make(cls, p, c):
        if np.ndim(p) != 2 or np.ndim(c) != 2 or p.shape[1] != c.shape[1]:
            raise ValueError("factor shapes mismatch")
        return cls(readonly(p, complex), readonly(c, complex))

    @property
    def rank(self):
        return self.basis.shape[1]

    @property
    def retained_bytes(self):
        return self.basis.nbytes+self.reduced.nbytes

    def forward(self, x):
        return self.reduced@(self.basis.conj().T@x)

    def adjoint(self, y):
        return self.basis@(self.reduced.conj().T@y)

    def with_multiplier(self, multiplier):
        w = np.asarray(multiplier)
        if w.shape != (self.reduced.shape[0],) or not np.all(np.isfinite(w)):
            raise ValueError("multiplier shape/value invalid")
        return PreparedRadialAction(self.basis, readonly(w[:, None]*self.reduced, complex))

    def normal_core(self, weight=None):
        w = np.ones(self.reduced.shape[0]) if weight is None else np.asarray(weight)
        if w.shape != (self.reduced.shape[0],) or np.iscomplexobj(w) or np.any(w < 0) or not np.all(np.isfinite(w)):
            raise ValueError("normal weight must be nonnegative real finite")
        return self.reduced.conj().T@(w[:, None]*self.reduced)

    def normal(self, x, core):
        return self.basis@(core@(self.basis.conj().T@x))


def prepare_projection(matrix, samples, tolerance):
    p, resolved = ordered_qr(samples)
    scale = linalg.svdvals(matrix)[0]
    k, checked = smallest_shared_prefix([matrix/scale], p, tolerance)
    if k is None or k == 0:
        raise ArithmeticError("available portable basis does not resolve tolerance")
    p = p[:, :k]
    return PreparedRadialAction.make(p, matrix@p), {"resolved": resolved, "selected_error": checked[k][0]}


def prepare_svd(matrix, tolerance):
    u, s, vh = robust_svd(matrix)
    k = int(np.count_nonzero(s > tolerance*s[0]))
    return PreparedRadialAction.make(vh[:k].conj().T, u[:, :k]*s[:k]), s


@dataclass(frozen=True)
class PortableNystromSVD:
    """Reference quadrature SVD extended by its kernel integral equation.

    Nyström evaluation still visits every reference node. Small singular values
    amplify numerical error; the retained floor is part of the prespecified plan.
    """
    order: int
    c: float
    nodes: np.ndarray
    extension_coefficients: np.ndarray
    singular_values: np.ndarray

    @classmethod
    def prepare(cls, order, c, reference_count, floor):
        x, w = gauss_unit_interval(reference_count)
        a = np.sqrt(w[:, None])*kernel(order, c, x, x)*np.sqrt(w[None, :])
        u, s, _ = robust_svd(a)
        keep = s > floor*s[0]
        coeff = np.sqrt(w[:, None])*u[:, keep]/s[keep]
        return cls(order, c, readonly(x), readonly(coeff), readonly(s[keep]))

    @property
    def retained_bytes(self):
        return self.nodes.nbytes+self.extension_coefficients.nbytes+self.singular_values.nbytes

    def sample(self, nodes, *, chunk=32):
        out = np.empty((len(nodes), self.singular_values.size))
        for j in range(0, len(nodes), chunk):
            out[j:j+chunk] = kernel(self.order, self.c, nodes[j:j+chunk], self.nodes)@self.extension_coefficients
        return out


@dataclass(frozen=True)
class HankelSpectrum:
    basis: ContinuousRadialBasis
    mu: np.ndarray
    eigen_residual_l2: np.ndarray
    quadrature: int

    @classmethod
    def prepare(cls, basis, quadrature, *, chunk=32):
        # No Nq*Nr kernel or dense Hankel Galerkin matrix is constructed.
        x, w = gauss_unit_interval(quadrature)
        norm, cross, image_norm = (np.zeros(basis.columns) for _ in range(3))
        for start in range(0, x.size, chunk):
            z, weights = x[start:start+chunk], w[start:start+chunk]
            psi = basis.sample(z, chunk=chunk)
            image = hankel_basis_images(basis.order, basis.c, basis.dimension, z)@basis.coefficients
            norm += np.sum(weights[:, None]*psi*psi, axis=0)
            cross += np.sum(weights[:, None]*psi*image, axis=0)
            image_norm += np.sum(weights[:, None]*image*image, axis=0)
        mu = cross/norm
        # Direct residual pass avoids catastrophic cancellation near unit mu.
        residual = np.zeros(basis.columns)
        for start in range(0, x.size, chunk):
            z, weights = x[start:start+chunk], w[start:start+chunk]
            psi = basis.sample(z, chunk=chunk)
            image = hankel_basis_images(basis.order, basis.c, basis.dimension, z)@basis.coefficients
            residual += np.sum(weights[:, None]*(image-psi*mu)**2, axis=0)
        return cls(basis, readonly(mu), readonly(np.sqrt(residual)), int(quadrature))

    def rank(self, relative_tail, guard=8):
        if not 0 < relative_tail < 1 or guard < 1:
            raise ValueError("invalid spectrum tolerance/guard")
        threshold = relative_tail*np.max(abs(self.mu))
        above = np.flatnonzero(abs(self.mu) > threshold)
        k = int(above[-1]+1) if above.size else 0
        if k == 0 or self.mu.size-k < guard:
            raise ArithmeticError("spectral search window unresolved; no silent rank cap")
        return k

    def bind(self, binding, relative_tail, *, chunk=32):
        k = self.rank(relative_tail)
        fx = binding.right[:, None]*self.basis.sample(binding.x, columns=k, chunk=chunk)
        if min(len(binding.x), len(binding.y)) < k:
            raise ValueError("grid under-resolved for spectral rank")
        # Q R = Fx; keep the metric adaptation R in the reduced action.
        p, r = linalg.qr(fx, mode="economic")
        if np.min(abs(np.diag(r))/linalg.norm(fx, axis=0)) <= 1e-12:
            raise ArithmeticError("sampled basis is rank deficient")
        fy = binding.left[:, None]*self.basis.sample(binding.y, columns=k, chunk=chunk)
        return PreparedRadialAction.make(p, (fy*self.mu[:k])@r.T)

    @property
    def retained_bytes(self):
        return self.basis.retained_bytes+self.mu.nbytes+self.eigen_residual_l2.nbytes
