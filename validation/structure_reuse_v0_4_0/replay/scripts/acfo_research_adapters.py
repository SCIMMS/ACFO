"""Validation-only adapters. No changes to the public ACFO API.

All Hankel matrices use c sqrt(x y) J_m(c x y) on L2(dx).
Zernike functions instead denote physical radial pupils on L2(r dr).
"""
from __future__ import annotations

import time
import numpy as np
from scipy import linalg, special
from waxs_cake.portable_cpswf import kernel, PreparedRadialAction, robust_svd


class BlockedHankel:
    """Matrix-free in storage, not in arithmetic; visits every kernel entry."""
    def __init__(self, binding, order, c, block=64):
        self.b, self.order, self.c, self.block = binding, order, c, block
        self.shape = (len(binding.y), len(binding.x))
        self.calls = 0

    def apply(self, v, adjoint=False):
        v = np.asarray(v)
        vector = v.ndim == 1
        if vector:
            v = v[:, None]
        nout = self.shape[1] if adjoint else self.shape[0]
        out = np.zeros((nout, v.shape[1]), dtype=np.result_type(v, float))
        for start in range(0, self.shape[0], self.block):
            end = min(start+self.block, self.shape[0])
            tile = (self.b.left[start:end, None]
                    * kernel(self.order, self.c, self.b.y[start:end], self.b.x)
                    * self.b.right[None, :])
            if adjoint:
                out += tile.T @ v[start:end]
            else:
                out[start:end] = tile @ v
        self.calls += 1
        return out[:, 0] if vector else out


def probe_error(operator, action, seed=919):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(operator.shape[1], 12))
    truth = operator.apply(x)
    return float(linalg.norm(truth-action.forward(x))/linalg.norm(truth))


def randomized_prepare(operator, tolerance, seed):
    """Range(A*) with QR-normalized power iteration, then a small SVD.

    No dense target kernel is assembled, even on a retry. Probe failures
    enlarge the range; final spectral certification belongs to the audit.
    """
    rng = np.random.default_rng(seed)
    width = min(min(operator.shape), int(np.ceil(operator.c/np.pi))+32)
    phases = dict(sampling=0.0, qr=0.0, reduced_svd=0.0, required_check=0.0)
    while True:
        t = time.perf_counter()
        z = operator.apply(rng.normal(size=(operator.shape[0], width)), True)
        phases['sampling'] += time.perf_counter()-t
        t = time.perf_counter(); q = linalg.qr(z, mode='economic')[0]
        phases['qr'] += time.perf_counter()-t
        # One subspace iteration; QR between A and A* avoids cubing small s.
        t = time.perf_counter(); y = operator.apply(q)
        phases['sampling'] += time.perf_counter()-t
        t = time.perf_counter(); u = linalg.qr(y, mode='economic')[0]
        phases['qr'] += time.perf_counter()-t
        t = time.perf_counter(); z = operator.apply(u, True)
        phases['sampling'] += time.perf_counter()-t
        t = time.perf_counter(); q = linalg.qr(z, mode='economic')[0]
        phases['qr'] += time.perf_counter()-t
        t = time.perf_counter(); y = operator.apply(q)
        phases['sampling'] += time.perf_counter()-t
        t = time.perf_counter(); u, s, vh = robust_svd(y)
        k = max(1, int(np.count_nonzero(s > tolerance*0.1*s[0])))
        action = PreparedRadialAction.make(q@vh[:k].T, u[:, :k]*s[:k])
        phases['reduced_svd'] += time.perf_counter()-t
        t = time.perf_counter(); err = probe_error(operator, action)
        phases['required_check'] += time.perf_counter()-t
        if err <= tolerance or width == min(operator.shape):
            return action, dict(phases, probe_error=err, width=width)
        width = min(min(operator.shape), width*2)


def zernike_samples(m, alpha, count, r):
    if m < 0 or alpha <= -1 or count < 1:
        raise ValueError('m>=0, alpha>-1, count>=1 required')
    r = np.asarray(r)
    return np.column_stack([r**m*(1-r*r)**alpha
        * special.eval_jacobi(n, alpha, m, 2*r*r-1) for n in range(count)])


def zernike_images(m, alpha, count, v):
    """Integral_0^1 R(r) J_m(v r) r dr, including the v=0 limit.

    Janssen arXiv:1110.2369, generalized Zernike Hankel image.
    """
    v = np.asarray(v, dtype=float)
    if np.any(v < 0):
        raise ValueError('nonnegative frequencies required')
    out = np.zeros((v.size, count))
    positive = v > 0
    for n in range(count):
        pref = (-1.)**n*np.exp(alpha*np.log(2)+special.gammaln(n+alpha+1)
                              - special.gammaln(n+1))
        out[positive, n] = pref*special.jv(m+2*n+alpha+1, v[positive])/v[positive]**(alpha+1)
        if m == 0 and n == 0:
            out[~positive, n] = 1/(2*(alpha+1))
    return out


def ball_coefficients(m, alpha, c, dimension):
    """d=2 radial weighted-ball PSWF Galerkin matrix.

    Jacobi modes r^m P_n^(alpha,m)(2r²-1) are normalized in
    r(1-r²)^alpha dr. D0 has eigenvalues 4n(n+m+alpha+1), up to
    an irrelevant constant for a fixed m; add c² r². Physical pupil
    for the weighted transform is (1-r²)^alpha times its eigenfunction.
    """
    n = np.arange(dimension)
    norm2 = np.exp(special.gammaln(n+alpha+1)+special.gammaln(n+m+1)
        - special.gammaln(n+1)-special.gammaln(n+alpha+m+1)) / (2*(2*n+alpha+m+1))
    z, w = special.roots_jacobi(dimension+2, alpha, m)
    p = np.column_stack([special.eval_jacobi(i, alpha, m, z) for i in n])
    # Mapping r²=(1+z)/2: normalized Jacobi integration.
    w = w/(2**(alpha+m+2))
    p = p/np.sqrt(norm2)
    x2 = p.T@((w*(1+z)/2)[:, None]*p)
    d = 4*n*(n+m+alpha+1)+c*c*np.diag(x2)
    e = c*c*np.diag(x2, 1)
    _, u = linalg.eigh_tridiagonal(d, e)
    return u/np.sqrt(norm2)[:, None]


def sonine_image(nu, delta, z, quadrature):
    """Sonine integral evaluated in float64, plus cancellation indicator.

    J_(nu+d)(z) = z^d/(2^d Gamma(d)) integral_0^1
    u^(nu/2) (1-u)^(d-1) J_nu(z sqrt(u)) du.
    Gauss-Jacobi weight uses u^(nu/2), not the normalized-j weight.
    """
    if nu < 0 or delta <= 0:
        raise ValueError('nu>=0, delta>0 required')
    z = np.asarray(z, float)
    a, w = special.roots_jacobi(quadrature, delta-1, nu/2)
    u = (1+a)/2
    # Normalize weights then supply the integral beta factor in log scale.
    w = w/w.sum()
    values = special.jv(nu, z[:, None]*np.sqrt(u)[None, :])
    integral = values@w
    abs_integral = abs(values)@w
    pref = np.zeros_like(z)
    nz = z > 0
    pref[nz] = np.exp(delta*np.log(z[nz]/2)+special.gammaln(nu/2+1)
                      -special.gammaln(nu/2+delta+1))
    return pref*integral, pref*abs_integral
