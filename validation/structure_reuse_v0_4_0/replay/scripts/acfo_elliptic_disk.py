"""Validation-only SO(2) disk calculus with Dirichlet Fourier--Jacobi states.

The polynomial ladder is known spectral calculus, not a new ACFO identity.
All angular functions use exp(i*m*phi)/sqrt(2*pi); radial integrals use r dr.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator, cg, splu
from scipy.special import eval_jacobi, roots_legendre


def radial_rule(order: int):
    x, w = roots_legendre(order)
    r = (x + 1) / 2
    return r, w * r / 2


def state_values(m: int, n: int, r: np.ndarray):
    """Return value, radial derivative and radial Laplacian independently.

    Derivatives are evaluated via scalar Jacobi derivative identities, not the
    coefficient gradient map used by assembly.
    """
    k = abs(m)
    t = r * r
    p = eval_jacobi(n, 1, k, 2*t-1)
    pt = ((n+k+2) * eval_jacobi(n-1, 2, k+1, 2*t-1)
          if n else np.zeros_like(r))
    ptt = ((n+k+2)*(n+k+3)*eval_jacobi(n-2, 3, k+2, 2*t-1)
           if n > 1 else np.zeros_like(r))
    norm = np.sqrt(2*(n+1)**2/(2*n+k+2))
    f = (1-t)*p/norm
    ft = ((1-t)*pt-p)/norm
    ftt = ((1-t)*ptt-2*pt)/norm
    value = r**k * f
    derivative = r**(k-1)*(k*f+2*t*ft)
    laplacian = 4*r**k*((k+1)*ft+t*ftt)
    return value, derivative, laplacian


def zernike_values(m: int, ns: list[int], r: np.ndarray):
    k = abs(m)
    return np.column_stack([
        np.sqrt(2*(2*n+k+1))*r**k*eval_jacobi(n, 0, k, 2*r*r-1)
        for n in ns
    ])


@dataclass(frozen=True)
class DiskSpace:
    modes: tuple[int, ...]
    radial_count: int

    @cached_property
    def labels(self):
        return tuple((m, n) for m in self.modes for n in range(self.radial_count))

    @cached_property
    def indices(self):
        return {label: i for i, label in enumerate(self.labels)}

    @property
    def size(self):
        return len(self.modes)*self.radial_count


def gradient(space: DiskSpace, sign: int, workspace: str = "reachable"):
    """Exact D_sign/sqrt(2), with one -1/sqrt(2) entry per state.

    The radial degree increases by one when |m| decreases; this is essential
    even if the derivative is evaluated at the same physical points.
    """
    if sign not in (-1, 1):
        raise ValueError("sign must be +1 or -1")
    targets = [(m+sign, n+(abs(m+sign) < abs(m))) for m, n in space.labels]
    if workspace == "reachable":
        labels = tuple(sorted(set(targets)))
    elif workspace == "padded":
        labels = tuple((m, n) for m in range(min(space.modes)-1, max(space.modes)+2)
                       for n in range(space.radial_count+1))
    elif workspace == "underpadded":
        labels = space.labels
    else:
        raise ValueError(workspace)
    indices = {label: i for i, label in enumerate(labels)}
    columns = [j for j, label in enumerate(targets) if label in indices]
    rows = [indices[targets[j]] for j in columns]
    g = sparse.csr_matrix((np.full(len(rows), -1/np.sqrt(2)), (rows, columns)),
                          shape=(len(labels), space.size))
    return g, labels


def _groups(labels):
    groups = {}
    for i, (m, n) in enumerate(labels):
        groups.setdefault(m, []).append((i, n))
    return groups


def multiplication(labels, harmonics: dict[int, float], *, axisymmetric: bool,
                   quadrature: int):
    """Orthogonal projection of a real scalar multiplier in a gradient space.

    Axisymmetric coefficient: 1+0.2*r^2. Perturbation: sum w*r^s*cos(s*phi).
    Structural zeros are from angular selection; radial roundoff zeros are
    removed at 1e-13 absolute (orthonormal basis, bounded multipliers).
    """
    r, w = radial_rule(quadrature)
    groups = _groups(labels)
    z = {m: zernike_values(m, [n for _, n in rows], r) for m, rows in groups.items()}
    offsets = [0] if axisymmetric else sorted({d for s in harmonics for d in (-s, s)})
    rr, cc, vv = [], [], []
    for m, rows in groups.items():
        ii = np.array([i for i, _ in rows])
        for delta in offsets:
            target = m+delta
            if target not in groups:
                continue
            jj = np.array([i for i, _ in groups[target]])
            coef = (1+0.2*r*r if axisymmetric else
                    0.5*harmonics[abs(delta)]*r**abs(delta))
            block = z[m].T @ ((w*coef)[:, None]*z[target])
            # Polynomial degree gives exact zeros, independently of quadrature.
            # Multiplication by r^s*exp(+-i*s*phi) is multiplication by z^s or
            # conjugate(z)^s; each changes total polynomial degree by at most s.
            degree = 2 if axisymmetric else abs(delta)
            row_degree = np.array([abs(m)+2*n for _, n in rows])
            col_degree = np.array([abs(target)+2*n for _, n in groups[target]])
            block[np.abs(row_degree[:, None]-col_degree[None, :]) > degree] = 0
            a, b = np.nonzero(np.abs(block) > 1e-13)
            rr.extend(ii[a]); cc.extend(jj[b]); vv.extend(block[a, b])
    result = sparse.csr_matrix((vv, (rr, cc)), shape=(len(labels), len(labels)))
    return (result+result.T)*0.5


def mass_matrix(space: DiskSpace, quadrature: int):
    r, w = radial_rule(quadrature)
    blocks = []
    for m in space.modes:
        b = np.column_stack([state_values(m, n, r)[0] for n in range(space.radial_count)])
        block = b.T @ (w[:, None]*b)
        indices = np.arange(space.radial_count)
        block[np.abs(indices[:, None]-indices[None, :]) > 1] = 0
        block[np.abs(block) < 1e-14] = 0
        blocks.append(sparse.csr_matrix(block))
    return sparse.block_diag(blocks, format="csr")


@dataclass
class DiskOperator:
    space: DiskSpace
    harmonics: dict[int, float]
    base: sparse.csr_matrix
    perturbation: sparse.csr_matrix
    mass: sparse.csr_matrix
    construction_s: float
    gradient_rows: int
    gradient_nnz: int
    quadrature: int
    sigma: float = 0.3

    def matrix(self, epsilon: float):
        return (self.base+epsilon*self.perturbation).tocsc()


def assemble(space: DiskSpace, harmonics: dict[int, float], *, quadrature=100,
             workspace="reachable", sigma=0.3):
    start = perf_counter()
    mass = mass_matrix(space, quadrature)
    k0 = sparse.csr_matrix((space.size, space.size))
    v = k0.copy()
    rows = nnz = 0
    for sign in (-1, 1):
        g, labels = gradient(space, sign, workspace)
        rows += g.shape[0]; nnz += g.nnz
        a = multiplication(labels, harmonics, axisymmetric=True, quadrature=quadrature)
        b = multiplication(labels, harmonics, axisymmetric=False, quadrature=quadrature)
        k0 = k0+g.T@a@g
        v = v+g.T@b@g
    return DiskOperator(space, harmonics, (k0+sigma*mass).tocsr(), v.tocsr(), mass,
                        perf_counter()-start, rows, nnz, quadrature, sigma)


def manufactured_radial(m, r, beta=-6.0):
    k = abs(m)
    t = r*r
    ex = np.exp(beta*t)
    f = (1-t)*ex
    ft = (beta*(1-t)-1)*ex
    ftt = (beta*beta*(1-t)-2*beta)*ex
    return r**k*f, r**(k-1)*(k*f+2*t*ft), 4*r**k*((k+1)*ft+t*ftt)


def apply_physical_operator(fields, r, harmonics, epsilon, sigma=0.3):
    """Analytic strong-form Fourier fields, independent of coefficient G/M.

    fields[m] contains u, partial_r u, radial Laplacian u at the nodes.
    """
    out = {}
    for m, (u, ur, lap) in fields.items():
        out[m] = out.get(m, 0)-(1+0.2*r*r)*lap-0.4*r*ur+sigma*u
        for s, weight in harmonics.items():
            b = weight*r**s
            br = s*weight*r**(s-1)
            for sign in (-1, 1):
                value = -0.5*epsilon*(b*lap+br*ur-sign*s*m*b*u/(r*r))
                out[m+sign*s] = out.get(m+sign*s, 0)+value
    return out


def load_vector(space, values, quadrature=150):
    r, w = radial_rule(quadrature)
    data = values(r)
    out = np.zeros(space.size, dtype=complex)
    for j, (m, n) in enumerate(space.labels):
        if m in data:
            out[j] = np.sum(w*state_values(m, n, r)[0]*data[m])
    return out


def source_values(mode: int, beta=-3.0):
    return lambda r: {mode: r**abs(mode)*np.exp(beta*r*r)}


def manufactured_fields(r, modes=(0, 19)):
    return {m: tuple(v*(1+0.15j*(i+1)) for v in manufactured_radial(m, r))
            for i, m in enumerate(modes)}


def evaluate(space, c, r):
    fields = {}
    for i, m in enumerate(space.modes):
        block = c[i*space.radial_count:(i+1)*space.radial_count]
        if not np.any(block):
            continue
        values = [state_values(m, n, r) for n in range(space.radial_count)]
        fields[m] = tuple(sum((block[n]*values[n][j] for n in range(space.radial_count)),
                             np.zeros_like(r, dtype=complex)) for j in range(3))
    return fields


def physical_errors(space, c, truth, *, harmonics, epsilon, rhs=None, quadrature=170):
    r, w = radial_rule(quadrature)
    got = evaluate(space, c, r)
    expected = truth(r)
    l2 = h1 = l2den = h1den = 0.0
    for m in set(got) | set(expected):
        z = np.zeros_like(r, dtype=complex)
        u, ur, _ = got.get(m, (z, z, z))
        v, vr, _ = expected.get(m, (z, z, z))
        l2 += np.sum(w*np.abs(u-v)**2)
        l2den += np.sum(w*np.abs(v)**2)
        h1 += np.sum(w*(np.abs(ur-vr)**2+m*m*np.abs((u-v)/r)**2))
        h1den += np.sum(w*(np.abs(vr)**2+m*m*np.abs(v/r)**2))
    result = {"physical_l2": float(np.sqrt(l2/l2den)),
              "h1_seminorm": float(np.sqrt(h1/h1den))}
    if rhs is not None:
        applied = apply_physical_operator(got, r, harmonics, epsilon)
        force = rhs(r)
        err = sum(np.sum(w*np.abs(applied.get(m, 0)-force.get(m, 0))**2)
                  for m in set(applied) | set(force))
        den = sum(np.sum(w*np.abs(f)**2) for f in force.values())
        result["strong_residual_l2"] = float(np.sqrt(err/den))
    return result


def embed(source: DiskSpace, c, target: DiskSpace):
    out = np.zeros(target.size, dtype=c.dtype)
    for label, j in source.indices.items():
        if label in target.indices:
            out[target.indices[label]] = c[j]
    return out


class ModalPreconditioner:
    def __init__(self, op: DiskOperator):
        self.space = op.space
        n = self.space.radial_count
        self.factors = [cho_factor(op.base[i*n:(i+1)*n, i*n:(i+1)*n].toarray(),
                                  lower=True, check_finite=False)
                        for i in range(len(self.space.modes))]
        self.applied_blocks = 0

    def solve(self, rhs, active_modes=None):
        n = self.space.radial_count
        out = np.zeros_like(rhs)
        allowed = None if active_modes is None else set(active_modes)
        for i, m in enumerate(self.space.modes):
            if allowed is not None and m not in allowed:
                continue
            block = rhs[i*n:(i+1)*n]
            if np.any(block):
                out[i*n:(i+1)*n] = cho_solve(self.factors[i], block, check_finite=False)
                self.applied_blocks += 1
        return out


def reachable_modes(seeds, harmonics, allowed):
    return set(allowed) & {m+d for m in seeds for s in harmonics for d in (-s, s)}


def connected_modes(seeds, harmonics, allowed):
    out = set(seeds) & set(allowed)
    while True:
        new = out | reachable_modes(out, harmonics, allowed)
        if new == out:
            return tuple(sorted(out))
        out = new


def solve_iterative(op, rhs, epsilon, method, tolerance=1e-10, maxiter=160):
    start = perf_counter()
    pre = ModalPreconditioner(op)
    matrix = op.matrix(epsilon)
    prep_s = perf_counter()-start
    iterations = 0
    start_solve = perf_counter()
    if method == "pcg":
        def callback(_):
            nonlocal iterations
            iterations += 1
        inv = LinearOperator(matrix.shape, matvec=pre.solve, dtype=rhs.dtype)
        u, info = cg(matrix, rhs, M=inv, rtol=tolerance, atol=0, maxiter=maxiter,
                     callback=callback)
    elif method == "neumann":
        term = pre.solve(rhs)
        u = term.copy()
        info = maxiter
        for iterations in range(maxiter+1):
            if np.linalg.norm(rhs-matrix@u) <= tolerance*np.linalg.norm(rhs):
                info = 0
                break
            if iterations == maxiter:
                break
            term = -epsilon*pre.solve(op.perturbation@term)
            u += term
    else:
        raise ValueError(method)
    residual = rhs-matrix@u
    euclidean = np.linalg.norm(residual)/np.linalg.norm(rhs)
    pre_residual = np.sqrt(np.real(np.vdot(residual, pre.solve(residual))) /
                           np.real(np.vdot(rhs, pre.solve(rhs))))
    solve_s = perf_counter()-start_solve
    return u, {"method": method, "preconditioner_s": prep_s, "solve_and_check_s": solve_s,
               "total_solver_s": prep_s+solve_s, "iterations": iterations, "info": int(info),
               "relative_residual": float(euclidean),
               "relative_preconditioned_residual": float(pre_residual),
               "applied_modal_blocks": pre.applied_blocks}


def energy_error(op, epsilon, u, reference):
    a = op.matrix(epsilon)
    e = u-reference
    return float(np.sqrt(max(0, np.real(np.vdot(e, a@e))) /
                         np.real(np.vdot(reference, a@reference))))


def sparse_bytes(a):
    return a.data.nbytes+a.indices.nbytes+a.indptr.nbytes


def direct_solve(matrix, rhs):
    factor = splu(matrix)
    if np.iscomplexobj(rhs) and not np.iscomplexobj(matrix.data):
        return factor.solve(rhs.real)+1j*factor.solve(rhs.imag)
    return factor.solve(rhs)
