"""Validation-only precision-aware refresh in the existing affine elliptic family.

The zero-material anchor is fixed. Reusing multivariate Taylor moments is a
known parametric strategy; this adapter tests dependency selection and refresh.
Bounds are for infinite angular/axial labels at a fixed radial Galerkin degree,
in exact arithmetic. Radial and floating-point errors need separate audits.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from scripts.acfo_cylinder_coupling import (
    Bank, CylinderSpace, SHIFTS, alphas, make_plan, reachable, plus, minus,
    source_columns, observation, observe, matrix_parts, stack_field,
    observation_matrix,
)

SEEDS = ((0, 0), (5, -2))
ARMS = ('taylor_cold', 'taylor_blocks', 'taylor_full_reuse',
        'taylor_query_reuse', 'direct_cold', 'direct_reuse')


def spectral(x):
    return float(np.linalg.norm(x, 2))


def gram_norm(g):
    return float(np.sqrt(max(0., np.linalg.eigvalsh((g+g.conj().T)/2)[-1])))


def rho_bound(theta):
    # max_{0<=r<=1} (.6|t0| r²+.4|t1| r³)/(1+.2r²).
    rho = (.6*abs(theta[0])+.4*abs(theta[1]))/1.2
    if not rho < 1:
        raise ValueError('Anchor contraction bound fails: recenter/another solver required')
    return float(rho)


class Context:
    def __init__(self, nr=18, per_seed=2, quadrature=145):
        self.nr = nr
        self.bank = Bank(nr, quadrature)
        self.sources = source_columns(nr, SEEDS, per_seed)
        # Independent references also reintegrate the loads.
        if quadrature != 145:
            from scripts.acfo_cylinder_coupling import modal_load
            for i, mode in enumerate(SEEDS):
                for j in range(per_seed):
                    self.sources[mode][:, i*per_seed+j] = modal_load(
                        nr, mode, lambda r: r**abs(mode[0])*np.exp((-2.-.4*j)*r*r),
                        quadrature=quadrature)*(1+.13j*(j+1))
        self.ncols = 2*per_seed
        self.obs = observation(nr, [(m, q) for m in
            [(0, 0), (5, -2), (2, 1), (3, -2), (5, -1)] for q in (1.7, 4.3)], quadrature)
        self.rootw = np.sqrt(np.linspace(.4, 1.6, len(self.obs)))[:, None]
        self.u0 = {m: self.bank.solve(m, b) for m, b in self.sources.items()}
        self.source_energy = gram_norm(sum(u.conj().T@self.sources[m] for m, u in self.u0.items()))
        dual = np.zeros((len(self.obs), len(self.obs)))
        for mode in {m for m, _ in self.obs}:
            ids = [i for i, (m, _) in enumerate(self.obs) if m == mode]
            rows = np.array([self.obs[i][1] for i in ids])*self.rootw[ids]
            dual[np.ix_(ids, ids)] = rows@self.bank.solve(mode, rows.T)
        self.observation_dual = gram_norm(dual)
        self.y0norm = spectral(self.rootw*observe(self.u0, self.obs, self.ncols))

    def select(self, theta, tolerance, max_order=18):
        rho = rho_bound(theta)
        scale = self.source_energy*self.observation_dual
        # Reverse triangle inequality gives a theta-dependent, computable lower
        # bound on the exact response norm without consulting a reference.
        lower = self.y0norm-scale*rho/(1-rho)
        if lower <= 0:
            raise ValueError('Response lower bound is inconclusive; use residual-based solver')
        for p in range(max_order+1):
            tail = scale*rho**(p+1)/(1-rho)
            if tail <= tolerance*lower:
                return p, float(tail/lower), float(tail), rho
        raise ValueError('Maximum Taylor order reached without accuracy certificate')


class TaylorRefresh:
    def __init__(self, context, query=False, retain_values=True):
        self.ctx, self.query, self.retain_values = context, query, retain_values
        self.values = {(0, 0): dict(context.u0)}

    def action(self, theta, tolerance):
        c = self.ctx
        before = c.bank.factor_solves
        old_nodes = sum(map(len, self.values.values()))
        p, bound, absolute, rho = c.select(theta, tolerance)
        t = perf_counter()
        plan = make_plan(SEEDS, alphas(p), {m for m, _ in c.obs}, self.query)
        plan_s = perf_counter()-t
        t = perf_counter()
        if not self.retain_values:
            self.values = {(0, 0): dict(c.u0)}
        reused = added = 0
        for alpha, nodes in plan.items():
            field = self.values.setdefault(alpha, {})
            for target in sorted(nodes):
                if target in field:
                    reused += 1
                    continue
                rhs = np.zeros((c.nr, c.ncols), complex)
                for l in range(2):
                    if not alpha[l]:
                        continue
                    prev = list(alpha); prev[l] -= 1
                    previous = self.values[tuple(prev)]
                    # Only the plan's ancestors are required; cached extras do
                    # not change the recurrence because all true ancestors of
                    # a requested node are present.
                    for source in (minus(target, SHIFTS[l]), plus(target, SHIFTS[l])):
                        if source in previous:
                            rhs -= c.bank.edge(l, target, source)@previous[source]
                field[target] = c.bank.solve(target, rhs)
                added += 1
        build_s = perf_counter()-t
        t = perf_counter()
        answer = sum(theta[0]**a*theta[1]**b*observe(self.values[a, b], c.obs, c.ncols)
                     for a, b in alphas(p))
        apply_s = perf_counter()-t
        if not np.isfinite(answer).all():
            raise FloatingPointError('Nonfinite response')
        return answer, dict(order=p, rho=rho, relative_bound=bound, absolute_bound=absolute,
            planning_s=plan_s, extension_s=build_s, synthesis_s=apply_s,
            new_nodes=added, reused_nodes=reused, prior_nodes=old_nodes,
            requested_nodes=sum(map(len, plan.values())),
            retained_nodes=sum(map(len, self.values.values())),
            retained_values_bytes=sum(v.nbytes for f in self.values.values() for v in f.values()),
            bank_bytes=c.bank.bytes(), factor_keys=len(c.bank.factors),
            solve_calls=c.bank.factor_solves-before)


def residual_bound(context, space, matrix, b, u, theta):
    """Include exterior leakage; a small truncated-system residual is insufficient."""
    c = context
    interior = b-matrix@u
    fields = {m: u[i*c.nr:(i+1)*c.nr] for i, m in enumerate(space.modes)}
    residuals = {m: interior[i*c.nr:(i+1)*c.nr] for i, m in enumerate(space.modes)}
    for source, state in fields.items():
        for l, shift in enumerate(SHIFTS):
            if theta[l] == 0:
                continue
            for target in (plus(source, shift), minus(source, shift)):
                if target in space.indices:
                    continue
                if target not in residuals:
                    residuals[target] = np.zeros_like(state)
                residuals[target] -= theta[l]*(c.bank.edge(l, target, source)@state)
    gram = np.zeros((c.ncols, c.ncols), complex)
    for mode, r in residuals.items():
        gram += r.conj().T@c.bank.solve(mode, r)
    dual_residual = gram_norm(gram)
    bound = c.observation_dual*dual_residual/(1-rho_bound(theta))
    return float(bound), len(residuals)-len(space.modes)


class DirectRefresh:
    """Generic reusable affine assembly + block preconditioning + warm-start CG."""
    def __init__(self, context):
        self.ctx = context
        self.parts = {}
        self.last_field = {}
        self.last_radius = 2

    def action(self, theta, tolerance, max_radius=16):
        c = self.ctx
        rho_bound(theta)
        before = c.bank.factor_solves
        attempts = []; iterations = 0; assembly_s = solve_s = audit_s = 0.
        for radius in range(self.last_radius, max_radius+1, 2):
            t = perf_counter()
            if radius not in self.parts:
                modes = tuple(sorted(set().union(*reachable(SEEDS, radius).values())))
                space = CylinderSpace(modes, c.nr)
                base, vs = matrix_parts(c.bank, space)
                self.parts[radius] = space, base, vs
            space, base, vs = self.parts[radius]
            matrix = base+theta[0]*vs[0]+theta[1]*vs[1]
            b = stack_field(space, c.sources, c.ncols)
            x0 = stack_field(space, self.last_field, c.ncols)
            assembly_s += perf_counter()-t
            t = perf_counter()
            def pre(x):
                return np.concatenate([c.bank.solve(m, x[i*c.nr:(i+1)*c.nr])
                                       for i, m in enumerate(space.modes)])
            op = LinearOperator(matrix.shape, matvec=pre, dtype=np.complex128)
            columns = []
            counts = []
            for j in range(c.ncols):
                count = [0]
                def callback(_): count[0] += 1
                solve_tolerance = (min(1e-12, tolerance*1e-3) if tolerance <= 1e-10
                                   else tolerance*.01)
                x, info = cg(matrix, b[:, j], M=op, x0=x0[:, j],
                             rtol=solve_tolerance, atol=0., maxiter=200,
                             callback=callback)
                if info:
                    raise RuntimeError(f'CG failed: {info}')
                columns.append(x); counts.append(count[0])
            u = np.column_stack(columns)
            iterations += sum(counts)
            solve_s += perf_counter()-t
            self.last_field = {m: u[i*c.nr:(i+1)*c.nr] for i, m in enumerate(space.modes)}
            t = perf_counter()
            answer = observation_matrix(space, c.obs)@u
            absolute, exterior = residual_bound(c, space, matrix, b, u, theta)
            lower = spectral(c.rootw*answer)-absolute
            bound = absolute/lower if lower > 0 else float('inf')
            audit_s += perf_counter()-t
            attempts.append(dict(radius=radius, relative_bound=bound, modes=len(space.modes)))
            if bound <= tolerance:
                self.last_radius = radius
                return answer, dict(radius=radius, relative_bound=bound, absolute_bound=absolute,
                    attempts=attempts, modes=len(space.modes), exterior_modes_checked=exterior,
                    assembly_s=assembly_s, solve_s=solve_s, certificate_s=audit_s,
                    iterations=iterations, solve_calls=c.bank.factor_solves-before,
                    bank_bytes=c.bank.bytes(), factor_keys=len(c.bank.factors),
                    retained_values_bytes=sum(v.nbytes for v in self.last_field.values()),
                    retained_sparse_bytes=sum(sum(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes
                        for a in (base, *vs)) for _, base, vs in self.parts.values()))
        raise RuntimeError('Angular region limit reached without residual certificate')
