"""Validation-only ablation: one sparse executor, six preparation/reuse policies."""
from __future__ import annotations
from math import factorial
from time import perf_counter
import numpy as np
from scipy.sparse.linalg import splu
from scripts.acfo_cylinder_coupling import (
    Bank, CylinderSpace, alphas, reachable, make_plan, prepare_plan, evaluate_plan,
    source_columns, modal_load, observation, observe, observation_matrix,
    matrix_parts, stack_field,
)

ARMS = ("separate_preparation", "shared_preparation", "joint_forward",
        "joint_query", "cached_forward", "cached_query")
SEEDS = ((0, 0), (5, -2))


def request_geometry(cfg):
    requests = alphas(cfg["order"])
    full = reachable(SEEDS, cfg["order"])
    if cfg["observation"] == "full":
        modes = sorted(set().union(*full.values()))
    else:
        modes = [(0, 0), (5, -2), (2, 1), (3, -2)]
        if cfg["order"] >= 2:
            modes.append((5, -1))  # Nonzero mixed material derivative.
    return requests, [(mode, q) for mode in modes for q in (1.7, 4.3)]


def inputs(cfg, nr=None, reference=False):
    nr = nr or cfg["nr"]
    sources = source_columns(nr, SEEDS, cfg["per_seed"])
    if reference:
        # Reintegrate physical profiles at separate, finer source quadrature.
        for i, mode in enumerate(SEEDS):
            for j in range(cfg["per_seed"]):
                beta = -2. - .4*j
                sources[mode][:, i*cfg["per_seed"]+j] = modal_load(
                    nr, mode, lambda r: r**abs(mode[0])*np.exp(beta*r*r),
                    quadrature=max(230, 4*nr+30))*(1+.13j*(j+1))
    requests, geometry = request_geometry(cfg)
    obs = observation(nr, geometry, max(185 if reference else 145, 4*nr+20))
    weights = np.linspace(.4, 1.6, len(obs))
    return sources, obs, requests, weights


def collect(values, requests, obs, ncols):
    # Return actual derivatives, not Taylor coefficients.
    return np.array([factorial(a[0])*factorial(a[1])*observe(values[a], obs, ncols)
                     for a in requests])


def ancestor_subplan(prepared, alpha):
    ordered, operators = prepared
    return ({a: nodes for a, nodes in ordered.items()
             if all(x <= y for x, y in zip(a, alpha))}, operators)


def sparse_bytes(operators):
    return sum(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes for a in operators.values())


class Engine:
    def __init__(self, cfg, arm, sources, obs, requests):
        if arm not in ARMS:
            raise ValueError(arm)
        self.cfg, self.arm = cfg, arm
        self.sources, self.obs, self.requests = sources, obs, requests
        self.ncols = next(iter(sources.values())).shape[1]
        self.cached = arm.startswith("cached_")
        self.response = None
        self.solve_calls = 0
        self.solved_rhs_columns = 0
        self.state_peak_bytes = 0
        self.max_residual = 0.
        self.response_build_s = 0.
        self.cache_hits = 0
        t = perf_counter()
        query = arm in ("joint_query", "cached_query")
        self.plan = make_plan(SEEDS, requests, {m for m, _ in obs}, query)
        # Graph construction is timed for all arms, even cached responses.
        if arm == "separate_preparation":
            plans = [{a: nodes for a, nodes in self.plan.items()
                      if all(x <= y for x, y in zip(a, target))}
                     for target in requests]
        else:
            plans = [self.plan]
        self.planning_s = perf_counter()-t
        t = perf_counter()
        self.banks = [Bank(cfg["nr"], max(145, 4*cfg["nr"]+20)) for _ in plans]
        self.prepared = [prepare_plan(b, p, True) for b, p in zip(self.banks, plans)]
        self.block_preparation_s = perf_counter()-t
        self.bank_array_bytes = sum(b.bytes() for b in self.banks)
        self.stage_array_bytes = sum(sparse_bytes(p[1]) for p in self.prepared)
        self.unique_factor_keys = len(set().union(*(set(b.factors) for b in self.banks)))
        self.factorizations = sum(len(b.factors) for b in self.banks)
        self.prepared_nodes = sum(sum(map(len, p[0].values())) for p in self.prepared)

    def _eval(self, bank, prepared, sources):
        before = bank.factor_solves
        values, residual = evaluate_plan(bank, prepared, sources, True)
        solves = bank.factor_solves-before
        self.solve_calls += solves
        self.solved_rhs_columns += solves*self.ncols
        self.max_residual = max(self.max_residual, residual)
        self.state_peak_bytes = max(self.state_peak_bytes,
            sum(x.nbytes for field in values.values() for x in field.values()))
        return values

    def _compute(self, sources):
        if self.arm in ("separate_preparation", "shared_preparation"):
            answers = []
            for i, alpha in enumerate(self.requests):
                idx = i if self.arm == "separate_preparation" else 0
                prepared = self.prepared[idx]
                if self.arm == "shared_preparation":
                    prepared = ancestor_subplan(prepared, alpha)
                values = self._eval(self.banks[idx], prepared, sources)
                answers.append(factorial(alpha[0])*factorial(alpha[1])*
                               observe(values[alpha], self.obs, self.ncols))
                del values
            return np.array(answers)
        values = self._eval(self.banks[0], self.prepared[0], sources)
        return collect(values, self.requests, self.obs, self.ncols)

    def action(self, coefficients):
        if coefficients.shape != (self.ncols, self.ncols):
            raise ValueError("Expected a batch of source-coordinate mixtures")
        if self.cached:
            if self.response is None:
                t = perf_counter()
                self.response = self._compute(self.sources)
                self.response_build_s = perf_counter()-t
                # Fixed-family response needs no factors or stage matrices later.
                self.banks = []
                self.prepared = []
            else:
                self.cache_hits += 1
            result = self.response @ coefficients
        else:
            result = self._compute({m: b@coefficients for m, b in self.sources.items()})
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite derivative response")
        return result

    def metrics(self):
        return dict(planning_s=self.planning_s, block_preparation_s=self.block_preparation_s,
                    response_build_s=self.response_build_s,
                    bank_array_bytes=self.bank_array_bytes,
                    sparse_stage_array_bytes=self.stage_array_bytes,
                    prepared_response_array_bytes=0 if self.response is None else self.response.nbytes,
                    retained_bank_array_bytes=self.bank_array_bytes if self.banks else 0,
                    retained_stage_array_bytes=self.stage_array_bytes if self.prepared else 0,
                    retained_state_peak_bytes=self.state_peak_bytes,
                    unique_factor_keys=self.unique_factor_keys, factorizations=self.factorizations,
                    prepared_graph_nodes=self.prepared_nodes,
                    solve_calls=self.solve_calls, solved_rhs_columns=self.solved_rhs_columns,
                    cache_hits=self.cache_hits, recurrence_residual_max=self.max_residual)


def mixtures(ncols, calls):
    rng = np.random.default_rng(20260905)
    return [np.eye(ncols, dtype=complex)] + [
        (rng.normal(size=(ncols, ncols))+1j*rng.normal(size=(ncols, ncols)))/np.sqrt(2*ncols)
        for _ in range(calls-1)]


def relative_by_derivative(actual, reference, weights):
    errors = []
    rootw = np.sqrt(weights)[:, None]
    for a, b in zip(actual, reference):
        norm = np.linalg.norm(rootw*b, 2)
        error = np.linalg.norm(rootw*(a-b), 2)
        errors.append(float(error/norm if norm > 1e-25 else error))
    return errors


def full_sparse_reference(cfg, nr=None):
    """Global sparse LU recurrence, no graph-local evaluation or slicing."""
    nr = nr or cfg["nr"]
    sources, obs, requests, weights = inputs(cfg, nr, True)
    labels = tuple(sorted(set().union(*reachable(SEEDS, cfg["order"]).values())))
    space = CylinderSpace(labels, nr)
    bank = Bank(nr, max(185, 4*nr+30))
    base, vs = matrix_parts(bank, space)
    lu = splu(base.tocsc())
    def solve(rhs):
        return lu.solve(rhs.real)+1j*lu.solve(rhs.imag)
    field = {(0, 0): solve(stack_field(space, sources, 2*cfg["per_seed"]))}
    for alpha in requests[1:]:
        rhs = np.zeros_like(field[0, 0])
        for l in range(2):
            if alpha[l]:
                prev = list(alpha); prev[l] -= 1
                rhs -= vs[l]@field[tuple(prev)]
        field[alpha] = solve(rhs)
    c = observation_matrix(space, obs)
    response = np.array([factorial(a[0])*factorial(a[1])*(c@field[a]) for a in requests])
    return response, weights, dict(dof=space.size, modes=len(labels),
                                  reference_quadrature=max(185, 4*nr+30))
