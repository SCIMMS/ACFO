"""Validation adapter for observation changes; prior frozen adapters are unmodified."""
from time import perf_counter
import numpy as np
from scipy.special import jv
from scripts.acfo_elliptic_refresh import (
    Context, TaylorRefresh, DirectRefresh, spectral, gram_norm, SEEDS,
)
from scripts.acfo_cylinder_coupling import observation, observe, reachable

ARMS = ('query_cold', 'query_blocks', 'full_reuse', 'query_reuse', 'direct_field_reuse')
BASE_MODES = ((0, 0), (5, -2), (2, 1), (3, -2), (5, -1))
NEIGHBOR_MODES = BASE_MODES+((4, 2), (6, -4), (1, -3))
EXPANDED_MODES = NEIGHBOR_MODES+((6, 3), (9, -6), (7, 0), (-4, 5), (0, 7))


def geometry(spec):
    kind = spec['kind']
    if kind == 'base':
        modes = BASE_MODES
    elif kind == 'neighbor':
        modes = NEIGHBOR_MODES
    elif kind == 'expanded':
        modes = EXPANDED_MODES
    elif kind == 'shell':
        modes = tuple(sorted(set().union(*reachable(SEEDS, spec['radius']).values())))
    else:
        raise ValueError(kind)
    qs = spec.get('q', [1.7, 4.3])
    requests = tuple((m, float(q)) for m in modes for q in qs)
    weights = (np.linspace(.8, 1.2, len(requests)) if spec.get('weight') == 'ramp'
               else np.ones(len(requests)))
    return requests, weights


class ObservationContext(Context):
    def __init__(self, nr=18, per_seed=2, quadrature=145):
        super().__init__(nr, per_seed, quadrature)
        self.quadrature = quadrature
        self.observation_cache = {}
        self.row_cache = {}
        self.dual_row_cache = {}

    def set_observation(self, spec):
        start = perf_counter()
        requests, weights = geometry(spec)
        key = requests, tuple(float(x) for x in weights)
        cached = key in self.observation_cache
        factors = len(self.bank.factors)
        old_rows = len(self.row_cache)
        old_duals = len(self.dual_row_cache)
        if not cached:
            obs = []
            for mode, q in requests:
                radial_key = mode[0], q
                if radial_key not in self.row_cache:
                    if self.quadrature == 145:
                        row = self.bank.state(mode[0]).T@(self.bank.w*jv(mode[0], q*self.bank.r))
                    else:
                        # Keep the standalone finer-quadrature reference path.
                        row = observation(self.nr, [(mode, q)], self.quadrature)[0][1]
                    self.row_cache[radial_key] = row
                row = self.row_cache[radial_key]
                if (mode, q) not in self.dual_row_cache:
                    self.dual_row_cache[mode, q] = self.bank.solve(mode, row)
                obs.append((mode, row))
            rootw = np.sqrt(weights)[:, None]
            # CK0^-1C* is block diagonal in observed (m,k). Its spectral norm
            # is the maximum block norm, with no dense all-observation matrix.
            maximum = 0.
            groups = {}
            for i, (mode, row) in enumerate(obs):
                groups.setdefault(mode, []).append(i)
            for mode, ids in groups.items():
                rows = np.asarray([rootw[i, 0]*obs[i][1] for i in ids])
                duals = np.asarray([rootw[i, 0]*self.dual_row_cache[requests[i]] for i in ids])
                maximum = max(maximum, gram_norm(rows@duals.T))
            y0 = spectral(rootw*observe(self.u0, obs, self.ncols))
            self.observation_cache[key] = obs, rootw, maximum, y0
        self.obs, self.rootw, self.observation_dual, self.y0norm = self.observation_cache[key]
        return key, dict(observation_s=perf_counter()-start, observation_cache_hit=cached,
                         observation_rows=len(self.obs),
                         new_radial_rows=len(self.row_cache)-old_rows,
                         new_dual_rows=len(self.dual_row_cache)-old_duals,
                         observation_new_factors=len(self.bank.factors)-factors)

    def observation_bytes(self):
        arrays = list(self.row_cache.values())+list(self.dual_row_cache.values())
        arrays += [w for _, w, _, _ in self.observation_cache.values()]
        return sum(a.nbytes for a in {id(a): a for a in arrays}.values())


class DirectFieldReuse:
    """Transport an observation-independent energy error bound with the field."""
    def __init__(self, context):
        self.ctx = context
        self.solver = DirectRefresh(context)
        self.theta = None
        self.energy_error_bound = None

    def action(self, theta, tolerance):
        c = self.ctx
        start = perf_counter()
        if self.theta == tuple(theta):
            y = observe(self.solver.last_field, c.obs, c.ncols)
            absolute = c.observation_dual*self.energy_error_bound
            lower = spectral(c.rootw*y)-absolute
            bound = absolute/lower if lower > 0 else float('inf')
            if bound <= tolerance:
                return y, dict(relative_bound=bound, absolute_bound=absolute,
                    field_reused=True, solve_calls=0, iterations=0,
                    field_recertification_s=perf_counter()-start,
                    radius=self.solver.last_radius, new_nodes=0)
        y, meta = self.solver.action(theta, tolerance)
        self.theta = tuple(theta)
        self.energy_error_bound = meta['absolute_bound']/c.observation_dual
        return y, dict(meta, field_reused=False, new_nodes=0,
                       field_recertification_s=perf_counter()-start)


class Runner:
    def __init__(self, arm, nr=18, per_seed=2, quadrature=145):
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.args = nr, per_seed, quadrature
        self.ctx = self.engine = None
        self.responses = {}

    def step(self, event):
        start = perf_counter()
        setup = perf_counter()
        if self.ctx is None or self.arm == 'query_cold':
            self.ctx = ObservationContext(*self.args)
            if self.arm == 'direct_field_reuse':
                self.engine = DirectFieldReuse(self.ctx)
            else:
                self.engine = TaylorRefresh(self.ctx, query=self.arm != 'full_reuse',
                                           retain_values=self.arm != 'query_blocks')
        setup_s = perf_counter()-setup
        observation_key, obs_meta = self.ctx.set_observation(event['observation'])
        key = tuple(event['theta']), observation_key
        old = self.responses.get(key) if self.arm != 'query_cold' else None
        if old is not None and old[1]['relative_bound'] <= event['tolerance']:
            y, old_meta = old
            meta = dict(old_meta)
            for name in list(meta):
                if name.endswith('_s') or name in ('new_nodes', 'reused_nodes', 'solve_calls', 'iterations'):
                    meta[name] = 0
            meta['response_cache_hit'] = True
            meta['field_reused'] = False
        else:
            y, meta = self.engine.action(event['theta'], event['tolerance'])
            meta['response_cache_hit'] = False
            if self.arm != 'query_cold':
                self.responses[key] = y, dict(meta)
        meta.update(obs_meta)
        meta['setup_s'] = setup_s
        meta['bank_bytes'] = self.ctx.bank.bytes()
        meta['response_cache_bytes'] = sum(y.nbytes for y, _ in self.responses.values())
        meta['observation_cache_bytes'] = self.ctx.observation_bytes()
        if isinstance(self.engine, TaylorRefresh):
            meta['retained_nodes'] = sum(map(len, self.engine.values.values()))
            meta['state_bytes'] = sum(v.nbytes for f in self.engine.values.values() for v in f.values())
            meta['sparse_bytes'] = 0
        else:
            solver = self.engine.solver
            meta['retained_nodes'] = len(solver.last_field)
            meta['state_bytes'] = sum(v.nbytes for v in solver.last_field.values())
            meta['sparse_bytes'] = sum(sum(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes
                for a in (base, *vs)) for _, base, vs in solver.parts.values())
        meta['total_s'] = perf_counter()-start
        return y, meta
