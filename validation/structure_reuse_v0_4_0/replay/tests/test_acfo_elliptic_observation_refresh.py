import numpy as np
from scipy.sparse.linalg import splu
from scripts.acfo_elliptic_observation_refresh import Runner, ObservationContext
from scripts.acfo_elliptic_refresh import spectral, TaylorRefresh, SEEDS
from scripts.acfo_cylinder_coupling import (
    Bank, CylinderSpace, reachable, matrix_parts, stack_field, observation_matrix, observation,
)


def event(kind='base', q=(1.7, 4.3), weight='unit', radius=7):
    return dict(theta=[.12, -.08], tolerance=1e-8,
                observation=dict(kind=kind, q=list(q), weight=weight, radius=radius))


def reference(e, nr=6):
    c = ObservationContext(nr, 1, 181)
    c.set_observation(e['observation'])
    space = CylinderSpace(tuple(sorted(set().union(*reachable(SEEDS, 9).values()))), nr)
    base, vs = matrix_parts(Bank(nr, 181), space)
    a = base+sum(t*v for t,v in zip(e['theta'], vs))
    lu = splu(a.tocsc()); b = stack_field(space, c.sources, c.ncols)
    u = lu.solve(b.real)+1j*lu.solve(b.imag)
    return observation_matrix(space, c.obs)@u, c.rootw


def test_radial_observation_change_reuses_states_but_invalidates_response():
    runner = Runner('query_reuse', 6, 1)
    a, first = runner.step(event())
    b, second = runner.step(event(q=(1.9, 4.5)))
    assert second['order'] == first['order']
    assert second['new_nodes'] == 0 and not second['response_cache_hit']
    assert spectral(a-b) > 1e-5
    ref, w = reference(event(q=(1.9, 4.5)))
    assert spectral(w*(b-ref))/spectral(w*ref) < 1e-8


def test_new_modes_restore_missing_ancestors_without_mutating_old_states():
    r = Runner('query_reuse', 6, 1)
    r.step(event())
    before = {(a,m): v.copy() for a,f in r.engine.values.items() for m,v in f.items()}
    answer, meta = r.step(event('expanded'))
    assert meta['new_nodes'] > 0 and meta['reused_nodes'] > 0
    assert all(np.array_equal(v, r.engine.values[a][m]) for (a,m),v in before.items())
    ref, w = reference(event('expanded'))
    assert spectral(w*(answer-ref))/spectral(w*ref) < 1e-8


def test_new_neighbor_observations_can_already_exist_as_cached_ancestors():
    r = Runner('query_reuse', 6, 1)
    r.step(event())
    y, meta = r.step(event('neighbor'))
    assert meta['new_nodes'] == 0 and not meta['response_cache_hit']
    ref, w = reference(event('neighbor'))
    assert spectral(w*(y-ref))/spectral(w*ref) < 1e-8


def test_full_observation_coverage_restores_full_taylor_graph():
    q = Runner('query_reuse', 5, 1); f = Runner('full_reuse', 5, 1)
    q.step(event()); f.step(event())
    e = event('shell', radius=8)
    y, qm = q.step(e); z, fm = f.step(e)
    assert qm['order'] <= 8
    assert qm['retained_nodes'] == fm['retained_nodes']
    assert spectral(y-z) < 1e-13


def test_direct_field_reuse_recertifies_new_observation_without_cg():
    r = Runner('direct_field_reuse', 6, 1)
    r.step(event())
    e = event('expanded', q=(1.9, 4.5))
    y, meta = r.step(e)
    assert meta['field_reused'] and not meta['response_cache_hit']
    assert meta['solve_calls'] == meta['iterations'] == 0
    ref, w = reference(e)
    assert spectral(w*(y-ref))/spectral(w*ref) <= meta['relative_bound']+1e-11


def test_observation_weight_is_part_of_cache_key_and_return_uses_exact_cache():
    r = Runner('query_reuse', 5, 1)
    y, a = r.step(event())
    _, b = r.step(event(weight='ramp'))
    z, c = r.step(event())
    assert not b['response_cache_hit'] and not b['observation_cache_hit']
    assert b['new_radial_rows'] == b['new_dual_rows'] == 0
    assert c['response_cache_hit'] and c['observation_cache_hit']
    assert np.array_equal(y, z)
    assert c['new_nodes'] == c['solve_calls'] == 0


def test_cached_observation_rows_match_standalone_quadrature_with_signed_orders():
    from scripts.acfo_elliptic_observation_refresh import geometry
    c = ObservationContext(8, 1)
    spec = event('expanded')['observation']
    c.set_observation(spec)
    requests, _ = geometry(spec)
    expected = observation(c.nr, requests, 145)
    assert max(np.linalg.norm(a-b) for (_,a),(_,b) in zip(c.obs, expected)) < 1e-14
