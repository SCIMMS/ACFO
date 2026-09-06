import numpy as np
import pytest
from scripts.acfo_elliptic_refresh import (
    Context, TaylorRefresh, DirectRefresh, spectral, rho_bound,
)


def test_query_extension_matches_full_and_independent_nonzero_solve():
    c = Context(nr=7, per_seed=1)
    q = TaylorRefresh(c, query=True)
    for theta, tol in [((.04, .02), 1e-6), ((.12, -.08), 1e-8), ((.04, .02), 1e-8)]:
        answer, meta = q.action(theta, tol)
        full, _ = TaylorRefresh(Context(7, 1)).action(theta, tol)
        exact, ref = DirectRefresh(Context(7, 1, 177)).action(theta, 1e-11)
        assert spectral(c.rootw*(answer-full)) < 1e-13
        error = spectral(c.rootw*(answer-exact))/spectral(c.rootw*exact)
        assert error <= meta['relative_bound']+ref['relative_bound']+1e-12
        assert error < tol


def test_tighter_precision_extends_and_returning_request_reuses_without_solves():
    c = Context(6, 1)
    e = TaylorRefresh(c, query=True)
    _, loose = e.action((.12, .08), 1e-6)
    _, tight = e.action((.12, .08), 1e-8)
    _, again = e.action((.12, .08), 1e-6)
    assert tight['order'] > loose['order']
    assert tight['new_nodes'] > 0 and tight['reused_nodes'] > 0
    assert again['solve_calls'] == again['new_nodes'] == 0


def test_contraction_failure_is_explicit_and_material_strength_changes_region():
    c = Context(5, 1)
    assert c.select((.24, .16), 1e-8)[0] > c.select((.04, .02), 1e-8)[0]
    with pytest.raises(ValueError, match='contraction'):
        rho_bound((2., 0.))


def test_direct_certificate_checks_exterior_and_warm_solver_reuses_parts():
    c = Context(6, 1)
    d = DirectRefresh(c)
    _, first = d.action((.24, .16), 1e-8)
    count = len(d.parts)
    _, second = d.action((.24, .16), 1e-8)
    assert first['exterior_modes_checked'] > 0
    assert first['attempts'][0]['relative_bound'] > 1e-8
    assert second['relative_bound'] <= 1e-8
    assert len(d.parts) == count
    assert second['iterations'] <= first['iterations']
