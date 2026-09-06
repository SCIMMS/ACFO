"""Check ablation equivalence and independent material derivative references."""
import numpy as np
from scipy.sparse.linalg import splu
from scripts.acfo_cylinder_coupling import (
    Bank, CylinderSpace, reachable, matrix_parts, stack_field, observation_matrix,
)
from scripts.acfo_elliptic_ablation import (
    ARMS, SEEDS, Engine, inputs, mixtures, full_sparse_reference, relative_by_derivative,
)


def config(order=2, observation="selected"):
    return dict(id="test", order=order, nr=6, per_seed=2, observation=observation)


def test_six_policies_preserve_independent_global_response_and_source_mixtures():
    cfg = config()
    sources, obs, requests, weights = inputs(cfg)
    ref, _, _ = full_sparse_reference(cfg)
    for arm in ARMS:
        engine = Engine(cfg, arm, sources, obs, requests)
        for mix in mixtures(4, 3):
            assert max(relative_by_derivative(engine.action(mix), ref@mix, weights)) < 2e-11


def test_preparation_cse_support_and_response_cache_have_separate_operation_counts():
    cfg = config()
    sources, obs, requests, _ = inputs(cfg)
    engines = {arm: Engine(cfg, arm, sources, obs, requests) for arm in ARMS}
    for mix in mixtures(4, 3):
        for e in engines.values():
            e.action(mix)
    m = {a: e.metrics() for a, e in engines.items()}
    assert m["separate_preparation"]["factorizations"] > m["shared_preparation"]["factorizations"]
    assert m["separate_preparation"]["solve_calls"] == m["shared_preparation"]["solve_calls"]
    assert m["shared_preparation"]["solve_calls"] > m["joint_forward"]["solve_calls"]
    assert m["joint_forward"]["solve_calls"] > m["joint_query"]["solve_calls"]
    assert m["cached_forward"]["solve_calls"]*3 == m["joint_forward"]["solve_calls"]
    assert m["cached_query"]["solve_calls"]*3 == m["joint_query"]["solve_calls"]
    assert m["cached_query"]["cache_hits"] == 2


def test_full_observation_removes_support_pruning_advantage():
    cfg = config(observation="full")
    sources, obs, requests, _ = inputs(cfg)
    forward = Engine(cfg, "joint_forward", sources, obs, requests)
    query = Engine(cfg, "joint_query", sources, obs, requests)
    assert forward.plan == query.plan


def test_material_jacobian_and_hessian_agree_with_nonzero_parameter_solves():
    cfg = config()
    sources, obs, requests, _ = inputs(cfg)
    engine = Engine(cfg, "joint_query", sources, obs, requests)
    response = engine.action(np.eye(4, dtype=complex))
    labels = tuple(sorted(set().union(*reachable(SEEDS, 6).values())))
    space = CylinderSpace(labels, cfg["nr"])
    base, vs = matrix_parts(Bank(cfg["nr"], 173), space)
    b = stack_field(space, sources, 4); c = observation_matrix(space, obs)
    direction = np.array([.7, -.4])
    def value(t):
        lu = splu((base+t*(direction[0]*vs[0]+direction[1]*vs[1])).tocsc())
        return c@(lu.solve(b.real)+1j*lu.solve(b.imag))
    d = dict(zip(requests, response))
    jac = direction[0]*d[1, 0]+direction[1]*d[0, 1]
    hess = direction[0]**2*d[2, 0]+2*np.prod(direction)*d[1, 1]+direction[1]**2*d[0, 2]
    jac_fd=[]; hess_fd=[]
    for h in (.02, .01):
        plus, minus, zero = value(h), value(-h), value(0)
        jac_fd.append((plus-minus)/(2*h))
        hess_fd.append((plus-2*zero+minus)/(h*h))
    for exact, fd in [(jac, jac_fd), (hess, hess_fd)]:
        extrapolated = (4*fd[1]-fd[0])/3
        assert np.linalg.norm(extrapolated-exact)/np.linalg.norm(exact) < 1e-6
