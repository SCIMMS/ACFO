import numpy as np
import pytest
from scripts.acfo_elliptic_update_policy import (
    Context, Direct, IncrementalTaylor, Dispatcher, CostModel, bind_observation, spectral,
)
from scripts.acfo_elliptic_new_coupling import Taylor


def event(g=0.,obs='narrow',tol=1e-8):
    return dict(theta=[.12,-.08,g],observation=obs,tolerance=tol)


def model(update, direct):
    return CostModel({'update':{'coefficients':[update,0,0,0,0]},
                      'direct':{'coefficients':[direct,0,0,0,0,0]}})


def test_partial_work_cannot_be_returned_and_resumes_without_mutating_completed_nodes():
    c=Context(5,1);u=IncrementalTaylor(c)
    p=u.inspect((.12,-.08,.04),1e-8);n,_=u.advance(p,count=7)
    assert n==7
    saved={(a,m):v.copy() for a,f in u.values.items() for m,v in f.items()}
    with pytest.raises(RuntimeError,match='Unfinished'):u.finish(p)
    u.advance(p);y,meta=u.finish(p)
    expected,_=Taylor(Context(5,1),query=True).action((.12,-.08,.04),1e-8)
    assert spectral(y-expected)<1e-13 and meta['relative_bound']<=1e-8
    assert all(np.array_equal(v,u.values[a][m]) for (a,m),v in saved.items())


def test_cost_can_choose_either_route_without_affecting_accuracy():
    answer=[]
    for costs,route in [(model(1e-5,100),'sparse_update'),(model(100,1e-5),'cost_to_direct')]:
        d=Dispatcher('policy',costs,5,1);y,m=d.step(event(.04))
        assert m['route']==route and m['relative_bound']<=1e-8
        answer.append(y)
    assert spectral(answer[0]-answer[1])/spectral(answer[1])<1e-8


def test_precision_can_invalidate_reuse_of_tiny_change():
    d=Dispatcher('update',nr=5,per_seed=1)
    d.step(event());_,loose=d.step(event(1e-7,tol=1e-6));_,tight=d.step(event(1e-7))
    assert loose['route']=='perturbation_reuse'
    assert tight['route']=='sparse_update' and tight['relative_bound']<=1e-8


def test_observation_change_transports_field_error_without_resolving_pde():
    d=Dispatcher('direct',nr=5,per_seed=1)
    d.step(event(.04));before=d.c.bank.factor_solves
    y,m=d.step(event(.04,'wide'))
    assert m['route']=='field_transport'
    assert m['direct_s']==0
    c=Context(5,1);bind_observation(c,'wide');ref,_=Direct(c).action((.12,-.08,.04),1e-11)
    assert spectral(y-ref)/spectral(ref)<1e-8
    # Observation dual rows may require solves; the material field stays intact.
    assert d.field_theta==(.12,-.08,.04) and d.c.bank.factor_solves>=before


def test_order_guard_falls_back_to_certified_solve_not_a_physical_failure():
    d=Dispatcher('update',nr=5,per_seed=1,settings={'max_order':1})
    _,m=d.step(event(.12))
    assert m['route']=='guard_to_direct' and m['fallback_reason']=='Order limit exceeded'
    assert m['relative_bound']<=1e-8


def test_trial_budget_exhaustion_never_returns_partial_taylor_answer():
    d=Dispatcher('policy',model(.001,.002),5,1,
                 settings={'gray_ratio':0.,'trial_budget_fraction':0.})
    y,m=d.step(event(.04))
    assert m['fallback_reason']=='trial' and m['trial_nodes']==0
    assert d.field_theta==(.12,-.08,.04) and m['relative_bound']<=1e-8
    expected,_=Direct(Context(5,1)).action((.12,-.08,.04),1e-11)
    assert spectral(y-expected)/spectral(expected)<1e-8


def test_fallback_keeps_valid_completed_coefficient_nodes_for_later_queries():
    d=Dispatcher('policy',model(1e-5,100),5,1)
    d.step(event(.04))
    saved={(a,m):v.copy() for a,f in d.taylor.values.items() for m,v in f.items()}
    d.model=model(100,1e-5);_,m=d.step(event(.12))
    assert m['route']=='cost_to_direct'
    assert all(np.array_equal(v,d.taylor.values[a][mode]) for (a,mode),v in saved.items())
    d.model=model(1e-5,100);_,m=d.step(event(.03))
    assert m['route']=='recombine' and m['new_nodes']==0


def test_invalid_coercivity_is_not_hidden_by_cost_or_cache():
    d=Dispatcher('direct',nr=5,per_seed=1)
    with pytest.raises(ValueError,match='Contraction'):d.step(event(5))


def test_unsupported_physical_change_cannot_silently_reuse_the_old_family():
    d=Dispatcher('direct',nr=5,per_seed=1)
    d.step(event())
    with pytest.raises(ValueError,match='Unsupported change'):
        d.step(dict(event(),geometry='different_domain'))
