import pytest
from scripts.acfo_elliptic_perturbation_sweep import event,forecast,boundaries,warm_fingerprint,bisect_last_true
from scripts.acfo_elliptic_weak_coupling import WeakCouplingDispatcher
from tests.test_acfo_elliptic_structured_policy import model


@pytest.mark.parametrize('tol',[1e-6,1e-8])
def test_prospective_boundaries_bracket_reuse_layers_and_guard(tol):
    d=WeakCouplingDispatcher('cap_update',model(),5,1);y,e=d.step(event(0,tol))
    b=boundaries(d.c,y,e,tol)
    assert 0<b['reuse']<b['one_layer']<b['preparation']<.8
    assert forecast(d.c,y,e,b['reuse']*.95,tol)['phase']=='perturbation_reuse'
    assert forecast(d.c,y,e,b['reuse']*1.05,tol)['phase']=='correction'
    assert forecast(d.c,y,e,b['one_layer']*.95,tol)['q']==1
    assert forecast(d.c,y,e,b['one_layer']*1.05,tol)['q']>1
    assert forecast(d.c,y,e,b['preparation']*1.05,tol)['phase']=='preparation_guard'


def test_fresh_warm_state_is_identical_and_prediction_does_not_mutate_it():
    ds=[WeakCouplingDispatcher('cap_update',model(),5,1) for _ in range(2)]
    states=[]
    for d in ds:
        y,e=d.step(event(0,1e-8));before=warm_fingerprint(d)
        for g in (1e-8,1e-5,.001,.28,.8):forecast(d.c,y,e,g,1e-8)
        assert warm_fingerprint(d)==before
        states.append(before)
    assert states[0]==states[1]


def test_forecast_matches_frozen_adapter_from_common_warm_start():
    for gamma in (0,1e-9,1e-5,.001,.8):
        d=WeakCouplingDispatcher('cap_update',model(),5,1)
        y,e=d.step(event(0,1e-8));pred=forecast(d.c,y,e,gamma,1e-8)
        _,actual=d.step(event(gamma,1e-8))
        if pred['phase']=='correction':assert (pred['p'],pred['q'])==(actual['p'],actual['q'])
        elif pred['phase']=='preparation_guard':assert actual['route']=='guard_to_direct'
        else:assert actual['route'] in ('exact_response','perturbation_reuse')


def test_boundary_requires_bracket():
    with pytest.raises(ValueError,match='bracketed'):bisect_last_true(lambda g:True)
