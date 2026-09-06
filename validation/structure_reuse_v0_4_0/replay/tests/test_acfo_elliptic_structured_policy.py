import numpy as np
import pytest
from scripts.acfo_elliptic_update_policy import CostModel,Dispatcher,spectral
from scripts.acfo_elliptic_structured_policy import StructuredDispatcher


def model(u=.001,d=1.):
    return CostModel({'update':{'coefficients':[u,0,0,0,0]},'direct':{'coefficients':[d,0,0,0,0,0]}})


def event(g=.04,obs='narrow',tol=1e-8):
    return dict(theta=[.12,-.08,g],observation=obs,tolerance=tol)


def test_union_preparation_and_each_physical_norm_match_direct():
    events=[event(.12),event(.12,'radial'),event(.12,'wide')]
    u=StructuredDispatcher('batch_update',model(),5,1)
    d=Dispatcher('direct',nr=5,per_seed=1)
    for i,e in enumerate(events):
        y,m=u.step(e,events[i+1:]);ref,_=d.step(e)
        assert spectral(y-ref)/spectral(ref)<1e-8 and m['relative_bound']<=e['tolerance']
        if i==0:assert m['prefetched_responses']==3
        else:assert m['route']=='exact_response'


def test_direct_batch_transports_evidence_and_caches_responses():
    events=[event(.04),event(.04,'wide'),event(.04,'radial')]
    d=StructuredDispatcher('batch_direct',model(),5,1)
    for i,e in enumerate(events):
        _,m=d.step(e,events[i+1:])
        assert m['relative_bound']<=1e-8
        if i:assert m['route']=='exact_response' and m['direct_s']==0


def test_batch_cannot_hide_operator_changes_and_stream_cannot_read_future():
    d=StructuredDispatcher('scheduled',model(),5,1)
    with pytest.raises(ValueError,match='operator fixed'):d.step(event(),[event(.12)])
    d=StructuredDispatcher('remaining',model(),5,1)
    with pytest.raises(ValueError,match='future'):d.step(event(),[event(.04,'wide')])


def test_trial_is_sunk_and_completed_nodes_are_retained():
    d=StructuredDispatcher('remaining',model(8e-5,1e-4),5,1,settings={'trial_budget_fraction':1000.})
    y,m=d.step(event(.12))
    assert m['trial_nodes']==64 and m['trial_s']>m['predicted_remaining_direct_s']
    assert m['route']=='trial_update' and m['relative_bound']<=1e-8
    ref,_=Dispatcher('direct',nr=5,per_seed=1).step(event(.12))
    assert spectral(y-ref)/spectral(ref)<1e-8


def test_stricter_later_request_is_not_prefetched_with_weaker_certificate():
    e=event(.04,tol=1e-6);later=event(.04,tol=1e-8)
    d=StructuredDispatcher('batch_update',model(),5,1)
    d.step(e,[later]);_,m=d.step(later)
    assert m['route']=='exact_response' and m['relative_bound']<=1e-8


def test_cost_and_feasibility_fallback_are_separate():
    d=StructuredDispatcher('remaining',model(100,1e-5),5,1)
    _,m=d.step(event());assert m['route']=='cost_to_direct'
    d=StructuredDispatcher('remaining',model(),5,1,settings={'max_order':1})
    _,m=d.step(event());assert m['route']=='guard_to_direct' and m['relative_bound']<=1e-8


def test_failed_direct_cannot_leave_old_field_evidence_attached(monkeypatch):
    d=StructuredDispatcher('batch_direct',model(),5,1)
    d.step(event())
    def fail(*args,**kwargs):
        d.direct.field={};raise RuntimeError('injected solver failure')
    monkeypatch.setattr(d.direct,'action',fail)
    with pytest.raises(RuntimeError,match='injected'):d.step(event(.12))
    assert d.field_theta is None and d.field_energy is None
