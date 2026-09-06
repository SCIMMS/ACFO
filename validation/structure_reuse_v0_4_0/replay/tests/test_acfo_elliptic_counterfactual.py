import pickle
import numpy as np
from scripts.acfo_elliptic_update_policy import CostModel,spectral
from scripts.acfo_elliptic_counterfactual import initialized,checkpoint,branch,profile,state_counts


def model():
    return CostModel({'update':{'coefficients':[0,.08,.02,.1,.01]},
                      'direct':{'coefficients':[.02,.3,.2,1.,0,0]}})


def event(g=.2,obs='narrow'):
    return dict(theta=[.12,-.08,g],observation=obs,tolerance=1e-8)


def test_snapshot_preserves_aliases_but_branch_mutation_is_isolated():
    d=initialized(model(),5,1);blob=pickle.dumps(d,protocol=5)
    a=pickle.loads(blob);b=pickle.loads(blob)
    assert a.taylor.c is a.c and a.direct.c is a.c
    before=state_counts(b);a.step(event(.04))
    assert state_counts(b)==before and state_counts(d)==before
    assert a.c.bank.factors is not b.c.bank.factors
    for m,u in b.c.u0.items():assert np.array_equal(u,d.c.u0[m])


def test_same_state_branches_and_both_common_suffixes_preserve_accuracy():
    spec=dict(event=event(.04),suffix=[event(.06),event(.06,'wide')])
    d,_=checkpoint(model(),spec,5,1);blob=pickle.dumps(d,protocol=5)
    for suffix in ('update','direct'):
        ua,um=branch(pickle.loads(blob),spec,'update',suffix)
        da,dm=branch(pickle.loads(blob),spec,'direct',suffix)
        for u,v in zip(ua,da):assert spectral(u-v)/spectral(v)<1e-8
        assert all(m['relative_bound']<=1e-8 for m in um['events']+dm['events'])


def test_bank_prewarm_moves_cost_without_preparing_any_coefficients():
    d=initialized(model(),5,1);blob=pickle.dumps(d,protocol=5)
    stock,sm=profile(pickle.loads(blob),event(.04),False)
    warm,wm=profile(pickle.loads(blob),event(.04),True)
    assert spectral(stock-warm)<1e-13
    assert wm['prewarm']['before']['coefficient_nodes']==wm['prewarm']['after']['coefficient_nodes']
    assert wm['prewarm']['seconds']>0
    assert sum(b['new_factors']+b['new_edges'] for b in wm['blocks'])==0
    assert sum(b['new_factors']+b['new_edges'] for b in sm['blocks'])>0


def test_after_trial_checkpoint_is_partial_valid_work_not_a_response():
    spec=dict(event=event(.04),trial_nodes=7,suffix=[])
    d,m=checkpoint(model(),spec,5,1)
    assert m['trial']['nodes']==7 and len(d.responses)==0
    assert state_counts(d)['coefficient_nodes']==9
    before={(a,k):v.copy() for a,f in d.taylor.values.items() for k,v in f.items()}
    branch(d,spec,'update','direct')
    assert all(np.array_equal(v,d.taylor.values[a][k]) for (a,k),v in before.items())


def test_existing_field_transport_is_a_shared_control_not_forced_recomputation():
    spec=dict(prefix=[dict(arm='direct',event=event(.04))],event=event(.04,'wide'),suffix=[])
    d,_=checkpoint(model(),spec,5,1);blob=pickle.dumps(d,protocol=5)
    for arm in ('update','direct'):
        _,m=branch(pickle.loads(blob),spec,arm,'direct')
        assert m['events'][0]['route']=='field_transport'
