from itertools import combinations
import pytest
from scripts.acfo_elliptic_progressive_policy import reachable,GraphInvestigator,ProgressiveDispatcher
from scripts.acfo_elliptic_structured_policy import describe,StructuredDispatcher
from scripts.acfo_elliptic_new_coupling import forward,plan,Context,Direct
from scripts.acfo_elliptic_update_policy import IncrementalTaylor,Dispatcher,bind_observation,spectral
from tests.test_acfo_elliptic_structured_policy import model,event


def ready(mode='batch_update',cost=None):
    d=ProgressiveDispatcher(mode,cost or model(),5,1)
    d.c=Context(5,1);d.taylor=IncrementalTaylor(d.c,d.settings);d.direct=Direct(d.c)
    return d


def test_integer_oracle_matches_forward_including_negative_parity_and_all_active_sets():
    for n in range(4):
        for active in combinations(range(3),n):
            for a,nodes in forward(6,active).items():
                # The box covers and surrounds every reachable point at degree <= 6.
                actual={(m,k) for m in range(-20,26) for k in range(-15,14) if reachable(a,(m,k))}
                assert actual==nodes,(active,a)


@pytest.mark.parametrize('active',[(0,),(1,2),(0,1,2)])
def test_complete_query_graph_and_current_cache_features_match_frozen_planner(active):
    d=ready();theta=tuple(.04 if l in active else 0 for l in range(3))
    modes={(0,0),(5,-2),(3,1),(6,-2),(-7,4)}
    new=d.investigator.describe(d,theta,6,modes,1e-9,1e-10)
    old=describe(d,theta,6,modes,1e-9,1e-10)
    assert new['wanted']==plan(6,active,modes,True)
    assert new['missing']==old['missing'] and new['features']==old['features']
    d.taylor.advance(new,count=37)
    new=d.investigator.describe(d,theta,6,modes,1e-9,1e-10)
    old=describe(d,theta,6,modes,1e-9,1e-10)
    assert new['missing']==old['missing'] and new['features']==old['features']


def test_interrupted_graph_resumes_and_cannot_be_numerically_prepared():
    d=ready(cost=model(1.,1.));modes={(0,0),(5,-2),(3,1),(6,-2)}
    q=d.investigator.describe(d,(.12,-.08,.28),9,modes,1e-9,1e-10,ceiling=.1)
    assert q is None and d.investigator.last['graph_screened']
    assert not d.investigator.last['graph_complete']
    assert sum(map(len,d.taylor.values.values()))==2
    q=d.investigator.describe(d,(.12,-.08,.28),9,modes,1e-9,1e-10)
    assert q['wanted']==plan(9,(0,1,2),modes,True)
    assert d.investigator.last['graph_cache_hit']


def test_screen_is_monotone_model_bound_not_accuracy_certificate():
    d=ready('scheduled',model(100.,1e-5))
    _,m=d.step(event())
    assert m['graph_screened'] and m['route']=='cost_to_direct'
    assert m['relative_bound']<=1e-8
    assert m['model_lower_update_s']>m['model_screen_ceiling_s']
    bad=model();bad.data['update']['coefficients'][1]=-1
    d=ready(cost=bad)
    with pytest.raises(ValueError,match='nonnegative'):
        d.investigator.describe(d,(.12,-.08,.04),4,{(0,0)},1e-8,1e-10,ceiling=1.)


def test_materialized_graph_reuses_metadata_without_missing_node_scan():
    d=ready();modes={(0,0),(5,-2)}
    q=d.investigator.describe(d,(.1,0,0),4,modes,1e-8,1e-10)
    d.taylor.advance(q)
    d.investigator.describe(d,(.1,0,0),4,modes,1e-8,1e-10)
    q=d.investigator.describe(d,(.11,0,0),4,modes,1e-8,1e-10)
    assert not q['missing'] and d.investigator.last['graph_nodes_scanned']==0


def test_batch_and_trial_match_independent_direct_and_preserve_information_rules():
    events=[event(.12),event(.12,'radial'),event(.12,'wide')]
    d=ProgressiveDispatcher('scheduled',model(8e-5,1e-4),5,1,
                            settings={'trial_budget_fraction':1000.})
    ref=Dispatcher('direct',nr=5,per_seed=1)
    for i,e in enumerate(events):
        y,m=d.step(e,events[i+1:]);r,_=ref.step(e)
        assert spectral(y-r)/spectral(r)<1e-8 and m['relative_bound']<=1e-8
        if i==0:assert m['trial_nodes']==64 and m['route']=='trial_update'
        else:assert m['route']=='exact_response'
    with pytest.raises(ValueError,match='operator fixed'):d.step(event(),[event(.12)])
    stream=ProgressiveDispatcher('remaining',model(),5,1)
    with pytest.raises(ValueError,match='future'):stream.step(event(),[event(.04,'wide')])


def test_node_budget_still_falls_back_to_certified_direct():
    d=ProgressiveDispatcher('batch_update',model(),5,1,settings={'max_nodes':100})
    _,m=d.step(event())
    assert m['route']=='guard_to_direct' and m['relative_bound']<=1e-8


def test_maximum_production_order_wide_graph_and_model_lower_bound():
    d=ready();bind_observation(d.c,'wide')
    d.model.data['update']['coefficients']=[0.,.08,.015,.1,.015]
    modes={m for m,_ in d.c.obs};theta=(.12,-.08,.28)
    old=describe(d,theta,10,modes,1e-9,1e-10)
    full_cost=d.model.predict('update',old['features'])
    assert d.investigator.describe(d,theta,10,modes,1e-9,1e-10,ceiling=full_cost*.6) is None
    assert full_cost*.6<d.investigator.last['model_lower_update_s']<=full_cost
    q=d.investigator.describe(d,theta,10,modes,1e-9,1e-10)
    assert q['wanted']==old['wanted'] and q['missing']==old['missing']
    assert q['features']==old['features']
