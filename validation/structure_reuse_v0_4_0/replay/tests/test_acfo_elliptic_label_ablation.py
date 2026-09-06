import numpy as np
from scripts.acfo_elliptic_label_ablation import make_graph,forward_support,predecessors,Runner,numerical_digest
from scripts.acfo_elliptic_new_coupling import indices,Direct
from scripts.acfo_elliptic_update_policy import spectral

def test_generic_and_integer_oracle_graphs_agree_and_are_closed():
    for p,q in ((3,0),(5,1),(7,2),(5,5)):
        aa=[a for a in indices(p,(0,1,2)) if a[2]<=q]
        fwd=forward_support(aa)
        wanted,_=make_graph('generic_query',aa,{(0,0),(5,-2),(1,0),(-3,2)})
        other,_=make_graph('algebra_query',aa,{(0,0),(5,-2),(1,0),(-3,2)})
        assert wanted==other
        for a,ms in wanted.items():
            for m in ms:
                assert all(n in wanted[b] for b,n in predecessors(a,m) if n in fwd[b])

def test_full_observation_has_no_pruning_advantage():
    aa=[a for a in indices(5,(0,1,2)) if a[2]<=1]
    full=forward_support(aa);obs=set().union(*full.values())
    for arm in ('generic_query','algebra_query'):
        assert make_graph(arm,aa,obs)[0]==full

def test_same_warm_coefficients_and_requested_responses_match_direct():
    for gamma in (1e-5,.03):
        answers=[];warm=[]
        e=dict(theta=[.12,-.08,0.],tolerance=1e-8,observation='narrow')
        for arm in ('forward','generic_query','algebra_query'):
            r=Runner(arm,8,2);r.step(e,arm='generic_query');warm.append(numerical_digest(r.engine))
            y,m=r.step(dict(e,theta=[.12,-.08,gamma]));answers.append(y)
            ref,_=Direct(r.c).action((.12,-.08,gamma),1e-12)
            assert spectral(y-ref)/spectral(ref)<1e-8 and m['relative_bound']<=1e-8
        assert len(set(warm))==1
        for y in answers[1:]: np.testing.assert_allclose(y,answers[0],rtol=1e-12,atol=1e-14)
