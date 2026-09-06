from math import comb,isclose
import numpy as np
from scripts.acfo_elliptic_weak_coupling import majorant_tail,select_cap,CappedInvestigator,WeakCouplingDispatcher
from scripts.acfo_elliptic_new_coupling import forward,SHIFTS,rho
from scripts.acfo_elliptic_update_policy import Context,Direct,IncrementalTaylor,Dispatcher,bind_observation,spectral
from tests.test_acfo_elliptic_structured_policy import model,event


def test_tail_matches_independent_positive_word_count_and_limits():
    theta=(.4,-.3,.5);base=(.6*.4+.4*.3)/1.2;weak=.3*.5/1.2
    for p in (0,3,7):
        for q in range(p+1):
            explicit=sum(comb(d,c)*weak**c*base**(d-c) for d in range(100)
                         for c in range(d+1) if d>p or c>q)
            assert isclose(majorant_tail(theta,p,q),explicit,rel_tol=1e-13,abs_tol=1e-16)
        assert majorant_tail(theta,p,p)==rho(theta)**(p+1)/(1-rho(theta))
    assert majorant_tail((.12,-.08,0),7,0)==majorant_tail((.12,-.08,0),7,7)


def test_capped_graph_matches_independent_enumerated_ancestor_closure():
    d=WeakCouplingDispatcher('cap_update',model(),5,1);d.c=Context(5,1)
    d.taylor=IncrementalTaylor(d.c,d.settings);d.direct=Direct(d.c)
    active=(0,1,2);p=7;q=1;modes={(0,0),(5,-2),(1,0),(6,-2)}
    fwd={a:v for a,v in forward(p,active).items() if a[2]<=q}
    wanted={a:set() for a in fwd}
    stack=[(a,m) for a,v in fwd.items() for m in modes if m in v]
    while stack:
        a,m=stack.pop()
        if m in wanted[a]:continue
        wanted[a].add(m)
        for l in active:
            if not a[l]:continue
            b=list(a);b[l]-=1;b=tuple(b)
            for sign in (-1,1):
                n=tuple(x+sign*s for x,s in zip(m,SHIFTS[l]))
                if n in fwd[b]:stack.append((b,n))
    i=CappedInvestigator(q,{})
    prepared=i.describe(d,(.12,-.08,1e-5),p,modes,1e-8,1e-10)
    assert prepared['wanted']==wanted and not d.taylor.plans


def test_weak_correction_is_necessary_and_one_gamma_layer_is_sufficient():
    d=WeakCouplingDispatcher('cap_update',model(),8,2);ref=Dispatcher('direct',nr=8,per_seed=2)
    e0=event(0);y0,_=d.step(e0);d0={a:{m:v.copy() for m,v in f.items()} for a,f in d.taylor.values.items()}
    e=event(1e-5);y,m=d.step(e);r,_=ref.step(e)
    assert m['q']==1 and m['p']==7 and m['relative_bound']<=1e-8
    assert spectral(y-r)/spectral(r)<1e-8
    assert spectral(y0-r)/spectral(r)>1e-8
    for a,f in d0.items():
        for mode,v in f.items():assert np.array_equal(v,d.taylor.values[a][mode])
    assert all(a[2]<=1 for a in d.taylor.values)


def test_stricter_tolerance_and_larger_coupling_add_required_layers():
    d=WeakCouplingDispatcher('cap_update',model(),5,1)
    _,loose=d.step(event(1e-7,tol=1e-6))
    _,strict=d.step(event(1e-5,tol=1e-8))
    _,larger=d.step(event(.001))
    assert loose['q']==0 and strict['q']==1 and larger['q']==2
    ref=Dispatcher('direct',nr=5,per_seed=1);r,_=ref.step(event(.001))
    y,m=d.step(event(.001))
    assert m['route']=='exact_response' and spectral(y-r)/spectral(r)<1e-8


def test_sign_reversal_and_wider_observation_restore_ancestors():
    d=WeakCouplingDispatcher('cap_update',model(),5,1);ref=Dispatcher('direct',nr=5,per_seed=1)
    for e in [event(0),event(1e-5),event(-1e-5,'wide'),event(.001,'wide')]:
        y,m=d.step(e);r,_=ref.step(e)
        assert spectral(y-r)/spectral(r)<1e-8 and m['relative_bound']<=1e-8


def test_both_cost_and_feasibility_fallback_use_full_certificate():
    d=WeakCouplingDispatcher('cap_policy',model(100,1e-6),5,1)
    _,m=d.step(event(1e-5));assert m['route']=='cost_to_direct' and m['relative_bound']<=1e-8
    d=WeakCouplingDispatcher('cap_policy',model(),5,1)
    _,m=d.step(event(.8));assert m['route']=='guard_to_direct' and m['relative_bound']<=1e-8


def test_select_uses_physical_norm_and_obeys_common_degree_guard():
    c=Context(5,1)
    for obs in ('narrow','wide'):
        bind_observation(c,obs)
        for g in (1e-5,.001,.04,.12,.28):
            p,q,bound,absolute=select_cap(c,(.12,-.08,g),1e-8,10)
            assert 0<=q<=p<=10 and bound<=1e-8 and absolute>0
