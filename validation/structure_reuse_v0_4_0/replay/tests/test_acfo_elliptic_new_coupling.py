import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import splu
from scripts.acfo_elliptic_new_coupling import (
    Context,ExtendedBank,Taylor,Direct,Runner,mode_ball,component,rho,field_certificate,
)
from scripts.acfo_elliptic_refresh import Context as OldContext, TaylorRefresh, spectral
from scripts.acfo_cylinder_coupling import CylinderSpace,stack_field,observation_matrix
from scripts.acfo_elliptic_disk import radial_rule,state_values


def reference(theta,nr=5):
    c=Context(nr,1,181)
    space=CylinderSpace(mode_ball(9,tuple(l for l,t in enumerate(theta) if t)),nr)
    a=sparse.block_diag([c.bank.base(m) for m in space.modes],format='csr')
    for l,t in enumerate(theta):
        if t:a+=t*component(c.bank,space,l)
    b=stack_field(space,c.sources,c.ncols);lu=splu(a.tocsc())
    return observation_matrix(space,c.obs)@(lu.solve(b.real)+1j*lu.solve(b.imag))


def test_new_edge_matches_independent_cylindrical_weak_gradient():
    c=Context(6,1);r,w=radial_rule(193)
    for source,target in [((0,0),(1,0)),((-2,3),(-1,3)),((4,-2),(3,-2))]:
        ms,ks=source;mt,kt=target
        sv=[state_values(ms,n,r) for n in range(6)];tv=[state_values(mt,n,r) for n in range(6)]
        s=np.column_stack([x[0] for x in sv]);sd=np.column_stack([x[1] for x in sv])
        t=np.column_stack([x[0] for x in tv]);td=np.column_stack([x[1] for x in tv])
        weight=.15*w*r
        expected=td.T@(weight[:,None]*sd)+t.T@((weight*(mt*ms/r**2+kt*ks))[:,None]*s)
        assert spectral(c.bank.edge(2,target,source)-expected)/spectral(expected)<1e-11


def test_actual_old_two_index_objects_embed_and_remain_unchanged():
    c=Context(5,1);oldc=OldContext(5,1)
    for key in ('obs','rootw','observation_dual','y0norm'):
        setattr(oldc,key,getattr(c,key))
    old=TaylorRefresh(oldc,query=True);old.action((.12,-.08),1e-8)
    before={(a,m):u.copy() for a,f in old.values.items() for m,u in f.items()}
    old_blocks={key:v.copy() for key,v in oldc.bank.edges.items()}
    c.bank=ExtendedBank.inherit(oldc.bank)
    new=Taylor(c,query=True,legacy=old.values)
    assert all(new.values[a+(0,)][m] is old.values[a][m] for a,m in before)
    y,meta=new.action((.12,-.08,.04),1e-8)
    assert meta['new_generator_nodes']>0
    assert all(np.array_equal(u,new.values[a+(0,)][m]) for (a,m),u in before.items())
    assert all(np.array_equal(v,c.bank.edges[k]) for k,v in old_blocks.items())
    assert spectral(y-reference((.12,-.08,.04)))/spectral(y)<1e-8


def test_full_and_query_match_after_extension_and_precision_change():
    full=Taylor(Context(5,1));query=Taylor(Context(5,1),query=True)
    for theta,tol in [((.12,-.08,0),1e-8),((.12,-.08,.04),1e-8),((.12,-.08,.12),1e-8)]:
        a,_=full.action(theta,tol);b,_=query.action(theta,tol)
        assert spectral(a-b)<1e-13
    assert spectral(b-reference((.12,-.08,.12)))/spectral(b)<1e-8


def test_new_coupling_changes_old_observations_as_well_as_new_channels():
    a=reference((.12,-.08,0));b=reference((.12,-.08,.12))
    assert spectral(a[10:])<1e-15
    assert spectral(b[10:])>1e-8
    assert spectral(a[:10]-b[:10])/spectral(a[:10])>1e-7


def test_tiny_change_can_reuse_response_but_tighter_precision_requires_new_certificate():
    r=Runner('query_reuse',5,1)
    r.step(dict(theta=[.12,-.08,0],tolerance=1e-8))
    _,loose=r.step(dict(theta=[.12,-.08,1e-7],tolerance=1e-6))
    y,tight=r.step(dict(theta=[.12,-.08,1e-7],tolerance=1e-8))
    assert loose['perturbation_reused'] and loose['new_nodes']==0
    assert not tight['exact_cache_hit']
    assert tight['relative_bound']<=1e-8
    assert spectral(y-reference((.12,-.08,1e-7)))/spectral(y)<1e-8


def test_direct_rechecks_residual_under_changed_operator():
    d=Direct(Context(5,1))
    d.action((.12,-.08,0),1e-8)
    y,meta=d.action((.12,-.08,.12),1e-8)
    assert not meta['field_reused'] and meta['iterations']>0
    assert meta['exterior_modes_checked']>0
    assert spectral(y-reference((.12,-.08,.12)))/spectral(y)<1e-8
    _,sparse_bound,_,_=d.certify((.12,-.08,.04))
    _,local_bound,_,_=field_certificate(d.c,d.field,(.12,-.08,.04))
    assert abs(sparse_bound-local_bound)<1e-12


def test_bad_contraction_is_explicit():
    with pytest.raises(ValueError,match='Contraction'):
        rho((0,0,5))
