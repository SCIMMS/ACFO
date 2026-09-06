import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from slab_update_adapter import SlabFactors,quadrature
from waxs_cake import ScalarUniaxialSlabSommerfeldOperator,ScalarSubstrateAnnularBornOperator


def test_adapter_matches_existing_production_forward_and_both_derivatives():
    q,w=quadrature(16);rs=np.array([.16,.31,.48]);rt=np.array([.12,.29,.46])
    hs=np.array([.22,.28,.34]);ht=np.array([.31,.39,.47]);sw=np.array([.045,.061,.054])
    fac=SlabFactors(rs,rt,hs,ht,sw,q,w)
    rng=np.random.default_rng(9);f=rng.normal(size=(3,192))+1j*rng.normal(size=(3,192))
    for d,z in ((.31,0),(.43,-.02)):
        slab=ScalarUniaxialSlabSommerfeldOperator(q,w,4,1,2.4+.025j,-1.2+.04j,d,.05,'extraordinary',torch=torch)
        op=ScalarSubstrateAnnularBornOperator(slab,rs,rt,hs,ht+z,sw,192,support_q_max=50)
        ref=np.stack([op.forward(f).numpy(),op.substrate_thickness_jvp(f).numpy(),op.geometry_jvp(f,np.zeros(3),np.zeros(3),np.ones(3),np.zeros(3)).numpy()])
        for name in ('modal_cached','factor_separate','factor_shared'):
            got=fac.apply(name,fac.prepare(name,d,z),f)
            np.testing.assert_allclose(got,ref,rtol=2e-11,atol=2e-14)


def test_thickness_and_height_derivatives_against_five_point_difference():
    q,w=quadrature(12);r=np.linspace(.1,.48,3);z=np.linspace(.22,.34,3)
    fac=SlabFactors(r,r,z,z,np.ones(3),q,w);rng=np.random.default_rng(41)
    f=rng.normal(size=(3,192))+1j*rng.normal(size=(3,192));d=.37;h=.03;step=2e-5
    exact=fac.apply('factor_shared',fac.prepare('factor_shared',d,h),f)
    for axis in (0,1):
        vals=[]
        for delta in (-2,-1,1,2):
            p=[d,h];p[axis]+=delta*step
            vals.append(fac.apply('factor_shared',fac.prepare('factor_shared',*p),f)[0])
        fd=(vals[0]-8*vals[1]+8*vals[2]-vals[3])/(12*step)
        assert np.linalg.norm(fd-exact[axis+1])/np.linalg.norm(exact[axis+1])<1e-8
