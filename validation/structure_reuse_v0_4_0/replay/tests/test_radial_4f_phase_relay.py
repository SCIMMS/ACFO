import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from experiment_radial_4f_phase_relay import phase_vectors,update,apply,certify


def test_phase_derivative_and_unit_norm():
    x=np.linspace(0,1,70);a,b=4.,-2.;h=1e-4
    p,dp=phase_vectors(x,a,b)
    actual=(-phase_vectors(x,a+2*h,b)[0]+8*phase_vectors(x,a+h,b)[0]-8*phase_vectors(x,a-h,b)[0]+phase_vectors(x,a-2*h,b)[0])/(12*h)
    np.testing.assert_allclose(actual,dp,rtol=1e-10,atol=1e-11)
    np.testing.assert_allclose(abs(p),1,atol=3e-16)


def test_nonorthogonal_complex_factor_composite_matches_direct():
    # Production factors are real before complex casting; transpose is used.
    rng=np.random.default_rng(58);a=rng.normal(size=(11,4)).astype(complex);b=rng.normal(size=(11,4)).astype(complex)
    k=a@b.T;fac=dict(a=a,b=b);x=np.linspace(0,1,11)
    f=rng.normal(size=(11,3))+1j*rng.normal(size=(11,3))
    truth=apply('dense_factor',None,k,update('dense_factor',None,k,x,3,2),f)
    for name in ('jacobi_value','jacobi_core'):
        result=apply(name,fac,k,update(name,fac,k,x,3,2),f)
        np.testing.assert_allclose(result,truth,rtol=1e-12,atol=1e-10)


def test_two_sided_bound_for_nonorthogonal_factors_and_derivative():
    rng=np.random.default_rng(61)
    a=rng.normal(size=(13,4));b=rng.normal(size=(13,4))
    khat=a@b.T;k=khat+1e-4*rng.normal(size=(13,13))
    fac=dict(a=a,b=b,ra=np.linalg.qr(a,mode='reduced')[1],
             rb=np.linalg.qr(b,mode='reduced')[1],delta=np.linalg.norm(k-khat,2),
             hn=np.linalg.norm(khat,2))
    x=np.linspace(0,1,13);p,dp=phase_vectors(x,10,6)
    results=[]
    for name in ('jacobi_value','jacobi_core'):
        plan=update(name,fac,k,x,10,6)
        bound,relative_bound=certify(fac,np.linalg.norm(k,2),plan,x,10,6)
        results.append((bound,relative_bound))
        for mask in (p,dp):
            truth=k@(mask[:,None]*k);estimate=khat@(mask[:,None]*khat)
            assert np.linalg.norm(truth-estimate,2)<=bound
        truth=k@(p[:,None]*k)
        assert np.linalg.norm(truth-khat@(p[:,None]*khat),2)/np.linalg.norm(truth,2)<=relative_bound
    np.testing.assert_allclose(results[0],results[1],rtol=1e-12)
