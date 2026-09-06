"""Independent checks of new mathematical and matrix-free adapters."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pytest
from scipy import special
from waxs_cake.finite_hankel import gauss_unit_interval
from waxs_cake.portable_cpswf import RadialBinding, ContinuousRadialBasis
from acfo_research_adapters import (BlockedHankel, randomized_prepare,
    zernike_samples,zernike_images,ball_coefficients,sonine_image)


def test_blocked_complex_adjoint_and_randomized_spectral_residual():
    x,w=gauss_unit_interval(80);y,v=gauss_unit_interval(91)
    b=RadialBinding.make(x,y,w,v,c=25);op=BlockedHankel(b,4,25,17)
    rng=np.random.default_rng(23)
    rhs=rng.normal(size=(80,3))+1j*rng.normal(size=(80,3))
    dual=rng.normal(size=(91,3))+1j*rng.normal(size=(91,3))
    exact=b.matrix(4,25)
    np.testing.assert_allclose(op.apply(rhs),exact@rhs,atol=2e-14)
    np.testing.assert_allclose(op.apply(dual,True),exact.T@dual,atol=2e-14)
    action,_=randomized_prepare(op,1e-8,41)
    assert np.linalg.norm(exact-action.reduced@action.basis.conj().T,2)<1e-8


@pytest.mark.parametrize('m,alpha',[(0,0),(4,0),(0,.5),(4,1.5)])
def test_zernike_image_independent_weighted_quadrature(m,alpha):
    z,w=special.roots_jacobi(180,alpha,m/2)
    u=(z+1)/2;r=np.sqrt(u);w=w/2**(alpha+m/2+2)
    v=np.array([0.,.01,2.,20.,45.])
    pol=np.column_stack([special.eval_jacobi(n,alpha,m,2*u-1) for n in range(8)])
    ref=(special.jv(m,v[:,None]*r)*w)@pol
    np.testing.assert_allclose(zernike_images(m,alpha,8,v),ref,atol=2e-13,rtol=1e-10)


def test_weighted_ball_alpha_zero_matches_existing_cpswf():
    x,_=gauss_unit_interval(120)
    coef=ball_coefficients(4,0,25,48)
    got=np.sqrt(x[:,None])*zernike_samples(4,0,48,x)@coef[:,:12]
    ref=ContinuousRadialBasis.prepare(4,25,48,12).sample(x)
    signs=np.sign(np.sum(got*ref,axis=0))
    np.testing.assert_allclose(got*signs,ref,atol=3e-12,rtol=1e-10)


def test_sonine_small_argument_and_domain():
    z=np.linspace(0,8,71)
    got,_=sonine_image(0,4,z,100)
    np.testing.assert_allclose(got,special.jv(4,z),atol=2e-12)
    with pytest.raises(ValueError):sonine_image(0,0,z,100)
