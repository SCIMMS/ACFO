import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from odt_weight_coupling_adapter import WeightGeometry,profile


def test_weighted_gram_fourier_sign_wrap_and_phase_convention():
    n=12;freq=np.array([0,1,5,7,11]);phase=np.exp(1j*np.arange(5)*.3)
    psi=torch.tensor(np.exp(2j*np.pi*np.arange(n)[:,None]*freq/n)*phase,dtype=torch.complex128)
    geo=WeightGeometry(psi);torch.manual_seed(19);x=torch.randn((17,5),dtype=torch.complex128)
    for name in ('harmonic1','harmonic3','view_drop'):
        w=profile(name,n,.3);ref=geo.apply('explicit',geo.prepare('explicit',w),x)
        for method in ('dense_gram','fft_dense','fourier_sparse','view_update','fft_view'):
            actual=geo.apply(method,geo.prepare(method,w),x)
            torch.testing.assert_close(actual,ref,rtol=1e-12,atol=1e-12)
        gram=geo.prepare('dense_gram',w)['g']
        torch.testing.assert_close(gram,gram.conj().T,rtol=1e-12,atol=1e-12)
        assert torch.linalg.eigvalsh(gram).min()>0


def test_one_view_change_is_rank_one_even_with_broad_fourier_support():
    n=120;l=np.arange(-32,33)
    psi=torch.tensor(np.exp(-2j*np.pi*np.arange(n)[:,None]*l/n),dtype=torch.complex128)
    geo=WeightGeometry(psi);w=profile('view_drop',n)
    delta=geo.prepare('dense_gram',w)['g']-torch.diag(geo.diag)
    s=torch.linalg.svdvals(delta)
    assert s[1]<1e-11*s[0]
    assert len(geo.prepare('fourier_sparse',w)['active'])==120
