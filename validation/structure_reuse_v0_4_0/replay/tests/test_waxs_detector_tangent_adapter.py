import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from waxs_detector_tangent_adapter import geometry,form_factor,carbon_coefficients,evaluate
from waxs_cake import TorchDifferentiableAxisymmetricOperator,torch_xray_f0_form_factors


def test_full_chain_matches_production_autograd_and_finite_difference():
    rho=torch.linspace(.03,1.5,12,dtype=torch.float64);p=torch.tensor([1.,.02],dtype=torch.float64)
    qp,qz,*_=geometry(rho,p)
    op=TorchDifferentiableAxisymmetricOperator([0.,.3,.7],[-.4,0,.5],qp,qz,128,harmonic_padding=24,q_block_size=5,support_q_perp=[8.])
    torch.manual_seed(12);f=torch.randn(op.object_shape,dtype=torch.complex128)
    target=torch.rand(op.data_shape,dtype=torch.float64)*1000;weight=torch.ones_like(target);norm=(target**2).sum()
    coeff=carbon_coefficients('cpu')
    ref=evaluate('production_backward',op,f,rho,p,target,weight,norm,coeff)
    for name in ('coordinate_fields','tangent_full','tangent_stream'):
        result=evaluate(name,op,f,rho,p,target,weight,norm,coeff)
        torch.testing.assert_close(result,ref,rtol=2e-11,atol=2e-10)
    step=1e-5
    for axis in (0,1):
        terms=[]
        for j in (-2,-1,1,2):
            pj=p.clone();pj[axis]+=j*step
            terms.append(evaluate('coordinate_fields',op,f,rho,pj,target,weight,norm,coeff)[0])
        fd=(terms[0]-8*terms[1]+8*terms[2]-terms[3])/(12*step)
        torch.testing.assert_close(fd,ref[1][axis],rtol=1e-7,atol=1e-8)


def test_carbon_derivative_matches_existing_torch_formula():
    q=torch.linspace(.1,14,17,dtype=torch.float64,requires_grad=True)
    ref=torch_xray_f0_form_factors(['C'],q,torch=torch)['C']
    grad=torch.autograd.grad(ref.sum(),q)[0]
    f,df=form_factor(q,carbon_coefficients('cpu'))
    torch.testing.assert_close(f,ref,rtol=1e-13,atol=1e-13)
    torch.testing.assert_close(df,grad,rtol=1e-13,atol=1e-13)
