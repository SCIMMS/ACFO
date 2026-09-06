"""Validation-only geometry loss actions on the existing cylindrical WAXS map.

Detector z(rho)=D+kappa*rho^2, with D dimensionless relative to fixed D0.
Both parameters share the ringwise detector-height tangent. Intensities are
point samples, not area-integrated counts; detector loss weights are fixed.
"""
import torch
from periodictable import cromermann


def carbon_coefficients(device):
    cm=cromermann.getCMformula('C')
    return tuple(torch.as_tensor(x,dtype=torch.float64,device=device) for x in (cm.a,cm.b,cm.c))


def geometry(rho,params):
    z=params[0]+params[1]*rho**2
    length=torch.sqrt(rho**2+z**2)
    qp=8*rho/length;qz=8*(z/length-1)
    dqp=-8*rho*z/length**3;dqz=8*rho**2/length**3
    q=torch.sqrt(qp**2+qz**2)
    dq=(qp*dqp+qz*dqz)/q
    return qp,qz,dqp,dqz,q,dq


def form_factor(q,coeff):
    a,b,c=coeff;s=q/(40*torch.pi)
    e=torch.exp(-b[:,None]*s[None,:]**2)
    ff=torch.sum(a[:,None]*e,dim=0)+c
    df=torch.sum(-2*a[:,None]*b[:,None]*s[None,:]*e,dim=0)/(40*torch.pi)
    return ff,df


def blocks(op,values,qp,qz):
    """Same Miller and contraction primitives as production; each block once."""
    bymode=op._selected_object_modes(values).permute(2,1,0).contiguous()
    zp=op._z_phase(qz);dzp=1j*op.z_centers[None,:]*zp
    radial=op.r_centers[None,None,:]
    for start in range(0,op.n_q,op.q_block_size):
        stop=min(start+op.q_block_size,op.n_q)
        pos=op._positive_kernel(qp[start:stop],op.derivative_max_order)
        mask=op._mode_mask(start,stop)
        buffer=torch.empty((stop-start,op.object_shape[0],op.n_modes),dtype=op.complex_dtype,device=op.device)
        ax=op._axial_contraction(zp[start:stop],bymode)
        axz=op._axial_contraction(dzp[start:stop],bymode)
        kernel=op._fill_kernel_buffer(pos,op.mode_abs,mask,buffer)
        f=torch.sum(ax*kernel,dim=-1).T
        fz=torch.sum(axz*kernel,dim=-1).T
        ar=ax*radial
        kernel=op._fill_kernel_buffer(pos,op.left_neighbor_abs,mask,buffer)
        fp=torch.sum(ar*kernel,dim=-1).T
        kernel=op._fill_kernel_buffer(pos,op.right_neighbor_abs,mask,buffer)
        fp=0.5j*(fp+torch.sum(ar*kernel,dim=-1).T)
        yield start,stop,f,fp,fz


def synthesis(op,selected):
    arr=torch.zeros((selected.shape[0],op.n_phi),dtype=op.complex_dtype,device=op.device)
    arr.index_copy_(-1,op.mode_indices,selected)
    return torch.fft.ifft(arr,dim=-1)


def reduce_loss(amplitude,tangent,target,weight,normalizer,rho):
    residual=abs(amplitude)**2-target
    loss=torch.sum(weight*residual**2)/(2*normalizer)
    ring=torch.sum(2*weight*residual*torch.real(amplitude.conj()*tangent),dim=-1)/normalizer
    return loss,torch.stack((ring.sum(),torch.sum(ring*rho**2)))


def evaluate(method,op,values,rho,params,target,weight,normalizer,coeff):
    if method=='production_backward':
        p=params.detach().clone().requires_grad_(True)
        qp,qz,_,_,q,_=geometry(rho,p)
        ff,_=form_factor(q,coeff)
        f=op.autograd_forward(values,qp,qz)
        residual=abs(ff[:,None]*f)**2-target
        loss=torch.sum(weight*residual**2)/(2*normalizer)
        grad=torch.autograd.grad(loss,p)[0]
        return loss.detach(),grad.detach()
    with torch.no_grad():
        qp,qz,dqp,dqz,q,dq=geometry(rho,params)
        qp,qz=op._resolve_geometry(qp,qz)
        values=op._object_tensor(values)
        ff,df=form_factor(q,coeff)
        if method=='coordinate_fields':
            f,fp,fz=op.forward_with_geometry_derivatives(values,qp,qz)
            tangent=ff[:,None]*(dqp[:,None]*fp+dqz[:,None]*fz)+(df*dq)[:,None]*f
            return reduce_loss(ff[:,None]*f,tangent,target,weight,normalizer,rho)
        loss=torch.zeros((),device=op.device,dtype=torch.float64)
        grad=torch.zeros(2,device=op.device,dtype=torch.float64)
        if method=='tangent_full':
            sf=torch.empty((op.n_q,op.n_modes),device=op.device,dtype=op.complex_dtype)
            st=torch.empty_like(sf)
        elif method!='tangent_stream':raise ValueError(method)
        for a,b,f,fp,fz in blocks(op,values,qp,qz):
            # Detector parameters share one tangent direction in each q row.
            tangent=dqp[a:b,None]*fp+dqz[a:b,None]*fz
            if method=='tangent_full':
                sf[a:b]=f;st[a:b]=tangent
            else:
                realf=synthesis(op,f);realt=synthesis(op,tangent)
                amp=ff[a:b,None]*realf
                derivative=ff[a:b,None]*realt+(df[a:b]*dq[a:b])[:,None]*realf
                l,g=reduce_loss(amp,derivative,target[a:b],weight[a:b],normalizer,rho[a:b])
                loss+=l;grad+=g
        if method=='tangent_full':
            f=synthesis(op,sf);ft=synthesis(op,st)
            return reduce_loss(ff[:,None]*f,ff[:,None]*ft+(df*dq)[:,None]*f,target,weight,normalizer,rho)
        return loss,grad
