"""Validation-only illumination-weight updates in the existing ODT latent.

Row convention: latent @ Psi.T @ diag(w) @ conj(Psi). Rotational view
weights become Fourier-lag couplings; isolated view changes are low rank.
"""
import numpy as np
import torch
from benchmark_odt_fused_modal_normal import cone_reduced_h_l_u


class WeightGeometry:
    def __init__(self,psi):
        self.psi=psi;self.pt=psi.T.contiguous();self.pc=psi.conj().contiguous()
        self.n,self.l=psi.shape
        array=psi.detach().cpu().numpy()
        freq=np.argmax(abs(np.fft.fft(array,axis=0)),axis=0)
        phase=array[0]
        ref=np.exp(2j*np.pi*np.arange(self.n)[:,None]*freq[None,:]/self.n)*phase[None,:]
        if np.linalg.norm(ref-array)/np.linalg.norm(array)>1e-10:raise ValueError('not complete uniform Fourier orbits')
        if len(set(freq))!=self.l:raise ValueError('aliased illumination modes')
        self.lags=torch.as_tensor((freq[None,:]-freq[:,None])%self.n,device=psi.device,dtype=torch.long)
        self.phases=torch.as_tensor(phase[:,None]*phase.conj()[None,:],device=psi.device,dtype=psi.dtype)
        self.diag=torch.as_tensor(self.n*abs(phase)**2,device=psi.device,dtype=psi.dtype)
        self.freq=torch.as_tensor(freq,device=psi.device,dtype=torch.long)
        self.phase=torch.as_tensor(phase,device=psi.device,dtype=psi.dtype)
        self.pairs=[]
        for lag in range(self.n):
            a,b=np.nonzero((freq[None,:]-freq[:,None])%self.n==lag)
            self.pairs.append((torch.as_tensor(a,device=psi.device),torch.as_tensor(b,device=psi.device),self.phases[a,b]))

    def prepare(self,method,w_np):
        # The benchmark contract supplies host-resident weights. Extract support
        # there, avoiding an unnecessary device-to-host synchronization roundtrip.
        if method in ('fft_dense','fourier_sparse'):
            hat_np=np.fft.fft(np.asarray(w_np))
            hat=torch.as_tensor(hat_np,device=self.psi.device,dtype=self.psi.dtype)
            if method=='fft_dense':return dict(g=hat[self.lags]*self.phases)
            threshold=1e-12*float(np.max(abs(hat_np)))
            active=np.flatnonzero(abs(hat_np)>threshold).tolist()
            tail=float(np.sum(abs(hat_np[abs(hat_np)<=threshold])))
            return dict(hat=hat,active=active,discarded_l1=tail)
        w=torch.as_tensor(np.ascontiguousarray(w_np),device=self.psi.device,dtype=self.psi.real.dtype)
        if method in ('explicit','fft_view'):return dict(w=w)
        if method=='dense_gram':return dict(g=(self.pt*w[None,:])@self.pc)
        if method=='view_update':
            changed=np.flatnonzero(np.asarray(w_np)!=1.)
            selected=self.psi[torch.as_tensor(changed,device=self.psi.device)]
            return dict(pt=selected.T.contiguous(),pc=selected.conj().contiguous(),dw=w[changed]-1)
        raise ValueError(method)

    def apply(self,method,plan,rows):
        if method=='explicit':return ((rows@self.pt)*plan['w'][None,:])@self.pc
        if method=='fft_view':
            spectrum=torch.zeros((len(rows),self.n),device=rows.device,dtype=rows.dtype)
            spectrum.index_copy_(1,self.freq,rows*self.phase[None,:])
            views=torch.fft.ifft(spectrum,dim=1)*self.n
            back=torch.fft.fft(views*plan['w'][None,:],dim=1)
            return back.index_select(1,self.freq)*self.phase.conj()[None,:]
        if method in ('dense_gram','fft_dense'):return rows@plan['g']
        if method=='view_update':return rows*self.diag[None,:]+((rows@plan['pt'])*plan['dw'][None,:])@plan['pc']
        out=torch.zeros_like(rows)
        for lag in plan['active']:
            a,b,phase=self.pairs[lag]
            if len(a):out.index_add_(1,b,rows.index_select(1,a)*(phase*plan['hat'][lag])[None,:])
        return out


def weighted_normal(cones,geometries,method,plans,coeff,index):
    result=None
    for cone,geo,plan in zip(cones,geometries,plans):
        if method=='explicit':
            y=cone.forward_selected_z_modes(coeff,index).reshape(cone.n_illum,cone.cap_radial,cone.n_h)
            value=cone.adjoint_selected_z_modes((y*plan['w'][:,None,None]).reshape(-1),index)
        else:
            latent=cone_reduced_h_l_u(cone,coeff,index)
            power=(cone.mode_phase*cone.mode_phase_conj).reshape(cone.n_h,1,1)
            rows=(latent*power).permute(2,0,1).reshape(cone.cap_radial*cone.n_h,cone.n_l)
            mixed=geo.apply(method,plan,rows)*float(cone.cap_phi)
            value=cone._adjoint_selected_z_from_illumination_mixed(mixed,index)
        result=value if result is None else result+value
    return result


def profile(name,n,phase=0.):
    if n==1:return np.ones(1)
    phi=2*np.pi*np.arange(n)/n
    if name=='harmonic1':return 1+.25*np.cos(phi+phase)
    if name=='harmonic3':return 1+.20*np.cos(phi+phase)+.15*np.sin(3*phi-.4+phase)
    w=np.ones(n);w[17%n]=.2+.05*np.cos(phase)
    return w
