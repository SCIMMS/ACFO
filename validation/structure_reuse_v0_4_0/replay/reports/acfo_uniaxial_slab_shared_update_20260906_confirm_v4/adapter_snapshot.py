"""Validation-only factorization of the existing scalar annular slab operator.

Fixed radii, source quadrature, material and frequency; only slab thickness
and a uniform target-height offset vary. No additional radial compression.
"""
import numpy as np
from scipy import special


def root(z):
    v=np.sqrt(np.asarray(z,dtype=complex))
    return np.where((v.imag<0)|((v.imag==0)&(v.real<0)),-v,v)


def quadrature(nodes):
    # Resolve the damped ambient light line without changing the physics.
    breaks=[*np.linspace(0.,10.,41),*np.linspace(10.5,30.,40),*np.linspace(32.,50.,10)]
    x,w=np.polynomial.legendre.leggauss(nodes)
    return (np.concatenate([(a+b)/2+(b-a)*x/2 for a,b in zip(breaks[:-1],breaks[1:])]),
            np.concatenate([(b-a)*w/2 for a,b in zip(breaks[:-1],breaks[1:])]))


class SlabFactors:
    def __init__(self,rs,rt,hs,ht,sw,q,w,branch='extraordinary',eps_z=-1.2+.04j,nphi=192):
        self.q=q;self.nphi=nphi;self.ns=len(rs);self.nt=len(rt)
        self.h=int(np.ceil(50*max(max(rs),max(rt))))+48
        if self.h>=nphi//2:raise ValueError('support reaches Nyquist')
        modes=np.rint(np.fft.fftfreq(nphi)*nphi).astype(int)
        self.indices=np.flatnonzero(abs(modes)<=self.h)
        orders=abs(modes[self.indices])
        self.kz=root((4.+.05j)**2-q*q)
        ep=2.4+.025j
        self.ks=root(ep*(4.+.05j)**2-(q*q if branch=='ordinary' else ep/eps_z*q*q))
        y=self.ks if branch=='ordinary' else self.ks/ep
        self.r=(self.kz-y)/(self.kz+y)
        self.prefactor=1j/(4*np.pi)*w*q/self.kz
        positive=np.arange(self.h+1)[:,None,None]
        js=special.jv(positive,q[None,:,None]*rs[None,None,:])[orders]
        jt=special.jv(positive,q[None,:,None]*rt[None,None,:])[orders]
        self.b=np.ascontiguousarray(js*np.exp(1j*self.kz[:,None]*hs[None,:])[None,:,:]*sw[None,None,:])
        self.a=np.ascontiguousarray((jt*np.exp(1j*self.kz[:,None]*ht[None,:])[None,:,:]).transpose(0,2,1))

    def weights(self,d,height):
        e=np.exp(2j*self.ks*d);den=1-self.r**2*e
        refl=self.r*(1-e)/den
        dr=2j*self.ks*e*self.r*(self.r**2-1)/den**2
        phase=self.prefactor*np.exp(1j*self.kz*height)
        weights=np.stack((phase*refl,phase*dr,phase*refl*1j*self.kz))
        if not np.isfinite(weights).all():raise FloatingPointError('nonfinite weights')
        return weights

    def prepare(self,method,d,height):
        w=self.weights(d,height)
        if method=='modal_cached':
            return np.stack([(self.a[:self.h+1]*wi[None,None,:])@self.b[:self.h+1] for wi in w])
        return w

    def source_modes(self,f):
        return np.ascontiguousarray(np.fft.fft(f,axis=-1)[:,self.indices].T[:,:,None])

    def synthesize(self,values):
        out=np.zeros((3,self.nt,self.nphi),dtype=complex)
        out[:,:,self.indices]=2*np.pi*values.transpose(0,2,1)
        return np.fft.ifft(out,axis=-1)

    def apply(self,method,plan,f):
        fm=self.source_modes(f)
        if method=='modal_cached':
            pos=(plan@fm[None,:self.h+1,:,:])[...,0]
            neg=(plan[:,1:][:,::-1]@fm[None,self.h+1:,:,:])[...,0]
            result=np.concatenate((pos,neg),axis=1)
        elif method=='factor_shared':
            projected=self.b@fm
            result=np.stack([(self.a@(w[None,:,None]*projected))[...,0] for w in plan])
        elif method=='factor_separate':
            result=np.stack([(self.a@(w[None,:,None]*(self.b@fm)))[...,0] for w in plan])
        else:raise ValueError(method)
        return self.synthesize(result)

    @property
    def retained_bytes(self):
        return sum(v.nbytes for v in vars(self).values() if isinstance(v,np.ndarray))
