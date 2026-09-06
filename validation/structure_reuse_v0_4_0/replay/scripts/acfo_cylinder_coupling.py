"""Validation adapter: known Fourier-Jacobi calculus on disk x periodic axis.

States: phi_mj(r)*exp(i*m*phi+i*k*z)/(2*pi); volume metric r dr dphi dz.
Loads and observations are dual functionals, not Euclidean point samples.
No circular indexing. No public API changes or new algebra identities.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
from time import perf_counter
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator, cg
from scipy.special import jv
from scripts.acfo_elliptic_disk import radial_rule, state_values, zernike_values

SHIFTS = ((2, 1), (3, -2))
WEIGHTS = (0.6, 0.4)


def plus(a, b):
    return tuple(x+y for x,y in zip(a,b))


def minus(a, b):
    return tuple(x-y for x,y in zip(a,b))


def alphas(order):
    return [(a,d-a) for d in range(order+1) for a in range(d+1)]


@dataclass(frozen=True)
class CylinderSpace:
    modes: tuple[tuple[int,int], ...]
    nr: int

    @cached_property
    def indices(self):
        return {mode:i for i,mode in enumerate(self.modes)}

    @property
    def size(self):
        return len(self.modes)*self.nr


class Bank:
    """Lazy local radial actions shared by every competing realization."""
    def __init__(self, nr, quadrature=90, sigma=0.3):
        self.nr, self.sigma = nr, sigma
        self.r, self.w = radial_rule(quadrature)
        self.states, self.gradients = {}, {}
        self.bases, self.masses, self.factors, self.edges = {}, {}, {}, {}
        self.factor_solves = 0
        self.max_structural_zero_removed = 0.

    def structural_zeros(self,block,mask):
        self.max_structural_zero_removed=max(self.max_structural_zero_removed,
                                             float(np.max(np.abs(block[mask]),initial=0)))
        block[mask]=0
        return block

    def state(self,m):
        m=abs(m)
        if m not in self.states:
            self.states[m]=np.column_stack([state_values(m,n,self.r)[0] for n in range(self.nr)])
        return self.states[m]

    def grad(self,m,sign):
        key=(m,sign)
        if key not in self.gradients:
            target=m+sign
            ns=[n+int(abs(target)<abs(m)) for n in range(self.nr)]
            self.gradients[key]=-zernike_values(target,ns,self.r)/np.sqrt(2)
        return self.gradients[key]

    def mass(self,m):
        m=abs(m)
        if m not in self.masses:
            b=self.state(m)
            value=b.T@(self.w[:,None]*b)
            ii=np.arange(self.nr)
            self.masses[m]=self.structural_zeros(value,abs(ii[:,None]-ii[None,:])>1)
        return self.masses[m]

    def base(self,mode):
        m,k=map(abs,mode)
        key=(m,k)
        if key not in self.bases:
            b=self.state(m); wa=self.w*(1+0.2*self.r**2)
            value=k*k*(b.T@(wa[:,None]*b))+self.sigma*self.mass(m)
            for sign in (-1,1):
                g=self.grad(m,sign)
                value+=g.T@(wa[:,None]*g)
            ii=np.arange(self.nr)
            # State mass is tridiagonal; multiplication by r^2 adds one band.
            self.structural_zeros(value,abs(ii[:,None]-ii[None,:])>2)
            self.bases[key]=(value+value.T)/2
        return self.bases[key]

    def edge(self,l,target,source):
        difference=minus(target,source)
        if difference not in (SHIFTS[l],tuple(-x for x in SHIFTS[l])):
            raise ValueError("illegal joint-label edge")
        key=(l,target,source)
        reverse=(l,source,target)
        if reverse in self.edges:
            return self.edges[reverse].T
        if key not in self.edges:
            mt,kt=target; ms,ks=source
            weight=self.w*0.5*WEIGHTS[l]*self.r**abs(SHIFTS[l][0])
            block=kt*ks*(self.state(mt).T@(weight[:,None]*self.state(ms)))
            for sign in (-1,1):
                block+=self.grad(mt,sign).T@(weight[:,None]*self.grad(ms,sign))
            # Polynomial multiplication and Jacobi orthogonality give this
            # exact degree bound (the state mass contributes two extra degrees).
            dt=abs(mt)+2*np.arange(self.nr);ds=abs(ms)+2*np.arange(self.nr)
            self.structural_zeros(block,abs(dt[:,None]-ds[None,:])>abs(SHIFTS[l][0])+2)
            # No claim that arbitrary nonpolynomial radial blocks are banded.
            block[np.abs(block)<2e-14]=0
            self.edges[key]=block
        return self.edges[key]

    def solve(self,mode,rhs):
        key=tuple(map(abs,mode))
        if key not in self.factors:
            self.factors[key]=cho_factor(self.base(mode),lower=True,check_finite=False)
        self.factor_solves+=1
        return cho_solve(self.factors[key],rhs,check_finite=False)

    def bytes(self):
        arrays=list(self.states.values())+list(self.gradients.values())+list(self.bases.values())+list(self.masses.values())+list(self.edges.values())
        arrays += [f[0] for f in self.factors.values()]
        return sum(a.nbytes for a in arrays)


def block_sparse(rows,cols,blocks,nr):
    rr=[];cc=[];vv=[]
    for i,j,b in blocks:
        r,c=np.nonzero(b)
        rr.extend((i*nr+r).tolist());cc.extend((j*nr+c).tolist());vv.extend(b[r,c].tolist())
    return sparse.csr_matrix((vv,(rr,cc)),shape=(len(rows)*nr,len(cols)*nr))


def matrix_parts(bank,space):
    base=sparse.block_diag([bank.base(x) for x in space.modes],format="csr")
    vs=[]
    for l,shift in enumerate(SHIFTS):
        blocks=[]
        for j,source in enumerate(space.modes):
            for delta in (shift,tuple(-x for x in shift)):
                target=plus(source,delta)
                if target in space.indices:
                    blocks.append((space.indices[target],j,bank.edge(l,target,source)))
        vs.append(block_sparse(space.modes,space.modes,blocks,bank.nr))
    return base,vs


def solve_pde(bank,space,rhs,theta,tol=2e-12):
    base,vs=matrix_parts(bank,space)
    matrix=base+sum((t*v for t,v in zip(theta,vs)),sparse.csr_matrix(base.shape))
    def pre(x):
        blocks=x.reshape(len(space.modes),bank.nr)
        return np.concatenate([bank.solve(mode,b) if np.any(b) else b for mode,b in zip(space.modes,blocks)])
    preop=LinearOperator(matrix.shape,matvec=pre,dtype=rhs.dtype)
    columns=rhs[:,None] if rhs.ndim==1 else rhs
    solutions=[];iterations=[]
    for column in columns.T:
        count=[0]
        def callback(_): count[0]+=1
        u,info=cg(matrix,column,M=preop,rtol=tol,atol=0,maxiter=300,callback=callback)
        if info: raise RuntimeError(f"PCG did not converge: {info}")
        solutions.append(u);iterations.append(count[0])
    result=np.column_stack(solutions)
    return (result[:,0] if rhs.ndim==1 else result),matrix,vs,iterations


def modal_load(nr,mode,radial_function,quadrature=130):
    r,w=radial_rule(quadrature)
    b=np.column_stack([state_values(mode[0],n,r)[0] for n in range(nr)])
    return b.T@(w*radial_function(r))


def source_columns(nr,seeds,per_seed=1):
    out={};total=len(seeds)*per_seed
    for i,mode in enumerate(seeds):
        rhs=np.zeros((nr,total),complex)
        for j in range(per_seed):
            beta=-2.-0.4*j
            rhs[:,i*per_seed+j]=modal_load(nr,mode,lambda r:r**abs(mode[0])*np.exp(beta*r*r))*(1+0.13j*(j+1))
        out[mode]=rhs
    return out


def stack_field(space,field,ncols=None):
    if ncols is None:
        return np.concatenate([field.get(mode,np.zeros(space.nr,complex)) for mode in space.modes])
    return np.vstack([field.get(mode,np.zeros((space.nr,ncols),complex)) for mode in space.modes])


def unstack_field(space,vector):
    return {mode:vector[i*space.nr:(i+1)*space.nr] for i,mode in enumerate(space.modes)}


def observation(nr,requests,quadrature=140):
    """Requests=(m,k,q); C u is a radial Hankel moment of the normalized mode.

    Inner product with J_m(q*r)*exp(i*m*phi+i*k*z)/(2*pi).
    """
    r,w=radial_rule(quadrature)
    return [(mode, np.array([np.sum(w*jv(mode[0],q*r)*state_values(mode[0],n,r)[0])
                             for n in range(nr)])) for mode,q in requests]


def observation_matrix(space,obs):
    c=np.zeros((len(obs),space.size))
    for j,(mode,row) in enumerate(obs):
        if mode in space.indices:
            i=space.indices[mode];c[j,i*space.nr:(i+1)*space.nr]=row
    return c


def observe(field,obs,ncols):
    return np.array([row@field[mode] if mode in field else np.zeros(ncols,complex)
                     for mode,row in obs])


def reachable(seeds,order):
    table={(0,0):set(seeds)}
    for alpha in alphas(order)[1:]:
        active=set()
        for l in range(2):
            if alpha[l]:
                prev=list(alpha);prev[l]-=1
                for source in table[tuple(prev)]:
                    active.add(plus(source,SHIFTS[l]));active.add(minus(source,SHIFTS[l]))
        table[alpha]=active
    return table


def make_plan(seeds,requests,output_modes,query_directed):
    forward=reachable(seeds,max(map(sum,requests)))
    if not query_directed:
        return forward
    needed={a:set() for a in forward}
    stack=[(a,node) for a in requests for node in output_modes if node in forward[a]]
    while stack:
        alpha,node=stack.pop()
        if node in needed[alpha]: continue
        needed[alpha].add(node)
        for l in range(2):
            if alpha[l]:
                prev=list(alpha);prev[l]-=1;prev=tuple(prev)
                for source in (minus(node,SHIFTS[l]),plus(node,SHIFTS[l])):
                    if source in forward[prev]: stack.append((prev,source))
    return needed


def prepare_plan(bank,plan,native_sparse=False):
    """Materialize all required local blocks/factors; count full preparation."""
    operators={}
    ordered={a:tuple(sorted(nodes)) for a,nodes in plan.items()}
    for alpha,nodes in ordered.items():
        for node in nodes:
            key=tuple(map(abs,node))
            if key not in bank.factors:
                bank.factors[key]=cho_factor(bank.base(node),lower=True,check_finite=False)
        for l in range(2):
            if not alpha[l]: continue
            prev=list(alpha);prev[l]-=1;prev=tuple(prev)
            cols=ordered[prev]; indices={node:i for i,node in enumerate(cols)}
            blocks=[]
            for i,target in enumerate(nodes):
                for source in (minus(target,SHIFTS[l]),plus(target,SHIFTS[l])):
                    if source in indices:
                        block=bank.edge(l,target,source)
                        if native_sparse: blocks.append((i,indices[source],block))
            if native_sparse:
                operators[alpha,l]=block_sparse(nodes,cols,blocks,bank.nr)
    return ordered,operators


def evaluate_plan(bank,prepared,sources,native_sparse=False):
    ordered,operators=prepared
    ncols=next(iter(sources.values())).shape[1]
    values={};max_residual=0.;residual_scale=0.
    for alpha,nodes in ordered.items():
        fields={}
        if alpha==(0,0):
            loads={node:sources[node] for node in nodes}
        elif native_sparse:
            rhs=np.zeros((len(nodes)*bank.nr,ncols),complex)
            for l in range(2):
                if alpha[l]:
                    prev=list(alpha);prev[l]-=1;prev=tuple(prev)
                    if ordered[prev]:
                        rhs-=operators[alpha,l]@np.vstack([values[prev][n] for n in ordered[prev]])
            loads={node:rhs[i*bank.nr:(i+1)*bank.nr] for i,node in enumerate(nodes)}
        else:
            loads={node:np.zeros((bank.nr,ncols),complex) for node in nodes}
            for l in range(2):
                if alpha[l]:
                    prev=list(alpha);prev[l]-=1;prev=tuple(prev)
                    for target in nodes:
                        for source in (minus(target,SHIFTS[l]),plus(target,SHIFTS[l])):
                            if source in values[prev]:
                                loads[target]-=bank.edge(l,target,source)@values[prev][source]
        for node,rhs in loads.items():
            fields[node]=bank.solve(node,rhs)
            max_residual=max(max_residual,float(np.linalg.norm(bank.base(node)@fields[node]-rhs)))
            residual_scale=max(residual_scale,float(np.linalg.norm(rhs)))
        values[alpha]=fields
    return values,max_residual/max(residual_scale,1e-300)


def radial_fields(nr,field,r):
    out={}
    for (m,k),c in field.items():
        values=[state_values(m,n,r) for n in range(nr)]
        out[m,k]=tuple(sum((c[n]*values[n][j] for n in range(nr)),np.zeros_like(r,dtype=complex)) for j in range(3))
    return out


def strong_action(fields,r,theta,sigma=0.3):
    """Independent analytic strong form (does not use coefficient gradient)."""
    out={}
    for mode,(u,ur,lap) in fields.items():
        m,k=mode
        out[mode]=out.get(mode,0)-(1+0.2*r*r)*(lap-k*k*u)-0.4*r*ur+sigma*u
        for l,(s,t) in enumerate(SHIFTS):
            b=0.5*WEIGHTS[l]*r**abs(s);br=abs(s)*b/r
            for sign in (-1,1):
                target=plus(mode,(sign*s,sign*t))
                value=-b*lap-br*ur+(k*k+sign*t*k+sign*s*m/(r*r))*b*u
                out[target]=out.get(target,0)+theta[l]*value
    return out


def physical_norm(fields,r,w,h1=False):
    value=0.
    for (m,k),(u,ur,_) in fields.items():
        value+=np.sum(w*(np.abs(ur)**2+(m*m/(r*r)+k*k)*np.abs(u)**2)) if h1 else np.sum(w*np.abs(u)**2)
    return float(np.sqrt(value))


def difference(a,b,r):
    zero=(np.zeros_like(r,complex),)*3
    return {mode:tuple(x-y for x,y in zip(a.get(mode,zero),b.get(mode,zero))) for mode in set(a)|set(b)}


def connected_box(seeds,M,K):
    active={s for s in seeds if abs(s[0])<=M and abs(s[1])<=K};stack=list(active)
    while stack:
        node=stack.pop()
        for shift in SHIFTS:
            for target in (plus(node,shift),minus(node,shift)):
                if abs(target[0])<=M and abs(target[1])<=K and target not in active:
                    active.add(target);stack.append(target)
    return tuple(sorted(active))
