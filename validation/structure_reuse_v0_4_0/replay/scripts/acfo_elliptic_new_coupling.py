"""Append a known harmonic generator without modifying prior frozen adapters."""
from time import perf_counter
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg
from scripts.acfo_cylinder_coupling import (
    Bank, CylinderSpace, plus, minus, block_sparse, stack_field, observe,
)
from scripts.acfo_elliptic_refresh import spectral, gram_norm, SEEDS
from scripts.acfo_elliptic_observation_refresh import ObservationContext

SHIFTS = ((2,1), (3,-2), (1,0))
AMPLITUDES = (.6, .4, .3)
ARMS = ('full_cold', 'full_blocks', 'full_reuse', 'query_reuse', 'direct_reuse')


class ExtendedBank(Bank):
    @classmethod
    def inherit(cls, bank):
        result = cls.__new__(cls)
        result.__dict__ = bank.__dict__.copy()
        return result

    def edge(self, l, target, source):
        if l < 2:
            return super().edge(l, target, source)
        if l != 2 or minus(target,source) not in ((1,0),(-1,0)):
            raise ValueError('Invalid appended harmonic edge')
        key, reverse = (l,target,source), (l,source,target)
        if reverse in self.edges:
            return self.edges[reverse].T
        if key not in self.edges:
            mt,kt = target; ms,ks = source
            weight = self.w*.15*self.r
            block = kt*ks*(self.state(mt).T@(weight[:,None]*self.state(ms)))
            for sign in (-1,1):
                block += self.grad(mt,sign).T@(weight[:,None]*self.grad(ms,sign))
            dt = abs(mt)+2*np.arange(self.nr); ds = abs(ms)+2*np.arange(self.nr)
            self.structural_zeros(block, abs(dt[:,None]-ds[None,:])>3)
            block[np.abs(block)<2e-14] = 0
            self.edges[key] = block
        return self.edges[key]


def rho(theta):
    value = sum(w*abs(t) for w,t in zip(AMPLITUDES,theta))/1.2
    if not value < 1:
        raise ValueError('Contraction bound fails')
    return value


class Context(ObservationContext):
    def __init__(self, nr=18, per_seed=2, quadrature=145):
        super().__init__(nr,per_seed,quadrature)
        self.bank = ExtendedBank.inherit(self.bank)
        # Existing observation geometry with first-new-generator channels.
        # Build explicitly; do not change the prior geometry catalogue.
        from scripts.acfo_cylinder_coupling import observation
        requests = [(m,q) for m in ((0,0),(5,-2),(2,1),(3,-2),(5,-1),
                                    (1,0),(6,-2),(3,1),(4,-2)) for q in (1.7,4.3)]
        self.obs = observation(nr,requests,quadrature)
        self.rootw = np.ones((len(self.obs),1))
        self.observation_dual = 0.
        for m in {m for m,_ in self.obs}:
            rows = np.array([r for mode,r in self.obs if mode==m])
            self.observation_dual = max(self.observation_dual,
                gram_norm(rows@self.bank.solve(m,rows.T)))
        self.y0norm = spectral(observe(self.u0,self.obs,self.ncols))

    def select(self, theta, tolerance, max_order=14):
        r = rho(theta); scale = self.source_energy*self.observation_dual
        lower = self.y0norm-scale*r/(1-r)
        if lower <= 0: raise ValueError('Response lower bound inconclusive')
        for p in range(max_order+1):
            absolute = scale*r**(p+1)/(1-r)
            if absolute/lower <= tolerance: return p,absolute/lower,absolute
        raise ValueError('Order limit exceeded')


def indices(p, active):
    return [(a,b,d-a-b) for d in range(p+1) for a in range(d+1) for b in range(d-a+1)
            if all(v==0 or l in active for l,v in enumerate((a,b,d-a-b)))]


def forward(p, active):
    result = {(0,0,0): set(SEEDS)}
    for alpha in indices(p,active)[1:]:
        nodes = set()
        for l in active:
            if alpha[l]:
                prev=list(alpha);prev[l]-=1
                for source in result[tuple(prev)]:
                    nodes.update((plus(source,SHIFTS[l]),minus(source,SHIFTS[l])))
        result[alpha] = nodes
    return result


def plan(p, active, output_modes, query):
    fwd = forward(p,active)
    if not query: return fwd
    needed = {a:set() for a in fwd}
    stack = [(a,m) for a,nodes in fwd.items() for m in output_modes if m in nodes]
    while stack:
        a,m = stack.pop()
        if m in needed[a]: continue
        needed[a].add(m)
        for l in active:
            if a[l]:
                prev=list(a);prev[l]-=1;prev=tuple(prev)
                for n in (plus(m,SHIFTS[l]),minus(m,SHIFTS[l])):
                    if n in fwd[prev]: stack.append((prev,n))
    return needed


class Taylor:
    def __init__(self,c,query=False,retain=True,legacy=None):
        self.c,self.query,self.retain = c,query,retain
        self.values = ({(a,b,0):dict(field) for (a,b),field in legacy.items()} if legacy is not None
                       else {(0,0,0):dict(c.u0)})

    def action(self,theta,tolerance):
        c=self.c; active=tuple(l for l,t in enumerate(theta) if t!=0)
        p,bound,absolute=c.select(theta,tolerance)
        t=perf_counter(); wanted=plan(p,active,{m for m,_ in c.obs},self.query)
        planning=perf_counter()-t
        if not self.retain: self.values={(0,0,0):dict(c.u0)}
        reused=old_added=new_added=0; before=c.bank.factor_solves
        t=perf_counter()
        for a,nodes in wanted.items():
            field=self.values.setdefault(a,{})
            for target in sorted(nodes):
                if target in field:
                    reused+=1; continue
                rhs=np.zeros((c.nr,c.ncols),complex)
                for l in active:
                    if a[l]:
                        prev=list(a);prev[l]-=1
                        prior=self.values[tuple(prev)]
                        for source in (plus(target,SHIFTS[l]),minus(target,SHIFTS[l])):
                            if source in prior: rhs-=c.bank.edge(l,target,source)@prior[source]
                field[target]=c.bank.solve(target,rhs)
                if a[2]==0: old_added+=1
                else: new_added+=1
        build=perf_counter()-t;t=perf_counter()
        y=sum(np.prod(np.asarray(theta)**np.asarray(a))*observe(self.values[a],c.obs,c.ncols)
              for a in wanted)
        return y,dict(order=p,relative_bound=bound,absolute_bound=absolute,
            planning_s=planning,extension_s=build,synthesis_s=perf_counter()-t,
            reused_nodes=reused,restored_old_plane=old_added,new_generator_nodes=new_added,
            new_nodes=old_added+new_added,requested_nodes=sum(map(len,wanted.values())),
            solve_calls=c.bank.factor_solves-before)


def mode_ball(radius,active):
    seen=set(SEEDS);front=set(SEEDS)
    for _ in range(radius):
        layer={target for m in front for l in active
               for target in (plus(m,SHIFTS[l]),minus(m,SHIFTS[l]))}-seen
        seen |= layer;front=layer
    return tuple(sorted(seen))


def component(bank,space,l):
    blocks=[]
    for j,source in enumerate(space.modes):
        for target in (plus(source,SHIFTS[l]),minus(source,SHIFTS[l])):
            if target in space.indices:
                blocks.append((space.indices[target],j,bank.edge(l,target,source)))
    return block_sparse(space.modes,space.modes,blocks,bank.nr)


def field_certificate(c,field,theta):
    """Full residual including every active exterior edge after the change."""
    residual={m:b.copy() for m,b in c.sources.items()}
    for m,u in field.items():
        residual.setdefault(m,np.zeros_like(u))
        residual[m]-=c.bank.base(m)@u
        for l,t in enumerate(theta):
            if t:
                for target in (plus(m,SHIFTS[l]),minus(m,SHIFTS[l])):
                    residual.setdefault(target,np.zeros_like(u))
                    residual[target]-=t*(c.bank.edge(l,target,m)@u)
    gram=sum(r.conj().T@c.bank.solve(m,r) for m,r in residual.items())
    absolute=c.observation_dual*gram_norm(gram)/(1-rho(theta))
    y=observe(field,c.obs,c.ncols);lower=spectral(c.rootw*y)-absolute
    return y,absolute/lower if lower>0 else float('inf'),absolute,len(residual)-len(field)


class Direct:
    def __init__(self,c):
        self.c=c;self.field={};self.parts={};self.radius=2

    def certify(self,theta):
        c=self.c;modes=tuple(sorted(self.field))
        space,base,vs=self.parts[modes]
        active=tuple(l for l,t in enumerate(theta) if t)
        for l in active:
            if l not in vs:vs[l]=component(c.bank,space,l)
        a=base+sum((theta[l]*vs[l] for l in active),sparse.csr_matrix(base.shape))
        u=stack_field(space,self.field,c.ncols)
        residual=stack_field(space,c.sources,c.ncols)-a@u
        fields={m:residual[i*c.nr:(i+1)*c.nr].copy() for i,m in enumerate(modes)}
        # Interior action uses the same cached affine CSR parts as the solve.
        # Only the genuinely exterior leakage needs local edge accumulation.
        for source,state in self.field.items():
            for l in active:
                for target in (plus(source,SHIFTS[l]),minus(source,SHIFTS[l])):
                    if target not in space.indices:
                        fields.setdefault(target,np.zeros_like(state))
                        fields[target]-=theta[l]*(c.bank.edge(l,target,source)@state)
        gram=sum(r.conj().T@c.bank.solve(m,r) for m,r in fields.items())
        absolute=c.observation_dual*gram_norm(gram)/(1-rho(theta))
        y=observe(self.field,c.obs,c.ncols);lower=spectral(c.rootw*y)-absolute
        return y,absolute/lower if lower>0 else float('inf'),absolute,len(fields)-len(modes)

    def action(self,theta,tolerance,max_radius=14):
        c=self.c;before=c.bank.factor_solves;start=perf_counter()
        attempts=[];iterations=0
        if self.field:
            y,bound,absolute,exterior=self.certify(theta)
            if bound<=tolerance:
                return y,dict(relative_bound=bound,absolute_bound=absolute,field_reused=True,
                    radius=self.radius,iterations=0,solve_calls=c.bank.factor_solves-before,
                    direct_s=perf_counter()-start,exterior_modes_checked=exterior)
        active=tuple(l for l,t in enumerate(theta) if t)
        for radius in range(self.radius,max_radius+1,2):
            modes=mode_ball(radius,active)
            if modes not in self.parts:
                space=CylinderSpace(modes,c.nr)
                self.parts[modes]=space,sparse.block_diag([c.bank.base(m) for m in modes],format='csr'),{}
            space,base,vs=self.parts[modes]
            for l in active:
                if l not in vs: vs[l]=component(c.bank,space,l)
            a=base+sum((theta[l]*vs[l] for l in active),sparse.csr_matrix(base.shape))
            b=stack_field(space,c.sources,c.ncols);x0=stack_field(space,self.field,c.ncols)
            def pre(x):
                return np.concatenate([c.bank.solve(m,x[i*c.nr:(i+1)*c.nr]) for i,m in enumerate(modes)])
            preop=LinearOperator(a.shape,matvec=pre,dtype=np.complex128)
            columns=[]
            for j in range(c.ncols):
                count=[0]
                def callback(_):count[0]+=1
                tol=min(1e-12,tolerance*1e-3) if tolerance<=1e-10 else tolerance*.01
                u,info=cg(a,b[:,j],x0=x0[:,j],M=preop,rtol=tol,atol=0,maxiter=200,callback=callback)
                if info:raise RuntimeError(f'CG failed: {info}')
                columns.append(u);iterations+=count[0]
            u=np.column_stack(columns)
            self.field={m:u[i*c.nr:(i+1)*c.nr] for i,m in enumerate(modes)}
            y,bound,absolute,exterior=self.certify(theta)
            attempts.append(dict(radius=radius,modes=len(modes),bound=bound))
            if bound<=tolerance:
                self.radius=radius
                return y,dict(relative_bound=bound,absolute_bound=absolute,field_reused=False,
                    radius=radius,iterations=iterations,solve_calls=c.bank.factor_solves-before,
                    direct_s=perf_counter()-start,attempts=attempts,exterior_modes_checked=exterior)
        raise RuntimeError('Direct region limit exceeded')


class Runner:
    def __init__(self,arm,nr=18,per_seed=2,quadrature=145):
        if arm not in ARMS:raise ValueError(arm)
        self.arm=arm;self.args=nr,per_seed,quadrature;self.c=self.engine=None
        self.responses={}

    def step(self,event):
        start=perf_counter();setup=perf_counter()
        cold=self.arm=='full_cold'
        if self.c is None or cold:
            self.c=Context(*self.args)
            self.engine=(Direct(self.c) if self.arm=='direct_reuse' else
                         Taylor(self.c,query=self.arm=='query_reuse',retain=self.arm!='full_blocks'))
        setup_s=perf_counter()-setup;c=self.c;theta=tuple(event['theta']);tol=event['tolerance']
        saved=self.responses.get(theta) if not cold else None
        y=meta=None
        if saved is not None and saved[1]['relative_bound']<=tol:
            y,old=saved;meta=dict(old,exact_cache_hit=True,perturbation_reused=False)
        elif not cold:
            # Known resolvent bound lets every warm method skip sufficiently
            # small operator changes, not only the Taylor methods.
            for previous,(candidate,old) in reversed(list(self.responses.items())):
                delta=sum(w*abs(a-b) for w,a,b in zip(AMPLITUDES,theta,previous))/1.2
                change=c.observation_dual*c.source_energy*delta/((1-rho(theta))*(1-rho(previous)))
                absolute=old['absolute_bound']+change
                lower=spectral(c.rootw*candidate)-absolute
                bound=absolute/lower if lower>0 else float('inf')
                if bound<=tol:
                    y=candidate;meta=dict(relative_bound=bound,absolute_bound=absolute,
                        perturbation_reused=True,exact_cache_hit=False,origin_theta=previous)
                    break
        if y is not None:
            for k in list(meta):
                if k.endswith('_s') or k in ('solve_calls','iterations','new_nodes','reused_nodes',
                                               'restored_old_plane','new_generator_nodes'):
                    meta[k]=0
            meta.update(new_nodes=0,solve_calls=0,restored_old_plane=0,new_generator_nodes=0)
            meta['field_reused']=False
        else:
            y,meta=self.engine.action(theta,tol)
            meta.update(exact_cache_hit=False,perturbation_reused=False)
        if not cold:self.responses[theta]=y,dict(meta)
        meta['setup_s']=setup_s;meta['bank_bytes']=c.bank.bytes()
        meta['response_cache_bytes']=sum(y.nbytes for y,_ in self.responses.values())
        meta['edge_blocks_by_generator']=[sum(k[0]==l for k in c.bank.edges) for l in range(3)]
        if isinstance(self.engine,Taylor):
            meta['retained_nodes']=sum(map(len,self.engine.values.values()))
            meta['retained_old_plane']=sum(len(v) for a,v in self.engine.values.items() if a[2]==0)
            meta['state_bytes']=sum(v.nbytes for f in self.engine.values.values() for v in f.values())
            meta['sparse_bytes']=0
        else:
            meta['retained_nodes']=len(self.engine.field)
            meta['state_bytes']=sum(v.nbytes for v in self.engine.field.values())
            meta['sparse_bytes']=sum(sum(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes
                for a in (base,*vs.values())) for _,base,vs in self.engine.parts.values())
        meta['total_s']=perf_counter()-start
        return y,meta
