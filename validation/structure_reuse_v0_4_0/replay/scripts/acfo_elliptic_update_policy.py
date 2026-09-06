"""Certified, cost-aware update dispatch for the fixed elliptic operator family.

An experimental adapter; prior frozen preparation/solve APIs are unchanged.
Cost estimates never certify correctness and never read reference responses.
"""
from time import perf_counter
import math
import numpy as np
from scipy.special import jv
from scripts.acfo_elliptic_new_coupling import (
    Context, Direct, SHIFTS, AMPLITUDES, rho, plan, mode_ball,
)
from scripts.acfo_cylinder_coupling import plus, minus, observe, observation
from scripts.acfo_elliptic_refresh import spectral, gram_norm

ARMS = ('update', 'direct', 'threshold', 'policy')
DEFAULTS = dict(max_order=10, max_nodes=18000, threshold_rho=.15,
                trial_nodes=64, gray_ratio=.65, update_margin=.95,
                trial_budget_fraction=.25)


def bind_observation(c, kind):
    """Known fixed geometry/source family; only the observation changes here."""
    if not hasattr(c, '_policy_obs'):
        c._policy_obs = {'narrow': (c.obs, c.rootw, c.observation_dual, c.y0norm)}
        for i,(mode,row) in enumerate(c.obs):
            c.row_cache[(mode[0],(1.7,4.3)[i%2])]=row
    hit = kind in c._policy_obs
    if not hit:
        if kind == 'wide':
            modes = mode_ball(3, (0, 1, 2)); qs = (1.7, 4.3)
        elif kind == 'radial':
            modes = sorted({m for m, _ in c._policy_obs['narrow'][0]}); qs = (2.1, 5.1)
        else:
            raise ValueError('Unknown observation family: '+kind)
        obs=[]
        for m in modes:
            for q in qs:
                radial_key=(m[0],q)
                if radial_key not in c.row_cache:
                    # Same row/dual reuse granted to every baseline. The finer
                    # reference retains the independent scalar quadrature path.
                    c.row_cache[radial_key]=(c.bank.state(m[0]).T@(c.bank.w*jv(m[0],q*c.bank.r))
                        if c.quadrature==145 else observation(c.nr,[(m,q)],c.quadrature)[0][1])
                row=c.row_cache[radial_key]
                if (m,q) not in c.dual_row_cache:c.dual_row_cache[(m,q)]=c.bank.solve(m,row)
                obs.append((m,row))
        rootw = np.ones((len(obs),1)); dual = 0.
        for m in modes:
            rows = np.array([r for mode,r in obs if mode == m])
            dual = max(dual, gram_norm(rows@np.column_stack([c.dual_row_cache[(m,q)] for q in qs])))
        c._policy_obs[kind] = obs,rootw,dual,spectral(observe(c.u0,obs,c.ncols))
    c.obs,c.rootw,c.observation_dual,c.y0norm = c._policy_obs[kind]
    return hit


class IncrementalTaylor:
    """Inspectable preparation with resumable, atomic coefficient nodes."""
    def __init__(self, c, settings=None):
        self.c = c
        self.settings = dict(DEFAULTS, **(settings or {}))
        self.values = {(0,0,0): dict(c.u0)}
        self.plans = {}

    def inspect(self, theta, tolerance):
        start=perf_counter(); c=self.c
        p,bound,absolute=c.select(theta,tolerance,self.settings['max_order'])
        active=tuple(l for l,t in enumerate(theta) if t)
        graph_key=p,active,tuple(sorted({m for m,_ in c.obs}))
        graph_hit=graph_key in self.plans
        wanted=self.plans[graph_key] if graph_hit else plan(p,active,set(graph_key[2]),True)
        requested=sum(map(len,wanted.values()))
        if requested>self.settings['max_nodes']:
            raise ValueError('Preparation node budget exceeded')
        self.plans[graph_key]=wanted
        missing=[(a,m) for a,nodes in wanted.items() for m in sorted(nodes)
                 if m not in self.values.get(a,{})]
        edges=set(); factors=set()
        for a,m in missing:
            key=(abs(m[0]),abs(m[1]))
            if key not in c.bank.factors: factors.add(key)
            for l in active:
                if not a[l]:continue
                prev=list(a);prev[l]-=1;prev=tuple(prev)
                for source in (plus(m,SHIFTS[l]),minus(m,SHIFTS[l])):
                    if source not in wanted[prev]:continue
                    if (l,m,source) not in c.bank.edges and (l,source,m) not in c.bank.edges:
                        edges.add((l,*sorted((m,source))))
        features=[1.,len(missing)/1000,len(edges)/1000,len(factors)/100,
                  len(wanted)*len(c.obs)/10000]
        return dict(order=p,relative_bound=bound,absolute_bound=absolute,
                    theta=tuple(theta),active=active,wanted=wanted,missing=missing,
                    cursor=0,requested_nodes=requested,features=features,
                    graph_cache_hit=graph_hit,
                    planning_s=perf_counter()-start)

    def advance(self, prepared, count=None, budget_s=None):
        """Stop only between nodes; an incomplete plan is never an answer."""
        start=perf_counter(); c=self.c; initial=prepared['cursor']
        while prepared['cursor']<len(prepared['missing']):
            if count is not None and prepared['cursor']-initial>=count:break
            if budget_s is not None and perf_counter()-start>=budget_s:break
            a,target=prepared['missing'][prepared['cursor']]
            rhs=np.zeros((c.nr,c.ncols),complex)
            for l in prepared['active']:
                if not a[l]:continue
                prev=list(a);prev[l]-=1
                prior=self.values.get(tuple(prev),{})
                for source in (plus(target,SHIFTS[l]),minus(target,SHIFTS[l])):
                    if source in prior:rhs-=c.bank.edge(l,target,source)@prior[source]
            self.values.setdefault(a,{})[target]=c.bank.solve(target,rhs)
            prepared['cursor']+=1
        return prepared['cursor']-initial,perf_counter()-start

    def finish(self, prepared):
        if prepared['cursor']!=len(prepared['missing']):
            raise RuntimeError('Unfinished coefficient plan cannot be certified')
        c=self.c;start=perf_counter()
        y=sum(np.prod(np.asarray(prepared['theta'])**np.asarray(a))*
              observe(self.values.get(a,{}),c.obs,c.ncols) for a in prepared['wanted'])
        return y,dict(order=prepared['order'],relative_bound=prepared['relative_bound'],
            absolute_bound=prepared['absolute_bound'],new_nodes=prepared['cursor'],
            requested_nodes=prepared['requested_nodes'],synthesis_s=perf_counter()-start)


def direct_features(c, direct, theta, tolerance):
    """Cheap current-state features, with no hypothetical PDE solve."""
    r=rho(theta)
    degree=0 if r==0 else max(2,min(14,int(math.ceil(math.log(tolerance*.01)/math.log(r)))))
    radius=max(direct.radius,2*int(math.ceil(degree/2)))
    # A cardinality upper model is enough; do not construct a graph to price a solve.
    nactive=sum(t!=0 for t in theta)
    growth=(2*radius+1)**min(nactive,2)
    return [1.,growth/1000, float(not direct.field),r,
            len(c.obs)/1000, len(direct.field)/1000]


class CostModel:
    def __init__(self, data):self.data=data
    def predict(self, kind, features):
        return max(1e-6,float(np.dot(self.data[kind]['coefficients'],features)))


class Dispatcher:
    def __init__(self, arm='policy', model=None, nr=18, per_seed=2, quadrature=145, settings=None):
        if arm not in ARMS:raise ValueError(arm)
        self.arm,self.args=arm,(nr,per_seed,quadrature)
        self.settings=dict(DEFAULTS,**(settings or {}));self.model=model
        self.c=self.taylor=self.direct=None;self.responses={}
        self.field_theta=None;self.field_energy=None

    def step(self, event):
        if set(event)-{'theta','tolerance','observation'}:
            raise ValueError('Unsupported change outside the fixed operator/source family')
        if len(event['theta'])!=3 or not np.all(np.isfinite(event['theta'])):
            raise ValueError('Three finite material parameters required')
        if not np.isfinite(event['tolerance']) or event['tolerance']<=0:
            raise ValueError('Positive finite tolerance required')
        start=perf_counter(); setup=start
        if self.c is None:
            self.c=Context(*self.args)
            self.taylor=IncrementalTaylor(self.c,self.settings);self.direct=Direct(self.c)
        c=self.c;meta=dict(setup_s=perf_counter()-setup,new_nodes=0,trial_nodes=0,
            trial_s=0.,abandoned_trial_s=0.,update_s=0.,direct_s=0.,planning_s=0.,
            cost_estimation_s=0.,predicted_update_s=None,predicted_direct_s=None)
        t=perf_counter();obskey=event.get('observation','narrow')
        meta['observation_cache_hit']=bind_observation(c,obskey)
        meta['observation_s']=perf_counter()-t
        theta=tuple(event['theta']);tol=event['tolerance'];rho(theta)
        key=(theta,obskey);t=perf_counter();y=None
        saved=self.responses.get(key)
        if saved is not None and saved[1]['relative_bound']<=tol:
            y,old=saved;meta.update(route='exact_response',**old)
        if y is None:
            for (previous,obs),(candidate,old) in reversed(list(self.responses.items())):
                if obs!=obskey:continue
                delta=sum(w*abs(a-b) for w,a,b in zip(AMPLITUDES,theta,previous))/1.2
                change=c.observation_dual*c.source_energy*delta/((1-rho(theta))*(1-rho(previous)))
                absolute=old['absolute_bound']+change
                lower=spectral(c.rootw*candidate)-absolute
                bound=absolute/lower if lower>0 else float('inf')
                if bound<=tol:
                    y=candidate;meta.update(route='perturbation_reuse',relative_bound=bound,
                                          absolute_bound=absolute);break
        if y is None and self.field_theta==theta and self.field_energy is not None:
            candidate=observe(self.direct.field,c.obs,c.ncols)
            absolute=c.observation_dual*self.field_energy
            lower=spectral(c.rootw*candidate)-absolute
            bound=absolute/lower if lower>0 else float('inf')
            if bound<=tol:
                y=candidate;meta.update(route='field_transport',relative_bound=bound,absolute_bound=absolute)
        meta['reuse_check_s']=perf_counter()-t
        if y is None:
            prepared=None;reason=None;t=perf_counter()
            try_update=self.arm!='direct'
            if self.arm=='threshold' and rho(theta)>self.settings['threshold_rho']:
                try_update=False;reason='threshold'
            if try_update:
                try:
                    prepared=self.taylor.inspect(theta,tol)
                    meta['graph_cache_hit']=prepared['graph_cache_hit']
                except ValueError as exc:reason=str(exc)
            meta['planning_s']=perf_counter()-t
            use_update=prepared is not None
            if self.arm=='policy' and prepared is not None:
                if self.model is None:raise ValueError('Policy requires an independently fitted cost model')
                t=perf_counter()
                tu=self.model.predict('update',prepared['features'])
                td=self.model.predict('direct',direct_features(c,self.direct,theta,tol))
                meta.update(predicted_update_s=tu,predicted_direct_s=td)
                meta['cost_estimation_s']=perf_counter()-t
                use_update=tu<=td*self.settings['update_margin']
                reason='cost'
                # Only borderline, otherwise promising updates are probed.
                if use_update and tu>td*self.settings['gray_ratio'] and prepared['missing']:
                    n,spent=self.taylor.advance(prepared,self.settings['trial_nodes'],
                                               td*self.settings['trial_budget_fraction'])
                    meta.update(trial_nodes=n,trial_s=spent,new_nodes=n)
                    remaining=len(prepared['missing'])-prepared['cursor']
                    estimate=spent/max(n,1)*remaining if remaining else 0.
                    meta['trial_remaining_estimate_s']=estimate
                    use_update=remaining==0 or (n>0 and estimate+spent<=td*self.settings['update_margin'])
                    reason='trial'
                    if not use_update:meta['abandoned_trial_s']=spent
            if use_update:
                n,spent=self.taylor.advance(prepared)
                y,result=self.taylor.finish(prepared)
                meta.update(result);meta['update_s']=spent+meta['trial_s']+result['synthesis_s']
                meta['route']=('trial_update' if meta['trial_nodes'] else
                               'sparse_update' if result['new_nodes'] else 'recombine')
            else:
                y,result=self.direct.action(theta,tol)
                self.field_theta=theta;self.field_energy=result['absolute_bound']/c.observation_dual
                meta.update(result);meta['route']=('trial_to_direct' if reason=='trial' else
                    'cost_to_direct' if reason=='cost' else 'direct' if self.arm=='direct' else 'guard_to_direct')
                meta['fallback_reason']=reason
        if not np.isfinite(meta['relative_bound']) or meta['relative_bound']>tol:
            raise RuntimeError('No certified response; refusing to return an answer')
        evidence={k:meta[k] for k in ('relative_bound','absolute_bound')}
        self.responses[key]=y,evidence
        meta['bank_bytes']=c.bank.bytes()
        meta['retained_nodes']=sum(map(len,self.taylor.values.values()))
        meta['state_bytes']=sum(u.nbytes for f in self.taylor.values.values() for u in f.values())
        meta['state_bytes']+=sum(u.nbytes for u in self.direct.field.values())
        meta['sparse_bytes']=sum(sum(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes for a in (base,*vs.values()))
                                 for _,base,vs in self.direct.parts.values())
        meta['response_bytes']=sum(v.nbytes for v,_ in self.responses.values())
        observation_arrays=list(c.row_cache.values())+list(c.dual_row_cache.values())
        for obs,w,_,_ in c._policy_obs.values():observation_arrays.extend([w,*[r for _,r in obs]])
        meta['observation_bytes']=sum(a.nbytes for a in {id(a):a for a in observation_arrays}.values())
        meta['total_s']=perf_counter()-start
        return y,meta
