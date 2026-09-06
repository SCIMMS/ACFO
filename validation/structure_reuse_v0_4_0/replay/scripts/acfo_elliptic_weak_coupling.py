"""Certified cap on weak-generator degree in the fixed three-shift family.

The coefficient recurrence stays unchanged. Only the downward-closed index
set and its positive majorant tail change. This is a validation adapter.
"""
from math import comb
from time import perf_counter
import numpy as np
from scripts.acfo_elliptic_structured_policy import StructuredDispatcher
from scripts.acfo_elliptic_progressive_policy import GraphInvestigator,GraphRecord
from scripts.acfo_elliptic_update_policy import (
    Context,Direct,IncrementalTaylor,bind_observation,direct_features,
)
from scripts.acfo_elliptic_new_coupling import indices,rho,AMPLITUDES


def majorant_tail(theta,p,q):
    """Sum outside {total degree <= p, gamma degree <= q}, no subtraction.

    Sum over the two other indices first: choose(d,c) r_weak^c r_base^(d-c).
    The Neumann words can be noncommutative; the norm majorant counts them.
    """
    r=rho(theta);base=(AMPLITUDES[0]*abs(theta[0])+AMPLITUDES[1]*abs(theta[1]))/1.2
    weak=AMPLITUDES[2]*abs(theta[2])/1.2
    return r**(p+1)/(1-r)+sum(comb(d,c)*weak**c*base**(d-c)
        for d in range(q+1,p+1) for c in range(q+1,d+1))


def select_cap(c,theta,tolerance,max_order):
    r=rho(theta);scale=c.observation_dual*c.source_energy
    lower=c.y0norm-scale*r/(1-r)
    if lower<=0:raise ValueError('Response lower bound inconclusive')
    active=tuple(l for l,t in enumerate(theta) if t);candidates=[]
    for p in range(max_order+1):
        alphas=indices(p,active)
        for q in range(p+1 if theta[2] else 1):
            absolute=scale*majorant_tail(theta,p,q)
            if absolute/lower<=tolerance:
                count=sum(a[2]<=q for a in alphas)
                candidates.append((count,p,q,absolute/lower,absolute))
                break
    if not candidates:raise ValueError('Order limit exceeded')
    # Cheap deterministic structural proxy; not an oracle for graph size/time.
    _,p,q,bound,absolute=min(candidates)
    return p,q,bound,absolute


class CappedInvestigator(GraphInvestigator):
    def __init__(self,q,reach):
        super().__init__();self.q=q;self.reach=reach

    def describe(self,d,theta,p,modes,bound,absolute,**kwargs):
        active=tuple(l for l,t in enumerate(theta) if t);key=p,active,tuple(sorted(modes))
        hit=key in self.graphs
        if not hit:
            g=GraphRecord(p,active,modes)
            g.alphas=[a for a in g.alphas if a[2]<=self.q]
            g.wanted={a:set() for a in g.alphas}
            g.roots=iter((a,m) for a in g.alphas for m in sorted(modes))
            self.graphs[key]=g
        out=super().describe(d,theta,p,modes,bound,absolute,**kwargs)
        self.last['graph_cache_hit']=hit
        if out is not None:out['graph_cache_hit']=hit
        # The inherited numeric engine does not consume taylor.plans. Its
        # uncapped keys lack q, so never publish a capped graph under those keys.
        d.taylor.plans.pop(key,None)
        return out


class WeakCouplingDispatcher(StructuredDispatcher):
    def __init__(self,mode,model,*args,**kwargs):
        if mode not in ('cap_update','cap_policy'):raise ValueError(mode)
        super().__init__('remaining',model,*args,**kwargs)
        self.weak_mode=mode;self.investigators={};self.reach={}

    def step(self,event):
        self._validate(event);start=perf_counter();setup=start
        if self.c is None:
            self.c=Context(*self.args);self.taylor=IncrementalTaylor(self.c,self.settings);self.direct=Direct(self.c)
        c=self.c;theta=tuple(event['theta']);tol=event['tolerance']
        m=dict(setup_s=perf_counter()-setup,selection_s=0.,planning_s=0.,update_s=0.,direct_s=0.,
            observation_s=0.,reuse_check_s=0.,new_nodes=0,p=None,q=None,
            predicted_direct_s=None,predicted_update_s=None)
        t=perf_counter();bind_observation(c,event.get('observation','narrow'));m['observation_s']=perf_counter()-t
        t=perf_counter();y,reuse=self._reuse(event);m['reuse_check_s']=perf_counter()-t
        if y is not None:m.update(reuse)
        else:
            guard=None;prepared=None;screened=False;t=perf_counter()
            try:
                p,q,bound,absolute=select_cap(c,theta,tol,self.settings['max_order'])
                m.update(p=p,q=q,full_degree_indices=len(indices(p,tuple(l for l,t in enumerate(theta) if t))))
            except ValueError as exc:guard=str(exc)
            m['selection_s']=perf_counter()-t;t=perf_counter()
            if guard is None:
                investigator=self.investigators.setdefault(q,CappedInvestigator(q,self.reach))
                ceiling=None
                if self.weak_mode=='cap_policy':
                    td=self.model.predict('direct',direct_features(c,self.direct,theta,tol))
                    m['predicted_direct_s']=td;ceiling=td*self.settings['update_margin']
                try:
                    prepared=investigator.describe(self,theta,p,{mode for mode,_ in c.obs},bound,absolute,ceiling=ceiling)
                    m.update(investigator.last);screened=prepared is None
                    if prepared is not None:m['predicted_update_s']=self.model.predict('update',prepared['features'])
                except ValueError as exc:guard=str(exc)
            m['planning_s']=perf_counter()-t
            if prepared is not None:
                _,spent=self.taylor.advance(prepared)
                prepared['graph_record'].materialized=True
                y,evidence=self.taylor.finish(prepared);m.update(evidence)
                m['update_s']=spent+evidence['synthesis_s']
                m['route']='weak_correction' if m['new_nodes'] and q else 'base_prepare' if m['new_nodes'] else 'recombine'
            else:
                t=perf_counter();y,evidence=self._direct(event);m['direct_s']=perf_counter()-t;m.update(evidence)
                m['route']='guard_to_direct' if guard else 'cost_to_direct'
                m['fallback_reason']=guard or 'model_screen'
        self._save(event,y,m)
        m['retained_nodes']=sum(map(len,self.taylor.values.values()))
        m['retained_nodes_by_gamma_degree']={str(q):sum(len(v) for a,v in self.taylor.values.items() if a[2]==q)
            for q in sorted({a[2] for a in self.taylor.values})}
        m['field_modes']=len(self.direct.field);m['bank_bytes']=c.bank.bytes()
        m['state_bytes']=sum(v.nbytes for f in self.taylor.values.values() for v in f.values())+sum(v.nbytes for v in self.direct.field.values())
        m['reachability_entries']=len(self.reach)
        m['retained_graph_nodes']=sum(len(g.dependencies) for i in self.investigators.values() for g in i.graphs.values())
        m['total_s']=perf_counter()-start
        return y,m
