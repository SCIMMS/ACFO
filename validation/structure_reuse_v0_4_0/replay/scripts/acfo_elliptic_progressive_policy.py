"""Exact support oracle and interruptible graph investigation for one elliptic family.

The cost screen bounds a nonnegative frozen cost MODEL, never wall time or error.
Only a complete ancestor closure can be sent to numerical preparation.
"""
from time import perf_counter
import numpy as np
from scripts.acfo_elliptic_structured_policy import StructuredDispatcher
from scripts.acfo_elliptic_update_policy import (
    Context, Direct, IncrementalTaylor, bind_observation, direct_features,
)
from scripts.acfo_elliptic_new_coupling import indices, SHIFTS
from scripts.acfo_elliptic_refresh import SEEDS


def reachable(alpha, mode):
    """Exact for SEEDS + t0*(2,1) + t1*(3,-2) + t2*(1,0).

    Each t_l is between -alpha_l and alpha_l with alpha_l parity.
    Eliminating t0=k+2*t1 and t2=m-2*k-7*t1 leaves an integer interval.
    """
    a,b,c=alpha
    for sm,sk in SEEDS:
        m,k=mode[0]-sm,mode[1]-sk
        if (k-a)%2 or (m-b-c)%2:continue
        low=max(-b,-((a+k)//2),-((-(m-2*k-c))//7))
        high=min(b,(a-k)//2,(m-2*k+c)//7)
        low+=(b-low)%2
        if low<=high:return True
    return False


class GraphRecord:
    def __init__(self,p,active,modes):
        self.alphas=indices(p,active)
        self.wanted={a:set() for a in self.alphas}
        self.roots=iter((a,m) for a in self.alphas for m in sorted(modes))
        self.stack=[];self.dependencies={};self.complete=False;self.materialized=False
        self.ordered=None


class GraphInvestigator:
    """Persistent structural metadata; numerical cache counts are read afresh.

    No graph or reachability cache is shared across fresh-process workloads.
    """
    def __init__(self):
        self.graphs={};self.reach={};self.last={}

    def _reachable(self,a,m):
        key=a,m
        if key not in self.reach:self.reach[key]=reachable(a,m)
        return self.reach[key]

    def describe(self,d,theta,p,modes,bound,absolute,rows=None,ceiling=None):
        active=tuple(l for l,t in enumerate(theta) if t)
        key=p,active,tuple(sorted(modes));hit=key in self.graphs
        g=self.graphs.setdefault(key,GraphRecord(p,active,modes)) if not hit else self.graphs[key]
        values=d.taylor.values;bank=d.c.bank;edges=set();factors=set();missing=[]
        features=[1.,0.,0.,0.,len(g.alphas)*(rows if rows is not None else len(d.c.obs))/10000]
        # A negative coefficient invalidates monotonic lower-bound screening.
        if ceiling is not None and any(v<0 for v in d.model.data['update']['coefficients']):
            raise ValueError('Progressive cost screen requires nonnegative model coefficients')
        start_size=len(g.dependencies);scanned=0

        def account(node,deps):
            nonlocal scanned
            scanned+=1;a,m=node
            if m in values.get(a,{}):return
            missing.append(node)
            factor=abs(m[0]),abs(m[1])
            if factor not in bank.factors:factors.add(factor)
            for l,source in deps:
                if (l,m,source) not in bank.edges and (l,source,m) not in bank.edges:
                    edges.add((l,*sorted((m,source))))

        def screened():
            features[1:4]=len(missing)/1000,len(edges)/1000,len(factors)/100
            return ceiling is not None and d.model.predict('update',features)>ceiling

        def receipt(stopped):
            self.last=dict(graph_cache_hit=hit,graph_complete=g.complete,
                graph_screened=stopped,graph_nodes_known=len(g.dependencies),
                graph_nodes_added=len(g.dependencies)-start_size,graph_nodes_scanned=scanned,
                model_lower_update_s=d.model.predict('update',features) if ceiling is not None else None,
                model_screen_ceiling_s=ceiling)

        if not g.materialized:
            for node in (g.ordered if g.complete else g.dependencies):
                account(node,g.dependencies[node])
                if scanned%128==0 and screened():
                    receipt(True);return None
        while not g.complete:
            if not g.stack:
                for node in g.roots:
                    if self._reachable(*node) and node[1] not in g.wanted[node[0]]:
                        g.stack.append(node);break
                else:
                    g.complete=True;break
            a,m=g.stack.pop()
            if m in g.wanted[a]:continue
            deps=[];g.wanted[a].add(m)
            for l in active:
                if not a[l]:continue
                prev=list(a);prev[l]-=1;prev=tuple(prev);dm,dk=SHIFTS[l]
                for source in ((m[0]+dm,m[1]+dk),(m[0]-dm,m[1]-dk)):
                    if self._reachable(prev,source):
                        deps.append((l,source))
                        if source not in g.wanted[prev]:g.stack.append((prev,source))
            g.dependencies[a,m]=tuple(deps);account((a,m),deps)
            if len(g.dependencies)>d.settings['max_nodes']:
                receipt(False);raise ValueError('Preparation node budget exceeded')
            if scanned%128==0 and screened():
                receipt(True);return None
        if g.ordered is None:
            g.ordered=[(a,m) for a in g.alphas for m in sorted(g.wanted[a])]
        if len(g.dependencies)>d.settings['max_nodes']:
            receipt(False);raise ValueError('Preparation node budget exceeded')
        stopped=screened();receipt(stopped)
        if stopped:return None
        if not missing:g.materialized=True
        # DFS discovery is not a topological numerical execution order.
        if missing:
            needed=set(missing);missing=[node for node in g.ordered if node in needed]
        d.taylor.plans[key]=g.wanted
        return dict(theta=theta,active=active,order=p,wanted=g.wanted,missing=missing,cursor=0,
            requested_nodes=len(g.dependencies),graph_cache_hit=hit,relative_bound=bound,
            absolute_bound=absolute,features=features,graph_record=g)


class ProgressiveDispatcher(StructuredDispatcher):
    def __init__(self,*args,screen=True,**kwargs):
        super().__init__(*args,**kwargs)
        self.investigator=GraphInvestigator();self.screen=screen

    def step(self,event,upcoming=()):
        self._validate(event)
        if self.mode=='remaining' and upcoming:raise ValueError('Stream policy cannot inspect future requests')
        targets=self._targets(event,upcoming)
        start=perf_counter();setup=start
        if self.c is None:
            self.c=Context(*self.args);self.taylor=IncrementalTaylor(self.c,self.settings);self.direct=Direct(self.c)
        c=self.c;key=event.get('observation','narrow');theta=tuple(event['theta'])
        m=dict(setup_s=perf_counter()-setup,planning_s=0.,reprice_s=0.,trial_s=0.,trial_nodes=0,
            abandoned_trial_s=0.,update_s=0.,direct_s=0.,new_nodes=0,observation_s=0.,reuse_check_s=0.,
            predicted_update_s=None,predicted_direct_s=None,horizon_queries=len(targets),prefetched_responses=0)
        t=perf_counter();bind_observation(c,key);m['observation_s']+=perf_counter()-t
        t=perf_counter();y,reused=self._reuse(event);m['reuse_check_s']+=perf_counter()-t
        if y is not None:
            m.update(reused);self._save(event,y,m)
        else:
            # Build only the advertised same-operator batch, never an unseen suffix.
            t=perf_counter();certificates={};modes=set();maxp=0;guard=None
            direct_only=self.mode=='batch_direct' or (self.mode=='coverage' and any(e.get('observation')=='wide' for e in targets))
            for e in targets:
                obs=e.get('observation','narrow');bind_observation(c,obs);modes.update(mode for mode,_ in c.obs)
                if not direct_only:
                    try:
                        p,bound,absolute=c.select(theta,e['tolerance'],self.settings['max_order'])
                        certificates[obs]=(p,bound,absolute);maxp=max(maxp,p)
                    except ValueError as exc:guard=str(exc)
            bind_observation(c,key);m['observation_s']+=perf_counter()-t
            prepared=None;t=perf_counter()
            rows=sum(len(c._policy_obs[e.get('observation','narrow')][0]) for e in targets)
            screened=False;ceiling=None
            if self.screen and self.mode in ('remaining','scheduled'):
                td=self.model.predict('direct',direct_features(c,self.direct,theta,min(e['tolerance'] for e in targets)))
                ceiling=td*self.settings['update_margin'];m['predicted_direct_s']=td
            if not direct_only and guard is None:
                try:
                    prepared=self.investigator.describe(self,theta,maxp,modes,*certificates[key][1:],rows=rows,ceiling=ceiling)
                    m.update(self.investigator.last);screened=prepared is None
                except ValueError as exc:guard=str(exc)
            m['planning_s']=perf_counter()-t
            use_update=prepared is not None;reason='guard' if guard else 'cost' if screened else 'fixed_direct'
            if use_update and self.mode in ('remaining','scheduled'):
                t=perf_counter()
                # Synthesis features account for every distinct requested response.
                rows=sum(len(c._policy_obs[e.get('observation','narrow')][0]) for e in targets)
                prepared['features'][-1]=len(prepared['wanted'])*rows/10000
                tu=self.model.predict('update',prepared['features'])
                td=self.model.predict('direct',direct_features(c,self.direct,theta,min(e['tolerance'] for e in targets)))
                m.update(predicted_update_s=tu,predicted_direct_s=td)
                use_update=tu<=td*self.settings['update_margin'];reason='cost'
                m['reprice_s']+=perf_counter()-t
                if use_update and tu>td*self.settings['gray_ratio'] and prepared['missing']:
                    n,spent=self.taylor.advance(prepared,self.settings['trial_nodes'],td*self.settings['trial_budget_fraction'])
                    m.update(trial_nodes=n,trial_s=spent,new_nodes=n)
                    t=perf_counter();remaining=self.investigator.describe(self,theta,maxp,modes,*certificates[key][1:],rows=rows)
                    remaining['features'][-1]=len(remaining['wanted'])*rows/10000
                    ru=self.model.predict('update',remaining['features'])
                    rd=self.model.predict('direct',direct_features(c,self.direct,theta,min(e['tolerance'] for e in targets)))
                    m.update(predicted_remaining_update_s=ru,predicted_remaining_direct_s=rd)
                    m['reprice_s']+=perf_counter()-t
                    # Trial is sunk in both alternatives. Reprice at CURRENT state.
                    use_update=not remaining['missing'] or ru<=rd*self.settings['update_margin']
                    reason='trial';prepared=remaining
                    if not use_update:m['abandoned_trial_s']=spent
            if use_update:
                _,spent=self.taylor.advance(prepared)
                m['new_nodes']=m['trial_nodes']+prepared['cursor'];m['update_s']=m['trial_s']+spent
                m['order']=maxp;m['requested_nodes']=prepared['requested_nodes']
                prepared['graph_record'].materialized=True
                for e in targets:
                    obs=e.get('observation','narrow');bind_observation(c,obs)
                    # The union is complete at maxp. Each observation retains its
                    # own physical norm and conservative minimum-order bound.
                    prepared['relative_bound'],prepared['absolute_bound']=certificates[obs][1:]
                    answer,evidence=self.taylor.finish(prepared);m['update_s']+=evidence['synthesis_s']
                    self._save(e,answer,evidence)
                m['route']='trial_update' if m['trial_nodes'] else 'sparse_update' if m['new_nodes'] else 'recombine'
            else:
                for e in targets:
                    bind_observation(c,e.get('observation','narrow'));answer,evidence=self._reuse(e)
                    if answer is None:
                        t=perf_counter();answer,evidence=self._direct(e);m['direct_s']+=perf_counter()-t
                    self._save(e,answer,evidence)
                m['route']='trial_to_direct' if reason=='trial' else 'cost_to_direct' if reason=='cost' else 'guard_to_direct' if guard else 'direct'
                m['fallback_reason']=guard or reason
            bind_observation(c,key);y,evidence=self.responses[(theta,key)];m.update(evidence)
            m['prefetched_responses']=len(targets)
        m['retained_nodes']=sum(map(len,self.taylor.values.values()));m['field_modes']=len(self.direct.field)
        m['bank_bytes']=c.bank.bytes()
        m['graph_records']=len(self.investigator.graphs);m['reachability_entries']=len(self.investigator.reach)
        m['retained_graph_nodes']=sum(len(g.dependencies) for g in self.investigator.graphs.values())
        m['state_bytes']=sum(v.nbytes for f in self.taylor.values.values() for v in f.values())+sum(v.nbytes for v in self.direct.field.values())
        m['total_s']=perf_counter()-start
        return y,m
