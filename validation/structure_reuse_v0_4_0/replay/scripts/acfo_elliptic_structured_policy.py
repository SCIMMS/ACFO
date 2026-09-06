"""Current-state remainder pricing and declared same-operator observation batches."""
from time import perf_counter
import numpy as np
from scripts.acfo_elliptic_update_policy import (
    Dispatcher,Context,Direct,IncrementalTaylor,bind_observation,direct_features,
    spectral,rho,AMPLITUDES,SHIFTS,plus,minus,observe,
)
from scripts.acfo_elliptic_new_coupling import plan

MODES=('remaining','scheduled','batch_update','batch_direct','coverage')


def describe(d,theta,p,modes,bound,absolute):
    """A single union graph, priced against actual current coefficient/bank caches."""
    c=d.c;u=d.taylor;active=tuple(l for l,t in enumerate(theta) if t)
    key=p,active,tuple(sorted(modes));hit=key in u.plans
    wanted=u.plans[key] if hit else plan(p,active,set(modes),True)
    size=sum(map(len,wanted.values()))
    if size>d.settings['max_nodes']:raise ValueError('Preparation node budget exceeded')
    u.plans[key]=wanted
    missing=[(a,m) for a,nodes in wanted.items() for m in sorted(nodes) if m not in u.values.get(a,{})]
    edges=set();factors=set()
    for a,m in missing:
        if tuple(map(abs,m)) not in c.bank.factors:factors.add(tuple(map(abs,m)))
        for l in active:
            if not a[l]:continue
            prev=list(a);prev[l]-=1;prev=tuple(prev)
            for source in (plus(m,SHIFTS[l]),minus(m,SHIFTS[l])):
                if source not in wanted[prev]:continue
                if (l,m,source) not in c.bank.edges and (l,source,m) not in c.bank.edges:
                    edges.add((l,*sorted((m,source))))
    return dict(theta=theta,active=active,order=p,wanted=wanted,missing=missing,cursor=0,
        requested_nodes=size,graph_cache_hit=hit,relative_bound=bound,absolute_bound=absolute,
        features=[1.,len(missing)/1000,len(edges)/1000,len(factors)/100,len(wanted)*len(c.obs)/10000])


class StructuredDispatcher(Dispatcher):
    def __init__(self,mode,model,nr=18,per_seed=2,quadrature=145,settings=None):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('policy',model,nr,per_seed,quadrature,settings)
        self.mode=mode

    def _validate(self,e):
        if set(e)-{'theta','tolerance','observation'}:raise ValueError('Unsupported physical change')
        if len(e['theta'])!=3 or not np.all(np.isfinite(e['theta'])):raise ValueError('Invalid theta')
        if not np.isfinite(e['tolerance']) or e['tolerance']<=0:raise ValueError('Invalid tolerance')
        rho(e['theta'])

    def _reuse(self,e):
        c=self.c;theta=tuple(e['theta']);obs=e.get('observation','narrow');tol=e['tolerance']
        saved=self.responses.get((theta,obs))
        if saved and saved[1]['relative_bound']<=tol:return saved[0],dict(saved[1],route='exact_response')
        for (previous,key),(y,old) in reversed(list(self.responses.items())):
            if key!=obs:continue
            delta=sum(w*abs(a-b) for w,a,b in zip(AMPLITUDES,theta,previous))/1.2
            absolute=old['absolute_bound']+c.observation_dual*c.source_energy*delta/((1-rho(theta))*(1-rho(previous)))
            lower=spectral(c.rootw*y)-absolute
            if lower>0 and absolute/lower<=tol:
                return y,dict(relative_bound=absolute/lower,absolute_bound=absolute,route='perturbation_reuse')
        if self.field_theta==theta and self.field_energy is not None:
            y=observe(self.direct.field,c.obs,c.ncols);absolute=c.observation_dual*self.field_energy
            lower=spectral(c.rootw*y)-absolute
            if lower>0 and absolute/lower<=tol:
                return y,dict(relative_bound=absolute/lower,absolute_bound=absolute,route='field_transport')
        return None,None

    def _save(self,e,y,m):
        if not np.isfinite(m['relative_bound']) or m['relative_bound']>e['tolerance']:
            raise RuntimeError('Refusing uncertified response')
        self.responses[(tuple(e['theta']),e.get('observation','narrow'))]=(y,
            {k:m[k] for k in ('relative_bound','absolute_bound')})

    def _direct(self,e):
        # Direct may partially mutate its field before raising. Old evidence
        # must never be attached to that field during a subsequent request.
        self.field_theta=self.field_energy=None
        y,m=self.direct.action(tuple(e['theta']),e['tolerance'])
        self.field_theta=tuple(e['theta']);self.field_energy=m['absolute_bound']/self.c.observation_dual
        return y,m

    def _targets(self,event,upcoming):
        targets={event.get('observation','narrow'):dict(event)}
        for e in upcoming:
            self._validate(e)
            if tuple(e['theta'])!=tuple(event['theta']):raise ValueError('Declared batch must keep operator fixed')
            key=e.get('observation','narrow')
            if key not in targets or e['tolerance']<targets[key]['tolerance']:targets[key]=dict(e)
        return list(targets.values())

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
            if not direct_only and guard is None:
                try:prepared=describe(self,theta,maxp,modes,*certificates[key][1:])
                except ValueError as exc:guard=str(exc)
            m['planning_s']=perf_counter()-t
            use_update=prepared is not None;reason='guard' if guard else 'fixed_direct'
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
                    t=perf_counter();remaining=describe(self,theta,maxp,modes,*certificates[key][1:])
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
        m['state_bytes']=sum(v.nbytes for f in self.taylor.values.values() for v in f.values())+sum(v.nbytes for v in self.direct.field.values())
        m['total_s']=perf_counter()-start
        return y,m
