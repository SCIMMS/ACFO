"""Same-state branch diagnostics; no changes to the frozen online dispatcher."""
from time import perf_counter
import copy
import numpy as np
from scripts.acfo_elliptic_update_policy import (
    Context, Direct, IncrementalTaylor, Dispatcher, CostModel, bind_observation,
    direct_features, SHIFTS, plus, minus,
)


def initialized(model, nr=18, per_seed=2, quadrature=145):
    d=Dispatcher('update',model,nr,per_seed,quadrature)
    d.c=Context(nr,per_seed,quadrature)
    d.taylor=IncrementalTaylor(d.c,d.settings);d.direct=Direct(d.c)
    bind_observation(d.c,'narrow')
    return d


def state_counts(d):
    c=d.c
    return dict(coefficient_nodes=sum(map(len,d.taylor.values.values())),
        factors=len(c.bank.factors),edges=len(c.bank.edges),graphs=len(d.taylor.plans),
        field_modes=len(d.direct.field),csr_regions=len(d.direct.parts),
        responses=len(d.responses),observation_sets=len(c._policy_obs))


def checkpoint(model, spec, nr=18, per_seed=2, quadrature=145):
    start=perf_counter();d=initialized(model,nr,per_seed,quadrature);prefix=[]
    for item in spec.get('prefix',[]):
        d.arm=item['arm'];_,meta=d.step(item['event']);prefix.append(meta)
    bind_observation(d.c,spec['event']['observation'])
    trial=None
    if spec.get('trial_nodes',0):
        prepared=d.taylor.inspect(spec['event']['theta'],spec['event']['tolerance'])
        n,seconds=d.taylor.advance(prepared,count=spec['trial_nodes'])
        trial=dict(nodes=n,seconds=seconds,planning_s=prepared['planning_s'])
    return d,dict(prefix=prefix,trial=trial,common_preparation_s=perf_counter()-start,
                  counts=state_counts(d))


def branch(d, spec, first_arm, suffix_arm):
    """Both branches keep all common cheap reuse checks and certification."""
    start=perf_counter();answers=[];rows=[]
    d.arm=first_arm
    y,m=d.step(spec['event']);answers.append(y);rows.append(m)
    for event in spec.get('suffix',[]):
        d.arm=suffix_arm;y,m=d.step(event);answers.append(y);rows.append(m)
    return answers,dict(events=rows,total_s=perf_counter()-start,
        first_s=rows[0]['total_s'],suffix_s=sum(r['total_s'] for r in rows[1:]),
        final_counts=state_counts(d))


def prewarm_bank(d,prepared):
    """Relocate required bank creation only; charge this cost separately.

    Nodes/response/field are untouched. This is a mechanism ablation, not free
    prior preparation and not a proposed speedup against the cold workload.
    """
    start=perf_counter();bank=d.c.bank;before=state_counts(d)
    for a,target in prepared['missing']:
        key=tuple(map(abs,target))
        if key not in bank.factors:bank.solve(target,np.zeros((d.c.nr,1)))
        for l in prepared['active']:
            if not a[l]:continue
            prev=list(a);prev[l]-=1;prev=tuple(prev)
            for source in (plus(target,SHIFTS[l]),minus(target,SHIFTS[l])):
                if source in prepared['wanted'][prev]:bank.edge(l,target,source)
    return dict(seconds=perf_counter()-start,before=before,after=state_counts(d))


def profile(d,event,warm=False,chunk=64):
    start=perf_counter();u=d.taylor;c=d.c
    prepared=u.inspect(event['theta'],event['tolerance']);planning=perf_counter()-start
    model=d.model;predicted_full=model.predict('update',prepared['features'])
    predicted_direct=model.predict('direct',direct_features(c,d.direct,event['theta'],event['tolerance']))
    warming=prewarm_bank(d,prepared) if warm else dict(seconds=0.)
    blocks=[];remaining_prediction=None;remaining_estimation_s=0.;naive=None
    while prepared['cursor']<len(prepared['missing']):
        before=state_counts(d);n,elapsed=u.advance(prepared,count=chunk);after=state_counts(d)
        blocks.append(dict(nodes=n,seconds=elapsed,new_factors=after['factors']-before['factors'],
                           new_edges=after['edges']-before['edges'],cursor=prepared['cursor']))
        if len(blocks)==1:
            # Diagnostic only: this estimate is recorded, never used to choose
            # the branch or terminate the experiment.
            naive=elapsed/n*(len(prepared['missing'])-prepared['cursor'])
            t=perf_counter();remaining=u.inspect(event['theta'],event['tolerance'])
            remaining_prediction=model.predict('update',remaining['features'])
            remaining_estimation_s=perf_counter()-t
    y,meta=u.finish(prepared)
    remainder=sum(b['seconds'] for b in blocks[1:])+meta['synthesis_s']
    total=sum(b['seconds'] for b in blocks)+meta['synthesis_s']
    return y,dict(blocks=blocks,prewarm=warming,planning_s=planning,
        computation_s=total,charged_total_s=planning+warming['seconds']+total,
        diagnostic_remaining_estimation_s=remaining_estimation_s,
        profiled_wall_s=perf_counter()-start,naive_remaining_s=naive,
        structural_remaining_s=remaining_prediction,actual_remaining_s=remainder,
        predicted_full_s=predicted_full,predicted_direct_s=predicted_direct,
        relative_bound=meta['relative_bound'],absolute_bound=meta['absolute_bound'],
        initial_missing_nodes=len(prepared['missing']),final_counts=state_counts(d))
