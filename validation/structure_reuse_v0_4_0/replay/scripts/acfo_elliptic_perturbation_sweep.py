"""Prospective accuracy regimes for the frozen weak-coupling adapter.

Forecasts use only a gamma-zero prepared response, physical bounds and the
existing selector. They do not predict the optimum runtime branch.
"""
import hashlib
import numpy as np
from scripts.acfo_elliptic_weak_coupling import select_cap
from scripts.acfo_elliptic_update_policy import spectral
from scripts.acfo_elliptic_new_coupling import rho


def event(gamma,tolerance):
    return dict(theta=[.12,-.08,float(gamma)],tolerance=float(tolerance),observation='wide')


def forecast(c,base_y,base_evidence,gamma,tolerance,max_order=10):
    theta=(.12,-.08,float(gamma));r=rho(theta);rb=rho((.12,-.08,0))
    weak=.3*abs(gamma)/1.2
    absolute=base_evidence['absolute_bound']+c.observation_dual*c.source_energy*weak/((1-r)*(1-rb))
    lower=spectral(c.rootw*base_y)-absolute
    reuse_bound=absolute/lower if lower>0 else float('inf')
    if gamma==0:return dict(phase='exact_reuse',p=None,q=None,reuse_bound=reuse_bound,eta_upper=0.)
    if reuse_bound<=tolerance:
        return dict(phase='perturbation_reuse',p=None,q=None,reuse_bound=reuse_bound,eta_upper=weak/(1-rb))
    try:
        p,q,bound,absolute=select_cap(c,theta,tolerance,max_order)
        return dict(phase='correction',p=p,q=q,relative_bound=bound,absolute_bound=absolute,
            reuse_bound=reuse_bound,eta_upper=weak/(1-rb))
    except ValueError as exc:
        return dict(phase='preparation_guard',p=None,q=None,reason=str(exc),reuse_bound=reuse_bound,eta_upper=weak/(1-rb))


def bisect_last_true(predicate,upper=.8,steps=44):
    if not predicate(0.) or predicate(upper):raise ValueError('Boundary is not bracketed')
    lo,hi=0.,upper
    for _ in range(steps):
        mid=(lo+hi)/2
        if predicate(mid):lo=mid
        else:hi=mid
    return (lo+hi)/2


def boundaries(c,y,evidence,tolerance):
    def prediction(g):return forecast(c,y,evidence,g,tolerance)
    def one_layer(g):
        r=prediction(g)
        return r['phase'] in ('exact_reuse','perturbation_reuse') or (r['phase']=='correction' and r['q']<=1)
    # The q boundary is a boundary of the frozen index-count selector, not of
    # all possible representations. The local bracket is verified below.
    predicates=dict(reuse=lambda g:prediction(g)['phase'] in ('exact_reuse','perturbation_reuse'),
                    one_layer=one_layer,preparation=lambda g:prediction(g)['phase']!='preparation_guard')
    result={name:bisect_last_true(fn) for name,fn in predicates.items()}
    for name,g in result.items():
        if not predicates[name](g*.95) or predicates[name](g*1.05):
            raise RuntimeError('Boundary neighborhood is not monotone')
    return result


def warm_fingerprint(d):
    """Offline audit of deterministic numerical state, excluding timings/ids."""
    h=hashlib.sha256()
    def arrays(name,items):
        for key,value in sorted(items):
            h.update((name+repr(key)).encode());a=np.ascontiguousarray(value)
            h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
    arrays('coefficient',[( (a,m),v) for a,f in d.taylor.values.items() for m,v in f.items()])
    arrays('field',list(d.direct.field.items()))
    arrays('base',list(d.c.bank.bases.items()))
    arrays('edge',list(d.c.bank.edges.items()))
    for key,(answer,evidence) in sorted(d.responses.items()):
        arrays('response',[(key,answer)])
        h.update(repr(sorted(evidence.items())).encode())
    return dict(sha256=h.hexdigest(),coefficient_nodes=sum(map(len,d.taylor.values.values())),
                field_modes=len(d.direct.field),factors=len(d.c.bank.factors),edges=len(d.c.bank.edges))
