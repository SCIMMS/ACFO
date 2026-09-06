"""Same-index, same-executor comparison against generic graph slicing."""
from time import perf_counter
import hashlib
import json
from scripts.acfo_elliptic_new_coupling import indices, SHIFTS
from scripts.acfo_elliptic_refresh import SEEDS
from scripts.acfo_elliptic_progressive_policy import reachable
from scripts.acfo_elliptic_update_policy import Context, IncrementalTaylor, bind_observation
from scripts.acfo_elliptic_weak_coupling import select_cap

ARMS = ('forward', 'generic_query', 'algebra_query')

def predecessors(alpha, mode):
    for l, n in enumerate(alpha):
        if n:
            b = list(alpha); b[l] -= 1
            for sign in (-1, 1):
                yield tuple(b), tuple(x + sign*s for x,s in zip(mode,SHIFTS[l]))

def forward_support(alphas):
    support = {(0,0,0): set(SEEDS)}
    for a in alphas[1:]:
        support[a] = set()
        for l,n in enumerate(a):
            if n:
                b=list(a); b[l]-=1
                for m in support[tuple(b)]:
                    for sign in (-1,1):
                        support[a].add(tuple(x+sign*s for x,s in zip(m,SHIFTS[l])))
    return support

def graph_digest(wanted):
    return hashlib.sha256(json.dumps([(a,sorted(v)) for a,v in wanted.items()]).encode()).hexdigest()

def make_graph(arm, alphas, output_modes):
    """Generic and algebra arms share exact backward slicing and sorting.

    Only the membership predicate differs: forward-set membership versus the
    specialized integer oracle. No dense fields, padded rectangles or numeric
    zero tests are imposed on the generic baseline.
    """
    if arm not in ARMS: raise ValueError(arm)
    fwd = None if arm == 'algebra_query' else forward_support(alphas)
    checks = 0; cache = {}
    def contains(a,m):
        nonlocal checks
        checks += 1
        if fwd is not None: return m in fwd[a]
        if (a,m) not in cache: cache[a,m] = reachable(a,m)
        return cache[a,m]
    if arm == 'forward': wanted = fwd
    else:
        wanted = {a:set() for a in alphas}
        stack = [(a,m) for a in alphas for m in sorted(output_modes) if contains(a,m)]
        while stack:
            a,m = stack.pop()
            if m in wanted[a]: continue
            wanted[a].add(m)
            for b,n in predecessors(a,m):
                if contains(b,n) and n not in wanted[b]: stack.append((b,n))
    return wanted, dict(membership_checks=checks, oracle_entries=len(cache),
        forward_metadata_nodes=sum(map(len,fwd.values())) if fwd is not None else 0,
        graph_nodes=sum(map(len,wanted.values())))

def numerical_digest(engine):
    h=hashlib.sha256()
    for a,f in sorted(engine.values.items()):
        for m,v in sorted(f.items()):
            h.update(repr((a,m,v.shape)).encode());h.update(v.tobytes())
    return h.hexdigest()

class Runner:
    def __init__(self, arm, nr=18, per_seed=2, quadrature=145):
        self.arm=arm; self.c=Context(nr,per_seed,quadrature)
        self.engine=IncrementalTaylor(self.c)

    def step(self,event,arm=None):
        start=perf_counter(); c=self.c;theta=tuple(event['theta'])
        t=perf_counter();bind_observation(c,event['observation']);obs_s=perf_counter()-t
        t=perf_counter();p,q,bound,absolute=select_cap(c,theta,event['tolerance'],10)
        alphas=[a for a in indices(p,tuple(l for l,v in enumerate(theta) if v)) if a[2]<=q]
        selection_s=perf_counter()-t
        t=perf_counter();wanted,meta=make_graph(arm or self.arm,alphas,{m for m,_ in c.obs})
        missing=[(a,m) for a,nodes in wanted.items() for m in sorted(nodes) if m not in self.engine.values.get(a,{})]
        prepared=dict(wanted=wanted,missing=missing,cursor=0,theta=theta,
            active=tuple(l for l,v in enumerate(theta) if v),order=p,
            relative_bound=bound,absolute_bound=absolute,requested_nodes=meta['graph_nodes'])
        planning_s=perf_counter()-t
        before=c.bank.factor_solves
        _,numeric_s=self.engine.advance(prepared)
        y,evidence=self.engine.finish(prepared)
        meta.update(evidence,p=p,q=q,alpha_count=len(alphas),observation_s=obs_s,
            selection_s=selection_s,planning_s=planning_s,numeric_s=numeric_s,
            solve_calls=c.bank.factor_solves-before,
            bank_bytes=c.bank.bytes(),state_bytes=sum(v.nbytes for f in self.engine.values.values() for v in f.values()),
            total_s=perf_counter()-start)
        # Offline structural receipt is excluded from recorded step time.
        meta['graph_sha256']=graph_digest(wanted)
        return y,meta
