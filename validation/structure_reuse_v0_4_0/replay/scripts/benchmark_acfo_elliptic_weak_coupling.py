"""Weak-coupling correction versus total-degree preparation and direct reuse."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,subprocess
from time import perf_counter
import numpy as np
import psutil
from scripts.acfo_elliptic_weak_coupling import WeakCouplingDispatcher
from scripts.acfo_elliptic_progressive_policy import ProgressiveDispatcher
from scripts.acfo_elliptic_structured_policy import StructuredDispatcher
from scripts.acfo_elliptic_update_policy import Context,Direct,CostModel,bind_observation,spectral
from scripts.benchmark_acfo_elliptic_progressive_policy import CODE as PREVIOUS_CODE
from scripts.benchmark_acfo_elliptic_update_policy import read,dump,sha,refname,check_new

OUT=ROOT/'reports/acfo_elliptic_weak_coupling_20260905_v1'
PILOT=ROOT/'validation_contracts/acfo_elliptic_weak_coupling_20260905_pilot.json'
FROZEN=PILOT.with_name('acfo_elliptic_weak_coupling_20260905_confirmation.json')
CODE=(*PREVIOUS_CODE,'scripts/acfo_elliptic_weak_coupling.py',
      'scripts/benchmark_acfo_elliptic_weak_coupling.py','tests/test_acfo_elliptic_weak_coupling.py')
ARMS=['total_update','cap_update','cap_policy','direct']


def event(g,obs):return dict(theta=[.12,-.08,g],tolerance=1e-8,observation=obs)


def initialize():
    check_new(PILOT)
    workloads={
        'weak_narrow':[event(g,'narrow') for g in (0,1e-5,2e-5,-1e-5,1e-4,1e-3)],
        'weak_wide':[event(g,'wide') for g in (0,1e-5,2e-5,1e-4,1e-3)],
        'coupling_transition':[event(g,'wide') for g in (0,1e-5,.01,.12,.28,.45,.8)],
        'broad_coupling':[event(g,'wide') for g in (.28,.29)],
    }
    dump(PILOT,dict(nr=18,per_seed=2,quadrature=145,reference_nr=26,reference_quadrature=211,
        reference_tolerance=1e-12,radial_gate=2e-10,confirmation_pairs=4,arms=ARMS,workloads=workloads,
        model='reports/acfo_elliptic_update_policy_20260905_v1/calibration.json',
        support='Fixed generator shifts and fixed observation norm. Cap gamma multi-index degree q as well as total degree p.',
        selector='Minimize accepted alpha-count proxy over p<=10 and q<=p; positive majorant tail, then optional frozen-cost screen.',
        scope='One fixed elliptic family. Only current request is provided. All arms retain bank/coefficients/field and response caches as applicable.',
        baseline='Total-degree update receives same arithmetic reachability and graph metadata improvements. Direct gets prior field, bank, and response reuse.',
        omission='Ideal gamma-zero fine reference is an offline diagnostic of whether correction is necessary, not a valid timed competitor.',
        ordering='A, reverse(A), rotate(A), reverse(rotate(A)): each pair of arms has 2:2 relative order.',
        acceptance='Report positive and adverse cases; first setup and all selection/preparation/certification costs included. Freeze after pilot, no new cost fit.'))


def references():
    cfg=read(PILOT);known=read(ROOT/'reports/acfo_elliptic_progressive_policy_20260905_v1/references.json')
    es={refname(e):e for seq in cfg['workloads'].values() for e in seq}
    for obs in ('narrow','wide'):
        e=event(0,obs);es[refname(e)]=e
    refs={}
    for key,e in es.items():
        if key in known:
            row=known[key];p=ROOT/row['path']
            if sha(p)!=row['sha256']:raise RuntimeError('Prior reference changed')
            refs[key]=dict(row,reused=True);continue
        p=OUT/'references'/(key+'.npz');check_new(p);p.parent.mkdir(parents=True,exist_ok=True)
        start=perf_counter();values=[];metadata=[]
        for nr in (cfg['nr'],cfg['reference_nr']):
            c=Context(nr,cfg['per_seed'],cfg['reference_quadrature']);bind_observation(c,e['observation'])
            y,m=Direct(c).action(e['theta'],cfg['reference_tolerance']);values.append(y);metadata.append(m)
        error=spectral(c.rootw*(values[0]-values[1]))/spectral(c.rootw*values[1])
        np.savez_compressed(p,finite=values[0],finer=values[1],rootw=c.rootw)
        passed=error<=cfg['radial_gate'] and all(m['relative_bound']<=cfg['reference_tolerance'] for m in metadata)
        dump(p.with_suffix('.json'),dict(event=e,radial_change=error,metadata=metadata,passed=passed,seconds=perf_counter()-start))
        if not passed:raise RuntimeError('Reference gate failed')
        refs[key]=dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),reused=False)
        print(json.dumps(dict(reference=key,gamma=e['theta'][2],obs=e['observation'],radial_error=error)),flush=True)
    dump(OUT/'references.json',refs)


def worker(workload,arm,pair,stage):
    cfg=read(PILOT)
    if stage=='confirmation':
        for path,h in read(FROZEN)['hashes'].items():
            if sha(ROOT/path)!=h:raise RuntimeError('Frozen input changed: '+path)
    path=OUT/stage/f'{pair:02d}_{workload}_{arm}.json';check_new(path)
    model=CostModel(read(ROOT/cfg['model'])['model']);args=cfg['nr'],cfg['per_seed'],cfg['quadrature']
    if arm=='total_update':d=ProgressiveDispatcher('batch_update',model,*args,screen=False)
    elif arm=='direct':d=StructuredDispatcher('batch_direct',model,*args)
    else:d=WeakCouplingDispatcher(arm,model,*args)
    rows=[];ys=[];start=perf_counter()
    for e in cfg['workloads'][workload]:
        y,m=d.step(e);ys.append(y);memory=psutil.Process().memory_info()
        rows.append(dict(**e,**m,process_peak_bytes=getattr(memory,'peak_wset',memory.rss)))
    total=perf_counter()-start;t=perf_counter();refs=read(OUT/'references.json')
    for y,r in zip(ys,rows):
        entry=refs[refname(r)];p=ROOT/entry['path']
        if sha(p)!=entry['sha256']:raise RuntimeError('Reference changed')
        base_entry=refs[refname(event(0,r['observation']))];bp=ROOT/base_entry['path']
        if sha(bp)!=base_entry['sha256']:raise RuntimeError('Base diagnostic changed')
        with np.load(p) as ref,np.load(bp) as base:
            norm=spectral(ref['rootw']*ref['finer'])
            r['relative_error']=spectral(ref['rootw']*(y-ref['finer']))/norm
            r['ideal_omission_error']=spectral(ref['rootw']*(base['finer']-ref['finer']))/norm
            labels=[m for m,_ in d.c._policy_obs[r['observation']][0]]
            outside=np.array([(m-2*k)%7 not in (0,2) for m,k in labels])
            r['new_residue_response_norm_relative']=spectral((ref['rootw']*ref['finer'])[outside])/norm
            r['old_residue_change_relative']=spectral((ref['rootw']*(base['finer']-ref['finer']))[~outside])/norm
        r['passed']=r['relative_error']<=r['tolerance'] and r['relative_bound']<=r['tolerance']
    result=dict(workload=workload,arm=arm,pair=pair,stage=stage,events=rows,total_s=total,
                independent_audit_s=perf_counter()-t,passed=all(e['passed'] for e in rows))
    dump(path,result)
    print(json.dumps(dict(workload=workload,arm=arm,pair=pair,seconds=total,
        caps=[(e.get('p'),e.get('q')) for e in rows],routes=[e['route'] for e in rows],passed=result['passed'])),flush=True)
    if not result['passed']:raise RuntimeError('Accuracy gate failed')


def orders(arms):
    other=arms[len(arms)//2:]+arms[:len(arms)//2]
    return [arms,list(reversed(arms)),other,list(reversed(other))]


def batch(stage):
    cfg=read(PILOT)
    for pair in range(1 if stage=='pilot' else cfg['confirmation_pairs']):
        for workload in cfg['workloads']:
            for arm in orders(cfg['arms'])[pair]:
                subprocess.run([sys.executable,str(Path(__file__)),'--worker',arm,'--workload',workload,
                    '--pair',str(pair),'--stage',stage],cwd=ROOT,check=True)


def freeze():
    check_new(FROZEN);cfg=read(PILOT);pilot=[read(p) for p in (OUT/'pilot').glob('*.json')]
    if len(pilot)!=len(cfg['workloads'])*len(cfg['arms']) or not all(r['passed'] for r in pilot):
        raise RuntimeError('Complete passing pilot required')
    files=[ROOT/p for p in CODE]+[PILOT,ROOT/cfg['model'],OUT/'unit_tests.xml',OUT/'references.json']
    files+=list((OUT/'references').glob('*'))+[ROOT/r['path'] for r in read(OUT/'references.json').values()]
    dump(FROZEN,dict(protocol=cfg,hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files}))
    print('Frozen weak-coupling experiment')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for mode in ('initialize','reference','pilot','freeze','confirm'):p.add_argument('--'+mode,action='store_true')
    p.add_argument('--worker');p.add_argument('--workload');p.add_argument('--pair',type=int,default=0);p.add_argument('--stage',default='pilot')
    a=p.parse_args()
    if a.initialize:initialize()
    elif a.reference:references()
    elif a.pilot:batch('pilot')
    elif a.freeze:freeze()
    elif a.confirm:batch('confirmation')
    elif a.worker:worker(a.workload,a.worker,a.pair,a.stage)
