"""End-to-end stream and declared observation-batch validation in one system."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,subprocess
from time import perf_counter
import numpy as np
import psutil
from scripts.acfo_elliptic_structured_policy import StructuredDispatcher
from scripts.acfo_elliptic_progressive_policy import ProgressiveDispatcher
from scripts.acfo_elliptic_new_coupling import rho
from scripts.acfo_elliptic_update_policy import Dispatcher,CostModel,Context,Direct,bind_observation,spectral
from scripts.benchmark_acfo_elliptic_update_policy import (
    OUT as OLD_OUT,PILOT as OLD_PILOT,CODE as OLD_CODE,read,dump,sha,refname,check_new,
)

OUT=ROOT/'reports/acfo_elliptic_progressive_policy_20260905_v1'
PILOT=ROOT/'validation_contracts/acfo_elliptic_progressive_policy_20260905_pilot.json'
FROZEN=PILOT.with_name('acfo_elliptic_progressive_policy_20260905_confirmation.json')
CODE=(*OLD_CODE,'scripts/acfo_elliptic_structured_policy.py',
      'scripts/acfo_elliptic_progressive_policy.py','scripts/benchmark_acfo_elliptic_progressive_policy.py',
      'tests/test_acfo_elliptic_progressive_policy.py')


def initialize():
    check_new(PILOT)
    cfg=read(ROOT/'validation_contracts/acfo_elliptic_structured_policy_20260905_pilot.json')
    for w in cfg['workloads'].values():
        w['arms']=(['old_remaining','fast_remaining','progressive_remaining','fast_update','direct','fast_threshold']
                   if w['kind']=='stream' else
                   ['old_scheduled','fast_scheduled','progressive_scheduled','fast_batch_update','batch_direct','fast_coverage'])
    cfg.update(policy='Exact integer support oracle, persistent dependency metadata, progressive lower bound on frozen nonnegative update cost model; screen every 128 inspected nodes.',
        baselines='Identical fast graph backend for forced update, threshold, and coverage. No-screen fast policy isolates the effect of early stopping.',
        scope='Same six frozen workloads and same fine independent references. One fixed elliptic system; no new model fit.',
        information='Declared same-theta block only for batch arms. Every stream arm receives only current event.',
        acceptance='Preserve all workloads and all adverse results; 4 balanced fresh-process paired repeats after passing pilot.')
    dump(PILOT,cfg)
    dump(OUT/'references.json',read(ROOT/'reports/acfo_elliptic_structured_policy_20260905_v1/references.json'))



def worker(workload,arm,pair,stage):
    cfg=read(PILOT)
    if stage=='confirmation':
        for path,h in read(FROZEN)['hashes'].items():
            if sha(ROOT/path)!=h:raise RuntimeError('Frozen input changed: '+path)
    path=OUT/stage/f'{pair:02d}_{workload}_{arm}.json';check_new(path)
    model=CostModel(read(ROOT/cfg['model'])['model']);args=cfg['nr'],cfg['per_seed'],cfg['quadrature']
    if arm in ('old_remaining','old_scheduled'):
        d=StructuredDispatcher(arm.removeprefix('old_'),model,*args)
    elif arm in ('direct','batch_direct'):
        d=StructuredDispatcher('batch_direct',model,*args)
    else:
        mode={'fast_update':'batch_update','fast_threshold':'batch_update',
              'fast_batch_update':'batch_update','fast_coverage':'coverage'}.get(arm,
              arm.removeprefix('fast_').removeprefix('progressive_'))
        d=ProgressiveDispatcher(mode,model,*args,screen=arm.startswith('progressive_'))
    w=cfg['workloads'][workload];rows=[];ys=[];start=perf_counter()
    for bidx,block in enumerate(w['blocks']):
        for i,e in enumerate(block):
            if arm=='fast_threshold':d.mode='batch_update' if rho(e['theta'])<=d.settings['threshold_rho'] else 'batch_direct'
            if w['kind']=='batch':y,m=d.step(e,block[i+1:])
            else:y,m=d.step(e)
            mem=psutil.Process().memory_info()
            rows.append(dict(**e,**m,block=bidx,process_peak_bytes=getattr(mem,'peak_wset',mem.rss)))
            ys.append(y)
    total=perf_counter()-start;t=perf_counter();refs=read(OUT/'references.json')
    for y,r in zip(ys,rows):
        entry=refs[refname(r)];p=ROOT/entry['path']
        if sha(p)!=entry['sha256']:raise RuntimeError('Reference changed')
        with np.load(p) as ref:r['relative_error']=spectral(ref['rootw']*(y-ref['finer']))/spectral(ref['rootw']*ref['finer'])
        r['passed']=r['relative_error']<=r['tolerance'] and r['relative_bound']<=r['tolerance']
    result=dict(workload=workload,kind=w['kind'],arm=arm,pair=pair,stage=stage,events=rows,total_s=total,
                independent_audit_s=perf_counter()-t,passed=all(r['passed'] for r in rows))
    dump(path,result)
    print(json.dumps(dict(workload=workload,arm=arm,pair=pair,seconds=total,
                          routes=[r['route'] for r in rows] if arm not in ('update','direct','threshold','old') else None,
                          passed=result['passed'])),flush=True)
    if not result['passed']:raise RuntimeError('Accuracy gate failed')


def orders(arms):
    other=arms[len(arms)//2:]+arms[:len(arms)//2]
    return [arms,list(reversed(arms)),other,list(reversed(other))]


def batch(stage):
    cfg=read(PILOT)
    for pair in range(1 if stage=='pilot' else cfg['confirmation_pairs']):
        for name,w in cfg['workloads'].items():
            for arm in orders(w['arms'])[pair]:
                subprocess.run([sys.executable,str(Path(__file__)),'--worker',arm,'--workload',name,
                                '--pair',str(pair),'--stage',stage],cwd=ROOT,check=True)


def freeze():
    check_new(FROZEN);cfg=read(PILOT);rows=[read(p) for p in (OUT/'pilot').glob('*.json')]
    if len(rows)!=sum(len(w['arms']) for w in cfg['workloads'].values()) or not all(r['passed'] for r in rows):
        raise RuntimeError('Complete passing pilot required')
    files=[ROOT/p for p in CODE]+[PILOT,ROOT/cfg['model'],OUT/'unit_tests.xml',OUT/'references.json']
    files+=list((OUT/'references').glob('*'))
    files+=[ROOT/r['path'] for r in read(OUT/'references.json').values()]
    dump(FROZEN,dict(protocol=cfg,hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files}))
    print('Frozen progressive policy inputs')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for mode in ('initialize','reference','pilot','freeze','confirm'):p.add_argument('--'+mode,action='store_true')
    p.add_argument('--worker');p.add_argument('--workload');p.add_argument('--pair',type=int,default=0);p.add_argument('--stage',default='pilot')
    a=p.parse_args()
    if a.initialize:initialize()
    elif a.reference:raise RuntimeError('All references reused by initialize')
    elif a.pilot:batch('pilot')
    elif a.freeze:freeze()
    elif a.confirm:batch('confirmation')
    elif a.worker:worker(a.workload,a.worker,a.pair,a.stage)
