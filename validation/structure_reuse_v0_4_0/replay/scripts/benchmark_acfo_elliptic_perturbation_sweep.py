"""Fresh-process perturbation sweep; every point starts at gamma zero."""
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,subprocess
from time import perf_counter
import numpy as np
import psutil
from scripts.acfo_elliptic_perturbation_sweep import event,forecast,boundaries,warm_fingerprint
from scripts.acfo_elliptic_weak_coupling import WeakCouplingDispatcher
from scripts.acfo_elliptic_progressive_policy import ProgressiveDispatcher
from scripts.acfo_elliptic_structured_policy import StructuredDispatcher
from scripts.acfo_elliptic_update_policy import CostModel,Context,Direct,bind_observation,spectral
from scripts.benchmark_acfo_elliptic_weak_coupling import CODE as OLD_CODE
from scripts.benchmark_acfo_elliptic_update_policy import read,dump,sha,refname,check_new

OUT=ROOT/'reports/acfo_elliptic_perturbation_sweep_20260905_v1'
PILOT=ROOT/'validation_contracts/acfo_elliptic_perturbation_sweep_20260905_pilot.json'
FROZEN=PILOT.with_name('acfo_elliptic_perturbation_sweep_20260905_confirmation.json')
PROSPECTIVE=OUT/'prospective_contract.json'
CODE=(*OLD_CODE,'scripts/acfo_elliptic_perturbation_sweep.py',
      'scripts/benchmark_acfo_elliptic_perturbation_sweep.py','tests/test_acfo_elliptic_perturbation_sweep.py')
ARMS=['total_update','cap_update','cap_policy','direct']


def dispatcher(arm,cfg):
    model=CostModel(read(ROOT/cfg['model'])['model']);args=cfg['nr'],cfg['per_seed'],cfg['quadrature']
    if arm=='total_update':return ProgressiveDispatcher('batch_update',model,*args,screen=False)
    if arm=='direct':return StructuredDispatcher('batch_direct',model,*args)
    return WeakCouplingDispatcher(arm,model,*args)


def initialize():
    check_new(PILOT);check_new(PROSPECTIVE)
    cfg=dict(nr=18,per_seed=2,quadrature=145,reference_nr=26,reference_quadrature=211,
        reference_tolerance=1e-12,radial_gate=2e-10,pairs=4,arms=ARMS,cases={},
        model='reports/acfo_elliptic_update_policy_20260905_v1/calibration.json',
        information='Only current request; each fresh worker prepares gamma=0 at the target tolerance, then one target gamma. No cross-point cache.',
        timing='Warm setup and target refresh recorded separately and added. Offline state fingerprint and reference audit excluded and separately timed.',
        predictions='Reuse, chosen p/q, preparation guard predicted using gamma-zero response only. Positive/negative runtimes are measured separately.',
        acceptance=dict(accuracy='Every target and base pass 1e-6 or 1e-8; reference gate and frozen predictions checked.',
            state='Identical warm numerical fingerprint within each arm/tolerance across points and pairs.',
            positive='At least two adjacent fixed-log-grid nonzero points need correction by independent omission error and beat total_update and direct in all four refresh pairs using cap_update.',
            boundaries='Prospective +/-5% reuse, q<=1 selector, and preparation-guard boundary probes match their forecast and all accuracy gates.',
            cost='Report every disagreement between cost policy and fastest measured baseline; no universal runtime-optimality requirement.'),
        scope='Existing wide observation and fixed elliptic family; no new geometry or physics; no new cost fit.')
    start=perf_counter();predictions={};thresholds={}
    grid=[0.,1e-8,1e-6,1e-5,1e-4,1e-3,.01,.12,.28,.8]
    for power in (6,8):
        tol=10.**(-power);d=dispatcher('cap_update',cfg);y,m=d.step(event(0,tol))
        limits=boundaries(d.c,y,m,tol);thresholds[str(power)]=limits
        points=[(g,'log') for g in grid]
        points += [(g*factor,name+suffix) for name,g in limits.items() for factor,suffix in ((.95,'_below'),(1.05,'_above'))]
        for i,(gamma,tag) in enumerate(sorted(points)):
            key=f'e{power}_{i:02d}';e=event(gamma,tol)
            cfg['cases'][key]=dict(event=e,tag=tag,pilot=(gamma in (1e-5,.28) or tag in ('reuse_below','one_layer_above','preparation_above')))
            predictions[key]=forecast(d.c,y,m,gamma,tol)
    dump(PILOT,cfg)
    dump(OUT/'forecast.json',dict(thresholds=thresholds,cases=predictions,construction_s=perf_counter()-start,
        note='Constructed before target references or pilot. Selector boundaries are sufficient-method boundaries, not minimal physical correction thresholds.'))
    files=[ROOT/p for p in CODE]+[PILOT,OUT/'forecast.json',ROOT/cfg['model']]
    dump(PROSPECTIVE,dict(hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files}))
    print(json.dumps(dict(cases=len(cfg['cases']),thresholds=thresholds),indent=2))


def check_prospective():
    for p,h in read(PROSPECTIVE)['hashes'].items():
        if sha(ROOT/p)!=h:raise RuntimeError('Prospective input changed: '+p)


def references():
    check_prospective();cfg=read(PILOT)
    known=read(ROOT/'reports/acfo_elliptic_weak_coupling_20260905_v1/references.json')
    es={refname(r['event']):r['event'] for r in cfg['cases'].values()};refs={}
    for key,e in es.items():
        if key in known:
            row=known[key];assert sha(ROOT/row['path'])==row['sha256'];refs[key]=dict(row,reused=True);continue
        p=OUT/'references'/(key+'.npz');check_new(p);p.parent.mkdir(parents=True,exist_ok=True)
        start=perf_counter();values=[];metadata=[]
        for nr in (cfg['nr'],cfg['reference_nr']):
            c=Context(nr,cfg['per_seed'],cfg['reference_quadrature']);bind_observation(c,'wide')
            y,m=Direct(c).action(e['theta'],cfg['reference_tolerance']);values.append(y);metadata.append(m)
        error=spectral(c.rootw*(values[0]-values[1]))/spectral(c.rootw*values[1])
        np.savez_compressed(p,finite=values[0],finer=values[1],rootw=c.rootw)
        passed=error<=cfg['radial_gate'] and all(m['relative_bound']<=cfg['reference_tolerance'] for m in metadata)
        dump(p.with_suffix('.json'),dict(event=e,radial_change=error,metadata=metadata,passed=passed,seconds=perf_counter()-start))
        if not passed:raise RuntimeError('Reference gate failed')
        refs[key]=dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),reused=False)
        print(json.dumps(dict(reference=key,gamma=e['theta'][2],passed=passed)),flush=True)
    dump(OUT/'references.json',refs)


def worker(case,arm,pair,stage):
    check_prospective();cfg=read(PILOT)
    if stage=='confirmation':
        for p,h in read(FROZEN)['hashes'].items():
            if sha(ROOT/p)!=h:raise RuntimeError('Frozen input changed: '+p)
    path=OUT/stage/f'{pair:02d}_{case}_{arm}.json';check_new(path)
    e=cfg['cases'][case]['event'];base=event(0,e['tolerance']);d=dispatcher(arm,cfg)
    start=perf_counter();yb,mb=d.step(base);warm_s=perf_counter()-start
    start=perf_counter();fingerprint=warm_fingerprint(d);state_audit_s=perf_counter()-start
    start=perf_counter();y,m=d.step(e);memory=psutil.Process().memory_info();refresh_s=perf_counter()-start
    m['process_peak_bytes']=getattr(memory,'peak_wset',memory.rss)
    start=perf_counter();refs=read(OUT/'references.json');answers=[]
    for candidate,ev,meta in ((yb,base,mb),(y,e,m)):
        rr=refs[refname(ev)];rp=ROOT/rr['path'];assert sha(rp)==rr['sha256']
        with np.load(rp) as r:
            norm=spectral(r['rootw']*r['finer'])
            meta['relative_error']=spectral(r['rootw']*(candidate-r['finer']))/norm
            if ev is e:
                rb=refs[refname(base)]
                with np.load(ROOT/rb['path']) as br:
                    meta['ideal_omission_error']=spectral(r['rootw']*(br['finer']-r['finer']))/norm
        meta['passed']=meta['relative_error']<=ev['tolerance'] and meta['relative_bound']<=ev['tolerance']
        answers.append(dict(**ev,**meta))
    pred=read(OUT/'forecast.json')['cases'][case]
    prediction_match=None
    if arm in ('cap_update','cap_policy'):
        if pred['phase']=='correction':prediction_match=(m.get('p'),m.get('q'))==(pred['p'],pred['q'])
        elif pred['phase']=='preparation_guard':prediction_match=m['route']=='guard_to_direct'
        elif pred['phase']=='exact_reuse':prediction_match=m['route']=='exact_response'
        else:prediction_match=m['route']=='perturbation_reuse'
    passed=all(r['passed'] for r in answers) and prediction_match is not False
    result=dict(case=case,arm=arm,pair=pair,stage=stage,warm_s=warm_s,refresh_s=refresh_s,
        total_s=warm_s+refresh_s,state_audit_s=state_audit_s,independent_audit_s=perf_counter()-start,
        warm_fingerprint=fingerprint,events=answers,prediction_match=prediction_match,passed=passed)
    dump(path,result)
    print(json.dumps(dict(case=case,arm=arm,pair=pair,refresh_s=refresh_s,route=m['route'],passed=passed)),flush=True)
    if not passed:raise RuntimeError('Accuracy/prediction gate failed')


def orders(arms):
    other=arms[len(arms)//2:]+arms[:len(arms)//2]
    return [arms,list(reversed(arms)),other,list(reversed(other))]


def batch(stage):
    cfg=read(PILOT)
    for pair in range(1 if stage=='pilot' else cfg['pairs']):
        cases=[k for k,r in cfg['cases'].items() if stage!='pilot' or r['pilot']]
        # Alternate increasing/decreasing perturbation order across fresh pairs.
        if pair%2:cases=list(reversed(cases))
        for case in cases:
            for arm in orders(cfg['arms'])[pair]:
                subprocess.run([sys.executable,str(Path(__file__)),'--worker',arm,'--case',case,
                    '--pair',str(pair),'--stage',stage],cwd=ROOT,check=True)


def freeze():
    check_prospective();check_new(FROZEN);cfg=read(PILOT);rows=[read(p) for p in (OUT/'pilot').glob('*.json')]
    assert len(rows)==sum(r['pilot'] for r in cfg['cases'].values())*len(ARMS) and all(r['passed'] for r in rows)
    files=[ROOT/p for p in CODE]+[PILOT,PROSPECTIVE,OUT/'forecast.json',OUT/'unit_tests.xml',OUT/'references.json',ROOT/cfg['model']]
    files+=list((OUT/'references').glob('*'))+[ROOT/r['path'] for r in read(OUT/'references.json').values()]
    dump(FROZEN,dict(protocol=cfg,hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files}))
    print('Frozen common-warm-state perturbation sweep')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for mode in ('initialize','reference','pilot','freeze','confirm'):p.add_argument('--'+mode,action='store_true')
    p.add_argument('--worker');p.add_argument('--case');p.add_argument('--pair',type=int,default=0);p.add_argument('--stage',default='pilot')
    a=p.parse_args()
    if a.initialize:initialize()
    elif a.reference:references()
    elif a.pilot:batch('pilot')
    elif a.freeze:freeze()
    elif a.confirm:batch('confirmation')
    elif a.worker:worker(a.case,a.worker,a.pair,a.stage)
