"""Independent calibration, frozen paired trajectories, and reference audit."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse, hashlib, json, subprocess
from time import perf_counter
import numpy as np
import psutil
from scipy.optimize import nnls
from scripts.acfo_elliptic_update_policy import (
    Context, Direct, IncrementalTaylor, Dispatcher, CostModel, bind_observation,
    direct_features, spectral, DEFAULTS,
)

OUT=ROOT/'reports/acfo_elliptic_update_policy_20260905_v1'
PILOT=ROOT/'validation_contracts/acfo_elliptic_update_policy_20260905_pilot.json'
FROZEN=PILOT.with_name('acfo_elliptic_update_policy_20260905_confirmation.json')
CODE=('scripts/acfo_elliptic_update_policy.py','scripts/benchmark_acfo_elliptic_update_policy.py',
      'tests/test_acfo_elliptic_update_policy.py','scripts/acfo_elliptic_new_coupling.py',
      'scripts/acfo_elliptic_observation_refresh.py','scripts/acfo_elliptic_refresh.py',
      'scripts/acfo_cylinder_coupling.py','scripts/acfo_elliptic_disk.py')


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def refname(e):return hashlib.sha256(json.dumps([e['theta'],e.get('observation','narrow')]).encode()).hexdigest()[:16]
def check_new(p):
    if p.exists():raise FileExistsError(p)


def initialize():
    check_new(PILOT)
    def event(g,obs='narrow',tol=1e-8,a=.12,b=-.08):
        return dict(theta=[a,b,g],observation=obs,tolerance=tol)
    cfg=dict(nr=18,per_seed=2,quadrature=145,reference_nr=26,reference_quadrature=211,
        reference_tolerance=1e-12,radial_gate=2e-10,confirmation_pairs=4,
        arms=['update','direct','threshold','policy'],settings=DEFAULTS,
        trajectories={
            'local':[event(0),event(1e-7,tol=1e-6),event(1e-7),event(.04),event(.045),
                     event(.06),event(.055,a=.13),event(.055,'radial',a=.13)],
            'expansion':[event(.12),event(.13),event(.13,'wide'),event(.14,'wide'),
                         event(.14,'radial'),event(.14),event(.14,'wide',1e-6),event(.14,'wide')],
            'strong':[event(.2),event(.28),event(.45),event(.8),event(.78),
                      event(.78,'radial'),event(.03),event(.031)]},
        calibration=[
            [event(.015,a=.10,b=-.06),event(.075,a=.10,b=-.06),event(.076,a=.10,b=-.06),
             event(.076,'wide',a=.10,b=-.06),event(.076,'radial',a=.10,b=-.06)],
            [event(.03,tol=1e-6,a=.16,b=.07),event(.18,a=.16,b=.07),event(.19,a=.16,b=.07),
             event(.19,'wide',a=.16,b=.07),event(.30,a=.16,b=.07)],
            [event(.24,a=.09,b=-.11),event(.25,'wide',a=.09,b=-.11),
             event(.25,'radial',a=.09,b=-.11),event(.52,a=.09,b=-.11)],
            [event(0,a=.15,b=.04),event(.02,a=.15,b=.04),event(.021,a=.15,b=.04),
             event(.021,'wide',a=.15,b=.04)]] ,
        scope='Fixed geometry/base/source; finite radial Galerkin certificate plus independent radial audit. No online reference data.',
        calibration_accounting='Offline training wall time reported separately; never included as zero-cost one-off deployment.',
        comparisons='Always update means update whenever common feasibility guards permit; direct retains factors/affine parts/field and all arms share response error transport.',
        baseline_oracle='Best fixed arm per complete trajectory; not a state-dependent global oracle.')
    dump(PILOT,cfg)


def calibrate():
    path=OUT/'calibration.json';check_new(path)
    cfg=read(PILOT);start=perf_counter();rows=[]
    for seq,events in enumerate(cfg['calibration']):
        cu=Context(cfg['nr'],cfg['per_seed'],cfg['quadrature']);cd=Context(cfg['nr'],cfg['per_seed'],cfg['quadrature'])
        u=IncrementalTaylor(cu,cfg['settings']);d=Direct(cd)
        for event in events:
            theta=event['theta'];tol=event['tolerance']
            bind_observation(cu,event['observation']);bind_observation(cd,event['observation'])
            row=dict(sequence=seq,event=event)
            try:
                prepared=u.inspect(theta,tol);t=perf_counter()
                u.advance(prepared);_,meta=u.finish(prepared)
                row['update']=dict(features=prepared['features'],seconds=perf_counter()-t,
                                   new_nodes=meta['new_nodes'],planning_s=prepared['planning_s'])
            except ValueError as exc:row['update_unavailable']=str(exc)
            features=direct_features(cd,d,theta,tol);t=perf_counter()
            _,meta=d.action(theta,tol)
            row['direct']=dict(features=features,seconds=perf_counter()-t,radius=meta['radius'])
            rows.append(row)
            print(json.dumps(dict(calibration=seq,theta=theta,observation=event['observation'],
                                  update=row.get('update',{}).get('seconds'),direct=row['direct']['seconds'])),flush=True)
    model={}
    for kind in ('update','direct'):
        records=[r[kind] for r in rows if kind in r]
        x=np.array([r['features'] for r in records]);y=np.array([r['seconds'] for r in records])
        coef,_=nnls(x,y)
        predicted=x@coef
        model[kind]=dict(coefficients=coef.tolist(),training_rows=len(records),
            training_median_absolute_log_ratio=float(np.median(np.abs(np.log(np.maximum(predicted,1e-6)/y)))))
    dump(path,dict(model=model,rows=rows,total_calibration_s=perf_counter()-start,
                   protocol_sha256=sha(PILOT),data_split='Only calibration sequences; all evaluation sequences held out'))


def references():
    cfg=read(PILOT);rows=[]
    events={refname(e):e for events in cfg['trajectories'].values() for e in events}
    for name,event in events.items():
        path=OUT/'references'/(name+'.npz');check_new(path);path.parent.mkdir(parents=True,exist_ok=True)
        start=perf_counter();values=[];metadata=[]
        for nr in (cfg['nr'],cfg['reference_nr']):
            c=Context(nr,cfg['per_seed'],cfg['reference_quadrature']);bind_observation(c,event['observation'])
            y,m=Direct(c).action(event['theta'],cfg['reference_tolerance'])
            values.append(y);metadata.append(m)
        error=spectral(c.rootw*(values[0]-values[1]))/spectral(c.rootw*values[1])
        np.savez_compressed(path,finite=values[0],finer=values[1],rootw=c.rootw)
        row=dict(event=event,radial_change=error,metadata=metadata,array_sha256=sha(path),
                 passed=bool(error<=cfg['radial_gate']),seconds=perf_counter()-start)
        dump(path.with_suffix('.json'),row);rows.append(row)
        print(json.dumps(dict(reference=name,error=error,seconds=row['seconds'],passed=row['passed'])),flush=True)
        if not row['passed']:raise RuntimeError('Reference radial gate failed')
    dump(OUT/'reference_summary.json',dict(rows=rows,passed=all(r['passed'] for r in rows)))


def worker(arm,pair,stage,trajectory):
    cfg=read(PILOT)
    if stage=='confirmation':
        for p,h in read(FROZEN)['hashes'].items():
            if sha(ROOT/p)!=h:raise RuntimeError('Frozen input changed: '+p)
    path=OUT/stage/f'{pair:02d}_{trajectory}_{arm}.json';check_new(path)
    model=CostModel(read(OUT/'calibration.json')['model'])
    runner=Dispatcher(arm,model,cfg['nr'],cfg['per_seed'],cfg['quadrature'],cfg['settings'])
    rows=[];answers=[];start=perf_counter()
    for event in cfg['trajectories'][trajectory]:
        y,meta=runner.step(event);mem=psutil.Process().memory_info()
        rows.append(dict(**event,**meta,process_peak_bytes=getattr(mem,'peak_wset',mem.rss)))
        answers.append(y)
    elapsed=perf_counter()-start;audit=perf_counter()
    for row,y in zip(rows,answers):
        with np.load(OUT/'references'/(refname(row)+'.npz')) as ref:
            row['relative_error']=spectral(ref['rootw']*(y-ref['finer']))/spectral(ref['rootw']*ref['finer'])
        row['passed']=bool(row['relative_error']<=row['tolerance'] and row['relative_bound']<=row['tolerance'])
    result=dict(arm=arm,pair=pair,stage=stage,trajectory=trajectory,events=rows,
        trajectory_s=elapsed,independent_audit_s=perf_counter()-audit,passed=all(r['passed'] for r in rows))
    dump(path,result)
    print(json.dumps(dict(arm=arm,pair=pair,trajectory=trajectory,seconds=elapsed,
                          routes=[r['route'] for r in rows],passed=result['passed'])),flush=True)
    if not result['passed']:raise RuntimeError('Accuracy gate failed')


def batch(stage):
    cfg=read(PILOT)
    if stage=='confirmation' and not FROZEN.exists():raise RuntimeError('Freeze first')
    for pair in range(1 if stage=='pilot' else cfg['confirmation_pairs']):
        arms=cfg['arms'][pair:]+cfg['arms'][:pair]
        if pair%2:arms=list(reversed(arms))
        for trajectory in cfg['trajectories']:
            for arm in arms:
                subprocess.run([sys.executable,str(Path(__file__)),'--worker',arm,'--pair',str(pair),
                                '--stage',stage,'--trajectory',trajectory],cwd=ROOT,check=True)


def freeze():
    check_new(FROZEN);cfg=read(PILOT);pilots=list((OUT/'pilot').glob('*.json'))
    if len(pilots)!=len(cfg['arms'])*len(cfg['trajectories']) or not all(read(p)['passed'] for p in pilots):
        raise RuntimeError('Passing complete pilot required')
    files=[ROOT/p for p in CODE]+[PILOT,OUT/'unit_tests.xml',OUT/'calibration.json',OUT/'reference_summary.json']
    files+=list((OUT/'references').glob('*'))
    dump(FROZEN,dict(protocol=cfg,hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files}))
    print('Frozen confirmation inputs')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for mode in ('initialize','calibrate','reference','pilot','freeze','confirm'):
        p.add_argument('--'+mode,action='store_true')
    p.add_argument('--worker');p.add_argument('--pair',type=int,default=0)
    p.add_argument('--stage',default='pilot');p.add_argument('--trajectory')
    a=p.parse_args()
    if a.initialize:initialize()
    elif a.calibrate:calibrate()
    elif a.reference:references()
    elif a.pilot:batch('pilot')
    elif a.freeze:freeze()
    elif a.confirm:batch('confirmation')
    elif a.worker:worker(a.worker,a.pair,a.stage,a.trajectory)
