"""Paired update/application ablation on the existing annular slab physics."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import numpy as np
from scipy import special
from slab_update_adapter import SlabFactors,quadrature,root

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'validation_contracts/uniaxial_slab_shared_update_v1.json'


def boundary_reflection(q,d,branch,eps_z):
    # Independent four-boundary solve, not the round-trip formula.
    ep=2.4+.025j;freq=4+.05j;k0=root(freq**2-q*q)
    ks=root(ep*freq**2-(q*q if branch=='ordinary' else ep/eps_z*q*q))
    y=ks if branch=='ordinary' else ks/ep
    plus=np.exp(1j*ks*d);minus=1/plus
    mat=np.zeros((len(q),4,4),complex)
    mat[:,0,:]=[1,-1,-1,0]
    mat[:,1,0]=-k0;mat[:,1,1]=-y;mat[:,1,2]=y
    mat[:,2,1]=plus;mat[:,2,2]=minus;mat[:,2,3]=-1
    mat[:,3,1]=y*plus;mat[:,3,2]=-y*minus;mat[:,3,3]=-k0
    rhs=np.zeros((len(q),4,1),complex);rhs[:,0,0]=-1;rhs[:,1,0]=-k0
    return np.linalg.solve(mat,rhs)[:,0,0]


def green_reference(nodes,d,height,branch,eps_z):
    q,w=quadrature(nodes)
    x,v=np.polynomial.legendre.leggauss(nodes)
    q=np.r_[q,62.5+12.5*x];w=np.r_[w,12.5*v]  # independent longer tail
    kz=root((4+.05j)**2-q*q)
    reflect=boundary_reflection(q,d,branch,eps_z)
    step=2e-5
    rr=[boundary_reflection(q,d+j*step,branch,eps_z) for j in (-2,-1,1,2)]
    dr=(rr[0]-8*rr[1]+8*rr[2]-rr[3])/(12*step)
    rho=np.linspace(0,.94,15);heights=np.linspace(.51,.81,15)+height
    geom=special.jv(0,rho[:,None]*q[None,:])*np.exp(1j*heights[:,None]*kz[None,:])
    pref=1j/(4*np.pi)*w*q/kz
    return np.stack([geom@(pref*reflect),geom@(pref*dr),geom@(pref*reflect*1j*kz)])


def audit(fac,case,states,nodes):
    errors=[]
    for d,h in states:
        ref1=green_reference(nodes*2,d,h,case['branch'],complex(*case['eps_z']))
        ref2=green_reference(nodes*4,d,h,case['branch'],complex(*case['eps_z']))
        rho=np.linspace(0,.94,15);heights=np.linspace(.51,.81,15)
        geom=special.jv(0,rho[:,None]*fac.q[None,:])*np.exp(1j*heights[:,None]*fac.kz[None,:])
        actual=np.stack([geom@v for v in fac.weights(d,h)])
        rel=lambda a,b:[float(np.linalg.norm(x-y)/np.linalg.norm(y)) for x,y in zip(a,b)]
        errors.append(dict(state=[d,h],independent_reference_errors=rel(actual,ref2),reference_convergence=rel(ref1,ref2)))
    return errors


def worker(case,design,seed):
    rng=np.random.default_rng(seed);n=case['rings'];start=perf_counter()
    rs=np.linspace(.16,.48,n);rt=np.linspace(.12,.46,n)
    hs=np.linspace(.22,.34,n);ht=np.linspace(.31,.47,n);sw=np.linspace(.045,.061,n)/n
    q,w=quadrature(design['nodes_per_segment'])
    fac=SlabFactors(rs,rt,hs,ht,sw,q,w,case['branch'],complex(*case['eps_z']),design['nphi'])
    common=perf_counter()-start
    result=dict(case=case,seed=seed,common_prepare_seconds=common,common_retained_bytes=fac.retained_bytes,
                quadrature_nodes=len(q),harmonic_cutoff=fac.h,methods={},audit=audit(fac,case,design['states'],design['nodes_per_segment']))
    for a in result['audit']:
        if max(a['reference_convergence'])>design['reference_tolerance']:raise ArithmeticError(('reference not resolved',a))
        if a['independent_reference_errors'][0]>design['operator_tolerance']:raise ArithmeticError(('field audit',a))
        if max(a['independent_reference_errors'][1:])>design['derivative_tolerance']:raise ArithmeticError(('derivative audit',a))
    # Inputs can change per application; no cached source spectrum across calls.
    inputs=[np.ascontiguousarray(rng.normal(size=(n,design['nphi']))+1j*rng.normal(size=(n,design['nphi']))) for _ in range(design['hot_inner'])]
    truth=[fac.apply('factor_shared',fac.prepare('factor_shared',*p),inputs[0]) for p in design['states']]
    for method in rng.permutation(design['methods']):
        updates=[];hot=[];errs=[];maxbytes=0
        for j,p in enumerate(design['states']):
            start=perf_counter();plan=fac.prepare(method,*p);updates.append(perf_counter()-start)
            actual=fac.apply(method,plan,inputs[0])
            e=[float(np.linalg.norm(x-y)/np.linalg.norm(y)) for x,y in zip(actual,truth[j])]
            if max(e)>1e-10:raise ArithmeticError((method,e))
            errs.append(e);maxbytes=max(maxbytes,plan.nbytes)
            fac.apply(method,plan,inputs[-1])
            start=perf_counter()
            for f in inputs:fac.apply(method,plan,f)
            hot.append((perf_counter()-start)/len(inputs))
        totals={str(c):dict(warm_seconds=sum(updates)+c*sum(hot),cold_seconds=common+sum(updates)+c*sum(hot)) for c in design['calls_per_update']}
        result['methods'][str(method)]=dict(update_seconds=updates,apply_seconds=hot,totals=totals,
                  agreement_errors=errs,retained_bytes=fac.retained_bytes+maxbytes)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','confirm'],default='pilot')
    ap.add_argument('--worker',type=Path);ap.add_argument('--result',type=Path);args=ap.parse_args()
    design=json.loads(CONTRACT.read_text())
    if args.worker:
        task=json.loads(args.worker.read_text());r=worker(task['case'],design,task['seed'])
        args.result.write_text(json.dumps(r,indent=2,allow_nan=False));return
    out=ROOT/f'reports/acfo_uniaxial_slab_shared_update_20260906_{args.phase}_v4'
    out.mkdir(exist_ok=False,parents=True)
    (out/'protocol.json').write_bytes(CONTRACT.read_bytes())
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (out/'adapter_snapshot.py').write_bytes((ROOT/'scripts/slab_update_adapter.py').read_bytes())
    cases=design['cases'] if args.phase=='pilot' else design['cases'][-1:]
    results=[]
    for i in range(1 if args.phase=='pilot' else 5):
        for j,case in enumerate(cases):
            task=out/f'task_{i}_{j}.json';dest=out/f'worker_{i}_{j}.json'
            task.write_text(json.dumps(dict(case=case,seed=20260906+i*100+j)))
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(task),'--result',str(dest)],check=True)
            results.append(json.loads(dest.read_text()));print(f'{args.phase} {i} {case} PASS',flush=True)
    (out/'results.json').write_text(json.dumps(results,indent=2))
    files=[*out.iterdir(),CONTRACT,Path(__file__),ROOT/'scripts/slab_update_adapter.py']
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
