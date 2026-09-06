"""Frozen A/B/C research experiments; D is in the companion inverse runner.

Example: python scripts/validate_acfo_research_abcd.py campaign --out reports/...
Every A arm is a fresh process. Builders never assemble the target matrix;
the separately timed post-preparation audit does. Results are create-only.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import threading
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import psutil
from scipy import linalg, special
from waxs_cake.finite_hankel import gauss_unit_interval
from waxs_cake.portable_cpswf import (RadialBinding, ContinuousRadialBasis,
    HankelSpectrum, PreparedRadialAction, robust_svd)
from acfo_research_adapters import (BlockedHankel, probe_error, randomized_prepare,
    zernike_samples, zernike_images, ball_coefficients, sonine_image)

CONTRACT = ROOT/'validation_contracts/acfo_research_abcd_v1.json'


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)


def freeze(out):
    files = [CONTRACT, Path(__file__), Path(__file__).with_name('acfo_research_adapters.py')]
    inverse = Path(__file__).with_name('validate_acfo_research_inverse.py')
    if inverse.exists(): files.append(inverse)
    save(out/'provenance.json', {'contract':json.loads(CONTRACT.read_text()),
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
        'cpu':platform.processor(),'threads':1,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})


class Memory:
    def __enter__(self):
        self.proc=psutil.Process(); self.base=self.proc.memory_info().rss
        self.peak=self.base; self.stop=threading.Event()
        def sample():
            while not self.stop.wait(.005):
                self.peak=max(self.peak,self.proc.memory_info().rss)
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start()
        return self
    def __exit__(self,*args):
        self.peak=max(self.peak,self.proc.memory_info().rss)
        info=self.proc.memory_info()
        self.os_hwm=getattr(info,'peak_wset',None)
        self.stop.set();self.thread.join()
    def report(self):
        return {'baseline_rss':self.base,'peak_rss':self.peak,'increment_rss':self.peak-self.base,'os_hwm':self.os_hwm}


def a_worker(args,p):
    n,c,m=p['A']['cases'][args.case]
    x,w=gauss_unit_interval(n)
    binding=RadialBinding.make(x,x,w,w,c=c)
    op=BlockedHankel(binding,m,c)
    rows=[]
    for tol in p['operator_tolerances']:
        with Memory() as mem:
            start=time.perf_counter(); phases={}
            if args.arm=='randomized':
                action,phases=randomized_prepare(op,tol,p['seed']+args.repeat)
            else:
                t=time.perf_counter()
                dim=int(np.ceil(c))+m+48;cols=int(np.ceil(c/2))+16
                basis=ContinuousRadialBasis.prepare(m,c,dim,cols)
                phases['eigensolve']=time.perf_counter()-t
                t=time.perf_counter()
                spectrum=HankelSpectrum.prepare(basis,quadrature=2*dim+m+32)
                phases['spectrum']=time.perf_counter()-t
                t=time.perf_counter()
                if args.arm=='cpswf':
                    action=spectrum.bind(binding,relative_tail=tol*.01)
                else:
                    k=spectrum.rank(tol*.01)
                    q=linalg.qr(binding.right[:,None]*basis.sample(x,columns=k),mode='economic')[0]
                    action=PreparedRadialAction.make(q,op.apply(q))
                phases['sampling_qr_binding']=time.perf_counter()-t
                t=time.perf_counter(); phases['probe_error']=probe_error(op,action)
                phases['required_check']=time.perf_counter()-t
            prep=time.perf_counter()-start
        # Independent full spectral audit, excluded from operational prep/memory.
        t=time.perf_counter(); exact=binding.matrix(m,c)
        scale=linalg.svdvals(exact)[0]
        err=float(linalg.svdvals(exact-action.reduced@action.basis.conj().T)[0]/scale)
        audit=time.perf_counter()-t
        rng=np.random.default_rng(813);rhs=rng.normal(size=(n,8))+1j*rng.normal(size=(n,8))
        y=rng.normal(size=(n,8))+1j*rng.normal(size=(n,8))
        dot=float(abs(np.vdot(action.forward(rhs),y)-np.vdot(rhs,action.adjoint(y)))/(linalg.norm(rhs)*linalg.norm(y)))
        multiplier=.2+.8*(1-x*x)
        bounded=float(linalg.norm(multiplier[:,None]*(action.forward(rhs)-exact@rhs))/linalg.norm(exact@rhs))
        action.forward(rhs)
        times=[]
        for _ in range(7):
            t=time.perf_counter()
            for _ in range(10):action.forward(rhs)
            times.append((time.perf_counter()-t)/10)
        rows.append(dict(n=n,c=c,m=m,arm=args.arm,tolerance=tol,repeat=args.repeat,
            prep_seconds=prep,phases=phases,memory=mem.report(),rank=action.rank,
            retained_bytes=action.retained_bytes,hot_seconds=float(np.median(times)),
            spectral_error=err,adjoint_error=dot,bounded_action_error=bounded,
            audit_seconds=audit,passed=bool(err<=tol and dot<1e-12)))
        # Preserve actual factors for primary accuracy independent replay.
        if tol==p['operator_tolerances'][0]:
            np.savez_compressed(args.out/f'A_c{args.case}_{args.arm}_r{args.repeat}.npz',
                basis=action.basis,reduced=action.reduced,x=x,w=w,c=c,m=m)
        del exact,action
    save(args.out/f'A_c{args.case}_{args.arm}_r{args.repeat}.json',rows)
    print(json.dumps({'case':args.case,'arm':args.arm,'rows':rows}),flush=True)


def profiles(r,m):
    base=r**m
    off=base*np.exp(-.6*r*r)*np.exp(.35j*np.cos(7*r*r))*(1-.2*np.exp(-((r-.81)/.07)**2))
    return {'uniform':base,'smooth_apodization':base*np.exp(-.75*r*r)*(1-.3*r*r)**.25,
            'edge_power':base*(1-r*r)**.5,
            'edge_layer':base*(1-np.exp(-(1-r*r)/.025)),
            'off_basis':off,'edge_off_basis':off*np.sqrt(1-r*r),
            'edge_mismatch':off*(1-r*r)**.7}


def b_run(out,p):
    cfg=p['B'];xf,wf=gauss_unit_interval(cfg['fit_nodes'])
    # v2: independent change of variables resolves the square-root edge profile.
    # Same r dr measure, not a favorable weighted error metric.
    ta,va=gauss_unit_interval(cfg['audit_nodes']);tr,vr=gauss_unit_interval(cfg['reference_nodes'])
    xa=np.sin(np.pi*ta/2);wa=va*np.pi/2*np.cos(np.pi*ta/2)
    xr=np.sin(np.pi*tr/2);wr=vr*np.pi/2*np.cos(np.pi*tr/2)
    y,wy=gauss_unit_interval(cfg['output_nodes']);rows=[];identities=[]
    for c in cfg['bandwidths']:
      for m in cfg['orders']:
        fa,fr,ff=({name:value for name,value in profiles(nodes,m).items() if name in cfg['profiles']}
                  for nodes in [xa,xr,xf])
        direct=lambda x,w: special.jv(m,c*y[:,None]*x[None,:])*(w*x)[None,:]
        ka,kr,kf=direct(xa,wa),direct(xr,wr),direct(xf,wf)
        truths={name:kr@v for name,v in fr.items()}
        refs={name:float(linalg.norm(np.sqrt(wy*y)*(ka@fa[name]-truths[name]))/linalg.norm(np.sqrt(wy*y)*truths[name])) for name in fa}
        # shared SVD of the physical transform; output singular vectors are not used for source fitting.
        t=time.perf_counter()
        weighted=np.sqrt(wy*y)[:,None]*special.jv(m,c*y[:,None]*xf[None,:])*np.sqrt(wf*xf)[None,:]
        _,ss,vh=robust_svd(weighted)
        svd_time=time.perf_counter()-t
        for alpha in cfg['alphas']:
          count=max(cfg['ranks'])
          t=time.perf_counter()
          sf=zernike_samples(m,alpha,count,xf);sa=zernike_samples(m,alpha,count,xa)
          images=zernike_images(m,alpha,count,c*y)
          common_preparation=time.perf_counter()-t
          # Jacobi-weight quadrature integrates endpoint powers independently.
          z,wj=special.roots_jacobi(512,alpha,m/2)
          u=(1+z)/2;r=np.sqrt(u);wj=wj/2**(alpha+m/2+2)
          pol=np.column_stack([special.eval_jacobi(n,alpha,m,2*u-1) for n in range(12)])
          numerical=(special.jv(m,c*y[:,None]*r[None,:])*wj[None,:])@pol
          identities.append(dict(c=c,m=m,alpha=alpha,max_abs_error=float(abs(numerical-images[:,:12]).max())))
          coeffs={'generalized_zernike':np.eye(count)}
          if alpha==0:coeffs={'ordinary_zernike':np.eye(count)}
          t=time.perf_counter();bc=ball_coefficients(m,alpha,c,count)
          ball_time=time.perf_counter()-t
          coeffs['cpswf' if alpha==0 else 'weighted_ball_pswf']=bc
          for arm,coef in coeffs.items():
            for k in cfg['ranks']:
              t=time.perf_counter()
              bfit=sf@coef[:,:k];baudit=sa@coef[:,:k];trans=images@coef[:,:k]
              weighted_fit=np.sqrt(wf*xf)[:,None]*bfit
              # Scaling avoids confusing arbitrary polynomial normalization with conditioning.
              norms=linalg.norm(weighted_fit,axis=0);weighted_fit=weighted_fit/norms
              pinv=linalg.pinv(weighted_fit,rtol=1e-13)
              prep=time.perf_counter()-t+common_preparation+(ball_time if 'pswf' in arm else 0)
              for name,f in ff.items():
                t=time.perf_counter();a=(pinv@(np.sqrt(wf*xf)*f))/norms
                pred=trans@a;apply=time.perf_counter()-t
                se=float(linalg.norm(np.sqrt(wa*xa)*(baudit@a-fa[name]))/linalg.norm(np.sqrt(wa*xa)*fa[name]))
                te=float(linalg.norm(np.sqrt(wy*y)*(pred-truths[name]))/linalg.norm(np.sqrt(wy*y)*truths[name]))
                rows.append(dict(c=c,m=m,alpha=alpha,arm=arm,rank=k,profile=name,source_error=se,transform_error=te,
                    combined_error=float(np.hypot(se,te)),reference_error=refs[name],prep_seconds=prep,apply_seconds=apply,
                    retained_bytes=bfit.nbytes+trans.nbytes+pinv.nbytes))
        # Portable Nyström continuation of right singular functions to audit nodes.
        # Source metric mapping is undone explicitly. Guard tiny singular values.
        left=weighted@vh.T
        cross=np.sqrt(wy*y)[:,None]*special.jv(m,c*y[:,None]*xa[None,:])
        for k in cfg['ranks']:
          eligible=min(k,int(np.count_nonzero(ss>1e-12*ss[0])))
          vf=vh[:eligible].T
          va=cross.T@(left[:,:eligible]/ss[:eligible]**2)
          for name,f in ff.items():
            a=vf.T@(np.sqrt(wf*xf)*f)
            pred=(left[:,:eligible]@a)/np.sqrt(wy*y)
            se=float(linalg.norm(np.sqrt(wa*xa)*(va@a-fa[name]))/linalg.norm(np.sqrt(wa*xa)*fa[name]))
            te=float(linalg.norm(np.sqrt(wy*y)*(pred-truths[name]))/linalg.norm(np.sqrt(wy*y)*truths[name]))
            rows.append(dict(c=c,m=m,alpha=0,arm='shared_svd',rank=eligible,requested_rank=k,profile=name,
              source_error=se,transform_error=te,combined_error=float(np.hypot(se,te)),reference_error=refs[name],
              prep_seconds=svd_time,retained_bytes=vf.nbytes+left[:,:eligible].nbytes))
        print(f'B c={c} m={m}: reference max={max(refs.values()):.3g}',flush=True)
    save(out/'B.json',{'rows':rows,'identities':identities,
        'amendment':'v2 preserves v1 results; audit/reference r=sin(pi*t/2) quadrature with exact Jacobian replaces unresolved endpoint GL rule. Same physical norm. Preparation charges common basis and image construction; local timings remain exploratory.',
        'timing_scope':'basis construction, source fitting map and transformed modes; audit-grid basis evaluation conservatively charged to polynomial arms, SVD continuation audit excluded. Do not use these times for cross-arm performance claims.'})


def c_run(out,p):
    rows=[]
    for c in p['C']['bandwidths']:
      z=np.linspace(.001,c,401)
      for nu,target in p['C']['order_pairs']:
        reference=special.jv(target,z);scale=max(abs(reference).max(),1e-300)
        for nq in p['C']['quadratures']:
          t=time.perf_counter();got,absolute=sonine_image(nu,target-nu,z,nq)
          elapsed=time.perf_counter()-t
          err=float(abs(got-reference).max()/scale)
          # Global amplification against max target magnitude, robust at zeros.
          amp=float(absolute.max()/scale)
          rows.append(dict(c=c,source_order=nu,target_order=target,quadrature=nq,
            error=err,amplification=amp,seconds=elapsed,
            gate=bool(err<=p['C']['identity_tolerance'] and amp<=p['C']['conditioning_stop'])))
    # Numerical shared-input baseline over the actual five validated orders.
    x,w=gauss_unit_interval(256);c=441.3799661723198
    b=RadialBinding.make(x,x,w,w,c=c)
    t=time.perf_counter();family=np.concatenate([b.matrix(m,c) for m in [0,4,16,64,128]],axis=0)
    _,s,_=robust_svd(family);elapsed=time.perf_counter()-t
    save(out/'C.json',{'rows':rows,'joint_numerical_baseline':{
        'grid':256,'orders':[0,4,16,64,128],'c':c,'seconds':elapsed,'matrix_bytes':family.nbytes,
        'rank_1e8':int(np.count_nonzero(s>1e-8*s[0])),
        'scope':'shared right basis of stacked matrices; dense numerical reference, not a Sonine algorithm'},
        'compressed_connection':'not attempted where float64 identity/conditioning gate fails; small stable cases are toolbox identities only'})
    print('C complete',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['campaign','a','b','c','freeze'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--case',type=int,default=0)
    parser.add_argument('--arm',default='cpswf');parser.add_argument('--repeat',type=int,default=0)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True);p=json.loads(CONTRACT.read_text())
    if args.stage=='freeze':freeze(args.out)
    elif args.stage=='a':a_worker(args,p)
    elif args.stage=='b':b_run(args.out,p)
    elif args.stage=='c':c_run(args.out,p)
    else:
        freeze(args.out)
        for i in range(len(p['A']['cases'])):
          for arm in p['A']['arms']:
            command=[sys.executable,str(Path(__file__)),'a','--out',str(args.out),'--case',str(i),'--arm',arm]
            with (args.out/f'A_c{i}_{arm}.log').open('x') as f:
                r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
            print(f'A case={i} arm={arm} exit={r.returncode}',flush=True)
        b_run(args.out,p);c_run(args.out,p)


if __name__=='__main__':main()
