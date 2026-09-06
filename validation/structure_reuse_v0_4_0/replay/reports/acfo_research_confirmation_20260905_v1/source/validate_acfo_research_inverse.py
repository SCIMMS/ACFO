"""Independent-observation noisy multistart translation inverse comparison.

Truth uses direct quadrature at two higher orders. No compressed truth.
Complex vector fields are observed; this is not intensity-only phase retrieval.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy import linalg,optimize
from waxs_cake.vector_debye import gauss_sine_theta_grid
from waxs_cake.composite_wave_operator import PreparedLayeredVectorCompositeOperator
from waxs_cake.direct_prepared_composite import DirectPreparedComposite
from validate_acfo_research_abcd import Memory,save,freeze,CONTRACT

COEFF=np.array([.68-.13j,-.27+.19j,.16+.07j,-.11j,.09+.12j])
K=5.9


def source(theta,phi,theta_max):
    # Fixed physical normalization, unlike normalization by a sampled maximum.
    r=np.sin(theta)[:,None]/np.sin(theta_max);a=phi[None,:];e=np.exp(-.75*r*r)
    b=np.zeros((5,2,len(theta),len(phi)),complex)
    b[0,0]=e;b[1,1]=e*np.exp(1j*a)
    b[2,0]=.71*e*np.exp(-2j*a);b[2,1]=.28j*e*np.exp(-2j*a)
    b[3,0]=.43*e*np.exp(3j*a);b[3,1]=.57*e*np.exp(3j*a)
    phase=np.exp(.22j*r*r*np.cos(2*a))
    b[4,0]=.52*e*phase;b[4,1]=.34j*e*np.exp(-1j*a)*phase
    return b


def build(cfg,quadrature,arm):
    nt,np_=quadrature
    if cfg.get('quadrature_rule')=='theta_gauss_with_sin_weight':
        u,v=np.polynomial.legendre.leggauss(nt)
        theta=(u+1)*cfg['theta_max']/2
        w=v*cfg['theta_max']/2*np.sin(theta)
    else:
        theta,w=gauss_sine_theta_grid(nt,cfg['theta_max'])
    phi=.04+np.arange(np_)*2*np.pi/np_
    args=dict(theta=theta,theta_weights=w,phi=phi,rho_axis=np.array(cfg['rho']),
        psi_axis=np.arange(cfg['npsi'])*2*np.pi/cfg['npsi'],z_axis=np.array(cfg['z']),
        upper_wavenumber=K,lower_wavenumber=7.6,damping=.05,source_height=.44,
        source_basis=source(theta,phi,cfg['theta_max']),lateral_displacement=tuple(cfg['base_pose']))
    cls=DirectPreparedComposite if arm=='direct' else PreparedLayeredVectorCompositeOperator
    return cls.build(**args)


def truth(out,p):
    cfg=p['D'];t=time.perf_counter()
    a=build(cfg,cfg['truth_quadrature'],'direct').materialize_lateral_update(cfg['truth_delta'])
    b=build(cfg,cfg['audit_quadrature'],'direct').materialize_lateral_update(cfg['truth_delta'])
    matrix_error=float(linalg.norm(a.operator_matrix-b.operator_matrix)/linalg.norm(b.operator_matrix))
    derivative_error=float(linalg.norm(a.lateral_derivative_matrices-b.lateral_derivative_matrices)/linalg.norm(b.lateral_derivative_matrices))
    if max(matrix_error,derivative_error)>p['reference_tolerance']:
        save(out/'D_truth_failure.json',dict(matrix_error=matrix_error,derivative_error=derivative_error))
        raise ArithmeticError('Independent truth quadrature is unresolved')
    # Check reconstruction grid against the higher-quadrature response at truth.
    model={}
    for arm in ['acfo','direct']:
        obj=build(cfg,cfg['model_quadrature'],arm)
        action=obj.materialize_lateral_update(cfg['truth_delta'])
        error=float(linalg.norm(action.operator_matrix-b.operator_matrix)/linalg.norm(b.operator_matrix))
        # Nondimensional derivative w.r.t. eta=k*delta, checked at two steps.
        fd=[]
        for step in [1e-5,5e-6]:
          errors=[]
          for axis in range(2):
            h=np.eye(2)[axis]*step/K
            plus=obj.materialize_lateral_update(np.array(cfg['truth_delta'])+h).operator_matrix
            minus=obj.materialize_lateral_update(np.array(cfg['truth_delta'])-h).operator_matrix
            der=(plus-minus)/(2*step)
            expected=action.lateral_derivative_matrices[axis]/K
            errors.append(float(linalg.norm(der-expected)/linalg.norm(expected)))
          fd.append(dict(step=step,error=max(errors)))
        model[arm]=dict(operator_error=error,derivative_fd=fd)
    rng=np.random.default_rng(p['seed']);clean=[]
    for condition in cfg['source_conditions']:
        amps=np.ones(5) if condition=='known' else np.array(cfg['truth_amplitudes'])
        clean.append(b.forward(COEFF*amps).reshape(-1))
    clean=np.array(clean)
    noises=rng.normal(size=clean.shape)+1j*rng.normal(size=clean.shape)
    noises*=linalg.norm(clean,axis=1)[:,None]/linalg.norm(noises,axis=1)[:,None]
    np.savez_compressed(out/'D_truth.npz',clean=clean,noise=noises,
        operator=b.operator_matrix,derivatives=b.lateral_derivative_matrices)
    save(out/'D_truth.json',dict(matrix_error=matrix_error,derivative_error=derivative_error,
        model_audit=model,seconds=time.perf_counter()-t,
        observations='complex vector fields; independent quadrature, same physical Jones Green model'))
    if any(v['operator_error']>1e-8 or max(x['error'] for x in v['derivative_fd'])>1e-7 for v in model.values()):
        raise ArithmeticError('Model/reference or derivative accuracy gate failed')


def cases(cfg):
    return [(ic,noise,start) for ic in range(2) for noise in cfg['noise_relative_rms'] for start in cfg['starts']]


def worker(args,p):
    cfg=p['D'];ic,noise,start=cases(cfg)[args.case]
    data=np.load(args.out/'D_truth.npz');clean=data['clean'][ic];observed=clean+noise*data['noise'][ic]
    hold=np.arange(clean.size)%4==0;train=~hold;scale=linalg.norm(observed[train])
    unknown=bool(ic);true_amp=np.array(cfg['truth_amplitudes']) if unknown else np.ones(5)
    refresh_seconds=0.;refresh_count=0;cached_eta=None;cached_action=None
    def evaluate(x):
        nonlocal refresh_seconds,refresh_count,cached_eta,cached_action
        if cached_eta is None or not np.array_equal(x[:2],cached_eta):
            t=time.perf_counter();cached_action=obj.materialize_lateral_update(x[:2]/K)
            refresh_seconds+=time.perf_counter()-t;refresh_count+=1;cached_eta=x[:2].copy()
        amps=x[2:] if unknown else np.ones(5)
        prediction=cached_action.forward(COEFF*amps).reshape(-1)
        d=np.einsum('aiub,b->aiu',cached_action.lateral_derivative_matrices,COEFF*amps).reshape(2,-1).T/K
        if unknown:
            d=np.column_stack((d,cached_action.operator_matrix.reshape(-1,5)*COEFF))
        residual=(prediction[train]-observed[train])/scale
        j=d[train]/scale
        res=np.r_[residual.real,residual.imag]
        jac=np.r_[j.real,j.imag]
        if unknown:
            res=np.r_[res,1e-4*(amps-1)]
            prior=np.zeros((5,7));prior[:,2:]=1e-4*np.eye(5);jac=np.r_[jac,prior]
        return res,jac,prediction
    x0=np.r_[np.array(start)*K,np.ones(5)] if unknown else np.array(start)*K
    lower=np.r_[np.full(2,-.3*K),np.full(5,.5)] if unknown else np.full(2,-.3*K)
    upper=np.r_[np.full(2,.3*K),np.full(5,1.5)] if unknown else np.full(2,.3*K)
    with Memory() as mem:
        t=time.perf_counter();obj=build(cfg,cfg['model_quadrature'],args.arm);cold=time.perf_counter()-t
        t=time.perf_counter()
        result=optimize.least_squares(lambda x:evaluate(x)[0],x0,jac=lambda x:evaluate(x)[1],
            bounds=(lower,upper),method='trf',ftol=cfg['ftol'],xtol=cfg['xtol'],gtol=cfg['gtol'],max_nfev=60)
        _,j,pred=evaluate(result.x);opt_seconds=time.perf_counter()-t
    eta_error=float(linalg.norm(result.x[:2]-np.array(cfg['truth_delta'])*K))
    amp=result.x[2:] if unknown else np.ones(5)
    singular=linalg.svdvals(j[:-5] if unknown else j)
    retained=(obj.refresh_retained_bytes if args.arm=='direct' else obj.reference_chain_bytes+obj.materialized_bytes)
    row=dict(arm=args.arm,case=args.case,repeat=args.repeat,source_condition=cfg['source_conditions'][ic],
        noise=noise,start=start,estimate_delta=(result.x[:2]/K).tolist(),estimated_amplitudes=amp.tolist(),
        dimensionless_translation_error=eta_error,amplitude_relative_error=float(linalg.norm(amp-true_amp)/linalg.norm(true_amp)),
        heldout_clean_relative_error=float(linalg.norm((pred-clean)[hold])/linalg.norm(clean[hold])),
        heldout_noisy_relative_error=float(linalg.norm((pred-observed)[hold])/linalg.norm(observed[hold])),
        train_clean_relative_error=float(linalg.norm((pred-clean)[train])/linalg.norm(clean[train])),
        optimizer_success=bool(result.success),success=bool(result.success and eta_error<=max(.005,5*noise)),
        nfev=result.nfev,njev=result.njev,refresh_count=refresh_count,refresh_seconds=refresh_seconds,
        cold_seconds=cold,optimization_seconds=opt_seconds,total_seconds=cold+opt_seconds,memory=mem.report(),
        retained_bytes=int(retained),retained_scope='direct refresh arrays; ACFO documented reference_chain+materialized (not full Python object graph)',
        jacobian_condition=float(singular[0]/singular[-1]),final_cost=float(result.cost))
    save(args.out/f'D_c{args.case}_{args.arm}_r{args.repeat}.json',row)
    print(json.dumps(row),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['campaign','truth','worker'])
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--arm',default='acfo')
    ap.add_argument('--case',type=int,default=0);ap.add_argument('--repeat',type=int,default=0)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);p=json.loads(CONTRACT.read_text())
    # Preserve v1 failed quadrature gate; v2 changes integration variable only.
    p['D']['quadrature_rule']='theta_gauss_with_sin_weight'
    p['D']['amendment']='v1 cos(theta) quadrature 48/64 reference failed (2.40e-6). Angular modes lack radial vanishing at the pole; integrate in theta so the integrand is smooth. Same physical sin(theta)dtheta measure and node counts.'
    if args.stage=='truth':truth(args.out,p)
    elif args.stage=='worker':worker(args,p)
    else:
        freeze(args.out);save(args.out/'D_protocol_v2.json',p);truth(args.out,p)
        for i in range(len(cases(p['D']))):
          for arm in ['acfo','direct']:
            with (args.out/f'D_c{i}_{arm}.log').open('x') as f:
                r=subprocess.run([sys.executable,str(Path(__file__)),'worker','--out',str(args.out),
                    '--case',str(i),'--arm',arm],stdout=f,stderr=subprocess.STDOUT)
            print(f'D case={i} arm={arm} exit={r.returncode}',flush=True)


if __name__=='__main__':main()
