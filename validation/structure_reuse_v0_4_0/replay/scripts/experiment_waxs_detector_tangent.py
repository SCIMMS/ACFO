"""Curved-detector WAXS geometry loss: physical tangent and streamed reduction."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from waxs_cake import TorchDifferentiableAxisymmetricOperator
from waxs_detector_tangent_adapter import geometry,form_factor,carbon_coefficients,evaluate

CONTRACT=ROOT/'validation_contracts/waxs_detector_tangent_v1.json'
os.environ.setdefault('CUPY_CACHE_DIR',str(ROOT/'.cupy_cache'))


def forward_loss(op,values,rho,p,target,weight,norm,coeff):
    with torch.no_grad():
        qp,qz,_,_,q,_=geometry(rho,p);ff,_=form_factor(q,coeff)
        f=op.forward(values,qp,qz)
        return torch.sum(weight*(abs(ff[:,None]*f)**2-target)**2)/(2*norm)


def direct_audit(op,values,rho,p,coeff):
    with torch.no_grad():
        qp,qz,_,_,q,_=geometry(rho,p);ff,_=form_factor(q,coeff)
        pred=ff[:,None]*op.forward(values,qp,qz)
    r=op.r_centers.cpu().numpy();z=op.z_centers.cpu().numpy();phi=2*np.pi*np.arange(op.n_phi)/op.n_phi
    rr,zz,pp=np.meshgrid(r,z,phi,indexing='ij');x=(rr*np.cos(pp)).ravel();y=(rr*np.sin(pp)).ravel();z=zz.ravel()
    f=values.cpu().numpy().ravel();actual=[];reference=[]
    qp=qp.cpu().numpy();qz=qz.cpu().numpy();ff=ff.cpu().numpy()
    for iq in (0,op.n_q//3,op.n_q-1):
        for ip in (0,op.n_phi//5):
            phase=qp[iq]*(x*np.cos(phi[ip])+y*np.sin(phi[ip]))+qz[iq]*z
            reference.append(ff[iq]*np.sum(f*np.exp(1j*phase)))
            actual.append(complex(pred[iq,ip].cpu()))
    return float(np.linalg.norm(np.asarray(actual)-reference)/np.linalg.norm(reference))


def worker(case,design,seed):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required by frozen contract')
    torch.set_num_threads(1);rng=np.random.default_rng(seed);device='cuda'
    t=perf_counter()
    r=np.linspace(0,1.5,case['nr']);z=np.linspace(-1.5,1.5,case['nz'])
    rho=torch.as_tensor(np.tan(np.linspace(.025,1.,case['nq'])),device=device,dtype=torch.float64)
    states=[torch.tensor(p,device=device,dtype=torch.float64) for p in design['states']]
    qp,qz,*_=geometry(rho,states[0])
    op=TorchDifferentiableAxisymmetricOperator(r,z,qp,qz,case['nphi'],device=device,torch=torch,
        harmonic_padding=48,miller_margin=64,q_block_size=case['qblock'],support_q_perp=[8.])
    shape=op.object_shape;size=np.prod(shape)
    f=(rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2*size)
    values=torch.as_tensor(np.ascontiguousarray(f),device=device,dtype=torch.complex128)
    target=torch.as_tensor(rng.uniform(18,54,op.data_shape),device=device,dtype=torch.float64)
    weight=torch.ones_like(target);weight[:,case['nphi']//7:case['nphi']//6]=0
    norm=torch.sum(weight*target**2);coeff=carbon_coefficients(device)
    torch.cuda.synchronize();prepare=perf_counter()-t
    # Complete kernel/library warmup cost is recorded separately, not hidden in cold claims.
    t=perf_counter()
    evaluate('tangent_stream',op,values,rho,states[0],target,weight,norm,coeff)
    torch.cuda.synchronize();initialization=perf_counter()-t
    result=dict(case=case,seed=seed,device=torch.cuda.get_device_name(0),torch_version=torch.__version__,
        prepare_seconds=prepare,first_shared_call_seconds=initialization,max_cutoff=op.max_cutoff,methods={},audit=[])
    result['operator_source']=str(Path(sys.modules[TorchDifferentiableAxisymmetricOperator.__module__].__file__).resolve())
    assert Path(result['operator_source'])==ROOT/'src/waxs_cake/differentiable_acfo.py'
    reference=[]
    for p in states:
        l,g=evaluate('production_backward',op,values,rho,p,target,weight,norm,coeff)
        reference.append((float(l.cpu()),g.cpu().numpy()))
        direct=direct_audit(op,values,rho,p,coeff)
        fd=[];step=1e-5
        for axis in (0,1):
            vals=[]
            for j in (-2,-1,1,2):
                pj=p.clone();pj[axis]+=j*step
                vals.append(float(forward_loss(op,values,rho,pj,target,weight,norm,coeff).cpu()))
            fd.append((vals[0]-8*vals[1]+8*vals[2]-vals[3])/(12*step))
        error=float(np.linalg.norm(np.asarray(fd)-reference[-1][1])/max(1.,np.linalg.norm(reference[-1][1])))
        if direct>design['accuracy']['independent_amplitude_relative'] or error>design['accuracy']['finite_difference_normalized']:
            raise ArithmeticError(dict(direct=direct,finite_difference=error))
        result['audit'].append(dict(params=p.cpu().tolist(),direct_amplitude_relative=direct,finite_difference_normalized=error))
    for method in rng.permutation(design['methods']):
        errors=[];times=[];peaks=[]
        for j,p in enumerate(states):
            l,g=evaluate(method,op,values,rho,p,target,weight,norm,coeff)
            le=abs(float(l.cpu())-reference[j][0]);ge=float(np.linalg.norm(g.cpu().numpy()-reference[j][1])/max(np.linalg.norm(reference[j][1]),1e-30))
            if le>design['accuracy']['loss_absolute'] or ge>design['accuracy']['gradient_relative']:raise ArithmeticError((method,le,ge))
            errors.append(dict(loss_absolute=le,gradient_relative=ge));del l,g
            for _ in range(design['warmup']):evaluate(method,op,values,rho,p,target,weight,norm,coeff)
            torch.cuda.synchronize();samples=[]
            for _ in range(design['samples']):
                torch.cuda.synchronize();start=perf_counter()
                evaluate(method,op,values,rho,p,target,weight,norm,coeff)
                torch.cuda.synchronize();samples.append(perf_counter()-start)
            times.append(samples)
            torch.cuda.empty_cache();torch.cuda.synchronize()
            baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            evaluate(method,op,values,rho,p,target,weight,norm,coeff);torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated()-baseline)
        result['methods'][str(method)]=dict(errors=errors,sample_seconds=times,
            three_state_seconds=float(sum(np.median(v) for v in times)),incremental_torch_peak_bytes=peaks)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',type=Path);ap.add_argument('--result',type=Path)
    ap.add_argument('--phase',choices=['pilot','confirm'],default='pilot');args=ap.parse_args()
    design=json.loads(CONTRACT.read_text())
    if args.worker:
        task=json.loads(args.worker.read_text());r=worker(task['case'],design,task['seed'])
        args.result.write_text(json.dumps(r,indent=2,allow_nan=False));return
    out=ROOT/f'reports/acfo_waxs_detector_tangent_20260906_{args.phase}_v1';out.mkdir(parents=True,exist_ok=False)
    for src,name in ((CONTRACT,'protocol.json'),(Path(__file__),'source_snapshot.py'),(ROOT/'scripts/waxs_detector_tangent_adapter.py','adapter_snapshot.py')):
        (out/name).write_bytes(src.read_bytes())
    results=[];cases=design['cases'] if args.phase=='pilot' else design['cases'][-1:]
    for i in range(1 if args.phase=='pilot' else 5):
        for j,c in enumerate(cases):
            task=out/f'task_{i}_{j}.json';dest=out/f'worker_{i}_{j}.json'
            task.write_text(json.dumps(dict(case=c,seed=20260906+i*100+j)))
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(task),'--result',str(dest)],check=True)
            results.append(json.loads(dest.read_text()));print(f'{args.phase} {i} {c} PASS',flush=True)
    (out/'results.json').write_text(json.dumps(results,indent=2))
    files=[*out.iterdir(),CONTRACT,Path(__file__),ROOT/'scripts/waxs_detector_tangent_adapter.py',ROOT/'src/waxs_cake/differentiable_acfo.py',ROOT/'src/waxs_cake/gpu_miller.py']
    (out/'manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},indent=2))


if __name__=='__main__':main()
