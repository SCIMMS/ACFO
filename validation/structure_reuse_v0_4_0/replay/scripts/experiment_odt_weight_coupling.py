"""Dynamic per-illumination confidence in the existing packed ODT normal."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
os.environ.setdefault('CUPY_CACHE_DIR',str(ROOT/'.cupy_cache'))
import numpy as np
import torch
from benchmark_odt_fused_modal_normal import parser,iter_cones,cone_reduced_h_l_u
from benchmark_odt_banded_detector import build_variant
from odt_weight_coupling_adapter import WeightGeometry,weighted_normal,profile
CONTRACT=ROOT/'validation_contracts/odt_weight_coupling_v1.json'


def timed(fn,n):
    samples=[]
    for _ in range(n):
        torch.cuda.synchronize();start=perf_counter();value=fn();torch.cuda.synchronize()
        samples.append(perf_counter()-start)
    return samples,value


def worker(size,design,seed):
    torch.set_num_threads(1);torch.manual_seed(seed)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    args=parser().parse_args([])
    args.n_r=args.n_z=size;args.n_beta=design['n_beta'];args.dtype=design['dtype'];args.cpp_threads=1
    args.seed=seed;args.skip_axis_illumination=False
    start=perf_counter()
    plan,context,bands,context_s,plan_s=build_variant(torch=torch,device=torch.device('cuda'),base_args=args,label=design['variant'])
    cones=iter_cones(plan);geos=[WeightGeometry(c.psi_phase) for c in cones]
    torch.cuda.synchronize();setup=perf_counter()-start
    index=torch.arange(size,device='cuda',dtype=torch.long)
    x=torch.randn((size,size,design['n_beta']),device='cuda',dtype=torch.complex128)
    probe=torch.randn_like(x)
    result=dict(size=size,seed=seed,setup_seconds=setup,context_seconds=context_s,plan_seconds=plan_s,
        device=torch.cuda.get_device_name(0),torch_version=torch.__version__,profiles={},
        geometry=[dict(n_illum=c.n_illum,n_l=c.n_l,n_h=c.n_h,cap_radial=c.cap_radial,cap_phi=c.cap_phi) for c in cones],
        inherited_approximation=dict(h_cutoff=args.h_cutoff,axial_rank=args.axial_lowrank_rank,adaptive_l=args.ring_adaptive_l_packed_threshold))
    rng=np.random.default_rng(seed)
    with torch.inference_mode():
        rows=[cone_reduced_h_l_u(c,x,index).permute(2,0,1).reshape(c.cap_radial*c.n_h,c.n_l) for c in cones]
        for name in design['profiles']:
            weights=[profile(name,c.n_illum,.3) for c in cones]
            ep=[g.prepare('explicit',w) for g,w in zip(geos,weights)]
            ref=weighted_normal(cones,geos,'explicit',ep,x,index)
            energy=0
            for c,w in zip(cones,weights):
                y=c.forward_selected_z_modes(x,index).reshape(c.n_illum,c.cap_radial,c.n_h)
                energy+=torch.sum(abs(y)**2*torch.as_tensor(w,device='cuda')[:,None,None])
            normal_energy=torch.vdot(x.reshape(-1),ref.reshape(-1))
            energy_error=float((abs(normal_energy-energy)/abs(energy)).cpu())
            if energy_error>design['accuracy_tolerance']:raise ArithmeticError(('energy',energy_error))
            r=dict(energy_error=energy_error,weight_min=min(float(w.min()) for w in weights),methods={})
            for method in rng.permutation(design['methods']):
                prepare=lambda:[g.prepare(method,w) for g,w in zip(geos,weights)]
                prepare() # library initialization excluded; same geometry reuse
                updates,prepared=timed(prepare,design['samples'])
                fn=lambda:weighted_normal(cones,geos,method,prepared,x,index)
                got=fn();rel=float((torch.linalg.vector_norm(got-ref)/torch.linalg.vector_norm(ref)).cpu())
                other=weighted_normal(cones,geos,method,prepared,probe,index)
                lhs=torch.vdot(probe.reshape(-1),got.reshape(-1));rhs=torch.vdot(other.reshape(-1),x.reshape(-1))
                herm=float((abs(lhs-rhs)/(abs(lhs)+abs(rhs))).cpu())
                if rel>design['accuracy_tolerance'] or herm>design['hermitian_tolerance']:raise ArithmeticError((name,method,rel,herm))
                for _ in range(design['warmup']):fn()
                samples,got=timed(fn,design['samples'])
                core=lambda:[g.apply(method,p,v) for g,p,v in zip(geos,prepared,rows)]
                core();core_times,core_result=timed(core,design['samples'])
                core_reference=[g.apply('explicit',p,v) for g,p,v in zip(geos,ep,rows)]
                core_error=max(float((torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b)).cpu()) for a,b in zip(core_result,core_reference))
                if core_error>design['accuracy_tolerance']:raise ArithmeticError(('core',core_error))
                del got,other,core_result,core_reference
                torch.cuda.synchronize();torch.cuda.empty_cache();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                fn();torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()-baseline
                update=float(np.median(updates));apply=float(np.median(samples))
                r['methods'][str(method)]=dict(update_samples=updates,normal_samples=samples,core_samples=core_times,
                    update_seconds=update,normal_seconds=apply,core_seconds=float(np.median(core_times)),
                    agreement_relative=rel,hermitian_error=herm,core_error=core_error,incremental_torch_peak_bytes=peak,
                    lag_counts=[len(p.get('active',[])) for p in prepared],
                    view_update_ranks=[int(p['dw'].numel()) if 'dw' in p else None for p in prepared],
                    discarded_l1=[p.get('discarded_l1',0.) for p in prepared],
                    totals={str(n):update+n*apply for n in design['calls_per_update']})
            result['profiles'][name]=r
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','confirm'],default='pilot')
    ap.add_argument('--worker',type=Path);ap.add_argument('--result',type=Path);args=ap.parse_args()
    design=json.loads(CONTRACT.read_text())
    if args.worker:
        t=json.loads(args.worker.read_text());r=worker(t['size'],design,t['seed']);args.result.write_text(json.dumps(r,indent=2));return
    out=ROOT/f'reports/acfo_odt_weight_coupling_20260906_{args.phase}_v3';out.mkdir(parents=True,exist_ok=False)
    for src,name in ((CONTRACT,'protocol.json'),(Path(__file__),'source_snapshot.py'),(ROOT/'scripts/odt_weight_coupling_adapter.py','adapter_snapshot.py')):(out/name).write_bytes(src.read_bytes())
    results=[]
    sizes=design['sizes'] if args.phase=='pilot' else design['sizes'][-1:]
    for i in range(1 if args.phase=='pilot' else 5):
        for j,size in enumerate(sizes):
            task=out/f'task_{i}_{j}.json';dest=out/f'worker_{i}_{j}.json'
            task.write_text(json.dumps(dict(size=size,seed=20260906+i*100+j)))
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(task),'--result',str(dest)],check=True)
            results.append(json.loads(dest.read_text()));print(f'{args.phase} {i} size={size} PASS',flush=True)
    (out/'results.json').write_text(json.dumps(results,indent=2))
    dependencies=[*out.iterdir(),CONTRACT,Path(__file__),ROOT/'scripts/odt_weight_coupling_adapter.py',ROOT/'scripts/benchmark_odt_fused_modal_normal.py',ROOT/'scripts/benchmark_odt_torch_gpu_reconstruction.py']
    (out/'manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies if p.is_file()},indent=2))


if __name__=='__main__':main()
