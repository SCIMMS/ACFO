"""Finite-aperture 4f phase-plate relay with paired forward/phase derivative.

This tests two-sided bandlimiting, not CPSWF-only superiority. Dense factor,
analytic Jacobi, Nyström and full SVD receive identical mask reuse.
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[_key]="1"
import argparse
import json
import hashlib
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import numpy as np
from scipy import linalg
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from experiment_cpswf_operator_algebra import (accurate_gauss,physical_map,eigensystem,
    radial_jacobi_basis,hankel_basis_images,norm,relative)


def phase_vectors(x,alpha,beta):
    phase=np.exp(1j*(alpha*x*x+beta*x**4))
    return phase,1j*x*x*phase


def make_factors(name,case,x,w,k,design):
    start=perf_counter();m,c=case["m"],case["c"]
    floor=design["kernel_cutoff"]
    if name=="svd":
        u,s,vh=linalg.svd(k,full_matrices=False)
        keep=s>floor
        a,b=u[:,keep]*s[keep],vh[keep].T
    elif name=="nystrom":
        z,wz=accurate_gauss(design["reference_nodes"])
        small=physical_map(m,c,z,wz,z,wz)
        lam,vectors=linalg.eigh(small)
        keep=abs(lam)>floor
        lam,vectors=lam[keep],vectors[:,keep]
        a=physical_map(m,c,z,wz,x,w)@vectors
        b=a/lam
    else:
        n=design["spectral_dimension"]
        z,wz=accurate_gauss(design["reference_nodes"])
        images=np.sqrt(w[:,None])*hankel_basis_images(m,c,n,x)
        basis=np.sqrt(w[:,None])*radial_jacobi_basis(m,n,x)
        if name=="jacobi":
            # Analytic image-tail choice; the full discrete residual is then
            # checked and charged below. No claim of a proven infinite tail.
            tail=np.sqrt(np.cumsum(np.sum(abs(images)**2,axis=0)[::-1])[::-1])
            indices=np.flatnonzero(tail<=floor)
            rank=int(indices[0]) if len(indices) else n
            a,b=images[:,:rank],basis[:,:rank]
        elif name=="cpswf":
            _,u=eigensystem(m,c,n)
            pz=radial_jacobi_basis(m,n,z)
            hz=hankel_basis_images(m,c,n,z)
            galerkin=(pz.T*wz)@hz
            lam=np.diag(u.T@galerkin@u)
            keep=abs(lam)>floor
            b=basis@u[:,keep]
            a=b*lam[keep]
        else:raise ValueError(name)
    elapsed=perf_counter()-start
    start=perf_counter()
    delta=norm(k-a@b.T)
    qa,ra=linalg.qr(a,mode="economic")
    qb,rb=linalg.qr(b,mode="economic")
    hn=norm(ra@rb.T)
    # Converting runtime factors is part of preparation, not free.
    a,b=np.ascontiguousarray(a,dtype=complex),np.ascontiguousarray(b,dtype=complex)
    checked=perf_counter()-start
    if delta>5*floor:raise ArithmeticError(f"{name}: kernel error {delta}")
    return dict(a=a,b=b,ra=ra,rb=rb,delta=delta,hn=hn,rank=a.shape[1],
                prepare_seconds=elapsed,required_check_seconds=checked,
                retained_bytes=a.nbytes+b.nbytes+ra.nbytes+rb.nbytes)


def update(name,factor,k,x,alpha,beta):
    p,dp=phase_vectors(x,alpha,beta)
    if name=="dense_factor":return dict(p=p,dp=dp)
    if name=="dense_fused":return dict(t=k@(p[:,None]*k),dt=k@(dp[:,None]*k))
    if name.endswith("_value"):return dict(p=p,dp=dp)
    return dict(g=factor["b"].T@(p[:,None]*factor["a"]),
                dg=factor["b"].T@(dp[:,None]*factor["a"]))


def apply(name,factor,k,plan,f):
    if name=="dense_factor":
        first=k@f
        return k@np.hstack((plan["p"][:,None]*first,plan["dp"][:,None]*first))
    if name=="dense_fused":return np.hstack((plan["t"]@f,plan["dt"]@f))
    a,b=factor["a"],factor["b"]
    coeff=b.T@f
    if name.endswith("_value"):
        first=a@coeff
        middle=b.T@np.hstack((plan["p"][:,None]*first,plan["dp"][:,None]*first))
    else:middle=np.hstack((plan["g"]@coeff,plan["dg"]@coeff))
    return a@middle


def certify(factor,kn,plan,x,alpha,beta):
    """Relative operator-norm bound, not relative error for every output."""
    if "g" in plan:
        g=plan["g"]
    else:
        p,_=phase_vectors(x,alpha,beta)
        g=factor["b"].T@(p[:,None]*factor["a"])
    tnorm=norm(factor["ra"]@g@factor["rb"].T)
    bound=factor["delta"]*(kn+factor["hn"])
    return bound,bound/(tnorm-bound) if tnorm>bound else float("inf")


def reference_sources(case,x,w,alpha,beta,design):
    m,c=case["m"],case["c"]
    ns=design["audit_source_modes"]
    actual=[]
    for q in design["physical_reference_nodes"]:
        z,wz=accurate_gauss(q)
        # First transform is the independent analytic Jacobi-Bessel image.
        first=np.sqrt(wz[:,None])*hankel_basis_images(m,c,max(ns)+1,z)[:,ns]
        p,dp=phase_vectors(z,alpha,beta)
        ko=physical_map(m,c,z,wz,x,w)
        actual.append(ko@np.hstack((p[:,None]*first,dp[:,None]*first)))
    source=np.sqrt(w[:,None])*radial_jacobi_basis(m,max(ns)+1,x)[:,ns]
    return source,actual[-1],relative(actual[0],actual[-1])


def worker(case,design,seed):
    start=perf_counter()
    x,w=accurate_gauss(case["q"])
    k=physical_map(case["m"],case["c"],x,w,x,w)
    kc=np.ascontiguousarray(k,dtype=complex)
    common=perf_counter()-start
    start=perf_counter();kn=norm(k);common_check=perf_counter()-start
    rng=np.random.default_rng(seed)
    factors={}
    for name in rng.permutation(["cpswf","jacobi","nystrom","svd"]):
        factors[name]=make_factors(name,case,x,w,k,design)
    refs=[]
    for alpha,beta in design["phases"]:
        source,ref,error=reference_sources(case,x,w,alpha,beta,design)
        if error>1e-10:raise ArithmeticError(f"physical reference {error}")
        refs.append((source,ref,error))
    rhs={n:np.ascontiguousarray(rng.normal(size=(len(x),n))+1j*rng.normal(size=(len(x),n)))
         for n in design["rhs_counts"]}
    result=dict(case=case,seed=seed,common_kernel_seconds=common,common_norm_check_seconds=common_check,
                kernel_norm=kn,kernel_bytes=kc.nbytes,methods={},factor_metrics={})
    for name,fac in factors.items():
        result["factor_metrics"][name]={key:fac[key] for key in ("delta","hn","rank","prepare_seconds","required_check_seconds","retained_bytes")}
    for name in rng.permutation(design["methods"]):
        fac=None if name.startswith("dense") else factors[name.split("_")[0]]
        validation=[];updates=[];checks=[];plans=[]
        for (alpha,beta),(source,ref,referr) in zip(design["phases"],refs):
            start=perf_counter();plan=update(name,fac,kc,x,alpha,beta);updates.append(perf_counter()-start)
            plans.append(plan)
            start=perf_counter()
            if fac is not None:
                bound,relative_bound=certify(fac,kn,plan,x,alpha,beta)
                if relative_bound>design["tolerance"]:raise ArithmeticError(f"{name}: operator bound {relative_bound}")
                checks.append(perf_counter()-start)
            else:
                bound=relative_bound=0.0
                checks.append(0.0)
            actual=apply(name,fac,kc,plan,source)
            error=relative(actual,ref)
            if error>design["tolerance"]:raise ArithmeticError(f"{name}: physical error {error}")
            validation.append(dict(phase=[alpha,beta],physical_error=error,quadrature_error=referr,
                                   operator_relative_bound=relative_bound,
                                   derivative_absolute_bound_per_unit_input=bound))
        timings={}
        for count,f in rhs.items():
            values=[]
            for plan in plans:
                apply(name,fac,kc,plan,f)
                start=perf_counter()
                for _ in range(design["timing_inner"]):apply(name,fac,kc,plan,f)
                values.append((perf_counter()-start)/design["timing_inner"])
            prepare=common+(0 if fac is None else common_check+fac["prepare_seconds"]+fac["required_check_seconds"])
            totals={str(n):dict(warm_seconds=sum(updates)+sum(checks)+n*sum(values),
                          cold_seconds=prepare+sum(updates)+sum(checks)+n*sum(values)) for n in design["calls_per_mask"]}
            timings[str(count)]=dict(apply_seconds=values,totals=totals)
        # Only one plan retained in an operational update stream. The runner
        # keeps three to isolate phase timings; do not call this peak RSS.
        state_bytes=(kc.nbytes if fac is None else fac["retained_bytes"])
        plan_bytes=max(sum(a.nbytes for a in plan.values()) for plan in plans)
        result["methods"][name]=dict(validation=validation,update_seconds=updates,required_update_check_seconds=checks,timings=timings,
                 runtime_state_and_one_plan_bytes=state_bytes+plan_bytes)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--worker",type=Path);ap.add_argument("--result",type=Path)
    ap.add_argument("--phase",choices=["pilot","confirm"],default="pilot");args=ap.parse_args()
    contract=ROOT/"validation_contracts/radial_4f_phase_relay_v1.json"
    design=json.loads(contract.read_text())
    if args.worker:
        task=json.loads(args.worker.read_text());r=worker(task["case"],design,task["seed"])
        args.result.write_text(json.dumps(r,indent=2,allow_nan=False));return
    out=ROOT/f"reports/acfo_radial_4f_phase_relay_20260906_{args.phase}_v2"
    out.mkdir(parents=True,exist_ok=False)
    (out/"protocol.json").write_bytes(contract.read_bytes());(out/"source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    cases=design["cases"] if args.phase=="pilot" else design["cases"][-1:]
    results=[]
    for repeat in range(1 if args.phase=="pilot" else 5):
        for j,case in enumerate(cases):
            task=out/f"task_{repeat}_{j}.json";dest=out/f"worker_{repeat}_{j}.json"
            task.write_text(json.dumps(dict(case=case,seed=20260906+100*repeat+j),indent=2))
            subprocess.run([sys.executable,str(Path(__file__).resolve()),"--worker",str(task),"--result",str(dest)],check=True)
            results.append(json.loads(dest.read_text()));print(f"{args.phase} repeat={repeat} case={case} passed",flush=True)
    (out/"results.json").write_text(json.dumps(results,indent=2))
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*out.iterdir(),contract,Path(__file__),ROOT/"scripts/experiment_cpswf_operator_algebra.py"] if p.is_file()}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))


if __name__=="__main__":main()
