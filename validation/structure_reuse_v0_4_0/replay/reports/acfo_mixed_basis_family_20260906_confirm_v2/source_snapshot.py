"""Compare actual mixed-basis composite paths for a fixed affine profile family.

All comparators get affine reuse. Common generator catalogue costs are explicit.
Only fixed finite-input operator norms are certified; this is not a PDE solve.
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import numpy as np
from scipy import linalg, special

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from experiment_cpswf_operator_algebra import (
    accurate_gauss, radial_jacobi_basis, differential_columns, physical_map,
    eigensystem, ladder_apply, norm, relative, tri, x_squared_tridiagonal,
    hankel_basis_images,
)


def profile(name, r):
    if name == "r2":
        return r*r, 2*r
    if name == "edge_erf":
        z = 18*(r*r-0.72)
        return (0.5*(1+special.erf(z)), 36*r/np.sqrt(np.pi)*np.exp(-z*z))
    raise ValueError(name)


def lower_bound(f0, f1):
    """Analytic minimum of ||(F0+theta F1)v|| for one fixed unit v."""
    v = linalg.svd(f0, full_matrices=False)[2][0].conj()
    a, b = f0@v, f1@v
    theta = float(np.clip(-np.vdot(a, b).real/max(np.vdot(b, b).real, 1e-300), 0, 1))
    return float(linalg.norm(a+theta*b)), theta


def retained_bytes(state):
    seen = set()
    def visit(a):
        if isinstance(a, np.ndarray):
            while isinstance(a.base, np.ndarray):
                a = a.base
            if id(a) in seen:
                return 0
            seen.add(id(a))
            return a.nbytes
        if isinstance(a, dict):
            return sum(visit(x) for x in a.values())
        return 0
    return int(visit(state))


def catalogue(m, c, nin, profile_name, design):
    start = perf_counter()
    n = design["dimension"]
    x, w = accurate_gauss(design["quadrature"])
    y, v = accurate_gauss(design["output_quadrature"])
    qp = np.sqrt(w[:, None])*radial_jacobi_basis(m+1, n, x)
    b, _ = profile(profile_name, x)
    e = np.eye(n)[:, :nin]
    x0 = ladder_apply(m, e)
    if profile_name == "r2":
        d, off = x_squared_tridiagonal(m+1, n)
        mult = tri(d, off)
        x1 = mult@x0
    else:
        d, off = np.array([]), np.array([])
        x1 = qp.T@(b[:, None]*(qp@x0))
    h = np.sqrt(v[:, None])*hankel_basis_images(m, c, n, y)/c**2
    down = h@ladder_apply(m, np.eye(n), lower=True)
    f0, f1 = down@x0, down@x1
    bound, argmin = lower_bound(f0, f1)
    result = dict(n=n, nin=nin, m=m, c=c, profile=profile_name, qp=qp, b=b,
                  e=e, x0=x0, x1=x1, h=h, down=down, f0=f0, f1=f1,
                  d=d, off=off, bound=bound, bound_argmin=argmin)
    result["seconds"] = perf_counter()-start
    result["retained_bytes"] = retained_bytes(result)
    return result


def independent_audit(cat, design):
    start = perf_counter()
    m,c,nin = cat["m"],cat["c"],cat["nin"]
    y,v = accurate_gauss(design["output_quadrature"])
    refs=[]
    for count in (design["quadrature"], design["audit_quadrature"]):
        x,w=accurate_gauss(count)
        sw=np.sqrt(w[:, None])
        b,bp=profile(cat["profile"],x)
        lap=sw*differential_columns(m,nin,x,"laplacian")
        raise_=sw*differential_columns(m,nin,x,"raise")
        k=physical_map(m,c,x,w,y,v)/c**2
        # Product rule is independent of coefficient multiplication/ladders.
        refs.append((k@lap, k@(b[:,None]*lap+bp[:,None]*raise_)))
    f0,f1=refs[-1]
    scale=cat["bound"]
    out={"catalogue_reference_error": (norm(cat["f0"]-f0)+norm(cat["f1"]-f1))/scale,
         "quadrature_error": (norm(refs[0][0]-f0)+norm(refs[0][1]-f1))/scale,
         "seconds": perf_counter()-start}
    return out, (f0,f1)


def choose_basis(v, cat, tol, output=False):
    """All-prefix family gate, includes required scan/check cost."""
    curves=[]
    for k in range(1,v.shape[1]+1):
        vk=v[:,:k]
        if output:
            left=vk
            r0,r1=vk.T@cat["f0"],vk.T@cat["f1"]
        else:
            left=cat["down"]@vk
            r0,r1=vk.T@cat["x0"],vk.T@cat["x1"]
        error=(norm(left@r0-cat["f0"])+norm(left@r1-cat["f1"]))/cat["bound"]
        curves.append(error)
        if error<=tol/2:
            return {"left":left,"r0":r0,"r1":r1,"rank":k,"family_bound":error,
                    "gate_curve":curves}
    raise ArithmeticError(f"basis cannot resolve family, best={min(curves)}")


def build(name, cat, tol):
    start=perf_counter()
    if name=="direct_affine":
        s={"f0":cat["f0"].copy(),"f1":cat["f1"].copy(),"rank":cat["nin"],"family_bound":0.0}
    elif name=="jacobi_native":
        s={k:cat[k] for k in ("m","profile","nin")}
        if cat["profile"]=="r2":
            # D+ lowers the radial polynomial index; r² restores at most one.
            # The first nin modes therefore close exactly for this action.
            k=cat["nin"]
            s.update(h=cat["h"][:,:k],d=cat["d"][:k],off=cat["off"][:k-1],rank=k)
        else:
            s.update(h=cat["h"],qp=cat["qp"],b=cat["b"],rank=cat["n"])
        s["family_bound"]=0.0
    elif name=="value_native":
        # Exact pointwise product rule is a strong alternative to mixed bases.
        x,w=accurate_gauss(len(cat["b"]))
        y,v=accurate_gauss(cat["h"].shape[0])
        b,bp=profile(cat["profile"],x)
        s={"kernel":physical_map(cat["m"],cat["c"],x,w,y,v)/cat["c"]**2,
           "lap":np.sqrt(w[:,None])*differential_columns(cat["m"],cat["nin"],x,"laplacian"),
           "raise":np.sqrt(w[:,None])*differential_columns(cat["m"],cat["nin"],x,"raise"),
           "b":b,"bp":bp,"rank":len(x),"family_bound":None}
    elif name.startswith("cpswf"):
        _,vectors=eigensystem(cat["m"]+1,cat["c"],cat["n"])
        s=choose_basis(vectors,cat,tol)
    elif name.startswith("shared_image"):
        vectors=linalg.svd(np.hstack((cat["x0"],cat["x1"])),full_matrices=False)[0]
        s=choose_basis(vectors,cat,tol)
    elif name=="shared_output":
        vectors=linalg.svd(np.hstack((cat["f0"],cat["f1"])),full_matrices=False)[0]
        s=choose_basis(vectors,cat,tol,output=True)
    elif name=="action_svd":
        s={"f0":cat["f0"].copy(),"f1":cat["f1"].copy(),"tolerance":tol,
           "rank":None,"family_bound":None}
    else:
        raise ValueError(name)
    if name.endswith("_fused"):
        s["f0"],s["f1"]=s["left"]@s["r0"],s["left"]@s["r1"]
        del s["left"],s["r0"],s["r1"]
    # Same complex128 runtime convention in every arm.
    for key,value in list(s.items()):
        if isinstance(value,np.ndarray):
            s[key]=np.ascontiguousarray(value,dtype=np.complex128)
    elapsed=perf_counter()-start
    return s,elapsed


def update(name,s,theta):
    if name=="action_svd":
        f=s["f0"]+theta*s["f1"]
        u,singular,vh=linalg.svd(f,full_matrices=False)
        k=max(1,int(np.count_nonzero(singular>s["tolerance"]/2*singular[0])))
        return {"left":np.ascontiguousarray(u[:,:k]*singular[:k]),"core":np.ascontiguousarray(vh[:k]),"rank":k}
    if "f0" in s:
        return {"matrix":s["f0"]+theta*s["f1"]}
    if "left" in s:
        return {"left":s["left"],"core":s["r0"]+theta*s["r1"]}
    return {"theta":theta}


def apply(name,s,plan,z):
    if "matrix" in plan:
        return plan["matrix"]@z
    if "left" in plan:
        return plan["left"]@(plan["core"]@z)
    theta=plan["theta"]
    if name=="value_native":
        g=(1+theta*s["b"][:,None])*(s["lap"]@z)+theta*s["bp"][:,None]*(s["raise"]@z)
        return s["kernel"]@g
    g=ladder_apply(s["m"],z)
    if s["profile"]=="r2":
        mg=s["d"][:,None]*g
        mg[1:]+=s["off"][:,None]*g[:-1]
        mg[:-1]+=s["off"][:,None]*g[1:]
    else:
        mg=s["qp"].T@(s["b"][:,None]*(s["qp"][:,:s["nin"]]@g))
        g=np.vstack((g,np.zeros((mg.shape[0]-g.shape[0],g.shape[1]),dtype=g.dtype)))
    return s["h"]@ladder_apply(s["m"],g+theta*mg,lower=True)


ARMS=["direct_affine","jacobi_native","value_native","cpswf_factor","cpswf_fused",
      "shared_image_factor","shared_image_fused","shared_output","action_svd"]


def worker(case,design,seed):
    cat=catalogue(case["m"],case["c"],case["nin"],case["profile"],design)
    audit,refs=independent_audit(cat,design)
    if max(audit["catalogue_reference_error"],audit["quadrature_error"])>design["reference_threshold"]:
        raise ArithmeticError(f"reference gate failed {audit}")
    rng=np.random.default_rng(seed)
    result={"case":case,"seed":seed,"catalogue_seconds":cat["seconds"],
            "catalogue_retained_bytes":cat["retained_bytes"],"audit":audit,"arms":{},
            "family_norm_lower_bound":cat["bound"],"bound_argmin":cat["bound_argmin"]}
    eye=np.eye(cat["nin"],dtype=complex)
    rhs_values={rhs:np.ascontiguousarray(rng.normal(size=(cat["nin"],rhs))+1j*rng.normal(size=(cat["nin"],rhs)))
                for rhs in design["rhs_counts"]}
    for name in rng.permutation(ARMS):
        s,prep=build(name,cat,case["tolerance"])
        # Operational correctness checks charged to backend preparation.
        start=perf_counter()
        errors=[]; outputs=[]
        for theta in [0.0,1.0,*design["heldout_theta"]]:
            p=update(name,s,theta)
            actual=apply(name,s,p,eye)
            errors.append(relative(actual,cat["f0"]+theta*cat["f1"]))
            outputs.append((theta,actual))
        check=perf_counter()-start
        physical_errors=[relative(actual,refs[0]+theta*refs[1]) for theta,actual in outputs]
        if max([*errors,*physical_errors])>case["tolerance"]:
            raise ArithmeticError(f"{name} failed accuracy {max(errors)}")
        times={}
        for rhs in design["rhs_counts"]:
            z=rhs_values[rhs]
            refresh=[]; hot=[]; ranks=[]
            for theta in design["update_sequence"]:
                begin=perf_counter(); p=update(name,s,theta); refresh.append(perf_counter()-begin)
                apply(name,s,p,z)
                begin=perf_counter()
                for _ in range(design["timing_inner"]):
                    apply(name,s,p,z)
                hot.append((perf_counter()-begin)/design["timing_inner"])
                ranks.append(p.get("rank",s["rank"]))
            workloads={}
            for count in design["calls_per_update"]:
                refresh_apply=sum(refresh)+count*sum(hot)
                workloads[str(count)]={"sequence_seconds":refresh_apply,
                    "cold_plus_sequence_seconds":cat["seconds"]+prep+check+refresh_apply}
            times[str(rhs)]={"update_seconds":refresh,"apply_seconds":hot,"ranks":ranks,"workloads":workloads}
        result["arms"][name]={"backend_prepare_seconds":prep,"required_check_seconds":check,
            "retained_backend_bytes":retained_bytes(s),"retained_current_plan_bytes":retained_bytes(p),
            "retained_backend_and_plan_bytes":retained_bytes({"state":s,"plan":p}),
            "selected_modes":s["rank"],"family_bound":s["family_bound"],
            "max_catalogue_error":max(errors),"max_reference_error":max(physical_errors),"timings":times}
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--worker",type=Path)
    ap.add_argument("--result",type=Path)
    ap.add_argument("--phase",choices=["pilot","confirm"],default="pilot")
    args=ap.parse_args()
    protocol=ROOT/"validation_contracts/cpswf_mixed_basis_family_v1.json"
    design=json.loads(protocol.read_text())
    if args.worker:
        task=json.loads(args.worker.read_text())
        result=worker(task["case"],design,task["seed"])
        args.result.write_text(json.dumps(result,indent=2,allow_nan=False))
        return
    out=ROOT/f"reports/acfo_mixed_basis_family_20260906_{args.phase}_v2"
    out.mkdir(parents=True,exist_ok=False)
    (out/"protocol.json").write_bytes(protocol.read_bytes())
    (out/"source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    cases=[]
    for cp in design["cases"]:
        for m in design["orders"]:
            for profile_name in design["profiles"]:
                for tol in design["tolerances"]:
                    if args.phase=="confirm" and (cp["c"]!=64 or m!=0 or tol!=1e-8):
                        continue
                    cases.append(dict(cp,m=m,profile=profile_name,tolerance=tol))
    results=[]
    for repeat in range(1 if args.phase=="pilot" else 5):
        for j,case in enumerate(cases):
            task=out/f"task_{repeat}_{j}.json"
            dest=out/f"worker_{repeat}_{j}.json"
            task.write_text(json.dumps({"case":case,"seed":20260906+100*repeat+j},indent=2))
            subprocess.run([sys.executable,str(Path(__file__).resolve()),"--worker",str(task),"--result",str(dest)],check=True)
            r=json.loads(dest.read_text());results.append(r)
            print(f"{args.phase} repeat={repeat} case={case} passed",flush=True)
    (out/"results.json").write_text(json.dumps(results,indent=2))
    files=[*out.iterdir(),protocol,Path(__file__),ROOT/"scripts/experiment_cpswf_operator_algebra.py"]
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))


if __name__=="__main__":
    main()
