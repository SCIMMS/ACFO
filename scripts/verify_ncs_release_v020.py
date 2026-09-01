"""Verify the public ACFO v0.2.0 evidence bundle."""
from __future__ import annotations
import hashlib, json, math, re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; RELEASE=ROOT/'validation/ncs_v0_2_0'; TEXT={'.csv','.json','.md','.txt'}
BAD=re.compile(r'(?i)(github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+|(?:api[_-]?key|password|secret)\s*[:=]\s*[^\s,;]+|(?:141\.223\.|172\.17\.)|/home/(?:user|compu)/|[A-Za-z]:\\Users\\)')
def nbytes(p):
    b=p.read_bytes(); return b.replace(b'\r\n',b'\n') if p.suffix.lower() in TEXT else b
def inventory():
    out=[]
    for p in sorted(x for x in RELEASE.rglob('*') if x.is_file() and x.name!='MANIFEST.json'):
        b=nbytes(p); out.append({'path':p.relative_to(RELEASE).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
    return out
def req(ok,msg,errors):
    if not ok: errors.append(msg)
def close(a,b): return math.isclose(float(a),float(b),rel_tol=2e-12,abs_tol=1e-15)
def waxs_boot(a,b,seed=20260825,draws=10000):
    rng=np.random.default_rng(seed); r=np.empty(draws)
    for i in range(draws):
        q=rng.integers(0,a.size,size=a.size); r[i]=np.median(b[q])/np.median(a[q])
    return float(np.median(b)/np.median(a)),float(np.quantile(r,.025)),float(np.quantile(r,.975))
def numag_boot(a,b,sa,sb,seed=20260825,draws=10000):
    rng=np.random.default_rng(seed); r=np.empty(draws)
    for i in range(draws):
        q=rng.integers(0,a.size,size=a.size); r[i]=(sb+b[q].sum())/(sa+a[q].sum())
    return float((sb+b.sum())/(sa+a.sum())),float(np.quantile(r,.025)),float(np.quantile(r,.975))
def main():
    e=[]; m=json.loads((RELEASE/'MANIFEST.json').read_text(encoding='utf-8'))
    req(m.get('schema')=='acfo-ncs-release-v0.2.0-manifest','manifest schema mismatch',e); req(m.get('files')==inventory(),'manifest hash mismatch',e)
    for p in sorted(x for x in RELEASE.rglob('*') if x.is_file() and x.suffix.lower() in TEXT):
        if BAD.search(p.read_text(encoding='utf-8-sig',errors='replace')): e.append(f'local path or credential pattern in {p.relative_to(RELEASE)}')
    f=json.loads((RELEASE/'source_data/figure_source_data.json').read_text(encoding='utf-8')); w=f['figure_2']['waxs_timing']; a=np.array(w['acfo_seconds']['samples_s']); b=np.array(w['baseline_seconds']['samples_s']); wr=waxs_boot(a,b); wo=w['baseline_over_acfo_speedup']
    for x,y,n in zip(wr,[wo['point'],wo['lower_95'],wo['upper_95']],['point','lower','upper']): req(close(x,y),f'WAXS {n} mismatch',e)
    req(f['figure_2']['waxs']['same_top1'] and close(f['figure_2']['waxs']['spearman'],1),'WAXS ranking mismatch',e)
    n=json.loads((RELEASE/'evidence/numagsans_timing_and_accuracy.json').read_text(encoding='utf-8')); a=np.array(n['orientation_fourier_samples_seconds']['acfo']); b=np.array(n['orientation_fourier_samples_seconds']['affine_type2']); s=n['setup_seconds']; nr=numag_boot(a,b,float(s['acfo_total_excluding_jit'])+float(s['gpu_miller_jit_cold_start']),float(s['selected_baseline'])); no=n['cold_total_speedup_selected_baseline_over_acfo']
    for x,y,k in zip(nr,[no['point'],no['lower_95'],no['upper_95']],['point','lower','upper']): req(close(x,y),f'NuMagSANS {k} mismatch',e)
    req(a.size==b.size==800,'NuMagSANS orientation count mismatch',e); req(n['strongest_baseline_frozen_before_orientation_2']=='affine_type2','NuMagSANS comparator mismatch',e)
    u=json.loads((RELEASE/'evidence/uob_native_grid_summary.json').read_text(encoding='utf-8')); req(close(u['ranking']['primary_best_J_over_T'],.13413167796828418),'UOB candidate mismatch',e); req(u['accuracy']['acfo_vs_fft_relative_l2_max']<2e-14 and u['accuracy']['fft_vs_atomic_direct_relative_l2_max']<3e-14,'UOB agreement failed',e)
    o=json.loads((RELEASE/'evidence/odt_native_representation_aggregate.json').read_text(encoding='utf-8')); req(o.get('integrity_passed') is True,'ODT aggregate integrity failed',e); req(len(o.get('rows',[]))==6,'ODT row count mismatch',e); req(all(r.get('integrity_passed') for r in o.get('rows',[])),'ODT row integrity failed',e)
    figs=[p for p in (RELEASE/'figures').iterdir() if p.is_file() and p.suffix.lower() in {'.pdf','.png','.svg'}]; req(len(figs)==15,'expected 15 figure files',e)
    out={'schema':'acfo-ncs-release-v0.2.0-audit','passed':not e,'file_count':len(inventory()),'waxs':dict(zip(['point','lower_95','upper_95'],wr)),'numagsans_fourier_backbone':{'point':nr[0],'lower_95_orientation_resampling':nr[1],'upper_95_orientation_resampling':nr[2],'orientation_count':int(a.size),'packing_reductions_in_ratio':False},'uob_primary_J_over_T':u['ranking']['primary_best_J_over_T'],'errors':e}; print(json.dumps(out,indent=2,ensure_ascii=False)); return 0 if not e else 1
if __name__=='__main__': raise SystemExit(main())
