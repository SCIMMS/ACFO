"""Build the normalized evidence bundle for ACFO release v0.2.0."""
from __future__ import annotations
import argparse, hashlib, json, re, shutil, zipfile
from datetime import datetime, timezone
from pathlib import Path

TEXT={'.csv','.json','.md','.txt'}
def hbytes(b): return hashlib.sha256(b).hexdigest()
def hfile(p): return hbytes(p.read_bytes())
def norm_str(s, root):
    s=s.replace(str(root),'<WORKSPACE>').replace(root.as_posix(),'<WORKSPACE>')
    s=re.sub(r'/home/(?:user|compu)/ACFO','<ACFO_WORKSPACE>',s)
    s=re.sub(r'/home/(?:user|compu)/\.conda/envs/[^/\s]+','<CONDA_ENV>',s)
    s=re.sub(r'/home/(?:user|compu)/anaconda3','<CONDA_ROOT>',s)
    s=s.replace('/home/conda/feedstock_root','<CONDA_BUILD_ROOT>')
    return re.sub(r'[A-Za-z]:\\Users\\[^\\\s]+','<USER_HOME>',s)
def norm(v, root):
    if isinstance(v,str): return norm_str(v,root)
    if isinstance(v,list): return [norm(x,root) for x in v]
    if isinstance(v,dict): return {k:norm(x,root) for k,x in v.items()}
    return v
def wjson(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def wtext(p,s):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(s.replace('\r\n','\n'),encoding='utf-8',newline='\n')
def inventory(root):
    out=[]
    for p in sorted(x for x in root.rglob('*') if x.is_file() and x.name!='MANIFEST.json'):
        b=p.read_bytes(); b=b.replace(b'\r\n',b'\n') if p.suffix.lower() in TEXT else b
        out.append({'path':p.relative_to(root).as_posix(),'bytes':len(b),'sha256':hbytes(b)})
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source-workspace',type=Path,required=True); ap.add_argument('--repository-root',type=Path,default=Path(__file__).resolve().parents[1]); a=ap.parse_args()
    src=a.source_workspace.resolve(); repo=a.repository_root.resolve(); out=repo/'validation/ncs_v0_2_0'; out.mkdir(parents=True,exist_ok=True)
    r3=src/'reports/acfo_ncs_manuscript_final_hardening_r3_20260902'; prov=[]
    def cp(source,dest,kind='json'):
        dest.parent.mkdir(parents=True,exist_ok=True)
        if kind=='json': wjson(dest,norm(json.loads(source.read_text(encoding='utf-8-sig')),src)); n='paths normalized; scientific values unchanged'
        elif kind=='text': wtext(dest,source.read_text(encoding='utf-8-sig')); n='UTF-8 LF'
        else: shutil.copyfile(source,dest); n='none'
        prov.append({'source':source.relative_to(src).as_posix(),'source_sha256':hfile(source),'public_copy':dest.relative_to(out).as_posix(),'public_sha256':hfile(dest),'normalization':n})
    sd=r3/'figures/source_data'
    for n in ['figure_source_data.json','figure2_uob_adapter_sensitivity.csv','figure2_uob_native_grid.csv','figure2_waxs.csv']:
        cp(sd/n,out/'source_data'/n,'json' if n.endswith('.json') else 'text')
    for p in sorted((r3/'figures').iterdir()):
        if p.is_file() and p.suffix.lower() in {'.pdf','.png','.svg'}: cp(p,out/'figures'/p.name,'text' if p.suffix.lower()=='.svg' else 'binary')
    sources={
      'uob_native_grid_summary.json':'reports/uob100dy_scatty_native_grid_external_request_v1_score_return/unpacked/evidence/summary.json',
      'uob_estimator_sensitivity_summary.json':'reports/uob100dy_nd_estimator_sensitivity_external_request_v1_score_return/unpacked/evidence/summary.json',
      'uob_atomic_decoration_summary.json':'reports/uob100dy_atomic_decoration_external_request_v4_score_return/unpacked/evidence/summary.json',
      'odt_native_representation_aggregate.json':'reports/odt_native_representation_external_rtx3090_20260806/odt_native_representation_external_return/evidence/aggregate.json',
      'numagsans_condensed_evidence.json':'benchmark_results/numagsans_example3_strongest_baseline_external_evidence_20260825.json'}
    for n,p in sources.items(): cp(src/p,out/'evidence'/n)
    cp(src/'reports/acfo_full_calculation_fairness_audit_20260901/comparison_ledger.csv',out/'evidence/comparator_fairness_ledger.csv','text')
    archives=[
      ('benchmark_results/external_validation/waxs_100_state_eps_frontier_external_request_v1_162d3bbfd821_20260825_195415_c9eea9a53a20492fa2212737e6ccc746/return/waxs_100_state_eps_frontier_external_return.zip',{
        'waxs_100_state_eps_frontier_external_return/WAXS_100_STATE_EPS_FRONTIER_PROTOCOL.json':'waxs_protocol.json','waxs_100_state_eps_frontier_external_return/evidence/SUMMARY.json':'waxs_external_summary.json','waxs_100_state_eps_frontier_external_return/evidence/EXTERNAL_RETURN_AUDIT.json':'waxs_external_audit.json','waxs_100_state_eps_frontier_external_return/evidence/waxs_100_state_eps_frontier.json':'waxs_timing_and_accuracy.json','waxs_100_state_eps_frontier_external_return/evidence/waxs_100_state_eps_frontier.csv':'waxs_frontier.csv'}),
      ('benchmark_results/external_validation/numagsans_example3_strongest_baseline_438e2555ef28_20260825_200600_bd7dfd67307e42ea832c037c4628ded2/return/numagsans_example3_strongest_baseline_external_return.zip',{
        'numagsans_example3_strongest_baseline_external_return/NUMAGSANS_EXAMPLE3_STRONGEST_BASELINE_PROTOCOL.json':'numagsans_protocol.json','numagsans_example3_strongest_baseline_external_return/evidence/AUDIT.json':'numagsans_external_audit.json','numagsans_example3_strongest_baseline_external_return/evidence/numagsans_example3_strongest_baseline.json':'numagsans_timing_and_accuracy.json','numagsans_example3_strongest_baseline_external_return/evidence/environment.json':'numagsans_environment.json'})]
    for rel,members in archives:
        arc=src/rel
        with zipfile.ZipFile(arc) as z:
            for member,name in members.items():
                b=z.read(member); dest=out/'evidence'/name
                if name.endswith('.json'): wjson(dest,norm(json.loads(b.decode('utf-8-sig')),src))
                else: wtext(dest,b.decode('utf-8-sig'))
                prov.append({'source':rel,'source_archive_sha256':hfile(arc),'archive_member':member,'member_sha256':hbytes(b),'public_copy':dest.relative_to(out).as_posix(),'public_sha256':hfile(dest),'normalization':'paths normalized and/or UTF-8 LF; scientific values unchanged'})
    for n in ['MAIN_AUDIT.json','SI_AUDIT.json','NUMERICAL_AUDIT.json','FIGURE_AUDIT.json','STRUCTURE_AUDIT.json','FAIRNESS_CLAIM_AUDIT.json','FINAL_RELEASE_QA_RECEIPT.json','MANUSCRIPT_BUILD_RECEIPT.json']: cp(r3/n,out/'audits'/n)
    readme='''# ACFO v0.2.0 manuscript evidence

This directory is the public evidence and figure-source bundle for *Geometry factorization and representation dispatch for repeated curved Fourier inference*.

- WAXS: 8.279-fold versus the strongest eligible fused FINUFFT Type-3 comparator (95% paired-bootstrap interval 8.166-8.396), with preserved candidate ordering for the prespecified noise realization.
- NuMagSANS Example 3: 1.687-fold for the shared 800-orientation Fourier backbone versus affine Type-2 (95% orientation-resampling interval 1.684-1.690). The five packing reductions are excluded from this ratio.
- UOB-100(Dy): measured-data correctness example with primary candidate J/T=0.134132 and numerical agreement among ACFO, exact FFT and direct sums. A specialized eligible representation remains faster.
- ODT: native Type-2 forward and Type-1 adjoint comparisons, with setup, hot application and dtype reported separately.
- High-NA Debye-Wolf: Supplementary accuracy-control evidence only.

Run `python scripts/verify_ncs_release_v020.py` from the repository root. Third-party raw archives are not redistributed. `PROVENANCE.json` records original and normalized-copy hashes; normalization changes machine-local paths and line endings only.
'''
    wtext(out/'README.md',readme)
    wjson(out/'PROVENANCE.json',{'schema':'acfo-ncs-release-v0.2.0-provenance','created_utc':datetime.now(timezone.utc).isoformat(),'third_party_raw_archives_redistributed':False,'artifacts':prov})
    wjson(out/'MANIFEST.json',{'schema':'acfo-ncs-release-v0.2.0-manifest','created_utc':datetime.now(timezone.utc).isoformat(),'repository':'https://github.com/SCIMMS/ACFO','version':'0.2.0','license':'ACFO Citation-Required License 1.0','files':inventory(out)})
    print(out)
if __name__=='__main__': main()
