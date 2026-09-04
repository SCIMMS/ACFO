"""Verify the public distribution and numerical tables, without benchmarking."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parent
def sha(b): return hashlib.sha256(b).hexdigest()
def load(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(n): return list(csv.DictReader((ROOT/'source_data'/n).read_text(encoding='utf-8-sig').splitlines()))
def archive(b):
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        assert len(z.namelist())==len(set(z.namelist())), 'Duplicate archive member'
        assert z.testzip() is None, 'ZIP CRC failure'
        for n in z.namelist():
            assert not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts and '\\' not in n and ':' not in n
        return {n:z.read(n) for n in z.namelist() if not n.endswith('/')}
def manifest(files,m,exclude):
    assert set(files)-{exclude}==set(m['files']), 'Manifest inventory mismatch'
    for n,r in m['files'].items():
        assert len(files[n])==r['bytes'] and sha(files[n])==r['sha256'], n

def verify():
    checks={}
    m=load(ROOT/'PUBLIC_MANIFEST.json')
    files={p.relative_to(ROOT).as_posix():p.read_bytes() for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '_runs' not in p.parts}
    manifest(files,m,'PUBLIC_MANIFEST.json'); checks['public_manifest_files']=len(m['files'])
    rf={n[len('request/'):]:b for n,b in files.items() if n.startswith('request/')}
    rm=json.loads(rf['PACKAGE_MANIFEST.json']); manifest(rf,rm,'PACKAGE_MANIFEST.json')
    assert sha(rf['CLAIM_FREEZE.json'])==rm['claim_freeze_sha256']=='2bec572b6ae8b70ef9f4a3f47b07b4e5f294364189078e1bf852ce36aa8a6f56'
    assert sha(rf['PROTOCOL.json'])==rm['protocol_sha256']
    index=json.loads(rf['COMPONENT_INDEX.json'])
    for c in index['components'].values():
        b=rf[c['archive']]; assert sha(b)==c['archive_sha256'] and len(b)==c['archive_bytes']
        cf={n[len(c['root'])+1:]:v for n,v in archive(b).items()}
        assert sha(cf[c['manifest']])==c['packaged_manifest_sha256']
        manifest(cf,json.loads(cf[c['manifest']]),c['manifest'])
        if c['root']=='extension_component':
            geometry=archive(cf['benchmark_results/aidt_diatom_public_contract.npz'])
            assert 'data.npy' not in geometry and 'source_na_xy.npy' in geometry
    checks['request_components']=len(index['components'])
    provenance=load(ROOT/'PUBLICATION_PROVENANCE.json')
    returns={}
    for n,r in provenance['evidence_archives'].items():
        b=files[n]; assert sha(b)==r['public']['sha256'] and len(b)==r['public']['bytes']
        returns[n]=archive(b)
        for name,blob in returns[n].items():
            if name.endswith(('.zip','.npz')): archive(blob)
    original=json.loads(returns['evidence/server59_original_return.public.zip']['outputs/summary.json'])
    assert original['all_stage_acceptance_passed'] is False
    odt=next(s for s in original['steps'] if s['id']=='odt_cartesian')
    assert odt['returncode']!=0
    repair=load(ROOT/'audits/server59_corrective_odt/repair_receipt.json')
    assert repair['after']['returncode']==0 and repair['scientific_parameters_changed'] is False
    combined=load(ROOT/'audits/campaign_audit.json')
    assert combined['scoped_constructive_numerical_evidence_supported'] is True
    assert combined['all_strict_local_audits_passed'] is False
    assert combined['all_frozen_publication_requirements_fulfilled'] is False
    checks['original_failure_and_scientific_limits_preserved']=True
    waxs=rows('figure_2_waxs.csv'); assert len(waxs)==600
    for machine in ('server36','server59'):
        selected=[r for r in waxs if r['machine']==machine]
        seeds=sorted({r['seed'] for r in selected}); assert len(seeds)==3
        for seed in seeds:
            group=[r for r in selected if r['seed']==seed]; assert len(group)==100
            a=sorted(group,key=lambda r:float(r['acfo_score']))
            b=sorted(group,key=lambda r:float(r['finufft_score']))
            assert [r['state_index'] for r in a]==[r['state_index'] for r in b]
    checks['waxs_rankings_recomputed']=6
    checks['waxs_max_intensity_relative_l2']=max(float(r['intensity_relative_l2']) for r in waxs)
    assert checks['waxs_max_intensity_relative_l2']<=2.91e-7
    radial=rows('figure_3_common_radial.csv'); assert len(radial)==90
    for r in radial:
        assert float(r['max_spectral_residual'])<=float(r['tolerance'])
        assert 0<=int(r['cpswf_rank'])-int(r['shared_svd_rank'])<=1
        assert 0<=int(r['cpswf_rank'])-int(r['individual_svd_lower_bound'])<=1
    checks['radial_spectral_rank_rows']=len(radial)
    coupled=rows('figure_4_coupled.csv'); assert len(coupled)==8
    assert max(float(r['dense_probe_l2']) for r in coupled)<=7.50e-9
    assert max(float(r['prepared_composed_l2']) for r in coupled)<=6.77e-16
    checks['coupled_normal_rows']=len(coupled)
    timing=rows('figure_5_timing.csv'); assert len(timing)==2
    for r in timing:
        assert float(r['interval_low'])<=float(r['ratio'])<=float(r['interval_high'])
        assert int(r['unfused_bytes'])==640032 and int(r['materialized_bytes'])==373248
    checks['retained_array_reduction_percent']=100*(1-373248/640032)
    checks['csv_files']=len(list((ROOT/'source_data').glob('*.csv')))
    assert checks['csv_files']==13
    return {'distribution_verification_passed':True,'checks':checks,
      'scientific_assessment':'Scoped constructive evidence; recorded strict and publication failures remain. No universal speed, full-support or measured-reconstruction claim.',
      'new_external_benchmarks_run':False}

def smoke():
    # A new temporary root prevents modification of the release or saved evidence.
    with tempfile.TemporaryDirectory(prefix='acfo-v030-smoke-') as tmp:
        b=(ROOT/'request/components/extension_request.zip').read_bytes()
        for n,v in archive(b).items():
            p=Path(tmp)/n; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(v)
        work=Path(tmp)/'extension_component'; env=os.environ.copy()
        env.update(PYTHONPATH=os.pathsep.join((str(work/'src'),str(work/'scripts'))),
                   PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        tests=['finite_hankel_cpswf','radial_cpswf','radial_aidt','prepared_cpswf_aidt']
        result=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*[f'tests/test_{n}.py' for n in tests]],cwd=work,env=env,timeout=300)
        assert result.returncode==0,'Focused CPU smoke tests failed'

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); result=verify()
    if args.smoke: smoke(); result['focused_cpu_smoke_tests_passed']=True
    print(json.dumps(result,indent=2))
