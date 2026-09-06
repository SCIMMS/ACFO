"""Public archive integrity and frozen numerical-gate verification; no timing run."""
from pathlib import Path, PurePosixPath
import argparse, csv, hashlib, json, os, shutil, subprocess, sys, tempfile, zipfile
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    files=read(ROOT/'PUBLIC_MANIFEST.json')['files']
    for name,r in files.items():
        assert not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
        p=ROOT/name;assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],name
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
            and not any(x in p.parts for x in ('__pycache__','.pytest_cache','_runs'))
            and p.suffix not in ('.pyc','.pyd','.so')}
    assert actual==set(files)|{'PUBLIC_MANIFEST.json'},'Unmanifested distribution file'
    work=ROOT/'replay';sys.path.insert(0,str(work/'scripts'))
    from acfo_structure_replication_io import verify_request,jobs,result_metrics
    count=verify_request(work);request=read(work/'REQUEST.json');workers=0
    for host in ('server36','server59'):
        with zipfile.ZipFile(ROOT/'evidence'/(host+'_confirm.public.zip')) as z:
            assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
            for name in z.namelist():
                assert not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
            run=json.loads(z.read('RUN.json'));expected=jobs(work,'all','confirm')
            assert set(run['jobs'])=={x['id'] for x in expected}
            for item in expected:
                name=item['id'];assert run['jobs'][name]['status']=='passed'
                task=json.loads(z.read(name+'/task.json'));assert task==item['task']
                design=read(work/request['systems'][item['system']]['contract'])
                result_metrics(item['system'],json.loads(z.read(name+'/result.json')),design,task)
                workers+=1
    ell=work/'reports/acfo_elliptic_perturbation_sweep_20260905_v1'
    rows=list((ell/'confirmation').glob('*.json'));assert len(rows)==512
    assert all(read(p)['passed'] for p in rows)
    prep=work/'reports/acfo_research_confirmation_20260905_v1'
    records=list(prep.glob('A_c*_r*.json'));assert len(records)==48
    assert all(all(r['passed'] for r in read(p)) for p in records)
    assert len(list(prep.glob('A_c*_r*.npz')))==48
    closure=work/'reports/acfo_cpswf_operator_algebra_20260905_v2/comparison.csv'
    assert len(list(csv.DictReader(closure.read_text(encoding='utf-8-sig').splitlines())))==432
    for rel in ['reports/acfo_elliptic_perturbation_sweep_20260905_v1/prospective_contract.json',
                'validation_contracts/acfo_elliptic_perturbation_sweep_20260905_confirmation.json']:
        for p,h in read(work/rel)['hashes'].items():assert sha(work/p)==h,p
    return dict(public_files=len(files),replay_files=count,external_workers=workers,
                elliptic_workers=len(rows),preparation_workers=len(records),closure_rows=432)
def smoke():
    with tempfile.TemporaryDirectory(prefix='acfo-v040-') as tmp:
        dest=Path(tmp)/'replay';shutil.copytree(ROOT/'replay',dest)
        env=dict(os.environ,PYTHONPATH=str(dest/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        tests=['test_slab_update_adapter.py','test_radial_4f_phase_relay.py',
               'test_waxs_detector_tangent_adapter.py','test_odt_weight_coupling_adapter.py',
               'test_acfo_structure_replication_package.py','test_acfo_research_adapters.py',
               'test_cpswf_operator_algebra_pilot.py','test_acfo_elliptic_perturbation_sweep.py']
        subprocess.run([sys.executable,'-m','pytest','-q',*[str(Path('tests')/p) for p in tests]],
                       cwd=dest,env=env,check=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    print(json.dumps(verify(),indent=2))
    if args.smoke:smoke()
