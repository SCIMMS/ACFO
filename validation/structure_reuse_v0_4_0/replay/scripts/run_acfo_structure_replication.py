"""Run frozen workers sequentially; preserve failures and return auditable data."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from acfo_structure_replication_io import read,write,sha,verify_request,jobs,result_metrics,aggregate,zip_return,audit_return

ROOT=Path(__file__).resolve().parents[1]


def command(args):
    try:
        r=subprocess.run(args,capture_output=True,text=True,timeout=30)
        return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
    except (OSError,subprocess.TimeoutExpired) as e:return dict(error=str(e))


def gpu_snapshot():
    return dict(devices=command(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.total,memory.free,utilization.gpu','--format=csv,noheader']),
                processes=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader']))


def probe(profile):
    # Executed in its own process; release CUDA contexts before timing workers.
    sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
    import numpy as np
    import scipy
    import waxs_cake
    info=dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),cpu=platform.processor(),
        cpu_count=os.cpu_count(),numpy=np.__version__,scipy=scipy.__version__,
        source=str(Path(waxs_cake.__file__).resolve()),versions={},errors=[])
    if not Path(waxs_cake.__file__).resolve().is_relative_to(ROOT/'src'):info['errors'].append('Wrong waxs_cake source')
    for name in ('periodictable','pytest','pybind11','setuptools','torch','cupy-cuda12x','cupy-cuda13x'):
        try:info['versions'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    if profile in ('gpu','all'):
        try:
            import torch
            import cupy
            from waxs_cake import _cpp_odt
            info['native_source']=str(Path(_cpp_odt.__file__).resolve())
            info['native_sha256']=sha(_cpp_odt.__file__)
            if not Path(_cpp_odt.__file__).resolve().is_relative_to(ROOT/'src'):raise RuntimeError('Native module outside request')
            if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
            info['gpu']=torch.cuda.get_device_name(0);info['torch_cuda']=torch.version.cuda
            info['gpu_memory_bytes']=torch.cuda.get_device_properties(0).total_memory
            info['cupy_runtime']=cupy.cuda.runtime.runtimeGetVersion()
        except Exception as e:info['errors'].append(repr(e))
    print(__import__('json').dumps(info,indent=2))
    return int(bool(info['errors']))


def environment():
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join([str(ROOT/'src'),str(ROOT/'scripts')])
    env['PYTHONNOUSERSITE']='1'
    env['CUPY_CACHE_DIR']=str(ROOT/'.cupy_cache')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS'):env[key]='1'
    return env


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=['cpu','gpu','all'],default='all')
    p.add_argument('--phase',choices=['smoke','confirm'],default='smoke')
    p.add_argument('--run-id');p.add_argument('--preflight',action='store_true')
    p.add_argument('--probe',action='store_true');p.add_argument('--build-native',action='store_true')
    p.add_argument('--verify-return',type=Path)
    args=p.parse_args()
    if args.probe:return probe(args.profile)
    verify_request(ROOT)
    if args.verify_return:
        result=audit_return(args.verify_return,ROOT)
        write(Path(str(args.verify_return)+'.audit.json'),result)
        print(__import__('json').dumps({k:v for k,v in result.items() if k!='summary'},indent=2))
        return 0 if result['numerical_passed'] else 1
    env=environment()
    if args.build_native:
        env['WAXS_CPP_EXTENSIONS']='odt'
        # Build locally from frozen C++ sources. No installer/download is run.
        return subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=ROOT,env=env).returncode
    probe_result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--probe','--profile',args.profile],
        cwd=ROOT,env=env,capture_output=True,text=True)
    try:metadata=__import__('json').loads(probe_result.stdout)
    except ValueError:metadata=dict(errors=['Probe failed'],stdout=probe_result.stdout,stderr=probe_result.stderr)
    metadata['thread_environment']={k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS')}
    metadata['cuda_visible_devices']=env.get('CUDA_VISIBLE_DEVICES')
    if args.preflight:
        print(__import__('json').dumps(dict(environment=metadata,gpu=gpu_snapshot() if args.profile!='cpu' else None),indent=2))
        return 0 if not metadata.get('errors') else 1
    if metadata.get('errors'):raise RuntimeError(metadata)
    ident=args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',ident):raise ValueError('run-id must contain letters, digits, underscore or hyphen')
    out=ROOT/'_runs'/ident;out.mkdir(parents=True,exist_ok=False)
    archive=out.with_suffix('.zip')
    if archive.exists():raise FileExistsError(archive)
    plan=jobs(ROOT,args.profile,args.phase);spec=read(ROOT/'REQUEST.json')
    run=dict(schema='acfo-structure-return-v1',profile=args.profile,phase=args.phase,
        request_manifest_sha256=sha(ROOT/'PACKAGE_MANIFEST.json'),started_utc=datetime.now(timezone.utc).isoformat(),
        jobs={j['id']:dict(status='pending') for j in plan})
    write(out/'environment.json',metadata);write(out/'REQUEST.json',spec)
    for j in plan:write(out/j['id']/'task.json',j['task'])
    write(out/'RUN.json',run);rows=[]
    try:
        for j in plan:
            name=j['id'];folder=out/name;record=run['jobs'][name]
            if spec['systems'][j['system']]['device']=='gpu':
                snapshot=gpu_snapshot();write(folder/'gpu_before.json',snapshot)
                proc=snapshot['processes']
                if args.phase=='confirm' and proc.get('returncode')==0 and proc.get('stdout','').strip():
                    record.update(status='blocked',reason='GPU compute process present before worker')
                    write(out/'RUN.json',run);print(name+' BLOCKED: GPU busy',flush=True);continue
                if args.phase=='smoke':record['timing_not_for_performance']='Correctness-only smoke permits observed desktop GPU processes; snapshots retained.'
            start=time.perf_counter()
            with (folder/'worker.log').open('w',encoding='utf-8') as log:
                try:
                    r=subprocess.run([sys.executable,str(ROOT/j['script']),'--worker',str(folder/'task.json'),'--result',str(folder/'result.json')],
                        cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                    record['returncode']=r.returncode
                    if r.returncode!=0:raise RuntimeError(f'Worker exit {r.returncode}; see worker.log')
                    contract=read(ROOT/spec['systems'][j['system']]['contract'])
                    metrics=result_metrics(j['system'],read(folder/'result.json'),contract,j['task'])
                    rows.append(dict(system=j['system'],worker=name,metrics=metrics));record['status']='passed'
                except Exception as e:record.update(status='failed',reason=repr(e))
            record['wall_seconds']=time.perf_counter()-start
            if spec['systems'][j['system']]['device']=='gpu':write(folder/'gpu_after.json',gpu_snapshot())
            write(out/'RUN.json',run);print(name+' '+record['status'].upper(),flush=True)
    finally:
        run['finished_utc']=datetime.now(timezone.utc).isoformat();write(out/'RUN.json',run)
        write(out/'summary.json',dict(phase=args.phase,independent_samples_per_system=1 if args.phase=='smoke' else spec['fresh_processes'],
            timing_interpretation='Smoke is packaging QA only. Confirm uses fresh-process paired ratios; phase-sum totals are estimates, not optimizer trajectories.',
            systems=aggregate(rows) if rows else {}))
        zip_return(out,archive)
        audited=audit_return(archive,ROOT);write(Path(str(archive)+'.audit.json'),audited)
        print('RETURN '+str(archive),flush=True)
    return 0 if all(v['status']=='passed' for v in run['jobs'].values()) else 1


if __name__=='__main__':raise SystemExit(main())
