"""Create-only adapters for individual preserved preparation and elliptic workers."""
from pathlib import Path
import argparse, json, os, shutil, subprocess, sys
ROOT=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser();ap.add_argument('experiment',choices=['preparation','elliptic'])
    ap.add_argument('--case',required=True);ap.add_argument('--arm',required=True)
    ap.add_argument('--repeat',type=int,default=0);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();out=a.output.resolve()
    if out.exists():raise FileExistsError('Choose a new output directory')
    from verify import verify
    verify();work=ROOT/'replay';env=dict(os.environ,PYTHONPATH=str(work/'src'),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    if a.experiment=='preparation':
        assert a.arm in ('cpswf','randomized') and int(a.case) in range(4) and a.repeat in range(6)
        subprocess.run([sys.executable,str(work/'scripts/confirm_acfo_research.py'),'worker',
            '--out',str(out),'--task','A','--case',a.case,'--arm',a.arm,'--repeat',str(a.repeat)],
            cwd=work,env=env,check=True)
    else:
        assert a.arm in ('total_update','cap_update','cap_policy','direct') and a.repeat in range(4)
        os.environ.update({k:env[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})
        sys.path[:0]=[str(work),str(work/'scripts'),str(work/'src')]
        from scripts import benchmark_acfo_elliptic_perturbation_sweep as b
        cfg=json.loads(b.PILOT.read_text(encoding='utf-8-sig'));assert a.case in cfg['cases']
        out.mkdir(parents=True)
        for name in ('references.json','forecast.json'):shutil.copy2(b.OUT/name,out/name)
        b.OUT=out
        b.worker(a.case,a.arm,a.repeat,'confirmation')
if __name__=='__main__':main()
