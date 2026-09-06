"""Fresh-process paired confirmation, fixed cases and seeds before execution."""
import argparse,json,hashlib,shutil,subprocess,sys
from pathlib import Path
from validate_acfo_research_abcd import ROOT,CONTRACT,save,freeze,a_worker
from validate_acfo_research_inverse import worker as d_worker


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['large-pilot','confirm','worker'])
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--task',default='A')
    ap.add_argument('--case',type=int,default=0);ap.add_argument('--arm',default='cpswf')
    ap.add_argument('--repeat',type=int,default=0);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    p=json.loads(CONTRACT.read_text())
    config={'A_cases':[[350,441.3799661723198,0],[1024,441.3799661723198,0],[2048,441.3799661723198,0],[512,220.6899830861599,0]],
        'A_arms':['cpswf','randomized'],'D_cases':[7,16,17],'D_arms':['acfo','direct'],
        'repeats':6,'pair_order':'alternate AB/BA by repeat; both arms get the same seed',
        'sample_unit':'fresh process per arm; primary tolerance memory only (secondary process HWM includes prior audit)',
        'status':'local CPU evidence, six paired process repeats; selected after pilot, no population or hardware universality'}
    p['A']['cases']=config['A_cases'];p['D']['quadrature_rule']='theta_gauss_with_sin_weight'
    if args.stage=='worker':
        if args.task=='A':a_worker(args,p)
        else:d_worker(args,p)
        return
    freeze(args.out)
    save(args.out/'confirmation_contract.json',config)
    # Source snapshots make the hash receipt replayable after later report edits.
    snapshots=args.out/'source';snapshots.mkdir()
    for name in ['confirm_acfo_research.py','validate_acfo_research_abcd.py','validate_acfo_research_inverse.py','acfo_research_adapters.py']:
        shutil.copyfile(Path(__file__).with_name(name),snapshots/name)
    if args.stage=='large-pilot':
        jobs=[('A',2,arm,0) for arm in config['A_arms']]
    else:
        truthdir=ROOT/'reports/acfo_research_inverse_20260905_pilot_v2'
        for name in ['D_truth.npz','D_truth.json','D_protocol_v2.json']:
            shutil.copyfile(truthdir/name,args.out/name)
        jobs=[]
        for rep in range(config['repeats']):
            for task,ids,arms in [('A',range(4),config['A_arms']),('D',config['D_cases'],config['D_arms'])]:
                for case in ids:
                    for arm in (arms if rep%2==0 else arms[::-1]):jobs.append((task,case,arm,rep))
    failures=[]
    for task,case,arm,rep in jobs:
        log=args.out/f'{task}_c{case}_{arm}_r{rep}.log'
        with log.open('x') as f:
            r=subprocess.run([sys.executable,str(Path(__file__)),'worker','--out',str(args.out),
                '--task',task,'--case',str(case),'--arm',arm,'--repeat',str(rep)],stdout=f,stderr=subprocess.STDOUT)
        print(f'{task} case={case} arm={arm} repeat={rep} exit={r.returncode}',flush=True)
        if r.returncode:failures.append(str(log))
    save(args.out/'completion.json',{'jobs':len(jobs),'failed_logs':failures})


if __name__=='__main__':main()
