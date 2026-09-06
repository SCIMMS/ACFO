"""Uncontended confirmation replay with the unchanged validated worker.

The first confirmation could overlap a GPU regression test. Its outputs are
retained but not used for performance conclusions. Run this only after it ends.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/acfo_waxs_detector_tangent_20260906_confirm_v2'
WORKER=ROOT/'scripts/experiment_waxs_detector_tangent.py'
CONTRACT=ROOT/'validation_contracts/waxs_detector_tangent_v1.json'


def main():
    OUT.mkdir(exist_ok=False,parents=True)
    design=json.loads(CONTRACT.read_text())
    for src,name in ((CONTRACT,'protocol.json'),(WORKER,'source_snapshot.py'),(ROOT/'scripts/waxs_detector_tangent_adapter.py','adapter_snapshot.py'),(Path(__file__),'launcher_snapshot.py')):
        (OUT/name).write_bytes(src.read_bytes())
    results=[]
    for i in range(5):
        task=OUT/f'task_{i}.json';dest=OUT/f'worker_{i}.json'
        task.write_text(json.dumps(dict(case=design['cases'][-1],seed=20260906+i*100)))
        subprocess.run([sys.executable,str(WORKER),'--worker',str(task),'--result',str(dest)],check=True)
        results.append(json.loads(dest.read_text()));print(f'uncontended confirmation {i} PASS',flush=True)
    (OUT/'results.json').write_text(json.dumps(results,indent=2))
    files=[*OUT.iterdir(),WORKER,CONTRACT,Path(__file__),ROOT/'scripts/waxs_detector_tangent_adapter.py',ROOT/'src/waxs_cake/differentiable_acfo.py',ROOT/'src/waxs_cake/gpu_miller.py']
    (OUT/'manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},indent=2))


if __name__=='__main__':main()
