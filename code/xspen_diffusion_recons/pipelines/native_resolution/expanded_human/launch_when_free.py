"""Run one authorized reconstruction job only after three idle GPU checks.

Only the explicitly selected GPUs are considered. This output directory is locked.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from campaign import gpu_inventory
from run_guard import acquire_run_lock


def write_json(path,value):
    temp=path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', nargs='+', type=int, required=True, help='Physical GPU indices in preferred order')
    args = parser.parse_args()
    lock=acquire_run_lock(HERE/'reconstruction_controller')
    selection=HERE/'selection.json'
    if not selection.exists():raise FileNotFoundError(selection)
    command=[sys.executable,str(HERE/'reconstruct_expanded.py'),'--selection',str(selection),
             '--out',str(HERE/'evaluation'),'--device','cuda']
    counts={}
    history=[]
    while True:
        available=gpu_inventory()
        history.append(dict(time=time.time(),gpus=available))
        allowed=[v for v in available.values() if v['index'] in args.gpus]
        allowed.sort(key=lambda g: args.gpus.index(g['index']))
        for gpu in allowed:
            uuid=gpu['uuid']
            counts[uuid]=counts.get(uuid,0)+1 if gpu['available'] else 0
        stable=[g for g in allowed if counts[g['uuid']]>=3]
        write_json(HERE/'gpu_launch_status.json',dict(stage='waiting_for_idle_gpu',checks=history[-3:]))
        if stable:
            selected=stable[0]
            # Recheck compute processes immediately before launch.
            if gpu_inventory()[selected['uuid']]['available']:break
        time.sleep(10)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=selected['uuid'],OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    with (HERE/'reconstruction.log').open('a') as log:
        proc=subprocess.Popen(command,env=env,cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
        record=dict(stage='running',pid=proc.pid,command=command,gpu=selected,launched_unix=time.time(),
                    idle_checks=history[-3:],action_policy='Launch on observed idle GPU only; no process signals or GPU resets')
        write_json(HERE/'gpu_launch_status.json',record)
        print(json.dumps(record),flush=True)
        code=proc.wait()
    record.update(stage='complete' if code==0 else 'failed',exit_code=code,finished_unix=time.time())
    write_json(HERE/'gpu_launch_status.json',record)
    print(json.dumps(record),flush=True)
    if code:sys.exit(code)


if __name__=='__main__':main()
