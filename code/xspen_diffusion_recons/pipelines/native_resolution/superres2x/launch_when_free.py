"""Run this 2x reconstruction only on an observed idle GPU; never signal other jobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from campaign import gpu_inventory
from run_guard import acquire_run_lock


def write_status(value):
    temporary = HERE / 'gpu_launch_status.json.tmp'
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(HERE / 'gpu_launch_status.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', nargs='+', type=int, required=True, help='Physical GPU indices in preferred order')
    args = parser.parse_args()
    lock = acquire_run_lock(HERE / 'controller')
    command = [sys.executable, str(HERE / 'reconstruct_2x.py'), '--device', 'cuda']
    order = args.gpus
    counts = {}
    history = []
    while True:
        inventory = gpu_inventory()
        history.append(dict(time=time.time(), gpus=inventory))
        allowed = sorted((g for g in inventory.values() if g['index'] in order), key=lambda g: order.index(g['index']))
        for gpu in allowed:
            counts[gpu['uuid']] = counts.get(gpu['uuid'], 0) + 1 if gpu['available'] else 0
        stable = [g for g in allowed if counts[g['uuid']] >= 3]
        write_status(dict(stage='waiting_for_idle_gpu', checks=history[-3:]))
        if stable:
            selected = stable[0]
            if gpu_inventory()[selected['uuid']]['available']:
                break
        time.sleep(10)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=selected['uuid'], OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    with (HERE / 'reconstruction.log').open('a') as log:
        proc = subprocess.Popen(command, env=env, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        record = dict(stage='running', pid=proc.pid, command=command, gpu=selected,
                      launched_unix=time.time(), idle_checks=history[-3:],
                      action_policy='Observed idle GPU only. No signals, GPU resets, or checkpoint copies.')
        write_status(record)
        print(json.dumps(record), flush=True)
        code = proc.wait()
    record.update(stage='complete' if code == 0 else 'failed', exit_code=code, finished_unix=time.time())
    write_status(record)
    print(json.dumps(record), flush=True)
    if code:
        sys.exit(code)


if __name__ == '__main__':
    main()
