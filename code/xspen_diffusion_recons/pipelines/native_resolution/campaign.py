"""Persistent free-GPU queue for eight independently trained protocol/grid priors."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from run_guard import acquire_run_lock
from utils import write_json
from report_status import build_report


def gpu_inventory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=15)
    values = {}
    for line in result.stdout.splitlines():
        index, uuid, memory, util = [x.strip() for x in line.split(',')]
        values[uuid] = dict(index=int(index), uuid=uuid, memory_mb=int(memory), utilization=int(util))
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=15)
    occupied = {line.split(',')[0].strip() for line in result.stdout.splitlines() if ',' in line}
    for uuid, row in values.items():
        row['available'] = row['memory_mb'] < 128 and row['utilization'] <= 5 and uuid not in occupied
    return values


def process_matches(job):
    try:
        actual = Path(f'/proc/{job["pid"]}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
        return actual == job['command']
    except (KeyError, FileNotFoundError, PermissionError, ProcessLookupError):
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', nargs='+', required=True, help='Explicit physical GPU UUID allowlist')
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    lock = acquire_run_lock(HERE/'campaign')
    if not (HERE/'startup_verified.json').exists():
        raise RuntimeError('Missing recorded preflight verification')
    protocols = json.loads((HERE/'protocols.json').read_text())
    profiles = protocols['profiles']
    jobs = [dict(id=profile['id']+'_'+grid['id'], profile=profile['id'], grid=grid['id'],
                 shape=grid['shape'], native=grid['acquired_native'], stage='pending')
            for profile in profiles for grid in profile['grids']]
    jobs.sort(key=lambda j: (not j['native'], -min(j['shape']), j['id']))
    active = {}
    free_counts = {g: 0 for g in args.gpus}
    status_path = HERE/'campaign/status.json'
    if status_path.exists() and not args.resume:
        raise FileExistsError('Campaign already exists; inspect status and use --resume only after its owner exits')
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    if previous and (previous['steps_per_model'] != args.steps or previous['batch'] != args.batch):
        raise ValueError('Campaign resume must preserve steps and batch')
    previous_jobs = {j['id']: j for j in previous.get('jobs', [])}
    for job in jobs:
        out = HERE/'runs'/job['id']
        old = previous_jobs.get(job['id'], {})
        if old.get('evaluation_out'):
            job['evaluation_out'] = old['evaluation_out']
        if (out/'completed.json').exists():
            job['stage'] = 'trained'
        if (Path(job.get('evaluation_out', HERE/'evaluation'/job['id']))/'summary.json').exists():
            job['stage'] = 'complete'
        if process_matches(old):
            job.update(old)
            job['stage'] = 'external_evaluating' if 'evaluat' in old['stage'] else 'external_training'
    def persist(stage='running', **extra):
        write_json(status_path, dict(stage=stage, pid=os.getpid(), time=time.time(), steps_per_model=args.steps,
                                    batch=args.batch, allowed_gpus=args.gpus, jobs=jobs, **extra))
        build_report()
    def launch(job, gpu, evaluation=False):
        out = HERE/'runs'/job['id']
        out.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
        if evaluation:
            evaluation_out = HERE/'evaluation'/job['id']
            if evaluation_out.exists() and any(evaluation_out.iterdir()):
                evaluation_out = HERE/'evaluation'/f'{job["id"]}_attempt_{time.time_ns()}'
            job['evaluation_out'] = str(evaluation_out)
            command = [sys.executable, '-u', str(HERE/'evaluate_native.py'), '--checkpoint', str(out/'model_ema.pt'),
                       '--data', str(HERE/'data'/job['id']), '--out', str(evaluation_out),
                       '--steps', '60', '--limit', '12']
        else:
            manifest = json.loads((HERE/'data'/job['id']/'manifest.json').read_text())
            if not manifest.get('complete', False):
                raise ValueError('Full campaign cannot train on pilot data')
            command = [sys.executable, '-u', str(HERE/'train_native.py'), '--data', str(HERE/'data'/job['id']),
                       '--out', str(out), '--steps', str(args.steps), '--batch', str(args.batch), '--deterministic']
            if (out/'latest.pt').exists():
                if not args.resume:
                    raise FileExistsError(f'Existing checkpoint for {job["id"]}')
                command += ['--resume', str(out/'latest.pt')]
        log = (out/('evaluation.log' if evaluation else 'train.log')).open('a')
        proc = subprocess.Popen(command, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
        active[gpu] = (proc, log, job)
        job.update(stage='evaluating' if evaluation else 'training', gpu=gpu, pid=proc.pid, command=command, launched_unix=time.time())
        print(json.dumps(dict(event='launch', **job)), flush=True)
    persist()
    try:
        while True:
            for job in jobs:
                if job['stage'].startswith('external_') and not process_matches(job):
                    trained = (HERE/'runs'/job['id']/'completed.json').exists()
                    evaluated = (Path(job.get('evaluation_out', HERE/'evaluation'/job['id']))/'summary.json').exists()
                    if evaluated:
                        job['stage'] = 'complete'
                    elif trained and job['stage'] == 'external_training':
                        job['stage'] = 'trained'
                    else:
                        job['stage'] = 'failed'
            for gpu, (proc, log, job) in list(active.items()):
                result = proc.poll()
                if result is None:
                    continue
                log.close()
                del active[gpu]
                free_counts[gpu] = 0
                job['exit_code'] = result
                if result:
                    job['stage'] = 'failed'
                else:
                    job['stage'] = 'trained' if job['stage'] == 'training' else 'complete'
                print(json.dumps(dict(event='exit', **job)), flush=True)
            if all(j['stage'] in ('complete', 'failed') for j in jobs):
                persist('failed' if any(j['stage'] == 'failed' for j in jobs) else 'complete')
                return
            inventory = gpu_inventory()
            adopted_gpus = {j.get('gpu') for j in jobs if j['stage'].startswith('external_')}
            verification_path = HERE/'audit/native_dataset_verification.json'
            verified = verification_path.exists() and json.loads(verification_path.read_text()).get('passed', False)
            for gpu in args.gpus:
                if gpu in active or gpu in adopted_gpus:
                    continue
                free_counts[gpu] = free_counts[gpu]+1 if inventory.get(gpu, {}).get('available') else 0
                if free_counts[gpu] < 3:
                    continue
                if not verified:
                    continue
                # Finish evaluation of a trained model before accepting another long run.
                candidates = [j for j in jobs if j['stage'] == 'trained']
                if candidates:
                    launch(candidates[0], gpu, evaluation=True)
                    continue
                candidates = [j for j in jobs if j['stage'] == 'pending' and (HERE/'data'/j['id']/'manifest.json').exists()]
                if candidates:
                    launch(candidates[0], gpu)
            persist(gpu_inventory=inventory)
            time.sleep(10)
    except Exception as exc:
        persist('controller_failed', error=repr(exc), children_may_still_be_running=True)
        raise


if __name__ == '__main__':
    main()
