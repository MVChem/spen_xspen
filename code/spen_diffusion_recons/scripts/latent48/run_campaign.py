"""Run a checked latent-DiT experiment and its post-training prior diagnostics."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--vae', required=True, type=Path)
    p.add_argument('--cases', required=True, type=Path)
    p.add_argument('--gpus', default='5,6')
    p.add_argument('--local-batch', type=int, required=True)
    p.add_argument('--micro-batch', type=int)
    p.add_argument('--normalization', type=Path)
    p.add_argument('--steps', type=int, default=60000)
    args = p.parse_args()
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / '.campaign.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    acceptance = json.loads((run / 'vae_acceptance.json').read_text())
    if acceptance.get('accepted') is not True:
        raise ValueError('VAE quality review has not passed')
    if Path(acceptance['vae_path']).resolve() != args.vae.resolve():
        raise ValueError('The selected VAE differs from the reviewed model')
    for report, expected in acceptance['reports'].items():
        if hashlib.sha256(Path(report).read_bytes()).hexdigest() != expected:
            raise ValueError(f'VAE quality report changed: {report}')
    gpus = args.gpus.split(',')
    if len(gpus) != 2 or len(set(gpus)) != 2:
        raise ValueError('Use two distinct GPUs')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, OMP_NUM_THREADS='2',
               NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1', PYTHONUNBUFFERED='1',
               PYTHONPATH=str(PROJECT.parent / 'spenpy'))
    train = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
             str(HERE / 'train_latent_ddp.py'), '--data', str(args.data.resolve()),
             '--vae', str(args.vae.resolve()), '--out', str(run / 'train'),
             '--steps', str(args.steps), '--local-batch', str(args.local_batch)]
    if args.micro_batch:
        train += ['--micro-batch', str(args.micro_batch)]
    if args.normalization:
        train += ['--normalization', str(args.normalization.resolve())]
    if (run / 'train/latest.pt').exists():
        train += ['--resume']
    evaluate = [sys.executable, str(HERE / 'evaluate_prior.py'),
                '--checkpoint', str(run / 'train/model_ema.pt'), '--data', str(args.data.resolve()),
                '--cases', str(args.cases.resolve()), '--out', str(run / 'prior_diagnostics'),
                '--device', 'cuda']
    plan = dict(pid=os.getpid(), started_at=time.time(), gpus=args.gpus,
                target_steps=args.steps, local_batch=args.local_batch,
                global_batch=2 * args.local_batch, vae_acceptance=acceptance,
                stages=dict(train=train, prior_diagnostics=evaluate),
                sequence='VAE quality accepted -> 60000 optimizer updates -> final EMA latent-prior diagnostics',
                evaluation_scope='Latent Gaussian denoising and unconditional samples; no SPEN inversion')
    atomic_json(run / 'campaign_plan.json', plan)
    for stage, command in plan['stages'].items():
        if (run / stage / 'completed.json').exists():
            done = json.loads((run / stage / 'completed.json').read_text())
            if stage == 'train' and done['step'] != args.steps:
                raise ValueError('Wrong target step in completed training')
            continue
        if stage != 'train':
            done = json.loads((run / 'train/completed.json').read_text())
            if done['step'] != args.steps:
                raise ValueError('Training has not reached the target')
        stage_env = env if stage == 'train' else dict(env, CUDA_VISIBLE_DEVICES=gpus[0])
        started = time.time()
        state = dict(status='running', stage=stage, pid=os.getpid(), command=command, started_at=started)
        atomic_json(run / 'campaign_status.json', state)
        with (run / f'{stage}.log').open('a') as log:
            child = subprocess.Popen(command, cwd=PROJECT, env=stage_env, stdout=log, stderr=subprocess.STDOUT)
            atomic_json(run / 'campaign_status.json', dict(state, child_pid=child.pid))
            code = child.wait()
        if code:
            atomic_json(run / 'campaign_status.json', dict(state, status='failed', returncode=code, finished_at=time.time()))
            raise SystemExit(code)
        if not (run / stage / 'completed.json').exists():
            atomic_json(run / 'campaign_status.json', dict(state, status='failed', reason='Missing completion marker'))
            raise SystemExit(1)
    atomic_json(run / 'campaign_status.json', dict(status='completed', pid=os.getpid(), finished_at=time.time()))


if __name__ == '__main__':
    main()
