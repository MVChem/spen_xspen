"""Train on GPU 1, then automatically run the frozen simulation/real comparison."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DATA = PROJECT.parent/'data'
GPU1 = 'GPU-0d7eec80-ebfc-ed01-d338-5172ed91d565'


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=PROJECT/'runs/rodent96_dit20m_260917')
    parser.add_argument('--data', type=Path, default=DATA/'rodent96_expanded_260917')
    parser.add_argument('--gpu-uuid', default=GPU1)
    parser.add_argument('--sampling', choices=['uniform', 'balanced'], default='uniform')
    parser.add_argument('--select-lambda', action='store_true')
    args = parser.parse_args()
    args.out = args.out.resolve(); args.data = args.data.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out/'.campaign.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        PYTHONUNBUFFERED='1',
        SPEN_REFERENCE_ROOT=str(DATA/'prior96_0911_260916/scanner_reference'))
    env.pop('PYTHONPATH', None)  # Use the pip-installed SPENPy release.
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'ROLE_RANK',
                'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(key, None)
    # Resolve the venv directory, not its Python symlink (which points outside the venv).
    python = str(Path(sys.prefix).resolve()/'bin/python')
    train_out = args.out/'training'
    command = [python, '-u', str(HERE/'train.py'), '--data', str(args.data), '--out', str(train_out),
        '--steps', '60000', '--microbatch', '48', '--accumulation', '2', '--lr', '0.0002',
        '--warmup', '500', '--save-every', '1000', '--sample-every', '5000', '--seed', '19',
        '--sampling', args.sampling]
    inverse = [python, '-u', str(HERE/'evaluate_inverse.py'), '--data', str(args.data),
        '--checkpoint', str(train_out/'model_ema.pt'), '--expected-step', '60000', '--out', str(args.out/'evaluation')]
    if args.select_lambda: inverse.append('--select-lambda')
    if args.sampling == 'balanced': inverse += ['--model-label', 'DiT · balanced data']
    config = dict(gpu_uuid=args.gpu_uuid, requested_gpu_index=1, steps=60000, train_command=command,
                  inverse_command=inverse, data=str(args.data), python=python,
                  sampling=args.sampling, select_lambda=args.select_lambda)
    if (args.out/'campaign_config.json').exists():
        if json.loads((args.out/'campaign_config.json').read_text()) != config:
            raise ValueError('Campaign config changed; choose a new output directory')
    else:
        write_json(args.out/'campaign_config.json', config)
        # Save a reviewable copy of all source files used by the new campaign.
        sources = list(HERE.glob('*.py')) + list((HERE.parent/'prior96').glob('*.py')) + list((HERE.parent/'core').glob('*.py')) + [HERE.parent/'latent48/dit.py']
        source_hashes = {}
        for source in sources:
            relative = source.relative_to(HERE.parent)
            content = source.read_bytes()
            dest = args.out/'source_snapshot'/relative; dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content); source_hashes[str(relative)] = hashlib.sha256(content).hexdigest()
        write_json(args.out/'source_sha256.json', source_hashes)
    status = dict(pid=os.getpid(), gpu_uuid=args.gpu_uuid, target_steps=60000)
    def update(**values):
        status.update(**values, updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.out/'campaign_status.json', status)
        print(json.dumps(status), flush=True)
    child = None
    def terminate(signum, frame):
        if child and child.poll() is None: child.terminate()
        update(status='interrupted', signal=signum)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, terminate); signal.signal(signal.SIGINT, terminate)
    try:
        for stage, cmd, completion in [('training', command, train_out/'completed.json'),
                                      ('inverse_evaluation', inverse, args.out/'evaluation/completed.json')]:
            if completion.exists():
                record = json.loads(completion.read_text())
                if record.get('step', record.get('checkpoint_step')) != 60000:
                    raise ValueError(f'{stage} completion is not at 60000 steps')
                continue
            if stage == 'training' and (train_out/'latest.pt').exists():
                cmd = cmd + ['--resume', str(train_out/'latest.pt')]
            if stage == 'inverse_evaluation':
                for relative, digest in json.loads((args.out/'source_sha256.json').read_text()).items():
                    if hashlib.sha256((HERE.parent/relative).read_bytes()).hexdigest() != digest:
                        raise ValueError(f'Source changed during training: {relative}; review before evaluating')
            with (args.out/f'{stage}.log').open('a') as log:
                child = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
                update(status='running', stage=stage, child_pid=child.pid, command=cmd)
                result = child.wait()
                if result: raise RuntimeError(f'{stage} failed with exit {result}; see {stage}.log')
            if not completion.exists(): raise RuntimeError(f'{stage} exited without completion marker')
        update(status='complete', stage='complete', child_pid=None)
    except Exception as error:
        update(status='failed', error=str(error), child_pid=None)
        raise


if __name__ == '__main__': main()
