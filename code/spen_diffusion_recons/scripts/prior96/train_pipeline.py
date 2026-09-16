"""Train the two-stage magnitude prior from scratch on one GPU.

Restarting this launcher resumes saved local checkpoints and skips completed
stages. A pipeline lock and the trainer's own lock exclude duplicate writers.
"""
import argparse
import fcntl
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpu', required=True, help='One physical GPU UUID')
    args = p.parse_args()
    args.inputs = args.inputs.resolve(); args.out = args.out.resolve()
    if ',' in args.gpu:
        raise ValueError('This experiment uses exactly one GPU')
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out/'.pipeline.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='2',
               PYTHONPATH=str(PROJECT.parent/'spenpy'),
               SPEN_REFERENCE_ROOT=str(args.inputs/'scanner_reference'))
    for name in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE'):
        env.pop(name, None)
    manifest = args.out/'launch.json'
    if not manifest.exists():
        write_json(manifest, dict(
            command=sys.argv, python=sys.executable, hostname=platform.node(),
            started_utc=datetime.now(timezone.utc).isoformat(), gpu_uuid=args.gpu,
            inputs=str(args.inputs), cwd=str(PROJECT),
            effective_batch=96, microbatch=24, accumulation_steps=4,
            initialization='Random initialization -> new rat step-5000 EMA -> new mouse-mixed training',
            old_weights='Rat V1 checkpoint is used only to print a validation baseline, never for initialization',
            pretraining='5000 optimizer steps; original rat augmentation/validation; cosine horizon 30000',
            mixed_training='30000 optimizer steps; 80% mouse / 20% rat sampling; best validation EMA',
            evaluation='Frozen FOV16 observations, 60 DiffPIR steps, original validation-selected parameters',
            numerical_scope='Same protocol/effective batch; single-GPU RNG stream differs from four-rank training'))
        shutil.copyfile(args.inputs/'provenance.json', args.out/'input_provenance.json')
    state = dict(pid=os.getpid(), gpu_uuid=args.gpu)

    def run(stage, command):
        state.update(status='running', stage=stage, command=command,
                     updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.out/'status.json', state)
        print(json.dumps(state), flush=True)
        with (args.out/f'{stage}.log').open('a') as log:
            child = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid
            write_json(args.out/'status.json', state)
            code = child.wait()
        state.pop('child_pid', None)
        if code:
            raise RuntimeError(f'{stage} exited {code}; see {stage}.log')

    common = [sys.executable, '-u', str(HERE/'train_strong.py'),
              '--local-batch', '24', '--accumulation-steps', '4',
              '--lr-schedule-steps', '30000', '--lr', '0.0002', '--seed', '19',
              '--base-ch', '64', '--save-every', '1000', '--sample-every', '5000',
              '--reference-checkpoint', str(args.inputs/'rat_baseline.pt')]
    try:
        for stage, dataset, steps, protocol in [
                ('rat_pretrain', 'rat_pretrain', 5000, 'rat_pretrain'),
                ('mouse_mixed', 'mouse_mixed', 30000, 'mouse')]:
            out = args.out/stage
            if (out/'completed.json').exists():
                assert json.loads((out/'completed.json').read_text())['step'] == steps
                continue
            command = common + ['--data', str(args.inputs/dataset), '--out', str(out),
                                '--steps', str(steps), '--data-protocol', protocol]
            if (out/'latest.pt').exists():
                command += ['--resume', str(out/'latest.pt')]
            elif stage == 'mouse_mixed':
                command += ['--init-from', str(args.out/'rat_pretrain/latest.pt')]
            run(stage, command)
        evaluation = args.out/'evaluation'
        if not (evaluation/'completed.json').exists():
            # Keep a failed partial evaluation intact, then use a fresh directory.
            if evaluation.exists():
                evaluation.rename(args.out/f'evaluation_incomplete_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{os.getpid()}')
            run('evaluation', [sys.executable, '-u', str(HERE/'evaluate_reconstruction.py'),
                               '--checkpoint', str(args.out/'mouse_mixed/model_ema.pt'),
                               '--inputs', str(args.inputs), '--out', str(evaluation)])
        state.update(status='complete', stage='complete', updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.out/'status.json', state)
    except Exception as exc:
        state.update(status='failed', error=str(exc), updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.out/'status.json', state)
        raise


if __name__ == '__main__':
    main()
