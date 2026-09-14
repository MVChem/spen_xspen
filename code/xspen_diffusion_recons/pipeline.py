"""Train the 128-square EDM prior, then evaluate its checkpoint."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from paths import IXI_DATA, SCANNER_DATA
from utils import HERE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', nargs='+', required=True, help='Explicit GPU indices or UUIDs')
    parser.add_argument('--data', type=Path, default=IXI_DATA)
    parser.add_argument('--scanner', type=Path, default=SCANNER_DATA)
    parser.add_argument('--out', type=Path, default=HERE/'runs/human128')
    parser.add_argument('--evaluation-out', type=Path, default=HERE/'runs/evaluation')
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--local-batch', type=int, default=16)
    parser.add_argument('--backend', choices=['gloo', 'nccl'], default='gloo')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if not (args.data/'manifest.json').is_file():
        parser.error('Prepare the data first or pass --data with an existing prepared dataset')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=','.join(args.gpus), OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               f'--nproc_per_node={len(args.gpus)}', str(HERE/'train.py'),
               '--data', str(args.data.resolve()), '--out', str(args.out.resolve()),
               '--steps', str(args.steps), '--local-batch', str(args.local_batch), '--backend', args.backend]
    if args.resume:
        command += ['--resume', str(args.resume.resolve())]
    subprocess.run(command, cwd=HERE, env=env, check=True)
    env['CUDA_VISIBLE_DEVICES'] = args.gpus[0]
    subprocess.run([sys.executable, str(HERE/'evaluate.py'), '--checkpoint', str(args.out.resolve()/'model_ema.pt'),
                    '--data', str(args.data.resolve()), '--scanner', str(args.scanner.resolve()),
                    '--out', str(args.evaluation_out.resolve())], cwd=HERE, env=env, check=True)


if __name__ == '__main__':
    main()
