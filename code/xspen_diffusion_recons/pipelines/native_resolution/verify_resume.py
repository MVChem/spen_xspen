"""Verify actual train_native entrypoint across a fresh-process checkpoint resume.

Two isolated four-step runs share a learning-rate horizon. The second resumes
the first run's step-2 snapshot; compare their actual step-3 inputs and updates.
This verifies restart mechanics, not convergence or reconstruction performance.
"""
import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from native_model import NativePrior
from native_data import NativeData
import train_native
from utils import sha256, write_json


def digest_tensor(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str((value.dtype, tuple(value.shape))).encode())
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def digest_state(state):
    if isinstance(state, torch.Tensor):
        return digest_tensor(state)
    return hashlib.sha256(pickle.dumps(state, protocol=5)).hexdigest()


def worker(args):
    capture = dict(last_optimizer_step=2 if args.resume else 0)
    original_sample = NativeData.sample
    original_forward = NativePrior.forward
    original_step = torch.optim.AdamW.step
    original_save = train_native.save_checkpoint
    original_rng = train_native.rng_state

    # Deliberate sentinel draws ensure all four RNG states are nontrivial when
    # saved. These are test instrumentation, identical in both execution arms.
    def snapshot_rng():
        import random
        random.random()
        np.random.random(7)
        torch.rand(7)
        torch.rand(7, device='cuda')
        return original_rng()

    def sample(self, batch):
        before = original_rng()
        result = original_sample(self, batch)
        capture['rng_before_batch'] = {key: digest_state(value) for key, value in before.items()}
        capture['batch'] = result.detach()
        return result

    def forward(self, x, sigma):
        if self.training and capture['last_optimizer_step'] == 2:
            capture['model_before_forward'] = {name: digest_tensor(value) for name, value in self.state_dict().items()}
        pred = original_forward(self, x, sigma)
        if self.training:
            capture.update(net=self, noisy=x.detach(), sigma=sigma.detach(), pred=pred.detach())
        return pred

    def step(self, *a, **kw):
        parameter = self.param_groups[0]['params'][0]
        previous = int(self.state.get(parameter, {}).get('step', 0))
        if previous == 2:
            capture['optimizer_before_update'] = {str(key): {name: digest_state(value) for name, value in state.items()}
                                                 for key, state in self.state_dict()['state'].items()}
        result = original_step(self, *a, **kw)
        index = int(self.state[parameter]['step'])
        capture['last_optimizer_step'] = index
        if index == 3:
            net = capture['net']
            sigma = capture['sigma']
            weight = (sigma.square() + net.sigma_data**2) / (sigma * net.sigma_data).square()
            loss = (weight * (capture['pred'] - capture['batch']).square()).mean()
            model = {name: value.detach().cpu().clone() for name, value in net.state_dict().items()}
            optimizer = self.state_dict()
            optimizer_hashes = {str(key): {name: digest_state(value) for name, value in state.items()}
                                for key, state in optimizer['state'].items()}
            summary = dict(step=index, loss=float(loss),
                           batch_sha256=digest_tensor(capture['batch']),
                           sigma_sha256=digest_tensor(sigma),
                           noisy_input_sha256=digest_tensor(capture['noisy']),
                           prediction_sha256=digest_tensor(capture['pred']),
                           model_before_forward=capture['model_before_forward'],
                           optimizer_before_update=capture['optimizer_before_update'],
                           rng_before_batch=capture['rng_before_batch'],
                           rng_after_update={key: digest_state(value) for key, value in original_rng().items()},
                           model_sha256={name: digest_tensor(value) for name, value in model.items()},
                           optimizer_sha256=optimizer_hashes,
                           optimizer_param_groups=optimizer['param_groups'])
            write_json(args.work / 'step3.json', summary)
            torch.save(model, args.work / 'step3_model.pt')
        return result

    def save(path, value):
        original_save(path, value)
        if Path(path).name == 'latest.pt' and value.get('step') == 2 and not args.resume:
            shutil.copyfile(path, args.work / 'checkpoint_step2.pt')

    NativeData.sample = sample
    NativePrior.forward = forward
    torch.optim.AdamW.step = step
    train_native.save_checkpoint = save
    train_native.rng_state = snapshot_rng
    sys.argv = ['train_native.py', '--data', str(args.data), '--out', str(args.work),
                '--steps', '4', '--batch', str(args.batch), '--save-every', '2']
    if args.resume:
        sys.argv += ['--resume', str(args.resume)]
    if args.deterministic:
        sys.argv += ['--deterministic']
    train_native.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=HERE / 'data_pilot/p4mm_mm1p5')
    parser.add_argument('--work', type=Path, default=HERE / 'smoke/resume_verification')
    parser.add_argument('--out', type=Path, default=HERE / 'smoke/resume_verification.json')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--deterministic', action='store_true', help='Pass the production --deterministic flag to train_native.py')
    args = parser.parse_args()
    args.data = args.data.resolve(); args.work = args.work.resolve()
    if args.worker:
        worker(args)
        return
    if args.work.exists():
        raise FileExistsError('Use a fresh verification work directory')
    args.work.mkdir(parents=True)
    continuous = args.work / 'continuous'
    resumed = args.work / 'resumed'
    base_cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', '--data', str(args.data), '--batch', str(args.batch)]
    if args.deterministic:
        base_cmd += ['--deterministic']
    with (args.work / 'continuous.log').open('w') as log:
        subprocess.run(base_cmd + ['--work', str(continuous)], check=True, stdout=log, stderr=subprocess.STDOUT)
    resumed.mkdir()
    shutil.copyfile(continuous / 'config.json', resumed / 'config.json')
    with (args.work / 'resumed.log').open('w') as log:
        subprocess.run(base_cmd + ['--work', str(resumed), '--resume', str(continuous / 'checkpoint_step2.pt')],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    a = json.loads((continuous / 'step3.json').read_text())
    b = json.loads((resumed / 'step3.json').read_text())
    comparisons = {key: a[key] == b[key] for key in a}
    ma = torch.load(continuous / 'step3_model.pt', map_location='cpu', weights_only=True)
    mb = torch.load(resumed / 'step3_model.pt', map_location='cpu', weights_only=True)
    max_difference = max(float((ma[name] - mb[name]).abs().max()) for name in ma)
    report = dict(passed=all(comparisons.values()), comparisons=comparisons,
                  next_step=3, restored_checkpoint_step=2, total_steps=4, batch=args.batch,
                  data=str(args.data), manifest_sha256=sha256(args.data / 'manifest.json'),
                  checkpoint_sha256=sha256(continuous / 'checkpoint_step2.pt'),
                  continuous_loss=a['loss'], resumed_loss=b['loss'],
                  maximum_model_weight_difference=max_difference,
                  model_tensors_compared=len(ma), visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  torch=torch.__version__, work=str(args.work),
                  production_deterministic_flag=args.deterministic,
                  deterministic_diagnostic_override=False,
                  train_source_sha256=sha256(HERE / 'train_native.py'),
                  verifier_sha256=sha256(Path(__file__)),
                  method='Two fresh processes execute the actual train_native.main; resume at step2 and compare step3 batch, sigma, noisy input, prediction, loss, all four RNG states, optimizer and model tensors.',
                  instrumentation='Hooks record actual inputs/updates. Sentinel draws from Python/NumPy/CPU torch/CUDA torch immediately precede RNG snapshots so restoration is tested with nontrivial state. Determinism, when requested, is configured only by the production train_native --deterministic CLI flag.',
                  interpretation='Checkpoint restart mechanics only; these short-run losses do not estimate model quality.')
    write_json(args.out, report)
    print(json.dumps(report, indent=2), flush=True)
    if not report['passed']:
        raise AssertionError('Fresh-process exact resume differs; inspect report')


if __name__ == '__main__':
    main()
