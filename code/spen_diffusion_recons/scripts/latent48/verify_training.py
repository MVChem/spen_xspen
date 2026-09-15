"""Read-only CPU verification of a paused / resumed latent DiT training smoke run.

Example:
    python scripts/latent48/verify_training.py --run runs/SMOKE \
        --before runs/SMOKE/latest_step000004.pt
    python scripts/latent48/verify_training.py --cpu-only

The checker never initializes CUDA, trains the production model, or changes a
run. JSON evidence is written to stdout so the caller can choose a report path.
Checkpoint files must be trusted local artifacts produced by this trainer.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'prior192'))
from dit import LatentEDM
from train_png_ddp import microbatch_loss_backward


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_microbatch():
    """Unequal final micro-batch must preserve every gradient and the scalar loss."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(260915)
        full = LatentEDM(input_size=8, hidden_size=32, depth=2, num_heads=4,
                         use_bf16=False)
        # Nonzero gates/head exercise the entire network, including attention.
        with torch.no_grad():
            for block in full.net.blocks:
                block.modulation[-1].weight.normal_(std=.035)
            full.net.final_layer.projection.weight.normal_(std=.035)
        micro = copy.deepcopy(full)
        mean = torch.tensor([.4, -.3, .1, 1.7]).reshape(1, 4, 1, 1)
        std = torch.tensor([.2, .5, 1.3, 2.1]).reshape(1, 4, 1, 1)
        raw = torch.randn(7, 4, 8, 8) * std + mean
        clean = (raw - mean) / std
        sigma = torch.tensor([.002, .03, .2, .5, 1., 3., 40.]).reshape(7, 1, 1, 1)
        noisy = clean + sigma * torch.randn_like(clean)
        weight = (sigma.square() + 1.) / sigma.square()
        loss_full = (weight * (full(noisy, sigma) - clean).square()).mean()
        loss_full.backward()
        loss_micro = microbatch_loss_backward(micro, noisy, clean, sigma, weight,
                                             micro_batch=3, distributed=False)
        require(loss_full.dtype == torch.float32 and loss_micro.dtype == torch.float32,
                'CPU loss must remain FP32')
        torch.testing.assert_close(loss_micro, loss_full.detach(), rtol=2e-6, atol=2e-7)
        max_difference = 0.
        nonzero_gradients = 0
        for (name, expected), actual in zip(full.named_parameters(), micro.parameters()):
            require(expected.grad is not None and actual.grad is not None,
                    f'Missing gradient for {name}')
            require(torch.isfinite(actual.grad).all().item(), f'Nonfinite micro gradient: {name}')
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-4, atol=2e-6,
                                       msg=lambda message, name=name: f'{name}: {message}')
            max_difference = max(max_difference, float((actual.grad - expected.grad).abs().max()))
            nonzero_gradients += int(torch.count_nonzero(actual.grad) > 0)
        require(nonzero_gradients == len(list(full.parameters())),
                'Gradient comparison must exercise every trainable parameter')
        return dict(status='passed', batch_size=7, micro_batch_sizes=[3, 3, 1],
                    normalized_channels=4, loss=float(loss_full.detach()),
                    parameter_tensors_with_nonzero_gradients=nonzero_gradients,
                    max_gradient_absolute_difference=max_difference, precision='float32', device='cpu')


def meta_model(config):
    with torch.device('meta'):
        return LatentEDM(**config)


def check_finite_state(state, expected, label):
    require(state.keys() == expected.keys(), f'{label} keys must match exactly the DiT model')
    for name, value in state.items():
        require(isinstance(value, torch.Tensor), f'{label}/{name} is not a tensor')
        require(value.shape == expected[name].shape, f'{label}/{name} shape differs')
        require(value.dtype == torch.float32, f'{label}/{name} must retain FP32 storage')
        require(torch.isfinite(value).all().item(), f'{label}/{name} contains nonfinite values')


def check_optimizer(saved, named_parameters, expected_step):
    optimizer = saved['optimizer']
    ids = [identifier for group in optimizer['param_groups'] for identifier in group['params']]
    require(len(ids) == len(named_parameters) and len(set(ids)) == len(ids),
            'Optimizer must contain exactly one slot per DiT parameter, and no extra VAE slots')
    require(set(ids) == set(optimizer['state']), 'Missing or unexpected AdamW state slots')
    total_elements = 0
    for identifier, (name, parameter) in zip(ids, named_parameters):
        state = optimizer['state'][identifier]
        step = int(state['step'].item() if isinstance(state['step'], torch.Tensor) else state['step'])
        require(step == expected_step, f'AdamW step {step} != {expected_step} for {name}')
        for key in ('exp_avg', 'exp_avg_sq'):
            require(state[key].shape == parameter.shape, f'Optimizer {key} shape differs for {name}')
            require(state[key].dtype == torch.float32, f'Optimizer {key} must be FP32 for {name}')
            require(torch.isfinite(state[key]).all().item(), f'Optimizer {key} is nonfinite for {name}')
        require((state['exp_avg_sq'] >= 0).all().item(), f'Negative AdamW second moment for {name}')
        total_elements += parameter.numel()
    return dict(parameter_tensors=len(ids), parameter_elements=total_elements, step=expected_step,
                vae_parameter_slots=0)


def check_checkpoint(saved, config, expected_step, expected_world):
    require(saved['step'] == expected_step, f'Checkpoint step must be {expected_step}')
    for key in ('model_config', 'manifest_sha256', 'vae_sha256', 'latent_normalization',
                'global_batch', 'training_mode', 'selection'):
        require(saved[key] == config[key], f'Checkpoint/config differ: {key}')
    require(saved['img_resolution'] == 192 and saved['latent_resolution'] == 48,
            'Checkpoint must retain image192 / latent48 resolutions')
    model = meta_model(config['model_config'])
    expected = model.state_dict()
    check_finite_state(saved['model'], expected, 'model')
    check_finite_state(saved['ema'], expected, 'ema')
    named_parameters = list(model.named_parameters())
    optimizer = check_optimizer(saved, named_parameters, expected_step)
    require(optimizer['parameter_elements'] == config['parameters'], 'Parameter count differs from run config')
    states, counters = saved['rng_states'], saved['sampling_counts_by_rank']
    require(len(states) == expected_world and len(counters) == expected_world,
            'Checkpoint must contain RNG / sample counts for every DDP rank')
    totals = []
    for rank, (state, counts) in enumerate(zip(states, counters)):
        require(set(state) == {'cpu', 'cuda'}, f'Unexpected rank {rank} RNG state keys')
        for name, rng in state.items():
            require(isinstance(rng, torch.Tensor) and rng.dtype == torch.uint8
                    and rng.ndim == 1 and rng.numel() >= (100 if name == 'cpu' else 16),
                    f'Invalid rank {rank} {name} RNG state')
        # Validate CPU RNG bytes without initializing any CUDA context.
        torch.Generator(device='cpu').set_state(state['cpu'])
        counts = np.asarray(counts)
        require(counts.shape == (config['image_counts']['train'],) and np.issubdtype(counts.dtype, np.integer)
                and (counts >= 0).all(), f'Invalid rank {rank} sampling counts')
        total = int(counts.sum())
        require(total == expected_step * config['local_batch'], f'Rank {rank} sampling total differs')
        totals.append(total)
    require(sum(totals) == expected_step * config['global_batch'], 'Global sample count differs')
    require(any(not torch.equal(states[0][key], states[1][key]) for key in ('cpu', 'cuda'))
            if expected_world > 1 else True, 'DDP ranks must have distinct random streams')
    return dict(step=expected_step, ranks=expected_world, sample_totals_by_rank=totals,
                total_samples=sum(totals), model_and_ema_finite=True, optimizer=optimizer)


def check_vae(config):
    from vae_codec import FrozenVAE
    root = Path(config['vae'])
    for name, expected in config['vae_sha256'].items():
        require(sha256(root / name) == expected, f'VAE file changed: {name}')
    codec = FrozenVAE(root, device='cpu')
    parameters = list(codec.parameters())
    require(parameters and all(not parameter.requires_grad for parameter in parameters),
            'VAE must freeze all of its parameters')
    codec.train(True)
    require(all(not module.training for module in codec.modules()),
            'VAE must remain in evaluation mode when train() is called')
    require(codec.downsample_factor == 4, 'VAE must downsample by a factor of four')
    require(codec.latent_channels == config['latent_shape'][0], 'VAE latent channels differ')
    return dict(status='passed', device='cpu', parameter_tensors=len(parameters),
                parameter_elements=sum(parameter.numel() for parameter in parameters),
                all_parameters_frozen=True, stays_in_eval_mode=True, checkpoint_hashes_match=True)


def verify_run(run, before_path, after_path, before_step, after_step, target_steps, world):
    config, status = read_json(run / 'config.json'), read_json(run / 'status.json')
    require(config['steps'] == target_steps and status['target_steps'] == target_steps,
            'Pause / resume must retain the original target steps')
    require(config['world_size'] == world and config['global_batch'] == world * config['local_batch'],
            'Run must retain the expected DDP world size / global batch')
    require(status['step'] == after_step and status['status'] == 'paused',
            'Expected a paused resumed smoke run')
    require(not (run / 'completed.json').exists(), 'A smoke pause cannot claim training completed')
    require(config['online_augmentation'] and config['vae_posterior'] == 'mode'
            and config['vae_autocast'] == 'bfloat16', 'Unexpected codec / augmentation training settings')
    norm = read_json(run / 'latent_normalization.json')
    require(norm == config['latent_normalization'], 'Saved normalization differs from config')
    for key in ('manifest_sha256', 'vae_sha256', 'vae_autocast', 'augmentation'):
        require(norm[key] == config[key], f'Normalization identity differs: {key}')
    mean, std = np.asarray(norm['mean']), np.asarray(norm['std'])
    require(mean.shape == std.shape == (config['latent_shape'][0],)
            and np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all(),
            'Invalid latent channel normalization')
    require(sha256(Path(config['data']) / 'manifest.json') == config['manifest_sha256'],
            'Training manifest changed')
    # mmap avoids copying both large AdamW checkpoints into RAM up front.
    before = torch.load(before_path, map_location='cpu', weights_only=False, mmap=True)
    after = torch.load(after_path, map_location='cpu', weights_only=False, mmap=True)
    checks = {'before': check_checkpoint(before, config, before_step, world),
              'after': check_checkpoint(after, config, after_step, world)}
    updated = {}
    named_parameters = dict(meta_model(config['model_config']).named_parameters())
    for state_name in ('model', 'ema'):
        names = [name for name in named_parameters
                 if not torch.equal(before[state_name][name], after[state_name][name])]
        require(names, f'{state_name} did not update during resumed steps')
        require('net.final_layer.projection.weight' in names,
                f'{state_name} output head did not update during resumed steps')
        # Short warmup + adaLN-Zero may leave early layers below an FP32 ULP.
        updated[state_name] = dict(total=len(names), attention_qkv_tensors=sum(
            '.attention.qkv.weight' in name for name in names))
    manifest = read_json(Path(config['data']) / 'manifest.json')
    sampling_weights = torch.tensor([row['sample_weight'] for row in manifest['records']['train']],
                                    dtype=torch.float64)
    replay_checks = []
    for rank in range(world):
        counts_before, counts_after = (np.asarray(state['sampling_counts_by_rank'][rank])
                                      for state in (before, after))
        delta = counts_after - counts_before
        require((delta >= 0).all() and int(delta.sum()) == (after_step - before_step) * config['local_batch'],
                f'Rank {rank} sampling counters were reset or skipped on resume')
        generator = torch.Generator(device='cpu')
        generator.set_state(before['rng_states'][rank]['cpu'])
        expected_delta = np.zeros_like(delta)
        for _ in range(after_step - before_step):
            indices = torch.multinomial(sampling_weights, config['local_batch'], replacement=True,
                                        generator=generator).numpy()
            np.add.at(expected_delta, indices, 1)
        require(np.array_equal(delta, expected_delta),
                f'Rank {rank} resumed sampling differs from replay of its saved CPU RNG')
        require(torch.equal(generator.get_state(), after['rng_states'][rank]['cpu']),
                f'Rank {rank} final CPU RNG differs from exact resumed sampling replay')
        replay_checks.append(dict(rank=rank, resumed_steps=after_step - before_step,
                                  per_image_count_increments_match=True, final_cpu_rng_matches=True))
        for key in ('cpu', 'cuda'):
            require(not torch.equal(before['rng_states'][rank][key], after['rng_states'][rank][key]),
                    f'Rank {rank} {key} RNG did not advance during resume')
    public = torch.load(run / 'model_ema.pt', map_location='cpu', weights_only=False, mmap=True)
    require(public['step'] == after_step, 'Public EMA checkpoint was not updated on resume')
    require(public['latent_normalization'] == norm and public['vae_sha256'] == config['vae_sha256'],
            'Public EMA lost codec provenance')
    for name in after['ema']:
        require(torch.equal(after['ema'][name], public['ema'][name]), f'Public EMA differs: {name}')
    coverage = read_json(run / 'sampling_coverage.json')
    require(coverage['step'] == after_step and coverage['total_samples'] == after_step * config['global_batch'],
            'Sampling coverage report differs from resumed checkpoint')
    metrics = [json.loads(line) for line in (run / 'train_metrics.jsonl').read_text().splitlines() if line]
    require(any(row['step'] == before_step for row in metrics)
            and any(row['step'] == after_step for row in metrics), 'Missing before / after training metrics')
    require(all(np.isfinite(row['loss']) and np.isfinite(row['grad_norm']) for row in metrics),
            'Nonfinite loss or gradient norm in training metrics')
    checks.update(target_steps=target_steps, updated_parameter_tensors=updated,
                  normalization_unchanged=True, per_rank_rng_advanced=True,
                  public_ema_matches=True, sampling_counts_continued=True,
                  cpu_sampling_rng_exact_replay=replay_checks,
                  vae=check_vae(config), status='passed')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--before', type=Path)
    parser.add_argument('--after', type=Path)
    parser.add_argument('--before-step', type=int, default=4)
    parser.add_argument('--after-step', type=int, default=6)
    parser.add_argument('--target-steps', type=int, default=60000)
    parser.add_argument('--world-size', type=int, default=2)
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    if not args.cpu_only and (args.run is None or args.before is None):
        parser.error('--run and --before are required unless --cpu-only is used')
    torch.set_num_threads(2)
    result = {'microbatch': verify_microbatch()}
    if not args.cpu_only:
        result['resume'] = verify_run(args.run.resolve(), args.before.resolve(),
            (args.after or args.run / 'latest.pt').resolve(), args.before_step, args.after_step,
            args.target_steps, args.world_size)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
