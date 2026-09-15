"""Online image augmentation -> frozen f4 VAE -> unconditional latent DiT EDM.

All source images train the prior. No holdout or validation-based selection.
Checkpoints bind the data, codec, latent normalization, and per-rank RNG state.
"""
import argparse
import copy
from datetime import timedelta
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'prior192'))
from train_png_ddp import (load_data, tensor_batch, augment_training,
    augmentation_config, sha256, microbatch_loss_backward, save_grid,
    atomic_json, atomic_torch, acquire_run_lock, log_json,
    sampling_source_probabilities, augmentation_preview)
from dit import LatentEDM, sample_latent
from vae_codec import FrozenVAE


def encode_chunks(codec, images, batch):
    with torch.no_grad():
        return torch.cat([codec.encode(x) for x in images.split(batch)])


def normalization_tensors(normalization, device):
    mean, std = tuple(torch.tensor(normalization[k], dtype=torch.float32, device=device)
                      .reshape(1, -1, 1, 1) for k in ('mean', 'std'))
    if mean.shape != std.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or bool((std <= 0).any()):
        raise ValueError('Invalid latent normalization moments')
    return mean, std


@torch.no_grad()
def calibrate(codec, array, weights, args, device):
    """Estimate channel moments from the TRAIN sampling and augmentation distribution."""
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(args.seed + 919)
        ids = torch.multinomial(weights, args.calibration_images, replacement=True).numpy()
        total = total2 = None
        count = 0
        for start in range(0, len(ids), args.encode_batch):
            x, _ = augment_training(tensor_batch(array, ids[start:start + args.encode_batch], device), args)
            z = codec.encode(x * 2 - 1).double()
            if total is None:
                total = torch.zeros(z.shape[1], dtype=torch.float64, device=device)
                total2 = torch.zeros_like(total)
            total += z.sum((0, 2, 3))
            total2 += z.square().sum((0, 2, 3))
            count += z.shape[0] * z.shape[2] * z.shape[3]
        mean = total / count
        std = (total2 / count - mean.square()).clamp_min(1e-12).sqrt()
        if not torch.isfinite(std).all() or bool((std < 1e-6).any()):
            raise ValueError('Degenerate latent calibration')
        return dict(mean=mean.cpu().tolist(), std=std.cpu().tolist(),
                    images=len(ids), sampled_indices=ids.tolist(), seed=args.seed + 919,
                    distribution='subject-equal training images with unchanged ONLINE image augmentation',
                    posterior='mode', affine='standardized = (codec_latent - mean) / std')


def source_paths():
    return [HERE / name for name in ('train_latent_ddp.py', 'dit.py', 'vae_codec.py')] + [
        HERE.parent / 'prior192' / name for name in ('train_png_ddp.py', 'train_hr.py')]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--vae', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--normalization', type=Path)
    p.add_argument('--steps', type=int, default=60000)
    p.add_argument('--local-batch', type=int, default=64)
    p.add_argument('--micro-batch', type=int)
    p.add_argument('--encode-batch', type=int, default=8)
    p.add_argument('--hidden-size', type=int, default=768)
    p.add_argument('--depth', type=int, default=12)
    p.add_argument('--heads', type=int, default=12)
    p.add_argument('--patch-size', type=int, default=2)
    p.add_argument('--activation-checkpoint', action='store_true')
    p.add_argument('--p-mean', type=float, default=-.5,
                   help='Log sigma mean; shifted from pixel -1.2 for unit-variance latents')
    p.add_argument('--p-std', type=float, default=1.2)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--seed', type=int, default=20260915)
    p.add_argument('--calibration-images', type=int, default=1024)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--sample-every', type=int, default=5000)
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--backend', default='nccl', choices=['nccl', 'gloo'])
    p.add_argument('--stop-after', type=int)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--benchmark', action='store_true', help='Measure steps, with no checkpoint or samples')
    p.add_argument('--contrast-range', type=float, nargs=2, default=[.9, 1.1])
    p.add_argument('--contrast-gate', type=float, default=.05)
    args = p.parse_args()
    args.micro_batch = args.micro_batch or args.local_batch
    if min(args.steps, args.local_batch, args.micro_batch, args.encode_batch,
           args.calibration_images, args.save_every, args.sample_every, args.log_every, args.warmup) < 1:
        p.error('Counts must be positive')
    if args.micro_batch > args.local_batch or (args.stop_after is not None and args.stop_after < 1):
        p.error('Invalid micro batch or stop-after')
    if args.benchmark and args.resume:
        p.error('Benchmarks cannot resume')
    for name in ('data', 'vae', 'out', 'normalization'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    return args


def main():
    args = parse_args()
    rank, local, world = (int(os.environ.get(k, d)) for k, d in
                          [('RANK', '0'), ('LOCAL_RANK', '0'), ('WORLD_SIZE', '1')])
    torch.set_num_threads(2)
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if world > 1:
        init_options = dict(device_id=device) if args.backend == 'nccl' else {}
        dist.init_process_group(args.backend, timeout=timedelta(minutes=30), **init_options)
    def barrier():
        if world > 1:
            dist.barrier()
    args.out.mkdir(parents=True, exist_ok=True)
    lock = acquire_run_lock(args.out) if rank == 0 else None
    if (args.out / 'config.json').exists() and not args.resume:
        raise FileExistsError('Existing training directory; use --resume')
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    manifest, arrays, weights, val_ids = load_data(args.data)
    if manifest['dataset'] != 'rodent_native192_png_all' or val_ids:
        raise ValueError('This experiment requires all-image training without holdout')
    codec = FrozenVAE(args.vae, device=device, autocast_dtype=torch.bfloat16)
    if codec.downsample_factor != 4:
        raise ValueError('Codec must map 192 directly to 48')
    vae_hashes = {str(p.relative_to(args.vae)): sha256(p) for p in sorted(args.vae.rglob('*'))
                  if p.is_file() and (p.name == 'config.json' or p.suffix in ('.safetensors', '.bin', '.ckpt'))
                  and '.cache' not in p.parts}
    if not vae_hashes:
        raise ValueError('No local VAE configuration or weight files found')
    norm_identity = dict(manifest_sha256=sha256(args.data / 'manifest.json'),
        vae_sha256=vae_hashes, vae_autocast='bfloat16',
        augmentation=augmentation_config(args.contrast_range, args.contrast_gate))
    norm_path = args.out / 'latent_normalization.json'
    if rank == 0 and not args.resume:
        norm = (json.loads(args.normalization.read_text()) if args.normalization else
                calibrate(codec, arrays['train'], weights, args, device))
        if args.normalization and any(norm.get(k) != v for k, v in norm_identity.items()):
            raise ValueError('Supplied normalization differs in data, VAE, precision, or augmentation')
        norm.update(norm_identity)
        atomic_json(norm_path, norm)
    barrier()
    norm = json.loads(norm_path.read_text())
    if any(norm.get(k) != v for k, v in norm_identity.items()):
        raise ValueError('Normalization differs in data, VAE, precision, or augmentation')
    mean, std = normalization_tensors(norm, device)
    if mean.shape[1] != codec.latent_channels:
        raise ValueError('Normalization channel count differs from VAE')
    model_config = dict(input_size=48, patch_size=args.patch_size,
        in_channels=codec.latent_channels, hidden_size=args.hidden_size, depth=args.depth,
        num_heads=args.heads, sigma_data=1., p_mean=args.p_mean, p_std=args.p_std,
        activation_checkpoint=args.activation_checkpoint)
    net = LatentEDM(**model_config).to(device)
    # Retain an EMA on both ranks so batch probing matches the heavier rank.
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0., fused=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(model_config=net.config, parameters=sum(p.numel() for p in net.parameters()),
        world_size=world, global_batch=world * args.local_batch, image_shape=[1, 192, 192],
        latent_shape=[codec.latent_channels, 48, 48], manifest_sha256=sha256(args.data / 'manifest.json'),
        vae_sha256=vae_hashes, latent_normalization=norm,
        source_sha256={str(p): sha256(p) for p in source_paths()},
        training_mode='all_data_no_holdout', selection='final target-step EMA; no validation selection',
        image_counts=manifest['image_counts'], augmentation=augmentation_config(args.contrast_range, args.contrast_gate),
        sampling_source_probability=sampling_source_probabilities(manifest),
        precision='FP32 parameters/EDM; BF16 autocast DiT and frozen VAE; codec output FP32',
        online_augmentation=True, vae_posterior='mode', vae_autocast='bfloat16', torch_version=torch.__version__,
        package_versions={name: version(name) for name in ('diffusers', 'transformers', 'safetensors', 'numpy')})
    step0, elapsed_before = 0, 0.
    sample_counts = np.zeros(len(weights), dtype=np.int64)
    saved = None
    if args.resume:
        previous = json.loads((args.out / 'config.json').read_text())
        for key in ('data', 'vae', 'steps', 'local_batch', 'micro_batch', 'encode_batch', 'world_size',
                    'lr', 'warmup', 'seed', 'model_config', 'manifest_sha256', 'vae_sha256',
                    'latent_normalization', 'augmentation', 'source_sha256'):
            if previous[key] != config[key]:
                raise ValueError(f'Resume changes {key}')
        saved = torch.load(args.out / 'latest.pt', map_location='cpu', weights_only=False)
        if saved['manifest_sha256'] != config['manifest_sha256'] or saved['vae_sha256'] != vae_hashes:
            raise ValueError('Checkpoint provenance differs')
        for key in ('model_config', 'latent_normalization', 'augmentation', 'vae_autocast'):
            if saved[key] != config[key]:
                raise ValueError(f'Checkpoint differs from the run configuration: {key}')
        if (not 0 <= saved['step'] <= args.steps or saved['global_batch'] != world * args.local_batch
                or len(saved['rng_states']) != world or len(saved['sampling_counts_by_rank']) != world):
            raise ValueError('Checkpoint step, batch, or rank state differs')
        net.load_state_dict(saved['model'])
        ema.load_state_dict(saved['ema'])
        optimizer.load_state_dict(saved['optimizer'])
        step0, elapsed_before = saved['step'], saved['elapsed_sec']
        sample_counts = saved['sampling_counts_by_rank'][rank].copy()
        if (sample_counts.shape != (len(weights),) or bool((sample_counts < 0).any())
                or int(sample_counts.sum()) != step0 * args.local_batch):
            raise ValueError('Checkpoint sampling counts are inconsistent')
    model = DDP(net, device_ids=[local], broadcast_buffers=False,
                gradient_as_bucket_view=True) if world > 1 else net
    if args.resume:
        torch.set_rng_state(saved['rng_states'][rank]['cpu'])
        torch.cuda.set_rng_state(saved['rng_states'][rank]['cuda'], device)
        del saved
    if rank == 0:
        if not args.resume:
            atomic_json(args.out / 'config.json', config)
            snap = args.out / 'source_snapshot'
            snap.mkdir(exist_ok=True)
            for path in source_paths():
                shutil.copy2(path, snap / path.name)
            if not args.benchmark:
                augmentation_preview(arrays['train'], weights, device, args, args.out)
        atomic_json(args.out / 'process.json', dict(pid=os.getpid(), argv=sys.argv,
                    physical_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'), started_at=time.time()))
        print(json.dumps(dict(event='start', step=step0, target_steps=args.steps,
                              global_batch=config['global_batch'], parameters=config['parameters'])), flush=True)
    barrier()
    torch.cuda.reset_peak_memory_stats(device)
    started = last_time = time.monotonic()
    last_report = step0
    last_step = min(args.steps, step0 + args.stop_after) if args.stop_after else args.steps
    losses, clips = [], []
    for step in range(step0 + 1, last_step + 1):
        model.train()
        ids = torch.multinomial(weights, args.local_batch, replacement=True).numpy()
        np.add.at(sample_counts, ids, 1)
        images, stats = augment_training(tensor_batch(arrays['train'], ids, device), args)
        x = (encode_chunks(codec, images * 2 - 1, args.encode_batch).float() - mean) / std
        if x.shape[1:] != (codec.latent_channels, 48, 48) or not torch.isfinite(x).all():
            raise ValueError('Invalid encoded latent batch')
        sigma = (torch.randn(len(x), 1, 1, 1, device=device) * net.p_std + net.p_mean).exp()
        weight = (sigma.square() + net.sigma_data ** 2) / (sigma * net.sigma_data).square()
        noisy = x + sigma * torch.randn_like(x)
        lr = args.lr * min(step / args.warmup, 1.) * (.15 + .85 * .5 * (1 + math.cos(math.pi * step / args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = microbatch_loss_backward(model, noisy, x, sigma, weight, args.micro_batch, world > 1)
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if rank == 0:
            with torch.no_grad():
                decay = min(.9995, (1 + step) / (10 + step))
                torch._foreach_lerp_(list(ema.parameters()), list(net.parameters()), 1 - decay)
        losses.append(loss.detach())
        clips.append(stats['clip_fraction'].detach())
        if step % args.log_every == 0 or step in (step0 + 1, last_step):
            values = torch.stack([torch.stack(losses).mean(), torch.stack(clips).mean()])
            if world > 1:
                dist.all_reduce(values)
                values /= world
            torch.cuda.synchronize(device)
            now = time.monotonic()
            rate = (step - last_report) / (now - last_time)
            if rank == 0:
                row = dict(step=step, loss=float(values[0]), contrast_clip_fraction=float(values[1]),
                    lr=lr, grad_norm=float(grad_norm), elapsed_sec=elapsed_before + now - started,
                    recent_steps_per_sec=rate, images_per_sec=rate * args.local_batch * world,
                    eta_hours=(args.steps - step) / rate / 3600,
                    images_seen=step * args.local_batch * world,
                    max_memory_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                    max_memory_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)
                log_json(args.out / 'train_metrics.jsonl', row)
                atomic_json(args.out / 'status.json', dict(status='training', target_steps=args.steps, **row))
            losses.clear(); clips.clear()
            last_report, last_time = step, now
        if not args.benchmark and (step % args.save_every == 0 or step == last_step):
            del images, x, sigma, weight, noisy, loss
            optimizer.zero_grad(set_to_none=True)
            barrier()
            rng = dict(cpu=torch.get_rng_state().cpu(), cuda=torch.cuda.get_rng_state(device).cpu(),
                       counts=sample_counts.copy())
            states = [None] * world
            if world > 1:
                dist.all_gather_object(states, rng)
            else:
                states = [rng]
            counts = [s.pop('counts') for s in states]
            if rank == 0:
                common = dict(step=step, model_config=net.config, ema=ema.state_dict(),
                    manifest_sha256=config['manifest_sha256'], vae_sha256=vae_hashes, vae_path=str(args.vae),
                    latent_normalization=norm, img_resolution=192, latent_resolution=48,
                    augmentation=config['augmentation'], vae_autocast='bfloat16',
                    training_mode=config['training_mode'], selection=config['selection'])
                state = dict(**common, model=net.state_dict(), optimizer=optimizer.state_dict(),
                    rng_states=states, sampling_counts_by_rank=counts, global_batch=args.local_batch * world,
                    elapsed_sec=elapsed_before + time.monotonic() - started)
                atomic_torch(args.out / 'latest.pt', state)
                atomic_torch(args.out / 'model_ema.pt', common)
                if step % 5000 == 0 or step == args.steps:
                    atomic_torch(args.out / f'ema_{step:06d}.pt', common)
                total_counts = np.stack(counts).sum(0)
                atomic_json(args.out / 'sampling_coverage.json', dict(step=step,
                    images=len(total_counts), sampled_images=int((total_counts > 0).sum()),
                    all_images_sampled=bool((total_counts > 0).all()), total_samples=int(total_counts.sum()),
                    minimum_samples=int(total_counts.min()), maximum_samples=int(total_counts.max())))
                if step % args.sample_every == 0 or step == args.steps:
                    with torch.random.fork_rng(devices=[local]), torch.no_grad():
                        z = sample_latent(ema, count=8, steps=64, seed=args.seed)
                        decoded = torch.cat([codec.decode(v) for v in (z * std + mean).split(2)])
                        if not torch.isfinite(decoded).all():
                            raise FloatingPointError('Nonfinite decoded sample')
                        save_grid(decoded, args.out / f'samples_{step:06d}.png')
                        np.save(args.out / f'samples_{step:06d}.npy', decoded.cpu().numpy())
                del state, common
            barrier()
    if rank == 0:
        completed = last_step == args.steps and not args.benchmark
        # A restart may begin at the final saved update, after interruption
        # between checkpoint and sample writes. Complete that pending artifact.
        if completed and not (args.out / f'samples_{args.steps:06d}.png').exists():
            with torch.random.fork_rng(devices=[local]), torch.no_grad():
                z = sample_latent(ema, count=8, steps=64, seed=args.seed)
                decoded = torch.cat([codec.decode(v) for v in (z * std + mean).split(2)])
                if not torch.isfinite(decoded).all():
                    raise FloatingPointError('Nonfinite final decoded sample')
                save_grid(decoded, args.out / f'samples_{args.steps:06d}.png')
                np.save(args.out / f'samples_{args.steps:06d}.npy', decoded.cpu().numpy())
        status = dict(status='completed' if completed else ('benchmark_completed' if args.benchmark else 'paused'),
                      step=last_step, target_steps=args.steps, elapsed_sec=elapsed_before + time.monotonic() - started)
        atomic_json(args.out / 'status.json', status)
        if completed:
            atomic_json(args.out / 'completed.json', status)
        print(json.dumps(status), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
