"""Generate reproducible unconditional samples from the three trained SPEN priors.

No images, measurements, encoders, or inverse solvers are used to initialize
sampling. The latent prior only uses the frozen VAE decoder after sampling.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
PROJECT = SCRIPTS.parent
sys.path[:0] = [str(SCRIPTS / name) for name in ('dit96', 'prior96', 'latent48', 'core')]
from pixel_model import load_pixel_dit
from model_v2 import load_strong_prior
from dit import LatentEDM, sigma_schedule

MODELS = {
    'unet96': dict(checkpoint='retrain_0911_260916/mouse_mixed/model_ema.pt', step=30000,
                   label='U-Net', detail='96 x 96 | 14.43M',
                   training='18,037 mixed images; 5k rat pretraining + 30k mixed steps'),
    'dit96': dict(checkpoint='rodent96_dit20m_260917/training/model_ema.pt', step=60000,
                  label='Pixel DiT', detail='96 x 96 | 20.73M',
                  training='28,160 expanded images; 60k steps'),
    'vae_dit192': dict(checkpoint='rodent192_latent_dit_260915/train/model_ema.pt', step=60000,
                       label='VAE + DiT', detail='192 x 192 | DiT 129.53M',
                       training='10,633 images; 60k steps; all_data_no_holdout'),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


@torch.no_grad()
def sample(net, seeds, steps):
    """Common EDM Heun sampler; independent recorded RNG seed for every image."""
    device = next(net.parameters()).device
    shape = (net.img_channels, net.img_resolution, net.img_resolution)
    noise = torch.stack([torch.randn(shape, dtype=torch.float32, device=device,
                         generator=torch.Generator(device=device).manual_seed(seed)) for seed in seeds])
    sigmas = sigma_schedule(steps, sigma_max=80., sigma_min=.002, rho=7., device=device)
    x = noise * sigmas[0]
    for index, (sigma, following) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        derivative = (x - net(x, sigma)) / sigma
        proposal = x + (following - sigma) * derivative
        if index < steps - 1:
            next_derivative = (proposal - net(proposal, following)) / following
            proposal = x + (following - sigma) * (derivative + next_derivative) * .5
        x = proposal
    if not torch.isfinite(x).all():
        raise FloatingPointError('Nonfinite generated sample')
    return x, noise


def unit(raw):
    return np.clip((raw[:, 0] + 1.) / 2., 0., 1.)


def common96(images):
    if images.shape[-2:] == (96, 96):
        return images
    if images.shape[-2:] != (192, 192):
        raise ValueError('Unexpected image shape')
    return images.reshape(-1, 96, 2, 96, 2).mean(axis=(2, 4))


def render(out, args, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'font.size': 10})
    arrays = {key: np.load(out / key / 'samples.npz')['images_unit'] for key in MODELS}
    for start in range(0, args.count, 8):
        indices = list(range(start, min(start + 8, args.count)))
        for matched in (False, True):
            fig, axes = plt.subplots(3, len(indices), figsize=(2 * len(indices) + 2.2, 7.3), squeeze=False)
            fig.subplots_adjust(left=.13, right=.99, top=.84, bottom=.09, wspace=.025, hspace=.06)
            for row, (key, model) in enumerate(MODELS.items()):
                images = common96(arrays[key]) if matched else arrays[key]
                for col, index in enumerate(indices):
                    ax = axes[row, col]
                    ax.imshow(images[index], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                    ax.set_xticks([]); ax.set_yticks([])
                    for spine in ax.spines.values(): spine.set_visible(False)
                    if row == 0: ax.set_title(f'#{index + 1:02d}', fontsize=11, pad=7)
                    if col == 0:
                        detail = model['detail'] if not matched else (
                            'native 96 x 96' if key != 'vae_dit192' else '192 -> 96 (area mean)')
                        ax.set_ylabel(model['label'] + '\n' + detail, rotation=0, ha='right', va='center', labelpad=12)
            fig.suptitle('Unconditional generation from pure Gaussian noise', fontsize=18, y=.965, weight='bold')
            fig.text(.56, .902, f'Final EMA | {args.steps}-step EDM Heun | samples {start + 1}-{indices[-1] + 1} in seed order',
                     ha='center', fontsize=11)
            fig.text(.56, .045, 'Fixed grayscale [0, 1]; no per-image rescaling. Same noise for U-Net / pixel DiT; latent noise has a different shape.',
                     ha='center', fontsize=9)
            suffix = 'common96' if matched else 'native'
            stem = out / f'comparison_{suffix}_{start // 8 + 1:02d}'
            fig.savefig(stem.with_suffix('.png'), dpi=180, facecolor='white')
            fig.savefig(stem.with_suffix('.pdf'), facecolor='white')
            plt.close(fig)
    for key, model in MODELS.items():
        rows = (args.count + 7) // 8
        fig, axes = plt.subplots(rows, 8, figsize=(16, rows * 2.1 + 1), squeeze=False)
        fig.subplots_adjust(left=.02, right=.98, top=.89, bottom=.04, wspace=.035, hspace=.16)
        for index, ax in enumerate(axes.flat):
            if index < args.count:
                ax.imshow(arrays[key][index], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                ax.set_title(f'#{index + 1:02d} | seed {args.seed + index}', fontsize=8, pad=3)
            ax.axis('off')
        fig.suptitle(f"{model['label']} | all {args.count} unselected samples | {model['detail']}", fontsize=16)
        fig.text(.5, .025, f'Pure Gaussian noise; {args.steps}-step Heun; fixed grayscale [0, 1]; final EMA', ha='center', fontsize=10)
        fig.savefig(out / key / 'all_samples.png', dpi=180, facecolor='white')
        fig.savefig(out / key / 'all_samples.pdf', facecolor='white')
        plt.close(fig)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20260917)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if args.count < 1 or args.batch_size < 1 or args.steps < 2:
        parser.error('Require count/batch-size >= 1 and steps >= 2')
    args.out = args.out.resolve()
    if args.render_only:
        saved = json.loads((args.out / 'protocol.json').read_text())
        for key in ('count', 'steps', 'seed'): setattr(args, key, saved[key])
        render(args.out, args, json.loads((args.out / 'summary.json').read_text()))
        return
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Use a fresh output directory: {args.out}')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    source_paths = [Path(__file__), SCRIPTS/'core/model.py', SCRIPTS/'core/tiny_unet.py',
                    SCRIPTS/'prior96/model_v2.py', SCRIPTS/'prior96/tiny_unet_v2.py',
                    SCRIPTS/'dit96/pixel_model.py', SCRIPTS/'latent48/dit.py', SCRIPTS/'latent48/vae_codec.py']
    protocol = dict(count=args.count, steps=args.steps, seed=args.seed,
                    seeds=list(range(args.seed, args.seed + args.count)), batch_size=args.batch_size,
                    sampler='EDM Heun', denoiser_evaluations_per_image=2 * args.steps - 1,
                    sigma_max=80., sigma_min=.002, rho=7., conditioning=None,
                    initialization='independent standard normal per sample multiplied by sigma_max',
                    display='clip((raw + 1) / 2, 0, 1); fixed gray [0,1]; nearest display',
                    selection='all generated samples in seed order; no filtering or rerolling',
                    common96='clip at native resolution, then exact 2x2 area mean for VAE output',
                    matched_noise='U-Net and pixel DiT have byte-identical initial noise; latent dimensions differ',
                    caveat='Data, resolution, model size and training budgets differ; not an architecture ablation',
                    runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, python=sys.executable,
                                 device=torch.cuda.get_device_name(args.device)),
                    source_sha256={str(p): sha256(p) for p in source_paths},
                    created_utc=datetime.now(timezone.utc).isoformat())
    save_json(args.out / 'protocol.json', protocol)
    results = {}
    for key, model in MODELS.items():
        path = PROJECT / 'runs' / model['checkpoint']
        folder = args.out / key; folder.mkdir()
        checkpoint_hash = sha256(path)
        if key == 'unet96':
            net, state = load_strong_prior(path, args.device)
        elif key == 'dit96':
            net, state = load_pixel_dit(path, args.device)
        else:
            from vae_codec import FrozenVAE
            state = torch.load(path, map_location='cpu', weights_only=False)
            net = LatentEDM(**state['model_config']).to(args.device)
            net.load_state_dict(state['ema'], strict=True)
            net.eval().requires_grad_(False)
            vae_path = Path(state['vae_path'])
            for relative, expected in state['vae_sha256'].items():
                if sha256(vae_path / relative) != expected:
                    raise ValueError(f'VAE hash mismatch: {relative}')
            codec = FrozenVAE(vae_path, device=args.device, autocast_dtype=torch.bfloat16, decode_batch_size=4)
            mean, std = [torch.tensor(state['latent_normalization'][field], device=args.device,
                          dtype=torch.float32).reshape(1, -1, 1, 1) for field in ('mean', 'std')]
            if codec.latent_channels != net.img_channels or not bool((std > 0).all()):
                raise ValueError('Invalid latent codec or normalization')
        if state['step'] != model['step']:
            raise ValueError(f'Expected final checkpoint {model["step"]}, got {state["step"]}')
        info = dict(model, checkpoint=str(path), checkpoint_sha256=checkpoint_hash,
                    parameters=sum(p.numel() for p in net.parameters()), model_config=state['model_config'],
                    img_channels=net.img_channels, diffusion_resolution=net.img_resolution)
        if key == 'vae_dit192':
            info.update(vae_path=str(vae_path), vae_sha256=state['vae_sha256'],
                        latent_normalization={k: state['latent_normalization'][k] for k in ('mean', 'std')},
                        frozen_vae_parameters=sum(p.numel() for p in codec.parameters()))
        del state
        image_batches, diffusion_batches, noise_batches = [], [], []
        torch.cuda.synchronize(); model_started = time.monotonic()
        for start in range(0, args.count, args.batch_size):
            seeds = protocol['seeds'][start:start + args.batch_size]
            generated, noise = sample(net, seeds, args.steps)
            raw = codec.decode(generated * std + mean, clamp=False) if key == 'vae_dit192' else generated
            if not torch.isfinite(raw).all(): raise FloatingPointError('Nonfinite decoded output')
            image_batches.append(raw.cpu().numpy())
            diffusion_batches.append(generated.cpu().numpy())
            noise_batches.append(noise.cpu().numpy())
            print(json.dumps(dict(model=key, generated=start + len(seeds), total=args.count)), flush=True)
        torch.cuda.synchronize()
        raw = np.concatenate(image_batches); generated = np.concatenate(diffusion_batches); noise = np.concatenate(noise_batches)
        np.savez_compressed(folder / 'samples.npz', raw_images=raw, images_unit=unit(raw),
                            diffusion_output=generated, initial_standard_normal=noise,
                            seeds=np.array(protocol['seeds'], dtype=np.int64))
        info.update(sampling_and_decoding_seconds=time.monotonic() - model_started,
                    output_shape=list(raw.shape), finite=bool(np.isfinite(raw).all()),
                    raw_min=float(raw.min()), raw_max=float(raw.max()),
                    fraction_outside_display_range=float(np.mean((raw < -1) | (raw > 1))),
                    initial_noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest())
        save_json(folder / 'model.json', info)
        results[key] = info
        save_json(args.out / 'summary.json', results)
        del net
        if key == 'vae_dit192': del codec
        torch.cuda.empty_cache()
    if results['unet96']['initial_noise_sha256'] != results['dit96']['initial_noise_sha256']:
        raise AssertionError('96x96 priors must have identical initial noise')
    render(args.out, args, results)
    save_json(args.out / 'completed.json', dict(status='complete', models=list(MODELS),
              samples_per_model=args.count, same_pixel_noise_verified=True,
              elapsed_seconds=time.monotonic() - started))
    print(json.dumps(dict(status='complete', output=str(args.out))), flush=True)


if __name__ == '__main__':
    main()
