"""Train an unconditional EDM prior on the local rat brain images."""
import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import audit, load_images, sha256
from model import EDMPrior, sample_prior
from project_paths import RUNS


def save_checkpoint(path, state):
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    os.replace(temp, path)


def save_grid(x, path, columns=4):
    a = ((x.detach().float().cpu().numpy()[:, 0] + 1) / 2).clip(0, 1)
    rows = math.ceil(len(a) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns*2, rows*2), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, im in zip(axes.flat, a):
        ax.imshow(im, cmap='gray', vmin=0, vmax=1)
    fig.tight_layout(pad=.1)
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def validate(net, clean):
    gen = torch.Generator(device=clean.device).manual_seed(20260908)
    sigma = (torch.randn(len(clean), 1, 1, 1, device=clean.device, generator=gen)*1.2 - 1.2).exp()
    noise = torch.randn(clean.shape, device=clean.device, generator=gen)
    weight = (sigma.square() + net.sigma_data**2) / (sigma * net.sigma_data).square()
    total = 0.
    for i in range(0, len(clean), 32):
        sl = slice(i, i+32)
        pred = net(clean[sl] + sigma[sl]*noise[sl], sigma[sl])
        total += float(((pred-clean[sl]).square()*weight[sl]).flatten(1).mean(1).sum())
    return total / len(clean)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=RUNS/'core/edm_rat96')
    p.add_argument('--steps', type=int, default=12000)
    p.add_argument('--batch', type=int, default=48)
    p.add_argument('--base-ch', type=int, default=32)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--resume', type=Path)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if min(args.steps, args.batch, args.save_every) < 1:
        p.error('steps, batch and save-every must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'latest.pt').exists() and not args.resume:
        raise FileExistsError('Existing run; use --resume or a different --out')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    # Keep the numerical physics checks elsewhere in full precision.
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    names = audit(args.out/'data_manifest.json')
    train = load_images(names['train']).to(device)
    val = load_images(names['val']).to(device)
    net = EDMPrior(base_ch=args.base_ch).to(device)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.)
    step0, best = 0, float('inf')
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state['model_config'] != net.config:
            raise ValueError('Resume model architecture mismatch')
        if state['data_manifest_sha256'] != sha256(args.out/'data_manifest.json'):
            raise ValueError('Resume data manifest mismatch')
        net.load_state_dict(state['model']); ema.load_state_dict(state['ema'])
        opt.load_state_dict(state['optimizer'])
        step0, best = state['step'], state['best_val']
        torch.set_rng_state(state['rng_cpu'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state(state['rng_cuda'].cpu(), device)
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config.update(model=net.config, parameters=sum(p.numel() for p in net.parameters()),
                  torch=torch.__version__, device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
                  training_objective='EDM weighted denoising MSE; log(sigma) ~ N(-1.2, 1.2^2)',
                  source_sha256={f: sha256(Path(__file__).parent/f) for f in ['model.py','tiny_unet.py','train.py','data.py']})
    (args.out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    print(json.dumps(dict(event='start', **config)), flush=True)
    if not args.resume:
        baseline = validate(ema, val)
        print(json.dumps(dict(event='untrained_validation', val_loss=baseline)), flush=True)
        (args.out/'baseline.json').write_text(json.dumps(dict(val_loss=baseline))+'\n')
        save_grid(train[:16], args.out/'training_examples.png')
    start = time.monotonic()
    losses = []
    for step in range(step0+1, args.steps+1):
        net.train()
        ids = torch.randint(len(train), (args.batch,), device=device)
        x = train[ids]
        sigma = (torch.randn(args.batch,1,1,1,device=device)*1.2 - 1.2).exp()
        noisy = x + sigma * torch.randn_like(x)
        pred = net(noisy, sigma)
        weight = (sigma.square()+net.sigma_data**2)/(sigma*net.sigma_data).square()
        loss = (weight * (pred-x).square()).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite training loss at {step}')
        lr = args.lr * min(step/200., 1.) * (.2 + .8*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in opt.param_groups:
            group['lr'] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        decay = min(.999, (1+step)/(10+step))
        with torch.no_grad():
            for ep, np_ in zip(ema.parameters(), net.parameters()):
                ep.lerp_(np_, 1-decay)
        losses.append(float(loss.detach()))
        if step % 50 == 0 or step == args.steps:
            row = dict(step=step, loss=float(np.mean(losses)), lr=lr, grad_norm=float(norm),
                       elapsed_sec=time.monotonic()-start, steps_per_sec=(step-step0)/(time.monotonic()-start))
            print(json.dumps(row), flush=True)
            with (args.out/'train_metrics.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            losses.clear()
        if step % args.save_every == 0 or step == args.steps:
            vl = validate(ema, val)
            is_best = vl < best
            best = min(best, vl)
            state = dict(step=step, model=net.state_dict(), ema=ema.state_dict(), model_config=net.config,
                         optimizer=opt.state_dict(), val_loss=vl, best_val=best,
                         rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
                         data_manifest_sha256=sha256(args.out/'data_manifest.json'))
            save_checkpoint(args.out/'latest.pt', state)
            if is_best:
                save_checkpoint(args.out/'best.pt', state)
            row = dict(step=step, val_loss=vl, best_val=best)
            with (args.out/'val_metrics.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            print(json.dumps(dict(event='checkpoint', **row)), flush=True)
            if step % 2000 == 0 or step == args.steps:
                save_grid(sample_prior(ema, 8), args.out/f'samples_{step:06d}.png')
    (args.out/'completed.json').write_text(json.dumps(dict(step=args.steps, best_val=best, elapsed_sec=time.monotonic()-start))+'\n')
    print('TRAINING_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
