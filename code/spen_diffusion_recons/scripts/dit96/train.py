"""Single-GPU, resumable 60,000-step pixel DiT training on the expanded PNGs."""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from PIL import Image
from pixel_model import PixelDiT, sample_prior
from png_data import PNGData


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def atomic_torch(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def save_grid(value, path):
    images = ((value[:, 0].detach().float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    canvas = np.zeros((math.ceil(len(images)/4)*96, 4*96), dtype=np.uint8)
    for i, image in enumerate(images):
        canvas[i//4*96:(i//4+1)*96, i%4*96:(i%4+1)*96] = image
    Image.fromarray(canvas).save(path)


@torch.no_grad()
def validate(net, clean):
    generator = torch.Generator(device=clean.device).manual_seed(20260908)
    sigma = (torch.randn(len(clean), 1, 1, 1, device=clean.device, generator=generator)*1.2-1.2).exp()
    noise = torch.randn(clean.shape, device=clean.device, generator=generator)
    losses = [net.loss(clean[i:i+24], sigma[i:i+24], noise[i:i+24]) * len(clean[i:i+24])
              for i in range(0, len(clean), 24)]
    return float(torch.stack(losses).sum() / len(clean))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=60000)
    parser.add_argument('--microbatch', type=int, default=24)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--sample-every', type=int, default=5000)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--sampling', choices=['uniform', 'balanced'], default='uniform')
    parser.add_argument('--legacy-manifest', type=Path,
        default=Path(__file__).resolve().parents[3]/'data/prior96_0911_260916/mouse_mixed/manifest.json')
    parser.add_argument('--stop-after', type=int, help='Smoke test only; preserve full LR horizon')
    args = parser.parse_args()
    if min(args.steps, args.microbatch, args.accumulation, args.warmup, args.save_every,
           args.sample_every, args.log_every) < 1 or args.lr <= 0:
        parser.error('Training settings must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / '.training.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.out/'config.json').exists() and not args.resume:
        raise FileExistsError('Existing run; use --resume')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly the requested GPU via CUDA_VISIBLE_DEVICES')
    device = torch.device('cuda:0')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    net = PixelDiT().to(device)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(.9, .999), weight_decay=0., fused=True)
    print(json.dumps(dict(event='loading_pngs', data=str(args.data), parameters=sum(p.numel() for p in net.parameters()))), flush=True)
    data = PNGData(args.data, device, sampling=args.sampling, legacy_manifest=args.legacy_manifest)
    val = data.validation()
    protocol = dict(steps=args.steps, microbatch=args.microbatch, accumulation=args.accumulation,
                    effective_batch=args.microbatch*args.accumulation, lr=args.lr, warmup=args.warmup,
                    seed=args.seed, sigma_distribution=[-1.2, 1.2], weight_decay=0.,
                    ema_max=.9995, lr_floor=.1, sampling=data.audit['sampling'])
    source_files = [Path(__file__), Path(__file__).with_name('pixel_model.py'), Path(__file__).with_name('png_data.py'),
                    Path(__file__).resolve().parents[1]/'latent48/dit.py',
                    Path(__file__).resolve().parents[1]/'prior96/data_v2.py',
                    Path(__file__).with_name('balanced_sampling.py')]
    source_hashes = {str(p.relative_to(Path(__file__).resolve().parents[1])):
                     hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    config = dict(architecture='pixel96_dit', model=net.config, parameters=sum(p.numel() for p in net.parameters()),
                  data=str(args.data.resolve()), data_fingerprint=data.fingerprint, protocol=protocol,
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, source_sha256=source_hashes, python=os.sys.executable)
    step0, best = 0, float('inf')
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        for key, expected in [('model_config', net.config), ('data_fingerprint', data.fingerprint), ('protocol', protocol),
                              ('source_sha256', source_hashes)]:
            if saved[key] != expected:
                raise ValueError(f'Resume mismatch: {key}')
        net.load_state_dict(saved['model']); ema.load_state_dict(saved['ema'])
        optimizer.load_state_dict(saved['optimizer'])
        step0, best = saved['step'], saved['best_val']
        random.setstate(saved['rng_python']); np.random.set_state(saved['rng_numpy'])
        torch.set_rng_state(saved['rng_cpu']); torch.cuda.set_rng_state(saved['rng_cuda'], device)
        del saved
    else:
        atomic_json(args.out/'config.json', config)
        atomic_json(args.out/'data_audit.json', data.audit)
        if data.weights is not None:
            atomic_json(args.out/'sampling_audit.json', data.audit['sampling'])
            np.save(args.out/'sampling_probabilities.npy', data.sampling_probabilities)
        for src in source_files:
            dst = args.out/'source_snapshot'/src.relative_to(Path(__file__).resolve().parents[1])
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
        atomic_json(args.out/'baseline.json', dict(untrained_val_loss=validate(ema, val)))
        save_grid(data.arrays['train'][np.linspace(0, len(data.arrays['train'])-1, 16, dtype=int)]*2-1,
                  args.out/'training_examples.png')
    if step0 > args.steps:
        raise ValueError('Checkpoint exceeds requested total steps')
    print(json.dumps(dict(event='start', initial_step=step0, pid=os.getpid(), **config)), flush=True)
    atomic_json(args.out/'status.json', dict(status='training', step=step0, total_steps=args.steps, pid=os.getpid()))
    start = time.monotonic(); losses = []
    final_step = min(args.steps, args.stop_after) if args.stop_after else args.steps
    for step in range(step0+1, final_step+1):
        lr = args.lr * min(step/args.warmup, 1.) * (.1 + .9*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups: group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device)
        for _ in range(args.accumulation):
            loss = net.loss(data.sample(args.microbatch)) / args.accumulation
            loss.backward(); total_loss += loss.detach()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        if not torch.isfinite(total_loss): raise FloatingPointError(f'Nonfinite loss at step {step}')
        optimizer.step()
        with torch.no_grad():
            decay = min(.9995, (1+step)/(10+step))
            torch._foreach_lerp_(list(ema.parameters()), list(net.parameters()), 1-decay)
        losses.append(float(total_loss))
        if step % args.log_every == 0 or step == final_step or step == 1:
            elapsed = time.monotonic()-start
            row = dict(step=step, total_steps=args.steps, loss=float(np.mean(losses)), lr=lr, grad_norm=float(norm),
                       elapsed_sec=elapsed, seconds_per_step=elapsed/(step-step0),
                       gpu_memory_peak_gb=torch.cuda.max_memory_allocated()/2**30)
            print(json.dumps(row), flush=True)
            with (args.out/'train_metrics.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
            atomic_json(args.out/'status.json', dict(status='training', pid=os.getpid(), **row))
            losses.clear()
        if step % args.save_every == 0 or step == final_step:
            vl = validate(ema, val)
            if not math.isfinite(vl): raise FloatingPointError('Nonfinite validation loss')
            improved = vl < best; best = min(best, vl)
            common = dict(architecture='pixel96_dit', step=step, model_config=net.config, ema=ema.state_dict(),
                          data_fingerprint=data.fingerprint, protocol=protocol, source_sha256=source_hashes,
                          val_loss=vl, best_val=best)
            state = dict(**common, model=net.state_dict(), optimizer=optimizer.state_dict(),
                         rng_python=random.getstate(), rng_numpy=np.random.get_state(),
                         rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(device))
            atomic_torch(args.out/'latest.pt', state)
            atomic_torch(args.out/'model_ema.pt', common)
            if improved: atomic_torch(args.out/'best_ema.pt', common)
            row = dict(event='checkpoint', step=step, val_loss=vl, best_val=best)
            print(json.dumps(row), flush=True)
            with (args.out/'val_metrics.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
        if step % args.sample_every == 0 or step == args.steps:
            save_grid(sample_prior(ema, count=8, steps=64, seed=123), args.out/f'samples_{step:06d}.png')
    state = dict(status='complete' if final_step == args.steps else 'smoke_paused',
                 step=final_step, total_steps=args.steps, best_val=best)
    atomic_json(args.out/'status.json', state)
    if final_step == args.steps: atomic_json(args.out/'completed.json', state)
    print(json.dumps(state), flush=True)


if __name__ == '__main__': main()
