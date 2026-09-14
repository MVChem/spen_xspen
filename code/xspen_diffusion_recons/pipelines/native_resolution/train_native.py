"""Independent, resumable native-grid EDM fine-tuning from the human128 prior."""
import argparse
import copy
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from native_model import NativePrior, PROJECT
from native_data import NativeData
from utils import sha256, save_checkpoint, save_grid, write_json, validate
from run_guard import acquire_run_lock


def rng_state():
    return dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                numpy=np.random.get_state(), python=random.getstate())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--seed', type=int, default=20260913)
    p.add_argument('--init-from', type=Path, default=PROJECT/'runs/human128/model_ema.pt')
    p.add_argument('--resume', type=Path)
    p.add_argument('--deterministic', action='store_true')
    args = p.parse_args()
    if min(args.steps, args.batch, args.save_every) <= 0:
        raise ValueError('Positive steps, batch, save interval required')
    lock = acquire_run_lock(args.out)
    if (args.out/'config.json').exists() and not args.resume:
        raise FileExistsError('Existing run; use --resume or a fresh output directory')
    torch.set_num_threads(2)
    if args.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = not args.deterministic
    torch.backends.cudnn.deterministic = args.deterministic
    torch.use_deterministic_algorithms(args.deterministic)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    data = NativeData(args.data)
    val, val_keys = data.validation()
    initial = torch.load(args.init_from, map_location='cpu', weights_only=False)
    base_config = {k: v for k, v in initial['model_config'].items() if k != 'image_shape'}
    net = NativePrior(data.image_shape, **base_config).cuda()
    net.load_state_dict(initial['ema'])
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.)
    manifest_hash = sha256(args.data/'manifest.json')
    init_hash = sha256(args.init_from)
    step0, best = 0, float('inf')
    config = dict(data=str(args.data.resolve()), out=str(args.out.resolve()), steps=args.steps,
                  batch=args.batch, lr=args.lr, seed=args.seed, save_every=args.save_every,
                  model_config=net.config, manifest_sha256=manifest_hash,
                  initialization=str(args.init_from.resolve()), initialization_sha256=init_hash,
                  image_shape=list(data.image_shape), validation_keys=val_keys,
                  deterministic=args.deterministic,
                  counts=data.manifest['counts'], subject_counts={s: len(v) for s, v in data.manifest['subjects'].items()},
                  parameters=sum(v.numel() for v in net.parameters()), visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  objective='Unconditional magnitude EDM on physical IXI grids; xSPEN degradation enters DiffPIR at reconstruction',
                  padding='Symmetric to next multiple of 8, known background -1; cropped before loss and DC',
                  start_unix=time.time(), torch=torch.__version__)
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        if ckpt.get('deterministic', False) != args.deterministic:
            raise ValueError('Resume deterministic mode mismatch')
        for key, expected in [('model_config', net.config), ('manifest_sha256', manifest_hash),
                              ('total_steps', args.steps), ('batch', args.batch), ('lr', args.lr), ('seed', args.seed)]:
            if ckpt[key] != expected:
                raise ValueError(f'Resume mismatch: {key}')
        old = json.loads((args.out/'config.json').read_text())
        if old['validation_keys'] != val_keys or old['initialization_sha256'] != init_hash:
            raise ValueError('Resume validation/initialization mismatch')
        net.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['optimizer'])
        step0, best = ckpt['step'], ckpt['best_val']
        state = ckpt['rng_state']
        torch.set_rng_state(state['cpu'])
        torch.cuda.set_rng_state(state['cuda'])
        np.random.set_state(state['numpy'])
        random.setstate(state['python'])
        with (args.out/'resume_events.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=time.time(), step=step0, pid=os.getpid()))+'\n')
    else:
        write_json(args.out/'config.json', config)
        source = args.out/'source_snapshot'
        source.mkdir()
        for path in [*Path(__file__).parent.glob('*.py'), PROJECT/'model.py', PROJECT/'tiny_unet.py', PROJECT/'edm.py', PROJECT/'utils.py', PROJECT/'run_guard.py', PROJECT/'paths.py']:
            shutil.copyfile(path, source/path.name)
        write_json(source/'hashes.json', {f.name: sha256(f) for f in source.glob('*.py')})
        best = validate(ema, val)
        write_json(args.out/'baseline.json', dict(initial_native_denoising_loss=best, initialization_sha256=init_hash))
        with torch.random.fork_rng(devices=[0]):
            save_grid(data.sample(12), args.out/'training_examples.png', columns=4)
            save_grid(val[:12], args.out/'validation_examples.png', columns=4)
        # A valid initial EMA exists even if no later validation improves it.
        save_checkpoint(args.out/'model_ema.pt', dict(step=0, model_config=net.config, ema=ema.state_dict(),
                        val_loss=best, manifest_sha256=manifest_hash, config=config))
        save_checkpoint(args.out/'latest.pt', dict(step=0, model_config=net.config, model=net.state_dict(),
                        ema=ema.state_dict(), optimizer=optimizer.state_dict(), val_loss=best,
                        best_val=best, rng_state=rng_state(), manifest_sha256=manifest_hash,
                        total_steps=args.steps, batch=args.batch, lr=args.lr, seed=args.seed,
                        deterministic=args.deterministic, config=config))
    print(json.dumps(dict(event='start', step=step0, shape=data.image_shape, initial_val=best, pid=os.getpid())), flush=True)
    start = time.monotonic()
    losses = []
    try:
        for step in range(step0+1, args.steps+1):
            net.train()
            x = data.sample(args.batch)
            sigma = (torch.randn(len(x), 1, 1, 1, device='cuda')*1.2-1.2).exp()
            pred = net(x+sigma*torch.randn_like(x), sigma)
            weight = (sigma.square()+net.sigma_data**2)/(sigma*net.sigma_data).square()
            loss = (weight*(pred-x).square()).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss at {step}')
            lr = args.lr*min(step/200., 1.)*(.15+.85*.5*(1+math.cos(math.pi*step/args.steps)))
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                decay = min(.9995, (1+step)/(10+step))
                for ep, np_ in zip(ema.parameters(), net.parameters()):
                    ep.lerp_(np_, 1-decay)
            losses.append(float(loss.detach()))
            if step % 50 == 0 or step == args.steps:
                elapsed = time.monotonic()-start
                row = dict(step=step, loss=float(np.mean(losses)), grad_norm=float(grad), lr=lr,
                           images_seen=step*args.batch, elapsed_sec=elapsed, steps_per_sec=(step-step0)/elapsed)
                with (args.out/'train_metrics.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
                print(json.dumps(row), flush=True)
                write_json(args.out/'status.json', dict(stage='training', pid=os.getpid(), time=time.time(), **row))
                losses.clear()
            if step % args.save_every == 0 or step == args.steps:
                vl = validate(ema, val)
                improved = vl < best
                best = min(best, vl)
                ckpt = dict(step=step, model_config=net.config, model=net.state_dict(), ema=ema.state_dict(),
                            optimizer=optimizer.state_dict(), val_loss=vl, best_val=best, rng_state=rng_state(),
                            manifest_sha256=manifest_hash, total_steps=args.steps, batch=args.batch,
                            lr=args.lr, seed=args.seed, deterministic=args.deterministic, config=config)
                save_checkpoint(args.out/'latest.pt', ckpt)
                if improved:
                    save_checkpoint(args.out/'best.pt', ckpt)
                    save_checkpoint(args.out/'model_ema.pt', {k: ckpt[k] for k in ['step', 'model_config', 'ema', 'val_loss', 'manifest_sha256', 'config']})
                row = dict(step=step, val_loss=vl, best_val=best)
                with (args.out/'val_metrics.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
                print(json.dumps(dict(event='checkpoint', **row)), flush=True)
        write_json(args.out/'completed.json', dict(step=args.steps, best_val=best, elapsed_sec=time.monotonic()-start))
        write_json(args.out/'status.json', dict(stage='complete', step=args.steps, best_val=best, time=time.time()))
    except Exception as exc:
        write_json(args.out/'status.json', dict(stage='failed', error=repr(exc), time=time.time(), pid=os.getpid()))
        raise


if __name__ == '__main__':
    main()
