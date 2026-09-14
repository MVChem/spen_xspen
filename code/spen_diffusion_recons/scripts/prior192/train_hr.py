"""Fine-tune the frozen-source 96 prior on native-derived 192 mouse images."""
import argparse
from collections import Counter, defaultdict
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
V2 = HERE.parent / 'prior96'
sys.path.insert(0, str(HERE.parent / 'core'))
sys.path.insert(0, str(V2))
from project_paths import CORE, RUNS, PRIOR96_DATA, PRIOR192_DATA, MOUSE_RAW
from model_v2 import StrongPrior, load_strong_prior
from prepare_data import sha256


def atomic_json(path, state):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    os.replace(temp, path)


def atomic_torch(path, state):
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    os.replace(temp, path)


def learning_rate(step, config):
    """Keep a recorded continuation schedule fixed across process restarts."""
    peak, end = config['lr'], config['steps']
    extension = config.get('continuation')
    if extension:
        origin = extension['start_step']
        warmup = extension['warmup_steps']
        if step <= origin + warmup:
            fraction = (step - origin) / warmup
            return extension['start_lr'] + fraction * (peak - extension['start_lr'])
        progress = (step - origin - warmup) / (end - origin - warmup)
        return peak * (.2 + .8 * .5 * (1 + math.cos(math.pi * progress)))
    return peak * min(step / 50., 1.) * (.2 + .8 * .5 * (1 + math.cos(math.pi * step / end)))


def source_plane_key(record):
    """Count acquired planes once, independent of FOV copies or augmentation."""
    key = f"{record['source_sha256']}:{record['slice_axis']}:{record['slice_index']}"
    declared = record.get('source_plane_key')
    if declared is not None and declared != key:
        raise ValueError('source_plane_key does not identify the original acquired plane')
    return key


def load_training_arrays(data):
    """Validate immutable data on CPU and choose the fixed validation subset.

    The original ds005236 manifest still selects every validation image in its
    existing order. Expanded data may contain additional validation sources, but
    selection_indices.val chooses only the established scanner-domain cohort.
    Test files are hashed and their array headers checked, never supplied to the
    optimizer or the model-selection denoiser.
    """
    manifest = json.loads((data / 'manifest.json').read_text())
    if manifest['size'] != 192 or manifest['dataset'] not in ('ds005236', 'mouse_multisource_native192'):
        raise ValueError('Expected an original or multisource native-derived 192 mouse dataset')
    parts = ('train', 'val', 'test')
    expanded = manifest['dataset'] == 'mouse_multisource_native192'
    for field in ('subjects', 'split_groups'):
        if field not in manifest:
            raise ValueError(f'Manifest missing {field}')
        for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            if set(manifest[field][a]) & set(manifest[field][b]):
                raise ValueError(f'{field} overlaps between {a} and {b}')
    arrays, hashes, plane_sets, subject_counts, source_counts = {}, {}, {}, {}, {}
    for part in parts:
        records = manifest['records'][part]
        path = data / (part + '.npy')
        expected_hash = manifest.get('npy_sha256', {}).get(part)
        if not expected_hash or sha256(path) != expected_hash:
            raise ValueError(f'{part}.npy does not match the manifest SHA256')
        hashes[part] = expected_hash
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if array.dtype != np.uint16 or array.shape != (len(records), 192, 192) or not len(records):
            raise ValueError(f'{part}.npy must be uint16 with one 192x192 image per nonempty record')
        if part in ('train', 'val'):
            for start in range(0, len(array), 256):
                if not np.isfinite(array[start:start + 256]).all():
                    raise ValueError(f'{part}.npy contains nonfinite images')
            arrays[part] = array
        keys = [record['key'] for record in records]
        if len(keys) != len(set(keys)):
            raise ValueError(f'Duplicate image record key in {part}')
        subjects = {record['subject'] for record in records}
        groups = {record.get('split_group', record['subject']) for record in records}
        if subjects != set(manifest['subjects'][part]) or groups != set(manifest['split_groups'][part]):
            raise ValueError(f'{part} record identities disagree with the manifest split lists')
        if any(record.get('species', 'mouse') != 'mouse' for record in records):
            raise ValueError('The HR mouse prior received a non-mouse record')
        subject_counts[part] = len(subjects)
        plane_sets[part] = {source_plane_key(record) for record in records}
        if expanded and part == 'train' and len(plane_sets[part]) != len(records):
            raise ValueError(f'Expanded {part} repeats acquired source planes as separate images')
        if 'unique_plane_counts' in manifest and manifest['unique_plane_counts'][part] != len(plane_sets[part]):
            raise ValueError(f'{part} unique-plane count differs from the original source records')
        if manifest.get('image_counts', {}).get(part, len(records)) != len(records):
            raise ValueError(f'{part} declared image count differs from its records')
        if manifest.get('subject_counts', {}).get(part, len(subjects)) != len(subjects):
            raise ValueError(f'{part} declared subject count differs from its records')
        per_source = defaultdict(set)
        for record in records:
            per_source[record['dataset']].add(source_plane_key(record))
        source_counts[part] = {name: {'image_records': sum(r['dataset'] == name for r in records),
                                      'unique_acquired_planes': len(planes)}
                               for name, planes in sorted(per_source.items())}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if plane_sets[a] & plane_sets[b]:
            raise ValueError(f'Original acquired planes overlap between {a} and {b}')
    indices = manifest.get('selection_indices', {}).get('val')
    if expanded and indices is None:
        raise ValueError('Expanded data must explicitly preserve selection_indices.val')
    if indices is None:
        indices = list(range(len(manifest['records']['val'])))
    if (not indices or any(type(i) is not int or i < 0 or i >= len(arrays['val']) for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError('Validation selection indices are empty, duplicated or out of bounds')
    val_records = [manifest['records']['val'][i] for i in indices]
    if expanded and any(r['dataset'] != 'ds005236' for r in val_records):
        raise ValueError('Expanded checkpoint selection must use only the established ds005236 validation cohort')
    if expanded and indices != [i for i, r in enumerate(manifest['records']['val']) if r['dataset'] == 'ds005236']:
        raise ValueError('Checkpoint selection must retain every ds005236 validation image in its original order')
    val = arrays['val'][indices].astype(np.float32) / 65535.
    # Match the old CUDA validation transformation exactly, in the same order.
    selected = [i for i, r in enumerate(val_records) if r['slice_index'] % 2 == 0]
    val[selected] = np.rot90(val[selected], 2, (-2, -1))
    val_keys = [r['key'] + (':rot180' if r['slice_index'] % 2 == 0 else '') for r in val_records]
    weights = np.asarray([r['sample_weight'] for r in manifest['records']['train']], dtype=np.float32)
    if not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isfinite(weights.sum()) or weights.sum() <= 0:
        raise ValueError('Every training sample must have a finite positive sampling weight')
    probability = defaultdict(float)
    for record, weight in zip(manifest['records']['train'], weights.astype(np.float64) / weights.astype(np.float64).sum()):
        probability[record['dataset']] += float(weight)
    audit = {'dataset': manifest['dataset'], 'npy_sha256': hashes,
             'image_counts': {part: len(manifest['records'][part]) for part in parts},
             'unique_plane_counts': {part: len(plane_sets[part]) for part in parts},
             'subject_counts': subject_counts, 'source_counts': source_counts,
             'training_source_probability': dict(sorted(probability.items())),
             'sampling_weight_sum': float(weights.astype(np.float64).sum()),
             'validation_selection_indices': indices, 'validation_selection_image_count': len(indices),
             'validation_selection_source_counts': dict(Counter(r['dataset'] for r in val_records)),
             'validation_selection_sha256': hashlib.sha256(np.ascontiguousarray(val * 2 - 1).tobytes()).hexdigest(),
             'test_usage': 'SHA256 and array-header integrity only; no test pixels enter training or checkpoint selection'}
    return manifest, arrays['train'], val[:, None] * 2 - 1, weights, val_keys, audit


def augment_magnitude(x):
    """Same conservative transform as data_v2, with size-dependent bias field."""
    b, _, height, width = x.shape
    device = x.device
    rotate = torch.rand(b, 1, 1, 1, device=device) < .5
    x = torch.where(rotate, torch.rot90(x, 2, (-2, -1)), x)
    angle = (torch.rand(b, device=device) * 2 - 1) * .12
    scale = torch.empty(b, device=device).uniform_(.88, 1.12)
    flip = torch.where(torch.rand(b, device=device) < .5, -1., 1.)
    theta = torch.zeros(b, 2, 3, device=device)
    theta[:, 0, 0] = angle.cos() * scale * flip
    theta[:, 0, 1] = -angle.sin() * scale
    theta[:, 1, 0] = angle.sin() * scale * flip
    theta[:, 1, 1] = angle.cos() * scale
    theta[:, :, 2] = torch.empty(b, 2, device=device).uniform_(-.14, .14)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    out = F.grid_sample(x, grid, align_corners=False, padding_mode='zeros')
    gamma = torch.empty(b, 1, 1, 1, device=device).uniform_(.85, 1.15)
    gain = torch.empty(b, 1, 1, 1, device=device).uniform_(.90, 1.08)
    field = F.interpolate(torch.randn(b, 1, 4, 4, device=device) * .08,
                          size=(height, width), mode='bicubic', align_corners=False).exp()
    return (out.clamp_min(0).pow(gamma) * gain * field).clamp(0, 1)


@torch.no_grad()
def validate(net, clean, batch):
    net.eval()
    gen = torch.Generator(device=clean.device).manual_seed(20260914)
    sigma = (torch.randn(len(clean), 1, 1, 1, device=clean.device, generator=gen) * 1.2 - 1.2).exp()
    noise = torch.randn(clean.shape, device=clean.device, generator=gen)
    weight = (sigma.square() + net.sigma_data ** 2) / (sigma * net.sigma_data).square()
    loss = 0.
    for index in range(0, len(clean), batch):
        sl = slice(index, index + batch)
        # Both checkpoints use FP32 inputs/EDM arithmetic, with inherited
        # EDMPrior.forward internally autocasting the CUDA denoiser to BF16.
        pred = net(clean[sl] + sigma[sl] * noise[sl], sigma[sl])
        loss += float(((pred - clean[sl]).square() * weight[sl]).flatten(1).mean(1).sum())
    return loss / len(clean)


def plot_curves(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [json.loads(s) for s in (out/'train_metrics.jsonl').read_text().splitlines()]
    vals = [json.loads(s) for s in (out/'val_metrics.jsonl').read_text().splitlines()]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot([r['step'] for r in rows], [r['loss'] for r in rows])
    axes[0].set(title='192 prior training', xlabel='Fine-tuning step', ylabel='EDM loss')
    axes[1].plot([r['step'] for r in vals], [r['val_loss'] for r in vals], marker='o')
    axes[1].axhline(vals[0]['val_loss'], linestyle='--', color='gray', label='Old 96 prior at 192')
    axes[1].set(title='Fixed-noise validation', xlabel='Fine-tuning step', ylabel='EDM loss')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'learning_curve.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=PRIOR192_DATA)
    parser.add_argument('--out', type=Path, default=RUNS/'prior192/train')
    parser.add_argument('--init-from', type=Path, default=RUNS/'prior96/strong_mouse96/model_ema.pt')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--save-every', type=int, default=250)
    parser.add_argument('--seed', type=int, default=20260914)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--continue-from', type=Path,
                        help='Branch a completed run, preserving model/optimizer/RNG/global steps')
    parser.add_argument('--stop-after', type=int,
                        help='Run only this many additional steps without changing the target schedule')
    args = parser.parse_args()
    if min(args.steps, args.batch, args.save_every) < 1:
        raise ValueError('steps, batch, save-every must be positive')
    if args.continue_from and args.resume:
        raise ValueError('Use --continue-from for a new branch or --resume for its existing state')
    if args.stop_after is not None and args.stop_after < 1:
        raise ValueError('stop-after must be positive')
    if args.continue_from and args.continue_from.resolve() == args.out.resolve():
        raise ValueError('Continuation must use a fresh output directory')
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out/'.training.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.out/'completed.json').exists():
        raise FileExistsError('Training already completed')
    if (args.out/'config.json').exists() and not args.resume:
        raise FileExistsError('Existing run; specify --resume')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda:0')
    atomic_json(args.out/'process.json', dict(pid=os.getpid(), started_at=time.time(), physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), argv=sys.argv))
    manifest, train_array, val_array, weight_array, val_keys, data_audit = load_training_arrays(args.data)
    train = torch.from_numpy(train_array.astype(np.float32) / 65535.)[:, None].to(device)
    val = torch.from_numpy(val_array).to(device)
    weights = torch.from_numpy(weight_array).to(device)
    original = torch.load(args.init_from, map_location='cpu', weights_only=False)
    net = StrongPrior(**original['model_config']).to(device)
    net.load_state_dict(original['ema'])
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.)
    manifest_hash = sha256(args.data/'manifest.json')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(model_config=net.config, initialization_sha256=sha256(args.init_from),
        initialization_step=original['step'], manifest_sha256=manifest_hash,
        input_shape=[args.batch, 1, 192, 192],
        precision='Training and validation: FP32 tensors/EDM arithmetic; internal CUDA denoiser BF16 autocast. Identical validation precision for old and adapted priors.',
        validation_seed=20260914, validation_keys=val_keys, selection='Lowest fixed-noise validation EMA loss including step 0',
        subject_counts=manifest['subject_counts'], torch=torch.__version__,
        data_integrity=data_audit, dataset=manifest['dataset'],
        unique_plane_counts=data_audit['unique_plane_counts'],
        data_source_counts=data_audit['source_counts'],
        training_source_probability=data_audit['training_source_probability'],
        validation_selection_indices=data_audit['validation_selection_indices'],
        data_builder_source_sha256=manifest.get('source_code_sha256', {}),
        gpu=torch.cuda.get_device_name(), physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        objective='Unconditional EDM weighted denoising MSE, log(sigma) ~ N(-1.2,1.2^2)',
        augmentation='Same as data_v2; bias field interpolated to current image shape',
        note=f"Native-derived {manifest['dataset']} 192-grid fine-tune; {data_audit['unique_plane_counts']['train']} distinct acquired training planes. Source/FOV differences are recorded in the data manifest. Test pixels never enter training or checkpoint selection.")
    sources = [Path(__file__), HERE/'prepare_hr.py', V2/'model_v2.py', V2/'tiny_unet_v2.py', CORE/'model.py', V2/'prepare_data.py', V2/'data_v2.py']
    config['source_sha256'] = {str(p): sha256(p) for p in sources}
    # The initial source_snapshot stays immutable on resume. Record the actual
    # code used by this process separately, including compatible loader updates.
    atomic_json(args.out/'execution_provenance.json', dict(started_at=time.time(), argv=sys.argv,
        manifest_sha256=manifest_hash, source_sha256=config['source_sha256'], data_integrity=data_audit))
    step0, best, best_step = 0, float('inf'), 0
    if args.resume or args.continue_from:
        source = args.out if args.resume else args.continue_from
        previous = json.loads((source/'config.json').read_text())
        state = torch.load(source/'latest.pt', map_location='cpu', weights_only=False)
        assert state['manifest_sha256'] == manifest_hash and state['model_config'] == net.config
        assert state['global_batch'] == args.batch
        if args.steps <= state['step']:
            raise ValueError('Requested target must exceed the saved global step')
        for key in ('batch', 'lr', 'seed', 'manifest_sha256', 'initialization_sha256'):
            if previous[key] != config[key]:
                raise ValueError(f'Resume/continuation changes {key}')
        if previous.get('validation_keys') != val_keys:
            raise ValueError('Resume/continuation changes fixed-noise validation identities or order')
        previous_audit = previous.get('data_integrity')
        if previous_audit is not None and previous_audit != data_audit:
            raise ValueError('Resume/continuation changes verified data or validation selection')
        if args.resume:
            for key in ('steps', 'save_every'):
                if previous[key] != config[key]:
                    raise ValueError(f'Resume changes {key}; create an explicit continuation branch')
            config = previous
        else:
            warmup = min(250, max(1, (args.steps - state['step']) // 4))
            config['continuation'] = dict(source_run=str(source.resolve()),
                source_latest_sha256=sha256(source/'latest.pt'),
                source_best_sha256=sha256(source/'model_ema.pt'),
                source_config_sha256=sha256(source/'config.json'),
                start_step=state['step'],start_lr=state['optimizer']['param_groups'][0]['lr'],
                warmup_steps=warmup,previous_target_steps=previous['steps'],
                schedule='Linear warm restart from saved LR, then cosine to 20% of peak at total target; previous completed schedule is retained as history.',
                meaning='steps is the cumulative number of 192-grid fine-tuning updates, not the number of additional updates.')
            config['note'] = 'Continuation of the native-derived 192 prior; original 2000-step experiment is preserved.'
            for name in ('model_ema.pt','baseline.json','train_metrics.jsonl','val_metrics.jsonl'):
                shutil.copyfile(source/name,args.out/name)
            atomic_json(args.out/'config.json',config)
            snapshot=args.out/'source_snapshot';snapshot.mkdir(exist_ok=True)
            for path in sources:shutil.copyfile(path,snapshot/path.name)
        net.load_state_dict(state['model'])
        ema.load_state_dict(state['ema'])
        opt.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng_cpu'].cpu())
        torch.cuda.set_rng_state(state['rng_cuda'].cpu())
        step0, best, best_step = state['step'], state['best_val'], state['best_step']
        # Record restoration before any training update. Tensor/optimizer bytes
        # are loaded directly; all stochastic initialization has already ended.
        atomic_json(args.out/'restoration.json',dict(source_run=str(source.resolve()),
            restored_step=step0,restored_best_step=best_step,restored_best_val=best,
            restored_optimizer_lr=opt.param_groups[0]['lr'],global_batch=args.batch,
            target_step=args.steps,additional_target_steps=args.steps-step0,
            cpu_rng_equal=bool(torch.equal(torch.get_rng_state(),state['rng_cpu'].cpu())),
            cuda_rng_equal=bool(torch.equal(torch.cuda.get_rng_state(),state['rng_cuda'].cpu()))))
    else:
        atomic_json(args.out/'config.json', config)
        snapshot = args.out/'source_snapshot'
        snapshot.mkdir(exist_ok=True)
        for path in sources:
            shutil.copyfile(path, snapshot/path.name)
    print(json.dumps(dict(event='start', steps=args.steps, batch=args.batch, device=config['physical_gpu'], smoke=args.smoke)), flush=True)
    if not args.resume and not args.continue_from and not args.smoke:
        best = validate(ema, val, args.batch)
        atomic_json(args.out/'baseline.json', dict(old_prior_at_192_loss=best, initialization_step=original['step'], images=len(val), validation_seed=20260914))
        row = dict(step=0, val_loss=best, best_val=best, best_step=0)
        (args.out/'val_metrics.jsonl').write_text(json.dumps(row) + '\n')
        atomic_torch(args.out/'model_ema.pt', dict(step=0, model_config=net.config, ema=ema.state_dict(), val_loss=best, manifest_sha256=manifest_hash, resolution=192))
        print(json.dumps(dict(event='baseline', **row)), flush=True)
    start = time.monotonic()
    losses = []
    steady_start = None
    torch.cuda.reset_peak_memory_stats()
    end_step = args.steps if args.stop_after is None else min(args.steps,step0+args.stop_after)
    for step in range(step0 + 1, end_step + 1):
        net.train()
        ids = torch.multinomial(weights, args.batch, replacement=True)
        x = augment_magnitude(train[ids]) * 2 - 1
        sigma = (torch.randn(args.batch, 1, 1, 1, device=device) * 1.2 - 1.2).exp()
        noisy = x + sigma * torch.randn_like(x)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            pred = net(noisy, sigma)
        weight = (sigma.square() + net.sigma_data ** 2) / (sigma * net.sigma_data).square()
        loss = (weight * (pred.float() - x).square()).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at step {step}')
        lr = learning_rate(step, config)
        for group in opt.param_groups:
            group['lr'] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        decay = min(.9995, (1 + step) / (10 + step))
        with torch.no_grad():
            torch._foreach_lerp_(list(ema.parameters()), list(net.parameters()), 1 - decay)
        losses.append(float(loss.detach()))
        if step == step0 + 3:
            torch.cuda.synchronize()
            steady_start = time.monotonic()
        if step % 50 == 0 or step == end_step:
            elapsed = time.monotonic() - start
            row = dict(step=step, loss=float(np.mean(losses)), lr=lr, grad_norm=float(grad_norm),
                elapsed_sec=elapsed, steps_per_sec=(step-step0)/elapsed, images_seen=step*args.batch,
                peak_memory_gb=torch.cuda.max_memory_allocated()/1e9)
            with (args.out/'train_metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
            losses.clear()
        if not args.smoke and (step % args.save_every == 0 or step == end_step):
            loss_val = validate(ema, val, args.batch)
            improved = loss_val < best
            if improved:
                best, best_step = loss_val, step
            state = dict(step=step, model_config=net.config, model=net.state_dict(), ema=ema.state_dict(),
                optimizer=opt.state_dict(), val_loss=loss_val, best_val=best, best_step=best_step,
                manifest_sha256=manifest_hash, global_batch=args.batch, resolution=192,
                rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state())
            atomic_torch(args.out/'latest.pt', state)
            if improved:
                atomic_torch(args.out/'model_ema.pt', {k: state[k] for k in ['step', 'model_config', 'ema', 'val_loss', 'manifest_sha256', 'resolution']})
            row = dict(step=step, val_loss=loss_val, best_val=best, best_step=best_step)
            with (args.out/'val_metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            print(json.dumps(dict(event='checkpoint', **row)), flush=True)
            plot_curves(args.out)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    if args.smoke:
        timing = dict(start_step=step0,end_step=end_step,target_steps=args.steps, batch=args.batch, elapsed_sec=elapsed,
            steady_steps_per_sec=(end_step-step0-3)/(time.monotonic()-steady_start) if steady_start is not None else None,
            peak_memory_gb=torch.cuda.max_memory_allocated()/1e9)
        atomic_json(args.out/'smoke.json', timing)
        print(json.dumps(dict(event='smoke_complete', **timing)), flush=True)
        return
    if end_step < args.steps:
        atomic_json(args.out/'paused.json',dict(step=end_step,target_step=args.steps,elapsed_sec=elapsed))
        print(json.dumps(dict(event='PAUSED_AT_REQUESTED_STEP',step=end_step)),flush=True)
        return
    del net, ema, opt, original
    torch.cuda.empty_cache()
    best_model, best_ckpt = load_strong_prior(args.out/'model_ema.pt', device)
    with torch.inference_mode():
        check = best_model(val[:1] + .1 * torch.randn_like(val[:1]), torch.tensor([.1], device=device))
    assert check.shape == (1, 1, 192, 192) and torch.isfinite(check).all()
    completed = dict(step=args.steps, best_step=best_step, best_val=best, elapsed_sec=elapsed,
        start_step=step0,additional_steps_this_process=args.steps-step0,
        output_shape=list(check.shape), output_finite=True, checkpoint_sha256=sha256(args.out/'model_ema.pt'),
        selected_checkpoint_step=best_ckpt['step'], completed_at=time.time())
    atomic_json(args.out/'completed.json', completed)
    print(json.dumps(dict(event='TRAINING_COMPLETE', **completed)), flush=True)


if __name__ == '__main__':
    main()
