"""Train a native 192 rodent EDM prior from PNG-derived arrays using torchrun.

The acquisition operator is used only after training. Clean uint16 images stay
in CPU memory maps; only the sampled batch is transferred to each GPU.
"""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import copy
from datetime import timedelta
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

HERE = Path(__file__).resolve().parent
CORE = HERE.parent / 'core'
V2 = HERE.parent / 'prior96'
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(V2))
from model_v2 import StrongPrior
from model import sigma_schedule
from run_guard import acquire_run_lock
from train_hr import augment_magnitude, atomic_json, atomic_torch


def augmentation_config(contrast_range=(.9, 1.1), contrast_gate=.05, legacy=False):
    result = dict(
        version='rodent192_v1' if legacy else 'rodent192_v2_foreground_contrast',
        rotate_180_probability=.5, affine_rotation_radians=[-.12, .12],
        affine_coordinate_scale=[.88, 1.12], horizontal_flip_probability=.5,
        affine_translation_normalized=[-.14, .14], gamma=[.85, 1.15], gain=[.90, 1.08],
        bias_field=dict(coarse_shape=[4, 4], log_standard_deviation=.08,
                        interpolation='bicubic', multiplicative=True),
        output_range=[0., 1.], intent='Conservative intensity variability; not a simulation of an MRI sequence',
    )
    if not legacy:
        result['foreground_contrast'] = dict(
            factor_range=list(contrast_range), soft_gate=float(contrast_gate),
            gate='x / (x + soft_gate)', mean='sum(gate * x) / sum(gate)',
            transform='clamp(x + gate * (factor - 1) * (x - mean), 0, 1)',
            applied_after='legacy geometric, gamma, gain, and bias field augmentation',
        )
    return result


def foreground_contrast(x, factor_range=(.9, 1.1), soft_gate=.05):
    """Perturb foreground contrast while keeping exact zero background fixed."""
    gate = x / (x + soft_gate)
    dims = tuple(range(1, x.ndim))
    mean = (gate * x).sum(dims, keepdim=True) / gate.sum(dims, keepdim=True).clamp_min(1e-8)
    factor = torch.empty(len(x), 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(*factor_range)
    raw = x + gate * (factor - 1) * (x - mean)
    stats = dict(clip_fraction=((raw < 0) | (raw > 1)).float().mean(),
                 clip_low_fraction=(raw < 0).float().mean(),
                 clip_high_fraction=(raw > 1).float().mean(), factor_mean=factor.mean())
    return raw.clamp(0, 1), stats


def augment_training(x, args):
    return foreground_contrast(augment_magnitude(x), args.contrast_range, args.contrast_gate)


def microbatch_loss_backward(model, noisy, clean, sigma, weight, micro_batch, distributed=False):
    """Accumulate the full logical-batch EDM loss with one final DDP sync."""
    batch_loss = clean.new_zeros(())
    for start in range(0, len(clean), micro_batch):
        end = min(start + micro_batch, len(clean))
        context = model.no_sync() if distributed and end < len(clean) else nullcontext()
        with context:
            pred = model(noisy[start:end], sigma[start:end])
            loss = (weight[start:end] * (pred - clean[start:end]).square()).mean()
            loss = loss * ((end - start) / len(clean))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite micro-batch EDM loss')
            loss.backward()
        batch_loss.add_(loss.detach())
    return batch_loss


def validate_resume_config(previous, current, extend_run=False, adopt_all_data=False):
    if extend_run and adopt_all_data:
        raise ValueError('Extend target steps and adopt all data in separate audited transitions')
    immutable = ('local_batch', 'world_size', 'global_batch', 'lr', 'warmup', 'seed',
                 'base_ch', 'model_config', 'image_shape', 'save_every', 'sample_every',
                 'val_batch', 'backend', 'log_every')
    if not adopt_all_data:
        immutable += ('data', 'manifest_sha256', 'validation_indices', 'selection')
    for key in immutable:
        if previous.get(key) != current.get(key):
            raise ValueError(f'Resume changes {key}')
    if not adopt_all_data and previous.get('micro_batch', previous['local_batch']) != current['micro_batch']:
        raise ValueError('Resume changes micro_batch; use an explicitly audited data adoption')
    old_sources, new_sources = previous['source_sha256'], current['source_sha256']
    if old_sources.keys() != new_sources.keys():
        raise ValueError('Resume changes source file set')
    trainer_path = str(Path(__file__).resolve())
    for path, digest in old_sources.items():
        if new_sources[path] != digest and not ((extend_run or adopt_all_data) and path == trainer_path):
            raise ValueError(f'Resume changes source_sha256: {path}')
    if extend_run:
        if current['steps'] <= previous['steps']:
            raise ValueError('--extend-run requires a strictly larger target step count')
    else:
        if current['steps'] != previous['steps']:
            raise ValueError('Resume changes steps; use --extend-run to increase the target')
        if previous.get('augmentation', augmentation_config(legacy=True)) != current['augmentation']:
            raise ValueError('Resume changes augmentation; an explicit run extension is required')
    if adopt_all_data:
        if previous.get('training_mode') == 'all_data_no_holdout' or not previous['validation_indices']:
            raise ValueError('Run already uses all data; use ordinary --resume')
        if current.get('training_mode') != 'all_data_no_holdout' or current['validation_indices']:
            raise ValueError('--adopt-all-data requires a dataset with no holdout or validation selection')


def original_schedule(config):
    return dict(kind='warmup_cosine', end_step=config['steps'], peak_lr=config['lr'],
                warmup_steps=config['warmup'], min_lr=.15 * config['lr'])


def scheduled_lr(step, schedule):
    if schedule['kind'] == 'checkpoint_cosine':
        progress = (step - schedule['start_step']) / (schedule['end_step'] - schedule['start_step'])
        progress = min(max(progress, 0.), 1.)
        return schedule['min_lr'] + (schedule['start_lr'] - schedule['min_lr']) * .5 * (1 + math.cos(math.pi * progress))
    if schedule['kind'] != 'warmup_cosine':
        raise ValueError(f"Unknown learning-rate schedule: {schedule['kind']}")
    return schedule['peak_lr'] * min(step / schedule['warmup_steps'], 1.) * (
        .15 + .85 * .5 * (1 + math.cos(math.pi * step / schedule['end_step'])))


def checkpoint_lr(checkpoint):
    values = [float(group['lr']) for group in checkpoint['optimizer']['param_groups']]
    if not values or any(not math.isfinite(v) or v <= 0 for v in values) or len(set(values)) != 1:
        raise ValueError('Expected the same finite positive checkpoint LR in every optimizer group')
    return values[0]


def check_checkpoint_schedule(checkpoint, schedule):
    saved = checkpoint.get('lr_schedule')
    if saved == schedule or (saved is None and schedule['kind'] == 'warmup_cosine'):
        return
    # config.json may have committed an extension before its first new checkpoint.
    if (schedule['kind'] == 'checkpoint_cosine' and checkpoint['step'] == schedule['start_step']
            and checkpoint_lr(checkpoint) == schedule['start_lr']):
        return
    raise ValueError('Checkpoint learning-rate schedule disagrees with config.json')


def extension_schedule(previous, checkpoint, target_steps):
    start_step, start_lr = int(checkpoint['step']), checkpoint_lr(checkpoint)
    min_lr = .15 * previous['lr']
    if target_steps <= max(start_step, previous['steps']):
        raise ValueError('Extension must increase the previous target and exceed checkpoint step')
    if start_lr < min_lr:
        raise ValueError('Checkpoint LR is below the requested cosine floor; cannot make a decreasing extension')
    return dict(kind='checkpoint_cosine', start_step=start_step, end_step=target_steps,
                start_lr=start_lr, min_lr=min_lr, peak_lr=previous['lr'],
                previous_target_steps=previous['steps'], source='checkpoint optimizer param_groups LR')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sampling_source_probabilities(manifest):
    totals, images, groups = defaultdict(float), defaultdict(int), defaultdict(set)
    for record in manifest['records']['train']:
        source = record['dataset']
        totals[source] += record['sample_weight']
        images[source] += 1
        groups[source].add(record['subject_group'])
    normalization = sum(totals.values())
    return {source: dict(probability=totals[source] / normalization, images=images[source],
                         subject_groups=len(groups[source])) for source in sorted(totals)}


def load_data(root, verify_hashes=True):
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['dataset'] not in ('rodent_native192_png', 'rodent_native192_png_all') or manifest['size'] != 192:
        raise ValueError('Expected native rodent PNG-derived 192 dataset')
    all_training = manifest['dataset'] == 'rodent_native192_png_all'
    if all_training:
        for part in ('val', 'test'):
            if (manifest['records'][part] or manifest['split_groups'][part]
                    or manifest['image_counts'][part] or manifest['selection_indices'].get(part, [])):
                raise ValueError(f'All-data mode must not retain a {part} subset or monitor selection')
    arrays, identities = {}, {}
    parts = ('train',) if all_training else ('train', 'val', 'test')
    for part in parts:
        records = manifest['records'][part]
        path = root / f'{part}.npy'
        if verify_hashes and sha256(path) != manifest['npy_sha256'][part]:
            raise ValueError(f'{part} array SHA256 differs from manifest')
        a = np.load(path, mmap_mode='r', allow_pickle=False)
        if a.dtype != np.uint16 or a.shape != (len(records), 192, 192) or not len(a):
            raise ValueError(f'Invalid {part} array shape/dtype/count')
        arrays[part] = a
        identities[part] = {}
        for key in ('key', 'split_group', 'physical_plane_key'):
            identities[part][key] = {r[key] for r in records}
        if len(identities[part]['key']) != len(records):
            raise ValueError(f'Duplicate keys in {part}')
        if identities[part]['split_group'] != set(manifest['split_groups'][part]):
            raise ValueError(f'{part} split groups disagree with records')
    for a, b in (() if all_training else (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        for key in identities[a]:
            if identities[a][key] & identities[b][key]:
                raise ValueError(f'{key} overlap: {a}/{b}')
    weights = torch.tensor([r['sample_weight'] for r in manifest['records']['train']], dtype=torch.float64)
    if not torch.isfinite(weights).all() or not bool((weights > 0).all()):
        raise ValueError('Invalid sampling weights')
    if all_training:
        groups = defaultdict(list)
        for i, record in enumerate(manifest['records']['train']):
            groups[record['subject_group']].append(i)
        for indices in groups.values():
            expected = 1. / (len(groups) * len(indices))
            if not torch.allclose(weights[indices], torch.full_like(weights[indices], expected), rtol=1e-10, atol=1e-14):
                raise ValueError('All-data weights must sample every subject equally and images uniformly within subject')
        return manifest, arrays, weights, []
    ids = manifest['selection_indices']['val']
    if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= len(arrays['val']) for i in ids):
        raise ValueError('Invalid fixed validation selection')
    # Test arrays are checked above but no test pixels are used by the trainer.
    del arrays['test']
    return manifest, arrays, weights, ids


def verify_all_data_adoption(previous, manifest):
    """Prove that the new training set is exactly the union of the parent splits."""
    parent_path = Path(previous['data']) / 'manifest.json'
    parent_hash = sha256(parent_path)
    if (parent_hash != previous['manifest_sha256']
            or manifest.get('parent_manifest_sha256') != parent_hash
            or manifest['dataset'] != 'rodent_native192_png_all'):
        raise ValueError('All-data parent manifest does not match the resumed run')
    parent = json.loads(parent_path.read_text())
    if parent['dataset'] != 'rodent_native192_png':
        raise ValueError('Expected the original three-split parent dataset')
    if manifest.get('parent_npy_sha256') != parent['npy_sha256']:
        raise ValueError('All-data parent array provenance differs')
    if manifest.get('parent_image_counts') != parent['image_counts']:
        raise ValueError('All-data parent split counts differ')
    for key in ('size', 'normalization', 'source_manifest_sha256', 'source_export_provenance_sha256'):
        if manifest.get(key) != parent.get(key):
            raise ValueError(f'All-data migration changes {key}')
    originals = {}
    for part in ('train', 'val', 'test'):
        for index, record in enumerate(parent['records'][part]):
            if record['key'] in originals:
                raise ValueError('Duplicate key in parent dataset')
            originals[record['key']] = (part, index, record)
    new_records = manifest['records']['train']
    if len(new_records) != len(originals) or {r['key'] for r in new_records} != set(originals):
        raise ValueError('All-data train records must equal the complete parent split union')
    changed_fields = {'split', 'sample_weight', 'original_split', 'original_split_index'}
    for record in new_records:
        part, index, original = originals[record['key']]
        if (record.get('split') != 'train' or record.get('original_split') != part
                or record.get('original_split_index') != index):
            raise ValueError(f'Parent split provenance changed for {record["key"]}')
        clean = {k: v for k, v in record.items() if k not in changed_fields}
        old_clean = {k: v for k, v in original.items() if k not in changed_fields}
        if clean != old_clean:
            raise ValueError(f'Image pixels or source provenance changed for {record["key"]}')
    return dict(parent_manifest_sha256=parent_hash, original_split_counts=parent['image_counts'],
                parent_sampling_source_probability=sampling_source_probabilities(parent),
                all_training_images=len(new_records), proof='Exact parent split union; unchanged per-image pixel, PNG, and source provenance')


def validate_checkpoint_manifest(checkpoint, previous, current, adopt_all_data=False, checkpoint_path=None):
    if adopt_all_data:
        if checkpoint['manifest_sha256'] != previous['manifest_sha256']:
            raise ValueError('Checkpoint does not match the parent run manifest')
        return
    if checkpoint['manifest_sha256'] == current['manifest_sha256']:
        return
    # A committed adoption may still have the untouched parent latest.pt until
    # its first checkpoint. The archived digest makes this restart unambiguous.
    history = previous.get('data_adoption_history', [])
    if history:
        event = history[-1]
        if (checkpoint['step'] == event['checkpoint_step']
                and checkpoint['manifest_sha256'] == event['previous_manifest_sha256']
                and current['manifest_sha256'] == event['manifest_sha256']):
            if checkpoint_path is not None and sha256(checkpoint_path) != event['checkpoint_sha256']:
                raise ValueError('Pending all-data adoption checkpoint bytes changed')
            return
    raise ValueError('Resume checkpoint manifest differs from the current dataset')


def light_checkpoint(state, manifest_hash=None, all_training=False):
    light = {key: state[key] for key in ('step', 'model_config', 'ema', 'img_resolution')}
    light['manifest_sha256'] = manifest_hash or state['manifest_sha256']
    light['val_loss'] = None if all_training else state['val_loss']
    if all_training:
        light.update(selection='current EMA at every checkpoint; final target-step EMA',
                     training_mode='all_data_no_holdout')
    return light


def tensor_batch(array, indices, device):
    cpu = torch.from_numpy(array[indices].astype(np.float32) / 65535.)[:, None]
    return cpu.pin_memory().to(device, non_blocking=True)


@torch.no_grad()
def validation(net, array, records, indices, batch, device):
    net.eval()
    gen = torch.Generator(device=device).manual_seed(20260914)
    sigma = (torch.randn(len(indices), 1, 1, 1, generator=gen, device=device) * 1.2 - 1.2).exp()
    noise = torch.randn(len(indices), 1, 192, 192, generator=gen, device=device)
    grouped = defaultdict(list)
    for start in range(0, len(indices), batch):
        selected = indices[start:start + batch]
        x = tensor_batch(array, selected, device) * 2 - 1
        s = sigma[start:start + len(x)]
        pred = net(x + s * noise[start:start + len(x)], s)
        weight = (s.square() + net.sigma_data ** 2) / (s * net.sigma_data).square()
        losses = (weight * (pred - x).square()).flatten(1).mean(1).cpu().tolist()
        for index, loss in zip(selected, losses):
            grouped[records[index]['split_group']].append(loss)
    result = float(np.mean([np.mean(v) for v in grouped.values()]))
    if not math.isfinite(result):
        raise FloatingPointError('Nonfinite validation loss')
    return result


def save_grid(x, path, columns=4):
    from PIL import Image
    a = ((x.detach().float().cpu().numpy()[:, 0] + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
    canvas = np.zeros((math.ceil(len(a) / columns) * 192, columns * 192), dtype=np.uint8)
    for i, tile in enumerate(a):
        row, col = divmod(i, columns)
        canvas[row * 192:(row + 1) * 192, col * 192:(col + 1) * 192] = tile
    Image.fromarray(canvas).save(path)


@torch.no_grad()
def sample192(net, count=8, steps=64, seed=20260914):
    device = next(net.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    sigmas = sigma_schedule(steps, device=device)
    x = torch.randn(count, 1, 192, 192, device=device, generator=gen) * sigmas[0]
    for i, (s, sn) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        d = (x - net(x, s)) / s
        nxt = x + (sn - s) * d
        if i < len(sigmas) - 2:
            dn = (nxt - net(nxt, sn)) / sn
            nxt = x + (sn - s) * (d + dn) / 2
        x = nxt
    return x


def log_json(path, row):
    with path.open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)


@torch.no_grad()
def augmentation_preview(array, weights, device, args, directory):
    directory.mkdir(parents=True, exist_ok=True)
    devices = [device.index] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(20260914)
        indices = torch.multinomial(weights, 8, replacement=True).numpy()
        clean = torch.from_numpy(array[indices].astype(np.float32) / 65535.)[:, None].to(device)
        base = augment_magnitude(clean)
        augmented, stats = foreground_contrast(base, args.contrast_range, args.contrast_gate)
        tiles = torch.stack((clean, base, augmented), dim=1).flatten(0, 1)
        save_grid(tiles * 2 - 1, directory / 'augmentation_preview.png', columns=3)
        row = dict(columns=['original', 'existing augmentation', 'existing + foreground contrast'],
                   training_indices=indices.tolist(), augmentation=augmentation_config(args.contrast_range, args.contrast_gate),
                   **{key: float(value) for key, value in stats.items()},
                   zero_background_preserved=bool((augmented[base == 0] == 0).all()),
                   existing_at_zero_fraction=float((base == 0).float().mean()),
                   existing_at_one_fraction=float((base == 1).float().mean()))
        atomic_json(directory / 'augmentation_preview.json', row)
    return row


def archive_extension(out, previous, config, checkpoint_step, source_paths):
    """Commit extension metadata without changing any checkpoint tensors."""
    stamp = time.time_ns()
    archive = out / 'extensions' / f'step{checkpoint_step}_to{config["steps"]}_{stamp}'
    archive.mkdir(parents=True)
    shutil.copyfile(out / 'config.json', archive / 'config_before.json')
    snapshot = out / f'extension_source_step{checkpoint_step}'
    if snapshot.exists():
        snapshot = out / f'extension_source_step{checkpoint_step}_{stamp}'
    snapshot.mkdir()
    for path in source_paths:
        shutil.copyfile(path, snapshot / path.name)
    rollbacks = {}
    for name in ('train_metrics.jsonl', 'val_metrics.jsonl'):
        path = out / name
        if not path.exists():
            continue
        content = path.read_text()
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        kept = [row for row in rows if row.get('step', checkpoint_step) <= checkpoint_step]
        shutil.copyfile(path, archive / name)
        temp = path.with_suffix(path.suffix + '.extension.tmp')
        temp.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in kept))
        os.replace(temp, path)
        rollbacks[name] = dict(archived_rows=len(rows), retained_rows=len(kept), removed_rows=len(rows) - len(kept))
    for name in ('status.json', 'completed.json', 'augmentation_preview.json', 'augmentation_preview.png'):
        if (out / name).exists():
            shutil.copyfile(out / name, archive / name)
    event = dict(event='extend_run', started_at=time.time(), checkpoint_step=checkpoint_step,
                 previous_target_steps=previous['steps'], target_steps=config['steps'],
                 checkpoint_sha256=sha256(out / 'latest.pt'),
                 previous_config=str(archive / 'config_before.json'), source_snapshot=str(snapshot),
                 previous_source_sha256=previous['source_sha256'], source_sha256=config['source_sha256'],
                 previous_augmentation=previous.get('augmentation', augmentation_config(legacy=True)),
                 augmentation=config['augmentation'], lr_schedule=config['lr_schedule'],
                 metric_rollback=rollbacks,
                 preserved_state=['model', 'ema', 'optimizer', 'rng_states', 'step', 'best_val', 'best_step'])
    config['extension_history'] = previous.get('extension_history', []) + [event]
    atomic_json(archive / 'extension.json', event)
    atomic_json(out / 'config.json', config)
    if (out / 'completed.json').exists():
        (out / 'completed.json').unlink()
    atomic_json(out / 'status.json', dict(status='resuming', step=checkpoint_step,
                                         target_steps=config['steps'], lr_schedule=config['lr_schedule']))
    log_json(out / 'extension_events.jsonl', event)
    return archive


def archive_data_adoption(out, previous, config, checkpoint_step, source_paths, proof):
    archive = out / 'data_adoptions' / f'step{checkpoint_step}_{time.time_ns()}'
    archive.mkdir(parents=True)
    shutil.copyfile(out / 'config.json', archive / 'config_before.json')
    shutil.copyfile(Path(previous['data']) / 'manifest.json', archive / 'parent_manifest.json')
    # latest.pt is always replaced atomically, so this hard link retains the
    # exact parent checkpoint without a second 200+ MB write at migration time.
    os.link(out / 'latest.pt', archive / 'parent_latest.pt')
    snapshot = archive / 'source_snapshot'
    snapshot.mkdir()
    for path in source_paths:
        shutil.copyfile(path, snapshot / path.name)
    rollbacks = {}
    for name in ('train_metrics.jsonl', 'val_metrics.jsonl'):
        path = out / name
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        kept = [row for row in rows if row.get('step', checkpoint_step) <= checkpoint_step]
        shutil.copyfile(path, archive / name)
        rollbacks[name] = dict(archived_rows=len(rows), retained_rows=len(kept), removed_rows=len(rows) - len(kept))
        if name == 'val_metrics.jsonl':
            path.unlink()
        else:
            temp = path.with_suffix(path.suffix + '.adoption.tmp')
            temp.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in kept))
            os.replace(temp, path)
    for name in ('status.json', 'model_ema.pt', 'baseline.json', 'validation_examples.png',
                 'augmentation_preview.json', 'augmentation_preview.png', 'sampling_coverage.json'):
        if (out / name).exists():
            shutil.copyfile(out / name, archive / name)
    for name in ('baseline.json', 'validation_examples.png'):
        if (out / name).exists():
            (out / name).unlink()
    event = dict(event='adopt_all_data', started_at=time.time(), checkpoint_step=checkpoint_step,
                 target_steps=config['steps'], previous_manifest_sha256=previous['manifest_sha256'],
                 manifest_sha256=config['manifest_sha256'], checkpoint_sha256=sha256(archive / 'parent_latest.pt'),
                 previous_config=str(archive / 'config_before.json'), parent_checkpoint=str(archive / 'parent_latest.pt'),
                 source_snapshot=str(snapshot), previous_source_sha256=previous['source_sha256'],
                 source_sha256=config['source_sha256'], proof=proof,
                 previous_selection=previous['selection'], selection=config['selection'],
                 previous_sampling_source_probability=proof['parent_sampling_source_probability'],
                 sampling_source_probability=config['sampling_source_probability'],
                 previous_micro_batch=previous.get('micro_batch', previous['local_batch']),
                 micro_batch=config['micro_batch'], global_batch=config['global_batch'],
                 lr_schedule=config['lr_schedule'], augmentation=config['augmentation'],
                 metric_rollback=rollbacks, preserved_state=['model', 'ema', 'optimizer', 'rng_states', 'step', 'lr_schedule'])
    config['extension_history'] = previous.get('extension_history', [])
    config['data_adoption_history'] = previous.get('data_adoption_history', []) + [event]
    config['sampling_coverage_start_step'] = checkpoint_step
    atomic_json(archive / 'adoption.json', event)
    atomic_json(out / 'config.json', config)
    atomic_json(out / 'status.json', dict(status='resuming', step=checkpoint_step, target_steps=config['steps'],
                                         training_mode='all_data_no_holdout', lr_schedule=config['lr_schedule']))
    log_json(out / 'data_adoption_events.jsonl', event)
    return archive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--local-batch', type=int, default=16)
    p.add_argument('--micro-batch', type=int,
                   help='Forward/backward chunk size; optimizer batch remains local-batch (default: local-batch)')
    p.add_argument('--base-ch', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--seed', type=int, default=20260914)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--sample-every', type=int, default=5000)
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--val-batch', type=int, default=8)
    p.add_argument('--backend', choices=('nccl', 'gloo'), default='nccl')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--extend-run', action='store_true',
                   help='With --resume, increase target steps and audit the new schedule / augmentation')
    p.add_argument('--adopt-all-data', action='store_true',
                   help='With --resume, merge the proven parent splits into training and disable validation')
    p.add_argument('--contrast-range', type=float, nargs=2, default=(.9, 1.1), metavar=('LOW', 'HIGH'),
                   help='Foreground contrast multiplier range, applied after existing augmentation')
    p.add_argument('--contrast-gate', type=float, default=.05,
                   help='Soft foreground gate x / (x + value), preserving zero background')
    p.add_argument('--stop-after', type=int, help='Checkpoint and pause after N additional steps; schedule stays fixed')
    args = p.parse_args()
    if args.micro_batch is None:
        args.micro_batch = args.local_batch
    for name in ('steps', 'local_batch', 'warmup', 'save_every', 'sample_every', 'log_every', 'val_batch'):
        if getattr(args, name) < 1:
            p.error(f'{name} must be positive')
    if args.stop_after is not None and args.stop_after < 1:
        p.error('stop-after must be positive')
    if not 1 <= args.micro_batch <= args.local_batch:
        p.error('micro-batch must be between 1 and local-batch')
    if args.extend_run and not args.resume:
        p.error('--extend-run requires --resume')
    if args.adopt_all_data and (not args.resume or args.extend_run):
        p.error('--adopt-all-data requires --resume and cannot be combined with --extend-run')
    if (not all(math.isfinite(v) for v in args.contrast_range)
            or not .5 <= args.contrast_range[0] <= args.contrast_range[1] <= 1.5):
        p.error('contrast-range must satisfy 0.5 <= LOW <= HIGH <= 1.5')
    if not math.isfinite(args.contrast_gate) or args.contrast_gate <= 0:
        p.error('contrast-gate must be finite and positive')
    args.data, args.out = args.data.resolve(), args.out.resolve()
    rank, world, local = (int(os.environ.get(k, default)) for k, default in
                          (('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')))
    run_lock = acquire_run_lock(args.out) if rank == 0 else None
    if rank == 0:
        if (args.out / 'completed.json').exists() and not args.extend_run:
            raise FileExistsError('Training already complete')
        if (args.out / 'config.json').exists() and not args.resume:
            raise FileExistsError('Existing run; use --resume')
    torch.set_num_threads(2)
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if world > 1:
        options = dict(timeout=timedelta(minutes=30))
        if args.backend == 'nccl':
            options['device_id'] = device
        dist.init_process_group(args.backend, **options)
    barrier = lambda: dist.barrier() if world > 1 else None
    manifest, arrays, weights, val_indices = load_data(args.data, verify_hashes=(rank == 0))
    all_training = manifest['dataset'] == 'rodent_native192_png_all'
    barrier()
    torch.manual_seed(args.seed)
    net = StrongPrior(base_ch=args.base_ch).to(device)
    net.img_resolution = 192
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.)
    ema = copy.deepcopy(net).eval().requires_grad_(False) if rank == 0 else None
    manifest_hash = sha256(args.data / 'manifest.json')
    source_paths = [Path(__file__), HERE / 'train_hr.py', HERE / 'prepare_png.py',
                    V2 / 'model_v2.py', V2 / 'tiny_unet_v2.py', V2 / 'run_guard.py', CORE / 'model.py']
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(world_size=world, global_batch=world * args.local_batch, model_config=net.config,
                  image_shape=[1, 192, 192], parameters=sum(x.numel() for x in net.parameters()),
                  manifest_sha256=manifest_hash, image_counts=manifest['image_counts'],
                  subject_counts=manifest['subject_counts'], initialization='random; no old checkpoint',
                  precision='FP32 EDM arithmetic and weights; BF16 autocast in denoiser',
                  validation_indices=val_indices,
                  training_mode='all_data_no_holdout' if all_training else 'heldout_validation',
                  selection=('current EMA at every checkpoint; final target-step EMA' if all_training else
                             'lowest EMA validation loss, group-equal mean, fixed noise'),
                  sampling='equal probability per training subject_group, uniform images within group, with replacement',
                  sampling_source_probability=sampling_source_probabilities(manifest),
                  test_usage=('No heldout or monitor images; all parent images participate in training' if all_training else
                              'manifest, SHA256 and array headers only; no test pixels in training/model selection'),
                  torch=torch.__version__, python=sys.version, executable=sys.executable,
                  physical_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu=torch.cuda.get_device_name(),
                  augmentation=augmentation_config(args.contrast_range, args.contrast_gate),
                  source_sha256={str(path): sha256(path) for path in source_paths})
    step0, best, best_step, elapsed_before = 0, float('inf'), 0, 0.
    lr_schedule = original_schedule(config)
    config['lr_schedule'] = lr_schedule
    previous = None
    adoption_proof = None
    sample_counts = np.zeros(len(arrays['train']), dtype=np.int64)
    sampling_start_step = 0
    ckpt = None
    if args.resume:
        previous = json.loads((args.out / 'config.json').read_text())
        for key in ('extension_history', 'data_adoption_history'):
            if key in previous:
                config[key] = previous[key]
        validate_resume_config(previous, config, args.extend_run, args.adopt_all_data)
        if args.adopt_all_data:
            adoption_proof = verify_all_data_adoption(previous, manifest)
        ckpt = torch.load(args.out / 'latest.pt', map_location='cpu', weights_only=False)
        validate_checkpoint_manifest(ckpt, previous, config, args.adopt_all_data,
                                     args.out / 'latest.pt' if rank == 0 else None)
        if ckpt['model_config'] != net.config:
            raise ValueError('Resume model mismatch')
        if ckpt['global_batch'] != world * args.local_batch or len(ckpt['rng_states']) != world:
            raise ValueError('Resume requires same batch and GPU count')
        lr_schedule = previous.get('lr_schedule', original_schedule(previous))
        check_checkpoint_schedule(ckpt, lr_schedule)
        if args.extend_run:
            lr_schedule = extension_schedule(previous, ckpt, args.steps)
        config['lr_schedule'] = lr_schedule
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if rank == 0:
            ema.load_state_dict(ckpt['ema'])
        step0, best, best_step = ckpt['step'], ckpt['best_val'], ckpt['best_step']
        if not args.adopt_all_data and 'sampling_counts_by_rank' in ckpt:
            counts = ckpt['sampling_counts_by_rank']
            if len(counts) != world or np.asarray(counts[rank]).shape != sample_counts.shape:
                raise ValueError('Checkpoint sampling counters do not match the data or GPU count')
            sample_counts = np.array(counts[rank], dtype=np.int64, copy=True)
            if (sample_counts < 0).any():
                raise ValueError('Negative sampling counts in checkpoint')
            sampling_start_step = ckpt['sampling_coverage_start_step']
            if (sampling_start_step > step0 or sample_counts.sum() !=
                    (step0 - sampling_start_step) * args.local_batch):
                raise ValueError('Checkpoint sampling coverage does not match elapsed training steps')
        else:
            if all_training and not args.adopt_all_data and ckpt['manifest_sha256'] == manifest_hash:
                raise ValueError('All-data checkpoint is missing its persistent sampling counters')
            sampling_start_step = step0
        config['sampling_coverage_start_step'] = sampling_start_step
        elapsed_before = ckpt['elapsed_sec']
        if step0 > args.steps:
            raise ValueError('Checkpoint exceeds requested target')
        if rank == 0 and not all_training:
            # latest.pt and the public best EMA are separate atomic writes.
            # Repair an interruption between those writes before continuing.
            if best_step == step0:
                light = light_checkpoint(ckpt)
                atomic_torch(args.out / 'model_ema.pt', light)
            else:
                saved_best = torch.load(args.out / 'model_ema.pt', map_location='cpu', weights_only=False)
                if saved_best['step'] != best_step or saved_best['manifest_sha256'] != manifest_hash:
                    raise ValueError('Best checkpoint is missing or inconsistent with latest.pt')
                del saved_best
    if all_training:
        best, best_step = None, None
    model = DDP(net, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True) if world > 1 else net
    torch.manual_seed(args.seed + 1009 * rank)
    if ckpt is not None:
        torch.set_rng_state(ckpt['rng_states'][rank]['cpu'])
        torch.cuda.set_rng_state(ckpt['rng_states'][rank]['cuda'], device)
        del ckpt
    if rank == 0:
        if args.resume:
            if args.extend_run:
                archive = archive_extension(args.out, previous, config, step0, source_paths)
                preview = augmentation_preview(arrays['train'], weights, device, args, args.out)
                for name in ('augmentation_preview.json', 'augmentation_preview.png'):
                    shutil.copyfile(args.out / name, archive / f'new_{name}')
                print(json.dumps(dict(event='augmentation_preview', **preview)), flush=True)
            if args.adopt_all_data:
                archive_data_adoption(args.out, previous, config, step0, source_paths, adoption_proof)
            if all_training:
                # Repair the public EMA after migration or any interruption
                # between latest.pt and model_ema.pt atomic writes.
                atomic_torch(args.out / 'model_ema.pt', light_checkpoint(dict(
                    step=step0, model_config=net.config, ema=ema.state_dict(), img_resolution=192,
                    manifest_sha256=manifest_hash), all_training=True))
            log_json(args.out / 'resume_events.jsonl', dict(event='resume', step=step0, started_at=time.time(),
                                                           target_steps=args.steps, extended=args.extend_run,
                                                           adopted_all_data=args.adopt_all_data,
                                                           lr_schedule=lr_schedule))
        else:
            atomic_json(args.out / 'config.json', config)
            snapshot = args.out / 'source_snapshot'
            snapshot.mkdir()
            for path in source_paths:
                shutil.copyfile(path, snapshot / path.name)
            with torch.random.fork_rng(devices=[local]):
                ids = torch.multinomial(weights, 16, replacement=True).numpy()
                example, _ = augment_training(tensor_batch(arrays['train'], ids, device), args)
                example = example * 2 - 1
                save_grid(example, args.out / 'training_examples.png')
                if not all_training:
                    save_grid(tensor_batch(arrays['val'], val_indices[:16], device) * 2 - 1, args.out / 'validation_examples.png')
            augmentation_preview(arrays['train'], weights, device, args, args.out)
            if not all_training:
                baseline = validation(ema, arrays['val'], manifest['records']['val'], val_indices, args.val_batch, device)
                atomic_json(args.out / 'baseline.json', dict(step=0, val_loss=baseline))
        atomic_json(args.out / 'process.json', dict(pid=os.getpid(), rank=rank, started_at=time.time(), argv=sys.argv,
                                                   physical_gpus=os.environ.get('CUDA_VISIBLE_DEVICES')))
        print(json.dumps(dict(event='start', step=step0, target_steps=args.steps, local_batch=args.local_batch,
                              global_batch=world * args.local_batch, image_counts=manifest['image_counts'])), flush=True)
    barrier()
    start, losses, contrast_clips = time.monotonic(), [], []
    last_report_time, last_report_step = start, step0
    last_step = min(args.steps, step0 + args.stop_after) if args.stop_after else args.steps
    for step in range(step0 + 1, last_step + 1):
        model.train()
        ids = torch.multinomial(weights, args.local_batch, replacement=True).numpy()
        np.add.at(sample_counts, ids, 1)
        x, aug_stats = augment_training(tensor_batch(arrays['train'], ids, device), args)
        x = x * 2 - 1
        contrast_clips.append(aug_stats['clip_fraction'].detach())
        sigma = (torch.randn(len(x), 1, 1, 1, device=device) * 1.2 - 1.2).exp()
        weight = (sigma.square() + net.sigma_data ** 2) / (sigma * net.sigma_data).square()
        optimizer.zero_grad(set_to_none=True)
        noisy = x + sigma * torch.randn_like(x)
        lr = scheduled_lr(step, lr_schedule)
        for group in optimizer.param_groups:
            group['lr'] = lr
        loss = microbatch_loss_backward(model, noisy, x, sigma, weight, args.micro_batch, world > 1)
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if rank == 0:
            decay = min(.9995, (1 + step) / (10 + step))
            with torch.no_grad():
                for ep, np_ in zip(ema.parameters(), net.parameters()):
                    ep.lerp_(np_, 1 - decay)
        losses.append(loss.detach())
        if step % args.log_every == 0 or step == last_step or step == step0 + 1:
            report_values = torch.stack((torch.stack(losses).mean(), torch.stack(contrast_clips).mean()))
            if world > 1:
                dist.all_reduce(report_values)
                report_values /= world
            mean_loss, mean_contrast_clip = report_values
            now = time.monotonic()
            elapsed = now - start
            recent_rate = ((step - last_report_step) / (now - last_report_time)
                           if last_report_step > step0 else None)
            if rank == 0:
                row = dict(step=step, loss=float(mean_loss), lr=lr, grad_norm=float(norm),
                           contrast_clip_fraction=float(mean_contrast_clip),
                           elapsed_sec=elapsed_before + elapsed, steps_per_sec=(step - step0) / elapsed,
                           recent_steps_per_sec=recent_rate,
                           images_seen=step * world * args.local_batch,
                           eta_hours=(args.steps - step) / recent_rate / 3600 if recent_rate else None,
                           max_memory_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                           max_memory_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)
                log_json(args.out / 'train_metrics.jsonl', row)
                atomic_json(args.out / 'status.json', dict(status='training', target_steps=args.steps, **row))
            losses.clear()
            contrast_clips.clear()
            last_report_time, last_report_step = now, step
        if step % args.save_every == 0 or step == last_step:
            # Release training activations before EMA validation and sampling.
            del x, sigma, weight, noisy, loss
            optimizer.zero_grad(set_to_none=True)
            barrier()
            rng = dict(cpu=torch.get_rng_state().cpu(), cuda=torch.cuda.get_rng_state(device).cpu(),
                       sampling_counts=sample_counts.copy())
            states = [None] * world
            if world > 1:
                dist.all_gather_object(states, rng)
            else:
                states = [rng]
            sampling_counts_by_rank = [state.pop('sampling_counts') for state in states]
            if rank == 0:
                val_loss = (None if all_training else validation(
                    ema, arrays['val'], manifest['records']['val'], val_indices, args.val_batch, device))
                improved = not all_training and val_loss < best
                if improved:
                    best, best_step = val_loss, step
                state = dict(step=step, model_config=net.config, model=net.state_dict(), ema=ema.state_dict(),
                             optimizer=optimizer.state_dict(), rng_states=states, global_batch=world * args.local_batch,
                             val_loss=val_loss, best_val=best, best_step=best_step, manifest_sha256=manifest_hash,
                             lr_schedule=lr_schedule, augmentation=config['augmentation'],
                             training_mode=config['training_mode'], selection=config['selection'],
                             micro_batch=args.micro_batch,
                             sampling_counts_by_rank=sampling_counts_by_rank,
                             sampling_coverage_start_step=sampling_start_step,
                             img_resolution=192, elapsed_sec=elapsed_before + time.monotonic() - start)
                atomic_torch(args.out / 'latest.pt', state)
                light = light_checkpoint(state, all_training=all_training)
                if improved or all_training:
                    atomic_torch(args.out / 'model_ema.pt', light)
                if step % 5000 == 0 or step == args.steps:
                    atomic_torch(args.out / f'ema_{step:06d}.pt', light)
                aggregate_counts = np.stack(sampling_counts_by_rank).sum(axis=0)
                atomic_json(args.out / 'sampling_coverage.json', dict(
                    step=step, counted_since_step=sampling_start_step,
                    images=len(aggregate_counts), sampled_images=int((aggregate_counts > 0).sum()),
                    all_images_sampled=bool((aggregate_counts > 0).all()),
                    minimum_samples=int(aggregate_counts.min()), maximum_samples=int(aggregate_counts.max()),
                    total_samples=int(aggregate_counts.sum()), manifest_sha256=manifest_hash))
                if all_training:
                    log_json(args.out / 'checkpoint_metrics.jsonl', dict(
                        event='checkpoint', step=step, selection=config['selection'],
                        sampled_images=int((aggregate_counts > 0).sum()), training_images=len(aggregate_counts)))
                else:
                    log_json(args.out / 'val_metrics.jsonl', dict(event='checkpoint', step=step, val_loss=val_loss,
                                                                 best_val=best, best_step=best_step))
                if step % args.sample_every == 0 or step == args.steps:
                    with torch.random.fork_rng(devices=[local]):
                        save_grid(sample192(ema), args.out / f'samples_{step:06d}.png')
            barrier()
    if rank == 0:
        completed = last_step == args.steps
        if completed and not (args.out / f'samples_{args.steps:06d}.png').exists():
            with torch.random.fork_rng(devices=[local]):
                save_grid(sample192(ema), args.out / f'samples_{args.steps:06d}.png')
        status = dict(status='completed' if completed else 'paused', step=last_step, target_steps=args.steps,
                      best_val=best, best_step=best_step, training_mode=config['training_mode'],
                      selection=config['selection'], elapsed_sec=elapsed_before + time.monotonic() - start)
        atomic_json(args.out / 'status.json', status)
        if completed:
            atomic_json(args.out / 'completed.json', status)
        print(json.dumps(status), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
