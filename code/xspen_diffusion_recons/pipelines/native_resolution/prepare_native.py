"""Prepare physical xSPEN image grids from IXI, retaining the frozen subject split.

Each original volume is decoded once for all requested profiles/grids. Intermediate
shards are atomic and keyed by the complete preparation recipe and source identity.
The final manifest is published only after all arrays and leakage checks succeed.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
VERSION = 1


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    os.replace(temp, path)


def slab_plane(volume, axis, center, thickness_mm, spacing_mm):
    """Exact box integration of piecewise-constant source voxels along slice axis."""
    half = thickness_mm / spacing_mm / 2
    lo, hi = center - half, center + half
    indices = np.arange(max(0, int(np.floor(lo - .5))),
                        min(volume.shape[axis], int(np.ceil(hi + .5)) + 1))
    weights = np.maximum(0., np.minimum(indices + .5, hi) - np.maximum(indices - .5, lo))
    # Outside the acquired volume is zero; preserve full physical slab denominator.
    weights = (weights / (2 * half)).astype(np.float32)
    return np.tensordot(np.take(volume, indices, axis=axis), weights, axes=(axis, 0))


def resample_plane(plane, source_pixel_mm, center, fov_mm, shape):
    target_pixel_mm = np.asarray(fov_mm) / np.asarray(shape)
    sigma = .5 * np.sqrt(np.maximum((target_pixel_mm / source_pixel_mm) ** 2 - 1, 0))
    filtered = gaussian_filter(plane, sigma=sigma, mode='constant', cval=0)
    coords = [((np.arange(n) + .5 - n / 2) * f / n) / s + c
              for n, f, s, c in zip(shape, fov_mm, source_pixel_mm, center)]
    yy, xx = np.meshgrid(*coords, indexing='ij')
    return map_coordinates(filtered, [yy, xx], order=1, mode='constant', cval=0)


def process_volume(job):
    source, targets, shard_root, recipe_id, max_slices = job
    path = Path(source['path'])
    key = path.name.removesuffix('.nii.gz')
    meta_path = Path(shard_root) / f'{key}.json'
    source_stat = path.stat()
    identity = dict(path=str(path), size=source_stat.st_size, mtime_ns=source_stat.st_mtime_ns,
                    expected_sha256=source['sha256'])
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['recipe_id'] == recipe_id and meta['source_identity'] == identity:
            if all(Path(v['cache']).exists() and
                   list(np.load(v['cache'], mmap_mode='r').shape) == v['array_shape']
                   for v in meta['targets'].values()):
                return dict(path=str(meta_path), reused=True)
    actual_hash = digest_file(path)
    if actual_hash != source['sha256']:
        raise ValueError(f'Source hash changed since frozen IXI manifest: {path}')
    image = nib.as_closest_canonical(nib.load(path))
    volume = image.get_fdata(dtype=np.float32)
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError(f'Invalid source image: {path}')
    positive = volume[volume > 0]
    if not len(positive):
        raise ValueError(f'Empty source: {path}')
    scale = float(np.percentile(positive, 99.5))
    del positive
    volume = np.clip(volume / scale, 0, 1)
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
    foreground = np.where(volume > .12)
    if not len(foreground[0]):
        raise ValueError(f'No foreground: {path}')
    center = np.array([(np.percentile(c, 2) + np.percentile(c, 98)) / 2 for c in foreground])
    stacks = {t['id']: [] for t in targets}
    records = {t['id']: [] for t in targets}
    for view, axis, inplane in [('axial', 2, (1, 0)), ('sagittal', 0, (2, 1))]:
        low, high = np.percentile(foreground[axis], [5, 95]).astype(int)
        count = min(max_slices, max(1, (high - low) // 2 + 1))
        indices = np.unique(np.linspace(low, high, count).round().astype(int))
        remaining = [a for a in range(3) if a != axis]
        order = [remaining.index(a) for a in inplane]
        for index in indices:
            slabs = {}
            for target in targets:
                if view not in target['views']:
                    continue
                thickness = target['thickness_mm']
                if thickness not in slabs:
                    slabs[thickness] = np.transpose(
                        slab_plane(volume, axis, int(index), thickness, spacing[axis]), order)
                out = resample_plane(slabs[thickness], spacing[list(inplane)],
                                     center[list(inplane)], target['fov_mm'], target['shape'])
                if np.mean(out > .08) < .12:
                    continue
                # Superior/anterior at the top, matching existing IXI preparation.
                out = np.ascontiguousarray(np.flipud(out).clip(0, 1) * 65535, dtype=np.uint16)
                records[target['id']].append(dict(
                    key=f'{key}:{view}:{index}', subject=source['subject'],
                    modality=source['modality'], view=view, source=str(path),
                    slice_index=int(index), thickness_mm=thickness,
                    slice_sha256=hashlib.sha256(out.tobytes()).hexdigest()))
                stacks[target['id']].append(out)
    meta = dict(recipe_id=recipe_id, source_identity=identity, source_sha256=actual_hash,
                subject=source['subject'], modality=source['modality'], split=source['split'],
                canonical_shape=list(image.shape), source_spacing_mm=spacing.tolist(),
                canonical_affine=image.affine.tolist(), volume_scale=scale,
                foreground_center_voxel=center.tolist(), targets={})
    for target in targets:
        tid = target['id']
        if not stacks[tid]:
            raise ValueError(f'No usable slices for {key} {tid}')
        array = np.stack(stacks[tid])
        dest = Path(shard_root) / tid / f'{key}.npy'
        temp = dest.with_name(dest.name + f'.{os.getpid()}.tmp')
        with temp.open('wb') as f:
            np.save(f, array)
        os.replace(temp, dest)
        meta['targets'][tid] = dict(cache=str(dest), array_shape=list(array.shape),
                                   array_sha256=digest_file(dest), records=records[tid])
    atomic_json(meta_path, meta)
    return dict(path=str(meta_path), reused=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocols', type=Path, default=HERE / 'protocols.json')
    p.add_argument('--source-manifest', type=Path, default=PROJECT / 'data/ixi128/manifest.json')
    p.add_argument('--out', type=Path, default=HERE / 'data')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--max-slices-per-view', type=int, default=24)
    p.add_argument('--pilot-subjects-per-split', type=int, default=0)
    args = p.parse_args()
    if args.workers < 1 or args.max_slices_per_view < 1 or args.pilot_subjects_per_split < 0:
        p.error('workers/slices must be positive; pilot subjects must be nonnegative')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(args.source_manifest.read_text())
    protocols = json.loads(args.protocols.read_text())
    targets = []
    for profile in protocols['profiles']:
        for grid in profile['grids']:
            targets.append(dict(id=profile['id'] + '_' + grid['id'], profile_id=profile['id'],
                                grid_id=grid['id'], shape=grid['shape'], pixel_mm=grid['pixel_mm'],
                                acquired_native=grid['acquired_native'], fov_mm=profile['fov_mm'],
                                thickness_mm=profile['thickness_mm'], views=profile.get('views', ['axial', 'sagittal']),
                                native_shape=profile['native_shape'], scan_ids=profile['scan_ids']))
    subjects = source_manifest['subjects']
    if args.pilot_subjects_per_split:
        subjects = {sp: ids[:args.pilot_subjects_per_split] for sp, ids in subjects.items()}
    by_subject = {}
    for sp, ids in subjects.items():
        for sid in ids:
            if sid in by_subject:
                raise ValueError(f'Cross-split subject: {sid}')
            by_subject[sid] = sp
    volumes = [v for v in source_manifest['volumes'] if v['subject'] in by_subject]
    assert all(by_subject[v['subject']] == v['split'] for v in volumes)
    recipe = dict(version=VERSION, source_manifest_sha256=digest_file(args.source_manifest),
                  preparation_script_sha256=digest_file(__file__), targets=targets,
                  max_slices_per_view=args.max_slices_per_view, subjects=subjects,
                  pilot=bool(args.pilot_subjects_per_split),
                  normalization='Per-volume positive-voxel p99.5, clip [0,1]',
                  through_plane='Box integral of piecewise-constant source voxels, zero outside acquisition',
                  inplane='Gaussian anti-alias filter then physical-mm bilinear sampling; fixed FOV',
                  foreground='Volume-local >0.12 threshold; p2/p98 center; p5/p95 slice range',
                  numpy_version=np.__version__, nibabel_version=nib.__version__)
    recipe_id = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    previous = args.out / 'preparation_recipe.json'
    if previous.exists() and json.loads(previous.read_text())['recipe_id'] != recipe_id:
        raise ValueError('Output already belongs to a different recipe; use a fresh --out')
    atomic_json(previous, dict(recipe_id=recipe_id, **recipe))
    shard_root = args.out / '_shards'
    for target in targets:
        (shard_root / target['id']).mkdir(parents=True, exist_ok=True)
    jobs = [(v, targets, str(shard_root), recipe_id, args.max_slices_per_view) for v in volumes]
    started = time.time()
    status_path = args.out / 'preparation_status.json'
    completed = []
    reused = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(process_volume, jobs, chunksize=1):
            completed.append(result['path'])
            reused += result['reused']
            status = dict(event='prepare', volumes=len(completed), total=len(jobs), reused=reused,
                          elapsed_seconds=round(time.time() - started, 2), recipe_id=recipe_id)
            atomic_json(status_path, status)
            if len(completed) % 10 == 0 or len(completed) == len(jobs):
                print(json.dumps(status), flush=True)
    # Hold metadata only; concatenate one target/split at a time via memory maps.
    metadata = [json.loads(Path(path).read_text()) for path in completed]
    source_splits = {}
    for meta in metadata:
        digest, split = meta['source_sha256'], meta['split']
        if digest in source_splits and source_splits[digest] != split:
            raise ValueError('Cross-split original-volume duplicate')
        source_splits[digest] = split
    summaries = {}
    for target in targets:
        dest = args.out / target['id']
        dest.mkdir(exist_ok=True)
        if (dest / 'manifest.json').exists():
            done = json.loads((dest / 'manifest.json').read_text())
            if done.get('recipe_id') == recipe_id and all((dest / f'{sp}.npy').exists() for sp in subjects):
                summaries[target['id']] = done['counts']
                print(json.dumps(dict(event='reuse_dataset', dataset=target['id'])), flush=True)
                continue
        records = {sp: [] for sp in subjects}
        slice_splits = {}
        volume_records = []
        for meta in metadata:
            shard = meta['targets'][target['id']]
            sp = meta['split']
            for row in shard['records']:
                digest = row['slice_sha256']
                if digest in slice_splits and slice_splits[digest] != sp:
                    raise ValueError(f'Cross-split image duplicate: {target["id"]}')
                slice_splits[digest] = sp
            records[sp].extend(shard['records'])
            volume_records.append({k: v for k, v in meta.items() if k != 'targets'} |
                                  dict(cache=shard['cache'], array_sha256=shard['array_sha256']))
        array_hashes = {}
        for sp in subjects:
            shape = (len(records[sp]), *target['shape'])
            temp = dest / f'{sp}.npy.tmp'
            array = np.lib.format.open_memmap(temp, mode='w+', dtype=np.uint16, shape=shape)
            offset = 0
            for meta in metadata:
                if meta['split'] != sp:
                    continue
                shard = meta['targets'][target['id']]
                if digest_file(shard['cache']) != shard['array_sha256']:
                    raise ValueError(f'Corrupted prepared shard: {shard["cache"]}')
                data = np.load(shard['cache'], mmap_mode='r')
                array[offset:offset + len(data)] = data
                offset += len(data)
            assert offset == len(records[sp])
            array.flush()
            del array
            os.replace(temp, dest / f'{sp}.npy')
            array_hashes[sp] = digest_file(dest / f'{sp}.npy')
        manifest = dict(dataset='IXI', seed=source_manifest['seed'], recipe_id=recipe_id,
                        **target, resolution=target['shape'], subjects=subjects,
                        counts={sp: len(rows) for sp, rows in records.items()}, records=records,
                        volumes=volume_records, modality_probability=source_manifest['modality_probability'],
                        array_sha256=array_hashes, dtype='uint16', scale=65535,
                        source_root=source_manifest['source_root'],
                        source_url=source_manifest['source_url'], license=source_manifest['license'],
                        source_manifest=str(args.source_manifest.resolve()),
                        source_manifest_sha256=recipe['source_manifest_sha256'],
                        preparation_recipe=str(previous), complete=not recipe['pilot'],
                        note='IXI structural T2/PD magnitude priors with physical slice integration. '
                             'No xSPEN raw pairs, no acquired complex phase, no diffusion-weighted GT. '
                             'All modalities/views/resolutions of each subject retain frozen IXI split.')
        atomic_json(dest / 'manifest.json', manifest)
        summaries[target['id']] = manifest['counts']
        print(json.dumps(dict(event='dataset_complete', dataset=target['id'], counts=manifest['counts'])), flush=True)
    atomic_json(status_path, dict(event='complete', volumes=len(volumes), datasets=summaries,
                                 subjects={sp: len(ids) for sp, ids in subjects.items()}, recipe_id=recipe_id,
                                 elapsed_seconds=round(time.time() - started, 2), pilot=recipe['pilot']))
    print(status_path.read_text(), flush=True)


if __name__ == '__main__':
    main()
