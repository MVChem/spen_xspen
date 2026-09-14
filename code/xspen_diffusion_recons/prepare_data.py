"""Local IXI T2/PD, deterministic subject split and physical 210-mm views."""
import argparse
import concurrent.futures
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from utils import HERE, sha256, write_json

from paths import IXI_SOURCE

SOURCE = IXI_SOURCE

def process_volume(job):
    path, split, dest = job
    path = Path(path)
    subject = path.name.split('-')[0]
    modality = path.name.split('-')[-1].split('.')[0]
    key = path.name.replace('.nii.gz', '')
    image = nib.as_closest_canonical(nib.load(path))
    vol = image.get_fdata(dtype=np.float32)
    if vol.ndim != 3 or not np.isfinite(vol).all():
        raise ValueError(f'Invalid source volume: {path}')
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
    scale = float(np.percentile(vol[vol > 0], 99.5))
    if scale <= 0:
        raise ValueError(f'Empty source volume: {path}')
    vol = np.clip(vol/scale, 0, 1)
    # The center of the acquired head foreground; no atlas or test-data fitting.
    coords = np.where(vol > .12)
    center = np.array([(np.percentile(c, 2)+np.percentile(c, 98))/2 for c in coords])
    records, slices = [], []
    for view, axis, inplane in [('axial', 2, (1, 0)), ('sagittal', 0, (2, 1))]:
        low, high = np.percentile(coords[axis], [5, 95]).astype(int)
        # Every second acquired plane: avoid a large count of near-identical slices.
        for index in range(low, high+1, 2):
            plane = np.take(vol, index, axis=axis)
            remaining = [a for a in range(3) if a != axis]
            plane = np.transpose(plane, [remaining.index(a) for a in inplane])
            pix = spacing[list(inplane)]
            sigma = np.maximum(0., .5*np.sqrt(np.maximum((210/128/pix)**2-1, 0)))
            plane = gaussian_filter(plane, sigma=sigma)
            q = (np.arange(128)+.5-64)*210/128
            yy, xx = np.meshgrid(q/pix[0]+center[inplane[0]], q/pix[1]+center[inplane[1]], indexing='ij')
            out = map_coordinates(plane, [yy, xx], order=1, mode='constant', cval=0)
            if np.mean(out > .08) < .12:
                continue
            # Anatomical superior/anterior at the top for both views.
            out = np.ascontiguousarray(np.flipud(out).clip(0, 1)*65535, dtype=np.uint16)
            records.append(dict(key=f'{key}:{view}:{index}', subject=subject,
                                modality=modality, view=view, source=str(path), slice_index=int(index),
                                slice_sha256=hashlib.sha256(out.tobytes()).hexdigest()))
            slices.append(out)
    if not slices:
        raise ValueError(f'No usable slices: {path}')
    target = Path(dest)/'volumes'/f'{key}.npy'
    np.save(target, np.stack(slices))
    return dict(path=str(path), sha256=sha256(path), subject=subject, modality=modality,
                split=split, original_shape=list(image.shape), spacing_mm=spacing.tolist(),
                volume_scale=scale, cache=str(target), records=records)

def main():
    global SOURCE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=SOURCE, help='External IXI NIfTI root (or XSPEN_IXI_SOURCE)')
    p.add_argument('--out', type=Path, default=HERE/'data/ixi128')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    SOURCE = args.source.expanduser().resolve()
    if not SOURCE.is_dir():
        p.error(f'IXI source directory does not exist: {SOURCE}')
    if (args.out/'manifest.json').exists():
        print('Existing prepared dataset; reuse immutable manifest.', flush=True)
        return
    (args.out/'volumes').mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in SOURCE.rglob('*.nii.gz') if p.name.endswith(('-T2.nii.gz', '-PD.nii.gz')))
    subjects = sorted({p.name.split('-')[0] for p in paths})
    np.random.default_rng(20260909).shuffle(subjects)
    n = len(subjects); nval = round(n*.1)
    split_ids = dict(train=sorted(subjects[2*nval:]), val=sorted(subjects[nval:2*nval]), test=sorted(subjects[:nval]))
    by_subject = {s: split for split, ids in split_ids.items() for s in ids}
    if n < 100:
        raise ValueError('Unexpectedly incomplete local IXI source')
    jobs = [(str(path), by_subject[path.name.split('-')[0]], str(args.out)) for path in paths]
    outputs = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, out in enumerate(pool.map(process_volume, jobs)):
            outputs.append(out)
            if (i+1) % 25 == 0:
                print(json.dumps(dict(event='prepare', volumes=i+1, total=len(jobs))), flush=True)
    records = {s: [] for s in split_ids}
    slice_split, source_split = {}, {}
    for out in outputs:
        sp = out['split']
        if out['sha256'] in source_split and source_split[out['sha256']] != sp:
            raise ValueError('Cross-split source duplicate')
        source_split[out['sha256']] = sp
        for row in out['records']:
            digest = row['slice_sha256']
            if digest in slice_split and slice_split[digest] != sp:
                raise ValueError('Cross-split image duplicate')
            slice_split[digest] = sp
        records[sp].extend(out['records'])
    for sp in split_ids:
        array = np.lib.format.open_memmap(args.out/f'{sp}.npy', mode='w+', dtype=np.uint16,
                                          shape=(len(records[sp]), 128, 128))
        start = 0
        for out in outputs:
            if out['split'] != sp:
                continue
            a = np.load(out['cache'], mmap_mode='r')
            array[start:start+len(a)] = a
            start += len(a)
        array.flush()
        del array
    manifest = dict(seed=20260909, dataset='IXI', source_root=str(SOURCE),
                    source_url='https://brain-development.org/ixi-dataset/', license='CC BY-SA 3.0',
                    resolution=128, fov_mm=[210, 210], subjects=split_ids,
                    counts={sp: len(rr) for sp, rr in records.items()}, records=records,
                    volumes=[{k: v for k, v in out.items() if k != 'records'} for out in outputs],
                    modality_probability={'T2': .8, 'PD': .2},
                    array_sha256={s: sha256(args.out/f'{s}.npy') for s in split_ids},
                    note='Actual human structural magnitude data, not xSPEN raw or clean paired xSPEN GT. Same subject and all views/modalities remain in one split.')
    write_json(args.out/'manifest.json', manifest)
    print(json.dumps(dict(event='data_complete', counts=manifest['counts'], subjects={s: len(ids) for s, ids in split_ids.items()})), flush=True)

if __name__ == '__main__':
    main()
