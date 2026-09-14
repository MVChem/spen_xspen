"""Re-extract 192-pixel ds005236 acquired AP slices using the original animal split."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parent / 'prior96'
sys.path.insert(0, str(HERE.parent / 'core'))
sys.path.insert(0, str(V2))
from project_paths import CORE, RUNS, PRIOR96_DATA, PRIOR192_DATA, MOUSE_RAW
from prepare_data import physical_slice, sha256


def resolve_source(record):
    path = Path(record['source'])
    if path.exists():
        return path.resolve()
    # The workspace was reorganized after the original manifest was written.
    relative = path.parts[path.parts.index('mouse_raw') + 1:]
    return (MOUSE_RAW).joinpath(*relative).resolve(strict=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=RUNS / 'prepared/prior192')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'manifest.json').exists():
        raise FileExistsError('Completed HR dataset already exists')
    old_path = PRIOR96_DATA / 'manifest.json'
    old = json.loads(old_path.read_text())
    records = {p: [] for p in ('train', 'val', 'test')}
    arrays = {p: [] for p in records}
    volume_cache, source_hashes, volume_info = {}, {}, {}
    for part in records:
        for old_index, record in enumerate(old['records'][part]):
            if record.get('dataset') != 'ds005236':
                continue
            assert record['species'] == 'mouse'
            assert old['assignments'][record['subject']]['part'] == part
            path = resolve_source(record)
            if path not in volume_cache:
                digest = sha256(path)
                if digest != record['source_sha256']:
                    raise ValueError(f'Source hash changed: {path}')
                image = nib.load(path)
                volume = image.get_fdata(dtype=np.float32).squeeze()
                codes = nib.aff2axcodes(image.affine)
                spacing = tuple(map(float, image.header.get_zooms()[:3]))
                ap = next(i for i, c in enumerate(codes) if c in ('A', 'P'))
                si = next(i for i, c in enumerate(codes) if c in ('S', 'I'))
                ri = next(i for i, c in enumerate(codes) if c in ('R', 'L'))
                assert volume.ndim == 3 and volume.shape[si] == volume.shape[ri] == 256
                assert max(spacing[si], spacing[ri]) < 0.06
                volume_cache[path] = volume, codes, spacing, ap, si, ri
                source_hashes[str(path)] = digest
                volume_info[str(path)] = dict(source=str(path), source_sha256=digest,
                    subject=record['subject'], split=part, shape=list(image.shape),
                    native_codes=list(codes), voxel_spacing_mm=spacing,
                    acquired_slice_axis=ap, plane_axes=[si, ri])
            volume, codes, spacing, ap, si, ri = volume_cache[path]
            assert record['slice_axis'] == ap
            remaining = [i for i in range(3) if i != ap]
            plane = np.take(volume, record['slice_index'], axis=ap)
            plane = plane.transpose(remaining.index(si), remaining.index(ri))
            if codes[si] == 'S':
                plane = plane[::-1]
            if codes[ri] == 'R':
                plane = plane[:, ::-1]
            result = physical_slice(plane, (spacing[si], spacing[ri]), record['fov_mm'], size=192)
            assert result.shape == (192, 192) and np.isfinite(result).all()
            new_record = dict(record, source=str(path), original_source=record['source'],
                source_manifest_index=old_index, hr_shape=[192, 192],
                native_plane_shape=list(plane.shape), native_codes=list(codes),
                native_plane_spacing_mm=[spacing[si], spacing[ri]],
                hr_pixel_spacing_mm=[record['fov_mm'] / 192] * 2)
            new_record.pop('sample_weight', None)
            records[part].append(new_record)
            arrays[part].append(np.round(result * 65535).astype(np.uint16))
        print(json.dumps(dict(event='prepared', split=part, images=len(arrays[part]))), flush=True)
    subjects = {p: sorted({r['subject'] for r in rows}) for p, rows in records.items()}
    groups = {p: {r['split_group'] for r in rows} for p, rows in records.items()}
    hashes = {p: {hashlib.sha256(x.tobytes()).hexdigest() for x in images} for p, images in arrays.items()}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert not set(subjects[a]) & set(subjects[b]), (a, b, 'animal overlap')
        assert not groups[a] & groups[b], (a, b, 'split-group overlap')
        assert not hashes[a] & hashes[b], (a, b, 'duplicate-image overlap')
    counts = Counter((r['subject'], r['view']) for r in records['train'])
    view_probability = {'mouse_fov16': 0.6, 'mouse_fov24': 0.4}
    for record in records['train']:
        record['sample_weight'] = view_probability[record['view']] / len(subjects['train']) / counts[(record['subject'], record['view'])]
    array_hashes = {}
    for part, images in arrays.items():
        tensor = np.stack(images)
        np.save(args.out / f'{part}.npy', tensor)
        array_hashes[part] = sha256(args.out / f'{part}.npy')
    manifest = dict(version=1, size=192, dataset='ds005236', species='mouse',
        records=records, subjects=subjects, split_groups={p: sorted(v) for p, v in groups.items()},
        assignments={s: old['assignments'][s] for ss in subjects.values() for s in ss},
        counts={p: dict(Counter(r['view'] for r in rows)) for p, rows in records.items()},
        image_counts={p: len(rows) for p, rows in records.items()},
        subject_counts={p: len(ss) for p, ss in subjects.items()},
        volume_info=list(volume_info.values()), volume_sha256=source_hashes,
        source_manifest=str(old_path), source_manifest_sha256=sha256(old_path),
        source_code_sha256={str(Path(__file__)): sha256(Path(__file__)), str(V2/'prepare_data.py'): sha256(V2/'prepare_data.py')},
        train_probability=view_probability, npy_sha256=array_hashes,
        normalization='physical_slice: nonnegative, per-image q0.995 scaling, clip [0,1], uint16; model [-1,1]',
        audit=dict(animal_overlap=False, split_group_overlap=False, cross_split_byte_duplicates=False, all_source_hashes_verified=True),
        note='Native acquired AP planes, SI x RI orientation. Re-extracted from 256 x 256 NIfTI; never upsampled from the saved 96-pixel arrays. Original accepted keys/splits retained without HR-dependent rejection. FOV 16/24 mm; grid spacing 83.33/125 um, not the acquired 58.6 um resolution.')
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(event='complete', images=manifest['image_counts'], subjects=manifest['subject_counts'], manifest_sha256=sha256(args.out/'manifest.json'))), flush=True)


if __name__ == '__main__':
    main()
