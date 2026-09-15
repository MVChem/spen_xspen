"""Merge the complete native PNG cohort into training, retaining old split provenance.

The user explicitly requested all 10,633 images for diffusion training. This
creates an independent dataset and never modifies the earlier split or pixels.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
PARTS = ('train', 'val', 'test')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def prepare_all(data, out, expected_count=10633):
    data, out = data.resolve(strict=True), out.resolve()
    if out == data or data in out.parents:
        raise ValueError('The all-image dataset must be separate from its parent dataset')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output is not empty: {out}')
    parent_manifest = data / 'manifest.json'
    parent_hash = sha256(parent_manifest)
    parent = json.loads(parent_manifest.read_text())
    if parent['dataset'] != 'rodent_native192_png' or parent['size'] != 192:
        raise ValueError('Expected the original native 192 PNG split dataset')
    source_manifest = Path(parent['source_manifest'])
    if sha256(source_manifest) != parent['source_manifest_sha256']:
        raise ValueError('Original PNG export manifest changed')
    source_records = [json.loads(line) for line in source_manifest.read_text().splitlines()]
    if len(source_records) != expected_count:
        raise ValueError(f'Expected exactly {expected_count} original PNG records')
    arrays, entries, parent_npy_hashes = {}, [], {}
    for part in PARTS:
        path = data / f'{part}.npy'
        parent_npy_hashes[part] = sha256(path)
        if parent_npy_hashes[part] != parent['npy_sha256'][part]:
            raise ValueError(f'Parent {part}.npy changed')
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        records = parent['records'][part]
        if array.dtype != np.uint16 or array.shape != (len(records), 192, 192):
            raise ValueError(f'Parent {part} array shape or dtype is invalid')
        arrays[part] = array
        for index, record in enumerate(records):
            entries.append((record['source_manifest_index'], part, index, record))
    entries.sort(key=lambda item: item[0])
    if [item[0] for item in entries] != list(range(expected_count)):
        raise ValueError('The parent splits do not contain the complete original PNG cohort exactly once')
    for field in ('key', 'png_path', 'png_sha256', 'pixel_sha256', 'source_plane_key'):
        if len({row[field] for _, _, _, row in entries}) != expected_count:
            raise ValueError(f'Duplicate original image identity: {field}')
    groups = Counter(row['subject_group'] for _, _, _, row in entries)
    records = []
    for original_index, old_part, old_index, row in entries:
        source = source_records[original_index]
        for field in ('num', 'filename', 'source_id', 'slice_index', 'subject_group',
                      'source_sha256', 'png_sha256', 'pixel_sha256'):
            if row[field] != source[field]:
                raise ValueError(f'Parent image differs from original PNG provenance: {field}')
        if row['subject'] != row['subject_group'] or row['split_group'] != row['subject_group']:
            raise ValueError('Parent animal grouping is inconsistent')
        if (row['transform']['upsampled'] or row['transform']['added_padding']
                or min(row['transform']['native_plane_shape']) < 192):
            raise ValueError('An image fails the native no-upsampling requirements')
        records.append(dict(row, original_split=old_part, original_split_index=old_index,
            split='train', sample_weight=1 / (len(groups) * groups[row['subject_group']])))
    assignments = {group: dict(value, original_split=value['part'], part='train')
                   for group, value in parent['assignments'].items()}
    if set(assignments) != set(groups):
        raise ValueError('Original animal assignments differ from image records')
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{out.name}.preparing-', dir=out.parent))
    try:
        path = staging / 'train.npy'
        array = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint16,
                                         shape=(expected_count, 192, 192))
        for output_index, (_, old_part, old_index, row) in enumerate(entries):
            pixels = arrays[old_part][old_index]
            if hashlib.sha256(pixels.tobytes()).hexdigest() != row['pixel_sha256']:
                raise ValueError(f'Parent array image pixels changed: {row["key"]}')
            binary = Path(row['png_path']).read_bytes()
            if (binary[:8] != b'\x89PNG\r\n\x1a\n' or binary[24:26] != bytes((16, 0))
                    or hashlib.sha256(binary).hexdigest() != row['png_sha256']):
                raise ValueError(f'Original PNG file or bit depth changed: {row["png_path"]}')
            array[output_index] = pixels
        array.flush()
        del array
        merged = np.load(path, mmap_mode='r', allow_pickle=False)
        for index, row in enumerate(records):
            if hashlib.sha256(merged[index].tobytes()).hexdigest() != row['pixel_sha256']:
                raise ValueError(f'Merged output pixels differ: {row["key"]}')
        del merged
        npy_hash = sha256(path)
        if sha256(parent_manifest) != parent_hash:
            raise ValueError('Parent manifest changed during preparation')
        for part in PARTS:
            if sha256(data / f'{part}.npy') != parent_npy_hashes[part]:
                raise ValueError(f'Parent {part} array changed during preparation')
        probability = defaultdict(float)
        group_probability = defaultdict(float)
        for row in records:
            probability[row['dataset']] += row['sample_weight']
            group_probability[row['subject_group']] += row['sample_weight']
        if not np.allclose(list(group_probability.values()), 1 / len(groups), atol=1e-12):
            raise ValueError('Merged training weights are not equal per animal group')
        subjects = dict(train=sorted(groups), val=[], test=[])
        image_counts = dict(train=expected_count, val=0, test=0)
        subject_counts = dict(train=len(groups), val=0, test=0)
        source_counts = dict(train=dict(Counter(row['dataset'] for row in records)), val={}, test={})
        audit = dict(passed=True, checked_images=expected_count,
            source_png_file_sha256_verified=expected_count,
            parent_array_image_pixel_sha256_verified=expected_count,
            merged_array_image_pixel_sha256_verified=expected_count,
            uint16_pixels_unchanged=True, parent_arrays_unchanged=True, parent_manifest_unchanged=True,
            complete_original_png_membership=True, duplicate_keys=0, duplicate_pixels=0,
            original_export_order_restored=True, no_resampling_or_new_normalization=True,
            no_png_copies=True, upsampled_images=0, added_padding_images=0,
            training_weight_sum=float(sum(row['sample_weight'] for row in records)),
            training_source_probability=dict(sorted(probability.items())))
        manifest = dict(version=1, dataset='rodent_native192_png_all', size=192, species=parent['species'],
            created_at=datetime.now(timezone.utc).isoformat(), records=dict(train=records, val=[], test=[]),
            subjects=subjects, split_groups=subjects, assignments=assignments,
            image_counts=image_counts, subject_counts=subject_counts, source_counts=source_counts,
            species_counts=dict(train=dict(Counter(row['species'] for row in records)), val={}, test={}),
            sequence_counts=dict(train=dict(Counter(row['sequence'] for row in records)), val={}, test={}),
            counts=dict(train=dict(Counter(row['view'] for row in records)), val={}, test={}),
            unique_plane_counts=dict(train=len({row['source_plane_key'] for row in records}), val=0, test=0),
            unique_physical_plane_counts=dict(train=len({row['physical_plane_key'] for row in records}), val=0, test=0),
            train_probability={'native_crop192': 1.0},
            sampling='Uniform probability over all subject_group identities, then uniform accepted image within that group; per-record sample_weight',
            split_design=dict(method='All original PNG images are assigned to train in source_manifest_index order',
                user_request='用户明确要求 10,633 张图像全部用于 diffusion model 训练，取消训练阶段的 holdout。',
                holdout=False, original_split_meaning='Provenance only; original val/test images are now training data'),
            selection_indices={'val': []},
            validation_selection=dict(images=0, subject_counts=0, source_counts={}, method='None: no validation holdout'),
            normalization=parent['normalization'], parent_manifest=str(parent_manifest),
            parent_manifest_sha256=parent_hash, parent_npy_sha256=parent_npy_hashes,
            parent_image_counts=parent['image_counts'], parent_subject_counts=parent['subject_counts'],
            parent_source_code_sha256=parent['source_code_sha256'],
            source_manifest=str(source_manifest), source_manifest_sha256=parent['source_manifest_sha256'],
            source_export=parent['source_export'], source_export_summary=parent['source_export_summary'],
            source_export_provenance_sha256=parent['source_export_provenance_sha256'],
            source_code_sha256={str(Path(__file__).resolve()): sha256(__file__)},
            npy_sha256={'train': npy_hash}, audit=audit,
            note='All images, including former validation/test groups, now train the diffusion prior by explicit user request. Later reconstruction checks using these images are in-sample demonstrations or calibration, not held-out generalization estimates. Original species and native geometry are retained; no old data files are modified.')
        write_json(staging / 'manifest.json', manifest)
        manifest_hash = sha256(staging / 'manifest.json')
        write_json(staging / 'verification.json', dict(audit, manifest_sha256=manifest_hash,
            npy_sha256={'train': npy_hash}, image_counts=image_counts, subject_counts=subject_counts,
            parent_manifest_sha256=parent_hash))
        write_json(staging / 'summary.json', {key: manifest[key] for key in (
            'dataset', 'size', 'image_counts', 'subject_counts', 'source_counts', 'species_counts',
            'unique_plane_counts', 'unique_physical_plane_counts', 'split_design',
            'parent_manifest', 'parent_manifest_sha256', 'parent_image_counts', 'normalization')})
        staging.replace(out)
        print(json.dumps(dict(event='complete', output=str(out), image_counts=image_counts,
            subject_counts=subject_counts, manifest_sha256=manifest_hash, train_npy_sha256=npy_hash,
            training_source_probability=dict(sorted(probability.items())))), flush=True)
    except BaseException:
        shutil.rmtree(staging)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=PROJECT / 'runs/rodent192_spen2x_260914/data')
    p.add_argument('--out', type=Path, default=PROJECT / 'runs/rodent192_spen2x_260914/data_all')
    p.add_argument('--expected-count', type=int, default=10633)
    args = p.parse_args()
    if args.expected_count < 1:
        p.error('expected-count must be positive')
    prepare_all(args.data, args.out, args.expected_count)


if __name__ == '__main__':
    main()
