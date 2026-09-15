"""Pack a verified native 192 PNG export with reproducible animal-group splits.

The original PNG pixels and species labels are retained. No image resampling,
renormalization or PNG copying occurs. The training weight is uniform over
subject groups, then uniform over all accepted images of the selected group.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image


PARTS = ('train', 'val', 'test')
PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT.parent / 'data_preprocessing/runs/rodent_brain192_native_crop_260914'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def stable_order(seed, key):
    return hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()


def allocate(capacities, limit):
    """Water-fill integer quotas; small strata retain all their available items."""
    quotas = {key: 0 for key in sorted(capacities)}
    remaining = min(limit, sum(capacities.values()))
    while remaining:
        for key in quotas:
            if quotas[key] < capacities[key] and remaining:
                quotas[key] += 1
                remaining -= 1
    return quotas


def assign_groups(records, seed, fractions):
    groups = defaultdict(list)
    for record in records:
        groups[record['subject_group']].append(record)
    # Cross-dataset identity groups form one indivisible stratum member.
    strata = defaultdict(list)
    for group, rows in groups.items():
        signature = tuple(sorted({(r['dataset'], r['species']) for r in rows}))
        strata[signature].append(group)
    assignments, strata_report = {}, []
    for signature, members in sorted(strata.items()):
        members.sort(key=lambda group: stable_order(seed, group))
        exact = np.asarray(fractions) * len(members)
        counts = np.floor(exact).astype(int)
        remainder_order = sorted(range(3), key=lambda i: (-(exact[i] - counts[i]), i))
        for index in remainder_order[:len(members) - int(counts.sum())]:
            counts[index] += 1
        if len(members) >= 3:
            for index in (1, 2):
                if counts[index] == 0:
                    counts[0] -= 1
                    counts[index] = 1
        offset = 0
        for part, count in zip(PARTS, counts):
            for group in members[offset:offset + count]:
                assignments[group] = dict(part=part,
                    datasets=sorted({r['dataset'] for r in groups[group]}),
                    species=sorted({r['species'] for r in groups[group]}),
                    images=len(groups[group]))
            offset += count
        strata_report.append(dict(dataset_species=[list(pair) for pair in signature],
            group_counts={part: int(n) for part, n in zip(PARTS, counts)},
            total_groups=len(members)))
    return assignments, strata_report


def select_validation(records, limit, seed):
    """Represent sources equally, then animal groups, using spread-out slices."""
    by_dataset = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(records):
        by_dataset[record['dataset']][record['subject']].append(index)
    source_quotas = allocate({dataset: sum(map(len, groups.values()))
                              for dataset, groups in by_dataset.items()}, limit)
    selected = []
    for dataset, groups in sorted(by_dataset.items()):
        # Seeded ranks avoid privileging lexicographically early animal IDs.
        ordered = sorted(groups, key=lambda group: stable_order(seed, group))
        ranked = {f'{rank:06d}': len(groups[group]) for rank, group in enumerate(ordered)}
        quotas = allocate(ranked, source_quotas[dataset])
        for rank, group in enumerate(ordered):
            candidates = sorted(groups[group], key=lambda i: (
                records[i]['source_id'], records[i]['slice_index'], records[i]['key']))
            count = quotas[f'{rank:06d}']
            # Midpoints of equal bins give a central slice when count == 1.
            positions = ((np.arange(count) + .5) * len(candidates) / max(count, 1)).astype(int)
            selected.extend(candidates[index] for index in positions)
    return sorted(selected)


def validate_geometry(record):
    transform = record['transform']
    shape = np.asarray(transform['native_plane_shape'])
    output_shape = np.asarray(transform['output_shape'])
    sampling = np.asarray(transform['sampling_yx'], dtype=float)
    offset = np.asarray(transform['offset_yx'], dtype=float)
    last = offset + sampling * (output_shape - 1)
    if shape.shape != (2,) or np.any(shape < 192) or list(output_shape) != [192, 192]:
        raise ValueError('Image is not derived from an adequate native matrix')
    if transform['bit_depth'] != 16 or transform['upsampled'] or transform['added_padding']:
        raise ValueError('Expected a 16-bit export without upsampling or added padding')
    if (not np.isfinite(sampling).all() or not np.isfinite(offset).all()
            or np.any(sampling < 1 - 1e-4) or np.any(offset < -1e-6)
            or np.any(last > shape - 1 + 1e-6)):
        raise ValueError('Output sampling enlarges the native data or leaves its FOV')
    window = np.asarray(transform['normalization_window'], dtype=float)
    if window.shape != (2,) or not np.isfinite(window).all() or window[0] != 0 or window[1] <= 0:
        raise ValueError('Invalid exported normalization window')
    if transform['normalization'] != 'per-source positive q99.5':
        raise ValueError('Unexpected export normalization; review it before packing')


def prepare(source, out, seed, fractions, validation_limit):
    source, out = source.resolve(strict=True), out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output is not empty: {out}')
    source_manifest = source / 'manifest.jsonl'
    source_records = [json.loads(line) for line in source_manifest.read_text().splitlines()]
    summary = json.loads((source / 'summary.json').read_text())
    verification = json.loads((source / 'verification.json').read_text())
    processed = {row['source_id']: row for row in json.loads((source / 'processed_sources.json').read_text())}
    if (summary['status'] != 'complete' or summary['image_count'] != len(source_records)
            or not verification['passed'] or verification['checked_images'] != len(source_records)):
        raise ValueError('Input export is incomplete or disagrees with its verification')
    if summary['size'] != [192, 192] or summary['bit_depth'] != 16 or summary['upsampled_images'] != 0:
        raise ValueError('Expected the complete native 192x192, 16-bit, non-upsampled export')
    if {row['filename'] for row in source_records} != {path.name for path in (source / 'images').iterdir()}:
        raise ValueError('Export images and manifest membership differ')
    if dict(Counter(row['dataset'] for row in source_records)) != summary['images_by_dataset']:
        raise ValueError('Dataset counts differ from the export summary')
    assignments, strata = assign_groups(source_records, seed, fractions)
    records = {part: [] for part in PARTS}
    for index, original in enumerate(source_records):
        if original['num'] != index + 1:
            raise ValueError('Input manifest is not contiguously numbered')
        validate_geometry(original)
        group = original['subject_group']
        part = assignments[group]['part']
        source_info = processed[original['source_id']]
        if source_info['source_sha256'] != original['source_sha256']:
            raise ValueError('Source provenance SHA256 differs across export records')
        axis = source_info['metadata'].get('slice_axis')
        if axis is None:
            if source_info['metadata'].get('array_axis_order') != ['slice', 'row', 'column']:
                raise ValueError('Unknown source stack axis convention')
            axis = 0
        plane = f"{original['source_sha256']}:{axis}:{original['slice_index']}"
        echo = original.get('echo_index')
        transform = original['transform']
        record = dict(original, original_subject=original['subject'], subject=group,
            split_group=group, split=part, source=str(Path(original['source_path']).resolve()),
            png_path=str((source / original['image']).resolve(strict=True)),
            key=f"native192:{original['source_id']}:sl{original['slice_index']:03d}",
            view='native_crop192', slice_axis=axis,
            source_plane_key=plane + (f':echo={echo}' if echo is not None else ''),
            physical_plane_key=plane, source_manifest_index=index,
            hr_shape=[192, 192], native_plane_shape=transform['native_plane_shape'],
            native_plane_spacing_mm=transform['native_spacing_yx'],
            hr_pixel_spacing_mm=transform['output_spacing_yx'], fov_mm=transform['square_fov'])
        records[part].append(record)
    if any(not rows for rows in records.values()):
        raise ValueError('All three splits must contain images')
    subjects = {part: sorted({row['subject'] for row in rows}) for part, rows in records.items()}
    # Source volumes, physical planes, echo frames and known animal identities
    # must all stay in the same split, regardless of contrast or PNG cropping.
    overlap_fields = ('subject', 'source_sha256', 'source_plane_key', 'physical_plane_key', 'pixel_sha256', 'key')
    for field in overlap_fields:
        sets = {part: {row[field] for row in rows} for part, rows in records.items()}
        for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            if sets[a] & sets[b]:
                raise ValueError(f'Cross-split overlap: {field}, {a}, {b}')
    for field in ('pixel_sha256', 'source_plane_key', 'key'):
        if len({row[field] for rows in records.values() for row in rows}) != len(source_records):
            raise ValueError(f'Duplicate image or echo-frame identity: {field}')
    train_counts = Counter(row['subject'] for row in records['train'])
    for row in records['train']:
        row['sample_weight'] = 1 / (len(train_counts) * train_counts[row['subject']])
    selection = select_validation(records['val'], validation_limit, seed)
    source_files = [source_manifest] + [source / name for name in (
        'summary.json', 'verification.json', 'config.json', 'selected_sources.json', 'processed_sources.json')]
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{out.name}.preparing-', dir=out.parent))
    try:
        npy_hashes, pixel_statistics = {}, {}
        for part, rows in records.items():
            path = staging / f'{part}.npy'
            array = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint16,
                                             shape=(len(rows), 192, 192))
            minimum, maximum, total, sum_squared, nonzero, clipped = 65535, 0, 0., 0., 0, 0
            for index, row in enumerate(rows):
                binary = Path(row['png_path']).read_bytes()
                if (binary[:8] != b'\x89PNG\r\n\x1a\n' or binary[24:26] != bytes((16, 0))
                        or hashlib.sha256(binary).hexdigest() != row['png_sha256']):
                    raise ValueError(f'PNG bit depth, type or SHA256 mismatch: {row["png_path"]}')
                with Image.open(io.BytesIO(binary)) as image:
                    raw = np.asarray(image)
                    if (raw.shape != (192, 192) or raw.dtype.kind not in 'ui'
                            or int(raw.min()) < 0 or int(raw.max()) > 65535):
                        raise ValueError(f'Invalid uint16 image: {row["png_path"]}')
                    pixels = raw.astype(np.uint16)
                if (hashlib.sha256(pixels.tobytes()).hexdigest() != row['pixel_sha256']
                        or pixels.min() == pixels.max()):
                    raise ValueError(f'Pixel SHA256 mismatch or constant image: {row["png_path"]}')
                array[index] = pixels
                minimum, maximum = min(minimum, int(pixels.min())), max(maximum, int(pixels.max()))
                total += float(pixels.sum(dtype=np.float64))
                sum_squared += float(np.square(pixels, dtype=np.float64).sum())
                nonzero += int(np.count_nonzero(pixels))
                clipped += int(np.count_nonzero(pixels == 65535))
            array.flush()
            del array
            npy_hashes[part] = sha256(path)
            count = len(rows) * 192 * 192
            mean, square_mean = total / count / 65535, sum_squared / count / 65535 ** 2
            pixel_statistics[part] = dict(stored_min=minimum, stored_max=maximum,
                normalized_min=minimum / 65535, normalized_max=maximum / 65535,
                normalized_mean=mean, normalized_std=float(np.sqrt(max(0., square_mean - mean ** 2))),
                nonzero_fraction=nonzero / count, saturated_fraction=clipped / count,
                stored_dtype='uint16', normalized_dtype='float32', shape=[len(rows), 192, 192])
            print(json.dumps(dict(event='packed', split=part, images=len(rows), subjects=len(subjects[part]))), flush=True)
        source_probabilities = defaultdict(float)
        group_probabilities = defaultdict(float)
        for row in records['train']:
            source_probabilities[row['dataset']] += row['sample_weight']
            group_probabilities[row['subject']] += row['sample_weight']
        if not np.allclose(list(group_probabilities.values()), 1 / len(train_counts), atol=1e-12):
            raise ValueError('Training sampling does not balance subject groups')
        audit = dict(passed=True, checked_images=len(source_records), all_png_sha256_verified=True,
            all_pixel_sha256_verified=True, original_uint16_pixels_preserved=True,
            no_png_copies=True, no_resampling_or_new_normalization=True,
            minimum_native_matrix_side=min(min(row['transform']['native_plane_shape']) for row in source_records),
            upsampled_images=0, added_padding_images=0,
            cross_split_overlap={field: False for field in overlap_fields},
            duplicate_pixels=0, duplicate_echo_frame_keys=0,
            known_cross_dataset_groups=[group for group, info in assignments.items() if len(info['datasets']) > 1],
            training_weight_sum=sum(row['sample_weight'] for row in records['train']),
            training_source_probability=dict(sorted(source_probabilities.items())),
            normalized_pixel_statistics=pixel_statistics,
            original_raw_source_hashes='Recorded from verified export provenance; raw volume bytes are not re-read by this packer')
        manifest = dict(version=1, dataset='rodent_native192_png', size=192, species='mixed_rodent',
            created_at=datetime.now(timezone.utc).isoformat(), records=records, subjects=subjects,
            split_groups=subjects, assignments=assignments,
            image_counts={part: len(rows) for part, rows in records.items()},
            subject_counts={part: len(groups) for part, groups in subjects.items()},
            unique_plane_counts={part: len({row['source_plane_key'] for row in rows}) for part, rows in records.items()},
            unique_physical_plane_counts={part: len({row['physical_plane_key'] for row in rows}) for part, rows in records.items()},
            counts={part: dict(Counter(row['view'] for row in rows)) for part, rows in records.items()},
            source_counts={part: dict(Counter(row['dataset'] for row in rows)) for part, rows in records.items()},
            species_counts={part: dict(Counter(row['species'] for row in rows)) for part, rows in records.items()},
            sequence_counts={part: dict(Counter(row['sequence'] for row in rows)) for part, rows in records.items()},
            train_probability={'native_crop192': 1.0},
            sampling='Uniform subject_group, then uniform accepted image within that group; per-record sample_weight',
            split_design=dict(seed=seed, fractions=dict(zip(PARTS, fractions)), strata=strata,
                method='SHA256-seeded group order within dataset/species signature; largest-remainder group counts; whole cross-dataset groups stay together',
                grouping='Exporter subject_group identities, including known cross-dataset animal matches; no new animal identity inference',
                initialization_caveat='These new held-out groups are suitable for training from scratch; initialization from a previous prior requires a separate pretraining overlap audit'),
            selection_indices={'val': selection},
            validation_selection=dict(limit=validation_limit, images=len(selection),
                method='Equal source quotas with capacity redistribution, then equal subject-group quotas and uniformly spaced slices; original manifest order',
                source_counts=dict(Counter(records['val'][index]['dataset'] for index in selection)),
                subject_counts=len({records['val'][index]['subject'] for index in selection})),
            normalization='Preserve exported uint16 PNG exactly; clean x01=float32(pixels)/65535; diffusion input=2*x01-1; no per-image rescaling',
            source_manifest=str(source_manifest), source_manifest_sha256=sha256(source_manifest),
            source_export=str(source), source_export_provenance_sha256={str(path): sha256(path) for path in source_files},
            source_code_sha256={str(Path(__file__).resolve()): sha256(__file__)},
            npy_sha256=npy_hashes, source_export_summary=summary, audit=audit,
            note='Native reconstructed matrix is not effective acquired spatial resolution. Species labels are preserved, including rat and rodent_unspecified. Multiple echoes remain separate images but their source volumes and physical slice identities never cross splits.')
        write_json(staging / 'manifest.json', manifest)
        write_json(staging / 'verification.json', dict(audit, manifest_sha256=sha256(staging / 'manifest.json'),
            npy_sha256=npy_hashes, image_counts=manifest['image_counts'], subject_counts=manifest['subject_counts']))
        write_json(staging / 'summary.json', {key: manifest[key] for key in (
            'dataset', 'size', 'image_counts', 'subject_counts', 'source_counts', 'species_counts',
            'unique_plane_counts', 'unique_physical_plane_counts', 'validation_selection', 'normalization')})
        staging.replace(out)
        print(json.dumps(dict(event='complete', output=str(out), images=manifest['image_counts'],
            subjects=manifest['subject_counts'], manifest_sha256=sha256(out / 'manifest.json'))), flush=True)
    except BaseException:
        shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--out', type=Path, default=PROJECT / 'runs/rodent192_spen2x_260914/data')
    parser.add_argument('--seed', type=int, default=20260914)
    parser.add_argument('--val-fraction', type=float, default=.1)
    parser.add_argument('--test-fraction', type=float, default=.1)
    parser.add_argument('--validation-limit', type=int, default=256)
    args = parser.parse_args()
    fractions = (1 - args.val_fraction - args.test_fraction, args.val_fraction, args.test_fraction)
    if min(fractions) <= 0 or args.validation_limit < 1:
        parser.error('All split fractions and validation-limit must be positive')
    prepare(args.source, args.out, args.seed, fractions, args.validation_limit)


if __name__ == '__main__':
    main()
