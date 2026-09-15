"""Export traceable 192 x 192 rodent brain candidate slices from the local data catalog."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

from image_processing import assess_slice, fit_plane, volume_window
from nifti_sources import discover_nifti, load_nifti
from bruker_sources import discover_bruker, load_bruker

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value):
    return re.sub(r'[^A-Za-z0-9-]+', '-', str(value)).strip('-') or 'unknown'


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default) + '\n')


def choose_preview(sources, count):
    groups = defaultdict(list)
    for source in sources:
        groups[source['dataset']].append(source)
    selected = []
    for group in groups.values():
        indices = np.linspace(0, len(group) - 1, min(count, len(group))).round().astype(int)
        selected.extend(group[i] for i in indices)
    return selected


def filter_native_size(sources, minimum):
    kept, excluded = [], []
    for source in sources:
        shape = source.get('native_plane_shape')
        if shape is None and source.get('native_shape'):
            shape = [source['native_shape'][1], source['native_shape'][0]]
        if shape is None:
            raise ValueError(f"Missing native plane shape: {source['source_id']}")
        if min(shape) < minimum:
            excluded.append(dict(dataset=source['dataset'], source_id=source['source_id'],
                                 path=source['path'], native_plane_shape=list(shape),
                                 minimum_native_size=minimum, reason='native_matrix_below_minimum'))
        else:
            kept.append(source)
    return kept, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=WORKSPACE / 'code/data/rodent_mri')
    parser.add_argument('--out', type=Path, required=True, help='A new directory under this project runs/')
    parser.add_argument('--size', type=int, default=192)
    parser.add_argument('--bit-depth', type=int, choices=(8, 16), default=16)
    parser.add_argument('--min-native-size', type=int, default=192,
                        help='Exclude sources whose native in-plane matrix has either side below this size')
    parser.add_argument('--crop-mode', choices=('foreground', 'full'), default='foreground')
    parser.add_argument('--crop-margin', type=float, default=0.10, help='Fractional margin on each side of detected tissue')
    parser.add_argument('--trim-fraction', type=float, default=0.12, help='Exclude this fraction at each stack end')
    parser.add_argument('--preview-per-dataset', type=int, default=0)
    parser.add_argument('--sources', choices=('all', 'public', 'lab'), default='all')
    args = parser.parse_args()
    if not 0 <= args.trim_fraction < 0.5:
        parser.error('--trim-fraction must be in [0, 0.5)')
    if args.size < 2 or args.preview_per_dataset < 0:
        parser.error('size must be >= 2 and preview count >= 0')
    if args.min_native_size < args.size:
        parser.error('--min-native-size must be >= --size; this exporter does not enlarge small matrices')
    if not 0 <= args.crop_margin <= 0.5:
        parser.error('--crop-margin must be in [0, 0.5]')
    out = args.out.resolve()
    if not out.is_relative_to(HERE / 'runs'):
        parser.error('--out must be under code/data_preprocessing/runs/')
    if out.exists():
        parser.error('Output directory already exists; choose a new run directory')
    sources, exclusions = [], []
    for discover, kind in [(discover_nifti, 'public'), (discover_bruker, 'lab')]:
        if args.sources in ('all', kind):
            selected, skipped = discover(args.input.resolve())
            for source in selected:
                source['reader'] = kind
            sources.extend(selected)
            exclusions.extend(skipped)
    candidate_source_count = len(sources)
    sources, resolution_exclusions = filter_native_size(sources, args.min_native_size)
    exclusions.extend(resolution_exclusions)
    sources.sort(key=lambda s: (s['dataset'], s['source_id']))
    if args.preview_per_dataset:
        sources = choose_preview(sources, args.preview_per_dataset)
    if not sources:
        raise RuntimeError('No eligible source volumes found')
    (out / 'images').mkdir(parents=True)
    dump(out / 'config.json', dict(arguments=vars(args), created_at=datetime.now().astimezone().isoformat(),
                                 command=[sys.executable, *sys.argv],
                                 source_code_sha256={p.name: sha256(p) for p in HERE.glob('*.py')},
                                 python=sys.version))
    dump(out / 'selected_sources.json', sources)
    dump(out / 'excluded_sources.json', exclusions)
    dump(out / 'native_resolution_exclusions.json', resolution_exclusions)
    records, source_results, failures = [], [], []
    pixel_hashes, file_hashes = {}, {}
    rejected_counts, dataset_counts, sequence_counts = Counter(), Counter(), Counter()
    print(json.dumps(dict(event='started', sources=len(sources), excluded_sources=len(exclusions))), flush=True)
    with (out / 'manifest.jsonl').open('w') as manifest, (out / 'rejected_slices.jsonl').open('w') as rejects:
        for source_number, source in enumerate(sources, 1):
            try:
                volume, metadata = (load_nifti if source['reader'] == 'public' else load_bruker)(source)
                if volume.ndim != 3:
                    raise ValueError(f'Loader returned shape {volume.shape}')
                if min(volume.shape[1:]) < args.min_native_size:
                    raise ValueError('Loaded native matrix differs from discovery size filter')
                spacing = metadata.get('spacing_yx', metadata.get('pixel_spacing_mm'))
                if spacing is None:
                    raise ValueError('Missing in-plane spacing')
                upper = volume_window(volume)
                source_path = Path(source['path'])
                cache_key = str(source_path.resolve())
                if cache_key not in file_hashes:
                    file_hashes[cache_key] = sha256(source_path)
                digest = file_hashes[cache_key]
                expected = source.get('expected_sha256')
                if expected and digest != expected:
                    raise ValueError('Source SHA256 differs from provenance')
                n = len(volume)
                trim = int(np.floor(n * args.trim_fraction))
                start = max(trim, int(source.get('slice_start', 0)))
                stop = min(n - trim, int(source.get('slice_stop', n)))
                accepted = 0
                for index, plane in enumerate(volume):
                    stats = {}
                    if not start <= index < stop:
                        ok, reason = False, 'stack_edge_or_source_range'
                    else:
                        ok, reason, stats = assess_slice(plane, upper)
                    if ok:
                        try:
                            pixels, transform = fit_plane(plane, spacing, upper, args.size, args.bit_depth,
                                                          args.crop_mode, args.crop_margin)
                        except ValueError as error:
                            if str(error) not in {'crop_foreground_not_found', 'physical_square_requires_upsampling',
                                                  'foreground_does_not_fit_unpadded_square'}:
                                raise
                            ok, reason = False, str(error)
                    if ok:
                        pixel_digest = hashlib.sha256(pixels.tobytes()).hexdigest()
                        if pixel_digest in pixel_hashes:
                            ok, reason = False, 'duplicate_output_pixels'
                            stats['duplicate_of'] = pixel_hashes[pixel_digest]
                    if not ok:
                        rejected_counts[reason] += 1
                        rejects.write(json.dumps(dict(source_id=source['source_id'], slice_index=index,
                                                      reason=reason, qc=stats), default=json_default) + '\n')
                        continue
                    number = len(records) + 1
                    name = f"{number}_{safe_name(source['source_id'])}-sl{index:03d}_{safe_name(source['sequence'])}.png"
                    destination = out / 'images' / name
                    Image.fromarray(pixels).save(destination)
                    record = dict(num=number, filename=name, image=f'images/{name}',
                                  dataset=source['dataset'], subject=source['subject'],
                                  species=source.get('species', 'rodent'), sequence=source['sequence'],
                                  source_id=source['source_id'], source_path=str(source_path),
                                  source_sha256=digest, slice_index=index,
                                  echo_index=source.get('echo_index'), echo_time_ms=source.get('echo_time_ms'),
                                  transform=transform, qc=stats, pixel_sha256=pixel_digest,
                                  png_sha256=sha256(destination))
                    if source.get('split_group'):
                        record['split_group'] = source['split_group']
                    record['subject_group'] = source.get('subject_group', f"{source['dataset']}:{source['subject']}")
                    if metadata.get('source_frame_indices') is not None:
                        record['source_frame_index'] = metadata['source_frame_indices'][index]
                    manifest.write(json.dumps(record, ensure_ascii=False, default=json_default) + '\n')
                    records.append(record)
                    pixel_hashes[pixel_digest] = name
                    dataset_counts[source['dataset']] += 1
                    sequence_counts[source['sequence']] += 1
                    accepted += 1
                source_results.append(dict(source_id=source['source_id'], source_sha256=digest,
                                           accepted_images=accepted, slices=n, selected_range=[start, stop],
                                           normalization_upper=upper, metadata=metadata))
            except Exception as error:
                failure = dict(source_id=source['source_id'], path=str(source['path']),
                               error=f'{type(error).__name__}: {error}')
                failures.append(failure)
                print(json.dumps(dict(event='source_error', **failure)), flush=True)
            if source_number % 20 == 0 or source_number == len(sources):
                manifest.flush()
                print(json.dumps(dict(event='progress', sources=source_number, total_sources=len(sources),
                                      images=len(records), failures=len(failures))), flush=True)
    dump(out / 'processed_sources.json', source_results)
    dump(out / 'failed_sources.json', failures)
    summary = dict(status='completed_with_source_errors' if failures else 'complete', image_count=len(records),
                   size=[args.size, args.size], bit_depth=args.bit_depth,
                   crop_mode=args.crop_mode, minimum_native_size=args.min_native_size,
                   candidate_sources=candidate_source_count,
                   native_resolution_excluded_sources=len(resolution_exclusions),
                   selected_sources=len(sources), loaded_sources=len(source_results),
                   sources_with_images=sum(s['accepted_images'] > 0 for s in source_results),
                   failed_sources=len(failures), excluded_sources=len(exclusions),
                   images_by_dataset=dict(dataset_counts), images_by_sequence=dict(sequence_counts),
                   subject_groups=len({r['subject_group'] for r in records}),
                   rejected_slices=dict(rejected_counts),
                   upsampled_images=sum(r['transform']['upsampled'] for r in records),
                   direct_native_crops=sum(r['transform']['direct_native_crop'] for r in records),
                   images_with_added_padding=sum(r['transform']['added_padding'] for r in records),
                   note='Automatic brain-candidate export; no skull stripping, registration, or train/test split. '
                        'Output matrix size is not acquired resolution. Animal IDs can overlap across sessions/datasets.')
    dump(out / 'summary.json', summary)
    print(json.dumps(dict(event='complete', **summary)), flush=True)
    if not records or failures:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
