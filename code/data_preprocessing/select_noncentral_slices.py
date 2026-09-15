"""Copy a traceable, stratified subset of native stack-end PNGs without reprocessing."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

from export_rodent_brain import dump, sha256


def allocate_quotas(capacities, count, minimums=None, seed=260914):
    """Largest-remainder allocation, with source/side coverage when count allows."""
    minimums = minimums or {}
    if count < 1 or count > sum(capacities.values()):
        raise ValueError('Requested count exceeds eligible unique images')
    floor = {key: max(minimums.get(key, 0), int(count >= len(capacities))) for key in capacities}
    if any(floor[key] > capacity for key, capacity in capacities.items()) or sum(floor.values()) > count:
        raise ValueError('Required images cannot fit requested count')
    total = sum(capacities.values())
    ideal = {key: count * capacity / total for key, capacity in capacities.items()}
    quotas = {key: max(floor[key], int(ideal[key])) for key in capacities}
    tie = {key: hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest() for key in capacities}
    while sum(quotas.values()) > count:
        key = max((key for key in capacities if quotas[key] > floor[key]),
                  key=lambda key: (quotas[key] - ideal[key], tie[key]))
        quotas[key] -= 1
    while sum(quotas.values()) < count:
        key = max((key for key in capacities if quotas[key] < capacities[key]),
                  key=lambda key: (ideal[key] - quotas[key], tie[key]))
        quotas[key] += 1
    return quotas


def select_records(records, processed, count=4000, edge_fraction=0.25,
                   required_nums=(1442, 10493), min_p99=0.4, min_std=0.08, seed=260914):
    if not 0 < edge_fraction < 0.5:
        raise ValueError('Edge fraction must be between 0 and 0.5')
    processed = {source['source_id']: source for source in processed}
    required_nums = set(required_nums)
    groups, exclusions = defaultdict(list), []
    candidates = []
    for record in records:
        source = processed[record['source_id']]
        slices, index = source['slices'], record['slice_index']
        if slices < 2 or not 0 <= index < slices:
            raise ValueError('Invalid native slice index or source length')
        position = index / (slices - 1)
        if edge_fraction < position < 1 - edge_fraction:
            exclusions.append(dict(filename=record['filename'], num=record['num'], reason='central_stack_position'))
            continue
        side = 'low_index' if position <= edge_fraction else 'high_index'
        qc = record['qc']
        weak = qc['normalized_p99'] < min_p99 or qc['normalized_std'] < min_std
        quality = 0.5 * min(qc['normalized_p99'] / 0.8, 1) + 0.5 * min(qc['normalized_std'] / 0.16, 1)
        annotation = dict(native_slice_count=slices, native_slice_index=index,
                          native_relative_position=position, stack_end=side,
                          distance_from_stack_midpoint=abs(position - 0.5),
                          quality_rank_score=quality, weak_signal=weak,
                          user_example=record['num'] in required_nums,
                          note='Native stack position, not an anatomical label or brain segmentation')
        annotated = {**record, 'selection': annotation}
        candidates.append(annotated)
        if weak:
            exclusions.append(dict(filename=record['filename'], num=record['num'],
                                   reason='weak_signal_for_this_subset', selection=annotation))
            continue
        groups[(record['source_id'], side)].append(annotated)
    present = {record['num'] for group in groups.values() for record in group}
    if required_nums - present:
        raise ValueError(f'Required examples are missing or fail selection: {sorted(required_nums - present)}')
    required_count = {key: sum(record['num'] in required_nums for record in group) for key, group in groups.items()}
    quotas = allocate_quotas({key: len(group) for key, group in groups.items()}, count, required_count, seed)
    reference_edge = float(np.mean([min(record['selection']['native_relative_position'],
                                         1 - record['selection']['native_relative_position'])
                                    for record in candidates if record['num'] in required_nums])) if required_nums else 0.175
    selected = []
    for key, group in sorted(groups.items()):
        def rank(record):
            selection = record['selection']
            edge_position = min(selection['native_relative_position'], 1 - selection['native_relative_position'])
            tie = hashlib.sha256(f"{seed}:{record['filename']}".encode()).hexdigest()
            return (record['num'] in required_nums, selection['quality_rank_score'],
                    -abs(edge_position - reference_edge), tie)
        ordered = sorted(group, key=rank, reverse=True)
        selected.extend(ordered[:quotas[key]])
        exclusions.extend(dict(filename=record['filename'], num=record['num'],
                               reason='stratified_count_limit', selection=record['selection'])
                          for record in ordered[quotas[key]:])
    selected.sort(key=lambda record: record['num'])
    if len(selected) != count or len({record['pixel_sha256'] for record in selected}) != count:
        raise ValueError('Selection is not the requested number of unique pixel arrays')
    return selected, exclusions, dict(noncentral_candidates=len(candidates),
                                      eligible_after_signal_screen=sum(map(len, groups.values())),
                                      eligible_source_end_groups=len(groups),
                                      reference_edge_position=reference_edge)


def copy_and_verify(parent, out, records, size, bit_depth):
    destination = out / 'images'
    destination.mkdir(parents=True)
    pixel_hashes = set()
    for record in records:
        source = parent / record['image']
        if source.resolve().parent != (parent / 'images').resolve() or source.name != record['filename']:
            raise ValueError('Unexpected parent image path')
        target = destination / record['filename']
        shutil.copy2(source, target)
        binary = target.read_bytes()
        if hashlib.sha256(binary).hexdigest() != record['png_sha256']:
            raise ValueError(f'PNG content mismatch: {source.name}')
        if binary[24] != bit_depth or binary[25] != 0:
            raise ValueError('Unexpected PNG bit depth or color type')
        with Image.open(target) as image:
            if list(image.size) != size:
                raise ValueError('Unexpected output size')
            pixels = np.asarray(image, dtype=np.uint16 if bit_depth == 16 else np.uint8)
        digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        if digest != record['pixel_sha256'] or digest in pixel_hashes:
            raise ValueError('Pixel content mismatch or duplicate')
        pixel_hashes.add(digest)
        transform = record['transform']
        if min(transform['native_plane_shape']) < 192 or transform['upsampled'] or transform['added_padding']:
            raise ValueError('Subset violates inherited native-resolution/crop requirements')
    return dict(passed=True, checked_images=len(records), file_hashes_checked=True,
                pixel_hashes_checked=True, source_png_bytes_unchanged=True,
                unique_pixel_arrays=True, original_filenames_preserved=True,
                native_minimum=192, no_upsampling=True, no_added_padding=True,
                size=size, bit_depth=bit_depth, errors=[])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Parent completed export run')
    parser.add_argument('--out', type=Path, required=True, help='New subset directory within parent run, outside images/')
    parser.add_argument('--count', type=int, default=4000)
    parser.add_argument('--edge-fraction', type=float, default=0.25)
    parser.add_argument('--include-num', type=int, nargs='*', default=[1442, 10493])
    parser.add_argument('--min-p99', type=float, default=0.4)
    parser.add_argument('--min-std', type=float, default=0.08)
    parser.add_argument('--seed', type=int, default=260914)
    args = parser.parse_args()
    parent, out = args.run.resolve(), args.out.resolve()
    if out == parent or not out.is_relative_to(parent) or out.is_relative_to(parent / 'images'):
        parser.error('--out must be a new child directory of parent run, outside images/')
    if out.exists():
        parser.error('Output directory already exists; choose a new directory')
    manifest_hash = sha256(parent / 'manifest.jsonl')
    records = [json.loads(line) for line in (parent / 'manifest.jsonl').read_text().splitlines()]
    processed = json.loads((parent / 'processed_sources.json').read_text())
    summary = json.loads((parent / 'summary.json').read_text())
    selected, exclusions, statistics = select_records(records, processed, args.count, args.edge_fraction,
                                                      args.include_num, args.min_p99, args.min_std, args.seed)
    verified = copy_and_verify(parent, out, selected, summary['size'], summary['bit_depth'])
    by_source = defaultdict(list)
    for rank, record in enumerate(selected, 1):
        record['selection']['subset_rank'] = rank
        by_source[record['source_id']].append(record)
    (out / 'manifest.jsonl').write_text(''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in selected))
    (out / 'not_selected.jsonl').write_text(''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in exclusions))
    sources = json.loads((parent / 'selected_sources.json').read_text())
    dump(out / 'selected_sources.json', [source for source in sources if source['source_id'] in by_source])
    dump(out / 'processed_sources.json', [dict(source, parent_accepted_images=source['accepted_images'],
                                              accepted_images=len(by_source[source['source_id']]),
                                              subset_slice_indices=[record['slice_index'] for record in by_source[source['source_id']]])
                                          for source in processed if source['source_id'] in by_source])
    dump(out / 'failed_sources.json', [])
    dataset_counts = Counter(record['dataset'] for record in selected)
    subset_summary = dict(status='complete', image_count=len(selected), parent_image_count=len(records),
                          parent_run=str(parent), size=summary['size'], bit_depth=summary['bit_depth'],
                          selected_sources=len(by_source), sources_with_images=len(by_source),
                          images_by_dataset=dict(dataset_counts),
                          images_by_sequence=dict(Counter(record['sequence'] for record in selected)),
                          images_by_stack_end=dict(Counter(record['selection']['stack_end'] for record in selected)),
                          subject_groups=len({record['subject_group'] for record in selected}),
                          selection_rule=f'native slice_index / (native_slice_count - 1) <= {args.edge_fraction} or >= {1 - args.edge_fraction}',
                          signal_screen=dict(min_normalized_p99=args.min_p99, min_normalized_std=args.min_std,
                                             brain_area_threshold=None),
                          not_selected_by_reason=dict(Counter(record['reason'] for record in exclusions)),
                          required_examples=args.include_num, upsampled_images=0, images_with_added_padding=0,
                          direct_native_crops=sum(record['transform']['direct_native_crop'] for record in selected),
                          original_filenames_preserved=True, numbering='Parent IDs retained; subset_rank is contiguous',
                          **statistics)
    dump(out / 'summary.json', subset_summary)
    dump(out / 'config.json', dict(arguments=vars(args), created_at=datetime.now().astimezone().isoformat(),
                                  command=[sys.executable, *sys.argv], parent_manifest_sha256=manifest_hash,
                                  script_sha256=sha256(Path(__file__)), method='Source x stack-end stratified sampling; PNG byte copies'))
    if sha256(parent / 'manifest.jsonl') != manifest_hash:
        raise ValueError('Parent manifest changed during selection')
    verified['parent_manifest_unchanged'] = True
    verified['required_examples_present'] = all(number in {record['num'] for record in selected} for number in args.include_num)
    dump(out / 'verification.json', verified)
    print(json.dumps(subset_summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
