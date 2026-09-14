"""Verify completed native datasets against the frozen IXI split and row hashes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=HERE / 'data')
    p.add_argument('--out', type=Path, default=HERE / 'audit/native_dataset_verification.json')
    args = p.parse_args()
    frozen = json.loads((HERE.parents[1] / 'data/ixi128/manifest.json').read_text())
    recipe = json.loads((args.data / 'preparation_recipe.json').read_text())
    results = {}
    keys = {}
    all_subject_split = {}
    for target in recipe['targets']:
        directory = args.data / target['id']
        m = json.loads((directory / 'manifest.json').read_text())
        assert m['complete'] and not recipe['pilot'], target['id']
        assert m['subjects'] == frozen['subjects']
        assert m['recipe_id'] == recipe['recipe_id']
        assert m['shape'] == target['shape']
        assert len(m['volumes']) == len(frozen['volumes']) == 1156
        split_checks = {}
        seen_hashes = {}
        keys[target['id']] = {}
        for sp, subjects in m['subjects'].items():
            a = np.load(directory / f'{sp}.npy', mmap_mode='r')
            rows = m['records'][sp]
            assert list(a.shape) == [len(rows), *m['shape']]
            assert str(a.dtype) == 'uint16'
            assert len(rows) == m['counts'][sp]
            assert set(r['subject'] for r in rows) == set(subjects)
            for row in rows:
                assert row['subject'] in subjects
                assert row['modality'] in ['T2', 'PD']
                assert row['view'] in ['axial', 'sagittal']
                assert row['thickness_mm'] == m['thickness_mm']
                previous = all_subject_split.setdefault(row['subject'], sp)
                assert previous == sp
                digest = row['slice_sha256']
                previous = seen_hashes.setdefault(digest, sp)
                assert previous == sp
            selected = np.unique(np.linspace(0, len(rows)-1, min(256, len(rows))).round().astype(int))
            for i in selected:
                assert hashlib.sha256(a[i].tobytes()).hexdigest() == rows[i]['slice_sha256']
            keys[target['id']][sp] = {r['key'] for r in rows}
            assert len(keys[target['id']][sp]) == len(rows)
            split_checks[sp] = dict(subjects=len(subjects), slices=len(rows),
                modalities=dict(Counter(r['modality'] for r in rows)),
                views=dict(Counter(r['view'] for r in rows)),
                sampled_row_hashes=len(selected), array_bytes=(directory/f'{sp}.npy').stat().st_size)
        results[target['id']] = dict(shape=m['shape'], pixel_mm=m['pixel_mm'],
            thickness_mm=m['thickness_mm'], splits=split_checks,
            checks='Frozen split, full record membership, shape/dtype/count, all-volume coverage, '
                   'cross-split output hash exclusion, deterministic sampled pixel hashes passed')
    common = {}
    for profile in sorted({t['profile_id'] for t in recipe['targets']}):
        tids = [t['id'] for t in recipe['targets'] if t['profile_id'] == profile]
        common[profile] = {sp:len(set.intersection(*(keys[tid][sp] for tid in tids)))
                           for sp in frozen['subjects']}
    result = dict(passed=True, recipe_id=recipe['recipe_id'], datasets=results,
        common_record_keys_across_each_profile_grids=common,
        unique_subjects_across_all_datasets=len(all_subject_split),
        note='Preparation already hashed every original NIfTI, every shard and each final array. '
             'This independent check validates all metadata and 256 evenly spaced pixel rows per split per dataset; '
             'it does not reread all array bytes a second time.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=True, datasets=len(results), unique_subjects=len(all_subject_split),
                         common_keys=common, output=str(args.out))))


if __name__ == '__main__':
    main()
