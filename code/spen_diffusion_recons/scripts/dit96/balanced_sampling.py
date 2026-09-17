"""Explicit training mixture, preserving the established mouse/FOV balance.

All expanded PNGs remain eligible. Unknown lab rodent species is not guessed.
Weights are defined before training and never fitted to validation/test scores.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np

GROUP_MASS = {'old_mouse': .80, 'old_rat': .10, 'lab': .08,
              'ds005186': .01, 'figshare-aging-28433102': .01}


def safe(value):
    return re.sub(r'[^A-Za-z0-9-]+', '-', str(value)).strip('-') or 'unknown'


def _hierarchical_mass(entries, probability, weights):
    """Equal subjects -> equal scans -> equal echoes -> equal slices."""
    subjects = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for index, record in entries:
        scan = record['scan']
        acquisition = re.sub(r'-e\d+$', '', scan)
        subjects[record['subject']][acquisition][scan].append(index)
    for scans in subjects.values():
        for echoes in scans.values():
            for indices in echoes.values():
                mass = probability / len(subjects) / len(scans) / len(echoes)
                weights[indices] = mass / len(indices)


def balanced_weights(records, manifest_path):
    """Map PNG IDs to audited old weights, then allocate the new source mixture."""
    manifest_path = Path(manifest_path)
    content = manifest_path.read_bytes()
    legacy = json.loads(content)['records']['train']
    weights = np.zeros(len(records), dtype=np.float64)
    groups = defaultdict(list)
    old_seen = set()
    labels = []
    for index, record in enumerate(records):
        if record['batch'] == 'old':
            old_index = record['old_index']
            if old_index in old_seen or not 0 <= old_index < len(legacy):
                raise ValueError('Duplicate or invalid old training index')
            old_seen.add(old_index)
            original = legacy[old_index]
            expected = f"old__{safe(original['dataset'])}__{safe(original['subject'])}__{old_index:06d}__{safe(original['view'])}.png"
            if record['filename'] != expected:
                raise ValueError(f'PNG/legacy identity mismatch: {record["filename"]}')
            group = 'old_' + original['species']
            if group not in ('old_mouse', 'old_rat'):
                raise ValueError('Unrecognized legacy species')
            weight = float(original['sample_weight'])
            if not np.isfinite(weight) or weight <= 0:
                raise ValueError('Invalid legacy sampling weight')
            weights[index] = weight
        else:
            group = 'lab' if record['source'] == 'lab-RAT' else record['source']
            if group not in GROUP_MASS or group.startswith('old_'):
                raise ValueError(f'Unrecognized new source: {record["source"]}')
        groups[group].append((index, record))
        labels.append(group)
    if old_seen != set(range(len(legacy))):
        raise ValueError('Expanded PNG dataset does not contain every legacy training image')
    if set(groups) != set(GROUP_MASS):
        raise ValueError(f'Missing sampling groups: {set(GROUP_MASS)-set(groups)}')
    for group, mass in GROUP_MASS.items():
        if group.startswith('old_'):
            indices = [i for i, _ in groups[group]]
            weights[indices] *= mass / weights[indices].sum()
        else:
            _hierarchical_mass(groups[group], mass, weights)
    if not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.):
        raise ValueError('Invalid balanced probability vector')
    details = {}
    for group, entries in groups.items():
        ids = [i for i, _ in entries]
        subject_mass, source_mass, view_mass = defaultdict(float), defaultdict(float), defaultdict(float)
        for index, record in entries:
            subject_mass[record['subject']] += float(weights[index])
            source_mass[record['source']] += float(weights[index])
            view_mass[record['view']] += float(weights[index])
        details[group] = dict(probability=float(weights[ids].sum()), images=len(ids),
            subjects=len(subject_mass), per_subject_probability=dict(sorted(subject_mass.items())),
            per_source_probability=dict(sorted(source_mass.items())),
            per_view_probability=dict(sorted(view_mass.items())))
    audit = dict(policy='balanced_v1', group_probability=GROUP_MASS, groups=details,
        legacy_manifest=str(manifest_path.resolve()), legacy_manifest_sha256=hashlib.sha256(content).hexdigest(),
        probability_sha256=hashlib.sha256(weights.astype('<f8').tobytes()).hexdigest(),
        ordering='same sorted PNG order as data audit train images',
        legacy_balance='preserve old within-species sample_weight ratios; mouse FOV16/FOV24 = 60/40',
        new_balance='source quota, then equal labeled subjects, acquisitions, echoes, and slices',
        lab_species='unspecified; lab-RAT is a source label, not a biological assertion',
        no_test_selection=True, all_training_images_positive=True)
    return weights, audit
