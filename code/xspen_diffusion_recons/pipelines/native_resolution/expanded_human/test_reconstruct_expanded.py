"""Guard the irreversible interpretation errors: resized native grids and stale reuse."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconstruct_expanded import atomic_json, atomic_npz, canonical_hash, completed_case, sha256, validate_scan


def test_grid_and_case_validation():
    entry = dict(scan='MID51', cases=[dict(slice_index=12, repeat=0)])
    validate_scan(entry, (18, 48, 32, 46, 48), (46, 48))
    with pytest.raises(ValueError, match='resizing is forbidden'):
        validate_scan(entry, (18, 48, 32, 46, 48), (60, 64))
    with pytest.raises(ValueError, match='Duplicate'):
        validate_scan(dict(entry, cases=entry['cases']*2), (18,48,32,46,48), (46,48))
    with pytest.raises(ValueError, match='bounds'):
        validate_scan(dict(entry, cases=[dict(slice_index=48, repeat=0)]), (18,48,32,46,48), (46,48))


def test_completed_case_rejects_stale_or_corrupt_artifacts(tmp_path):
    npz_path, json_path = tmp_path/'case.npz', tmp_path/'case.json'
    signature = canonical_hash(dict(checkpoint='abc', h5='def', steps=60))
    assert completed_case(npz_path, json_path, signature) is None
    atomic_npz(npz_path, dict(x=np.ones((1,1,46,48))))
    info = dict(status='complete', computation_sha256=signature, npz_sha256=sha256(npz_path))
    atomic_json(json_path, info)
    assert completed_case(npz_path, json_path, signature) == info
    with pytest.raises(FileExistsError, match='differs'):
        completed_case(npz_path, json_path, canonical_hash(dict(checkpoint='abc', h5='def', steps=2)))
    with npz_path.open('ab') as target:
        target.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        completed_case(npz_path, json_path, signature)
