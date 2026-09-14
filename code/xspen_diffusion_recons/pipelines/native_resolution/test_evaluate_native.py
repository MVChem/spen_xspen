import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate_native import resolve_geometry, select_clean, synthetic_observation
from native_scanner import load_case

torch.set_num_threads(2)


def test_protocol_lookup_requires_consistent_shape():
    profile, grid = resolve_geometry(dict(profile_id='p4mm', grid_id='mm3', image_shape=[62, 64]), HERE)
    assert profile['native_shape'] == [46, 48]
    assert grid['shape'] == [62, 64]
    with pytest.raises(ValueError, match='disagree'):
        resolve_geometry(dict(profile_id='p4mm', grid_id='mm3', image_shape=[60, 64]), HERE)


@pytest.mark.parametrize('profile_id,shape', [('p3mm', (46, 48)), ('p3mm', (60, 64)), ('p4mm', (62, 64))])
def test_matched_rectangular_simulation_adjoint_and_proximal(profile_id, shape):
    profile = next(p for p in json.loads((HERE / 'protocols.json').read_text())['profiles'] if p['id'] == profile_id)
    x = torch.randn(1, 1, *shape)
    op, y = synthetic_observation(x, profile, acceleration=2)
    assert y.shape == (1, 32, *profile['native_shape'])
    assert torch.count_nonzero(y[:, :, 1::2]) == 0
    dual = torch.randn_like(y)
    torch.testing.assert_close((op.linear(x).conj() * dual).real.sum(), (x * op.adjoint(dual)).sum(), atol=2e-5, rtol=2e-5)
    z = torch.zeros_like(x)
    solution = op.proximal(z, y, .1)
    residual = op.adjoint(op.forward(solution) - y) + .1 * solution
    assert float(residual.norm()) < 2e-4


def test_scanner_nuisance_and_measurement_units_are_grid_independent(tmp_path):
    m, k = 8, 8
    meta = dict(r_value=8., beta=.5, fov_mm=[24., 24.], thickness_mm=3.,
                positions_lps_mm=[[0, 0, 0]], source_sha256='test', calibration_status='synthetic_fixture')
    raw = np.random.default_rng(31).normal(size=(1, 1, 4, m, k)) + 1j * np.random.default_rng(32).normal(size=(1, 1, 4, m, k))
    path = tmp_path / 'test.h5'
    with h5py.File(path, 'w') as source:
        source.attrs['metadata'] = json.dumps(meta)
        source.create_dataset('kspace', data=raw.astype(np.complex64))
    op1, y1, _, _, _, info1 = load_case(path, 0, image_shape=(8, 8))
    op2, y2, _, _, _, info2 = load_case(path, 0, image_shape=(12, 16))
    torch.testing.assert_close(y1, y2, atol=0, rtol=0)
    torch.testing.assert_close(op1.coils, op2.coils, atol=0, rtol=0)
    assert info1['magnitude_scale'] == info2['magnitude_scale']
    assert info1['gain'] == info2['gain']


def test_heldout_selection_uses_distinct_subjects_and_integer_units(tmp_path):
    records = [dict(subject=f's{i}', modality='T2', view='axial', key=f's{i}:axial:0') for i in range(3)]
    manifest = dict(subjects={'test': ['s0', 's1', 's2']}, records={'test': records})
    np.save(tmp_path / 'test.npy', np.full((3, 8, 12), 65535, np.uint16))
    clean, keys = select_clean(tmp_path, manifest, 'test', 2, 'cpu')
    assert keys == ['s0:axial:0', 's2:axial:0']
    torch.testing.assert_close(clean, torch.ones(2, 1, 8, 12))
