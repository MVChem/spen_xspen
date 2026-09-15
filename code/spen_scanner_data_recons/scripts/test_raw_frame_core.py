"""Regression of frame axes, echo sign and equivalent trajectory evaluation."""
from pathlib import Path

import numpy as np
import pytest
import torch

from raw_frame_core import (prepare_scan, reconstruct_frame, _regrid_all,
                            _normalize_legacy, _read_segmented_multiecho)
from spenpy._legacy.recon.gridding import one_d_regridding_pv360, one_d_regridding_pv6
from spenpy._legacy.recon.spen_recon import reconstruct_odd_segments

DATA = Path(__file__).resolve().parents[2] / 'data/spen_acquired_260915'
PILOT = Path(__file__).resolve().parents[1] / 'runs/raw_pilot_260915/arrays'


def test_volume_coil_echo_axes_are_not_folded_together():
    canonical = np.arange(7*8*3*2*4*2).reshape(7, 8, 3, 2, 4, 2)
    old = canonical.reshape(7, 8, 3, 8, 2, 1, order='F')
    np.testing.assert_array_equal(_normalize_legacy(old, 4, 2, 1), canonical)


@pytest.mark.parametrize('segments', [4, 5])
def test_segmented_multiecho_acquisition_packets_keep_all_frames(tmp_path, segments):
    # Emit acquisition packets in the scanner's outer-loop order. Distinct
    # values identify each volume, slice, coil, echo, PE line and ADC sample.
    ro, pe, slices, volumes, coils, echoes = 6, segments*4, 2, 2, 2, 2
    order = [1, 0]
    expected = np.empty((ro, pe, slices, volumes, coils, echoes), complex)
    packets = []
    for volume in range(volumes):
        for shot in range(segments):
            for acquired_slice in range(slices):
                for echo in range(echoes):
                    for coil in range(coils):
                        for line in range(pe//segments):
                            values = 1 + np.arange(ro) + 10*line + 100*shot + 1000*coil + 10000*echo + 100000*acquired_slice + 1000000*volume
                            values = values + 1j*(values+2)
                            reflected = line % 2 == (0 if segments % 2 == 0 or shot % 2 else 1)
                            expected[:, shot+line*segments, order[acquired_slice], volume, coil, echo] = values[::-1] if reflected else values
                            for value in values:
                                packets.extend([int(value.real), int(value.imag)])
    np.asarray(packets, dtype='<i4').tofile(tmp_path/'rawdata.job0')
    (tmp_path/'method').write_text('##$PVM_ObjOrderList=( 2 )\n1 0\n')
    params = dict(matrix_ro_pe=[ro, pe], n_segments=segments, slices=slices,
                  volumes=volumes, coils=coils, echoes=echoes)
    actual = _read_segmented_multiecho(tmp_path, params)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('segments', [1, 4, 5])
@pytest.mark.parametrize('flavor,reference', [('pv360', one_d_regridding_pv360), ('pv5', one_d_regridding_pv6)])
def test_batched_regridding_matches_preserved_scalar_algorithm(segments, flavor, reference):
    rng = np.random.default_rng(915)
    raw = (rng.normal(size=(32, 20, 2, 1, 2, 2)) + 1j*rng.normal(size=(32, 20, 2, 1, 2, 2))).astype(np.complex64)
    trajectory = np.linspace(0, 27, 32) + .08*np.sin(np.linspace(0, np.pi, 32))
    result = _regrid_all(raw, trajectory, [32, 20], segments, flavor)
    for sl, coil, echo in [(0, 0, 0), (1, 1, 1)]:
        expected = reference(raw[:, :, sl, 0, coil, echo], trajectory, segments, [32, 20])
        np.testing.assert_allclose(result[:, :, sl, 0, coil, echo], expected, atol=2e-7, rtol=2e-6)


@pytest.mark.skipif(not PILOT.exists(), reason='Imported pilot data are not available')
@pytest.mark.parametrize('name,relative,flavor,sl,vol', [
    ('mouse96_fov16', '20240321_204022_lxj_spen_mouse_240321_1_1_1/15', 'pv360', 0, 0),
    ('pv5_five_slices', 'lxj_motionRARE_SPEN_230904.lG2/16', 'pv5', 4, 0),
])
def test_existing_pilot_complex_regression(name, relative, flavor, sl, vol):
    torch.set_num_threads(1)
    path = DATA/'raw'/relative
    context = prepare_scan(path, flavor, path.parent/'15' if name == 'pv5_five_slices' else None)
    arrays, metadata = reconstruct_frame(context, sl, vol, 0)
    old = np.load(PILOT/f'{name}_v{vol:02d}_s{sl:02d}.npz')
    assert metadata['phase_map_status'] == 'applied'
    for key in ('rofft_original', 'rofft_corrected', 'inva_corrected', 'encoding'):
        relative_error = np.linalg.norm(arrays[key]-old[key]) / np.linalg.norm(old[key])
        assert relative_error < 1e-6, (key, relative_error)


@pytest.mark.skipif(not DATA.exists(), reason='Imported source data are not available')
def test_diffusion_second_volume_against_original_double_precision_reader():
    # The first pilot went through a complex64 public acquisition wrapper.
    # Validate the corrected double-ADC path against the original scalar
    # gridding/phase pipeline, rather than freezing that early rounding.
    from spenpy._legacy.bruker.raw import read_bruker_kspace_pv360_fid_multichannel
    from spenpy._legacy.bruker.param import read_pv_param
    torch.set_num_threads(1)
    path = DATA/'raw/20220721_095453_lxj_SPEN_diffusion_test_0721_water_1_1/17'
    raw = read_bruker_kspace_pv360_fid_multichannel(str(path))
    trajectory = read_pv_param(str(path), 'PVM_EpiTrajAdjkx')
    sampled = np.stack([one_d_regridding_pv360(raw[:, :, 0, v, 0], trajectory, 1, [64, 64])
                        for v in range(2)], axis=2).astype(np.complex64)
    old = reconstruct_odd_segments(str(path), kfield=sampled[:, :, None, :, None],
                                   input_stage='regridded_kspace')
    arrays, _ = reconstruct_frame(prepare_scan(path), 0, 1, 0)
    for key, expected in [('rofft_original', old.roffted_data_origin),
                          ('rofft_corrected', old.roffted_data_corrected)]:
        expected = expected.numpy()[:, :, 0, 1:2]
        assert np.linalg.norm(arrays[key]-expected)/np.linalg.norm(expected) < 1e-6


@pytest.mark.skipif(not DATA.exists(), reason='Imported source data are not available')
def test_second_echo_matches_legacy_echo_loop_without_discarding_first_echo():
    torch.set_num_threads(1)
    path = DATA/'raw/20211203_152911_lxj_SPEN_water_20211203_1_1/27'
    context = prepare_scan(path)
    assert context['counts']['echoes'] == 2
    first, _ = reconstruct_frame(context, 0, 0, 0)
    second, meta = reconstruct_frame(context, 0, 0, 1)
    raw = context['regridded_samples'][:, :, :, 0, :, :]
    old = reconstruct_odd_segments(str(path), kfield=raw, input_stage='regridded_kspace')
    np.testing.assert_allclose(second['rofft_corrected'], old.roffted_data_corrected.numpy()[:, :, 0, :], atol=1e-7, rtol=1e-7)
    assert meta['even_echo_pe_reversed']
    assert not np.allclose(first['rofft_original'], second['rofft_original'])
    assert not np.allclose(first['encoding'], second['encoding'])


@pytest.mark.skipif(not DATA.exists(), reason='Imported source data are not available')
def test_unvalidated_protocols_never_claim_phase_map():
    cases = [('20220509_151932_lxj_SPEN_multi_shot_0509_1_1/5', 'partial_inva_without_phase'),
             ('lxj_SPEN_test_230103.hK2/2', 'preview_only')]
    for relative, status in cases:
        context = prepare_scan(DATA/'raw'/relative)
        assert context['scope']['status'] == status
        for echo in range(context['counts']['echoes']):
            arrays, metadata = reconstruct_frame(context, 0, 0, echo)
            assert metadata['phase_map_status'] == 'not_applied'
            assert 'inva_corrected' not in arrays
            assert np.isfinite(arrays['rofft_original']).all()


@pytest.mark.skipif(not DATA.exists(), reason='Imported source data are not available')
def test_archived_scan8_redundant_counters_use_verified_thirteen_volumes():
    context = prepare_scan(DATA/'raw/lxj_motionRARE_SPEN_230904.lG2/8', 'pv5')
    assert context['counts']['volumes'] == 13
    assert context['dimension_correction']['declared_method_volumes'] == 169
    arrays, metadata = reconstruct_frame(context, 0, 12, 0)
    assert arrays['sorted_samples'].shape == (64, 64, 4)
    assert metadata['volume_index'] == 12


@pytest.mark.skipif(not DATA.exists(), reason='Imported source data are not available')
def test_large_int32_adc_preserves_reference_phase_optimizer_result():
    # Rounding these ~8e7 ADC integers to complex64 before gridding perturbs
    # the phase optimizer enough to change the corrected signal by 11%.
    import json
    from scipy.io import loadmat
    torch.set_num_threads(1)
    relative = 'raw/20231011_130630_lxj_SPEN_data_231011_3_1_3/72'
    record = next(r for r in json.loads((DATA/'manifest.json').read_text())['records']
                  if r['raw_scan_path'] == relative)
    context = prepare_scan(DATA/relative)
    assert context['sorted_samples'].dtype == np.complex128
    assert np.abs(context['sorted_samples']).max() > 2**24
    arrays, _ = reconstruct_frame(context, 0, 0, 0)
    reference = loadmat(DATA/record['mat_path'])
    for key, old_key in [('rofft_original', 'spen_original_signal_rofft'),
                         ('rofft_corrected', 'spen_phase_corrected_signal_rofft'),
                         ('inva_corrected', 'traditional_sr_data')]:
        old = reference[old_key][:, :, 0, :]
        error = np.linalg.norm(arrays[key]-old) / np.linalg.norm(old)
        assert error < 1e-6, (key, error)


@pytest.mark.skipif(not DATA.exists(), reason='Imported source data are not available')
def test_degenerate_trajectory_does_not_turn_nonzero_raw_into_successful_blank_recon():
    context = prepare_scan(DATA/'raw/20230911_092258_lxj_spen_test_230911_1_1_1/72')
    assert context['scope']['status'] == 'preview_only'
    assert 'leaves no measured signal' in context['scope']['regrid_error']
    arrays, meta = reconstruct_frame(context, 0, 0, 0)
    assert np.linalg.norm(arrays['rofft_original']) > 0
    assert 'inva_corrected' not in arrays
