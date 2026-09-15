import numpy as np
import pytest

from image_processing import assess_slice, fit_plane, volume_window


def test_physical_circle_stays_round_with_anisotropic_pixels():
    y, x = np.mgrid[:80, :160]
    # Input is rectangular in pixels but square in physical FOV.
    circle = ((((y - 39.5) * 2) ** 2 + (x - 79.5) ** 2) <= 30 ** 2).astype(np.float32)
    output, info = fit_plane(circle, [2, 1], 1)
    yy, xx = np.nonzero(output > 32767)
    assert abs(np.ptp(yy) - np.ptp(xx)) <= 1
    assert output.shape == (192, 192) and output.dtype == np.uint16
    assert info['native_fov_yx'] == [160, 160]


def test_full_fov_padding_preserves_center_and_intensity():
    plane = np.ones((80, 160), np.float32) * 25
    output, info = fit_plane(plane, [1, 1], 100, bit_depth=8)
    assert output[96, 96] == 64
    assert not output[:40].any() and not output[-40:].any()
    assert info['normalization_window'] == [0., 100.]


@pytest.mark.parametrize('value', [0., 1., np.nan])
def test_blank_constant_and_invalid_slices_rejected(value):
    ok, reason, stats = assess_slice(np.full((192, 192), value), 1.)
    assert not ok


def test_signal_phantom_accepted_and_window_uses_positive_voxels():
    plane = np.zeros((100, 100), np.float32)
    plane[25:75, 25:75] = np.linspace(1, 100, 2500).reshape(50, 50)
    upper = volume_window(plane[None])
    assert 99 < upper < 100
    assert assess_slice(plane, upper)[0]


def test_antialias_and_geometry_validation():
    checker = (np.indices((768, 768)).sum(axis=0) % 2).astype(np.float32)
    output, info = fit_plane(checker, [1, 1], 1)
    assert output[10:-10, 10:-10].std() < 100
    assert not info['upsampled']
    with pytest.raises(ValueError):
        fit_plane(checker, [0, 1], 1)
