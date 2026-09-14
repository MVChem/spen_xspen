import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from traditional_phase import build_matrices, reconstruct_traditional, reconstruct_even_odd_inva

torch.set_num_threads(2)


@pytest.mark.parametrize('shape,r_value', [((60, 64), 60.), ((46, 48), 46.)])
def test_native_readout_and_complex_adjoint(shape, r_value):
    m, k = shape
    matrices = build_matrices(m, k, r_value, dtype=torch.complex128)
    generator = torch.Generator().manual_seed(72)
    raw = torch.randn(3, m, k, generator=generator, dtype=torch.complex128)
    actual = raw @ matrices['readout'].conj()
    expected = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(raw, dim=-1), dim=-1, norm='ortho'), dim=-1)
    torch.testing.assert_close(actual, expected, atol=3e-13, rtol=3e-13)
    torch.testing.assert_close(actual @ matrices['readout'].T, raw, atol=3e-13, rtol=3e-13)
    assert matrices['inv_a'].shape == (m, m)
    assert matrices['inv_odd'].shape == (m // 2, m // 2)
    assert matrices['inv_even'].shape == (m // 2, m // 2)
    assert not torch.allclose(matrices['inv_odd'], matrices['inv_even'])


def test_phase_sign_corrected_parity_and_complex_coil_inverse():
    m, k, c = 12, 20, 3
    y, x = torch.meshgrid(torch.linspace(-1., 1., m // 2), torch.linspace(-1., 1., k), indexing='ij')
    phi = .63 + .15*x - .11*y + .03*x*y
    odd = torch.stack([(1 + .1*x + .03*y) * torch.exp(1j*(q*.7 + .1*x)) for q in range(c)], -1).to(torch.complex128)
    frame = torch.zeros(m, k, c, dtype=torch.complex128)
    frame[::2] = odd
    frame[1::2] = odd * torch.exp(1j*phi[..., None])
    inverse = torch.diag(torch.exp(1j*torch.arange(m, dtype=torch.float64)*.19))
    half = torch.eye(m // 2, dtype=torch.complex128)
    result = reconstruct_even_odd_inva(frame, inverse, half, half, estimator='quadratic', coil_combination='rss')
    torch.testing.assert_close(result.corrected_ro_image[::2], frame[::2], atol=0, rtol=0)
    torch.testing.assert_close(result.corrected_ro_image.abs(), frame.abs(), atol=2e-7, rtol=2e-7)
    torch.testing.assert_close(result.corrected_ro_image[1::2], odd, atol=2e-3, rtol=2e-3)
    expected = torch.einsum('ym,mxc->yxc', inverse, result.corrected_ro_image)
    torch.testing.assert_close(result.coil_images, expected, atol=0, rtol=0)
    torch.testing.assert_close(result.magnitude, expected.abs().square().sum(-1).sqrt())


def test_empty_input_and_ablation_are_finite_and_native():
    raw = np.zeros((2, 8, 12), np.complex64)
    result = reconstruct_traditional(raw, dict(r_value=8., beta=.5), magnitude_scale=3.)
    assert result['magnitude'].shape == (8, 12)
    assert result['coil_images'].shape == (2, 8, 12)
    assert not result['magnitude'].any()
    assert not result['windowed_adjoint_only'].any()
    assert result['metadata']['phase_map_supplied'] is False
    assert 'not a Tikhonov' in result['metadata']['inv_a_kind']
