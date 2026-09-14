import numpy as np
import pytest
import torch
from operators import XSPENOperator, cg_solve, encoding_matrices, synthetic_coils

torch.set_num_threads(2)

def make_op(masked=False):
    a, f = encoding_matrices(8, 8, 8, 8, 6., dtype=torch.complex128)
    c = synthetic_coils(8, 8, 3, dtype=torch.complex128)
    mask = torch.ones(8, 8, dtype=torch.float64)
    if masked:
        mask[1::2] = 0
    return XSPENOperator(a, f, c, mask, cg_iterations=180)

@pytest.mark.parametrize('masked', [False, True])
def test_real_adjoint_and_gradient(masked):
    torch.manual_seed(12)
    op = make_op(masked)
    x = torch.randn(2, 1, 8, 8, dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, 3, 8, 8, dtype=torch.complex128)
    left = (op.linear(x).conj()*y).real.sum()
    right = (x*op.adjoint(y)).sum()
    torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-10)
    grad = torch.autograd.grad((op.forward(x)-y).abs().square().sum(), x)[0]
    torch.testing.assert_close(grad, 2*op.adjoint(op.forward(x)-y), rtol=1e-10, atol=1e-10)

@pytest.mark.parametrize('masked', [False, True])
def test_affine_proximal_against_dense_solve(masked):
    torch.manual_seed(23)
    op = make_op(masked)
    basis = torch.eye(64, dtype=torch.float64).reshape(64, 1, 8, 8)
    linear = op.linear(basis).flatten(1).T
    gram = (linear.conj().T@linear).real
    z = torch.randn(1, 1, 8, 8, dtype=torch.float64)
    y = torch.randn(1, 3, 8, 8, dtype=torch.complex128)
    rho = .03
    rhs = op.adjoint(y-op.forward(torch.zeros_like(z))).flatten()+rho*z.flatten()
    expected = torch.linalg.solve(gram+rho*torch.eye(64), rhs).reshape_as(z)
    torch.testing.assert_close(op.proximal(z, y, rho), expected, rtol=2e-4, atol=3e-5)

def test_native_readout_matches_legacy_ifft():
    _, f = encoding_matrices(8, 8, 8, 8, 6., dtype=torch.complex128)
    x = torch.randn(8, dtype=torch.complex128)
    expected = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(x), norm='ortho'))
    torch.testing.assert_close(f@x, expected, rtol=1e-10, atol=1e-10)

def test_sinc_kernel_against_explicit_slice_and_pixel_integrals():
    m, h, r = 8, 16, 6.
    a, _ = encoding_matrices(m, 8, h, 16, r, dtype=torch.complex128)
    z = (np.arange(32768)+.5)/32768-.5
    nodes, weights = np.polynomial.legendre.leggauss(16)
    for row, col in [(0, 0), (3, 8), (7, 13)]:
        y = (col+.5+nodes/2)/h-.5
        focus = (row-m//2)/m
        phase = r*(y[:, None]-focus)*z
        value = (np.exp(2j*np.pi*phase).mean(1)*weights/2).sum()*m/h
        assert abs(complex(a[row, col])-value) < 2e-8

def test_independent_hr_grid_shapes_and_cg_residual():
    a, f = encoding_matrices(12, 16, 32, 32, 12.)
    op = XSPENOperator(a, f, synthetic_coils(32, 32), cg_iterations=120)
    x = torch.randn(1, 1, 32, 32)
    y = op.forward(x)
    assert y.shape == (1, 4, 12, 16)
    z = torch.zeros_like(x)
    got = op.proximal(z, y, .1)
    normal_residual = op.adjoint(op.forward(got)-y)+.1*(got-z)
    assert float(normal_residual.norm()) < 2e-4

@pytest.mark.parametrize('slope', [.2, 12.8])
def test_even_odd_phase_from_known_smooth_complex_signal(slope):
    from scanner import correct_even_odd
    yy, xx = torch.meshgrid(torch.arange(32)/16-1, torch.arange(32)/16-1, indexing='ij')
    clean = torch.exp(1j*(.15*xx+.1*yy))[None, None].repeat(1, 3, 1, 1)
    field = .7+slope*xx+.1*yy
    field[::2] = 0
    corrupted = clean*torch.exp(1j*field)
    got, estimated, _ = correct_even_odd(corrupted)
    torch.testing.assert_close(got, clean, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(estimated, field, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got.abs(), corrupted.abs(), rtol=1e-6, atol=1e-6)

def test_native_grid_projection_adjoint_and_dense_proximal():
    from operators import NativeGridXSPENOperator, pixel_average_matrix
    a, f = encoding_matrices(6, 8, 6, 8, 6., dtype=torch.complex128)
    coils = synthetic_coils(6, 8, 3, dtype=torch.complex128)
    # A rapid, native-only phase is intentionally retained without interpolation.
    coils = coils*torch.exp(2.5j*torch.arange(6, dtype=torch.float64)[:, None])
    op = NativeGridXSPENOperator(a, f, coils, image_shape=(8, 12), cg_iterations=200)
    torch.testing.assert_close(op.project(torch.ones(1, 1, 8, 12, dtype=torch.float64)), torch.ones(1, 1, 6, 8, dtype=torch.float64))
    x = torch.randn(1, 1, 8, 12, dtype=torch.float64)
    y = torch.randn(1, 3, 6, 8, dtype=torch.complex128)
    torch.testing.assert_close((op.linear(x).conj()*y).real.sum(), (x*op.adjoint(y)).sum(), rtol=1e-10, atol=1e-10)
    basis = torch.eye(96, dtype=torch.float64).reshape(96, 1, 8, 12)
    linear = op.linear(basis).flatten(1).T
    rhs = op.adjoint(y-op.forward(torch.zeros_like(x))).flatten()+.1*x.flatten()
    expected = torch.linalg.solve((linear.conj().T@linear).real+.1*torch.eye(96), rhs).reshape_as(x)
    torch.testing.assert_close(op.proximal(x, y, .1), expected, rtol=3e-5, atol=3e-5)
