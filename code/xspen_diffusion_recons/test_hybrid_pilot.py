"""Independent equations for the hybrid RO-domain finite-volume adapter."""
import numpy as np
import torch

from hybrid_pilot import make_operator, traditional
from operators import pixel_average_matrix


def test_hybrid_native_forward_real_adjoint_and_dense_proximal():
    torch.manual_seed(91)
    a = torch.randn(6, 6, dtype=torch.complex64)
    coils = torch.randn(2, 6, 5, dtype=torch.complex64)
    case = dict(a=a.numpy(), coils=coils.numpy())
    op = make_operator(case, 8, 'cpu')
    x = torch.randn(1, 1, 8, 8)
    y = torch.randn(1, 2, 6, 5, dtype=torch.complex64)
    p, q = pixel_average_matrix(6, 8), pixel_average_matrix(5, 8)
    expected = a @ (coils * (p @ ((x + 1) / 2) @ q.T))
    torch.testing.assert_close(op.forward(x), expected, atol=2e-6, rtol=2e-6)
    left = (op.linear(x).conj() * y).sum().real
    right = (x * op.adjoint(y)).sum()
    torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)
    basis = torch.eye(64).reshape(64, 1, 8, 8)
    complex_matrix = op.linear(basis).reshape(64, -1).T
    matrix = torch.cat([complex_matrix.real, complex_matrix.imag])
    observed = (y - op.forward(torch.zeros_like(x))).flatten()
    observed = torch.cat([observed.real, observed.imag])
    rho = .2
    expected = torch.linalg.solve(matrix.T @ matrix + rho * torch.eye(64),
                                  matrix.T @ observed + rho * x.flatten())
    actual = op.proximal(x, y, rho).flatten()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_traditional_recovers_complex_diagonal_model_and_reports_residual():
    a = torch.diag(torch.tensor([1+1j, 2-1j, .5+.7j], dtype=torch.complex128))
    coils = torch.randn(2, 3, 4, dtype=torch.complex128)
    y = a @ coils
    inverse = torch.linalg.inv(a)
    recovered, inva, checks = traditional(a, inverse, y)
    ridge = .01 * torch.linalg.matrix_norm(a, ord=2).square()
    expected = coils * (a.diagonal().abs().square() / (a.diagonal().abs().square() + ridge))[None, :, None]
    torch.testing.assert_close(recovered, expected)
    torch.testing.assert_close(inva, coils.abs().square().sum(0).sqrt())
    assert checks['tikhonov_normal_residual'] < 1e-12
    assert np.isfinite(checks['inva_gain']).all()
