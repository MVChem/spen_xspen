"""Numerical contract checks before using SPEN in an inverse solver."""
import numpy as np
import pytest
import torch
from operators import SpenMagnitudeOperator, load_scanner_case, SCANNER, CACHE
from model import EDMPrior

torch.set_num_threads(4)


def small_operator(masked=False):
    gen = torch.Generator().manual_seed(37)
    a = torch.randn(7,6, generator=gen) + 1j*torch.randn(7,6,generator=gen)
    c = torch.randn(2,6,5,generator=gen) + 1j*torch.randn(2,6,5,generator=gen)
    mask = torch.tensor([True,False,True,False,True,False,True]) if masked else None
    return SpenMagnitudeOperator(a/6,c/3,mask)


@pytest.mark.parametrize('masked',[False,True])
def test_real_complex_adjoint_and_gradient(masked):
    op = small_operator(masked)
    torch.manual_seed(91)
    x = torch.randn(3,1,6,5)
    y = torch.randn_like(op.forward(x))
    lhs = (op.linear(x).conj()*y).sum().real
    rhs = (x*op.adjoint(y)).sum()
    torch.testing.assert_close(lhs,rhs,atol=1e-5,rtol=1e-5)
    v = x.clone().requires_grad_()
    actual = torch.autograd.grad(op.loss(v,y).sum(),v)[0]
    torch.testing.assert_close(actual,op.gradient(x,y),atol=1e-5,rtol=1e-5)
    direction = torch.randn_like(x)
    eps=.001
    fd=(op.loss(x+eps*direction,y).sum()-op.loss(x-eps*direction,y).sum())/(2*eps)
    torch.testing.assert_close(fd,(actual*direction).sum(),atol=.015,rtol=.003)


@pytest.mark.parametrize('masked',[False,True])
def test_proximal_against_independent_dense_real_solve(masked):
    op=small_operator(masked)
    n=30
    eye=torch.eye(n).reshape(n,1,6,5)
    columns=op.linear(eye).flatten(1).numpy().T
    matrix=np.concatenate([columns.real,columns.imag],axis=0).astype(np.float64)
    z=torch.randn(1,1,6,5)
    y=torch.randn_like(op.forward(z))
    y0=(y-op.forward(torch.zeros_like(z))).flatten().numpy()
    real_y=np.concatenate([y0.real,y0.imag])
    rho=.013
    expected=np.linalg.solve(matrix.T@matrix+rho*np.eye(n),matrix.T@real_y+rho*z.flatten().numpy())
    actual=op.proximal(z,y,rho)
    np.testing.assert_allclose(actual.flatten().numpy(),expected,rtol=2e-4,atol=2e-4)
    normal_res=op.gradient(actual,y)/2+rho*(actual-z)
    assert normal_res.abs().max()<1e-5


@pytest.mark.integration
@pytest.mark.skipif(not (SCANNER/'slice_13.mat').exists() or not (CACHE/'slice_13.npz').exists(), reason='External scanner reference data not configured')
def test_scanner_scaling_and_data_fit():
    op,y,anchor,_,_,meta=load_scanner_case(13,'cpu')
    assert meta['normalization_error']<2e-5
    out=op.proximal(anchor,y,.01)
    assert float(op.relative_residual(out,y))<float(op.relative_residual(anchor,y))


def test_edm_interface_and_extreme_sigmas():
    torch.set_num_threads(2)
    net=EDMPrior(base_ch=8).eval()
    x=torch.randn(2,1,96,96)
    with torch.no_grad():
        for sigma in [.002,.1,80.]:
            y=net(x,sigma)
            assert y.shape==x.shape and torch.isfinite(y).all()
    y=net(x,torch.tensor([.1,1.]))
    y.mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
