"""Crossed-chirp slice-integral PE encoding and Fourier RO, complex multicoil DC.

The scanner adapter is a header-informed reduced model, not waveform/B0 calibration.
The image is a real magnitude in [-1,1]; fixed coil factors include object phase.
"""
import math
import numpy as np
import torch

def encoding_matrices(m, k, h, w, r_value, beta=.5, focus_offset=0., device='cpu', dtype=torch.complex64):
    real = torch.float64
    nodes, weights = np.polynomial.legendre.leggauss(16)
    y = ((torch.arange(h, dtype=real, device=device)+.5)/h-.5)[:, None]
    y = y+torch.tensor(nodes/2, device=device)/h
    focus = (torch.arange(m, device=device, dtype=real)-m//2+focus_offset)/m
    resolving_power = 4*r_value*beta*(1-beta)
    a = (torch.sinc(resolving_power*(y[None]-focus[:, None, None]))*
         torch.tensor(weights/2, device=device)).sum(-1)*(m/h)
    x = (torch.arange(w, dtype=real, device=device)+.5)/w-.5-.5/k
    freq = torch.arange(k, dtype=real, device=device)-k//2
    # Same constant-image response at native and independent HR grids.
    f = torch.exp(2j*math.pi*freq[:, None]*x[None])*(math.sqrt(k)/w)
    return a.to(dtype), f.to(dtype)

def cg_solve(matvec, rhs, initial=None, iterations=40, tolerance=1e-6):
    x = torch.zeros_like(rhs) if initial is None else initial.clone()
    r = rhs-matvec(x)
    p = r.clone()
    dot = lambda a, b: (a.conj()*b).real.flatten(1).sum(1).reshape(-1, 1, 1, 1)
    rr = dot(r, r)
    threshold = dot(rhs, rhs).clamp_min(1e-20)*tolerance**2
    for _ in range(iterations):
        ap = matvec(p)
        active = rr > threshold
        alpha = torch.where(active, rr/dot(p, ap).clamp_min(1e-30), 0.)
        x = x+alpha*p
        r = r-alpha*ap
        new = dot(r, r)
        p = r+torch.where(active, new/rr.clamp_min(1e-30), 0.)*p
        rr = new
        if not bool((rr > threshold).any()):
            break
    return x

class XSPENOperator:
    def __init__(self, a, f, coils, mask=None, cg_iterations=40):
        self.a, self.f = a, f
        self.coils = coils[None] if coils.ndim == 3 else coils
        self.image_shape = tuple(self.coils.shape[-2:])
        self.device = a.device
        self.mask = torch.ones(a.shape[0], f.shape[0], device=a.device) if mask is None else mask.to(a.device)
        self.cg_iterations = cg_iterations
        if self.coils.shape[-2:] != (a.shape[1], f.shape[1]):
            raise ValueError('Coil and image grid mismatch')
        if self.mask.shape != (a.shape[0], f.shape[0]) or not self.mask.any():
            raise ValueError('Invalid sampling mask')
        if not all(torch.isfinite(v).all() for v in [a, f, self.coils, self.mask]):
            raise ValueError('Nonfinite operator')

    def complex_forward(self, image):
        temp = torch.matmul(self.a, image.to(self.a.dtype))
        return torch.matmul(temp, self.f.T)*self.mask

    def complex_adjoint(self, y):
        return torch.matmul(self.a.conj().T, torch.matmul(y*self.mask, self.f.conj()))

    def linear(self, x):
        return self.complex_forward(self.coils*x*.5)

    def forward(self, x):
        return self.linear(x+1.)

    def adjoint(self, y):
        return .5*(self.coils.conj()*self.complex_adjoint(y)).sum(1, keepdim=True).real

    def proximal(self, z, observation, rho):
        rho = torch.as_tensor(rho, device=z.device, dtype=z.dtype)
        if rho.numel() != 1 or float(rho) <= 0:
            raise ValueError('Positive scalar rho required')
        rhs = self.adjoint(observation-self.forward(torch.zeros_like(z)))+rho*z
        return cg_solve(lambda x: self.adjoint(self.linear(x))+rho*x, rhs,
                        initial=z, iterations=self.cg_iterations)

    def relative_residual(self, x, observation):
        return ((self.forward(x)-observation).abs().square().flatten(1).sum(1).sqrt()/
                observation.abs().square().flatten(1).sum(1).sqrt().clamp_min(1e-12))

def pixel_average_matrix(native, high, device='cpu', dtype=torch.float32):
    """Exact overlap averages for two piecewise-constant grids over the same FOV."""
    n = torch.arange(native, device=device, dtype=torch.float64)
    h = torch.arange(high, device=device, dtype=torch.float64)
    left = torch.maximum(n[:, None]/native, h[None]/high)
    right = torch.minimum((n[:, None]+1)/native, (h[None]+1)/high)
    return ((right-left).clamp_min(0)*native).to(dtype)

class NativeGridXSPENOperator(XSPENOperator):
    """Scanner finite-volume model with nuisance phase kept on its measured grid.

    y = A_native [S_native * (P m Q^T)] F_native^T.
    This uses a piecewise-constant native voxel/coil approximation. It avoids
    inventing unmeasured HR coil/object phase by interpolation. Resolution finer
    than this native image grid is supplied by the prior, not resolved by data.
    """
    def __init__(self, a, f, coils, image_shape=(128, 128), **kwargs):
        super().__init__(a, f, coils, **kwargs)
        self.image_shape = tuple(image_shape)
        dtype = self.coils.real.dtype
        self.p = pixel_average_matrix(a.shape[1], image_shape[0], a.device, dtype)
        self.q = pixel_average_matrix(f.shape[1], image_shape[1], a.device, dtype)

    def project(self, x):
        return torch.matmul(self.p, torch.matmul(x, self.q.T))

    def linear(self, x):
        return self.complex_forward(self.coils*self.project(x)*.5)

    def adjoint(self, y):
        native = super().adjoint(y)
        return torch.matmul(self.p.T, torch.matmul(native, self.q))

def synthetic_coils(h, w, count=4, device='cpu', dtype=torch.complex64):
    y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=device), torch.linspace(-1, 1, w, device=device), indexing='ij')
    coils = []
    for i in range(count):
        angle = 2*math.pi*i/count
        amp = torch.exp(-((x-.6*math.cos(angle))**2+(y-.6*math.sin(angle))**2)/1.8)
        coils.append(amp*torch.exp(1j*(.4*x*math.cos(angle)+.4*y*math.sin(angle)+.2*x*y)))
    c = torch.stack(coils).to(dtype)
    return c/c.abs().square().sum(0, keepdim=True).sqrt()
