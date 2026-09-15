"""Batch the preserved FGG algorithm without changing its summation order.

All PE lines, slices, volumes and receivers share trajectory weights. Unlike
FFT-of-basis followed by a matrix product, spreading the actual ADC samples
also preserves the scalar implementation's rounding before complex64 storage.
"""
from functools import lru_cache
import numpy as np

from spenpy._legacy.recon.gridding import smooth_trajectory, _place_regridded_block


@lru_cache(maxsize=256)
def _spread_plan(knots_tuple, nx):
    knots = np.asarray(knots_tuple, np.float64)
    r, m_sp = 2, 6
    tau = np.pi*m_sp/(nx*nx*r*(r-.5))
    m_r = r*nx
    kmin, kmax = float(np.min(knots)), float(np.max(knots))
    scale = (nx-1)/(kmax-kmin)
    shift = -nx/2-kmin*scale
    knots = scale*knots+shift
    knots = np.mod(2*np.pi*knots/nx, 2*np.pi)
    e3_pos = np.exp(-((np.pi*np.arange(1, m_sp+1)/m_r)**2)/tau)
    e3 = np.concatenate([e3_pos[:m_sp-1][::-1], np.ones(1), e3_pos])
    steps = []
    for knotx in knots:
        m1 = int(np.floor(m_r*knotx/(2*np.pi)))
        x = knotx-m1*np.pi/(m_r/2)
        e1 = np.exp(-(x*x)/(4*tau))
        e2_dummy = np.exp(x*np.pi/(m_r*tau))
        e2_dummy_inv = 1./e2_dummy
        e2 = np.empty(2*m_sp)
        e2[m_sp-1] = 1.
        for j in range(m_sp, 2*m_sp):
            e2[j] = e2_dummy*e2[j-1]
        for j in range(m_sp-2, -1, -1):
            e2[j] = e2[j+1]*e2_dummy_inv
        entries = []
        for l1 in range(1-m_sp, m_sp+1):
            lx = int((m1+l1+m_r/2) >= 0)
            rx = int((m1+l1) < m_r/2)
            row = int(m1+l1+(rx-lx)*m_r+m_r/2)
            entries.append((row, e2[m_sp+l1-1]*e3[m_sp+l1-1]))
        steps.append((e1, tuple(entries)))
    kx = np.arange(-nx/2, nx/2)
    e4 = np.sqrt(np.pi/tau)*np.exp(tau*(kx**2))
    return tuple(steps), e4


def _readout_batch(signal, knots, nx):
    signal = np.asarray(signal, np.complex128)
    shape = signal.shape
    flat = signal.reshape(shape[0], -1)
    steps, e4 = _spread_plan(tuple(np.asarray(knots, float)), nx)
    if len(steps) != shape[0]:
        raise ValueError('Trajectory and ADC sample count differ')
    spread = np.zeros((2*nx, flat.shape[1]), np.complex128)
    for datum, (e1, entries) in zip(flat, steps):
        v0 = datum*e1
        for row, e23 in entries:
            spread[row] += v0*e23
    transformed = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(spread, axes=0), axes=(0,)), axes=0)
    chop = int(np.floor(.5*nx+.5))
    transformed = transformed[chop:chop+nx]*e4[:, None]/(shape[0]*2)
    result = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(transformed, axes=0), axis=0), axes=0)
    result[:3] = 0
    result[-3:] = 0
    return result.reshape(nx, *shape[1:])


def regrid_all_preserving_scalar(raw, trajectory, matrix, segments, flavor):
    trajectory, values = smooth_trajectory(np.asarray(trajectory, float))
    n_out = int(np.floor(float(trajectory.max())+.5))
    if n_out <= 6:
        raise ValueError(f'Readout trajectory yields only {n_out} grid points; the reference 3+3 edge removal leaves no measured signal')
    if not np.isfinite(trajectory).all() or np.ptp(trajectory) <= 0:
        raise ValueError('Invalid readout trajectory')
    offset = float(values.max() if flavor == 'pv360' else trajectory.max())
    placement = int(np.floor(offset+.5))
    out = np.empty((int(matrix[0]), *raw.shape[1:]), np.complex64)
    indices = np.arange(raw.shape[1])
    reverse = (indices//segments)%2 == 0 if segments%2 == 0 else indices%2 == 1
    for mask, knots in [(~reverse, trajectory), (reverse, -trajectory[::-1]+offset)]:
        block = _readout_batch(raw[:, mask], knots, n_out)
        placed = _place_regridded_block(block.reshape(n_out, -1), int(matrix[0]), placement)
        out[:, mask] = placed.reshape(int(matrix[0]), *block.shape[1:])
    return out
