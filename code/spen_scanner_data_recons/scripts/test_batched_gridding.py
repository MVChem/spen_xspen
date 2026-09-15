"""Require scalar FGG reproduction after the pipeline's complex64 storage.

Comparison calls the original scalar implementation for every separate
slice/volume/receiver/echo. Tests intentionally use exact array equality,
because small gridding roundoff can alter downstream PhaseMap optimization.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent / "spenpy"))

from batched_gridding import regrid_all_preserving_scalar
from spenpy._legacy.bruker.param import read_pv_param
from spenpy._legacy.bruker.raw import read_bruker_kspace_pv360_fid_multichannel
from spenpy._legacy.recon.gridding import one_d_regridding_pv360, one_d_regridding_pv6
from segmented_raw_reader import read_segmented_raw

DATA = PROJECT.parent / "data/spen_acquired_260915"


def scalar_frames(raw, trajectory, matrix, segments, flavor):
    reference = one_d_regridding_pv360 if flavor == "pv360" else one_d_regridding_pv6
    expected = np.empty((int(matrix[0]), *raw.shape[1:]), np.complex64)
    for indices in np.ndindex(*raw.shape[2:]):
        selector = (slice(None), slice(None), *indices)
        expected[selector] = reference(raw[selector], trajectory, segments, matrix)
    return expected


@pytest.mark.parametrize("flavor", ["pv5", "pv360"])
@pytest.mark.parametrize("segments", [1, 3, 4, 5])
@pytest.mark.parametrize("grid_max", [18.2, 19.2])
def test_full_int32_all_frame_axes_match_scalar_exactly(flavor, segments, grid_max):
    rng = np.random.default_rng(20260915 + segments)
    shape = (24, 3 * segments, 2, 2, 2, 2)
    real = rng.integers(-(2**31), 2**31, size=shape, dtype=np.int32)
    imag = rng.integers(-(2**31), 2**31, size=shape, dtype=np.int32)
    real.flat[0], imag.flat[0] = 2**31 - 1, -(2**31)
    raw = real.astype(np.float64) + 1j * imag.astype(np.float64)
    trajectory = np.linspace(0.15, grid_max, shape[0]) + .12 * np.sin(np.linspace(0, np.pi, shape[0]))
    matrix = [24, shape[1]]
    actual = regrid_all_preserving_scalar(raw, trajectory, matrix, segments, flavor)
    expected = scalar_frames(raw, trajectory, matrix, segments, flavor)
    assert actual.dtype == np.complex64
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("flavor", ["pv5", "pv360"])
def test_noncontiguous_small_signal_and_trailing_singletons_match(flavor):
    rng = np.random.default_rng(915)
    raw = (rng.normal(size=(32, 20, 2, 1, 2, 2))
           + 1j * rng.normal(size=(32, 20, 2, 1, 2, 2))).astype(np.complex64)
    raw = raw[:, ::2, ::-1, :, :, ::-1]
    assert not raw.flags.c_contiguous
    trajectory = np.linspace(0, 27, 32) + .08 * np.sin(np.linspace(0, np.pi, 32))
    actual = regrid_all_preserving_scalar(raw, trajectory, [32, 10], 4, flavor)
    np.testing.assert_array_equal(actual, scalar_frames(raw, trajectory, [32, 10], 4, flavor))


def _raw_case(relative):
    scan_dir = DATA / "raw" / relative
    p = lambda name: read_pv_param(str(scan_dir), name)
    segments = int(p("NSegments") or 1)
    diffusion = p("PVM_DwNDiffExp") or p("DwNDiffExp") or 1
    params = {"matrix_ro_pe": p("PVM_Matrix"), "n_segments": segments,
              "slices": int(np.sum(p("PVM_SPackArrNSlices") or 1)),
              "volumes": int(diffusion) * int(p("PVM_NRepetitions") or 1),
              "coils": int(p("PVM_EncNReceivers") or 1),
              "echoes": int(p("PVM_NEchoImages") or 1)}
    if segments > 1:
        raw, _ = read_segmented_raw(scan_dir, params)
    else:
        # Do not call prepare_scan: it already applies the function under test.
        old = read_bruker_kspace_pv360_fid_multichannel(str(scan_dir))
        echoes = params["echoes"]
        if echoes == 1:
            assert old.ndim == 5 and old.shape[-1] == 1
            flat = old
        else:
            assert old.ndim == 6 and old.shape[-2:] == (echoes, 1)
            flat = old[..., 0]
        raw = flat.reshape(flat.shape[0], flat.shape[1], params["slices"],
                           params["volumes"], params["coils"], echoes, order="F")
    assert raw.dtype == np.complex128
    return scan_dir, params, raw


REAL_CASES = [
    # All five MAT cases that exposed PhaseMap sensitivity to summation order.
    ("20231207_150817_lxj_spen_231207_1_1_1/58", "pv360", None),
    ("20231207_185834_lxj_spen_mouse2_231207_1_1_1/33", "pv360", None),
    ("20231207_185834_lxj_spen_mouse2_231207_1_1_1/36", "pv360", None),
    ("20231207_185834_lxj_spen_mouse2_231207_1_1_1/40", "pv360", None),
    ("20231207_221933_lxj_spen_mouse3_231207_1_1_1/31", "pv360", None),
    # Explicit PV5 calibration pairing, all ten slices and all four receivers.
    ("lxj_motionRARE_SPEN_230904.lG2/18", "pv5", 17),
    # Both echoes remain present for PV6 odd/even segmented acquisitions.
    ("20220509_151932_lxj_SPEN_multi_shot_0509_1_1/5", "pv5", None),
    ("20220724_161419_lxj_SPEN_xly_0724_1_2/8", "pv5", None),
    # Native packet geometry contains 95 PE lines despite declared PE=96.
    ("20220721_095453_lxj_SPEN_diffusion_test_0721_water_1_1/35", "pv360", None),
]


@pytest.mark.skipif(not DATA.exists(), reason="Imported scanner data are unavailable")
@pytest.mark.parametrize("relative,flavor,trajectory_id", REAL_CASES)
def test_real_acquisitions_match_every_scalar_frame_exactly(relative, flavor, trajectory_id):
    scan_dir, params, raw = _raw_case(relative)
    trajectory_dir = scan_dir if trajectory_id is None else scan_dir.parent / str(trajectory_id)
    trajectory = read_pv_param(str(trajectory_dir), "PVM_EpiTrajAdjkx")
    expected = scalar_frames(raw, trajectory, params["matrix_ro_pe"], params["n_segments"], flavor)
    actual = regrid_all_preserving_scalar(raw, trajectory, params["matrix_ro_pe"], params["n_segments"], flavor)
    assert actual.shape[2:] == tuple(params[k] for k in ("slices", "volumes", "coils", "echoes"))
    np.testing.assert_array_equal(actual, expected)
