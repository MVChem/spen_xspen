"""Independent complex normal-equation checks for the reconstruction solver."""
import numpy as np
import pytest

from reconstruct_raw import tikhonov_reconstruct


def test_complex_tikhonov_matches_independent_normal_solve():
    rng = np.random.default_rng(915)
    # Rectangular, complex A makes conjugation and axis mistakes observable.
    a = rng.normal(size=(7, 5)) + 1j * rng.normal(size=(7, 5))
    truth = rng.normal(size=(5, 4, 3)) + 1j * rng.normal(size=(5, 4, 3))
    y = np.einsum('ij,jrc->irc', a, truth)
    y += 0.01 * (rng.normal(size=y.shape) + 1j * rng.normal(size=y.shape))
    x, diagnostics = tikhonov_reconstruct(a, y, 0.01)
    expected = np.linalg.solve(
        a.conj().T @ a + 0.01 * np.linalg.norm(a, 2)**2 * np.eye(a.shape[1]),
        a.conj().T @ y.reshape(7, -1),
    ).reshape(truth.shape)
    np.testing.assert_allclose(x, expected, rtol=1e-12, atol=1e-12)
    assert diagnostics['relative_normal_residual'] < 1e-12
    stronger, _ = tikhonov_reconstruct(a, y, 0.1)
    assert np.linalg.norm(stronger) < np.linalg.norm(x)


def test_encoding_unit_change_preserves_tikhonov_solution():
    rng = np.random.default_rng(14)
    a = rng.normal(size=(8, 6)) + 1j*rng.normal(size=(8, 6))
    y = rng.normal(size=(8, 5, 2)) + 1j*rng.normal(size=(8, 5, 2))
    x, first = tikhonov_reconstruct(a, y, 0.03)
    changed, second = tikhonov_reconstruct(100*a, 100*y, 0.03)
    np.testing.assert_allclose(x, changed, rtol=1e-12, atol=1e-12)
    assert second['lambda_absolute'] == pytest.approx(10000*first['lambda_absolute'])
    with pytest.raises(ValueError):
        tikhonov_reconstruct(a, y, 0)


def test_collection_cache_reuses_operator_without_reusing_observation():
    from run_collection import TikhonovCache
    rng = np.random.default_rng(2026)
    a = rng.normal(size=(9, 7)) + 1j*rng.normal(size=(9, 7))
    cache = TikhonovCache(.01)
    previous = None
    for _ in range(2):
        y = rng.normal(size=(9, 5, 3)) + 1j*rng.normal(size=(9, 5, 3))
        actual, checks = cache.solve(a, y)
        expected, _ = tikhonov_reconstruct(a, y, .01)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        assert checks['relative_normal_residual'] < 1e-12
        if previous is not None:
            assert not np.allclose(previous, actual)
        previous = actual
    assert len(cache.entries) == 1
