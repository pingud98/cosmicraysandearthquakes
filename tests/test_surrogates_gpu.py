"""
tests/test_surrogates_gpu.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for src/crq/stats/surrogates_gpu.py.

GPU tests are skipped automatically when CuPy is unavailable.
The CPU-fallback path is always tested.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from crq.stats.surrogates_gpu import (
    auto_batch_size,
    gpu_available,
    phase_randomise_batch_gpu,
    iaaft_batch_gpu,
    surrogate_xcorr_test_gpu,
    _pearson_lag_array_cpu,
    _pearson_lag_batch_gpu,
)

GPU = pytest.mark.skipif(not gpu_available(), reason="CuPy / CUDA not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ar1_512():
    rng = np.random.default_rng(0)
    N, rho = 512, 0.85
    x = np.empty(N, dtype=np.float32)
    x[0] = rng.standard_normal()
    for i in range(1, N):
        x[i] = rho * x[i-1] + math.sqrt(1 - rho**2) * rng.standard_normal()
    return x


@pytest.fixture
def short_wn():
    return np.random.default_rng(7).standard_normal(200).astype(np.float32)


# ---------------------------------------------------------------------------
# auto_batch_size
# ---------------------------------------------------------------------------

class TestAutoBatchSize:
    def test_returns_positive(self):
        assert auto_batch_size(3215) > 0

    def test_large_T_gives_smaller_batch(self):
        b_small = auto_batch_size(1000)
        b_large = auto_batch_size(100_000)
        assert b_large < b_small

    def test_headroom(self):
        # With 10 GB budget and headroom=0.20 we should stay under 10 GB
        T = 16_000
        batch = auto_batch_size(T, vram_budget_gb=10.0, headroom=0.20)
        # bytes_per: T*4 + T*4 + T*4 + (T//2+1)*8 ≈ T*20
        bytes_used = batch * (T * 4 + T * 4 + T * 4 + (T // 2 + 1) * 8)
        assert bytes_used <= 10e9

    def test_16k_fits_under_50k(self):
        # For T=16,000 float32 with 10 GB VRAM, batch should be < 50,000
        # (the spec says "max batch ≈ 50,000 series per pass")
        batch = auto_batch_size(16_000, vram_budget_gb=10.0)
        assert batch <= 50_000


# ---------------------------------------------------------------------------
# _pearson_lag_array_cpu
# ---------------------------------------------------------------------------

class TestPearsonLagCpu:
    def test_lag_zero_equals_pearsonr(self):
        import scipy.stats
        rng = np.random.default_rng(1)
        x = rng.standard_normal(100).astype(np.float32)
        y = rng.standard_normal(100).astype(np.float32)
        r_cpu = _pearson_lag_array_cpu(x, y, np.array([0]))[0]
        r_ref, _ = scipy.stats.pearsonr(x, y)
        assert abs(float(r_cpu) - float(r_ref)) < 1e-4

    def test_shape(self):
        x = np.random.default_rng(0).standard_normal(200).astype(np.float32)
        y = np.random.default_rng(1).standard_normal(200).astype(np.float32)
        lags = np.arange(-10, 11)
        r = _pearson_lag_array_cpu(x, y, lags)
        assert r.shape == (len(lags),)


# ---------------------------------------------------------------------------
# _pearson_lag_batch_gpu (CPU-side vectorised path)
# ---------------------------------------------------------------------------

class TestPearsonLagBatch:
    def test_matches_scalar_cpu(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(100).astype(np.float32)
        y = rng.standard_normal(100).astype(np.float32)
        lags = np.arange(-5, 6)
        # single surrogate
        ref = _pearson_lag_array_cpu(x, y, lags)
        batch = _pearson_lag_batch_gpu(x[np.newaxis, :], y, lags)
        np.testing.assert_allclose(ref, batch[0], atol=1e-4)

    def test_output_shape(self):
        rng = np.random.default_rng(3)
        surr = rng.standard_normal((50, 200)).astype(np.float32)
        y    = rng.standard_normal(200).astype(np.float32)
        lags = np.arange(-10, 11)
        out  = _pearson_lag_batch_gpu(surr, y, lags)
        assert out.shape == (50, len(lags))


# ---------------------------------------------------------------------------
# phase_randomise_batch_gpu  (GPU tests)
# ---------------------------------------------------------------------------

@GPU
class TestPhaseRandomiseBatchGpu:
    def test_output_shape(self, ar1_512):
        out = phase_randomise_batch_gpu(ar1_512, n_surrogates=10, seed=0)
        assert out.shape == (10, len(ar1_512))

    def test_spectrum_preserved(self, ar1_512):
        out = phase_randomise_batch_gpu(ar1_512, n_surrogates=5, seed=1)
        orig_amp = np.abs(np.fft.rfft(ar1_512))
        for i in range(5):
            surr_amp = np.abs(np.fft.rfft(out[i]))
            np.testing.assert_allclose(surr_amp, orig_amp, rtol=1e-4)

    def test_mean_preserved(self, ar1_512):
        out = phase_randomise_batch_gpu(ar1_512, n_surrogates=10, seed=2)
        for i in range(10):
            assert abs(out[i].mean() - ar1_512.mean()) < 1e-4

    def test_seed_reproducibility(self, ar1_512):
        a = phase_randomise_batch_gpu(ar1_512, n_surrogates=5, seed=42)
        b = phase_randomise_batch_gpu(ar1_512, n_surrogates=5, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self, ar1_512):
        a = phase_randomise_batch_gpu(ar1_512, n_surrogates=5, seed=10)
        b = phase_randomise_batch_gpu(ar1_512, n_surrogates=5, seed=11)
        assert not np.allclose(a, b)

    def test_surrogates_differ_from_input(self, ar1_512):
        out = phase_randomise_batch_gpu(ar1_512, n_surrogates=5, seed=3)
        for i in range(5):
            assert not np.allclose(out[i], ar1_512)


# ---------------------------------------------------------------------------
# iaaft_batch_gpu  (GPU tests)
# ---------------------------------------------------------------------------

@GPU
class TestIaaftBatchGpu:
    def test_output_shape(self, ar1_512):
        out = iaaft_batch_gpu(ar1_512, n_surrogates=5, seed=0, n_iter=10)
        assert out.shape == (5, len(ar1_512))

    def test_amplitude_distribution_preserved(self, ar1_512):
        out = iaaft_batch_gpu(ar1_512, n_surrogates=5, seed=1, n_iter=50)
        for i in range(5):
            np.testing.assert_array_almost_equal(
                np.sort(out[i]), np.sort(ar1_512), decimal=4,
            )

    def test_seed_reproducibility(self, ar1_512):
        a = iaaft_batch_gpu(ar1_512, n_surrogates=5, seed=99, n_iter=10)
        b = iaaft_batch_gpu(ar1_512, n_surrogates=5, seed=99, n_iter=10)
        np.testing.assert_array_equal(a, b)

    def test_spectrum_approximately_preserved(self, ar1_512):
        out = iaaft_batch_gpu(ar1_512, n_surrogates=3, seed=2, n_iter=100)
        orig_amp = np.abs(np.fft.rfft(ar1_512))
        for i in range(3):
            surr_amp = np.abs(np.fft.rfft(out[i]))
            rel_err = np.abs(surr_amp - orig_amp) / (orig_amp + 1e-12)
            assert np.median(rel_err) < 0.05


# ---------------------------------------------------------------------------
# surrogate_xcorr_test_gpu  (GPU + fallback)
# ---------------------------------------------------------------------------

class TestSurrogateXcorrTestGpu:
    """Tests that run on CPU-fallback path regardless of GPU availability."""

    def test_return_keys(self, short_wn):
        y = np.random.default_rng(0).standard_normal(len(short_wn)).astype(np.float32)
        out = surrogate_xcorr_test_gpu(
            short_wn, y, np.arange(-5, 6),
            n_surrogates=30, method="phase", seed=0,
        )
        required = {
            "observed_r", "observed_peak_r", "observed_peak_lag",
            "p_global", "p_at_lag", "surrogate_r_arrays", "surrogate_max_r",
        }
        assert required.issubset(out.keys())

    def test_p_global_in_range(self, short_wn):
        y = np.random.default_rng(1).standard_normal(len(short_wn)).astype(np.float32)
        out = surrogate_xcorr_test_gpu(
            short_wn, y, np.arange(-5, 6),
            n_surrogates=30, method="phase", seed=0,
        )
        assert 0.0 <= out["p_global"] <= 1.0

    def test_shapes(self, short_wn):
        y    = np.random.default_rng(2).standard_normal(len(short_wn)).astype(np.float32)
        lags = np.arange(-10, 11)
        out  = surrogate_xcorr_test_gpu(
            short_wn, y, lags,
            n_surrogates=20, method="phase", seed=0,
        )
        assert out["surrogate_r_arrays"].shape == (20, len(lags))
        assert out["surrogate_max_r"].shape    == (20,)
        assert out["p_at_lag"].shape           == (len(lags),)

    def test_rejects_nan(self):
        x = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        y = np.ones(3, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN"):
            surrogate_xcorr_test_gpu(x, y, np.array([0]))


@GPU
class TestSurrogateXcorrTestGpuNumerics:
    """GPU-specific: check equivalence between CPU and GPU p-values."""

    def test_p_global_agrees_with_cpu(self):
        """
        GPU and CPU p_global must agree to within ±2/sqrt(n_surrogates)
        Monte Carlo tolerance when using the same input.
        """
        from crq.stats.surrogates import surrogate_xcorr_test

        rng = np.random.default_rng(0)
        N = 512
        rho = 0.85
        x = np.empty(N, dtype=np.float32)
        x[0] = rng.standard_normal()
        for i in range(1, N):
            x[i] = rho * x[i-1] + math.sqrt(1 - rho**2) * rng.standard_normal()
        y = rng.standard_normal(N).astype(np.float32)
        lags = np.arange(-20, 21)
        n_surr = 500

        cpu_out = surrogate_xcorr_test(
            x, y, lags, n_surrogates=n_surr, method="phase", seed=42, n_jobs=1,
        )
        gpu_out = surrogate_xcorr_test_gpu(
            x, y, lags, n_surrogates=n_surr, method="phase", seed=42,
        )

        tol = 2.0 / math.sqrt(n_surr)
        delta = abs(cpu_out["p_global"] - gpu_out["p_global"])
        assert delta <= tol, (
            f"|Δp_global| = {delta:.4f} exceeds MC tolerance {tol:.4f}"
        )
