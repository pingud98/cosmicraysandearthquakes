"""
tests/test_surrogates.py
~~~~~~~~~~~~~~~~~~~~~~~~
Tests for src/crq/stats/surrogates.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.stats

from crq.stats.surrogates import (
    iaaft,
    n_eff_bretherton,
    p_to_sigma,
    phase_randomise,
    surrogate_xcorr_test,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ar1_series() -> np.ndarray:
    """AR(1) process, ρ = 0.85, N = 512 (power-of-2 for clean tests)."""
    rng = np.random.default_rng(0)
    N, rho = 512, 0.85
    x = np.empty(N)
    x[0] = rng.standard_normal()
    for i in range(1, N):
        x[i] = rho * x[i - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal()
    return x


@pytest.fixture
def short_series() -> np.ndarray:
    """Short white-noise series for speed-critical tests."""
    return np.random.default_rng(7).standard_normal(200)


# ---------------------------------------------------------------------------
# phase_randomise
# ---------------------------------------------------------------------------

class TestPhaseRandomise:
    def test_preserves_spectrum(self, ar1_series):
        surr = phase_randomise(ar1_series, seed=1)
        orig_amp = np.abs(np.fft.rfft(ar1_series))
        surr_amp = np.abs(np.fft.rfft(surr))
        np.testing.assert_allclose(surr_amp, orig_amp, rtol=1e-6,
                                   err_msg="power spectrum must be preserved")

    def test_preserves_mean(self, ar1_series):
        surr = phase_randomise(ar1_series, seed=2)
        assert abs(surr.mean() - ar1_series.mean()) < 1e-9

    def test_output_length(self, ar1_series):
        assert len(phase_randomise(ar1_series, seed=3)) == len(ar1_series)

    def test_output_is_different_from_input(self, ar1_series):
        surr = phase_randomise(ar1_series, seed=4)
        assert not np.allclose(surr, ar1_series), "surrogate should differ from original"

    def test_seed_reproducibility(self, ar1_series):
        s1 = phase_randomise(ar1_series, seed=42)
        s2 = phase_randomise(ar1_series, seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_give_different_surrogates(self, ar1_series):
        s1 = phase_randomise(ar1_series, seed=10)
        s2 = phase_randomise(ar1_series, seed=11)
        assert not np.allclose(s1, s2)

    def test_odd_length(self):
        x = np.random.default_rng(0).standard_normal(201)
        surr = phase_randomise(x, seed=5)
        assert len(surr) == 201
        np.testing.assert_allclose(
            np.abs(np.fft.rfft(surr)),
            np.abs(np.fft.rfft(x)),
            rtol=1e-5,
        )

    def test_even_length(self):
        x = np.random.default_rng(0).standard_normal(200)
        surr = phase_randomise(x, seed=6)
        assert len(surr) == 200

    def test_rejects_nan(self):
        x = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="NaN"):
            phase_randomise(x)

    def test_phases_are_random(self, ar1_series):
        """Phases of the surrogate FFT should differ from original phases."""
        X_orig = np.fft.rfft(ar1_series)
        s1 = phase_randomise(ar1_series, seed=7)
        X_surr = np.fft.rfft(s1)
        orig_phases = np.angle(X_orig[1:-1])
        surr_phases = np.angle(X_surr[1:-1])
        n_same = np.sum(np.abs(orig_phases - surr_phases) < 0.01)
        assert n_same < 5, "most phases should be randomised"


# ---------------------------------------------------------------------------
# iaaft
# ---------------------------------------------------------------------------

class TestIaaft:
    def test_preserves_amplitude_distribution(self, ar1_series):
        surr = iaaft(ar1_series, seed=1, n_iter=100)
        np.testing.assert_array_almost_equal(
            np.sort(surr), np.sort(ar1_series), decimal=10,
            err_msg="IAAFT must preserve the amplitude distribution exactly",
        )

    def test_preserves_spectrum_approximately(self, ar1_series):
        """After 100 iters the spectrum should be close but not exact."""
        surr = iaaft(ar1_series, seed=2, n_iter=100)
        orig_amp = np.abs(np.fft.rfft(ar1_series))
        surr_amp = np.abs(np.fft.rfft(surr))
        rel_err = np.abs(surr_amp - orig_amp) / (orig_amp + 1e-12)
        assert np.median(rel_err) < 0.05, "median spectral error should be < 5%"

    def test_output_length(self, ar1_series):
        assert len(iaaft(ar1_series, seed=3)) == len(ar1_series)

    def test_seed_reproducibility(self, ar1_series):
        s1 = iaaft(ar1_series, seed=99, n_iter=20)
        s2 = iaaft(ar1_series, seed=99, n_iter=20)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_differ(self, ar1_series):
        s1 = iaaft(ar1_series, seed=10, n_iter=20)
        s2 = iaaft(ar1_series, seed=11, n_iter=20)
        assert not np.allclose(s1, s2)

    def test_rejects_nan(self):
        x = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="NaN"):
            iaaft(x)

    def test_odd_length(self):
        x = np.random.default_rng(0).standard_normal(201)
        surr = iaaft(x, seed=5, n_iter=10)
        assert len(surr) == 201
        np.testing.assert_array_almost_equal(np.sort(surr), np.sort(x), decimal=10)

    def test_more_iterations_improves_spectrum(self, ar1_series):
        """Increasing n_iter should bring spectrum closer to target."""
        surr10  = iaaft(ar1_series, seed=1, n_iter=10)
        surr100 = iaaft(ar1_series, seed=1, n_iter=100)
        orig_amp  = np.abs(np.fft.rfft(ar1_series))
        err10  = np.mean(np.abs(np.abs(np.fft.rfft(surr10))  - orig_amp))
        err100 = np.mean(np.abs(np.abs(np.fft.rfft(surr100)) - orig_amp))
        assert err100 <= err10 * 1.5, "more iterations should not worsen spectrum"


# ---------------------------------------------------------------------------
# n_eff_bretherton
# ---------------------------------------------------------------------------

class TestNEffBretherton:
    def test_white_noise_gives_approx_n(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(1000)
        y = rng.standard_normal(1000)
        n_eff = n_eff_bretherton(x, y)
        # For white noise ρ₁ ≈ 0 so N_eff ≈ N
        assert 800 < n_eff <= 1000, f"N_eff={n_eff} expected near 1000 for white noise"

    def test_high_autocorrelation_reduces_n_eff(self):
        # AR(1) with ρ=0.95: N_eff << N
        rng = np.random.default_rng(1)
        N, rho = 500, 0.95
        x = np.zeros(N); x[0] = rng.standard_normal()
        for i in range(1, N):
            x[i] = rho * x[i-1] + np.sqrt(1-rho**2) * rng.standard_normal()
        y = x + 0.1 * rng.standard_normal(N)
        n_eff = n_eff_bretherton(x, y)
        assert n_eff < N / 5, f"N_eff={n_eff} should be much less than N={N}"

    def test_minimum_clamp(self):
        # Extreme autocorrelation
        x = np.ones(100)
        y = np.ones(100)
        assert n_eff_bretherton(x, y) >= 3.0

    def test_short_series(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        assert n_eff_bretherton(x, y) >= 3.0

    def test_symmetry(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(300)
        y = rng.standard_normal(300)
        assert abs(n_eff_bretherton(x, y) - n_eff_bretherton(y, x)) < 1e-6


# ---------------------------------------------------------------------------
# p_to_sigma
# ---------------------------------------------------------------------------

class TestPToSigma:
    def test_known_values(self):
        assert abs(p_to_sigma(0.05) - 1.96) < 0.01
        assert abs(p_to_sigma(0.003) - 3.0) < 0.05

    def test_zero_p_returns_lower_bound(self):
        sigma = p_to_sigma(0.0, n_trials=10_000)
        assert 3.0 < sigma < 5.0

    def test_one_p_returns_zero(self):
        assert p_to_sigma(1.0) < 0.01

    def test_very_small_p(self):
        sigma = p_to_sigma(1e-10)
        assert sigma > 6.0


# ---------------------------------------------------------------------------
# surrogate_xcorr_test
# ---------------------------------------------------------------------------

class TestSurrogateXcorrTest:
    """Functional tests on short synthetic series (n_surrogates=200 for speed)."""

    N_SURR = 200

    def test_uncorrelated_gives_uniform_p(self, short_series):
        """For two independent series, p_global should not be consistently near 0."""
        rng = np.random.default_rng(99)
        y   = rng.standard_normal(len(short_series))
        lags = np.arange(-20, 21)
        out = surrogate_xcorr_test(
            short_series, y, lags,
            n_surrogates=self.N_SURR, method="phase", seed=42, n_jobs=1,
        )
        # For independent series p_global should be > 0.01 most of the time
        # (not consistently significant) — use a generous threshold
        assert out["p_global"] > 0.001, (
            f"p_global={out['p_global']} is suspiciously small for uncorrelated series"
        )

    def test_correlated_at_known_lag_gives_low_p(self):
        """For a series with a planted 5-lag cross-correlation, p_global should be small."""
        rng = np.random.default_rng(5)
        N   = 300
        x   = rng.standard_normal(N)
        # Plant strong signal: y[t] ≈ 3*x[t-5] + noise
        y   = np.zeros(N)
        y[5:] = 3.0 * x[:N-5] + 0.1 * rng.standard_normal(N - 5)
        y[:5] = rng.standard_normal(5)

        lags = np.arange(-30, 31)
        out  = surrogate_xcorr_test(
            x, y, lags,
            n_surrogates=self.N_SURR, method="phase", seed=1, n_jobs=1,
        )
        assert out["p_global"] < 0.05, (
            f"p_global={out['p_global']} should be small for a strongly planted signal"
        )

    def test_return_dict_keys(self, short_series):
        y = np.random.default_rng(0).standard_normal(len(short_series))
        out = surrogate_xcorr_test(
            short_series, y, np.arange(-5, 6),
            n_surrogates=50, method="phase", seed=0, n_jobs=1,
        )
        required = {
            "observed_r", "observed_peak_r", "observed_peak_lag",
            "p_global", "p_at_lag",
            "surrogate_r_arrays", "surrogate_max_r",
            "n_surrogates", "method", "seed",
        }
        assert required.issubset(out.keys())

    def test_surrogate_array_shape(self, short_series):
        y    = np.random.default_rng(1).standard_normal(len(short_series))
        lags = np.arange(-10, 11)
        out  = surrogate_xcorr_test(
            short_series, y, lags,
            n_surrogates=100, method="phase", seed=0, n_jobs=1,
        )
        assert out["surrogate_r_arrays"].shape == (100, len(lags))
        assert out["surrogate_max_r"].shape    == (100,)
        assert out["p_at_lag"].shape           == (len(lags),)

    def test_observed_r_matches_direct_computation(self, short_series):
        from crq.stats.surrogates import _pearson_lag_array
        y    = np.random.default_rng(2).standard_normal(len(short_series))
        lags = np.arange(-5, 6)
        out  = surrogate_xcorr_test(
            short_series, y, lags,
            n_surrogates=50, method="phase", seed=0, n_jobs=1,
        )
        direct = _pearson_lag_array(short_series, y, lags)
        np.testing.assert_allclose(
            out["observed_r"].astype(np.float64),
            direct.astype(np.float64),
            atol=1e-5,
        )

    def test_seed_reproducibility(self, short_series):
        y    = np.random.default_rng(3).standard_normal(len(short_series))
        lags = np.arange(-5, 6)
        kw   = dict(n_surrogates=50, method="phase", seed=77, n_jobs=1)
        out1 = surrogate_xcorr_test(short_series, y, lags, **kw)
        out2 = surrogate_xcorr_test(short_series, y, lags, **kw)
        np.testing.assert_array_equal(
            out1["surrogate_r_arrays"], out2["surrogate_r_arrays"],
        )

    def test_iaaft_method_runs(self, short_series):
        y    = np.random.default_rng(4).standard_normal(len(short_series))
        lags = np.arange(-5, 6)
        out  = surrogate_xcorr_test(
            short_series, y, lags,
            n_surrogates=30, method="iaaft", seed=0, n_jobs=1, iaaft_n_iter=10,
        )
        assert 0.0 <= out["p_global"] <= 1.0

    def test_p_global_is_fraction(self, short_series):
        y   = np.random.default_rng(5).standard_normal(len(short_series))
        out = surrogate_xcorr_test(
            short_series, y, np.arange(-5, 6),
            n_surrogates=100, method="phase", seed=0, n_jobs=1,
        )
        assert 0.0 <= out["p_global"] <= 1.0
