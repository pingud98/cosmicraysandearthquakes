"""
tests/test_detrend.py
~~~~~~~~~~~~~~~~~~~~~
Tests for src/crq/preprocess/detrend.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from crq.preprocess.detrend import (
    hp_filter_detrend,
    stl_detrend,
    sunspot_regression_detrend,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trend_plus_noise() -> tuple[np.ndarray, np.ndarray]:
    """Slow sinusoidal trend + white noise; 512 points."""
    rng = np.random.default_rng(0)
    N = 512
    t = np.arange(N, dtype=float)
    trend = 5.0 * np.sin(2 * np.pi * t / 400)  # low-frequency trend
    noise = rng.standard_normal(N)
    return trend, trend + noise


@pytest.fixture
def seasonal_series() -> np.ndarray:
    """Series with period-64 seasonal component + trend + noise; 512 points."""
    rng = np.random.default_rng(1)
    N = 512
    t = np.arange(N, dtype=float)
    seasonal = 3.0 * np.sin(2 * np.pi * t / 64)
    trend = 0.01 * t
    return trend + seasonal + rng.standard_normal(N) * 0.5


# ---------------------------------------------------------------------------
# hp_filter_detrend
# ---------------------------------------------------------------------------

class TestHpFilterDetrend:
    def test_output_length(self, trend_plus_noise):
        _, x = trend_plus_noise
        assert len(hp_filter_detrend(x)) == len(x)

    def test_output_dtype_preserved(self, trend_plus_noise):
        _, x = trend_plus_noise
        assert hp_filter_detrend(x).dtype == np.float64

    def test_removes_slow_trend(self, trend_plus_noise):
        trend, x = trend_plus_noise
        residual = hp_filter_detrend(x, lamb=1600.0)
        # Residual variance should be much smaller than original trend variance
        assert np.var(residual) < np.var(trend) * 0.2, (
            "HP filter should substantially remove the low-frequency trend"
        )

    def test_residual_mean_near_zero(self, trend_plus_noise):
        _, x = trend_plus_noise
        residual = hp_filter_detrend(x)
        assert abs(residual.mean()) < 0.5

    def test_rejects_multidim(self):
        with pytest.raises(ValueError, match="1-D"):
            hp_filter_detrend(np.ones((10, 2)))

    def test_default_lambda_runs(self):
        x = np.random.default_rng(0).standard_normal(200)
        out = hp_filter_detrend(x)
        assert len(out) == 200

    def test_lower_lambda_tracks_trend_more_closely(self):
        """With lower λ the trend is more flexible and tracks x more closely,
        so the residual (cycle) has lower variance than with high λ."""
        rng = np.random.default_rng(0)
        N = 256
        t = np.arange(N, dtype=float)
        x = 10.0 * np.sin(2 * np.pi * t / 200) + rng.standard_normal(N)
        res_low  = hp_filter_detrend(x, lamb=1e3)   # flexible trend — small residual
        res_high = hp_filter_detrend(x, lamb=1e8)   # rigid trend  — large residual
        assert np.var(res_low) < np.var(res_high), (
            "lower lambda (more flexible trend) should produce a smaller-variance cycle component"
        )


# ---------------------------------------------------------------------------
# stl_detrend
# ---------------------------------------------------------------------------

class TestStlDetrend:
    def test_output_length(self, seasonal_series):
        out = stl_detrend(seasonal_series, period=64, seasonal_jump=10, trend_jump=10)
        assert len(out) == len(seasonal_series)

    def test_output_dtype_preserved(self, seasonal_series):
        out = stl_detrend(seasonal_series, period=64, seasonal_jump=10, trend_jump=10)
        assert out.dtype == np.float64

    def test_removes_seasonal_component(self, seasonal_series):
        """After STL the seasonal amplitude should be largely gone."""
        residual = stl_detrend(seasonal_series, period=64, seasonal_jump=10, trend_jump=10)
        orig_std = np.std(seasonal_series)
        resid_std = np.std(residual)
        assert resid_std < orig_std * 0.5, (
            f"STL residual std {resid_std:.3f} should be much less than original {orig_std:.3f}"
        )

    def test_rejects_multidim(self):
        with pytest.raises(ValueError, match="1-D"):
            stl_detrend(np.ones((10, 2)), period=5)

    def test_jump_params_do_not_change_length(self, seasonal_series):
        out1 = stl_detrend(seasonal_series, period=64, seasonal_jump=1,   trend_jump=1)
        out2 = stl_detrend(seasonal_series, period=64, seasonal_jump=100, trend_jump=100)
        assert len(out1) == len(out2) == len(seasonal_series)

    def test_residual_less_autocorrelated(self, seasonal_series):
        """Residual lag-1 ACF should be smaller than original."""
        def acf1(s):
            s = s - s.mean()
            return float(np.corrcoef(s[:-1], s[1:])[0, 1])
        residual = stl_detrend(seasonal_series, period=64, seasonal_jump=10, trend_jump=10)
        assert abs(acf1(residual)) < abs(acf1(seasonal_series)), (
            "STL residual should be less autocorrelated than the raw series"
        )


# ---------------------------------------------------------------------------
# sunspot_regression_detrend
# ---------------------------------------------------------------------------

class TestSunspotRegressionDetrend:
    def test_output_length(self):
        rng = np.random.default_rng(0)
        N = 300
        x = rng.standard_normal(N)
        ss = np.abs(rng.standard_normal(N)) * 50
        out = sunspot_regression_detrend(x, ss)
        assert len(out) == N

    def test_removes_correlated_component(self):
        """If x is mostly sunspot, residual should have much lower variance."""
        rng = np.random.default_rng(0)
        N = 400
        ss = np.abs(np.sin(np.arange(N, dtype=float) * 2 * np.pi / 200)) * 100
        x = 2.0 * ss + 0.1 * rng.standard_normal(N)
        residual = sunspot_regression_detrend(x, ss, lag_days=(0,), bin_days=5)
        assert np.var(residual) < np.var(x) * 0.05, (
            "Most variance should be explained by contemporaneous sunspot regression"
        )

    def test_rejects_length_mismatch(self):
        x  = np.ones(100)
        ss = np.ones(99)
        with pytest.raises(ValueError, match="same length"):
            sunspot_regression_detrend(x, ss)

    def test_rejects_multidim_x(self):
        ss = np.ones(10)
        with pytest.raises(ValueError, match="1-D"):
            sunspot_regression_detrend(np.ones((10, 2)), ss)

    def test_rejects_multidim_sunspot(self):
        x = np.ones(10)
        with pytest.raises(ValueError, match="1-D"):
            sunspot_regression_detrend(x, np.ones((10, 2)))

    def test_output_dtype_preserved(self):
        rng = np.random.default_rng(1)
        N = 300
        x  = rng.standard_normal(N)
        ss = np.abs(rng.standard_normal(N))
        out = sunspot_regression_detrend(x, ss)
        assert out.dtype == np.float64

    def test_no_nan_in_output(self):
        rng = np.random.default_rng(2)
        N = 300
        x  = rng.standard_normal(N)
        ss = np.abs(rng.standard_normal(N)) * 50
        out = sunspot_regression_detrend(x, ss, lag_days=(0, 30, 90), bin_days=5)
        assert not np.any(np.isnan(out)), "output must contain no NaN"

    def test_lag_zero_only_runs(self):
        rng = np.random.default_rng(3)
        N = 200
        x  = rng.standard_normal(N)
        ss = np.abs(rng.standard_normal(N))
        out = sunspot_regression_detrend(x, ss, lag_days=(0,), bin_days=5)
        assert len(out) == N

    def test_multiple_lags_run(self):
        rng = np.random.default_rng(4)
        N = 400
        x  = rng.standard_normal(N)
        ss = np.abs(rng.standard_normal(N))
        out = sunspot_regression_detrend(x, ss, lag_days=(0, 30, 90, 180), bin_days=5)
        assert len(out) == N
        assert not np.any(np.isnan(out))
