"""
src/crq/preprocess/detrend.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Three solar-cycle detrending methods for cosmic-ray / seismic time series.

All functions take a 1-D numpy array *x* (already binned to uniform spacing)
and return a detrended residual array of the same length and dtype.

Methods
-------
hp_filter_detrend  — Hodrick-Prescott filter (Hodrick & Prescott 1997)
stl_detrend        — Seasonal-Trend decomposition by Loess (Cleveland et al. 1990)
sunspot_regression_detrend — OLS regression on contemporaneous + lagged sunspot numbers

References
----------
Hodrick & Prescott 1997:
    "Postwar U.S. Business Cycles: An Empirical Investigation."
    J. Money, Credit Banking 29(1), 1-16.
Ravn & Uhlig 2002:
    "On Adjusting the Hodrick-Prescott Filter for the Frequency of Observations."
    Rev. Econ. Stat. 84(2), 371-376.
Cleveland et al. 1990:
    "STL: A Seasonal-Trend Decomposition Procedure Based on Loess."
    J. Off. Stat. 6(1), 3-33.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "hp_filter_detrend",
    "stl_detrend",
    "sunspot_regression_detrend",
]


# ---------------------------------------------------------------------------
# 1. Hodrick-Prescott filter
# ---------------------------------------------------------------------------

def hp_filter_detrend(x: np.ndarray, lamb: float = 1.29e5) -> np.ndarray:
    """
    Remove trend with the Hodrick-Prescott filter and return the residual.

    Parameters
    ----------
    x : array-like, shape (N,)
        Input time series (uniform spacing).
    lamb : float
        Smoothing parameter λ.  The default 1.29e5 is calibrated for 5-day
        bins targeting removal of variations longer than ~2000 days (Ravn &
        Uhlig 2002: λ = 1600 × (annual_freq / study_freq)^4; for 5-day bins
        relative to quarterly: (365.25/4 / 5)^4 × 1600 ≈ 1.29 × 10^5).

    Returns
    -------
    residual : np.ndarray, shape (N,)
        x minus the HP trend component.
    """
    from statsmodels.tsa.filters.hp_filter import hpfilter  # type: ignore

    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1-D")
    cycle, _ = hpfilter(x, lamb=lamb)
    return cycle.astype(x.dtype)


# ---------------------------------------------------------------------------
# 2. STL decomposition
# ---------------------------------------------------------------------------

def stl_detrend(
    x: np.ndarray,
    period: int,
    seasonal_jump: int = 100,
    trend_jump: int = 100,
) -> np.ndarray:
    """
    Remove trend + seasonal via STL and return the residual.

    Parameters
    ----------
    x : array-like, shape (N,)
        Input time series (uniform spacing).
    period : int
        Number of bins per solar cycle (e.g. 803 for 11-year cycle with
        5-day bins: 11 × 365.25 / 5 ≈ 803).
    seasonal_jump, trend_jump : int
        Step sizes passed to STL; larger values give ~3× speedup with
        negligible quality loss.  Defaults tuned for 3215-point series.

    Returns
    -------
    residual : np.ndarray, shape (N,)
        STL residual component (x − trend − seasonal).
    """
    from statsmodels.tsa.seasonal import STL  # type: ignore

    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1-D")

    result = STL(
        x,
        period=period,
        seasonal_jump=seasonal_jump,
        trend_jump=trend_jump,
    ).fit()
    return result.resid.astype(x.dtype)


# ---------------------------------------------------------------------------
# 3. Sunspot regression detrend
# ---------------------------------------------------------------------------

def sunspot_regression_detrend(
    x: np.ndarray,
    sunspot_series: np.ndarray,
    lag_days: Sequence[int] = (0, 30, 90, 180),
    bin_days: int = 5,
) -> np.ndarray:
    """
    Remove solar-cycle variation via OLS regression on sunspot numbers.

    Constructs a design matrix of contemporaneous and lagged (binned)
    sunspot values and subtracts the fitted component.

    Parameters
    ----------
    x : array-like, shape (N,)
        Input time series, already binned to *bin_days* spacing.
    sunspot_series : array-like, shape (N,)
        Sunspot numbers binned to the same *bin_days* grid as *x*.
        Must be aligned (same length, same time axis).
    lag_days : sequence of int
        Lag offsets in days.  Each is converted to bin steps
        (``lag_bins = lag_days // bin_days``).  ``0`` gives
        contemporaneous regression; positive values give lagged.
    bin_days : int
        Bin width in days (used only to convert *lag_days* → bin steps).

    Returns
    -------
    residual : np.ndarray, shape (N,)
        x minus the fitted solar-cycle component.
    """
    import statsmodels.api as sm  # type: ignore

    x = np.asarray(x, dtype=float)
    sunspot = np.asarray(sunspot_series, dtype=float)
    if x.ndim != 1 or sunspot.ndim != 1:
        raise ValueError("x and sunspot_series must be 1-D")
    if len(x) != len(sunspot):
        raise ValueError(
            f"x (len {len(x)}) and sunspot_series (len {len(sunspot)}) must have the same length"
        )

    N = len(x)
    lag_bins = [int(d) // bin_days for d in lag_days]

    # Build design matrix: column for each (unique) lag bin
    cols: list[np.ndarray] = []
    for lb in lag_bins:
        col = np.full(N, np.nan)
        if lb == 0:
            col[:] = sunspot
        else:
            col[lb:] = sunspot[: N - lb]
        cols.append(col)

    design = np.column_stack(cols)  # (N, n_lags)
    design = sm.add_constant(design)  # prepend intercept column

    # Drop rows where any predictor is NaN (from lagged edges)
    valid = ~np.any(np.isnan(design), axis=1)
    if valid.sum() < 10:
        logger.warning(
            "sunspot_regression_detrend: only %d valid rows — returning x unchanged",
            valid.sum(),
        )
        return x.copy()

    ols = sm.OLS(x[valid], design[valid]).fit()

    fitted = np.full(N, np.nan)
    fitted[valid] = ols.fittedvalues

    # Fill leading NaN rows (lagged edges) with mean of fitted to avoid NaN residuals
    if not valid[0]:
        first_valid = np.argmax(valid)
        fitted[:first_valid] = np.nanmean(fitted)

    residual = x - fitted
    return residual.astype(x.dtype)
