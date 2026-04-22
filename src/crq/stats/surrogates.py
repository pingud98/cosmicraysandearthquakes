"""
src/crq/stats/surrogates.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase-randomised and IAAFT surrogate null models for time-series hypothesis
testing.

Both methods generate synthetic realisations of a series x that preserve its
power spectrum (hence linear autocorrelation structure) while destroying any
specific phase relationship between frequencies.  Correlations seen in many
surrogates arise from coincidental phase alignment alone; an observed
correlation that exceeds the surrogate distribution provides evidence for a
genuine cross-signal.

References
----------
Theiler et al. 1992:
    "Testing for nonlinearity in time series: the method of surrogate data."
    Physica D 58(1-4), 77-94.
Schreiber & Schmitz 1996:
    "Improved Surrogate Data for Nonlinearity Tests."
    Phys. Rev. Lett. 77(4), 635-638.
Schreiber & Schmitz 2000:
    "Surrogate time series." Physica D 142(3-4), 346-382.
Bretherton et al. 1999:
    "The Effective Number of Spatial Degrees of Freedom of a Time-Varying
    Field." J. Climate 12(7), 1990-2009.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

import numpy as np
import scipy.stats
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

__all__ = [
    "phase_randomise",
    "iaaft",
    "n_eff_bretherton",
    "surrogate_xcorr_test",
    "p_to_sigma",
]

# Pad FFTs to next power of 2 for speed.  3215 = 5 × 643 (643 is prime) is a
# pathologically bad FFT size; 4096 is 3× faster for irfft/rfft.
_FFT_PAD_POWER2 = True


def _next_pow2(n: int) -> int:
    """Smallest power of 2 ≥ n."""
    return 1 << (n - 1).bit_length()


# ---------------------------------------------------------------------------
# Surrogate generators
# ---------------------------------------------------------------------------

def phase_randomise(
    x: np.ndarray,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """
    Fourier phase-randomisation surrogate (FT surrogate; Theiler et al. 1992).

    Randomises the Fourier phases of *x* uniformly on [0, 2π) while keeping
    every amplitude unchanged.  The surrogate has the same power spectrum (and
    therefore the same linear autocorrelation structure) as *x* but random
    cross-frequency phase relationships.

    Parameters
    ----------
    x :
        1-D finite-valued float array.
    seed :
        RNG seed or ``np.random.Generator`` instance.

    Returns
    -------
    np.ndarray
        Surrogate array, same length and dtype as *x*.

    Notes
    -----
    * DC (index 0) and Nyquist (index −1, even N) keep their original values
      so that the inverse FFT is exactly real and the mean is preserved.
    * Raises ``ValueError`` if *x* contains NaN or Inf.
    """
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("phase_randomise: x must contain only finite values (no NaN/Inf)")

    rng = np.random.default_rng(seed)
    n = len(x)
    # NOTE: No padding here.  Padding would change which spectrum is being
    # randomised — irfft(padded_surr, n=original_n) doesn't round-trip back
    # to the original unpadded amplitudes.  Spectrum preservation is the core
    # guarantee of this method, so always use n_fft = n.
    X = np.fft.rfft(x)
    n_rfft = len(X)

    phases = rng.uniform(0.0, 2.0 * np.pi, n_rfft)
    phases[0] = 0.0       # DC: keep real, preserves mean
    if n % 2 == 0:
        phases[-1] = 0.0  # Nyquist: keep real

    X_surr = np.abs(X) * np.exp(1j * phases)
    X_surr[0] = X[0]      # exact DC value (preserves mean including sign)
    if n % 2 == 0:
        X_surr[-1] = X[-1]

    return np.fft.irfft(X_surr, n=n)


def iaaft(
    x: np.ndarray,
    seed: int | np.random.Generator | None = None,
    n_iter: int = 100,
) -> np.ndarray:
    """
    Iterative Amplitude-Adjusted Fourier Transform surrogate
    (Schreiber & Schmitz 1996).

    Preserves both the power spectrum **and** the amplitude distribution of
    *x*.  This is the gold-standard null model for nonlinear time series,
    because it rules out explanations based on the measured distribution as
    well as the autocorrelation structure.

    Algorithm
    ---------
    1.  Compute sorted values (target amplitudes) and |FFT| (target spectrum).
    2.  Initialise with a random permutation of *x*.
    3.  Iterate *n_iter* times:

        a.  Spectral amplitude adjustment: replace |FFT| of the current
            iterate with the target spectrum while preserving phases.
        b.  Rank-order rescaling: rearrange elements so that their ranks
            match the current iterate but values come from the sorted *x*.

    4.  Return the last iterate.

    Parameters
    ----------
    x :
        1-D finite-valued float array.
    seed :
        RNG seed or ``np.random.Generator`` instance.
    n_iter :
        Iterations (100 is the standard default; convergence is typically
        fast for smooth spectra).

    Returns
    -------
    np.ndarray
        Surrogate array with the same amplitude distribution as *x* and
        approximately the same power spectrum, same length as *x*.

    Notes
    -----
    * FFTs are zero-padded to the next power of 2 for speed, then truncated
      back to ``len(x)`` via ``irfft(..., n=len(x))``.
    * Raises ``ValueError`` if *x* contains NaN or Inf.
    """
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("iaaft: x must contain only finite values (no NaN/Inf)")

    rng = np.random.default_rng(seed)
    n = len(x)
    n_fft = _next_pow2(n) if _FFT_PAD_POWER2 else n

    x_sorted     = np.sort(x)
    target_amp   = np.abs(np.fft.rfft(x, n=n_fft))

    s = rng.permutation(x)

    for _ in range(n_iter):
        # a. Spectral amplitude adjustment (preserve phases, replace amplitudes)
        S      = np.fft.rfft(s, n=n_fft)
        phases = np.angle(S)
        s      = np.fft.irfft(target_amp * np.exp(1j * phases), n=n)

        # b. Rank-order rescaling (preserve ranks, replace values)
        ranks  = np.argsort(np.argsort(s))
        s      = x_sorted[ranks]

    return s


# ---------------------------------------------------------------------------
# Effective sample size
# ---------------------------------------------------------------------------

def n_eff_bretherton(x: np.ndarray, y: np.ndarray) -> float:
    """
    Bretherton et al. 1999 effective sample size for the correlation r(x, y).

    Uses the first-order approximation:

        N_eff ≈ N × (1 − ρ₁ₓ ρ₁ᵧ) / (1 + ρ₁ₓ ρ₁ᵧ)

    where ρ₁ₓ, ρ₁ᵧ are the lag-1 autocorrelation coefficients of *x* and *y*
    respectively.  For white noise (ρ₁ = 0), N_eff = N as expected.  For
    highly autocorrelated series (ρ₁ → 1), N_eff → 0.

    Parameters
    ----------
    x, y : 1-D arrays of equal length.  NaN values are excluded when
           estimating lag-1 ACF.

    Returns
    -------
    float
        Effective sample size, clamped to [3, N].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))

    def _lag1_acf(v: np.ndarray) -> float:
        v = v[np.isfinite(v)]
        if len(v) < 4:
            return 0.0
        vc  = v - v.mean()
        var = np.dot(vc, vc)
        if var < 1e-15:
            return 0.0
        r1 = float(np.dot(vc[:-1], vc[1:]) / var)
        return float(np.clip(r1, -0.9999, 0.9999))

    r1x = _lag1_acf(x)
    r1y = _lag1_acf(y)

    denom = 1.0 + r1x * r1y
    if abs(denom) < 1e-12:
        return 3.0

    n_eff = n * (1.0 - r1x * r1y) / denom
    return float(np.clip(n_eff, 3.0, n))


# ---------------------------------------------------------------------------
# Fast Pearson r across all lags (inner loop for surrogate test)
# ---------------------------------------------------------------------------

def _pearson_lag_array(
    x: np.ndarray,
    y: np.ndarray,
    lag_bins: np.ndarray,
) -> np.ndarray:
    """
    Pearson r(τ) for every lag in *lag_bins*.

    Convention: lag > 0 means x leads y.
    Returns float32 array of length ``len(lag_bins)`` (NaN where n_valid < 10).
    """
    N   = len(x)
    rs  = np.full(len(lag_bins), np.nan, dtype=np.float32)

    for i, lag in enumerate(lag_bins):
        if lag >= 0:
            n = N - lag
            if n < 10:
                continue
            xa = x[:n]
            ya = y[lag : lag + n]
        else:                   # lag < 0
            n = N + lag         # N - |lag|
            if n < 10:
                continue
            xa = x[-lag:]
            ya = y[:n]

        valid = np.isfinite(xa) & np.isfinite(ya)
        nv    = int(valid.sum())
        if nv < 10:
            continue

        xa_v = xa[valid].astype(np.float64)
        ya_v = ya[valid].astype(np.float64)
        xa_c = xa_v - xa_v.mean()
        ya_c = ya_v - ya_v.mean()
        denom = np.sqrt(np.dot(xa_c, xa_c) * np.dot(ya_c, ya_c))
        if denom > 0.0:
            rs[i] = float(np.dot(xa_c, ya_c) / denom)

    return rs


# ---------------------------------------------------------------------------
# Batch worker (must be module-level for joblib pickle compatibility)
# ---------------------------------------------------------------------------

def _batch_surrogates(
    i_start: int,
    i_end: int,
    x: np.ndarray,
    y: np.ndarray,
    lag_bins: np.ndarray,
    method: str,
    base_seed: int,
    iaaft_n_iter: int,
) -> np.ndarray:
    """
    Generate and cross-correlate surrogates for indices [i_start, i_end).

    Returns float32 array of shape ``(i_end - i_start, len(lag_bins))``.
    Designed as a joblib worker: each call handles multiple surrogates to
    amortise inter-process communication overhead.
    """
    gen_fn     = iaaft if method == "iaaft" else phase_randomise
    gen_kwargs = {"n_iter": iaaft_n_iter} if method == "iaaft" else {}

    count   = i_end - i_start
    n_lags  = len(lag_bins)
    results = np.full((count, n_lags), np.nan, dtype=np.float32)

    for k, i in enumerate(range(i_start, i_end)):
        s           = gen_fn(x, seed=base_seed + i, **gen_kwargs)
        results[k]  = _pearson_lag_array(s, y, lag_bins)

    return results


# ---------------------------------------------------------------------------
# Main surrogate cross-correlation test
# ---------------------------------------------------------------------------

def surrogate_xcorr_test(
    x: np.ndarray,
    y: np.ndarray,
    lag_bins: np.ndarray,
    n_surrogates: int = 10_000,
    method: Literal["phase", "iaaft"] = "iaaft",
    seed: int = 42,
    n_jobs: int = -1,
    iaaft_n_iter: int = 100,
) -> dict:
    """
    Surrogate cross-correlation test with global multiple-lag correction.

    Generates *n_surrogates* synthetic realisations of *x* that preserve its
    autocorrelation structure but have random phase, computes r(τ) for each
    against the real *y*, and returns:

    * **p_global**: fraction of surrogates whose **peak** |r(τ)| across all
      lags equals or exceeds the observed peak |r(τ)|.  This is the correct
      multiple-comparison-corrected p-value for the "best lag" search.
    * **p_at_lag**: per-lag fraction of surrogates exceeding the observed |r|
      at each individual lag.
    * The full surrogate r(τ) distribution for envelope plotting.

    Parameters
    ----------
    x :
        CR index (1-D, float; should be NaN-free for best results).
    y :
        Seismic metric (1-D, float; aligned to *x*).
    lag_bins :
        Integer lag offsets in bin units, e.g. ``np.arange(-200, 201)``.
        A lag of +k means x is advanced k bins relative to y.
    n_surrogates :
        Number of surrogates.  10 000 is recommended for stable 99.9th-
        percentile estimates.
    method :
        ``"phase"`` — fast FT surrogate (preserves spectrum only).
        ``"iaaft"`` — conservative IAAFT (preserves spectrum + amplitude).
    seed :
        Base seed; surrogate i uses ``seed + i`` for reproducibility.
    n_jobs :
        Joblib parallel workers (``-1`` = all logical CPUs).
    iaaft_n_iter :
        IAAFT iterations (ignored for ``method="phase"``).

    Returns
    -------
    dict
        ``observed_r``          float32 (n_lags) — observed r(τ)
        ``observed_peak_r``     float   — peak |r| observed
        ``observed_peak_lag``   int     — lag bin of peak |r|
        ``p_global``            float   — global-corrected p-value
        ``p_at_lag``            float32 (n_lags) — per-lag p-values
        ``surrogate_r_arrays``  float32 (n_surr × n_lags) — full distribution
        ``surrogate_max_r``     float32 (n_surr) — per-surrogate max |r|
        ``n_surrogates``        int
        ``method``              str
        ``seed``                int
    """
    x        = np.asarray(x, dtype=float)
    y        = np.asarray(y, dtype=float)
    lag_bins = np.asarray(lag_bins, dtype=int)
    n_lags   = len(lag_bins)

    # ── Observed correlation ──────────────────────────────────────────────
    obs_r        = _pearson_lag_array(x, y, lag_bins)
    obs_peak_idx = int(np.nanargmax(np.abs(obs_r)))
    obs_peak_r   = float(np.abs(obs_r[obs_peak_idx]))
    obs_peak_lag = int(lag_bins[obs_peak_idx])

    logger.info(
        "Surrogate test (%s × %d): observed peak |r| = %.4f at lag bin %+d",
        method, n_surrogates, obs_peak_r, obs_peak_lag,
    )

    # ── Batch jobs ────────────────────────────────────────────────────────
    n_workers   = n_jobs if n_jobs > 0 else (os.cpu_count() or 1)
    batch_size  = max(1, n_surrogates // (n_workers * 4))
    batches     = [
        (i, min(i + batch_size, n_surrogates))
        for i in range(0, n_surrogates, batch_size)
    ]

    logger.info(
        "Dispatching %d batches (batch_size=%d) across %s workers …",
        len(batches), batch_size,
        f"{n_workers}" if n_jobs > 0 else "all",
    )

    raw = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_batch_surrogates)(
            i0, i1, x, y, lag_bins, method, seed, iaaft_n_iter
        )
        for i0, i1 in batches
    )

    surr_r_arrays = np.vstack(raw).astype(np.float32)   # (n_surr, n_lags)
    surr_max_r    = np.nanmax(np.abs(surr_r_arrays), axis=1)  # (n_surr,)

    # ── Global p-value ────────────────────────────────────────────────────
    p_global = float(np.mean(surr_max_r >= obs_peak_r))

    # ── Per-lag p-values ──────────────────────────────────────────────────
    abs_obs = np.abs(obs_r).astype(np.float64)
    abs_surr = np.abs(surr_r_arrays).astype(np.float64)
    p_at_lag = np.mean(abs_surr >= abs_obs[None, :], axis=0).astype(np.float32)

    n_exceed = int(np.sum(surr_max_r >= obs_peak_r))
    sigma_g  = p_to_sigma(p_global, n_surrogates)
    logger.info(
        "p_global = %.4f  (%.2fσ equiv.)  |  %d / %d surrogates exceeded peak",
        p_global, sigma_g, n_exceed, n_surrogates,
    )

    return {
        "observed_r":         obs_r,
        "observed_peak_r":    obs_peak_r,
        "observed_peak_lag":  obs_peak_lag,
        "p_global":           p_global,
        "p_at_lag":           p_at_lag,
        "surrogate_r_arrays": surr_r_arrays,
        "surrogate_max_r":    surr_max_r,
        "n_surrogates":       n_surrogates,
        "method":             method,
        "seed":               seed,
    }


# ---------------------------------------------------------------------------
# Sigma conversion
# ---------------------------------------------------------------------------

def p_to_sigma(p: float, n_trials: int | None = None) -> float:
    """
    Convert a two-tailed p-value to the equivalent Gaussian sigma.

    When *p* = 0 (no surrogate exceeded the threshold), returns a lower bound
    using the one-sided p ≥ 1 / *n_trials*.
    """
    if np.isfinite(p) and p > 0.0:
        return float(scipy.stats.norm.isf(p / 2.0))
    if n_trials is not None and n_trials > 0:
        p_lb = 1.0 / n_trials
        return float(scipy.stats.norm.isf(p_lb / 2.0))
    return np.inf
