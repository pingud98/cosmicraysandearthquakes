#!/usr/bin/env python3
"""
scripts/10_robustness.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Robustness checks for the CR–seismic correlation analysis.

2b  HP filter λ derivation and detrend comparison
    • Prints the exact λ_p = 1600·(365/p)^4 formula
    • 4-panel cross-correlation figure: raw / HP / Butterworth highpass
      (2-yr cutoff) / 12-month rolling-mean subtraction

2c  Effective-N comparison
    • Bretherton 1999 (full sum over lags)
    • Bartlett 1946 first-order  (current implementation)
    • Monte Carlo via 1 000-replicate phase randomisation
    • Supplementary table written to results/neff_comparison.json

2d  Magnitude threshold sensitivity
    • Cross-correlation r(τ) for M≥4.5, M≥5.0, M≥6.0
    • 3-panel figure; flags if peak lags or signs diverge across thresholds

All figures saved to results/figs/.  Runs in ~2 min on CPU (no GPU needed).
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from crq.ingest.nmdb import load_station, resample_daily
from crq.ingest.usgs import load_usgs, seismic_energy_per_bin
from crq.stats.surrogates import n_eff_bretherton, phase_randomise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BIN_DAYS      = 5
STUDY_START   = "1976-01-01"
STUDY_END     = "2019-12-31"
LAG_RANGE     = 1000          # ±1000 days
MIN_STATIONS  = 3
COV_THRESH    = 0.60
SEED          = 42
OUT_DIR       = ROOT / "results"
FIG_DIR       = ROOT / "results" / "figs"
NMDB_DIR      = ROOT / "data" / "raw" / "nmdb"
USGS_DIR      = ROOT / "data" / "raw" / "usgs"
CFG_FILE      = ROOT / "config" / "stations.yaml"


# ---------------------------------------------------------------------------
# Shared data loading
# ---------------------------------------------------------------------------

def _station_names() -> list[str]:
    with open(CFG_FILE) as fh:
        return list(yaml.safe_load(fh)["stations"].keys())


def _load_cr(study_start: str, study_end: str) -> pd.Series:
    """Global CR index: station-normalised mean over 5-day bins."""
    t0 = pd.Timestamp(study_start)
    t1 = pd.Timestamp(study_end)
    start_yr, end_yr = t0.year, t1.year

    norm_cols: dict[str, pd.Series] = {}
    for stn in _station_names():
        hourly = load_station(stn, start_yr, end_yr, NMDB_DIR)
        if hourly.empty:
            continue
        daily = resample_daily(hourly, stn, coverage_threshold=COV_THRESH)[stn]
        daily = daily.loc[study_start:study_end]
        n_valid = daily.notna().sum()
        if n_valid < 30:
            continue
        mu = daily.mean()
        if not np.isfinite(mu) or mu <= 0:
            continue
        norm_cols[stn] = daily / mu

    if not norm_cols:
        raise RuntimeError("No NMDB stations loaded.")

    mat = pd.DataFrame(norm_cols)
    n_valid = mat.notna().sum(axis=1)
    global_daily = mat.mean(axis=1)
    global_daily[n_valid < MIN_STATIONS] = np.nan

    # 5-day bins
    days = (global_daily.index - t0).days
    bin_num = days // BIN_DAYS
    bin_dates = t0 + pd.to_timedelta(bin_num * BIN_DAYS, unit="D")
    cr = global_daily.groupby(bin_dates).mean()
    cr.name = "cr_index"
    log.info("CR loaded: %d bins, %d stations", len(cr), len(norm_cols))
    return cr


def _load_seismic(
    study_start: str,
    study_end: str,
    min_mag: float = 4.5,
    ref_index: "pd.DatetimeIndex | None" = None,
) -> pd.Series:
    """log10(summed seismic energy) per 5-day bin."""
    t0 = pd.Timestamp(study_start)
    events = load_usgs(t0.year, pd.Timestamp(study_end).year, USGS_DIR)
    sei = seismic_energy_per_bin(
        events, study_start, study_end, BIN_DAYS, t0, min_mag=min_mag,
    )
    if ref_index is not None:
        floor = float(sei.min())
        sei = sei.reindex(ref_index, fill_value=floor)
    log.info("Seismic loaded (M≥%.1f): %d bins, range [%.1f, %.1f]",
             min_mag, len(sei), float(sei.min()), float(sei.max()))
    return sei


# ---------------------------------------------------------------------------
# Cross-correlation utility
# ---------------------------------------------------------------------------

def xcorr(x: np.ndarray, y: np.ndarray, lag_bins: np.ndarray) -> np.ndarray:
    """Pearson r(τ) for each lag in *lag_bins* (bin units, τ>0 = x leads y)."""
    N = len(x)
    rs = np.full(len(lag_bins), np.nan)
    for i, lag in enumerate(lag_bins):
        if lag >= 0:
            xa, ya = x[:N - lag], y[lag:]
        else:
            absl = -lag
            xa, ya = x[absl:], y[:N - absl]
        ok = np.isfinite(xa) & np.isfinite(ya)
        n = ok.sum()
        if n >= 10:
            r, _ = scipy.stats.pearsonr(xa[ok], ya[ok])
            rs[i] = r
    return rs


# ---------------------------------------------------------------------------
# Detrending helpers
# ---------------------------------------------------------------------------

def _hp_detrend(arr: np.ndarray, lam: float) -> np.ndarray:
    """Return HP-filter residual."""
    from crq.preprocess.detrend import hp_filter_detrend
    return hp_filter_detrend(arr, lamb=lam)


def _butterworth_highpass(arr: np.ndarray, cutoff_years: float = 2.0) -> np.ndarray:
    """
    3rd-order Butterworth highpass with cutoff at *cutoff_years* years.
    Removes variability with periods longer than cutoff; keeps shorter-period signal.
    """
    nyq = 1.0 / (2 * BIN_DAYS)                    # Nyquist in cycles/day
    cutoff_cy_per_day = 1.0 / (cutoff_years * 365.25)
    Wn = cutoff_cy_per_day / nyq                   # normalised frequency
    Wn = float(np.clip(Wn, 1e-6, 0.999))
    sos = scipy.signal.butter(3, Wn, btype="high", output="sos")
    clean = np.where(np.isfinite(arr), arr, np.nanmean(arr))
    filtered = scipy.signal.sosfiltfilt(sos, clean)
    # Restore NaN positions
    filtered[~np.isfinite(arr)] = np.nan
    return filtered


def _rolling_mean_subtract(arr: np.ndarray, window_bins: int = 73) -> np.ndarray:
    """Subtract a centred rolling mean (73 bins ≈ 365 days at 5-day bins)."""
    s = pd.Series(arr)
    trend = s.rolling(window=window_bins, center=True, min_periods=window_bins // 2).mean()
    return (s - trend).to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# 2b: HP λ derivation + detrend robustness
# ---------------------------------------------------------------------------

def _hp_lambda_derivation() -> float:
    """Print derivation and return λ_5."""
    lam_annual = 1600.0
    p = BIN_DAYS
    lam_p = lam_annual * (365.0 / p) ** 4
    log.info("=== HP λ derivation ===")
    log.info("  λ_annual = 1600  (Ravn & Uhlig 2002 standard for annual data)")
    log.info("  For bin period p = %d days:", p)
    log.info("  λ_p = 1600 × (365/p)^4 = 1600 × (%.1f)^4 = %.4e", 365.0 / p, lam_p)
    log.info("  → λ_5 ≈ %.2e  (used throughout this analysis)", lam_p)
    return lam_p


def run_2b(cr: pd.Series, sei: pd.Series) -> None:
    lam = _hp_lambda_derivation()

    lag_bins = np.arange(-LAG_RANGE // BIN_DAYS, LAG_RANGE // BIN_DAYS + 1)
    lags_days = lag_bins * BIN_DAYS

    idx = cr.index.intersection(sei.index)
    cr_a  = cr.reindex(idx).to_numpy(float)
    sei_a = sei.reindex(idx).to_numpy(float)

    # Four detrending variants
    variants = [
        ("Raw (no detrending)",           cr_a,                          sei_a),
        ("HP filter  λ=1.29×10⁵",        _hp_detrend(cr_a, lam),        _hp_detrend(sei_a, lam)),
        ("Butterworth highpass  (2-yr)",  _butterworth_highpass(cr_a),   _butterworth_highpass(sei_a)),
        ("12-month rolling mean removal", _rolling_mean_subtract(cr_a),  _rolling_mean_subtract(sei_a)),
    ]

    colors = ["#555555", "#1b7837", "#2166ac", "#d6604d"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=False)
    fig.suptitle(
        "Detrending robustness: CR–seismic cross-correlation r(τ)\n"
        "In-sample window 1976–2019, corrected seismic metric log₁₀(Σ E [J])",
        fontsize=11, fontweight="bold",
    )

    for ax, (label, cr_v, sei_v), color in zip(axes.flat, variants, colors):
        r_vals = xcorr(cr_v, sei_v, lag_bins)

        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(15, color="crimson", ls="--", lw=0.9, alpha=0.7, label="τ=+15 d")
        ax.axvline(0, color="k", lw=0.5, alpha=0.3)
        ax.plot(lags_days, r_vals, color=color, lw=1.3)

        # Annotate peak and τ=+15d
        valid = np.isfinite(r_vals)
        if valid.any():
            pk_i  = np.nanargmax(np.abs(r_vals))
            pk_r  = r_vals[pk_i]
            pk_lg = lags_days[pk_i]
            r15   = r_vals[np.searchsorted(lag_bins, 3)]   # lag+15d = bin 3
            ax.scatter([pk_lg], [pk_r], color=color, s=40, zorder=5)
            ax.text(0.02, 0.97,
                    f"peak r={pk_r:+.3f} @ τ={pk_lg:+d}d\nr(+15d)={r15:+.3f}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85))

        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(lags_days[0], lags_days[-1])

    for ax in axes[1]:
        ax.set_xlabel("Lag τ (days)  [τ>0 = CR leads seismic]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Pearson r")

    fig.tight_layout()
    path = FIG_DIR / "detrend_robustness.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("2b figure saved: %s", path)


# ---------------------------------------------------------------------------
# 2c: Neff comparison
# ---------------------------------------------------------------------------

def _n_eff_bretherton_full(x: np.ndarray, y: np.ndarray, K: int = 200) -> float:
    """
    Full Bretherton 1999 Neff summing over lags 1…K:
        Neff = N / (1 + 2 Σ_{k=1}^{K} r_xx(k) r_yy(k))
    """
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    n = min(len(x), len(y))
    if n < 10:
        return float(n)
    x = x[:n]; y = y[:n]
    xc = x - x.mean(); yc = y - y.mean()
    var_x = np.dot(xc, xc); var_y = np.dot(yc, yc)
    if var_x < 1e-15 or var_y < 1e-15:
        return float(n)
    K_use = min(K, n - 2)
    s = 0.0
    for k in range(1, K_use + 1):
        rxx = np.dot(xc[:-k], xc[k:]) / var_x
        ryy = np.dot(yc[:-k], yc[k:]) / var_y
        s += rxx * ryy
    denom = 1.0 + 2.0 * s
    return float(np.clip(n / denom, 3.0, n))


def _n_eff_bartlett(x: np.ndarray, y: np.ndarray) -> float:
    """
    Bartlett 1946 first-order estimate:
        Neff = N · (1 − ρ₁ₓ ρ₁ᵧ) / (1 + ρ₁ₓ ρ₁ᵧ)
    """
    # This is what the current n_eff_bretherton() in surrogates.py computes.
    def r1(v):
        v = v[np.isfinite(v)]
        if len(v) < 4:
            return 0.0
        vc = v - v.mean()
        var = np.dot(vc, vc)
        if var < 1e-15:
            return 0.0
        return float(np.clip(np.dot(vc[:-1], vc[1:]) / var, -0.9999, 0.9999))

    n = min(len(x), len(y))
    r1x, r1y = r1(x), r1(y)
    denom = 1.0 + r1x * r1y
    if abs(denom) < 1e-12:
        return 3.0
    return float(np.clip(n * (1.0 - r1x * r1y) / denom, 3.0, n))


def _n_eff_monte_carlo(
    x: np.ndarray,
    y: np.ndarray,
    lag_bin: int = 3,
    n_reps: int = 1000,
    seed: int = SEED,
) -> float:
    """
    Monte Carlo Neff via phase randomisation of y.
    SE_r = std of r(lag_bin) under the null ≈ 1/sqrt(Neff)
    → Neff_mc = 1 / SE_r²
    """
    rng = np.random.default_rng(seed)
    N   = len(x)
    ok  = np.isfinite(x) & np.isfinite(y)
    x_c = x[ok]; y_c = y[ok]
    n   = len(x_c)

    rs = np.empty(n_reps)
    for i in range(n_reps):
        y_surr = phase_randomise(y_c, seed=rng)
        # Align at given lag_bin (positive = x leads y)
        if lag_bin >= 0:
            xa = x_c[:n - lag_bin]; ya = y_surr[lag_bin:]
        else:
            xa = x_c[-lag_bin:]; ya = y_surr[:n + lag_bin]
        if len(xa) < 4:
            rs[i] = 0.0
            continue
        rs[i], _ = scipy.stats.pearsonr(xa, ya)

    se = float(np.std(rs, ddof=1))
    if se < 1e-12:
        return float(n)
    return float(np.clip(1.0 / se**2, 3.0, n))


def run_2c(cr: pd.Series, sei: pd.Series) -> dict:
    log.info("=== 2c: Neff comparison ===")
    idx = cr.index.intersection(sei.index)
    x = cr.reindex(idx).to_numpy(float)
    y = sei.reindex(idx).to_numpy(float)
    N = (~np.isnan(x) & ~np.isnan(y)).sum()

    r15, _ = scipy.stats.pearsonr(
        x[np.isfinite(x) & np.isfinite(y)],
        y[np.isfinite(x) & np.isfinite(y)],
    )
    # Actually compute at lag +15d (bin 3)
    ok = np.isfinite(x) & np.isfinite(y)
    lag = 3
    xa, ya = x[:len(x) - lag], y[lag:]
    ok2 = np.isfinite(xa) & np.isfinite(ya)
    r15, _ = scipy.stats.pearsonr(xa[ok2], ya[ok2])

    neff_bartlett   = _n_eff_bartlett(x, y)
    neff_breth_full = _n_eff_bretherton_full(x, y)
    log.info("Running Monte Carlo Neff (1000 surrogates) …")
    neff_mc         = _n_eff_monte_carlo(x, y, lag_bin=3, n_reps=1000)

    def _ci(r, n):
        if n < 4 or not np.isfinite(r):
            return np.nan, np.nan
        se = 1.0 / np.sqrt(n - 3)
        z  = np.arctanh(r)
        return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))

    results = {
        "N_raw": int(N),
        "r_at_tau_plus15d": round(float(r15), 5),
        "methods": {
            "Bartlett_1946_first_order": {
                "Neff": round(neff_bartlett, 1),
                "CI_95_r": _ci(r15, neff_bartlett),
                "note": "N·(1−ρ₁ₓρ₁ᵧ)/(1+ρ₁ₓρ₁ᵧ)  [current implementation]",
            },
            "Bretherton_1999_full_sum": {
                "Neff": round(neff_breth_full, 1),
                "CI_95_r": _ci(r15, neff_breth_full),
                "note": "N / (1 + 2Σ_k ρ_xx(k)ρ_yy(k))  K=200 lags",
            },
            "Monte_Carlo_phase_surrogates": {
                "Neff": round(neff_mc, 1),
                "CI_95_r": _ci(r15, neff_mc),
                "note": "1/Var(r_null) from 1000 phase-randomised y series",
            },
        },
    }
    log.info("  N_raw=%d  r(+15d)=%.4f", N, r15)
    for k, v in results["methods"].items():
        ci = v["CI_95_r"]
        log.info("  %-40s  Neff=%6.0f  95%%CI=[%.3f, %.3f]", k, v["Neff"], ci[0], ci[1])

    out = OUT_DIR / "neff_comparison.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    log.info("2c results saved: %s", out)
    return results


# ---------------------------------------------------------------------------
# 2d: Magnitude threshold sensitivity
# ---------------------------------------------------------------------------

def run_2d(cr: pd.Series) -> None:
    log.info("=== 2d: Magnitude threshold sensitivity ===")
    thresholds = [4.5, 5.0, 6.0]
    lag_bins  = np.arange(-LAG_RANGE // BIN_DAYS, LAG_RANGE // BIN_DAYS + 1)
    lags_days = lag_bins * BIN_DAYS

    colors = ["#1b7837", "#2166ac", "#d6604d"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    fig.suptitle(
        "Magnitude threshold sensitivity: CR–seismic cross-correlation r(τ)\n"
        "In-sample 1976–2019, no detrending, log₁₀(Σ E [J]) metric",
        fontsize=10, fontweight="bold",
    )

    summary = {}
    for ax, Mmin, color in zip(axes, thresholds, colors):
        log.info("  Loading M≥%.1f seismic series …", Mmin)
        sei = _load_seismic(STUDY_START, STUDY_END, min_mag=Mmin, ref_index=cr.index)
        idx = cr.index.intersection(sei.index)
        cr_a  = cr.reindex(idx).to_numpy(float)
        sei_a = sei.reindex(idx).to_numpy(float)

        n_events_approx = sei_a[np.isfinite(sei_a)].size
        r_vals = xcorr(cr_a, sei_a, lag_bins)

        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(15, color="crimson", ls="--", lw=0.9, alpha=0.7)
        ax.axvline(0, color="k", lw=0.5, alpha=0.3)
        ax.plot(lags_days, r_vals, color=color, lw=1.3)

        valid = np.isfinite(r_vals)
        r15  = float(r_vals[np.searchsorted(lag_bins, 3)])   # τ=+15d bin
        pk_i = int(np.nanargmax(np.abs(r_vals)))
        pk_r  = float(r_vals[pk_i])
        pk_lg = int(lags_days[pk_i])

        # Flag empty-bin fraction (relevant at M≥6.0)
        n_nan = int(np.isnan(sei_a).sum())
        frac_empty = 100.0 * n_nan / len(sei_a)

        ax.scatter([pk_lg], [pk_r], color=color, s=40, zorder=5)
        note = f"⚠ {frac_empty:.0f}% empty bins" if frac_empty > 5 else ""
        ax.text(0.02, 0.97,
                f"peak r={pk_r:+.3f} @ τ={pk_lg:+d}d\nr(+15d)={r15:+.3f}\n{note}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85))

        ax.set_title(f"M ≥ {Mmin:.1f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Lag τ (days)  [τ>0 = CR leads seismic]")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(lags_days[0], lags_days[-1])

        summary[f"M_ge_{Mmin}"] = {
            "r_at_tau_plus15d": round(r15, 4),
            "peak_r": round(pk_r, 4),
            "peak_lag_days": pk_lg,
            "frac_empty_bins_pct": round(frac_empty, 1),
        }
        log.info("  M≥%.1f: r(+15d)=%.4f  peak r=%.4f @ %+d d  empty=%.1f%%",
                 Mmin, r15, pk_r, pk_lg, frac_empty)

    axes[0].set_ylabel("Pearson r")
    fig.tight_layout()
    path = FIG_DIR / "magnitude_threshold.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("2d figure saved: %s", path)

    # Flag divergence
    r15_vals = [summary[k]["r_at_tau_plus15d"] for k in summary]
    if max(r15_vals) - min(r15_vals) > 0.05:
        log.warning(
            "r(+15d) shifts substantially across thresholds (range %.3f)."
            " May indicate catalogue incompleteness or aftershock contamination "
            "at lower magnitudes.", max(r15_vals) - min(r15_vals),
        )

    out = OUT_DIR / "magnitude_sensitivity.json"
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("2d results saved: %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading CR index …")
    cr = _load_cr(STUDY_START, STUDY_END)

    log.info("Loading seismic metric (M≥4.5) …")
    sei = _load_seismic(STUDY_START, STUDY_END, min_mag=4.5, ref_index=cr.index)

    log.info("Running 2b: HP λ derivation + detrend robustness …")
    run_2b(cr, sei)

    log.info("Running 2c: Neff comparison …")
    neff_results = run_2c(cr, sei)

    log.info("Running 2d: Magnitude threshold sensitivity …")
    run_2d(cr)

    log.info("All robustness checks complete.")

    # Print Neff summary table
    print("\n" + "=" * 70)
    print("  Neff COMPARISON  (in-sample 1976–2019, raw series, τ=+15 d)")
    print("=" * 70)
    print(f"  N_raw = {neff_results['N_raw']}    r(+15d) = {neff_results['r_at_tau_plus15d']:.4f}")
    print()
    for method, vals in neff_results["methods"].items():
        ci = vals["CI_95_r"]
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if all(np.isfinite(c) for c in ci) else "N/A"
        print(f"  {method:<42}  Neff={vals['Neff']:>6.0f}  95%CI_r={ci_str}")
    print("=" * 70)


if __name__ == "__main__":
    main()
