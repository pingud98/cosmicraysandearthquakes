#!/usr/bin/env python3
"""
scripts/11_additional_robustness.py
====================================
Seven additional robustness checks for the CR–seismic correlation analysis.

  3a  Block-bootstrap surrogates vs IAAFT null
      Circular block bootstrap (block ≈ 1 solar cycle = 803 bins ≈ 11 yr at 5-day
      resolution), 5 000 surrogates; compare p-values to stored IAAFT results.

  3b  Partial correlation controlling for sunspots (raw, unfiltered data).
      Seismic_resid = seismic − β·sunspots (OLS); compute r(τ) between CR and
      Seismic_resid without any HP-filter preprocessing.

  3c  Spectral coherence + mutual information (kNN estimator).
      Magnitude-squared coherence at each frequency; MI at lag=0 and lag=+3 bins
      (+15 d) vs shuffle null (1 000 permutations).

  3d  Missing-data impact.
      Fraction of NaN bins per station threshold (2, 3, 5), temporal clustering of
      NaN bins near solar maxima, and r(+15 d) sensitivity.

  3e  Bin-size sensitivity (1-day, 5-day, 27-day Bartels rotation bins).

  3f  Gardner–Knopoff earthquake declustering.
      Remove aftershock sequences; recompute seismic energy from mainshock-only
      catalogue; compare r(τ) to the clustered result.

  3g  Sub-period analysis (per-solar-cycle cross-correlations).
      Cycles 21–24 (≈ 1976–2019); independent r(τ) per cycle.

Outputs
-------
  results/additional_robustness.json        — machine-readable summary
  results/figs/block_bootstrap_null.png     — 3a
  results/figs/partial_correlation.png      — 3b
  results/figs/spectral_coherence.png       — 3c
  results/figs/bin_size_sensitivity.png     — 3e
  results/figs/declustered_xcorr.png        — 3f
  results/figs/solar_cycle_xcorr.png        — 3g
  (3d is table-only, included in JSON)
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
import scipy.spatial
import scipy.stats
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from crq.ingest.nmdb import load_station, resample_daily
from crq.ingest.usgs import load_usgs, seismic_energy_per_bin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STUDY_START  = "1976-01-01"
STUDY_END    = "2019-12-31"
BIN_DAYS     = 5
MIN_MAG      = 4.5
COV_THRESH   = 0.60
MIN_STATIONS = 3
SEED         = 42
LAG_BINS     = np.arange(-200, 205, 1)    # ±1000 d in 5-d steps
CLAIM_BIN    = 3                           # +15 d = bin 3

OUT_DIR  = ROOT / "results"
FIG_DIR  = ROOT / "results" / "figs"
NMDB_DIR = ROOT / "data" / "raw" / "nmdb"
USGS_DIR = ROOT / "data" / "raw" / "usgs"
CFG_FILE = ROOT / "config" / "stations.yaml"
SN_FILE  = ROOT / "data" / "raw" / "sidc" / "sunspots.csv"

FIG_DIR.mkdir(parents=True, exist_ok=True)

SOLAR_CYCLES = {
    21: ("1976-03-01", "1986-09-01"),
    22: ("1986-09-01", "1996-08-01"),
    23: ("1996-08-01", "2008-12-01"),
    24: ("2008-12-01", "2019-12-31"),
}


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _station_names() -> list[str]:
    with open(CFG_FILE) as fh:
        return list(yaml.safe_load(fh)["stations"].keys())


def _bin_daily(daily: pd.Series, t0: pd.Timestamp, bin_days: int,
               agg: str = "mean") -> pd.Series:
    days = (daily.index - t0).days
    bin_num = days // bin_days
    bin_dates = t0 + pd.to_timedelta(bin_num * bin_days, unit="D")
    g = daily.groupby(bin_dates)
    return g.sum() if agg == "sum" else g.mean()


def _load_cr(study_start: str, study_end: str,
             bin_days: int = BIN_DAYS,
             min_stations: int = MIN_STATIONS) -> tuple[pd.Series, pd.DataFrame]:
    """Return (cr_index, station_daily_norm)."""
    t0 = pd.Timestamp(study_start)
    t1 = pd.Timestamp(study_end)
    norm: dict[str, pd.Series] = {}
    for stn in _station_names():
        hourly = load_station(stn, t0.year, t1.year, NMDB_DIR)
        if hourly.empty:
            continue
        daily = resample_daily(hourly, stn, coverage_threshold=COV_THRESH)[stn]
        daily = daily.loc[study_start:study_end]
        if daily.notna().sum() < 30:
            continue
        mu = daily.mean()
        if not (np.isfinite(mu) and mu > 0):
            continue
        norm[stn] = daily / mu

    if not norm:
        raise RuntimeError("No NMDB data.")

    mat = pd.DataFrame(norm)
    n_valid = mat.notna().sum(axis=1)
    global_daily = mat.mean(axis=1)
    global_daily[n_valid < min_stations] = np.nan

    cr = _bin_daily(global_daily, t0, bin_days)
    cr.name = "cr_index"
    log.info("CR (%d-d bins, min_stn=%d): %d bins, %d stations",
             bin_days, min_stations, len(cr), len(norm))
    return cr, mat


def _load_seismic_events(study_start: str, study_end: str,
                          min_mag: float = MIN_MAG) -> pd.DataFrame:
    """Return raw USGS event DataFrame with latitude, longitude, mag columns."""
    t0 = pd.Timestamp(study_start)
    t1 = pd.Timestamp(study_end)
    ev = load_usgs(t0.year, t1.year, USGS_DIR)
    ev = ev.loc[study_start:study_end]
    ev = ev[ev["mag"] >= min_mag].copy()
    log.info("Events M≥%.1f: %d in %s–%s", min_mag, len(ev),
             study_start, study_end)
    return ev


def _energy_metric(events: pd.DataFrame, study_start: str, study_end: str,
                   bin_days: int, ref_index: pd.DatetimeIndex | None = None,
                   min_mag: float = MIN_MAG) -> pd.Series:
    t0 = pd.Timestamp(study_start)
    s = seismic_energy_per_bin(events, study_start, study_end, bin_days, t0,
                               min_mag=min_mag)
    if ref_index is not None:
        s = s.reindex(ref_index, fill_value=float(s.min()))
    else:
        s = s.fillna(float(s.min()))
    return s


def _load_sunspots_binned(study_start: str, study_end: str,
                           bin_days: int = BIN_DAYS) -> pd.Series:
    """Parse the KSO comma-sep sunspot CSV → 5-day binned daily mean."""
    df = pd.read_csv(SN_FILE, header=0)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["Date"].str.strip(), errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    sn = pd.to_numeric(df["Total"].astype(str).str.strip(), errors="coerce")
    sn = sn.loc[study_start:study_end].rename("sunspot")
    # Resample to daily (data is already monthly-ish — forward-fill)
    daily_idx = pd.date_range(study_start, study_end, freq="D")
    sn = sn.reindex(daily_idx).interpolate("linear").ffill().bfill()
    t0 = pd.Timestamp(study_start)
    return _bin_daily(sn, t0, bin_days)


def xcorr(x: np.ndarray, y: np.ndarray, lag_bins: np.ndarray) -> np.ndarray:
    """Pearson r(τ) for each lag bin; τ>0 means x leads y."""
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
            xo = xa[ok] - xa[ok].mean()
            yo = ya[ok] - ya[ok].mean()
            denom = np.sqrt((xo ** 2).sum() * (yo ** 2).sum())
            if denom > 1e-12:
                rs[i] = np.dot(xo, yo) / denom
    return rs


# ---------------------------------------------------------------------------
# 3a — Block-bootstrap surrogates
# ---------------------------------------------------------------------------

def _circular_block_bootstrap(
    x: np.ndarray, y: np.ndarray,
    block_len: int,
    n_surr: int,
    lag_bins: np.ndarray,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Circular block bootstrap (CBB) null distribution.

    Independently resample *x* and *y* with replacement using blocks of length
    *block_len* (circular, so the series wraps).  Returns arrays of shape
    (n_surr,) containing the surrogate r(+15d) and peak |r| values.

    Vectorised: all surrogates are built into a matrix (n_surr × N) and
    the correlation is computed for all surrogates simultaneously per lag,
    avoiding a 5000 × 401 = 2 M Python-loop iterations.
    """
    rng = np.random.default_rng(seed)
    N = len(x)
    n_blocks = int(np.ceil(N / block_len))
    n_lags = len(lag_bins)

    # Build surrogate matrices (n_surr × N) — Python loop only over n_surr
    log.info("  Building %d surrogate series (block_len=%d) …", n_surr, block_len)
    xs_mat = np.empty((n_surr, N), dtype=np.float64)
    ys_mat = np.empty((n_surr, N), dtype=np.float64)
    for i in range(n_surr):
        starts_x = rng.integers(0, N, size=n_blocks)
        starts_y = rng.integers(0, N, size=n_blocks)
        xs_mat[i] = np.concatenate([np.roll(x, -s)[:block_len] for s in starts_x])[:N]
        ys_mat[i] = np.concatenate([np.roll(y, -s)[:block_len] for s in starts_y])[:N]

    # Vectorised xcorr: loop over lags only (all surrogates in parallel)
    log.info("  Vectorised xcorr over %d lags …", n_lags)
    rs_mat = np.full((n_surr, n_lags), np.nan, dtype=np.float64)
    for j, lag in enumerate(lag_bins):
        if lag >= 0:
            xa = xs_mat[:, :N - lag] if lag > 0 else xs_mat
            ya = ys_mat[:, lag:]     if lag > 0 else ys_mat
        else:
            absl = -lag
            xa = xs_mat[:, absl:]
            ya = ys_mat[:, :N - absl]
        xm = xa - xa.mean(axis=1, keepdims=True)
        ym = ya - ya.mean(axis=1, keepdims=True)
        num   = (xm * ym).sum(axis=1)
        denom = np.sqrt((xm ** 2).sum(axis=1) * (ym ** 2).sum(axis=1))
        valid = denom > 1e-12
        rs_mat[valid, j] = num[valid] / denom[valid]

    claim_idx = np.where(lag_bins == CLAIM_BIN)[0]
    rs_15 = rs_mat[:, claim_idx[0]] if len(claim_idx) else np.full(n_surr, np.nan)
    rs_pk = np.nanmax(np.abs(rs_mat), axis=1)
    return rs_15, rs_pk


def run_3a(cr: pd.Series, sei: pd.Series) -> dict:
    log.info("=== 3a: Block-bootstrap surrogates ===")
    x = cr.values.copy()
    y = sei.values.copy()

    # Fill NaN with series mean for bootstrap (preserves length)
    x = np.where(np.isfinite(x), x, np.nanmean(x))
    y = np.where(np.isfinite(y), y, np.nanmean(y))

    N = len(x)
    # Block length ≈ 1 solar cycle (11 yr ≈ 4016 d ÷ 5 d/bin = 803 bins)
    block_len = int(round(11 * 365.25 / BIN_DAYS))
    log.info("  N=%d  block_len=%d bins (≈%.1f yr)  surrogates=5000",
             N, block_len, block_len * BIN_DAYS / 365.25)

    rs_15, rs_pk = _circular_block_bootstrap(x, y, block_len, 5000, LAG_BINS)

    obs_rv = xcorr(x, y, LAG_BINS)
    obs_15 = obs_rv[LAG_BINS == CLAIM_BIN][0]
    obs_pk = np.nanmax(np.abs(obs_rv))

    p_15 = float(np.mean(np.abs(rs_15) >= np.abs(obs_15)))
    p_pk = float(np.mean(rs_pk >= obs_pk))
    ci95_15 = (float(np.percentile(rs_15, 2.5)), float(np.percentile(rs_15, 97.5)))

    log.info("  obs r(+15d)=%.4f  block-bootstrap p(+15d)=%.4f  p(peak)=%.4f",
             obs_15, p_15, p_pk)

    # Load IAAFT p-value for comparison
    iaaft_raw = None
    try:
        with open(OUT_DIR / "detrended_results.json") as fh:
            det = json.load(fh)
        iaaft_raw = det["results"][0]["p_global_iaaft"]
    except Exception:
        pass

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(rs_15, bins=60, color="steelblue", alpha=0.7,
                 label="Block-bootstrap null")
    axes[0].axvline(obs_15, color="red", lw=2, label=f"Observed r={obs_15:.3f}")
    axes[0].axvline(ci95_15[0], color="grey", ls="--")
    axes[0].axvline(ci95_15[1], color="grey", ls="--", label="95% CI")
    axes[0].set_xlabel("r(+15 d)")
    axes[0].set_title(f"Block bootstrap null\np(+15d) = {p_15:.3f}")
    axes[0].legend(fontsize=8)

    axes[1].hist(rs_pk, bins=60, color="darkorange", alpha=0.7,
                 label="Block-bootstrap null")
    axes[1].axvline(obs_pk, color="red", lw=2, label=f"Observed peak |r|={obs_pk:.3f}")
    axes[1].set_xlabel("Peak |r|")
    axes[1].set_title(f"Peak |r| null\np(peak) = {p_pk:.3f}")
    axes[1].legend(fontsize=8)

    if iaaft_raw is not None:
        axes[0].set_xlabel("r(+15 d)" + f"\n(IAAFT p_global={iaaft_raw:.4f})")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "block_bootstrap_null.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("3a figure saved.")

    return dict(
        block_len_bins=block_len,
        obs_r_15d=float(obs_15),
        obs_peak_r=float(obs_pk),
        p_block_15d=p_15,
        p_block_peak=p_pk,
        ci95_15d=list(ci95_15),
        iaaft_p_global_raw=iaaft_raw,
        interpretation=(
            "p-value similar to IAAFT" if iaaft_raw is None or abs(p_15 - iaaft_raw) < 0.1
            else "p-value changes materially vs IAAFT"
        ),
    )


# ---------------------------------------------------------------------------
# 3b — Partial correlation controlling for sunspots (raw data)
# ---------------------------------------------------------------------------

def run_3b(cr: pd.Series, sei: pd.Series) -> dict:
    log.info("=== 3b: Partial correlation (sunspot control) ===")
    sn = _load_sunspots_binned(STUDY_START, STUDY_END)

    # Align all three series
    idx = cr.index.intersection(sei.index).intersection(sn.index)
    x = cr.reindex(idx).values.astype(float)
    y = sei.reindex(idx).values.astype(float)
    z = sn.reindex(idx).values.astype(float)

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    log.info("  Aligned N=%d, valid=%d", len(x), ok.sum())

    # OLS: regress seismic on sunspot
    z_ok = z[ok]; y_ok = y[ok]; x_ok = x[ok]
    slope, intercept, *_ = scipy.stats.linregress(z_ok, y_ok)
    y_resid = y_ok - (slope * z_ok + intercept)
    log.info("  Sunspot regression on seismic: slope=%.4f  intercept=%.4f", slope, intercept)

    r_partial, pv = scipy.stats.pearsonr(x_ok, y_resid)
    log.info("  Partial r(CR, Seismic|Sunspot) at zero lag = %.4f  p=%.4e",
             r_partial, pv)

    # Full cross-correlation of CR vs seismic residual
    x_full = np.full(len(idx), np.nan)
    y_res_full = np.full(len(idx), np.nan)
    x_full[ok] = x_ok
    y_res_full[ok] = y_resid

    rv_partial = xcorr(x_full, y_res_full, LAG_BINS)
    rv_raw     = xcorr(x, y, LAG_BINS)

    r_partial_15 = rv_partial[LAG_BINS == CLAIM_BIN][0]
    r_raw_15     = rv_raw[LAG_BINS == CLAIM_BIN][0]
    peak_partial = float(np.nanmax(np.abs(rv_partial)))
    log.info("  r(+15d) raw=%.4f  partial=%.4f", r_raw_15, r_partial_15)

    # Figure
    fig, ax = plt.subplots(figsize=(9, 4))
    lags_d = LAG_BINS * BIN_DAYS
    ax.plot(lags_d, rv_raw, color="steelblue", alpha=0.7, label="Raw seismic")
    ax.plot(lags_d, rv_partial, color="darkorange", lw=1.5,
            label="Seismic residual (sunspot removed)")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(15, color="grey", ls="--", lw=1.2, label="+15 d")
    ax.set_xlabel("Lag τ (days)")
    ax.set_ylabel("Pearson r")
    ax.set_title(f"Partial correlation: CR vs seismic after sunspot regression\n"
                 f"r_raw(+15d)={r_raw_15:.3f}  r_partial(+15d)={r_partial_15:.3f}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "partial_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("3b figure saved.")

    drop_frac = (abs(r_raw_15) - abs(r_partial_15)) / max(abs(r_raw_15), 1e-9)
    return dict(
        sunspot_slope=float(slope),
        r_raw_15d=float(r_raw_15),
        r_partial_15d=float(r_partial_15),
        drop_fraction_at_15d=float(drop_frac),
        peak_r_partial=float(peak_partial),
        interpretation=(
            "Partial correlation near zero — solar-cycle confounding confirmed"
            if abs(r_partial_15) < 0.05 else
            "Partial correlation non-trivial — residual signal remains after sunspot control"
        ),
    )


# ---------------------------------------------------------------------------
# 3c — Spectral coherence + mutual information
# ---------------------------------------------------------------------------

def _mi_knn(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """
    Kraskov et al. (2004) kNN mutual information estimator (estimator 1).
    Uses Chebyshev (L-inf) metric in joint space; marginal counts via
    vectorised searchsorted — O(N log N), no Python loop over points.
    Requires finite values only.
    """
    from scipy.special import digamma
    N = len(x)
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    xy = np.column_stack([x, y])

    # k-NN in joint space using Chebyshev (L-inf) metric
    tree_xy = scipy.spatial.cKDTree(xy)
    dists, _ = tree_xy.query(xy, k=k + 1, p=np.inf, workers=-1)
    eps = dists[:, -1]  # k-th neighbour Chebyshev distance

    # Count marginal neighbours via sorted binary search — O(N log N), vectorised
    xs = np.sort(x)
    ys = np.sort(y)
    nx = (np.searchsorted(xs, x + eps, side="right")
          - np.searchsorted(xs, x - eps, side="left") - 1)
    ny = (np.searchsorted(ys, y + eps, side="right")
          - np.searchsorted(ys, y - eps, side="left") - 1)
    nx = np.maximum(nx, 0)
    ny = np.maximum(ny, 0)

    mi = digamma(k) + digamma(N) - np.mean(digamma(nx + 1)) - np.mean(digamma(ny + 1))
    return float(max(mi, 0.0))


def run_3c(cr: pd.Series, sei: pd.Series) -> dict:
    log.info("=== 3c: Spectral coherence + mutual information ===")
    rng = np.random.default_rng(SEED)

    idx = cr.index.intersection(sei.index)
    x = cr.reindex(idx).values.astype(float)
    y = sei.reindex(idx).values.astype(float)
    ok = np.isfinite(x) & np.isfinite(y)
    xf, yf = x[ok], y[ok]

    # --- Spectral coherence ---
    # nperseg=2048 gives freq. resolution (1/5)/2048 = 0.036 cycles/yr,
    # sufficient to resolve the solar-cycle band (0.08–0.115 cycles/yr).
    # nperseg=512 only gives 0.143 cycles/yr — no bins fall in the SC band.
    nperseg = 2048
    fs = 1.0 / BIN_DAYS   # cycles per day
    f_d, Cxy = scipy.signal.coherence(xf, yf, fs=fs, nperseg=nperseg,
                                       window="hann", noverlap=nperseg * 3 // 4)
    f_yr = f_d * 365.25   # cycles per year

    # Solar cycle band: 0.08–0.11 cycles/yr (9–12 year period)
    sc_mask = (f_yr >= 0.08) & (f_yr <= 0.115)
    coh_sc = float(np.mean(Cxy[sc_mask]))
    log.info("  Mean coherence in solar-cycle band: %.4f", coh_sc)

    # Significance: 95th percentile of coherence for white-noise pairs
    noverlap_used = nperseg * 3 // 4
    step = nperseg - noverlap_used
    n_seg = max(1, (ok.sum() - nperseg) // step + 1)
    coh_95 = 1 - 0.05 ** (1.0 / (n_seg - 1)) if n_seg > 1 else np.nan
    log.info("  Coherence 95%% significance threshold (for %d segments): %.4f",
             n_seg, coh_95)

    # --- Mutual information ---
    N = len(xf)
    mi_lag0 = _mi_knn(xf, yf)
    # MI at lag +15d (=+3 bins)
    lag15 = CLAIM_BIN
    xi15, yi15 = xf[:N - lag15], yf[lag15:]
    ok15 = np.isfinite(xi15) & np.isfinite(yi15)
    mi_lag15 = _mi_knn(xi15[ok15], yi15[ok15])
    log.info("  MI(lag=0)=%.4f  MI(lag=+15d)=%.4f", mi_lag0, mi_lag15)

    # Shuffle null for MI (1000 permutations)
    n_shuf = 1000
    mi_null_0  = np.empty(n_shuf)
    mi_null_15 = np.empty(n_shuf)
    for i in range(n_shuf):
        ys = rng.permutation(yf)
        mi_null_0[i]  = _mi_knn(xf, ys)
        ys15 = rng.permutation(yf[lag15:])
        mi_null_15[i] = _mi_knn(xf[:N - lag15][ok15], ys15[ok15])

    p_mi_0  = float(np.mean(mi_null_0  >= mi_lag0))
    p_mi_15 = float(np.mean(mi_null_15 >= mi_lag15))
    log.info("  p(MI lag=0)=%.4f  p(MI lag=+15d)=%.4f", p_mi_0, p_mi_15)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Coherence panel
    ax = axes[0]
    ax.plot(f_yr, Cxy, color="steelblue", lw=1, label="Coherence")
    ax.axvspan(0.08, 0.115, alpha=0.15, color="orange", label="Solar-cycle band (0.08–0.115 yr⁻¹)")
    if np.isfinite(coh_95):
        ax.axhline(coh_95, color="red", ls="--", lw=1.2,
                   label=f"95% sig. ({coh_95:.3f}, K={n_seg} seg.)")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Frequency (cycles yr⁻¹)")
    ax.set_ylabel("Magnitude-squared coherence")
    coh_sc_str = f"{coh_sc:.3f}" if np.isfinite(coh_sc) else "N/A"
    ax.set_title(f"Spectral coherence (nperseg={nperseg})\nMean coh. in SC band: {coh_sc_str}")
    ax.legend(fontsize=8)

    # MI panel
    ax = axes[1]
    ax.hist(mi_null_0, bins=40, color="steelblue", alpha=0.6, label="Shuffle null (lag=0)")
    ax.hist(mi_null_15, bins=40, color="darkorange", alpha=0.6, label="Shuffle null (lag=+15d)")
    ax.axvline(mi_lag0,  color="steelblue", lw=2, ls="-",  label=f"Obs MI(0)={mi_lag0:.3f}")
    ax.axvline(mi_lag15, color="darkorange", lw=2, ls="--",
               label=f"Obs MI(+15d)={mi_lag15:.3f}")
    ax.set_xlabel("Mutual information (nats)")
    ax.set_title(f"kNN mutual information\np(lag=0)={p_mi_0:.3f}  p(lag=+15d)={p_mi_15:.3f}")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "spectral_coherence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("3c figure saved.")

    return dict(
        coherence_solar_cycle_band=coh_sc,
        coherence_95pct_threshold=coh_95 if np.isfinite(coh_95) else None,
        mi_lag0=mi_lag0,
        mi_lag15d=mi_lag15,
        p_mi_lag0=p_mi_0,
        p_mi_lag15d=p_mi_15,
    )


# ---------------------------------------------------------------------------
# 3d — Missing-data sensitivity
# ---------------------------------------------------------------------------

SOLAR_MAXIMA = [1979.7, 1989.6, 2000.3, 2014.2]   # approximate decimal years


def _cr_nan_analysis(study_start: str, study_end: str,
                     min_stations: int) -> dict:
    """Return NaN stats and r(+15d) for the given station threshold."""
    cr, _ = _load_cr(study_start, study_end, bin_days=BIN_DAYS,
                     min_stations=min_stations)
    ev = _load_seismic_events(study_start, study_end)
    sei = _energy_metric(ev, study_start, study_end, BIN_DAYS, cr.index)

    nan_frac = float(cr.isna().mean())

    # Are NaN bins concentrated near solar maxima?
    is_nan = cr.isna()
    bin_years = cr.index.year + cr.index.dayofyear / 365.25
    # Flag years within ±1.5 yr of any solar maximum
    near_max = np.zeros(len(cr), dtype=bool)
    for sm in SOLAR_MAXIMA:
        near_max |= np.abs(bin_years - sm) <= 1.5
    nan_near = float(is_nan[near_max].mean())
    nan_far  = float(is_nan[~near_max].mean())

    # r(+15d)
    ok = np.isfinite(cr.values) & np.isfinite(sei.values)
    x, y = cr.values[ok], sei.values[ok]
    r15, _ = scipy.stats.pearsonr(x, y)   # zero-lag as proxy; lag-3 below
    # Actual lag-3
    N = len(cr)
    xa = cr.values[:N - CLAIM_BIN]; ya = sei.values[CLAIM_BIN:]
    ok2 = np.isfinite(xa) & np.isfinite(ya)
    r_15d, _ = scipy.stats.pearsonr(xa[ok2], ya[ok2])

    return dict(
        min_stations=min_stations,
        nan_fraction=nan_frac,
        nan_fraction_near_solar_max=nan_near,
        nan_fraction_far_from_solar_max=nan_far,
        clustering_ratio=nan_near / max(nan_far, 1e-9),
        r_at_15d=float(r_15d),
    )


def run_3d() -> dict:
    log.info("=== 3d: Missing-data sensitivity ===")
    results = []
    for thr in (2, 3, 5):
        log.info("  Station threshold = %d …", thr)
        res = _cr_nan_analysis(STUDY_START, STUDY_END, thr)
        results.append(res)
        log.info("    NaN=%.1f%%  near-max=%.1f%%  r(+15d)=%.4f",
                 100 * res["nan_fraction"],
                 100 * res["nan_fraction_near_solar_max"],
                 res["r_at_15d"])
    return dict(station_threshold_sensitivity=results)


# ---------------------------------------------------------------------------
# 3e — Bin-size sensitivity
# ---------------------------------------------------------------------------

def _run_for_bin_size(bin_days: int) -> dict:
    """Load CR and seismic at *bin_days* resolution, compute r(τ)."""
    cr, _ = _load_cr(STUDY_START, STUDY_END, bin_days=bin_days)
    ev = _load_seismic_events(STUDY_START, STUDY_END)
    sei = _energy_metric(ev, STUDY_START, STUDY_END, bin_days, cr.index)

    # Lag range: ±1000 days in steps of bin_days
    max_lag_bins = int(1000 / bin_days)
    lbs = np.arange(-max_lag_bins, max_lag_bins + 1)
    rv  = xcorr(cr.values, sei.values, lbs)

    # Find claimed lag bin (closest to +15d)
    claim_bin = int(round(15 / bin_days))
    r15 = rv[lbs == claim_bin][0] if np.any(lbs == claim_bin) else np.nan
    peak_idx = np.nanargmax(np.abs(rv))
    peak_r   = rv[peak_idx]
    peak_lag = int(lbs[peak_idx]) * bin_days   # days

    log.info("  %d-d bins: r(+%dd)=%.4f  peak r=%.4f @ τ=%dd",
             bin_days, claim_bin * bin_days, r15, peak_r, peak_lag)
    return dict(
        bin_days=bin_days,
        r_at_claimed_lag=float(r15),
        claimed_lag_days=claim_bin * bin_days,
        peak_r=float(peak_r),
        peak_lag_days=peak_lag,
        lag_bins=lbs.tolist(),
        r_values=rv.tolist(),
    )


def run_3e() -> dict:
    log.info("=== 3e: Bin-size sensitivity ===")
    results = []
    rv_by_bin = {}
    for bd in (1, 5, 27):
        r = _run_for_bin_size(bd)
        results.append({k: v for k, v in r.items() if k not in ("lag_bins", "r_values")})
        rv_by_bin[bd] = (r["lag_bins"], r["r_values"])

    # Figure: 3-panel
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    colours = ["steelblue", "darkorange", "seagreen"]
    for ax, (bd, (lbs, rv)), col in zip(axes, rv_by_bin.items(), colours):
        lags_d = np.array(lbs) * bd
        rv = np.array(rv)
        ax.plot(lags_d, rv, color=col, lw=0.8)
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(15, color="grey", ls="--", lw=1, label="+15 d")
        closest = lbs[np.argmin(np.abs(np.array(lbs) - round(15 / bd)))]
        pk_d = lbs[np.nanargmax(np.abs(rv))] * bd
        ax.axvline(pk_d, color="red", ls=":", lw=1, label=f"Peak τ={pk_d}d")
        ax.set_title(f"{bd}-day bins\nr(+15d≈{int(round(15/bd)*bd)}d)="
                     f"{rv[np.argmin(np.abs(np.array(lbs)-round(15/bd)))]:.3f}")
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("Pearson r")
        ax.legend(fontsize=7)
    fig.suptitle("Bin-size sensitivity: 1-day, 5-day, 27-day bins", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bin_size_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("3e figure saved.")

    return dict(bin_size_sensitivity=results)


# ---------------------------------------------------------------------------
# 3f — Gardner–Knopoff declustering
# ---------------------------------------------------------------------------

def _gk_time_window(mag: float) -> float:
    """After-shock time window T(M) in days (Gardner & Knopoff 1974)."""
    if mag < 6.5:
        return 10 ** (0.5 * mag - 1.8)
    else:
        return 10 ** (0.46 * mag - 2.31)


def _gk_dist_window(mag: float) -> float:
    """After-shock distance window L(M) in km (Gardner & Knopoff 1974)."""
    return 10 ** (0.1238 * mag + 0.983)


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray,
                  lat2: float, lon2: float) -> np.ndarray:
    """Vectorised haversine distance in km."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _gardner_knopoff_decluster(events: pd.DataFrame) -> pd.DataFrame:
    """
    Remove aftershocks from *events* using the Gardner-Knopoff (1974) algorithm.

    Returns the mainshock-only DataFrame.  Uses forward-only time windows
    (each mainshock's aftershock zone applies to all future events within T(M)).
    """
    ev = events.dropna(subset=["latitude", "longitude", "mag"]).copy()
    ev = ev.sort_index()
    n = len(ev)

    times  = ev.index.values.astype("datetime64[ns]").astype(np.int64) / 1e9 / 86400  # days
    lats   = ev["latitude"].values
    lons   = ev["longitude"].values
    mags   = ev["mag"].values
    is_as  = np.zeros(n, dtype=bool)

    log.info("  GK declustering: %d events …", n)
    for i in range(n):
        if is_as[i]:
            continue
        tw = _gk_time_window(mags[i])
        dw = _gk_dist_window(mags[i])
        # Forward window in time
        t_end = times[i] + tw
        j_lo = i + 1
        # Find j_hi via binary search (times is sorted)
        j_hi = np.searchsorted(times, t_end, side="right")
        if j_lo >= j_hi:
            continue
        cands = np.arange(j_lo, j_hi)
        cands = cands[~is_as[cands]]   # skip already-flagged
        if len(cands) == 0:
            continue
        dists = _haversine_km(lats[cands], lons[cands], lats[i], lons[i])
        is_as[cands[dists <= dw]] = True

    n_main = int((~is_as).sum())
    n_as   = int(is_as.sum())
    log.info("  GK result: %d mainshocks  %d aftershocks (%.1f%% removed)",
             n_main, n_as, 100 * n_as / max(n, 1))
    return ev[~is_as]


def run_3f(cr: pd.Series) -> dict:
    log.info("=== 3f: Gardner–Knopoff declustering ===")
    events = _load_seismic_events(STUDY_START, STUDY_END, min_mag=MIN_MAG)
    sei_full = _energy_metric(events, STUDY_START, STUDY_END, BIN_DAYS, cr.index)

    mainshocks = _gardner_knopoff_decluster(events)
    sei_decl = _energy_metric(mainshocks, STUDY_START, STUDY_END, BIN_DAYS, cr.index)

    rv_full = xcorr(cr.values, sei_full.values, LAG_BINS)
    rv_decl = xcorr(cr.values, sei_decl.values, LAG_BINS)

    r15_full = float(rv_full[LAG_BINS == CLAIM_BIN][0])
    r15_decl = float(rv_decl[LAG_BINS == CLAIM_BIN][0])
    pk_full  = float(np.nanmax(np.abs(rv_full)))
    pk_decl  = float(np.nanmax(np.abs(rv_decl)))
    log.info("  r(+15d) full=%.4f  declustered=%.4f", r15_full, r15_decl)

    n_removed_frac = float(1 - len(mainshocks) / len(events))

    # Figure
    fig, ax = plt.subplots(figsize=(9, 4))
    lags_d = LAG_BINS * BIN_DAYS
    ax.plot(lags_d, rv_full, color="steelblue", alpha=0.8,
            label=f"Full catalogue (n={len(events):,})")
    ax.plot(lags_d, rv_decl, color="darkorange", lw=1.5,
            label=f"Declustered / mainshocks only (n={len(mainshocks):,})")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(15, color="grey", ls="--", lw=1.2, label="+15 d")
    ax.set_xlabel("Lag τ (days)")
    ax.set_ylabel("Pearson r")
    ax.set_title(f"Gardner–Knopoff declustering\n"
                 f"r(+15d): full={r15_full:.3f}  declustered={r15_decl:.3f}  "
                 f"({100*n_removed_frac:.0f}% removed)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "declustered_xcorr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("3f figure saved.")

    return dict(
        n_events_full=len(events),
        n_mainshocks=len(mainshocks),
        fraction_removed=n_removed_frac,
        r_15d_full=r15_full,
        r_15d_declustered=r15_decl,
        peak_r_full=pk_full,
        peak_r_declustered=pk_decl,
        interpretation=(
            "Aftershock clustering is NOT a primary confound — result stable after GK declustering"
            if abs(r15_full - r15_decl) < 0.03 else
            "Result changes after GK declustering — aftershock contamination is a concern"
        ),
    )


# ---------------------------------------------------------------------------
# 3g — Sub-period analysis (per solar cycle)
# ---------------------------------------------------------------------------

def run_3g(cr: pd.Series, sei: pd.Series) -> dict:
    log.info("=== 3g: Sub-period analysis (per solar cycle) ===")
    lag_range_bins = np.arange(-40, 41)   # ±200 d in 5-d steps (short cycles)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    results = {}

    for ax, (cycle_num, (t_start, t_end)) in zip(axes.flat, SOLAR_CYCLES.items()):
        # Crop pre-aligned global CR/seismic series — avoids bin-date misalignment
        cr_c  = cr.loc[t_start:t_end]
        sei_c = sei.loc[t_start:t_end]
        if len(cr_c) < 40:
            log.warning("  Cycle %d: insufficient data (%d bins)", cycle_num, len(cr_c))
            ax.set_title(f"Cycle {cycle_num}: insufficient data")
            results[f"cycle_{cycle_num}"] = None
            continue

        rv = xcorr(cr_c.values, sei_c.values, lag_range_bins)

        if np.all(np.isnan(rv)):
            log.warning("  Cycle %d: all-NaN xcorr — skipping", cycle_num)
            ax.set_title(f"Cycle {cycle_num}: all-NaN")
            results[f"cycle_{cycle_num}"] = None
            continue

        r15 = float(rv[lag_range_bins == CLAIM_BIN][0]) if np.any(lag_range_bins == CLAIM_BIN) else np.nan
        pk_idx = np.nanargmax(np.abs(rv))
        pk_r   = float(rv[pk_idx])
        pk_lag = int(lag_range_bins[pk_idx]) * BIN_DAYS
        n_bins = len(cr_c)
        log.info("  Cycle %d (%s–%s): N=%d  r(+15d)=%.4f  peak=%.4f@%dd",
                 cycle_num, t_start, t_end, n_bins, r15, pk_r, pk_lag)

        lags_d = lag_range_bins * BIN_DAYS
        ax.plot(lags_d, rv, color="steelblue", lw=1.2)
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(15, color="grey", ls="--", lw=1, label="+15 d")
        ax.axvline(pk_lag, color="red", ls=":", lw=1,
                   label=f"Peak τ={pk_lag}d  r={pk_r:.3f}")
        ax.set_title(f"Solar Cycle {cycle_num} ({t_start[:4]}–{t_end[:4]})\n"
                     f"N={n_bins} bins   r(+15d)={r15:.3f}")
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("Pearson r")
        ax.legend(fontsize=7)

        results[f"cycle_{cycle_num}"] = dict(
            start=t_start, end=t_end,
            n_bins=n_bins,
            r_at_15d=r15,
            peak_r=pk_r,
            peak_lag_days=pk_lag,
        )

    fig.suptitle("Cross-correlation per solar cycle (1976–2019)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "solar_cycle_xcorr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("3g figure saved.")

    r15_vals = [v["r_at_15d"] for v in results.values() if v is not None]
    sign_consistent = all(r > 0 for r in r15_vals) or all(r < 0 for r in r15_vals)
    return dict(
        per_cycle=results,
        r15d_range=[float(min(r15_vals)), float(max(r15_vals))],
        sign_consistent_across_cycles=sign_consistent,
        interpretation=(
            "r(+15d) consistent in sign across all solar cycles"
            if sign_consistent else
            "r(+15d) sign INCONSISTENT across solar cycles — confirms solar-cycle artefact"
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Loading base CR and seismic data …")
    cr, _stn = _load_cr(STUDY_START, STUDY_END)
    events   = _load_seismic_events(STUDY_START, STUDY_END)
    sei      = _energy_metric(events, STUDY_START, STUDY_END, BIN_DAYS, cr.index)

    out = {}

    log.info("Running 3a …")
    out["3a_block_bootstrap"] = run_3a(cr, sei)

    log.info("Running 3b …")
    out["3b_partial_correlation"] = run_3b(cr, sei)

    log.info("Running 3c …")
    out["3c_coherence_mi"] = run_3c(cr, sei)

    log.info("Running 3d …")
    out["3d_missing_data"] = run_3d()

    log.info("Running 3e …")
    out["3e_bin_size"] = run_3e()

    log.info("Running 3f …")
    out["3f_declustering"] = run_3f(cr)

    log.info("Running 3g …")
    out["3g_solar_cycles"] = run_3g(cr, sei)

    # Save JSON
    out_path = OUT_DIR / "additional_robustness.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    log.info("Results saved: %s", out_path)

    # ── Summary printout ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ADDITIONAL ROBUSTNESS SUMMARY")
    print("=" * 70)
    r = out["3a_block_bootstrap"]
    print(f"  3a  Block bootstrap  p(+15d)={r['p_block_15d']:.4f}  "
          f"p(peak)={r['p_block_peak']:.4f}")
    r = out["3b_partial_correlation"]
    print(f"  3b  Partial corr     r_raw(+15d)={r['r_raw_15d']:.4f}  "
          f"r_partial={r['r_partial_15d']:.4f}")
    r = out["3c_coherence_mi"]
    print(f"  3c  Coherence SC-band={r['coherence_solar_cycle_band']:.4f}  "
          f"MI p(lag=0)={r['p_mi_lag0']:.3f}  p(+15d)={r['p_mi_lag15d']:.3f}")
    for res in out["3d_missing_data"]["station_threshold_sensitivity"]:
        print(f"  3d  min_stn={res['min_stations']}  "
              f"NaN={100*res['nan_fraction']:.1f}%  r(+15d)={res['r_at_15d']:.4f}")
    for res in out["3e_bin_size"]["bin_size_sensitivity"]:
        print(f"  3e  {res['bin_days']:2d}-d bins  "
              f"r(+{res['claimed_lag_days']}d)={res['r_at_claimed_lag']:.4f}  "
              f"peak={res['peak_r']:.4f}@{res['peak_lag_days']}d")
    r = out["3f_declustering"]
    print(f"  3f  Declustering     removed={100*r['fraction_removed']:.0f}%  "
          f"r_full={r['r_15d_full']:.4f}  r_decl={r['r_15d_declustered']:.4f}")
    r = out["3g_solar_cycles"]
    print(f"  3g  Solar cycles     r(+15d) range={r['r15d_range']}  "
          f"sign_consistent={r['sign_consistent_across_cycles']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
