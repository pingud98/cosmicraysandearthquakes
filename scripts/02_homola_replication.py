#!/usr/bin/env python3
"""
scripts/02_homola_replication.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reproduce the Homola et al. 2023 (JASTP 247, 106068) cross-correlation analysis
as literally as possible from the paper.

Method (from the paper)
-----------------------
* Data:  NMDB pressure-corrected hourly neutron counts; USGS earthquake catalogue.
* CR index: Each station is normalised to its own long-term mean count rate
  (so all stations become dimensionless relative-rate series around 1.0).
  The global index is the mean across all stations that have valid data at
  each 5-day bin.
* Seismic metric: sum of Mw for all M ≥ 4.0 events per 5-day bin (Homola
  metric; zero for bins with no events).
* Binning: non-overlapping 5-day bins anchored at study_start.
* Correlation: Pearson r(τ) for τ ∈ [-1000, +1000] days  (step = 5 days).
  Convention: τ > 0 means CR leads seismic (at lag τ the seismic series is
  advanced τ days into the future relative to the CR series).
* Significance: "naive" Pearson t-test p-value (treats bins as i.i.d. — this
  is the inflated significance that Homola reports; reproduced here as the
  baseline before phase-randomised correction).

Outputs
-------
results/figs/homola_replication.png   — r(τ) plot with peak annotated
results/homola_replication.json       — machine-readable summary
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crq.ingest.nmdb import load_station, resample_daily
from crq.ingest.usgs import load_usgs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("crq.homola")

# ---------------------------------------------------------------------------
# Constants (matching Homola et al. 2023)
# ---------------------------------------------------------------------------
STUDY_START   = "1976-01-01"
STUDY_END     = "2019-12-31"
BIN_DAYS      = 5
LAG_MIN_DAYS  = -1000
LAG_MAX_DAYS  = +1000
MIN_MAG       = 4.0
SEED          = 42

NMDB_COVERAGE_THRESHOLD = 0.60   # min valid hourly fraction per station-day
MIN_STATIONS_PER_BIN    = 3      # global index → NaN if fewer stations valid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _load_station_names(config_path: Path) -> list[str]:
    with config_path.open() as fh:
        return list(yaml.safe_load(fh)["stations"].keys())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--study-start",  default=STUDY_START)
    p.add_argument("--study-end",    default=STUDY_END)
    p.add_argument("--bin-days",     type=int,   default=BIN_DAYS)
    p.add_argument("--lag-min",      type=int,   default=LAG_MIN_DAYS)
    p.add_argument("--lag-max",      type=int,   default=LAG_MAX_DAYS)
    p.add_argument("--min-mag",      type=float, default=MIN_MAG)
    p.add_argument("--min-stations", type=int,   default=MIN_STATIONS_PER_BIN)
    p.add_argument("--nmdb-dir",     type=Path,  default=PROJECT_ROOT / "data" / "raw" / "nmdb")
    p.add_argument("--usgs-dir",     type=Path,  default=PROJECT_ROOT / "data" / "raw" / "usgs")
    p.add_argument("--output-dir",   type=Path,  default=PROJECT_ROOT / "results")
    p.add_argument("--config",       type=Path,  default=PROJECT_ROOT / "config" / "stations.yaml")
    p.add_argument("--seed",         type=int,   default=SEED)
    return p.parse_args()


def _bin_series(
    series: pd.Series,
    study_start: str,
    bin_days: int,
    agg: str = "mean",
) -> pd.Series:
    """
    Resample a daily Series to *bin_days*-day bins anchored exactly at
    *study_start*.

    Uses manual floor-division binning so the anchor date is always respected,
    regardless of pandas version or calendar effects.

    Parameters
    ----------
    agg : ``"mean"`` (default) or ``"sum"``
        Aggregation function.  Use ``"mean"`` for normalised CR rates;
        use ``"sum"`` for the seismic Mw total.
    """
    t0 = pd.Timestamp(study_start)
    days_from_origin = (series.index - t0).days
    bin_number = days_from_origin // bin_days
    bin_dates  = t0 + pd.to_timedelta(bin_number * bin_days, unit="D")
    grouped = series.groupby(bin_dates)
    return grouped.sum() if agg == "sum" else grouped.mean()


# ---------------------------------------------------------------------------
# Step 1 — Global cosmic-ray index
# ---------------------------------------------------------------------------

def build_cr_index(
    station_names:     list[str],
    study_start:       str,
    study_end:         str,
    bin_days:          int,
    nmdb_dir:          Path,
    *,
    coverage_threshold: float = NMDB_COVERAGE_THRESHOLD,
    min_stations:      int   = MIN_STATIONS_PER_BIN,
) -> tuple[pd.Series, pd.DataFrame, int]:
    """
    Load all available NMDB stations, normalise each to its mean count rate
    over the study window, and return a *bin_days*-day-binned global index.

    Returns
    -------
    cr_index          : binned global CR series (NaN if < min_stations valid)
    station_norm_daily: daily normalised series per station (for diagnostics)
    effective_min     : the min_stations threshold actually applied
    """
    t0 = pd.Timestamp(study_start)
    t1 = pd.Timestamp(study_end)
    start_year, end_year = t0.year, t1.year

    norm_cols: dict[str, pd.Series] = {}

    for station in station_names:
        hourly = load_station(station, start_year, end_year, nmdb_dir)
        if hourly.empty:
            continue

        daily_df = resample_daily(hourly, station, coverage_threshold=coverage_threshold)
        daily    = daily_df[station].loc[study_start:study_end]

        n_valid = int(daily.notna().sum())
        if n_valid < 30:
            continue

        station_mean = daily.mean()
        if not np.isfinite(station_mean) or station_mean <= 0:
            logger.warning("station %s: non-positive mean — skipping", station)
            continue

        norm_cols[station] = daily / station_mean   # dimensionless ~1.0
        n_total    = (t1 - t0).days + 1
        coverage   = 100 * n_valid / n_total
        logger.info(
            "station %-6s  valid_days=%5d  coverage=%5.1f%%  mean=%.1f",
            station, n_valid, coverage, station_mean,
        )

    n_loaded = len(norm_cols)
    logger.info("Loaded %d stations with valid data.", n_loaded)

    if n_loaded == 0:
        raise RuntimeError(
            "No valid station data found — run scripts/01_download_data.py first."
        )

    # Auto-adjust min_stations when data is sparse (e.g. subset test)
    effective_min = min_stations
    if n_loaded < min_stations:
        effective_min = max(1, n_loaded - 1) if n_loaded > 1 else 1
        logger.warning(
            "Only %d stations loaded but min_stations=%d requested; "
            "using effective_min=%d.  Results with <%d stations are unreliable.",
            n_loaded, min_stations, effective_min, min_stations,
        )

    station_norm_daily = pd.DataFrame(norm_cols)

    # Global daily index
    n_valid_per_day = station_norm_daily.notna().sum(axis=1)
    global_daily    = station_norm_daily.mean(axis=1)            # NaN-safe
    global_daily[n_valid_per_day < effective_min] = np.nan

    # Bin to bin_days
    cr_index = _bin_series(global_daily, study_start, bin_days)
    cr_index.name = "cr_index"

    n_valid_bins = int(cr_index.notna().sum())
    logger.info(
        "CR index: %d %d-day bins, %d non-NaN (%.1f%%)",
        len(cr_index), bin_days, n_valid_bins,
        100 * n_valid_bins / max(len(cr_index), 1),
    )
    return cr_index, station_norm_daily, effective_min


# ---------------------------------------------------------------------------
# Step 2 — Seismic metric
# ---------------------------------------------------------------------------

def build_seismic_metric(
    study_start: str,
    study_end:   str,
    bin_days:    int,
    usgs_dir:    Path,
    *,
    min_mag:  float = MIN_MAG,
    cr_index: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Sum of Mw for all M ≥ *min_mag* events per *bin_days*-day bin.

    Aligned to *cr_index*'s index if supplied; otherwise built from scratch.
    Bins with no events → 0.0.
    """
    t0 = pd.Timestamp(study_start)
    t1 = pd.Timestamp(study_end)

    events = load_usgs(t0.year, t1.year, usgs_dir)
    if events.empty:
        raise RuntimeError("No USGS data — run scripts/01_download_data.py first.")

    events = events.loc[study_start:study_end]
    events = events[events["mag"] >= min_mag]
    logger.info(
        "Seismic: %d events M≥%.1f in %s–%s",
        len(events), min_mag, study_start, study_end,
    )

    # Daily sum-of-Mw then bin to 5-day totals (sum, not mean)
    daily_mw = events["mag"].resample("1D").sum()
    daily_mw = daily_mw.reindex(
        pd.date_range(study_start, study_end, freq="D"), fill_value=0.0
    )
    seismic = _bin_series(daily_mw, study_start, bin_days, agg="sum").fillna(0.0)
    seismic.name = "seismic_mw_sum"

    # Align to CR index bins
    if cr_index is not None:
        seismic = seismic.reindex(cr_index.index, fill_value=0.0)

    nonzero_pct = 100 * (seismic > 0).mean()
    logger.info(
        "Seismic metric: %d bins, %.1f%% non-zero", len(seismic), nonzero_pct
    )
    return seismic


# ---------------------------------------------------------------------------
# Step 3 — Cross-correlation
# ---------------------------------------------------------------------------

def pearson_lag_correlation(
    x: pd.Series,
    y: pd.Series,
    lag_bins: np.ndarray,
) -> pd.DataFrame:
    """
    Pearson r(τ) for each lag in *lag_bins* (bin units).

    Convention: τ > 0 means x leads y.
    At lag τ > 0:  x[0..N-τ]  vs  y[τ..N]
    At lag τ < 0:  x[|τ|..N]  vs  y[0..N-|τ|]

    Returns DataFrame: lag_bins, r, p_value, n_pairs.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    N = len(x_arr)

    rs, ps, ns = [], [], []
    for lag in lag_bins:
        if lag >= 0:
            end = N - lag
            if end <= 0:
                rs.append(np.nan); ps.append(np.nan); ns.append(0)
                continue
            xa = x_arr[:end]
            ya = y_arr[lag : lag + end]
        else:
            abslag = -lag
            end = N - abslag
            if end <= 0:
                rs.append(np.nan); ps.append(np.nan); ns.append(0)
                continue
            xa = x_arr[abslag : abslag + end]
            ya = y_arr[:end]

        valid = np.isfinite(xa) & np.isfinite(ya)
        n = int(valid.sum())
        ns.append(n)
        if n < 10:
            rs.append(np.nan); ps.append(np.nan)
            continue

        r, p = scipy.stats.pearsonr(xa[valid], ya[valid])
        rs.append(float(r)); ps.append(float(p))

    return pd.DataFrame({"lag_bins": lag_bins, "r": rs, "p_value": ps, "n_pairs": ns})


def naive_sigma(p_value: float) -> float:
    """Two-tailed p → Gaussian sigma  z = Φ⁻¹(1 − p/2)."""
    if not np.isfinite(p_value) or p_value <= 0:
        return np.inf
    return float(scipy.stats.norm.isf(p_value / 2.0))


# ---------------------------------------------------------------------------
# Step 4 — Figure
# ---------------------------------------------------------------------------

def make_figure(
    lag_df:         pd.DataFrame,
    bin_days:       int,
    peak_lag_days:  float,
    peak_r:         float,
    n_stations:     int,
    study_start:    str,
    study_end:      str,
    naive_p:        float,
    sigma:          float,
    cr_5d:          pd.Series,
    seismic_5d:     pd.Series,
    n_stations_eff: int,
    r_at_15:        float,
    sigma_at_15:    float,
) -> plt.Figure:
    lags_days = lag_df["lag_bins"].values * bin_days
    r_vals    = lag_df["r"].values

    n0 = int(lag_df.loc[lag_df["lag_bins"] == 0, "n_pairs"].iloc[0])
    naive_ci95 = 1.96 / np.sqrt(n0) if n0 > 4 else 0.0

    # Naive r thresholds for σ=3 and σ=6 (two-tailed, N=n0)
    def _r_for_sigma(s: float, n: int) -> float:
        """r value corresponding to naive σ=s for sample size n."""
        p = 2.0 * scipy.stats.norm.sf(s)
        t_crit = scipy.stats.t.isf(p / 2.0, df=n - 2)
        return float(t_crit / np.sqrt(t_crit**2 + n - 2))

    r_sigma3 = _r_for_sigma(3.0, n0) if n0 > 4 else np.nan
    r_sigma6 = _r_for_sigma(6.0, n0) if n0 > 4 else np.nan

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 12),
        gridspec_kw={"height_ratios": [3, 1.4, 1.4]},
    )

    # ── Panel 1: r(τ) ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.axhline(0,           color="k",   lw=0.8)
    ax.axhline( naive_ci95, color="C0",  lw=0.8, ls="--",
                label=f"±{naive_ci95:.3f} naive 95 % CI  (n={n0:,})")
    ax.axhline(-naive_ci95, color="C0",  lw=0.8, ls="--")

    if np.isfinite(r_sigma3):
        ax.axhline( r_sigma3, color="C2", lw=0.9, ls=":", alpha=0.7,
                    label=f"Naive 3σ  (r={r_sigma3:.3f})")
        ax.axhline(-r_sigma3, color="C2", lw=0.9, ls=":", alpha=0.7)
    if np.isfinite(r_sigma6):
        ax.axhline( r_sigma6, color="C1", lw=0.9, ls="-.", alpha=0.8,
                    label=f"Naive 6σ  (r={r_sigma6:.3f})  ← Homola claim")
        ax.axhline(-r_sigma6, color="C1", lw=0.9, ls="-.", alpha=0.8)

    ax.axvline(0,  color="k", lw=0.6, alpha=0.35)

    ax.plot(lags_days, r_vals, color="k", lw=1.1, zorder=3)

    # Mark the τ=+15d bin (Homola claimed signal)
    lag15_bins = 15 // bin_days
    row15 = lag_df[lag_df["lag_bins"] == lag15_bins]
    if len(row15):
        r15 = float(row15["r"].iloc[0])
        if np.isfinite(r15):
            s15_str = f"{sigma_at_15:.1f}σ" if np.isfinite(sigma_at_15) and sigma_at_15 < 100 else ">8σ"
            ax.scatter([15], [r15], color="darkorange", s=60, zorder=6,
                       label=f"τ = +15 d: r = {r15:+.4f} ({s15_str} naive)")
            ax.axvline(15, color="darkorange", lw=1.0, ls=":", alpha=0.6)

    # Mark the overall peak
    ax.axvline(peak_lag_days, color="crimson", lw=1.3, ls="--",
               label=f"Peak τ = {peak_lag_days:+.0f} d,  r = {peak_r:+.4f}")
    ax.scatter([peak_lag_days], [peak_r], color="crimson", s=50, zorder=5)

    r_clean = r_vals[np.isfinite(r_vals)]
    ylo = min(-0.15, r_clean.min() * 1.15) if len(r_clean) else -0.15
    yhi = max( 0.15, r_clean.max() * 1.15) if len(r_clean) else  0.15
    ax.set_xlim(lags_days[0], lags_days[-1])
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("Lag τ (days)  [τ > 0: CR leads seismic]")
    ax.set_ylabel("Pearson r")
    ax.set_title(
        f"Homola et al. 2023 — replication baseline  |  "
        f"{n_stations} NMDB stations  |  "
        f"{study_start[:4]}–{study_end[:4]}  |  "
        f"5-day bins  |  M≥{MIN_MAG:.1f}",
        fontsize=10,
    )
    ax.legend(fontsize=8.5, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.22)

    sig_str = f"{sigma:.1f}σ" if np.isfinite(sigma) and sigma < 100 else ">8σ"
    info = (
        f"Peak: r = {peak_r:+.5f}  at  τ = {peak_lag_days:+.0f} d  ({sig_str} naive)\n"
        f"τ = +15 d: r = {r_at_15:+.5f}  "
        + (f"({sigma_at_15:.1f}σ naive)" if np.isfinite(sigma_at_15) and sigma_at_15 < 100 else "")
        + f"\nn = {n0:,} valid bin-pairs at τ=0\n"
        f"⚠ Naive — ignores autocorr., solar cycle, lag-scan penalty"
    )
    ax.text(
        0.014, 0.975, info, transform=ax.transAxes,
        va="top", ha="left", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="lightyellow", alpha=0.92),
    )

    # ── Panel 2: CR index ──────────────────────────────────────────────────
    ax2 = axes[1]
    cr_dev = (cr_5d - 1.0) * 100   # % deviation from station-normalised mean
    ax2.plot(cr_dev.index, cr_dev.values, lw=0.7, color="navy", alpha=0.85)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylabel("CR deviation (%)")
    ax2.set_title(
        f"Global CR index  ({n_stations} stations, each normalised to their long-term mean)",
        fontsize=9,
    )
    ax2.grid(True, alpha=0.18)
    ax2.set_xlim(cr_5d.index[0], cr_5d.index[-1])

    # ── Panel 3: seismic metric ────────────────────────────────────────────
    ax3 = axes[2]
    ax3.fill_between(seismic_5d.index, seismic_5d.values, color="firebrick", alpha=0.6, lw=0)
    ax3.set_ylabel(f"Σ Mw  (M≥{MIN_MAG:.1f})")
    ax3.set_title(f"Global seismic metric  (sum of all Mw in each 5-day bin)", fontsize=9)
    ax3.grid(True, alpha=0.18)
    ax3.set_xlim(seismic_5d.index[0], seismic_5d.index[-1])

    fig.tight_layout(pad=1.2)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    np.random.seed(args.seed)
    git_sha = _get_git_sha()

    logger.info("Homola replication  |  git=%s  seed=%d", git_sha, args.seed)
    logger.info("Study window: %s – %s", args.study_start, args.study_end)
    logger.info("Bin: %d days  |  Lags: %+d … %+d days",
                args.bin_days, args.lag_min, args.lag_max)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = args.output_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. CR index
    station_names = _load_station_names(args.config)
    cr_5d, station_norm_daily, effective_min = build_cr_index(
        station_names,
        args.study_start, args.study_end,
        args.bin_days, args.nmdb_dir,
        coverage_threshold=NMDB_COVERAGE_THRESHOLD,
        min_stations=args.min_stations,
    )
    n_stations = station_norm_daily.shape[1]

    # ── 2. Seismic metric
    seismic_5d = build_seismic_metric(
        args.study_start, args.study_end,
        args.bin_days, args.usgs_dir,
        min_mag=args.min_mag,
        cr_index=cr_5d,
    )

    # ── 3. Data quality assessment
    n_cr_bins   = int(cr_5d.notna().sum())
    study_days  = (pd.Timestamp(args.study_end) - pd.Timestamp(args.study_start)).days
    expect_bins = study_days // args.bin_days
    coverage_pct = 100.0 * n_cr_bins / max(expect_bins, 1)

    if coverage_pct < 50:
        logger.warning(
            "⚠  CR data coverage = %.1f%% of study window.  "
            "Results are not meaningful with less than half the data.  "
            "Run: python scripts/01_download_data.py  (full run, no --subset).",
            coverage_pct,
        )

    # ── 4. Cross-correlation
    lag_bins = np.arange(
        args.lag_min // args.bin_days,
        args.lag_max // args.bin_days + 1,
        dtype=int,
    )
    logger.info("Computing Pearson r(τ) for %d lags …", len(lag_bins))
    lag_df = pearson_lag_correlation(cr_5d, seismic_5d, lag_bins)

    # ── 5. Peak statistics
    valid_mask = lag_df["r"].notna()
    if not valid_mask.any():
        raise RuntimeError(
            "All correlation values are NaN — data is insufficient for analysis."
        )

    peak_idx      = lag_df.loc[valid_mask, "r"].abs().idxmax()
    peak_row      = lag_df.loc[peak_idx]
    peak_lag_bins = int(peak_row["lag_bins"])
    peak_lag_days = peak_lag_bins * args.bin_days
    peak_r        = float(peak_row["r"])
    naive_p       = float(peak_row["p_value"])
    n_at_peak     = int(peak_row["n_pairs"])
    sigma         = naive_sigma(naive_p)

    # Correlation specifically at the Homola ±15-day claim
    lag15_bins   = 15 // args.bin_days   # = 3 bins
    row15        = lag_df[lag_df["lag_bins"] == lag15_bins]
    r_at_15      = float(row15["r"].iloc[0]) if len(row15) else np.nan
    p_at_15      = float(row15["p_value"].iloc[0]) if len(row15) else np.nan
    sigma_at_15  = naive_sigma(p_at_15) if np.isfinite(p_at_15) else np.nan

    logger.info("─" * 55)
    logger.info("Peak r = %+.5f  at  τ = %+d days (%+d bins)",
                peak_r, peak_lag_days, peak_lag_bins)
    logger.info("Naive p = %.3e  (%.1fσ)", naive_p,
                sigma if np.isfinite(sigma) else 999)
    logger.info("n pairs at peak = %d", n_at_peak)
    logger.info("r at τ = +15 d   = %+.5f  (naive p = %.2e, %.1fσ)",
                r_at_15, p_at_15 if np.isfinite(p_at_15) else 1,
                sigma_at_15 if np.isfinite(sigma_at_15) else 0)
    logger.info("─" * 55)
    logger.info(
        "NOTE: naive significance ignores autocorrelation, the shared solar-"
        "cycle trend, and scanning over %d lags — expect gross over-significance.",
        len(lag_bins),
    )

    # ── 6. Figure
    fig = make_figure(
        lag_df, args.bin_days, peak_lag_days, peak_r,
        n_stations, args.study_start, args.study_end,
        naive_p, sigma, cr_5d, seismic_5d,
        n_stations_eff=effective_min,
        r_at_15=r_at_15,
        sigma_at_15=sigma_at_15,
    )
    fig_path = figs_dir / "homola_replication.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", fig_path)

    # ── 7. JSON log
    results = {
        "script":               "scripts/02_homola_replication.py",
        "git_sha":              git_sha,
        "seed":                 args.seed,
        "timestamp_utc":        datetime.now(timezone.utc).isoformat(),
        "study_window":         [args.study_start, args.study_end],
        "bin_days":             args.bin_days,
        "lag_range_days":       [args.lag_min, args.lag_max],
        "min_magnitude":        args.min_mag,
        "nmdb_coverage_threshold": NMDB_COVERAGE_THRESHOLD,
        "min_stations_requested": args.min_stations,
        "min_stations_effective": effective_min,
        "n_stations_used":      n_stations,
        "n_cr_bins_total":      int(len(cr_5d)),
        "n_cr_bins_valid":      n_cr_bins,
        "cr_data_coverage_pct": round(coverage_pct, 2),
        "peak_lag_days":        peak_lag_days,
        "peak_lag_bins":        peak_lag_bins,
        "peak_r":               round(peak_r, 6),
        "naive_p_value":        float(f"{naive_p:.6e}"),
        "naive_sigma":          round(sigma, 3) if np.isfinite(sigma) and sigma < 1e6 else None,
        "n_pairs_at_peak":      n_at_peak,
        "r_at_tau_15d":         round(r_at_15,  6) if np.isfinite(r_at_15)      else None,
        "p_at_tau_15d":         float(f"{p_at_15:.6e}") if np.isfinite(p_at_15) else None,
        "sigma_at_tau_15d":     round(sigma_at_15, 3) if np.isfinite(sigma_at_15) else None,
        "n_lags_scanned":       len(lag_bins),
        "note_significance": (
            "Naive p from Pearson t-test treating 5-day bins as i.i.d. "
            "Not corrected for: (1) temporal autocorrelation, "
            "(2) shared solar-cycle trend, (3) scan over multiple lags. "
            "Expected to be grossly over-significant — corrected tests in phase 2."
        ),
    }
    json_path = args.output_dir / "homola_replication.json"
    json_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved: %s", json_path)

    # ── Human summary
    print()
    print("=" * 62)
    print("  HOMOLA ET AL. 2023 — REPLICATION SUMMARY")
    print("=" * 62)
    print(f"  Stations used           : {n_stations}")
    print(f"  CR data coverage        : {coverage_pct:.1f}% of study window")
    print(f"  Valid 5-day bins (CR)   : {n_cr_bins:,}")
    print()
    print(f"  Peak |r|                : {abs(peak_r):.5f}  (r = {peak_r:+.5f})")
    print(f"  Peak lag τ              : {peak_lag_days:+d} days")
    print(f"  Naive p-value at peak   : {naive_p:.3e}")
    sstr = f"{sigma:.2f}σ" if np.isfinite(sigma) and sigma < 1e6 else ">8σ"
    print(f"  Naive sigma at peak     : {sstr}")
    print()
    s15str = f"{sigma_at_15:.2f}σ" if np.isfinite(sigma_at_15) else "N/A"
    print(f"  r at τ = +15 d (claim)  : {r_at_15:+.5f}  ({s15str})")
    print()
    if coverage_pct < 50:
        print("  ⚠  WARNING: data coverage < 50% — results not meaningful.")
        print("             python scripts/01_download_data.py  (full run)")
    else:
        print("  ✓  Data coverage sufficient.")
    print("=" * 62)


if __name__ == "__main__":
    main()
