#!/usr/bin/env python3
"""
09_raw_pairwise_correlations.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Raw pairwise correlations between galactic cosmic-ray flux (CR), global
seismicity, and solar activity (sunspot number) across three time windows:

  in-sample  : 1976-01-01 to 2019-12-31
  OOS        : 2020-01-01 to 2025-04-29
  combined   : 1976-01-01 to 2025-04-29

No HP filtering or detrending is applied.  Missing OOS data (NMDB 2020-2025,
USGS 2020-2025) are downloaded automatically.

CR is represented by its per-bin distribution across NMDB stations (p5, p50,
p95, min, max).  Seismic energy uses the physically correct E ∝ 10^(1.5·Mw)
sum.  Two CR variants are correlated: the station-median (p50) and station-p95.

Outputs
-------
results/raw_pairwise_correlations.json
results/figs/raw_corr_insample.png
results/figs/raw_corr_oos.png
results/figs/raw_corr_combined.png
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
NMDB_DIR = ROOT / "data" / "raw" / "nmdb"
USGS_DIR = ROOT / "data" / "raw" / "usgs"
SIDC_DIR = ROOT / "data" / "raw" / "sidc"
AVAIL_FILE = ROOT / "results" / "data_availability.json"
OUT_DIR = ROOT / "results"
FIG_DIR = ROOT / "results" / "figs"

# Add src/ to path
sys.path.insert(0, str(ROOT / "src"))
from crq.ingest.nmdb import download_station_year, load_station, resample_daily
from crq.ingest.usgs import download_year as download_usgs_year, load_usgs

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BIN_DAYS = 5
EPOCH = pd.Timestamp("1976-01-01")
IN_SAMPLE_START = "1976-01-01"
IN_SAMPLE_END = "2019-12-31"
OOS_START = "2020-01-01"
OOS_END = "2025-04-29"
COMBINED_START = "1976-01-01"
COMBINED_END = "2025-04-29"
MIN_MAG = 4.5
MIN_STATIONS = 3
COVERAGE_THRESHOLD = 0.60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper: 5-day bin index
# ---------------------------------------------------------------------------

def _bin_index(ts: pd.Timestamp) -> int:
    return (ts - EPOCH).days // BIN_DAYS


def _bin_start(b: int) -> pd.Timestamp:
    return EPOCH + pd.Timedelta(days=b * BIN_DAYS)


def _decimal_year(ts: pd.DatetimeIndex) -> np.ndarray:
    """Convert DatetimeIndex to decimal year (for scatter colouring)."""
    yr = ts.year.values.astype(float)
    day_of_yr = ts.day_of_year.values.astype(float)
    days_in_yr = np.where(
        pd.DatetimeIndex(ts).is_leap_year, 366.0, 365.0
    )
    return yr + (day_of_yr - 1.0) / days_in_yr


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _ensure_usgs(years: range) -> None:
    """Download any missing USGS yearly files."""
    for yr in years:
        dest = USGS_DIR / f"usgs-{yr}.csv"
        if dest.exists() and dest.stat().st_size > 100:
            continue
        log.info("Downloading USGS %d …", yr)
        try:
            download_usgs_year(yr, USGS_DIR, min_magnitude=MIN_MAG)
        except Exception as exc:
            log.warning("USGS %d download failed: %s", yr, exc)


def _oos_stations() -> list[str]:
    """Return OOS-good station list from data_availability.json."""
    if AVAIL_FILE.exists():
        with open(AVAIL_FILE) as fh:
            data = json.load(fh)
        return data.get("good_stations_oos", [])
    # Fall back to all stations present in config
    import yaml
    cfg = yaml.safe_load((ROOT / "config" / "stations.yaml").read_text())
    return list(cfg["stations"].keys())


def _ensure_nmdb_oos(stations: list[str], years: range) -> None:
    """Download any missing NMDB station-year files for OOS window."""
    total = len(stations) * len(years)
    done = 0
    for stn in stations:
        for yr in years:
            dest = NMDB_DIR / f"{stn}{yr}.csv"
            if dest.exists() and dest.stat().st_size > 0:
                done += 1
                continue
            log.info("[%d/%d] Downloading NMDB %s %d …", done + 1, total, stn, yr)
            try:
                download_station_year(stn, yr, NMDB_DIR, sleep_s=0.5)
            except Exception as exc:
                log.warning("NMDB %s %d download failed: %s", stn, yr, exc)
            done += 1


# ---------------------------------------------------------------------------
# Load NMDB: per-bin station distribution
# ---------------------------------------------------------------------------

def _load_nmdb_bins(
    start: str,
    end: str,
    coverage_thr: float = COVERAGE_THRESHOLD,
) -> pd.DataFrame:
    """
    Load all NMDB stations for [start, end] and return a DataFrame of
    per-5d-bin statistics across stations:
      cr_p05, cr_p25, cr_p50, cr_p75, cr_p95, cr_min, cr_max, cr_n
    Each station is normalised by its long-run mean before aggregation.
    """
    t0 = pd.Timestamp(start)
    t1 = pd.Timestamp(end)
    start_yr = t0.year
    end_yr = t1.year

    # Determine which stations have files in this window
    station_files: dict[str, list[Path]] = {}
    for p in sorted(NMDB_DIR.glob("*.csv")):
        stem = p.stem  # e.g. AATA2018
        stn = "".join(c for c in stem if not c.isdigit())
        yr_str = "".join(c for c in stem if c.isdigit())
        if not yr_str:
            continue
        yr = int(yr_str)
        if yr < start_yr or yr > end_yr:
            continue
        station_files.setdefault(stn, []).append(p)

    if not station_files:
        log.warning("No NMDB files found for %s–%s", start, end)
        return pd.DataFrame()

    # Build bin grid
    b0 = _bin_index(t0)
    b1 = _bin_index(t1)
    bin_idx = np.arange(b0, b1 + 1)
    bin_starts = pd.DatetimeIndex([_bin_start(b) for b in bin_idx])

    station_means: dict[str, pd.Series] = {}  # station -> per-bin mean (normalised)

    for stn, _ in station_files.items():
        try:
            hourly = load_station(stn, start_yr, end_yr, NMDB_DIR)
        except Exception as exc:
            log.warning("load_station %s failed: %s", stn, exc)
            continue
        if hourly.empty or stn not in hourly.columns:
            continue

        daily = resample_daily(hourly, stn, coverage_threshold=coverage_thr)
        daily_vals = daily[stn].loc[start:end]
        if daily_vals.isna().all():
            continue

        # Normalise by station long-run mean (ignore NaN)
        station_mean = daily_vals.mean(skipna=True)
        if np.isnan(station_mean) or station_mean == 0:
            continue
        daily_norm = daily_vals / station_mean

        # Resample to 5-day bins (mean within bin)
        # Use bin index as grouper
        day_index = daily_norm.index
        bin_of_day = ((day_index - EPOCH).days // BIN_DAYS)
        grp = daily_norm.groupby(bin_of_day)
        bin_mean = grp.mean()  # index = bin integer
        bin_mean.index = [_bin_start(b) for b in bin_mean.index]
        bin_mean = bin_mean.reindex(bin_starts)
        station_means[stn] = bin_mean

    if not station_means:
        log.warning("No valid stations for %s–%s", start, end)
        return pd.DataFrame()

    # Stack into (bins × stations) matrix
    mat = pd.DataFrame(station_means).reindex(bin_starts)

    # Per-bin statistics across stations (ignore NaN)
    n_valid = mat.notna().sum(axis=1)
    mask = n_valid >= MIN_STATIONS  # require at least MIN_STATIONS stations

    result = pd.DataFrame(index=bin_starts)
    result["cr_p05"] = mat.quantile(0.05, axis=1)
    result["cr_p25"] = mat.quantile(0.25, axis=1)
    result["cr_p50"] = mat.quantile(0.50, axis=1)
    result["cr_p75"] = mat.quantile(0.75, axis=1)
    result["cr_p95"] = mat.quantile(0.95, axis=1)
    result["cr_min"] = mat.min(axis=1)
    result["cr_max"] = mat.max(axis=1)
    result["cr_n"] = n_valid.values

    # Mask bins with fewer than MIN_STATIONS stations
    for col in result.columns:
        if col != "cr_n":
            result.loc[~mask, col] = np.nan

    log.info("NMDB %s–%s: %d bins, %d stations, %.1f%% valid bins",
             start, end, len(result), len(station_means),
             100.0 * mask.sum() / len(result))
    return result


# ---------------------------------------------------------------------------
# Load USGS: per-bin seismic energy E ∝ 10^(1.5·Mw)
# ---------------------------------------------------------------------------

def _load_seismic_energy(start: str, end: str) -> pd.Series:
    """
    Load USGS events for [start, end] and compute per-5d-bin summed seismic
    energy E = Σ 10^(1.5 · Mw).  Returns log10(E); bins with no events → NaN.
    """
    t0 = pd.Timestamp(start)
    t1 = pd.Timestamp(end)
    events = load_usgs(t0.year, t1.year, USGS_DIR)

    if events.empty or "mag" not in events.columns:
        log.warning("No USGS events loaded for %s–%s", start, end)
        return pd.Series(dtype=float)

    events = events.loc[start:end]
    events = events[events["mag"] >= MIN_MAG].copy()
    events["energy"] = np.power(10.0, 1.5 * events["mag"].values)

    # 5-day binning
    b0 = _bin_index(t0)
    b1 = _bin_index(t1)
    bin_idx = np.arange(b0, b1 + 1)
    bin_starts = pd.DatetimeIndex([_bin_start(b) for b in bin_idx])

    day_of_event = events.index.normalize()
    bin_of_event = ((day_of_event - EPOCH).days // BIN_DAYS)
    events["bin"] = bin_of_event.values

    bin_energy = events.groupby("bin")["energy"].sum()

    result = bin_energy.reindex(bin_idx)
    result.index = bin_starts
    log.info("Seismic %s–%s: %d events, %.1f%% bins non-zero",
             start, end, len(events),
             100.0 * result.notna().sum() / len(result))
    return np.log10(result)  # log10(E) for all operations


# ---------------------------------------------------------------------------
# Load SIDC sunspots: bin to 5-day, carry raw spread
# ---------------------------------------------------------------------------

def _load_sunspot_bins(start: str, end: str) -> pd.DataFrame:
    """
    Load KSO/SIDC daily sunspot CSV and return a per-5d-bin DataFrame with:
      sn_raw_mean, sn_raw_min, sn_raw_max, sn_smooth (365-day rolling mean)
    The smoothed series is computed on the daily series before binning.
    """
    path = SIDC_DIR / "sunspots.csv"
    if not path.exists():
        raise FileNotFoundError(f"Sunspot file not found: {path}")

    # KSO format: Date,Total,North,South,Diff  (comma-separated)
    # Standard SIDC format: Year;Month;Day;FracYear;SN;StdDev;Nobs;Definitive
    # Detect format by reading header
    header = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    if ";" in header and "Year" in header:
        # SIDC SILSO format
        df = pd.read_csv(
            path, sep=";",
            names=["Year", "Month", "Day", "FracYear", "SN", "StdDev", "Nobs", "Def"],
            header=0,
        )
        df["date"] = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=df["Day"]))
        df = df.set_index("date")[["SN"]].rename(columns={"SN": "sn"})
        df["sn"] = pd.to_numeric(df["sn"], errors="coerce")
    else:
        # KSO comma-separated format
        df = pd.read_csv(
            path, sep=",",
            names=["date", "total", "north", "south", "diff"],
            header=0,
        )
        df["date"] = pd.to_datetime(df["date"].str.strip(), errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.set_index("date")
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.rename(columns={"total": "sn"})[["sn"]]

    df = df.sort_index()
    df = df.loc[~df.index.duplicated(keep="first")]

    # 365-day rolling mean (smoothed solar cycle)
    df["sn_smooth"] = df["sn"].rolling(window=365, center=True, min_periods=180).mean()

    # Ensure DatetimeIndex
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    df = df.sort_index()

    # Clip to window
    df = df.loc[start:end]

    t0 = pd.Timestamp(start)
    t1 = pd.Timestamp(end)
    b0 = _bin_index(t0)
    b1 = _bin_index(t1)
    bin_idx = np.arange(b0, b1 + 1)
    bin_starts = pd.DatetimeIndex([_bin_start(b) for b in bin_idx])

    day_idx = pd.DatetimeIndex(df.index).normalize()
    bin_of_day = ((day_idx - EPOCH).days // BIN_DAYS)
    df = df.copy()
    df["bin"] = bin_of_day.values

    grp = df.groupby("bin")
    sn_mean = grp["sn"].mean().reindex(bin_idx)
    sn_min = grp["sn"].min().reindex(bin_idx)
    sn_max = grp["sn"].max().reindex(bin_idx)
    sn_smooth = grp["sn_smooth"].mean().reindex(bin_idx)
    result = pd.DataFrame({
        "sn_mean": sn_mean.values,
        "sn_min": sn_min.values,
        "sn_max": sn_max.values,
        "sn_smooth": sn_smooth.values,
    }, index=bin_starts)

    log.info("Sunspot %s–%s: %d bins, %d daily records",
             start, end, len(result), len(df))
    return result


# ---------------------------------------------------------------------------
# Correlation statistics
# ---------------------------------------------------------------------------

def _pearson_with_ci(x: np.ndarray, y: np.ndarray, alpha: float = 0.05):
    """
    Pearson r, p-value, and (1-alpha) CI via Fisher z-transform.
    Returns (r, p, ci_lo, ci_hi, n).
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 4:
        return np.nan, np.nan, np.nan, np.nan, n
    r, p = stats.pearsonr(x, y)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1.0 - alpha / 2)
    ci_lo = float(np.tanh(z - z_crit * se))
    ci_hi = float(np.tanh(z + z_crit * se))
    return float(r), float(p), ci_lo, ci_hi, n


def _spearman(x: np.ndarray, y: np.ndarray):
    """Spearman ρ and p-value. Returns (rho, p, n)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 4:
        return np.nan, np.nan, n
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p), n


def _correlate_pair(
    x: np.ndarray, y: np.ndarray, label: str, window: str, n_tests: int
) -> dict:
    """Compute Pearson + Spearman for one (x, y) pair with Bonferroni correction."""
    pr, pp, ci_lo, ci_hi, n = _pearson_with_ci(x, y)
    rho, sp, _ = _spearman(x, y)
    return {
        "label": label,
        "window": window,
        "n_bins": n,
        "pearson_r": pr,
        "pearson_p": pp,
        "pearson_ci_lo": ci_lo,
        "pearson_ci_hi": ci_hi,
        "pearson_p_bonf": min(1.0, pp * n_tests) if np.isfinite(pp) else np.nan,
        "spearman_rho": rho,
        "spearman_p": sp,
        "spearman_p_bonf": min(1.0, sp * n_tests) if np.isfinite(sp) else np.nan,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _lowess_line(x: np.ndarray, y: np.ndarray, frac: float = 0.4):
    """Return (x_sorted, y_smooth) for a LOWESS trend line."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.array([]), np.array([])
    xl, yl = x[mask], y[mask]
    order = np.argsort(xl)
    xl, yl = xl[order], yl[order]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm = lowess(yl, xl, frac=frac, return_sorted=True)
    return sm[:, 0], sm[:, 1]


def _scatter_panel(
    ax: plt.Axes,
    x_med: np.ndarray,
    x_lo: np.ndarray,
    x_hi: np.ndarray,
    x_xlo: np.ndarray,   # extreme low (min)
    x_xhi: np.ndarray,   # extreme high (max)
    y: np.ndarray,
    times: np.ndarray,   # decimal year for colour
    xlabel: str,
    ylabel: str,
    title: str,
    corr_text: str,
    cmap: str = "plasma",
    show_x_bands: bool = True,
) -> None:
    """
    Scatter plot with:
    - Points coloured by time (decimal year)
    - p5–p95 band as horizontal error bars (light)
    - min–max band as even lighter error bars
    - LOWESS trend line
    - Correlation annotation
    """
    mask = np.isfinite(x_med) & np.isfinite(y)
    xm, ym, tm = x_med[mask], y[mask], times[mask]

    if len(tm) == 0:
        ax.set_title(title + "\n(no data)", fontsize=9)
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")
        return

    norm = mcolors.Normalize(vmin=tm.min(), vmax=tm.max())
    sc = ax.scatter(xm, ym, c=tm, cmap=cmap, norm=norm, s=8, alpha=0.65, zorder=3)

    # Uncertainty bands on x (station percentile spread)
    if show_x_bands and x_lo is not None and x_hi is not None:
        xlo_m = x_lo[mask]
        xhi_m = x_hi[mask]
        xerr_lo = np.clip(xm - xlo_m, 0, None)
        xerr_hi = np.clip(xhi_m - xm, 0, None)
        ax.errorbar(
            xm, ym,
            xerr=[xerr_lo, xerr_hi],
            fmt="none", ecolor="steelblue", alpha=0.12, linewidth=0.5, zorder=2,
        )
    if show_x_bands and x_xlo is not None and x_xhi is not None:
        xxlo_m = x_xlo[mask]
        xxhi_m = x_xhi[mask]
        xerr_lo2 = np.clip(xm - xxlo_m, 0, None)
        xerr_hi2 = np.clip(xxhi_m - xm, 0, None)
        ax.errorbar(
            xm, ym,
            xerr=[xerr_lo2, xerr_hi2],
            fmt="none", ecolor="steelblue", alpha=0.06, linewidth=0.4, zorder=1,
        )

    # LOWESS trend
    xl, yl = _lowess_line(xm, ym)
    if len(xl):
        ax.plot(xl, yl, "k-", linewidth=1.5, zorder=4, label="LOWESS")

    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.text(
        0.03, 0.97, corr_text,
        transform=ax.transAxes, fontsize=7,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    # Colourbar
    cbar = plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label("Year", fontsize=7)
    cbar.ax.tick_params(labelsize=6)


def _make_figure(
    window_label: str,
    cr_bins: pd.DataFrame,
    seis_log_e: pd.Series,
    sun_bins: pd.DataFrame,
    stats_list: list[dict],
) -> plt.Figure:
    """
    Three-panel figure for one time window:
      Panel 1: CR (p50) vs log10(Seismic energy)
      Panel 2: CR (p50) vs Sunspot (smoothed)
      Panel 3: Sunspot (smoothed) vs log10(Seismic energy)
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"Raw pairwise correlations — {window_label} (no detrending)",
        fontsize=11, fontweight="bold", y=1.01,
    )

    # Align all series to same index
    idx = cr_bins.index
    cr_p50 = cr_bins["cr_p50"].reindex(idx).values
    cr_p05 = cr_bins["cr_p05"].reindex(idx).values
    cr_p95 = cr_bins["cr_p95"].reindex(idx).values
    cr_min = cr_bins["cr_min"].reindex(idx).values
    cr_max = cr_bins["cr_max"].reindex(idx).values
    seis = seis_log_e.reindex(idx).values
    sn_sm = sun_bins["sn_smooth"].reindex(idx).values
    sn_raw = sun_bins["sn_mean"].reindex(idx).values
    sn_min = sun_bins["sn_min"].reindex(idx).values
    sn_max = sun_bins["sn_max"].reindex(idx).values
    times = _decimal_year(idx)

    def _fmt_stat(d: dict | None, key_r: str, key_rho: str) -> str:
        if d is None:
            return ""
        r = d.get(key_r, np.nan)
        rho = d.get(key_rho, np.nan)
        pp = d.get("pearson_p", np.nan)
        sp = d.get("spearman_p", np.nan)
        def _pstr(p):
            if not np.isfinite(p):
                return "—"
            if p < 0.001:
                return "p<0.001"
            return f"p={p:.3f}"
        rs = f"r={r:.3f}" if np.isfinite(r) else "r=—"
        rhos = f"ρ={rho:.3f}" if np.isfinite(rho) else "ρ=—"
        return f"{rs} {_pstr(pp)}\n{rhos} {_pstr(sp)}"

    # Find stat records for this window
    def _find(label: str) -> dict | None:
        for d in stats_list:
            if d["window"] == window_label and d["label"] == label:
                return d
        return None

    # Panel 1: CR vs Seismic
    ax = axes[0]
    _scatter_panel(
        ax, cr_p50, cr_p05, cr_p95, cr_min, cr_max, seis, times,
        xlabel="CR index (station median, norm.)",
        ylabel="log₁₀(Seismic energy)",
        title="CR vs Seismicity",
        corr_text=_fmt_stat(_find("CR_p50 vs Seismic"), "pearson_r", "spearman_rho"),
        show_x_bands=True,
    )

    # Panel 2: CR vs Sunspot
    ax = axes[1]
    _scatter_panel(
        ax, cr_p50, cr_p05, cr_p95, cr_min, cr_max, sn_sm, times,
        xlabel="CR index (station median, norm.)",
        ylabel="Sunspot number (smoothed)",
        title="CR vs Sunspot Number",
        corr_text=_fmt_stat(_find("CR_p50 vs Sunspot"), "pearson_r", "spearman_rho"),
        show_x_bands=True,
    )

    # Panel 3: Sunspot vs Seismic (sunspot on x with daily spread as error)
    ax = axes[2]
    _scatter_panel(
        ax, sn_sm, sn_min, sn_max, None, None, seis, times,
        xlabel="Sunspot number (365d smoothed)",
        ylabel="log₁₀(Seismic energy)",
        title="Sunspot Number vs Seismicity",
        corr_text=_fmt_stat(_find("Sunspot vs Seismic"), "pearson_r", "spearman_rho"),
        show_x_bands=True,
    )
    # Add raw sunspot spread as lighter error bars
    mask3 = np.isfinite(sn_sm) & np.isfinite(seis)
    ax.errorbar(
        sn_sm[mask3], seis[mask3],
        xerr=[
            np.clip(sn_sm[mask3] - sn_min[mask3], 0, None),
            np.clip(sn_max[mask3] - sn_sm[mask3], 0, None),
        ],
        fmt="none", ecolor="orange", alpha=0.08, linewidth=0.4, zorder=1,
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Download missing OOS data ────────────────────────────────────────
    log.info("Checking OOS USGS data …")
    _ensure_usgs(range(2020, 2026))

    log.info("Checking OOS NMDB data …")
    oos_stations = _oos_stations()
    log.info("OOS stations to download: %d", len(oos_stations))
    _ensure_nmdb_oos(oos_stations, range(2020, 2026))

    # ── 2. Define windows ───────────────────────────────────────────────────
    windows = {
        "In-sample (1976–2019)": (IN_SAMPLE_START, IN_SAMPLE_END),
        "OOS (2020–2025)": (OOS_START, OOS_END),
        "Combined (1976–2025)": (COMBINED_START, COMBINED_END),
    }

    # ── 3. Compute correlations ─────────────────────────────────────────────
    n_tests = 9  # 3 pairs × 3 windows, Bonferroni denominator

    all_stats: list[dict] = []
    window_data: dict[str, tuple] = {}

    for win_label, (wstart, wend) in windows.items():
        log.info("=== Window: %s ===", win_label)

        cr_bins = _load_nmdb_bins(wstart, wend)
        seis_log_e = _load_seismic_energy(wstart, wend)
        sun_bins = _load_sunspot_bins(wstart, wend)

        if cr_bins.empty:
            log.warning("No CR data for %s — skipping", win_label)
            continue

        window_data[win_label] = (cr_bins, seis_log_e, sun_bins)

        # Align to common index
        idx = cr_bins.index
        cr_p50 = cr_bins["cr_p50"].reindex(idx).values
        cr_p95 = cr_bins["cr_p95"].reindex(idx).values
        seis = seis_log_e.reindex(idx).values
        sn = sun_bins["sn_smooth"].reindex(idx).values

        pairs = [
            ("CR_p50 vs Seismic",  cr_p50, seis),
            ("CR_p95 vs Seismic",  cr_p95, seis),
            ("CR_p50 vs Sunspot",  cr_p50, sn),
            ("CR_p95 vs Sunspot",  cr_p95, sn),
            ("Sunspot vs Seismic", sn,     seis),
        ]

        for label, x, y in pairs:
            rec = _correlate_pair(x, y, label, win_label, n_tests)
            all_stats.append(rec)
            log.info(
                "  %-30s  r=% .3f (p=%.3g)  ρ=% .3f (p=%.3g)  n=%d",
                label,
                rec["pearson_r"], rec["pearson_p"],
                rec["spearman_rho"], rec["spearman_p"],
                rec["n_bins"],
            )

    # ── 4. Save JSON ─────────────────────────────────────────────────────────
    out_json = OUT_DIR / "raw_pairwise_correlations.json"

    def _nan_to_none(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: _nan_to_none(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_nan_to_none(v) for v in obj]
        return obj

    with open(out_json, "w") as fh:
        json.dump(_nan_to_none({"n_tests_bonferroni": n_tests, "results": all_stats}), fh, indent=2)
    log.info("Saved %s", out_json)

    # ── 5. Print LaTeX table ─────────────────────────────────────────────────
    _print_latex_table(all_stats)

    # ── 6. Produce figures ───────────────────────────────────────────────────
    fig_names = {
        "In-sample (1976–2019)": "raw_corr_insample.png",
        "OOS (2020–2025)": "raw_corr_oos.png",
        "Combined (1976–2025)": "raw_corr_combined.png",
    }

    for win_label, (cr_bins, seis_log_e, sun_bins) in window_data.items():
        fig = _make_figure(win_label, cr_bins, seis_log_e, sun_bins, all_stats)
        fname = FIG_DIR / fig_names[win_label]
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved %s", fname)

    log.info("Done.")


# ---------------------------------------------------------------------------
# LaTeX table helper
# ---------------------------------------------------------------------------

def _print_latex_table(stats: list[dict]) -> None:
    """Print a LaTeX longtable fragment to stdout."""
    # Primary 9 pairs (CR_p50 + Sunspot vs Seismic)
    primary_labels = ["CR_p50 vs Seismic", "CR_p50 vs Sunspot", "Sunspot vs Seismic"]
    window_order = [
        "In-sample (1976–2019)",
        "OOS (2020–2025)",
        "Combined (1976–2025)",
    ]

    def _lookup(label, window):
        for d in stats:
            if d["label"] == label and d["window"] == window:
                return d
        return {}

    def _rf(v, fmt=".3f"):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return format(v, fmt)

    def _pstar(p_bonf):
        if p_bonf is None or (isinstance(p_bonf, float) and np.isnan(p_bonf)):
            return ""
        if p_bonf < 0.001:
            return "$^{***}$"
        if p_bonf < 0.01:
            return "$^{**}$"
        if p_bonf < 0.05:
            return "$^{*}$"
        return ""

    # Map labels to display names
    label_display = {
        "CR_p50 vs Seismic": r"CR (med.) vs Seismicity",
        "CR_p50 vs Sunspot": r"CR (med.) vs Sunspot",
        "Sunspot vs Seismic": r"Sunspot vs Seismicity",
    }
    win_display = {
        "In-sample (1976–2019)": r"In-sample (1976--2019)",
        "OOS (2020–2025)": r"OOS (2020--2025)",
        "Combined (1976–2025)": r"Combined (1976--2025)",
    }

    lines = []
    lines.append(r"""% Auto-generated by 09_raw_pairwise_correlations.py
\begin{table}[htbp]
  \centering
  \caption{Raw pairwise correlation statistics across three time windows.
           Bonferroni correction applied for $3 \times 3 = 9$ tests.
           CR uses the per-bin station-median index.
           Seismic energy is $\log_{10}\!\left(\sum 10^{1.5 M_W}\right)$.
           Sunspot is the 365-day smoothed daily count.
           $^{*}p_\text{Bonf}<0.05$, $^{**}p_\text{Bonf}<0.01$,
           $^{***}p_\text{Bonf}<0.001$.}
  \label{tab:rawcorr}
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{llrrrrrrr}
    \toprule
    Pair & Window & $N$ &
      $r$ & 95\% CI &
      $p$ (raw) & $p$ (Bonf.) &
      $\rho$ & $p_\rho$ (Bonf.) \\
    \midrule""")

    for lbl in primary_labels:
        disp = label_display[lbl]
        first = True
        for win in window_order:
            d = _lookup(lbl, win)
            r = d.get("pearson_r")
            ci_lo = d.get("pearson_ci_lo")
            ci_hi = d.get("pearson_ci_hi")
            pp = d.get("pearson_p")
            pp_b = d.get("pearson_p_bonf")
            rho = d.get("spearman_rho")
            sp_b = d.get("spearman_p_bonf")
            n = d.get("n_bins", 0)
            star = _pstar(pp_b)
            rho_star = _pstar(sp_b)

            ci_str = f"[{_rf(ci_lo)}, {_rf(ci_hi)}]" if ci_lo is not None else "—"
            row_lbl = disp if first else ""
            first = False

            lines.append(
                f"    {row_lbl} & {win_display.get(win, win)} & {n} & "
                f"{_rf(r)}{star} & {ci_str} & "
                f"{_rf(pp, '.3g')} & {_rf(pp_b, '.3g')} & "
                f"{_rf(rho)}{rho_star} & {_rf(sp_b, '.3g')} \\\\"
            )
        lines.append(r"    \addlinespace")

    lines.append(r"""    \bottomrule
  \end{tabular}
  \bigskip

  \textit{Note: CR\textsubscript{p95} variant (station 95th percentile instead of median)
  gives similar structure; see \texttt{results/raw\_pairwise\_correlations.json}.}
\end{table}""")

    table_text = "\n".join(lines)
    print(table_text)
    # Also save to file
    out = OUT_DIR / "raw_pairwise_table.tex"
    out.write_text(table_text + "\n", encoding="utf-8")
    log.info("LaTeX table written to %s", out)


if __name__ == "__main__":
    main()
