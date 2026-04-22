#!/usr/bin/env python3
"""
scripts/05_geographic_localisation.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests whether the CR–seismic cross-correlation is geographically localised.

For each pair (NMDB station s, 10°×10° seismic grid cell g):
  1. Compute the great-circle distance d(s, g).
  2. Compute the cross-correlation r_{s,g}(τ) and its peak lag τ*(s,g).
  3. Test significance via GPU phase-randomisation surrogates.

Vectorisation: for a fixed station s all N_cells seismic series are evaluated
simultaneously in a single GPU pass (one cuBLAS matmul per lag bin), so the
scan costs O(N_stations) GPU calls rather than O(N_stations × N_cells).

Multiple-testing control: Benjamini–Hochberg FDR at q = 0.05 across all
(station, cell) pairs that pass the minimum-events filter.

Scientific hypotheses
---------------------
H_CR    (cosmic-ray): τ*(s,g) is independent of d(s,g) — CRs are near-isotropic
H_local (radon/EM) : amplitude decays or lag grows with d(s,g) — local coupling

Outputs
-------
results/figs/geo_heatmap.png        — world grid: −log₁₀(min-p) + n_sig_stations
results/figs/geo_distance_lag.png   — distance vs peak-lag + distance vs |r|
results/geo_localisation.json       — full (station, cell) result table
results/geo_localisation_report.md  — narrative summary

Usage
-----
python scripts/05_geographic_localisation.py
python scripts/05_geographic_localisation.py --n-surrogates 5000 --method phase
python scripts/05_geographic_localisation.py --lag-max 100 --min-events 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import scipy.stats
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crq.ingest.nmdb import load_station, resample_daily
from crq.ingest.usgs import load_usgs
from crq.stats.surrogates_gpu import (
    gpu_available,
    phase_randomise_batch_gpu,
    _GPU_REASON,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("crq.geo")

CELL_SIZE = 10          # degrees
N_LAT_CELLS = 180 // CELL_SIZE   # 18
N_LON_CELLS = 360 // CELL_SIZE   # 36
N_CELLS_TOTAL = N_LAT_CELLS * N_LON_CELLS  # 648


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    dφ = np.radians(lat2 - lat1)
    dλ = np.radians(lon2 - lon1)
    a = np.sin(dφ / 2) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(dλ / 2) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def assign_cell_index(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Row-major cell index: lat-bands outer, lon-bands inner."""
    lat_i = np.floor((lat + 90.0) / CELL_SIZE).astype(int).clip(0, N_LAT_CELLS - 1)
    lon_i = np.floor((lon + 180.0) / CELL_SIZE).astype(int).clip(0, N_LON_CELLS - 1)
    return lat_i * N_LON_CELLS + lon_i


def cell_centers() -> tuple[np.ndarray, np.ndarray]:
    """Return (lat_center, lon_center) arrays, shape (N_CELLS_TOTAL,)."""
    lats, lons = [], []
    for lat0 in range(-90, 90, CELL_SIZE):
        for lon0 in range(-180, 180, CELL_SIZE):
            lats.append(lat0 + CELL_SIZE / 2)
            lons.append(lon0 + CELL_SIZE / 2)
    return np.array(lats), np.array(lons)


# ---------------------------------------------------------------------------
# Binning helper (consistent with homola scripts)
# ---------------------------------------------------------------------------

def _bin_series(
    series: pd.Series, study_start: str, bin_days: int, agg: str = "mean"
) -> pd.Series:
    t0 = pd.Timestamp(study_start)
    days = (series.index - t0).days
    bin_num = days // bin_days
    bin_dates = t0 + pd.to_timedelta(bin_num * bin_days, unit="D")
    grouped = series.groupby(bin_dates)
    return grouped.sum() if agg == "sum" else grouped.mean()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_station_series(
    station_ids: list[str],
    start_year: int,
    end_year: int,
    nmdb_dir: Path,
    study_start: str,
    study_end: str,
    bin_days: int,
    coverage_threshold: float = 0.60,
    min_valid_bins: int = 500,
) -> dict[str, np.ndarray]:
    """
    Load every NMDB station, normalise to its mean rate, bin to bin_days.

    Returns mapping station_id → float32 array of length T_ref (NaN where
    the station was not operational).  Stations with < min_valid_bins non-NaN
    bins are dropped.
    """
    t0 = pd.Timestamp(study_start)
    t1 = pd.Timestamp(study_end)
    ref_dates = pd.date_range(study_start, study_end, freq=f"{bin_days}D")
    # Trim to multiples of bin_days from origin
    days_total = (t1 - t0).days
    n_bins = days_total // bin_days + 1
    ref_index = pd.DatetimeIndex(
        [t0 + pd.Timedelta(days=i * bin_days) for i in range(n_bins)]
    )

    out: dict[str, np.ndarray] = {}
    for station in station_ids:
        hourly = load_station(station, start_year, end_year, nmdb_dir)
        if hourly.empty:
            logger.debug("station %s: no data files", station)
            continue
        daily_df = resample_daily(hourly, station, coverage_threshold=coverage_threshold)
        daily = daily_df[station].loc[study_start:study_end]
        n_valid = int(daily.notna().sum())
        if n_valid < 30:
            continue
        mean_ = daily.mean()
        if not np.isfinite(mean_) or mean_ <= 0:
            continue
        norm = (daily / mean_).dropna()
        binned = _bin_series(norm, study_start, bin_days).reindex(ref_index)
        arr = binned.to_numpy(dtype=np.float32)
        n_valid_bins = int(np.isfinite(arr).sum())
        if n_valid_bins < min_valid_bins:
            logger.info("station %s: only %d valid bins — skipped", station, n_valid_bins)
            continue
        out[station] = arr
        logger.info("station %-6s  valid_bins=%d / %d", station, n_valid_bins, len(arr))

    logger.info("Loaded %d / %d stations with sufficient data", len(out), len(station_ids))
    return out, ref_index


def build_cell_seismic_matrix(
    events: pd.DataFrame,
    ref_index: pd.DatetimeIndex,
    study_start: str,
    bin_days: int,
    min_events: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign events to 10°×10° cells and build (N_CELLS_TOTAL, T) float32 matrix.

    Returns
    -------
    Y           : (N_CELLS_TOTAL, T) Mw-sum per cell per bin (zeros where no events)
    n_events    : (N_CELLS_TOTAL,) total event counts per cell
    active_mask : (N_CELLS_TOTAL,) bool — cells with n_events >= min_events
    """
    T = len(ref_index)
    t0 = pd.Timestamp(study_start)

    # Map ref_index bins to integer positions
    ref_bin_num = ((ref_index - t0).days.values // bin_days)
    bin_to_pos = {int(b): i for i, b in enumerate(ref_bin_num)}

    Y = np.zeros((N_CELLS_TOTAL, T), dtype=np.float32)
    n_events = np.zeros(N_CELLS_TOTAL, dtype=np.int32)

    cell_idx = assign_cell_index(events["latitude"].values, events["longitude"].values)
    event_day = (events.index.normalize() - t0).days.values
    event_bin = event_day // bin_days
    mag_vals = events["mag"].values

    # Vectorised accumulation using pandas groupby
    tmp = pd.DataFrame({
        "cell": cell_idx,
        "bin_num": event_bin.astype(int),
        "mag": mag_vals,
    }).dropna(subset=["mag"])

    n_events_ser = tmp.groupby("cell")["mag"].count()
    n_events[n_events_ser.index.values] = n_events_ser.values.astype(np.int32)

    mw_sum = tmp.groupby(["cell", "bin_num"])["mag"].sum()
    for (ci, bn), total in mw_sum.items():
        pos = bin_to_pos.get(int(bn))
        if pos is not None and 0 <= ci < N_CELLS_TOTAL:
            Y[ci, pos] += float(total)

    active_mask = n_events >= min_events
    logger.info(
        "Grid: %d / %d cells have >= %d events",
        active_mask.sum(), N_CELLS_TOTAL, min_events,
    )
    return Y, n_events, active_mask


# ---------------------------------------------------------------------------
# GPU-vectorised geographic surrogate test
# ---------------------------------------------------------------------------

def _pearson_1_vs_cells(
    x: np.ndarray,        # (T,) float32 — station CR, already valid (no NaN)
    Y: np.ndarray,        # (N_cells, T) float32
    lag_bins: np.ndarray, # (L,) int
) -> np.ndarray:          # (N_cells, L) float32
    """Observed Pearson r(τ): one station series vs all cell seismic series."""
    T = len(x)
    N_cells = len(Y)
    L = len(lag_bins)

    x_z = ((x - x.mean()) / (x.std() + 1e-15)).astype(np.float32)
    mu = Y.mean(axis=1, keepdims=True)
    sd = Y.std(axis=1, keepdims=True) + 1e-15
    Y_z = ((Y - mu) / sd).astype(np.float32)

    out = np.zeros((N_cells, L), dtype=np.float32)
    for k, lag in enumerate(lag_bins):
        if lag >= 0:
            n = T - lag
            if n < 2:
                continue
            # correlate x[0:n] with y[:,lag:lag+n]
            out[:, k] = Y_z[:, lag : lag + n] @ x_z[:n] / n
        else:
            n = T + lag          # lag is negative
            if n < 2:
                continue
            # correlate x[|lag|:|lag|+n] with y[:,0:n]
            out[:, k] = Y_z[:, :n] @ x_z[-lag : -lag + n] / n
    return out


def _surr_peak_cupy(
    X_surr: np.ndarray,   # (S, T) float32 — surrogate series
    Y_z: np.ndarray,      # (N_cells, T) float32 — pre-z-scored cell series
    lag_bins: np.ndarray,
) -> np.ndarray:          # (S, N_cells) float32 — max |r| over lags
    """GPU: compute surrogate peak |r| for every (surrogate, cell) pair."""
    import cupy as cp

    S, T = X_surr.shape
    N_cells = Y_z.shape[0]

    X_gpu = cp.asarray(X_surr)
    Y_gpu = cp.asarray(Y_z)

    mu_x = X_gpu.mean(axis=1, keepdims=True)
    sd_x = X_gpu.std(axis=1, keepdims=True) + cp.float32(1e-15)
    X_z_gpu = (X_gpu - mu_x) / sd_x           # (S, T)

    peak = cp.zeros((S, N_cells), dtype=cp.float32)

    for lag in lag_bins:
        if lag >= 0:
            n = T - lag
            if n < 2:
                continue
            r = X_z_gpu[:, :n] @ Y_gpu[:, lag : lag + n].T / n   # (S, N_cells)
        else:
            n = T + lag
            if n < 2:
                continue
            r = X_z_gpu[:, -lag : -lag + n] @ Y_gpu[:, :n].T / n

        peak = cp.maximum(peak, cp.abs(r))

    return cp.asnumpy(peak)


def _surr_peak_cpu(
    X_surr: np.ndarray,   # (S, T) float32
    Y_z: np.ndarray,      # (N_cells, T) float32
    lag_bins: np.ndarray,
) -> np.ndarray:          # (S, N_cells) float32
    """CPU fallback for surrogate peak computation."""
    S, T = X_surr.shape
    N_cells = Y_z.shape[0]

    mu_x = X_surr.mean(axis=1, keepdims=True)
    sd_x = X_surr.std(axis=1, keepdims=True) + 1e-15
    X_z = ((X_surr - mu_x) / sd_x).astype(np.float32)

    peak = np.zeros((S, N_cells), dtype=np.float32)

    for lag in lag_bins:
        if lag >= 0:
            n = T - lag
            if n < 2:
                continue
            r = X_z[:, :n] @ Y_z[:, lag : lag + n].T / n
        else:
            n = T + lag
            if n < 2:
                continue
            r = X_z[:, -lag : -lag + n] @ Y_z[:, :n].T / n

        np.maximum(peak, np.abs(r), out=peak)

    return peak


def geo_surrogate_batch(
    x_station: np.ndarray,   # (T,) float32 — no NaN
    Y_cells: np.ndarray,     # (N_cells, T) float32
    lag_bins: np.ndarray,
    n_surr: int,
    seed: int,
    method: str = "phase",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised surrogate test: one station CR vs all N_cells seismic series.

    Phase surrogates of x_station are generated on GPU (or CPU fallback), then
    for each surrogate the peak |r| over lags is computed against every cell in
    a single batched matrix multiply, giving (S, N_cells) in one GPU pass.

    Returns
    -------
    obs_r        : (N_cells, L) observed correlations at each lag
    obs_peak_r   : (N_cells,)  max |r| over lags
    obs_peak_lag : (N_cells,)  lag bin at peak (in lag_bins units)
    p_global     : (N_cells,)  surrogate p-value
    """
    # Observed cross-correlations
    obs_r = _pearson_1_vs_cells(x_station, Y_cells, lag_bins)
    obs_abs = np.abs(obs_r)
    peak_idx = obs_abs.argmax(axis=1)
    obs_peak_r = obs_abs[np.arange(len(Y_cells)), peak_idx]
    obs_peak_lag = lag_bins[peak_idx]

    # Pre-z-score Y once (reused across all surrogate lags)
    mu = Y_cells.mean(axis=1, keepdims=True)
    sd = Y_cells.std(axis=1, keepdims=True) + 1e-15
    Y_z = ((Y_cells - mu) / sd).astype(np.float32)

    # Generate surrogates of x_station
    if gpu_available() and method == "phase":
        X_surr = phase_randomise_batch_gpu(
            x_station.astype(np.float32), n_surr, seed
        ).astype(np.float32)
    else:
        from crq.stats.surrogates import phase_randomise
        rng = np.random.default_rng(seed)
        seeds_ = rng.integers(0, 2**31, size=n_surr)
        X_surr = np.stack([
            phase_randomise(x_station.astype(np.float64), seed=int(s)).astype(np.float32)
            for s in seeds_
        ])

    # Compute surrogate peak |r| for every (surrogate, cell) pair
    if gpu_available():
        surr_peak = _surr_peak_cupy(X_surr, Y_z, lag_bins)
    else:
        surr_peak = _surr_peak_cpu(X_surr, Y_z, lag_bins)

    # p_global[g] = fraction of surrogates whose peak ≥ observed peak
    p_global = (surr_peak >= obs_peak_r[np.newaxis, :]).mean(axis=0).astype(np.float64)

    return obs_r, obs_peak_r, obs_peak_lag, p_global


# ---------------------------------------------------------------------------
# Benjamini–Hochberg FDR correction
# ---------------------------------------------------------------------------

def benjamini_hochberg(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    """
    Return boolean significance mask at FDR level q.

    Implements Benjamini & Hochberg (1995), assuming independence (or positive
    dependence — PRDS).  Returns True for every test whose adjusted p-value
    (step-up procedure) meets the threshold.
    """
    n = len(p_values)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    thresholds = (np.arange(1, n + 1) / n) * q
    below = sorted_p <= thresholds
    if not below.any():
        return np.zeros(n, dtype=bool)
    # All tests up to the largest k where p_(k) ≤ (k/m)*q are significant
    last_sig = int(below.nonzero()[0][-1])
    sig = np.zeros(n, dtype=bool)
    sig[order[: last_sig + 1]] = True
    return sig


# ---------------------------------------------------------------------------
# Distance-lag regression
# ---------------------------------------------------------------------------

def distance_lag_regression(
    distances: np.ndarray,  # (N,)
    peak_lags: np.ndarray,  # (N,) in days
    peak_rs: np.ndarray,    # (N,)
) -> dict:
    """OLS regression of peak_lag ~ distance and |peak_r| ~ distance."""
    n = len(distances)
    if n < 3:
        return {"n": n, "lag_slope": np.nan, "lag_p": np.nan,
                "r_slope": np.nan, "r_p": np.nan}

    sl, ic, rv, pv, _ = scipy.stats.linregress(distances, peak_lags)
    sl2, ic2, rv2, pv2, _ = scipy.stats.linregress(distances, np.abs(peak_rs))
    return {
        "n": n,
        "lag_slope_days_per_1000km": float(sl * 1000),
        "lag_intercept_days": float(ic),
        "lag_r2": float(rv ** 2),
        "lag_pvalue": float(pv),
        "r_slope_per_1000km": float(sl2 * 1000),
        "r_intercept": float(ic2),
        "r_r2": float(rv2 ** 2),
        "r_pvalue": float(pv2),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_heatmap_figure(
    all_results: list[dict],
    station_meta: dict,
    n_active_cells: int,
    bh_sig: np.ndarray,  # bool mask over all_results
    output_path: Path,
    study_start: str,
    study_end: str,
    n_surr: int,
    fdr_q: float,
) -> None:
    """
    Two-panel world heatmap.
    Panel (a): −log₁₀(min p_global across stations) per cell.
    Panel (b): number of BH-significant stations per cell.
    """
    cell_lat, cell_lon = cell_centers()   # (648,)

    # Aggregate per cell
    min_p = np.ones(N_CELLS_TOTAL)          # 1.0 = no test
    n_sig = np.zeros(N_CELLS_TOTAL, dtype=int)
    n_tested = np.zeros(N_CELLS_TOTAL, dtype=int)

    for i, row in enumerate(all_results):
        ci = row["cell_idx"]
        n_tested[ci] += 1
        p = row["p_global"]
        if p < min_p[ci]:
            min_p[ci] = p
        if bh_sig[i]:
            n_sig[ci] += 1

    # Only show tested cells
    tested_mask = n_tested > 0
    neg_log_p = np.where(tested_mask, -np.log10(np.clip(min_p, 1e-4, 1.0)), np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f"Geographic localisation of CR–seismic cross-correlation  "
        f"({study_start[:4]}–{study_end[:4]}, {n_surr:,} surrogates, BH q={fdr_q})",
        fontsize=12, fontweight="bold",
    )

    for ax, values, cmap, label, title in [
        (axes[0], neg_log_p, "YlOrRd",
         "−log₁₀(min p_global across stations)",
         "(a) Strongest signal per cell"),
        (axes[1], n_sig.astype(float), "Blues",
         f"Number of BH-significant stations (q={fdr_q})",
         "(b) Significant station count per cell"),
    ]:
        ax.set_facecolor("#d0d8e0")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.set_title(title, fontsize=10)

        # Draw grid lines
        ax.grid(True, color="white", linewidth=0.3, alpha=0.5)

        # Scatter cells (size proportional to CELL_SIZE, uniform squares)
        sc = ax.scatter(
            cell_lon[tested_mask],
            cell_lat[tested_mask],
            c=values[tested_mask],
            cmap=cmap,
            s=60,
            marker="s",
            linewidths=0,
            vmin=0,
            vmax=np.nanpercentile(values[tested_mask], 98) if tested_mask.any() else 1,
            zorder=3,
        )
        plt.colorbar(sc, ax=ax, label=label, fraction=0.03, pad=0.02)

        # NMDB stations
        for sid, smeta in station_meta.items():
            ax.scatter(
                smeta["lon"], smeta["lat"],
                marker="^", s=30, c="black", zorder=5, linewidths=0,
            )

        # Reference lines
        ax.axhline(0, color="k", linewidth=0.4, alpha=0.4)
        ax.axvline(0, color="k", linewidth=0.4, alpha=0.4)

        # Annotate Bonferroni / BH thresholds on panel (a)
        if label.startswith("−log"):
            n_tests = len(all_results)
            bh_thresh = fdr_q / n_tests * 5  # rough BH level for rank 5
            ax.axhline(
                -np.log10(fdr_q / n_tests) if n_tests > 0 else 3,
                color="red", linewidth=0.8, linestyle="--", alpha=0.6,
                label=f"Bonferroni −log₁₀(p)",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Heatmap saved: %s", output_path)


def make_distance_lag_figure(
    all_results: list[dict],
    bh_sig: np.ndarray,
    lag_reg_all: dict,
    lag_reg_sig: dict,
    r_reg_all: dict,
    bin_days: int,
    output_path: Path,
) -> None:
    """
    Two-panel distance-lag analysis.
    Panel (a): d(s,g) vs τ*(s,g) for all pairs, highlight significant.
    Panel (b): d(s,g) vs |peak r| with regression.
    """
    dists = np.array([r["distance_km"] for r in all_results])
    lags = np.array([r["peak_lag_days"] for r in all_results])
    rs = np.abs(np.array([r["peak_r"] for r in all_results]))
    ps = np.array([r["p_global"] for r in all_results])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel (a): distance vs lag
    ax = axes[0]
    sc_ns = ax.scatter(
        dists[~bh_sig], lags[~bh_sig],
        c=rs[~bh_sig], cmap="Greys",
        s=5, alpha=0.3, vmin=0, vmax=0.15,
        label="Not significant",
    )
    if bh_sig.any():
        sc_s = ax.scatter(
            dists[bh_sig], lags[bh_sig],
            c=rs[bh_sig], cmap="YlOrRd",
            s=25, alpha=0.85, zorder=5,
            vmin=0, vmax=0.15,
            label="BH significant",
            edgecolors="k", linewidths=0.3,
        )
        plt.colorbar(sc_s, ax=ax, label="|peak r|", fraction=0.03, pad=0.02)

    # Regression line over all pairs
    if lag_reg_all["n"] >= 10:
        x_line = np.linspace(dists.min(), dists.max(), 200)
        ic = lag_reg_all["lag_intercept_days"]
        sl = lag_reg_all["lag_slope_days_per_1000km"] / 1000
        ax.plot(
            x_line, ic + sl * x_line,
            color="steelblue", linewidth=1.5, linestyle="--",
            label=(
                f"OLS all pairs: β={lag_reg_all['lag_slope_days_per_1000km']:.2f} d/1000 km  "
                f"(p={lag_reg_all['lag_pvalue']:.3f})"
            ),
        )

    ax.axhline(0, color="k", linewidth=0.5)
    ax.axhline(15, color="darkorange", linewidth=0.8, linestyle=":", alpha=0.7,
               label="τ = +15 d  (Homola claimed lag)")
    ax.set_xlabel("Great-circle distance d(s,g)  [km]")
    ax.set_ylabel("Peak lag τ*  [days]")
    ax.set_title(
        "(a) Distance vs optimal cross-correlation lag\n"
        "H₀ (CR isotropic): no slope expected",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel (b): distance vs |peak r|
    ax = axes[1]
    ax.scatter(
        dists[~bh_sig], rs[~bh_sig],
        c="lightgray", s=5, alpha=0.4, label="Not significant",
    )
    if bh_sig.any():
        ax.scatter(
            dists[bh_sig], rs[bh_sig],
            c="tomato", s=25, alpha=0.85, zorder=5,
            label="BH significant", edgecolors="k", linewidths=0.3,
        )

    if r_reg_all["n"] >= 10:
        x_line = np.linspace(dists.min(), dists.max(), 200)
        ic = r_reg_all["r_intercept"]
        sl = r_reg_all["r_slope_per_1000km"] / 1000
        ax.plot(
            x_line, ic + sl * x_line,
            color="steelblue", linewidth=1.5, linestyle="--",
            label=(
                f"OLS all pairs: β={r_reg_all['r_slope_per_1000km']:.4f}/1000 km  "
                f"(p={r_reg_all['r_pvalue']:.3f})"
            ),
        )

    ax.set_xlabel("Great-circle distance d(s,g)  [km]")
    ax.set_ylabel("|Peak Pearson r|")
    ax.set_title(
        "(b) Distance vs correlation amplitude\n"
        "H_local (radon/EM): amplitude should decay with distance",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Distance-lag figure saved: %s", output_path)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    all_results: list[dict],
    bh_sig: np.ndarray,
    lag_reg_all: dict,
    lag_reg_sig: dict,
    r_reg_all: dict,
    station_meta: dict,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    n_total = len(all_results)
    n_sig = int(bh_sig.sum())
    fdr_expected = args.fdr_q * n_total

    # Count unique significant cells and stations
    sig_cells = set(r["cell_idx"] for r, s in zip(all_results, bh_sig) if s)
    sig_stations = set(r["station"] for r, s in zip(all_results, bh_sig) if s)

    if n_sig == 0:
        conclusion = (
            f"**No significant (station, cell) pairs detected** at BH q={args.fdr_q} "
            f"out of {n_total:,} tests. "
            f"Expected false discoveries under H₀: {fdr_expected:.1f}. "
            "The absence of significant localisation is consistent with **H_CR** "
            "(cosmic rays are near-isotropic; no geographic structure expected) "
            "and inconsistent with a radon or EM propagation mechanism whose "
            "amplitude or lag would vary with source–detector distance."
        )
    elif n_sig <= fdr_expected * 1.5:
        conclusion = (
            f"**{n_sig} significant pairs** (BH q={args.fdr_q}), "
            f"barely exceeding the expected false-discovery count ({fdr_expected:.1f}). "
            "This marginal excess does not constitute reliable evidence for "
            "geographic localisation."
        )
    else:
        conclusion = (
            f"**{n_sig} significant pairs** (BH q={args.fdr_q}) in "
            f"{len(sig_cells)} cells and {len(sig_stations)} stations, "
            f"vs {fdr_expected:.1f} expected false discoveries. "
            "See distance-lag regression for whether amplitude or lag "
            "shows distance dependence."
        )

    # Distance-lag interpretation
    lag_p = lag_reg_all.get("lag_pvalue", np.nan)
    r_p = r_reg_all.get("lag_pvalue", np.nan)
    if np.isfinite(lag_p) and lag_p < 0.05:
        dist_interp = (
            f"The OLS regression of τ* on d shows a statistically significant slope "
            f"(β = {lag_reg_all['lag_slope_days_per_1000km']:.2f} d/1000 km, "
            f"p = {lag_p:.4f}, R² = {lag_reg_all['lag_r2']:.3f}).  "
            "This is inconsistent with H_CR (isotropic cosmic rays) and "
            "suggests a propagating medium-velocity mechanism."
        )
    else:
        dist_interp = (
            f"The OLS regression of τ* on d is not significant "
            f"(β = {lag_reg_all.get('lag_slope_days_per_1000km', float('nan')):.2f} d/1000 km, "
            f"p = {lag_p:.4f}).  "
            "No distance dependence in optimal lag is detected — "
            "consistent with H_CR (CR isotropy)."
        )

    md = f"""# Geographic Localisation of CR–Seismic Cross-Correlation

Generated: {timestamp}
Study period: {args.study_start} – {args.study_end}
Bin size: {args.bin_days} days
Lag range: {args.lag_min}…{args.lag_max} days (step {args.bin_days} d)
Surrogates: {args.n_surrogates} × phase-randomisation (GPU: {_GPU_REASON})
Min events per cell: {args.min_events}
Grid: {CELL_SIZE}°×{CELL_SIZE}° ({N_CELLS_TOTAL} cells total)
Stations loaded: {len(station_meta)}
Total (station, cell) tests: {n_total:,}
BH q: {args.fdr_q}

## Main finding

{conclusion}

## Distance–lag analysis (all {n_total:,} pairs)

{dist_interp}

| Regression | slope (per 1000 km) | R² | p-value |
|---|---|---|---|
| τ*(s,g) ~ d | {lag_reg_all.get('lag_slope_days_per_1000km', float('nan')):.3f} d | {lag_reg_all.get('lag_r2', float('nan')):.4f} | {lag_reg_all.get('lag_pvalue', float('nan')):.4f} |
| |r*|(s,g) ~ d | {r_reg_all.get('r_slope_per_1000km', float('nan')):.5f} | {r_reg_all.get('r_r2', float('nan')):.4f} | {r_reg_all.get('r_pvalue', float('nan')):.4f} |

## Significant pairs (BH q={args.fdr_q})

- Total significant pairs: **{n_sig}** / {n_total:,}
- Expected false discoveries: **{fdr_expected:.1f}**
- Significant cells: {len(sig_cells)}
- Stations contributing significant pairs: {len(sig_stations)}

## Scientific context

Homola et al. (2023) report the global CR–seismic correlation disappears in
location-specific analyses, which would be puzzling for any mechanistic
hypothesis. This analysis tests that claim quantitatively by controlling
the false-discovery rate across all {n_total:,} geographic pairs.

Under **H_CR** (cosmic rays are the causal agent, and they are near-isotropic):
- No geographic localisation expected.
- τ*(s,g) should be independent of d(s,g).
- |r*(s,g)| should be independent of d(s,g).

Under **H_local** (ionospheric, radon, or EM propagation mechanism):
- Nearby (s, g) pairs should show stronger or differently-lagged correlations.
- τ*(s,g) or |r*(s,g)| should vary systematically with d(s,g).

## Figures

- `results/figs/geo_heatmap.png` — −log₁₀(min p) per cell + BH-significant stations
- `results/figs/geo_distance_lag.png` — distance vs peak lag and distance vs |r|
"""

    md_path = output_dir / "geo_localisation_report.md"
    md_path.write_text(md, encoding="utf-8")
    logger.info("Report saved: %s", md_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--study-start",  default="1976-01-01")
    p.add_argument("--study-end",    default="2019-12-31")
    p.add_argument("--bin-days",     type=int,   default=5)
    p.add_argument("--lag-min",      type=int,   default=-200,
                   help="Min lag in days")
    p.add_argument("--lag-max",      type=int,   default=+200,
                   help="Max lag in days")
    p.add_argument("--min-mag",      type=float, default=4.0)
    p.add_argument("--min-events",   type=int,   default=100,
                   help="Min events per cell to include it in the scan")
    p.add_argument("--min-valid-bins", type=int, default=500,
                   help="Min valid 5-day bins for a station to be included")
    p.add_argument("--n-surrogates", type=int,   default=1_000)
    p.add_argument("--method",       default="phase", choices=["phase"])
    p.add_argument("--fdr-q",        type=float, default=0.05)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--nmdb-dir",  type=Path, default=PROJECT_ROOT/"data"/"raw"/"nmdb")
    p.add_argument("--usgs-dir",  type=Path, default=PROJECT_ROOT/"data"/"raw"/"usgs")
    p.add_argument("--config",    type=Path, default=PROJECT_ROOT/"config"/"stations.yaml")
    p.add_argument("--output-dir",type=Path, default=PROJECT_ROOT/"results")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figs").mkdir(exist_ok=True)

    t_start = pd.Timestamp(args.study_start)
    t_end   = pd.Timestamp(args.study_end)
    start_year, end_year = t_start.year, t_end.year

    # ------------------------------------------------------------------ #
    # 1. Station metadata                                                  #
    # ------------------------------------------------------------------ #
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    station_meta: dict[str, dict] = cfg["stations"]
    station_ids = list(station_meta.keys())
    logger.info("Config: %d station definitions", len(station_ids))

    # ------------------------------------------------------------------ #
    # 2. Load per-station CR series (5-day bins)                          #
    # ------------------------------------------------------------------ #
    logger.info("Loading per-station CR series …")
    station_series, ref_index = load_all_station_series(
        station_ids,
        start_year, end_year,
        args.nmdb_dir,
        args.study_start, args.study_end,
        args.bin_days,
        min_valid_bins=args.min_valid_bins,
    )
    T = len(ref_index)
    logger.info("Reference time axis: T=%d bins (%s – %s)",
                T, ref_index[0].date(), ref_index[-1].date())

    if not station_series:
        raise RuntimeError("No station data loaded.  Run scripts/01_download_data.py first.")

    # ------------------------------------------------------------------ #
    # 3. Load USGS events                                                  #
    # ------------------------------------------------------------------ #
    logger.info("Loading USGS events …")
    events = load_usgs(start_year, end_year, args.usgs_dir)
    if events.empty:
        raise RuntimeError("No USGS data.  Run scripts/01_download_data.py first.")
    events = events.loc[args.study_start:args.study_end]
    events = events[events["mag"] >= args.min_mag].copy()
    logger.info("Events M≥%.1f: %d", args.min_mag, len(events))

    # ------------------------------------------------------------------ #
    # 4. Build per-cell seismic series                                     #
    # ------------------------------------------------------------------ #
    logger.info("Building %d×%d° seismic grid …", CELL_SIZE, CELL_SIZE)
    Y_cells, n_events_per_cell, active_mask = build_cell_seismic_matrix(
        events, ref_index, args.study_start, args.bin_days, args.min_events,
    )
    active_indices = np.where(active_mask)[0]
    n_active = len(active_indices)
    logger.info("Active cells: %d  (min_events=%d)", n_active, args.min_events)

    cell_lat_c, cell_lon_c = cell_centers()

    # ------------------------------------------------------------------ #
    # 5. Lag bins                                                          #
    # ------------------------------------------------------------------ #
    lag_min_b = args.lag_min // args.bin_days
    lag_max_b = args.lag_max // args.bin_days
    lag_bins  = np.arange(lag_min_b, lag_max_b + 1, dtype=int)
    n_lags    = len(lag_bins)
    logger.info("Lag range: %d … %d days  (%d bins)", args.lag_min, args.lag_max, n_lags)

    if gpu_available():
        logger.info("GPU: %s", _GPU_REASON)
    else:
        logger.warning("GPU unavailable (%s) — using CPU fallback (will be slow)", _GPU_REASON)

    # ------------------------------------------------------------------ #
    # 6. Geographic surrogate scan                                         #
    # ------------------------------------------------------------------ #
    Y_active = Y_cells[active_indices]    # (n_active, T)
    cell_lat_active = cell_lat_c[active_indices]
    cell_lon_active = cell_lon_c[active_indices]

    all_results: list[dict] = []
    station_ids_loaded = sorted(station_series.keys())
    n_stations = len(station_ids_loaded)

    logger.info(
        "Scan: %d stations × %d cells = %d pairs  (n_surr=%d, method=%s)",
        n_stations, n_active, n_stations * n_active, args.n_surrogates, args.method,
    )

    t_scan_start = time.perf_counter()

    for s_idx, station_id in enumerate(station_ids_loaded):
        x_full = station_series[station_id]         # (T,) with possible NaN
        valid  = np.isfinite(x_full)
        n_valid = int(valid.sum())
        if n_valid < args.min_valid_bins:
            continue

        # Restrict to station's valid window (same mask applied to all cells)
        x_v = x_full[valid].astype(np.float32)     # (n_valid,)
        Y_v = Y_active[:, valid].astype(np.float32) # (n_active, n_valid)

        # Drop cells with zero variance in this window (no events in window)
        cell_std = Y_v.std(axis=1)
        cell_ok  = cell_std > 1e-6
        if not cell_ok.any():
            logger.debug("station %s: no cells with variance — skip", station_id)
            continue

        s_lat = station_meta[station_id]["lat"]
        s_lon = station_meta[station_id]["lon"]

        logger.info(
            "[%d/%d] %s  n_valid=%d  cells_with_data=%d",
            s_idx + 1, n_stations, station_id, n_valid, int(cell_ok.sum()),
        )

        obs_r, obs_peak_r, obs_peak_lag, p_global = geo_surrogate_batch(
            x_v,
            Y_v[cell_ok],
            lag_bins,
            n_surr=args.n_surrogates,
            seed=args.seed + s_idx * 9973,   # unique seed per station
            method=args.method,
        )

        ok_indices = np.where(cell_ok)[0]
        for i, ai in enumerate(ok_indices):
            g_lat = float(cell_lat_active[ai])
            g_lon = float(cell_lon_active[ai])
            dist  = haversine_km(s_lat, s_lon, g_lat, g_lon)
            all_results.append({
                "station":         station_id,
                "station_lat":     float(s_lat),
                "station_lon":     float(s_lon),
                "cell_idx":        int(active_indices[ai]),
                "cell_lat_center": g_lat,
                "cell_lon_center": g_lon,
                "n_valid_bins":    n_valid,
                "n_cell_events":   int(n_events_per_cell[active_indices[ai]]),
                "distance_km":     float(dist),
                "p_global":        float(p_global[i]),
                "peak_r":          float(obs_peak_r[i]),
                "peak_lag_bins":   int(obs_peak_lag[i]),
                "peak_lag_days":   int(obs_peak_lag[i]) * args.bin_days,
            })

    elapsed = time.perf_counter() - t_scan_start
    n_tests = len(all_results)
    logger.info("Scan complete: %d (station, cell) pairs in %.1f s", n_tests, elapsed)

    if n_tests == 0:
        logger.error("No pairs evaluated — check data availability.")
        return

    # ------------------------------------------------------------------ #
    # 7. Benjamini–Hochberg FDR correction                                 #
    # ------------------------------------------------------------------ #
    p_arr   = np.array([r["p_global"] for r in all_results])
    bh_sig  = benjamini_hochberg(p_arr, q=args.fdr_q)
    n_sig   = int(bh_sig.sum())
    fdr_exp = args.fdr_q * n_tests

    logger.info(
        "BH FDR (q=%.2f): %d / %d pairs significant  (expected FP: %.1f)",
        args.fdr_q, n_sig, n_tests, fdr_exp,
    )

    # ------------------------------------------------------------------ #
    # 8. Distance-lag regression                                           #
    # ------------------------------------------------------------------ #
    dists_all = np.array([r["distance_km"] for r in all_results])
    lags_all  = np.array([r["peak_lag_days"] for r in all_results])
    rs_all    = np.abs(np.array([r["peak_r"] for r in all_results]))

    lag_reg_all = distance_lag_regression(dists_all, lags_all, rs_all)
    r_reg_all   = distance_lag_regression(dists_all, rs_all, rs_all)

    if bh_sig.any():
        lag_reg_sig = distance_lag_regression(
            dists_all[bh_sig], lags_all[bh_sig], rs_all[bh_sig]
        )
    else:
        lag_reg_sig = {"n": 0}

    logger.info(
        "Distance–lag OLS (all pairs): β=%.3f d/1000 km  p=%.4f  R²=%.4f",
        lag_reg_all.get("lag_slope_days_per_1000km", np.nan),
        lag_reg_all.get("lag_pvalue", np.nan),
        lag_reg_all.get("lag_r2", np.nan),
    )
    logger.info(
        "Distance–|r| OLS (all pairs): β=%.5f/1000 km  p=%.4f  R²=%.4f",
        r_reg_all.get("r_slope_per_1000km", np.nan),
        r_reg_all.get("r_pvalue", np.nan),
        r_reg_all.get("r_r2", np.nan),
    )

    # ------------------------------------------------------------------ #
    # 9. Summary table                                                     #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 72)
    print(f"  GEOGRAPHIC LOCALISATION SUMMARY")
    print(f"  Grid: {CELL_SIZE}°×{CELL_SIZE}°  |  Stations: {len(station_series)}  "
          f"|  Active cells: {n_active}")
    print(f"  Total (station, cell) tests: {n_tests:,}")
    print(f"  Surrogates: {args.n_surrogates:,}  |  GPU: {_GPU_REASON}")
    print("=" * 72)
    print(f"  BH q={args.fdr_q}: {n_sig:,} / {n_tests:,} significant  "
          f"(expected FP: {fdr_exp:.1f})")
    print()
    print(f"  Distance–lag  β = {lag_reg_all.get('lag_slope_days_per_1000km', float('nan')):+.3f} d / 1000 km  "
          f"p = {lag_reg_all.get('lag_pvalue', float('nan')):.4f}")
    print(f"  Distance–|r|  β = {r_reg_all.get('r_slope_per_1000km', float('nan')):+.6f} / 1000 km  "
          f"p = {r_reg_all.get('r_pvalue', float('nan')):.4f}")
    print()
    if n_sig == 0:
        verdict = "NO localisation detected — consistent with CR isotropy (H_CR)"
    elif n_sig <= fdr_exp * 2:
        verdict = "Marginal excess; not reliable evidence of localisation"
    else:
        slope_p = lag_reg_all.get("lag_pvalue", 1.0)
        if np.isfinite(slope_p) and slope_p < 0.05:
            verdict = "Significant pairs AND distance-dependent lag — suggests propagating mechanism"
        else:
            verdict = "Significant pairs but NO distance dependence — ambiguous"
    print(f"  Verdict: {verdict}")
    print("=" * 72)
    print()

    # ------------------------------------------------------------------ #
    # 10. Figures                                                          #
    # ------------------------------------------------------------------ #
    make_heatmap_figure(
        all_results, station_meta, n_active, bh_sig,
        output_path=args.output_dir / "figs" / "geo_heatmap.png",
        study_start=args.study_start,
        study_end=args.study_end,
        n_surr=args.n_surrogates,
        fdr_q=args.fdr_q,
    )

    make_distance_lag_figure(
        all_results, bh_sig,
        lag_reg_all, lag_reg_sig, r_reg_all,
        bin_days=args.bin_days,
        output_path=args.output_dir / "figs" / "geo_distance_lag.png",
    )

    # ------------------------------------------------------------------ #
    # 11. JSON output                                                      #
    # ------------------------------------------------------------------ #
    json_results = []
    for row, sig in zip(all_results, bh_sig):
        out_row = dict(row)
        out_row["bh_significant"] = bool(sig)
        json_results.append(out_row)

    json_payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "study_start": args.study_start,
        "study_end":   args.study_end,
        "bin_days":    args.bin_days,
        "lag_min_days": args.lag_min,
        "lag_max_days": args.lag_max,
        "n_surrogates": args.n_surrogates,
        "method":       args.method,
        "fdr_q":        args.fdr_q,
        "n_stations_loaded": len(station_series),
        "n_active_cells":    n_active,
        "n_tests":           n_tests,
        "n_significant_bh":  n_sig,
        "fdr_expected_fp":   round(fdr_exp, 2),
        "gpu_device":        _GPU_REASON,
        "scan_elapsed_s":    round(elapsed, 1),
        "distance_lag_regression_all": lag_reg_all,
        "distance_r_regression_all":   r_reg_all,
        "distance_lag_regression_sig": lag_reg_sig,
        "results": json_results,
    }
    json_path = args.output_dir / "geo_localisation.json"
    json_path.write_text(json.dumps(json_payload, indent=2))
    logger.info("JSON saved: %s", json_path)

    # ------------------------------------------------------------------ #
    # 12. Markdown report                                                  #
    # ------------------------------------------------------------------ #
    write_report(
        all_results, bh_sig,
        lag_reg_all, lag_reg_sig, r_reg_all,
        station_meta, args, args.output_dir,
    )

    logger.info("Done.")


if __name__ == "__main__":
    run(_parse_args())
