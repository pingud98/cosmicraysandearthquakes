#!/usr/bin/env python3
"""
scripts/08_combined_timeseries.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Combined analysis on the FULL series (1976-to-end), comparing in-sample
(1976-2019) vs out-of-sample (2020-to-end) vs combined significance.

Key analyses
------------
1. Overall significance on the full window (phase surrogates, GPU)
2. Rolling r(τ=+15 d) in 3-year windows, 1-year steps — full 50-year span
3. Sinusoid fit: period ∈ [9, 13] years to the rolling r time series
4. Bayes factor: constant vs 11-year sinusoidal model (BIC approximation)
5. Does appending OOS data strengthen or weaken significance?
6. Station roster A/B/C comparison

Outputs
-------
results/figs/full_series_with_envelope_fit.png
results/combined_analysis.json
results/combined_analysis_report.md

Usage
-----
python scripts/08_combined_timeseries.py
python scripts/08_combined_timeseries.py --study-end 2026-01-01 --n-surrogates 10000
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crq.ingest.nmdb import load_station, resample_daily
from crq.ingest.usgs import load_usgs, seismic_energy_per_bin
from crq.stats.surrogates_gpu import (
    surrogate_xcorr_test_gpu,
    gpu_available,
    _GPU_REASON,
)
from crq.stats.surrogates import n_eff_bretherton, p_to_sigma
from crq.ingest.station_roster import (
    station_cr_series,
    global_cr_index,
    probe_station_coverage,
    classify_stations,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("crq.combined")

BIN_DAYS         = 5
LAG_15D_BINS     = 15 // BIN_DAYS          # 3 bins
INSAMPLE_START   = "1976-01-01"
INSAMPLE_END     = "2019-12-31"
OOS_START        = "2020-01-01"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _ref_index(study_start: str, study_end: str, bin_days: int) -> pd.DatetimeIndex:
    t0 = pd.Timestamp(study_start)
    t1 = pd.Timestamp(study_end)
    n  = (t1 - t0).days // bin_days + 1
    return pd.DatetimeIndex([t0 + pd.Timedelta(days=i * bin_days) for i in range(n)])


def _bin_series(s: pd.Series, t0: pd.Timestamp, bin_days: int, agg: str = "mean") -> pd.Series:
    days = (s.index - t0).days
    bn   = days // bin_days
    bd   = t0 + pd.to_timedelta(bn * bin_days, unit="D")
    grp  = s.groupby(bd)
    return grp.sum() if agg == "sum" else grp.mean()


def load_full_window(
    station_ids: list[str],
    study_start: str,
    study_end: str,
    nmdb_dir: Path,
    usgs_dir: Path,
    bin_days: int = BIN_DAYS,
    min_mag: float = 4.0,
    min_stations: int = 3,
    coverage_threshold: float = 0.60,
) -> tuple[pd.Series, pd.Series, pd.DatetimeIndex, int]:
    t0         = pd.Timestamp(study_start)
    t1         = pd.Timestamp(study_end)
    start_year = t0.year
    end_year   = t1.year
    ref_index  = _ref_index(study_start, study_end, bin_days)

    # Per-station CR
    norm_cols: dict[str, pd.Series] = {}
    for station in station_ids:
        hourly = load_station(station, start_year, end_year, nmdb_dir)
        if hourly.empty:
            continue
        daily_df = resample_daily(hourly, station, coverage_threshold=coverage_threshold)
        daily    = daily_df[station].loc[study_start:study_end]
        n_valid  = int(daily.notna().sum())
        if n_valid < 30:
            continue
        mean_    = daily.mean()
        if not (np.isfinite(mean_) and mean_ > 0):
            continue
        norm_cols[station] = (daily / mean_).dropna()

    n_stations = len(norm_cols)
    if n_stations == 0:
        raise RuntimeError(f"No CR station data for {study_start}–{study_end}")

    df_norm      = pd.DataFrame(norm_cols)
    n_valid_day  = df_norm.notna().sum(axis=1)
    global_daily = df_norm.mean(axis=1)
    global_daily[n_valid_day < min(min_stations, n_stations)] = np.nan

    cr_bin = _bin_series(global_daily, t0, bin_days).reindex(ref_index)

    # Seismic: log10 of summed seismic energy (E = 10^(1.5·Mw+4.8) J)
    events = load_usgs(start_year, end_year, usgs_dir)
    seismic_bin = seismic_energy_per_bin(
        events, study_start, study_end, bin_days, t0, min_mag=min_mag,
    )
    floor_val = float(seismic_bin.min())
    seismic_bin = seismic_bin.reindex(ref_index, fill_value=floor_val)

    common      = cr_bin.index.intersection(seismic_bin.index)
    cr_bin      = cr_bin.reindex(common)
    seismic_bin = seismic_bin.reindex(common)

    logger.info(
        "Window %s–%s: %d CR stations, %d bins, %d events",
        study_start[:4], study_end[:4], n_stations, len(common), len(events),
    )
    return cr_bin, seismic_bin, ref_index, n_stations


# ---------------------------------------------------------------------------
# Rolling r(τ=15)
# ---------------------------------------------------------------------------

def compute_rolling_r15(
    cr: pd.Series,
    seismic: pd.Series,
    window_bins: int,
    step_bins: int,
    lag_bins: int = LAG_15D_BINS,
) -> list[dict]:
    """
    Rolling Pearson r at a fixed lag over the full series.
    Returns list of {center_date, center_year, r, n_pairs, n_eff}.
    """
    cr_arr  = cr.to_numpy(dtype=np.float64)
    sei_arr = seismic.to_numpy(dtype=np.float64)
    dates   = cr.index
    T       = len(cr_arr)

    results = []
    for i0 in range(0, T - window_bins + 1, step_bins):
        i1  = i0 + window_bins
        x   = cr_arr[i0:i1]
        y   = sei_arr[i0:i1]

        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 30:
            continue

        x_v = x[valid]; y_v = y[valid]
        n   = len(x_v)
        L   = lag_bins

        if L >= n:
            continue

        n_pairs = n - L
        xa = x_v[:n_pairs]; ya = y_v[L:L + n_pairs]
        if n_pairs < 10:
            continue

        try:
            r, _ = scipy.stats.pearsonr(xa, ya)
        except Exception:
            r = np.nan

        n_eff = float(n_eff_bretherton(x_v, y_v))
        center = dates[i0 + window_bins // 2]
        results.append({
            "center_date":  str(center.date()),
            "center_year":  center.year + center.month / 12,
            "r":            float(r),
            "n_pairs":      n_pairs,
            "n_eff":        round(n_eff, 1),
        })

    return results


# ---------------------------------------------------------------------------
# Sinusoid fitting + Bayes factor (BIC)
# ---------------------------------------------------------------------------

def sinusoid_model(t: np.ndarray, A: float, P: float, phi: float, mu: float) -> np.ndarray:
    return A * np.sin(2 * np.pi / P * t + phi) + mu


def fit_sinusoid(
    years: np.ndarray,
    r_vals: np.ndarray,
    period_bounds: tuple[float, float] = (9.0, 13.0),
) -> dict:
    """
    Fit constant and sinusoidal models to the rolling r time series.

    Returns dict with model parameters, residuals, BIC, and Bayes factor.
    """
    finite = np.isfinite(years) & np.isfinite(r_vals)
    t = years[finite] - years[finite].mean()   # centre time axis
    r = r_vals[finite]
    n = len(r)
    if n < 8:
        return {"status": "too_few_points", "n": n}

    # Model A: constant (1 free parameter: mu)
    mu_A    = r.mean()
    rss_A   = float(np.sum((r - mu_A) ** 2))
    k_A     = 1
    bic_A   = n * math.log(rss_A / n) + k_A * math.log(n)

    # Model B: A*sin(2π/P*t + φ) + mu (4 free parameters: A, P, φ, mu)
    best_res = None
    best_rss = np.inf

    for P_init in np.linspace(period_bounds[0], period_bounds[1], 5):
        for phi_init in np.linspace(0, 2 * math.pi, 4, endpoint=False):
            try:
                popt, _ = scipy.optimize.curve_fit(
                    sinusoid_model, t, r,
                    p0=[0.02, P_init, phi_init, mu_A],
                    bounds=(
                        [-0.3, period_bounds[0], -2*math.pi, -0.5],
                        [ 0.3, period_bounds[1],  4*math.pi,  0.5],
                    ),
                    maxfev=10_000,
                )
                r_hat = sinusoid_model(t, *popt)
                rss   = float(np.sum((r - r_hat) ** 2))
                if rss < best_rss:
                    best_rss = rss
                    best_res = popt
            except Exception:
                pass

    if best_res is None:
        return {"status": "fit_failed", "bic_A": bic_A, "rss_A": rss_A, "n": n}

    A_fit, P_fit, phi_fit, mu_fit = best_res
    k_B    = 4
    rss_B  = best_rss
    bic_B  = n * math.log(max(rss_B, 1e-20) / n) + k_B * math.log(n)
    d_bic  = bic_A - bic_B                        # positive = B preferred
    bf     = math.exp(d_bic / 2)                  # Bayes factor B vs A

    return {
        "status":          "ok",
        "n":               n,
        "model_A_mu":      float(mu_A),
        "model_A_rss":     float(rss_A),
        "model_A_bic":     float(bic_A),
        "model_B_A":       float(A_fit),
        "model_B_period":  float(P_fit),
        "model_B_phi":     float(phi_fit),
        "model_B_mu":      float(mu_fit),
        "model_B_rss":     float(rss_B),
        "model_B_bic":     float(bic_B),
        "delta_bic":       float(d_bic),
        "bayes_factor":    float(bf),
        "preferred_model": "B (sinusoidal)" if d_bic > 0 else "A (constant)",
    }


# ---------------------------------------------------------------------------
# Station roster analysis
# ---------------------------------------------------------------------------

def roster_analysis(
    station_ids: list[str],
    nmdb_dir: Path,
    insample_start: str,
    insample_end: str,
    oos_start: str,
    oos_end: str,
    usgs_dir: Path,
    bin_days: int,
    n_surr: int,
    lag_bins: np.ndarray,
    seed: int,
) -> dict:
    """
    Compare surrogate-test p_global for station rosters A, B_oos, C.

    Only runs if roster C (new stations) is non-empty.
    Returns dict of roster_name -> p_global.
    """
    windows = {
        "in_sample":     (insample_start, insample_end),
        "out_of_sample": (oos_start,      oos_end),
    }
    logger.info("Probing station coverage for roster classification …")
    cov: dict[str, dict[str, float]] = {}
    for sid in station_ids:
        try:
            cov[sid] = probe_station_coverage(sid, windows, nmdb_dir)
        except Exception as exc:
            logger.debug("station %s probe error: %s", sid, exc)
            cov[sid] = {k: 0.0 for k in windows}

    rosters = classify_stations(station_ids, cov,
                                in_sample_key="in_sample",
                                oos_key="out_of_sample")
    logger.info(
        "Rosters — A: %d  B_oos: %d  C (new): %d",
        len(rosters["A"]), len(rosters["B_oos"]), len(rosters["C"]),
    )

    results = {
        "roster_sizes": {k: len(v) for k, v in rosters.items()},
        "rosters":      rosters,
        "p_global":     {},
    }

    start_year = int(oos_start[:4])
    end_year   = int(oos_end[:4])
    t0         = pd.Timestamp(oos_start)
    ref_idx    = _ref_index(oos_start, oos_end, bin_days)

    # Seismic series (log10 summed energy, same for all rosters)
    events = load_usgs(start_year, end_year, usgs_dir)
    sei_bin = seismic_energy_per_bin(
        events, oos_start, oos_end, bin_days, t0, min_mag=4.0,
    )
    floor_val = float(sei_bin.min())
    sei_bin = sei_bin.reindex(ref_idx, fill_value=floor_val)

    for label, roster in [("A", rosters["A"]),
                          ("B_oos", rosters["B_oos"]),
                          ("C", rosters["C"])]:
        if not roster:
            results["p_global"][label] = None
            logger.info("Roster %s: empty — skip", label)
            continue

        norm_cols: dict[str, pd.Series] = {}
        for sid in roster:
            hourly = load_station(sid, start_year, end_year, nmdb_dir)
            if hourly.empty:
                continue
            daily_df = resample_daily(hourly, sid)
            daily    = daily_df[sid].loc[oos_start:oos_end]
            mean_    = daily.mean()
            if not (np.isfinite(mean_) and mean_ > 0):
                continue
            norm_cols[sid] = (daily / mean_).dropna()

        if not norm_cols:
            results["p_global"][label] = None
            continue

        df_norm = pd.DataFrame(norm_cols)
        n_valid_day = df_norm.notna().sum(axis=1)
        global_daily = df_norm.mean(axis=1)
        global_daily[n_valid_day < 1] = np.nan
        cr_bin = _bin_series(global_daily, t0, bin_days).reindex(ref_idx)

        common = cr_bin.notna() & sei_bin.notna()
        x_arr  = cr_bin[common].to_numpy(dtype=np.float32)
        y_arr  = sei_bin[common].to_numpy(dtype=np.float32)

        if len(x_arr) < 50:
            results["p_global"][label] = None
            continue

        sr = surrogate_xcorr_test_gpu(
            x_arr, y_arr, lag_bins,
            n_surrogates=n_surr, method="phase", seed=seed + hash(label) % 1000,
        )
        results["p_global"][label] = round(float(sr["p_global"]), 6)
        logger.info("Roster %s (%d stations): p_global = %.4f",
                    label, len(roster), results["p_global"][label])

    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_combined_figure(
    rolling_full: list[dict],
    sinusoid_fit: dict,
    insample_end: str,
    oos_start: str,
    study_start: str,
    study_end: str,
    output_path: Path,
) -> None:
    years = np.array([rw["center_year"] for rw in rolling_full])
    rs    = np.array([rw["r"]           for rw in rolling_full])

    fig = plt.figure(figsize=(16, 7))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ── Panel 1: rolling r(τ=+15 d) ─────────────────────────────────────
    insample_end_year = float(insample_end[:4]) + 1.0
    oos_start_year    = float(oos_start[:4])

    in_mask  = years <= insample_end_year
    oos_mask = years > oos_start_year - 0.5

    ax1.scatter(years[in_mask],  rs[in_mask],  s=20, color="steelblue",  zorder=4,
                label="In-sample (1976–2019)")
    ax1.scatter(years[oos_mask], rs[oos_mask], s=20, color="tomato",     zorder=4,
                label="Out-of-sample (2020–)")
    ax1.plot(years, rs, color="k", linewidth=0.6, alpha=0.4)

    # Sinusoid fit overlay
    if sinusoid_fit.get("status") == "ok":
        P    = sinusoid_fit["model_B_period"]
        A    = sinusoid_fit["model_B_A"]
        phi  = sinusoid_fit["model_B_phi"]
        mu   = sinusoid_fit["model_B_mu"]
        t_c  = years.mean()
        t_dense = np.linspace(years.min(), years.max(), 500)
        fit_vals = sinusoid_model(t_dense - t_c, A, P, phi, mu)
        bf = sinusoid_fit["bayes_factor"]
        ax1.plot(t_dense, fit_vals, color="darkorange", linewidth=2.0,
                 label=f"Sinusoid fit: P={P:.1f} yr, A={A:.3f}  (BF={bf:.2f})")

    ax1.axhline(0, color="k", linewidth=0.5)
    ax1.axvline(oos_start_year, color="gray", linewidth=1.2,
                linestyle="--", alpha=0.7, label="In-sample / OOS boundary")
    ax1.set_ylabel(f"Pearson r(τ = +15 d)")
    ax1.set_title(
        f"Full-series rolling r(τ=+15 d) in 3-year windows (1-year step)  |  "
        f"{study_start[:4]}–{study_end[:4]}",
        fontsize=10,
    )
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: rolling n_eff ───────────────────────────────────────────
    n_effs = np.array([rw["n_eff"] for rw in rolling_full])
    ax2.fill_between(years, 0, n_effs, alpha=0.4, color="steelblue")
    ax2.axvline(oos_start_year, color="gray", linewidth=1.2, linestyle="--", alpha=0.7)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Effective N")
    ax2.set_title("Bretherton effective sample size per rolling window", fontsize=9)
    ax2.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Combined figure saved: %s", output_path)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fv(v, fmt: str) -> str:
    """Format a possibly-None float; returns 'N/A' when None/NaN."""
    if v is None:
        return "N/A"
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return "N/A"


def write_combined_report(
    results: dict,
    sinusoid_fit: dict,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    ts   = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    bf   = sinusoid_fit.get("bayes_factor", float("nan"))
    P    = sinusoid_fit.get("model_B_period", float("nan"))

    # Interpret Bayes factor (Jeffreys scale)
    if np.isnan(bf):
        bf_interp = "Could not be computed"
    elif bf < 1:
        bf_interp = f"BF = {bf:.2f} < 1: evidence FAVOURS constant model (no envelope)"
    elif bf < 3:
        bf_interp = f"BF = {bf:.2f}: anecdotal evidence for sinusoidal envelope"
    elif bf < 10:
        bf_interp = f"BF = {bf:.2f}: moderate evidence for sinusoidal envelope"
    elif bf < 30:
        bf_interp = f"BF = {bf:.2f}: strong evidence for sinusoidal envelope"
    else:
        bf_interp = f"BF = {bf:.2f}: very strong evidence for sinusoidal envelope"

    md = f"""# Combined Full-Series Analysis (1976–{args.study_end[:4]})

Generated: {ts}
Full window: {args.study_start} → {args.study_end}
In-sample: {INSAMPLE_START} → {INSAMPLE_END}
Out-of-sample: {OOS_START} → {args.study_end}
GPU: {_GPU_REASON}
Surrogates: {args.n_surrogates:,} per window

## Does appending OOS data strengthen or weaken significance?

| Window | p_global | σ_surrogate | peak lag |
|---|---|---|---|
| In-sample (1976–2019) | {_fv(results.get('p_global_insample'), '.4f')} | {_fv(results.get('sigma_insample'), '.2f')} | {results.get('peak_lag_insample', 'N/A')} d |
| Out-of-sample (2020–{args.study_end[:4]}) | {_fv(results.get('p_global_oos'), '.4f')} | {_fv(results.get('sigma_oos'), '.2f')} | {results.get('peak_lag_oos', 'N/A')} d |
| Combined (1976–{args.study_end[:4]}) | {_fv(results.get('p_global_full'), '.4f')} | {_fv(results.get('sigma_full'), '.2f')} | {results.get('peak_lag_full', 'N/A')} d |

## Sinusoidal envelope fit

{bf_interp}

Best-fit period: **{P:.2f} years** (constrained to [9, 13] years)

| Parameter | Value |
|---|---|
| Period P | {sinusoid_fit.get('model_B_period', float('nan')):.2f} yr |
| Amplitude A | {sinusoid_fit.get('model_B_A', float('nan')):.4f} |
| Phase φ | {sinusoid_fit.get('model_B_phi', float('nan')):.2f} rad |
| Baseline μ | {sinusoid_fit.get('model_B_mu', float('nan')):.4f} |
| Model B BIC | {sinusoid_fit.get('model_B_bic', float('nan')):.2f} |
| Model A BIC | {sinusoid_fit.get('model_A_bic', float('nan')):.2f} |
| ΔBIC (A−B) | {sinusoid_fit.get('delta_bic', float('nan')):.2f} |
| Bayes factor (BF) | {bf:.3f} |

## Station roster comparison (OOS window)

| Roster | Description | Stations | p_global |
|---|---|---|---|
| A | In BOTH windows | {results.get('roster', {}).get('roster_sizes', {}).get('A', '?')} | {results.get('roster', {}).get('p_global', {}).get('A', 'N/A')} |
| B_oos | All OOS stations | {results.get('roster', {}).get('roster_sizes', {}).get('B_oos', '?')} | {results.get('roster', {}).get('p_global', {}).get('B_oos', 'N/A')} |
| C | New OOS-only | {results.get('roster', {}).get('roster_sizes', {}).get('C', '?')} | {results.get('roster', {}).get('p_global', {}).get('C', 'N/A')} |

A real effect should appear consistently across all three rosters.
Divergence (e.g., significant only in A) would suggest station-selection bias.

## Figure
`results/figs/full_series_with_envelope_fit.png`
"""
    path = output_dir / "combined_analysis_report.md"
    path.write_text(md, encoding="utf-8")
    logger.info("Combined report saved: %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--study-start",  default=INSAMPLE_START)
    p.add_argument("--study-end",    default=None,
                   help="Override end date (default: read from data_availability.json)")
    p.add_argument("--bin-days",     type=int,   default=BIN_DAYS)
    p.add_argument("--lag-min",      type=int,   default=-200)
    p.add_argument("--lag-max",      type=int,   default=+200)
    p.add_argument("--min-mag",      type=float, default=4.0)
    p.add_argument("--n-surrogates", type=int,   default=10_000,
                   help="Surrogates per window surrogate test (default 10 000)")
    p.add_argument("--roll-window",  type=int,   default=None,
                   help="Rolling window bins (default: 3 years)")
    p.add_argument("--roll-step",    type=int,   default=None,
                   help="Rolling step bins (default: 1 year)")
    p.add_argument("--period-min",   type=float, default=9.0)
    p.add_argument("--period-max",   type=float, default=13.0)
    p.add_argument("--roster",       action="store_true",
                   help="Run station roster A/B/C comparison (slow)")
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

    # Determine study end
    avail_path = args.output_dir / "data_availability.json"
    if args.study_end is None:
        if avail_path.exists():
            avail          = json.loads(avail_path.read_text())
            args.study_end = avail["oos_end"]
            logger.info("Study end from data_availability.json: %s", args.study_end)
        else:
            from datetime import date, timedelta
            args.study_end = str(date.today() - timedelta(days=60))
            logger.warning("data_availability.json not found; using %s", args.study_end)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    station_ids = list(cfg["stations"].keys())

    lag_min_b = args.lag_min // args.bin_days
    lag_max_b = args.lag_max // args.bin_days
    lag_bins  = np.arange(lag_min_b, lag_max_b + 1, dtype=int)

    # Rolling window / step defaults
    roll_window = args.roll_window or int(3 * 365.25 / args.bin_days)   # 3 years
    roll_step   = args.roll_step   or int(1 * 365.25 / args.bin_days)   # 1 year

    # ------------------------------------------------------------------ #
    # 1. Load three windows                                                #
    # ------------------------------------------------------------------ #
    logger.info("Loading full window %s–%s …", args.study_start, args.study_end)
    cr_full, sei_full, _, n_st_full = load_full_window(
        station_ids, args.study_start, args.study_end,
        args.nmdb_dir, args.usgs_dir, args.bin_days, args.min_mag,
    )

    logger.info("Loading in-sample window %s–%s …", INSAMPLE_START, INSAMPLE_END)
    try:
        cr_in, sei_in, _, _ = load_full_window(
            station_ids, INSAMPLE_START, INSAMPLE_END,
            args.nmdb_dir, args.usgs_dir, args.bin_days, args.min_mag,
        )
    except Exception as exc:
        logger.warning("In-sample load failed: %s", exc)
        cr_in = sei_in = None

    logger.info("Loading OOS window %s–%s …", OOS_START, args.study_end)
    try:
        cr_oos, sei_oos, _, _ = load_full_window(
            station_ids, OOS_START, args.study_end,
            args.nmdb_dir, args.usgs_dir, args.bin_days, args.min_mag,
        )
    except Exception as exc:
        logger.warning("OOS load failed: %s", exc)
        cr_oos = sei_oos = None

    # ------------------------------------------------------------------ #
    # 2. Surrogate significance per window                                 #
    # ------------------------------------------------------------------ #
    def _run_surr(cr: pd.Series, sei: pd.Series, label: str) -> dict:
        valid = cr.notna() & sei.notna()
        x = cr[valid].to_numpy(dtype=np.float32)
        y = sei[valid].to_numpy(dtype=np.float32)
        if len(x) < 50:
            return {"p_global": np.nan, "sigma": np.nan, "peak_lag_days": None}
        sr = surrogate_xcorr_test_gpu(
            x, y, lag_bins, n_surrogates=args.n_surrogates,
            method="phase", seed=args.seed,
        )
        p   = float(sr["p_global"])
        sig = p_to_sigma(p) if p > 0 else float("inf")
        pk  = int(sr["observed_peak_lag"]) * args.bin_days
        logger.info("%s: p_global=%.4f (%.2fσ), peak=%+d d", label, p, sig, pk)
        return {"p_global": round(p, 6), "sigma": round(sig, 3),
                "peak_lag_days": pk, "T": len(x)}

    surr_full    = _run_surr(cr_full, sei_full, "full")
    surr_insample = _run_surr(cr_in,  sei_in,   "in-sample")  if cr_in  is not None else {}
    surr_oos      = _run_surr(cr_oos, sei_oos,  "OOS")         if cr_oos is not None else {}

    # ------------------------------------------------------------------ #
    # 3. Rolling r(τ=+15 d) — full series                                 #
    # ------------------------------------------------------------------ #
    logger.info("Computing rolling r(τ=+15 d), window=%d bins, step=%d bins …",
                roll_window, roll_step)
    rolling_full = compute_rolling_r15(
        cr_full, sei_full, roll_window, roll_step, lag_bins=LAG_15D_BINS,
    )
    logger.info("Rolling windows: %d", len(rolling_full))

    # ------------------------------------------------------------------ #
    # 4. Sinusoid fit + Bayes factor                                       #
    # ------------------------------------------------------------------ #
    if rolling_full:
        years_arr = np.array([rw["center_year"] for rw in rolling_full])
        r_arr     = np.array([rw["r"]           for rw in rolling_full])
        sinusoid_fit = fit_sinusoid(
            years_arr, r_arr,
            period_bounds=(args.period_min, args.period_max),
        )
        logger.info(
            "Sinusoid fit: P=%.2f yr, BF=%.3f, ΔBIC=%.2f",
            sinusoid_fit.get("model_B_period", np.nan),
            sinusoid_fit.get("bayes_factor",   np.nan),
            sinusoid_fit.get("delta_bic",      np.nan),
        )
    else:
        sinusoid_fit = {"status": "no_rolling_data"}

    # ------------------------------------------------------------------ #
    # 5. Station roster comparison (optional)                              #
    # ------------------------------------------------------------------ #
    roster_res = {}
    if args.roster:
        roster_res = roster_analysis(
            station_ids, args.nmdb_dir,
            INSAMPLE_START, INSAMPLE_END,
            OOS_START, args.study_end,
            args.usgs_dir, args.bin_days,
            min(args.n_surrogates, 1000), lag_bins, args.seed,
        )

    # ------------------------------------------------------------------ #
    # 6. Figure                                                            #
    # ------------------------------------------------------------------ #
    make_combined_figure(
        rolling_full, sinusoid_fit,
        INSAMPLE_END, OOS_START,
        args.study_start, args.study_end,
        args.output_dir / "figs" / "full_series_with_envelope_fit.png",
    )

    # ------------------------------------------------------------------ #
    # 7. JSON + report                                                     #
    # ------------------------------------------------------------------ #
    combined_results = {
        "study_start":      args.study_start,
        "study_end":        args.study_end,
        "n_surrogates":     args.n_surrogates,
        "gpu_device":       _GPU_REASON,
        "roll_window_bins": roll_window,
        "roll_step_bins":   roll_step,
        "p_global_full":    surr_full.get("p_global"),
        "sigma_full":       surr_full.get("sigma"),
        "peak_lag_full":    surr_full.get("peak_lag_days"),
        "T_full":           surr_full.get("T"),
        "p_global_insample": surr_insample.get("p_global"),
        "sigma_insample":    surr_insample.get("sigma"),
        "peak_lag_insample": surr_insample.get("peak_lag_days"),
        "p_global_oos":     surr_oos.get("p_global"),
        "sigma_oos":        surr_oos.get("sigma"),
        "peak_lag_oos":     surr_oos.get("peak_lag_days"),
        "sinusoid_fit":     sinusoid_fit,
        "roster":           roster_res,
        "n_rolling_windows": len(rolling_full),
        "rolling_windows":  rolling_full,
    }

    json_path = args.output_dir / "combined_analysis.json"
    json_path.write_text(
        json.dumps(combined_results, indent=2, default=str), encoding="utf-8"
    )
    logger.info("JSON saved: %s", json_path)

    write_combined_report(combined_results, sinusoid_fit, args, args.output_dir)

    # Summary
    print()
    print("=" * 72)
    print(f"  COMBINED ANALYSIS:  {args.study_start[:4]}–{args.study_end[:4]}")
    print("=" * 72)
    print(f"  Full:       p={surr_full.get('p_global', float('nan')):.4f}  "
          f"σ={surr_full.get('sigma', float('nan')):.2f}  "
          f"peak={surr_full.get('peak_lag_days', '?')} d")
    print(f"  In-sample:  p={surr_insample.get('p_global', float('nan')):.4f}")
    print(f"  OOS:        p={surr_oos.get('p_global', float('nan')):.4f}")
    bf_val = sinusoid_fit.get("bayes_factor", float("nan"))
    P_val  = sinusoid_fit.get("model_B_period", float("nan"))
    print(f"  Sinusoid:   P={P_val:.2f} yr  BF={bf_val:.3f}  "
          f"({'preferred' if bf_val > 1 else 'not preferred'} vs constant)")
    print("=" * 72)
    print()
    logger.info("Done.")


if __name__ == "__main__":
    run(_parse_args())
