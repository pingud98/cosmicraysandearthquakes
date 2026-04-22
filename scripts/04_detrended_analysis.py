#!/usr/bin/env python3
"""
scripts/04_detrended_analysis.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Does the Homola et al. 2023 CR–seismic cross-correlation SURVIVE solar-cycle
detrending?  This script applies three detrending methods to both the CR index
and seismic metric, then re-runs the full IAAFT surrogate stress test on each.

The dominant peak in the raw r(τ) is at τ ≈ −525 d (half a solar cycle), not
τ = +15 d.  If that peak disappears after detrending the result is consistent
with a spurious solar-cycle confound rather than a genuine CR–seismic link.

Detrending methods
------------------
1. Hodrick-Prescott (HP) filter  — λ = 1.29 × 10^5 (calibrated for 5-day bins)
2. STL decomposition — period = 803 bins ≈ 11-year solar cycle
3. Sunspot OLS regression — contemporaneous + 30/90/180-day lagged sunspot numbers

Outputs
-------
results/figs/detrended_xcorr.png    — 2×2 figure (raw / HP / STL / sunspot-reg)
results/detrended_report.md         — narrative summary
results/detrended_results.json      — machine-readable table

Usage
-----
python scripts/04_detrended_analysis.py
python scripts/04_detrended_analysis.py --n-surrogates 1000 --n-jobs 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import scipy.stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crq.stats.surrogates import (
    n_eff_bretherton,
    p_to_sigma,
    surrogate_xcorr_test,
)
from crq.preprocess.detrend import (
    hp_filter_detrend,
    stl_detrend,
    sunspot_regression_detrend,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("crq.detrend_analysis")


# ---------------------------------------------------------------------------
# Import shared data from 02_homola_replication.py
# ---------------------------------------------------------------------------

def _import_homola():
    spec = importlib.util.spec_from_file_location(
        "homola_replication",
        PROJECT_ROOT / "scripts" / "02_homola_replication.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Load and bin sunspot data
# ---------------------------------------------------------------------------

def _load_sunspots(
    sidc_path: Path,
    study_start: str,
    study_end: str,
    bin_days: int,
    bin_origin: pd.Timestamp,
) -> np.ndarray | None:
    """
    Load SIDC daily sunspot numbers, bin to *bin_days*, return aligned array.

    Returns None if the file is missing or empty.
    """
    if not sidc_path.exists():
        logger.warning("Sunspot file not found: %s", sidc_path)
        return None

    df = pd.read_csv(sidc_path, comment="#")
    # Normalise column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Identify date column (may be 'date' or first column)
    date_col = "date" if "date" in df.columns else df.columns[0]
    # Convert to datetime robustly (avoid ArrowDtype issues)
    df[date_col] = pd.to_datetime(df[date_col].astype(str).str.strip(), errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col)
    df.index = pd.DatetimeIndex(df.index)

    # Find the sunspot total column
    for candidate in ("total", "ssn", "sunspot", "value"):
        if candidate in df.columns:
            ss_col = candidate
            break
    else:
        ss_col = df.columns[0]
        logger.warning("Guessing sunspot column as %r", ss_col)

    # Parse as numeric — strip whitespace first
    ss = pd.to_numeric(df[ss_col].astype(str).str.strip(), errors="coerce")
    ss = ss.dropna()

    # Restrict to study window
    t_start = pd.Timestamp(study_start)
    t_end   = pd.Timestamp(study_end)
    ss = ss[(ss.index >= t_start) & (ss.index <= t_end)]
    if ss.empty:
        logger.warning("No sunspot data in study window")
        return None

    # Bin onto the same 5-day grid as the seismic / CR series
    # using the same floor-division approach as _bin_series in homola script
    days_from_origin = (ss.index - bin_origin).days
    bin_number = days_from_origin // bin_days
    bin_dates  = bin_origin + pd.to_timedelta(bin_number * bin_days, unit="D")
    ss_binned  = ss.groupby(bin_dates).mean()

    return ss_binned


def _align_sunspots(
    ss_binned: pd.Series,
    ref_index: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Reindex sunspot series onto *ref_index*, filling gaps with interpolation
    then linear extrapolation (edge fill).  Returns float64 array.
    """
    aligned = ss_binned.reindex(ref_index)
    aligned = aligned.interpolate(method="linear", limit_direction="both")
    aligned = aligned.ffill().bfill()
    return aligned.to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# Naive Pearson significance
# ---------------------------------------------------------------------------

def _naive_pearson(r: float, n: float) -> tuple[float, float]:
    if n < 3 or not np.isfinite(r) or abs(r) >= 1.0:
        return np.nan, np.nan
    t = r * np.sqrt((n - 2) / (1.0 - r ** 2))
    p = float(2.0 * scipy.stats.t.sf(abs(t), df=n - 2))
    return p, p_to_sigma(p)


# ---------------------------------------------------------------------------
# Run surrogate test for one (cr, seismic) pair
# ---------------------------------------------------------------------------

def _run_test(
    cr: np.ndarray,
    seismic: np.ndarray,
    lags_bins: np.ndarray,
    bin_days: int,
    n_surrogates: int,
    n_jobs: int,
    iaaft_iter: int,
    seed: int,
    label: str,
) -> dict:
    """Run IAAFT surrogate test and return a summary dict."""
    logger.info("Running IAAFT surrogate test: %s  (N=%d, %d surrogates)",
                label, len(cr), n_surrogates)

    result = surrogate_xcorr_test(
        cr, seismic, lags_bins,
        n_surrogates=n_surrogates,
        method="iaaft",
        seed=seed,
        n_jobs=n_jobs,
        iaaft_n_iter=iaaft_iter,   # surrogate_xcorr_test param name
    )

    obs_r     = result["observed_r"]
    peak_r    = float(result["observed_peak_r"])
    peak_lag  = int(result["observed_peak_lag"]) * bin_days  # convert to days
    p_global  = float(result["p_global"])

    n     = len(cr)
    n_eff = n_eff_bretherton(cr, seismic)
    r_15  = float(obs_r[np.searchsorted(lags_bins, 15 // bin_days)])
    _, sigma_15_naive = _naive_pearson(r_15, n)
    _, sigma_15_breth = _naive_pearson(r_15, n_eff)
    _, sigma_pk_naive = _naive_pearson(peak_r, n)
    sigma_iaaft       = p_to_sigma(p_global, n_trials=n_surrogates)

    return {
        "label":          label,
        "n":              n,
        "n_eff":          round(float(n_eff), 1),
        "r_at_15d":       round(r_15, 4),
        "sigma_15_naive": round(float(sigma_15_naive), 2) if np.isfinite(sigma_15_naive) else None,
        "sigma_15_breth": round(float(sigma_15_breth), 2) if np.isfinite(sigma_15_breth) else None,
        "peak_r":         round(peak_r, 4),
        "peak_lag_days":  peak_lag,
        "sigma_pk_naive": round(float(sigma_pk_naive), 2) if np.isfinite(sigma_pk_naive) else None,
        "p_global_iaaft": round(p_global, 4),
        "sigma_iaaft":    round(sigma_iaaft, 2),
        # arrays for figure
        "_obs_r":         obs_r,
        "_surr_arrays":   result["surrogate_r_arrays"],
        "_lags_bins":     lags_bins,
    }


# ---------------------------------------------------------------------------
# 2×2 figure
# ---------------------------------------------------------------------------

PANEL_LABELS = ["(a) Raw", "(b) HP filter", "(c) STL", "(d) Sunspot regression"]
PANEL_COLORS = ["#1b7837", "#762a83", "#e08214", "#2166ac"]

def _make_figure(
    results: list[dict],
    bin_days: int,
    study_start: str,
    study_end: str,
    n_stations: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=False, sharex=True)
    fig.suptitle(
        f"CR–Seismic Cross-Correlation: Raw vs Detrended  "
        f"({study_start[:4]}–{study_end[:4]}, {n_stations} CR stations, {bin_days}-day bins)",
        fontsize=13, fontweight="bold",
    )

    for ax, res, label, color in zip(axes.flat, results, PANEL_LABELS, PANEL_COLORS):
        lags_days = res["_lags_bins"] * bin_days
        obs_r     = res["_obs_r"]
        surr      = res["_surr_arrays"]  # (n_surr, n_lags)

        # Percentile envelopes
        p95 = np.percentile(surr, 97.5, axis=0)
        p05 = np.percentile(surr,  2.5, axis=0)
        p99 = np.percentile(surr, 99.5, axis=0)
        p01 = np.percentile(surr,  0.5, axis=0)

        ax.fill_between(lags_days, p01, p99, alpha=0.15, color=color, label="99% envelope")
        ax.fill_between(lags_days, p05, p95, alpha=0.30, color=color, label="95% envelope")
        ax.plot(lags_days, obs_r, color=color, linewidth=1.5, label="Observed r(τ)")

        # τ = +15 d marker
        ax.axvline(15, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label="τ = +15 d")
        ax.axhline(0,  color="black", linewidth=0.5)

        # Annotations
        sig_str  = f"p_global={res['p_global_iaaft']:.3f} ({res['sigma_iaaft']:.1f}σ IAAFT)"
        peak_str = f"peak: τ={res['peak_lag_days']}d, r={res['peak_r']:.3f}"
        r15_str  = f"r(+15d)={res['r_at_15d']:.3f} ({res.get('sigma_15_breth') or 'N/A'}σ Breth.)"
        ax.set_title(f"{label}\n{peak_str}  |  {r15_str}", fontsize=9)
        ax.text(0.02, 0.98, sig_str, transform=ax.transAxes,
                fontsize=8, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

        ax.set_xlim(lags_days[0], lags_days[-1])
        ax.grid(True, alpha=0.3)

    for ax in axes[1, :]:
        ax.set_xlabel("Lag τ (days)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Pearson r")

    # Shared legend from first panel
    handles, lbls = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s", output_path)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _write_report(
    results: list[dict],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Markdown table
    header = (
        "| Method | N_eff | r(+15d) | σ_Breth(15d) | Peak r | Peak lag | "
        "p_global (IAAFT) | σ_IAAFT |"
    )
    sep = "|---|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for r in results:
        rows.append(
            f"| {r['label']} | {r['n_eff']:.0f} | {r['r_at_15d']:.4f} | "
            f"{r['sigma_15_breth'] or 'N/A'} | {r['peak_r']:.4f} | "
            f"{r['peak_lag_days']} d | {r['p_global_iaaft']:.4f} | "
            f"{r['sigma_iaaft']:.1f}σ |"
        )
    table = "\n".join(rows)

    # Interpret surviving peaks
    survived = [r for r in results if r["p_global_iaaft"] < 0.05]
    if survived:
        interpret = (
            "**CAUTION**: The following variants retain p_global < 0.05 after detrending: "
            + ", ".join(r["label"] for r in survived)
            + ".  Further investigation required."
        )
    else:
        interpret = (
            "**Conclusion**: After all three solar-cycle detrending methods the global "
            "IAAFT p-value is ≥ 0.05 in every case.  The observed cross-correlation is "
            "fully consistent with a shared solar-cycle confound; there is no residual "
            "evidence for a genuine CR–seismic link."
        )

    md = f"""# Detrended CR–Seismic Cross-Correlation Analysis

Generated: {timestamp}
Study period: {args.study_start} – {args.study_end}
Bin size: {args.bin_days} days
Surrogates: {args.n_surrogates} IAAFT
Lag range: {args.lag_min}…{args.lag_max} days

## Significance table

{table}

## Interpretation

{interpret}

## Methods

### HP filter (Hodrick-Prescott)
λ = {args.hp_lambda:.2e} calibrated for {args.bin_days}-day bins targeting removal of
variations longer than ~2000 days (Ravn & Uhlig 2002 scaling of the standard λ=1600).

### STL decomposition
Period = {args.stl_period} bins ≈ {args.stl_period * args.bin_days / 365.25:.1f} years
(11-year solar cycle). seasonal_jump=100, trend_jump=100 for computational efficiency.
The *residual* component (x − trend − seasonal) is used.

### Sunspot OLS regression
Contemporaneous + {args.sunspot_lags}-day lagged sunspot numbers from SIDC WDC-SILSO.
The fitted solar-proxy component is subtracted from each series.

## Figure
`results/figs/detrended_xcorr.png`
"""
    md_path = output_dir / "detrended_report.md"
    md_path.write_text(md)
    logger.info("Report saved: %s", md_path)

    # JSON (strip numpy arrays)
    json_rows = []
    for r in results:
        row = {k: v for k, v in r.items() if not k.startswith("_")}
        json_rows.append(row)
    json_data = {
        "generated": timestamp,
        "study_start": args.study_start,
        "study_end":   args.study_end,
        "bin_days":    args.bin_days,
        "n_surrogates": args.n_surrogates,
        "results":     json_rows,
    }
    json_path = output_dir / "detrended_results.json"
    json_path.write_text(json.dumps(json_data, indent=2))
    logger.info("JSON saved: %s", json_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--study-start",   default="1976-01-01")
    p.add_argument("--study-end",     default="2019-12-31")
    p.add_argument("--bin-days",      type=int,   default=5)
    p.add_argument("--lag-min",       type=int,   default=-1000)
    p.add_argument("--lag-max",       type=int,   default=1000)
    p.add_argument("--min-mag",       type=float, default=4.0)
    p.add_argument("--min-stations",  type=int,   default=3)
    p.add_argument("--n-surrogates",  type=int,   default=10_000)
    p.add_argument("--n-jobs",        type=int,   default=-1)
    p.add_argument("--iaaft-iter",    type=int,   default=100)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--hp-lambda",     type=float, default=1.29e5)
    p.add_argument("--stl-period",    type=int,   default=803,
                   help="STL period in bins (default 803 ≈ 11-yr solar cycle at 5-day bins)")
    p.add_argument("--sunspot-lags",  type=int,   nargs="+", default=[0, 30, 90, 180],
                   metavar="DAYS",
                   help="Sunspot lag offsets in days (default: 0 30 90 180)")
    p.add_argument("--nmdb-dir",  type=Path, default=PROJECT_ROOT/"data"/"raw"/"nmdb")
    p.add_argument("--usgs-dir",  type=Path, default=PROJECT_ROOT/"data"/"raw"/"usgs")
    p.add_argument("--sidc-dir",  type=Path, default=PROJECT_ROOT/"data"/"raw"/"sidc")
    p.add_argument("--config",    type=Path, default=PROJECT_ROOT/"config"/"stations.yaml")
    p.add_argument("--output-dir",type=Path, default=PROJECT_ROOT/"results")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figs").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: build raw CR and seismic series via homola script
    # ------------------------------------------------------------------
    logger.info("Loading raw data via 02_homola_replication …")
    homola = _import_homola()

    import yaml
    with open(args.config) as fh:
        stations_cfg = yaml.safe_load(fh)
    stations_val = stations_cfg["stations"]
    if isinstance(stations_val, dict):
        station_ids = list(stations_val.keys())
    else:
        station_ids = [s["id"] for s in stations_val]

    bin_days = args.bin_days
    t0       = pd.Timestamp(args.study_start)

    cr_5d, station_norm_daily, n_stations = homola.build_cr_index(
        station_ids,
        args.study_start, args.study_end,
        bin_days,
        args.nmdb_dir,
        min_stations=args.min_stations,
    )
    cr_5d = cr_5d.dropna()

    seismic_5d = homola.build_seismic_metric(
        args.study_start, args.study_end,
        bin_days,
        args.usgs_dir,
        min_mag=args.min_mag,
        cr_index=cr_5d,
    )

    # Align on common index
    common_idx = cr_5d.index.intersection(seismic_5d.index)
    cr_5d      = cr_5d.reindex(common_idx)
    seismic_5d = seismic_5d.reindex(common_idx)
    cr_arr      = cr_5d.to_numpy(dtype=np.float64)
    seismic_arr = seismic_5d.to_numpy(dtype=np.float64)
    N           = len(cr_arr)
    logger.info("Common bins: %d  (%s – %s)", N,
                common_idx[0].date(), common_idx[-1].date())

    # ------------------------------------------------------------------
    # Step 2: load and align sunspot data
    # ------------------------------------------------------------------
    sidc_path   = args.sidc_dir / "sunspots.csv"
    ss_binned   = _load_sunspots(sidc_path, args.study_start, args.study_end,
                                 bin_days, t0)
    if ss_binned is not None:
        ss_arr = _align_sunspots(ss_binned, common_idx)
        logger.info("Sunspot series loaded, len=%d, NaN=%.0f%%",
                    len(ss_arr), 100 * np.mean(np.isnan(ss_arr)))
    else:
        ss_arr = None
        logger.warning("Sunspot data unavailable — sunspot-regression panel will be skipped")

    # ------------------------------------------------------------------
    # Step 3: build lags array (bins)
    # ------------------------------------------------------------------
    lag_min_b = args.lag_min // bin_days
    lag_max_b = args.lag_max // bin_days
    lags_bins = np.arange(lag_min_b, lag_max_b + 1, dtype=int)

    # ------------------------------------------------------------------
    # Step 4: detrend
    # ------------------------------------------------------------------
    logger.info("Applying detrending methods …")

    # HP filter
    cr_hp      = hp_filter_detrend(cr_arr,      lamb=args.hp_lambda)
    seismic_hp = hp_filter_detrend(seismic_arr, lamb=args.hp_lambda)

    # STL
    period = args.stl_period
    logger.info("  STL: period=%d bins (~%.1f yr)", period, period*bin_days/365.25)
    cr_stl      = stl_detrend(cr_arr,      period=period, seasonal_jump=100, trend_jump=100)
    seismic_stl = stl_detrend(seismic_arr, period=period, seasonal_jump=100, trend_jump=100)

    # Sunspot regression
    if ss_arr is not None and not np.all(np.isnan(ss_arr)):
        cr_ssreg      = sunspot_regression_detrend(
            cr_arr, ss_arr, lag_days=args.sunspot_lags, bin_days=bin_days)
        seismic_ssreg = sunspot_regression_detrend(
            seismic_arr, ss_arr, lag_days=args.sunspot_lags, bin_days=bin_days)
    else:
        cr_ssreg      = None
        seismic_ssreg = None

    # ------------------------------------------------------------------
    # Step 5: surrogate tests on each variant
    # ------------------------------------------------------------------
    seed = args.seed
    kw   = dict(n_surrogates=args.n_surrogates, n_jobs=args.n_jobs,
                iaaft_iter=args.iaaft_iter, seed=seed)

    results = []

    results.append(_run_test(
        cr_arr, seismic_arr, lags_bins, bin_days, label="Raw", **kw))

    results.append(_run_test(
        cr_hp, seismic_hp, lags_bins, bin_days, label="HP filter", **kw))

    results.append(_run_test(
        cr_stl, seismic_stl, lags_bins, bin_days, label="STL", **kw))

    if cr_ssreg is not None:
        results.append(_run_test(
            cr_ssreg, seismic_ssreg, lags_bins, bin_days,
            label="Sunspot regression", **kw))
    else:
        # Placeholder so the figure still has 4 panels
        logger.warning("Sunspot regression skipped — using HP result as placeholder")
        ph = dict(results[1])
        ph["label"] = "Sunspot regression (N/A)"
        results.append(ph)

    # ------------------------------------------------------------------
    # Step 6: print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"{'Method':<25} {'N_eff':>6} {'r(+15d)':>8} {'σ_Breth':>8} "
          f"{'Peak r':>8} {'Peak lag':>9} {'p_IAAFT':>8} {'σ_IAAFT':>8}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['label']:<25} {r['n_eff']:>6.0f} {r['r_at_15d']:>8.4f} "
            f"{str(r['sigma_15_breth'] or 'N/A'):>8} "
            f"{r['peak_r']:>8.4f} {r['peak_lag_days']:>8}d "
            f"{r['p_global_iaaft']:>8.4f} {r['sigma_iaaft']:>7.1f}σ"
        )
    print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # Step 7: figure
    # ------------------------------------------------------------------
    _make_figure(
        results, bin_days, args.study_start, args.study_end, n_stations,
        args.output_dir / "figs" / "detrended_xcorr.png",
    )

    # ------------------------------------------------------------------
    # Step 8: report + JSON
    # ------------------------------------------------------------------
    _write_report(results, args.output_dir, args)

    logger.info("Done.")


if __name__ == "__main__":
    run(_parse_args())
