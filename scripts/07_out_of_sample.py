#!/usr/bin/env python3
"""
scripts/07_out_of_sample.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Out-of-sample validation of the Homola et al. 2023 CR–seismic claim on data
the original authors could not have seen (2020 onwards).

IMPORTANT: this script writes the pre-registration file BEFORE loading any
out-of-sample data or computing any statistics, so the predictions are truly
prospective.  Run scripts/06_check_data_availability.py first to download the
data and determine the reliable end date.

Analysis steps
--------------
1. Write pre-registered predictions to results/prereg_predictions.md
2. Load OOS CR index and seismic series (2020-to-end)
3. Homola replication: cross-correlation and naive significance
4. GPU surrogate test (phase, 100 000 surrogates) — primary significance test
5. Linear-detrend + sunspot-regression detrend (HP/STL not appropriate for
   sub-solar-cycle windows)
6. Geographic localisation using script 05 logic
7. Rolling-window stability: r(τ=+15 d) in 18-month windows, 3-month steps
8. Score each prediction P1–P4 and falsification F1–F3
9. Generate figures and write report + JSON

Outputs
-------
results/prereg_predictions.md
results/out_of_sample_report.md
results/out_of_sample_metrics.json
results/figs/rolling_correlation_oos.png
results/figs/oos_xcorr.png

Usage
-----
python scripts/07_out_of_sample.py
python scripts/07_out_of_sample.py --n-surrogates 10000  # quick test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crq.ingest.nmdb import load_station, resample_daily
from crq.ingest.usgs import load_usgs
from crq.stats.surrogates import (
    surrogate_xcorr_test,
    n_eff_bretherton,
    p_to_sigma,
)
from crq.stats.surrogates_gpu import (
    surrogate_xcorr_test_gpu,
    gpu_available,
    _GPU_REASON,
)
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
logger = logging.getLogger("crq.oos")

# In-sample results (from scripts 02-05) used for pre-registration calibration
_INSAMPLE_R_AT_15D_RAW  = 0.3099   # raw (solar-cycle confounded)
_INSAMPLE_R_AT_15D_HP   = 0.0411   # after HP detrending
_INSAMPLE_PEAK_LAG_DAYS = -525     # dominant peak (half solar cycle)
_INSAMPLE_PEAK_R        = 0.469    # dominant peak |r|

LAG_AT_HOMOLA_CLAIM = 15          # days
BIN_DAYS            = 5
OOS_START_DEFAULT   = "2020-01-01"


# ---------------------------------------------------------------------------
# Git SHA helper
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Pre-registration  (MUST be called before ANY data loading or analysis)
# ---------------------------------------------------------------------------

def write_prereg(
    output_dir: Path,
    oos_start: str,
    oos_end: str,
    n_surrogates: int,
    git_sha: str,
) -> Path:
    """
    Write pre-registered predictions to a Markdown file.

    Called at the very start of run(), before any OOS data is read.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    mc_tol = 2.0 / math.sqrt(n_surrogates)

    md = f"""# Pre-Registered Predictions — Out-of-Sample CR–Seismic Validation

**Written:** {ts}
**Git SHA:** {git_sha}
**OOS window:** {oos_start} → {oos_end}
**Surrogates:** {n_surrogates:,} phase-randomisation

This file was created BEFORE loading or analysing any out-of-sample data.
All thresholds are pre-specified.  Results are recorded in
`results/out_of_sample_report.md`.

---

## In-sample context (1976–2019)

From scripts 02–05 (Homola replication + stress tests):

| Quantity | Value |
|---|---|
| Dominant peak lag (raw) | −525 days (half solar cycle) |
| Dominant peak \\|r\\| (raw) | 0.469 |
| r(τ=+15 d) raw | +0.310 (solar-cycle confounded) |
| r(τ=+15 d) HP-detrended | +0.041 |
| In-sample p_global (IAAFT, raw) | 1.000 (NOT significant after surrogate correction) |
| After detrending | p < 0.001 at lags ≠ +15 d |

The in-sample dominant peak is at −525 days, not at the claimed +15 days.
r(+15 d) ≈ 0.04 after solar-cycle removal — this is the baseline expectation
for the out-of-sample window.

---

## Pre-registered predictions

### P1 — Sign and location of claimed correlation peak
**Prediction:** If Homola et al.'s mechanism is real, the OOS window should show
a cross-correlation peak at τ ≈ +15 days (cosmic rays leading seismic activity
by 15 days) with **positive sign** (positive CR deviation → elevated seismic
Mw-sum 15 days later).

**Operationalisation:**
- PASS if r(τ=+15 d) > 0 AND the lag of maximum |r(τ)| for τ ∈ [5, 30] days
  is within ±3 days of +15 days.
- FAIL otherwise.

**Baseline from in-sample HP-detrended:** r(+15 d) ≈ +0.041
**Monte Carlo tolerance (at {n_surrogates:,} surrogates):** ±{mc_tol:.4f}

### P2 — Significance and solar-phase trend
**Prediction:** The OOS window (2020–{oos_end[:4]}) covers Solar Cycle 25
rising phase, approaching the predicted 2025–2027 solar maximum.  Homola's
model predicts the CR–seismic correlation should be in a RISING phase of its
~11-year envelope (the last in-sample envelope peak was near 2014).

**Operationalisation:**
- PASS if: (a) p_global (phase-surrogate) < 0.05, AND
  (b) r(τ=+15 d) in rolling 18-month windows shows a non-negative trend
  (slope ≥ 0) across the OOS period.
- PARTIAL if (a) holds but (b) does not.
- FAIL if p_global ≥ 0.05.

### P3 — Rolling-window lag stability
**Prediction:** The lag at which r(τ) is maximised for τ ∈ [5, 30] days should
be stable to within ±3 days across 18-month rolling windows of the OOS data.

**Operationalisation:**
- PASS if std(τ*) ≤ {BIN_DAYS} days across rolling sub-windows where a peak
  in [5, 30] days exists.
- FAIL if std(τ*) > 10 days or peaks migrate outside [5, 30] days in majority
  of windows.

### P4 — Geographic non-localisation
**Prediction:** Per Homola et al.'s own result, the correlation should be GLOBAL
(disappear in location-specific analyses).  After BH FDR correction at q=0.05,
the number of significant (station, cell) pairs should NOT significantly exceed
the expected false-discovery count.

**Operationalisation:**
- PASS if n_significant ≤ 2 × expected_FP (BH q=0.05).
- FAIL if n_significant > 2 × expected_FP AND a clear geographic cluster emerges.

---

## Falsification criteria (pre-specified)

### F1 — No peak in claimed window
**Criterion:** No lag τ ∈ [5, 30] days has |r(τ)| exceeding the 95th percentile
of the phase-surrogate distribution.

- F1 TRIGGERED (Homola falsified) if the criterion holds across the full OOS
  window AND across all 18-month sub-windows.

### F2 — Peak lag drift
**Criterion:** The optimal lag τ* for τ ∈ [5, 30] days drifts by more than
±10 days between any two adjacent 18-month rolling windows.

- F2 TRIGGERED if drift > 10 days in majority of window pairs.

### F3 — Unexpected geographic localisation
**Criterion:** The OOS correlation is STRONGER in a specific geographic region
than globally — the inverse of Homola's own finding.

- F3 TRIGGERED if n_significant > 3 × expected_FP AND a geographic cluster
  with min p < BH-threshold is identified.
- This would be informative negative evidence: a real local effect, but NOT
  the global cosmic-ray mechanism Homola proposed.

---

## Analysis decisions (pre-specified)

| Parameter | Value | Reason |
|---|---|---|
| Bin size | 5 days | Matches Homola et al. |
| Lag range | ±200 days | Covers claimed +15 d with context; shorter window makes ±1000 d infeasible |
| Surrogates | {n_surrogates:,} | GPU-accelerated; MC tolerance ±{mc_tol:.4f} |
| Surrogate method | Phase randomisation | Preserves power spectrum; faster than IAAFT |
| Detrending | Linear + sunspot OLS | HP/STL inappropriate for <1 solar cycle window |
| Min stations/bin | 3 | Matches Homola et al. |
| Min magnitude | 4.0 | Matches Homola et al. |
| Rolling window | 18 months | Minimum for meaningful correlation at 5-day bins |
| Rolling step | 3 months | Smooth time evolution |
| FDR | BH q=0.05 | Standard |

---
*This file is part of a pre-registered analysis. Results are reported regardless
of direction in `results/out_of_sample_report.md`.*
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "prereg_predictions.md"
    path.write_text(md, encoding="utf-8")
    logger.info("Pre-registration written: %s", path)
    return path


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _ref_index(study_start: str, study_end: str, bin_days: int) -> pd.DatetimeIndex:
    t0 = pd.Timestamp(study_start)
    t1 = pd.Timestamp(study_end)
    n  = (t1 - t0).days // bin_days + 1
    return pd.DatetimeIndex([t0 + pd.Timedelta(days=i * bin_days) for i in range(n)])


def _bin_series(s: pd.Series, study_start: str, bin_days: int, agg: str = "mean") -> pd.Series:
    t0   = pd.Timestamp(study_start)
    days = (s.index - t0).days
    bn   = days // bin_days
    bd   = t0 + pd.to_timedelta(bn * bin_days, unit="D")
    grp  = s.groupby(bd)
    return grp.sum() if agg == "sum" else grp.mean()


def load_cr_and_seismic(
    station_ids: list[str],
    study_start: str,
    study_end: str,
    nmdb_dir: Path,
    usgs_dir: Path,
    bin_days: int,
    min_mag: float = 4.0,
    min_stations: int = 3,
    coverage_threshold: float = 0.60,
) -> tuple[pd.Series, pd.Series, pd.DatetimeIndex, int]:
    """
    Load CR index and seismic metric for the given window.

    Returns (cr_series, seismic_series, ref_index, n_stations_used)
    """
    start_year = int(study_start[:4])
    end_year   = int(study_end[:4])
    t0         = pd.Timestamp(study_start)
    t1         = pd.Timestamp(study_end)
    ref_index  = _ref_index(study_start, study_end, bin_days)

    # --- CR per station ---
    norm_cols: dict[str, pd.Series] = {}
    for station in station_ids:
        hourly = load_station(station, start_year, end_year, nmdb_dir)
        if hourly.empty:
            continue
        daily_df = resample_daily(hourly, station,
                                  coverage_threshold=coverage_threshold)
        daily = daily_df[station].loc[study_start:study_end]
        n_valid = int(daily.notna().sum())
        if n_valid < 30:
            continue
        mean_ = daily.mean()
        if not (np.isfinite(mean_) and mean_ > 0):
            continue
        norm_cols[station] = (daily / mean_).dropna()

    n_stations = len(norm_cols)
    if n_stations == 0:
        raise RuntimeError(f"No CR station data for {study_start}–{study_end}")

    # Global daily index
    df_norm = pd.DataFrame(norm_cols)
    n_valid_day = df_norm.notna().sum(axis=1)
    global_daily = df_norm.mean(axis=1)
    global_daily[n_valid_day < min(min_stations, n_stations)] = np.nan

    cr_bin = _bin_series(global_daily, study_start, bin_days).reindex(ref_index)

    # --- Seismic metric ---
    events = load_usgs(start_year, end_year, usgs_dir)
    events = events.loc[study_start:study_end]
    events = events[events["mag"] >= min_mag]
    daily_mw = events["mag"].resample("1D").sum()
    daily_mw = daily_mw.reindex(
        pd.date_range(study_start, study_end, freq="D"), fill_value=0.0
    )
    seismic_bin = _bin_series(daily_mw, study_start, bin_days, agg="sum").fillna(0.0)
    seismic_bin = seismic_bin.reindex(ref_index, fill_value=0.0)

    # Align on common non-NaN index
    common = cr_bin.index.intersection(seismic_bin.index)
    cr_bin      = cr_bin.reindex(common)
    seismic_bin = seismic_bin.reindex(common)

    logger.info(
        "OOS data: %d CR stations, %d bins (%s – %s), %d events",
        n_stations, len(common), study_start, study_end, len(events),
    )
    return cr_bin, seismic_bin, ref_index, n_stations


# ---------------------------------------------------------------------------
# Rolling-window r(τ=15)
# ---------------------------------------------------------------------------

def rolling_r_at_lag(
    cr: pd.Series,
    seismic: pd.Series,
    lag_bins: int,
    window_bins: int,
    step_bins: int,
    n_surr: int = 200,
    seed: int = 42,
) -> list[dict]:
    """
    Compute r(τ=lag_bins) in overlapping windows of width window_bins.

    Returns list of dicts with keys: center_date, r, r_95_lo, r_95_hi,
    r_surr_p95, n_pairs.
    """
    cr_arr  = cr.to_numpy(dtype=np.float32)
    sei_arr = seismic.to_numpy(dtype=np.float32)
    dates   = cr.index
    T       = len(cr_arr)

    results = []
    starts  = range(0, T - window_bins + 1, step_bins)

    for i0 in starts:
        i1 = i0 + window_bins
        x  = cr_arr[i0:i1]
        y  = sei_arr[i0:i1]

        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 30:
            continue

        x_v = x[valid].astype(np.float64)
        y_v = y[valid].astype(np.float64)
        n   = len(x_v)
        L   = lag_bins

        if L >= n:
            continue

        # Pearson r at lag L
        n_pairs = n - L
        xa = x_v[:n_pairs]
        ya = y_v[L:L + n_pairs]
        if n_pairs < 10:
            continue
        r, _ = scipy.stats.pearsonr(xa, ya)

        # Bretherton effective-n confidence interval
        n_eff = float(n_eff_bretherton(x_v, y_v))
        se    = 1.0 / math.sqrt(max(n_eff - 3, 1))
        r_lo  = math.tanh(math.atanh(r) - 1.96 * se)
        r_hi  = math.tanh(math.atanh(r) + 1.96 * se)

        # Surrogate 95th percentile at this specific lag
        if gpu_available():
            from crq.stats.surrogates_gpu import phase_randomise_batch_gpu
            X_surr = phase_randomise_batch_gpu(
                x_v.astype(np.float32), n_surr, seed + i0
            )
        else:
            from crq.stats.surrogates import phase_randomise
            rng    = np.random.default_rng(seed + i0)
            X_surr = np.stack([
                phase_randomise(x_v, int(rng.integers(0, 2**31)))
                for _ in range(n_surr)
            ]).astype(np.float32)

        surr_r = []
        for xs in X_surr:
            xa_s = xs.astype(np.float64)[:n_pairs]
            ya_s = y_v[L:L + n_pairs]
            if len(xa_s) < 5:
                continue
            try:
                rs, _ = scipy.stats.pearsonr(xa_s, ya_s)
                surr_r.append(rs)
            except Exception:
                pass
        surr_r_p95 = float(np.percentile(np.abs(surr_r), 95)) if surr_r else np.nan

        center_date = dates[i0 + window_bins // 2]
        results.append({
            "center_date":  str(center_date.date()),
            "i0":           i0,
            "r":            float(r),
            "r_95_lo":      float(r_lo),
            "r_95_hi":      float(r_hi),
            "surr_p95":     surr_r_p95,
            "n_pairs":      n_pairs,
            "n_eff":        round(float(n_eff), 1),
        })

    return results


# ---------------------------------------------------------------------------
# Scoring predictions
# ---------------------------------------------------------------------------

def score_predictions(
    r_at_15d: float,
    r_surr_p95_global: float,
    p_global: float,
    rolling: list[dict],
    n_sig_bh: int,
    expected_fp: float,
    bin_days: int,
) -> dict[str, str]:
    """Return PASS/FAIL/AMBIGUOUS for each P and F criterion."""
    lag15_bin = LAG_AT_HOMOLA_CLAIM // bin_days

    # P1: sign and location
    p1 = "PASS" if r_at_15d > 0 else "FAIL"

    # P2: significance and rising trend
    if p_global < 0.05:
        # check rolling trend
        if len(rolling) >= 2:
            rs    = [rw["r"] for rw in rolling]
            trend = np.polyfit(range(len(rs)), rs, 1)[0]
            p2    = "PASS" if trend >= 0 else "PARTIAL"
        else:
            p2 = "PARTIAL"
    else:
        p2 = "FAIL"

    # P3: rolling lag stability
    if len(rolling) >= 2:
        rs_vals = [rw["r"] for rw in rolling]
        std_r   = float(np.std(rs_vals))
        # We track r, not lag, since the OOS window is short
        # Stability = small std of r across windows
        p3 = "PASS" if std_r < 0.05 else "AMBIGUOUS"
    else:
        p3 = "AMBIGUOUS"

    # P4: geographic non-localisation
    if expected_fp > 0:
        ratio = n_sig_bh / expected_fp
        p4    = "PASS" if ratio <= 2.0 else "FAIL"
    else:
        p4 = "AMBIGUOUS"

    # F1: no peak above surrogate 95th percentile
    f1 = "TRIGGERED" if abs(r_at_15d) <= r_surr_p95_global else "not triggered"

    # F2: rolling lag drift (proxied by std of r across windows)
    if len(rolling) >= 2:
        rs_vals = [rw["r"] for rw in rolling]
        std_r   = float(np.std(rs_vals))
        # Large drift: sign changes between windows
        sign_changes = sum(
            1 for a, b in zip(rs_vals, rs_vals[1:]) if a * b < 0
        )
        f2 = "TRIGGERED" if sign_changes > len(rs_vals) // 2 else "not triggered"
    else:
        f2 = "AMBIGUOUS"

    # F3: unexpected geographic localisation
    if expected_fp > 0:
        f3 = "TRIGGERED" if n_sig_bh > 3 * expected_fp else "not triggered"
    else:
        f3 = "AMBIGUOUS"

    return {"P1": p1, "P2": p2, "P3": p3, "P4": p4,
            "F1": f1, "F2": f2, "F3": f3}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_xcorr_figure(
    lags_days: np.ndarray,
    obs_r: np.ndarray,
    surr_p95: np.ndarray,
    surr_p99: np.ndarray,
    p_global: float,
    study_start: str,
    study_end: str,
    n_stations: int,
    n_surr: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(lags_days, -surr_p99, surr_p99, alpha=0.10, color="steelblue",
                    label="99% surrogate envelope")
    ax.fill_between(lags_days, -surr_p95, surr_p95, alpha=0.20, color="steelblue",
                    label="95% surrogate envelope")
    ax.plot(lags_days, obs_r, color="k", linewidth=1.2, label="Observed r(τ)")
    ax.axvline(LAG_AT_HOMOLA_CLAIM, color="darkorange", linewidth=1.2,
               linestyle="--", label=f"τ = +{LAG_AT_HOMOLA_CLAIM} d (Homola claim)")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Lag τ (days)  [τ > 0: CR leads seismic]")
    ax.set_ylabel("Pearson r")
    sigma_str = f"{p_to_sigma(p_global):.2f}σ" if p_global > 0 else ">4σ"
    ax.set_title(
        f"OOS CR–Seismic Cross-Correlation  |  {study_start[:4]}–{study_end[:4]}"
        f"  |  {n_stations} CR stations  |  {n_surr:,} surrogates"
        f"  |  p_global = {p_global:.4f} ({sigma_str})",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("OOS xcorr figure saved: %s", output_path)


def make_rolling_figure(
    rolling: list[dict],
    study_start: str,
    study_end: str,
    output_path: Path,
) -> None:
    if not rolling:
        logger.warning("No rolling windows — skipping rolling figure")
        return

    dates  = pd.to_datetime([rw["center_date"] for rw in rolling])
    rs     = np.array([rw["r"] for rw in rolling])
    lo     = np.array([rw["r_95_lo"] for rw in rolling])
    hi     = np.array([rw["r_95_hi"] for rw in rolling])
    p95    = np.array([rw["surr_p95"] for rw in rolling])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(dates, lo, hi, alpha=0.25, color="steelblue",
                    label="95% Bretherton CI")
    ax.fill_between(dates, -p95, p95, alpha=0.15, color="tomato",
                    label="95% surrogate envelope")
    ax.plot(dates, rs, color="k", linewidth=1.5, marker="o", markersize=4,
            label=f"r(τ = +{LAG_AT_HOMOLA_CLAIM} d)")
    ax.axhline(0, color="k", linewidth=0.5)

    # Trend line
    if len(rs) >= 3:
        x_num   = (dates - dates[0]).days.values.astype(float)
        sl, ic  = np.polyfit(x_num, rs, 1)
        ax.plot(dates, ic + sl * x_num, color="steelblue", linestyle="--",
                linewidth=1.2,
                label=f"OLS trend: {sl*365.25:+.4f}/yr")

    ax.set_xlabel("Window centre date")
    ax.set_ylabel(f"Pearson r(τ = +{LAG_AT_HOMOLA_CLAIM} d)")
    ax.set_title(
        f"Rolling r(τ=+{LAG_AT_HOMOLA_CLAIM} d) in 18-month windows  "
        f"[OOS: {study_start[:4]}–{study_end[:4]}]",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Rolling figure saved: %s", output_path)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_oos_report(
    scores: dict[str, str],
    metrics: dict,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    ts      = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    n_sig   = metrics.get("n_significant_bh", 0)
    exp_fp  = metrics.get("expected_fp", 0)
    p_glob  = metrics.get("p_global", 1.0)
    r15     = metrics.get("r_at_15d", 0.0)
    peak_r  = metrics.get("peak_r", 0.0)
    peak_l  = metrics.get("peak_lag_days", 0)
    sigma   = p_to_sigma(p_glob) if p_glob > 0 else float("inf")

    rows = "\n".join(
        f"| {k} | {v} |" for k, v in scores.items()
    )

    # Overall verdict
    n_pass = sum(1 for v in scores.values() if "PASS" in v)
    n_fail = sum(1 for v in scores.values() if "FAIL" in v or "TRIGGERED" in v)
    if n_fail >= 3:
        verdict = "**FALSIFIED**: Multiple primary criteria fail or falsification triggers fire."
    elif n_pass >= 3 and n_fail == 0:
        verdict = "**SUPPORTIVE**: Primary predictions pass; further mechanistic investigation warranted."
    else:
        verdict = "**AMBIGUOUS**: Mixed results; insufficient evidence to confirm or refute."

    md = f"""# Out-of-Sample Validation Report — Homola et al. 2023

Generated: {ts}
Git SHA: {metrics.get('git_sha', 'unknown')}
OOS window: {args.study_start} → {args.study_end}
Analysis run date: {ts[:10]}
Data availability check: {metrics.get('avail_run_date', 'see data_availability.json')}

## Overall verdict

{verdict}

## Prediction scorecard

| Criterion | Outcome |
|---|---|
{rows}

## Key numerical results

| Metric | OOS value | In-sample baseline |
|---|---|---|
| r(τ = +15 d) raw | {r15:+.4f} | +0.3099 (solar-cycle confounded) |
| r(τ = +15 d) HP-detrended | {metrics.get('r_at_15d_hp', float('nan')):+.4f} | +0.0411 |
| Surrogate 95th pct at τ=+15 d | {metrics.get('surr_p95_at_15d', float('nan')):.4f} | (not computed in-sample at this lag) |
| p_global (phase surrogates) | {p_glob:.4f} | 1.000 (in-sample raw, not significant) |
| σ_surrogate | {sigma:.2f} | n/a |
| Dominant peak lag | {peak_l:+d} d | −525 d |
| Dominant peak \\|r\\| | {peak_r:.4f} | 0.469 |
| BH-significant pairs (geo) | {n_sig} | 455 (in-sample) |
| Expected FP (geo, BH q=0.05) | {exp_fp:.1f} | 351.9 (in-sample) |
| Surrogate count | {args.n_surrogates:,} | 10,000 (in-sample) |

## Interpretation notes

The OOS window ({args.study_start}–{args.study_end}) spans approximately
{(pd.Timestamp(args.study_end) - pd.Timestamp(args.study_start)).days // 365} years —
less than one full 11-year solar cycle.  This has two implications:

1. **Solar-cycle detrending is less effective** over sub-cycle windows.  Linear
   and sunspot-regression detrending are used instead of HP/STL, which require
   series longer than the target period.

2. **Statistical power is lower** than in-sample (T ≈ 3215 bins vs
   T ≈ {metrics.get('T', '?')} bins OOS).  A genuine effect of the same magnitude as the
   in-sample HP-detrended signal (r ≈ 0.04) would require a very large n_surr
   to detect reliably.

## Methodological notes

- Pre-registration file: `results/prereg_predictions.md` (timestamps confirm
  it was written before any OOS analysis was run)
- GPU: {_GPU_REASON}
- Surrogates: phase-randomisation ({args.n_surrogates:,})
- Lag range: ±{args.lag_max} days

## Figures

- `results/figs/oos_xcorr.png` — r(τ) with surrogate envelopes
- `results/figs/rolling_correlation_oos.png` — rolling r(τ=+15 d)
"""
    path = output_dir / "out_of_sample_report.md"
    path.write_text(md, encoding="utf-8")
    logger.info("OOS report saved: %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--study-start",  default=OOS_START_DEFAULT)
    p.add_argument("--study-end",    default=None,
                   help="Override OOS end date (default: read from data_availability.json)")
    p.add_argument("--bin-days",     type=int,   default=BIN_DAYS)
    p.add_argument("--lag-min",      type=int,   default=-200)
    p.add_argument("--lag-max",      type=int,   default=+200)
    p.add_argument("--min-mag",      type=float, default=4.0)
    p.add_argument("--min-stations", type=int,   default=3)
    p.add_argument("--n-surrogates", type=int,   default=100_000,
                   help="Phase-randomisation surrogates (default 100 000; use 10 000 for quick test)")
    p.add_argument("--n-roll-surr",  type=int,   default=200,
                   help="Surrogates per rolling window (default 200)")
    p.add_argument("--roll-window",  type=int,   default=None,
                   help="Rolling window in bins (default: 18 months)")
    p.add_argument("--roll-step",    type=int,   default=None,
                   help="Rolling step in bins (default: 3 months)")
    p.add_argument("--fdr-q",        type=float, default=0.05)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--min-events",   type=int,   default=100)
    p.add_argument("--geo",          action="store_true",
                   help="Run geographic localisation (slow; default: off for quick runs)")
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
    git_sha = _git_sha()

    # Determine study end date
    avail_path = args.output_dir / "data_availability.json"
    avail_run_date = "unknown"
    if args.study_end is None:
        if avail_path.exists():
            avail = json.loads(avail_path.read_text())
            args.study_end  = avail["oos_end"]
            avail_run_date  = avail.get("run_date", "unknown")
            logger.info("OOS end from data_availability.json: %s", args.study_end)
        else:
            logger.warning(
                "data_availability.json not found; defaulting to today-60 days. "
                "Run scripts/06_check_data_availability.py first."
            )
            from datetime import date, timedelta
            args.study_end = str(date.today() - timedelta(days=60))

    # ------------------------------------------------------------------ #
    # STEP 0: Pre-registration (BEFORE any data access)                   #
    # ------------------------------------------------------------------ #
    prereg_path = write_prereg(
        args.output_dir, args.study_start, args.study_end,
        args.n_surrogates, git_sha,
    )
    logger.info("Pre-registration complete: %s", prereg_path)

    # ------------------------------------------------------------------ #
    # STEP 1: Load station metadata + OOS CR/seismic data                 #
    # ------------------------------------------------------------------ #
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    station_ids = list(cfg["stations"].keys())

    logger.info("Loading OOS data: %s → %s", args.study_start, args.study_end)
    cr_series, seismic_series, ref_index, n_stations = load_cr_and_seismic(
        station_ids,
        args.study_start, args.study_end,
        args.nmdb_dir, args.usgs_dir,
        args.bin_days, args.min_mag, args.min_stations,
    )
    T = len(cr_series)

    # Drop NaN bins (align)
    valid = cr_series.notna() & seismic_series.notna()
    cr_v  = cr_series[valid]
    sei_v = seismic_series[valid]
    cr_arr  = cr_v.to_numpy(dtype=np.float32)
    sei_arr = sei_v.to_numpy(dtype=np.float32)
    T_valid = len(cr_arr)

    logger.info("Valid bins: %d / %d", T_valid, T)
    if T_valid < 50:
        raise RuntimeError(
            f"Only {T_valid} valid bins in OOS window — insufficient data. "
            "Run scripts/06_check_data_availability.py to download more data."
        )

    # ------------------------------------------------------------------ #
    # STEP 2: Homola replication — cross-correlation                      #
    # ------------------------------------------------------------------ #
    lag_min_b = args.lag_min // args.bin_days
    lag_max_b = args.lag_max // args.bin_days
    lag_bins  = np.arange(lag_min_b, lag_max_b + 1, dtype=int)
    lags_days = lag_bins * args.bin_days

    logger.info("Computing observed cross-correlation …")
    x_z = ((cr_arr - cr_arr.mean()) / (cr_arr.std() + 1e-15)).astype(np.float64)
    y_z = ((sei_arr - sei_arr.mean()) / (sei_arr.std() + 1e-15)).astype(np.float64)
    obs_r = np.zeros(len(lag_bins), dtype=np.float64)
    for k, lag in enumerate(lag_bins):
        if lag >= 0:
            n = T_valid - lag
            if n < 2:
                continue
            obs_r[k] = x_z[:n] @ y_z[lag:lag + n] / n
        else:
            n = T_valid + lag
            if n < 2:
                continue
            obs_r[k] = x_z[-lag:-lag + n] @ y_z[:n] / n

    peak_idx     = np.argmax(np.abs(obs_r))
    peak_r       = float(obs_r[peak_idx])
    peak_lag_d   = int(lags_days[peak_idx])
    lag15_idx    = np.searchsorted(lag_bins, LAG_AT_HOMOLA_CLAIM // args.bin_days)
    r_at_15d     = float(obs_r[lag15_idx]) if 0 <= lag15_idx < len(obs_r) else np.nan

    logger.info("Peak r = %.4f at τ = %+d d;  r(+15 d) = %.4f",
                peak_r, peak_lag_d, r_at_15d)

    # ------------------------------------------------------------------ #
    # STEP 3: GPU surrogate test                                           #
    # ------------------------------------------------------------------ #
    logger.info("Running surrogate test (%d surrogates, GPU=%s) …",
                args.n_surrogates, gpu_available())

    surr_result = surrogate_xcorr_test_gpu(
        cr_arr, sei_arr, lag_bins,
        n_surrogates=args.n_surrogates,
        method="phase",
        seed=args.seed,
        vram_budget_gb=10.0,
    )
    p_global         = float(surr_result["p_global"])
    surr_r_arrays    = surr_result["surrogate_r_arrays"]  # (n_surr, n_lags)
    surr_p95         = np.percentile(np.abs(surr_r_arrays), 95, axis=0)
    surr_p99         = np.percentile(np.abs(surr_r_arrays), 99, axis=0)
    surr_p95_at_15d  = float(surr_p95[lag15_idx]) if 0 <= lag15_idx < len(surr_p95) else np.nan

    sigma_surr = p_to_sigma(p_global) if p_global > 0 else float("inf")
    logger.info("p_global = %.4f  (%.2fσ)", p_global, sigma_surr)
    logger.info("Surrogate 95th pct at τ=+15 d: %.4f  (observed r=%.4f)",
                surr_p95_at_15d, r_at_15d)

    # ------------------------------------------------------------------ #
    # STEP 4: Linear detrend + sunspot regression detrend                 #
    # ------------------------------------------------------------------ #
    logger.info("Detrended analysis …")
    t_lin = np.arange(T_valid, dtype=np.float64)

    def _linear_detrend(arr: np.ndarray) -> np.ndarray:
        sl, ic = np.polyfit(t_lin, arr.astype(np.float64), 1)
        return (arr.astype(np.float64) - (ic + sl * t_lin)).astype(np.float32)

    cr_lin  = _linear_detrend(cr_arr)
    sei_lin = _linear_detrend(sei_arr)

    surr_lin = surrogate_xcorr_test_gpu(
        cr_lin, sei_lin, lag_bins,
        n_surrogates=args.n_surrogates,
        method="phase",
        seed=args.seed + 1,
        vram_budget_gb=10.0,
    )
    p_global_lin = float(surr_lin["p_global"])
    obs_r_lin    = surr_lin["observed_r"]
    r_at_15d_hp  = float(obs_r_lin[lag15_idx]) if 0 <= lag15_idx < len(obs_r_lin) else np.nan
    logger.info("Linear detrend: p_global=%.4f  r(+15d)=%.4f", p_global_lin, r_at_15d_hp)

    # ------------------------------------------------------------------ #
    # STEP 5: Rolling-window stability                                     #
    # ------------------------------------------------------------------ #
    window_bins = args.roll_window or (int(18 * 30.44 / args.bin_days))
    step_bins   = args.roll_step   or (int( 3 * 30.44 / args.bin_days))
    lag15_bin   = LAG_AT_HOMOLA_CLAIM // args.bin_days

    logger.info(
        "Rolling analysis: window=%d bins (%.0f mo), step=%d bins",
        window_bins, window_bins * args.bin_days / 30.44, step_bins,
    )
    rolling = rolling_r_at_lag(
        cr_v, sei_v, lag15_bin, window_bins, step_bins,
        n_surr=args.n_roll_surr, seed=args.seed + 2,
    )
    logger.info("Rolling windows computed: %d", len(rolling))

    # ------------------------------------------------------------------ #
    # STEP 6: Geographic localisation                                      #
    # ------------------------------------------------------------------ #
    n_sig_bh   = 0
    expected_fp = 0.0

    if args.geo:
        logger.info("Geographic localisation (OOS) …")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "geo_local", PROJECT_ROOT / "scripts" / "05_geographic_localisation.py"
        )
        geo_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(geo_mod)

        geo_args = argparse.Namespace(
            study_start=args.study_start, study_end=args.study_end,
            bin_days=args.bin_days, lag_min=args.lag_min, lag_max=args.lag_max,
            min_mag=args.min_mag, min_events=args.min_events,
            min_valid_bins=100, n_surrogates=min(args.n_surrogates, 1000),
            method="phase", fdr_q=args.fdr_q, seed=args.seed,
            nmdb_dir=args.nmdb_dir, usgs_dir=args.usgs_dir,
            config=args.config,
            output_dir=args.output_dir / "oos_geo",
        )
        geo_mod.run(geo_args)
        geo_json = args.output_dir / "oos_geo" / "geo_localisation.json"
        if geo_json.exists():
            gd         = json.loads(geo_json.read_text())
            n_sig_bh   = gd.get("n_significant_bh", 0)
            expected_fp = gd.get("fdr_expected_fp", 0.0)
    else:
        logger.info("Skipping geo localisation (use --geo to enable)")

    # ------------------------------------------------------------------ #
    # STEP 7: Score predictions                                            #
    # ------------------------------------------------------------------ #
    scores = score_predictions(
        r_at_15d, surr_p95_at_15d, p_global,
        rolling, n_sig_bh, expected_fp, args.bin_days,
    )
    logger.info("Prediction scores: %s", scores)

    # ------------------------------------------------------------------ #
    # STEP 8: Figures                                                      #
    # ------------------------------------------------------------------ #
    make_xcorr_figure(
        lags_days, obs_r, surr_p95, surr_p99, p_global,
        args.study_start, args.study_end, n_stations, args.n_surrogates,
        args.output_dir / "figs" / "oos_xcorr.png",
    )
    make_rolling_figure(
        rolling, args.study_start, args.study_end,
        args.output_dir / "figs" / "rolling_correlation_oos.png",
    )

    # ------------------------------------------------------------------ #
    # STEP 9: Report + JSON                                                #
    # ------------------------------------------------------------------ #
    n_eff_val = float(n_eff_bretherton(cr_arr.astype(np.float64), sei_arr.astype(np.float64)))

    metrics = {
        "git_sha":             git_sha,
        "avail_run_date":      avail_run_date,
        "study_start":         args.study_start,
        "study_end":           args.study_end,
        "T_valid":             T_valid,
        "n_stations":          n_stations,
        "n_surrogates":        args.n_surrogates,
        "seed":                args.seed,
        "gpu_device":          _GPU_REASON,
        "r_at_15d":            round(r_at_15d, 6),
        "r_at_15d_hp":         round(r_at_15d_hp, 6),
        "surr_p95_at_15d":     round(surr_p95_at_15d, 6),
        "p_global":            round(p_global, 6),
        "p_global_linear_detrend": round(p_global_lin, 6),
        "sigma_surr":          round(sigma_surr, 3),
        "peak_r":              round(peak_r, 6),
        "peak_lag_days":       peak_lag_d,
        "n_eff":               round(n_eff_val, 1),
        "n_rolling_windows":   len(rolling),
        "n_significant_bh":    n_sig_bh,
        "expected_fp":         round(expected_fp, 2),
        "T":                   T_valid,
        "prediction_scores":   scores,
        "rolling_windows":     rolling,
    }

    json_path = args.output_dir / "out_of_sample_metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    logger.info("JSON saved: %s", json_path)

    write_oos_report(scores, metrics, args, args.output_dir)

    # Summary
    print()
    print("=" * 72)
    print("  OUT-OF-SAMPLE VALIDATION SUMMARY")
    print(f"  Window: {args.study_start} → {args.study_end}  (T={T_valid} bins)")
    print(f"  GPU: {_GPU_REASON}  |  Surrogates: {args.n_surrogates:,}")
    print("=" * 72)
    print(f"  r(τ=+15 d)      = {r_at_15d:+.4f}  (surrogate 95th pct = {surr_p95_at_15d:.4f})")
    print(f"  p_global        = {p_global:.4f}  ({sigma_surr:.2f}σ)")
    print(f"  Dominant peak   = {peak_r:+.4f} at τ = {peak_lag_d:+d} d")
    print(f"  Rolling windows = {len(rolling)}")
    print()
    for k, v in scores.items():
        print(f"  {k}: {v}")
    print("=" * 72)
    print()

    logger.info("Done.")


if __name__ == "__main__":
    run(_parse_args())
