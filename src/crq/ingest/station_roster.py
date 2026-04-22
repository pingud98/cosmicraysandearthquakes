"""
src/crq/ingest/station_roster.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Station roster management across study windows.

Three analytic sub-sets for comparative analysis:
  A — stations present in BOTH the in-sample (1976-2019) AND out-of-sample
      (2020-end) windows with ≥ min_coverage fractional coverage in each.
  B — all stations available in each respective window (maximises power).
  C — NEW stations: data coverage only post-2020 (fully independent detectors).

The choice of subset affects how much of the apparent correlation is
attributable to station-selection artefacts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from crq.ingest.nmdb import load_station, resample_daily

logger = logging.getLogger(__name__)

# Station is considered "present" in a window if it has this fraction of
# non-NaN daily bins over that window.
DEFAULT_COVERAGE = 0.50


def probe_station_coverage(
    station_id: str,
    windows: dict[str, tuple[str, str]],
    nmdb_dir: Path,
    coverage_threshold: float = 0.60,
) -> dict[str, float]:
    """
    Return fractional coverage of *station_id* in each named time window.

    Parameters
    ----------
    windows : mapping of window_name -> (study_start, study_end) ISO strings
    coverage_threshold : hourly coverage fraction required to count a day as valid

    Returns
    -------
    dict window_name -> fraction of days in window that are non-NaN
    """
    t_ranges = {
        name: (pd.Timestamp(s), pd.Timestamp(e))
        for name, (s, e) in windows.items()
    }
    all_start = min(t.year for t, _ in t_ranges.values())
    all_end   = max(t.year for _, t in t_ranges.values())

    hourly = load_station(station_id, all_start, all_end, nmdb_dir)
    if hourly.empty:
        return {name: 0.0 for name in windows}

    daily_df = resample_daily(hourly, station_id, coverage_threshold=coverage_threshold)
    daily    = daily_df[station_id]

    result = {}
    for name, (t0, t1) in t_ranges.items():
        window_days  = daily.loc[t0:t1]
        n_total      = (t1 - t0).days + 1
        n_valid      = int(window_days.notna().sum())
        result[name] = n_valid / max(n_total, 1)

    return result


def classify_stations(
    station_ids: list[str],
    coverage_by_station: dict[str, dict[str, float]],
    in_sample_key: str = "in_sample",
    oos_key: str = "out_of_sample",
    min_coverage: float = DEFAULT_COVERAGE,
) -> dict[str, list[str]]:
    """
    Partition stations into categories A, B, C.

    Parameters
    ----------
    coverage_by_station : station_id -> {window_name: coverage_fraction}
    in_sample_key       : key in coverage dicts for the 1976-2019 window
    oos_key             : key in coverage dicts for the 2020-end window

    Returns
    -------
    dict with keys "A", "B_in_sample", "B_oos", "C"
    """
    has_insample = {
        sid for sid in station_ids
        if coverage_by_station.get(sid, {}).get(in_sample_key, 0.0) >= min_coverage
    }
    has_oos = {
        sid for sid in station_ids
        if coverage_by_station.get(sid, {}).get(oos_key, 0.0) >= min_coverage
    }

    return {
        "A":           sorted(has_insample & has_oos),       # both windows
        "B_in_sample": sorted(has_insample),                 # in-sample best
        "B_oos":       sorted(has_oos),                      # OOS best
        "C":           sorted(has_oos - has_insample),       # new stations only
    }


def station_cr_series(
    station_ids: list[str],
    start_year: int,
    end_year: int,
    nmdb_dir: Path,
    study_start: str,
    study_end: str,
    bin_days: int,
    ref_index: pd.DatetimeIndex,
    coverage_threshold: float = 0.60,
    min_valid_bins: int = 30,
) -> dict[str, np.ndarray]:
    """
    Load, normalise, and 5-day-bin per-station CR series.

    Returns mapping station_id -> float32 array aligned to ref_index (NaN
    where station was not operational).
    """
    t0 = pd.Timestamp(study_start)

    def _bin(s: pd.Series) -> pd.Series:
        days = (s.index - t0).days
        bn   = days // bin_days
        bd   = t0 + pd.to_timedelta(bn * bin_days, unit="D")
        return s.groupby(bd).mean()

    out: dict[str, np.ndarray] = {}
    for station in station_ids:
        hourly = load_station(station, start_year, end_year, nmdb_dir)
        if hourly.empty:
            continue
        daily_df = resample_daily(hourly, station, coverage_threshold=coverage_threshold)
        daily    = daily_df[station].loc[study_start:study_end]
        mean_    = daily.mean()
        if not (np.isfinite(mean_) and mean_ > 0):
            continue
        norm   = (daily / mean_).dropna()
        binned = _bin(norm).reindex(ref_index)
        arr    = binned.to_numpy(dtype=np.float32)
        if int(np.isfinite(arr).sum()) < min_valid_bins:
            continue
        out[station] = arr

    return out


def global_cr_index(
    station_series: dict[str, np.ndarray],
    min_stations: int = 3,
) -> np.ndarray:
    """
    Mean across available stations per bin (NaN if < min_stations).

    Parameters
    ----------
    station_series : station_id -> (T,) float32 array with possible NaN

    Returns
    -------
    (T,) float64 global CR index
    """
    if not station_series:
        raise ValueError("No station series provided")
    T = next(iter(station_series.values())).shape[0]
    mat = np.full((len(station_series), T), np.nan, dtype=np.float64)
    for i, arr in enumerate(station_series.values()):
        mat[i] = arr.astype(np.float64)

    n_valid = np.isfinite(mat).sum(axis=0)
    with np.errstate(all="ignore"):
        mean_ = np.nanmean(mat, axis=0)
    mean_[n_valid < min_stations] = np.nan
    return mean_
