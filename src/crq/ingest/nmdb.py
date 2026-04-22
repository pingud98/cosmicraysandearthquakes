"""
src/crq/ingest/nmdb.py
~~~~~~~~~~~~~~~~~~~~~~
Download and parse hourly pressure-corrected neutron monitor data from NMDB.

Bug fixed vs. original notebook: yearly files are concatenated (not merged-and-added),
so NaN rows remain NaN.  A per-station daily coverage fraction is computed before
resampling; station-days below ``coverage_threshold`` are set to NaN.
"""

from __future__ import annotations

import logging
import time
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NMDB API
# ---------------------------------------------------------------------------
_NMDB_BASE = "https://www.nmdb.eu/nest/draw_graph.php"

_NMDB_PARAMS = dict(
    wget=1,
    tabchoice="1h",
    dtype="corr_for_pressure",
    output="ascii",
    date_choice="bydate",
    last_label="days_label",
    tresolution=60,
    force=0,
    yunits=0,
)

# Sentinel values NMDB uses for missing data
_MISSING_SENTINELS: tuple[float, ...] = (-999.9, -9999.0, 0.0)


def _nmdb_url(station: str, year: int) -> str:
    """Build NMDB query URL for a full calendar year."""
    params = {
        **_NMDB_PARAMS,
        "stations[]": station,
        "start_day": 1,
        "start_month": 1,
        "start_year": year,
        "start_hour": 0,
        "start_min": 0,
        "end_day": 31,
        "end_month": 12,
        "end_year": year,
        "end_hour": 23,
        "end_min": 59,
    }
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{_NMDB_BASE}?{qs}"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_station_year(
    station: str,
    year: int,
    out_dir: Path,
    *,
    sleep_s: float = 1.0,
    timeout_s: float = 60.0,
    retries: int = 3,
) -> Path:
    """
    Download one station-year from NMDB and save to *out_dir*.

    Idempotent: if the file already exists and is non-empty it is not re-fetched.

    Returns the path to the saved file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{station}{year}.csv"

    if dest.exists() and dest.stat().st_size > 0:
        logger.debug("skip %s (already exists)", dest)
        return dest

    url = _nmdb_url(station, year)
    logger.info("GET %s", url)

    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout_s)
            resp.raise_for_status()
            text = resp.text
            break
        except Exception as exc:
            last_exc = exc
            logger.warning("attempt %d/%d failed for %s %d: %s", attempt, retries, station, year, exc)
            if attempt < retries:
                time.sleep(sleep_s * attempt)
    else:
        raise RuntimeError(f"All {retries} attempts failed for {station} {year}") from last_exc

    # Strip preamble — keep from the 'start_date_time' header onward
    lines = text.splitlines(keepends=True)
    header_idx = next(
        (i for i, ln in enumerate(lines) if "start_date_time" in ln),
        None,
    )
    if header_idx is None:
        logger.warning("no data header found for %s %d — writing empty file", station, year)
        dest.write_text("")
        return dest

    payload = "".join(lines[header_idx:])
    dest.write_text(payload, encoding="utf-8")
    time.sleep(sleep_s)
    return dest


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_nmdb_csv(path: Path, station: str) -> pd.DataFrame:
    """
    Parse a single NMDB ASCII file (as saved by :func:`download_station_year`).

    Returns a DataFrame indexed by ``start_date_time`` (UTC, tz-naive) with one
    column named *station*.  Sentinel / non-positive values become NaN.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return pd.DataFrame(columns=[station])

    # File starts with the 'start_date_time;STATION' header row
    df = pd.read_csv(
        StringIO(text),
        sep=";",
        header=0,
        names=["start_date_time", station],
        parse_dates=["start_date_time"],
        index_col="start_date_time",
        dtype={station: float},
    )

    # Convert sentinel / zero / negative counts to NaN
    # (zero counts are physically impossible for a running NM)
    df[station] = df[station].where(df[station] > 0, np.nan)

    # Drop duplicate timestamps (can appear at DST transitions in some files)
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Loading + coverage
# ---------------------------------------------------------------------------

def load_station(
    station: str,
    start_year: int,
    end_year: int,
    raw_dir: Path,
) -> pd.DataFrame:
    """
    Concatenate all yearly files for *station* into a single hourly DataFrame.

    Missing years are skipped with a warning.  NaN rows are preserved (no
    fill-with-zero).
    """
    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        path = Path(raw_dir) / f"{station}{year}.csv"
        if not path.exists():
            logger.debug("missing file: %s", path)
            continue
        try:
            df = parse_nmdb_csv(path, station)
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            logger.warning("parse error %s: %s", path, exc)

    if not frames:
        logger.warning("no data loaded for station %s", station)
        return pd.DataFrame(columns=[station])

    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined.sort_index(inplace=True)
    return combined


def compute_daily_coverage(hourly: pd.Series) -> pd.Series:
    """
    Compute the fraction of valid (non-NaN) hourly observations per calendar day.

    Returns a daily Series with values in [0, 1].  A value of 1.0 means all
    24 expected hourly readings were present.
    """
    daily_count = hourly.resample("1D").count()   # counts non-NaN
    return daily_count / 24.0


def resample_daily(
    hourly: pd.DataFrame,
    station: str,
    coverage_threshold: float = 0.60,
) -> pd.DataFrame:
    """
    Resample hourly neutron-monitor data to daily means with NaN propagation.

    Days where the fraction of valid hourly readings is below
    *coverage_threshold* are flagged and their daily mean is set to NaN,
    rather than being silently biased by the available partial data.

    Parameters
    ----------
    hourly:
        DataFrame with a DatetimeIndex and a column named *station*.
    station:
        Column name of the count-rate series.
    coverage_threshold:
        Minimum fraction of valid hours required to keep a daily mean.
        Default 0.60 (≥ 14.4 out of 24 hours).

    Returns
    -------
    DataFrame with columns:

    * ``<station>``          — daily mean (NaN when flagged)
    * ``<station>_coverage`` — fraction of valid hours [0, 1]
    * ``<station>_flagged``  — True where coverage < threshold
    """
    series = hourly[station]
    daily_mean = series.resample("1D").mean()          # NaN if all values are NaN
    coverage = compute_daily_coverage(series)

    flagged = coverage < coverage_threshold
    daily_mean = daily_mean.where(~flagged, np.nan)

    return pd.DataFrame(
        {
            station: daily_mean,
            f"{station}_coverage": coverage,
            f"{station}_flagged": flagged,
        }
    )


# ---------------------------------------------------------------------------
# Convenience: log-power average (matches notebook's log_avg)
# ---------------------------------------------------------------------------

def log_avg(values: "np.ndarray | pd.Series") -> float:
    """
    Power-law (dB-domain) average: 10·log₁₀(mean(10^(x/10))).

    Used for averaging seismic magnitudes and cosmic-ray counts in the
    original notebook.  NaN values are ignored.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan
    return float(10.0 * np.log10(np.mean(np.power(10.0, finite / 10.0))))
