"""
src/crq/ingest/usgs.py
~~~~~~~~~~~~~~~~~~~~~~
Download and parse USGS earthquake catalogue via the FDSN event web service.

The notebook used local files named ``siesmic-YYYY.csv`` (note the original
typo); this module downloads from the live API and stores files as
``usgs-YYYY.csv`` in *raw_dir*/usgs/.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_USGS_API = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Columns we need (positions 0-4 in the USGS CSV)
_USGS_COLS = ["time", "latitude", "longitude", "depth", "mag"]


def _usgs_url(year: int, min_magnitude: float) -> str:
    return (
        f"{_USGS_API}?format=csv"
        f"&starttime={year}-01-01"
        f"&endtime={year}-12-31"
        f"&minmagnitude={min_magnitude}"
        f"&orderby=time"
    )


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_year(
    year: int,
    out_dir: Path,
    *,
    min_magnitude: float = 4.5,
    sleep_s: float = 2.0,
    timeout_s: float = 120.0,
    retries: int = 3,
) -> Path:
    """
    Download one year of USGS events (M ≥ *min_magnitude*) to *out_dir*.

    Idempotent: skips download if the file already exists and is non-empty.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"usgs-{year}.csv"

    if dest.exists() and dest.stat().st_size > 0:
        logger.debug("skip %s (already exists)", dest)
        return dest

    url = _usgs_url(year, min_magnitude)
    logger.info("GET %s", url)

    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout_s)
            resp.raise_for_status()
            dest.write_text(resp.text, encoding="utf-8")
            break
        except Exception as exc:
            last_exc = exc
            logger.warning("attempt %d/%d failed for USGS %d: %s", attempt, retries, year, exc)
            if attempt < retries:
                time.sleep(sleep_s * attempt)
    else:
        raise RuntimeError(f"All {retries} USGS download attempts failed for {year}") from last_exc

    time.sleep(sleep_s)
    return dest


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_usgs_csv(path: Path) -> pd.DataFrame:
    """
    Parse a USGS FDSN CSV file.

    Returns a DataFrame indexed by UTC datetime with columns:
    latitude, longitude, depth, mag (all float).
    """
    df = pd.read_csv(
        path,
        usecols=range(5),
        names=_USGS_COLS,
        header=0,
        parse_dates=["time"],
        index_col="time",
        dtype={"latitude": str, "longitude": str, "depth": str, "mag": str},
    )
    for col in ("latitude", "longitude", "depth", "mag"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove timezone info to match NMDB index
    df.index = df.index.tz_convert(None)
    df.sort_index(inplace=True)
    return df


def load_usgs(
    start_year: int,
    end_year: int,
    raw_dir: Path,
) -> pd.DataFrame:
    """Load and concatenate all yearly USGS files."""
    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        path = Path(raw_dir) / f"usgs-{year}.csv"
        if not path.exists():
            logger.debug("missing USGS file: %s", path)
            continue
        try:
            frames.append(parse_usgs_csv(path))
        except Exception as exc:
            logger.warning("parse error %s: %s", path, exc)

    if not frames:
        return pd.DataFrame(columns=_USGS_COLS[1:])
    return pd.concat(frames).sort_index()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def seismic_energy_per_bin(
    events: pd.DataFrame,
    study_start: str,
    study_end: str,
    bin_days: int,
    t0: "pd.Timestamp | None" = None,
    *,
    min_mag: float = 4.5,
) -> pd.Series:
    """
    Physically correct seismic energy metric per *bin_days*-day bin.

    Each earthquake contributes its radiated energy
        E_i = 10^(1.5 · Mw_i + 4.8)  [joules]
    (Kanamori 1977).  Values are summed over all events in the bin and the
    result is returned as log10(E_bin_sum).  Empty bins → NaN.

    Parameters
    ----------
    events      : DataFrame with DatetimeIndex and a "mag" column (from load_usgs).
    study_start : ISO date string (inclusive).
    study_end   : ISO date string (inclusive).
    bin_days    : Bin width in days.
    t0          : Anchor timestamp for floor-division binning; defaults to
                  pd.Timestamp(study_start).
    min_mag     : Minimum magnitude threshold.

    Returns
    -------
    pd.Series with DatetimeIndex of bin start-dates and float64 values
    (log10 of summed seismic energy in joules).
    """
    if t0 is None:
        t0 = pd.Timestamp(study_start)

    ev = events.loc[study_start:study_end]
    ev = ev[ev["mag"] >= min_mag].copy()

    ev["energy"] = np.power(10.0, 1.5 * ev["mag"].values + 4.8)

    # Daily sum of energy
    daily_e = ev["energy"].resample("1D").sum()
    full_day_idx = pd.date_range(study_start, study_end, freq="D")
    daily_e = daily_e.reindex(full_day_idx, fill_value=0.0)

    # Floor-division binning anchored at t0
    days_from_t0 = (daily_e.index - t0).days
    bin_num = days_from_t0 // bin_days
    bin_dates = t0 + pd.to_timedelta(bin_num * bin_days, unit="D")
    bin_energy = daily_e.groupby(bin_dates).sum()

    # log10; empty bins → NaN
    result = np.log10(bin_energy.where(bin_energy > 0, other=np.nan))
    result.name = "log10_seismic_energy_J"
    return result


def compute_daily_seismic(
    events: pd.DataFrame,
    interval: str = "1D",
    origin: str = "1960-01-01",
) -> pd.DataFrame:
    """
    .. deprecated::
        This function applied a dB-domain average to Mw values, which is
        physically invalid (Mw is already logarithmic so 10·log10(mean(10^(Mw/10)))
        does not represent energy).  Use :func:`seismic_energy_per_bin` instead.

    Retained for backwards compatibility with existing callers only.
    """
    mag = events["mag"].copy()

    def _energy_log_mean(s: pd.Series) -> float:
        arr = s.dropna().values
        if arr.size == 0:
            return np.nan
        # Correct: sum seismic energy, return log10
        energies = np.power(10.0, 1.5 * arr + 4.8)
        return float(np.log10(np.sum(energies)))

    daily = mag.resample(interval).apply(_energy_log_mean)
    return daily.rename("mag").to_frame()
