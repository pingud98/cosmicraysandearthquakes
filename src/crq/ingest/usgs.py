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

def compute_daily_seismic(
    events: pd.DataFrame,
    interval: str = "1D",
    origin: str = "1960-01-01",
) -> pd.DataFrame:
    """
    Resample earthquake events to *interval* using the log-power average of
    magnitude (same formula as the notebook's ``log_avg``).

    NaN days (no events) remain NaN — **not** zero.
    """
    mag = events["mag"].copy()

    def _log_avg(s: pd.Series) -> float:
        arr = s.dropna().values
        if arr.size == 0:
            return np.nan
        return float(10.0 * np.log10(np.mean(np.power(10.0, arr / 10.0))))

    daily = mag.resample(interval).apply(_log_avg)
    return daily.rename("mag").to_frame()
