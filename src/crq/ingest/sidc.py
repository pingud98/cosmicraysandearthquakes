"""
src/crq/ingest/sidc.py
~~~~~~~~~~~~~~~~~~~~~~
Download and parse daily sunspot numbers from the SIDC / KSO solar monitor.

Original notebook source: http://cesar.kso.ac.at/sunspot_numbers/daily_sn.csv
Semicolon-delimited; columns: time, tot, sn, ss, diff.
"""

from __future__ import annotations

import logging
import time
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_SIDC_URL = "https://cesar.kso.ac.at/sunspot_numbers/daily_sn.csv"
_SIDC_COLS = ["time", "tot", "sn", "ss", "diff"]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_sunspots(
    out_dir: Path,
    *,
    url: str = _SIDC_URL,
    sleep_s: float = 1.0,
    timeout_s: float = 60.0,
    retries: int = 3,
) -> Path:
    """
    Download the daily sunspot CSV to *out_dir*/sunspots.csv.

    Idempotent: skips if the file already exists and is non-empty.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "sunspots.csv"

    if dest.exists() and dest.stat().st_size > 0:
        logger.debug("skip %s (already exists)", dest)
        return dest

    logger.info("GET %s", url)
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": "crq/0.1"})
            resp.raise_for_status()
            dest.write_text(resp.text, encoding="utf-8")
            break
        except Exception as exc:
            last_exc = exc
            logger.warning("attempt %d/%d failed for sunspots: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(sleep_s * attempt)
    else:
        raise RuntimeError("All sunspot download attempts failed") from last_exc

    time.sleep(sleep_s)
    return dest


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_sidc_csv(path: Path) -> pd.DataFrame:
    """
    Parse the SIDC daily sunspot CSV.

    Returns a DataFrame indexed by date with numeric columns:
    tot (int), sn, ss, diff (float).
    """
    df = pd.read_csv(
        path,
        sep=";",
        names=_SIDC_COLS,
        header=0,
        parse_dates=["time"],
        index_col="time",
        dtype={c: str for c in _SIDC_COLS[1:]},
    )
    df["tot"] = pd.to_numeric(df["tot"], errors="coerce")
    for col in ("sn", "ss", "diff"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def load_sunspots(raw_dir: Path) -> pd.DataFrame:
    """Load sunspot data from *raw_dir*/sunspots.csv."""
    path = Path(raw_dir) / "sunspots.csv"
    if not path.exists():
        raise FileNotFoundError(f"Sunspot file not found: {path}")
    return parse_sidc_csv(path)


def resample_daily(df: pd.DataFrame, interval: str = "1D") -> pd.DataFrame:
    """Resample (already daily) sunspot data to *interval* mean."""
    return df.resample(interval).mean()
