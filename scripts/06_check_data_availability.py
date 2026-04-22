#!/usr/bin/env python3
"""
scripts/06_check_data_availability.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Determine the most recent date on which all three data sources are reliably
available, then download missing data for the 2020-to-present window.

Data sources
------------
1. NMDB  — hourly pressure-corrected neutron monitor counts.
           Reliable end = last date with ≥ 60% hourly coverage, minus 30 days
           to allow for processing delays.  Flags stations with > 30-day gaps.
2. USGS  — M ≥ 4.5 global earthquake catalogue via FDSN.
           Catalogue is generally complete within ~30 days.
           Reliable end = today − 45 days.
3. SIDC  — SILSO daily sunspot numbers.
           Definitive values: ~6-month lag.  Provisional: ~30-day lag.
           Reliable end (definitive) = today − 180 days.
           This script uses the provisional series with a note, so
           reliable end = today − 30 days.

Common window end = min(NMDB_reliable, USGS_reliable, SIDC_reliable).
Window start is fixed at 2020-01-01 (first date post-Homola study period).

Outputs
-------
results/data_availability.json   — window dates + per-source details
results/data_availability.txt    — human-readable report

Usage
-----
python scripts/06_check_data_availability.py
python scripts/06_check_data_availability.py --no-download   # check only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crq.ingest.nmdb import load_station, resample_daily, download_station_year
from crq.ingest.usgs import download_year as usgs_download_year, load_usgs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("crq.avail")

OOS_START = "2020-01-01"
COVERAGE_THRESHOLD = 0.60
MIN_COVERAGE_FRACTION = 0.50   # station needs 50% valid bins in OOS window
GAP_WARN_DAYS = 30


# ---------------------------------------------------------------------------
# SIDC download
# ---------------------------------------------------------------------------

_SIDC_URL = (
    "https://www.sidc.be/silso/INFO/sndhcsv.php"
)
_SIDC_URL_ALT = (
    "https://www.sidc.be/silso/DATA/SN_d_tot_V2.0.csv"
)


def download_sidc(sidc_dir: Path, timeout: int = 60) -> Path | None:
    """Download SIDC daily total sunspot number (V2.0). Returns path or None."""
    sidc_dir.mkdir(parents=True, exist_ok=True)
    dest = sidc_dir / "sunspots.csv"

    for url in (_SIDC_URL, _SIDC_URL_ALT):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            dest.write_text(resp.text, encoding="utf-8")
            logger.info("SIDC downloaded: %d bytes → %s", len(resp.text), dest)
            return dest
        except Exception as exc:
            logger.warning("SIDC download failed (%s): %s", url, exc)

    # Fall back to any existing file
    if dest.exists() and dest.stat().st_size > 0:
        logger.warning("SIDC download failed — using cached file %s", dest)
        return dest
    return None


def check_sidc(sidc_dir: Path, today: date) -> dict:
    """Parse SIDC file and determine reliable end date."""
    path = sidc_dir / "sunspots.csv"
    if not path.exists():
        return {"status": "missing", "last_date": None, "reliable_end": None}

    try:
        # SIDC V2.0 CSV: year;month;day;fracyear;SSN;std;Nobs;provisional
        df = pd.read_csv(
            path,
            sep=";",
            header=None,
            names=["year", "month", "day", "fracyear", "ssn", "std", "nobs", "prov"],
            comment="#",
            dtype=str,
        )
        df = df[df["year"].str.isnumeric()]
        df["date"] = pd.to_datetime(
            df["year"].str.strip() + "-" + df["month"].str.strip() + "-" + df["day"].str.strip(),
            errors="coerce",
        )
        df = df.dropna(subset=["date"])
        df["ssn"] = pd.to_numeric(df["ssn"].str.strip(), errors="coerce")
        df["prov"] = pd.to_numeric(df["prov"].str.strip(), errors="coerce").fillna(1).astype(int)

        last_date    = df["date"].max().date()
        # Provisional records (prov=1) may be revised; definitive = prov=0
        definitive   = df[df["prov"] == 0]["date"].max().date() if (df["prov"] == 0).any() else None
        # Reliable end: today minus 30 days (provisional is good enough)
        reliable_end = today - timedelta(days=30)

        return {
            "status":          "ok",
            "last_date":       str(last_date),
            "definitive_end":  str(definitive) if definitive else None,
            "reliable_end":    str(min(reliable_end, last_date)),
            "n_records":       len(df),
            "note":            "Using provisional values (prov=1); definitive lag ~6 months",
        }
    except Exception as exc:
        logger.warning("SIDC parse error: %s", exc)
        return {"status": "parse_error", "error": str(exc), "reliable_end": None}


# ---------------------------------------------------------------------------
# NMDB download + check
# ---------------------------------------------------------------------------

def download_nmdb_oos(
    station_ids: list[str],
    nmdb_dir: Path,
    oos_start_year: int,
    oos_end_year: int,
    sleep_between: float = 0.3,
) -> dict[str, list[int]]:
    """
    Download all station-years in [oos_start_year, oos_end_year] that are
    missing from nmdb_dir.  Returns dict station_id -> list of years downloaded.
    """
    downloaded: dict[str, list[int]] = {s: [] for s in station_ids}
    for station in station_ids:
        for year in range(oos_start_year, oos_end_year + 1):
            dest = nmdb_dir / f"{station}{year}.csv"
            if dest.exists() and dest.stat().st_size > 0:
                logger.debug("skip %s %d (exists)", station, year)
                continue
            try:
                download_station_year(station, year, nmdb_dir)
                downloaded[station].append(year)
                time.sleep(sleep_between)
            except Exception as exc:
                logger.warning("NMDB %s %d: %s", station, year, exc)
    return downloaded


def check_nmdb_stations(
    station_ids: list[str],
    nmdb_dir: Path,
    oos_start: str,
    today: date,
) -> dict[str, dict]:
    """
    For each station, determine coverage fraction in OOS window and
    the most recent date with data.
    """
    oos_start_ts = pd.Timestamp(oos_start)
    oos_end_ts   = pd.Timestamp(today.isoformat())
    start_year   = int(oos_start[:4])
    end_year     = today.year

    station_info = {}
    for station in station_ids:
        hourly = load_station(station, start_year, end_year, nmdb_dir)
        if hourly.empty:
            station_info[station] = {
                "status": "no_data",
                "coverage_oos": 0.0,
                "last_date": None,
                "gap_days": None,
            }
            continue

        hourly_oos = hourly.loc[oos_start:]
        if hourly_oos.empty:
            station_info[station] = {
                "status": "no_oos_data",
                "coverage_oos": 0.0,
                "last_date": None,
                "gap_days": None,
            }
            continue

        daily_df  = resample_daily(hourly_oos, station, coverage_threshold=COVERAGE_THRESHOLD)
        daily     = daily_df[station]
        n_total   = (oos_end_ts - oos_start_ts).days + 1
        n_valid   = int(daily.notna().sum())
        coverage  = n_valid / n_total

        last_valid = daily.dropna().index.max().date() if not daily.dropna().empty else None
        gap_days   = (today - last_valid).days if last_valid else None

        station_info[station] = {
            "status":       "ok" if coverage >= MIN_COVERAGE_FRACTION else "low_coverage",
            "coverage_oos": round(coverage, 4),
            "last_date":    str(last_valid) if last_valid else None,
            "gap_days":     gap_days,
            "flag_gap":     gap_days > GAP_WARN_DAYS if gap_days is not None else True,
        }
        logger.info(
            "NMDB %-6s  coverage=%.1f%%  last=%s  gap=%s d",
            station, 100 * coverage,
            last_valid or "N/A",
            gap_days or "N/A",
        )

    return station_info


def nmdb_reliable_end(station_info: dict[str, dict], today: date) -> date:
    """
    NMDB reliable end: median last_date among stations with good coverage,
    minus 30 days.
    """
    last_dates = []
    for info in station_info.values():
        if info.get("coverage_oos", 0) >= MIN_COVERAGE_FRACTION and info.get("last_date"):
            last_dates.append(date.fromisoformat(info["last_date"]))
    if not last_dates:
        return today - timedelta(days=90)
    # Use the 25th percentile to be conservative
    last_dates.sort()
    p25_idx = max(0, len(last_dates) // 4)
    return last_dates[p25_idx] - timedelta(days=30)


# ---------------------------------------------------------------------------
# USGS download + check
# ---------------------------------------------------------------------------

def download_usgs_oos(
    usgs_dir: Path,
    oos_start_year: int,
    oos_end_year: int,
    min_magnitude: float = 4.5,
) -> None:
    """Download missing USGS yearly files for OOS window."""
    for year in range(oos_start_year, oos_end_year + 1):
        dest = usgs_dir / f"usgs-{year}.csv"
        if dest.exists() and dest.stat().st_size > 0:
            logger.debug("USGS %d: skip (exists)", year)
            continue
        try:
            usgs_download_year(year, usgs_dir, min_magnitude=min_magnitude)
            logger.info("USGS %d: downloaded", year)
        except Exception as exc:
            logger.warning("USGS %d: %s", year, exc)


def check_usgs(usgs_dir: Path, today: date, oos_start: str) -> dict:
    """Determine USGS coverage and reliable end date."""
    start_year   = int(oos_start[:4])
    end_year     = today.year
    available    = []
    total_events = 0

    for year in range(start_year, end_year + 1):
        p = usgs_dir / f"usgs-{year}.csv"
        if p.exists() and p.stat().st_size > 0:
            available.append(year)
            try:
                df = pd.read_csv(p, usecols=["time", "mag"])
                total_events += len(df)
            except Exception:
                pass

    reliable_end = today - timedelta(days=45)
    return {
        "status":        "ok" if available else "missing",
        "years_present": available,
        "total_events":  total_events,
        "reliable_end":  str(reliable_end),
        "note":          "Catalogue stability: complete within ~30 days; using today-45 days",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--no-download", action="store_true",
                   help="Skip download attempts; check existing files only")
    p.add_argument("--min-mag",     type=float, default=4.5,
                   help="Minimum magnitude for USGS download (default 4.5)")
    p.add_argument("--nmdb-dir",    type=Path, default=PROJECT_ROOT/"data"/"raw"/"nmdb")
    p.add_argument("--usgs-dir",    type=Path, default=PROJECT_ROOT/"data"/"raw"/"usgs")
    p.add_argument("--sidc-dir",    type=Path, default=PROJECT_ROOT/"data"/"raw"/"sidc")
    p.add_argument("--config",      type=Path, default=PROJECT_ROOT/"config"/"stations.yaml")
    p.add_argument("--output-dir",  type=Path, default=PROJECT_ROOT/"results")
    return p.parse_args()


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    today = date.today()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    station_ids = list(cfg["stations"].keys())
    oos_start_year = int(OOS_START[:4])

    # ------------------------------------------------------------------ #
    # 1. Download missing data                                             #
    # ------------------------------------------------------------------ #
    if not args.no_download:
        logger.info("Downloading missing NMDB OOS data (%d-%d) …", oos_start_year, today.year)
        download_nmdb_oos(station_ids, args.nmdb_dir, oos_start_year, today.year)

        logger.info("Downloading missing USGS OOS data (%d-%d) …", oos_start_year, today.year)
        download_usgs_oos(args.usgs_dir, oos_start_year, today.year, min_magnitude=args.min_mag)

        logger.info("Downloading SIDC sunspot data …")
        download_sidc(args.sidc_dir)
    else:
        logger.info("--no-download: skipping download, checking existing files only")

    # ------------------------------------------------------------------ #
    # 2. Check each source                                                 #
    # ------------------------------------------------------------------ #
    logger.info("Checking NMDB station coverage …")
    nmdb_info = check_nmdb_stations(station_ids, args.nmdb_dir, OOS_START, today)

    good_stations = [
        sid for sid, info in nmdb_info.items()
        if info.get("coverage_oos", 0) >= MIN_COVERAGE_FRACTION
    ]
    flagged = [
        sid for sid, info in nmdb_info.items()
        if info.get("flag_gap") and info.get("coverage_oos", 0) > 0
    ]

    nmdb_end  = nmdb_reliable_end(nmdb_info, today)
    usgs_info = check_usgs(args.usgs_dir, today, OOS_START)
    sidc_info = check_sidc(args.sidc_dir, today)

    usgs_end  = date.fromisoformat(usgs_info["reliable_end"])
    sidc_end  = date.fromisoformat(sidc_info["reliable_end"]) if sidc_info.get("reliable_end") else today - timedelta(days=90)

    common_end = min(nmdb_end, usgs_end, sidc_end)
    constraining = {
        "NMDB":  nmdb_end,
        "USGS":  usgs_end,
        "SIDC":  sidc_end,
    }
    constrained_by = min(constraining, key=constraining.get)

    # ------------------------------------------------------------------ #
    # 3. Print summary                                                     #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 72)
    print("  OUT-OF-SAMPLE DATA AVAILABILITY")
    print(f"  Run date: {today}")
    print("=" * 72)
    print(f"\n  OOS window start: {OOS_START}")
    print(f"  NMDB reliable end:  {nmdb_end}  ({len(good_stations)} stations ≥{MIN_COVERAGE_FRACTION*100:.0f}% coverage)")
    print(f"  USGS reliable end:  {usgs_end}")
    print(f"  SIDC reliable end:  {sidc_end}")
    print(f"\n  *** Common reliable end: {common_end}  (constrained by {constrained_by}) ***")
    print(f"\n  OOS window: {OOS_START} → {common_end}")
    print(f"  Duration: {(date.fromisoformat(str(common_end)) - date.fromisoformat(OOS_START)).days} days")
    print()
    print(f"  NMDB stations with ≥{MIN_COVERAGE_FRACTION*100:.0f}% OOS coverage ({len(good_stations)}):")
    for sid in sorted(good_stations):
        info = nmdb_info[sid]
        flag = "  *** GAP > 30d ***" if info.get("flag_gap") else ""
        print(f"    {sid:<8} coverage={info['coverage_oos']*100:5.1f}%  last={info['last_date']}{flag}")
    if flagged:
        print(f"\n  Stations with >30-day gap (may be offline): {', '.join(sorted(flagged))}")
    print("=" * 72)
    print()

    # ------------------------------------------------------------------ #
    # 4. Save JSON and text report                                         #
    # ------------------------------------------------------------------ #
    payload = {
        "run_date":        str(today),
        "oos_start":       OOS_START,
        "oos_end":         str(common_end),
        "constrained_by":  constrained_by,
        "nmdb_reliable_end":  str(nmdb_end),
        "usgs_reliable_end":  str(usgs_end),
        "sidc_reliable_end":  str(sidc_end),
        "good_stations_oos":  sorted(good_stations),
        "flagged_stations":   sorted(flagged),
        "nmdb_station_detail": {
            sid: {k: v for k, v in info.items() if k != "flag_gap"}
            for sid, info in nmdb_info.items()
        },
        "usgs_detail": usgs_info,
        "sidc_detail":  sidc_info,
    }
    json_path = args.output_dir / "data_availability.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("JSON saved: %s", json_path)

    txt_lines = [
        "OUT-OF-SAMPLE DATA AVAILABILITY REPORT",
        f"Run date: {today}",
        f"OOS window: {OOS_START} → {common_end}  (constrained by {constrained_by})",
        "",
        f"NMDB stations with ≥{MIN_COVERAGE_FRACTION*100:.0f}% OOS coverage: {len(good_stations)}",
        *[
            f"  {sid:<8} coverage={nmdb_info[sid]['coverage_oos']*100:5.1f}%  last={nmdb_info[sid]['last_date']}"
            for sid in sorted(good_stations)
        ],
        "",
        f"USGS: years available = {usgs_info['years_present']}  events = {usgs_info['total_events']:,}",
        f"SIDC: last_date = {sidc_info.get('last_date')}  (provisional note: {sidc_info.get('note', '')})",
    ]
    txt_path = args.output_dir / "data_availability.txt"
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
    logger.info("Text report saved: %s", txt_path)

    return payload


if __name__ == "__main__":
    run(_parse_args())
