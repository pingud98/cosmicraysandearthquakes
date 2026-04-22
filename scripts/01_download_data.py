#!/usr/bin/env python3
"""
scripts/01_download_data.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Idempotent data downloader for the CRQ project.

Downloads:
  1. NMDB hourly pressure-corrected neutron counts (all stations, year-by-year)
  2. USGS earthquake catalogue (M≥4.5, year-by-year)
  3. SIDC daily sunspot numbers

Files are skipped if they already exist and are non-empty.  Failed downloads
are logged and skipped (the script continues rather than aborting).

Usage
-----
# Full run (1960-2019, all 44 stations) — takes several hours on first run
python scripts/01_download_data.py

# Quick subset: 2 stations, 2 years — useful for CI / smoke-testing
python scripts/01_download_data.py --subset

# Custom range
python scripts/01_download_data.py --start-year 2000 --end-year 2002 --stations OULU THUL
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Ensure project src is importable when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crq.ingest import nmdb as _nmdb
from crq.ingest import usgs as _usgs
from crq.ingest import sidc as _sidc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("crq.download")


def _load_station_names(config_path: Path) -> list[str]:
    with config_path.open() as fh:
        cfg = yaml.safe_load(fh)
    return list(cfg["stations"].keys())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", action="store_true",
                   help="Quick 2-station / 2-year smoke-test (OULU + THUL, 2018-2019)")
    p.add_argument("--start-year", type=int, default=1960)
    p.add_argument("--end-year", type=int, default=2019)
    p.add_argument("--stations", nargs="+", metavar="CODE",
                   help="Override station list (default: all from config/stations.yaml)")
    p.add_argument("--min-magnitude", type=float, default=4.5)
    p.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "stations.yaml")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.subset:
        stations = ["OULU", "THUL"]
        start_year, end_year = 2018, 2019
        logger.info("SUBSET mode: stations=%s years=%d-%d", stations, start_year, end_year)
    else:
        stations = args.stations or _load_station_names(args.config)
        start_year, end_year = args.start_year, args.end_year

    nmdb_dir = args.data_dir / "nmdb"
    usgs_dir = args.data_dir / "usgs"
    sidc_dir = args.data_dir / "sidc"

    # ------------------------------------------------------------------
    # 1. NMDB
    # ------------------------------------------------------------------
    logger.info("=== NMDB download: %d stations × %d years ===",
                len(stations), end_year - start_year + 1)
    nmdb_ok = nmdb_err = 0
    for station in stations:
        for year in range(start_year, end_year + 1):
            try:
                _nmdb.download_station_year(station, year, nmdb_dir)
                nmdb_ok += 1
            except Exception as exc:
                logger.error("NMDB %s %d: %s", station, year, exc)
                nmdb_err += 1
        logger.info("station %s done", station)
    logger.info("NMDB: %d ok, %d errors", nmdb_ok, nmdb_err)

    # ------------------------------------------------------------------
    # 2. USGS
    # ------------------------------------------------------------------
    logger.info("=== USGS download: years %d-%d, M≥%.1f ===",
                start_year, end_year, args.min_magnitude)
    usgs_ok = usgs_err = 0
    for year in range(start_year, end_year + 1):
        try:
            _usgs.download_year(year, usgs_dir, min_magnitude=args.min_magnitude)
            usgs_ok += 1
        except Exception as exc:
            logger.error("USGS %d: %s", year, exc)
            usgs_err += 1
    logger.info("USGS: %d ok, %d errors", usgs_ok, usgs_err)

    # ------------------------------------------------------------------
    # 3. SIDC sunspots
    # ------------------------------------------------------------------
    logger.info("=== SIDC sunspot download ===")
    try:
        _sidc.download_sunspots(sidc_dir)
        logger.info("sunspots ok")
    except Exception as exc:
        logger.error("sunspots: %s", exc)

    logger.info("Download complete.")


if __name__ == "__main__":
    main()
