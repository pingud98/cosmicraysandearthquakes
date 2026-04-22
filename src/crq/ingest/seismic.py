"""
src/crq/ingest/seismic.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Convert the USGS earthquake catalogue into analysis-ready time series.

Public API
----------
seismic_moment(mw)              — M₀ = 10^(1.5·Mw + 9.1) N·m (Hanks & Kanamori 1979)
assign_depth_stratum(depth_km)  — shallow / intermediate / deep categorical
assign_grid_cell(lat, lon)      — 10°×10° bin corners (floor)
build_global_daily_moment()     — sum(M₀) per day, log10, gapless
build_global_daily_magnitude()  — sum(Mw) per day, gapless (Homola metric)
build_regional_daily_moment()   — 10°×10° grid, sum(M₀) per cell per day
build_depth_stratified_daily()  — three depth strata, sum(M₀) per day
process_catalogue()             — entry point: filter → enrich → write four parquets
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
MIN_MAGNITUDE: float = 4.5          # global catalogue completeness threshold since ~1976
CELL_DEG: float = 10.0              # 10°×10° lat/lon grid

DEPTH_BINS: list[float]   = [-np.inf, 70.0, 300.0, np.inf]
DEPTH_LABELS: list[str]   = ["shallow", "intermediate", "deep"]   # <70, 70-300, >300 km

DepthStratum = Literal["shallow", "intermediate", "deep"]


# ---------------------------------------------------------------------------
# Core physics
# ---------------------------------------------------------------------------

def seismic_moment(mw: "float | np.ndarray | pd.Series") -> "float | np.ndarray":
    """
    Convert moment magnitude Mw to scalar seismic moment M₀ in N·m.

    Formula (Hanks & Kanamori 1979):
        M₀ = 10^(1.5 · Mw + 9.1)  N·m

    Examples
    --------
    >>> seismic_moment(9.0)   # ~3.98e22 N·m
    3.981071705534969e+22
    >>> seismic_moment(9.1)   # ~5.62e22 N·m  (USGS Tohoku value)
    5.623413251903491e+22
    """
    return np.power(10.0, 1.5 * np.asarray(mw, dtype=float) + 9.1)


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def assign_depth_stratum(depth_km: pd.Series) -> pd.Categorical:
    """
    Classify focal depths into three standard seismological strata.

    Returns
    -------
    pd.Categorical with ordered categories ["shallow", "intermediate", "deep"].
    NaN depth → NaN category.

    Boundaries (km)
    ---------------
    shallow       < 70
    intermediate  70 – 300
    deep          > 300
    """
    return pd.cut(
        depth_km,
        bins=DEPTH_BINS,
        labels=DEPTH_LABELS,
        right=False,      # bins: [-inf, 70), [70, 300), [300, inf)
    )


def assign_grid_cell(
    lat: "pd.Series | np.ndarray",
    lon: "pd.Series | np.ndarray",
    cell_deg: float = CELL_DEG,
) -> tuple[pd.Series, pd.Series]:
    """
    Assign each event to the lower-left corner of its *cell_deg* × *cell_deg* grid cell.

    Returns (lat_bin, lon_bin) as int Series, e.g. lat=37.5 → lat_bin=30 for cell_deg=10.

    Conventions
    -----------
    * lat_bin ∈ [-90, -80, …, 80]  (18 rows)
    * lon_bin ∈ [-180, -170, …, 170] (36 cols)
    * Total 648 cells at 10°×10°
    """
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)
    lat_bin = (np.floor(lat_arr / cell_deg) * cell_deg).astype(int)
    lon_bin = (np.floor(lon_arr / cell_deg) * cell_deg).astype(int)
    return pd.Series(lat_bin, index=getattr(lat, "index", None), name="lat_bin"), \
           pd.Series(lon_bin, index=getattr(lon, "index", None), name="lon_bin")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _enrich(events: pd.DataFrame, min_magnitude: float) -> pd.DataFrame:
    """
    Filter events and attach derived columns.

    Adds: ``moment``, ``depth_stratum``, ``lat_bin``, ``lon_bin``, ``date``.
    """
    ev = events.copy()
    ev = ev.dropna(subset=["mag", "latitude", "longitude"])
    ev = ev[ev["mag"] >= min_magnitude]

    ev["moment"]        = seismic_moment(ev["mag"].values)
    ev["depth_stratum"] = assign_depth_stratum(ev["depth"])
    ev["lat_bin"], ev["lon_bin"] = assign_grid_cell(ev["latitude"], ev["longitude"])
    ev.index = pd.to_datetime(ev.index).normalize()   # floor to midnight UTC
    ev.index.name = "date"
    return ev


def _log10_safe(s: pd.Series) -> pd.Series:
    """log10 of a non-negative Series; returns NaN where s == 0."""
    out = s.copy().astype(float)
    out[out == 0] = np.nan
    return np.log10(out)


def _make_date_range(events: pd.DataFrame) -> pd.DatetimeIndex:
    """Gapless daily date range covering the full event catalogue."""
    return pd.date_range(
        start=events.index.min(),
        end=events.index.max(),
        freq="D",
        name="date",
    )


# ---------------------------------------------------------------------------
# Aggregation builders
# ---------------------------------------------------------------------------

def build_global_daily_moment(
    events: pd.DataFrame,
    date_range: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Sum scalar seismic moment M₀ across all global events per calendar day.

    Parameters
    ----------
    events:
        Enriched DataFrame produced by :func:`_enrich`.
    date_range:
        Complete gapless DatetimeIndex (typically from :func:`_make_date_range`).

    Returns
    -------
    DataFrame indexed by ``date`` with columns:

    * ``log10_moment_sum`` — log₁₀ of the daily M₀ sum; NaN for zero-event days
    * ``event_count``      — integer, 0 for zero-event days
    """
    agg = (
        events["moment"]
        .groupby(level="date")
        .agg(moment_sum="sum", event_count="count")
    )
    agg = agg.reindex(date_range, fill_value=0)
    agg["log10_moment_sum"] = _log10_safe(agg["moment_sum"])
    return agg[["log10_moment_sum", "event_count"]]


def build_global_daily_magnitude(
    events: pd.DataFrame,
    date_range: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Sum raw moment magnitudes Mw per calendar day (Homola et al. 2023 metric).

    Returns
    -------
    DataFrame with columns ``magnitude_sum`` (0.0 for no-event days) and
    ``event_count``.
    """
    agg = (
        events["mag"]
        .groupby(level="date")
        .agg(magnitude_sum="sum", event_count="count")
    )
    agg = agg.reindex(date_range, fill_value=0)
    return agg[["magnitude_sum", "event_count"]]


def build_regional_daily_moment(
    events: pd.DataFrame,
    date_range: pd.DatetimeIndex,
    cell_deg: float = CELL_DEG,
) -> pd.DataFrame:
    """
    Sum M₀ per 10°×10° grid cell per calendar day.

    Returns
    -------
    DataFrame with a three-level MultiIndex ``(date, lat_bin, lon_bin)`` and
    columns ``log10_moment_sum`` and ``event_count``.

    Zero-fill policy
    ----------------
    Every date in *date_range* appears for each grid cell that had at least
    one event in the entire catalogue.  Cells that never recorded an event are
    omitted entirely.
    """
    agg = (
        events
        .groupby([pd.Grouper(level="date", freq="D"), "lat_bin", "lon_bin"])["moment"]
        .agg(moment_sum="sum", event_count="count")
    )
    # Identify cells that ever had events
    active_cells = agg.index.droplevel("date").unique()  # MultiIndex (lat_bin, lon_bin)
    lat_bins = sorted(active_cells.get_level_values("lat_bin").unique())
    lon_bins = sorted(active_cells.get_level_values("lon_bin").unique())

    # Unstack to (date × lat_bin × lon_bin) cube, fill, then stack back
    unstacked = (
        agg["moment_sum"]
        .unstack(["lat_bin", "lon_bin"])         # date rows, (lat, lon) columns
        .reindex(date_range)                     # insert missing dates
    )
    count_unstacked = (
        agg["event_count"]
        .unstack(["lat_bin", "lon_bin"])
        .reindex(date_range, fill_value=0)
        .fillna(0)
        .astype(int)
    )
    unstacked_filled = unstacked.fillna(0.0)

    # Remove cells that appear in the Cartesian product but never had events
    # (artifacts of the unstack if lat_bins or lon_bins don't form a full grid)
    active_col_mask = count_unstacked.sum(axis=0) > 0
    count_unstacked = count_unstacked.loc[:, active_col_mask]
    unstacked_filled = unstacked_filled.loc[:, active_col_mask]

    moment_long   = unstacked_filled.stack(["lat_bin", "lon_bin"], future_stack=True)
    count_long    = count_unstacked.stack(["lat_bin", "lon_bin"], future_stack=True)

    result = pd.DataFrame({
        "log10_moment_sum": _log10_safe(moment_long),
        "event_count": count_long,
    })
    result.index.names = ["date", "lat_bin", "lon_bin"]
    return result


def build_depth_stratified_daily(
    events: pd.DataFrame,
    date_range: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Sum M₀ per depth stratum per calendar day.

    Returns
    -------
    DataFrame with MultiIndex ``(date, depth_stratum)`` — all three strata appear
    for every date, zero-filled when no events of that type occurred.
    Columns: ``log10_moment_sum``, ``event_count``.
    """
    ev = events.dropna(subset=["depth_stratum"])
    agg = (
        ev
        .groupby([pd.Grouper(level="date", freq="D"), "depth_stratum"])["moment"]
        .agg(moment_sum="sum", event_count="count")
    )
    # Unstack strata → date rows, 3 stratum columns
    moment_wide = agg["moment_sum"].unstack("depth_stratum").reindex(date_range).fillna(0.0)
    count_wide  = agg["event_count"].unstack("depth_stratum").reindex(date_range, fill_value=0).fillna(0).astype(int)

    # Ensure all three strata are present even if some never appeared
    for stratum in DEPTH_LABELS:
        if stratum not in moment_wide.columns:
            moment_wide[stratum] = 0.0
            count_wide[stratum]  = 0

    moment_wide = moment_wide[DEPTH_LABELS]
    count_wide  = count_wide[DEPTH_LABELS]

    moment_long = moment_wide.stack("depth_stratum", future_stack=True)
    count_long  = count_wide.stack("depth_stratum", future_stack=True)

    result = pd.DataFrame({
        "log10_moment_sum": _log10_safe(moment_long),
        "event_count": count_long,
    })
    result.index.names = ["date", "depth_stratum"]
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_catalogue(
    events: pd.DataFrame,
    output_dir: Path,
    min_magnitude: float = MIN_MAGNITUDE,
    cell_deg: float = CELL_DEG,
) -> dict[str, Path]:
    """
    Filter the USGS catalogue, compute seismic moment, and write four parquet files.

    Parameters
    ----------
    events:
        Raw DataFrame loaded by :func:`crq.ingest.usgs.load_usgs`.
    output_dir:
        Directory where parquet files are written.
    min_magnitude:
        Events below this Mw are dropped.
    cell_deg:
        Grid cell size in degrees for the regional output.

    Returns
    -------
    Dict mapping logical name → output path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Filtering catalogue: %d raw events, M ≥ %.1f", len(events), min_magnitude)
    ev = _enrich(events, min_magnitude)
    logger.info("Retained %d events after filter", len(ev))

    date_range = _make_date_range(ev)
    logger.info("Date range: %s → %s (%d days)", date_range[0].date(), date_range[-1].date(), len(date_range))

    outputs: dict[str, Path] = {}

    # 1. Global daily moment sum
    logger.info("Building global daily moment sum …")
    gm = build_global_daily_moment(ev, date_range)
    p = output_dir / "global_daily_moment_sum.parquet"
    gm.to_parquet(p, engine="pyarrow")
    outputs["global_daily_moment_sum"] = p
    logger.info("  wrote %s  (%d rows)", p.name, len(gm))

    # 2. Global daily magnitude sum (Homola metric)
    logger.info("Building global daily magnitude sum …")
    gmag = build_global_daily_magnitude(ev, date_range)
    p = output_dir / "global_daily_magnitude_sum.parquet"
    gmag.to_parquet(p, engine="pyarrow")
    outputs["global_daily_magnitude_sum"] = p
    logger.info("  wrote %s  (%d rows)", p.name, len(gmag))

    # 3. Regional 10°×10°
    logger.info("Building regional daily moment (10°×10° grid) …")
    reg = build_regional_daily_moment(ev, date_range, cell_deg=cell_deg)
    p = output_dir / "regional_daily_moment.parquet"
    reg.to_parquet(p, engine="pyarrow")
    outputs["regional_daily_moment"] = p
    logger.info("  wrote %s  (%d rows, %d active cells)",
                p.name, len(reg),
                len(reg.index.droplevel("date").unique()))

    # 4. Depth-stratified
    logger.info("Building depth-stratified daily moment …")
    dep = build_depth_stratified_daily(ev, date_range)
    p = output_dir / "depth_stratified_daily_moment.parquet"
    dep.to_parquet(p, engine="pyarrow")
    outputs["depth_stratified_daily_moment"] = p
    logger.info("  wrote %s  (%d rows)", p.name, len(dep))

    return outputs
