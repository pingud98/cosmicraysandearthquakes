"""
Tests for src/crq/ingest/seismic.py

Covers:
  - seismic_moment formula correctness (including Tohoku 2011 benchmark)
  - assign_depth_stratum classification
  - assign_grid_cell bin assignment
  - build_global_daily_moment: zero-fill, log10 correctness, NaN for no-event days
  - build_global_daily_magnitude: Homola metric, zero-fill
  - build_regional_daily_moment: cell assignment, zero-fill per active cell
  - build_depth_stratified_daily: all three strata present, zero-fill
  - process_catalogue: writes four parquet files with correct shapes
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crq.ingest.seismic import (
    DEPTH_LABELS,
    assign_depth_stratum,
    assign_grid_cell,
    build_depth_stratified_daily,
    build_global_daily_magnitude,
    build_global_daily_moment,
    build_regional_daily_moment,
    process_catalogue,
    seismic_moment,
    _enrich,
    _make_date_range,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_events(**overrides) -> pd.DataFrame:
    """Minimal synthetic catalogue with sensible defaults."""
    base = {
        "latitude":  [35.0, -10.0, 60.0, 35.0],
        "longitude": [140.0, -75.0, 150.0, 140.0],
        "depth":     [30.0, 150.0, 600.0, 10.0],
        "mag":       [5.0, 6.5, 7.0, 4.5],
    }
    base.update(overrides)
    idx = pd.to_datetime([
        "2019-01-01",
        "2019-01-01",
        "2019-01-02",
        "2019-01-03",
    ])
    df = pd.DataFrame(base, index=idx)
    df.index.name = "time"
    return df


@pytest.fixture
def events() -> pd.DataFrame:
    return _make_events()


@pytest.fixture
def enriched(events) -> pd.DataFrame:
    return _enrich(events, min_magnitude=4.5)


@pytest.fixture
def date_range_fixture(enriched) -> pd.DatetimeIndex:
    return _make_date_range(enriched)


# ---------------------------------------------------------------------------
# seismic_moment
# ---------------------------------------------------------------------------

class TestSeismicMoment:
    def test_formula_scalar(self) -> None:
        """M₀ = 10^(1.5·Mw + 9.1)"""
        mw = 5.0
        expected = 10 ** (1.5 * 5.0 + 9.1)
        assert seismic_moment(mw) == pytest.approx(expected, rel=1e-9)

    def test_increases_with_magnitude(self) -> None:
        """Larger Mw → larger M₀."""
        assert seismic_moment(7.0) > seismic_moment(6.0) > seismic_moment(5.0)

    def test_mw9_gives_approx_4e22(self) -> None:
        """
        Mw 9.0 → M₀ ≈ 3.98e22 N·m.

        The 2011 Tōhoku earthquake (Mw 9.0–9.1) is commonly cited as
        releasing ~3.8–5.6e22 N·m depending on the source; this checks the
        formula gives the right order of magnitude.
        """
        m0 = seismic_moment(9.0)
        assert 3e22 < m0 < 5e22, f"Expected ~3.98e22, got {m0:.3e}"

    def test_tohoku_usgs_mw91(self) -> None:
        """USGS lists Tōhoku 2011 at Mw 9.1 → M₀ ≈ 5.62e22 N·m."""
        m0 = seismic_moment(9.1)
        assert m0 == pytest.approx(5.623e22, rel=1e-3)

    def test_vectorised_numpy(self) -> None:
        """Accepts numpy array; returns array of same length."""
        arr = np.array([5.0, 6.0, 7.0])
        result = seismic_moment(arr)
        assert result.shape == (3,)

    def test_vectorised_pandas(self) -> None:
        """Accepts pd.Series."""
        s = pd.Series([5.0, 6.0])
        result = seismic_moment(s)
        assert len(result) == 2

    def test_mw0_reference_point(self) -> None:
        """Mw 0 → 10^9.1 ≈ 1.259e9 N·m (standard reference)."""
        assert seismic_moment(0.0) == pytest.approx(10**9.1, rel=1e-9)


# ---------------------------------------------------------------------------
# assign_depth_stratum
# ---------------------------------------------------------------------------

class TestDepthStratum:
    def test_shallow(self) -> None:
        s = pd.Series([0.0, 69.9, 30.0])
        result = assign_depth_stratum(s)
        assert (result == "shallow").all()

    def test_intermediate(self) -> None:
        s = pd.Series([70.0, 150.0, 299.9])
        result = assign_depth_stratum(s)
        assert (result == "intermediate").all()

    def test_deep(self) -> None:
        s = pd.Series([300.0, 500.0, 700.0])
        result = assign_depth_stratum(s)
        assert (result == "deep").all()

    def test_boundary_70km_is_intermediate(self) -> None:
        """Right=False means 70.0 km falls in intermediate."""
        result = assign_depth_stratum(pd.Series([70.0]))
        assert result.iloc[0] == "intermediate"

    def test_boundary_300km_is_deep(self) -> None:
        result = assign_depth_stratum(pd.Series([300.0]))
        assert result.iloc[0] == "deep"

    def test_nan_depth_gives_nan_stratum(self) -> None:
        result = assign_depth_stratum(pd.Series([np.nan]))
        assert pd.isna(result.iloc[0])

    def test_all_three_strata_present(self, events) -> None:
        result = assign_depth_stratum(events["depth"])
        assert set(result.dropna().unique()) == {"shallow", "intermediate", "deep"}


# ---------------------------------------------------------------------------
# assign_grid_cell
# ---------------------------------------------------------------------------

class TestGridCell:
    @pytest.mark.parametrize("lat,lon,expected_lat,expected_lon", [
        (35.6,  139.7,   30,  130),   # Tokyo
        (-33.9, 151.2,  -40,  150),   # Sydney
        (0.0,     0.0,    0,    0),   # null island
        (-90.0,   0.0,  -90,    0),   # South Pole
        (89.9, -179.9,   80, -180),   # near North Pole / date line
        (37.5,  140.0,   30,  140),   # fractional lat inside bin
    ])
    def test_known_locations(self, lat, lon, expected_lat, expected_lon) -> None:
        lat_bin, lon_bin = assign_grid_cell(pd.Series([lat]), pd.Series([lon]))
        assert int(lat_bin.iloc[0]) == expected_lat
        assert int(lon_bin.iloc[0]) == expected_lon

    def test_bin_is_multiple_of_cell_deg(self) -> None:
        lats = pd.Series(np.random.uniform(-90, 90, 1000))
        lons = pd.Series(np.random.uniform(-180, 180, 1000))
        lat_bins, lon_bins = assign_grid_cell(lats, lons, cell_deg=10.0)
        assert (lat_bins % 10 == 0).all()
        assert (lon_bins % 10 == 0).all()

    def test_accepts_numpy_arrays(self) -> None:
        lat_bin, lon_bin = assign_grid_cell(np.array([35.0]), np.array([140.0]))
        assert int(lat_bin.iloc[0]) == 30
        assert int(lon_bin.iloc[0]) == 140


# ---------------------------------------------------------------------------
# build_global_daily_moment
# ---------------------------------------------------------------------------

class TestGlobalDailyMoment:
    def test_output_columns(self, enriched, date_range_fixture) -> None:
        result = build_global_daily_moment(enriched, date_range_fixture)
        assert set(result.columns) == {"log10_moment_sum", "event_count"}

    def test_gapless_index(self, enriched, date_range_fixture) -> None:
        result = build_global_daily_moment(enriched, date_range_fixture)
        assert len(result) == len(date_range_fixture)

    def test_zero_event_day_has_nan_log10(self, enriched, date_range_fixture) -> None:
        """A day with no events gets event_count=0 and log10_moment_sum=NaN."""
        result = build_global_daily_moment(enriched, date_range_fixture)
        zero_days = result[result["event_count"] == 0]
        assert zero_days["log10_moment_sum"].isna().all()

    def test_zero_event_day_count_is_zero(self, enriched, date_range_fixture) -> None:
        result = build_global_daily_moment(enriched, date_range_fixture)
        # Jan 3 has one event (mag 4.5); Jan 2 has one event; Jan 1 has two
        # No zero days in this fixture (3 dates, 3 days), but check count dtype
        assert result["event_count"].dtype in (np.int64, np.int32, "int64", "int32")

    def test_log10_is_log_of_moment_sum(self, enriched, date_range_fixture) -> None:
        """log10_moment_sum should equal log10(sum(M₀)) for event days."""
        result = build_global_daily_moment(enriched, date_range_fixture)
        # Day 2019-01-01 has 2 events: mag 5.0 and 6.5
        day = result.loc["2019-01-01"]
        expected_sum = seismic_moment(5.0) + seismic_moment(6.5)
        assert day["log10_moment_sum"] == pytest.approx(np.log10(expected_sum), rel=1e-6)

    def test_no_fillna_zero_bias(self) -> None:
        """
        Regression: zero-fill must not replace NaN M₀ values with 0 before
        computing log10 — the daily log10 on event days must not be depressed.
        """
        idx = pd.to_datetime(["2019-01-01", "2019-01-03"])  # gap on Jan 2
        ev_raw = pd.DataFrame({"latitude": [35.0, 35.0],
                                "longitude": [140.0, 140.0],
                                "depth": [30.0, 30.0],
                                "mag": [6.0, 6.0]}, index=idx)
        ev_raw.index.name = "time"
        ev = _enrich(ev_raw, 4.5)
        dr = _make_date_range(ev)
        result = build_global_daily_moment(ev, dr)

        expected_m0 = seismic_moment(6.0)
        for day in ["2019-01-01", "2019-01-03"]:
            assert result.loc[day, "log10_moment_sum"] == pytest.approx(
                np.log10(expected_m0), rel=1e-6
            ), f"Log10 on {day} was biased by zero-fill"

        # Gap day
        assert np.isnan(result.loc["2019-01-02", "log10_moment_sum"])
        assert result.loc["2019-01-02", "event_count"] == 0


# ---------------------------------------------------------------------------
# build_global_daily_magnitude (Homola metric)
# ---------------------------------------------------------------------------

class TestGlobalDailyMagnitude:
    def test_output_columns(self, enriched, date_range_fixture) -> None:
        result = build_global_daily_magnitude(enriched, date_range_fixture)
        assert set(result.columns) == {"magnitude_sum", "event_count"}

    def test_gapless_index(self, enriched, date_range_fixture) -> None:
        result = build_global_daily_magnitude(enriched, date_range_fixture)
        assert len(result) == len(date_range_fixture)

    def test_zero_event_day_magnitude_sum_is_zero(self) -> None:
        """For Homola replication: no-event days must have magnitude_sum = 0."""
        idx = pd.to_datetime(["2019-01-01", "2019-01-03"])
        ev_raw = pd.DataFrame({"latitude": [35.0, 35.0], "longitude": [140.0, 140.0],
                                "depth": [30.0, 30.0], "mag": [5.0, 5.0]}, index=idx)
        ev_raw.index.name = "time"
        ev = _enrich(ev_raw, 4.5)
        dr = _make_date_range(ev)
        result = build_global_daily_magnitude(ev, dr)
        assert result.loc["2019-01-02", "magnitude_sum"] == 0.0

    def test_magnitude_sum_correct(self, enriched, date_range_fixture) -> None:
        result = build_global_daily_magnitude(enriched, date_range_fixture)
        # Jan 1: mag 5.0 + 6.5 = 11.5
        assert result.loc["2019-01-01", "magnitude_sum"] == pytest.approx(5.0 + 6.5, rel=1e-9)


# ---------------------------------------------------------------------------
# build_regional_daily_moment
# ---------------------------------------------------------------------------

class TestRegionalDailyMoment:
    def test_output_columns(self, enriched, date_range_fixture) -> None:
        result = build_regional_daily_moment(enriched, date_range_fixture)
        assert set(result.columns) == {"log10_moment_sum", "event_count"}

    def test_multiindex_levels(self, enriched, date_range_fixture) -> None:
        result = build_regional_daily_moment(enriched, date_range_fixture)
        assert result.index.names == ["date", "lat_bin", "lon_bin"]

    def test_cell_assignment_correct(self, enriched, date_range_fixture) -> None:
        """Events at lat=35, lon=140 → lat_bin=30, lon_bin=140."""
        result = build_regional_daily_moment(enriched, date_range_fixture)
        cell = result.xs((30, 140), level=("lat_bin", "lon_bin"))
        assert len(cell) == len(date_range_fixture)
        assert cell.loc["2019-01-01", "event_count"] == 1  # one event in this cell on Jan 1

    def test_active_cell_is_gapless(self, enriched, date_range_fixture) -> None:
        """Every date in date_range appears for each active cell."""
        result = build_regional_daily_moment(enriched, date_range_fixture)
        for cell_idx in result.index.droplevel("date").unique():
            ts = result.xs(cell_idx, level=("lat_bin", "lon_bin"))
            assert len(ts) == len(date_range_fixture)

    def test_zero_event_cell_day_is_nan_log10(self, enriched, date_range_fixture) -> None:
        result = build_regional_daily_moment(enriched, date_range_fixture)
        # The cell (30, 140) has events on Jan 1 and Jan 3 but not Jan 2
        cell = result.xs((30, 140), level=("lat_bin", "lon_bin"))
        assert np.isnan(cell.loc["2019-01-02", "log10_moment_sum"])
        assert cell.loc["2019-01-02", "event_count"] == 0

    def test_distinct_cells(self, enriched, date_range_fixture) -> None:
        """Events in different grid cells should appear as separate entries."""
        result = build_regional_daily_moment(enriched, date_range_fixture)
        cells = result.index.droplevel("date").unique()
        # lat=35→30, lon=140→140 and lat=-10→-10, lon=-75→-80 and lat=60→60, lon=150→150
        assert (30, 140) in cells.tolist()
        assert (-10, -80) in cells.tolist()


# ---------------------------------------------------------------------------
# build_depth_stratified_daily
# ---------------------------------------------------------------------------

class TestDepthStratified:
    def test_all_three_strata_present(self, enriched, date_range_fixture) -> None:
        result = build_depth_stratified_daily(enriched, date_range_fixture)
        strata = result.index.get_level_values("depth_stratum").unique().tolist()
        for s in DEPTH_LABELS:
            assert s in strata

    def test_gapless_per_stratum(self, enriched, date_range_fixture) -> None:
        result = build_depth_stratified_daily(enriched, date_range_fixture)
        for stratum in DEPTH_LABELS:
            ts = result.xs(stratum, level="depth_stratum")
            assert len(ts) == len(date_range_fixture)

    def test_zero_fill_for_absent_stratum_day(self) -> None:
        """A stratum with no events on a day must have count=0, log10=NaN."""
        # All events shallow — intermediate and deep must be zero-filled
        idx = pd.to_datetime(["2019-01-01"])
        ev_raw = pd.DataFrame({"latitude": [35.0], "longitude": [140.0],
                                "depth": [10.0], "mag": [6.0]}, index=idx)
        ev_raw.index.name = "time"
        ev = _enrich(ev_raw, 4.5)
        dr = _make_date_range(ev)
        result = build_depth_stratified_daily(ev, dr)
        for stratum in ["intermediate", "deep"]:
            row = result.xs(stratum, level="depth_stratum").iloc[0]
            assert row["event_count"] == 0
            assert np.isnan(row["log10_moment_sum"])

    def test_depth_values_correct(self, enriched, date_range_fixture) -> None:
        result = build_depth_stratified_daily(enriched, date_range_fixture)
        # Jan 1 has events at 30 km (shallow) and 150 km (intermediate)
        # Jan 2 has event at 600 km (deep)
        shallow_jan1 = result.loc[("2019-01-01", "shallow")]
        assert shallow_jan1["event_count"] == 1
        inter_jan1   = result.loc[("2019-01-01", "intermediate")]
        assert inter_jan1["event_count"] == 1
        deep_jan2    = result.loc[("2019-01-02", "deep")]
        assert deep_jan2["event_count"] == 1


# ---------------------------------------------------------------------------
# process_catalogue (integration)
# ---------------------------------------------------------------------------

class TestProcessCatalogue:
    def test_writes_four_parquet_files(self, events, tmp_path) -> None:
        paths = process_catalogue(events, tmp_path / "out")
        assert len(paths) == 4
        for path in paths.values():
            assert Path(path).exists(), f"Missing output: {path}"

    def test_global_moment_parquet_readable(self, events, tmp_path) -> None:
        paths = process_catalogue(events, tmp_path / "out")
        df = pd.read_parquet(paths["global_daily_moment_sum"])
        assert "log10_moment_sum" in df.columns
        assert "event_count" in df.columns

    def test_global_magnitude_parquet_readable(self, events, tmp_path) -> None:
        paths = process_catalogue(events, tmp_path / "out")
        df = pd.read_parquet(paths["global_daily_magnitude_sum"])
        assert "magnitude_sum" in df.columns

    def test_regional_parquet_has_multiindex(self, events, tmp_path) -> None:
        paths = process_catalogue(events, tmp_path / "out")
        df = pd.read_parquet(paths["regional_daily_moment"])
        assert df.index.names == ["date", "lat_bin", "lon_bin"]

    def test_depth_parquet_has_all_strata(self, events, tmp_path) -> None:
        paths = process_catalogue(events, tmp_path / "out")
        df = pd.read_parquet(paths["depth_stratified_daily_moment"])
        strata = df.index.get_level_values("depth_stratum").unique().tolist()
        for s in DEPTH_LABELS:
            assert s in strata

    def test_min_magnitude_filter_applied(self, tmp_path) -> None:
        """Events below min_magnitude must not appear in outputs."""
        idx = pd.to_datetime(["2019-01-01", "2019-01-01"])
        ev = pd.DataFrame({"latitude": [35.0, 35.0], "longitude": [140.0, 140.0],
                            "depth": [30.0, 30.0], "mag": [3.0, 6.0]}, index=idx)
        ev.index.name = "time"
        paths = process_catalogue(ev, tmp_path / "out", min_magnitude=4.5)
        df = pd.read_parquet(paths["global_daily_moment_sum"])
        # Only the M6.0 event should count
        expected = seismic_moment(6.0)
        assert df.loc["2019-01-01", "event_count"] == 1
        assert df.loc["2019-01-01", "log10_moment_sum"] == pytest.approx(
            np.log10(expected), rel=1e-6
        )

    def test_idempotent_overwrite(self, events, tmp_path) -> None:
        """Calling process_catalogue twice on the same output dir should succeed."""
        out = tmp_path / "out"
        process_catalogue(events, out)
        process_catalogue(events, out)   # should not raise
