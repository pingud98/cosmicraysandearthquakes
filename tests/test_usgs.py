"""Tests for src/crq/ingest/usgs.py"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crq.ingest.usgs import compute_daily_seismic, parse_usgs_csv


USGS_CSV = textwrap.dedent("""\
    time,latitude,longitude,depth,mag
    2019-01-01T01:00:00.000Z,-55.877,-1.89,15,6.3
    2019-01-01T05:30:00.000Z,35.556,-121.351,6,4.04
    2019-01-02T12:00:00.000Z,43.7,84.542,15,5.7
""")


@pytest.fixture
def usgs_csv_file(tmp_path: Path) -> Path:
    p = tmp_path / "usgs-2019.csv"
    p.write_text(USGS_CSV)
    return p


class TestParseUsgsCsv:
    def test_row_count(self, usgs_csv_file: Path) -> None:
        df = parse_usgs_csv(usgs_csv_file)
        assert len(df) == 3

    def test_index_is_tz_naive(self, usgs_csv_file: Path) -> None:
        df = parse_usgs_csv(usgs_csv_file)
        assert df.index.tz is None

    def test_numeric_columns(self, usgs_csv_file: Path) -> None:
        df = parse_usgs_csv(usgs_csv_file)
        assert df["mag"].dtype == float
        assert pd.api.types.is_numeric_dtype(df["depth"])


class TestDailySeismic:
    def test_returns_series(self, usgs_csv_file: Path) -> None:
        events = parse_usgs_csv(usgs_csv_file)
        daily = compute_daily_seismic(events)
        assert "mag" in daily.columns

    def test_log_avg_is_not_arithmetic_mean(self, usgs_csv_file: Path) -> None:
        """The log-power average is always ≥ arithmetic mean for positive values."""
        events = parse_usgs_csv(usgs_csv_file)
        daily = compute_daily_seismic(events)
        day1_mag = daily["mag"].iloc[0]
        arith = (6.3 + 4.04) / 2
        assert day1_mag >= arith

    def test_no_events_day_is_nan(self) -> None:
        """Days with no events should be NaN, not 0."""
        idx = pd.to_datetime(["2019-01-01T00:00:00"])
        events = pd.DataFrame({"mag": [5.0]}, index=idx)
        daily = compute_daily_seismic(events)
        # 2019-01-02 should be NaN
        if pd.Timestamp("2019-01-02") in daily.index:
            assert np.isnan(daily.loc["2019-01-02", "mag"])
