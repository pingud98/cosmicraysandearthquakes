"""Tests for src/crq/ingest/nmdb.py"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crq.ingest.nmdb import (
    compute_daily_coverage,
    log_avg,
    parse_nmdb_csv,
    resample_daily,
)


# ---------------------------------------------------------------------------
# log_avg
# ---------------------------------------------------------------------------

class TestLogAvg:
    def test_single_value(self) -> None:
        """log_avg of a single value x should equal x."""
        assert log_avg(np.array([40.0])) == pytest.approx(40.0, rel=1e-9)

    def test_symmetric_pair(self) -> None:
        """log_avg([10, 10]) == 10."""
        assert log_avg(np.array([10.0, 10.0])) == pytest.approx(10.0, rel=1e-9)

    def test_known_value(self) -> None:
        """log_avg([20, 30]) = 10*log10(mean(10^2, 10^3)) = 10*log10(550) ≈ 27.404."""
        expected = 10.0 * np.log10(np.mean([10**2, 10**3]))
        assert log_avg(np.array([20.0, 30.0])) == pytest.approx(expected, rel=1e-9)

    def test_nan_ignored(self) -> None:
        """NaN values should be excluded from the average."""
        result = log_avg(np.array([10.0, np.nan, 10.0]))
        assert result == pytest.approx(10.0, rel=1e-9)

    def test_all_nan_returns_nan(self) -> None:
        assert np.isnan(log_avg(np.array([np.nan, np.nan])))

    def test_pandas_series(self) -> None:
        """Accept a pd.Series as input."""
        s = pd.Series([10.0, 10.0])
        assert log_avg(s) == pytest.approx(10.0, rel=1e-9)


# ---------------------------------------------------------------------------
# parse_nmdb_csv
# ---------------------------------------------------------------------------

class TestParseNmdbCsv:
    def test_returns_dataframe(self, nmdb_csv_file: Path) -> None:
        df = parse_nmdb_csv(nmdb_csv_file, "OULU")
        assert isinstance(df, pd.DataFrame)
        assert "OULU" in df.columns

    def test_index_is_datetime(self, nmdb_csv_file: Path) -> None:
        df = parse_nmdb_csv(nmdb_csv_file, "OULU")
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_row_count(self, nmdb_csv_file: Path) -> None:
        df = parse_nmdb_csv(nmdb_csv_file, "OULU")
        assert len(df) == 24

    def test_values_are_positive_floats(self, nmdb_csv_file: Path) -> None:
        df = parse_nmdb_csv(nmdb_csv_file, "OULU")
        assert (df["OULU"] > 0).all()

    def test_sentinel_zero_becomes_nan(self, tmp_path: Path) -> None:
        """Zero-valued rows (sentinel for missing) must become NaN."""
        csv = tmp_path / "OULU2019.csv"
        csv.write_text(
            "start_date_time;OULU\n"
            "2019-01-01 00:59:00;0.0\n"
            "2019-01-01 01:59:00;6850.0\n"
        )
        df = parse_nmdb_csv(csv, "OULU")
        assert np.isnan(df["OULU"].iloc[0])
        assert df["OULU"].iloc[1] == pytest.approx(6850.0)

    def test_empty_file_returns_empty_df(self, tmp_path: Path) -> None:
        csv = tmp_path / "empty.csv"
        csv.write_text("")
        df = parse_nmdb_csv(csv, "OULU")
        assert df.empty

    def test_duplicate_timestamps_dropped(self, tmp_path: Path) -> None:
        csv = tmp_path / "dup.csv"
        csv.write_text(
            "start_date_time;OULU\n"
            "2019-01-01 00:59:00;6850.0\n"
            "2019-01-01 00:59:00;6900.0\n"   # duplicate
        )
        df = parse_nmdb_csv(csv, "OULU")
        assert len(df) == 1
        assert df["OULU"].iloc[0] == pytest.approx(6850.0)


# ---------------------------------------------------------------------------
# compute_daily_coverage
# ---------------------------------------------------------------------------

class TestDailyCoverage:
    def test_full_day_coverage_is_one(self, nmdb_csv_file: Path) -> None:
        df = parse_nmdb_csv(nmdb_csv_file, "OULU")
        cov = compute_daily_coverage(df["OULU"])
        assert float(cov.iloc[0]) == pytest.approx(1.0)

    def test_half_day_coverage(self, nmdb_csv_partial: Path) -> None:
        df = parse_nmdb_csv(nmdb_csv_partial, "OULU")
        cov = compute_daily_coverage(df["OULU"])
        assert float(cov.iloc[0]) == pytest.approx(0.5)

    def test_coverage_range(self, nmdb_csv_file: Path) -> None:
        df = parse_nmdb_csv(nmdb_csv_file, "OULU")
        cov = compute_daily_coverage(df["OULU"])
        assert (cov >= 0).all() and (cov <= 1).all()

    def test_all_nan_gives_zero_coverage(self) -> None:
        idx = pd.date_range("2019-01-01", periods=24, freq="h")
        s = pd.Series([np.nan] * 24, index=idx)
        cov = compute_daily_coverage(s)
        assert float(cov.iloc[0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# resample_daily — coverage threshold + NaN propagation
# ---------------------------------------------------------------------------

class TestResampleDaily:
    def test_full_day_is_not_flagged(self, nmdb_csv_file: Path) -> None:
        hourly = parse_nmdb_csv(nmdb_csv_file, "OULU")
        daily = resample_daily(hourly, "OULU")
        assert not daily["OULU_flagged"].iloc[0]
        assert not np.isnan(daily["OULU"].iloc[0])

    def test_partial_day_below_threshold_is_nan(self, nmdb_csv_partial: Path) -> None:
        """12/24 hours = 0.5 coverage < default 0.60 threshold → NaN."""
        hourly = parse_nmdb_csv(nmdb_csv_partial, "OULU")
        daily = resample_daily(hourly, "OULU", coverage_threshold=0.60)
        assert daily["OULU_flagged"].iloc[0]
        assert np.isnan(daily["OULU"].iloc[0])

    def test_partial_day_above_threshold_is_kept(self, nmdb_csv_partial: Path) -> None:
        """12/24 = 0.5 coverage — kept when threshold is 0.40."""
        hourly = parse_nmdb_csv(nmdb_csv_partial, "OULU")
        daily = resample_daily(hourly, "OULU", coverage_threshold=0.40)
        assert not daily["OULU_flagged"].iloc[0]
        assert not np.isnan(daily["OULU"].iloc[0])

    def test_two_days_mixed_coverage(self, nmdb_csv_two_days: Path) -> None:
        """Day 1 = 24 hrs (ok), Day 2 = 10 hrs (flagged at 0.60 threshold)."""
        hourly = parse_nmdb_csv(nmdb_csv_two_days, "OULU")
        daily = resample_daily(hourly, "OULU", coverage_threshold=0.60)
        assert not daily["OULU_flagged"].iloc[0], "Day 1 should not be flagged"
        assert daily["OULU_flagged"].iloc[1], "Day 2 should be flagged"
        assert not np.isnan(daily["OULU"].iloc[0])
        assert np.isnan(daily["OULU"].iloc[1])

    def test_output_columns(self, nmdb_csv_file: Path) -> None:
        hourly = parse_nmdb_csv(nmdb_csv_file, "OULU")
        daily = resample_daily(hourly, "OULU")
        assert set(daily.columns) == {"OULU", "OULU_coverage", "OULU_flagged"}

    def test_no_fillna_zero_bias(self) -> None:
        """Regression: NaN gaps must NOT be replaced with 0 before averaging."""
        idx = pd.date_range("2019-01-01", periods=24, freq="h")
        vals = [6850.0] * 12 + [np.nan] * 12
        hourly = pd.DataFrame({"OULU": vals}, index=idx)
        daily = resample_daily(hourly, "OULU", coverage_threshold=0.0)
        # Mean of the 12 valid values — must NOT be dragged down toward zero
        expected_mean = np.mean([6850.0] * 12)
        assert daily["OULU"].iloc[0] == pytest.approx(expected_mean, rel=1e-6)
