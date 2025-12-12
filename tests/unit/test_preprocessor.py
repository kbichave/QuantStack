# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for quantcore.data.preprocessor module."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import pytz

from quantcore.config.timeframes import Timeframe
from quantcore.data.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Test DataPreprocessor class."""

    @pytest.fixture
    def preprocessor(self) -> DataPreprocessor:
        """Create preprocessor instance."""
        return DataPreprocessor()

    @pytest.fixture
    def sample_ohlcv(self) -> pd.DataFrame:
        """Create sample OHLCV data."""
        dates = pd.date_range(
            start="2023-01-03 09:00",
            periods=100,
            freq="1h",
            tz="America/New_York",
        )
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        high = close + np.abs(np.random.randn(100) * 0.3)
        low = close - np.abs(np.random.randn(100) * 0.3)
        open_ = close + np.random.randn(100) * 0.2

        return pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(10000, 100000, 100),
            },
            index=dates,
        )

    def test_preprocess_empty_dataframe(self, preprocessor):
        """Test preprocessing empty DataFrame."""
        df = pd.DataFrame()
        result = preprocessor.preprocess(df, Timeframe.H1)
        assert result.empty

    def test_preprocess_basic(self, preprocessor, sample_ohlcv):
        """Test basic preprocessing."""
        result = preprocessor.preprocess(sample_ohlcv, Timeframe.H1)
        assert not result.empty
        assert len(result) > 0

    def test_preprocess_removes_duplicates(self, preprocessor, sample_ohlcv):
        """Test that duplicates are removed."""
        # Add duplicate row
        df = pd.concat([sample_ohlcv, sample_ohlcv.iloc[[0]]])
        result = preprocessor.preprocess(df, Timeframe.H1)
        assert len(result) <= len(sample_ohlcv)
        assert not result.index.duplicated().any()

    def test_preprocess_sorts_by_index(self, preprocessor, sample_ohlcv):
        """Test that result is sorted by index."""
        # Shuffle the DataFrame
        shuffled = sample_ohlcv.sample(frac=1)
        result = preprocessor.preprocess(shuffled, Timeframe.H1)
        assert result.index.is_monotonic_increasing

    def test_timezone_normalization_utc(self, preprocessor):
        """Test timezone normalization from UTC."""
        dates = pd.date_range(start="2023-01-03 14:00", periods=10, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {
                "open": np.ones(10) * 100,
                "high": np.ones(10) * 101,
                "low": np.ones(10) * 99,
                "close": np.ones(10) * 100.5,
                "volume": np.ones(10, dtype=int) * 1000,
            },
            index=dates,
        )

        result = preprocessor.preprocess(df, Timeframe.H1, market_hours_only=False)
        assert str(result.index.tz) == "America/New_York"

    def test_timezone_normalization_naive(self, preprocessor):
        """Test timezone normalization for naive timestamps."""
        dates = pd.date_range(start="2023-01-03 09:00", periods=10, freq="1h")
        df = pd.DataFrame(
            {
                "open": np.ones(10) * 100,
                "high": np.ones(10) * 101,
                "low": np.ones(10) * 99,
                "close": np.ones(10) * 100.5,
                "volume": np.ones(10, dtype=int) * 1000,
            },
            index=dates,
        )

        result = preprocessor.preprocess(df, Timeframe.H1, market_hours_only=False)
        assert result.index.tz is not None

    def test_market_hours_filter_hourly(self, preprocessor, sample_ohlcv):
        """Test market hours filtering for hourly data."""
        result = preprocessor.preprocess(
            sample_ohlcv, Timeframe.H1, market_hours_only=True
        )

        # All hours should be within market hours
        hours = result.index.hour
        assert all((hours >= 9) & (hours < 16))

    def test_market_hours_filter_not_applied_to_daily(self, preprocessor):
        """Test market hours filter not applied to daily data."""
        dates = pd.date_range(
            start="2023-01-03", periods=30, freq="1D", tz="America/New_York"
        )
        df = pd.DataFrame(
            {
                "open": np.ones(30) * 100,
                "high": np.ones(30) * 101,
                "low": np.ones(30) * 99,
                "close": np.ones(30) * 100.5,
                "volume": np.ones(30, dtype=int) * 1000,
            },
            index=dates,
        )

        result = preprocessor.preprocess(df, Timeframe.D1, market_hours_only=True)
        # Should still have data (filter only applies to intraday)
        assert len(result) > 0


class TestValidateOHLCV:
    """Test OHLCV validation."""

    @pytest.fixture
    def preprocessor(self) -> DataPreprocessor:
        return DataPreprocessor()

    def test_fix_high_less_than_low(self, preprocessor):
        """Test fixing high < low."""
        dates = pd.date_range(
            start="2023-01-03 09:00", periods=5, freq="1h", tz="America/New_York"
        )
        df = pd.DataFrame(
            {
                "open": [100, 100, 100, 100, 100],
                "high": [99, 101, 101, 101, 101],  # First row: high < low
                "low": [101, 99, 99, 99, 99],  # First row: low > high
                "close": [100, 100, 100, 100, 100],
                "volume": [1000, 1000, 1000, 1000, 1000],
            },
            index=dates,
        )

        result = preprocessor._validate_ohlcv(df)
        # High should always be >= low after fix
        assert (result["high"] >= result["low"]).all()

    def test_fix_high_less_than_close(self, preprocessor):
        """Test fixing high < close."""
        dates = pd.date_range(
            start="2023-01-03 09:00", periods=3, freq="1h", tz="America/New_York"
        )
        df = pd.DataFrame(
            {
                "open": [100, 100, 100],
                "high": [100, 100, 100],  # High < close in row 0
                "low": [98, 98, 98],
                "close": [105, 100, 100],  # Close > high in row 0
                "volume": [1000, 1000, 1000],
            },
            index=dates,
        )

        result = preprocessor._validate_ohlcv(df)
        # High should always be >= max(open, close)
        max_oc = result[["open", "close"]].max(axis=1)
        assert (result["high"] >= max_oc).all()

    def test_remove_non_positive_prices(self, preprocessor):
        """Test removing rows with non-positive prices."""
        dates = pd.date_range(
            start="2023-01-03 09:00", periods=5, freq="1h", tz="America/New_York"
        )
        df = pd.DataFrame(
            {
                "open": [100, -1, 100, 100, 100],  # Negative in row 1
                "high": [101, 101, 101, 101, 101],
                "low": [99, 99, 99, 99, 99],
                "close": [100, 100, 0, 100, 100],  # Zero in row 2
                "volume": [1000, 1000, 1000, 1000, 1000],
            },
            index=dates,
        )

        result = preprocessor._validate_ohlcv(df)
        assert len(result) < 5
        assert (result[["open", "high", "low", "close"]] > 0).all().all()


class TestGapDetection:
    """Test gap detection functionality."""

    @pytest.fixture
    def preprocessor(self) -> DataPreprocessor:
        return DataPreprocessor()

    def test_detect_gaps_no_gaps(self, preprocessor):
        """Test gap detection with no gaps."""
        dates = pd.date_range(
            start="2023-01-03 09:00", periods=10, freq="1h", tz="America/New_York"
        )
        df = pd.DataFrame(
            {
                "open": np.ones(10) * 100,
                "high": np.ones(10) * 101,
                "low": np.ones(10) * 99,
                "close": np.ones(10) * 100.5,
                "volume": np.ones(10, dtype=int) * 1000,
            },
            index=dates,
        )

        gaps = preprocessor.detect_gaps(df, Timeframe.H1)
        assert len(gaps) == 0

    def test_detect_gaps_with_gaps(self, preprocessor):
        """Test gap detection with gaps."""
        dates = pd.to_datetime(
            [
                "2023-01-03 09:00",
                "2023-01-03 10:00",
                "2023-01-03 11:00",
                # Gap of 3 hours
                "2023-01-03 15:00",
                "2023-01-03 16:00",
            ]
        ).tz_localize("America/New_York")

        df = pd.DataFrame(
            {
                "open": np.ones(5) * 100,
                "high": np.ones(5) * 101,
                "low": np.ones(5) * 99,
                "close": np.ones(5) * 100.5,
                "volume": np.ones(5, dtype=int) * 1000,
            },
            index=dates,
        )

        gaps = preprocessor.detect_gaps(df, Timeframe.H1)
        assert len(gaps) > 0

    def test_detect_gaps_empty_dataframe(self, preprocessor):
        """Test gap detection with empty DataFrame."""
        df = pd.DataFrame()
        gaps = preprocessor.detect_gaps(df, Timeframe.H1)
        assert len(gaps) == 0


class TestDataQualityReport:
    """Test data quality report generation."""

    @pytest.fixture
    def preprocessor(self) -> DataPreprocessor:
        return DataPreprocessor()

    def test_quality_report_empty_df(self, preprocessor):
        """Test quality report for empty DataFrame."""
        df = pd.DataFrame()
        report = preprocessor.get_data_quality_report(df, Timeframe.H1)
        assert report["status"] == "empty"
        assert report["rows"] == 0

    def test_quality_report_structure(self, preprocessor):
        """Test quality report has expected structure."""
        dates = pd.date_range(
            start="2023-01-03 09:00", periods=50, freq="1h", tz="America/New_York"
        )
        df = pd.DataFrame(
            {
                "open": np.random.uniform(99, 101, 50),
                "high": np.random.uniform(100, 102, 50),
                "low": np.random.uniform(98, 100, 50),
                "close": np.random.uniform(99, 101, 50),
                "volume": np.random.randint(1000, 10000, 50),
            },
            index=dates,
        )

        report = preprocessor.get_data_quality_report(df, Timeframe.H1)

        assert "rows" in report
        assert "start_date" in report
        assert "end_date" in report
        assert "missing_values" in report
        assert "duplicates" in report
        assert "gaps" in report
        assert "invalid_ohlcv" in report
        assert "price_stats" in report

    def test_quality_report_price_stats(self, preprocessor):
        """Test price statistics in quality report."""
        dates = pd.date_range(
            start="2023-01-03 09:00", periods=100, freq="1h", tz="America/New_York"
        )
        close_prices = np.random.uniform(95, 105, 100)
        df = pd.DataFrame(
            {
                "open": np.random.uniform(99, 101, 100),
                "high": np.random.uniform(100, 106, 100),
                "low": np.random.uniform(94, 100, 100),
                "close": close_prices,
                "volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )

        report = preprocessor.get_data_quality_report(df, Timeframe.H1)
        stats = report["price_stats"]

        # pandas std uses ddof=1 by default, numpy uses ddof=0
        assert stats["mean"] == pytest.approx(close_prices.mean(), rel=1e-5)
        assert stats["std"] == pytest.approx(pd.Series(close_prices).std(), rel=1e-5)
        assert stats["min"] == close_prices.min()
        assert stats["max"] == close_prices.max()
