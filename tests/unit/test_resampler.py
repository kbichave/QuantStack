# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for quantcore.data.resampler module."""

import numpy as np
import pandas as pd
import pytest

from quantcore.config.timeframes import Timeframe
from quantcore.data.resampler import (
    TimeframeResampler,
    OHLCVResampler,
    build_multi_tf_from_hourly,
    align_daily_to_hourly,
)


class TestTimeframeResampler:
    """Test TimeframeResampler class."""

    @pytest.fixture
    def resampler(self) -> TimeframeResampler:
        """Create resampler instance."""
        return TimeframeResampler()

    @pytest.fixture
    def hourly_data(self) -> pd.DataFrame:
        """Create sample hourly data."""
        # Generate 2 weeks of hourly data (7 bars per day, 5 days per week)
        dates = pd.date_range(
            start="2023-01-02 09:00",
            periods=70,  # 2 weeks
            freq="1h",
            tz="America/New_York",
        )
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(70) * 0.5)
        high = close + np.abs(np.random.randn(70) * 0.3)
        low = close - np.abs(np.random.randn(70) * 0.3)
        open_ = close + np.random.randn(70) * 0.2

        return pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(10000, 100000, 70),
            },
            index=dates,
        )

    def test_resample_empty_dataframe(self, resampler):
        """Test resampling empty DataFrame."""
        df = pd.DataFrame()
        result = resampler.resample_to_higher_tf(df, Timeframe.D1)
        assert result.empty

    def test_resample_to_4h(self, resampler, hourly_data):
        """Test resampling hourly to 4-hour."""
        result = resampler.resample_to_higher_tf(hourly_data, Timeframe.H4)

        # Should have fewer bars than hourly
        assert len(result) < len(hourly_data)
        assert len(result) > 0

        # Should have OHLCV columns
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns

    def test_resample_to_daily(self, resampler, hourly_data):
        """Test resampling hourly to daily."""
        result = resampler.resample_to_higher_tf(hourly_data, Timeframe.D1)

        # Should have even fewer bars than 4H
        assert len(result) < len(hourly_data) / 4
        assert len(result) > 0

    def test_resample_produces_valid_ohlcv(self, resampler, hourly_data):
        """Test that resampling produces valid OHLCV structure."""
        result = resampler.resample_to_higher_tf(hourly_data, Timeframe.D1)

        # High should be >= low
        assert (result["high"] >= result["low"]).all()

        # All prices should be positive
        assert (result[["open", "high", "low", "close"]] > 0).all().all()

        # Volume should be non-negative
        assert (result["volume"] >= 0).all()

    def test_build_4h_from_1h(self, resampler, hourly_data):
        """Test build_4h_from_1h method."""
        result = resampler.build_4h_from_1h(hourly_data)
        assert len(result) > 0
        assert len(result) < len(hourly_data)

    def test_build_daily_from_intraday(self, resampler, hourly_data):
        """Test build_daily_from_intraday method."""
        result = resampler.build_daily_from_intraday(hourly_data)
        assert len(result) > 0

    def test_build_weekly_from_daily(self, resampler):
        """Test build_weekly_from_daily method."""
        dates = pd.date_range(
            start="2023-01-02",
            periods=30,
            freq="1D",
            tz="America/New_York",
        )
        daily_data = pd.DataFrame(
            {
                "open": np.ones(30) * 100,
                "high": np.ones(30) * 101,
                "low": np.ones(30) * 99,
                "close": np.ones(30) * 100.5,
                "volume": np.ones(30, dtype=int) * 1000,
            },
            index=dates,
        )

        result = resampler.build_weekly_from_daily(daily_data)
        assert len(result) > 0
        assert len(result) <= 5  # ~4 weeks in 30 days


class TestAlignHigherTFToLower:
    """Test higher timeframe alignment to lower."""

    @pytest.fixture
    def resampler(self) -> TimeframeResampler:
        return TimeframeResampler()

    @pytest.fixture
    def hourly_data(self) -> pd.DataFrame:
        dates = pd.date_range(
            start="2023-01-02 09:00",
            periods=14,
            freq="1h",
            tz="America/New_York",
        )
        return pd.DataFrame(
            {
                "open": np.arange(100, 114),
                "high": np.arange(101, 115),
                "low": np.arange(99, 113),
                "close": np.arange(100.5, 114.5),
                "volume": np.ones(14, dtype=int) * 1000,
            },
            index=dates,
        )

    @pytest.fixture
    def daily_data(self) -> pd.DataFrame:
        # Need 3 days so that after shift(1), we still have data covering the hourly range
        # Daily data: Jan 1, Jan 2, Jan 3
        # After shift(1): Jan 1=NaN, Jan 2=Jan1 data, Jan 3=Jan2 data
        # Hourly data goes from Jan 2 09:00 to Jan 2 22:00
        # Forward fill from Jan 2 daily (which has Jan 1's data after shift)
        dates = pd.date_range(
            start="2023-01-01",
            periods=3,
            freq="1D",
            tz="America/New_York",
        )
        return pd.DataFrame(
            {
                "open": [90, 100, 110],
                "high": [105, 115, 125],
                "low": [85, 95, 105],
                "close": [100, 110, 120],
                "volume": [8000, 10000, 12000],
            },
            index=dates,
        )

    def test_align_empty_lower(self, resampler, daily_data):
        """Test alignment with empty lower TF data."""
        df_lower = pd.DataFrame()
        result = resampler.align_higher_tf_to_lower(df_lower, daily_data)
        assert result.empty

    def test_align_empty_higher(self, resampler, hourly_data):
        """Test alignment with empty higher TF data."""
        df_higher = pd.DataFrame()
        result = resampler.align_higher_tf_to_lower(hourly_data, df_higher)
        assert result.equals(hourly_data)

    def test_align_with_prefix(self, resampler, hourly_data, daily_data):
        """Test alignment adds prefix to columns."""
        result = resampler.align_higher_tf_to_lower(
            hourly_data, daily_data, prefix="D1"
        )

        # All columns should have prefix
        for col in result.columns:
            assert col.startswith("D1_")

    def test_align_forward_fills(self, resampler, hourly_data, daily_data):
        """Test that alignment forward fills values."""
        result = resampler.align_higher_tf_to_lower(
            hourly_data, daily_data, prefix="D1"
        )

        # Should have same length as lower TF
        assert len(result) == len(hourly_data)

        # Should not have NaN values after forward fill
        # (except possibly at the start before first higher TF bar)
        last_values = result.iloc[-5:]  # Last 5 rows should definitely be filled
        assert not last_values.isna().any().any()


class TestMultiTimeframeDataset:
    """Test multi-timeframe dataset building."""

    @pytest.fixture
    def resampler(self) -> TimeframeResampler:
        return TimeframeResampler()

    @pytest.fixture
    def hourly_data(self) -> pd.DataFrame:
        dates = pd.date_range(
            start="2023-01-02 09:00",
            periods=200,
            freq="1h",
            tz="America/New_York",
        )
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)
        return pd.DataFrame(
            {
                "open": close + np.random.randn(200) * 0.2,
                "high": close + np.abs(np.random.randn(200) * 0.3),
                "low": close - np.abs(np.random.randn(200) * 0.3),
                "close": close,
                "volume": np.random.randint(10000, 100000, 200),
            },
            index=dates,
        )

    def test_build_multi_timeframe_dataset(self, resampler, hourly_data):
        """Test building multi-TF dataset."""
        result = resampler.build_multi_timeframe_dataset(hourly_data)

        assert Timeframe.H1 in result
        assert Timeframe.H4 in result
        assert Timeframe.D1 in result
        assert Timeframe.W1 in result

    def test_build_multi_timeframe_selective(self, resampler, hourly_data):
        """Test building multi-TF dataset with selection."""
        result = resampler.build_multi_timeframe_dataset(
            hourly_data,
            include_4h=True,
            include_daily=False,
            include_weekly=False,
        )

        assert Timeframe.H1 in result
        assert Timeframe.H4 in result
        assert Timeframe.D1 not in result
        assert Timeframe.W1 not in result


class TestGetLastCompletedBar:
    """Test getting last completed bar."""

    @pytest.fixture
    def resampler(self) -> TimeframeResampler:
        return TimeframeResampler()

    def test_get_last_completed_bar(self, resampler):
        """Test getting last completed bar."""
        dates = pd.date_range(
            start="2023-01-02 09:00",
            periods=10,
            freq="1h",
            tz="America/New_York",
        )
        df = pd.DataFrame(
            {
                "open": np.arange(100, 110),
                "high": np.arange(101, 111),
                "low": np.arange(99, 109),
                "close": np.arange(100.5, 110.5),
                "volume": np.ones(10, dtype=int) * 1000,
            },
            index=dates,
        )

        as_of = pd.Timestamp("2023-01-02 14:30", tz="America/New_York")
        result = resampler.get_last_completed_bar(df, Timeframe.H1, as_of)

        assert result is not None
        assert result.name < as_of

    def test_get_last_completed_bar_empty(self, resampler):
        """Test getting last completed bar from empty DataFrame."""
        df = pd.DataFrame()
        as_of = pd.Timestamp("2023-01-02 14:30", tz="America/New_York")
        result = resampler.get_last_completed_bar(df, Timeframe.H1, as_of)

        assert result is None

    def test_get_last_completed_bar_none_before(self, resampler):
        """Test when no bars are completed before as_of."""
        dates = pd.date_range(
            start="2023-01-02 15:00",
            periods=5,
            freq="1h",
            tz="America/New_York",
        )
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

        as_of = pd.Timestamp("2023-01-02 14:30", tz="America/New_York")
        result = resampler.get_last_completed_bar(df, Timeframe.H1, as_of)

        assert result is None


class TestConvenienceFunctions:
    """Test module convenience functions."""

    @pytest.fixture
    def hourly_data(self) -> pd.DataFrame:
        dates = pd.date_range(
            start="2023-01-02 09:00",
            periods=100,
            freq="1h",
            tz="America/New_York",
        )
        return pd.DataFrame(
            {
                "open": np.ones(100) * 100,
                "high": np.ones(100) * 101,
                "low": np.ones(100) * 99,
                "close": np.ones(100) * 100.5,
                "volume": np.ones(100, dtype=int) * 1000,
            },
            index=dates,
        )

    def test_build_multi_tf_from_hourly(self, hourly_data):
        """Test build_multi_tf_from_hourly function."""
        result = build_multi_tf_from_hourly(hourly_data)

        assert Timeframe.H1 in result
        assert Timeframe.H4 in result
        assert Timeframe.D1 in result
        assert Timeframe.W1 in result

    def test_align_daily_to_hourly(self, hourly_data):
        """Test align_daily_to_hourly function."""
        dates = pd.date_range(
            start="2023-01-02",
            periods=10,
            freq="1D",
            tz="America/New_York",
        )
        daily_data = pd.DataFrame(
            {
                "open": np.arange(100, 110),
                "high": np.arange(101, 111),
                "low": np.arange(99, 109),
                "close": np.arange(100.5, 110.5),
                "volume": np.ones(10, dtype=int) * 10000,
            },
            index=dates,
        )

        result = align_daily_to_hourly(hourly_data, daily_data)
        assert len(result) == len(hourly_data)
        for col in result.columns:
            assert col.startswith("D1_")


class TestBackwardCompatibility:
    """Test backward compatibility aliases."""

    def test_ohlcv_resampler_alias(self):
        """Test OHLCVResampler is alias for TimeframeResampler."""
        assert OHLCVResampler is TimeframeResampler

        resampler = OHLCVResampler()
        assert isinstance(resampler, TimeframeResampler)
