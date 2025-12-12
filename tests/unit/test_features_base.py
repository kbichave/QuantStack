# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for quantcore.features.base module."""

from typing import List

import numpy as np
import pandas as pd
import pytest

from quantcore.config.timeframes import Timeframe, TIMEFRAME_PARAMS
from quantcore.features.base import FeatureBase


class ConcreteFeature(FeatureBase):
    """Concrete implementation for testing."""

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["test_ema"] = self.ema(df["close"], self.params.ema_fast)
        result["test_sma"] = self.sma(df["close"], self.params.ema_slow)
        return result

    def get_feature_names(self) -> List[str]:
        return ["test_ema", "test_sma"]


class TestFeatureBaseInit:
    """Test FeatureBase initialization."""

    def test_init_with_timeframe(self):
        """Test initialization with timeframe."""
        feature = ConcreteFeature(Timeframe.D1)

        assert feature.timeframe == Timeframe.D1
        assert feature.params == TIMEFRAME_PARAMS[Timeframe.D1]

    def test_init_all_timeframes(self):
        """Test initialization works for all timeframes."""
        for tf in Timeframe:
            feature = ConcreteFeature(tf)
            assert feature.timeframe == tf
            assert feature.params is not None

    def test_params_are_correct(self):
        """Test that params match expected timeframe params."""
        feature = ConcreteFeature(Timeframe.H1)
        expected = TIMEFRAME_PARAMS[Timeframe.H1]

        assert feature.params.ema_fast == expected.ema_fast
        assert feature.params.ema_slow == expected.ema_slow
        assert feature.params.rsi_period == expected.rsi_period


class TestEMA:
    """Test EMA calculation."""

    @pytest.fixture
    def prices(self) -> pd.Series:
        """Create sample price series."""
        return pd.Series([100.0, 101.0, 102.0, 101.5, 103.0, 102.5, 104.0])

    def test_ema_returns_series(self, prices):
        """Test EMA returns Series."""
        result = FeatureBase.ema(prices, period=3)
        assert isinstance(result, pd.Series)

    def test_ema_length(self, prices):
        """Test EMA has same length as input."""
        result = FeatureBase.ema(prices, period=3)
        assert len(result) == len(prices)

    def test_ema_smoothing(self, prices):
        """Test EMA is smoother than raw prices."""
        result = FeatureBase.ema(prices, period=5)
        # EMA should be between min and max of recent prices
        assert result.iloc[-1] >= prices.min()
        assert result.iloc[-1] <= prices.max()

    def test_ema_responsiveness(self, prices):
        """Test EMA responds to recent prices."""
        # Last few prices trend up, EMA should be above older values
        result = FeatureBase.ema(prices, period=3)
        assert result.iloc[-1] > result.iloc[0]


class TestSMA:
    """Test SMA calculation."""

    @pytest.fixture
    def prices(self) -> pd.Series:
        """Create sample price series."""
        return pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])

    def test_sma_returns_series(self, prices):
        """Test SMA returns Series."""
        result = FeatureBase.sma(prices, period=3)
        assert isinstance(result, pd.Series)

    def test_sma_first_values_nan(self, prices):
        """Test first period-1 values are NaN."""
        result = FeatureBase.sma(prices, period=3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert not pd.isna(result.iloc[2])

    def test_sma_calculation(self, prices):
        """Test SMA calculation is correct."""
        result = FeatureBase.sma(prices, period=3)
        # Third value should be average of first 3: (100+101+102)/3 = 101
        assert result.iloc[2] == pytest.approx(101.0)
        # Fourth value: (101+102+103)/3 = 102
        assert result.iloc[3] == pytest.approx(102.0)


class TestRollingStats:
    """Test rolling statistics methods."""

    @pytest.fixture
    def prices(self) -> pd.Series:
        """Create sample price series."""
        return pd.Series([100.0, 105.0, 95.0, 110.0, 90.0, 115.0])

    def test_rolling_std(self, prices):
        """Test rolling standard deviation."""
        result = FeatureBase.rolling_std(prices, period=3)
        assert isinstance(result, pd.Series)
        assert not pd.isna(result.iloc[2])

    def test_rolling_max(self, prices):
        """Test rolling maximum."""
        result = FeatureBase.rolling_max(prices, period=3)
        # At position 3, max of [95, 110, 90] (well, [105, 95, 110]) = 110
        assert result.iloc[3] == 110.0

    def test_rolling_min(self, prices):
        """Test rolling minimum."""
        result = FeatureBase.rolling_min(prices, period=3)
        # At position 3, min of [105, 95, 110] = 95
        assert result.iloc[3] == 95.0


class TestPctChange:
    """Test percentage change calculation."""

    def test_pct_change_basic(self):
        """Test basic percentage change."""
        prices = pd.Series([100.0, 102.0, 101.0])
        result = FeatureBase.pct_change(prices)

        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == pytest.approx(0.02)  # 2% increase
        assert result.iloc[2] == pytest.approx(-0.0098, rel=1e-2)  # ~1% decrease

    def test_pct_change_multi_period(self):
        """Test multi-period percentage change."""
        prices = pd.Series([100.0, 102.0, 104.0, 106.0])
        result = FeatureBase.pct_change(prices, periods=2)

        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(0.04)  # 4% increase over 2 periods


class TestLogReturn:
    """Test log return calculation."""

    def test_log_return_basic(self):
        """Test basic log return."""
        prices = pd.Series([100.0, 105.0, 110.0])
        result = FeatureBase.log_return(prices)

        assert pd.isna(result.iloc[0])
        expected_1 = np.log(105 / 100)
        assert result.iloc[1] == pytest.approx(expected_1)

    def test_log_return_additivity(self):
        """Test log returns are additive."""
        prices = pd.Series([100.0, 110.0, 121.0])
        result = FeatureBase.log_return(prices)

        # Sum of log returns should equal total log return
        total = np.log(121 / 100)
        sum_of_returns = result.iloc[1] + result.iloc[2]
        assert sum_of_returns == pytest.approx(total)


class TestZScore:
    """Test z-score calculation."""

    def test_zscore_basic(self):
        """Test basic z-score calculation."""
        # Mean-reverting series
        prices = pd.Series([100.0] * 20 + [110.0])  # 20 at 100, then jump to 110
        result = FeatureBase.zscore(prices, period=20)

        # Last value should have high positive z-score
        assert result.iloc[-1] > 2  # More than 2 std above mean

    def test_zscore_symmetry(self):
        """Test z-score symmetry."""
        prices = pd.Series([100.0] * 20 + [90.0])  # 20 at 100, then drop to 90
        result = FeatureBase.zscore(prices, period=20)

        # Last value should have high negative z-score
        assert result.iloc[-1] < -2

    def test_zscore_handles_zero_std(self):
        """Test z-score handles zero standard deviation."""
        prices = pd.Series([100.0] * 25)  # Constant values
        result = FeatureBase.zscore(prices, period=20)

        # Should return NaN when std is zero
        assert pd.isna(result.iloc[-1])


class TestNormalizeToRange:
    """Test normalize_to_range calculation."""

    def test_normalize_default_range(self):
        """Test normalization to default 0-100 range."""
        prices = pd.Series([100.0, 110.0, 90.0, 105.0, 95.0])
        result = FeatureBase.normalize_to_range(prices, period=3)

        # Values should be in 0-100 range
        valid_results = result.dropna()
        assert (valid_results >= 0).all()
        assert (valid_results <= 100).all()

    def test_normalize_custom_range(self):
        """Test normalization to custom range."""
        prices = pd.Series([100.0, 110.0, 90.0, 105.0, 95.0])
        result = FeatureBase.normalize_to_range(
            prices, period=3, min_val=-50, max_val=50
        )

        valid_results = result.dropna()
        assert (valid_results >= -50).all()
        assert (valid_results <= 50).all()

    def test_normalize_handles_zero_range(self):
        """Test normalization handles constant values."""
        prices = pd.Series([100.0] * 10)
        result = FeatureBase.normalize_to_range(prices, period=5)

        # Should return NaN when range is zero
        assert pd.isna(result.iloc[-1])


class TestLagFeatures:
    """Test lag_features method."""

    @pytest.fixture
    def df(self) -> pd.DataFrame:
        """Create sample DataFrame."""
        return pd.DataFrame(
            {
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "rsi": [30.0, 40.0, 50.0, 60.0, 70.0],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )

    def test_lag_single_column(self, df):
        """Test lagging single column."""
        result = FeatureBase.lag_features(df, ["rsi"], lag=1)

        assert pd.isna(result["rsi"].iloc[0])
        assert result["rsi"].iloc[1] == 30.0
        assert result["rsi"].iloc[2] == 40.0

    def test_lag_multiple_columns(self, df):
        """Test lagging multiple columns."""
        result = FeatureBase.lag_features(df, ["rsi", "volume"], lag=1)

        assert pd.isna(result["rsi"].iloc[0])
        assert pd.isna(result["volume"].iloc[0])
        assert result["rsi"].iloc[1] == 30.0
        assert result["volume"].iloc[1] == 1000

    def test_lag_doesnt_affect_other_columns(self, df):
        """Test lagging doesn't affect unspecified columns."""
        result = FeatureBase.lag_features(df, ["rsi"], lag=1)

        # Close should be unchanged
        pd.testing.assert_series_equal(result["close"], df["close"])

    def test_lag_nonexistent_column(self, df):
        """Test lagging ignores nonexistent columns."""
        result = FeatureBase.lag_features(df, ["nonexistent"], lag=1)

        # Should return unchanged DataFrame
        pd.testing.assert_frame_equal(result, df)

    def test_lag_multiple_periods(self, df):
        """Test lagging by multiple periods."""
        result = FeatureBase.lag_features(df, ["rsi"], lag=2)

        assert pd.isna(result["rsi"].iloc[0])
        assert pd.isna(result["rsi"].iloc[1])
        assert result["rsi"].iloc[2] == 30.0


class TestAbstractMethods:
    """Test abstract method enforcement."""

    def test_cannot_instantiate_base_class(self):
        """Test FeatureBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            FeatureBase(Timeframe.D1)

    def test_must_implement_compute(self):
        """Test subclass must implement compute."""

        class IncompleteFeature(FeatureBase):
            def get_feature_names(self) -> List[str]:
                return []

        with pytest.raises(TypeError):
            IncompleteFeature(Timeframe.D1)

    def test_must_implement_get_feature_names(self):
        """Test subclass must implement get_feature_names."""

        class IncompleteFeature(FeatureBase):
            def compute(self, df: pd.DataFrame) -> pd.DataFrame:
                return df

        with pytest.raises(TypeError):
            IncompleteFeature(Timeframe.D1)


class TestConcreteFeatureCompute:
    """Test concrete feature computation."""

    @pytest.fixture
    def feature(self) -> ConcreteFeature:
        """Create feature instance."""
        return ConcreteFeature(Timeframe.D1)

    @pytest.fixture
    def ohlcv(self) -> pd.DataFrame:
        """Create sample OHLCV data."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(50) * 0.5)
        return pd.DataFrame(
            {
                "open": close + np.random.randn(50) * 0.2,
                "high": close + np.abs(np.random.randn(50) * 0.3),
                "low": close - np.abs(np.random.randn(50) * 0.3),
                "close": close,
                "volume": np.random.randint(10000, 100000, 50),
            },
            index=dates,
        )

    def test_compute_returns_dataframe(self, feature, ohlcv):
        """Test compute returns DataFrame."""
        result = feature.compute(ohlcv)
        assert isinstance(result, pd.DataFrame)

    def test_compute_adds_features(self, feature, ohlcv):
        """Test compute adds expected features."""
        result = feature.compute(ohlcv)
        feature_names = feature.get_feature_names()

        for name in feature_names:
            assert name in result.columns

    def test_compute_preserves_original_columns(self, feature, ohlcv):
        """Test compute preserves original columns."""
        result = feature.compute(ohlcv)

        for col in ohlcv.columns:
            assert col in result.columns

    def test_compute_doesnt_modify_input(self, feature, ohlcv):
        """Test compute doesn't modify input DataFrame."""
        original = ohlcv.copy()
        feature.compute(ohlcv)

        pd.testing.assert_frame_equal(ohlcv, original)
