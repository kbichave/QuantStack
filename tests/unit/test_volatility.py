# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for quantcore.features.volatility module."""

import numpy as np
import pandas as pd
import pytest

from quantcore.config.timeframes import Timeframe
from quantcore.features.volatility import VolatilityFeatures


class TestVolatilityFeatures:
    """Test VolatilityFeatures class."""

    @pytest.fixture
    def features(self) -> VolatilityFeatures:
        """Create volatility features instance."""
        return VolatilityFeatures(Timeframe.D1)

    @pytest.fixture
    def ohlcv(self) -> pd.DataFrame:
        """Create sample OHLCV data."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        np.random.seed(42)

        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        high = close + np.abs(np.random.randn(100) * 0.5)
        low = close - np.abs(np.random.randn(100) * 0.5)
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

    def test_compute_returns_dataframe(self, features, ohlcv):
        """Test compute returns DataFrame."""
        result = features.compute(ohlcv)
        assert isinstance(result, pd.DataFrame)

    def test_compute_adds_all_features(self, features, ohlcv):
        """Test all expected features are added."""
        result = features.compute(ohlcv)
        feature_names = features.get_feature_names()

        for name in feature_names:
            assert name in result.columns, f"Missing feature: {name}"

    def test_feature_count(self, features):
        """Test correct number of features."""
        names = features.get_feature_names()
        assert len(names) >= 15  # At least 15 volatility features


class TestTrueRange:
    """Test True Range calculation."""

    @pytest.fixture
    def features(self) -> VolatilityFeatures:
        return VolatilityFeatures(Timeframe.D1)

    def test_true_range_positive(self, features):
        """Test True Range is always positive."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(50) * 0.5)

        df = pd.DataFrame(
            {
                "open": close,
                "high": close + np.abs(np.random.randn(50)),
                "low": close - np.abs(np.random.randn(50)),
                "close": close,
                "volume": [1000] * 50,
            },
            index=dates,
        )

        result = features.compute(df)
        tr = result["true_range"].dropna()

        assert (tr >= 0).all()

    def test_true_range_with_gap(self, features):
        """Test True Range accounts for gaps."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")

        # Create gap up scenario
        df = pd.DataFrame(
            {
                "open": [100, 105, 103, 104, 102],  # Gap up on day 2
                "high": [101, 106, 104, 105, 103],
                "low": [99, 104, 102, 103, 101],
                "close": [100, 105, 103, 104, 102],
                "volume": [1000] * 5,
            },
            index=dates,
        )

        result = features.compute(df)

        # True range on gap day should be larger than high-low
        high_low_range = df["high"].iloc[1] - df["low"].iloc[1]  # 106 - 104 = 2
        true_range = result["true_range"].iloc[1]

        # True range should account for gap from previous close (100)
        # TR = max(106-104, |106-100|, |104-100|) = max(2, 6, 4) = 6
        assert true_range >= high_low_range


class TestATR:
    """Test Average True Range calculation."""

    @pytest.fixture
    def features(self) -> VolatilityFeatures:
        return VolatilityFeatures(Timeframe.D1)

    @pytest.fixture
    def ohlcv(self) -> pd.DataFrame:
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(50) * 0.5)
        return pd.DataFrame(
            {
                "open": close,
                "high": close + np.abs(np.random.randn(50)),
                "low": close - np.abs(np.random.randn(50)),
                "close": close,
                "volume": [1000] * 50,
            },
            index=dates,
        )

    def test_atr_positive(self, features, ohlcv):
        """Test ATR is always positive."""
        result = features.compute(ohlcv)
        atr = result["atr"].dropna()

        assert (atr >= 0).all()

    def test_atr_pct_present(self, features, ohlcv):
        """Test ATR percentage is computed."""
        result = features.compute(ohlcv)

        assert "atr_pct" in result.columns
        atr_pct = result["atr_pct"].dropna()
        assert (atr_pct >= 0).all()


class TestBollingerBands:
    """Test Bollinger Bands calculation."""

    @pytest.fixture
    def features(self) -> VolatilityFeatures:
        return VolatilityFeatures(Timeframe.D1)

    @pytest.fixture
    def ohlcv(self) -> pd.DataFrame:
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(50) * 0.5)
        return pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": [1000] * 50,
            },
            index=dates,
        )

    def test_bollinger_bands_order(self, features, ohlcv):
        """Test Bollinger Bands maintain correct order."""
        result = features.compute(ohlcv)

        valid_idx = result["bb_upper"].notna()
        assert (
            result.loc[valid_idx, "bb_upper"] >= result.loc[valid_idx, "bb_middle"]
        ).all()
        assert (
            result.loc[valid_idx, "bb_middle"] >= result.loc[valid_idx, "bb_lower"]
        ).all()

    def test_bb_width_positive(self, features, ohlcv):
        """Test Bollinger Band width is positive."""
        result = features.compute(ohlcv)
        bb_width = result["bb_width"].dropna()

        assert (bb_width >= 0).all()

    def test_bb_position_range(self, features, ohlcv):
        """Test Bollinger Band position is roughly 0-100."""
        result = features.compute(ohlcv)
        bb_position = result["bb_position"].dropna()

        # Most values should be within bands (0-100)
        # Allow some values outside due to price breakouts
        within_bands = (bb_position >= -50) & (bb_position <= 150)
        assert within_bands.mean() > 0.8  # At least 80% within extended range

    def test_price_above_bb_binary(self, features, ohlcv):
        """Test price_above_bb is binary."""
        result = features.compute(ohlcv)
        assert set(result["price_above_bb"].unique()).issubset({0, 1})

    def test_price_below_bb_binary(self, features, ohlcv):
        """Test price_below_bb is binary."""
        result = features.compute(ohlcv)
        assert set(result["price_below_bb"].unique()).issubset({0, 1})


class TestRealizedVolatility:
    """Test realized volatility calculation."""

    @pytest.fixture
    def features(self) -> VolatilityFeatures:
        return VolatilityFeatures(Timeframe.D1)

    def test_realized_vol_non_negative(self, features):
        """Test realized volatility is non-negative."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(50) * 0.5)

        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": [1000] * 50,
            },
            index=dates,
        )

        result = features.compute(df)
        realized_vol = result["realized_vol"].dropna()

        assert (realized_vol >= 0).all()

    def test_high_vol_regime_detection(self, features):
        """Test volatility regime is computed."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        np.random.seed(42)

        close = 100 + np.cumsum(np.random.randn(100) * 0.5)

        df = pd.DataFrame(
            {
                "open": close,
                "high": close + np.abs(np.random.randn(100) * 0.5),
                "low": close - np.abs(np.random.randn(100) * 0.5),
                "close": close,
                "volume": [1000] * 100,
            },
            index=dates,
        )

        result = features.compute(df)

        # Vol regime should be present and have valid values
        assert "vol_regime" in result.columns
        vol_regime = result["vol_regime"].dropna()
        valid_values = vol_regime.unique()
        # Should be -1 (low), 0 (normal), or 1 (high)
        assert all(v in [-1, 0, 1, -1.0, 0.0, 1.0] or pd.isna(v) for v in valid_values)


class TestVolatilityZScore:
    """Test volatility z-score calculation."""

    @pytest.fixture
    def features(self) -> VolatilityFeatures:
        return VolatilityFeatures(Timeframe.D1)

    @pytest.fixture
    def ohlcv(self) -> pd.DataFrame:
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        return pd.DataFrame(
            {
                "open": close,
                "high": close + np.abs(np.random.randn(100)),
                "low": close - np.abs(np.random.randn(100)),
                "close": close,
                "volume": [1000] * 100,
            },
            index=dates,
        )

    def test_vol_zscore_computed(self, features, ohlcv):
        """Test vol_zscore is computed."""
        result = features.compute(ohlcv)
        assert "vol_zscore" in result.columns

    def test_vol_regime_values(self, features, ohlcv):
        """Test vol_regime has expected values."""
        result = features.compute(ohlcv)
        vol_regime = result["vol_regime"].dropna()

        # Should be -1 (low), 0 (normal), or 1 (high)
        valid_values = vol_regime.dropna()
        assert set(valid_values.unique()).issubset({-1, 0, 1, -1.0, 0.0, 1.0})


class TestHVolRatio:
    """Test historical volatility ratio calculation."""

    @pytest.fixture
    def features(self) -> VolatilityFeatures:
        return VolatilityFeatures(Timeframe.D1)

    @pytest.fixture
    def ohlcv(self) -> pd.DataFrame:
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        return pd.DataFrame(
            {
                "open": close,
                "high": close + np.abs(np.random.randn(100)),
                "low": close - np.abs(np.random.randn(100)),
                "close": close,
                "volume": [1000] * 100,
            },
            index=dates,
        )

    def test_hvol_ratio_computed(self, features, ohlcv):
        """Test hvol_ratio is computed."""
        result = features.compute(ohlcv)
        assert "hvol_ratio" in result.columns

    def test_hvol_ratio_positive(self, features, ohlcv):
        """Test hvol_ratio is positive."""
        result = features.compute(ohlcv)
        hvol = result["hvol_ratio"].dropna()

        assert (hvol > 0).all()


class TestIntradayRange:
    """Test intraday range calculations."""

    @pytest.fixture
    def features(self) -> VolatilityFeatures:
        return VolatilityFeatures(Timeframe.D1)

    @pytest.fixture
    def ohlcv(self) -> pd.DataFrame:
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        close = pd.Series(np.linspace(100, 110, 50), index=dates)
        return pd.DataFrame(
            {
                "open": close,
                "high": close + 1,  # Fixed range of 2
                "low": close - 1,
                "close": close,
                "volume": [1000] * 50,
            },
            index=dates,
        )

    def test_intraday_range_computed(self, features, ohlcv):
        """Test intraday_range is computed."""
        result = features.compute(ohlcv)
        assert "intraday_range" in result.columns

    def test_intraday_range_calculation(self, features, ohlcv):
        """Test intraday range calculation."""
        result = features.compute(ohlcv)

        # For fixed high-low of 2 and close around 100, should be ~2%
        intraday_range = result["intraday_range"]
        assert (intraday_range > 0).all()

    def test_range_expansion_computed(self, features, ohlcv):
        """Test range_expansion is computed."""
        result = features.compute(ohlcv)
        assert "range_expansion" in result.columns
        assert "range_ma" in result.columns
