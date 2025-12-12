# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for quantcore.data.synthetic module."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quantcore.data.synthetic import (
    SyntheticMarketConfig,
    generate_synthetic_ohlcv,
    generate_synthetic_multi_symbol,
    validate_synthetic_ohlcv,
    _get_bars_per_year,
)


class TestSyntheticMarketConfig:
    """Test SyntheticMarketConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SyntheticMarketConfig()

        assert config.periods == 2000
        assert config.freq == "1h"
        assert config.base_price == 100.0
        assert config.vol == 0.20
        assert config.seed == 42

    def test_custom_config(self):
        """Test custom configuration."""
        config = SyntheticMarketConfig(
            periods=500,
            freq="D",
            base_price=50.0,
            vol=0.30,
            seed=123,
        )

        assert config.periods == 500
        assert config.freq == "D"
        assert config.base_price == 50.0
        assert config.vol == 0.30
        assert config.seed == 123

    def test_validation_periods(self):
        """Test periods validation."""
        with pytest.raises(ValueError, match="periods must be >= 1"):
            SyntheticMarketConfig(periods=0)

        with pytest.raises(ValueError, match="periods must be >= 1"):
            SyntheticMarketConfig(periods=-1)

    def test_validation_base_price(self):
        """Test base_price validation."""
        with pytest.raises(ValueError, match="base_price must be positive"):
            SyntheticMarketConfig(base_price=0)

        with pytest.raises(ValueError, match="base_price must be positive"):
            SyntheticMarketConfig(base_price=-10)

    def test_validation_vol(self):
        """Test vol validation."""
        with pytest.raises(ValueError, match="vol must be non-negative"):
            SyntheticMarketConfig(vol=-0.1)

    def test_validation_regime_switch_prob(self):
        """Test regime_switch_prob validation."""
        with pytest.raises(ValueError, match="regime_switch_prob must be in"):
            SyntheticMarketConfig(regime_switch_prob=-0.1)

        with pytest.raises(ValueError, match="regime_switch_prob must be in"):
            SyntheticMarketConfig(regime_switch_prob=1.5)

    def test_validation_mean_reversion_strength(self):
        """Test mean_reversion_strength validation."""
        with pytest.raises(ValueError, match="mean_reversion_strength must be in"):
            SyntheticMarketConfig(mean_reversion_strength=-0.1)

        with pytest.raises(ValueError, match="mean_reversion_strength must be in"):
            SyntheticMarketConfig(mean_reversion_strength=1.5)


class TestGetBarsPerYear:
    """Test _get_bars_per_year helper function."""

    def test_hourly(self):
        """Test hourly bars per year."""
        assert _get_bars_per_year("1h") == 252 * 7
        assert _get_bars_per_year("H") == 252 * 7
        assert _get_bars_per_year("1H") == 252 * 7

    def test_four_hour(self):
        """Test 4-hour bars per year."""
        assert _get_bars_per_year("4h") == 252 * 2
        assert _get_bars_per_year("4H") == 252 * 2

    def test_daily(self):
        """Test daily bars per year."""
        assert _get_bars_per_year("d") == 252
        assert _get_bars_per_year("D") == 252
        assert _get_bars_per_year("1d") == 252
        assert _get_bars_per_year("1D") == 252

    def test_weekly(self):
        """Test weekly bars per year."""
        assert _get_bars_per_year("w") == 52
        assert _get_bars_per_year("W") == 52
        assert _get_bars_per_year("W-FRI") == 52

    def test_unknown_defaults_to_hourly(self):
        """Test unknown frequency defaults to hourly."""
        assert _get_bars_per_year("unknown") == 252 * 7


class TestGenerateSyntheticOHLCV:
    """Test generate_synthetic_ohlcv function."""

    def test_default_generation(self):
        """Test generation with defaults."""
        df = generate_synthetic_ohlcv()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2000
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_custom_periods(self):
        """Test generation with custom periods."""
        config = SyntheticMarketConfig(periods=100)
        df = generate_synthetic_ohlcv(config)

        assert len(df) == 100

    def test_deterministic_with_seed(self):
        """Test that generation is deterministic with same seed."""
        config1 = SyntheticMarketConfig(periods=100, seed=42)
        config2 = SyntheticMarketConfig(periods=100, seed=42)

        df1 = generate_synthetic_ohlcv(config1)
        df2 = generate_synthetic_ohlcv(config2)

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data."""
        config1 = SyntheticMarketConfig(periods=100, seed=42)
        config2 = SyntheticMarketConfig(periods=100, seed=123)

        df1 = generate_synthetic_ohlcv(config1)
        df2 = generate_synthetic_ohlcv(config2)

        assert not df1["close"].equals(df2["close"])

    def test_datetime_index(self):
        """Test that index is DatetimeIndex."""
        df = generate_synthetic_ohlcv()

        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is not None

    def test_ohlcv_constraints(self):
        """Test OHLCV constraints are satisfied."""
        config = SyntheticMarketConfig(periods=1000, seed=42)
        df = generate_synthetic_ohlcv(config)

        # High >= max(open, close)
        max_oc = df[["open", "close"]].max(axis=1)
        assert (df["high"] >= max_oc).all()

        # Low <= min(open, close)
        min_oc = df[["open", "close"]].min(axis=1)
        assert (df["low"] <= min_oc).all()

        # High >= Low
        assert (df["high"] >= df["low"]).all()

        # Positive prices
        assert (df[["open", "high", "low", "close"]] > 0).all().all()

        # Non-negative volume
        assert (df["volume"] >= 0).all()

    def test_no_nan_values(self):
        """Test that no NaN values are generated."""
        config = SyntheticMarketConfig(periods=500, seed=42)
        df = generate_synthetic_ohlcv(config)

        assert not df.isna().any().any()

    def test_different_frequencies(self):
        """Test generation with different frequencies."""
        for freq in ["1h", "4h", "D", "W"]:
            config = SyntheticMarketConfig(periods=100, freq=freq, seed=42)
            df = generate_synthetic_ohlcv(config)

            assert len(df) == 100
            assert isinstance(df.index, pd.DatetimeIndex)


class TestGenerateSyntheticMultiSymbol:
    """Test generate_synthetic_multi_symbol function."""

    def test_multiple_symbols(self):
        """Test generating data for multiple symbols."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        result = generate_synthetic_multi_symbol(symbols)

        assert len(result) == 3
        assert "AAPL" in result
        assert "MSFT" in result
        assert "GOOGL" in result

    def test_each_symbol_has_data(self):
        """Test each symbol has valid data."""
        symbols = ["SYM1", "SYM2"]
        config = SyntheticMarketConfig(periods=100)
        result = generate_synthetic_multi_symbol(symbols, config)

        for symbol in symbols:
            df = result[symbol]
            assert len(df) == 100
            assert "close" in df.columns

    def test_symbols_have_different_data(self):
        """Test that different symbols have different data."""
        symbols = ["SYM1", "SYM2"]
        result = generate_synthetic_multi_symbol(symbols)

        assert not result["SYM1"]["close"].equals(result["SYM2"]["close"])

    def test_deterministic_with_seed(self):
        """Test multi-symbol generation is deterministic."""
        symbols = ["A", "B"]
        config = SyntheticMarketConfig(periods=50, seed=42)

        result1 = generate_synthetic_multi_symbol(symbols, config)
        result2 = generate_synthetic_multi_symbol(symbols, config)

        pd.testing.assert_frame_equal(result1["A"], result2["A"])
        pd.testing.assert_frame_equal(result1["B"], result2["B"])


class TestValidateSyntheticOHLCV:
    """Test validate_synthetic_ohlcv function."""

    def test_valid_data(self):
        """Test validation of valid data."""
        df = generate_synthetic_ohlcv(SyntheticMarketConfig(periods=100))
        is_valid, errors = validate_synthetic_ohlcv(df)

        assert is_valid is True
        assert len(errors) == 0

    def test_missing_columns(self):
        """Test validation with missing columns."""
        df = pd.DataFrame({"open": [100], "close": [101]})
        is_valid, errors = validate_synthetic_ohlcv(df)

        assert is_valid is False
        assert any("Missing columns" in e for e in errors)

    def test_nan_values(self):
        """Test validation with NaN values."""
        df = pd.DataFrame(
            {
                "open": [100, np.nan],
                "high": [101, 102],
                "low": [99, 98],
                "close": [100.5, 101.5],
                "volume": [1000, 2000],
            }
        )
        is_valid, errors = validate_synthetic_ohlcv(df)

        assert is_valid is False
        assert any("NaN" in e for e in errors)

    def test_invalid_high_low(self):
        """Test validation with high < low."""
        df = pd.DataFrame(
            {
                "open": [100],
                "high": [99],  # Invalid: less than low
                "low": [101],
                "close": [100],
                "volume": [1000],
            }
        )
        is_valid, errors = validate_synthetic_ohlcv(df)

        assert is_valid is False
        assert any("high < low" in e for e in errors)

    def test_invalid_high_less_than_close(self):
        """Test validation with high < close."""
        df = pd.DataFrame(
            {
                "open": [100],
                "high": [100],  # Invalid: less than close
                "low": [99],
                "close": [105],
                "volume": [1000],
            }
        )
        is_valid, errors = validate_synthetic_ohlcv(df)

        assert is_valid is False
        assert any("high < max(open, close)" in e for e in errors)

    def test_invalid_low_greater_than_close(self):
        """Test validation with low > close."""
        df = pd.DataFrame(
            {
                "open": [100],
                "high": [110],
                "low": [105],  # Invalid: greater than close
                "close": [100],
                "volume": [1000],
            }
        )
        is_valid, errors = validate_synthetic_ohlcv(df)

        assert is_valid is False
        assert any("low > min(open, close)" in e for e in errors)

    def test_negative_prices(self):
        """Test validation with negative prices."""
        df = pd.DataFrame(
            {
                "open": [-100],
                "high": [101],
                "low": [99],
                "close": [100],
                "volume": [1000],
            }
        )
        is_valid, errors = validate_synthetic_ohlcv(df)

        assert is_valid is False
        assert any("non-positive prices" in e for e in errors)

    def test_negative_volume(self):
        """Test validation with negative volume."""
        df = pd.DataFrame(
            {
                "open": [100],
                "high": [101],
                "low": [99],
                "close": [100],
                "volume": [-1000],
            }
        )
        is_valid, errors = validate_synthetic_ohlcv(df)

        assert is_valid is False
        assert any("negative volume" in e for e in errors)

    def test_non_datetime_index(self):
        """Test validation with non-DatetimeIndex."""
        df = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [101, 102],
                "low": [99, 100],
                "close": [100, 101],
                "volume": [1000, 2000],
            },
            index=[0, 1],
        )  # Integer index

        is_valid, errors = validate_synthetic_ohlcv(df)

        assert is_valid is False
        assert any("DatetimeIndex" in e for e in errors)
