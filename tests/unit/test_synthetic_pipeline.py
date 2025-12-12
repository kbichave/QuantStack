"""
Tests for the synthetic data pipeline.

These tests verify that the synthetic data generation and full pipeline
work correctly. Uses small datasets for fast CI execution.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from quantcore.config.timeframes import Timeframe
from quantcore.data.synthetic import (
    SyntheticMarketConfig,
    generate_synthetic_ohlcv,
    generate_synthetic_multi_symbol,
    validate_synthetic_ohlcv,
)


def resample_all_timeframes(df_1h: pd.DataFrame) -> dict:
    """Helper function to resample 1H data to all timeframes.

    Args:
        df_1h: 1-hour OHLCV DataFrame

    Returns:
        Dictionary of Timeframe -> DataFrame
    """
    from quantcore.data.resampler import TimeframeResampler

    resampler = TimeframeResampler()

    return {
        Timeframe.H1: df_1h.copy(),
        Timeframe.H4: resampler.resample_to_higher_tf(df_1h, Timeframe.H4),
        Timeframe.D1: resampler.resample_to_higher_tf(df_1h, Timeframe.D1),
        Timeframe.W1: resampler.resample_to_higher_tf(df_1h, Timeframe.W1),
    }


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_config():
    """Small config for fast tests."""
    return SyntheticMarketConfig(
        start=datetime(2023, 1, 3, 9, 0),
        periods=300,
        freq="1h",
        base_price=100.0,
        vol=0.20,
        trend_strength=0.0002,
        regime_switch_prob=0.03,
        seed=42,
    )


# =============================================================================
# Synthetic OHLCV Generation Tests
# =============================================================================


class TestSyntheticOHLCVGeneration:
    """Tests for generate_synthetic_ohlcv function."""

    def test_generates_correct_number_of_bars(self, small_config):
        """Should generate exactly the requested number of bars."""
        df = generate_synthetic_ohlcv(small_config)
        assert len(df) == small_config.periods

    def test_has_required_columns(self, small_config):
        """Should have all OHLCV columns."""
        df = generate_synthetic_ohlcv(small_config)
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_has_datetime_index(self, small_config):
        """Should have DatetimeIndex."""
        df = generate_synthetic_ohlcv(small_config)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_high_low_constraints(self, small_config):
        """High should be >= open,close; Low should be <= open,close."""
        df = generate_synthetic_ohlcv(small_config)

        # High >= max(open, close)
        assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()

        # Low <= min(open, close)
        assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()

        # High >= Low
        assert (df["high"] >= df["low"]).all()

    def test_positive_prices(self, small_config):
        """All prices should be positive."""
        df = generate_synthetic_ohlcv(small_config)
        for col in ["open", "high", "low", "close"]:
            assert (df[col] > 0).all(), f"{col} has non-positive values"

    def test_non_negative_volume(self, small_config):
        """Volume should be non-negative."""
        df = generate_synthetic_ohlcv(small_config)
        assert (df["volume"] >= 0).all()

    def test_no_nan_values(self, small_config):
        """Should have no NaN values."""
        df = generate_synthetic_ohlcv(small_config)
        assert not df.isna().any().any()

    def test_deterministic_with_seed(self):
        """Same seed should produce identical data."""
        config1 = SyntheticMarketConfig(periods=100, seed=123)
        config2 = SyntheticMarketConfig(periods=100, seed=123)

        df1 = generate_synthetic_ohlcv(config1)
        df2 = generate_synthetic_ohlcv(config2)

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_different_data(self):
        """Different seeds should produce different data."""
        config1 = SyntheticMarketConfig(periods=100, seed=123)
        config2 = SyntheticMarketConfig(periods=100, seed=456)

        df1 = generate_synthetic_ohlcv(config1)
        df2 = generate_synthetic_ohlcv(config2)

        # Close prices should be different
        assert not np.allclose(df1["close"].values, df2["close"].values)

    def test_default_config_works(self):
        """Should work with default configuration."""
        df = generate_synthetic_ohlcv()
        assert len(df) == 2000  # Default periods
        is_valid, errors = validate_synthetic_ohlcv(df)
        assert is_valid, f"Validation failed: {errors}"

    def test_starts_at_base_price(self, small_config):
        """First bar should start at base price."""
        df = generate_synthetic_ohlcv(small_config)
        assert df["close"].iloc[0] == pytest.approx(small_config.base_price)
        assert df["open"].iloc[0] == pytest.approx(small_config.base_price)


class TestSyntheticConfigValidation:
    """Tests for SyntheticMarketConfig validation."""

    def test_invalid_periods(self):
        """Should reject periods < 1."""
        with pytest.raises(ValueError, match="periods must be >= 1"):
            SyntheticMarketConfig(periods=0)

    def test_invalid_base_price(self):
        """Should reject non-positive base price."""
        with pytest.raises(ValueError, match="base_price must be positive"):
            SyntheticMarketConfig(base_price=0)

        with pytest.raises(ValueError, match="base_price must be positive"):
            SyntheticMarketConfig(base_price=-100)

    def test_invalid_volatility(self):
        """Should reject negative volatility."""
        with pytest.raises(ValueError, match="vol must be non-negative"):
            SyntheticMarketConfig(vol=-0.1)

    def test_invalid_regime_switch_prob(self):
        """Should reject probability outside [0, 1]."""
        with pytest.raises(ValueError, match="regime_switch_prob must be in"):
            SyntheticMarketConfig(regime_switch_prob=1.5)

        with pytest.raises(ValueError, match="regime_switch_prob must be in"):
            SyntheticMarketConfig(regime_switch_prob=-0.1)


class TestValidateSyntheticOHLCV:
    """Tests for validate_synthetic_ohlcv function."""

    def test_valid_data_passes(self, small_config):
        """Valid synthetic data should pass validation."""
        df = generate_synthetic_ohlcv(small_config)
        is_valid, errors = validate_synthetic_ohlcv(df)
        assert is_valid, f"Should be valid but got errors: {errors}"

    def test_detects_missing_columns(self):
        """Should detect missing columns."""
        df = pd.DataFrame({"open": [100], "high": [101]})
        is_valid, errors = validate_synthetic_ohlcv(df)
        assert not is_valid
        assert any("Missing columns" in e for e in errors)

    def test_detects_invalid_high_low(self):
        """Should detect high < low violations."""
        df = pd.DataFrame(
            {
                "open": [100],
                "high": [98],  # Invalid: less than low
                "low": [99],
                "close": [100],
                "volume": [1000],
            },
            index=pd.date_range("2024-01-01", periods=1, freq="1h"),
        )

        is_valid, errors = validate_synthetic_ohlcv(df)
        assert not is_valid
        assert any("high < low" in e for e in errors)


class TestMultiSymbolGeneration:
    """Tests for generate_synthetic_multi_symbol function."""

    def test_generates_all_symbols(self):
        """Should generate data for all requested symbols."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        config = SyntheticMarketConfig(periods=100, seed=42)

        data = generate_synthetic_multi_symbol(symbols, config)

        assert len(data) == len(symbols)
        for symbol in symbols:
            assert symbol in data
            assert len(data[symbol]) == config.periods

    def test_different_symbols_different_data(self):
        """Each symbol should have different data."""
        symbols = ["AAPL", "GOOGL"]
        config = SyntheticMarketConfig(periods=100, seed=42)

        data = generate_synthetic_multi_symbol(symbols, config)

        # Close prices should be different
        assert not np.allclose(
            data["AAPL"]["close"].values,
            data["GOOGL"]["close"].values,
        )


# =============================================================================
# Resampling Tests
# =============================================================================


class TestSyntheticResampling:
    """Tests for resampling synthetic data."""

    def test_resampling_produces_all_timeframes(self, small_config):
        """Resampling should produce all timeframes."""
        df_1h = generate_synthetic_ohlcv(small_config)
        data = resample_all_timeframes(df_1h)

        assert Timeframe.H1 in data
        assert Timeframe.H4 in data
        assert Timeframe.D1 in data
        assert Timeframe.W1 in data

    def test_resampled_data_is_valid(self, small_config):
        """Resampled data should maintain OHLCV validity."""
        df_1h = generate_synthetic_ohlcv(small_config)
        data = resample_all_timeframes(df_1h)

        for tf, df in data.items():
            if len(df) > 0:
                # Check OHLCV constraints
                assert (df["high"] >= df["low"]).all(), f"{tf} has invalid high/low"
                assert (df["high"] >= df["open"]).all(), f"{tf} has high < open"
                assert (df["high"] >= df["close"]).all(), f"{tf} has high < close"

    def test_higher_timeframes_have_fewer_bars(self, small_config):
        """Higher timeframes should have fewer bars."""
        df_1h = generate_synthetic_ohlcv(small_config)
        data = resample_all_timeframes(df_1h)

        # H4 should have ~1/4 the bars of H1
        if len(data[Timeframe.H4]) > 0:
            assert len(data[Timeframe.H4]) < len(data[Timeframe.H1])

        # D1 should have fewer than H4
        if len(data[Timeframe.D1]) > 0 and len(data[Timeframe.H4]) > 0:
            assert len(data[Timeframe.D1]) <= len(data[Timeframe.H4])


# =============================================================================
# Feature Computation Tests
# =============================================================================


class TestSyntheticFeatures:
    """Tests for computing features on synthetic data."""

    def test_features_computed_without_error(self, small_config):
        """Feature computation should not raise errors."""
        from quantcore.features.factory import MultiTimeframeFeatureFactory

        df_1h = generate_synthetic_ohlcv(small_config)
        data = resample_all_timeframes(df_1h)

        factory = MultiTimeframeFeatureFactory(include_waves=False, include_rrg=False)
        features = factory.compute_all_timeframes(data, lag_features=True)

        # Should have features for each timeframe
        assert Timeframe.H1 in features

    def test_features_include_expected_columns(self, small_config):
        """Features should include key expected columns."""
        from quantcore.features.factory import MultiTimeframeFeatureFactory

        df_1h = generate_synthetic_ohlcv(small_config)
        data = resample_all_timeframes(df_1h)

        factory = MultiTimeframeFeatureFactory(include_waves=False, include_rrg=False)
        features = factory.compute_all_timeframes(data, lag_features=True)

        h1_features = features[Timeframe.H1]

        # Should have ATR (used for labels and sizing)
        assert "atr" in h1_features.columns

        # Should have RSI (common momentum indicator)
        assert "rsi" in h1_features.columns


class TestSyntheticMovingAverages:
    """Test all moving average indicators on synthetic data."""

    def test_sma_on_synthetic(self):
        """SMA should be computed correctly on synthetic data."""
        from quantcore.features.factory import MultiTimeframeFeatureFactory

        # Use larger dataset for H1 timeframe only (enough for all indicators)
        config = SyntheticMarketConfig(periods=300, seed=42)
        df_1h = generate_synthetic_ohlcv(config)

        factory = MultiTimeframeFeatureFactory(
            include_waves=False,
            include_rrg=False,
            enable_oscillators=False,
            enable_volatility_indicators=False,
            enable_volume_indicators=False,
        )
        # Only compute features for H1 (skip higher timeframes with fewer bars)
        features = factory._compute_features_for_timeframe(df_1h, Timeframe.H1)

        # Check SMA for standard periods (matching TechnicalIndicators.ma_periods)
        for period in [10, 20, 50, 200]:
            col = f"sma_{period}"
            assert col in features.columns, f"Missing {col}"
            sma = features[col].dropna()
            assert len(sma) > 0, f"{col} has no valid values"

    def test_ema_on_synthetic(self):
        """EMA should be computed correctly on synthetic data."""
        from quantcore.features.factory import MultiTimeframeFeatureFactory

        # Use larger dataset for H1 timeframe only
        config = SyntheticMarketConfig(periods=300, seed=42)
        df_1h = generate_synthetic_ohlcv(config)

        factory = MultiTimeframeFeatureFactory(
            include_waves=False,
            include_rrg=False,
            enable_oscillators=False,
            enable_volatility_indicators=False,
            enable_volume_indicators=False,
        )
        features = factory._compute_features_for_timeframe(df_1h, Timeframe.H1)

        # Check EMA for standard periods (matching TechnicalIndicators.ma_periods)
        for period in [10, 20, 50, 200]:
            col = f"ema_{period}"
            assert col in features.columns, f"Missing {col}"
            ema = features[col].dropna()
            assert len(ema) > 0, f"{col} has no valid values"

    def test_wma_dema_tema_on_synthetic(self):
        """WMA, DEMA, TEMA should be computed correctly on synthetic data."""
        from quantcore.features.factory import MultiTimeframeFeatureFactory

        # Use larger dataset for H1 timeframe only
        config = SyntheticMarketConfig(periods=300, seed=42)
        df_1h = generate_synthetic_ohlcv(config)

        factory = MultiTimeframeFeatureFactory(
            include_waves=False,
            include_rrg=False,
            enable_oscillators=False,
            enable_volatility_indicators=False,
            enable_volume_indicators=False,
        )
        features = factory._compute_features_for_timeframe(df_1h, Timeframe.H1)

        # Check WMA, DEMA, TEMA (matching TechnicalIndicators.ma_periods)
        for period in [10, 20, 50, 200]:
            for ma_type in ["wma", "dema", "tema"]:
                col = f"{ma_type}_{period}"
                assert col in features.columns, f"Missing {col}"

    def test_vwap_on_synthetic(self):
        """VWAP should be computed correctly on synthetic data."""
        from quantcore.features.factory import MultiTimeframeFeatureFactory

        # Use larger dataset for H1 timeframe only
        config = SyntheticMarketConfig(periods=300, seed=42)
        df_1h = generate_synthetic_ohlcv(config)

        factory = MultiTimeframeFeatureFactory(
            include_waves=False,
            include_rrg=False,
            enable_oscillators=False,
            enable_volatility_indicators=False,
            enable_volume_indicators=False,
        )
        features = factory._compute_features_for_timeframe(df_1h, Timeframe.H1)

        assert "vwap" in features.columns
        vwap = features["vwap"].dropna()
        assert len(vwap) > 0
        # VWAP should be within reasonable range of price
        assert (vwap <= features["high"].loc[vwap.index] * 1.5).all()
        assert (vwap >= features["low"].loc[vwap.index] * 0.5).all()


class TestSyntheticOscillators:
    """Test all oscillator indicators on synthetic data."""

    def test_rsi_on_synthetic(self, small_config):
        """RSI should be bounded 0-100 on synthetic data."""
        from quantcore.features.factory import MultiTimeframeFeatureFactory

        df_1h = generate_synthetic_ohlcv(small_config)
        data = resample_all_timeframes(df_1h)

        factory = MultiTimeframeFeatureFactory(
            include_waves=False,
            include_rrg=False,
            enable_moving_averages=False,
            enable_volatility_indicators=False,
            enable_volume_indicators=False,
        )
        features = factory.compute_all_timeframes(data, lag_features=False)
        h1_features = features[Timeframe.H1]

        assert "rsi" in h1_features.columns
        rsi = h1_features["rsi"].dropna()
        assert len(rsi) > 0
        # RSI must be 0-100
        assert (rsi >= 0).all(), "RSI has values below 0"
        assert (rsi <= 100).all(), "RSI has values above 100"

    def test_macd_on_synthetic(self, small_config):
        """MACD components should be computed correctly on synthetic data."""
        from quantcore.features.factory import MultiTimeframeFeatureFactory

        df_1h = generate_synthetic_ohlcv(small_config)
        data = resample_all_timeframes(df_1h)

        factory = MultiTimeframeFeatureFactory(
            include_waves=False,
            include_rrg=False,
            enable_moving_averages=False,
            enable_volatility_indicators=False,
            enable_volume_indicators=False,
        )
        features = factory.compute_all_timeframes(data, lag_features=False)
        h1_features = features[Timeframe.H1]

        # Check all MACD components exist
        assert "macd_line" in h1_features.columns
        assert "macd_signal" in h1_features.columns
        assert "macd_histogram" in h1_features.columns

        # Histogram should be line - signal
        macd_line = h1_features["macd_line"].dropna()
        macd_signal = h1_features["macd_signal"].dropna()
        macd_hist = h1_features["macd_histogram"].dropna()

        # Check on common indices
        common_idx = macd_line.index.intersection(macd_signal.index).intersection(
            macd_hist.index
        )
        if len(common_idx) > 0:
            expected_hist = macd_line.loc[common_idx] - macd_signal.loc[common_idx]
            np.testing.assert_allclose(
                macd_hist.loc[common_idx],
                expected_hist,
                rtol=1e-5,
            )

    def test_stochastic_on_synthetic(self, small_config):
        """Stochastic oscillator should be bounded 0-100 on synthetic data."""
        from quantcore.features.factory import MultiTimeframeFeatureFactory

        df_1h = generate_synthetic_ohlcv(small_config)
        data = resample_all_timeframes(df_1h)

        factory = MultiTimeframeFeatureFactory(
            include_waves=False,
            include_rrg=False,
            enable_moving_averages=False,
            enable_volatility_indicators=False,
            enable_volume_indicators=False,
        )
        features = factory.compute_all_timeframes(data, lag_features=False)
        h1_features = features[Timeframe.H1]

        assert "stoch_k" in h1_features.columns
        assert "stoch_d" in h1_features.columns

        stoch_k = h1_features["stoch_k"].dropna()
        stoch_d = h1_features["stoch_d"].dropna()

        # Both should be 0-100
        if len(stoch_k) > 0:
            assert (stoch_k >= 0).all()
            assert (stoch_k <= 100).all()
        if len(stoch_d) > 0:
            assert (stoch_d >= 0).all()
            assert (stoch_d <= 100).all()

    def test_adx_on_synthetic(self, small_config):
        """ADX should be bounded 0-100 on synthetic data."""
        from quantcore.features.factory import MultiTimeframeFeatureFactory

        df_1h = generate_synthetic_ohlcv(small_config)
        data = resample_all_timeframes(df_1h)

        factory = MultiTimeframeFeatureFactory(
            include_waves=False,
            include_rrg=False,
            enable_moving_averages=False,
            enable_volatility_indicators=False,
            enable_volume_indicators=False,
        )
        features = factory.compute_all_timeframes(data, lag_features=False)
        h1_features = features[Timeframe.H1]

        assert "adx" in h1_features.columns
        adx = h1_features["adx"].dropna()
        if len(adx) > 0:
            assert (adx >= 0).all()
            assert (adx <= 100).all()

    def test_williams_r_on_synthetic(self, small_config):
        """Williams %R should be bounded -100 to 0 on synthetic data."""
        from quantcore.features.factory import MultiTimeframeFeatureFactory

        df_1h = generate_synthetic_ohlcv(small_config)
        data = resample_all_timeframes(df_1h)

        factory = MultiTimeframeFeatureFactory(
            include_waves=False,
            include_rrg=False,
            enable_moving_averages=False,
            enable_volatility_indicators=False,
            enable_volume_indicators=False,
        )
        features = factory.compute_all_timeframes(data, lag_features=False)
        h1_features = features[Timeframe.H1]

        assert "willr" in h1_features.columns
        willr = h1_features["willr"].dropna()
        if len(willr) > 0:
            assert (willr >= -100).all()
            assert (willr <= 0).all()


class TestSyntheticVolatility:
    """Test all volatility indicators on synthetic data."""

    def test_bollinger_bands_on_synthetic(self, small_config):
        """Bollinger Bands should maintain ordering on synthetic data."""
        from quantcore.features.factory import MultiTimeframeFeatureFactory

        df_1h = generate_synthetic_ohlcv(small_config)
        data = resample_all_timeframes(df_1h)

        factory = MultiTimeframeFeatureFactory(
            include_waves=False,
            include_rrg=False,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volume_indicators=False,
        )
        features = factory.compute_all_timeframes(data, lag_features=False)
        h1_features = features[Timeframe.H1]

        assert "bb_upper" in h1_features.columns
        assert "bb_middle" in h1_features.columns
        assert "bb_lower" in h1_features.columns

        # Check ordering: upper >= middle >= lower
        valid_rows = h1_features[["bb_upper", "bb_middle", "bb_lower"]].dropna()
        if len(valid_rows) > 0:
            assert (valid_rows["bb_upper"] >= valid_rows["bb_middle"]).all()
            assert (valid_rows["bb_middle"] >= valid_rows["bb_lower"]).all()

    def test_atr_on_synthetic(self, small_config):
        """ATR should be positive on synthetic data."""
        from quantcore.features.factory import MultiTimeframeFeatureFactory

        df_1h = generate_synthetic_ohlcv(small_config)
        data = resample_all_timeframes(df_1h)

        factory = MultiTimeframeFeatureFactory(
            include_waves=False,
            include_rrg=False,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volume_indicators=False,
        )
        features = factory.compute_all_timeframes(data, lag_features=False)
        h1_features = features[Timeframe.H1]

        assert "atr" in h1_features.columns
        assert "natr" in h1_features.columns

        atr = h1_features["atr"].dropna()
        natr = h1_features["natr"].dropna()

        # Both should be positive
        if len(atr) > 0:
            assert (atr > 0).all(), "ATR should be positive"
        if len(natr) > 0:
            assert (natr > 0).all(), "NATR should be positive"

    def test_sar_on_synthetic(self, small_config):
        """Parabolic SAR should exist on synthetic data."""
        from quantcore.features.factory import MultiTimeframeFeatureFactory

        df_1h = generate_synthetic_ohlcv(small_config)
        data = resample_all_timeframes(df_1h)

        factory = MultiTimeframeFeatureFactory(
            include_waves=False,
            include_rrg=False,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volume_indicators=False,
        )
        features = factory.compute_all_timeframes(data, lag_features=False)
        h1_features = features[Timeframe.H1]

        assert "sar" in h1_features.columns
        sar = h1_features["sar"].dropna()
        assert len(sar) > 0, "SAR should have valid values"


class TestSyntheticVolume:
    """Test all volume indicators on synthetic data."""

    def test_obv_on_synthetic(self, small_config):
        """OBV should be computed correctly on synthetic data."""
        from quantcore.features.factory import MultiTimeframeFeatureFactory

        df_1h = generate_synthetic_ohlcv(small_config)
        data = resample_all_timeframes(df_1h)

        factory = MultiTimeframeFeatureFactory(
            include_waves=False,
            include_rrg=False,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volatility_indicators=False,
        )
        features = factory.compute_all_timeframes(data, lag_features=False)
        h1_features = features[Timeframe.H1]

        assert "obv" in h1_features.columns
        obv = h1_features["obv"].dropna()
        assert len(obv) > 0, "OBV should have valid values"
        # OBV should change over time
        assert obv.diff().abs().sum() > 0, "OBV should vary"

    def test_ad_on_synthetic(self, small_config):
        """Accumulation/Distribution should be finite on synthetic data."""
        from quantcore.features.factory import MultiTimeframeFeatureFactory

        df_1h = generate_synthetic_ohlcv(small_config)
        data = resample_all_timeframes(df_1h)

        factory = MultiTimeframeFeatureFactory(
            include_waves=False,
            include_rrg=False,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volatility_indicators=False,
        )
        features = factory.compute_all_timeframes(data, lag_features=False)
        h1_features = features[Timeframe.H1]

        assert "ad" in h1_features.columns
        ad = h1_features["ad"].dropna()
        assert len(ad) > 0, "A/D should have valid values"
        # Should not have inf
        assert not np.isinf(ad).any(), "A/D should be finite"

    def test_mfi_on_synthetic(self, small_config):
        """Money Flow Index should be bounded 0-100 on synthetic data."""
        from quantcore.features.factory import MultiTimeframeFeatureFactory

        df_1h = generate_synthetic_ohlcv(small_config)
        data = resample_all_timeframes(df_1h)

        factory = MultiTimeframeFeatureFactory(
            include_waves=False,
            include_rrg=False,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volatility_indicators=False,
        )
        features = factory.compute_all_timeframes(data, lag_features=False)
        h1_features = features[Timeframe.H1]

        assert "mfi" in h1_features.columns
        mfi = h1_features["mfi"].dropna()
        if len(mfi) > 0:
            assert (mfi >= 0).all(), "MFI should be >= 0"
            assert (mfi <= 100).all(), "MFI should be <= 100"


class TestSyntheticTAIntegration:
    """Integration tests for all TA indicators on synthetic data."""

    def test_all_ta_indicators_computed(self):
        """All TA indicators should be computed in the full pipeline."""
        from quantcore.features.factory import MultiTimeframeFeatureFactory
        from quantcore.features.technical_indicators import TechnicalIndicators

        # Use larger dataset for H1 timeframe only
        config = SyntheticMarketConfig(periods=300, seed=42)
        df_1h = generate_synthetic_ohlcv(config)

        # Create factory with all TA indicators enabled
        factory = MultiTimeframeFeatureFactory(
            include_waves=False,
            include_rrg=False,
            include_technical_indicators=True,
            enable_moving_averages=True,
            enable_oscillators=True,
            enable_volatility_indicators=True,
            enable_volume_indicators=True,
            enable_hilbert=False,  # Skip Hilbert (slow)
        )
        # Test only H1 timeframe (has enough bars for all indicators)
        features = factory._compute_features_for_timeframe(df_1h, Timeframe.H1)

        # Get expected feature names from TechnicalIndicators
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=True,
            enable_oscillators=True,
            enable_volatility=True,
            enable_volume=True,
            enable_hilbert=False,
        )
        expected_features = ti.get_feature_names()

        # Check all expected indicators are present
        missing = [f for f in expected_features if f not in features.columns]
        assert len(missing) == 0, f"Missing indicators: {missing}"

    def test_ta_no_lookahead_bias(self):
        """TA indicators should not have lookahead bias."""
        from quantcore.features.factory import MultiTimeframeFeatureFactory

        # Use larger dataset for H1 timeframe only
        config = SyntheticMarketConfig(periods=300, seed=42)
        df_1h = generate_synthetic_ohlcv(config)

        factory = MultiTimeframeFeatureFactory(
            include_waves=False,
            include_rrg=False,
            enable_hilbert=False,
        )
        # Test only H1 timeframe
        features = factory._compute_features_for_timeframe(df_1h, Timeframe.H1)

        # Check that key indicators aren't perfectly correlated with future prices
        future_close = features["close"].shift(-1)

        for col in ["rsi", "macd_line", "atr", "sma_20", "ema_20"]:
            if col in features.columns:
                corr = features[col].corr(future_close)
                assert (
                    abs(corr) < 0.95
                ), f"{col} suspiciously correlated with future: {corr}"

    def test_ta_performance_on_synthetic(self):
        """TA computation should complete in reasonable time on synthetic data."""
        import time
        from quantcore.features.factory import MultiTimeframeFeatureFactory

        # Larger dataset for performance testing
        config = SyntheticMarketConfig(periods=500, seed=42)
        df_1h = generate_synthetic_ohlcv(config)

        factory = MultiTimeframeFeatureFactory(
            include_waves=False,
            include_rrg=False,
            enable_hilbert=False,
        )

        start = time.time()
        # Test only H1 timeframe
        features = factory._compute_features_for_timeframe(df_1h, Timeframe.H1)
        elapsed = time.time() - start

        # Should complete in reasonable time (<2s for 500 bars on H1)
        assert elapsed < 2.0, f"TA computation too slow: {elapsed:.3f}s"

    def test_ta_features_for_ml_extraction(self):
        """Should be able to extract TA features for ML model input."""
        from quantcore.features.factory import MultiTimeframeFeatureFactory

        # Use larger dataset for H1 timeframe only
        config = SyntheticMarketConfig(periods=300, seed=42)
        df_1h = generate_synthetic_ohlcv(config)

        factory = MultiTimeframeFeatureFactory(
            include_waves=False,
            include_rrg=False,
            enable_hilbert=False,
        )
        # Test only H1 timeframe
        features = factory._compute_features_for_timeframe(df_1h, Timeframe.H1)

        # Extract ML feature names for H1 (without higher timeframes)
        ml_features = factory.get_feature_names_for_ml(
            Timeframe.H1,
            include_higher_tf=False,
            include_wave_features=False,
        )

        # Should have a reasonable number of features
        assert len(ml_features) > 50, "Should have substantial number of ML features"

        # All ML features should be in the computed features
        available = [f for f in ml_features if f in features.columns]

        # Most features should be available
        coverage = len(available) / len(ml_features)
        assert coverage > 0.8, f"Too few ML features available: {coverage:.1%}"


# =============================================================================
# Labeling Tests
# =============================================================================


class TestSyntheticLabels:
    """Tests for creating labels on synthetic data."""

    def test_labels_created(self, small_config):
        """Labels should be created for synthetic data."""
        from quantcore.features.factory import MultiTimeframeFeatureFactory
        from quantcore.labeling.event_labeler import EventLabeler

        df_1h = generate_synthetic_ohlcv(small_config)
        data = resample_all_timeframes(df_1h)

        factory = MultiTimeframeFeatureFactory(include_waves=False)
        features = factory.compute_all_timeframes(data, lag_features=True)

        labeler = EventLabeler()
        labeled = labeler.label_long_trades(features[Timeframe.H1])

        assert "label_long" in labeled.columns

    def test_labels_have_valid_values(self, small_config):
        """Labels should be 0 or 1 (or NaN for bars without labels)."""
        from quantcore.features.factory import MultiTimeframeFeatureFactory
        from quantcore.labeling.event_labeler import EventLabeler

        df_1h = generate_synthetic_ohlcv(small_config)
        data = resample_all_timeframes(df_1h)

        factory = MultiTimeframeFeatureFactory(include_waves=False)
        features = factory.compute_all_timeframes(data, lag_features=True)

        labeler = EventLabeler()
        labeled = labeler.label_long_trades(features[Timeframe.H1])

        # Labels should be 0, 1, or NaN
        valid_labels = labeled["label_long"].dropna()
        assert set(valid_labels.unique()).issubset({0, 1})

    def test_labels_statistics_available(self, small_config):
        """Should be able to get label statistics."""
        from quantcore.features.factory import MultiTimeframeFeatureFactory
        from quantcore.labeling.event_labeler import EventLabeler

        df_1h = generate_synthetic_ohlcv(small_config)
        data = resample_all_timeframes(df_1h)

        factory = MultiTimeframeFeatureFactory(include_waves=False)
        features = factory.compute_all_timeframes(data, lag_features=True)

        labeler = EventLabeler()
        labeled = labeler.label_long_trades(features[Timeframe.H1])

        stats = labeler.get_label_statistics(labeled, "label_long")

        assert "count" in stats
        assert stats["count"] > 0


# =============================================================================
# End-to-End Pipeline Test
# =============================================================================


class TestSyntheticEndToEnd:
    """End-to-end pipeline tests."""

    def test_full_pipeline_runs(self):
        """Full pipeline should run without errors."""
        from quantcore.features.factory import MultiTimeframeFeatureFactory
        from quantcore.labeling.event_labeler import EventLabeler
        from quantcore.models.trainer import ModelTrainer, TrainingConfig

        # 1. Generate data - need more bars for feature warmup periods
        config = SyntheticMarketConfig(
            periods=800,  # More bars to ensure enough valid samples after warmup
            seed=42,
        )
        df_1h = generate_synthetic_ohlcv(config)

        # 2. Resample
        data = resample_all_timeframes(df_1h)

        # 3. Compute features
        factory = MultiTimeframeFeatureFactory(include_waves=False)
        features = factory.compute_all_timeframes(data, lag_features=True)

        # 4. Create labels
        labeler = EventLabeler()
        h1_labeled = labeler.label_long_trades(features[Timeframe.H1])

        # 5. Prepare training data
        feature_names = factory.get_feature_names_for_ml(
            Timeframe.H1,
            include_higher_tf=True,
            include_wave_features=False,
        )
        available_features = [f for f in feature_names if f in h1_labeled.columns]

        X = h1_labeled[available_features].copy()
        y = h1_labeled["label_long"].copy()

        # Drop NaN labels
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]

        # Fill NaN/inf in features (from warmup periods) with 0
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        # Skip if not enough samples
        if len(X) < 50:
            pytest.skip("Not enough samples for training")

        # 6. Train model
        train_config = TrainingConfig(
            model_type="lightgbm",
            n_estimators=10,
            max_depth=3,
            n_splits=2,
        )
        trainer = ModelTrainer(train_config)
        model_result = trainer.train(X, y)

        # Model should have trained
        assert model_result.model is not None
        assert "auc" in model_result.metrics

    def test_backtest_produces_trades(self, small_config):
        """Backtest should produce at least one trade."""
        from quantcore.features.factory import MultiTimeframeFeatureFactory
        from quantcore.labeling.event_labeler import EventLabeler
        from quantcore.backtesting.engine import BacktestEngine, BacktestConfig

        # Generate more bars to ensure we get trades
        config = SyntheticMarketConfig(
            periods=500,  # More bars
            seed=42,
            vol=0.30,  # Higher volatility
        )

        # Generate and prepare data
        df_1h = generate_synthetic_ohlcv(config)
        data = resample_all_timeframes(df_1h)

        factory = MultiTimeframeFeatureFactory(include_waves=False)
        features = factory.compute_all_timeframes(data, lag_features=True)

        labeler = EventLabeler()
        h1_labeled = labeler.label_long_trades(features[Timeframe.H1])

        # Create simple signals
        signals = h1_labeled.copy()
        signals["signal"] = 0
        signals["signal_direction"] = "NONE"
        signals["signal_confidence"] = 0.0
        signals["signal_entry"] = np.nan
        signals["signal_tp"] = np.nan
        signals["signal_sl"] = np.nan
        signals["signal_alignment"] = 0.5

        # Generate signals every 20 bars
        for i in range(50, len(signals), 20):
            idx = signals.index[i]
            entry = signals.loc[idx, "close"]
            atr = signals.loc[idx, "atr"] if "atr" in signals.columns else entry * 0.01

            signals.loc[idx, "signal"] = 1
            signals.loc[idx, "signal_direction"] = "LONG"
            signals.loc[idx, "signal_confidence"] = 0.6
            signals.loc[idx, "signal_entry"] = entry
            signals.loc[idx, "signal_tp"] = entry + 1.5 * atr
            signals.loc[idx, "signal_sl"] = entry - 1.0 * atr

        # Run backtest
        bt_config = BacktestConfig(
            initial_capital=100000,
            max_concurrent_trades=3,
        )
        engine = BacktestEngine(bt_config)
        result = engine.run(signals, df_1h)

        # Should have produced at least one trade
        assert result.total_trades >= 1

    def test_backtest_metrics_are_finite(self, small_config):
        """Backtest metrics should be finite (not NaN/inf)."""
        from quantcore.backtesting.engine import BacktestEngine, BacktestConfig

        config = SyntheticMarketConfig(periods=300, seed=42)
        df_1h = generate_synthetic_ohlcv(config)
        data = resample_all_timeframes(df_1h)

        # Create signals
        signals = df_1h.copy()
        signals["signal"] = 0
        signals["signal_direction"] = "NONE"
        signals["signal_confidence"] = 0.0
        signals["signal_entry"] = np.nan
        signals["signal_tp"] = np.nan
        signals["signal_sl"] = np.nan
        signals["signal_alignment"] = 0.5

        # Add a few signals
        for i in [50, 100, 150, 200]:
            if i < len(signals):
                idx = signals.index[i]
                entry = signals.loc[idx, "close"]
                signals.loc[idx, "signal"] = 1
                signals.loc[idx, "signal_direction"] = "LONG"
                signals.loc[idx, "signal_confidence"] = 0.6
                signals.loc[idx, "signal_entry"] = entry
                signals.loc[idx, "signal_tp"] = entry * 1.02
                signals.loc[idx, "signal_sl"] = entry * 0.98

        # Run backtest
        engine = BacktestEngine()
        result = engine.run(signals, df_1h)

        # Metrics should be finite
        assert np.isfinite(result.total_return)
        assert np.isfinite(result.sharpe_ratio)
        assert np.isfinite(result.max_drawdown)
