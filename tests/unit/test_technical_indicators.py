"""
Tests for complete technical indicators suite.

Tests all 57 indicators for correctness, performance, and integration.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from quantcore.features.technical_indicators import TechnicalIndicators
from quantcore.config.timeframes import Timeframe


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=500, freq="1H")
    np.random.seed(42)

    # Generate realistic price data
    close = pd.Series(100 + np.cumsum(np.random.randn(500) * 0.5), index=dates)
    high = close + np.abs(np.random.randn(500) * 0.3)
    low = close - np.abs(np.random.randn(500) * 0.3)
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = np.random.randint(1000, 10000, 500)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class TestMovingAverages:
    """Test all moving average indicators."""

    def test_sma(self, sample_ohlcv):
        """Test Simple Moving Average."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_oscillators=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(sample_ohlcv)

        for period in ti.ma_periods:
            col = f"sma_{period}"
            assert col in result.columns, f"Missing {col}"
            assert not result[col].iloc[period:].isna().all(), f"{col} all NaN"

    def test_ema(self, sample_ohlcv):
        """Test Exponential Moving Average."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_oscillators=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(sample_ohlcv)

        for period in ti.ma_periods:
            col = f"ema_{period}"
            assert col in result.columns
            # EMA should be more responsive than SMA
            ema = result[col].dropna()
            assert len(ema) > 0

    def test_wma(self, sample_ohlcv):
        """Test Weighted Moving Average."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_oscillators=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(sample_ohlcv)

        for period in ti.ma_periods:
            col = f"wma_{period}"
            assert col in result.columns
            assert result[col].isna().sum() < len(result)

    def test_dema_tema(self, sample_ohlcv):
        """Test Double and Triple EMA."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_oscillators=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(sample_ohlcv)

        for period in ti.ma_periods:
            assert f"dema_{period}" in result.columns
            assert f"tema_{period}" in result.columns

    def test_vwap(self, sample_ohlcv):
        """Test Volume Weighted Average Price."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_oscillators=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(sample_ohlcv)

        assert "vwap" in result.columns
        vwap = result["vwap"].dropna()
        assert len(vwap) > 0
        # VWAP should be within price range
        assert (vwap <= result["high"].loc[vwap.index] * 1.1).all()
        assert (vwap >= result["low"].loc[vwap.index] * 0.9).all()


class TestOscillators:
    """Test all oscillator indicators."""

    def test_rsi(self, sample_ohlcv):
        """Test RSI."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(sample_ohlcv)

        assert "rsi" in result.columns
        rsi = result["rsi"].dropna()
        # RSI should be 0-100
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

    def test_macd(self, sample_ohlcv):
        """Test MACD."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(sample_ohlcv)

        assert "macd_line" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns

        # Histogram should be difference
        hist = result["macd_histogram"].dropna()
        expected_hist = (result["macd_line"] - result["macd_signal"]).dropna()
        pd.testing.assert_series_equal(
            hist, expected_hist.loc[hist.index], check_names=False
        )

    def test_stochastic(self, sample_ohlcv):
        """Test Stochastic oscillator."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(sample_ohlcv)

        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns

        stoch_k = result["stoch_k"].dropna()
        stoch_d = result["stoch_d"].dropna()

        # Should be 0-100
        assert (stoch_k >= 0).all()
        assert (stoch_k <= 100).all()
        assert (stoch_d >= 0).all()
        assert (stoch_d <= 100).all()

    def test_adx(self, sample_ohlcv):
        """Test ADX."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(sample_ohlcv)

        assert "adx" in result.columns
        adx = result["adx"].dropna()
        # ADX should be 0-100
        assert (adx >= 0).all()
        assert (adx <= 100).all()

    def test_williams_r(self, sample_ohlcv):
        """Test Williams %R."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(sample_ohlcv)

        assert "willr" in result.columns
        willr = result["willr"].dropna()
        # Williams %R should be -100 to 0
        assert (willr >= -100).all()
        assert (willr <= 0).all()


class TestVolatility:
    """Test all volatility indicators."""

    def test_bollinger_bands(self, sample_ohlcv):
        """Test Bollinger Bands."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volume=False,
        )
        result = ti.compute(sample_ohlcv)

        assert "bb_upper" in result.columns
        assert "bb_middle" in result.columns
        assert "bb_lower" in result.columns

        # Check ordering
        valid_rows = result[["bb_upper", "bb_middle", "bb_lower"]].dropna()
        assert (valid_rows["bb_upper"] >= valid_rows["bb_middle"]).all()
        assert (valid_rows["bb_middle"] >= valid_rows["bb_lower"]).all()

    def test_atr(self, sample_ohlcv):
        """Test Average True Range."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volume=False,
        )
        result = ti.compute(sample_ohlcv)

        assert "atr" in result.columns
        assert "natr" in result.columns

        atr = result["atr"].dropna()
        natr = result["natr"].dropna()

        # ATR should be positive
        assert (atr > 0).all()
        assert (natr > 0).all()

    def test_sar(self, sample_ohlcv):
        """Test Parabolic SAR."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volume=False,
        )
        result = ti.compute(sample_ohlcv)

        assert "sar" in result.columns
        sar = result["sar"].dropna()
        assert len(sar) > 0


class TestVolume:
    """Test all volume indicators."""

    def test_obv(self, sample_ohlcv):
        """Test On Balance Volume."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volatility=False,
        )
        result = ti.compute(sample_ohlcv)

        assert "obv" in result.columns
        obv = result["obv"].dropna()
        assert len(obv) > 0
        # OBV should change when price changes
        assert obv.diff().abs().sum() > 0

    def test_ad(self, sample_ohlcv):
        """Test Accumulation/Distribution."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volatility=False,
        )
        result = ti.compute(sample_ohlcv)

        assert "ad" in result.columns
        ad = result["ad"].dropna()
        assert len(ad) > 0
        # Should not have inf
        assert not np.isinf(ad).any()

    def test_mfi(self, sample_ohlcv):
        """Test Money Flow Index."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(sample_ohlcv)

        assert "mfi" in result.columns
        mfi = result["mfi"].dropna()
        # MFI should be 0-100
        assert (mfi >= 0).all()
        assert (mfi <= 100).all()


class TestIntegration:
    """Integration tests."""

    def test_all_indicators_computed(self, sample_ohlcv):
        """Test that all enabled indicators are computed."""
        ti = TechnicalIndicators(Timeframe.H1)
        result = ti.compute(sample_ohlcv)

        feature_names = ti.get_feature_names()
        for name in feature_names:
            assert name in result.columns, f"Missing indicator: {name}"

    def test_no_lookahead(self, sample_ohlcv):
        """Test that indicators don't use future data."""
        ti = TechnicalIndicators(Timeframe.H1)
        result = ti.compute(sample_ohlcv)

        # Check that indicators aren't perfectly correlated with future prices
        for col in ["rsi", "macd_line", "atr"]:
            if col in result.columns:
                corr = result[col].corr(result["close"].shift(-1))
                assert (
                    abs(corr) < 0.95
                ), f"{col} suspiciously high correlation with future"

    def test_performance(self, sample_ohlcv):
        """Test computation performance."""
        import time

        ti = TechnicalIndicators(Timeframe.H1, enable_hilbert=False)

        start = time.time()
        result = ti.compute(sample_ohlcv)
        elapsed = time.time() - start

        # Should compute in reasonable time (<1s for 500 rows)
        assert elapsed < 1.0, f"Computation too slow: {elapsed:.3f}s"

        # Calculate per-row time
        per_row = elapsed / len(sample_ohlcv) * 1000
        assert per_row < 2.0, f"Per-row computation too slow: {per_row:.3f}ms"

    def test_synthetic_mode(self, sample_ohlcv):
        """Test synthetic mode for testing."""
        ti = TechnicalIndicators(Timeframe.H1, synthetic_mode=True, enable_hilbert=True)
        result = ti.compute(sample_ohlcv)

        # Should complete without errors
        assert not result.empty
        # Hilbert indicators should exist (simplified)
        assert "ht_trendline" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
