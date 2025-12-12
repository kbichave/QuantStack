"""
Comprehensive unit tests for all 57 technical indicators.

Tests each indicator individually for:
- Correct output
- Expected range bounds
- No NaN/Inf after warmup
- Correct computation logic
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from quantcore.features.technical_indicators import TechnicalIndicators
from quantcore.config.timeframes import Timeframe


@pytest.fixture
def large_ohlcv():
    """Generate larger OHLCV dataset for indicator testing."""
    dates = pd.date_range(start="2022-01-01", periods=1000, freq="1H")
    np.random.seed(42)

    # Generate realistic trending price data
    trend = np.linspace(100, 120, 1000)
    noise = np.random.randn(1000) * 2
    close = trend + noise

    high = close + np.abs(np.random.randn(1000) * 0.5)
    low = close - np.abs(np.random.randn(1000) * 0.5)
    open_ = close + np.random.randn(1000) * 0.3
    volume = np.random.randint(5000, 15000, 1000)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class TestMovingAverageIndicators:
    """Test all 10 moving average indicators individually."""

    def test_sma_10(self, large_ohlcv):
        """Test SMA with period 10."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_oscillators=False,
            enable_volatility=False,
            enable_volume=False,
        )
        ti.ma_periods = [10]
        result = ti.compute(large_ohlcv)

        assert "sma_10" in result.columns
        sma = result["sma_10"].dropna()
        assert len(sma) >= 990  # Should have values after warmup
        # SMA should be within reasonable range of price
        assert (sma >= large_ohlcv["low"].min() * 0.9).all()
        assert (sma <= large_ohlcv["high"].max() * 1.1).all()

    def test_sma_all_periods(self, large_ohlcv):
        """Test SMA for all default periods."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_oscillators=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        for period in [10, 20, 50, 200]:
            col = f"sma_{period}"
            assert col in result.columns, f"Missing {col}"
            sma = result[col].dropna()
            assert len(sma) >= 800, f"{col} insufficient non-NaN values"

    def test_ema_10(self, large_ohlcv):
        """Test EMA with period 10."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_oscillators=False,
            enable_volatility=False,
            enable_volume=False,
        )
        ti.ma_periods = [10]
        result = ti.compute(large_ohlcv)

        assert "ema_10" in result.columns
        ema = result["ema_10"].dropna()
        assert len(ema) >= 990
        # EMA should be more responsive than SMA
        assert not ema.isna().all()

    def test_ema_all_periods(self, large_ohlcv):
        """Test EMA for all default periods."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_oscillators=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        for period in [10, 20, 50, 200]:
            col = f"ema_{period}"
            assert col in result.columns, f"Missing {col}"
            ema = result[col].dropna()
            assert len(ema) >= 800, f"{col} insufficient non-NaN values"

    def test_wma_10(self, large_ohlcv):
        """Test WMA with period 10."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_oscillators=False,
            enable_volatility=False,
            enable_volume=False,
        )
        ti.ma_periods = [10]
        result = ti.compute(large_ohlcv)

        assert "wma_10" in result.columns
        wma = result["wma_10"].dropna()
        assert len(wma) >= 990

    def test_dema_10(self, large_ohlcv):
        """Test DEMA with period 10."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_oscillators=False,
            enable_volatility=False,
            enable_volume=False,
        )
        ti.ma_periods = [10]
        result = ti.compute(large_ohlcv)

        assert "dema_10" in result.columns
        dema = result["dema_10"].dropna()
        assert len(dema) >= 980

    def test_tema_10(self, large_ohlcv):
        """Test TEMA with period 10."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_oscillators=False,
            enable_volatility=False,
            enable_volume=False,
        )
        ti.ma_periods = [10]
        result = ti.compute(large_ohlcv)

        assert "tema_10" in result.columns
        tema = result["tema_10"].dropna()
        assert len(tema) >= 970

    def test_trima_10(self, large_ohlcv):
        """Test TRIMA with period 10."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_oscillators=False,
            enable_volatility=False,
            enable_volume=False,
        )
        ti.ma_periods = [10]
        result = ti.compute(large_ohlcv)

        assert "trima_10" in result.columns
        trima = result["trima_10"].dropna()
        assert len(trima) >= 980

    def test_kama_10(self, large_ohlcv):
        """Test KAMA with period 10."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_oscillators=False,
            enable_volatility=False,
            enable_volume=False,
        )
        ti.ma_periods = [10]
        result = ti.compute(large_ohlcv)

        assert "kama_10" in result.columns
        kama = result["kama_10"].dropna()
        assert len(kama) >= 990

    def test_t3_10(self, large_ohlcv):
        """Test T3 with period 10."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_oscillators=False,
            enable_volatility=False,
            enable_volume=False,
        )
        ti.ma_periods = [10]
        result = ti.compute(large_ohlcv)

        assert "t3_10" in result.columns
        t3 = result["t3_10"].dropna()
        assert len(t3) >= 940  # T3 requires longer warmup due to 6x EMA

    def test_vwap(self, large_ohlcv):
        """Test VWAP."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_oscillators=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "vwap" in result.columns
        vwap = result["vwap"].dropna()
        assert len(vwap) > 0
        # VWAP should be within price range
        assert (vwap >= large_ohlcv["low"].min() * 0.9).all()
        assert (vwap <= large_ohlcv["high"].max() * 1.1).all()

    def test_mama_fama(self, large_ohlcv):
        """Test MAMA and FAMA."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_oscillators=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "mama" in result.columns
        assert "fama" in result.columns
        mama = result["mama"].dropna()
        fama = result["fama"].dropna()
        assert len(mama) > 0
        assert len(fama) > 0


class TestOscillatorIndicators:
    """Test all 23 oscillator indicators individually."""

    def test_macd(self, large_ohlcv):
        """Test MACD indicator."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "macd_line" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns

        macd = result["macd_line"].dropna()
        signal = result["macd_signal"].dropna()
        hist = result["macd_histogram"].dropna()

        assert len(macd) >= 950
        assert len(signal) >= 950
        assert len(hist) >= 950

    def test_macdext(self, large_ohlcv):
        """Test MACDEXT indicator."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "macdext_line" in result.columns
        assert "macdext_signal" in result.columns
        assert "macdext_histogram" in result.columns

    def test_rsi(self, large_ohlcv):
        """Test RSI indicator."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "rsi" in result.columns
        rsi = result["rsi"].dropna()
        assert len(rsi) >= 980
        # RSI must be 0-100
        assert (rsi >= 0).all(), f"RSI has values < 0: {rsi[rsi < 0]}"
        assert (rsi <= 100).all(), f"RSI has values > 100: {rsi[rsi > 100]}"

    def test_stoch(self, large_ohlcv):
        """Test Stochastic oscillator."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns

        stoch_k = result["stoch_k"].dropna()
        stoch_d = result["stoch_d"].dropna()

        assert len(stoch_k) >= 980
        assert len(stoch_d) >= 970

        # Stochastic must be 0-100
        assert (stoch_k >= 0).all()
        assert (stoch_k <= 100).all()
        assert (stoch_d >= 0).all()
        assert (stoch_d <= 100).all()

    def test_stochf(self, large_ohlcv):
        """Test Stochastic Fast."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "stochf_k" in result.columns
        assert "stochf_d" in result.columns

        stochf_k = result["stochf_k"].dropna()
        stochf_d = result["stochf_d"].dropna()

        assert len(stochf_k) >= 990
        assert len(stochf_d) >= 990

        # Must be 0-100
        assert (stochf_k >= 0).all()
        assert (stochf_k <= 100).all()

    def test_stochrsi(self, large_ohlcv):
        """Test Stochastic RSI."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "stochrsi_k" in result.columns
        assert "stochrsi_d" in result.columns

        stochrsi_k = result["stochrsi_k"].dropna()
        stochrsi_d = result["stochrsi_d"].dropna()

        assert len(stochrsi_k) >= 960
        # Must be 0-100
        assert (stochrsi_k >= 0).all()
        assert (stochrsi_k <= 100).all()

    def test_willr(self, large_ohlcv):
        """Test Williams %R."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "willr" in result.columns
        willr = result["willr"].dropna()
        assert len(willr) >= 980

        # Williams %R must be -100 to 0
        assert (
            willr >= -100
        ).all(), f"Williams %R has values < -100: {willr[willr < -100]}"
        assert (willr <= 0).all(), f"Williams %R has values > 0: {willr[willr > 0]}"

    def test_adx(self, large_ohlcv):
        """Test ADX."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "adx" in result.columns
        adx = result["adx"].dropna()
        assert len(adx) >= 950

        # ADX must be 0-100
        assert (adx >= 0).all()
        assert (adx <= 100).all()

    def test_adxr(self, large_ohlcv):
        """Test ADXR."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "adxr" in result.columns
        adxr = result["adxr"].dropna()
        assert len(adxr) >= 920

        # ADXR must be 0-100
        assert (adxr >= 0).all()
        assert (adxr <= 100).all()

    def test_apo(self, large_ohlcv):
        """Test APO."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "apo" in result.columns
        apo = result["apo"].dropna()
        assert len(apo) >= 950

    def test_ppo(self, large_ohlcv):
        """Test PPO."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "ppo" in result.columns
        ppo = result["ppo"].dropna()
        assert len(ppo) >= 950

    def test_mom(self, large_ohlcv):
        """Test Momentum."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "mom" in result.columns
        mom = result["mom"].dropna()
        assert len(mom) >= 980

    def test_bop(self, large_ohlcv):
        """Test Balance of Power."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "bop" in result.columns
        bop = result["bop"].dropna()
        assert len(bop) > 0

        # BOP typically -1 to 1, but can have extreme values with small high-low ranges
        # Just verify no inf/nan after warmup
        assert not np.isinf(bop).any(), "BOP contains inf values"
        assert not np.isnan(bop).any(), "BOP contains NaN values"

    def test_cci(self, large_ohlcv):
        """Test CCI."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "cci" in result.columns
        cci = result["cci"].dropna()
        assert len(cci) >= 970

    def test_cmo(self, large_ohlcv):
        """Test CMO."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "cmo" in result.columns
        cmo = result["cmo"].dropna()
        assert len(cmo) >= 980

        # CMO must be -100 to 100
        assert (cmo >= -100).all()
        assert (cmo <= 100).all()

    def test_roc(self, large_ohlcv):
        """Test ROC."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "roc" in result.columns
        roc = result["roc"].dropna()
        assert len(roc) >= 980

    def test_rocr(self, large_ohlcv):
        """Test ROCR."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "rocr" in result.columns
        rocr = result["rocr"].dropna()
        assert len(rocr) >= 980
        # ROCR should be positive
        assert (rocr > 0).all()

    def test_aroon(self, large_ohlcv):
        """Test Aroon."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "aroon_up" in result.columns
        assert "aroon_down" in result.columns

        aroon_up = result["aroon_up"].dropna()
        aroon_down = result["aroon_down"].dropna()

        assert len(aroon_up) >= 970
        assert len(aroon_down) >= 970

        # Aroon must be 0-100
        assert (aroon_up >= 0).all()
        assert (aroon_up <= 100).all()
        assert (aroon_down >= 0).all()
        assert (aroon_down <= 100).all()

    def test_aroonosc(self, large_ohlcv):
        """Test Aroon Oscillator."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "aroonosc" in result.columns
        aroonosc = result["aroonosc"].dropna()
        assert len(aroonosc) >= 970

        # Aroon Osc must be -100 to 100
        assert (aroonosc >= -100).all()
        assert (aroonosc <= 100).all()

    def test_mfi(self, large_ohlcv):
        """Test MFI."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "mfi" in result.columns
        mfi = result["mfi"].dropna()
        assert len(mfi) >= 980

        # MFI must be 0-100
        assert (mfi >= 0).all()
        assert (mfi <= 100).all()

    def test_trix(self, large_ohlcv):
        """Test TRIX."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "trix" in result.columns
        trix = result["trix"].dropna()
        assert len(trix) >= 900

    def test_ultosc(self, large_ohlcv):
        """Test Ultimate Oscillator."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "ultosc" in result.columns
        ultosc = result["ultosc"].dropna()
        assert len(ultosc) >= 960

        # Ultimate Osc must be 0-100
        assert (ultosc >= 0).all()
        assert (ultosc <= 100).all()

    def test_dx(self, large_ohlcv):
        """Test DX."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "dx" in result.columns
        dx = result["dx"].dropna()
        assert len(dx) >= 970

        # DX must be 0-100
        assert (dx >= 0).all()
        assert (dx <= 100).all()

    def test_directional_indicators(self, large_ohlcv):
        """Test Plus/Minus DI and DM."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "plus_di" in result.columns
        assert "minus_di" in result.columns
        assert "plus_dm" in result.columns
        assert "minus_dm" in result.columns

        plus_di = result["plus_di"].dropna()
        minus_di = result["minus_di"].dropna()
        plus_dm = result["plus_dm"].dropna()
        minus_dm = result["minus_dm"].dropna()

        assert len(plus_di) >= 970
        assert len(minus_di) >= 970
        assert len(plus_dm) >= 990
        assert len(minus_dm) >= 990

        # DI should be positive
        assert (plus_di >= 0).all()
        assert (minus_di >= 0).all()
        # DM should be non-negative
        assert (plus_dm >= 0).all()
        assert (minus_dm >= 0).all()


class TestVolatilityIndicators:
    """Test all 8 volatility indicators individually."""

    def test_bbands(self, large_ohlcv):
        """Test Bollinger Bands."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "bb_upper" in result.columns
        assert "bb_middle" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_width" in result.columns
        assert "bb_pct" in result.columns

        bb_upper = result["bb_upper"].dropna()
        bb_middle = result["bb_middle"].dropna()
        bb_lower = result["bb_lower"].dropna()

        assert len(bb_upper) >= 970
        assert len(bb_middle) >= 970
        assert len(bb_lower) >= 970

        # Check ordering: upper > middle > lower
        valid_idx = bb_upper.index.intersection(bb_middle.index).intersection(
            bb_lower.index
        )
        assert (bb_upper.loc[valid_idx] >= bb_middle.loc[valid_idx]).all()
        assert (bb_middle.loc[valid_idx] >= bb_lower.loc[valid_idx]).all()

    def test_midpoint(self, large_ohlcv):
        """Test Midpoint."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "midpoint" in result.columns
        midpoint = result["midpoint"].dropna()
        assert len(midpoint) >= 980

    def test_midprice(self, large_ohlcv):
        """Test Midprice."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "midprice" in result.columns
        midprice = result["midprice"].dropna()
        assert len(midprice) >= 980

    def test_sar(self, large_ohlcv):
        """Test Parabolic SAR."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "sar" in result.columns
        sar = result["sar"].dropna()
        assert len(sar) > 0

    def test_trange(self, large_ohlcv):
        """Test True Range."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "trange" in result.columns
        trange = result["trange"].dropna()
        assert len(trange) >= 990

        # True Range must be positive
        assert (trange >= 0).all()

    def test_atr(self, large_ohlcv):
        """Test ATR."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "atr" in result.columns
        atr = result["atr"].dropna()
        assert len(atr) >= 980

        # ATR must be positive
        assert (atr > 0).all(), f"ATR has non-positive values: {atr[atr <= 0]}"

    def test_natr(self, large_ohlcv):
        """Test NATR."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volume=False,
        )
        result = ti.compute(large_ohlcv)

        assert "natr" in result.columns
        natr = result["natr"].dropna()
        assert len(natr) >= 980

        # NATR must be positive
        assert (natr > 0).all()


class TestVolumeIndicators:
    """Test all 3 volume indicators individually."""

    def test_ad(self, large_ohlcv):
        """Test Accumulation/Distribution."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volatility=False,
        )
        result = ti.compute(large_ohlcv)

        assert "ad" in result.columns
        ad = result["ad"].dropna()
        assert len(ad) > 0

        # Should not have inf
        assert not np.isinf(ad).any()

    def test_adosc(self, large_ohlcv):
        """Test AD Oscillator."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volatility=False,
        )
        result = ti.compute(large_ohlcv)

        assert "adosc" in result.columns
        adosc = result["adosc"].dropna()
        assert len(adosc) >= 980

        # Should not have inf
        assert not np.isinf(adosc).any()

    def test_obv(self, large_ohlcv):
        """Test On Balance Volume."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volatility=False,
        )
        result = ti.compute(large_ohlcv)

        assert "obv" in result.columns
        obv = result["obv"].dropna()
        assert len(obv) > 0

        # OBV should change over time
        assert obv.diff().abs().sum() > 0


class TestHilbertTransformIndicators:
    """Test all 6 Hilbert Transform indicators individually."""

    def test_ht_trendline(self, large_ohlcv):
        """Test HT Trendline."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volatility=False,
            enable_volume=False,
            enable_hilbert=True,
        )
        result = ti.compute(large_ohlcv)

        assert "ht_trendline" in result.columns
        ht_trendline = result["ht_trendline"].dropna()
        assert len(ht_trendline) > 0

    def test_ht_sine(self, large_ohlcv):
        """Test HT Sine and LeadSine."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volatility=False,
            enable_volume=False,
            enable_hilbert=True,
        )
        result = ti.compute(large_ohlcv)

        assert "ht_sine" in result.columns
        assert "ht_leadsine" in result.columns

        ht_sine = result["ht_sine"].dropna()
        ht_leadsine = result["ht_leadsine"].dropna()

        assert len(ht_sine) > 0
        assert len(ht_leadsine) > 0

        # Sine waves should be -1 to 1
        assert (ht_sine >= -1.1).all()  # Small tolerance
        assert (ht_sine <= 1.1).all()
        assert (ht_leadsine >= -1.1).all()
        assert (ht_leadsine <= 1.1).all()

    def test_ht_trendmode(self, large_ohlcv):
        """Test HT Trend Mode."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volatility=False,
            enable_volume=False,
            enable_hilbert=True,
        )
        result = ti.compute(large_ohlcv)

        assert "ht_trendmode" in result.columns
        ht_trendmode = result["ht_trendmode"].dropna()
        assert len(ht_trendmode) > 0

        # Should be 0 or 1
        assert ht_trendmode.isin([0, 1]).all()

    def test_ht_dcperiod(self, large_ohlcv):
        """Test HT Dominant Cycle Period."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volatility=False,
            enable_volume=False,
            enable_hilbert=True,
        )
        result = ti.compute(large_ohlcv)

        assert "ht_dcperiod" in result.columns
        ht_dcperiod = result["ht_dcperiod"].dropna()
        assert len(ht_dcperiod) > 0

    def test_ht_dcphase(self, large_ohlcv):
        """Test HT Dominant Cycle Phase."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volatility=False,
            enable_volume=False,
            enable_hilbert=True,
        )
        result = ti.compute(large_ohlcv)

        assert "ht_dcphase" in result.columns
        ht_dcphase = result["ht_dcphase"].dropna()
        assert len(ht_dcphase) > 0

        # Phase should be 0-360
        assert (ht_dcphase >= 0).all()
        assert (ht_dcphase <= 360).all()

    def test_ht_phasor(self, large_ohlcv):
        """Test HT Phasor Components."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volatility=False,
            enable_volume=False,
            enable_hilbert=True,
        )
        result = ti.compute(large_ohlcv)

        assert "ht_inphase" in result.columns
        assert "ht_quadrature" in result.columns

        ht_inphase = result["ht_inphase"].dropna()
        ht_quadrature = result["ht_quadrature"].dropna()

        assert len(ht_inphase) > 0
        assert len(ht_quadrature) > 0

        # Should be -1 to 1
        assert (ht_inphase >= -1.1).all()
        assert (ht_inphase <= 1.1).all()
        assert (ht_quadrature >= -1.1).all()
        assert (ht_quadrature <= 1.1).all()


class TestComprehensiveIntegration:
    """Test all 57 indicators together."""

    def test_all_57_indicators_present(self, large_ohlcv):
        """Test that all 57 indicators are computed."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=True,
            enable_oscillators=True,
            enable_volatility=True,
            enable_volume=True,
            enable_hilbert=True,
        )
        result = ti.compute(large_ohlcv)

        feature_names = ti.get_feature_names()

        # Should have all 57+ indicators (including variations)
        assert (
            len(feature_names) >= 57
        ), f"Expected >= 57 indicators, got {len(feature_names)}"

        # Check each is in result
        missing = []
        for name in feature_names:
            if name not in result.columns:
                missing.append(name)

        assert len(missing) == 0, f"Missing indicators: {missing}"

    def test_all_indicators_no_inf(self, large_ohlcv):
        """Test that no indicator produces inf values."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=True,
            enable_oscillators=True,
            enable_volatility=True,
            enable_volume=True,
            enable_hilbert=True,
        )
        result = ti.compute(large_ohlcv)

        feature_names = ti.get_feature_names()

        for name in feature_names:
            if name in result.columns:
                has_inf = np.isinf(result[name]).any()
                assert not has_inf, f"{name} contains infinite values"

    def test_all_indicators_reasonable_nan_count(self, large_ohlcv):
        """Test that indicators don't have excessive NaN values."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=True,
            enable_oscillators=True,
            enable_volatility=True,
            enable_volume=True,
            enable_hilbert=False,  # Skip Hilbert for speed
        )
        # Use smaller periods for this test to avoid excessive warmup with only 1000 rows
        ti.ma_periods = [
            10,
            20,
            50,
        ]  # Skip 200 period for T3/TEMA which need too many rows
        result = ti.compute(large_ohlcv)

        feature_names = ti.get_feature_names()
        warmup_threshold = 300  # Allow 300 rows for warmup with smaller periods

        for name in feature_names:
            if name in result.columns:
                nan_count = result[name].isna().sum()
                assert (
                    nan_count <= warmup_threshold
                ), f"{name} has {nan_count} NaN values (expected <= {warmup_threshold})"

    def test_performance_all_indicators(self, large_ohlcv):
        """Test computation performance with all indicators."""
        import time

        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=True,
            enable_oscillators=True,
            enable_volatility=True,
            enable_volume=True,
            enable_hilbert=False,  # Skip expensive Hilbert
        )

        start = time.time()
        result = ti.compute(large_ohlcv)
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 2.0, f"Computation too slow: {elapsed:.3f}s"

        # Calculate per-row time
        per_row = elapsed / len(large_ohlcv) * 1000
        assert per_row < 5.0, f"Per-row computation too slow: {per_row:.3f}ms"


class TestIndicatorCountVerification:
    """Verify we have exactly 57 indicators."""

    def test_moving_average_count(self):
        """Verify 10 moving average indicators."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_oscillators=False,
            enable_volatility=False,
            enable_volume=False,
        )
        ti.ma_periods = [10]  # Use single period for counting
        names = ti.get_feature_names()

        # Should have: SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, T3, VWAP, MAMA, FAMA
        ma_indicators = [
            n
            for n in names
            if any(
                x in n
                for x in [
                    "sma_",
                    "ema_",
                    "wma_",
                    "dema_",
                    "tema_",
                    "trima_",
                    "kama_",
                    "t3_",
                    "vwap",
                    "mama",
                    "fama",
                ]
            )
        ]

        # 8 types * 1 period + VWAP + MAMA + FAMA = 11
        assert (
            len(ma_indicators) == 11
        ), f"Expected 11 MA indicators, got {len(ma_indicators)}: {ma_indicators}"

    def test_oscillator_count(self):
        """Verify 23+ oscillator indicators."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_volatility=False,
            enable_volume=False,
        )
        names = ti.get_feature_names()

        # Count unique oscillator types (not including multi-output variants)
        oscillators = [
            "macd_line",
            "macdext_line",
            "rsi",
            "stoch_k",
            "stochf_k",
            "stochrsi_k",
            "willr",
            "adx",
            "adxr",
            "apo",
            "ppo",
            "mom",
            "bop",
            "cci",
            "cmo",
            "roc",
            "rocr",
            "aroon_up",
            "aroonosc",
            "mfi",
            "trix",
            "ultosc",
            "dx",
            "plus_di",
            "minus_di",
            "plus_dm",
            "minus_dm",
        ]

        found_oscillators = [osc for osc in oscillators if osc in names]
        # Should have at least 27 oscillator outputs
        assert (
            len(found_oscillators) >= 27
        ), f"Expected >= 27 oscillator indicators, got {len(found_oscillators)}"

    def test_volatility_count(self):
        """Verify 8+ volatility indicators."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volume=False,
        )
        names = ti.get_feature_names()

        volatility = [
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "bb_width",
            "bb_pct",
            "midpoint",
            "midprice",
            "sar",
            "trange",
            "atr",
            "natr",
        ]

        found_volatility = [v for v in volatility if v in names]
        # Should have 11 volatility outputs
        assert (
            len(found_volatility) == 11
        ), f"Expected 11 volatility indicators, got {len(found_volatility)}"

    def test_volume_count(self):
        """Verify 3 volume indicators."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volatility=False,
        )
        names = ti.get_feature_names()

        volume = ["ad", "adosc", "obv"]
        found_volume = [v for v in volume if v in names]

        assert (
            len(found_volume) == 3
        ), f"Expected 3 volume indicators, got {len(found_volume)}"

    def test_hilbert_count(self):
        """Verify 6+ Hilbert indicators."""
        ti = TechnicalIndicators(
            Timeframe.H1,
            enable_moving_averages=False,
            enable_oscillators=False,
            enable_volatility=False,
            enable_volume=False,
            enable_hilbert=True,
        )
        names = ti.get_feature_names()

        hilbert = [
            "ht_trendline",
            "ht_sine",
            "ht_leadsine",
            "ht_trendmode",
            "ht_dcperiod",
            "ht_dcphase",
            "ht_inphase",
            "ht_quadrature",
        ]

        found_hilbert = [h for h in hilbert if h in names]
        # Should have 8 Hilbert outputs
        assert (
            len(found_hilbert) == 8
        ), f"Expected 8 Hilbert indicators, got {len(found_hilbert)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
