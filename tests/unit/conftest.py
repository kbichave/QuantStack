"""
Shared test fixtures and synthetic data generators for pytest.

Provides:
- Synthetic OHLCV DataFrame generators for various price patterns
- ATR computation helper
- Mock settings for avoiding environment dependencies
- Helper functions for constructing swing legs directly
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Literal, Optional
from unittest.mock import MagicMock, patch

from quantcore.config.timeframes import Timeframe
from quantcore.features.waves import SwingPoint, SwingLeg, WaveConfig


# =============================================================================
# Mock Settings
# =============================================================================


@pytest.fixture
def mock_settings():
    """
    Mock settings to avoid environment variable dependencies.
    Returns a MagicMock with default market timezone.
    """
    settings = MagicMock()
    settings.market_timezone = "America/New_York"
    return settings


@pytest.fixture
def patch_get_settings(mock_settings):
    """
    Patch get_settings() to return mock settings.
    Use this fixture when testing modules that depend on settings.
    """
    with patch("quantcore.config.settings.get_settings", return_value=mock_settings):
        yield mock_settings


# =============================================================================
# Synthetic OHLCV Generators
# =============================================================================


def make_ohlcv_df(
    prices: List[float],
    start: str = "2024-01-01 09:00",
    freq: str = "1h",
    volumes: Optional[List[int]] = None,
    spread_pct: float = 0.5,
) -> pd.DataFrame:
    """
    Create a synthetic OHLCV DataFrame from a list of close prices.

    The high/low are generated as close +/- spread_pct%.
    Open is set to previous close (or first close for bar 0).

    Args:
        prices: List of close prices
        start: Start timestamp string
        freq: Pandas frequency string (e.g., "1H", "4H", "D")
        volumes: Optional list of volumes (defaults to 1000 per bar)
        spread_pct: Percentage to add/subtract for high/low

    Returns:
        DataFrame with columns: open, high, low, close, volume
    """
    n = len(prices)
    if volumes is None:
        volumes = [1000] * n

    # Generate timestamps
    index = pd.date_range(start=start, periods=n, freq=freq, tz="America/New_York")

    # Build OHLCV
    opens = [prices[0]] + prices[:-1]  # Open = previous close
    spread = [p * spread_pct / 100 for p in prices]
    highs = [max(o, c) + s for o, c, s in zip(opens, prices, spread)]
    lows = [min(o, c) - s for o, c, s in zip(opens, prices, spread)]

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
        },
        index=index,
    )

    return df


def make_v_shape_ohlcv(
    start_price: float = 100.0,
    bottom_price: float = 90.0,
    end_price: float = 100.0,
    down_bars: int = 10,
    up_bars: int = 10,
    start: str = "2024-01-01 09:00",
    freq: str = "1h",
) -> pd.DataFrame:
    """
    Create a V-shaped price series: decline then rise.

    Scenario: Price starts at start_price, declines linearly to bottom_price,
    then rises linearly to end_price.
    """
    down_prices = np.linspace(start_price, bottom_price, down_bars).tolist()
    up_prices = np.linspace(bottom_price, end_price, up_bars + 1)[
        1:
    ].tolist()  # Skip duplicate bottom
    prices = down_prices + up_prices
    return make_ohlcv_df(prices, start=start, freq=freq)


def make_w_shape_ohlcv(
    start_price: float = 100.0,
    low1: float = 90.0,
    mid_high: float = 95.0,
    low2: float = 88.0,
    end_price: float = 100.0,
    bars_per_leg: int = 5,
    start: str = "2024-01-01 09:00",
    freq: str = "1h",
) -> pd.DataFrame:
    """
    Create a W-shaped price series: down, up, down, up.

    Scenario: Starts high, drops to low1, rises to mid_high,
    drops to low2, then rises to end_price.
    """
    leg1 = np.linspace(start_price, low1, bars_per_leg).tolist()
    leg2 = np.linspace(low1, mid_high, bars_per_leg + 1)[1:].tolist()
    leg3 = np.linspace(mid_high, low2, bars_per_leg + 1)[1:].tolist()
    leg4 = np.linspace(low2, end_price, bars_per_leg + 1)[1:].tolist()
    prices = leg1 + leg2 + leg3 + leg4
    return make_ohlcv_df(prices, start=start, freq=freq)


def make_monotonic_uptrend(
    start_price: float = 100.0,
    end_price: float = 120.0,
    n_bars: int = 20,
    start: str = "2024-01-01 09:00",
    freq: str = "1h",
) -> pd.DataFrame:
    """
    Create a monotonically increasing price series.

    Scenario: Steady uptrend with no pullbacks.
    """
    prices = np.linspace(start_price, end_price, n_bars).tolist()
    return make_ohlcv_df(prices, start=start, freq=freq, spread_pct=0.1)


def make_monotonic_downtrend(
    start_price: float = 100.0,
    end_price: float = 80.0,
    n_bars: int = 20,
    start: str = "2024-01-01 09:00",
    freq: str = "1h",
) -> pd.DataFrame:
    """
    Create a monotonically decreasing price series.

    Scenario: Steady downtrend with no bounces.
    """
    prices = np.linspace(start_price, end_price, n_bars).tolist()
    return make_ohlcv_df(prices, start=start, freq=freq, spread_pct=0.1)


def make_flat_market(
    price: float = 100.0,
    n_bars: int = 20,
    start: str = "2024-01-01 09:00",
    freq: str = "1h",
) -> pd.DataFrame:
    """
    Create a flat price series with no movement.

    Scenario: Price stays constant - no swings should be detected.
    """
    prices = [price] * n_bars
    return make_ohlcv_df(prices, start=start, freq=freq, spread_pct=0.05)


def make_impulse_wave_ohlcv(
    start_price: float = 100.0,
    wave1_end: float = 110.0,
    wave2_end: float = 105.0,
    wave3_end: float = 125.0,
    wave4_end: float = 118.0,
    wave5_end: float = 135.0,
    bars_per_wave: int = 5,
    start: str = "2024-01-01 09:00",
    freq: str = "1h",
) -> pd.DataFrame:
    """
    Create an impulse wave pattern (5 waves: up, down, up, down, up).

    Scenario: Classic Elliott Wave impulse structure suitable for
    testing wave pattern detection.
    """
    w1 = np.linspace(start_price, wave1_end, bars_per_wave).tolist()
    w2 = np.linspace(wave1_end, wave2_end, bars_per_wave + 1)[1:].tolist()
    w3 = np.linspace(wave2_end, wave3_end, bars_per_wave + 1)[1:].tolist()
    w4 = np.linspace(wave3_end, wave4_end, bars_per_wave + 1)[1:].tolist()
    w5 = np.linspace(wave4_end, wave5_end, bars_per_wave + 1)[1:].tolist()
    prices = w1 + w2 + w3 + w4 + w5
    return make_ohlcv_df(prices, start=start, freq=freq)


# =============================================================================
# ATR Helpers
# =============================================================================


def add_atr_column(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add ATR column to OHLCV DataFrame.

    Uses standard ATR calculation: EMA of true range.
    """
    result = df.copy()
    high = result["high"]
    low = result["low"]
    close = result["close"]

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR as EMA of TR
    result["atr"] = tr.ewm(span=period, adjust=False).mean()

    return result


# =============================================================================
# Swing Leg Helpers
# =============================================================================


def make_swing_leg(
    start_idx: int,
    end_idx: int,
    start_price: float,
    end_price: float,
    direction: Literal["up", "down"],
) -> SwingLeg:
    """
    Create a SwingLeg directly for testing wave detection.

    Computes ret_pct and length_bars automatically.
    """
    ret_pct = (end_price - start_price) / start_price if start_price != 0 else 0
    return SwingLeg(
        start_idx=start_idx,
        end_idx=end_idx,
        start_price=start_price,
        end_price=end_price,
        direction=direction,
        ret_pct=ret_pct,
        length_bars=end_idx - start_idx,
    )


def make_impulse_up_legs(
    wave1_ret: float = 0.10,
    wave2_ret: float = -0.05,
    wave3_ret: float = 0.15,
    wave4_ret: float = -0.04,
    wave5_ret: float = 0.08,
    bars_per_leg: int = 5,
    start_price: float = 100.0,
) -> List[SwingLeg]:
    """
    Create 5 legs forming a valid impulse-up pattern.

    Default returns:
    - Wave1: +10%
    - Wave2: -5% (50% retrace of wave1)
    - Wave3: +15% (larger than wave1)
    - Wave4: -4% (~27% retrace of wave3)
    - Wave5: +8%
    """
    legs = []
    price = start_price
    idx = 0

    returns = [wave1_ret, wave2_ret, wave3_ret, wave4_ret, wave5_ret]
    directions = ["up", "down", "up", "down", "up"]

    for ret, direction in zip(returns, directions):
        start_idx = idx
        end_idx = idx + bars_per_leg
        end_price = price * (1 + ret)

        legs.append(
            make_swing_leg(
                start_idx=start_idx,
                end_idx=end_idx,
                start_price=price,
                end_price=end_price,
                direction=direction,
            )
        )

        price = end_price
        idx = end_idx

    return legs


def make_swing_point(
    idx: int,
    price: float,
    direction: Literal["up", "down"],
    time: Optional[pd.Timestamp] = None,
) -> SwingPoint:
    """
    Create a SwingPoint for testing.
    """
    if time is None:
        time = pd.Timestamp("2024-01-01 09:00") + pd.Timedelta(hours=idx)
    return SwingPoint(idx=idx, time=time, price=price, direction=direction)


# =============================================================================
# Pytest Fixtures for Common Patterns
# =============================================================================


@pytest.fixture
def simple_hourly_ohlcv():
    """
    Simple 8-bar hourly OHLCV for basic resampling tests.

    Bars have distinct values to make aggregation verification easy.
    """
    return pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105, 106, 107],
            "high": [102, 103, 104, 105, 106, 107, 108, 109],
            "low": [99, 100, 101, 102, 103, 104, 105, 106],
            "close": [101, 102, 103, 104, 105, 106, 107, 108],
            "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700],
        },
        index=pd.date_range(
            "2024-01-02 09:00", periods=8, freq="1H", tz="America/New_York"
        ),
    )


@pytest.fixture
def v_shape_df():
    """V-shaped price series for swing detection tests."""
    return make_v_shape_ohlcv()


@pytest.fixture
def w_shape_df():
    """W-shaped price series for swing detection tests."""
    return make_w_shape_ohlcv()


@pytest.fixture
def monotonic_up_df():
    """Monotonic uptrend for edge case tests."""
    return make_monotonic_uptrend()


@pytest.fixture
def flat_df():
    """Flat market for edge case tests."""
    return make_flat_market()


@pytest.fixture
def impulse_wave_df():
    """5-wave impulse pattern for wave detection tests."""
    return make_impulse_wave_ohlcv()


@pytest.fixture
def valid_impulse_legs():
    """5 legs forming a valid impulse-up pattern."""
    return make_impulse_up_legs()


@pytest.fixture
def wave_config():
    """Default wave configuration for testing."""
    return WaveConfig()
