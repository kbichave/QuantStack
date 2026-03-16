"""
Consolidated OHLCV generator functions for test fixtures.

This module contains ALL synthetic data generators from both the root conftest
(flexible API, for quant_pod tests) and the unit conftest (strict API, for
wave/indicator tests). Functions are disambiguated by name to avoid signature
conflicts.

Root conftest originals (flexible API):
    make_ohlcv_df, make_v_shape_ohlcv, make_w_shape_ohlcv,
    make_monotonic_uptrend, make_monotonic_downtrend, make_flat_market,
    make_impulse_wave_ohlcv, make_impulse_up_legs, make_swing_leg,
    add_atr_column

Unit conftest originals (strict/explicit API), renamed here:
    make_ohlcv_df_from_prices, make_v_shape_from_params,
    make_w_shape_from_params, make_uptrend_from_params,
    make_downtrend_from_params, make_flat_from_params,
    make_impulse_from_params, make_typed_swing_leg,
    make_typed_impulse_up_legs, make_swing_point, add_atr
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


# =============================================================================
# Root conftest generators (flexible API)
# =============================================================================


def make_ohlcv_df(
    prices_or_n_bars=100,
    start_price: float = 100.0,
    vol: float = 0.02,
    seed: int = 42,
    spread_pct: float = 0.0,
    freq: str = "1D",
) -> pd.DataFrame:
    """Create a sample OHLCV DataFrame for testing.

    Args:
        prices_or_n_bars: Either a list of close prices, or an integer number of bars.
        start_price: Starting price (only used if n_bars is int).
        vol: Volatility for random walk (only used if n_bars is int).
        seed: Random seed.
        spread_pct: Percentage spread for high/low from close (0.5 = 0.5%).
        freq: Frequency for date_range (e.g., "1D", "1h", "4h").

    Returns:
        DataFrame with OHLCV columns.
    """
    np.random.seed(seed)

    # Handle both list of prices and integer n_bars
    if isinstance(prices_or_n_bars, (list, np.ndarray)):
        prices = np.array(prices_or_n_bars, dtype=float)
        n_bars = len(prices)
    else:
        n_bars = prices_or_n_bars
        returns = np.random.randn(n_bars) * vol
        prices = start_price * np.exp(np.cumsum(returns))

    # Calculate high/low - use spread_pct for tight spreads, or default random spread
    if spread_pct > 0:
        # Custom spread percentage - use additive spread
        spread = np.abs(prices) * (spread_pct / 100.0)
        high = prices + spread * (1 + np.abs(np.random.randn(n_bars)) * 0.5)
        low = prices - spread * (1 + np.abs(np.random.randn(n_bars)) * 0.5)
    else:
        # Default random spread (original behavior)
        high = prices * (1 + np.abs(np.random.randn(n_bars)) * 0.01)
        low = prices * (1 - np.abs(np.random.randn(n_bars)) * 0.01)

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_bars) * 0.002),
            "high": high,
            "low": low,
            "close": prices,
            "volume": np.random.randint(1000, 10000, n_bars),
        },
        index=pd.date_range("2024-01-01", periods=n_bars, freq=freq, tz="America/New_York"),
    )

    return df


def add_atr_column(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add ATR column to DataFrame."""

    result = df.copy()
    high = result["high"]
    low = result["low"]
    close = result["close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    result["atr"] = true_range.ewm(span=period, adjust=False).mean()

    return result


def make_impulse_wave_ohlcv(
    n_bars: int = 100,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Create OHLCV data with impulse wave pattern."""
    np.random.seed(42)

    # Create 5-wave impulse pattern (up-down-up-down-up)
    wave_lengths = [20, 10, 25, 15, 30]  # Bars per wave
    wave_directions = [1, -1, 1, -1, 1]  # 1=up, -1=down
    wave_magnitudes = [0.10, 0.05, 0.15, 0.07, 0.12]  # % move

    prices = [start_price]
    current_price = start_price

    for length, direction, magnitude in zip(
        wave_lengths, wave_directions, wave_magnitudes, strict=False
    ):
        target = current_price * (1 + direction * magnitude)
        wave_prices = np.linspace(current_price, target, length)
        prices.extend(wave_prices[1:])  # Skip first (already in list)
        current_price = target

    prices = np.array(prices[:n_bars])

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(len(prices)) * 0.002),
            "high": prices * (1 + np.abs(np.random.randn(len(prices))) * 0.005),
            "low": prices * (1 - np.abs(np.random.randn(len(prices))) * 0.005),
            "close": prices,
            "volume": np.random.randint(1000, 10000, len(prices)),
        },
        index=pd.date_range("2024-01-01", periods=len(prices), freq="1D", tz="America/New_York"),
    )

    return df


def make_impulse_up_legs():
    """Create list of swing legs for a valid upward impulse pattern.

    Returns a list of 5 swing legs forming a valid Elliott Wave impulse:
    - Wave 1: Up (100 -> 110)
    - Wave 2: Down (110 -> 106)  - 40% retracement (valid: < 100%)
    - Wave 3: Up (106 -> 125)    - Largest wave (valid: > wave 1)
    - Wave 4: Down (125 -> 120)  - 26% retracement (valid: < 100% of wave 3)
    - Wave 5: Up (120 -> 135)
    """
    return [
        make_swing_leg(0, 10, 100.0, 110.0, "up"),  # Wave 1: +10%
        make_swing_leg(10, 17, 110.0, 106.0, "down"),  # Wave 2: -4% (40% retrace of W1)
        make_swing_leg(17, 32, 106.0, 125.0, "up"),  # Wave 3: +19% (largest)
        make_swing_leg(32, 40, 125.0, 120.0, "down"),  # Wave 4: -5% (26% retrace of W3)
        make_swing_leg(40, 55, 120.0, 135.0, "up"),  # Wave 5: +15%
    ]


def make_swing_leg(
    start_idx_or_start_price,
    end_idx_or_end_price,
    start_price_or_direction=None,
    end_price=None,
    direction=None,
):
    """Create a swing leg for wave pattern testing.

    Supports two calling conventions -- the active convention is detected by
    whether ``direction`` (or ``end_price``) is supplied:

    **New (preferred):**
        ``make_swing_leg(start_idx, end_idx, start_price, end_price, direction)``
        All five arguments must be provided.

    **Legacy:**
        ``make_swing_leg(start_price, end_price, n_bars=10)``
        Only two positional args; ``direction`` is inferred from the price delta.

    Returns a MockSwingLeg dataclass with ``start_idx``, ``end_idx``,
    ``start_price``, ``end_price``, ``direction``, ``n_bars``, ``ret_pct``,
    and ``length_bars``.
    """
    from dataclasses import dataclass

    @dataclass
    class MockSwingLeg:
        start_idx: int
        end_idx: int
        start_price: float
        end_price: float
        direction: str
        n_bars: int = 10

        @property
        def ret_pct(self) -> float:
            """Return percentage."""
            if self.start_price == 0:
                return 0.0
            return (self.end_price - self.start_price) / self.start_price

        @property
        def length_bars(self) -> int:
            """Number of bars in leg (alias for n_bars)."""
            return self.n_bars

    # Detect calling convention
    if direction is not None or (start_price_or_direction is not None and end_price is not None):
        # New convention: make_swing_leg(start_idx, end_idx, start_price, end_price, direction)
        return MockSwingLeg(
            start_idx=int(start_idx_or_start_price),
            end_idx=int(end_idx_or_end_price),
            start_price=float(start_price_or_direction),
            end_price=float(end_price),
            direction=(
                direction
                if direction
                else ("up" if end_price > start_price_or_direction else "down")
            ),
            n_bars=int(end_idx_or_end_price) - int(start_idx_or_start_price),
        )
    else:
        # Legacy convention: make_swing_leg(start_price, end_price, n_bars=10)
        sp = float(start_idx_or_start_price)
        ep = float(end_idx_or_end_price)
        nb = int(start_price_or_direction) if start_price_or_direction else 10
        return MockSwingLeg(
            start_idx=0,
            end_idx=nb,
            start_price=sp,
            end_price=ep,
            direction="up" if ep > sp else "down",
            n_bars=nb,
        )


def make_monotonic_downtrend(
    n_bars: int = 50,
    start_price: float = 100.0,
    end_price: float = None,
) -> pd.DataFrame:
    """Create monotonic downtrend OHLCV data."""
    np.random.seed(42)
    if end_price is None:
        end_price = start_price * 0.8
    prices = np.linspace(start_price, end_price, n_bars)

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_bars) * 0.002),
            "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.005),
            "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.005),
            "close": prices,
            "volume": np.random.randint(1000, 10000, n_bars),
        },
        index=pd.date_range("2024-01-01", periods=n_bars, freq="1D", tz="America/New_York"),
    )

    return df


def make_monotonic_uptrend(
    n_bars: int = 50,
    start_price: float = 100.0,
    end_price: float = None,
) -> pd.DataFrame:
    """Create monotonic uptrend OHLCV data."""
    np.random.seed(42)
    if end_price is None:
        end_price = start_price * 1.2
    prices = np.linspace(start_price, end_price, n_bars)

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_bars) * 0.002),
            "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.005),
            "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.005),
            "close": prices,
            "volume": np.random.randint(1000, 10000, n_bars),
        },
        index=pd.date_range("2024-01-01", periods=n_bars, freq="1D", tz="America/New_York"),
    )

    return df


def make_flat_market(
    n_bars: int = 50,
    price: float = 100.0,
    noise: float = 0.005,
) -> pd.DataFrame:
    """Create flat/ranging market OHLCV data."""
    np.random.seed(42)
    prices = price + np.random.randn(n_bars) * price * noise

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_bars) * 0.002),
            "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.005),
            "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.005),
            "close": prices,
            "volume": np.random.randint(1000, 10000, n_bars),
        },
        index=pd.date_range("2024-01-01", periods=n_bars, freq="1D", tz="America/New_York"),
    )

    return df


def make_v_shape_ohlcv(
    n_bars: int = 50,
    start_price: float = 100.0,
    bottom_price: float = None,
    end_price: float = None,
    down_bars: int = None,
    up_bars: int = None,
) -> pd.DataFrame:
    """Create V-shape reversal OHLCV data.

    Args:
        n_bars: Total number of bars (used if down_bars/up_bars not specified).
        start_price: Starting price.
        bottom_price: Price at the bottom of the V.
        end_price: Ending price (defaults to start_price).
        down_bars: Number of bars in the down leg.
        up_bars: Number of bars in the up leg.

    Returns:
        DataFrame with V-shape price pattern.
    """
    np.random.seed(42)
    if bottom_price is None:
        bottom_price = start_price * 0.85
    if end_price is None:
        end_price = start_price

    # Calculate bar counts
    if down_bars is not None and up_bars is not None:
        total_bars = down_bars + up_bars
    else:
        total_bars = n_bars
        down_bars = n_bars // 2
        up_bars = n_bars - down_bars

    down_prices = np.linspace(start_price, bottom_price, down_bars)
    up_prices = np.linspace(bottom_price, end_price, up_bars)
    prices = np.concatenate([down_prices, up_prices[1:]])

    # Ensure correct length
    actual_bars = len(prices)
    if actual_bars < total_bars:
        prices = np.append(prices, [prices[-1]] * (total_bars - actual_bars))
    elif actual_bars > total_bars:
        prices = prices[:total_bars]

    actual_bars = len(prices)

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(actual_bars) * 0.002),
            "high": prices * (1 + np.abs(np.random.randn(actual_bars)) * 0.005),
            "low": prices * (1 - np.abs(np.random.randn(actual_bars)) * 0.005),
            "close": prices,
            "volume": np.random.randint(1000, 10000, actual_bars),
        },
        index=pd.date_range("2024-01-01", periods=actual_bars, freq="1D", tz="America/New_York"),
    )

    return df


def make_w_shape_ohlcv(
    n_bars: int = 60,
    start_price: float = 100.0,
    low1: float = None,
    mid_high: float = None,
    low2: float = None,
    end_price: float = None,
    bars_per_leg: int = None,
) -> pd.DataFrame:
    """Create W-shape (double bottom) OHLCV data.

    Args:
        n_bars: Total number of bars (used if bars_per_leg not specified).
        start_price: Starting price.
        low1: First bottom price.
        mid_high: Middle peak price.
        low2: Second bottom price.
        end_price: Ending price.
        bars_per_leg: Number of bars per leg (total bars = 4 * bars_per_leg).

    Returns:
        DataFrame with W-shape price pattern.
    """
    np.random.seed(42)

    # Set defaults
    if low1 is None:
        low1 = start_price * 0.85
    if mid_high is None:
        mid_high = start_price * 0.95
    if low2 is None:
        low2 = start_price * 0.85
    if end_price is None:
        end_price = start_price

    # Calculate bar counts
    if bars_per_leg is not None:
        quarter = bars_per_leg
        total_bars = 4 * bars_per_leg
    else:
        quarter = n_bars // 4
        total_bars = n_bars

    # Down to first bottom
    p1 = np.linspace(start_price, low1, quarter)
    # Up to middle peak
    p2 = np.linspace(low1, mid_high, quarter)
    # Down to second bottom
    p3 = np.linspace(mid_high, low2, quarter)
    # Up to recovery
    remaining_bars = total_bars - 3 * quarter
    p4 = np.linspace(low2, end_price, max(remaining_bars, 2))

    prices = np.concatenate([p1, p2[1:], p3[1:], p4[1:]])

    # Ensure correct length
    actual_bars = len(prices)
    if actual_bars < total_bars:
        prices = np.append(prices, [prices[-1]] * (total_bars - actual_bars))
    elif actual_bars > total_bars:
        prices = prices[:total_bars]

    actual_bars = len(prices)

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(actual_bars) * 0.002),
            "high": prices * (1 + np.abs(np.random.randn(actual_bars)) * 0.005),
            "low": prices * (1 - np.abs(np.random.randn(actual_bars)) * 0.005),
            "close": prices,
            "volume": np.random.randint(1000, 10000, actual_bars),
        },
        index=pd.date_range("2024-01-01", periods=actual_bars, freq="1D", tz="America/New_York"),
    )

    return df


# =============================================================================
# Unit conftest generators (strict/explicit API, renamed)
# =============================================================================


def make_ohlcv_df_from_prices(
    prices: list[float],
    start: str = "2024-01-01 09:00",
    freq: str = "1h",
    volumes: list[int] | None = None,
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
    highs = [max(o, c) + s for o, c, s in zip(opens, prices, spread, strict=False)]
    lows = [min(o, c) - s for o, c, s in zip(opens, prices, spread, strict=False)]

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


def make_v_shape_from_params(
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
    return make_ohlcv_df_from_prices(prices, start=start, freq=freq)


def make_w_shape_from_params(
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
    return make_ohlcv_df_from_prices(prices, start=start, freq=freq)


def make_uptrend_from_params(
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
    return make_ohlcv_df_from_prices(prices, start=start, freq=freq, spread_pct=0.1)


def make_downtrend_from_params(
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
    return make_ohlcv_df_from_prices(prices, start=start, freq=freq, spread_pct=0.1)


def make_flat_from_params(
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
    return make_ohlcv_df_from_prices(prices, start=start, freq=freq, spread_pct=0.05)


def make_impulse_from_params(
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
    return make_ohlcv_df_from_prices(prices, start=start, freq=freq)


# Alias: identical implementation to add_atr_column
add_atr = add_atr_column


def make_typed_swing_leg(
    start_idx: int,
    end_idx: int,
    start_price: float,
    end_price: float,
    direction: Literal["up", "down"],
):
    """
    Create a SwingLeg (from quantcore.features.waves) directly for testing.

    Computes ret_pct and length_bars automatically.
    """
    from quantcore.features.waves import SwingLeg

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


def make_typed_impulse_up_legs(
    wave1_ret: float = 0.10,
    wave2_ret: float = -0.05,
    wave3_ret: float = 0.15,
    wave4_ret: float = -0.04,
    wave5_ret: float = 0.08,
    bars_per_leg: int = 5,
    start_price: float = 100.0,
):
    """
    Create 5 legs forming a valid impulse-up pattern.

    Returns list of SwingLeg (from quantcore.features.waves).

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

    for ret, direction in zip(returns, directions, strict=False):
        start_idx = idx
        end_idx = idx + bars_per_leg
        end_price = price * (1 + ret)

        legs.append(
            make_typed_swing_leg(
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
    time: pd.Timestamp | None = None,
):
    """
    Create a SwingPoint (from quantcore.features.waves) for testing.
    """
    from quantcore.features.waves import SwingPoint

    if time is None:
        time = pd.Timestamp("2024-01-01 09:00") + pd.Timedelta(hours=idx)
    return SwingPoint(idx=idx, time=time, price=price, direction=direction)
