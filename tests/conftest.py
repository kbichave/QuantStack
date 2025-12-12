# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Root pytest configuration for QuantCore tests.

This conftest.py is automatically loaded by pytest and provides:
- Common fixtures available to all test modules
- pytest hooks for test collection and reporting
- Markers for test categorization
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator

import pytest


# Ensure src is in path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


# =============================================================================
# Pytest Hooks
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "benchmark: marks tests as benchmark tests")
    config.addinivalue_line(
        "markers", "requires_api: marks tests that require API access"
    )
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    """Modify test collection based on markers."""
    # Skip slow tests by default unless explicitly requested
    if config.getoption("-m") is None:
        skip_slow = pytest.mark.skip(reason="slow test - use -m slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


# =============================================================================
# Common Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    """Return the test data directory."""
    data_dir = project_root / "tests" / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def temp_db(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary DuckDB database for testing."""
    db_path = tmp_path / "test.duckdb"
    yield db_path
    # Cleanup handled by tmp_path fixture


@pytest.fixture(autouse=True)
def reset_random_seeds() -> None:
    """Reset random seeds before each test for reproducibility."""
    import numpy as np

    np.random.seed(42)

    try:
        import torch

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    except ImportError:
        pass


@pytest.fixture
def sample_ohlcv_df():
    """Create a sample OHLCV DataFrame for testing."""
    import pandas as pd
    import numpy as np

    n_bars = 100
    np.random.seed(42)

    # Generate random walk prices
    returns = np.random.randn(n_bars) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))

    # Create OHLCV
    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_bars) * 0.005),
            "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.01),
            "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.01),
            "close": prices,
            "volume": np.random.randint(1000, 10000, n_bars),
        },
        index=pd.date_range(
            "2024-01-01", periods=n_bars, freq="1D", tz="America/New_York"
        ),
    )

    return df


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    from unittest.mock import MagicMock

    settings = MagicMock()
    settings.market_timezone = "America/New_York"
    settings.database_path = ":memory:"
    settings.alpha_vantage_api_key = "test_key"
    settings.log_level = "DEBUG"
    return settings


# =============================================================================
# Helper Functions for Legacy Tests
# =============================================================================


def make_ohlcv_df(
    prices_or_n_bars=100,
    start_price: float = 100.0,
    vol: float = 0.02,
    seed: int = 42,
    spread_pct: float = 0.0,
    freq: str = "1D",
) -> "pd.DataFrame":
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
    import pandas as pd
    import numpy as np

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
        index=pd.date_range(
            "2024-01-01", periods=n_bars, freq=freq, tz="America/New_York"
        ),
    )

    return df


def add_atr_column(df: "pd.DataFrame", period: int = 14) -> "pd.DataFrame":
    """Add ATR column to DataFrame."""
    import numpy as np

    result = df.copy()
    high = result["high"]
    low = result["low"]
    close = result["close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    import pandas as pd

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    result["atr"] = true_range.ewm(span=period, adjust=False).mean()

    return result


def make_impulse_wave_ohlcv(
    n_bars: int = 100,
    start_price: float = 100.0,
) -> "pd.DataFrame":
    """Create OHLCV data with impulse wave pattern."""
    import pandas as pd
    import numpy as np

    np.random.seed(42)

    # Create 5-wave impulse pattern (up-down-up-down-up)
    wave_lengths = [20, 10, 25, 15, 30]  # Bars per wave
    wave_directions = [1, -1, 1, -1, 1]  # 1=up, -1=down
    wave_magnitudes = [0.10, 0.05, 0.15, 0.07, 0.12]  # % move

    prices = [start_price]
    current_price = start_price

    for length, direction, magnitude in zip(
        wave_lengths, wave_directions, wave_magnitudes
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
        index=pd.date_range(
            "2024-01-01", periods=len(prices), freq="1D", tz="America/New_York"
        ),
    )

    return df


def make_impulse_up_legs():
    """Create list of swing legs for a valid upward impulse pattern.

    Returns a list of 5 swing legs forming a valid Elliott Wave impulse:
    - Wave 1: Up (100 → 110)
    - Wave 2: Down (110 → 106)  - 40% retracement (valid: < 100%)
    - Wave 3: Up (106 → 125)    - Largest wave (valid: > wave 1)
    - Wave 4: Down (125 → 120)  - 26% retracement (valid: < 100% of wave 3)
    - Wave 5: Up (120 → 135)
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

    Supports two calling conventions:
    1. make_swing_leg(start_idx, end_idx, start_price, end_price, direction)
    2. make_swing_leg(start_price, end_price, n_bars=10)  # Legacy

    Returns a SwingLeg-like object with required attributes.
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
    if direction is not None or (
        start_price_or_direction is not None and end_price is not None
    ):
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
) -> "pd.DataFrame":
    """Create monotonic downtrend OHLCV data."""
    import pandas as pd
    import numpy as np

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
        index=pd.date_range(
            "2024-01-01", periods=n_bars, freq="1D", tz="America/New_York"
        ),
    )

    return df


def make_monotonic_uptrend(
    n_bars: int = 50,
    start_price: float = 100.0,
    end_price: float = None,
) -> "pd.DataFrame":
    """Create monotonic uptrend OHLCV data."""
    import pandas as pd
    import numpy as np

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
        index=pd.date_range(
            "2024-01-01", periods=n_bars, freq="1D", tz="America/New_York"
        ),
    )

    return df


def make_flat_market(
    n_bars: int = 50,
    price: float = 100.0,
    noise: float = 0.005,
) -> "pd.DataFrame":
    """Create flat/ranging market OHLCV data."""
    import pandas as pd
    import numpy as np

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
        index=pd.date_range(
            "2024-01-01", periods=n_bars, freq="1D", tz="America/New_York"
        ),
    )

    return df


def make_v_shape_ohlcv(
    n_bars: int = 50,
    start_price: float = 100.0,
    bottom_price: float = None,
    end_price: float = None,
    down_bars: int = None,
    up_bars: int = None,
) -> "pd.DataFrame":
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
    import pandas as pd
    import numpy as np

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
        index=pd.date_range(
            "2024-01-01", periods=actual_bars, freq="1D", tz="America/New_York"
        ),
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
) -> "pd.DataFrame":
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
    import pandas as pd
    import numpy as np

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
        index=pd.date_range(
            "2024-01-01", periods=actual_bars, freq="1D", tz="America/New_York"
        ),
    )

    return df
