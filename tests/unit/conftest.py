"""
Shared test fixtures and synthetic data generators for pytest.

Provides:
- Synthetic OHLCV DataFrame generators for various price patterns
- ATR computation helper
- Mock settings for avoiding environment dependencies
- Helper functions for constructing swing legs directly

Generator functions are imported from tests._fixtures.ohlcv_generators
(single source of truth) and re-exported under the original names used
by unit tests.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from quantstack.core.features.waves import WaveConfig

# Import renamed generators from shared fixtures and re-export under
# the original names that unit tests expect.
from tests._fixtures.ohlcv_generators import (
    add_atr_column,  # noqa: F401 — identical impl, same name
    make_downtrend_from_params as make_monotonic_downtrend,  # noqa: F401
    make_flat_from_params as make_flat_market,  # noqa: F401
    make_impulse_from_params as make_impulse_wave_ohlcv,  # noqa: F401
    make_ohlcv_df_from_prices as make_ohlcv_df,  # noqa: F401
    make_swing_point,  # noqa: F401 — same name, no conflict
    make_typed_impulse_up_legs as make_impulse_up_legs,  # noqa: F401
    make_typed_swing_leg as make_swing_leg,  # noqa: F401
    make_uptrend_from_params as make_monotonic_uptrend,  # noqa: F401
    make_v_shape_from_params as make_v_shape_ohlcv,  # noqa: F401
    make_w_shape_from_params as make_w_shape_ohlcv,  # noqa: F401
)

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
    with patch("quantstack.config.settings.get_settings", return_value=mock_settings):
        yield mock_settings


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


# =============================================================================
# Async Test Support
# =============================================================================


@pytest.fixture
def run_async():
    """
    Fixture that provides a clean way to run async functions in sync tests.

    This properly handles event loop lifecycle per test, avoiding the
    "coroutine was never awaited" issues that arise when using
    asyncio.get_event_loop().run_until_complete() directly across
    multiple tests in a suite.

    Usage:
        def test_something(run_async):
            result = run_async(my_async_function(arg1, arg2))
            assert result == expected
    """
    import asyncio

    def _run(coro):
        # Create a fresh event loop for this test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            # Clean up properly
            try:
                # Cancel all pending tasks
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                # Run loop once more to let tasks finish cancelling
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            finally:
                loop.close()
                # Remove the event loop to prevent pollution
                asyncio.set_event_loop(None)

    return _run
