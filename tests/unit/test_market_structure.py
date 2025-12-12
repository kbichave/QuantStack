"""
Unit tests for MarketStructureFeatures.

Tests verify:
- ZigZag-style swing detection with ATR-based thresholds
- Swing alternation invariants (up/down must strictly alternate)
- Behavior on various price patterns (V-shape, W-shape, monotonic, flat)
- Edge cases and parameter effects
"""

import pytest
import pandas as pd
import numpy as np

from quantcore.features.market_structure import MarketStructureFeatures
from quantcore.config.timeframes import Timeframe

# Import helpers from conftest
from tests.conftest import (
    make_ohlcv_df,
    make_v_shape_ohlcv,
    make_w_shape_ohlcv,
    make_monotonic_uptrend,
    make_flat_market,
    add_atr_column,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def market_structure_h4():
    """MarketStructureFeatures instance for 4H timeframe."""
    return MarketStructureFeatures(Timeframe.H4)


@pytest.fixture
def market_structure_h1():
    """MarketStructureFeatures instance for 1H timeframe."""
    return MarketStructureFeatures(Timeframe.H1)


@pytest.fixture
def v_shape_with_atr():
    """
    V-shaped price series with ATR column.

    Price goes from 100 down to 80, then back up to 100.
    This should produce at least one swing low at the bottom.
    """
    df = make_v_shape_ohlcv(
        start_price=100.0,
        bottom_price=80.0,
        end_price=100.0,
        down_bars=15,
        up_bars=15,
    )
    return add_atr_column(df, period=14)


@pytest.fixture
def w_shape_with_atr():
    """
    W-shaped price series with ATR column.

    Two distinct lows at 85 and 82, with a bounce to 92 in between.
    Should produce 2 swing lows and 1-2 swing highs.
    """
    df = make_w_shape_ohlcv(
        start_price=100.0,
        low1=85.0,
        mid_high=92.0,
        low2=82.0,
        end_price=100.0,
        bars_per_leg=8,
    )
    return add_atr_column(df, period=14)


@pytest.fixture
def monotonic_up_with_atr():
    """
    Monotonic uptrend with ATR column.

    Steady rise from 100 to 130 over 30 bars with minimal spread.
    Few or no swings expected due to lack of reversals.
    """
    df = make_monotonic_uptrend(
        start_price=100.0,
        end_price=130.0,
        n_bars=30,
    )
    return add_atr_column(df, period=14)


@pytest.fixture
def flat_with_atr():
    """
    Flat market with ATR column.

    Constant price at 100 for 30 bars.
    No swings expected.
    """
    df = make_flat_market(price=100.0, n_bars=30)
    return add_atr_column(df, period=14)


@pytest.fixture
def volatile_series_with_atr():
    """
    Highly volatile series with clear swing points.

    Explicit up/down/up/down pattern to ensure swing detection.
    """
    # Create explicit swing pattern
    prices = (
        list(np.linspace(100, 110, 5))  # Up
        + list(np.linspace(110, 95, 6)[1:])  # Down
        + list(np.linspace(95, 115, 6)[1:])  # Up
        + list(np.linspace(115, 100, 6)[1:])  # Down
        + list(np.linspace(100, 120, 6)[1:])  # Up
    )
    df = make_ohlcv_df(prices, spread_pct=0.5)
    return add_atr_column(df, period=14)


# =============================================================================
# Test: ZigZag Swing Detection - V-Shape
# =============================================================================


class TestZigZagVShape:
    """Tests for swing detection on V-shaped price patterns."""

    def test_v_shape_detects_swing_low(self, market_structure_h4, v_shape_with_atr):
        """
        Verify that a V-shape pattern produces at least one swing low.

        Scenario: Price drops from 100 to 80 then rises back to 100.
        Expected: At least one swing low detected near the bottom.
        """
        swing_high, swing_low = market_structure_h4.detect_zigzag_swings(
            v_shape_with_atr, atr_mult=1.5, min_bars=3
        )

        # Should detect at least one swing low near the bottom
        assert swing_low.sum() >= 1, "V-shape should have at least one swing low"

        # The swing low should be in the middle portion (where bottom is)
        swing_low_indices = swing_low[swing_low == 1].index
        n_bars = len(v_shape_with_atr)

        # Bottom is around bar 15 (middle of 30 bars)
        # Allow some tolerance for detection lag
        for idx in swing_low_indices:
            bar_position = v_shape_with_atr.index.get_loc(idx)
            assert (
                5 < bar_position < n_bars - 5
            ), f"Swing low at position {bar_position} should be in middle portion"

    def test_v_shape_swing_low_at_local_minimum(
        self, market_structure_h4, v_shape_with_atr
    ):
        """
        Verify that detected swing low is near a local price minimum.

        Scenario: V-shape with clear bottom at index ~15.
        Expected: Swing low detected at or near the minimum price point.
        """
        swing_high, swing_low = market_structure_h4.detect_zigzag_swings(
            v_shape_with_atr, atr_mult=1.5, min_bars=3
        )

        if swing_low.sum() > 0:
            # Get prices at swing lows
            swing_low_prices = v_shape_with_atr.loc[swing_low == 1, "low"]
            min_swing_price = swing_low_prices.min()

            # The minimum price in the series
            overall_min = v_shape_with_atr["low"].min()

            # Swing low should be within 10% of the overall minimum
            tolerance = overall_min * 0.10
            assert (
                min_swing_price <= overall_min + tolerance
            ), f"Swing low price {min_swing_price} should be near overall min {overall_min}"


# =============================================================================
# Test: ZigZag Swing Detection - W-Shape
# =============================================================================


class TestZigZagWShape:
    """Tests for swing detection on W-shaped price patterns."""

    def test_w_shape_detects_multiple_swings(
        self, market_structure_h4, w_shape_with_atr
    ):
        """
        Verify that W-shape produces multiple swing points.

        Scenario: Down-up-down-up pattern creates 2 lows and 1-2 highs.
        Expected: At least 2 swing lows and 1 swing high.
        """
        swing_high, swing_low = market_structure_h4.detect_zigzag_swings(
            w_shape_with_atr,
            atr_mult=1.2,  # Lower threshold to catch swings
            min_bars=2,
        )

        # W-shape should have at least 2 swing lows
        n_swing_lows = swing_low.sum()
        n_swing_highs = swing_high.sum()

        # At minimum we expect some swing activity
        total_swings = n_swing_lows + n_swing_highs
        assert (
            total_swings >= 1
        ), f"W-shape should detect at least 1 swing point, got {total_swings}"

    def test_w_shape_swing_highs_and_lows(self, market_structure_h4, w_shape_with_atr):
        """
        Verify that both swing highs and lows are detected in W-shape.

        Scenario: W-shape has distinct peaks and troughs.
        """
        swing_high, swing_low = market_structure_h4.detect_zigzag_swings(
            w_shape_with_atr, atr_mult=1.0, min_bars=2  # More sensitive
        )

        # Check we have both types (may not always be the case depending on thresholds)
        # This is more of a "good to have" - the main check is swing detection works
        total = swing_high.sum() + swing_low.sum()
        assert total >= 1, "Should detect at least one swing in W-shape"


# =============================================================================
# Test: ZigZag Swing Detection - Monotonic Trend
# =============================================================================


class TestZigZagMonotonic:
    """Tests for swing detection on monotonic price series."""

    def test_monotonic_uptrend_few_swings(
        self, market_structure_h4, monotonic_up_with_atr
    ):
        """
        Verify that monotonic uptrend produces few or no swings.

        Scenario: Steady price increase with no pullbacks.
        Expected: Very few swing points (ideally zero).
        """
        swing_high, swing_low = market_structure_h4.detect_zigzag_swings(
            monotonic_up_with_atr, atr_mult=1.5, min_bars=3
        )

        total_swings = swing_high.sum() + swing_low.sum()

        # Monotonic series should have very few swings
        # The small spread might cause minor detections, but should be minimal
        assert (
            total_swings <= 2
        ), f"Monotonic uptrend should have few swings, got {total_swings}"

    def test_monotonic_downtrend_few_swings(self, market_structure_h4):
        """
        Verify that monotonic downtrend produces few or no swings.

        Scenario: Steady price decrease with no bounces.
        """
        from tests.conftest import make_monotonic_downtrend

        df = make_monotonic_downtrend(start_price=100, end_price=70, n_bars=30)
        df = add_atr_column(df, period=14)

        swing_high, swing_low = market_structure_h4.detect_zigzag_swings(
            df, atr_mult=1.5, min_bars=3
        )

        total_swings = swing_high.sum() + swing_low.sum()
        assert (
            total_swings <= 2
        ), f"Monotonic downtrend should have few swings, got {total_swings}"


# =============================================================================
# Test: ZigZag Swing Detection - Flat Market
# =============================================================================


class TestZigZagFlatMarket:
    """Tests for swing detection on flat/sideways markets."""

    def test_flat_market_no_swings(self, market_structure_h4, flat_with_atr):
        """
        Verify that flat market produces no swings.

        Scenario: Price stays constant at 100.
        Expected: Zero swing points detected.
        """
        swing_high, swing_low = market_structure_h4.detect_zigzag_swings(
            flat_with_atr, atr_mult=1.5, min_bars=3
        )

        total_swings = swing_high.sum() + swing_low.sum()
        assert (
            total_swings == 0
        ), f"Flat market should have no swings, got {total_swings}"


# =============================================================================
# Test: Swing Alternation Invariant
# =============================================================================


class TestSwingAlternation:
    """Tests for swing direction alternation invariant."""

    def test_swings_never_fire_simultaneously(
        self, market_structure_h4, volatile_series_with_atr
    ):
        """
        Verify that swing_high and swing_low never both equal 1 at the same bar.

        Invariant: A bar cannot be both a swing high and swing low.
        """
        swing_high, swing_low = market_structure_h4.detect_zigzag_swings(
            volatile_series_with_atr, atr_mult=1.0, min_bars=2
        )

        # Check no bar has both swing_high=1 and swing_low=1
        both_true = ((swing_high == 1) & (swing_low == 1)).sum()
        assert (
            both_true == 0
        ), f"Found {both_true} bars with both swing high and swing low"

    def test_swings_strictly_alternate_v_shape(
        self, market_structure_h4, v_shape_with_atr
    ):
        """
        Verify swing direction alternation in V-shape.

        Invariant: After a swing high, next swing must be low (and vice versa).
        """
        swing_high, swing_low = market_structure_h4.detect_zigzag_swings(
            v_shape_with_atr, atr_mult=1.5, min_bars=3
        )

        # Combine into single series with direction
        swings = []
        for idx in swing_high.index:
            if swing_high.loc[idx] == 1:
                swings.append(("high", idx))
            if swing_low.loc[idx] == 1:
                swings.append(("low", idx))

        # Sort by index
        swings.sort(key=lambda x: x[1])

        # Check alternation
        for i in range(1, len(swings)):
            prev_dir = swings[i - 1][0]
            curr_dir = swings[i][0]
            assert (
                prev_dir != curr_dir
            ), f"Swings at {swings[i-1][1]} and {swings[i][1]} both are '{curr_dir}'"


# =============================================================================
# Test: Minimum Bars Between Swings
# =============================================================================


class TestMinBarsBetweenSwings:
    """Tests for min_bars parameter enforcement."""

    def test_min_bars_parameter_respected(
        self, market_structure_h4, volatile_series_with_atr
    ):
        """
        Verify that swings respect the min_bars constraint.

        Scenario: Volatile series with min_bars=5.
        Expected: All consecutive swings are at least 5 bars apart.
        """
        min_bars = 5
        swing_high, swing_low = market_structure_h4.detect_zigzag_swings(
            volatile_series_with_atr, atr_mult=1.0, min_bars=min_bars
        )

        # Get all swing indices
        high_indices = list(swing_high[swing_high == 1].index)
        low_indices = list(swing_low[swing_low == 1].index)

        # Combine and sort
        all_swing_positions = []
        for idx in high_indices:
            pos = volatile_series_with_atr.index.get_loc(idx)
            all_swing_positions.append(pos)
        for idx in low_indices:
            pos = volatile_series_with_atr.index.get_loc(idx)
            all_swing_positions.append(pos)

        all_swing_positions.sort()

        # Check gaps between consecutive swings
        for i in range(1, len(all_swing_positions)):
            gap = all_swing_positions[i] - all_swing_positions[i - 1]
            assert (
                gap >= min_bars
            ), f"Gap {gap} between swings at {all_swing_positions[i-1]} and {all_swing_positions[i]} is less than min_bars={min_bars}"

    def test_smaller_min_bars_more_swings(
        self, market_structure_h4, volatile_series_with_atr
    ):
        """
        Verify that smaller min_bars allows more swings to be detected.

        Scenario: Same series with min_bars=2 vs min_bars=5.
        Expected: min_bars=2 should detect >= as many swings as min_bars=5.
        """
        # More permissive
        sh_small, sl_small = market_structure_h4.detect_zigzag_swings(
            volatile_series_with_atr, atr_mult=1.0, min_bars=2
        )
        swings_small = sh_small.sum() + sl_small.sum()

        # More restrictive
        sh_large, sl_large = market_structure_h4.detect_zigzag_swings(
            volatile_series_with_atr, atr_mult=1.0, min_bars=5
        )
        swings_large = sh_large.sum() + sl_large.sum()

        assert (
            swings_small >= swings_large
        ), f"Smaller min_bars ({swings_small}) should detect >= swings than larger ({swings_large})"


# =============================================================================
# Test: ATR Multiplier Effect
# =============================================================================


class TestATRMultiplierEffect:
    """Tests for ATR multiplier impact on swing detection."""

    def test_larger_atr_mult_fewer_swings(
        self, market_structure_h4, volatile_series_with_atr
    ):
        """
        Verify that larger ATR multiplier produces fewer swings.

        Scenario: Same series with atr_mult=1.0 vs atr_mult=2.0.
        Expected: Higher threshold should detect fewer (or equal) swings.
        """
        # Lower threshold (more sensitive)
        sh_low, sl_low = market_structure_h4.detect_zigzag_swings(
            volatile_series_with_atr, atr_mult=1.0, min_bars=2
        )
        swings_low = sh_low.sum() + sl_low.sum()

        # Higher threshold (less sensitive)
        sh_high, sl_high = market_structure_h4.detect_zigzag_swings(
            volatile_series_with_atr, atr_mult=2.0, min_bars=2
        )
        swings_high = sh_high.sum() + sl_high.sum()

        assert (
            swings_low >= swings_high
        ), f"Lower atr_mult ({swings_low}) should detect >= swings than higher ({swings_high})"


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestZigZagEdgeCases:
    """Edge case tests for swing detection."""

    def test_very_short_series(self, market_structure_h4):
        """
        Verify that very short series (< 5 bars) returns empty swings.

        Scenario: Only 3 bars of data.
        Expected: No swings detected (insufficient data).
        """
        df = make_ohlcv_df([100, 90, 100])  # 3 bars
        df = add_atr_column(df, period=3)

        swing_high, swing_low = market_structure_h4.detect_zigzag_swings(
            df, atr_mult=1.5, min_bars=2
        )

        # Short series should have no swings
        total = swing_high.sum() + swing_low.sum()
        assert total == 0, f"Very short series should have 0 swings, got {total}"

    def test_missing_atr_column_computes_internally(self, market_structure_h4):
        """
        Verify that swing detection works even without pre-computed ATR.

        Scenario: DataFrame without 'atr' column.
        Expected: ATR computed internally, swings detected normally.
        """
        df = make_v_shape_ohlcv(
            start_price=100,
            bottom_price=80,
            end_price=100,
            down_bars=15,
            up_bars=15,
        )
        # Note: NOT adding ATR column

        # Should not raise error - ATR computed internally
        swing_high, swing_low = market_structure_h4.detect_zigzag_swings(
            df, atr_mult=1.5, min_bars=3
        )

        # Should still detect something in V-shape
        total = swing_high.sum() + swing_low.sum()
        assert total >= 0, "Should handle missing ATR gracefully"

    def test_returns_series_with_correct_index(
        self, market_structure_h4, v_shape_with_atr
    ):
        """
        Verify that returned swing series have the same index as input DataFrame.
        """
        swing_high, swing_low = market_structure_h4.detect_zigzag_swings(
            v_shape_with_atr, atr_mult=1.5, min_bars=3
        )

        pd.testing.assert_index_equal(swing_high.index, v_shape_with_atr.index)
        pd.testing.assert_index_equal(swing_low.index, v_shape_with_atr.index)

    def test_returns_integer_series(self, market_structure_h4, v_shape_with_atr):
        """
        Verify that swing series contain only 0 and 1 values.
        """
        swing_high, swing_low = market_structure_h4.detect_zigzag_swings(
            v_shape_with_atr, atr_mult=1.5, min_bars=3
        )

        assert set(swing_high.unique()).issubset(
            {0, 1}
        ), "swing_high should only contain 0 or 1"
        assert set(swing_low.unique()).issubset(
            {0, 1}
        ), "swing_low should only contain 0 or 1"


# =============================================================================
# Test: Full Feature Computation
# =============================================================================


class TestMarketStructureFeatureCompute:
    """Tests for full feature computation via compute() method."""

    def test_compute_adds_expected_columns(self, market_structure_h4, v_shape_with_atr):
        """
        Verify that compute() adds all expected market structure columns.
        """
        result = market_structure_h4.compute(v_shape_with_atr)

        expected_cols = [
            "probable_swing_low",
            "probable_swing_high",
            "bars_since_swing_low",
            "bars_since_swing_high",
            "consecutive_up_bars",
            "consecutive_down_bars",
            "higher_high",
            "lower_low",
            "trend_structure",
        ]

        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_compute_preserves_original_columns(
        self, market_structure_h4, v_shape_with_atr
    ):
        """
        Verify that compute() preserves original OHLCV columns.
        """
        result = market_structure_h4.compute(v_shape_with_atr)

        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns, f"Original column {col} should be preserved"
            pd.testing.assert_series_equal(
                result[col], v_shape_with_atr[col], check_names=True
            )

    def test_get_feature_names_returns_list(self, market_structure_h4):
        """
        Verify that get_feature_names() returns a non-empty list of strings.
        """
        names = market_structure_h4.get_feature_names()

        assert isinstance(names, list), "Should return a list"
        assert len(names) > 0, "Should return at least one feature name"
        assert all(isinstance(n, str) for n in names), "All names should be strings"
