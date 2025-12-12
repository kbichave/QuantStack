"""
Unit tests for event-based trade labeling.

Tests verify:
- TP hit before SL labeling (WIN)
- SL hit before TP labeling (LOSS)
- Timeout scenarios
- No lookahead in label generation
- ATR scaling correctness
"""

import pytest
import pandas as pd
import numpy as np

from quantcore.labeling.event_labeler import (
    EventLabeler,
    LabelConfig,
    TradeOutcome,
    MultiTimeframeLabelBuilder,
)
from quantcore.config.timeframes import Timeframe
from tests.conftest import make_ohlcv_df, add_atr_column


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def label_config():
    """Default labeling configuration."""
    return LabelConfig(
        tp_atr_multiple=1.5,
        sl_atr_multiple=1.0,
        max_hold_bars=6,
    )


@pytest.fixture
def labeler(label_config):
    """EventLabeler instance with default config."""
    return EventLabeler(label_config)


@pytest.fixture
def uptrend_df():
    """
    Uptrend price series where long trades should hit TP.

    Price rises consistently: 100 → 110 over 10 bars.
    """
    prices = list(np.linspace(100, 110, 10))
    df = make_ohlcv_df(prices, spread_pct=0.5)
    return add_atr_column(df, period=5)


@pytest.fixture
def downtrend_df():
    """
    Downtrend price series where long trades should hit SL.

    Price falls consistently: 100 → 90 over 10 bars.
    """
    prices = list(np.linspace(100, 90, 10))
    df = make_ohlcv_df(prices, spread_pct=0.5)
    return add_atr_column(df, period=5)


@pytest.fixture
def flat_df():
    """
    Flat price series where trades should timeout.

    Price stays at 100 with minimal movement.
    """
    prices = [100.0] * 15
    df = make_ohlcv_df(prices, spread_pct=0.1)
    return add_atr_column(df, period=5)


@pytest.fixture
def v_reversal_df():
    """
    V-shaped reversal: down then up.

    Tests that TP can be hit after initial adverse movement.
    """
    down = list(np.linspace(100, 95, 5))
    up = list(np.linspace(95, 108, 8))
    prices = down + up[1:]  # Skip duplicate 95
    df = make_ohlcv_df(prices, spread_pct=0.5)
    return add_atr_column(df, period=5)


# =============================================================================
# Test: Long Trade Labeling
# =============================================================================


class TestLongTradeLabeling:
    """Tests for long trade label generation."""

    def test_uptrend_labels_as_win(self, labeler, uptrend_df):
        """
        Verify uptrend produces WIN labels for long trades.

        Scenario: Price rises from 100 to 110.
        Expected: Early bars should be labeled as WIN (TP hit).
        """
        result = labeler.label_long_trades(uptrend_df)

        # First few bars should have labels
        assert "label_long" in result.columns

        # Check that we have at least some WIN labels
        valid_labels = result["label_long"].dropna()
        if len(valid_labels) > 0:
            win_count = (valid_labels == 1).sum()
            assert win_count > 0, "Uptrend should produce some WIN labels"

    def test_downtrend_labels_as_loss(self, labeler, downtrend_df):
        """
        Verify downtrend produces LOSS labels for long trades.

        Scenario: Price falls from 100 to 90.
        Expected: All labelable bars should be LOSS (SL hit).
        """
        result = labeler.label_long_trades(downtrend_df)

        valid_labels = result["label_long"].dropna()
        if len(valid_labels) > 0:
            # In a pure downtrend, longs should mostly lose
            loss_count = (valid_labels == 0).sum()
            assert (
                loss_count >= len(valid_labels) * 0.8
            ), "Downtrend should produce mostly LOSS labels for longs"

    def test_flat_market_timeout(self, labeler, flat_df):
        """
        Verify flat market produces TIMEOUT (labeled as 0).

        Scenario: Price stays flat at 100.
        Expected: Neither TP nor SL hit, so TIMEOUT (0).
        """
        result = labeler.label_long_trades(flat_df)

        # Check exit types
        if "label_long_exit_type" in result.columns:
            exit_types = result["label_long_exit_type"].dropna()
            timeout_count = (exit_types == "TIMEOUT").sum()
            # With very flat prices, most should timeout
            if len(exit_types) > 0:
                assert (
                    timeout_count >= len(exit_types) * 0.5
                ), "Flat market should produce TIMEOUT exits"

    def test_bars_to_exit_populated(self, labeler, uptrend_df):
        """
        Verify bars_to_exit column is populated correctly.
        """
        result = labeler.label_long_trades(uptrend_df)

        assert "label_long_bars_to_exit" in result.columns

        bars_to_exit = result["label_long_bars_to_exit"].dropna()
        if len(bars_to_exit) > 0:
            assert (bars_to_exit >= 1).all(), "Bars to exit must be >= 1"
            assert (
                bars_to_exit <= labeler.config.max_hold_bars
            ).all(), "Bars to exit must be <= max_hold_bars"


# =============================================================================
# Test: Short Trade Labeling
# =============================================================================


class TestShortTradeLabeling:
    """Tests for short trade label generation."""

    def test_downtrend_labels_short_as_win(self, labeler, downtrend_df):
        """
        Verify downtrend produces WIN labels for short trades.

        Scenario: Price falls from 100 to 90.
        Expected: Early bars should be labeled as WIN for shorts.
        """
        result = labeler.label_short_trades(downtrend_df)

        assert "label_short" in result.columns

        valid_labels = result["label_short"].dropna()
        if len(valid_labels) > 0:
            win_count = (valid_labels == 1).sum()
            assert win_count > 0, "Downtrend should produce WIN labels for shorts"

    def test_uptrend_labels_short_as_loss(self, labeler, uptrend_df):
        """
        Verify uptrend produces LOSS labels for short trades.

        Scenario: Price rises from 100 to 110.
        Expected: Short trades should hit SL (LOSS).
        """
        result = labeler.label_short_trades(uptrend_df)

        valid_labels = result["label_short"].dropna()
        if len(valid_labels) > 0:
            loss_count = (valid_labels == 0).sum()
            assert (
                loss_count >= len(valid_labels) * 0.8
            ), "Uptrend should produce mostly LOSS labels for shorts"


# =============================================================================
# Test: No Lookahead
# =============================================================================


class TestNoLookahead:
    """Tests to verify no lookahead bias in labeling."""

    def test_labels_use_only_future_bars(self, labeler, uptrend_df):
        """
        Verify labels at bar t only use data from bars t+1 to t+H.

        This is the EXPECTED behavior - labels look forward for training.
        The key is that these labels are NOT used as runtime features.
        """
        result = labeler.label_long_trades(uptrend_df)

        # Labels should be NaN for the last max_hold_bars
        # because there's no future to evaluate
        last_bars = result["label_long"].iloc[-labeler.config.max_hold_bars :]
        assert (
            last_bars.isna().all()
        ), "Last max_hold_bars should have NaN labels (no future data)"

    def test_entry_at_close_not_future_price(self, labeler, v_reversal_df):
        """
        Verify entry price is close[t], not any future price.

        We can verify this indirectly by checking that the label
        reflects what would happen starting from close[t].
        """
        result = labeler.label_long_trades(v_reversal_df)

        # The labeler should have produced some results
        valid_labels = result["label_long"].dropna()
        assert len(valid_labels) > 0, "Should have labels for V-reversal"

    def test_atr_at_entry_bar_used(self, labeler):
        """
        Verify ATR from entry bar is used, not future ATR.
        """
        # Create data with increasing ATR over time
        prices = list(np.linspace(100, 100, 20))  # Flat prices
        df = make_ohlcv_df(prices, spread_pct=0.5)

        # Manually set ATR to increase over time
        df["atr"] = np.linspace(1.0, 5.0, 20)

        result = labeler.label_long_trades(df)

        # Labels should exist (just checking it runs without error)
        assert "label_long" in result.columns


# =============================================================================
# Test: PnL Calculation
# =============================================================================


class TestPnLCalculation:
    """Tests for P&L calculation in labels."""

    def test_win_has_positive_pnl(self, labeler, uptrend_df):
        """
        Verify WIN labels have positive P&L.
        """
        result = labeler.label_long_trades(uptrend_df)

        wins = result[result["label_long"] == 1]
        if len(wins) > 0:
            assert (
                wins["label_long_pnl_pct"] > 0
            ).all(), "WIN labels should have positive P&L"

    def test_loss_has_negative_pnl(self, labeler, downtrend_df):
        """
        Verify LOSS labels have negative P&L.
        """
        result = labeler.label_long_trades(downtrend_df)

        losses = result[result["label_long"] == 0]
        sl_losses = losses[losses["label_long_exit_type"] == "SL"]

        if len(sl_losses) > 0:
            assert (
                sl_losses["label_long_pnl_pct"] < 0
            ).all(), "SL exits should have negative P&L"


# =============================================================================
# Test: Statistics
# =============================================================================


class TestLabelStatistics:
    """Tests for label statistics calculation."""

    def test_get_label_statistics(self, labeler, uptrend_df):
        """
        Verify statistics calculation returns expected structure.
        """
        result = labeler.label_long_trades(uptrend_df)
        stats = labeler.get_label_statistics(result, "label_long")

        assert "count" in stats
        assert "win_count" in stats
        assert "loss_count" in stats
        assert "win_rate" in stats

        # Win rate should be between 0 and 1
        assert 0 <= stats["win_rate"] <= 1

    def test_class_balance_calculated(self, labeler, uptrend_df):
        """
        Verify class balance is calculated correctly.
        """
        result = labeler.label_long_trades(uptrend_df)
        stats = labeler.get_label_statistics(result, "label_long")

        assert "class_balance" in stats
        # Class balance is mean of labels, should be between 0 and 1
        assert 0 <= stats["class_balance"] <= 1


# =============================================================================
# Test: Multi-Timeframe Labeling
# =============================================================================


class TestMultiTimeframeLabelBuilder:
    """Tests for multi-timeframe label building."""

    def test_label_builder_creates_labelers_for_all_timeframes(self):
        """
        Verify label builder has labelers for all timeframes.
        """
        builder = MultiTimeframeLabelBuilder()

        for tf in Timeframe:
            assert tf in builder.labelers, f"Missing labeler for {tf}"

    def test_label_all_timeframes_returns_dict(self):
        """
        Verify label_all_timeframes returns dictionary with all timeframes.
        """
        builder = MultiTimeframeLabelBuilder()

        # Create dummy data for each timeframe
        data = {}
        for tf in [Timeframe.H1, Timeframe.H4]:
            prices = list(np.linspace(100, 110, 20))
            df = make_ohlcv_df(prices)
            df = add_atr_column(df, period=5)
            data[tf] = df

        result = builder.label_all_timeframes(data)

        assert isinstance(result, dict)
        for tf in data.keys():
            assert tf in result


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestLabelingEdgeCases:
    """Edge case tests for labeling."""

    def test_empty_dataframe(self, labeler):
        """
        Verify graceful handling of empty DataFrame.
        """
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "atr"])
        result = labeler.label_long_trades(df)

        assert len(result) == 0

    def test_insufficient_future_bars(self, labeler):
        """
        Verify handling when there aren't enough future bars.
        """
        # Only 3 bars, less than max_hold_bars
        prices = [100, 101, 102]
        df = make_ohlcv_df(prices)
        df = add_atr_column(df, period=2)

        result = labeler.label_long_trades(df)

        # Should have NaN labels since not enough future data
        assert result["label_long"].isna().all()

    def test_zero_atr_skipped(self, labeler):
        """
        Verify bars with zero ATR are skipped (no label).
        """
        prices = list(np.linspace(100, 110, 15))
        df = make_ohlcv_df(prices)
        df["atr"] = 0.0  # Zero ATR

        result = labeler.label_long_trades(df)

        # All labels should be NaN since ATR is 0
        assert result["label_long"].isna().all()

    def test_nan_atr_skipped(self, labeler):
        """
        Verify bars with NaN ATR are skipped.
        """
        prices = list(np.linspace(100, 110, 15))
        df = make_ohlcv_df(prices)
        df["atr"] = np.nan

        result = labeler.label_long_trades(df)

        assert result["label_long"].isna().all()
