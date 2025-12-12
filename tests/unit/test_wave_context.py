"""
Unit tests for WaveContextAnalyzer and WaveContextSummary.

Tests verify:
- Wave context summarization from DataFrame with wave columns
- Boolean helper properties (is_corrective_down, is_late_impulse, etc.)
- Probability-based flag triggering
- for_long_mr() and for_short_mr() decision helpers
- Edge cases (empty data, missing columns)
"""

import pytest
import pandas as pd
import numpy as np

from quantcore.hierarchy.wave_context import (
    WaveContextAnalyzer,
    WaveContextSummary,
    MultiTimeframeWaveContext,
)
from quantcore.features.waves import WaveRole
from quantcore.config.timeframes import Timeframe

# Import helpers from conftest
from tests.conftest import (
    make_ohlcv_df,
    add_atr_column,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def wave_analyzer_h4():
    """WaveContextAnalyzer for 4H timeframe."""
    return WaveContextAnalyzer(Timeframe.H4)


@pytest.fixture
def wave_analyzer_d1():
    """WaveContextAnalyzer for Daily timeframe."""
    return WaveContextAnalyzer(Timeframe.D1)


def make_df_with_wave_columns(
    n_bars: int = 20,
    wave_role: str = "none",
    wave_stage: int = -1,
    wave_conf: float = 0.0,
    prob_impulse_up: float = 0.0,
    prob_impulse_down: float = 0.0,
    prob_corr_down: float = 0.0,
    prob_corr_up: float = 0.0,
) -> pd.DataFrame:
    """
    Create a DataFrame with pre-populated wave feature columns.

    This allows testing WaveContextAnalyzer without needing
    actual wave detection to run.
    """
    prices = [100 + i * 0.5 for i in range(n_bars)]
    df = make_ohlcv_df(prices)
    df = add_atr_column(df, period=14)

    # Add wave columns
    df["wave_role"] = wave_role
    df["wave_stage"] = wave_stage
    df["wave_conf"] = wave_conf
    df["prob_impulse_up"] = prob_impulse_up
    df["prob_impulse_down"] = prob_impulse_down
    df["prob_corr_down"] = prob_corr_down
    df["prob_corr_up"] = prob_corr_up

    return df


# =============================================================================
# Test: is_corrective_down Flag
# =============================================================================


class TestIsCorrectiveDown:
    """Tests for is_corrective_down boolean property."""

    def test_is_corrective_down_from_wave_role(self, wave_analyzer_h4):
        """
        Verify that wave_role="corr_down" sets is_corrective_down=True.

        Scenario: DataFrame with wave_role="corr_down", high confidence.
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.CORR_DOWN.value,
            wave_conf=0.8,
        )

        summary = wave_analyzer_h4.analyze(df)

        assert (
            summary.is_corrective_down is True
        ), "wave_role=corr_down should set is_corrective_down=True"

    def test_is_corrective_down_from_probability(self, wave_analyzer_h4):
        """
        Verify that prob_corr_down > 0.6 triggers is_corrective_down=True.

        Scenario: wave_role="none" but prob_corr_down=0.7.
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role="none",
            wave_conf=0.0,
            prob_corr_down=0.7,  # Above 0.6 threshold
        )

        summary = wave_analyzer_h4.analyze(df)

        assert (
            summary.is_corrective_down is True
        ), "prob_corr_down > 0.6 should trigger is_corrective_down=True"

    def test_is_corrective_down_false_when_neither_condition(self, wave_analyzer_h4):
        """
        Verify that is_corrective_down is False when neither condition met.

        Scenario: wave_role="none" and prob_corr_down=0.3 (below threshold).
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role="none",
            wave_conf=0.0,
            prob_corr_down=0.3,  # Below 0.6 threshold
        )

        summary = wave_analyzer_h4.analyze(df)

        assert summary.is_corrective_down is False


# =============================================================================
# Test: is_corrective_up Flag
# =============================================================================


class TestIsCorrectiveUp:
    """Tests for is_corrective_up boolean property."""

    def test_is_corrective_up_from_wave_role(self, wave_analyzer_h4):
        """
        Verify that wave_role="corr_up" sets is_corrective_up=True.
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.CORR_UP.value,
            wave_conf=0.8,
        )

        summary = wave_analyzer_h4.analyze(df)

        assert summary.is_corrective_up is True

    def test_is_corrective_up_from_probability(self, wave_analyzer_h4):
        """
        Verify that prob_corr_up > 0.6 triggers is_corrective_up=True.
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role="none",
            prob_corr_up=0.65,
        )

        summary = wave_analyzer_h4.analyze(df)

        assert summary.is_corrective_up is True


# =============================================================================
# Test: is_late_impulse Flag
# =============================================================================


class TestIsLateImpulse:
    """Tests for is_late_impulse boolean property."""

    def test_is_late_impulse_terminal_up(self, wave_analyzer_h4):
        """
        Verify that wave_role="impulse_up_terminal" sets is_late_impulse=True.

        Scenario: We're in wave 5 of an impulse up.
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.IMPULSE_UP_TERMINAL.value,
            wave_stage=5,
            wave_conf=0.7,
        )

        summary = wave_analyzer_h4.analyze(df)

        assert (
            summary.is_late_impulse is True
        ), "impulse_up_terminal should set is_late_impulse=True"

    def test_is_late_impulse_terminal_down(self, wave_analyzer_h4):
        """
        Verify that wave_role="impulse_down_terminal" sets is_late_impulse=True.
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.IMPULSE_DOWN_TERMINAL.value,
            wave_stage=5,
            wave_conf=0.7,
        )

        summary = wave_analyzer_h4.analyze(df)

        assert summary.is_late_impulse is True

    def test_is_late_impulse_from_stage_4(self, wave_analyzer_h4):
        """
        Verify that wave_stage=4 triggers is_late_impulse=True.

        Scenario: In wave 4 (corrective within late impulse).
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.CORR_DOWN.value,  # Wave 4 is corrective
            wave_stage=4,
            wave_conf=0.6,
        )

        summary = wave_analyzer_h4.analyze(df)

        assert (
            summary.is_late_impulse is True
        ), "wave_stage=4 should trigger is_late_impulse=True"

    def test_is_late_impulse_from_stage_5(self, wave_analyzer_h4):
        """
        Verify that wave_stage=5 triggers is_late_impulse=True.
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.IMPULSE_UP.value,
            wave_stage=5,
            wave_conf=0.6,
        )

        summary = wave_analyzer_h4.analyze(df)

        assert summary.is_late_impulse is True

    def test_is_late_impulse_false_early_waves(self, wave_analyzer_h4):
        """
        Verify that early wave stages (1, 2, 3) do not trigger is_late_impulse.
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.IMPULSE_UP.value,
            wave_stage=3,  # Wave 3 - not late
            wave_conf=0.7,
        )

        summary = wave_analyzer_h4.analyze(df)

        assert summary.is_late_impulse is False


# =============================================================================
# Test: is_impulse_active Flag
# =============================================================================


class TestIsImpulseActive:
    """Tests for is_impulse_active boolean property."""

    def test_is_impulse_active_impulse_up(self, wave_analyzer_h4):
        """
        Verify that wave_role="impulse_up" sets is_impulse_active=True.
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.IMPULSE_UP.value,
            wave_stage=1,
            wave_conf=0.6,
        )

        summary = wave_analyzer_h4.analyze(df)

        assert summary.is_impulse_active is True

    def test_is_impulse_active_impulse_down(self, wave_analyzer_h4):
        """
        Verify that wave_role="impulse_down" sets is_impulse_active=True.
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.IMPULSE_DOWN.value,
            wave_stage=3,
            wave_conf=0.6,
        )

        summary = wave_analyzer_h4.analyze(df)

        assert summary.is_impulse_active is True

    def test_is_impulse_active_false_for_corrective(self, wave_analyzer_h4):
        """
        Verify that corrective roles do not set is_impulse_active.
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.CORR_DOWN.value,
            wave_stage=2,
            wave_conf=0.6,
        )

        summary = wave_analyzer_h4.analyze(df)

        assert summary.is_impulse_active is False


# =============================================================================
# Test: for_long_mr() Method
# =============================================================================


class TestForLongMR:
    """Tests for for_long_mr() decision helper."""

    def test_for_long_mr_corrective_down(self, wave_analyzer_h4):
        """
        Verify that corrective down context returns for_long_mr()=True.

        Scenario: is_corrective_down=True (ideal for long mean reversion).
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.CORR_DOWN.value,
            wave_conf=0.7,
        )

        summary = wave_analyzer_h4.analyze(df)

        assert summary.for_long_mr() is True, "Corrective down should favor long MR"

    def test_for_long_mr_high_prob_corr_down(self, wave_analyzer_h4):
        """
        Verify that prob_corr_down > 0.6 triggers for_long_mr()=True.
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role="none",
            prob_corr_down=0.65,
        )

        summary = wave_analyzer_h4.analyze(df)

        assert summary.for_long_mr() is True

    def test_for_long_mr_wave_2_in_impulse_up(self, wave_analyzer_h4):
        """
        Verify that wave 2 in impulse up favors long MR.

        Scenario: prob_impulse_up > 0.5 and wave_stage=2.
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.CORR_DOWN.value,
            wave_stage=2,
            wave_conf=0.6,
            prob_impulse_up=0.6,
        )

        summary = wave_analyzer_h4.analyze(df)

        assert summary.for_long_mr() is True

    def test_for_long_mr_false_impulse_down(self, wave_analyzer_h4):
        """
        Verify that impulse down context does not favor long MR.
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.IMPULSE_DOWN.value,
            wave_stage=3,
            wave_conf=0.7,
            prob_impulse_down=0.8,
        )

        summary = wave_analyzer_h4.analyze(df)

        assert summary.for_long_mr() is False


# =============================================================================
# Test: for_short_mr() Method
# =============================================================================


class TestForShortMR:
    """Tests for for_short_mr() decision helper."""

    def test_for_short_mr_corrective_up(self, wave_analyzer_h4):
        """
        Verify that corrective up context returns for_short_mr()=True.
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.CORR_UP.value,
            wave_conf=0.7,
        )

        summary = wave_analyzer_h4.analyze(df)

        assert summary.for_short_mr() is True

    def test_for_short_mr_high_prob_corr_up(self, wave_analyzer_h4):
        """
        Verify that prob_corr_up > 0.6 triggers for_short_mr()=True.
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role="none",
            prob_corr_up=0.65,
        )

        summary = wave_analyzer_h4.analyze(df)

        assert summary.for_short_mr() is True


# =============================================================================
# Test: Caution Helpers
# =============================================================================


class TestCautionHelpers:
    """Tests for caution_for_long() and caution_for_short()."""

    def test_caution_for_long_late_impulse_up(self, wave_analyzer_h4):
        """
        Verify that late impulse up triggers caution for longs.

        Scenario: is_late_impulse=True and prob_impulse_up > 0.5.
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.IMPULSE_UP_TERMINAL.value,
            wave_stage=5,
            wave_conf=0.7,
            prob_impulse_up=0.6,
        )

        summary = wave_analyzer_h4.analyze(df)

        assert summary.caution_for_long() is True

    def test_caution_for_long_strong_impulse_down(self, wave_analyzer_h4):
        """
        Verify that strong impulse down triggers caution for longs.

        Scenario: prob_impulse_down > 0.7.
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.IMPULSE_DOWN.value,
            wave_stage=3,
            prob_impulse_down=0.75,
        )

        summary = wave_analyzer_h4.analyze(df)

        assert summary.caution_for_long() is True

    def test_caution_for_short_late_impulse_down(self, wave_analyzer_h4):
        """
        Verify that late impulse down triggers caution for shorts.
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.IMPULSE_DOWN_TERMINAL.value,
            wave_stage=5,
            wave_conf=0.7,
            prob_impulse_down=0.6,
        )

        summary = wave_analyzer_h4.analyze(df)

        assert summary.caution_for_short() is True

    def test_caution_for_short_strong_impulse_up(self, wave_analyzer_h4):
        """
        Verify that strong impulse up triggers caution for shorts.
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.IMPULSE_UP.value,
            wave_stage=3,
            prob_impulse_up=0.75,
        )

        summary = wave_analyzer_h4.analyze(df)

        assert summary.caution_for_short() is True


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestWaveContextEdgeCases:
    """Edge case tests for WaveContextAnalyzer."""

    def test_empty_data_returns_empty_context(self, wave_analyzer_h4):
        """
        Verify that empty DataFrame returns neutral/empty context.
        """
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        summary = wave_analyzer_h4.analyze(df)

        assert summary.wave_role == "none"
        assert summary.wave_stage == -1
        assert summary.wave_conf == 0.0
        assert summary.is_corrective_down is False
        assert summary.is_late_impulse is False

    def test_short_data_returns_empty_context(self, wave_analyzer_h4):
        """
        Verify that DataFrame with < 10 rows returns empty context.
        """
        df = make_df_with_wave_columns(n_bars=5)  # Less than 10

        summary = wave_analyzer_h4.analyze(df)

        assert summary.wave_role == "none"
        assert summary.wave_stage == -1

    def test_analyze_preserves_existing_wave_columns(self, wave_analyzer_h4):
        """
        Verify that if wave columns exist, they're used directly.

        Scenario: DataFrame already has wave_role/stage/conf columns.
        Expected: Those values used (no recomputation).
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role="custom_role",  # Non-standard role for testing
            wave_stage=99,
            wave_conf=0.99,
        )

        summary = wave_analyzer_h4.analyze(df)

        # Should use the pre-set values
        assert summary.wave_role == "custom_role"
        assert summary.wave_stage == 99
        assert summary.wave_conf == 0.99

    def test_missing_wave_columns_triggers_compute(self, wave_analyzer_h4):
        """
        Verify that missing wave columns trigger wave feature computation.
        """
        # Create DataFrame WITHOUT wave columns
        prices = [100 + i * 0.5 for i in range(20)]
        df = make_ohlcv_df(prices)
        df = add_atr_column(df, period=14)
        # Note: NOT adding wave columns

        # Should not raise error - will compute wave features
        summary = wave_analyzer_h4.analyze(df)

        # Should have valid summary
        assert isinstance(summary, WaveContextSummary)
        assert hasattr(summary, "wave_role")


# =============================================================================
# Test: WaveContextSummary Serialization
# =============================================================================


class TestWaveContextSummarySerialization:
    """Tests for WaveContextSummary.to_dict() method."""

    def test_to_dict_contains_all_fields(self, wave_analyzer_h4):
        """
        Verify that to_dict() includes all expected fields.
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.CORR_DOWN.value,
            wave_stage=2,
            wave_conf=0.7,
            prob_corr_down=0.65,
        )

        summary = wave_analyzer_h4.analyze(df)
        d = summary.to_dict()

        expected_keys = [
            "wave_role",
            "wave_stage",
            "wave_conf",
            "prob_impulse_up",
            "prob_impulse_down",
            "prob_corr_down",
            "prob_corr_up",
            "is_corrective_down",
            "is_corrective_up",
            "is_impulse_active",
            "is_late_impulse",
            "for_long_mr",
            "for_short_mr",
            "caution_long",
            "caution_short",
        ]

        for key in expected_keys:
            assert key in d, f"Missing key in to_dict(): {key}"

    def test_to_dict_values_match_properties(self, wave_analyzer_h4):
        """
        Verify that to_dict() values match property values.
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.CORR_DOWN.value,
            wave_stage=2,
            wave_conf=0.7,
        )

        summary = wave_analyzer_h4.analyze(df)
        d = summary.to_dict()

        assert d["wave_role"] == summary.wave_role
        assert d["wave_stage"] == summary.wave_stage
        assert d["wave_conf"] == summary.wave_conf
        assert d["is_corrective_down"] == summary.is_corrective_down
        assert d["for_long_mr"] == summary.for_long_mr()


# =============================================================================
# Test: MultiTimeframeWaveContext
# =============================================================================


class TestMultiTimeframeWaveContext:
    """Tests for MultiTimeframeWaveContext."""

    def test_analyze_returns_both_timeframes(self):
        """
        Verify that analyze() returns context for both H4 and D1.
        """
        mtf = MultiTimeframeWaveContext()

        df_h4 = make_df_with_wave_columns(n_bars=20, wave_role=WaveRole.CORR_DOWN.value)
        df_d1 = make_df_with_wave_columns(
            n_bars=20, wave_role=WaveRole.IMPULSE_UP.value
        )

        result = mtf.analyze(df_h4, df_d1)

        assert "H4" in result
        assert "D1" in result
        assert isinstance(result["H4"], WaveContextSummary)
        assert isinstance(result["D1"], WaveContextSummary)

    def test_analyze_without_daily_data(self):
        """
        Verify that analyze() works with only H4 data.
        """
        mtf = MultiTimeframeWaveContext()

        df_h4 = make_df_with_wave_columns(n_bars=20)

        result = mtf.analyze(df_h4, df_d1=None)

        assert "H4" in result
        assert "D1" in result
        # D1 should be empty context
        assert result["D1"].wave_role == "none"

    def test_get_combined_signal_quality_long(self):
        """
        Verify get_combined_signal_quality() for LONG direction.
        """
        mtf = MultiTimeframeWaveContext()

        df_h4 = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.CORR_DOWN.value,
            wave_conf=0.7,
        )
        df_d1 = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.IMPULSE_UP.value,
            prob_impulse_up=0.6,
        )

        contexts = mtf.analyze(df_h4, df_d1)
        quality = mtf.get_combined_signal_quality(contexts, "LONG")

        # Should be > 0.5 (favorable for long)
        assert 0 <= quality <= 1
        assert (
            quality > 0.5
        ), "Corrective down on H4 + impulse up on D1 should favor long"

    def test_get_combined_signal_quality_short(self):
        """
        Verify get_combined_signal_quality() for SHORT direction.
        """
        mtf = MultiTimeframeWaveContext()

        df_h4 = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.CORR_UP.value,
            wave_conf=0.7,
        )
        df_d1 = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.IMPULSE_DOWN.value,
            prob_impulse_down=0.6,
        )

        contexts = mtf.analyze(df_h4, df_d1)
        quality = mtf.get_combined_signal_quality(contexts, "SHORT")

        assert 0 <= quality <= 1
        assert (
            quality > 0.5
        ), "Corrective up on H4 + impulse down on D1 should favor short"


# =============================================================================
# Test: Wave Context Series (for Backtesting)
# =============================================================================


class TestWaveContextSeries:
    """Tests for get_wave_context_series() method."""

    def test_get_wave_context_series_adds_columns(self, wave_analyzer_h4):
        """
        Verify that get_wave_context_series() adds derived columns.
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.CORR_DOWN.value,
            wave_stage=2,
            wave_conf=0.7,
        )

        result = wave_analyzer_h4.get_wave_context_series(df)

        expected_cols = [
            "wave_is_corr_down",
            "wave_is_corr_up",
            "wave_is_late_impulse",
            "wave_favor_long_mr",
            "wave_favor_short_mr",
        ]

        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_get_wave_context_series_preserves_original(self, wave_analyzer_h4):
        """
        Verify that original columns are preserved.
        """
        df = make_df_with_wave_columns(n_bars=20)

        result = wave_analyzer_h4.get_wave_context_series(df)

        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_wave_favor_long_mr_column(self, wave_analyzer_h4):
        """
        Verify wave_favor_long_mr column logic.
        """
        df = make_df_with_wave_columns(
            n_bars=20,
            wave_role=WaveRole.CORR_DOWN.value,
            wave_stage=2,  # Not late impulse
            wave_conf=0.7,
        )

        result = wave_analyzer_h4.get_wave_context_series(df)

        # Corrective down + not late impulse should favor long
        assert result["wave_favor_long_mr"].iloc[-1] == 1
