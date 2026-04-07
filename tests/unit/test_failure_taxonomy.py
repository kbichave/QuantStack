"""
Tests for failure mode taxonomy and classification.
"""

import pytest
from quantstack.learning.failure_taxonomy import (
    FailureMode,
    classify_failure,
    compute_research_priority,
)


class TestFailureModeEnum:
    """Test FailureMode enum structure."""

    def test_seven_members_exist(self):
        """Verify all 7 failure modes are defined."""
        expected = {
            "REGIME_MISMATCH",
            "FACTOR_CROWDING",
            "DATA_STALE",
            "TIMING_ERROR",
            "THESIS_WRONG",
            "BLACK_SWAN",
            "UNCLASSIFIED",
        }
        actual = {m.name for m in FailureMode}
        assert actual == expected

    def test_str_compatible(self):
        """Verify enum values are string-compatible."""
        assert FailureMode.REGIME_MISMATCH.value == "regime_mismatch"
        assert FailureMode.DATA_STALE.value == "data_stale"
        assert isinstance(FailureMode.BLACK_SWAN.value, str)
        # Since FailureMode inherits from (str, Enum), value can be used as string
        assert FailureMode.REGIME_MISMATCH == "regime_mismatch"


class TestRuleBasedClassifier:
    """Test classify_failure rule-based heuristics."""

    def test_regime_mismatch_detected(self):
        """Rule 1: Detects regime change during trade."""
        result = classify_failure(
            realized_pnl_pct=-5.0,
            regime_at_entry="trending_up",
            regime_at_exit="ranging",
            strategy_id="test_strat",
            symbol="AAPL",
            entry_price=150.0,
            exit_price=142.5,
        )
        assert result == FailureMode.REGIME_MISMATCH

    def test_data_stale_detected(self):
        """Rule 2: Detects stale data (>60 min)."""
        result = classify_failure(
            realized_pnl_pct=-3.0,
            regime_at_entry="trending_up",
            regime_at_exit="trending_up",  # Same regime
            strategy_id="test_strat",
            symbol="AAPL",
            entry_price=150.0,
            exit_price=145.5,
            data_freshness=75.0,  # 75 minutes stale
        )
        assert result == FailureMode.DATA_STALE

    def test_black_swan_detected(self):
        """Rule 3: Detects loss >3 std from historical distribution."""
        # Historical losses: mean = -2%, std = 1%
        # Loss at -2% - 3*1% = -5% or worse is black swan
        historical = [-1.0, -2.0, -3.0, -2.5, -1.5] * 4  # 20 samples
        result = classify_failure(
            realized_pnl_pct=-8.0,  # Way beyond -5% threshold
            regime_at_entry="trending_up",
            regime_at_exit="trending_up",
            strategy_id="test_strat",
            symbol="AAPL",
            entry_price=150.0,
            exit_price=138.0,
            historical_losses=historical,
        )
        assert result == FailureMode.BLACK_SWAN

    def test_black_swan_skipped_insufficient_history(self):
        """Rule 3: Skips black swan check with <20 samples."""
        result = classify_failure(
            realized_pnl_pct=-10.0,
            regime_at_entry="trending_up",
            regime_at_exit="trending_up",
            strategy_id="test_strat",
            symbol="AAPL",
            entry_price=150.0,
            exit_price=135.0,
            historical_losses=[-1.0, -2.0, -3.0],  # Only 3 samples
        )
        # Should fall through to UNCLASSIFIED, not BLACK_SWAN
        assert result == FailureMode.UNCLASSIFIED

    def test_timing_error_detected(self):
        """Rule 4: Detects entry within 0.5% of key level."""
        result = classify_failure(
            realized_pnl_pct=-2.0,
            regime_at_entry="trending_up",
            regime_at_exit="trending_up",
            strategy_id="test_strat",
            symbol="AAPL",
            entry_price=150.0,
            exit_price=147.0,
            key_levels=[149.5, 155.0],  # 149.5 is 0.33% away from 150.0
        )
        assert result == FailureMode.TIMING_ERROR

    def test_timing_error_not_triggered_far_from_levels(self):
        """Rule 4: Does not trigger when entry far from key levels."""
        result = classify_failure(
            realized_pnl_pct=-2.0,
            regime_at_entry="trending_up",
            regime_at_exit="trending_up",
            strategy_id="test_strat",
            symbol="AAPL",
            entry_price=150.0,
            exit_price=147.0,
            key_levels=[140.0, 160.0],  # Both >5% away
        )
        assert result == FailureMode.UNCLASSIFIED

    def test_returns_unclassified_when_no_match(self):
        """Rule 5: Returns UNCLASSIFIED when no rule matches."""
        result = classify_failure(
            realized_pnl_pct=-2.0,
            regime_at_entry="trending_up",
            regime_at_exit="trending_up",
            strategy_id="test_strat",
            symbol="AAPL",
            entry_price=150.0,
            exit_price=147.0,
        )
        assert result == FailureMode.UNCLASSIFIED

    def test_first_matching_rule_wins(self):
        """Verify priority order: regime mismatch beats data stale."""
        result = classify_failure(
            realized_pnl_pct=-3.0,
            regime_at_entry="trending_up",
            regime_at_exit="ranging",  # Regime changed
            strategy_id="test_strat",
            symbol="AAPL",
            entry_price=150.0,
            exit_price=145.5,
            data_freshness=90.0,  # Also stale
        )
        # Should return REGIME_MISMATCH, not DATA_STALE
        assert result == FailureMode.REGIME_MISMATCH


class TestResearchQueuePriority:
    """Test research priority scoring."""

    def test_recent_large_loss_high_priority(self):
        """Recent large loss gets high priority."""
        priority = compute_research_priority(
            cumulative_loss_30d=1.0,  # 100% cumulative loss
            days_since_last_loss=0,   # Today
        )
        # 1.0 * 0.95^0 * 10 = 10, capped at 9
        assert priority == 9

    def test_old_loss_low_priority(self):
        """Old loss gets downweighted by recency."""
        priority = compute_research_priority(
            cumulative_loss_30d=0.5,  # 50% cumulative loss
            days_since_last_loss=10,  # 10 days ago
        )
        # 0.5 * 0.95^10 * 10 ≈ 0.5 * 0.599 * 10 ≈ 2.995
        assert priority == 2

    def test_small_recent_loss_low_priority(self):
        """Small recent loss gets low priority."""
        priority = compute_research_priority(
            cumulative_loss_30d=0.05,  # 5% cumulative loss
            days_since_last_loss=1,    # Yesterday
        )
        # 0.05 * 0.95^1 * 10 ≈ 0.475
        assert priority == 0

    def test_priority_capped_at_nine(self):
        """Priority cannot exceed 9."""
        priority = compute_research_priority(
            cumulative_loss_30d=5.0,  # 500% cumulative loss
            days_since_last_loss=0,
        )
        assert priority == 9

    def test_zero_loss_zero_priority(self):
        """Zero loss gives zero priority."""
        priority = compute_research_priority(
            cumulative_loss_30d=0.0,
            days_since_last_loss=0,
        )
        assert priority == 0
