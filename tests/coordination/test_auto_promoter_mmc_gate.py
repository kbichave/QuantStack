"""Tests for MMC gate in auto-promoter (section-11)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from quantstack.coordination.auto_promoter import (
    AutoPromoter,
    PromotionCriteria,
)


def _make_promoter_with_mmc(
    mmc_correlation: float | None = None,
    mmc_scalar: float | None = None,
    icir_value: float = 0.35,
    forward_test_days: int = 30,
    trade_count: int = 20,
    win_rate: float = 0.55,
    live_count: int = 3,
    bt_sharpe: float = 1.5,
) -> tuple[AutoPromoter, str, datetime, str]:
    """Build mock-backed AutoPromoter with configurable MMC gate inputs."""
    strategy_id = "test_mmc_gate_001"
    now = datetime.now(timezone.utc)
    updated_at = now - timedelta(days=forward_test_days)

    conn = MagicMock()

    def mock_execute(query, params=None):
        q = query.strip().lower()
        result = MagicMock()

        # _get_mmc
        if "strategy_mmc" in q:
            if mmc_correlation is not None:
                result.fetchone.return_value = (mmc_correlation, mmc_scalar)
            else:
                result.fetchone.return_value = None
            return result

        # _get_grandfathered_until
        if "ic_gate_grandfathered_until" in q:
            result.fetchone.return_value = (None,)
            return result

        # _get_icir
        if "icir_21d" in q and "signal_ic" in q:
            result.fetchone.return_value = (icir_value,)
            return result

        # _get_forward_test_outcomes
        if "strategy_outcomes" in q:
            outcomes = []
            wins = int(trade_count * win_rate)
            for i in range(wins):
                outcomes.append((0.015, "win", updated_at + timedelta(days=i), updated_at + timedelta(days=i + 1)))
            for i in range(trade_count - wins):
                outcomes.append((-0.005, "loss", updated_at + timedelta(days=wins + i), updated_at + timedelta(days=wins + i + 1)))
            result.fetchall.return_value = outcomes
            return result

        # _count_live_strategies
        if "count" in q and "status = 'live'" in q:
            result.fetchone.return_value = (live_count,)
            return result

        result.fetchone.return_value = None
        result.fetchall.return_value = []
        return result

    conn.execute = mock_execute

    bt_summary = json.dumps({"sharpe_ratio": bt_sharpe})
    promoter = AutoPromoter(conn=conn, criteria=PromotionCriteria())
    return promoter, strategy_id, updated_at, bt_summary


class TestMMCGate:
    def test_high_correlation_blocks_promotion(self):
        """correlation=0.75 > 0.70 → hold, reason contains 'correlated'."""
        promoter, sid, updated_at, bt = _make_promoter_with_mmc(
            mmc_correlation=0.75, mmc_scalar=0.0
        )
        decision = promoter._evaluate_one(sid, "test_strat", bt, updated_at)
        assert decision.decision == "hold"
        assert "correlated" in decision.reason.lower()
        assert decision.criteria_results.get("mmc_gate") is False

    def test_moderate_correlation_promotes_with_penalty(self):
        """correlation=0.60 (between 0.50 and 0.70) → promote with scalar=0.5."""
        promoter, sid, updated_at, bt = _make_promoter_with_mmc(
            mmc_correlation=0.60, mmc_scalar=0.5
        )
        decision = promoter._evaluate_one(sid, "test_strat", bt, updated_at)
        assert decision.decision == "promote"
        assert decision.evidence.get("capital_weight_scalar") == 0.5
        assert decision.criteria_results.get("mmc_gate") is True

    def test_low_correlation_promotes_full_weight(self):
        """correlation=0.40 < 0.50 → promote with scalar=1.0."""
        promoter, sid, updated_at, bt = _make_promoter_with_mmc(
            mmc_correlation=0.40, mmc_scalar=1.0
        )
        decision = promoter._evaluate_one(sid, "test_strat", bt, updated_at)
        assert decision.decision == "promote"
        assert decision.evidence.get("capital_weight_scalar") == 1.0

    def test_no_mmc_data_promotes_full_weight(self):
        """No rows in strategy_mmc → promote with scalar=1.0."""
        promoter, sid, updated_at, bt = _make_promoter_with_mmc(
            mmc_correlation=None
        )
        decision = promoter._evaluate_one(sid, "test_strat", bt, updated_at)
        assert decision.decision == "promote"
        assert decision.evidence.get("capital_weight_scalar") == 1.0
        assert decision.criteria_results.get("mmc_gate") is True

    def test_ic_gate_fails_before_mmc_checked(self):
        """When IC gate fails, MMC gate is NOT checked (short-circuit)."""
        promoter, sid, updated_at, bt = _make_promoter_with_mmc(
            mmc_correlation=0.75,  # Would block if reached
            icir_value=0.10,  # IC gate fails
        )
        decision = promoter._evaluate_one(sid, "test_strat", bt, updated_at)
        assert decision.decision == "hold"
        assert "ICIR below" in decision.reason
        # MMC gate should NOT be in criteria_results (short-circuited)
        assert "mmc_gate" not in decision.criteria_results
