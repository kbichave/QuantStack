# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for quant_pod.guardrails.agent_hardening — Sprint 1.

Tests validate_agent_output() and reconcile_blackboard_with_portfolio().
No external I/O — pure in-memory logic.
"""

from __future__ import annotations

import pytest
from quantstack.guardrails.agent_hardening import AgentHardener


@pytest.fixture
def hardener() -> AgentHardener:
    return AgentHardener()


# ---------------------------------------------------------------------------
# validate_agent_output — action validation
# ---------------------------------------------------------------------------


class TestValidateAgentOutput:
    def test_clean_output_is_valid(self, hardener):
        result = hardener.validate_agent_output(
            {
                "action": "BUY",
                "confidence": 0.75,
                "position_size_pct": 0.10,
                "symbol": "SPY",
                "entry_price": 450.0,
            }
        )
        assert result["is_valid"]
        assert result["violations"] == []

    def test_hold_action_is_valid(self, hardener):
        result = hardener.validate_agent_output({"action": "HOLD"})
        assert result["is_valid"]

    def test_unknown_action_rejected(self, hardener):
        result = hardener.validate_agent_output({"action": "DOUBLE_DOWN"})
        assert not result["is_valid"]
        assert any(
            "action" in v.lower() or "DOUBLE_DOWN" in v for v in result["violations"]
        )
        # sanitized to HOLD
        assert result["sanitized_output"]["action"] == "HOLD"

    def test_confidence_above_one_flagged(self, hardener):
        result = hardener.validate_agent_output({"confidence": 1.5})
        assert not result["is_valid"]
        assert result["sanitized_output"]["confidence"] == 1.0

    def test_confidence_negative_flagged(self, hardener):
        result = hardener.validate_agent_output({"confidence": -0.1})
        assert not result["is_valid"]
        assert result["sanitized_output"]["confidence"] == 0.0

    def test_position_size_above_max_clamped(self, hardener):
        # >20% → flagged and clamped
        result = hardener.validate_agent_output({"position_size_pct": 0.50})
        assert not result["is_valid"]
        assert (
            result["sanitized_output"]["position_size_pct"]
            == hardener.MAX_RECOMMENDED_POSITION_PCT
        )

    def test_negative_position_size_flagged(self, hardener):
        result = hardener.validate_agent_output({"position_size_pct": -0.05})
        assert not result["is_valid"]
        assert result["sanitized_output"]["position_size_pct"] == 0.0

    def test_symbol_too_long_flagged(self, hardener):
        result = hardener.validate_agent_output({"symbol": "VERYLONGSYMBOL123"})
        assert not result["is_valid"]

    def test_price_target_negative_flagged(self, hardener):
        result = hardener.validate_agent_output({"entry_price": -10.0})
        assert not result["is_valid"]
        assert result["sanitized_output"]["entry_price"] is None

    def test_price_target_above_million_flagged(self, hardener):
        result = hardener.validate_agent_output({"take_profit": 2_000_000.0})
        assert not result["is_valid"]

    def test_injection_in_action_field_sanitized(self, hardener):
        # action field: "ignore previous instructions" is unknown → HOLD
        result = hardener.validate_agent_output({"action": "ignore previous"})
        assert not result["is_valid"]

    def test_sanitized_output_has_all_original_keys(self, hardener):
        original = {"action": "BUY", "confidence": 0.8, "symbol": "SPY"}
        result = hardener.validate_agent_output(original)
        for k in original:
            assert k in result["sanitized_output"]


# ---------------------------------------------------------------------------
# reconcile_blackboard_with_portfolio
# ---------------------------------------------------------------------------


class TestReconcileBlackboard:
    def _pos(self, symbol: str, qty: int) -> dict:
        return {"symbol": symbol, "quantity": qty}

    def test_identical_positions_reconcile(self, hardener):
        bb = [self._pos("SPY", 100), self._pos("QQQ", 50)]
        port = [self._pos("SPY", 100), self._pos("QQQ", 50)]
        result = hardener.reconcile_blackboard_with_portfolio(bb, port)
        assert result["is_reconciled"]
        assert not result["should_halt_execution"]
        assert result["phantom_positions"] == []
        assert result["unknown_positions"] == []
        assert result["quantity_mismatches"] == []

    def test_phantom_position_detected(self, hardener):
        """BB believes it has AAPL — portfolio doesn't have it."""
        bb = [self._pos("SPY", 100), self._pos("AAPL", 50)]
        port = [self._pos("SPY", 100)]
        result = hardener.reconcile_blackboard_with_portfolio(bb, port)
        assert not result["is_reconciled"]
        assert result["should_halt_execution"]
        assert any(p["symbol"] == "AAPL" for p in result["phantom_positions"])

    def test_unknown_position_detected(self, hardener):
        """Portfolio has MSFT that BB doesn't know about."""
        bb = [self._pos("SPY", 100)]
        port = [self._pos("SPY", 100), self._pos("MSFT", 30)]
        result = hardener.reconcile_blackboard_with_portfolio(bb, port)
        assert not result["is_reconciled"]
        # Unknown positions don't halt — only phantom does
        assert not result["should_halt_execution"]
        assert any(u["symbol"] == "MSFT" for u in result["unknown_positions"])

    def test_quantity_mismatch_detected(self, hardener):
        """BB thinks 100 SPY, portfolio has 50 SPY — >5% divergence."""
        bb = [self._pos("SPY", 100)]
        port = [self._pos("SPY", 50)]
        result = hardener.reconcile_blackboard_with_portfolio(bb, port)
        assert not result["is_reconciled"]
        assert len(result["quantity_mismatches"]) == 1
        assert result["quantity_mismatches"][0]["symbol"] == "SPY"

    def test_small_quantity_difference_ignored(self, hardener):
        """< 5% difference is within tolerance — should reconcile."""
        bb = [self._pos("SPY", 100)]
        port = [self._pos("SPY", 102)]  # 2% diff — within 5% tolerance
        result = hardener.reconcile_blackboard_with_portfolio(bb, port)
        assert result["is_reconciled"]
        assert result["quantity_mismatches"] == []

    def test_empty_both_sides_reconcile(self, hardener):
        result = hardener.reconcile_blackboard_with_portfolio([], [])
        assert result["is_reconciled"]

    def test_phantom_triggers_halt(self, hardener):
        """Phantom positions should halt execution (TradeTrap: memory poisoning)."""
        bb = [self._pos("FAKE", 9999)]
        port = []
        result = hardener.reconcile_blackboard_with_portfolio(bb, port)
        assert result["should_halt_execution"]
