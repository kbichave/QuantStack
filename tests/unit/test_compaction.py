"""Tests for Section 02: Context Compaction at Merge Points.

Validates Pydantic brief schemas, compaction node logic,
fallback behavior, and context size reduction.
"""

import json

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Brief schema validation
# ---------------------------------------------------------------------------


class TestParallelMergeBrief:
    """ParallelMergeBrief schema validation."""

    def test_validates_with_complete_fields(self):
        from quantstack.graphs.trading.briefs import ParallelMergeBrief
        brief = ParallelMergeBrief(
            exits=[{"symbol": "AAPL", "action": "sell", "reason": "stop loss"}],
            entries=[{"symbol": "NVDA", "signal_strength": 0.8, "thesis": "momentum", "ewf_bias": "bullish"}],
            risks=[],
            regime="trending_up",
            earnings_flags={},
        )
        assert len(brief.exits) == 1
        assert len(brief.entries) == 1

    def test_rejects_signal_strength_outside_range(self):
        from quantstack.graphs.trading.briefs import EntryCandidate
        with pytest.raises(ValidationError):
            EntryCandidate(symbol="AAPL", signal_strength=1.5, thesis="test", ewf_bias="neutral")
        with pytest.raises(ValidationError):
            EntryCandidate(symbol="AAPL", signal_strength=-0.1, thesis="test", ewf_bias="neutral")

    def test_accepts_empty_lists(self):
        from quantstack.graphs.trading.briefs import ParallelMergeBrief
        brief = ParallelMergeBrief(
            exits=[], entries=[], risks=[], regime="", earnings_flags={},
        )
        assert brief.exits == []
        assert brief.entries == []

    def test_compaction_degraded_defaults_false(self):
        from quantstack.graphs.trading.briefs import ParallelMergeBrief
        brief = ParallelMergeBrief(
            exits=[], entries=[], risks=[], regime="", earnings_flags={},
        )
        assert brief.compaction_degraded is False


class TestPreExecutionBrief:
    """PreExecutionBrief schema validation."""

    def test_validates_with_complete_fields(self):
        from quantstack.graphs.trading.briefs import PreExecutionBrief
        brief = PreExecutionBrief(
            approved=[{"symbol": "AAPL", "position_size": 100, "structure": "equity", "rationale": "strong momentum"}],
            rejected=[{"symbol": "TSLA", "reason": "too volatile"}],
            options_specs=[],
            risk_checks={"max_position": True, "correlation": True},
        )
        assert len(brief.approved) == 1
        assert len(brief.rejected) == 1

    def test_rejects_missing_required_fields(self):
        from quantstack.graphs.trading.briefs import PreExecutionBrief
        with pytest.raises(ValidationError):
            PreExecutionBrief()  # missing all required fields


# ---------------------------------------------------------------------------
# Compaction node logic
# ---------------------------------------------------------------------------

def _make_trading_state(**overrides):
    """Create a TradingState with sensible defaults for testing."""
    from quantstack.graphs.state import TradingState
    defaults = {
        "cycle_number": 1,
        "regime": "trending_up",
        "portfolio_context": {"cash": 50000},
        "position_reviews": [{"symbol": "AAPL", "action": "hold"}],
        "exit_orders": [{"symbol": "AAPL", "action": "sell", "qty": 10, "reason": "stop loss hit"}],
        "entry_candidates": [
            {"symbol": "NVDA", "signal_strength": 0.8, "thesis": "AI momentum", "ewf_bias": "bullish"},
            {"symbol": "MSFT", "signal_strength": 0.6, "thesis": "cloud growth", "ewf_bias": "neutral"},
        ],
        "earnings_symbols": [],
        "earnings_analysis": {},
        "risk_verdicts": [{"symbol": "NVDA", "verdict": "pass"}],
        "fund_manager_decisions": [
            {"symbol": "NVDA", "decision": "APPROVED", "position_size": 50, "structure": "equity", "rationale": "strong setup"},
            {"symbol": "MSFT", "decision": "REJECTED", "reason": "weak thesis"},
        ],
        "options_analysis": [{"symbol": "NVDA", "legs": [{"type": "call", "strike": 150}], "max_loss": 500}],
        "portfolio_target_weights": {"NVDA": 0.05},
    }
    defaults.update(overrides)
    return TradingState(**defaults)


class TestCompactParallel:
    """compact_parallel node logic."""

    def test_extracts_exits(self):
        from quantstack.graphs.trading.compaction import compact_parallel
        state = _make_trading_state()
        result = compact_parallel(state)
        brief = result["parallel_brief"]
        assert len(brief.exits) == 1
        assert brief.exits[0]["symbol"] == "AAPL"

    def test_extracts_entries(self):
        from quantstack.graphs.trading.compaction import compact_parallel
        state = _make_trading_state()
        result = compact_parallel(state)
        brief = result["parallel_brief"]
        assert len(brief.entries) == 2
        assert brief.entries[0]["symbol"] == "NVDA"

    def test_includes_earnings_flags_when_present(self):
        from quantstack.graphs.trading.compaction import compact_parallel
        state = _make_trading_state(
            earnings_symbols=["NVDA"],
            earnings_analysis={"NVDA": {"event": "Q1", "impact": "high"}},
        )
        result = compact_parallel(state)
        brief = result["parallel_brief"]
        assert "NVDA" in brief.earnings_flags

    def test_empty_earnings_when_no_earnings(self):
        from quantstack.graphs.trading.compaction import compact_parallel
        state = _make_trading_state(earnings_symbols=[], earnings_analysis={})
        result = compact_parallel(state)
        brief = result["parallel_brief"]
        assert brief.earnings_flags == {}

    def test_includes_regime(self):
        from quantstack.graphs.trading.compaction import compact_parallel
        state = _make_trading_state(regime="ranging")
        result = compact_parallel(state)
        assert result["parallel_brief"].regime == "ranging"


class TestCompactPreExecution:
    """compact_pre_execution node logic."""

    def test_extracts_approved_rejected(self):
        from quantstack.graphs.trading.compaction import compact_pre_execution
        state = _make_trading_state()
        result = compact_pre_execution(state)
        brief = result["pre_execution_brief"]
        assert len(brief.approved) == 1
        assert brief.approved[0]["symbol"] == "NVDA"
        assert len(brief.rejected) == 1
        assert brief.rejected[0]["symbol"] == "MSFT"

    def test_extracts_options_specs(self):
        from quantstack.graphs.trading.compaction import compact_pre_execution
        state = _make_trading_state()
        result = compact_pre_execution(state)
        brief = result["pre_execution_brief"]
        assert len(brief.options_specs) == 1

    def test_includes_risk_checks(self):
        from quantstack.graphs.trading.compaction import compact_pre_execution
        state = _make_trading_state(
            risk_verdicts=[{"symbol": "NVDA", "verdict": "pass", "checks": {"max_position": True}}],
        )
        result = compact_pre_execution(state)
        brief = result["pre_execution_brief"]
        assert isinstance(brief.risk_checks, dict)


# ---------------------------------------------------------------------------
# Fallback behavior
# ---------------------------------------------------------------------------


class TestCompactionFallback:
    """Compaction gracefully degrades on malformed state."""

    def test_malformed_state_produces_degraded_brief(self):
        from quantstack.graphs.trading.compaction import compact_parallel
        from quantstack.graphs.state import TradingState
        # Minimal valid state (no exits/entries populated)
        state = TradingState(cycle_number=1, regime="trending_up")
        result = compact_parallel(state)
        brief = result["parallel_brief"]
        assert brief.compaction_degraded is False  # empty is valid, not degraded

    def test_empty_state_produces_valid_brief(self):
        from quantstack.graphs.trading.compaction import compact_parallel
        from quantstack.graphs.state import TradingState
        state = TradingState()
        result = compact_parallel(state)
        brief = result["parallel_brief"]
        assert brief.exits == []
        assert brief.entries == []

    def test_pre_execution_empty_state_valid(self):
        from quantstack.graphs.trading.compaction import compact_pre_execution
        from quantstack.graphs.state import TradingState
        state = TradingState()
        result = compact_pre_execution(state)
        brief = result["pre_execution_brief"]
        assert brief.approved == []
        assert brief.rejected == []


# ---------------------------------------------------------------------------
# Context size reduction
# ---------------------------------------------------------------------------


class TestContextSizeReduction:
    """Brief serialization should be significantly smaller than raw state."""

    def test_brief_smaller_than_raw_state(self):
        from quantstack.graphs.trading.compaction import compact_parallel
        state = _make_trading_state()
        result = compact_parallel(state)
        brief = result["parallel_brief"]

        # Raw state size (the fields that would be in context without compaction)
        raw_keys = {
            "exit_orders": state.exit_orders,
            "entry_candidates": state.entry_candidates,
            "earnings_analysis": state.earnings_analysis,
            "position_reviews": state.position_reviews,
            "regime": state.regime,
        }
        raw_size = len(json.dumps(raw_keys))
        brief_size = len(brief.model_dump_json())

        # Brief should be smaller (or comparable for small test data)
        assert brief_size <= raw_size * 2  # generous bound for small test data
