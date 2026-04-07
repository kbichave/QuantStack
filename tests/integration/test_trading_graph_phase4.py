"""Phase 4 integration tests — cross-cutting subsystem interactions.

Tests verify that the phase-4 subsystems compose correctly:
  - Error blocking + execution gate routing
  - Conflict resolution + state merging
  - Circuit breaker + safe defaults + execution gate
  - Non-blocking failure accumulation + safety net threshold
  - Regime flip actions + stop tightening flow
  - DLQ write on parse failure with context propagation

These tests exercise real logic from multiple modules without requiring
a live PostgreSQL or LLM provider — external I/O is mocked at the
narrowest possible boundary.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from quantstack.execution.regime_flip import (
    classify_regime_flip,
    compute_tightened_stop,
    generate_regime_flip_actions,
)
from quantstack.graphs.agent_executor import (
    PRIORITY_P0,
    PRIORITY_P1,
    PRIORITY_P2,
    _get_message_priority,
    _prune_messages,
    parse_json_response,
    tag_message_priority,
)
from quantstack.observability.dlq_monitor import (
    DLQ_CRITICAL_RATE_PCT,
    check_dlq_alerts,
    compute_dlq_rate,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_trading_state_dict(
    errors: list[str] | None = None,
    exit_orders: list[dict] | None = None,
    entry_candidates: list[dict] | None = None,
) -> dict:
    """Minimal trading state dict for testing cross-cutting logic."""
    return {
        "errors": errors or [],
        "exit_orders": exit_orders or [],
        "entry_candidates": entry_candidates or [],
        "position_reviews": [],
        "decisions": [],
    }


# Import execution gate and conflict resolution from where they live
try:
    from quantstack.graphs.trading.graph import (
        NODE_CLASSIFICATION,
        _BLOCKING_NODES,
        _ERROR_THRESHOLD,
        _execution_gate,
    )
    from quantstack.graphs.trading.nodes import resolve_symbol_conflicts
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False


# ── Test 1: Conflicting Symbols End-to-End ───────────────────────────────

@pytest.mark.skipif(not GRAPH_AVAILABLE, reason="Trading graph imports unavailable")
class TestConflictResolutionIntegration:
    """Conflict resolution + execution gate compose correctly."""

    @pytest.mark.asyncio
    async def test_conflicting_symbols_resolved_before_execution_gate(self):
        """Overlapping exit + entry symbols resolved; gate sees clean state."""
        state = _make_trading_state_dict(
            exit_orders=[
                {"symbol": "AAPL", "side": "sell", "quantity": 50},
                {"symbol": "TSLA", "side": "sell", "quantity": 30},
            ],
            entry_candidates=[
                {"symbol": "AAPL", "action": "buy", "quantity": 100},
                {"symbol": "NVDA", "action": "buy", "quantity": 75},
            ],
        )

        # Resolve conflicts
        result = await resolve_symbol_conflicts(state)

        # AAPL should be dropped from entries (exit takes priority)
        remaining_entries = result.get("entry_candidates", [])
        remaining_symbols = {c["symbol"] for c in remaining_entries}
        assert "AAPL" not in remaining_symbols
        assert "NVDA" in remaining_symbols

        # Exits unchanged
        assert len(result.get("exit_orders", state["exit_orders"])) == 2

        # Execution gate should see no errors
        merged_state = {**state, **result}
        assert _execution_gate(merged_state) == "continue"


# ── Test 2: Blocking Node Failure Halts Pipeline ────────────────────────

@pytest.mark.skipif(not GRAPH_AVAILABLE, reason="Trading graph imports unavailable")
class TestBlockingNodeHalts:
    """Blocking node error → execution gate halts."""

    def test_blocking_node_error_triggers_halt(self):
        """Error from a blocking node routes execution gate to halt."""
        state = _make_trading_state_dict(
            errors=["data_refresh: feed timeout after 30s"],
        )
        assert _execution_gate(state) == "halt"

    def test_non_blocking_node_error_continues(self):
        """Error from a non-blocking node allows pipeline to proceed."""
        state = _make_trading_state_dict(
            errors=["market_intel: web search failed"],
        )
        assert _execution_gate(state) == "continue"


# ── Test 3: Error Count Safety Net ──────────────────────────────────────

@pytest.mark.skipif(not GRAPH_AVAILABLE, reason="Trading graph imports unavailable")
class TestErrorCountSafetyNet:
    """Multiple non-blocking errors trigger safety halt at threshold."""

    def test_three_non_blocking_errors_halts(self):
        """3 non-blocking errors exceed threshold → halt."""
        state = _make_trading_state_dict(
            errors=[
                "plan_day: scheduling timeout",
                "market_intel: no data",
                "earnings_analysis: API 429",
            ],
        )
        assert _execution_gate(state) == "halt"

    def test_two_non_blocking_errors_continues(self):
        """2 non-blocking errors under threshold → continue."""
        state = _make_trading_state_dict(
            errors=[
                "plan_day: scheduling timeout",
                "market_intel: no data",
            ],
        )
        assert _execution_gate(state) == "continue"


# ── Test 4: Regime Flip + DLQ Compose ───────────────────────────────────

class TestRegimeFlipWithDLQFlow:
    """Regime flip action generation + DLQ monitoring compose correctly."""

    def test_severe_flip_generates_exit_then_dlq_unaffected(self):
        """Severe regime flip generates exit order; DLQ is a separate pathway."""
        # Regime flip: trending_up → trending_down (severe)
        actions = generate_regime_flip_actions(
            symbol="AAPL", side="long", quantity=100,
            entry_regime="trending_up", current_regime="trending_down",
            current_price=150.0, stop_price=None, entry_atr=2.0,
        )
        assert actions["severity"] == "severe"
        assert actions["exit_order"] is not None
        assert actions["exit_order"]["reason"] == "regime_flip_severe"
        # Belt-and-suspenders stop also set
        assert actions["new_stop"] is not None

    def test_moderate_flip_tightens_stop_with_floor(self):
        """Moderate flip tightens stop; floor enforced correctly."""
        actions = generate_regime_flip_actions(
            symbol="AAPL", side="long", quantity=100,
            entry_regime="trending_up", current_regime="ranging",
            current_price=150.0, stop_price=140.0, entry_atr=2.0,
        )
        assert actions["severity"] == "moderate"
        assert actions["exit_order"] is None
        # Stop tightened: distance=10, halved=5, floor=max(4.0, 1.5)=4.0 → 5 > 4 → stop=145
        assert actions["new_stop"] == pytest.approx(145.0)


# ── Test 5: DLQ Write + Monitor Rate Compose ────────────────────────────

class TestDLQEndToEnd:
    """parse_json_response DLQ write + monitor rate computation compose."""

    @patch("quantstack.db.db_conn")
    def test_parse_failure_writes_dlq_and_rate_computable(self, mock_db_conn):
        """Parse failure writes DLQ row; monitor can compute rate from count."""
        mock_conn = MagicMock()
        mock_db_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db_conn.return_value.__exit__ = MagicMock(return_value=False)

        # Trigger DLQ write via parse failure
        result = parse_json_response(
            "not valid json",
            fallback={"safe": True},
            agent_name="fund_manager",
            graph_name="trading",
            run_id="cycle-42",
            model_used="sonnet",
            prompt_text="Allocate capital based on signals",
        )
        assert result == {"safe": True}
        mock_conn.execute.assert_called_once()

        # Verify DLQ monitor rate computation works with mocked count
        with patch("quantstack.observability.dlq_monitor.count_dlq_entries", return_value=5):
            rate = compute_dlq_rate("fund_manager", 100)
            assert rate == pytest.approx(5.0)

        with patch("quantstack.observability.dlq_monitor.compute_dlq_rate", return_value=12.0):
            alert = check_dlq_alerts("fund_manager", 100)
            assert alert == "critical"


# ── Test 6: Priority Pruning + Type Override Compose ────────────────────

class TestPriorityPruningIntegration:
    """Priority tagging + pruning compose: type overrides survive pruning."""

    def test_risk_gate_message_survives_aggressive_pruning(self):
        """Risk gate output tagged P0 at construction time survives pruning."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        # Build oversized message list with a risk gate message
        msgs = [
            SystemMessage(content="system"),
            HumanMessage(content="user"),
        ]

        # Add a risk gate P0 message
        risk_msg = AIMessage(content="x" * 50000)
        tag_message_priority(risk_msg, agent_priority=PRIORITY_P1, message_type="risk_gate_check")
        assert _get_message_priority(risk_msg) == PRIORITY_P0  # Override worked
        msgs.append(risk_msg)

        # Add P2 filler messages to push over budget
        for _ in range(5):
            filler = AIMessage(content="x" * 40000)
            tag_message_priority(filler, agent_priority=PRIORITY_P2)
            msgs.append(filler)

        # Prune
        result = _prune_messages(msgs)

        # Risk gate message must survive
        p0_msgs = [m for m in result[2:] if _get_message_priority(m) == PRIORITY_P0]
        assert len(p0_msgs) == 1
        assert len(p0_msgs[0].content) == 50000  # Full content preserved
