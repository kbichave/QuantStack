"""Tests for the Trading Graph (Section 08).

Tests cover graph structure, node routing, parallel branch behavior,
and mandatory risk gate enforcement.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_config_watcher(tmp_path):
    """Create a ConfigWatcher with trading agent configs."""
    yaml_content = """
daily_planner:
  role: "Daily Trading Planner"
  goal: "Generate actionable daily trading plan."
  backstory: "Senior trading planner."
  llm_tier: medium
  max_iterations: 10
  timeout_seconds: 120
  tools:
    - signal_brief

position_monitor:
  role: "Position Monitor"
  goal: "Review open positions."
  backstory: "Position monitoring specialist."
  llm_tier: medium
  max_iterations: 15
  timeout_seconds: 300
  tools:
    - signal_brief

trade_debater:
  role: "Trade Entry Debater"
  goal: "Find entry candidates."
  backstory: "Market scanner."
  llm_tier: heavy
  max_iterations: 20
  timeout_seconds: 600
  tools:
    - signal_brief
    - fetch_market_data

risk_analyst:
  role: "Risk Analyst"
  goal: "Size positions and validate risk."
  backstory: "Risk management specialist."
  llm_tier: medium
  max_iterations: 10
  timeout_seconds: 120
  tools:
    - compute_risk_metrics

fund_manager:
  role: "Fund Manager"
  goal: "Review proposed entries for portfolio risk."
  backstory: "Portfolio-level approval gate."
  llm_tier: heavy
  max_iterations: 10
  timeout_seconds: 180
  tools:
    - fetch_portfolio

options_analyst:
  role: "Options Analyst"
  goal: "Select optimal options structures."
  backstory: "Options specialist."
  llm_tier: heavy
  max_iterations: 15
  timeout_seconds: 300
  tools:
    - fetch_portfolio

trade_reflector:
  role: "Trade Reflector"
  goal: "Analyze completed trades."
  backstory: "Post-trade analyst."
  llm_tier: medium
  max_iterations: 5
  timeout_seconds: 120
  tools:
    - signal_brief
"""
    yaml_file = tmp_path / "agents.yaml"
    yaml_file.write_text(yaml_content)
    from quantstack.graphs.config_watcher import ConfigWatcher
    watcher = ConfigWatcher(yaml_file)
    yield watcher
    watcher.stop()


def _make_mock_model(content='{"result": "ok"}'):
    """Create a mock ChatModel returning the given JSON content."""
    from langchain_core.messages import AIMessage
    model = MagicMock()
    model.bind_tools = MagicMock(return_value=model)
    model.ainvoke = AsyncMock(return_value=AIMessage(content=content))
    return model


class TestBuildTradingGraph:
    """Tests for the graph builder."""

    def test_returns_compiled_graph(self, mock_config_watcher):
        from langgraph.checkpoint.memory import MemorySaver
        from quantstack.graphs.trading.graph import build_trading_graph
        with patch("quantstack.graphs.trading.graph.get_chat_model") as mock_gcm:
            mock_gcm.return_value = MagicMock()
            graph = build_trading_graph(mock_config_watcher, MemorySaver())
        assert graph is not None
        assert hasattr(graph, "ainvoke")

    def test_graph_has_twelve_nodes(self, mock_config_watcher):
        from langgraph.checkpoint.memory import MemorySaver
        from quantstack.graphs.trading.graph import build_trading_graph
        with patch("quantstack.graphs.trading.graph.get_chat_model") as mock_gcm:
            mock_gcm.return_value = MagicMock()
            graph = build_trading_graph(mock_config_watcher, MemorySaver())
        node_names = set(graph.get_graph().nodes.keys())
        expected_nodes = {
            "safety_check", "plan_day", "position_review", "execute_exits",
            "entry_scan", "merge_parallel", "risk_sizing", "portfolio_review",
            "analyze_options", "execute_entries", "reflect",
        }
        assert expected_nodes.issubset(node_names), (
            f"Missing nodes: {expected_nodes - node_names}"
        )
        # 11 real nodes (merge_parallel is the 11th, __start__/__end__ excluded)
        real_nodes = node_names - {"__start__", "__end__"}
        assert len(real_nodes) == 11

    def test_reads_agent_configs(self, mock_config_watcher):
        from langgraph.checkpoint.memory import MemorySaver
        from quantstack.graphs.trading.graph import build_trading_graph
        with patch("quantstack.graphs.trading.graph.get_chat_model") as mock_gcm:
            mock_gcm.return_value = MagicMock()
            build_trading_graph(mock_config_watcher, MemorySaver())
        tier_calls = [call.args[0] for call in mock_gcm.call_args_list]
        assert "medium" in tier_calls
        assert "heavy" in tier_calls

    def test_parallel_branches_from_plan_day(self, mock_config_watcher):
        """plan_day has two outgoing edges to position_review AND entry_scan."""
        from langgraph.checkpoint.memory import MemorySaver
        from quantstack.graphs.trading.graph import build_trading_graph
        with patch("quantstack.graphs.trading.graph.get_chat_model") as mock_gcm:
            mock_gcm.return_value = MagicMock()
            graph = build_trading_graph(mock_config_watcher, MemorySaver())
        graph_data = graph.get_graph()
        dp_edges = [e for e in graph_data.edges if e.source == "plan_day"]
        targets = {e.target for e in dp_edges}
        assert "position_review" in targets
        assert "entry_scan" in targets


class TestSafetyCheckRouter:
    """Tests for the safety_check conditional router."""

    def test_routes_to_daily_plan_when_healthy(self):
        from quantstack.graphs.trading.graph import _safety_check_router
        state = {
            "decisions": [{"node": "safety_check", "halted": False}],
            "errors": [],
        }
        assert _safety_check_router(state) == "continue"

    def test_routes_to_end_when_halted(self):
        from quantstack.graphs.trading.graph import _safety_check_router
        state = {
            "decisions": [{"node": "safety_check", "halted": True}],
            "errors": [],
        }
        assert _safety_check_router(state) == "halt"

    def test_routes_to_end_on_safety_check_error(self):
        from quantstack.graphs.trading.graph import _safety_check_router
        state = {
            "decisions": [],
            "errors": ["safety_check: connection refused"],
        }
        assert _safety_check_router(state) == "halt"


class TestRiskGateRouter:
    """Tests for the risk gate conditional router."""

    def test_routes_to_portfolio_review_when_approved(self):
        from quantstack.graphs.trading.graph import _risk_gate_router
        state = {"risk_verdicts": [{"symbol": "SPY", "approved": True}]}
        assert _risk_gate_router(state) == "approved"

    def test_routes_to_end_when_rejected(self):
        from quantstack.graphs.trading.graph import _risk_gate_router
        state = {"risk_verdicts": [
            {"symbol": "SPY", "approved": True},
            {"symbol": "QQQ", "approved": False, "violations": ["too large"]},
        ]}
        assert _risk_gate_router(state) == "rejected"

    def test_routes_to_end_when_no_candidates(self):
        from quantstack.graphs.trading.graph import _risk_gate_router
        state = {"risk_verdicts": []}
        assert _risk_gate_router(state) == "rejected"

    def test_routes_to_end_when_missing_verdicts(self):
        from quantstack.graphs.trading.graph import _risk_gate_router
        state = {}
        assert _risk_gate_router(state) == "rejected"


class TestRiskGateStructural:
    """Structural test: no path from risk_sizing to execute_entries bypasses risk gate."""

    def test_no_bypass_path(self, mock_config_watcher):
        from langgraph.checkpoint.memory import MemorySaver
        from quantstack.graphs.trading.graph import build_trading_graph
        with patch("quantstack.graphs.trading.graph.get_chat_model") as mock_gcm:
            mock_gcm.return_value = MagicMock()
            graph = build_trading_graph(mock_config_watcher, MemorySaver())

        graph_data = graph.get_graph()
        # risk_sizing should only have conditional edges (no direct edge to execute_entries)
        rs_edges = [e for e in graph_data.edges if e.source == "risk_sizing"]
        rs_targets = {e.target for e in rs_edges}
        # Must NOT have a direct edge to execute_entries
        assert "execute_entries" not in rs_targets
        # Must route through portfolio_review (approved) or __end__ (rejected)
        assert "portfolio_review" in rs_targets or "__end__" in rs_targets


class TestTradingGraphExecution:
    """Tests for graph execution with mock LLM."""

    @pytest.mark.asyncio
    async def test_full_invocation_healthy_with_entries(self, mock_config_watcher):
        """Full graph run: healthy system, candidates found, risk approved."""
        from langchain_core.messages import AIMessage
        from langgraph.checkpoint.memory import MemorySaver

        mock_model = MagicMock()
        mock_model.bind_tools = MagicMock(return_value=mock_model)
        # Return JSON that navigates the full happy path
        mock_model.ainvoke = AsyncMock(return_value=AIMessage(
            content='{"halted": false, "plan": "test plan", '
            '"symbol": "SPY", "action": "HOLD", '
            '"strategy": "swing", "signal_strength": 0.8, '
            '"recommended_size_pct": 5.0, "reasoning": "ok", "confidence": 0.9, '
            '"decision": "APPROVED", "reason": "ok", '
            '"structure": "none", "params": {}, '
            '"order_id": "ord-1", "type": "market", '
            '"reflection": "good cycle", "lessons": []}'
        ))

        from quantstack.graphs.trading.graph import build_trading_graph
        with patch("quantstack.graphs.trading.graph.get_chat_model", return_value=mock_model):
            graph = build_trading_graph(mock_config_watcher, MemorySaver())

        result = await graph.ainvoke(
            {
                "cycle_number": 1,
                "regime": "trending_up",
                "portfolio_context": {
                    "positions": [],
                    "total_equity": 100000,
                    "daily_pnl_pct": 0.0,
                    "gross_exposure_pct": 0.5,
                    "average_daily_volume": 1000000,
                },
                "errors": [],
                "decisions": [],
            },
            config={"configurable": {"thread_id": "test-happy"}},
        )
        assert "decisions" in result
        assert isinstance(result["decisions"], list)
        assert len(result["decisions"]) > 0

    @pytest.mark.asyncio
    async def test_halted_system_stops_early(self, mock_config_watcher):
        """Graph terminates after safety_check when system is halted."""
        from langchain_core.messages import AIMessage
        from langgraph.checkpoint.memory import MemorySaver

        mock_model = MagicMock()
        mock_model.bind_tools = MagicMock(return_value=mock_model)
        mock_model.ainvoke = AsyncMock(return_value=AIMessage(
            content='{"halted": true, "reason": "daily loss halt triggered"}'
        ))

        from quantstack.graphs.trading.graph import build_trading_graph
        with patch("quantstack.graphs.trading.graph.get_chat_model", return_value=mock_model):
            graph = build_trading_graph(mock_config_watcher, MemorySaver())

        result = await graph.ainvoke(
            {
                "cycle_number": 1,
                "regime": "trending_down",
                "portfolio_context": {},
                "errors": [],
                "decisions": [],
            },
            config={"configurable": {"thread_id": "test-halted"}},
        )
        # Should have halted — no daily_plan in decisions
        node_names = {d.get("node") for d in result.get("decisions", [])}
        assert "safety_check" in node_names
        assert "daily_plan" not in node_names
        assert len(result.get("errors", [])) > 0

    @pytest.mark.asyncio
    async def test_risk_gate_rejection_no_entries(self, mock_config_watcher):
        """When risk gate rejects, entry_orders should be empty."""
        from langchain_core.messages import AIMessage
        from langgraph.checkpoint.memory import MemorySaver

        call_count = 0

        async def staged_responses(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # safety_check: healthy
            if call_count == 1:
                return AIMessage(content='{"halted": false}')
            # daily_plan
            if call_count == 2:
                return AIMessage(content='{"plan": "test"}')
            # position_review (parallel with entry_scan)
            if call_count in (3, 4):
                # One returns positions, other returns candidates
                # entry_scan returns a candidate that will be oversized
                return AIMessage(
                    content='[{"symbol": "MEGA", "strategy": "yolo", '
                    '"signal_strength": 0.9, "action": "HOLD", '
                    '"recommended_size_pct": 50.0, "reasoning": "huge", "confidence": 0.5}]'
                )
            # execute_exits
            if call_count == 5:
                return AIMessage(content='[]')
            # risk_sizing: returns oversized position that SafetyGate will reject
            if call_count == 6:
                return AIMessage(
                    content='[{"symbol": "MEGA", "recommended_size_pct": 50.0, '
                    '"reasoning": "yolo", "confidence": 0.5}]'
                )
            return AIMessage(content='{"result": "ok"}')

        mock_model = MagicMock()
        mock_model.bind_tools = MagicMock(return_value=mock_model)
        mock_model.ainvoke = AsyncMock(side_effect=staged_responses)

        from quantstack.graphs.trading.graph import build_trading_graph
        with patch("quantstack.graphs.trading.graph.get_chat_model", return_value=mock_model):
            graph = build_trading_graph(mock_config_watcher, MemorySaver())

        result = await graph.ainvoke(
            {
                "cycle_number": 1,
                "regime": "trending_up",
                "portfolio_context": {
                    "positions": [],
                    "total_equity": 100000,
                    "daily_pnl_pct": 0.0,
                    "gross_exposure_pct": 0.5,
                    "average_daily_volume": 1000000,
                },
                "errors": [],
                "decisions": [],
            },
            config={"configurable": {"thread_id": "test-rejected"}},
        )
        # Risk gate should reject — no execute_entries or portfolio_review
        node_names = {d.get("node") for d in result.get("decisions", [])}
        assert "execute_entries" not in node_names
        # entry_orders should be empty or absent
        assert result.get("entry_orders", []) == [] or "entry_orders" not in result
