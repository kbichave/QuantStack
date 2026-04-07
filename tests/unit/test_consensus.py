"""Unit tests for the 3-agent consensus subgraph.

All tests run without a database or LLM provider. DB interactions are
mocked; agent nodes are deterministic stubs.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from quantstack.graphs.trading.consensus import (
    AgentVote,
    ConsensusResult,
    ConsensusState,
    arbiter_node,
    bear_advocate_node,
    build_consensus_graph,
    bull_advocate_node,
    consensus_merge,
    fan_out_agents,
    should_run_consensus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_STATE: ConsensusState = {
    "signal_data": {"signal_id": "sig-1", "symbol": "AAPL", "notional": 6000.0},
    "market_data": {"regime": "trending_up"},
    "votes": [],
}


def _make_vote(role: str, vote: str, confidence: float = 0.7) -> AgentVote:
    return AgentVote(
        agent_role=role,
        vote=vote,
        confidence=confidence,
        reasoning=f"{role} reasoning",
    )


# ---------------------------------------------------------------------------
# Threshold routing
# ---------------------------------------------------------------------------


class TestShouldRunConsensus:
    def test_above_threshold_routes_to_consensus(self):
        assert should_run_consensus(5001.0) is True

    def test_at_threshold_bypasses_consensus(self):
        assert should_run_consensus(5000.0) is False

    def test_below_threshold_bypasses_consensus(self):
        assert should_run_consensus(3000.0) is False

    def test_threshold_configurable(self):
        assert should_run_consensus(7000.0, threshold=10000.0) is False
        assert should_run_consensus(10001.0, threshold=10000.0) is True

    def test_disabled_bypasses_all(self):
        assert should_run_consensus(999_999.0, consensus_enabled=False) is False


# ---------------------------------------------------------------------------
# Deterministic merge
# ---------------------------------------------------------------------------


class TestConsensusMerge:
    def test_unanimous_enter(self):
        state: ConsensusState = {
            **_BASE_STATE,
            "votes": [
                _make_vote("bull", "ENTER"),
                _make_vote("bear", "ENTER"),
                _make_vote("arbiter", "ENTER"),
            ],
        }
        with patch("quantstack.graphs.trading.consensus._log_consensus"):
            result = consensus_merge(state)
        assert result.final_sizing_pct == 1.0
        assert result.consensus_level == "unanimous"

    def test_majority_enter(self):
        state: ConsensusState = {
            **_BASE_STATE,
            "votes": [
                _make_vote("bull", "ENTER"),
                _make_vote("bear", "REJECT"),
                _make_vote("arbiter", "ENTER"),
            ],
        }
        with patch("quantstack.graphs.trading.consensus._log_consensus"):
            result = consensus_merge(state)
        assert result.final_sizing_pct == 0.5
        assert result.consensus_level == "majority"

    def test_minority_enter(self):
        state: ConsensusState = {
            **_BASE_STATE,
            "votes": [
                _make_vote("bull", "ENTER"),
                _make_vote("bear", "REJECT"),
                _make_vote("arbiter", "REJECT"),
            ],
        }
        with patch("quantstack.graphs.trading.consensus._log_consensus"):
            result = consensus_merge(state)
        assert result.final_sizing_pct == 0.0
        assert result.consensus_level == "minority"

    def test_all_reject(self):
        state: ConsensusState = {
            **_BASE_STATE,
            "votes": [
                _make_vote("bull", "REJECT"),
                _make_vote("bear", "REJECT"),
                _make_vote("arbiter", "REJECT"),
            ],
        }
        with patch("quantstack.graphs.trading.consensus._log_consensus"):
            result = consensus_merge(state)
        assert result.final_sizing_pct == 0.0
        assert result.consensus_level == "minority"


# ---------------------------------------------------------------------------
# Agent independence
# ---------------------------------------------------------------------------


class TestAgentIndependence:
    def test_agent_votes_are_independent(self):
        """Each agent produces its own vote from the same input state."""
        bull_result = bull_advocate_node(_BASE_STATE)
        bear_result = bear_advocate_node(_BASE_STATE)
        arbiter_result = arbiter_node(_BASE_STATE)

        bull_vote = bull_result["votes"][0]
        bear_vote = bear_result["votes"][0]
        arbiter_vote = arbiter_result["votes"][0]

        assert bull_vote.agent_role == "bull"
        assert bear_vote.agent_role == "bear"
        assert arbiter_vote.agent_role == "arbiter"

        # Votes can differ — they are independent assessments
        roles = {bull_vote.agent_role, bear_vote.agent_role, arbiter_vote.agent_role}
        assert len(roles) == 3

    def test_arbiter_vote_is_binary(self):
        result = arbiter_node(_BASE_STATE)
        vote = result["votes"][0]
        assert vote.vote in ("ENTER", "REJECT")


# ---------------------------------------------------------------------------
# Pydantic model validation
# ---------------------------------------------------------------------------


class TestModels:
    def test_agent_vote_model(self):
        vote = AgentVote(
            agent_role="bull", vote="ENTER", confidence=0.8, reasoning="looks good"
        )
        assert vote.agent_role == "bull"
        assert vote.confidence == 0.8

    def test_consensus_result_model(self):
        result = ConsensusResult(
            consensus_level="unanimous",
            final_sizing_pct=1.0,
            votes=[_make_vote("bull", "ENTER")],
            decision_id="abc123",
        )
        assert result.decision_id == "abc123"
        assert len(result.votes) == 1

    def test_invalid_vote_rejected(self):
        with pytest.raises(ValidationError):
            AgentVote(
                agent_role="bull",
                vote="MAYBE",
                confidence=0.5,
                reasoning="unsure",
            )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestLogging:
    @patch("quantstack.graphs.trading.consensus._log_consensus")
    def test_consensus_logged(self, mock_log):
        state: ConsensusState = {
            **_BASE_STATE,
            "votes": [
                _make_vote("bull", "ENTER"),
                _make_vote("bear", "REJECT"),
                _make_vote("arbiter", "ENTER"),
            ],
        }
        result = consensus_merge(state)

        # Verify _log_consensus was called with the result and state
        assert mock_log.called
        call_args = mock_log.call_args
        logged_result = call_args[0][0]
        assert logged_result.consensus_level == "majority"
        assert result.consensus_level == "majority"


# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------


class TestFanOut:
    def test_fan_out_returns_3_sends(self):
        sends = fan_out_agents(_BASE_STATE)
        assert len(sends) == 3

    def test_fan_out_send_targets(self):
        sends = fan_out_agents(_BASE_STATE)
        targets = {s.node for s in sends}
        assert targets == {"bull_advocate", "bear_advocate", "arbiter"}


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------


class TestGraphBuild:
    def test_graph_compiles(self):
        graph = build_consensus_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        graph = build_consensus_graph()
        # CompiledStateGraph exposes node names via .nodes
        node_names = set(graph.nodes.keys())
        for expected in ("bull_advocate", "bear_advocate", "arbiter", "merge"):
            assert expected in node_names, f"Missing node: {expected}"


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------


class TestFeatureFlag:
    def test_consensus_enabled_false_bypasses(self):
        """Feature flag off means no consensus regardless of notional."""
        for notional in (100.0, 5001.0, 50_000.0, 1_000_000.0):
            assert should_run_consensus(notional, consensus_enabled=False) is False
