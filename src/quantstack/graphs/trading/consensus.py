"""Three-agent consensus subgraph for trade signal validation.

Bull advocate, bear advocate, and neutral arbiter independently evaluate
trade signals above a configurable notional threshold. A deterministic
merge node tallies votes and produces a sizing decision.

Agent nodes are currently deterministic stubs. LLM-backed reasoning
will replace them once the consensus loop is proven in paper trading.
"""

import logging
import operator
import uuid
from typing import Annotated, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from pydantic import BaseModel
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AgentVote(BaseModel):
    """A single advocate's vote on a trade signal."""

    agent_role: str
    vote: Literal["ENTER", "REJECT"]
    confidence: float  # 0.0–1.0
    reasoning: str


class ConsensusResult(BaseModel):
    """Outcome of the 3-agent consensus process."""

    consensus_level: Literal["unanimous", "majority", "minority"]
    final_sizing_pct: float
    votes: list[AgentVote]
    decision_id: str


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class ConsensusState(TypedDict):
    signal_data: dict
    market_data: dict
    votes: Annotated[list, operator.add]


# ---------------------------------------------------------------------------
# Threshold gate
# ---------------------------------------------------------------------------


def should_run_consensus(
    notional: float,
    consensus_enabled: bool = True,
    threshold: float = 5000.0,
) -> bool:
    """Return True when a signal's notional value warrants consensus review.

    The check is strictly greater-than so that signals exactly at the
    threshold skip the (relatively expensive) multi-agent deliberation.
    """
    if not consensus_enabled:
        return False
    return notional > threshold


# ---------------------------------------------------------------------------
# Agent nodes (deterministic stubs — no LLM calls)
# ---------------------------------------------------------------------------


def bull_advocate_node(state: ConsensusState) -> dict:
    """Bull advocate: biased toward entering the trade."""
    vote = AgentVote(
        agent_role="bull",
        vote="ENTER",
        confidence=0.7,
        reasoning="Momentum indicators and recent price action support entry.",
    )
    return {"votes": [vote]}


def bear_advocate_node(state: ConsensusState) -> dict:
    """Bear advocate: biased toward rejecting the trade."""
    vote = AgentVote(
        agent_role="bear",
        vote="REJECT",
        confidence=0.6,
        reasoning="Elevated downside risk and unfavorable risk/reward ratio.",
    )
    return {"votes": [vote]}


def arbiter_node(state: ConsensusState) -> dict:
    """Neutral arbiter: casts the tie-breaking vote."""
    vote = AgentVote(
        agent_role="arbiter",
        vote="ENTER",
        confidence=0.5,
        reasoning="Signal meets minimum criteria; risk is within acceptable bounds.",
    )
    return {"votes": [vote]}


# ---------------------------------------------------------------------------
# Deterministic merge
# ---------------------------------------------------------------------------


def consensus_merge(state: ConsensusState) -> ConsensusResult:
    """Tally votes and produce a sizing decision.

    Sizing rules:
      - 3/3 ENTER  -> 1.0  (unanimous)
      - 2/3 ENTER  -> 0.5  (majority)
      - <2  ENTER  -> 0.0  (minority — do not trade)

    Logs the decision to the ``consensus_log`` table on a best-effort
    basis; DB failures never block the trading pipeline.
    """
    votes: list[AgentVote] = [
        v if isinstance(v, AgentVote) else AgentVote(**v) for v in state["votes"]
    ]
    enter_count = sum(1 for v in votes if v.vote == "ENTER")

    if enter_count == 3:
        consensus_level: Literal["unanimous", "majority", "minority"] = "unanimous"
        final_sizing_pct = 1.0
    elif enter_count == 2:
        consensus_level = "majority"
        final_sizing_pct = 0.5
    else:
        consensus_level = "minority"
        final_sizing_pct = 0.0

    decision_id = uuid.uuid4().hex[:16]

    result = ConsensusResult(
        consensus_level=consensus_level,
        final_sizing_pct=final_sizing_pct,
        votes=votes,
        decision_id=decision_id,
    )

    _log_consensus(result, state)
    return result


def _log_consensus(result: ConsensusResult, state: ConsensusState) -> None:
    """Best-effort write to consensus_log. Failures are logged, never raised."""
    try:
        from quantstack.db import db_conn

        votes_by_role = {v.agent_role: v for v in result.votes}
        bull = votes_by_role.get("bull")
        bear = votes_by_role.get("bear")
        arbiter = votes_by_role.get("arbiter")
        signal = state.get("signal_data", {})

        with db_conn() as conn:
            conn.execute(
                """
                INSERT INTO consensus_log (
                    decision_id, signal_id, symbol, notional,
                    bull_vote, bull_confidence, bull_reasoning,
                    bear_vote, bear_confidence, bear_reasoning,
                    arbiter_vote, arbiter_confidence, arbiter_reasoning,
                    consensus_level, final_sizing_pct
                ) VALUES (
                    %(decision_id)s, %(signal_id)s, %(symbol)s, %(notional)s,
                    %(bull_vote)s, %(bull_confidence)s, %(bull_reasoning)s,
                    %(bear_vote)s, %(bear_confidence)s, %(bear_reasoning)s,
                    %(arbiter_vote)s, %(arbiter_confidence)s, %(arbiter_reasoning)s,
                    %(consensus_level)s, %(final_sizing_pct)s
                )
                """,
                {
                    "decision_id": result.decision_id,
                    "signal_id": signal.get("signal_id", ""),
                    "symbol": signal.get("symbol", ""),
                    "notional": signal.get("notional", 0.0),
                    "bull_vote": bull.vote if bull else "",
                    "bull_confidence": bull.confidence if bull else 0.0,
                    "bull_reasoning": bull.reasoning if bull else "",
                    "bear_vote": bear.vote if bear else "",
                    "bear_confidence": bear.confidence if bear else 0.0,
                    "bear_reasoning": bear.reasoning if bear else "",
                    "arbiter_vote": arbiter.vote if arbiter else "",
                    "arbiter_confidence": arbiter.confidence if arbiter else 0.0,
                    "arbiter_reasoning": arbiter.reasoning if arbiter else "",
                    "consensus_level": result.consensus_level,
                    "final_sizing_pct": result.final_sizing_pct,
                },
            )
    except Exception:
        logger.warning("consensus_log write failed", exc_info=True)


# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------


def fan_out_agents(state: ConsensusState) -> list[Send]:
    """Dispatch bull, bear, and arbiter nodes in parallel via Send."""
    return [
        Send("bull_advocate", state),
        Send("bear_advocate", state),
        Send("arbiter", state),
    ]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_consensus_graph() -> CompiledStateGraph:
    """Build and compile the 3-agent consensus subgraph.

    Topology:
      START -> fan_out -> [bull_advocate, bear_advocate, arbiter] -> merge -> END
    """
    graph = StateGraph(ConsensusState)

    graph.add_node("bull_advocate", bull_advocate_node)
    graph.add_node("bear_advocate", bear_advocate_node)
    graph.add_node("arbiter", arbiter_node)
    graph.add_node("merge", consensus_merge)

    graph.add_conditional_edges(START, fan_out_agents)
    graph.add_edge("bull_advocate", "merge")
    graph.add_edge("bear_advocate", "merge")
    graph.add_edge("arbiter", "merge")
    graph.add_edge("merge", END)

    return graph.compile()
