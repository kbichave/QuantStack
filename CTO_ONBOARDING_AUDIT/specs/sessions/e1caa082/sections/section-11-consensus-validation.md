# Section 11: Consensus-Based Signal Validation (AR-6)

## Overview

The trading graph currently relies on a single agent to make entry decisions. If that agent has a directional blind spot (e.g., bullish bias in a topping market), there is no counterweight. The risk gate catches position sizing and exposure violations but cannot evaluate the quality of a directional call. This section adds a 3-agent consensus subgraph that independently evaluates trade signals above a configurable notional threshold, producing a deterministic sizing decision before the trade reaches the risk gate.

## Dependencies

- **section-01-db-migrations**: The `consensus_log` table must exist before consensus decisions can be persisted.
- **section-05-event-bus-extensions**: The `CONSENSUS_REQUIRED` and `CONSENSUS_REACHED` event types must be registered on the EventType enum.
- **section-10-knowledge-graph**: The bear advocate agent uses `check_factor_overlap` from the knowledge graph to detect factor crowding. If the KG is not yet populated, the bear agent operates without crowding data (graceful degradation, not a hard dependency).

## Blocks

- **section-13-meta-agents**: Meta agents need the consensus subgraph in place so they can tune consensus-related thresholds and prompts.

---

## Tests First

File: `tests/unit/test_consensus.py`

```python
"""Consensus-based signal validation tests.

Tests cover: threshold routing, deterministic merge logic, logging,
feature flag behavior, and agent state isolation.
"""

# --- Threshold Routing ---

# Test: trades > $5K route to consensus subgraph
# Given a signal with estimated notional $5,001, the entry scan node
# should route to the consensus subgraph (not directly to risk gate).

# Test: trades <= $5K bypass consensus (go directly to risk gate)
# Given a signal with estimated notional $5,000, the entry scan node
# should skip consensus and proceed to the risk gate.

# Test: $5K threshold configurable via CONSENSUS_THRESHOLD env var
# When CONSENSUS_THRESHOLD=10000 is set, a $7K trade should bypass
# consensus and a $10,001 trade should route to consensus.

# --- Deterministic Merge Logic ---

# Test: consensus_merge with 3/3 ENTER returns full position size (1.0)
# Given three agent votes all "ENTER", the merge node should return
# final_sizing_pct=1.0 and consensus_level="unanimous".

# Test: consensus_merge with 2/3 ENTER returns half position size (0.5)
# Given two "ENTER" and one "REJECT", the merge node should return
# final_sizing_pct=0.5 and consensus_level="majority".

# Test: consensus_merge with 1/3 ENTER returns reject (0.0)
# Given one "ENTER" and two "REJECT", the merge node should return
# final_sizing_pct=0.0 and consensus_level="minority".

# Test: consensus_merge with 0/3 ENTER returns reject (0.0)
# Given three "REJECT" votes, the merge node should return
# final_sizing_pct=0.0 and consensus_level="minority".

# --- Logging ---

# Test: consensus decision logged to consensus_log table
# After a consensus merge completes, a row should exist in consensus_log
# with all agent votes, reasoning, confidence scores, and the final decision.

# --- Feature Flag ---

# Test: CONSENSUS_ENABLED=false bypasses consensus entirely
# When the env var is false, all trades (regardless of notional) should
# skip the consensus subgraph and go directly to risk gate.

# --- Agent Independence ---

# Test: bull, bear, arbiter agents have independent state (no shared context)
# Each agent receives the same signal/market data but does NOT see other
# agents' arguments. Verify by checking that agent state objects are
# distinct instances with no shared references.

# Test: arbiter vote is binary (ENTER or REJECT), not a score
# The arbiter must return one of exactly two values. Any other value
# should raise a validation error.
```

---

## Implementation Details

### Architecture

The consensus subgraph is a LangGraph `StateGraph` embedded within the trading graph. It is invoked conditionally based on estimated trade notional. Three agents are spawned via the LangGraph `Send` API, each with independent state. Their votes converge at a deterministic merge node that applies fixed sizing rules with no LLM involvement.

### Trade Decision Routing

In the trading graph's entry scanning node (`src/quantstack/graphs/trading/nodes.py`), after a signal is generated and before the risk gate, add a conditional edge:

- Compute estimated notional from signal (price * proposed_shares).
- Read threshold from `CONSENSUS_THRESHOLD` env var (default: 5000).
- If notional > threshold AND `CONSENSUS_ENABLED` env var is true (default: true), route to consensus subgraph.
- Otherwise, proceed directly to risk gate (current behavior preserved).

### Consensus Subgraph

File: `src/quantstack/graphs/trading/consensus.py`

The subgraph contains four nodes:

**1. Bull Advocate** (Haiku tier)
- Receives: signal data, strategy metadata, market data (OHLCV, regime, signal brief).
- Task: Build the strongest possible case FOR entering the trade. Focus on technical momentum, catalysts, favorable regime alignment, historical precedent from knowledge graph.
- Tools bound: `signal_brief`, `fetch_market_data`, `search_knowledge_base`.
- Output: vote ("ENTER" or "REJECT"), confidence (0.0-1.0), reasoning (string).

**2. Bear Advocate** (Haiku tier)
- Receives: same inputs as bull (independently, not bull's output).
- Task: Build the strongest possible case AGAINST entering the trade. Look for divergences, overhead resistance, adverse macro conditions, factor crowding (via `check_factor_overlap` from the knowledge graph).
- Tools bound: same as bull plus `compute_risk_metrics`, `check_factor_overlap`.
- Output: vote ("ENTER" or "REJECT"), confidence (0.0-1.0), reasoning (string).

**3. Neutral Arbiter** (Haiku tier)
- Receives: same raw inputs as bull and bear (NOT their arguments — no anchoring bias).
- Task: Independently evaluate the trade signal. Score on evidence strength (1-5), logical coherence (1-5), and data recency (1-5). Return a binary vote.
- Tools bound: same as bull.
- Output: vote ("ENTER" or "REJECT"), confidence (0.0-1.0), reasoning (string).

The agents do not see each other's arguments. This is enforced by spawning them via `Send` with isolated state slices.

**4. Consensus Merge Node** (deterministic, no LLM)

```python
def consensus_merge(votes: list[AgentVote]) -> ConsensusResult:
    """Deterministic merge of 3 agent votes into a sizing decision.

    Rules:
        3/3 ENTER -> full position size (1.0), consensus_level="unanimous"
        2/3 ENTER -> half position size (0.5), consensus_level="majority"
        <2/3 ENTER -> reject (0.0), consensus_level="minority"

    No LLM call. Pure count-based logic.
    """
```

After computing the result, the merge node:
1. Writes a row to the `consensus_log` table (schema below).
2. Publishes a `CONSENSUS_REACHED` event to the event bus with decision_id, consensus_level, and final_sizing_pct.
3. If final_sizing_pct > 0, passes the adjusted position size to the risk gate.
4. If final_sizing_pct == 0, the trade is rejected (does not reach risk gate).

### Data Model

The `consensus_log` table (created by section-01-db-migrations) stores every consensus decision:

```python
@dataclass
class ConsensusDecision:
    decision_id: str          # UUID
    signal_id: str            # references the originating signal
    symbol: str
    notional: float           # estimated notional that triggered consensus
    bull_vote: str            # "ENTER" or "REJECT"
    bull_confidence: float
    bull_reasoning: str
    bear_vote: str
    bear_confidence: float
    bear_reasoning: str
    arbiter_vote: str
    arbiter_confidence: float
    arbiter_reasoning: str
    consensus_level: str      # "unanimous", "majority", "minority"
    final_sizing_pct: float   # 1.0, 0.5, or 0.0
    created_at: datetime      # TIMESTAMPTZ
```

### Agent Configuration

File: `src/quantstack/graphs/trading/config/agents.yaml`

Add three new agent entries:

```yaml
bull_advocate:
  description: "Builds the strongest case FOR a trade entry"
  tier: light          # Haiku
  tools:
    - signal_brief
    - fetch_market_data
    - search_knowledge_base

bear_advocate:
  description: "Builds the strongest case AGAINST a trade entry"
  tier: light          # Haiku
  tools:
    - signal_brief
    - fetch_market_data
    - search_knowledge_base
    - compute_risk_metrics
    - check_factor_overlap

neutral_arbiter:
  description: "Independent evaluator — binary ENTER/REJECT vote"
  tier: light          # Haiku
  tools:
    - signal_brief
    - fetch_market_data
    - search_knowledge_base
```

### Event Integration

Before spawning the consensus subgraph, publish a `CONSENSUS_REQUIRED` event with:
- `signal_id`: the signal being evaluated
- `symbol`: ticker
- `notional`: estimated trade size

After the merge node completes, publish a `CONSENSUS_REACHED` event with:
- `decision_id`: the consensus decision UUID
- `consensus_level`: "unanimous", "majority", or "minority"
- `final_sizing`: 1.0, 0.5, or 0.0

### Feature Flag

The `CONSENSUS_ENABLED` environment variable (default: `"true"`) controls whether consensus runs at all. When set to `"false"`, the routing logic in the entry scan node skips the consensus subgraph entirely and routes all trades directly to the risk gate, regardless of notional size. This allows disabling consensus if it adds latency without improving outcomes.

### Files to Create or Modify

| File | Action | What Changes |
|------|--------|-------------|
| `src/quantstack/graphs/trading/consensus.py` | CREATE | Consensus subgraph: bull, bear, arbiter agents + deterministic merge node |
| `src/quantstack/graphs/trading/nodes.py` | MODIFY | Add consensus routing conditional edge in entry scan node |
| `src/quantstack/graphs/trading/graph.py` | MODIFY | Wire consensus subgraph into the trading graph |
| `src/quantstack/graphs/trading/config/agents.yaml` | MODIFY | Add bull_advocate, bear_advocate, neutral_arbiter agent configs |
| `tests/unit/test_consensus.py` | CREATE | All unit tests listed above |

### Key Design Decisions

1. **$5K threshold, not per-trade consensus for everything.** Consensus adds ~30 seconds of LLM latency (3 parallel Haiku calls). For small trades, this latency is not worth the protection. The threshold is configurable via env var to allow tuning based on observed outcomes.

2. **Agents do not debate.** Bull and bear do not see each other's arguments. The arbiter does not see either argument. This prevents anchoring bias — each agent evaluates the raw signal independently. If agents saw each other's reasoning, the arbiter would likely anchor to the more articulate argument rather than the more correct one.

3. **Arbiter votes binary, not scores.** Scoring without commitment leads to indecisive middle-ground values (e.g., 6/10). Forcing a binary ENTER/REJECT decision requires the arbiter to commit to a position. The confidence score is logged for analysis but does not affect the merge logic.

4. **Deterministic merge, not LLM synthesis.** The merge node is pure arithmetic (count ENTER votes). This makes the decision auditable, predictable, and zero-cost. An LLM-based synthesis would add cost, latency, and unpredictability to a safety-critical decision point.

5. **Consensus before risk gate, not after.** Consensus adjusts position sizing (or rejects entirely). The risk gate then validates the adjusted size against exposure limits. If consensus were after the risk gate, the risk gate would validate a size that consensus might then reduce — wasting the risk gate computation.

### Failure Modes

- **One agent times out:** If any of the three agents fails to produce a vote within 30 seconds, treat it as a REJECT vote. This is conservative — a missing vote should not enable a trade.
- **All agents time out:** All three treated as REJECT. Trade is rejected. This is the correct degradation — if the system cannot evaluate a trade, it should not take it.
- **Consensus log write fails:** Log the failure, but do not block the trade decision. The consensus decision is in memory; the log write is for post-hoc analysis, not for correctness.
- **Knowledge graph unavailable for bear agent:** The bear agent's `check_factor_overlap` call returns empty results. The bear agent still evaluates based on technical analysis and risk metrics. Factor crowding is one input, not the only input.
