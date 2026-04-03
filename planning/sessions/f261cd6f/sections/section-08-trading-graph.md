# Section 08: Trading Graph

## Overview

Build the `TradingGraph` as a LangGraph `StateGraph` to replace the existing CrewAI `TradingCrew` (11-task sequential crew in `src/quantstack/crews/trading/crew.py`). The trading graph is the most complex of the three graphs: 12 nodes, two parallel branches with a join node, and a mandatory risk gate conditional edge enforced by the pure-Python `SafetyGate` in `src/quantstack/crews/risk/safety_gate.py`.

The existing `TradingCrew` runs 10 agents across 11 tasks in a strictly sequential `Process.sequential` flow with `memory=True` for implicit conversational context. The new graph replaces implicit memory with explicit typed state (`TradingState` from section-04), replaces sequential ordering with a graph topology that enables parallel execution of independent branches, and enforces the risk gate structurally via conditional edges rather than by convention.

**Key architectural constraint**: The `SafetyGate` is LAW. There must be no path through the compiled graph from `risk_sizing` to `execute_entries` that bypasses the risk gate conditional edge. This is enforced by graph topology, not by runtime checks.

## Dependencies

- **section-02-llm-provider**: `get_chat_model(tier)` must be available to construct `BaseChatModel` instances for agent nodes.
- **section-03-agent-config**: `AgentConfig` dataclass and `ConfigWatcher` must be implemented. The trading graph's `agents.yaml` will be loaded through this system.
- **section-04-state-schemas**: `TradingState` TypedDict must be defined (see schema below for reference).
- **section-05-tool-layer**: LLM-facing tools (`@tool` decorated) and node-callable functions must be split and registered in `TOOL_REGISTRY`.

## Tests (Write First)

All tests go in `tests/unit/test_trading_graph.py`. Write these before the implementation.

```python
# tests/unit/test_trading_graph.py

"""Unit tests for the trading LangGraph StateGraph.

Tests cover graph structure, node routing, parallel branch behavior,
and mandatory risk gate enforcement.
"""

import pytest


# --- Graph Structure ---

# Test: build_trading_graph() returns CompiledStateGraph
#   Call build_trading_graph(config_watcher, checkpointer) with test fixtures.
#   Assert return type is CompiledStateGraph.

# Test: trading graph has expected node count (12 including merge_parallel)
#   Compile the graph. Count nodes. Assert == 12.
#   Nodes: safety_check, daily_plan, position_review, execute_exits,
#          entry_scan, merge_parallel, risk_sizing, portfolio_review,
#          options_analysis, execute_entries, reflection, (implicit __start__/__end__ excluded)

# Test: safety_check routes to END when system halted
#   Invoke graph with state where get_system_status() returns halted.
#   Assert graph terminates after safety_check.
#   Assert errors field contains halt reason.

# Test: safety_check routes to daily_plan when system ok
#   Invoke graph with state where system is healthy.
#   Assert daily_plan node executes.


# --- Parallel Branches ---

# Test: daily_plan has two outgoing edges (position_review AND entry_scan)
#   Inspect compiled graph edges from daily_plan.
#   Assert both position_review and entry_scan are targets.
#   Note: Do NOT use Send() — dual edges are the correct pattern here.

# Test: position_review and entry_scan execute concurrently
#   Both branches write to different state fields (position_reviews vs entry_candidates).
#   Invoke graph with mock nodes that record execution timestamps.
#   Assert both started before either completed (or at minimum, both completed
#   before merge_parallel ran).

# Test: merge_parallel waits for both branches before proceeding
#   Invoke graph. Assert merge_parallel only runs after both
#   execute_exits and entry_scan have completed.


# --- Risk Gate (Critical) ---

# Test: risk_sizing routes to portfolio_review when SafetyGate approves
#   Provide state where all risk decisions pass SafetyGate.validate().
#   Assert portfolio_review node executes.

# Test: risk_sizing routes to END when SafetyGate rejects (with violations logged)
#   Provide state where SafetyGate.validate() returns approved=False.
#   Assert graph terminates at risk gate.
#   Assert errors field contains violation details.

# Test: risk gate edge is mandatory — no path from risk_sizing to execute_entries bypasses it
#   Inspect all paths through the compiled graph from risk_sizing to execute_entries.
#   Assert every path passes through the risk gate conditional edge.
#   This is a structural test on the graph topology, not a runtime test.

# Test: entry_orders is empty when risk gate rejects
#   Invoke graph with risk gate rejection.
#   Assert final state entry_orders == [].


# --- Full Graph ---

# Test: full graph invocation with mock LLM produces valid final state
#   Use MemorySaver checkpointer, mock all LLM calls.
#   Invoke graph with realistic test state.
#   Assert final state has all expected fields populated.
#   Assert decisions list is non-empty (audit trail).
```

## TradingState Reference

The `TradingState` TypedDict (defined in section-04, `src/quantstack/graphs/state.py`) that this graph operates on:

```python
class TradingState(TypedDict):
    # Input
    cycle_number: int
    regime: str
    portfolio_context: dict                  # positions, cash, exposure
    # Pipeline
    daily_plan: str                          # daily planner output
    position_reviews: list[dict]             # per-position HOLD/TRIM/CLOSE
    exit_orders: list[dict]                  # executed exits
    entry_candidates: list[dict]             # scanned candidates
    risk_verdicts: list[dict]                # per-candidate risk sizing
    fund_manager_decisions: list[dict]       # APPROVED/REJECTED per candidate
    options_analysis: list[dict]             # options structures if applicable
    entry_orders: list[dict]                 # executed entries
    reflection: str                          # trade reflector output
    # Accumulation (append-only via operator.add reducer)
    errors: Annotated[list[str], operator.add]
    decisions: Annotated[list[dict], operator.add]
```

**Append-only fields**: `errors` and `decisions` use `operator.add` reducers. Every node that returns these fields must return a `list` (even if empty `[]`). Returning `None` will fail. Nodes that have nothing to append should omit the field from their return dict entirely, or return `[]`.

## Graph Topology

```
START --> safety_check
  |
  v  (conditional: halted? --> END)
daily_plan
  |
  +---> position_review --> execute_exits --+
  |                                         |
  +---> entry_scan -------------------------+
  |                                         |
  v                                         v
                  merge_parallel
                      |
                      v
                  risk_sizing
                      |
              (conditional: SafetyGate)
              /                    \
         PASS                     FAIL --> END (violations logged)
          |
          v
     portfolio_review
          |
          v
     options_analysis
          |
          v
     execute_entries
          |
          v
       reflection
          |
          v
         END
```

## File: `src/quantstack/graphs/trading/graph.py`

This file exports `build_trading_graph()`. The builder:

1. Reads current agent configs from `config_watcher.get_config(agent_name)` for each of the trading agents (daily_planner, position_monitor, trade_debater, risk_analyst, fund_manager, options_analyst, trade_reflector).
2. Creates `BaseChatModel` instances via `get_chat_model(config.llm_tier)` for each agent.
3. Imports node functions from `src/quantstack/graphs/trading/nodes.py`.
4. Constructs the `StateGraph(TradingState)` with all edges.
5. Compiles with the provided checkpointer.

```python
# src/quantstack/graphs/trading/graph.py

"""Trading pipeline graph builder."""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from quantstack.graphs.state import TradingState
from quantstack.graphs.config_watcher import ConfigWatcher
from quantstack.graphs.trading.nodes import (
    safety_check,
    daily_plan,
    position_review,
    execute_exits,
    entry_scan,
    merge_parallel,
    risk_sizing,
    portfolio_review,
    options_analysis,
    execute_entries,
    reflection,
)


def _risk_gate_router(state: TradingState) -> str:
    """Route based on SafetyGate verdict. Pure Python, no LLM."""
    # Inspect risk_verdicts for any rejection
    ...


def _safety_check_router(state: TradingState) -> str:
    """Route to END if system is halted, daily_plan otherwise."""
    ...


def build_trading_graph(
    config_watcher: ConfigWatcher,
    checkpointer: BaseCheckpointSaver,
) -> "CompiledStateGraph":
    """Build the trading pipeline graph.

    Args:
        config_watcher: Provides current agent configs (hot-reloadable).
        checkpointer: Persistence backend (AsyncPostgresSaver in prod,
                      MemorySaver in tests).

    Returns:
        Compiled StateGraph ready for ainvoke().
    """
    ...
```

### Edge Wiring Details

The graph builder must wire edges as follows:

**Sequential edges:**
```python
graph.add_edge("position_review", "execute_exits")
graph.add_edge("execute_exits", "merge_parallel")
graph.add_edge("entry_scan", "merge_parallel")
graph.add_edge("merge_parallel", "risk_sizing")
graph.add_edge("portfolio_review", "options_analysis")
graph.add_edge("options_analysis", "execute_entries")
graph.add_edge("execute_entries", "reflection")
graph.add_edge("reflection", END)
```

**Conditional edges:**
```python
# Safety check: halted systems terminate immediately
graph.add_conditional_edges(
    "safety_check",
    _safety_check_router,
    {"continue": "daily_plan", "halt": END},
)

# Risk gate: mandatory, no bypass path
graph.add_conditional_edges(
    "risk_sizing",
    _risk_gate_router,
    {"approved": "portfolio_review", "rejected": END},
)
```

**Parallel branches (dual edges from daily_plan):**
```python
graph.add_edge("daily_plan", "position_review")
graph.add_edge("daily_plan", "entry_scan")
```

LangGraph executes both `position_review` and `entry_scan` concurrently because they share the same predecessor (`daily_plan`) and write to different state fields. They converge at `merge_parallel` which acts as a join node.

**Important**: Do NOT use `Send()` here. `Send()` is for map-reduce (fanning out the same node over a collection). Dual edges are the correct pattern for running two different nodes concurrently.

## File: `src/quantstack/graphs/trading/nodes.py`

Each node is an async function that accepts `TradingState` and returns a partial state update dict. Node functions are organized by type.

### Node Types and Implementations

**Tool nodes** (deterministic, no LLM):

- `safety_check(state)` -- Calls `get_system_status()`. If halted, sets a flag in state for the conditional router. No retry on failure (critical node -- fail fast).
- `execute_exits(state)` -- Reads `position_reviews` from state, executes exit orders via Alpaca API for positions marked TRIM/CLOSE. Returns `exit_orders` list. Retry once (tool node).
- `merge_parallel(state)` -- No-op join node. Returns empty dict `{}`. Exists solely as the convergence point for the two parallel branches.
- `execute_entries(state)` -- Reads `fund_manager_decisions` (approved entries) and `options_analysis`, executes via Alpaca API. Returns `entry_orders` list. Retry once.

**Agent nodes** (LLM reasoning):

- `daily_plan(state)` -- Agent: daily_planner. LLM generates a trading plan based on regime, calendar, and portfolio state. Returns `daily_plan` string. Retry up to 2 times.
- `position_review(state)` -- Agent: position_monitor. Reviews each open position in `portfolio_context`. Returns `position_reviews` with HOLD/TRIM/CLOSE per position. Retry up to 2 times.
- `entry_scan(state)` -- Agent: trade_debater. Scans for entry candidates based on regime, strategy registry, and signals. Returns `entry_candidates` list. Retry up to 2 times.
- `risk_sizing(state)` -- Tool+Agent hybrid. Computes position sizes via Kelly criterion (tool call), then risk analyst LLM reviews. Calls `SafetyGate.validate()` for each candidate. Returns `risk_verdicts` list. No retry on SafetyGate itself (critical).
- `portfolio_review(state)` -- Agent: fund_manager. Batch review of all proposed entries for correlation, allocation, diversity. Returns `fund_manager_decisions`. Retry up to 2 times.
- `options_analysis(state)` -- Agent: options_analyst. If any candidates are options-eligible, selects structures (spreads, straddles, etc.). Returns `options_analysis` list. Retry up to 2 times.
- `reflection(state)` -- Agent: trade_reflector. Analyzes completed trades and extracts lessons for the knowledge base. Returns `reflection` string. Retry up to 2 times.

```python
# src/quantstack/graphs/trading/nodes.py

"""Node functions for the trading graph.

Each function takes TradingState and returns a partial state update dict.
Agent nodes use LLM reasoning; tool nodes are deterministic.
"""

from quantstack.graphs.state import TradingState
from quantstack.crews.risk.safety_gate import SafetyGate, RiskDecision


async def safety_check(state: TradingState) -> dict:
    """Check system status. If halted, signal router to terminate."""
    ...


async def daily_plan(state: TradingState) -> dict:
    """Generate daily trading plan based on regime and portfolio."""
    ...


async def position_review(state: TradingState) -> dict:
    """Review open positions. Return HOLD/TRIM/CLOSE per position."""
    ...


async def execute_exits(state: TradingState) -> dict:
    """Execute exit orders for positions marked TRIM/CLOSE."""
    ...


async def entry_scan(state: TradingState) -> dict:
    """Scan for entry candidates matching current regime and strategies."""
    ...


async def merge_parallel(state: TradingState) -> dict:
    """No-op join node. Convergence point for parallel branches."""
    return {}


async def risk_sizing(state: TradingState) -> dict:
    """Compute position sizes and validate through SafetyGate.

    Uses Kelly criterion for sizing, then SafetyGate.validate()
    for each candidate. SafetyGate is pure Python -- no LLM.
    """
    ...


async def portfolio_review(state: TradingState) -> dict:
    """Fund manager batch review: correlation, allocation, diversity."""
    ...


async def options_analysis(state: TradingState) -> dict:
    """Select options structures for eligible candidates."""
    ...


async def execute_entries(state: TradingState) -> dict:
    """Execute approved entry orders via Alpaca API."""
    ...


async def reflection(state: TradingState) -> dict:
    """Analyze completed trades. Extract lessons for knowledge base."""
    ...
```

### Node Function Binding Pattern

Agent nodes need access to a `BaseChatModel` and tools. The graph builder binds these via closures or `functools.partial`:

```python
# In build_trading_graph():
planner_config = config_watcher.get_config("daily_planner")
planner_llm = get_chat_model(planner_config.llm_tier)
planner_tools = [TOOL_REGISTRY[t] for t in planner_config.tools]

# Bind LLM and tools to the node function
bound_daily_plan = functools.partial(
    daily_plan, llm=planner_llm, tools=planner_tools, config=planner_config
)
graph.add_node("daily_plan", bound_daily_plan)
```

The actual node function signatures therefore accept extra keyword arguments:

```python
async def daily_plan(
    state: TradingState,
    *,
    llm: BaseChatModel,
    tools: list,
    config: AgentConfig,
) -> dict:
    ...
```

## File: `src/quantstack/graphs/trading/config/agents.yaml`

Agent profiles for the trading graph. Adapted from the existing CrewAI `agents.yaml` format to the new `AgentConfig` schema (section-03).

```yaml
daily_planner:
  role: "Senior Daily Trading Planner"
  goal: "Generate actionable daily trading plan based on regime, calendar, and portfolio"
  backstory: |
    You are a senior trading planner...
  llm_tier: medium
  max_iterations: 10
  timeout_seconds: 120
  tools:
    - get_regime
    - get_portfolio_context
    - get_economic_calendar

position_monitor:
  role: "Position Monitor"
  goal: "Review open positions and recommend HOLD/TRIM/CLOSE actions"
  backstory: |
    You monitor all open positions...
  llm_tier: medium
  max_iterations: 15
  timeout_seconds: 300
  tools:
    - get_portfolio_context
    - signal_brief
    - get_position_details

trade_debater:
  role: "Trade Entry Debater"
  goal: "Find and debate entry candidates for the current regime"
  backstory: |
    You scan the market for entry opportunities...
  llm_tier: heavy
  max_iterations: 20
  timeout_seconds: 600
  tools:
    - signal_brief
    - multi_signal_brief
    - fetch_market_data
    - get_strategy_registry

risk_analyst:
  role: "Risk Analyst"
  goal: "Size positions using Kelly criterion and validate risk limits"
  backstory: |
    You compute optimal position sizes...
  llm_tier: medium
  max_iterations: 10
  timeout_seconds: 120
  tools:
    - compute_kelly_size
    - get_portfolio_context

fund_manager:
  role: "Fund Manager"
  goal: "Review proposed entries for portfolio-level risk: correlation, allocation, diversity"
  backstory: |
    You are the final human-equivalent approval gate...
  llm_tier: heavy
  max_iterations: 10
  timeout_seconds: 180
  tools:
    - get_portfolio_context
    - get_correlation_matrix

options_analyst:
  role: "Options Analyst"
  goal: "Select optimal options structures for eligible candidates"
  backstory: |
    You design options strategies...
  llm_tier: heavy
  max_iterations: 15
  timeout_seconds: 300
  tools:
    - get_options_chain
    - compute_greeks
    - signal_brief

trade_reflector:
  role: "Trade Reflector"
  goal: "Analyze completed trades and extract lessons"
  backstory: |
    You review all trades executed this cycle...
  llm_tier: medium
  max_iterations: 5
  timeout_seconds: 120
  tools:
    - get_trade_history
    - get_portfolio_context
```

## Risk Gate Implementation Detail

The risk gate is the most critical piece. It is a **conditional edge** (not a node that can be skipped). The router function `_risk_gate_router` runs after the `risk_sizing` node completes.

`risk_sizing` calls `SafetyGate.validate()` for each entry candidate and writes the results into `state["risk_verdicts"]`. The router inspects these verdicts:

```python
def _risk_gate_router(state: TradingState) -> str:
    """Route based on SafetyGate verdict.

    If ANY verdict is rejected, the entire batch is rejected.
    This is conservative by design -- partial approval is not supported
    because correlated entries could collectively breach limits.
    """
    verdicts = state.get("risk_verdicts", [])
    if not verdicts:
        # No candidates to trade -- skip to END
        return "rejected"

    for verdict in verdicts:
        if not verdict.get("approved", False):
            return "rejected"

    return "approved"
```

When rejected, the graph routes to END. The `errors` field should already contain violation details (appended by the `risk_sizing` node). No entry orders are executed.

The `SafetyGate` class itself (`src/quantstack/crews/risk/safety_gate.py`) is pure Python with no framework dependency. It checks:
- Daily loss halt threshold (3% default)
- Max position size (15% of equity)
- Minimum ADV liquidity (200k shares)
- Max gross exposure (200%)
- Max options premium at risk (10%)

This class is NOT modified during the migration. It is called from within the `risk_sizing` node function.

## Error Handling Strategy

Each node type has a different retry policy (applied via LangGraph's `retry_policy` parameter):

| Node Type | Retry Count | Rationale |
|-----------|-------------|-----------|
| Agent nodes (LLM calls) | 2 | Transient failures: rate limits, network timeouts |
| Tool nodes (deterministic) | 1 | Permanent failures: bad data, missing DB row |
| Critical nodes (safety_check, risk_sizing SafetyGate) | 0 | Fail fast. Retrying a halt check or safety gate is wrong |

On exhausted retries, the node appends the error to `state["errors"]` and the graph routes to END. The runner inspects the final state's `errors` field to determine cycle status and logs failures to LangFuse.

## Differences from Existing TradingCrew

| Aspect | CrewAI TradingCrew | LangGraph TradingGraph |
|--------|-------------------|----------------------|
| Execution | Strictly sequential (11 tasks) | Parallel branches (position_review and entry_scan run concurrently) |
| Risk gate | Convention (task ordering) | Structural (conditional edge, no bypass path) |
| State passing | Implicit conversational memory (`memory=True`) | Explicit typed state (`TradingState` TypedDict) |
| persist_state task | Explicit task at end | Not needed -- `AsyncPostgresSaver` checkpoints after every node automatically |
| Agent count | 10 agents | 7 agent nodes (executor absorbed into execute_exits/execute_entries tool nodes; earnings_analyst and market_intel folded into entry_scan's tool access) |
| Error handling | CrewAI default (log and continue) | Per-node retry policy with explicit error routing |
| Observability | CrewAI instrumentor | LangFuse callback handler per invocation |

## Shadow-Run Preparation

Before cutting over to the LangGraph trading graph for live/paper trading:

1. Run the new graph on paper trades for at least 2 full trading days alongside the old system.
2. Compare decisions made (entries, exits, risk rejections) between old and new.
3. Verify timing: every graph invocation must complete within the 5-minute trading cycle budget.
4. Verify LangFuse traces show complete coverage (every node, LLM call, and tool invocation traced).
5. Verify risk gate fires correctly on both pass and fail cases during live paper trading.

Shadow-run infrastructure is set up in section-11 (runners) and validated in section-14 (testing).
