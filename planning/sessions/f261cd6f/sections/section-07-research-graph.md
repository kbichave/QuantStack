# Section 07: Research Graph

## Overview

Build the `ResearchGraph` as a LangGraph `StateGraph` with 8 nodes and one conditional edge. This graph replaces the CrewAI research crew's hierarchical process (where `quant_researcher` acted as manager) with an explicit linear pipeline. The "manager" logic was really just task ordering -- LangGraph edges handle this deterministically.

The graph discovers and validates trading strategies through a pipeline: load context, select a research domain, generate a hypothesis, validate signals, backtest, run ML experiments, register the strategy, and update the knowledge base.

## Dependencies

- **section-02-llm-provider**: `get_chat_model(tier)` must be available to create `BaseChatModel` instances for agent nodes.
- **section-03-agent-config**: `AgentConfig` dataclass, YAML loader, and `ConfigWatcher` must be implemented. The research graph's `agents.yaml` is read through this system.
- **section-04-state-schemas**: `ResearchState` TypedDict must be defined in `src/quantstack/graphs/state.py`.
- **section-05-tool-layer**: LLM-facing tools (`@tool` decorated) and node-callable functions must be migrated and available in `TOOL_REGISTRY`.
- **section-09-rag-migration**: The `knowledge_update` node writes to pgvector. The RAG pipeline must be migrated before this node works end-to-end.

## Files to Create / Modify

| File | Action |
|------|--------|
| `src/quantstack/graphs/research/__init__.py` | Create -- export `build_research_graph` |
| `src/quantstack/graphs/research/graph.py` | Create -- graph builder function |
| `src/quantstack/graphs/research/nodes.py` | Create -- all 8 node functions |
| `src/quantstack/graphs/research/config/agents.yaml` | Create -- agent profiles for research domain |
| `tests/unit/test_research_graph.py` | Create -- unit + integration tests |

## Tests (Write First)

All tests go in `tests/unit/test_research_graph.py`. Use `MemorySaver` (LangGraph's in-memory checkpointer) for test isolation. Mock LLM calls -- do not hit real models in unit tests.

```python
# tests/unit/test_research_graph.py

import pytest

# --- Graph structure tests ---

# Test: build_research_graph() returns CompiledStateGraph
#   Call build_research_graph with a mock ConfigWatcher and MemorySaver.
#   Assert the return type is CompiledStateGraph.

# Test: research graph has expected node count (8)
#   Build the graph, inspect its nodes. Expect exactly 8:
#   context_load, domain_selection, hypothesis_generation,
#   signal_validation, backtest_validation, ml_experiment,
#   strategy_registration, knowledge_update.

# Test: research graph START connects to context_load
#   Inspect the graph edges. The START node must connect to context_load.

# Test: signal_validation routes to backtest_validation on pass
#   Build graph. Invoke the conditional router function with a state where
#   validation_result["passed"] is True. Assert it returns "backtest_validation".

# Test: signal_validation routes to END on fail
#   Build graph. Invoke the conditional router function with a state where
#   validation_result["passed"] is False. Assert it returns END.

# Test: all nodes in graph are callable (no missing implementations)
#   Build the graph. Iterate over all nodes and verify each is callable.

# --- Node unit tests ---

# Test: context_load node returns context_summary field
#   Call context_load with minimal ResearchState (cycle_number, regime).
#   Mock get_system_status() and get_regime(). Assert returned dict has
#   "context_summary" as a non-empty string.

# Test: domain_selection node returns selected_domain field
#   Call domain_selection with state containing context_summary.
#   Mock the LLM to return a structured domain choice.
#   Assert returned dict has "selected_domain" in {"swing", "investment", "options"}.

# --- Integration test ---

# Test: research graph invocation with mock LLM produces valid final state
#   Build graph with MemorySaver. Mock all LLM calls and external tool calls.
#   Invoke with a complete initial ResearchState. Assert:
#   - Final state has all expected fields populated
#   - errors list is empty (happy path)
#   - decisions list has entries (audit trail)

# --- Conditional routing test ---

# Test: full graph terminates early when signal_validation fails
#   Mock signal_validation to return validation_result={"passed": False}.
#   Invoke the full graph. Assert the graph reaches END without executing
#   backtest_validation, ml_experiment, strategy_registration, or knowledge_update.

# --- Error accumulation test ---

# Test: errors accumulate across nodes via operator.add reducer
#   Mock a node to append an error string. Assert the final state errors
#   list contains the error and any prior errors are preserved.
```

## Graph Topology

```
START --> context_load --> domain_selection --> hypothesis_generation
      --> signal_validation --> [CONDITIONAL]
                                  |-- passed=True  --> backtest_validation
                                  |-- passed=False --> END (error logged)
      --> ml_experiment --> strategy_registration --> knowledge_update --> END
```

This is a linear pipeline with a single conditional branch after `signal_validation`. If the signal does not pass validation, the graph terminates early. The error is appended to the `errors` accumulation field so the runner can inspect it.

## State Schema (from section-04)

The graph operates on `ResearchState`. Reproduced here for self-containment:

```python
class ResearchState(TypedDict):
    # Input
    cycle_number: int
    regime: str                              # current market regime
    # Pipeline
    context_summary: str                     # loaded context (text summary)
    selected_domain: str                     # "swing", "investment", "options"
    selected_symbols: list[str]
    hypothesis: str                          # generated hypothesis
    validation_result: dict                  # signal validation output
    backtest_id: str                         # reference to backtest results in DB
    ml_experiment_id: str                    # reference to ML experiment in DB
    registered_strategy_id: str              # newly registered strategy
    # Accumulation (append-only via operator.add)
    errors: Annotated[list[str], operator.add]
    decisions: Annotated[list[dict], operator.add]  # audit trail
```

Nodes return partial dicts containing only the fields they update. Append-only fields (`errors`, `decisions`) must return `[]` (empty list) if no entries are added -- never `None`.

## Node Implementations

All node functions go in `src/quantstack/graphs/research/nodes.py`. Each is an async function with signature `async def node_name(state: ResearchState) -> dict`. Agent nodes receive a bound LLM (`BaseChatModel`) via closure or `functools.partial` at graph build time.

### context_load (Tool Node -- Deterministic)

Calls `get_system_status()`, `get_regime()`, loads recent session handoffs and strategy registry state. Returns:
- `context_summary`: A text summary of current system state, regime, portfolio gaps, and recent session findings.
- `decisions`: Log entry recording what context was loaded.

No LLM involved. This is a pure data-gathering node.

### domain_selection (Agent Node)

LLM decides which research domain to pursue (`"swing"`, `"investment"`, or `"options"`) based on context summary, current regime, strategy gaps, and the regime-strategy matrix (trending_up + normal vol favors swing_momentum, ranging + low vol favors mean_reversion, etc.). Returns:
- `selected_domain`: One of the three domain strings.
- `selected_symbols`: List of ticker symbols to research.
- `decisions`: Log entry with reasoning for domain choice.

Bind the LLM from `get_chat_model(agent_config.llm_tier)`. The agent config comes from `agents.yaml` under the `quant_researcher` key.

### hypothesis_generation (Agent Node)

LLM generates a testable hypothesis for the selected domain and symbols. The hypothesis must be specific enough to validate with signal computation (e.g., "AAPL shows bullish divergence on RSI with institutional accumulation, targeting 5% upside over 2 weeks"). Returns:
- `hypothesis`: The hypothesis string.
- `decisions`: Log entry with hypothesis details.

### signal_validation (Tool Node)

Calls signal computation functions (from `tools/functions/`) to validate the hypothesis. Runs technical signals, checks for confluence. Returns:
- `validation_result`: Dict with at minimum `{"passed": bool, "signals": [...], "reason": str}`.
- `decisions`: Log entry with signal validation outcome.

### Conditional Router (after signal_validation)

A plain Python function, not a node. Inspects `state["validation_result"]["passed"]`:
- `True` --> route to `"backtest_validation"`
- `False` --> route to `END`

When routing to END on failure, the `signal_validation` node should have already appended the failure reason to `errors`.

```python
def route_after_validation(state: ResearchState) -> str:
    if state["validation_result"].get("passed", False):
        return "backtest_validation"
    return END
```

### backtest_validation (Tool Node)

Runs a backtest for the validated hypothesis. Stores full results in PostgreSQL (the heavy data stays in the DB, not in graph state). Returns:
- `backtest_id`: Reference ID for the stored backtest results.
- `decisions`: Log entry with backtest summary metrics (win rate, Sharpe, max drawdown).

### ml_experiment (Agent Node)

LLM designs an ML experiment informed by the hypothesis and backtest results. Calls ML tools to train/evaluate. Stores experiment artifacts in DB. Returns:
- `ml_experiment_id`: Reference ID for the stored experiment.
- `decisions`: Log entry with experiment design and results summary.

### strategy_registration (Tool Node)

Registers the validated strategy in the database strategy registry. Sets initial status to `"paper_ready"` (never `"live"` directly -- strategies must prove themselves in paper trading first). Returns:
- `registered_strategy_id`: The ID of the newly registered strategy.
- `decisions`: Log entry recording registration.

### knowledge_update (Tool Node)

Updates the pgvector knowledge base with research findings -- the hypothesis, validation results, backtest summary, and experiment summary. This enables future research cycles to build on past findings via RAG retrieval. Returns:
- `decisions`: Log entry confirming knowledge base update.

## Graph Builder

The graph builder goes in `src/quantstack/graphs/research/graph.py`.

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

def build_research_graph(
    config_watcher: ConfigWatcher,
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    """Build the research pipeline graph.

    Reads agent configs from config_watcher, creates ChatModel instances
    per agent tier, binds them to agent nodes via closures, defines the
    8-node topology with one conditional edge, and compiles with the
    checkpointer for state persistence.
    """
```

Builder steps:
1. Read agent configs from `config_watcher` (keys: `quant_researcher`, `ml_scientist`, etc.).
2. Create `BaseChatModel` instances via `get_chat_model(config.llm_tier)` for each agent.
3. Create node functions with bound LLMs (use `functools.partial` or closures).
4. Instantiate `StateGraph(ResearchState)`.
5. Add all 8 nodes.
6. Add edges: `START -> context_load -> domain_selection -> hypothesis_generation -> signal_validation`.
7. Add conditional edge: `signal_validation -> route_after_validation -> {backtest_validation, END}`.
8. Add edges: `backtest_validation -> ml_experiment -> strategy_registration -> knowledge_update -> END`.
9. Compile with checkpointer: `graph.compile(checkpointer=checkpointer)`.
10. Return compiled graph.

## Agent Configuration (YAML)

Create `src/quantstack/graphs/research/config/agents.yaml`:

```yaml
quant_researcher:
  role: "Senior Quantitative Researcher"
  goal: "Discover and validate alpha-generating strategies across swing, investment, and options domains"
  backstory: |
    You are a senior quant researcher at an autonomous trading firm.
    You analyze market regimes, identify strategy gaps, and generate
    testable hypotheses backed by technical signals and fundamentals.
  llm_tier: heavy
  max_iterations: 20
  timeout_seconds: 600
  tools:
    - signal_brief
    - multi_signal_brief
    - fetch_market_data
    - compute_features

ml_scientist:
  role: "Machine Learning Scientist"
  goal: "Design and execute ML experiments to validate trading hypotheses"
  backstory: |
    You design ML experiments for trading strategy validation.
    You select appropriate models, features, and evaluation metrics
    based on the hypothesis and available data.
  llm_tier: heavy
  max_iterations: 15
  timeout_seconds: 600
  tools:
    - fetch_market_data
    - compute_features
    - train_model
    - evaluate_model
```

Tool names must match keys in `TOOL_REGISTRY` (from section-05).

## Error Handling

Apply `retry_policy` based on node type when adding nodes to the graph:

- **Agent nodes** (`domain_selection`, `hypothesis_generation`, `ml_experiment`): Retry up to 2 times with exponential backoff. LLM calls fail transiently (rate limits, timeouts).
- **Tool nodes** (`context_load`, `signal_validation`, `backtest_validation`, `strategy_registration`, `knowledge_update`): Retry once. Deterministic failures are usually permanent.

When retries are exhausted, the node appends the error to the `errors` state field. The graph should have fallback routing so that a failed non-critical node still reaches END gracefully rather than crashing the runner.

The runner inspects `final_state["errors"]` after `graph.ainvoke()` completes to determine cycle health and log accordingly to LangFuse.

## Regression Baseline Capture

Before cutting over from CrewAI, capture 3-5 real research cycle I/O pairs from the current system:

1. Record the input state at cycle start (regime, cycle number, market conditions).
2. Record the output state at cycle end (selected domain, hypothesis, validation result, registered strategy ID).
3. After building the LangGraph research graph, replay the same inputs.
4. Compare outputs. Document any differences and whether they represent improvement or regression.

This is especially important because the research graph makes autonomous decisions about capital allocation strategies. Regression verification is not optional for a system that discovers trading strategies.

## Implementation Checklist

1. Write all tests in `tests/unit/test_research_graph.py` (tests above).
2. Create `src/quantstack/graphs/research/config/agents.yaml` with agent profiles.
3. Implement node functions in `src/quantstack/graphs/research/nodes.py` (8 functions + 1 router).
4. Implement `build_research_graph()` in `src/quantstack/graphs/research/graph.py`.
5. Export from `src/quantstack/graphs/research/__init__.py`.
6. Run tests. All structure tests and mock-LLM integration tests must pass.
7. Capture regression baselines from current CrewAI research crew (3-5 cycles).
8. Replay baselines through new graph and document comparison.
