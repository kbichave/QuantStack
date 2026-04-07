# Section 14: Budget Tracking & Context Compaction

## Overview

This section adds per-agent token/time/cost budget enforcement and context compaction at Trading Graph merge points. The goal is to prevent runaway agent compute (a single agent consuming unlimited tokens or time) and to reduce context bloat at convergence points where parallel branches merge.

**Current state:** Significant implementation already exists. The codebase has:
- `TokenBudgetTracker` in `src/quantstack/observability/cost_queries.py` — tracks token counts per agent and enforces a `max_tokens` limit.
- `AgentConfig.max_tokens_budget` field in `src/quantstack/graphs/config.py` — wired into YAML config loading.
- Budget checking in `agent_executor.py` (line ~379) — exits gracefully when `budget_tracker.budget_exceeded` is True.
- Deterministic compaction nodes (`compact_parallel`, `compact_pre_execution`) in `src/quantstack/graphs/trading/compaction.py` — already wired into `graph.py` at both merge points.
- Pydantic brief schemas (`ParallelMergeBrief`, `PreExecutionBrief`) in `src/quantstack/graphs/trading/briefs.py`.

**What remains:**
1. Wall-clock time tracking — `TokenBudgetTracker` tracks tokens but not elapsed time. A `max_wall_clock_seconds` budget is needed per agent.
2. Estimated cost tracking — `compute_cost_usd()` exists but is not wired into the per-agent budget tracker.
3. YAML config fields — `max_tokens_budget` exists but `max_wall_clock_seconds` does not. No agents in YAML currently set budget values.
4. Compaction effectiveness validation — the deterministic compaction nodes exist, but there is no measurement of actual context size reduction to confirm the 40%+ target.

**Dependencies:** Phase 2 complete (sections 7-11). Parallelizable with sections 12, 15, 16.

---

## Tests

All tests go in two files. Write these first; implementation follows.

### `tests/graphs/test_budget_tracker.py`

```python
# tests/graphs/test_budget_tracker.py

import time
from unittest.mock import patch

from quantstack.observability.cost_queries import TokenBudgetTracker, compute_cost_usd


# --- Token budget (existing behavior, regression tests) ---

# Test: budget tracker counts tokens per agent per cycle
#   Create a TokenBudgetTracker with max_tokens=10000.
#   Call add_usage(input_tokens=3000, output_tokens=1000) twice.
#   Assert total_tokens == 8000, input_tokens == 6000, output_tokens == 2000.

# Test: agent exits gracefully at node boundary when budget exhausted
#   Create a TokenBudgetTracker with max_tokens=5000.
#   Call add_usage(input_tokens=3000, output_tokens=3000) — total 6000 > 5000.
#   Assert budget_exceeded is True.

# Test: budget overshoot tolerance (exits at next boundary, not mid-generation)
#   Create a TokenBudgetTracker with max_tokens=5000.
#   Call add_usage(input_tokens=2000, output_tokens=2000) — total 4000, not exceeded.
#   Call add_usage(input_tokens=1500, output_tokens=500) — total 6000, now exceeded.
#   Assert budget_exceeded only becomes True after the second add_usage call.
#   (Overshoot is 1000/5000 = 20% — acceptable because we check at boundaries.)

# Test: no budget limit (max_tokens=None) never triggers exceeded
#   Create a TokenBudgetTracker with max_tokens=None.
#   Call add_usage(input_tokens=1_000_000, output_tokens=500_000).
#   Assert budget_exceeded is False.


# --- Wall-clock budget (new) ---

# Test: budget tracker counts wall-clock time per agent
#   Create a TokenBudgetTracker with max_wall_clock_seconds=10.
#   Start the tracker (call start_clock()).
#   Sleep briefly or mock time.monotonic to advance 5 seconds.
#   Assert elapsed_seconds is approximately 5.
#   Assert wall_clock_exceeded is False.

# Test: wall-clock budget exceeded triggers graceful exit
#   Create a TokenBudgetTracker with max_wall_clock_seconds=5.
#   Start the tracker. Mock time.monotonic to advance 6 seconds.
#   Assert wall_clock_exceeded is True.

# Test: wall-clock budget None means no time limit
#   Create a TokenBudgetTracker with max_wall_clock_seconds=None.
#   Assert wall_clock_exceeded is False regardless of elapsed time.

# Test: combined budget — either token OR wall-clock exceeded triggers exit
#   Create tracker with max_tokens=100000 and max_wall_clock_seconds=5.
#   Exceed wall-clock only. Assert budget_exceeded is True (either condition).
#   Create another tracker. Exceed tokens only. Assert budget_exceeded is True.


# --- Cost estimation ---

# Test: compute_cost_usd returns correct value for known model
#   compute_cost_usd(input_tokens=1_000_000, output_tokens=100_000, model="claude-sonnet-4-6")
#   Expected: (1_000_000 * 3.0 + 100_000 * 15.0) / 1_000_000 = 4.5

# Test: compute_cost_usd falls back to default pricing for unknown model
#   compute_cost_usd(input_tokens=1000, output_tokens=1000, model="unknown-model")
#   Should use _DEFAULT_PRICING (3.0, 15.0) — returns (1000*3.0 + 1000*15.0)/1e6

# Test: per-experiment ceiling prevents autoresearch runaway
#   Create tracker with max_tokens=50000 and max_wall_clock_seconds=300 (5 min).
#   Simulate exceeding time. Assert budget_exceeded is True.
#   (This tests that autoresearch experiments respect the per-experiment budget.)
```

### `tests/graphs/test_context_compaction.py`

```python
# tests/graphs/test_context_compaction.py

import json
from unittest.mock import MagicMock

from quantstack.graphs.trading.compaction import compact_parallel, compact_pre_execution
from quantstack.graphs.trading.briefs import ParallelMergeBrief, PreExecutionBrief


# --- compact_parallel ---

# Test: compact_context reduces context size by >= 40%
#   Build a TradingState mock with realistic verbose data:
#     - exit_orders: 3 exits with full reasoning text (~500 chars each)
#     - entry_candidates: 5 entries with full thesis (~800 chars each)
#     - position_reviews: 4 reviews with verbose analysis (~600 chars each)
#     - earnings_analysis: nested dict with multi-paragraph analysis
#     - regime: "trending_up"
#   Compute total size of raw state fields (json.dumps all the above).
#   Call compact_parallel(state).
#   Compute total size of the resulting ParallelMergeBrief (json.dumps).
#   Assert brief_size <= 0.6 * raw_state_size (i.e., >= 40% reduction).

# Test: compact_context preserves key decisions and action items
#   Build state with entry_candidates containing specific symbols.
#   Call compact_parallel(state).
#   Assert all symbols appear in the brief's entries list.

# Test: compact_context preserves risk flags
#   Build state with position_reviews containing action="flag" items.
#   Call compact_parallel(state).
#   Assert risks list is non-empty and contains the flagged items.

# Test: compaction handles empty state gracefully
#   Build state with all fields empty/None.
#   Call compact_parallel(state). Assert returns a valid ParallelMergeBrief
#   with empty lists and compaction_degraded=False.

# Test: compaction handles malformed state (degraded mode)
#   Build state where a field raises an exception on access.
#   Call compact_parallel(state). Assert compaction_degraded=True.


# --- compact_pre_execution ---

# Test: compact_pre_execution separates approved from rejected decisions
#   Build state with fund_manager_decisions: 2 APPROVED, 3 rejected.
#   Call compact_pre_execution(state).
#   Assert brief.approved has 2 items, brief.rejected has 3 items.

# Test: compact_pre_execution preserves options_specs
#   Build state with options_analysis containing 2 specs.
#   Call compact_pre_execution(state).
#   Assert brief.options_specs has 2 items.

# Test: compact_pre_execution aggregates risk_checks by symbol
#   Build state with risk_verdicts for symbols AAPL, MSFT.
#   Call compact_pre_execution(state).
#   Assert brief.risk_checks has keys "AAPL" and "MSFT".

# Test: compaction runs after merge_parallel node (integration)
#   Verify that the graph wiring in build_trading_graph() maps
#   "merge_parallel" to compact_parallel (not the old no-op).

# Test: compaction runs after merge_pre_execution node (integration)
#   Verify that the graph wiring in build_trading_graph() maps
#   "merge_pre_execution" to compact_pre_execution.
```

---

## Implementation Details

### 1. Add wall-clock tracking to `TokenBudgetTracker`

**File:** `src/quantstack/observability/cost_queries.py`

Extend `TokenBudgetTracker` to also track elapsed wall-clock time. Add:

- Constructor parameter: `max_wall_clock_seconds: float | None = None`
- Method: `start_clock()` — records `time.monotonic()` as the start time. Called once at the beginning of agent execution.
- Property: `elapsed_seconds -> float` — returns `time.monotonic() - start_time` (or 0.0 if clock not started).
- Property: `wall_clock_exceeded -> bool` — returns `True` if elapsed exceeds `max_wall_clock_seconds` (and limit is not None).
- Update `budget_exceeded` property to return `True` if *either* token budget or wall-clock budget is exceeded.

Signature changes:

```python
class TokenBudgetTracker:
    def __init__(
        self,
        max_tokens: int | None,
        max_wall_clock_seconds: float | None = None,
    ):
        ...
        self._max_wall_clock = max_wall_clock_seconds
        self._start_time: float | None = None

    def start_clock(self) -> None:
        """Record the start time. Call once per agent invocation."""
        self._start_time = time.monotonic()

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def wall_clock_exceeded(self) -> bool:
        if self._max_wall_clock is None:
            return False
        return self.elapsed_seconds > self._max_wall_clock

    @property
    def budget_exceeded(self) -> bool:
        if self._max_wall_clock is not None and self.wall_clock_exceeded:
            return True
        if self._max_tokens is not None and self.total_tokens > self._max_tokens:
            return True
        return False
```

### 2. Add `max_wall_clock_seconds` to `AgentConfig`

**File:** `src/quantstack/graphs/config.py`

Add a new field to the `AgentConfig` dataclass:

```python
max_wall_clock_seconds: float | None = None
```

Wire it in the `load_agent_configs` function:

```python
max_wall_clock_seconds=fields.get("max_wall_clock_seconds"),
```

### 3. Wire wall-clock budget into agent executor

**File:** `src/quantstack/graphs/agent_executor.py`

Where `TokenBudgetTracker` is instantiated (around line 347), pass the wall-clock config:

```python
budget_tracker = TokenBudgetTracker(
    max_tokens=config.max_tokens_budget,
    max_wall_clock_seconds=config.max_wall_clock_seconds,
)
budget_tracker.start_clock()
```

The existing budget check at line ~379 (`if budget_tracker.budget_exceeded`) already handles both conditions after the property update above — no further changes needed at the check site.

### 4. Add budget fields to agent YAML configs

**Files:**
- `src/quantstack/graphs/trading/config/agents.yaml`
- `src/quantstack/graphs/research/config/agents.yaml`
- `src/quantstack/graphs/supervisor/config/agents.yaml`

Add `max_tokens_budget` and `max_wall_clock_seconds` to each agent. Recommended defaults:

| Agent Category | max_tokens_budget | max_wall_clock_seconds | Rationale |
|---|---|---|---|
| Trading execution agents (executor, exit_manager) | 30,000 | 120 | Tight — these should decide fast |
| Trading analysis agents (entry_scanner, position_monitor) | 80,000 | 300 | Moderate — need tool rounds |
| Trading planning agents (daily_planner, fund_manager) | 100,000 | 300 | Higher — complex reasoning |
| Research agents (quant_researcher, ml_scientist) | 200,000 | 600 | Generous — deep analysis |
| Autoresearch experiments | 50,000 | 300 | Per-experiment ceiling (5 min) |
| Supervisor agents | 50,000 | 180 | Monitoring should be lightweight |

These are initial values. Tune based on Langfuse traces showing actual usage per agent.

### 5. Validate compaction effectiveness (no code changes needed)

The deterministic compaction nodes already exist and are wired in. The tests above validate that the 40%+ context reduction target is met. The key insight is that the existing implementation uses pure Python extraction (not LLM summarization), which is faster, cheaper, and more reliable.

Key files already in place:
- `src/quantstack/graphs/trading/compaction.py` — `compact_parallel()` and `compact_pre_execution()`
- `src/quantstack/graphs/trading/briefs.py` — `ParallelMergeBrief` and `PreExecutionBrief`
- `src/quantstack/graphs/trading/graph.py` (lines 249-250) — wires compaction nodes at merge points

No changes needed unless the 40% reduction target is not met in testing. If it is not met, the briefs schemas in `briefs.py` should be tightened (e.g., truncate `thesis` strings, drop verbose fields from `exits`/`entries` dicts).

---

## File Summary

| File | Action | What Changes |
|------|--------|-------------|
| `src/quantstack/observability/cost_queries.py` | Modify | Add wall-clock tracking to `TokenBudgetTracker` |
| `src/quantstack/graphs/config.py` | Modify | Add `max_wall_clock_seconds` field to `AgentConfig` and loader |
| `src/quantstack/graphs/agent_executor.py` | Modify | Pass `max_wall_clock_seconds` to tracker, call `start_clock()` |
| `src/quantstack/graphs/trading/config/agents.yaml` | Modify | Add `max_tokens_budget` and `max_wall_clock_seconds` per agent |
| `src/quantstack/graphs/research/config/agents.yaml` | Modify | Add budget fields per agent |
| `src/quantstack/graphs/supervisor/config/agents.yaml` | Modify | Add budget fields per agent |
| `tests/graphs/test_budget_tracker.py` | Create | Budget tracker unit tests |
| `tests/graphs/test_context_compaction.py` | Create | Compaction effectiveness tests |

Files that already exist and need no changes (verify only):
- `src/quantstack/graphs/trading/compaction.py` — compaction logic already implemented
- `src/quantstack/graphs/trading/briefs.py` — brief schemas already defined
- `src/quantstack/graphs/trading/graph.py` — compaction nodes already wired at merge points

---

## Verification Checklist

1. `TokenBudgetTracker` correctly tracks both tokens and wall-clock time
2. `budget_exceeded` returns True when either limit is hit
3. Agent executor calls `start_clock()` before the tool-calling loop
4. All agents in YAML have explicit budget values (no implicit unlimited)
5. Compaction nodes produce briefs that are >= 40% smaller than raw state
6. Compaction preserves all key decisions, action items, and risk flags
7. Compaction degrades gracefully (returns degraded brief, not crash) on malformed state
8. Autoresearch experiments (section 13) respect the 5-minute / 50K-token ceiling via the same budget mechanism
