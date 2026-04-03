# Section 13: Risk & Safety

This section validates that the CrewAI-to-LangGraph migration preserves all safety invariants: the risk gate remains mandatory, error handling follows node-type-specific retry policies, paper mode enforcement is framework-independent, and the kill switch terminates graph execution immediately. It also defines the shadow-run acceptance criteria that must pass before live cutover.

**Depends on:** section-08-trading-graph (the trading graph must be built before risk/safety wiring can be verified)

**Blocks:** section-14-testing (final validation requires risk safety to be verified)

---

## Background

QuantStack is an autonomous trading system where LLMs reason about position sizing, entry/exit decisions, and strategy selection. Because LLMs can hallucinate or be prompt-injected, every trade recommendation passes through a **programmatic safety boundary** (`SafetyGate`) before execution. This gate is pure Python -- no LLM involvement, no framework dependency. The migration must not weaken this boundary.

The `SafetyGate` lives at `src/quantstack/crews/risk/safety_gate.py` and exposes:

- `SafetyGateLimits` -- frozen dataclass with outer envelope limits (max position 15%, daily loss halt 3%, min ADV 200k, max gross exposure 200%, max options premium 10%)
- `RiskDecision` -- structured LLM output (symbol, recommended_size_pct, reasoning, confidence)
- `RiskVerdict` -- gate output (approved bool, violations list, violation_rule)
- `SafetyGate.validate(decision, portfolio_context) -> RiskVerdict` -- the core check

The `SafetyGate` class itself requires **zero changes** during migration. It is pure Python with no CrewAI or LangGraph imports. What changes is how it is wired into the orchestration layer: it moves from being called inside a CrewAI task to being a **mandatory conditional edge** in the LangGraph trading graph.

---

## Tests First

All tests go in `tests/unit/test_risk_safety.py` (renamed from `test_crewai_risk_safety.py`). The existing `SafetyGate` unit tests are preserved unchanged. The following new tests verify the migration-specific safety properties.

### 13.1 Error Retry Policy Tests

```python
# File: tests/unit/test_risk_safety.py (append to existing)

# Test: error retry_policy on agent nodes retries up to 2 times
#   - Build a mock agent node that fails twice then succeeds
#   - Verify the node is called exactly 3 times (2 retries + 1 success)
#   - Verify the retry_policy config on agent nodes specifies max_attempts=3

# Test: error retry_policy on tool nodes retries up to 1 time
#   - Build a mock tool node that fails once then succeeds
#   - Verify the node is called exactly 2 times (1 retry + 1 success)
#   - Verify the retry_policy config on tool nodes specifies max_attempts=2

# Test: critical nodes (safety_check, risk_sizing) have no retry -- fail fast
#   - Inspect the compiled trading graph's node configs
#   - Assert safety_check and risk_sizing nodes have retry_policy=None or max_attempts=1
#   - Verify that when safety_check raises, the graph terminates without retry

# Test: node error appends to errors state field
#   - Build a graph with a node that raises an exception
#   - Invoke the graph and inspect final state
#   - Assert the errors field contains a string describing the failure
```

### 13.2 Risk Gate Enforcement Tests

```python
# Test: SafetyGate.validate() called as conditional edge (not as node)
#   - Inspect the compiled trading graph edges
#   - Assert there is a conditional edge after risk_sizing
#   - Assert the condition function calls SafetyGate.validate()
#   - The gate must NOT be a regular node (it must be a routing function)

# Test: no path through trading graph bypasses risk gate
#   - Enumerate all paths from START to execute_entries in the graph
#   - Assert every path passes through the risk_sizing -> conditional edge
#   - This is the critical invariant: topology enforces the gate, not convention

# Test: risk_sizing routes to portfolio_review when SafetyGate approves
#   - Mock SafetyGate.validate() to return RiskVerdict(approved=True)
#   - Invoke the trading graph with valid state
#   - Assert the graph proceeds through portfolio_review -> execute_entries

# Test: risk_sizing routes to END when SafetyGate rejects (with violations logged)
#   - Mock SafetyGate.validate() to return RiskVerdict(approved=False, ...)
#   - Invoke the trading graph
#   - Assert the graph routes to END
#   - Assert the violations appear in the decisions or errors state field

# Test: entry_orders is empty when risk gate rejects
#   - Mock SafetyGate.validate() to return RiskVerdict(approved=False)
#   - Invoke the trading graph
#   - Assert final state entry_orders is empty list
```

### 13.3 Paper Mode Enforcement Tests

```python
# Test: paper mode enforced by execution tools regardless of graph framework
#   - Import the execution tool functions (execute_entries, execute_exits)
#   - Verify they check ALPACA_PAPER and USE_REAL_TRADING env vars
#   - With USE_REAL_TRADING=false, verify orders go to paper endpoint
#   - This test confirms the safety boundary lives in the tool layer,
#     not the orchestration layer, and is therefore framework-independent
```

### 13.4 Kill Switch Tests

```python
# Test: system halt detected by safety_check node -> graph terminates
#   - Mock get_system_status() to return halted=True
#   - Invoke the trading graph
#   - Assert the graph routes from safety_check directly to END
#   - Assert no trading nodes (daily_plan, position_review, etc.) execute
#   - Assert the decisions field logs the halt reason
```

### 13.5 Existing SafetyGate Tests (Preserved)

The existing test classes remain unchanged in the renamed file:

- `TestSafetyGateRejects` -- validates rejection for oversized positions, daily loss breaches, low ADV, excessive gross exposure, excessive options premium
- `TestSafetyGatePasses` -- validates approval for valid recommendations
- `TestRiskDecisionSchema` -- validates data model fields and defaults
- `TestSafetyGateLimits` -- validates default and custom limit values

These tests have zero framework dependency and pass identically before and after migration.

---

## Implementation Details

### 13.1 Error Handling Strategy

LangGraph supports `retry_policy` on individual nodes. Apply policies based on node type across all three graphs:

**Agent nodes** (LLM calls -- domain_selection, hypothesis_generation, daily_plan, position_review, entry_scan, etc.):
- Retry up to 2 times with exponential backoff
- Rationale: LLM calls fail transiently (rate limits, network timeouts, provider errors)
- After retries exhausted: append error string to the `errors` state field and route to END

**Tool nodes** (deterministic -- context_load, signal_validation, execute_exits, execute_entries, etc.):
- Retry once
- Rationale: deterministic failures are usually permanent (bad data, missing DB row, API rejection). A second attempt catches transient network issues without wasting time on permanent failures.
- After retry: append error and continue or route to END depending on criticality

**Critical nodes** (safety_check, risk_sizing):
- No retry (max_attempts=1). Fail fast.
- Rationale: retrying a safety check that returns "halted" is semantically wrong. If the risk gate computation fails, the correct behavior is to halt, not retry and hope the answer changes.

Implementation pattern in each graph builder:

```python
def build_trading_graph(...) -> CompiledStateGraph:
    graph = StateGraph(TradingState)

    # Agent nodes get retry
    graph.add_node("daily_plan", daily_plan_node, retry=RetryPolicy(max_attempts=3))
    graph.add_node("position_review", position_review_node, retry=RetryPolicy(max_attempts=3))

    # Tool nodes get limited retry
    graph.add_node("execute_exits", execute_exits_node, retry=RetryPolicy(max_attempts=2))

    # Critical nodes: no retry
    graph.add_node("safety_check", safety_check_node)  # no retry param
    graph.add_node("risk_sizing", risk_sizing_node)     # no retry param
```

Each graph should have a global error handler that catches unhandled exceptions from any node, appends to the `errors` state field, and routes to END. The runner inspects the final state's `errors` field after each cycle to determine success/failure and log accordingly.

### 13.2 Risk Gate as Conditional Edge

The risk gate is wired as a **conditional edge** in the trading graph, not as a regular node. This is a critical architectural choice: a conditional edge is a pure Python function that determines routing. It cannot be skipped, reordered, or bypassed by graph execution -- the topology enforces it.

Implementation in `src/quantstack/graphs/trading/graph.py`:

```python
def risk_gate_router(state: TradingState) -> str:
    """Mandatory risk gate -- routes based on SafetyGate verdict.

    This function is called as a conditional edge after risk_sizing.
    It validates each candidate through SafetyGate and routes accordingly.
    """
    gate = SafetyGate()
    verdicts = state.get("risk_verdicts", [])

    # If any verdict has approved=False, log violations and route to END
    rejected = [v for v in verdicts if not v.get("approved", False)]
    if rejected:
        # Violations are already in risk_verdicts; the END path logs them
        return "end"

    return "portfolio_review"
```

Wiring in the graph builder:

```python
graph.add_conditional_edges(
    "risk_sizing",
    risk_gate_router,
    {"portfolio_review": "portfolio_review", "end": END},
)
```

There must be **no** `graph.add_edge("risk_sizing", "portfolio_review")` or any other direct edge that would bypass the conditional routing. The only outbound edge from `risk_sizing` is the conditional edge through `risk_gate_router`.

### 13.3 Paper Mode Enforcement

Paper mode is enforced at the **tool layer**, not the orchestration layer. The execution functions check environment variables:

- `ALPACA_PAPER=true` -- uses paper trading endpoint
- `USE_REAL_TRADING=false` -- prevents live order submission

This design is intentional: safety boundaries live in the lowest layer possible, so they apply regardless of which framework (CrewAI, LangGraph, or direct script) calls them. No changes needed during migration -- just verify the behavior with a test.

### 13.4 Kill Switch (System Halt)

The `safety_check` node is the first node in the trading graph after START. It calls `get_system_status()` and checks for halt conditions. If the system is halted, the graph routes directly to END via a conditional edge:

```python
def safety_check_router(state: TradingState) -> str:
    """Route based on system health status."""
    health = state.get("health_status", {})
    if health.get("halted", False):
        return "end"
    return "daily_plan"

graph.add_conditional_edges(
    "safety_check",
    safety_check_router,
    {"daily_plan": "daily_plan", "end": END},
)
```

This replaces the CrewAI task that performed the same check. The behavior is identical but the enforcement is now topological (graph routing) rather than procedural (task code checking a flag and returning early).

### 13.5 Audit Trail

LangFuse callback handler (configured in section-10-observability) automatically traces every node execution, LLM call, and tool invocation. Combined with the custom trace helpers that remain unchanged:

- `trace_provider_failover()` -- logs LLM provider fallback events
- `trace_strategy_lifecycle()` -- logs strategy promotions/retirements
- `trace_safety_boundary_trigger()` -- logs when SafetyGate rejects a trade
- `trace_capital_allocation()` -- logs capital allocation decisions
- `trace_self_healing_event()` -- logs auto-fix events

The audit trail is more complete under LangGraph because every node transition is a traced span (vs. CrewAI where inter-task handoffs were opaque).

When the risk gate rejects a trade, the node should call `trace_safety_boundary_trigger()` to create an explicit LangFuse event with the violation details, symbol, and recommended size. This makes risk rejections searchable in the LangFuse dashboard.

### 13.6 Shadow-Run Acceptance Criteria

Before cutting over to LangGraph for live trading, all of the following must be verified:

1. **All three graphs pass unit tests and integration tests** -- zero test failures in `pytest tests/`
2. **Trading graph shadow-run**: at least 2 full trading days on paper trades using the new LangGraph trading graph. Capture all decisions and compare against what the old CrewAI system would have produced.
3. **Decision quality**: manual review of shadow-run decisions confirms they match or exceed CrewAI quality. Specifically: no trades that violate the regime-strategy matrix, no positions exceeding SafetyGate limits, no missed kill switch events.
4. **Timing budgets**: all graph invocations complete within cycle intervals (trading < 5 min, research < 10 min, supervisor < 5 min). Measured via timing benchmark tests.
5. **LangFuse trace coverage**: every node execution appears as a span, tool invocations are nested under parent nodes, session_id and thread_id are set correctly.
6. **Risk gate regression**: the specific test cases in section 13.2 all pass, confirming no path through the trading graph bypasses the risk gate.

### 13.7 Rollback Plan

If the migration must be reverted:

- Old crew code is preserved in git history
- PostgreSQL checkpoint tables (LangGraph) coexist with existing `crew_checkpoints` table -- no schema conflicts
- pgvector tables coexist with ChromaDB data -- ChromaDB service can be re-added to docker-compose
- Rollback procedure: `git revert` the migration commits, restore the `chromadb` service in docker-compose, restart old runners
- Database schema changes (pgvector extension, LangGraph checkpoint tables) are additive and do not require rollback -- they sit alongside the old schema safely

---

## File Changes Summary

| File | Action |
|------|--------|
| `tests/unit/test_crewai_risk_safety.py` | Rename to `tests/unit/test_risk_safety.py` |
| `tests/unit/test_risk_safety.py` | Add new test classes for retry policy, risk gate enforcement, paper mode, kill switch |
| `src/quantstack/graphs/trading/graph.py` | Wire `risk_gate_router` as conditional edge after `risk_sizing` (created in section-08) |
| `src/quantstack/graphs/trading/graph.py` | Wire `safety_check_router` as conditional edge after `safety_check` (created in section-08) |
| `src/quantstack/graphs/trading/nodes.py` | Ensure `risk_sizing` node populates `risk_verdicts` with SafetyGate output (created in section-08) |
| `src/quantstack/crews/risk/safety_gate.py` | No changes -- pure Python, framework-independent |

---

## Verification Checklist

- [ ] `tests/unit/test_risk_safety.py` exists with all test stubs from sections 13.1-13.4
- [ ] All existing SafetyGate tests pass unchanged after file rename
- [ ] Risk gate conditional edge is the **only** outbound edge from `risk_sizing` in the compiled trading graph
- [ ] Safety check conditional edge routes to END when system is halted
- [ ] Agent nodes have `retry_policy` with `max_attempts=3`
- [ ] Tool nodes have `retry_policy` with `max_attempts=2`
- [ ] Critical nodes (`safety_check`, `risk_sizing`) have no retry policy
- [ ] `trace_safety_boundary_trigger()` is called when risk gate rejects
- [ ] Paper mode enforcement is verified at the tool layer, not the graph layer
- [ ] Shadow-run acceptance criteria are documented and achievable
