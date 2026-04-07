# Research Findings — Phase 4: Agent Architecture & Coordination

## Part 1: Codebase Analysis

### 1. Trading Graph Structure

**File:** `src/quantstack/graphs/trading/graph.py`

- **16-node StateGraph** with 2 parallel branches and deterministic risk gating
- **Parallel branches:**
  - `position_review` -> `execute_exits`
  - `entry_scan` -> `earnings_analysis` (conditional) -> `merge_parallel`
- **Join node:** `merge_parallel()` (line ~439-441) is a **no-op convergence point**
- **Second join:** `merge_pre_execution()` (line ~444-446) merges `portfolio_review` + `analyze_options`

**Flow:**
1. START -> data_refresh -> safety_check -> [router]
2. If halted: END. If continue: market_intel -> plan_day
3. plan_day forks to parallel branches
4. Both converge at merge_parallel -> risk_sizing
5. risk_sizing routes to portfolio_construction (approved) or END (rejected)
6. portfolio_construction forks again (portfolio_review + analyze_options)
7. Reconverge at merge_pre_execution -> execute_entries -> reflect -> END

**Routers:**
- `_safety_check_router()` (lines 48-59): Checks `decisions` for halt flag, checks `errors` list
- `_earnings_router()` (lines 62-67): Routes based on `earnings_symbols` presence
- `_risk_gate_router()` (lines 70-80): Routes based on `alpha_signals` presence

**Retry Policy:** Most LLM nodes `RetryPolicy(max_attempts=3)` or `max_attempts=2`. Safety_check: no retry (fail fast).

**GAP:** `merge_parallel` and `merge_pre_execution` do NOT aggregate errors or validate state consistency. They are pure pass-through nodes.

---

### 2. State Management

**File:** `src/quantstack/graphs/state.py`

**TradingState (lines 54-91):** Uses `TypedDict` (NOT Pydantic). Key fields:
- Append-only: `errors: Annotated[list[str], operator.add]`, `decisions: Annotated[list[dict], operator.add]`
- Non-accumulating: `daily_plan: str`, `portfolio_context: dict`, `market_context: dict`, etc.

**ResearchState** (lines 17-43): Similar, includes `hypothesis_confidence`, `hypothesis_attempts`.
**SupervisorState** (lines 93-106): Simpler, includes `health_status`, `diagnosed_issues`.

**GAP:** No type validation on state create/update. No `extra="forbid"` — typos like `{"daly_plan": "..."}` silently accepted. No pre/post condition checks on node execution.

---

### 3. Error Handling in Graphs

**Safety check router** (lines 48-59) checks for halt flag and safety_check errors. This is the ONLY error-gating logic.

**GAPs:**
- Merge nodes do NOT check error count before proceeding
- No execution gate — errors accumulate in `errors` list but never trigger halt
- No circuit breaker per node
- If position_review has 5 errors, entry_scan still proceeds
- Error accumulation is one-way (never cleared within a cycle)

---

### 4. Agent Executor & Tool Binding

**Files:** `src/quantstack/graphs/tool_binding.py`, `src/quantstack/graphs/agent_executor.py`, `src/quantstack/tools/registry.py`

**Tool binding** (tool_binding.py lines 26-68): Three paths — Anthropic defer_loading, bigtool, or full loading.

**Agent config** (agents.yaml): Per-agent `tools:` list and optional `always_loaded_tools:` subset. Example: fund_manager gets 7 tools, 3 always loaded.

**Tool resolution:** `get_tools_for_agent()` (registry.py lines 320-335) — raises KeyError if tool not found. **No access control** — any agent can access any tool in TOOL_REGISTRY. Tool availability is config-only, not enforced at binding layer.

---

### 5. Event Bus

**File:** `src/quantstack/coordination/event_bus.py` (lines 160-253)

**Cursor update:** DELETE + INSERT (NOT atomic). Race condition window between operations. If process crashes between DELETE and INSERT, cursor is lost. Next poll restarts from beginning of 7-day event log.

**Event pruning:** TTL of 7 days, pruning on every `publish()`. Handles missing cursor gracefully.

**GAP:** No upsert. Could use `INSERT ... ON CONFLICT DO UPDATE`.

---

### 6. JSON Parsing / Error Handling

**File:** `src/quantstack/graphs/agent_executor.py` (lines 474-522)

`parse_json_response()`: Tries direct parse, then regex extraction of `{}` or `[]`. On failure: logs debug message (200 char truncated), returns fallback (empty dict or list).

**GAPs:**
- No dead letter queue — failed responses silently dropped
- No raw output logging on failure
- No tracking of parse failures per node per cycle
- Fallback values mask real failures (e.g., `market_context: {}` silently accepted)

---

### 7. Message/Conversation Management

**File:** `src/quantstack/graphs/agent_executor.py` (lines 32-143)

Constants: `MAX_TOOL_ROUNDS = 10`, `MAX_TOOL_RESULT_CHARS = 4000`, `MAX_MESSAGE_CHARS = 150_000` (~37k tokens).

Pruning: FIFO — oldest tool round pairs dropped first. Keeps system + user messages (first 2).

**GAPs:** No priority tagging. No distinction between critical (risk data) and verbose (market intel). Oldest rounds dropped regardless of importance.

---

### 8. Risk Gate

**File:** `src/quantstack/execution/risk_gate.py` (lines 1-982)

**Existing checks in `check()`:**
1. Daily halt (persisted to sentinel file)
2. Holding period
3. Trading window (instrument type + DTE)
4. Restricted symbols
5. Liquidity (ADV >= 500k)
6. Participation rate (<= 1% of ADV)
7. Execution quality scalar
8. Macro stress scalar
9. Options: DTE bounds, premium-at-risk
10. Equity: per-symbol position size, gross exposure

**Correlation check:** Only in `monitor()` (continuous), NOT pre-trade. 30-day rolling correlation, alerts > 0.80. Requires 20+ days common returns.

**Regime flip detection:** Also in `monitor()` only. Checks `entry_regimes` vs `current_regimes`. CRITICAL alert for opposite direction, WARNING for lateral.

**GAPs:**
- No pre-trade correlation check with existing positions
- No daily notional deployment cap
- No sector concentration check at all (no sector map in check())
- Regime-at-entry not stored in MonitoredPosition

---

### 9. Execution Monitor

**File:** `src/quantstack/execution/execution_monitor.py`

`MonitoredPosition` dataclass: tracks symbol, side, quantity, entry_price, stops, targets, trailing ATR. Exit rules evaluated in priority order: kill switch > hard stop > take profit > trailing stop > time stop > intraday flatten.

**GAP:** No explicit `regime_at_entry` field. Would need to be added for regime flip detection.

---

### 10. Testing Setup

**Framework:** pytest with AsyncMock, MagicMock. Tests in `tests/unit/`.

**Key test files:** test_trading_graph.py, test_research_graph.py, test_agent_executor.py, test_agent_config.py.

**Patterns:** ConfigWatcher fixtures with tmp_path, mock ChatModel factories, graph structure validation.

**GAPs:** No integration tests for parallel branch execution, no tests for error accumulation across branches, no event bus tests.

---

### 11. Circuit Breaker Patterns

**File:** `src/quantstack/execution/strategy_breaker.py`

**Strategy-level breaker exists:** ACTIVE -> SCALED (50% size) -> TRIPPED (halted). Thresholds: 3 consecutive losses or 5% drawdown trips, 2 losses or 3% scales down. 24h cooldown.

**GAP:** This is strategy-level ONLY. No node-level failure tracking across cycles. No node-level backoff logic.

---

## Part 2: Web Research — Best Practices (2026)

### LangGraph Parallel Branch Conflict Resolution

**State merging uses per-key reducers:**
- No reducer = last-write-wins (dangerous in parallel)
- `Annotated[list[T], operator.add]` = accumulates from all branches (safe)
- `add_messages` reducer = ID-based deduplication for message lists
- Custom reducers supported: `(existing_value, new_value) -> merged_value`

**Key patterns:**
- `Send` primitive for dynamic fan-out (unknown number of parallel tasks)
- `Overwrite` type to explicitly bypass reducers (document why)
- Custom priority-merge reducers for conflict resolution at fan-in points

**Recommendation:** Audit every state key written by parallel branches. Any key without a reducer uses last-write-wins, which means silent data loss.

---

### Circuit Breaker Patterns for LLM Agent Nodes

**Three-state model:** Closed (normal) -> Open (all calls fail fast) -> Half-Open (probe requests).

**LLM-specific failure types and handling:**
| Failure | Breaker Behavior |
|---------|-----------------|
| Rate limit (429) | Trip immediately, respect Retry-After |
| Token limit | Don't trip — route to message pruning |
| Model timeout | Count toward threshold |
| Malformed output | Count separately — signals prompt degradation |
| Provider outage | Trip immediately, switch to fallback model |

**Cross-cycle persistence:** Store breaker state in PostgreSQL, not in-memory. LangGraph's `RetryPolicy` handles within-task retries but does NOT persist across graph invocations.

**Safe defaults:** Return typed sentinel values (not None). Log circuit break event with context. Allow graph to continue — downstream nodes handle empty/low-confidence inputs. Emit Langfuse metric.

**Recommended schema:**
```sql
CREATE TABLE circuit_breaker_state (
    breaker_key TEXT PRIMARY KEY,  -- e.g., "trading/sonnet"
    state TEXT DEFAULT 'closed',
    failure_count INT DEFAULT 0,
    last_failure_at TIMESTAMPTZ,
    opened_at TIMESTAMPTZ,
    cooldown_seconds INT DEFAULT 30
);
```

---

### Pydantic State Validation in LangGraph

**LangGraph officially supports Pydantic BaseModel as state.** Less performant than TypedDict but negligible for agent systems (microseconds vs seconds for LLM calls).

**Key features:**
- `ConfigDict(extra="forbid")` catches typos in state keys at write time
- `field_validator` enforces domain invariants (e.g., position size bounds)
- `model_validator(mode="after")` for cross-field invariants
- Separate input/output schemas constrain graph boundaries
- Reducers (`Annotated`) work identically with Pydantic and TypedDict

**Recommendation:** Use Pydantic for main graph state with `extra="forbid"`. Keep TypedDict only if hot-loop performance matters (it doesn't for agent nodes).

---

### Dead Letter Queue + Priority-Based Message Pruning

**DLQ schema for agent outputs:**
```sql
CREATE TABLE agent_dlq (
    id SERIAL PRIMARY KEY,
    agent_name TEXT NOT NULL,
    graph_name TEXT NOT NULL,
    run_id TEXT NOT NULL,
    input_state JSONB,
    raw_output TEXT,
    error_type TEXT,  -- parse_error, validation_error, timeout, business_rule
    error_detail TEXT,
    prompt_hash TEXT,
    model_used TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    resolution TEXT
);
```

**DLQ rate as signal quality indicator:**
| Rate | Signal | Action |
|------|--------|--------|
| > 5% | Prompt degrading | Flag for review |
| > 20% | Critical failure | Circuit-break agent |
| Spike across all agents | Provider issue | Check model status |

**Priority tiers for message pruning:**
| Priority | Content | Rule |
|----------|---------|------|
| P0 (never prune) | Risk gate output, kill switch, position state | Always retained |
| P1 (summarize) | Signal briefs, trade decisions, regime | Summarize if > N tokens |
| P2 (prune first) | Raw analysis, verbose tool outputs | Drop at token limit |
| P3 (ephemeral) | Debug logs, trace metadata | Never in LLM context |

**LangGraph-specific compaction:** `RemoveMessage` + `add_messages` reducer enables rolling-window compaction with summarization.

---

## Cross-Cutting Synthesis

These patterns reinforce each other:
1. **Pydantic validation failures feed the DLQ** — unparseable output goes to DLQ instead of crashing
2. **Circuit breakers protect against DLQ growth** — if agent produces 50% bad output, break it
3. **Parallel branch reducers determine what reaches the DLQ** — custom reducers route malformed entries
4. **Message pruning prevents token-limit failures** that would otherwise trip breakers

**Recommended implementation order:**
1. Pydantic state with `extra="forbid"` (immediate safety, low effort)
2. Custom reducers for parallel branch conflict resolution (prevents silent data loss)
3. DLQ table + monitoring (observability before automation)
4. Circuit breakers in llm_routing.py (requires monitoring to tune thresholds)
5. Message compaction at fan-in points (optimization after correctness)
