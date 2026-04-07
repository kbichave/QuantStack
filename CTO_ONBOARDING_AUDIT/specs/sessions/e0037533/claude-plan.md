# Implementation Plan — Phase 4: Agent Architecture & Coordination

## Overview

QuantStack is an autonomous trading system with 3 LangGraph StateGraphs (Research, Trading, Supervisor) orchestrating 21 agents. The agents are well-specialized but don't coordinate safely: parallel branches can issue conflicting orders for the same symbol, errors accumulate without halting execution, state has no schema validation, and the risk gate lacks pre-trade portfolio-level checks.

This plan fixes 12 coordination and safety issues. All items deploy as a single atomic release. The guiding principle throughout: **anything that feeds risk calculations or manages existing exposure is blocking; anything that generates new opportunities can fail safely.**

---

## Section 1: Pydantic State Schema Migration

### What and Why

The 4 graph states (`TradingState`, `ResearchState`, `SupervisorState`, and `SymbolValidationState`) are currently `TypedDict` in `src/quantstack/graphs/state.py`. Nodes return plain dicts merged via `operator.add` or last-write-wins. A typo like `{"daly_plan": "..."}` is silently accepted — the real field stays stale with no error. This is the root cause of multiple hard-to-diagnose bugs.

Migrate all 4 states to Pydantic `BaseModel` with `ConfigDict(extra="forbid")`. This catches typos, type mismatches, and missing fields at merge time rather than downstream when a stale value causes a bad trade.

### Prerequisite: State Key Audit

Before writing any Pydantic models, run a dynamic audit to discover ghost fields. The `_risk_gate_router` references `alpha_signals` which does not appear in `TradingState` — this proves undeclared keys exist in the codebase. To find them all:

1. Instrument each graph to log every key returned by every node across 10+ cycles of paper trading
2. Compare logged keys against the TypedDict definitions
3. For each mismatch: either add the field to the Pydantic model or update the node to stop writing it
4. This audit prevents the Pydantic migration from being a "discover bugs one by one in production" exercise

### How

**State models** in `src/quantstack/graphs/state.py`:
- Convert each `TypedDict` to `BaseModel` with `extra="forbid"` (all 4 state classes including `SymbolValidationState`)
- Preserve `Annotated[list[T], operator.add]` reducers — LangGraph extracts these identically from Pydantic
- Add `field_validator` for domain invariants (e.g., `cycle_number >= 0`, `vol_state in {"low", "normal", "high", "extreme"}`)
- Add `model_validator(mode="after")` for cross-field invariants (e.g., `exit_orders` requires `position_reviews` to be non-empty)
- Define separate `input=` and `output=` schemas per graph to constrain boundaries

**Node output models**: Each node gets its own Pydantic output model. For example, `DataRefreshOutput`, `SafetyCheckOutput`, `PlanDayOutput`. These models define exactly which state fields the node is allowed to update. The node function returns an instance of its output model, which Pydantic validates before the state merge.

Place node output models in a new file `src/quantstack/graphs/trading/models.py` (and similarly for research/supervisor). Each model is a subset of the parent graph state — only the fields that node writes.

**Migration approach**: All 4 state classes at once. No half-migrated state. Update every node return to use its typed output model. Run the full test suite after migration — any existing test that constructs state dicts will need updating to match the Pydantic models.

### Key decisions

- `extra="forbid"` is non-negotiable — this is the primary safety mechanism
- Node output models add boilerplate but prevent the class of bugs where a node writes to a field it shouldn't
- Performance is negligible — microseconds of Pydantic validation vs seconds of LLM calls per node

---

## Section 2: Error Blocking & Node Classification

### What and Why

Currently every node catches exceptions and appends to `errors: Annotated[list[str], operator.add]`. After accumulating 5+ errors, the pipeline still reaches `execute_entries`. The only halt point is `safety_check_router`.

### How

**Node classification** — add a `node_classification` dict in the graph definition mapping each node name to `"blocking"` or `"non_blocking"`:

Blocking nodes (failure = potential unintended money loss):
- `data_refresh`, `safety_check`, `position_review`, `risk_sizing`, `execute_exits`

Non-blocking nodes (failure = missed opportunity):
- `plan_day`, `entry_scan`, `market_intel`, `portfolio_construction`, `execute_entries`, `resolve_symbol_conflicts`, `reflect`, `trade_reflector`, `earnings_analysis`, `analyze_options`, `portfolio_review`

**Error gating logic**: Add an `execution_gate` check before `execute_entries` and before `risk_sizing`:
1. Count errors from blocking nodes in current cycle
2. If any blocking node has errored → halt pipeline, log reason, end cycle
3. If total error count > 2 (from any source) → halt pipeline as a safety net
4. Non-blocking node failures → use safe defaults (defined per node in its output model as class method `safe_default()`)

**Implementation location**: The gate check runs as a conditional edge function (like `_safety_check_router`), not as a separate node. This keeps it deterministic and fast. The gate only needs to inspect the `errors` list in state — both caught exceptions and returned error states end up in the same accumulating list, so conditional edge access to state is sufficient.

**Tradeoff note**: `execute_exits` is classified as blocking. If it fails, the gate halts the entire pipeline including new entries. This is intentional — failing to close exposure is more dangerous than missing an entry opportunity. A transient `execute_exits` failure blocks entries for that cycle, which is the correct conservative behavior.

---

## Section 3: Race Condition Fix (Parallel Branch Conflict Resolution)

### What and Why

`position_review` and `entry_scan` run as parallel branches from `plan_day`. Both write to accumulating lists (`exit_orders` and `entry_candidates`). The lists don't conflict at the reducer level (both use `operator.add`), but they can contain the same symbol — meaning the system simultaneously wants to exit AND enter a position in the same symbol within the same cycle.

### How

Add a new node `resolve_symbol_conflicts` between `merge_parallel` and `risk_sizing` in the trading graph.

**Logic:**
1. Extract symbols from `exit_orders` and `entry_candidates`
2. Find intersection (symbols appearing in both)
3. For each conflicting symbol:
   - Remove from `entry_candidates` (exits take priority — risk-off bias)
   - Log conflict event: symbol, exit reasoning, entry reasoning, resolution
4. Return updated `entry_candidates` (exits untouched)

**Safe default on node failure**: Drop ALL entries that have any symbol overlap. This is more conservative than needed but guarantees no conflicting orders.

**Graph wiring**: `merge_parallel` → `resolve_symbol_conflicts` → (execution_gate check) → `risk_sizing`

---

## Section 4: Node Circuit Breaker

### What and Why

If `daily_planner` fails 5 consecutive cycles, the graph still calls it on cycle 6. No backoff, no circuit breaker, no fallback. The existing `StrategyBreaker` (in `src/quantstack/execution/strategy_breaker.py`) handles strategy-level P&L breakers — different concern from node health.

### How

**Decorator pattern**: `@circuit_breaker(threshold=3, alert_threshold=5)` applied to node functions.

**State storage**: New `circuit_breaker_state` table in PostgreSQL:

```
circuit_breaker_state
├── breaker_key TEXT PRIMARY KEY  (e.g., "trading/data_refresh")
├── state TEXT DEFAULT 'closed'   (closed / open / half_open)
├── failure_count INT DEFAULT 0
├── last_failure_at TIMESTAMPTZ
├── opened_at TIMESTAMPTZ
├── cooldown_seconds INT DEFAULT 300
└── last_success_at TIMESTAMPTZ
```

**Three-state model:**
- **Closed** (normal): failures counted. At `threshold` consecutive failures → Open.
- **Open**: node skipped, typed safe default returned. After `cooldown_seconds` (default 300s = 5 minutes, configurable per node) → Half-Open. The cooldown must exceed the cycle interval (60-300s) to ensure the breaker actually skips at least one full cycle.
- **Half-Open**: one probe request allowed. Success → Closed (reset counter). Failure → Open.

**Concurrency safety**: Use atomic increment for failure counts: `UPDATE circuit_breaker_state SET failure_count = failure_count + 1 WHERE breaker_key = ? RETURNING failure_count, state`. This prevents read-modify-write races when multiple graph cycles overlap.

**LLM-specific failure type discrimination:**
- Rate limit (429): trip immediately, respect Retry-After header as minimum cooldown
- Token limit exceeded: don't trip — route to message pruning (data problem, not service problem)
- Parse failures: count separately — high rate signals prompt degradation, not service failure
- Provider outage (5xx, connection error): trip immediately

**Safe defaults**: Each node's output model defines a `safe_default()` class method returning a typed neutral response. For example, `PlanDayOutput.safe_default()` returns neutral bias / no new trades. For blocking nodes, the safe default includes setting an error flag that the execution gate detects.

**Alert at threshold=5**: Emit Langfuse event + outbound notification. If 5 consecutive, halt the graph for that cycle.

**Implementation**: New file `src/quantstack/graphs/circuit_breaker.py` containing the decorator, DB operations, and state management. The decorator wraps the async node function, checks breaker state before invocation, and updates state after.

---

## Section 5: Tool Access Control

### What and Why

No access control exists. Any agent can call any tool registered in `TOOL_REGISTRY`. The only constraint is which tools are listed in an agent's `agents.yaml` config — but nothing enforces this at runtime. A misconfigured `agents.yaml` could let `hypothesis_critic` call `execute_order`.

### How

**Configuration**: Add `blocked_tools` field per graph (not per agent) in each graph's `agents.yaml`:

- Research graph: blocks `execute_order`, `cancel_order`, `activate_kill_switch`
- Trading graph: blocks `register_strategy`, `train_model`
- Supervisor graph: blocks all execution tools (read-only)

**Enforcement**: ~5-line guard in `src/quantstack/graphs/agent_executor.py` at the tool invocation point. Before calling any tool, check if the tool name appears in the current graph's `blocked_tools` list. If blocked:
1. Return an error message to the agent (not an exception — let the agent handle it)
2. Log as a security event via Langfuse with: agent name, tool name, graph name, timestamp
3. Do NOT circuit-break the agent — the violation is a config bug, not an agent failure

**Loading blocked_tools**: The `ConfigWatcher` already hot-reloads `agents.yaml`. Add `blocked_tools` as a graph-level key (not per-agent) since tool access is a graph boundary, not an agent boundary.

---

## Section 6: Event Bus Cursor Atomicity

### What and Why

The event bus (`src/quantstack/coordination/event_bus.py`) updates cursors via DELETE + INSERT. A crash between these operations loses the cursor, causing duplicate event processing on next poll.

### How

Replace the DELETE + INSERT pair with a single PostgreSQL upsert. The codebase uses `?` placeholders (not `$1` libpq style) via a DB abstraction layer — match that convention:

```sql
INSERT INTO loop_cursors (consumer_id, last_event_id, last_polled_at)
VALUES (?, ?, ?)
ON CONFLICT (consumer_id) DO UPDATE SET
  last_event_id = EXCLUDED.last_event_id,
  last_polled_at = EXCLUDED.last_polled_at
```

**Verify before implementing**: Confirm that the `PgConnection` wrapper supports `ON CONFLICT` pass-through. Some lightweight DB abstractions don't relay all PostgreSQL-specific SQL features.

This is atomic — either the cursor updates or it doesn't. No intermediate state.

**Verify**: `loop_cursors` table needs a UNIQUE constraint on `consumer_id`. Check if it exists; add if missing via migration.

---

## Section 7: Dead Letter Queue

### What and Why

`parse_json_response()` in `src/quantstack/graphs/agent_executor.py` fails silently — returns `{}` fallback, logs a 200-char debug message, and discards the raw LLM output. No record of what went wrong, how often it happens per agent, or what the agent actually said.

### How

**New table** `agent_dlq` in PostgreSQL:

```
agent_dlq
├── id SERIAL PRIMARY KEY
├── agent_name TEXT NOT NULL
├── graph_name TEXT NOT NULL
├── run_id TEXT NOT NULL
├── input_summary TEXT          (truncated input state for context)
├── raw_output TEXT             (the unparsed LLM output)
├── error_type TEXT             (parse_error, validation_error, timeout, business_rule)
├── error_detail TEXT
├── prompt_hash TEXT            (hash of prompt — cluster failures by prompt variant)
├── model_used TEXT
├── created_at TIMESTAMPTZ DEFAULT NOW()
├── resolved_at TIMESTAMPTZ
└── resolution TEXT             (manual_override, prompt_fixed, discarded)
```

**Integration point**: Modify `parse_json_response()` (or its callers) to accept agent context. When parsing fails, write to DLQ before returning fallback. The function signature gains optional kwargs: `agent_name`, `graph_name`, `run_id`, `model_used`.

**Monitoring** (Langfuse only, no self-healing):
- Track DLQ rate as Langfuse metric with 24h rolling window per agent
- Two alert thresholds: warn at 5%, critical at 10%
- Critical alert fires outbound notification (Slack webhook or email)
- DLQ metrics surfaced on the system dashboard alongside existing health metrics
- Alert payload includes: agent/node name, sample of failed messages, rate trend

**Why no self-healing integration**: The self-healing loop (`record_tool_error` → `bug_fix` task → AutoResearchClaw) was designed for deterministic tool errors with known fix patterns. Prompt degradation is a symptom with multiple causes (regime change, data drift, bad deployment, prompt rot). Auto-patching prompts that influence capital allocation without human sign-off crosses the wrong automation boundary. Revisit after 60+ days of paper trading data.

---

## Section 8: Priority-Based Message Pruning

### What and Why

When conversation exceeds 150K chars (~37k tokens), the `_prune_messages()` function drops oldest tool round pairs (FIFO). This means `fund_manager` (10th agent in the trading graph) may lose `position_review` results — the critical data it needs for allocation decisions.

### How

**Hybrid priority system**:

1. **Config defaults**: Add `priority_tier` field per agent in `agents.yaml` (values: `P0`, `P1`, `P2`, `P3`)
2. **Type overrides**: Certain message types are always P0 regardless of source agent:
   - Risk gate output
   - Kill switch status
   - Position state / portfolio context
   - Error messages from blocking nodes

**Priority tiers**:
- P0 (never prune): risk/execution state — always retained
- P1 (summarize when budget tight): signal briefs, trade decisions, regime assessments
- P2 (prune first): raw analysis text, verbose tool outputs, intermediate reasoning
- P3 (ephemeral): debug logs, trace metadata — never added to LLM context

**Pruning algorithm** (replace current FIFO):
1. Calculate current message budget usage
2. If over budget, prune all P3 messages (should already be excluded)
3. If still over, prune P2 messages oldest-first
4. If still over, summarize P1 messages using Haiku (cheap/fast) — replace verbose content with summary. **Hard timeout of 2 seconds** on the summarization call. If Haiku is slow or unavailable, fall back to truncation (first N chars) rather than stalling the pipeline.
5. P0 messages are never pruned or summarized

**Prefer pre-computing summaries** at merge points (compaction) over lazy summarization during pruning. Merge-point summaries run once per branch, while pruning-time summarization runs before every agent invocation.

**Compaction at merge points**: At `merge_parallel` and `merge_pre_execution`, summarize branch outputs before passing downstream. Use LangGraph's `RemoveMessage` with `add_messages` reducer for rolling-window compaction — remove old verbose messages, add summary message.

**Implementation**: Modify `_prune_messages()` in `src/quantstack/graphs/agent_executor.py`. Add priority tag to each message when constructed (metadata field). The pruning function reads this metadata to decide order.

---

## Section 9: Pre-Trade Risk Gate Additions

### What and Why

The risk gate (`src/quantstack/execution/risk_gate.py`) is the single enforcement point for all trade safety. Currently it checks per-position limits (size, liquidity, participation rate, holding period) but lacks portfolio-level pre-trade checks: correlation with existing positions, daily deployment limits, and sector concentration.

**Prerequisite**: Update `CLAUDE.md` to change "Never modify" to "Never weaken or bypass" on the risk gate rule. This preserves the intent (prevent weakening) while removing ambiguity about strengthening the gate.

### 4.9 Pre-Trade Correlation Check

Add to `risk_gate.check()`: before approving a new position, compute 30-day rolling correlation of daily returns between the candidate symbol and every existing position. Reject if any correlation > 0.7.

**Fail closed**: If correlation data is unavailable (insufficient history, data feed error), the check blocks. It does not pass. Missing data is treated as "unknown risk" not "no risk."

Requires minimum 20 days of common return data. For new symbols with less history, use sector-based proxy correlation as fallback (map symbol to sector, use sector ETF correlation). **Note**: this fallback depends on sector mapping data from 4.11 — implement them together.

### 4.10 Portfolio Heat Budget

Add to `risk_gate.check()`: max daily notional deployed (configurable, default 30% of equity/day). Track cumulative notional deployed today across all entries.

**Implementation**: The heat budget is **system-wide** (all 3 graph services deploy capital from the same portfolio). Query the positions DB table for today's entries on every `check()` call — do not use an in-memory accumulator, as it won't see deployments from other graph services. The query is cheap (indexed on entry date) and correctness is more important than the microseconds saved by caching.

### 4.11 Sector Concentration Pre-Trade

Add to `risk_gate.check()`: reject if adding this position would push any single sector above 40% concentration (configurable).

**New data requirement**: Sector mapping per symbol. Options:
- Maintain a `symbol_sectors` table populated from Alpha Vantage company overview data
- Use the universe.py metadata if it already includes sector information
- Fallback: if sector unknown, treat as its own sector (conservative — won't trigger concentration limit but won't benefit from diversification credit either)

### All 3 checks share these properties:
- Added inside `risk_gate.check()`, not as separate layers
- Each produces a `RiskViolation` with descriptive message on failure
- Each has configurable thresholds (correlation: 0.7, heat: 30%, sector: 40%)
- All fail closed on missing data

---

## Section 10: Regime Flip Forced Review

### What and Why

When a momentum trade entered in `trending_up` regime persists after the regime flips to `ranging` or `trending_down`, the system logs an alert but takes no action. The position stays open in a hostile regime.

Currently regime flip detection lives in `risk_gate.monitor()` (continuous monitoring). The `MonitoredPosition` dataclass has no `regime_at_entry` field.

### How

**Store regime at entry**:
- Add `regime_at_entry: str` field to `MonitoredPosition` dataclass
- Add `regime_at_entry` column to the positions DB table
- DB is source of truth; `MonitoredPosition` is runtime cache, reconstructed from DB on restart

**Regime comparison logic** (extend `monitor()` and add to position review):

| Entry Regime | Current Regime | Severity | Action |
|---|---|---|---|
| `trending_up` | `trending_down` | Severe | Auto-exit within 1 cycle |
| `trending_down` | `trending_up` | Severe (short positions) | Auto-exit within 1 cycle |
| `trending_up` | `ranging` | Moderate | Tighten stops by 50% |
| `trending_down` | `ranging` | Moderate | Tighten stops by 50% |
| `ranging` | `trending_*` | Moderate | Tighten stops by 50% |
| Any | `unknown` | Moderate | Tighten stops by 50% |

**Auto-exit implementation**: Severe mismatch generates an exit order injected into the next cycle's `exit_orders` list with reason `"regime_flip_severe"`. This flows through the normal execution pipeline (including risk gate) — not a bypass.

**Stop tightening**: Moderate mismatch modifies the `stop_price` on the `MonitoredPosition` to 50% of the distance between current price and existing stop. For example, if current = $100, stop = $90 (distance $10), new stop = $95 (distance $5).

**Minimum stop distance floor**: Tightening is capped at a minimum of max(2x ATR, 1% of current price). Without this floor, repeated moderate flips could push stops within bid/ask spread noise (e.g., two consecutive flips: $10 → $5 → $2.50).

**Handle `stop_price = None`**: If a position has no existing stop, the regime flip logic cannot tighten what doesn't exist. In this case, SET a stop at the floor distance (max(2x ATR, 1% of current price)). Positions without stops in hostile regimes are the highest-risk scenario — they need one.

---

## Section 11: Database Migration

All new tables and columns ship as a single migration script.

**New tables:**
1. `circuit_breaker_state` — node-level breaker state
2. `agent_dlq` — dead letter queue for parse/validation failures

**Modified tables:**
3. Positions table: add `regime_at_entry TEXT` column (nullable, backfilled as `'unknown'` for existing positions)
4. `loop_cursors`: ensure UNIQUE constraint on `consumer_id` (may already exist)

**Migration order**: Tables first (additive), then column additions (additive), then constraint verification. No destructive changes.

---

## Section 12: Testing Strategy

### Unit Tests

Each item gets focused unit tests. Key test patterns:

**4.1 Race condition**: Construct state with overlapping symbols in `exit_orders` and `entry_candidates` → verify `resolve_symbol_conflicts` removes entries, keeps exits, logs conflict.

**4.2 Error blocking**: Mock a blocking node failure → verify execution gate halts pipeline. Mock a non-blocking node failure → verify safe default used and pipeline continues.

**4.3 Pydantic validation**: Construct state dict with typo key → verify `ValidationError` raised. Construct with wrong type → verify rejection. Construct valid state → verify acceptance.

**4.4 Circuit breaker**: Fail a node 3 times consecutively → verify skip with safe default. Succeed after skip → verify counter reset. Verify DB state persistence across test invocations.

**4.5 Tool access**: Attempt blocked tool call → verify error returned to agent and security event logged. Attempt allowed tool call → verify execution proceeds normally.

**4.6 Cursor atomicity**: Verify single upsert statement. Simulate concurrent cursor updates → verify no lost cursors.

**4.7 DLQ**: Feed unparseable output → verify DLQ row written with full context. Verify DLQ rate calculation over rolling window.

**4.8 Message pruning**: Construct messages with mixed priorities → verify P2 pruned before P1, P0 never pruned. Verify compaction produces valid summary.

**4.9-4.11 Risk gate**: Mock portfolio with correlated positions → verify rejection at 0.7. Mock daily notional at 31% → verify rejection. Mock sector at 41% → verify rejection. Verify fail-closed on missing data.

**4.12 Regime flip**: Set `regime_at_entry = "trending_up"`, current = `"trending_down"` → verify auto-exit generated. Verify stop tightening math for moderate mismatches.

**Additional tests from review feedback:**

- **State key audit test**: Inventory all keys returned by all nodes → assert they exist in the Pydantic state model. Run this as a pre-migration validation gate.
- **Concurrent circuit breaker updates**: Two overlapping cycles both incrementing failure_count → verify atomic increment produces correct count (no lost increments).
- **Haiku summarization failure**: Summarization LLM call times out → verify fallback to truncation, pipeline continues.
- **Regime flip with stop_price=None**: Position has no stop → verify a stop is SET (not tightened) at the floor distance.
- **Tool access bypass attempt**: Attempt to call a blocked tool through `TOOL_REGISTRY` directly (not through agent executor) → verify the guard is at the invocation layer.
- **Circuit breaker + execution gate interaction**: `data_refresh` trips breaker in cycle N → verify safe default sets error flag → verify execution gate halts pipeline → verify graceful cycle end.

### Integration Tests

- Full trading graph cycle with parallel branches producing conflicting symbols → verify conflict resolved and execution proceeds safely
- Full cycle with induced blocking node failure → verify halt before execution
- Circuit breaker state persistence across graph invocations
- Circuit breaker trip on blocking node → execution gate halt → clean cycle termination (end-to-end interaction test)

---

## Dependency Order

Items within Phase 4 have internal dependencies:

1. **First** (parallel, zero-risk, unblocks everything):
   - CLAUDE.md update ("Never modify" → "Never weaken or bypass") — establishes policy for risk gate changes
   - DB migration (new tables + columns) — purely additive, unblocks circuit breaker, DLQ, and regime storage
   - State key audit (dynamic logging of all node returns) — prerequisite for Pydantic migration
2. **Second**: Pydantic state migration (4.3) — foundational, high-risk, needs focused attention. Depends on state key audit completing.
3. **Third** (parallel):
   - Error blocking (4.2) + node classification — uses new Pydantic models
   - Race condition fix (4.1) — uses new typed output models
   - Event bus fix (4.6) — independent
   - Circuit breaker (4.4) — depends on Pydantic models + DB tables
   - Tool access control (4.5) — independent
   - DLQ (4.7) — depends on DB tables
   - Regime-at-entry storage (4.12 part 1) — depends on DB tables
4. **Fourth** (parallel):
   - Message pruning (4.8) — depends on Pydantic models (for typed message metadata) and agents.yaml config
   - Pre-trade risk checks (4.9-4.11) — 4.9 and 4.11 share sector mapping data, implement together
   - Regime flip logic (4.12 part 2) — depends on regime-at-entry storage
5. **Last**: Integration tests across all items

---

## Files Changed Summary

| File | Changes |
|------|---------|
| `CLAUDE.md` | "Never modify" → "Never weaken or bypass" for risk gate |
| `src/quantstack/graphs/state.py` | TypedDict → Pydantic BaseModel (all 4 states including SymbolValidationState) |
| `src/quantstack/graphs/trading/models.py` | **NEW** — Node output models for trading graph |
| `src/quantstack/graphs/research/models.py` | **NEW** — Node output models for research graph |
| `src/quantstack/graphs/supervisor/models.py` | **NEW** — Node output models for supervisor graph |
| `src/quantstack/graphs/trading/graph.py` | Add `resolve_symbol_conflicts` node, error gating edges, node classification |
| `src/quantstack/graphs/trading/nodes.py` | Return typed output models, integrate circuit breaker decorator |
| `src/quantstack/graphs/circuit_breaker.py` | **NEW** — Decorator, DB operations, state machine |
| `src/quantstack/graphs/agent_executor.py` | Tool access guard, DLQ routing on parse failure, priority-based pruning |
| `src/quantstack/graphs/*/config/agents.yaml` | `blocked_tools` per graph, `priority_tier` per agent |
| `src/quantstack/coordination/event_bus.py` | DELETE+INSERT → upsert |
| `src/quantstack/execution/risk_gate.py` | Pre-trade correlation, heat budget, sector concentration |
| `src/quantstack/execution/execution_monitor.py` | `regime_at_entry` field, regime comparison + stop tightening |
| `migrations/phase4_*.sql` | **NEW** — Tables, columns, constraints |
| `tests/unit/test_*.py` | Tests for all 12 items |
