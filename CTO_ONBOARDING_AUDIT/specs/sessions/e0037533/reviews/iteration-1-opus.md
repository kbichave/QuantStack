# Opus Review

**Model:** claude-opus-4-6
**Generated:** 2026-04-06

---

# Plan Review: Phase 4 — Agent Architecture & Coordination

## Overall Assessment

This is a well-structured plan that correctly identifies the 12 spec items, provides clear implementation paths, and makes defensible architectural choices. The dependency ordering is largely correct, and the fail-closed philosophy is appropriate for a trading system. That said, there are several issues ranging from a likely state schema bug to gaps in the testing strategy and some implementation risks that need attention before execution.

---

## 1. Completeness

**All 12 spec items are covered.** The plan maps cleanly to items 4.1 through 4.12. No spec item is missing.

**Minor gaps:**

- **Section 1 (Pydantic migration)** does not mention `SymbolValidationState` (line 46 of `state.py`). This is a fourth state schema used by `Send()`-spawned validation workers in the research graph. It also uses `TypedDict` with `Annotated` reducers and needs the same migration treatment. Missing it would leave a hole where malformed validation worker state silently passes.

- **Section 9 (Pre-Trade Correlation Check)** mentions a sector-based proxy correlation fallback for new symbols, but does not specify where sector mapping data comes from for this specific use case. Section 11 (Sector Concentration) addresses sector mapping broadly, but the plan does not connect these two — the correlation fallback depends on sector data being available, creating an implicit dependency between 4.9 and 4.11 that is not captured in the dependency order.

- **CLAUDE.md update** is mentioned in Section 9 but not given its own sequencing. It should be the very first commit since it changes the contract that governs whether this work is even allowed. If someone reviews the risk gate changes before seeing the CLAUDE.md update, the "Never modify" rule would block the PR.

---

## 2. Correctness

### Ghost field: `alpha_signals` is not in `TradingState`

The `_risk_gate_router` at line 70-80 of `graph.py` checks `state.get("alpha_signals", [])`, but `alpha_signals` does not appear in `TradingState` in `state.py`. This is exactly the class of bug that Section 1 (Pydantic migration) is supposed to catch. However, the plan does not mention this specific case. When migrating to `extra="forbid"`, any node currently writing `alpha_signals` to state will immediately break with a `ValidationError`.

**Action needed:** Before writing any Pydantic models, run a dynamic audit. Instrument each graph to log every key returned by every node across 10+ cycles. Compare those keys against the TypedDict definitions. Any key that appears in node returns but not in TypedDict is a field that either needs to be added to the Pydantic model or removed from the node. This is a prerequisite discovery step the plan should explicitly call out.

### Event bus uses `?` placeholders, not `$1`

Section 6 shows the upsert using `$1, $2, $3` (PostgreSQL libpq parameter style), but the actual `event_bus.py` code (lines 232-250) uses `?` placeholders throughout. This suggests the codebase uses a DB abstraction layer (likely `PgConnection` wrapping something) that maps `?` to the appropriate driver syntax. The plan's SQL example should match the codebase convention, or the implementer may introduce a syntax mismatch.

### `resolve_symbol_conflicts` placement in the graph

The plan places this node between `merge_parallel` and `risk_sizing`. Looking at the actual graph (line 193), the edge is `merge_parallel -> risk_sizing`. The plan correctly identifies this insertion point. However, the plan says the `execution_gate` check runs as a conditional edge function between `resolve_symbol_conflicts` and `risk_sizing`. That means the graph flow becomes:

```
merge_parallel -> resolve_symbol_conflicts -> [execution_gate] -> risk_sizing
```

This is correct for blocking entries. But the execution gate also needs to block `execute_exits` failures from propagating. Currently `execute_exits` runs *before* `merge_parallel` (it is in the position_review branch). If `execute_exits` fails, the error is already in the `errors` list when it reaches the gate — that part works. But the plan classifies `execute_exits` as blocking. If it fails, the gate halts the entire pipeline including new entries. This is the correct behavior (failing to close exposure is dangerous), but it means a transient `execute_exits` failure also blocks the entry pipeline. The plan should acknowledge this tradeoff explicitly.

### Circuit breaker cooldown of 30 seconds is too short

For a system that runs in cycles (60-300 second iteration intervals per the event bus docs), a 30-second cooldown means a node that fails 3 times will be in "Open" state, but by the next cycle (60+ seconds later) it's already in "Half-Open" and will be probed. This effectively means the circuit breaker never actually skips a full cycle. For LLM nodes where failures tend to cluster (provider outages, rate limits), you want the cooldown to be at least 1 full cycle duration, probably 2-3x. Suggest making `cooldown_seconds` configurable per node with a default of 300 (5 minutes), not 30.

---

## 3. Risk

### High risk: All-at-once Pydantic migration

The plan explicitly says "All 3 graphs at once. No half-migrated state." This is the highest-risk item. Every node in every graph returns a dict that will now be validated against a Pydantic model. Any mismatch anywhere breaks the entire system. The `alpha_signals` ghost field is one example; there are likely others.

**Mitigation the plan should add:** Before writing any Pydantic models, run a dynamic audit. Instrument each graph to log every key returned by every node across 10+ cycles. Compare those keys against the TypedDict definitions. Any key that appears in node returns but not in TypedDict is a field that either needs to be added to the Pydantic model or removed from the node. This audit is cheap (a day of paper trading with logging) and prevents the migration from being a "discover bugs one by one in production" exercise.

### Medium risk: Haiku summarization in the pruning hot path

Section 8 proposes using Haiku to summarize P1 messages when the budget is tight. This introduces an LLM call into the message preparation path — the path that runs *before* every agent invocation. If Haiku is slow or unavailable, every agent call in the pipeline stalls. The plan does not specify a timeout or fallback for this summarization step.

**Suggestion:** Add a hard timeout (2 seconds) on the summarization call. If it fails, fall back to truncation (first N chars) rather than full content. Also consider pre-computing summaries at merge points (which the plan does mention as "compaction at merge points") rather than lazily summarizing during pruning.

### Medium risk: Stop tightening math

Section 10 says "new stop = 50% of the distance between current price and existing stop." The example is clear (100 -> 95), but there is no floor on stop distance. After two consecutive moderate regime flips, the stop would be at 97.5% of current price — likely within normal bid/ask spread noise for some instruments. The plan should specify a minimum stop distance (e.g., 2x ATR or 1% of price) below which tightening is capped.

### Low risk: Race condition in heat budget accumulator

Section 9 mentions maintaining an in-memory accumulator for daily notional deployed with "DB persistence for crash recovery." If multiple graph cycles overlap (the system runs 3 independent graph services), the in-memory accumulator won't see deployments from other graphs. The plan should clarify whether the heat budget is per-graph or system-wide, and if system-wide, it must query the DB every time rather than relying on an in-memory cache.

---

## 4. Dependencies

The dependency order in the plan is mostly correct but has two issues:

### Issue 1: DB migration should be first, not second

The plan puts Pydantic migration first, then DB migration in the second tier. But the circuit breaker (Section 4) needs its DB table to exist before the decorator can be tested. The DLQ (Section 7) needs its table too. Since DB migrations are purely additive (new tables, new columns, no destructive changes), they have zero risk and should be the very first step — even before Pydantic migration. This unblocks all DB-dependent work immediately.

**Corrected order:**
1. CLAUDE.md update + DB migration (zero-risk, unblocks everything)
2. Pydantic state migration (foundational, high-risk, needs focus)
3. Everything else as currently specified

### Issue 2: Message pruning depends on more than error blocking

The plan says message pruning (4.8) depends on error blocking (4.2) because it "needs priority tags." But priority tags come from `agents.yaml` configuration (Section 8's own design), not from error blocking. The actual dependency is on the Pydantic models (for typed message metadata) and the `agents.yaml` config changes. Error blocking is independent.

---

## 5. Testability

### Good coverage for unit tests

The test list in Section 12 covers the critical paths for each item. The proposed patterns (mock blocking node failure, construct conflicting symbol lists, verify Pydantic rejection) are the right ones.

### Missing tests

- **No test for the `alpha_signals` ghost field** or any other undeclared state keys. Before migrating to Pydantic, there should be a test that inventories all keys returned by all nodes and asserts they exist in the state model.

- **No test for concurrent circuit breaker updates.** If two graph cycles run close together, both could read "failure_count = 2" and both increment to 3, but only one actually writes. The circuit breaker DB operations need to use `UPDATE ... SET failure_count = failure_count + 1 WHERE breaker_key = $1 RETURNING failure_count` (atomic increment) rather than read-modify-write. The plan does not specify this, and the test list does not check for it.

- **No test for Haiku summarization failure** in the pruning path. What happens when the summarization LLM call fails during message preparation? The pruning path needs its own fallback test.

- **No test for regime flip with no existing stop.** Section 10 assumes a stop price exists to tighten. `MonitoredPosition.stop_price` is `float | None = None`. The tightening math will crash on `None`. The test matrix should include positions with no stop.

- **No negative test for tool access control bypass.** The test says "attempt blocked tool call -> verify error." It should also verify that the block list cannot be circumvented by calling the tool through an alias or through `TOOL_REGISTRY` directly (i.e., the guard is at the invocation layer, not just at binding time).

- **Integration test gap:** No test for the full cycle with circuit breaker tripping mid-pipeline. What happens when `data_refresh` trips its breaker in cycle 4? Does the entire pipeline skip gracefully? The safe default for a blocking node is supposed to set an error flag that the execution gate catches — this interaction between circuit breaker and execution gate needs an integration test.

---

## 6. Architecture

### Well-justified decisions

- **`extra="forbid"` on Pydantic models** — Correct choice. The performance cost is negligible vs the safety gain.
- **Exits take priority over entries** in conflict resolution — This is the right risk-off bias for an autonomous system.
- **No self-healing for DLQ** — Excellent restraint. Auto-patching prompts that influence capital allocation is exactly the wrong place to automate without extensive validation data.
- **Circuit breaker in PostgreSQL, not in-memory** — Correct for a multi-service architecture where state must survive restarts.
- **`blocked_tools` at graph level, not agent level** — Simpler, fewer config errors, correct boundary.

### Questionable decisions

**Node output models per node (Section 1):** The plan proposes a separate Pydantic model for every node's output (e.g., `DataRefreshOutput`, `SafetyCheckOutput`, `PlanDayOutput`). For 16 trading graph nodes + research + supervisor, this is 20+ output model classes. The benefit (preventing a node from writing to fields it shouldn't) is real but the cost is significant: every time a state field is added or renamed, you touch the parent state model AND every node output model that references it.

**Alternative worth considering:** Instead of per-node output models, use a `model_validator` on the parent state that checks which fields changed per update and validates them against a per-node allow-list (a simple `dict[str, set[str]]` mapping node names to allowed fields). Same safety, one-tenth the boilerplate, and the allow-list doubles as documentation. The per-node output models can always be added later if the allow-list approach proves insufficient.

**Execution gate as conditional edge (Section 2):** The plan says the gate runs as a conditional edge function, not a separate node. This is clean for the happy path, but conditional edge functions in LangGraph don't have access to node-level error context the same way nodes do. They see state, not execution metadata. If the gate needs to distinguish "data_refresh returned an error in state" from "data_refresh threw an exception that was caught by the retry policy," it needs to be a node with access to both state and error context. Verify that the conditional edge approach can access all the information the gate needs.

**Event bus placeholder mismatch (Section 6):** As noted above, the code uses `?` placeholders. But more fundamentally — the event bus uses a `PgConnection` that appears to be a custom wrapper. The plan should verify that this wrapper supports `ON CONFLICT` syntax before assuming the upsert will work. Some lightweight DB wrappers don't pass through all PostgreSQL-specific SQL features.

---

## Summary of Action Items (Priority Order)

1. **[Blocking] Audit all node return keys against TradingState/ResearchState/SupervisorState before Pydantic migration.** The `alpha_signals` ghost field proves undeclared keys exist. Migrating blind will cause runtime failures.

2. **[Blocking] Include `SymbolValidationState` in the Pydantic migration scope.** It's a fourth state schema currently missing from the plan.

3. **[High] Move DB migration to step 1 in dependency order.** It's zero-risk additive work that unblocks multiple parallel streams.

4. **[High] Add minimum stop distance floor to regime flip stop tightening.** Without it, repeated moderate flips produce stops within noise range.

5. **[High] Handle `stop_price = None` in regime flip logic.** The `MonitoredPosition` dataclass allows it.

6. **[Medium] Increase default circuit breaker cooldown from 30s to 300s** (or make it per-node configurable with a sane default).

7. **[Medium] Add timeout + fallback for Haiku summarization in pruning path.** An LLM call in the message preparation hot path needs a safety net.

8. **[Medium] Use atomic increment (`UPDATE ... RETURNING`) for circuit breaker failure counts.** Read-modify-write has a race condition under concurrent cycles.

9. **[Medium] Clarify whether heat budget is per-graph or system-wide,** and design the accumulator accordingly.

10. **[Low] Match SQL placeholder style to codebase convention** (`?` vs `$1`).

11. **[Low] Consider per-node field allow-list instead of per-node output models** to reduce boilerplate while maintaining the same safety guarantee.

12. **[Low] Commit CLAUDE.md update as the very first change** to establish the policy basis for risk gate modifications.
