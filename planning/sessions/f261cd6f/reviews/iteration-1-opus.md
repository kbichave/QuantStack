# Opus Review

**Model:** claude-opus-4-6
**Generated:** 2026-04-02

---

# Architectural Review: CrewAI to LangGraph Migration Plan

## 1. Completeness

**Verdict: Good coverage with notable gaps.**

### Gap 1: CrewAI `memory=True` semantics are unaddressed

All three crews use `memory=True`. The plan only addresses this via LangGraph checkpointing, but CrewAI's `memory=True` provides *cross-task conversational memory within a single crew invocation* — agents can reference what earlier agents said. LangGraph checkpointing is *cross-invocation* persistence. These are different things.

In the plan's LangGraph design, state fields carry structured data between nodes, which is the correct replacement for intra-invocation context passing. But the plan never explicitly calls out this tradeoff. The plan should audit the existing YAML task configs to determine if any task's `context` field references other tasks' full output text.

### Gap 2: Task YAML configs are not analyzed

The plan mentions existing `config/tasks.yaml` files will be replaced, but never looks at what's in them. Task configs define `expected_output`, `context` (upstream task dependencies), and `output_pydantic`/`output_json`. The `context` field is critical — it determines which prior task outputs are injected into a task's prompt. This is the real dependency graph.

### Gap 3: MCPBridge lifecycle

The `MCPBridge` class in `_bridge.py` is more complex than a simple BaseTool wrapper. It manages MCP server references, async-to-sync bridging, and response validation via `mcp_response_validator`. The plan's Phase 5 is vague. There are actually 8 files in `mcp_bridge/`, not 50.

### Gap 4: `guardrails/mcp_response_validator` dependency

`MCPBridge._bridge.py` imports from `quantstack.guardrails.mcp_response_validator`. Never mentioned in the plan.

### Gap 5: `duckduckgo-search` dependency removal

The spec lists `duckduckgo-search>=6.0.0` for removal but the plan never verifies whether non-CrewAI code uses it.

---

## 2. Correctness of LangGraph Patterns

### Issue 1: Misuse of `Send()` for parallel branches

Section 5.2 states `Send()` dispatches `position_review` and `entry_scan` concurrently. `Send()` is for *map-reduce* patterns — fanning out the **same node** over a collection. For two **different nodes** concurrently, the correct pattern is:

```python
graph.add_edge("daily_plan", "position_review")
graph.add_edge("daily_plan", "entry_scan")
```

LangGraph executes them concurrently if they share a common predecessor. But you need a **join node** that both feed into. The plan never specifies the join.

### Issue 2: `persist_state` node is redundant

Section 5.2 has a `persist_state` tool node but Section 9.5 says `AsyncPostgresSaver` handles this automatically. These contradict. Remove the node.

### Issue 3: Node return semantics

The plan never addresses what happens with `Annotated[list[str], operator.add]` fields when a node doesn't want to append. Must return empty list or omit from return dict. Common LangGraph pitfall.

---

## 3. Risk Assessment

### Risk 1: Big-bang rewrite of capital-handling system (HIGH)

The project CLAUDE.md says "Strangler fig, not big bang." The user chose clean-cut, but the plan should define a **shadow-run phase** where both old and new run in parallel on paper trades before cutting over.

### Risk 2: Synchronous runner becomes async without migration plan (MEDIUM)

`run_loop()` is fully synchronous. The plan says runners call `await graph.ainvoke()` but never addresses: how `run_loop()` becomes async, whether `AgentWatchdog` supports async, whether `GracefulShutdown` signal handlers work in async context.

### Risk 3: Hot-reload during trading cycles (MEDIUM)

If config reload triggers mid-cycle, what happens? The plan says per-cycle rebuild but `ConfigWatcher` with watchdog could trigger at any time.

### Risk 4: pgvector migration data loss (LOW but irreversible)

Count verification insufficient for embeddings. Should verify vector dimensions and do sample similarity search comparison.

---

## 4. Execution Ordering

### Phase 5 (mcp_bridge) should overlap with Phases 2-4

Graph nodes call tools. If tools aren't migrated yet, what do nodes call? Tools must migrate alongside their consuming graphs.

### Missing: Database schema migration phase

No explicit phase for running DB migrations (pgvector tables, checkpoint tables, crew_checkpoints handling).

---

## 5. Testing Strategy

### Weakness 1: No regression tests against current behavior
Capture current I/O pairs, replay through new graphs.

### Weakness 2: No load/timing tests
5-minute cycles. If `graph.ainvoke()` takes longer, cycles stack up.

### Weakness 3: Tool contract tests are thin
Should compare old-wrapper and new-wrapper outputs for identical inputs.

### Weakness 4: No LangFuse trace assertion tests
Should assert specific trace structure.

---

## 6. Missed Concerns

- **Error handling**: No retry/fallback topology for node failures. LangGraph supports `retry_policy`.
- **`nest_asyncio` removal timing**: Must happen after Phase 5, not Phase 1.
- **`ollama` dependency**: Error messages reference `quantstack[crewai]` install group which won't exist.
- **`start.sh` and tmux**: Never mentioned in plan. Will break if entry points change.
- **LangGraph versioning**: Pin to `>=0.4.0,<0.5.0` — pre-1.0, breaking changes between minors.
- **Existing tests**: Only some test files mentioned; need full list of all files importing CrewAI.

---

## Summary of Recommended Changes

| Priority | Section | Recommendation |
|----------|---------|----------------|
| **HIGH** | Risk | Add shadow-run acceptance criteria for capital-handling system rewrite |
| **HIGH** | Section 5.2 | Fix `Send()` misuse. Define correct parallel branch + join pattern |
| **HIGH** | Section 5.2/9.5 | Remove redundant `persist_state` node |
| **HIGH** | Phase 5 | Dissolve into Phases 2-4. Tools migrate with their consuming graphs |
| **HIGH** | Section 9 | Detail sync-to-async runner migration |
| **MEDIUM** | Section 4 | Audit tasks.yaml context fields to validate graph topologies |
| **MEDIUM** | Testing | Add regression tests: capture current I/O, replay through new graphs |
| **MEDIUM** | Phase 1 | Do not remove `nest-asyncio` until Phase 5 complete |
| **MEDIUM** | Dependencies | Pin LangGraph to `>=0.4.0,<0.5.0` |
| **LOW** | Section 7.3 | Migration script: verify vector dimensions + sample similarity |
| **LOW** | Missing | Update `start.sh`, tmux config, `ollama` error messages |
| **LOW** | Testing | Add timing benchmarks for graph invocations |
