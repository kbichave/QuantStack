# Integration Notes: Opus Review Feedback

## Integrating (HIGH priority)

### 1. Fix `Send()` misuse → parallel branch pattern
**Why**: Opus is correct. `Send()` is map-reduce, not multi-node parallel. The correct pattern is dual edges from `daily_plan` to both nodes, with a join node before `risk_sizing`.

### 2. Remove `persist_state` node
**Why**: Contradicts Section 9.5 which says checkpointing is automatic. Redundant.

### 3. Dissolve Phase 5 into Phases 2-4
**Why**: Graph nodes call tools. Tools must exist when graphs are built. Migrating tools per-graph alongside the graph that uses them is the only coherent approach.

### 4. Detail sync-to-async runner migration
**Why**: The runner transition from `crew.kickoff()` (sync) to `graph.ainvoke()` (async) is non-trivial. Need to address `asyncio.run()` entry point, watchdog compat, and shutdown handlers.

### 5. Pin LangGraph version
**Why**: Pre-1.0, breaking changes between minors. Pin `>=0.4.0,<0.5.0`.

### 6. `nest-asyncio` removal timing
**Why**: Can't remove in Phase 1 if mcp_bridge tools still use `run_async()`. Must wait until all tools migrated.

## Integrating (MEDIUM priority)

### 7. Add regression testing phase
**Why**: Capture current system I/O pairs before migration, replay through new graphs. Critical for a capital-handling system.

### 8. Add shadow-run acceptance criteria
**Why**: Big-bang rewrite of capital system. Plan should include paper-trade shadow-run phase before live cutover.

### 9. pgvector migration: verify dimensions + sample similarity
**Why**: Count check alone is insufficient for embeddings.

### 10. Add error handling / retry_policy per node
**Why**: Plan has no error handling topology. LangGraph supports `retry_policy`.

### 11. Clarify node return semantics for append-only fields
**Why**: Common pitfall — must return empty list or omit from dict for `Annotated[list, operator.add]` fields.

## NOT Integrating

### mcp_bridge file count (8 vs 50)
**Why**: The 50 count comes from the user's interview answer about `tools/mcp_bridge/` files. The reviewer may have counted only the top-level files. The actual count needs verification during implementation — this is an implementation detail, not a plan-level concern. Will note to audit actual count.

### `guardrails/mcp_response_validator` dependency
**Why**: Valid observation but too specific for the plan level. The implementation will discover and preserve this dependency during the mcp_bridge audit.

### `duckduckgo-search` verification
**Why**: Implementation-level check. The audit in Phase 5 (now dissolved) will handle this naturally.

### `loguru` vs `logging` standardization
**Why**: Out of scope for this migration. Technical debt to address separately.

### `start.sh` and tmux updates
**Why**: Good catch but trivial — the runner module paths aren't changing, only the internals. Entry points remain `python -m quantstack.runners.*`. Will add a note.

### `ollama` error message update
**Why**: Too specific for plan level. Will be caught during implementation.
