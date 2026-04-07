<!-- PROJECT_CONFIG
runtime: python-uv
test_command: uv run pytest
END_PROJECT_CONFIG -->

<!-- SECTION_MANIFEST
section-01-db-migration-and-policy
section-02-state-key-audit
section-03-pydantic-state-migration
section-04-node-output-models
section-05-error-blocking
section-06-race-condition-fix
section-07-circuit-breaker
section-08-tool-access-control
section-09-event-bus-cursor
section-10-dead-letter-queue
section-11-message-pruning
section-12-risk-gate-pretrade
section-13-regime-flip
section-14-integration-tests
END_MANIFEST -->

# Implementation Sections Index

## Dependency Graph

| Section | Depends On | Blocks | Parallelizable |
|---------|------------|--------|----------------|
| section-01-db-migration-and-policy | - | 02, 07, 10, 13 | Yes |
| section-02-state-key-audit | 01 | 03 | Yes |
| section-03-pydantic-state-migration | 02 | 04, 05, 06, 07, 08, 11 | No |
| section-04-node-output-models | 03 | 05, 06, 07 | No |
| section-05-error-blocking | 04 | 14 | Yes |
| section-06-race-condition-fix | 04 | 14 | Yes |
| section-07-circuit-breaker | 01, 04 | 14 | Yes |
| section-08-tool-access-control | 03 | 14 | Yes |
| section-09-event-bus-cursor | - | 14 | Yes |
| section-10-dead-letter-queue | 01 | 14 | Yes |
| section-11-message-pruning | 03 | 14 | Yes |
| section-12-risk-gate-pretrade | 01 | 14 | Yes |
| section-13-regime-flip | 01 | 14 | Yes |
| section-14-integration-tests | 05, 06, 07, 08, 09, 10, 11, 12, 13 | - | No |

## Execution Order

1. **Batch 1** (no dependencies, parallel):
   - section-01-db-migration-and-policy
   - section-09-event-bus-cursor

2. **Batch 2** (after batch 1):
   - section-02-state-key-audit

3. **Batch 3** (after batch 2):
   - section-03-pydantic-state-migration

4. **Batch 4** (after batch 3):
   - section-04-node-output-models

5. **Batch 5** (after batch 4, parallel):
   - section-05-error-blocking
   - section-06-race-condition-fix
   - section-07-circuit-breaker
   - section-08-tool-access-control
   - section-10-dead-letter-queue
   - section-11-message-pruning
   - section-12-risk-gate-pretrade
   - section-13-regime-flip

6. **Batch 6** (final, after all above):
   - section-14-integration-tests

## Section Summaries

### section-01-db-migration-and-policy
CLAUDE.md rule update ("Never modify" → "Never weaken or bypass"). DB migration: create `circuit_breaker_state`, `agent_dlq` tables; add `regime_at_entry` column to positions; ensure `loop_cursors` UNIQUE constraint. Zero-risk, additive only.

### section-02-state-key-audit
Dynamic audit of all node return keys across all 3 graphs. Log every key returned by every node, compare against TypedDict definitions. Resolve ghost fields (e.g., `alpha_signals`). Output: list of fields to add/remove before Pydantic migration.

### section-03-pydantic-state-migration
Convert `TradingState`, `ResearchState`, `SupervisorState`, `SymbolValidationState` from TypedDict to Pydantic BaseModel with `extra="forbid"`. Add field validators, model validators, input/output schemas per graph.

### section-04-node-output-models
Create typed Pydantic output models for every node in all 3 graphs. Each model defines which state fields the node writes. Includes `safe_default()` class method per model.

### section-05-error-blocking
Node classification (blocking vs non-blocking). Execution gate as conditional edge function. Error count gating before risk_sizing and execute_entries.

### section-06-race-condition-fix
New `resolve_symbol_conflicts` node between merge_parallel and risk_sizing. Exits take priority on symbol conflicts. Conflict event logging.

### section-07-circuit-breaker
`@circuit_breaker` decorator for node functions. PostgreSQL-backed state (3-state model). Atomic increment for concurrent safety. Per-node configurable cooldown (default 300s). LLM failure type discrimination.

### section-08-tool-access-control
`blocked_tools` per graph in agents.yaml. 5-line guard in agent_executor.py. Hard reject + security event log on violation.

### section-09-event-bus-cursor
Replace DELETE+INSERT cursor update with single PostgreSQL upsert. Verify PgConnection wrapper supports ON CONFLICT.

### section-10-dead-letter-queue
`agent_dlq` table integration into `parse_json_response()`. Langfuse metric for DLQ rate. Alert thresholds (5% warn, 10% critical). Dashboard + outbound notification on critical.

### section-11-message-pruning
Hybrid priority system (config defaults + type overrides). Replace FIFO pruning with priority-aware algorithm. Haiku summarization with 2s timeout + truncation fallback. Merge-point compaction.

### section-12-risk-gate-pretrade
Pre-trade correlation check (>0.7 rejection, fail closed). Portfolio heat budget (30% daily cap, system-wide DB query). Sector concentration (40% cap). Sector mapping data source.

### section-13-regime-flip
`regime_at_entry` field in MonitoredPosition + DB. Regime comparison logic in monitor(). Severity-based actions (severe=auto-exit, moderate=tighten stops). Minimum stop floor (2x ATR / 1%). Handle stop_price=None.

### section-14-integration-tests
Full trading graph cycle with conflicting symbols. Blocking node failure → execution gate halt. Circuit breaker + execution gate interaction. Cross-cycle breaker persistence.
