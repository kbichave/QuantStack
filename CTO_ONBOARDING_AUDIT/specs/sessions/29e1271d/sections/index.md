<!-- PROJECT_CONFIG
runtime: python-uv
test_command: uv run pytest
END_PROJECT_CONFIG -->

<!-- SECTION_MANIFEST
section-01-db-schema
section-02-system-alerts
section-03-corporate-actions
section-04-factor-exposure
section-05-performance-attribution
section-06-dashboard-alerts
section-07-eventbus-ack
section-08-multi-mode
section-09-llm-unification
section-10-research-fanout
END_MANIFEST -->

# Implementation Sections Index — Phase 9: Missing Roles & Scale

## Dependency Graph

| Section | Depends On | Blocks | Parallelizable With |
|---------|------------|--------|---------------------|
| section-01-db-schema | - | all others | - |
| section-02-system-alerts | 01 | 03, 04, 05, 06, 07 | - |
| section-03-corporate-actions | 01, 02 | - | 04, 05, 07, 09, 10 |
| section-04-factor-exposure | 01, 02 | 08 | 03, 05, 07, 09, 10 |
| section-05-performance-attribution | 01, 02 | 08 | 03, 04, 07, 09, 10 |
| section-06-dashboard-alerts | 01, 02 | - | 03, 04, 05, 07, 09, 10 |
| section-07-eventbus-ack | 01, 02 | - | 03, 04, 05, 06, 09, 10 |
| section-08-multi-mode | 01, 04, 05 | - | - |
| section-09-llm-unification | 01 | - | 03, 04, 05, 06, 07, 10 |
| section-10-research-fanout | 01 | - | 03, 04, 05, 06, 07, 09 |

## Execution Order (3 Batches)

**Batch 1 — Foundation (sequential):**
1. section-01-db-schema (no dependencies)
2. section-02-system-alerts (depends on 01 — foundational for all alert-emitting sections)

**Batch 2 — Core features (parallel):**
3. section-03-corporate-actions
4. section-04-factor-exposure
5. section-05-performance-attribution
6. section-06-dashboard-alerts
7. section-07-eventbus-ack
8. section-09-llm-unification
9. section-10-research-fanout

**Batch 3 — Integration (sequential):**
10. section-08-multi-mode (depends on 04, 05 for mode-aware computation)

## Section Summaries

### section-01-db-schema
All new tables: corporate_actions, split_adjustments, system_alerts, dead_letter_events, factor_config, factor_exposure_history, cycle_attribution. EventBus ACK columns. TradingState field update.

### section-02-system-alerts
System-level alert lifecycle: 5 LangChain tools (create, acknowledge, escalate, resolve, query) + internal emit_system_alert() helper. Tool registry registration. Supervisor agent YAML binding.

### section-03-corporate-actions
AV dividend/split collectors, EDGAR 8-K M&A detection, CIK mapping, split auto-adjustment with Alpaca reconciliation, scheduled daily job in supervisor graph.

### section-04-factor-exposure
Factor computation module (beta, sector, style, momentum crowding), configurable drift alerts, supervisor health_check integration, factor_exposure_history persistence.

### section-05-performance-attribution
Attribution engine (factor/timing/selection/cost decomposition), new trading graph node after reflect, cycle_attribution DB persistence, accounting identity assertion.

### section-06-dashboard-alerts
TUI alerts widget on Overview tab, FastAPI /api/alerts endpoint, SSE stream alert events, dashboard event publishing integration.

### section-07-eventbus-ack
EventBus publish/ack/monitor extensions, ACK_REQUIRED_EVENTS config, 600s fixed timeout, dead letter handling, supervisor ACK monitor, consumer-side ack() calls.

### section-08-multi-mode
OperatingMode enum (4 modes), get_operating_mode() detection, trading graph conditional edge for monitor-only mode, risk gate hard block on exposure-increasing orders, runner interval updates.

### section-09-llm-unification
Audit remaining hardcoded model strings, route all through get_chat_model(), llm_config DB table with env→DB→code precedence, provider health tracking.

### section-10-research-fanout
Flip RESEARCH_FAN_OUT_ENABLED default to true, add semaphore + rate limiter dual throttling, quota monitoring with 80% threshold throttle.
