<!-- PROJECT_CONFIG
runtime: python-uv
test_command: uv run pytest
END_PROJECT_CONFIG -->

<!-- SECTION_MANIFEST
section-01-stop-loss
section-02-output-schemas
section-03-tool-ordering-caching
section-04-injection-defense
section-05-backups-checkpoints
section-06-eventbus-cicd
section-07-circuit-breaker
section-08-email-alerting
section-09-risk-gate
section-10-signal-data-tools
section-11-kill-switch-logs
section-12-multi-mode
section-13-autoresearch-loss
section-14-budget-compaction
section-15-layered-breaker-greeks
section-16-kb-ic-urgency
END_MANIFEST -->

# Implementation Sections Index

## Dependency Graph

| Section | Depends On | Blocks | Parallelizable With |
|---------|------------|--------|---------------------|
| section-01-stop-loss | - | 05 (idempotency) | 02, 03, 04, 06 |
| section-02-output-schemas | - | - | 01, 03, 04, 06 |
| section-03-tool-ordering-caching | - | 10 (tool registry) | 01, 02, 04, 06 |
| section-04-injection-defense | - | - | 01, 02, 03, 06 |
| section-05-backups-checkpoints | 01 (idempotency) | 12 (durable state) | 04, 06 |
| section-06-eventbus-cicd | - | - | 01, 02, 03, 04 |
| section-07-circuit-breaker | Phase 1 complete | - | 08, 09, 10 |
| section-08-email-alerting | Phase 1 complete | 11 (email deps) | 07, 09, 10 |
| section-09-risk-gate | Phase 1 complete | - | 07, 08, 10 |
| section-10-signal-data-tools | 03 (tool ordering) | - | 07, 08, 09 |
| section-11-kill-switch-logs | 08 (email) | - | 07, 09, 10 |
| section-12-multi-mode | Phase 2 complete | 13 (overnight mode) | 14, 15 |
| section-13-autoresearch-loss | 12, 16-IC | - | 14, 15 |
| section-14-budget-compaction | Phase 2 complete | - | 12, 15, 16 |
| section-15-layered-breaker-greeks | Phase 2 complete | - | 12, 14, 16 |
| section-16-kb-ic-urgency | Phase 2 complete | 13 (IC tracker) | 12, 14, 15 |

## Execution Order (Batched)

**Batch 1 — Phase 1 (parallel):**
1. section-01-stop-loss, section-02-output-schemas, section-03-tool-ordering-caching, section-04-injection-defense, section-06-eventbus-cicd

**Batch 2 — Phase 1 (after batch 1):**
2. section-05-backups-checkpoints (needs idempotency from section-01)

**Batch 3 — Phase 2 (parallel):**
3. section-07-circuit-breaker, section-08-email-alerting, section-09-risk-gate, section-10-signal-data-tools

**Batch 4 — Phase 2 (after batch 3):**
4. section-11-kill-switch-logs (needs email alerting from section-08)

**Batch 5 — Phase 3 (parallel):**
5. section-12-multi-mode, section-14-budget-compaction, section-15-layered-breaker-greeks, section-16-kb-ic-urgency

**Batch 6 — Phase 3 (after batch 5):**
6. section-13-autoresearch-loss (needs multi-mode from 12 + IC tracker from 16)

## Section Summaries

### section-01-stop-loss
Mandatory stop-loss enforcement at OMS layer. Bracket order implementation with Alpaca API. Fallback to separate contingent orders. Startup reconciliation check. **Includes idempotency guards (client_order_id dedup) needed by section-05.**

### section-02-output-schemas
Pydantic output models for 5 critical agent nodes. Schema-aware parsing with retry. Safe conservative defaults. Groq compatibility benchmark.

### section-03-tool-ordering-caching
Deterministic tool ordering (sorted alphabetically). Prompt caching with cache_control breakpoints. Bedrock beta header wiring. Cache hit rate verification.

### section-04-injection-defense
Sanitization utility for untrusted data. XML-tagged structured templates replacing f-string interpolation. Pattern stripping and length truncation.

### section-05-backups-checkpoints
Daily pg_dump backup sidecar. PostgresSaver migration (direct cutover). AsyncConnectionPool per graph. Crash recovery verification.

### section-06-eventbus-cicd
EventBus wiring: Trading Graph polls for kill switch events. Kill switch publishes events. Scheduler containerization with health check. CI/CD pipeline re-enablement.

### section-07-circuit-breaker
LLM runtime failover with CircuitBreaker class. Per-provider health state. Retry with backoff. Provider cooldown. LangChain with_fallbacks() integration.

### section-08-email-alerting
Gmail SMTP alerting with 3 levels (INFO/WARNING/CRITICAL). Rate limiting. Local file fallback when SMTP fails.

### section-09-risk-gate
Pre-trade correlation check (>0.7 → 50% haircut). Market hours hard gating with TradingWindow enum. Daily notional deployment cap (30% equity).

### section-10-signal-data-tools
Signal cache auto-invalidation on refresh. Data staleness rejection in collectors. Tool registry split (ACTIVE/PLANNED). Per-agent temperature configuration.

### section-11-kill-switch-logs
Tiered kill switch recovery (transient auto-reset, permanent escalation). Fluent-bit → Loki → Grafana log aggregation wiring. Error rate alerting.

### section-12-multi-mode
ScheduleMode enum (MARKET_HOURS, EXTENDED_HOURS, OVERNIGHT_WEEKEND). Mode-aware graph routing. Timezone-aware scheduling (America/New_York).

### section-13-autoresearch-loss
Overnight autoresearch loop (70/30 new/refine). 5-min budget per experiment. IC > 0.02 gate. Loss analyzer with failure taxonomy. Research queue generation.

### section-14-budget-compaction
Per-agent token/time/cost budget tracker. Graceful exit at node boundaries. Haiku compaction at Trading Graph merge points (40-60% reduction).

### section-15-layered-breaker-greeks
Dual circuit breaker (daily P&L + portfolio HWM). Limit orders for emergency liquidation. Dead-man's switch (60s). Greeks integration in risk gate (delta/gamma/vega/theta limits).

### section-16-kb-ic-urgency
Knowledge base semantic search fix. HNSW index verification. Signal IC daily computation with collector gating. PG LISTEN/NOTIFY urgency channel.
