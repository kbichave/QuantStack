# TDD Plan: QuantStack 24/7 Autonomous Trading Readiness

**Testing framework:** pytest (configured in `pyproject.toml`, `asyncio_mode = "auto"`, paths: `tests/`, `src/quantstack/core/tests/`)
**Conventions:** Tests in `tests/` mirror `src/quantstack/` structure. Fixtures in `conftest.py`. Markers: `integration` (real DB), `regression` (behavioral contracts).

---

## Pre-Work: Section 0 — Commit Baseline

No tests needed — this is a git housekeeping step.

---

## Phase 1: Safety Hardening

### Section 1: Stop-Loss Enforcement & Bracket Orders

```python
# tests/execution/test_stop_loss_enforcement.py

# Test: submit_order rejects OrderRequest with stop_price=None → RiskViolation
# Test: submit_order accepts OrderRequest with valid stop_price
# Test: execute_bracket uses Alpaca bracket/OTO API when available
# Test: execute_bracket falls back to separate SL/TP when bracket API fails
# Test: partial fill places stop for filled quantity, not original quantity
# Test: stop price too close to market → widens to minimum distance + logs warning
# Test: startup reconciliation detects position without stop order → logs error
# Test: paper_broker mirrors bracket simulation and tracks linked stops
# Test: extended hours uses stop-limit as fallback when stop-market unavailable
```

### Section 2: Agent Output Schema Validation

```python
# tests/graphs/test_agent_output_schemas.py

# Test: DailyPlanOutput validates correct JSON structure
# Test: EntrySignal validates correct JSON structure
# Test: PositionReviewOutput validates correct JSON structure
# Test: FundManagerVerdict validates correct JSON structure
# Test: RiskSizingOutput validates correct JSON structure
# Test: parse failure on first attempt retries with schema hint
# Test: parse failure on second attempt returns safe default (not {})
# Test: safe default for fund_manager is all-reject (conservative)
# Test: safe default for entry_scan is empty list (no entries)
# Test: safe default for daily_plan is conservative plan with no candidates
# Test: raw LLM output logged as warning on fallback to safe default
# Test: with_structured_output works with json_schema method (Anthropic)
# Test: with_structured_output fallback to json_mode works (Groq)
```

### Section 3: Deterministic Tool Ordering & Prompt Caching

```python
# tests/tools/test_tool_ordering.py

# Test: tool_binding returns tools sorted alphabetically by name
# Test: two consecutive calls produce identical tool ordering
# Test: tool ordering is deterministic regardless of registry insertion order

# tests/graphs/test_prompt_caching.py

# Test: build_system_message includes cache_control in additional_kwargs
# Test: Bedrock model instantiation includes anthropic_beta header
# Test: cache_control only applied to Anthropic/Bedrock providers, not Groq
# Test: cache key stability: N consecutive calls with same config produce same prefix
```

### Section 4: Prompt Injection Defense

```python
# tests/llm/test_sanitize.py

# Test: sanitize_for_prompt escapes XML-like tags
# Test: sanitize_for_prompt strips "ignore previous instructions" patterns
# Test: sanitize_for_prompt truncates to max_length
# Test: sanitize_for_prompt handles empty string
# Test: sanitize_for_prompt handles None input gracefully
# Test: node templates use XML-tagged sections (not f-string interpolation)
```

### Section 5: Database Backups & Durable Checkpoints

```python
# tests/execution/test_order_idempotency.py

# Test: submit_order with duplicate client_order_id is rejected (not re-executed)
# Test: execute_bracket with duplicate client_order_id is rejected
# Test: unique client_order_id passes through normally

# tests/graphs/test_postgres_saver.py (integration marker)

# Test: AsyncPostgresSaver setup() is idempotent (call twice, no error)
# Test: graph checkpoint writes to PostgreSQL
# Test: graph resumes from last super-step after restart
# Test: CRITICAL integration test: full Trading Graph cycle → kill mid-cycle after
#       execute_entries → restart → verify no duplicate orders + correct state
# Test: each graph gets its own connection pool (no cross-contamination)

# tests/scripts/test_pg_backup.py

# Test: pg_backup.sh produces valid dump file
# Test: pg_restore_test.sh can restore from backup
```

### Section 6: EventBus Wiring & CI/CD

```python
# tests/coordination/test_eventbus_wiring.py

# Test: kill_switch.trigger() publishes KILL_SWITCH_TRIGGERED event to EventBus
# Test: Trading Graph safety_check node polls EventBus for KILL_SWITCH_TRIGGERED
# Test: Trading Graph safety_check node polls EventBus for RISK_EMERGENCY
# Test: Trading Graph safety_check node polls EventBus for IC_DECAY
# Test: safety_check aborts cycle when KILL_SWITCH_TRIGGERED received

# tests/scripts/test_scheduler_health.py

# Test: scheduler health endpoint returns 200 OK
# Test: scheduler health endpoint returns 503 when unhealthy
```

---

## Phase 2: Operational Resilience

### Section 7: LLM Circuit Breaker & Runtime Failover

```python
# tests/llm/test_circuit_breaker.py

# Test: retryable error (429) retries same provider with backoff
# Test: retryable error (500) retries same provider with backoff
# Test: 3rd consecutive failure switches to next provider in FALLBACK_ORDER
# Test: failed provider enters 5-minute cooldown
# Test: cooled-down provider re-enters rotation after 5 minutes
# Test: non-retryable error (400) fails immediately without retry
# Test: non-retryable error (401) fails immediately without retry
# Test: with_fallbacks chain exercises providers in correct order
# Test: health state is per-provider (provider A failure doesn't affect B)
```

### Section 8: Email Alerting System

```python
# tests/alerting/test_email_sender.py

# Test: send_alert sends email via SMTP for CRITICAL level
# Test: send_alert rate-limits INFO/WARNING to 1 per event type per 15 min
# Test: CRITICAL level bypasses rate limiting
# Test: SMTP failure falls back to local file logging
# Test: alert file fallback writes to /var/log/quantstack/alerts.log
# Test: AlertConfig loads from environment variables
```

### Section 9: Risk Gate Enhancements

```python
# tests/execution/test_risk_gate_enhancements.py

# Test: pre-trade correlation > 0.7 applies 50% haircut to position size
# Test: pre-trade correlation < 0.7 passes through unchanged
# Test: order outside market hours rejected (no extended_hours flag)
# Test: order outside market hours accepted with extended_hours=True
# Test: order within market hours accepted normally
# Test: daily notional cap exceeded → order rejected
# Test: daily notional cap resets at market open
# Test: daily notional tracks cumulative new deployments correctly
# Test: with $10K equity and 30% cap, max daily deployment is $3K
```

### Section 10: Signal & Data Quality Gates

```python
# tests/signal_engine/test_cache_invalidation.py

# Test: scheduled_refresh invalidates cache for refreshed symbol
# Test: cache returns fresh data after invalidation + new put

# tests/signal_engine/test_data_staleness.py

# Test: collector returns empty result when data staler than threshold
# Test: collector runs normally when data is fresh
# Test: staleness warning logged when data rejected

# tests/tools/test_tool_registry_split.py

# Test: ACTIVE_TOOLS contains only working tools
# Test: PLANNED_TOOLS contains only stubbed tools
# Test: agent bindings only reference ACTIVE_TOOLS

# tests/graphs/test_agent_temperature.py

# Test: execution agents (fund_manager, risk_sizing) have temperature=0.0
# Test: hypothesis_generation agent has temperature > 0 (e.g., 0.7)
# Test: agent_executor reads temperature from agent config
```

### Section 11: Kill Switch Auto-Recovery & Log Aggregation

```python
# tests/execution/test_kill_switch_recovery.py

# Test: transient trigger (broker_disconnect) auto-resets after 30-min cooldown
# Test: permanent trigger (daily_loss_limit) does NOT auto-reset
# Test: CRITICAL email sent immediately on trigger
# Test: escalation email sent after 4 hours if not reset
# Test: trigger reason classification is correct for each AutoTriggerMonitor condition
```

---

## Phase 3: Autonomy

### Section 12: Multi-Mode Operation

```python
# tests/graphs/test_multi_mode.py

# Test: MARKET_HOURS mode detected during 9:30-16:00 ET weekday
# Test: EXTENDED_HOURS mode detected during 16:00-20:00 ET weekday
# Test: OVERNIGHT_WEEKEND mode detected during 20:00-04:00 ET
# Test: OVERNIGHT_WEEKEND mode detected on Saturday
# Test: mode detection uses America/New_York timezone (not UTC)
# Test: DST transition (EST→EDT) correctly shifts mode boundaries
# Test: early close day (e.g., day before Thanksgiving) detected correctly
# Test: Trading Graph routes to monitoring-only in EXTENDED_HOURS
# Test: Research Graph routes to heavy compute in OVERNIGHT_WEEKEND
```

### Section 13: Overnight Autoresearch & Error-Driven Iteration

```python
# tests/graphs/test_autoresearch.py

# Test: experiment respects 5-minute wall-clock budget
# Test: experiment with IC > 0.02 registers as draft strategy
# Test: experiment with IC < 0.02 is rejected (not registered)
# Test: 70/30 budget split: 70% new hypotheses, 30% refinement
# Test: hung experiment (LLM timeout) is terminated at budget boundary
# Test: experiment results written to autoresearch_experiments table

# tests/graphs/test_loss_analyzer.py

# Test: losing trade classified correctly as regime_shift
# Test: losing trade classified correctly as signal_failure
# Test: losing trade classified correctly as sizing_error
# Test: 30-day failure frequency aggregation is correct
# Test: research tasks generated for top failure modes
# Test: research tasks feed research_queue table
```

### Section 14: Budget Tracking & Context Compaction

```python
# tests/graphs/test_budget_tracker.py

# Test: budget tracker counts tokens per agent per cycle
# Test: budget tracker counts wall-clock time per agent
# Test: agent exits gracefully at node boundary when budget exhausted
# Test: per-experiment ceiling prevents autoresearch runaway
# Test: budget overshoot < 10% (exits at next boundary, not mid-generation)

# tests/graphs/test_context_compaction.py

# Test: compact_context reduces context size by >= 40%
# Test: compact_context preserves key decisions and action items
# Test: compact_context preserves risk flags
# Test: compaction runs after merge_parallel node
# Test: compaction runs after merge_pre_execution node
```

### Section 15: Layered Circuit Breaker & Greeks Risk

```python
# tests/execution/test_layered_circuit_breaker.py

# Test: daily P&L -1.5% → halts new entries
# Test: daily P&L -2.5% → begins systematic exit
# Test: daily P&L -5% → emergency liquidation with limit orders (not market)
# Test: daily P&L layer resets at market open
# Test: portfolio HWM -3% → halts all trading
# Test: portfolio HWM -5% → defensive exit + kill switch + email
# Test: emergency liquidation uses limit orders with 1% collar
# Test: dead-man's switch: unfilled liquidation after 60s → kill switch + CRITICAL alert
# Test: both layers can trigger independently

# tests/execution/test_greeks_risk_gate.py

# Test: options position exceeding max delta exposure → rejected
# Test: options position exceeding gamma limit → rejected
# Test: options position exceeding vega limit → rejected
# Test: daily theta budget exceeded → new options rejected
# Test: portfolio-level Greeks aggregation is correct
# Test: equity positions bypass Greeks checks
```

### Section 16: Knowledge Base, IC Tracking & Urgency Channel

```python
# tests/tools/test_knowledge_base.py

# Test: search_knowledge_base returns semantically similar results (not recency)
# Test: search_knowledge_base respects top_k parameter
# Test: HNSW index exists on embeddings table

# tests/signal_engine/test_ic_tracker.py

# Test: IC computed correctly as rank correlation between signal and forward returns
# Test: IC computed for 1-day, 5-day, 20-day horizons
# Test: collector disabled when rolling_63d_IC < 0.02
# Test: collector re-enabled when rolling_63d_IC > 0.03 (hysteresis)
# Test: IC results written to signal_ic table

# tests/coordination/test_urgency_channel.py

# Test: PG LISTEN/NOTIFY delivers event sub-second
# Test: Trading Graph receives urgent event from Supervisor
# Test: urgency channel works across separate connections (simulating Docker containers)
# Test: multiple urgent events queued and delivered in order
```
