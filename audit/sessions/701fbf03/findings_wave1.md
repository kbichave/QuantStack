# Wave 1: Deep Analysis of Data Layer, Database Schema, Observability & Self-Healing

<!-- source: codebase-deep-audit, confidence: high, topic: data-ops, wave: 1, date: 2026-04-07 -->

## Executive Summary

QuantStack's data and operational infrastructure is **architecturally sound but operationally immature** in three critical areas:

1. **DB.py monolith** (3,473 LOC, 120 tables): Idempotent migration strategy with advisory locks is production-grade. Schema is well-normalized. **Concern**: no version tracking, no rollback mechanism, zero tests.

2. **Data layer** (67 files, 5 providers): Robust circuit-breaker failover with per-provider rate limiting. **Concern**: only Alpha Vantage truly primary; fallbacks rarely tested. Freshness checks exist but 8-hour staleness threshold is loose for intraday.

3. **Observability** (10 files): Prometheus + Grafana + Loki + Langfuse integrated. Custom metrics track trades, risk rejections, kill-switch, agent latency. **Concern**: dashboards cover tactical alerts; strategic cost tracking and per-cycle success metrics are missing.

4. **Self-healing** (AutoResearchClaw via Supervisor): Tool health monitor auto-disables tools below 50% success rate (5 invocations minimum). Bug-fix watcher dispatches tasks immediately. **Concern**: no validation before patching; no protected files list; overnight/morning runners are stubs.

5. **RAG + Memory** (4 files + Mem0): pgvector embeddings (1024D), Ollama embeddings, Mem0 persistence for cross-session learning. **Concern**: only 3 collections defined; zero agent integration.

6. **Governance** (CIO agent): Static mandate generation. **Concern**: todo to replace with LLM-based generation.

---

## Data Layer, DB Schema & Operations

### Database Schema Quality

**Strengths:**
- **Idempotent migrations**: All DDL uses `CREATE TABLE IF NOT EXISTS`, `ADD COLUMN IF NOT EXISTS`, no race conditions.
- **Advisory lock pattern**: `pg_try_advisory_lock(42)` ensures only one service runs migrations at startup. 8-second lock timeout prevents hangs.
- **Two-layer organization**:
  - Operational tables: positions, fills, cash_balance, decisions, strategies, regime_strategy_matrix, execution audit
  - Analytics tables: ml_experiments, research_queue, autoresearch_experiments, reflexion_episodes, feature_candidates
- **Connection pooling**: psycopg3 pool (4–20 size, 1h lifetime, 10m idle close), thread-safe via Lock()
- **Backward-compatible row factory**: `_DictRow` supports both dict access and tuple unpacking for legacy code

**Schema Coverage (120 tables):**
- Positions, fills, closed trades, cash_balance
- Strategies, regime matrix, strategy outcomes, strategy daily P&L
- Agent conversations, agent events, agent DLQ
- Research queue, research WIP, research plans
- ML experiments, model registry, feature candidates
- OHLCV (daily + 1-min), financial statements, insider trades, earnings calendar
- Signal snapshots, signal IC, regime state
- News sentiment, put/call ratio, options chains
- Tool health, tool demand signals, autoresearch experiments
- System state, system alerts, loop events, loop heartbeats

**Concerns:**

1. **No version tracking**: Migrations are sequential but no `schema_version` or `migration_history` table. Rollbacks require manual DDL.

2. **No schema tests**: 0 migration tests. If a DDL statement fails mid-phase, manual recovery needed.

3. **JSON parsing workaround** (legacy): `set_json_loads()` kept as string to support old `json.loads()` calls. Creates latent bugs if not migrated.

4. **Partition strategy missing**: OHLCV/fills tables will hit 100M+ rows (5 years × 100 symbols × 390 rows/day). No partitioning configured; index scans will slow.

5. **Column creep**: positions table has 25+ columns (strike, expiry, premium_at_entry mixed with equity fields). Suggests union anti-pattern instead of separate options_positions table.

**Recommendations:**
- Add `schema_versions` table to track applied migrations
- Create migration tests (pytest fixtures that run DDL → verify schema)
- Partition OHLCV on (symbol, year)
- Split options into dedicated schema (options_positions, options_chain_history)

### Connection Pool & Concurrency

**Current Model:**
```
Main app pool:        4–20 connections
Checkpointer pool:    2–6
Scheduler:            1–2
Backup jobs:          1
Total max:            ~29 (well below PostgreSQL default 100)
```

**Strategy:** Thread-safe `ConnectionPool` with MVCC (unlimited concurrent readers, writers never block readers).

**Strengths:**
- No file locks, no lock contention
- Per-process `_pg_pool` singleton with module-level Lock
- Connection lifetime 1h, idle timeout 10 min (prevents stale sockets)

**Concerns:**
- Pool size fixed at env var `PG_POOL_MAX` (default 20). No adaptive scaling if multiple graphs contest the pool.
- No connection pool metrics (active, idle, wait time).

---

## Data Providers & Acquisition Pipeline

### Provider Coverage

| Provider | Data Types | Rate Limit | Primary/Fallback |
|----------|-----------|-----------|-----------------|
| Alpha Vantage | ohlcv_daily, intraday, macro, fundamentals, earnings, insider, institutional, options, news, commodities | 75/min (premium) | Primary for all |
| FRED | macro_indicator | 120/min | Fallback for macro |
| EDGAR | sec_filings, insider, institutional | 10/sec | Fallback for insider/institutional |
| Alpaca | Intraday bars, real-time quotes | Per-plan | Paper trading only |
| IBKR | Historical + streaming | N/A | Stubs only |
| Polygon | Historical bars | Per-plan | Research mentions only |

**Acquisition phases** (12 sequential, all Alpha Vantage primary):
1. ohlcv_5min (50 symbols × 24 months = 1200 calls, daily delta)
2. ohlcv_1h (similar)
3. ohlcv_daily (50 calls, daily delta)
4. financials (50 × 3 = 150 calls, quarterly)
5. earnings_history (50 calls, quarterly)
6. macro (9 global series, monthly)
7. insider (50 calls, weekly)
8. institutional (50 calls, quarterly)
9. corporate_actions (50 × 2 = 100 calls, quarterly)
10. options (50 calls, daily)
11. news (10 batches, daily)
12. fundamentals (50 calls, weekly)

**Total daily API calls at full capacity**: ~1,500–2,000 (well within 75/min × 1440 min = 108k/day quota)

### Rate Limiting Strategy

1. **Primary**: PostgreSQL token bucket (`consume_token('alpha_vantage')`)
   - Shared across all containers
   - Respects 75/min premium tier
   - Falls back to per-process in-memory limiter if DB unavailable (60 retries)

2. **Daily quota gate**: Hard limit at 25,000 calls/day
   - Critical requests bypass at 100% budget
   - Normal requests skip at 80%+
   - Low-priority skip at 50%+

3. **Per-provider circuit breaker**:
   - Failure threshold: 3 consecutive failures
   - Cooldown: 10 minutes
   - Alerts fired via system_events table

### Data Freshness

```python
# data/validator.py
STALE_THRESHOLD_HOURS = 8
# Data is "stale" if most recent bar > 8 hours old
```

**Concerns:**

1. **Alpha Vantage over-reliance**: Only AV is truly primary for 9 of 12 data types. If AV down for 2+ hours during trading → 100% data vacuum.

2. **8-hour staleness threshold is loose for intraday**: During trading hours (09:30–16:00 ET), 8 hours old = missed entire session. Should be 30 minutes during market hours.

3. **No streaming redundancy**: Alpaca + IBKR streaming are optional. If unavailable, system falls back to batch OHLCV every 5 min.

4. **Rate limit fallback is best-effort**: If DB pool exhausted, in-memory limiter is per-process. Multiple concurrent threads can burst the rate limit.

5. **Acquisition pipeline has 24 NotImplementedError stubs**: Phases like ohlcv_1h, fundamentals have skeleton code only.

6. **No dedup logic**: If acquisition phase crashes mid-run, re-runs fetch overlapping data. Subsequent aggregation may double-count.

7. **Circuit breaker is time-based only**: If a provider is flaky (50% error rate), circuit breaker won't open until 3 failures in 10 minutes.

**Recommendations:**
- Reduce freshness threshold to 30 min during trading hours, 8 hours after-hours
- Wire FRED + EDGAR as true redundant sources (test quarterly)
- Add dedup on (symbol, timeframe, timestamp) with upsert-on-conflict
- Expand circuit breaker: open after 3 failures OR 50% error rate in 10 min

---

## Observability: Prometheus, Grafana, Langfuse

### Metrics Instrumentation

**Prometheus metrics deployed:**

| Metric | Type | Labels | Collection |
|--------|------|--------|------------|
| `quantstack_trades_executed_total` | Counter | symbol, side, speed | execution layer |
| `quantstack_risk_rejections_total` | Counter | violation_type | risk_gate |
| `quantstack_agent_latency_seconds` | Histogram | agent_name | LLM integration |
| `quantstack_signal_staleness_seconds` | Gauge | symbol | signal engine |
| `quantstack_portfolio_nav_dollars` | Gauge | none | 30s flusher |
| `quantstack_daily_pnl_dollars` | Gauge | none | 30s flusher |
| `quantstack_kill_switch_active` | Gauge | none | risk manager |
| `quantstack_tick_executor_lag_seconds` | Histogram | none | tick executor |
| `quantstack_cycle_success_rate` | Gauge | none | loop runner |
| `quantstack_strategy_generation_7d` | Gauge | none | research loop |

**Grafana dashboard** includes:
- Active alerts, alert history
- Kill switch status, OOM events
- Container memory usage, trades, risk rejections

**LangFuse integration:**
- Spans for LLM calls, tool invocations, research phases
- Custom events for provider failover, strategy lifecycle, auto-patch
- Best-effort: if Langfuse unavailable, tracing no-ops silently

**Concerns:**

1. **Circular metrics gap**: Prometheus tracks trades_executed but NOT:
   - Total trade decisions attempted (denominator for success rate)
   - Fill ratio (requested vs actual fills)
   - Average fill latency (tick arrival → confirmation)

2. **Cost tracking is decoupled**: `observability/cost_queries.py` has per-agent token budgets and model pricing, but NO metrics exported to Prometheus.

3. **Research throughput opacity**: No metrics for hypothesis generation rate, backtest latency, strategy promotion latency.

4. **Cycle health metrics are absent**:
   - Cycle duration (iteration time)
   - Tool failure count per cycle
   - Agent timeout count per cycle
   - Loop restart count

5. **No per-provider latency tracking**: Registry records failures but not latency histograms.

6. **Langfuse backend not monitored**: If Langfuse is slow/down, no alert fires.

**Recommendations:**
- Export cost tracking to Prometheus: `quantstack_agent_cost_usd`, `quantstack_daily_cost_usd`
- Add cycle health: `quantstack_cycle_duration_seconds`, `quantstack_cycle_tool_failures`
- Add provider latency: `quantstack_provider_call_latency_seconds`
- Add research pipeline: `quantstack_hypothesis_generation_rate`, `quantstack_backtest_latency_seconds`

---

## Self-Healing: Tool Health, Bug Fixing, AutoResearchClaw

### Tool Health Monitoring

**Tool health table:**
```sql
CREATE TABLE tool_health (
    tool_name TEXT PRIMARY KEY,
    invocation_count INTEGER,
    success_count INTEGER,
    failure_count INTEGER,
    avg_latency_ms DOUBLE PRECISION,
    last_invoked TIMESTAMPTZ,
    last_error TEXT,
    status TEXT DEFAULT 'active'
);
```

**Auto-disable logic** (daily):

```python
SUCCESS_RATE_THRESHOLD = 0.50  # 50% minimum
MIN_INVOCATIONS = 5            # 7-day rolling window

# If (success_count / invocation_count) < 0.50 AND invocations >= 5
# → move tool to DEGRADED status
```

**Concerns:**

1. **No success-rate model for patched code**: When AutoResearchClaw patches a tool, there's no validation or success tracking. Patched tool remains disabled until manual reset.

2. **Threshold is brittle**: 50% success rate = random coin flip. Should be 95%+ (industry standard).

3. **No protected files list**: AutoResearchClaw can patch ANY Python file, including risk_gate.py, db.py, execution/orders.py.

4. **Latency tracking but no alerts**: `avg_latency_ms` recorded but no alert if latency 10x.

5. **Failure mode stats are absent**: We track success/fail counts but NOT failure categories (timeout, API error, parsing, validation).

### Auto-Patch Mechanism

**Architecture:**

1. **Bug detection** (3 consecutive tool failures)
2. **Bug-fix dispatch** (supervisor `_bug_fix_watcher` polls bugs table every 60s)
3. **Patch execution** (90-minute timeout per task)

**Concerns:**

1. **No patch validation before deploy**: Generated patch is not tested before applying to live code.

2. **Code patch history is unclear**: Where are patches stored? No `code_patch_history` table visible.

3. **Dry-run not enforced**: ARC could patch critical files. No staging environment.

4. **Success rate of patches unknown**: No tracking for "patch applied" vs "patch failed" vs "system recovered."

5. **No notification to human on critical patches**: Silent notification via Langfuse, not Discord/email.

**Recommendations:**

1. Add patch validation:
   ```python
   # Workflow: Generate → Syntax check → Unit test → Apply → Track result
   ```

2. Create `code_patches` table with diff + validation status

3. Protected files list (cannot be patched):
   ```python
   PROTECTED_PATHS = {
       'src/quantstack/db.py',
       'src/quantstack/execution/risk_gate.py',
       'src/quantstack/execution/kill_switch.py',
       'src/quantstack/execution/orders.py',
   }
   ```

4. Alert humans on critical patches

### System Health Watchdog

**Watchdog checks** (every 60 seconds):

1. Database connectivity
2. Kill switch state
3. Daily equity snapshot freshness
4. Strategy breaker states
5. SignalEngine liveness
6. Execution pipeline liveness

**Auto-resume logic:**
- If watchdog tripped kill switch AND all checks pass for 5 consecutive minutes → auto-resume

**Concerns:**

1. **Watchdog checks are coarse**: "signal liveness" = "recent snapshots exist," not "signals are fresh & accurate."

2. **No adaptive alerting**: Every CRITICAL fires a Discord alert. Risk of alert fatigue.

3. **Strategy breaker logic is opaque**: Watchdog checks state but not the logic that sets breakers.

4. **Overnight/morning runners not monitored**: They have NotImplementedError stubs.

---

## Memory System: Mem0 Integration

**Architecture:**

1. **Mem0 singleton** (thread-safe)
2. **Vector storage**: Ollama (mxbai-embed-large, 1024D) backed by Qdrant (local)
3. **Memory categories**: MARKET_OBSERVATIONS, WAVE_SCENARIOS, REGIME_CONTEXT, TRADE_DECISIONS, PERFORMANCE_PATTERNS
4. **API**: store_memory(), search_memory(), get_recent()

**Concerns:**

1. **Zero integration with agents**: Memory methods are defined but NOT called from agent code.

2. **No ingestion pipeline**: How do memories get populated? Manual inserts only.

3. **Qdrant single point of failure**: No fallback to full-text search.

4. **No privacy/isolation**: All agents share one Mem0 instance.

5. **Embedding model is fixed**: mxbai-embed-large. Can't swap for domain-specific embeddings.

**Recommendations:**

1. Wire memory ingestion into agents (after trade closes, research completes)
2. Add memory retention policy (expire old, keep top-N relevant)
3. Add memory quality metrics (memory_count, avg_relevance, most_recent per category)

---

## RAG System: pgvector Embeddings

**Architecture:**

1. **Storage**: PostgreSQL with pgvector extension
2. **Collections**: 3 defined (trade_outcomes, strategy_knowledge, market_research)
3. **API**: store_embedding(), search_similar()
4. **Embedding model**: Ollama mxbai-embed-large (1024D)

**Concerns:**

1. **Ingestion not automated**: No pipeline to populate documents.

2. **Usage from agents is minimal**: Grep for `search_similar()` calls in agent code yields 0 results.

3. **No dedup on ingestion**: Duplicate documents = duplicate embeddings.

4. **Metadata filtering is manual**: No query planner.

5. **No chunk size strategy**: How long is `content`? Documents vs chunks unclear.

6. **Embedding model is fixed**: General-purpose. For finance domain, might want FinBERT.

**Recommendations:**

1. Wire ingestion (after strategy backtests, trades close)
2. Add document chunking (500-token sections with 50-token overlap)
3. Add retrieval metrics to Prometheus

---

## Dashboard: Streamlit + TUI

**Operator visibility:**

- Agent events
- Loop heartbeats
- Tool health
- Research queue
- Strategy breaker states
- Kill switch state
- Daily P&L
- Position summary

**Concerns:**

1. **No cost/budget visibility**: Dashboard has no pane showing LLM cost, token usage, cost anomaly threshold.

2. **No data freshness pane**: Can't see which data sources are stale, which providers rate-limited.

3. **No research pipeline transparency**: Can't see hypothesis generation rate, backtest queue, strategy promotion status.

4. **No trade quality metrics**: Can't see fill ratios, slippage, market impact, TCA.

**Recommendations:**

1. Add cost/budget pane
2. Add data freshness pane
3. Add research pipeline pane
4. Add trade quality pane

---

## Governance: CIO Agent & Daily Mandate

**Current state:**

```python
# governance/cio_agent.py: generate_daily_mandate()
mandate = DailyMandate(
    mandate_id=f"cio-{uuid.uuid4().hex[:8]}",
    date=today,
    regime_assessment="normal",
    allowed_sectors=["Technology", "Healthcare", ...],
    max_new_positions=5,
    max_daily_notional=50_000.0,
    ...
)
```

**Concerns:**

1. **Mandate is hardcoded**: No LLM input. Static default for every day.

2. **Regime assessment is ignored**: Generated but not used to modulate risk limits.

3. **No mandate validation**: Who approves? Human-in-the-loop?

4. **TODO in docstring**: Replace with LLM-based generation (per spec).

5. **Mandate enforcement unclear**: Is mandate enforced by risk_gate? Or advisory?

6. **No historical mandate log**: If mandate changes, no audit trail.

**Recommendations:**

1. Implement LLM-based mandate generation (per TODO)
2. Add mandate enforcement logic in risk_gate
3. Export mandate to dashboard (visible to ops team)

---

## Operational Maturity Assessment

### Strength Areas:

1. Migration strategy is production-grade (advisory lock + autocommit)
2. Connection pooling is well-designed (thread-safe, idle timeout)
3. Circuit breakers on data providers (prevents cascading failures)
4. Tool health monitoring exists (auto-disables broken tools daily)
5. Watchdog + auto-resume (fully autonomous recovery)

### Risk Areas:

| Risk | Severity | Recommendation |
|------|----------|----------------|
| No migration rollback mechanism | HIGH | Add schema_versions + rollback procedure |
| Alpha Vantage single point of failure | HIGH | Wire FRED/EDGAR as true redundant sources |
| No schema versioning | HIGH | Implement migration history table |
| 8-hour staleness threshold too loose | MEDIUM | Reduce to 30 min during trading hours |
| Auto-patch has no validation | MEDIUM | Add test run before patching |
| Patched code success not tracked | MEDIUM | Add `code_patch_results` metrics |
| No protected files list for ARC | MEDIUM | Define PROTECTED_PATHS, alert on critical |
| Tool health threshold too low (50%) | MEDIUM | Increase to 95%+ (industry standard) |
| Overnight/morning runners are stubs | MEDIUM | Implement backtest integration |
| CIO mandate is hardcoded | MEDIUM | Implement LLM-based generation |
| Research pipeline metrics missing | MEDIUM | Export metrics to Prometheus |
| Cost tracking not exposed | MEDIUM | Export cost metrics to Prometheus |
| Memory system not integrated | MEDIUM | Wire ingestion + semantic search |
| Dashboard lacks cost visibility | LOW | Add cost pane |
| No failure mode categorization | LOW | Add failure_modes table |

---

## Open Questions for Wave 2

1. **Patch rollback**: If AutoResearchClaw breaks a file at runtime, what's the rollback procedure?
2. **Partition strategy**: Is partitioning planned for OHLCV/fills tables (100M+ rows in 5 years)?
3. **Failure mode drill**: Has team practiced full failover (AV down, FRED/EDGAR only)? Latency impact?
4. **Cost control**: Total monthly spend on AV + LLM + AWS + Langfuse? Within budget?
5. **Data audit trail**: For compliance, do we log who requested which data when?
6. **Stale data recovery**: If staleness alert fires, what's the procedure? Re-fetch? Use cached?

---

**Report compiled**: 2026-04-07  
**Confidence**: HIGH  
**Wave**: 1 (Data Layer, DB Schema, Observability, Self-Healing)
