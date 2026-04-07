# Section 16: Knowledge Base, IC Tracking & Urgency Channel

## Summary

This section addresses three independent subsystems that share a Phase 3 boundary:

1. **Knowledge Base semantic search fix** — The `search_knowledge_base` tool already works correctly via `rag.query.search_knowledge_base()`, using pgvector cosine similarity with an HNSW index. Verification is needed to confirm the HNSW index exists and semantic search is functional.
2. **Signal IC daily computation** — A new `ic_tracker.py` module that computes daily Information Coefficient (rank correlation between signal strength and forward returns) for each signal collector, stores results in the existing `signal_ic` table, and gates weak collectors (rolling 63-day IC below 0.02) from the synthesis step.
3. **PG LISTEN/NOTIFY urgency channel** — A sub-second event delivery mechanism for urgent Supervisor-to-Trading events using PostgreSQL's built-in pub/sub, replacing the 5-10 minute poll latency for time-critical events.

## Dependencies

- **Phase 2 complete** — This section is in Phase 3, Batch 5.
- **Section 13 depends on this** — The IC tracker's `IC > 0.02` gate is used by the autoresearch loop in section-13 to decide whether experiments pass muster.

## Current State of Relevant Code

Before implementing, verify these existing pieces:

- `src/quantstack/tools/langchain/learning_tools.py` already calls `rag_search(query=query, n_results=top_k)` — the semantic search is already wired correctly. The original CTO audit finding (MC0: "search by recency, ignoring query") appears to have been fixed in the uncommitted baseline.
- `src/quantstack/rag/query.py` contains `search_knowledge_base()` which generates an embedding from the query, then calls `search_similar()` using pgvector's `<=>` cosine distance operator. This is correct semantic search.
- `src/quantstack/db.py` contains `_migrate_hnsw_index_pg()` which creates `idx_embeddings_hnsw` on `embeddings` using `hnsw (embedding vector_cosine_ops)` with `m=16, ef_construction=100`. This migration runs at startup.
- `src/quantstack/db.py` contains `_migrate_signal_ic_pg()` which creates the `signal_ic` table with columns: `date`, `strategy_id`, `horizon_days`, `rank_ic`, `ic_positive_rate`, `icir_21d`, `icir_63d`, `ic_tstat`, `n_symbols`. Primary key is `(date, strategy_id, horizon_days)`.
- `src/quantstack/coordination/event_bus.py` is poll-based with per-consumer cursors. It already has `IC_DECAY`, `RISK_EMERGENCY`, and `KILL_SWITCH_TRIGGERED` event types defined. There is no LISTEN/NOTIFY mechanism yet.
- `src/quantstack/signal_engine/engine.py` orchestrates collectors via `asyncio.gather`. There is no IC-based gating of collectors today.

---

## Tests

Write these test files first, then implement to make them pass.

### tests/tools/test_knowledge_base.py

```python
# Test: search_knowledge_base returns semantically similar results (not recency)
#   - Insert two embeddings: one semantically close to query, one recent but unrelated
#   - Verify the semantically close result ranks first regardless of timestamp
#
# Test: search_knowledge_base respects top_k parameter
#   - Insert 10 embeddings, search with top_k=3
#   - Verify exactly 3 results returned
#
# Test: HNSW index exists on embeddings table
#   - Query pg_indexes for idx_embeddings_hnsw
#   - Verify it exists and uses hnsw access method
```

These tests validate that the existing implementation is correct. Use the `integration` marker since they require a real PostgreSQL instance with pgvector.

### tests/signal_engine/test_ic_tracker.py

```python
# Test: IC computed correctly as rank correlation between signal and forward returns
#   - Provide known signal values and known forward returns
#   - Verify rank_ic matches scipy.stats.spearmanr result
#
# Test: IC computed for 1-day, 5-day, 20-day horizons
#   - Run compute_ic with a date and strategy
#   - Verify 3 rows written to signal_ic (one per horizon)
#
# Test: collector disabled when rolling_63d_IC < 0.02
#   - Seed signal_ic with 63 days of IC = 0.01 for a collector
#   - Call should_disable_collector() → returns True
#
# Test: collector re-enabled when rolling_63d_IC > 0.03 (hysteresis)
#   - Seed signal_ic with 63 days of IC = 0.035
#   - Collector was previously disabled
#   - Call should_enable_collector() → returns True
#   - Verify IC must exceed 0.03 (not 0.02) to re-enable — hysteresis prevents flapping
#
# Test: IC results written to signal_ic table
#   - Run compute_ic for a date + strategy
#   - Query signal_ic table directly
#   - Verify rows exist with correct primary key and non-null rank_ic
```

### tests/coordination/test_urgency_channel.py

```python
# Test: PG LISTEN/NOTIFY delivers event sub-second
#   - Set up listener on 'quantstack_urgent' channel
#   - Send NOTIFY from a separate connection
#   - Verify delivery latency < 1 second
#
# Test: Trading Graph receives urgent event from Supervisor
#   - Supervisor publishes urgent event via notify_urgent()
#   - Trading Graph's check_urgent_channel() picks it up
#   - Verify event payload is intact
#
# Test: urgency channel works across separate connections (simulating Docker containers)
#   - Open two independent PG connections (separate pools)
#   - One listens, other notifies
#   - Verify delivery works (confirms cross-container compatibility)
#
# Test: multiple urgent events queued and delivered in order
#   - Send 3 NOTIFY events in sequence
#   - Verify all 3 received in order
```

Use the `integration` marker for urgency channel tests since they require real PostgreSQL LISTEN/NOTIFY.

---

## Implementation

### Part 1: Knowledge Base HNSW Verification

**Goal:** Confirm that semantic search and the HNSW index are already working. No code changes expected — this is a verification step.

**Verification steps:**

1. Connect to the database and verify `idx_embeddings_hnsw` exists:
   ```sql
   SELECT indexname, indexdef FROM pg_indexes WHERE indexname = 'idx_embeddings_hnsw';
   ```
2. Run `EXPLAIN ANALYZE` on a sample vector search to confirm the HNSW index is used:
   ```sql
   EXPLAIN ANALYZE SELECT id, embedding <=> '[0.1, 0.2, ...]'::vector AS distance
   FROM embeddings ORDER BY distance LIMIT 5;
   ```
   The plan should show "Index Scan using idx_embeddings_hnsw", not "Seq Scan".
3. Verify `learning_tools.py` calls `rag_search(query=query, n_results=top_k)` — already confirmed in current code.

**If the HNSW index is missing:** The migration in `db.py` (`_migrate_hnsw_index_pg`) should create it on startup. If the embeddings table has fewer than ~1000 rows, PostgreSQL may choose a sequential scan anyway (cheaper for small tables). This is acceptable — the index exists for when the table grows.

**Files:** No modifications expected. Write the verification tests only.

### Part 2: Signal IC Tracker

**Goal:** Compute daily Information Coefficient for each signal collector/strategy, persist to `signal_ic`, and gate collectors whose signals have no predictive power.

**New file: `src/quantstack/signal_engine/ic_tracker.py`**

This module provides:

- `compute_daily_ic(date, strategy_id, conn)` — Computes rank IC (Spearman correlation) between signal strength on `date` and forward returns at 1-day, 5-day, and 20-day horizons. Reads from the `signals` table (signal values) and price data (forward returns). Writes results to the `signal_ic` table. Also computes rolling metrics: `icir_21d` (21-day IC information ratio), `icir_63d` (63-day IC information ratio), `ic_tstat` (t-statistic for IC significance), `ic_positive_rate` (fraction of days with positive IC), and `n_symbols` (cross-sectional breadth).

- `get_rolling_ic(strategy_id, horizon_days, window, conn)` — Returns the rolling mean IC over the specified window (default 63 trading days) for a given strategy and horizon.

- `should_disable_collector(strategy_id, conn)` — Returns `True` if `rolling_63d_IC < 0.02` for the 5-day horizon. This is the primary gating metric.

- `should_enable_collector(strategy_id, conn)` — Returns `True` if `rolling_63d_IC > 0.03` for the 5-day horizon. The asymmetric thresholds (0.02 to disable, 0.03 to re-enable) provide hysteresis to prevent flapping when IC oscillates around the boundary.

- `run_daily_ic_sweep(conn)` — Iterates over all active strategies, calls `compute_daily_ic` for each, and publishes `IC_DECAY` events via EventBus for any newly disabled collectors.

**Key design decisions:**

- IC is computed as cross-sectional rank correlation (Spearman), not time-series correlation. This matches the standard quant research definition.
- The 5-day horizon is the primary gating metric because it balances signal decay with statistical stability.
- Use `scipy.stats.spearmanr` for the rank correlation. It handles ties correctly.
- Forward returns use close-to-close percentage changes from price data already in the database.
- Minimum cross-section size: require at least 20 symbols per date to compute IC. Below this, the rank correlation is too noisy.

**Modify: `src/quantstack/signal_engine/engine.py`**

Add IC gating to the collector orchestration. Before running a collector, check `should_disable_collector(collector_strategy_id)`. If disabled, skip the collector and log a warning. This integrates into the existing `asyncio.gather` pattern — disabled collectors simply return an empty dict (same as a failed collector).

Track disabled collector state in memory (a `set[str]` of disabled strategy IDs) refreshed at the start of each engine run. This avoids hitting the database on every collector invocation.

**Modify: `scripts/scheduler.py`**

Add a daily job at 17:00 ET (after market close + 1 hour for data settlement):

```python
# Signature only — implementation follows scheduler's existing pattern
async def job_daily_ic_computation():
    """Compute daily IC for all active strategies."""
    ...
```

This job calls `run_daily_ic_sweep()`. It runs after the 16:10 P&L attribution job to ensure price data is fresh.

### Part 3: PG LISTEN/NOTIFY Urgency Channel

**Goal:** Provide sub-second event delivery from Supervisor to Trading Graph for urgent events, bypassing the 5-10 minute poll cycle.

**Modify: `src/quantstack/coordination/event_bus.py`**

Add two methods to the `EventBus` class:

- `notify_urgent(event)` — Publishes an event via both the existing append-only log (for durability and audit) AND `NOTIFY quantstack_urgent, '{event_json}'` (for sub-second delivery). The NOTIFY payload is a JSON string containing `event_type`, `event_id`, and `payload`. PG NOTIFY payloads are limited to 8000 bytes — truncate if needed and include a `truncated: true` flag so the consumer knows to fetch the full event from the log.

- `listen_urgent(callback, timeout_seconds)` — Sets up a `LISTEN quantstack_urgent` on the connection and calls `callback(event)` for each received notification. This is a blocking call — run it in a dedicated thread or asyncio task. Use `conn.notifies(timeout=timeout_seconds)` from psycopg3's async notification API.

**Why PG LISTEN/NOTIFY:** Zero additional dependencies (PostgreSQL is already running). Works across Docker containers (notifications go through the PG server, not shared memory). No shared filesystem needed. Built-in ordering guarantees (notifications delivered in commit order). The only limitation is the 8000-byte payload size, which is sufficient for event metadata — large payloads can reference the full event in the `loop_events` table.

**Modify: `src/quantstack/graphs/trading/nodes.py`**

Add an urgent event check in the `safety_check` node (or at the start of the trading cycle). This checks for any pending urgent notifications before proceeding with execution. If an urgent `KILL_SWITCH_TRIGGERED` or `RISK_EMERGENCY` event is received, abort the current cycle immediately.

The urgent check should be non-blocking with a short timeout (100ms). If no notification is pending, proceed normally. This adds negligible latency to the trading cycle.

**Integration pattern:**

```
Supervisor detects critical condition
  → EventBus.notify_urgent(KILL_SWITCH_TRIGGERED)
  → PG NOTIFY quantstack_urgent (sub-second)
  → Trading Graph safety_check receives notification
  → Aborts cycle
```

Compared to the current poll-based approach:
- Poll: 5-10 min latency (next iteration start)
- LISTEN/NOTIFY: <1 second latency

**Edge cases to handle:**
- PG connection drops: LISTEN is lost. On reconnect, re-issue LISTEN. Use the poll-based EventBus as a fallback — the event is also in `loop_events`.
- Multiple Trading Graph instances: Each instance opens its own LISTEN. PG delivers the notification to all listeners. This is correct behavior (all instances should react to urgent events).
- NOTIFY without active listener: The notification is silently dropped by PG. This is fine because the event is also in the durable `loop_events` table and will be picked up on the next poll.

---

## File Summary

| File | Action | Purpose |
|------|--------|---------|
| `tests/tools/test_knowledge_base.py` | Create | HNSW and semantic search verification tests |
| `tests/signal_engine/test_ic_tracker.py` | Create | IC computation and gating tests |
| `tests/coordination/test_urgency_channel.py` | Create | LISTEN/NOTIFY delivery tests |
| `src/quantstack/signal_engine/ic_tracker.py` | Create | IC computation, rolling metrics, collector gating |
| `src/quantstack/signal_engine/engine.py` | Modify | Add IC-based collector gating before running collectors |
| `src/quantstack/coordination/event_bus.py` | Modify | Add `notify_urgent()` and `listen_urgent()` methods |
| `src/quantstack/graphs/trading/nodes.py` | Modify | Add urgent channel check in safety_check node |
| `scripts/scheduler.py` | Modify | Add daily IC computation job at 17:00 ET |

## Verification Checklist

- [ ] HNSW index exists on embeddings table (query `pg_indexes`)
- [ ] `search_knowledge_base` returns results ranked by cosine similarity, not recency
- [ ] IC tracker computes rank IC matching scipy.stats.spearmanr for known inputs
- [ ] IC computed for all 3 horizons (1d, 5d, 20d) per strategy per day
- [ ] Collector disabled when rolling 63d IC < 0.02; re-enabled when > 0.03
- [ ] `IC_DECAY` event published when a collector is newly disabled
- [ ] PG NOTIFY delivers event in under 1 second in integration test
- [ ] Urgent event received by Trading Graph's safety_check node
- [ ] NOTIFY payload truncation handles events > 8000 bytes gracefully
- [ ] Scheduler runs IC sweep daily at 17:00 ET after P&L attribution
